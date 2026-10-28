"""CRAG grading with Jev (SPEC §7 ``grade``): relevance per passage + overall sufficiency.

ONE batch call carries every candidate: the ``state`` holds the query and each passage as a flat
field (``passage_01``, ``passage_02``, ...), and each question refers to its passage by name in
backticks (the only reference style TypeSafe documents). Questions are independent and see the
same state, so a Noul per passage plus one sufficiency Noul fit in a single request.

Jev's context is 32K tokens for ``state`` + the longest question: when the candidates do not fit
(``JEV_MAX_STATE_TOKENS``) they are split into several relevance-only calls, and sufficiency is then
asked once more over the retained passages.

Raw probabilities can be recorded (eval/debug only, never player data or keys) so thresholds can be
re-tuned without calling Jev again (SPEC §9 ``jev_judgments``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.clients.jev import (
    DecisionsResponse,
    JevClient,
    NoulAnswer,
    NoulQuestion,
    estimate_tokens,
    noul,
)
from app.config import Settings, get_settings
from app.rag.retrieve import Passage

MAX_PASSAGE_CHARS = 2_400  # ~480 tokens of English wiki text (the chunk maximum); cut beyond that
FIELD = "passage_{:02d}"

_RELEVANCE_YES = "The passage states facts needed to answer the question, or answers it directly."
_RELEVANCE_NO = (
    "The passage is about something else, only mentions the topic in passing, or does not "
    "contain what the question asks for."
)
_SUFFICIENT_YES = (
    "Together the passages contain all the facts needed to give a correct and complete answer."
)
_SUFFICIENT_NO = "Key information is missing, or the passages do not address the question."


class JudgmentLog(Protocol):
    async def record(self, kind: str, input_ref: str, probs: dict[str, float]) -> None: ...


@dataclass(frozen=True)
class GradedPassage:
    passage: Passage
    relevance: float
    kept: bool


@dataclass(frozen=True)
class GradeResult:
    graded: list[GradedPassage]
    sufficiency: float | None
    sufficient: bool
    cost_usd: float
    model: str
    calls: int

    @property
    def kept(self) -> list[GradedPassage]:
        """Retained passages, most relevant first."""
        return sorted((g for g in self.graded if g.kept), key=lambda g: -g.relevance)


def relevance_question(field: str) -> NoulQuestion:
    return noul(
        f"Does `{field}` contain information that helps answer the player's question in `query`? "
        "The question may be written in Spanish while the passage is English wiki text: judge by "
        "meaning, not by matching words.",
        yes=_RELEVANCE_YES,
        no=_RELEVANCE_NO,
    )


def sufficiency_question(fields: Sequence[str]) -> NoulQuestion:
    names = f"`{fields[0]}`" if len(fields) == 1 else f"`{fields[0]}` to `{fields[-1]}`"
    return noul(
        f"Taken together, do the passages in {names} contain enough information to answer the "
        "player's question in `query` correctly and completely? The question may be written in "
        "Spanish while the passages are English wiki text.",
        yes=_SUFFICIENT_YES,
        no=_SUFFICIENT_NO,
    )


def _clip(text: str) -> str:
    return text if len(text) <= MAX_PASSAGE_CHARS else text[:MAX_PASSAGE_CHARS].rstrip() + " ..."


def plan_batches(
    passages: Sequence[Passage], *, budget_tokens: int, chars_per_token: float
) -> list[list[Passage]]:
    """Greedy split so that each batch's state (+ its questions) stays inside the budget."""
    per_passage_overhead = 120  # the passage's own question, in tokens
    batches: list[list[Passage]] = [[]]
    used = 600  # query + sufficiency question + slack
    for passage in passages:
        cost = estimate_tokens(_clip(passage.content), chars_per_token) + per_passage_overhead
        if batches[-1] and used + cost > budget_tokens:
            batches.append([])
            used = 600
        batches[-1].append(passage)
        used += cost
    return [b for b in batches if b]


class Grader:
    def __init__(
        self,
        jev: JevClient,
        settings: Settings | None = None,
        judgments: JudgmentLog | None = None,
    ) -> None:
        self._jev = jev
        self._settings = settings or get_settings()
        self._judgments = judgments

    async def _decide(
        self, query: str, passages: Sequence[Passage], *, sufficiency: bool
    ) -> tuple[DecisionsResponse, list[str]]:
        fields = [FIELD.format(i + 1) for i in range(len(passages))]
        state = {
            "query": query,
            **{f: _clip(p.content) for f, p in zip(fields, passages, strict=True)},
        }
        questions = {f: relevance_question(f) for f in fields}
        if sufficiency:
            questions["sufficient"] = sufficiency_question(fields)
        return await self._jev.decide(state, questions), fields

    async def grade(
        self,
        query: str,
        passages: Sequence[Passage],
        *,
        query_ref: str = "",
        rel_threshold: float | None = None,
        sufficiency_threshold: float | None = None,
    ) -> GradeResult:
        rel_threshold = self._settings.rel_threshold if rel_threshold is None else rel_threshold
        suff_threshold = (
            self._settings.sufficiency_threshold
            if sufficiency_threshold is None
            else sufficiency_threshold
        )
        if not passages:
            return GradeResult([], None, False, 0.0, "", 0)

        batches = plan_batches(
            passages,
            budget_tokens=self._settings.jev_max_state_tokens,
            chars_per_token=self._settings.jev_chars_per_token,
        )
        single = len(batches) == 1
        decisions = await asyncio.gather(
            *(self._decide(query, batch, sufficiency=single) for batch in batches)
        )

        relevance: dict[int, float] = {}
        sufficiency: float | None = None
        cost, model, calls = 0.0, "", len(decisions)
        for batch, (decision, fields) in zip(batches, decisions, strict=True):
            cost += decision.usage.cost or 0.0
            model = decision.model or model
            for passage, field in zip(batch, fields, strict=True):
                relevance[passage.id] = decision.get(field, NoulAnswer).noul
            if single:
                sufficiency = decision.get("sufficient", NoulAnswer).noul

        graded = [
            GradedPassage(p, relevance[p.id], relevance[p.id] >= rel_threshold) for p in passages
        ]
        kept = [g.passage for g in sorted(graded, key=lambda g: -g.relevance) if g.kept]

        if not single and kept:
            # The candidates needed several calls: judge sufficiency once over what was retained.
            top = plan_batches(
                kept,
                budget_tokens=self._settings.jev_max_state_tokens,
                chars_per_token=self._settings.jev_chars_per_token,
            )[0]
            decision, _ = await self._decide(query, top, sufficiency=True)
            cost += decision.usage.cost or 0.0
            calls += 1
            sufficiency = decision.get("sufficient", NoulAnswer).noul

        sufficient = bool(kept) and sufficiency is not None and sufficiency >= suff_threshold
        await self._log(query_ref, graded, sufficiency)
        return GradeResult(graded, sufficiency, sufficient, cost, model, calls)

    async def _log(
        self, query_ref: str, graded: Sequence[GradedPassage], sufficiency: float | None
    ) -> None:
        if self._judgments is None:
            return
        for item in graded:
            await self._judgments.record(
                "relevance", f"{query_ref}:{item.passage.id}", {"relevance": item.relevance}
            )
        if sufficiency is not None:
            await self._judgments.record("sufficiency", query_ref, {"sufficiency": sufficiency})
