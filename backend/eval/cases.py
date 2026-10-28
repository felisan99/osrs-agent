"""Evaluation cases (SPEC §12): typed loader for ``questions.yaml`` and gold-label checks.

``load_cases`` resolves each case's ``profile`` into a ``player_state`` and validates the file
strictly (unknown keys, bad regexes, unknown intents or tools fail loudly), because a silently
mislabelled case would corrupt every metric computed from it.

``check_gold`` compares the labels with the loaded corpus: every ``gold_pages`` title must exist and
each ``evidence`` substring must occur in one of those pages' chunks, so a typo in a label is found
before it is blamed on the retriever.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.graph.intents import Intent

QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")
KNOWN_TOOLS = frozenset(
    {
        "get_item_price",
        "get_price_history",
        "calc_flip",
        "analyze_flip",
        "rank_flips",
        "rank_moneymakers",
    }
)
# Intents whose answers come from the CRAG pipeline (retrieve -> grade -> generate).
CRAG_INTENTS = frozenset({Intent.QUEST, Intent.MONEYMAKING, Intent.WIKI_GENERAL})


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    lang: Literal["en", "es"]
    question: str
    query_en: str | None = None  # hand-written English rewrite: what a perfect rewrite step yields
    intent_esperado: Intent
    player_state: dict[str, Any] | None = None
    history: list[Turn] = Field(default_factory=list)
    corpus: Literal["answerable", "unanswerable", "n/a"]
    gold_pages: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    behavior: Literal["answer", "no_evidence", "clarify", "refuse"]
    expected_facts: list[str] = Field(default_factory=list)
    avoid_facts: list[str] = Field(default_factory=list)
    avoid_patterns: list[str] = Field(default_factory=list)
    tools_expected: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("avoid_patterns")
    @classmethod
    def _compile(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:  # not a ValueError: pydantic would let it escape unwrapped
                raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        return patterns

    @field_validator("tools_expected")
    @classmethod
    def _known_tools(cls, tools: list[str]) -> list[str]:
        unknown = sorted(set(tools) - KNOWN_TOOLS)
        if unknown:
            raise ValueError(f"unknown tools {unknown}; known: {sorted(KNOWN_TOOLS)}")
        return tools

    @model_validator(mode="after")
    def _consistent(self) -> Case:
        if self.query_en and self.lang != "es":
            raise ValueError("query_en only makes sense for Spanish questions")
        if self.corpus == "answerable" and not (self.gold_pages and self.evidence):
            raise ValueError("an answerable case needs gold_pages and evidence")
        if self.corpus != "answerable" and (self.gold_pages or self.evidence):
            raise ValueError("only answerable cases carry gold_pages/evidence")
        if self.behavior == "answer" and self.corpus == "unanswerable":
            raise ValueError("an unanswerable case cannot expect a plain answer")
        if self.behavior == "refuse" and self.intent_esperado != Intent.OUT_OF_DOMAIN:
            raise ValueError("only out_of_domain cases expect a refusal")
        return self

    @property
    def reaches_crag(self) -> bool:
        """True when the answer is built from retrieved wiki passages."""
        return self.intent_esperado in CRAG_INTENTS


def load_cases(path: Path = QUESTIONS_PATH) -> list[Case]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    profiles: dict[str, dict[str, Any]] = raw.get("profiles", {})
    cases: list[Case] = []
    seen: set[str] = set()
    for item in raw["cases"]:
        item = dict(item)
        name = item.pop("profile", None)
        if name is not None:
            if name not in profiles:
                raise ValueError(f"case {item.get('id')!r} uses unknown profile {name!r}")
            if "player_state" in item:
                raise ValueError(f"case {item.get('id')!r} sets both profile and player_state")
            item["player_state"] = profiles[name]
        case = Case.model_validate(item)
        if case.id in seen:
            raise ValueError(f"duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    return cases


class _Connection(Protocol):
    def execute(self, query: str, params: Sequence[Any] = ...) -> Any: ...


def check_gold(conn: _Connection, cases: Sequence[Case]) -> list[str]:
    """Problems found when comparing gold labels with the loaded corpus (empty = all good)."""
    problems: list[str] = []
    for case in cases:
        if case.corpus != "answerable":
            continue
        found: dict[str, list[str]] = {}
        for title in case.gold_pages:
            rows = conn.execute(
                "select content from documents where page_title = %s", (title,)
            ).fetchall()
            if not rows:
                problems.append(f"{case.id}: gold page {title!r} is not in the corpus")
            found[title] = [row[0].lower() for row in rows]
        pooled = [text for texts in found.values() for text in texts]
        for needle in case.evidence:
            if pooled and not any(needle.lower() in text for text in pooled):
                problems.append(f"{case.id}: evidence {needle!r} not found in {case.gold_pages}")
    return problems
