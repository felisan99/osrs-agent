"""Milestone 4 acceptance check (SPEC §13.4): hybrid retrieval + Jev grading on the eval set.

For every eval case answered from wiki passages (quests, money-making details, general
knowledge) it retrieves the top-k candidates, grades them with ONE Jev batch call and reports:

* retrieval: is a passage that holds the answer (``evidence``) among the candidates?
* grading: does it survive the relevance threshold, and what else is retained (manual review)?
* sufficiency: is it high for answerable questions and low for the ones with no answer?

Only Jev is called (a few thousandths of a cent per question); the embeddings are local and no
reasoning LLM is involved. Follow-up questions that need the conversation history are skipped:
they depend on the query-condensing step of M5.

    uv run python -m scripts.m4_check [--only ID ...] [--k 20] [--max-spend 0.25] [--review 6]
    uv run python -m scripts.m4_check --retrieval-only     # no Jev call, no key: recall and rank
    uv run python -m scripts.m4_check --oracle-rewrite     # search with the hand-written English
                                                           # rewrite (`query_en`): retrieval ceiling

The raw probabilities are saved to ``backend/data/eval/`` so the thresholds can be re-tuned
offline without calling Jev again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.clients.jev import JevClient
from app.config import dev_api_key, get_settings
from app.db import MissingDatabaseUrl, async_connection, require_db_url
from app.devtools import BudgetExceeded, SpendGuard
from app.errors import AgentError
from app.rag.embed import Embedder
from app.rag.grade import Grader, GradeResult
from app.rag.retrieve import Passage, Retriever
from app.rag.titles import TitleIndex
from eval.cases import Case, load_cases

CONCURRENCY = 4


@dataclass
class Outcome:
    case: Case
    candidates: list[Passage]
    grade: GradeResult
    retrieval_s: float
    grading_s: float

    def holds_answer(self, passage: Passage) -> bool:
        text = passage.content.lower()
        return any(e.lower() in text for e in self.case.evidence)

    @property
    def evidence_candidates(self) -> list[Passage]:
        return [p for p in self.candidates if self.holds_answer(p)]

    @property
    def evidence_retained(self) -> list[Passage]:
        return [g.passage for g in self.grade.kept if self.holds_answer(g.passage)]

    @property
    def can_answer(self) -> bool:
        """Ground truth for sufficiency: the retained passages really hold the answer.

        An answerable question whose answer was not retrieved is NOT answerable from what was
        found, so calling that set insufficient is the right call (the CRAG loop then rewrites
        the query and searches again, M5).
        """
        return self.case.corpus == "answerable" and bool(self.evidence_retained)


def search_text(case: Case, args: argparse.Namespace) -> str:
    """The text retrieval searches with: the question, or its hand-written English rewrite."""
    return case.query_en if args.oracle_rewrite and case.query_en else case.question


def snippet(passage: Passage, evidence: list[str], width: int = 110) -> str:
    body = passage.content.split("\n", 1)[-1].replace("\n", " ")
    lowered = body.lower()
    for needle in evidence:
        at = lowered.find(needle.lower())
        if at >= 0:
            start = max(at - width // 3, 0)
            return "…" + body[start : start + width].strip() + "…"
    return body[:width].strip() + "…"


def label(passage: Passage) -> str:
    return passage.page_title + (f" > {passage.section}" if passage.section else "")


def print_case(outcome: Outcome, review: int) -> None:
    case, grade = outcome.case, outcome.grade
    kept = grade.kept
    suff = f"{grade.sufficiency:.2f}" if grade.sufficiency is not None else " n/a"
    want = case.corpus == "answerable"
    verdict = "SUFFICIENT" if grade.sufficient else "insufficient"
    ok = grade.sufficient == outcome.can_answer
    print(f"\n[{case.id}] ({case.corpus}) {case.question}")
    line = (
        f"   retained {len(kept)}/{len(outcome.candidates)} | sufficiency {suff} -> {verdict} "
        f"({'OK' if ok else 'MISMATCH'})"
    )
    if want:
        found = "yes" if outcome.evidence_candidates else "NO"
        kept_it = "yes" if outcome.evidence_retained else "NO"
        line += f" | answer in candidates: {found}, retained: {kept_it}"
    print(line)
    for item in kept[:review]:
        mark = "✓" if outcome.holds_answer(item.passage) else " "
        print(
            f"   {mark} {item.relevance:.2f}  {label(item.passage)}  "
            f"{snippet(item.passage, case.evidence)}"
        )
    dropped = sorted((g for g in grade.graded if not g.kept), key=lambda g: -g.relevance)
    for item in dropped[: max(review - len(kept), 0) or 2]:
        mark = "✗" if outcome.holds_answer(item.passage) else "·"
        print(
            f"   {mark} {item.relevance:.2f}  (dropped) {label(item.passage)}  "
            f"{snippet(item.passage, case.evidence, 70)}"
        )


def summarise(outcomes: list[Outcome]) -> None:
    answerable = [o for o in outcomes if o.case.corpus == "answerable"]
    print("\n" + "=" * 100)
    if answerable:
        in_cands = sum(bool(o.evidence_candidates) for o in answerable)
        in_kept = sum(bool(o.evidence_retained) for o in answerable)
        lost = [o.case.id for o in answerable if o.evidence_candidates and not o.evidence_retained]
        missed = [o.case.id for o in answerable if not o.evidence_candidates]
        print(
            f"retrieval  : answer present in the candidates for {in_cands}/{len(answerable)} "
            f"answerable questions; kept by grading for {in_kept}/{len(answerable)}"
        )
        if missed:
            print(f"             not retrieved: {missed}")
        if lost:
            print(f"             retrieved but dropped by grading: {lost}")
        precision = [
            sum(o.holds_answer(g.passage) for g in o.grade.kept) / len(o.grade.kept)
            for o in answerable
            if o.grade.kept
        ]
        mean_kept = sum(len(o.grade.kept) for o in answerable) / len(answerable)
        print(
            f"             retained passages per question: {mean_kept:.1f}; "
            f"share of them holding the evidence string: {sum(precision) / len(precision):.0%} "
            "(a lower bound: other passages may be legitimately relevant)"
        )
    sufficiency_right = sum(o.grade.sufficient == o.can_answer for o in outcomes)
    print(
        f"sufficiency: correct on {sufficiency_right}/{len(outcomes)} questions "
        "(sufficient exactly when the retained passages hold the answer)"
    )
    can = [o for o in outcomes if o.can_answer]
    cannot = [o for o in outcomes if not o.can_answer]
    for name, group in (("can answer", can), ("cannot", cannot)):
        if group:
            values = [o.grade.sufficiency or 0.0 for o in group]
            print(
                f"             {name:10} P(sufficient): min {min(values):.2f}  "
                f"mean {sum(values) / len(values):.2f}  max {max(values):.2f}  (n={len(group)})"
            )
    if can and cannot:
        low = min(o.grade.sufficiency or 0.0 for o in can)
        high = max(o.grade.sufficiency or 0.0 for o in cannot)
        gap = "clean gap" if low > high else "OVERLAP"
        print(f"             lowest 'can answer' {low:.2f} vs highest 'cannot' {high:.2f}: {gap}")
    wrong = [o.case.id for o in outcomes if o.grade.sufficient != o.can_answer]
    if wrong:
        print(f"             misjudged: {wrong}")
    cost = sum(o.grade.cost_usd for o in outcomes)
    retrieval = sorted(o.retrieval_s for o in outcomes)
    grading = sorted(o.grading_s for o in outcomes)
    print(
        f"latency    : retrieval median {retrieval[len(retrieval) // 2] * 1000:.0f} ms, "
        f"grading median {grading[len(grading) // 2]:.2f} s (max {grading[-1]:.2f} s)"
    )
    print(f"cost       : ${cost:.6f} total, ${cost / len(outcomes):.6f} per question (Jev)")


def report_retrieval(rows: list[tuple[Case, list[Passage]]], k: int) -> None:
    """Rank of the first passage holding the answer, for every answerable case (no Jev call)."""
    print(f"\n{'case':36} {'rank':>5}  via (vector/keyword/title rank of that passage)")
    ranks: list[int | None] = []
    for case, passages in rows:
        if case.corpus != "answerable":
            continue
        hit = next(
            (
                (i, p)
                for i, p in enumerate(passages, 1)
                if any(e.lower() in p.content.lower() for e in case.evidence)
            ),
            None,
        )
        ranks.append(hit[0] if hit else None)
        via = (
            f"{hit[1].vector_rank or '-'}/{hit[1].keyword_rank or '-'}/{hit[1].title_rank or '-'}"
            if hit
            else ""
        )
        print(f"{case.id:36} {hit[0] if hit else '-':>5}  {via}")
    found = [r for r in ranks if r]
    n = len(ranks)
    mrr = sum(1 / r for r in found) / n
    print(
        f"\nanswer in the top {k}: {len(found)}/{n} | top 10: {sum(r <= 10 for r in found)}/{n} | "
        f"top 5: {sum(r <= 5 for r in found)}/{n} | MRR {mrr:.3f}"
    )


def save(outcomes: list[Outcome], path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "created": datetime.now(UTC).isoformat(),
        "k": args.k,
        "model": next((o.grade.model for o in outcomes if o.grade.model), ""),
        "cases": [
            {
                "id": o.case.id,
                "question": o.case.question,
                "corpus": o.case.corpus,
                "sufficiency": o.grade.sufficiency,
                "cost_usd": o.grade.cost_usd,
                "passages": [
                    {
                        "id": g.passage.id,
                        "title": g.passage.page_title,
                        "section": g.passage.section,
                        "rrf": g.passage.score,
                        "vector_rank": g.passage.vector_rank,
                        "keyword_rank": g.passage.keyword_rank,
                        "relevance": g.relevance,
                        "holds_answer": o.holds_answer(g.passage),
                    }
                    for g in o.grade.graded
                ],
            }
            for o in outcomes
        ],
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        url = require_db_url(settings)
    except MissingDatabaseUrl as exc:
        print(exc, file=sys.stderr)
        return 2
    key = dev_api_key()
    if not key and not args.retrieval_only:
        print("No OPEN_ROUTER_KEY in the environment or .env", file=sys.stderr)
        return 2

    every = [c for c in load_cases() if c.reaches_crag and c.corpus != "n/a"]
    cases = [c for c in every if not c.history and (not args.only or c.id in args.only)]
    skipped = [c.id for c in every if c.history]
    print(f"{len(cases)} cases (skipping follow-ups that need the M5 condense step: {skipped})")

    print("loading embedding model...", flush=True)
    embedder = Embedder(settings)
    _ = embedder.model
    guard = SpendGuard(args.max_spend)
    gate = asyncio.Semaphore(CONCURRENCY)

    titles = await TitleIndex.load(lambda: async_connection(url))
    retriever = Retriever(lambda: async_connection(url), embedder.embed_query, settings, titles)
    if args.retrieval_only:
        rows = [(c, await retriever.search(search_text(c, args), k=args.k)) for c in cases]
        report_retrieval(rows, args.k)
        return 0

    async with httpx.AsyncClient(timeout=settings.http_timeout_s) as http:
        grader = Grader(JevClient(key or "", http=http, settings=settings), settings)

        async def evaluate(case: Case) -> Outcome:
            t0 = time.perf_counter()
            candidates = await retriever.search(search_text(case, args), k=args.k)
            t1 = time.perf_counter()
            async with gate:
                grade = await grader.grade(case.question, candidates, query_ref=case.id)
            guard.add(grade.cost_usd)
            return Outcome(case, candidates, grade, t1 - t0, time.perf_counter() - t1)

        try:
            results = await asyncio.gather(*(evaluate(c) for c in cases), return_exceptions=True)
        except BudgetExceeded as exc:  # pragma: no cover - guarded by gather(return_exceptions)
            print(f"STOPPED: {exc}")
            return 1

    outcomes: list[Outcome] = []
    for case, result in zip(cases, results, strict=True):
        if isinstance(result, BaseException):
            print(f"\n[{case.id}] ERROR {type(result).__name__}: {result}")
            if isinstance(result, AgentError) and result.http_status in (401, 402):
                return 1
            continue
        outcomes.append(result)
        print_case(result, args.review)
    if not outcomes:
        return 1
    summarise(outcomes)
    out = Path(settings.data_dir) / "eval" / f"m4-{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    save(outcomes, out, args)
    print(f"raw probabilities saved to {out}\n{guard.summary()}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--k", type=int, default=get_settings().crag_topk, help="candidates")
    parser.add_argument("--max-spend", type=float, default=0.25, help="USD cap for this run")
    parser.add_argument("--review", type=int, default=5, help="passages listed per question")
    parser.add_argument(
        "--retrieval-only", action="store_true", help="skip grading: no Jev call, no key needed"
    )
    parser.add_argument(
        "--oracle-rewrite",
        action="store_true",
        help="retrieve with each Spanish case's hand-written English `query_en`",
    )
    sys.exit(asyncio.run(main(parser.parse_args())))
