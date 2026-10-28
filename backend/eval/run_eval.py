"""Evaluation of the whole agent against ``questions.yaml`` (SPEC §12, milestone 8).

    uv run python -m eval.run_eval                       # every case, real services (dev key)
    uv run python -m eval.run_eval --cases eval/heldout.yaml     # the held-out set
    uv run python -m eval.run_eval --check-gold                  # labels vs the loaded corpus
    uv run python -m eval.run_eval --only quest-mm2-skill-reqs-en wiki-ge-tax-en
    uv run python -m eval.run_eval --sweep data/eval/run-<date>.json   # re-tune thresholds offline

Each case runs through the real graph in its own conversation thread (the earlier turns of a
follow-up are written into the thread's memory, not paid for). Reported, as SPEC §12 asks:

* router accuracy (also by language) and the misroutes;
* passage filtering precision/recall: is the passage holding the answer retained, and how much of
  what is retained holds it;
* the correct "no evidence" rate, and its opposite (answering "no evidence" when there was one);
* answer checks: expected facts present, forbidden facts and patterns absent, tools called;
* latency and cost per query (Jev and the reasoning model).

The qualitative behaviours (says it found nothing, asks to clarify) are judged by Jev with a Noul
question, which costs a hundredth of a cent. The raw per-case data is saved so ``--sweep`` can
re-tune ``REL_THRESHOLD`` and ``SUFFICIENCY_THRESHOLD`` without calling anything again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.api.schemas import PlayerState
from app.clients.jev import JevClient, NoulAnswer, noul
from app.config import dev_api_key, get_settings
from app.devtools import BudgetExceeded, SpendGuard
from app.errors import AgentError
from app.graph.grounding import numbers_in
from app.graph.prompts import FIXED
from app.service import AgentService
from eval.cases import QUESTIONS_PATH, Case, check_gold, load_cases

CONCURRENCY = 3
JUDGE_THRESHOLD = 0.5
NO_EVIDENCE_TEXTS = set(FIXED["no_evidence"].values())
REFUSAL_TEXTS = set(FIXED["refusal"].values())

JUDGES = {
    "no_evidence": (
        "Does the assistant's reply in `reply` state plainly that the information could not be "
        "found or is not available, WITHOUT giving a concrete answer to the player's `question`?",
        "The reply says it found nothing reliable, or that it does not know, and gives no answer.",
        "The reply gives a concrete answer, or invents details.",
    ),
    "clarify": (
        "Does the assistant's reply in `reply` ask the player for more details or to clarify the "
        "question, instead of answering it?",
        "The reply asks the player to say more, to be specific or to rephrase.",
        "The reply answers the question, or neither asks anything nor explains.",
    ),
}


# --------------------------------------------------------------------------- checks
_TYPOGRAPHIC = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})


def _fold(text: str) -> str:
    """Case-folded, with typographic quotes made plain: a model that writes "Legends\u2019 Quest"
    has not forgotten the quest the wiki spells "Legends' Quest"."""
    return text.translate(_TYPOGRAPHIC).casefold()


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", _fold(text))


def fact_present(answer: str, fact: str) -> bool:
    """Case- and whitespace-insensitive; ``a|b`` accepts either; numbers match by value."""
    for option in fact.split("|"):
        option = option.strip()
        if re.fullmatch(r"[\d.,]+", option):
            wanted = numbers_in(option)
            if wanted and any(
                abs(value - wanted[0][1]) <= 0.005 * max(wanted[0][1], 1)
                for _, value, _ in numbers_in(answer)
            ):
                return True
        if _squash(option) in _squash(answer):
            return True
    return False


def holds_answer(case: Case, passage: dict[str, Any]) -> bool:
    text = _fold(passage["content"])
    return any(_fold(e) in text for e in case.evidence)


@dataclass
class CaseResult:
    case: Case
    state: dict[str, Any] | None = None
    error: str | None = None
    seconds: float = 0.0
    judge_cost: float = 0.0
    checks: dict[str, bool | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and all(v is not False for v in self.checks.values())


def usage_cost(state: dict[str, Any]) -> tuple[float, float]:
    usage = state.get("usage", {})
    return usage.get("jev_cost", 0.0), usage.get("llm_cost", 0.0)


async def judge(jev: JevClient, behavior: str, case: Case, reply: str) -> tuple[bool, float]:
    instructions, yes, no = JUDGES[behavior]
    decision = await jev.decide(
        {"question": case.question, "reply": reply}, {"verdict": noul(instructions, yes=yes, no=no)}
    )
    answer = decision.get("verdict", NoulAnswer)
    return answer.noul >= JUDGE_THRESHOLD, decision.usage.cost or 0.0


async def evaluate(case: Case, state: dict[str, Any], jev: JevClient | None) -> CaseResult:
    result = CaseResult(case, state)
    answer = state.get("answer", "")
    checks = result.checks
    checks["router"] = state.get("intent") == case.intent_esperado
    if not checks["router"]:
        top = sorted(state.get("route_probs", {}).items(), key=lambda kv: -kv[1])[:3]
        shown = ", ".join(f"{k} {v:.2f}" for k, v in top)
        result.notes.append(
            f"routed to {state.get('intent')} ({shown}), expected {case.intent_esperado.value}"
        )

    fixed_no_evidence = answer in NO_EVIDENCE_TEXTS
    fixed_refusal = answer in REFUSAL_TEXTS
    behavior = case.behavior
    if behavior == "answer":
        checks["behavior"] = bool(answer.strip()) and not fixed_no_evidence and not fixed_refusal
        if fixed_no_evidence:
            result.notes.append("answered 'no evidence' although the question has an answer")
    elif behavior == "refuse":
        checks["behavior"] = fixed_refusal and state.get("usage", {}).get("llm_calls", 0) == 0
    else:  # no_evidence / clarify: the fixed text does both; otherwise ask Jev
        if fixed_no_evidence:
            checks["behavior"] = True
        elif jev is not None:
            checks["behavior"], cost = await judge(jev, behavior, case, answer)
            result.judge_cost += cost
        else:
            checks["behavior"] = None
        if checks["behavior"] is False:
            result.notes.append(f"expected '{behavior}', got: {answer[:100]!r}")

    if case.expected_facts and behavior == "answer":
        missing = [f for f in case.expected_facts if not fact_present(answer, f)]
        checks["facts"] = not missing
        if missing:
            result.notes.append(f"missing facts: {missing}")
    if case.avoid_facts:
        present = [f for f in case.avoid_facts if _fold(f) in _fold(answer)]
        checks["avoid_facts"] = not present
        if present:
            result.notes.append(f"forbidden facts present: {present}")
    if case.avoid_patterns:
        hits = [p for p in case.avoid_patterns if re.search(p, answer)]
        checks["avoid_patterns"] = not hits
        if hits:
            result.notes.append(f"forbidden patterns matched: {hits}")
    if case.tools_expected:
        called = set(state.get("tools_called", []))
        missing_tools = [t for t in case.tools_expected if t not in called]
        checks["tools"] = not missing_tools
        if missing_tools:
            result.notes.append(f"tools not called: {missing_tools} (called {sorted(called)})")

    if case.corpus == "answerable" and case.reaches_crag:
        kept = [d for d in state.get("graded", []) if d.get("kept")]
        checks["evidence_retained"] = any(holds_answer(case, d) for d in kept)
        if not checks["evidence_retained"]:
            result.notes.append("the passage with the answer was not retained")
    if case.corpus == "unanswerable":
        checks["no_false_answer"] = behavior != "answer"
    return result


# --------------------------------------------------------------------------- running
async def run_case(
    service: AgentService, key: str, case: Case, jev: JevClient, *, judging: bool
) -> CaseResult:
    thread = f"eval-{case.id}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread}}
    player = PlayerState.model_validate(case.player_state) if case.player_state else None
    started = time.perf_counter()
    try:
        if case.history:
            await service.graph.aupdate_state(
                config,
                {"messages": [{"role": t.role, "content": t.content} for t in case.history]},
                as_node="finalize",
            )
        state = await service.graph.ainvoke(
            {"messages": [{"role": "user", "content": case.question}]},
            config=config,
            context=service.context(key, player),
        )
    except AgentError as exc:
        return CaseResult(
            case, error=f"{exc.code}: {exc.message}", seconds=time.perf_counter() - started
        )
    seconds = time.perf_counter() - started
    result = await evaluate(case, state, jev if judging else None)
    result.seconds = seconds
    return result


# --------------------------------------------------------------------------- reporting
def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else "n/a"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)] if ordered else 0.0


def summarise(results: list[CaseResult]) -> str:
    done = [r for r in results if r.error is None]
    lines: list[str] = []
    out = lines.append
    out(f"cases run: {len(results)} ({len(results) - len(done)} failed with an error)")
    out(f"passed every check: {pct(sum(r.ok for r in done), len(done))}\n")

    routed = [r for r in done if r.checks.get("router") is not None]
    out(f"ROUTER accuracy: {pct(sum(bool(r.checks['router']) for r in routed), len(routed))}")
    for lang in ("en", "es"):
        subset = [r for r in routed if r.case.lang == lang]
        out(f"  {lang}: {pct(sum(bool(r.checks['router']) for r in subset), len(subset))}")
    by_expected: dict[str, list[bool]] = defaultdict(list)
    for r in routed:
        by_expected[r.case.intent_esperado.value].append(bool(r.checks["router"]))
    out(
        "  by expected intent: "
        + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in sorted(by_expected.items()))
    )
    for r in routed:
        if not r.checks["router"]:
            out(f"  MISROUTE {r.case.id}: {r.notes[0]}")

    crag = [r for r in done if r.case.corpus == "answerable" and r.case.reaches_crag]
    retained = [r for r in crag if r.checks.get("evidence_retained")]
    out(f"\nPASSAGE FILTERING (answerable wiki questions: {len(crag)})")
    out(f"  recall: the passage with the answer is retained for {pct(len(retained), len(crag))}")
    precisions = []
    for r in crag:
        kept = [d for d in r.state.get("graded", []) if d.get("kept")]  # type: ignore[union-attr]
        if kept:
            precisions.append(sum(holds_answer(r.case, d) for d in kept) / len(kept))
    if precisions:
        mean_kept = statistics.mean(
            len([d for d in r.state.get("graded", []) if d.get("kept")])
            for r in crag  # type: ignore[union-attr]
        )
        out(
            f"  precision (lower bound, share of retained passages holding the evidence string): "
            f"{statistics.mean(precisions):.0%}; retained per question: {mean_kept:.1f}"
        )
    rounds = Counter(r.state.get("retrieval_iterations", 0) for r in crag)  # type: ignore[union-attr]
    out(f"  searches per question: {dict(sorted(rounds.items()))}")

    no_ev = [r for r in done if r.case.behavior == "no_evidence"]
    clarify = [r for r in done if r.case.behavior == "clarify"]
    answerable = [r for r in done if r.case.behavior == "answer" and r.case.reaches_crag]
    false_no_ev = [r for r in answerable if r.state.get("answer") in NO_EVIDENCE_TEXTS]  # type: ignore[union-attr]
    out("\nNO EVIDENCE")
    out(
        f"  correct 'no evidence' rate (questions the wiki cannot answer): "
        f"{pct(sum(bool(r.checks.get('behavior')) for r in no_ev), len(no_ev))}"
    )
    out(
        f"  ambiguous questions that ask for detail: "
        f"{pct(sum(bool(r.checks.get('behavior')) for r in clarify), len(clarify))}"
    )
    false_rate = pct(len(false_no_ev), len(answerable))
    out(f"  false 'no evidence' (answerable wiki questions): {false_rate}")

    out("\nANSWER CHECKS")
    for name in ("behavior", "facts", "avoid_facts", "avoid_patterns", "tools", "no_false_answer"):
        applicable = [r for r in done if r.checks.get(name) is not None]
        if applicable:
            out(
                f"  {name:16} {pct(sum(bool(r.checks[name]) for r in applicable), len(applicable))}"
            )

    seconds = [r.seconds for r in done]
    jev_cost = [usage_cost(r.state)[0] for r in done]  # type: ignore[arg-type]
    llm_cost = [usage_cost(r.state)[1] for r in done]  # type: ignore[arg-type]
    out("\nLATENCY AND COST (per query)")
    if seconds:
        median, p95 = statistics.median(seconds), percentile(seconds, 0.95)
        out(f"  latency: median {median:.1f}s, p95 {p95:.1f}s")
        out(
            f"  Jev: ${statistics.mean(jev_cost):.5f} average; "
            f"reasoning model: ${statistics.mean(llm_cost):.5f} average"
        )
    total = sum(jev_cost) + sum(llm_cost) + sum(r.judge_cost for r in results)
    judging = sum(r.judge_cost for r in results)
    out(f"  total spent on this run: ${total:.4f} (judging included: ${judging:.5f})")

    failing = [r for r in results if not r.ok]
    if failing:
        out(f"\nFAILING CASES ({len(failing)})")
        for r in failing:
            reason = r.error or "; ".join(r.notes) or "check failed"
            out(f"  {r.case.id}: {reason}")
    return "\n".join(lines)


def to_json(results: list[CaseResult], settings_snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "created": datetime.now(UTC).isoformat(),
        "settings": settings_snapshot,
        "cases": [
            {
                "id": r.case.id,
                "question": r.case.question,
                "intent_expected": r.case.intent_esperado.value,
                "corpus": r.case.corpus,
                "behavior": r.case.behavior,
                "evidence": r.case.evidence,
                "error": r.error,
                "seconds": round(r.seconds, 2),
                "checks": r.checks,
                "notes": r.notes,
                **(
                    {
                        "intent": r.state.get("intent"),
                        "route_probs": r.state.get("route_probs"),
                        "answer": r.state.get("answer"),
                        "sources": r.state.get("sources"),
                        "tools_called": r.state.get("tools_called"),
                        "retrieval_iterations": r.state.get("retrieval_iterations"),
                        "sufficiency": r.state.get("sufficiency"),
                        "usage": r.state.get("usage"),
                        "graded": [
                            {
                                "title": d["page_title"],
                                "relevance": d["relevance"],
                                "kept": d["kept"],
                                "holds_answer": holds_answer(r.case, d),
                            }
                            for d in r.state.get("graded", [])
                        ],
                    }
                    if r.state
                    else {}
                ),
            }
            for r in results
        ],
    }


# --------------------------------------------------------------------------- threshold sweep
def sweep(report: dict[str, Any]) -> str:
    """Re-judge the last grading round of every wiki case under other thresholds, offline."""
    cases = [c for c in report["cases"] if c.get("graded") is not None and c["corpus"] != "n/a"]
    answerable = [c for c in cases if c["corpus"] == "answerable"]
    lines = [
        f"{len(cases)} graded cases ({len(answerable)} answerable), last grading round of each\n"
    ]
    lines.append(
        "REL_THRESHOLD: answerable cases whose evidence passage is retained, passages kept"
    )
    lines.append(
        f"{'thr':>5} {'evidence kept':>15} {'kept/question':>14} {'share holding evidence':>23}"
    )
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        kept_with_evidence = kept_total = holding = 0
        for c in answerable:
            kept = [g for g in c["graded"] if g["relevance"] >= threshold]
            kept_total += len(kept)
            holding += sum(g["holds_answer"] for g in kept)
            kept_with_evidence += any(g["holds_answer"] for g in kept)
        share = holding / kept_total if kept_total else 0
        lines.append(
            f"{threshold:5.1f} {kept_with_evidence:>10}/{len(answerable):<4} "
            f"{kept_total / max(len(answerable), 1):14.1f} {share:23.0%}"
        )
    lines.append(
        "\nSUFFICIENCY_THRESHOLD: does the verdict match 'the retained passages hold the answer'?"
    )
    lines.append(f"{'thr':>5} {'can answer -> sufficient':>26} {'cannot -> insufficient':>24}")
    can = [c for c in answerable if any(g["kept"] and g["holds_answer"] for g in c["graded"])]
    cannot = [c for c in cases if c not in can and c["sufficiency"] is not None]
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        yes = sum((c["sufficiency"] or 0) >= threshold for c in can)
        no = sum((c["sufficiency"] or 0) < threshold for c in cannot)
        lines.append(f"{threshold:5.1f} {yes:>12}/{len(can):<12} {no:>10}/{len(cannot):<12}")
    return "\n".join(lines)


def check_labels(settings: Any, cases: list[Case]) -> int:
    from app.db import require_db_url
    from app.ingest.load import connect

    with connect(require_db_url(settings)) as conn:
        problems = check_gold(conn, cases)
    print(
        "\n".join(problems) if problems else f"gold labels of {len(cases)} cases match the corpus"
    )
    return 1 if problems else 0


# --------------------------------------------------------------------------- main
async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    key = dev_api_key()
    if not key:
        print("No OPEN_ROUTER_KEY in the environment or .env", file=sys.stderr)
        return 2
    cases = [c for c in load_cases(Path(args.cases)) if not args.only or c.id in args.only]
    if args.check_gold:
        return check_labels(settings, cases)
    if not cases:
        print("no case matches --only", file=sys.stderr)
        return 2
    guard = SpendGuard(args.max_spend)
    gate = asyncio.Semaphore(CONCURRENCY)
    print(f"running {len(cases)} cases (spend cap ${args.max_spend:.2f})...", flush=True)
    async with AgentService.open(settings) as service:
        jev = JevClient(key, http=service.http, settings=settings)

        async def one(case: Case) -> CaseResult:
            async with gate:
                result = await run_case(service, key, case, jev, judging=not args.no_judge)
            if result.state:
                guard.add(sum(usage_cost(result.state)) + result.judge_cost)
            print(
                f"  {'ok  ' if result.ok else 'FAIL'} {case.id} ({result.seconds:.1f}s)", flush=True
            )
            return result

        try:
            results = await asyncio.gather(*(one(c) for c in cases))
        except BudgetExceeded as exc:
            print(f"STOPPED: {exc}", file=sys.stderr)
            return 1
    print("\n" + summarise(results))
    out = (
        Path(args.out)
        if args.out
        else settings.data_dir / "eval" / f"run-{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        k: getattr(settings, k)
        for k in ("rel_threshold", "sufficiency_threshold", "crag_topk", "max_crag_iterations",
                  "route_min_confidence", "out_of_domain_min_prob", "mm_price_basis",
                  "reasoning_model", "jev_model")
    }  # fmt: skip
    out.write_text(
        json.dumps(to_json(results, snapshot), indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nsaved {out}\n{guard.summary()}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", default=str(QUESTIONS_PATH), help="questions file")
    parser.add_argument("--check-gold", action="store_true", help="validate labels, run nothing")
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--max-spend", type=float, default=1.0, help="USD cap for this run")
    parser.add_argument("--no-judge", action="store_true", help="skip the Jev judge of behaviours")
    parser.add_argument("--out", help="where to save the raw results (JSON)")
    parser.add_argument("--sweep", help="re-tune thresholds offline from a saved run")
    arguments = parser.parse_args()
    if arguments.sweep:
        print(sweep(json.loads(Path(arguments.sweep).read_text(encoding="utf-8"))))
        sys.exit(0)
    sys.exit(asyncio.run(main(arguments)))
