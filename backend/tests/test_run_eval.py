"""The pure parts of the evaluation runner (eval/run_eval.py): checks, reports, threshold sweep."""

from __future__ import annotations

import re
from typing import Any

import pytest

from app.graph.prompts import fixed
from eval.cases import Case
from eval.run_eval import (
    CaseResult,
    evaluate,
    fact_present,
    holds_answer,
    summarise,
    sweep,
    to_json,
)


def case(**overrides: Any) -> Case:
    base = {
        "id": "c1",
        "lang": "en",
        "question": "q?",
        "intent_esperado": "wiki_general",
        "corpus": "n/a",
        "behavior": "answer",
    }
    return Case.model_validate(base | overrides)


def state(**overrides: Any) -> dict[str, Any]:
    base = {
        "intent": "wiki_general",
        "answer": "You need 70 Attack [Abyssal whip].",
        "usage": {"llm_calls": 2, "jev_calls": 2, "llm_cost": 0.001, "jev_cost": 0.0004},
        "tools_called": [],
        "graded": [],
        "retrieval_iterations": 1,
    }
    return base | overrides


# --------------------------------------------------------------------------- fact matching
@pytest.mark.parametrize(
    ("answer", "fact", "expected"),
    [
        ("You need 70 Attack", "70", True),
        ("Necesitás 70 de ATTACK", "attack", True),
        ("the tax is 2 % of the price", "2%", True),
        ("cap of 5,000,000 coins", "5 million|5,000,000|5m", True),
        ("capped at 5 million coins", "5 million|5,000,000|5m", True),
        ("pays 30.000 gp", "30,000", True),  # Spanish thousands separator
        ("pays 30000 gp", "30,000", True),
        ("pays 31,000 gp", "30,000", False),
        ("needs Desert Treasure I done", "Desert Treasure", True),
        ("done Legends\u2019 Quest first", "Legends' Quest", True),  # typographic apostrophe
        ("no mention", "Slayer", False),
    ],
)
def test_facts_are_matched_loosely_but_numbers_by_value(
    answer: str, fact: str, expected: bool
) -> None:
    assert fact_present(answer, fact) is expected


# --------------------------------------------------------------------------- per-case checks
async def test_a_correct_wiki_answer_passes_every_check() -> None:
    c = case(
        corpus="answerable",
        gold_pages=["Abyssal whip"],
        evidence=["Attack level of 70"],
        expected_facts=["70"],
    )
    s = state(
        graded=[
            {"page_title": "Abyssal whip", "relevance": 0.99, "kept": True,
             "content": "The whip needs an Attack level of 70."},
            {"page_title": "Cake", "relevance": 0.1, "kept": False, "content": "cake"},
        ]
    )  # fmt: skip
    result = await evaluate(c, s, None)
    assert result.ok and result.checks == {
        "router": True,
        "behavior": True,
        "facts": True,
        "evidence_retained": True,
    }


async def test_misrouting_missing_facts_and_a_dropped_evidence_passage_are_all_reported() -> None:
    c = case(
        corpus="answerable",
        gold_pages=["X"],
        evidence=["Attack level of 70"],
        expected_facts=["70", "Monkey Madness I"],
        avoid_facts=["Dragon Slayer"],
        avoid_patterns=[r"\d+ gp/h"],
        tools_expected=["calc_flip"],
    )
    s = state(
        intent="quest",
        answer="Needs 99 Attack. Dragon Slayer too. Pays 5 gp/h.",
        graded=[
            {"page_title": "X", "relevance": 0.4, "kept": False, "content": "Attack level of 70"}
        ],
    )
    result = await evaluate(c, s, None)
    assert not result.ok
    assert result.checks["router"] is False and result.checks["facts"] is False
    assert result.checks["avoid_facts"] is False and result.checks["avoid_patterns"] is False
    assert result.checks["tools"] is False and result.checks["evidence_retained"] is False
    joined = " | ".join(result.notes)
    assert "routed to quest" in joined and "Monkey Madness I" in joined and "calc_flip" in joined


async def test_a_refusal_must_be_the_fixed_text_and_cost_no_model_call() -> None:
    c = case(intent_esperado="out_of_domain", behavior="refuse")
    good = state(intent="out_of_domain", answer=fixed("refusal", "en"), usage={"llm_calls": 0})
    assert (await evaluate(c, good, None)).ok
    spent = state(intent="out_of_domain", answer=fixed("refusal", "en"), usage={"llm_calls": 1})
    assert (await evaluate(c, spent, None)).checks["behavior"] is False
    wrote_code = state(intent="out_of_domain", answer="def f(): ...", usage={"llm_calls": 0})
    assert (await evaluate(c, wrote_code, None)).checks["behavior"] is False


async def test_the_fixed_no_evidence_text_counts_for_both_no_evidence_and_clarify() -> None:
    for behavior, corpus in (("no_evidence", "unanswerable"), ("clarify", "unanswerable")):
        c = case(behavior=behavior, corpus=corpus)
        s = state(answer=fixed("no_evidence", "en"))
        assert (await evaluate(c, s, None)).checks["behavior"] is True
    # an ordinary answer to an unanswerable question is a failure (and cannot be judged offline)
    c = case(behavior="no_evidence", corpus="unanswerable")
    assert (await evaluate(c, state(answer="It likes shrimp."), None)).checks["behavior"] is None


async def test_answering_no_evidence_to_an_answerable_question_is_a_failure() -> None:
    c = case(corpus="answerable", gold_pages=["X"], evidence=["e"])
    result = await evaluate(c, state(answer=fixed("no_evidence", "en")), None)
    assert result.checks["behavior"] is False
    assert any("answered 'no evidence'" in note for note in result.notes)


async def test_an_unanswerable_case_answered_normally_fails_the_false_answer_check() -> None:
    c = case(behavior="no_evidence", corpus="unanswerable")
    result = await evaluate(c, state(answer=fixed("no_evidence", "en")), None)
    assert result.checks["no_false_answer"] is True


def test_holds_answer_is_case_insensitive() -> None:
    c = case(corpus="answerable", gold_pages=["X"], evidence=["Attack Level Of 70"])
    assert holds_answer(c, {"content": "needs an attack level of 70"})
    assert not holds_answer(c, {"content": "needs 60 attack"})


# --------------------------------------------------------------------------- report and sweep
def graded(*rows: tuple[str, float, bool, bool]) -> list[dict[str, Any]]:
    return [
        {"page_title": t, "relevance": r, "kept": k, "content": "EVIDENCE" if h else "other"}
        for t, r, k, h in rows
    ]


async def results_for_report() -> list[CaseResult]:
    good = case(
        id="good",
        corpus="answerable",
        gold_pages=["X"],
        evidence=["EVIDENCE"],
        expected_facts=["70"],
    )
    sg = state(
        graded=graded(("X", 0.95, True, True), ("Y", 0.75, True, False), ("Z", 0.2, False, False)),
        sufficiency=0.9,
    )
    nothing = case(id="nothing", behavior="no_evidence", corpus="unanswerable")
    sn = state(
        answer=fixed("no_evidence", "en"), graded=graded(("Q", 0.3, False, False)), sufficiency=0.1
    )
    miss = case(id="miss", corpus="answerable", gold_pages=["X"], evidence=["EVIDENCE"])
    sm = state(
        graded=graded(("X", 0.55, False, True)), sufficiency=0.4, answer=fixed("no_evidence", "en")
    )
    out = []
    for c, s in ((good, sg), (nothing, sn), (miss, sm)):
        result = await evaluate(c, s, None)
        result.seconds = 3.0
        out.append(result)
    out.append(CaseResult(case(id="broken"), error="invalid_key: nope"))
    return out


async def test_the_report_covers_every_metric_of_the_spec() -> None:
    text = summarise(await results_for_report())
    for heading in (
        "ROUTER accuracy",
        "PASSAGE FILTERING",
        "NO EVIDENCE",
        "ANSWER CHECKS",
        "LATENCY AND COST",
    ):
        assert heading in text
    assert "cases run: 4 (1 failed with an error)" in text
    assert "recall: the passage with the answer is retained for 1/2 (50%)" in text
    assert "correct 'no evidence' rate (questions the wiki cannot answer): 1/1" in text
    assert "false 'no evidence' (answerable wiki questions): 1/2 (50%)" in text
    assert "FAILING CASES (2)" in text and "broken: invalid_key: nope" in text


async def test_a_sweep_re_judges_the_saved_grades_under_other_thresholds() -> None:
    report = to_json(await results_for_report(), {"rel_threshold": 0.7})
    text = sweep(report)
    lines = {
        line.split()[0]: line
        for line in text.splitlines()
        if line[:1] == " " and line.split()[0][0].isdigit()
    }
    # at 0.7 'good' keeps X and Y (evidence in X); 'miss' keeps nothing (0.55 < 0.7)
    rel = next(line for line in text.splitlines() if line.strip().startswith("0.7"))
    assert "1/2" in rel
    # at 0.5 'miss' keeps its evidence passage as well
    assert "2/2" in next(line for line in text.splitlines() if line.strip().startswith("0.5"))
    assert lines  # the tables were produced
    assert "SUFFICIENCY_THRESHOLD" in text


def test_the_saved_json_carries_what_the_sweep_needs() -> None:
    import asyncio

    report = to_json(asyncio.run(results_for_report()), {"rel_threshold": 0.7})
    first = report["cases"][0]
    assert first["graded"][0] == {
        "title": "X",
        "relevance": 0.95,
        "kept": True,
        "holds_answer": True,
    }
    assert first["sufficiency"] == 0.9 and report["settings"] == {"rel_threshold": 0.7}
    assert (
        report["cases"][-1]["error"] == "invalid_key: nope" and "graded" not in report["cases"][-1]
    )


# ------------------------------------------------------------------ the two precise "avoid" checks
def _patterns(case_id: str) -> list[str]:
    from eval.cases import load_cases

    return next(c for c in load_cases() if c.id == case_id).avoid_patterns


def _flagged(case_id: str, answer: str) -> bool:
    return any(re.search(p, answer) for p in _patterns(case_id))


def test_the_ge_tax_check_allows_the_old_rate_as_history_but_not_as_the_current_one() -> None:
    history = (
        "The tax was 1% until 29 May 2025, when it was increased to 2%. It is capped at 5 million."
    )
    assert not _flagged("wiki-ge-tax-en", history)
    assert not _flagged("wiki-ge-tax-en", "It was implemented at 1%, and is 2% now.")
    assert _flagged("wiki-ge-tax-en", "The Grand Exchange tax is 1% of the sale price.")
    assert _flagged("wiki-ge-tax-en", "Currently 1 % is charged.")
    assert _flagged("wiki-ge-tax-en", "You pay a 5% fee.")


def test_the_fake_pet_check_flags_a_rate_for_the_invented_pet_only() -> None:
    honest = (
        "The Zulrah drop table lists the Pet snakeling with a drop rate of 1/4,000. There is no "
        'pet called "Blorb" in the wiki data.'
    )
    assert not _flagged("none-fake-pet-en", honest)
    assert _flagged("none-fake-pet-en", "The Blorb pet drops at 1/5000 from Zulrah.")
    assert _flagged("none-fake-pet-en", "A rate of 1/2000 for the Blorb pet.")
