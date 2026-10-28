from __future__ import annotations

import json

import httpx
import pytest

from app.clients.jev import JevClient
from app.config import Settings
from app.rag.grade import MAX_PASSAGE_CHARS, Grader, plan_batches
from app.rag.retrieve import Passage
from tests.conftest import FAKE_KEY, TEST_UA

URL = "https://openrouter.ai/api/alpha/decisions"


def passage(i: int, text: str | None = None) -> Passage:
    return Passage(
        id=i,
        page_title=f"Page {i}",
        section="",
        page_type="other",
        url=f"https://x/{i}",
        content=text or f"Passage {i} says something specific about topic {i}.",
        metadata={},
        score=0.1,
    )


def reply(
    relevance: dict[str, float], sufficiency: float | None = None, cost: float = 0.00001
) -> dict:
    answers = {k: {"type": "noul", "noul": v} for k, v in relevance.items()}
    if sufficiency is not None:
        answers["sufficient"] = {"type": "noul", "noul": sufficiency}
    return {
        "model": "typesafe/jev-1.13-20260917",
        "usage": {"input_tokens": 5, "output_tokens": 1, "cost": cost},
        "answers": answers,
    }


def grader(http, retry, judgments=None, **settings) -> Grader:
    s = Settings(_env_file=None, prices_user_agent=TEST_UA, **settings)
    return Grader(JevClient(FAKE_KEY, http=http, settings=s, retry=retry), s, judgments)


class Recorder:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, dict]] = []

    async def record(self, kind: str, input_ref: str, probs: dict[str, float]) -> None:
        self.rows.append((kind, input_ref, probs))


# --------------------------------------------------------------------------- one call
async def test_one_batch_call_with_flat_passage_fields_and_a_sufficiency_question(
    respx_mock, http, retry
) -> None:
    route = respx_mock.post(URL).mock(
        return_value=httpx.Response(
            200, json=reply({"passage_01": 0.95, "passage_02": 0.1, "passage_03": 0.7}, 0.8)
        )
    )
    await grader(http, retry).grade("how do I train prayer", [passage(1), passage(2), passage(3)])

    assert route.call_count == 1
    body = json.loads(route.calls.last.request.content)
    assert body["state"] == {
        "query": "how do I train prayer",
        "passage_01": passage(1).content,
        "passage_02": passage(2).content,
        "passage_03": passage(3).content,
    }
    assert list(body["questions"]) == ["passage_01", "passage_02", "passage_03", "sufficient"]
    first = body["questions"]["passage_01"]
    assert (
        first["type"] == "noul"
        and "`passage_01`" in first["instructions"]
        and "`query`" in first["instructions"]
    )
    assert set(first["criteria"]) == {"true", "false"}
    suff = body["questions"]["sufficient"]["instructions"]
    assert "`passage_01` to `passage_03`" in suff and "Spanish" in suff


async def test_thresholds_decide_which_passages_are_kept_and_whether_the_set_suffices(
    respx_mock, http, retry
) -> None:
    respx_mock.post(URL).mock(
        return_value=httpx.Response(
            200,
            json=reply({"passage_01": 0.95, "passage_02": 0.1, "passage_03": 0.7}, 0.8, 0.00002),
        )
    )
    result = await grader(http, retry).grade("q", [passage(1), passage(2), passage(3)])

    assert [(g.passage.id, g.relevance, g.kept) for g in result.graded] == [
        (1, 0.95, True),
        (2, 0.1, False),
        (3, 0.7, True),
    ]
    assert [g.passage.id for g in result.kept] == [1, 3]  # most relevant first
    assert result.sufficiency == 0.8 and result.sufficient is True
    assert result.cost_usd == pytest.approx(0.00002) and result.calls == 1
    assert result.model == "typesafe/jev-1.13-20260917"


async def test_thresholds_are_parameters_not_constants(respx_mock, http, retry) -> None:
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, json=reply({"passage_01": 0.6, "passage_02": 0.75}, 0.45))
    )
    strict = await grader(http, retry).grade("q", [passage(1), passage(2)])  # defaults: 0.70 / 0.50
    assert [g.kept for g in strict.graded] == [False, True] and strict.sufficient is False
    loose = await grader(http, retry).grade(
        "q", [passage(1), passage(2)], rel_threshold=0.5, sufficiency_threshold=0.4
    )
    assert [g.kept for g in loose.graded] == [True, True] and loose.sufficient is True


async def test_nothing_retained_means_not_sufficient_whatever_jev_says(
    respx_mock, http, retry
) -> None:
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, json=reply({"passage_01": 0.2, "passage_02": 0.3}, 0.99))
    )
    result = await grader(http, retry).grade("q", [passage(1), passage(2)])
    assert result.kept == [] and result.sufficiency == 0.99 and result.sufficient is False


async def test_no_candidates_means_no_call(respx_mock, http, retry) -> None:
    route = respx_mock.post(URL)
    result = await grader(http, retry).grade("q", [])
    assert (
        route.call_count == 0
        and result.graded == []
        and not result.sufficient
        and result.calls == 0
    )


async def test_very_long_passages_are_clipped_in_the_state(respx_mock, http, retry) -> None:
    route = respx_mock.post(URL).mock(
        return_value=httpx.Response(200, json=reply({"passage_01": 0.9}, 0.9))
    )
    await grader(http, retry).grade("q", [passage(1, "word " * 5_000)])
    state = json.loads(route.calls.last.request.content)["state"]
    assert len(state["passage_01"]) <= MAX_PASSAGE_CHARS + 4 and state["passage_01"].endswith("...")


# --------------------------------------------------------------------------- batching
def test_plan_batches_respects_the_budget_and_keeps_order() -> None:
    passages = [passage(i, "x" * 2_000) for i in range(1, 8)]  # ~667 tokens each at 3 chars/token
    batches = plan_batches(passages, budget_tokens=3_000, chars_per_token=3.0)
    assert [p.id for b in batches for p in b] == list(range(1, 8))
    assert len(batches) >= 2 and all(len(b) >= 1 for b in batches)
    assert plan_batches([], budget_tokens=3_000, chars_per_token=3.0) == []
    one = plan_batches(passages, budget_tokens=24_000, chars_per_token=3.0)
    assert len(one) == 1  # the normal case: 7 chunks fit one call


async def test_candidates_that_do_not_fit_are_graded_in_batches_then_sufficiency_is_asked_once(
    respx_mock, http, retry
) -> None:
    relevance_by_id = {1: 0.80, 2: 0.99, 3: 0.75, 4: 0.10}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        answers = {}
        for field, text in body["state"].items():
            if field.startswith("passage_"):
                answers[field] = {"type": "noul", "noul": relevance_by_id[int(text.split()[1])]}
        if "sufficient" in body["questions"]:
            answers["sufficient"] = {"type": "noul", "noul": 0.9}
        usage = {"input_tokens": 1, "output_tokens": 1}
        return httpx.Response(200, json={"model": "m", "usage": usage, "answers": answers})

    route = respx_mock.post(URL).mock(side_effect=handler)
    passages = [passage(i, f"Passage {i} " + "x" * 2_300) for i in range(1, 5)]
    # ~890 estimated tokens per passage: a 2,500-token budget fits two per call.
    result = await grader(http, retry, jev_max_state_tokens=2_500).grade("q", passages)

    bodies = [json.loads(c.request.content) for c in route.calls]
    relevance_calls = [b for b in bodies if "sufficient" not in b["questions"]]
    sufficiency_calls = [b for b in bodies if "sufficient" in b["questions"]]
    assert len(relevance_calls) == 2 and len(sufficiency_calls) == 1
    sizes = [len([k for k in b["state"] if k.startswith("passage_")]) for b in relevance_calls]
    assert sizes == [2, 2]
    assert result.calls == 3
    assert [g.passage.id for g in result.kept] == [2, 1, 3]  # most relevant first, 4 dropped
    # Sufficiency is judged over the most relevant retained passages that fit the budget.
    state = sufficiency_calls[0]["state"]
    assert [t.split()[1] for k, t in state.items() if k.startswith("passage_")] == ["2", "1"]
    assert result.sufficiency == 0.9 and result.sufficient is True


# --------------------------------------------------------------------------- judgments
async def test_raw_probabilities_are_recorded_for_threshold_tuning(respx_mock, http, retry) -> None:
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, json=reply({"passage_01": 0.9, "passage_02": 0.3}, 0.6))
    )
    log = Recorder()
    await grader(http, retry, judgments=log).grade(
        "q", [passage(11), passage(12)], query_ref="eval-07"
    )
    assert log.rows == [
        ("relevance", "eval-07:11", {"relevance": 0.9}),
        ("relevance", "eval-07:12", {"relevance": 0.3}),
        ("sufficiency", "eval-07", {"sufficiency": 0.6}),
    ]
