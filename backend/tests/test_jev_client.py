from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from app.clients.http import RetryPolicy
from app.clients.jev import (
    ChoiceAnswer,
    JevClient,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    choice,
    estimate_tokens,
    noul,
    score,
)
from app.config import Settings
from app.errors import (
    InsufficientCreditsError,
    InvalidKeyError,
    RateLimitedError,
    StateTooLargeError,
    UpstreamError,
)
from tests.conftest import FAKE_KEY

URL = "https://openrouter.ai/api/alpha/decisions"

# Response shape per OpenRouter's Decisions API reference (envelope + discriminated answers).
RESPONSE = {
    "id": "gen-dec-123",
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992},
    "answers": {
        "is_bug": {"type": "noul", "noul": 0.96},
        "team": {
            "type": "choice",
            "choice": "payments",
            "probabilities": {"payments": 0.78, "frontend": 0.22, "account": 0.0},
            "confidence": 0.67,
        },
        "urgency": {
            "type": "score",
            "score": 1.99,
            "legend": {"0": "low", "1": "medium", "2": "high"},
            "probabilities": {"0": 0.0, "1": 0.01, "2": 0.99},
            "confidence": 0.99,
        },
    },
}


def make(settings: Settings, http: httpx.AsyncClient, retry: RetryPolicy) -> JevClient:
    return JevClient(FAKE_KEY, http=http, settings=settings, retry=retry)


def questions():
    return {
        "is_bug": noul("Is `ticket` a software defect?", yes="It is a defect", no="It is not"),
        "team": choice(
            "Which team owns `ticket`?",
            {"payments": "billing", "frontend": None, "account": "login"},
        ),
        "urgency": score("How urgent is `ticket`?", ["low", "medium", "high"]),
    }


async def test_request_body_matches_the_decisions_api_shape(
    respx_mock, settings, http, retry
) -> None:
    route = respx_mock.post(URL).mock(return_value=httpx.Response(200, json=RESPONSE))
    state = {"ticket": "Charged twice", "customer_tier": "pro"}

    await make(settings, http, retry).decide(state, questions())

    request = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(request.content)
    assert body["model"] == "typesafe/jev-1.13"  # pinned, not the moving ~jev-latest alias
    assert body["state"] == state
    assert body["questions"]["is_bug"] == {
        "type": "noul",
        "instructions": "Is `ticket` a software defect?",
        # OpenRouter's reference marks BOTH keys required (TypeSafe's own docs say optional).
        "criteria": {"true": "It is a defect", "false": "It is not"},
    }
    assert body["questions"]["team"]["criteria"] == {
        "payments": "billing",
        "frontend": None,
        "account": "login",
    }
    assert body["questions"]["urgency"]["criteria"] == ["low", "medium", "high"]


async def test_parses_all_three_answer_types_usage_and_serving_model(
    respx_mock, settings, http, retry
) -> None:
    respx_mock.post(URL).mock(return_value=httpx.Response(200, json=RESPONSE))
    decision = await make(settings, http, retry).decide({"ticket": "x"}, questions())

    assert decision.get("is_bug", NoulAnswer).noul == pytest.approx(0.96)
    team = decision.get("team", ChoiceAnswer)
    assert team.choice == "payments" and team.probabilities["frontend"] == pytest.approx(0.22)
    assert team.confidence == pytest.approx(0.67)
    urgency = decision.get("urgency", ScoreAnswer)
    assert urgency.score == pytest.approx(1.99) and urgency.legend["2"] == "high"
    assert decision.model == "typesafe/jev-1.13-20260917"  # log it: aliases move under you
    assert decision.usage.cost == pytest.approx(0.000019992) and decision.usage.input_tokens == 476


async def test_get_with_wrong_kind_or_missing_id_is_an_upstream_error(
    respx_mock, settings, http, retry
) -> None:
    respx_mock.post(URL).mock(return_value=httpx.Response(200, json=RESPONSE))
    decision = await make(settings, http, retry).decide({"ticket": "x"}, questions())
    with pytest.raises(UpstreamError, match="not a ChoiceAnswer"):
        decision.get("is_bug", ChoiceAnswer)
    with pytest.raises(UpstreamError, match="missing"):
        decision.get("nope", NoulAnswer)


async def test_choose_convenience_for_the_router(respx_mock, settings, http, retry) -> None:
    route = respx_mock.post(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "m",
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "answers": {
                    "choice": {
                        "type": "choice",
                        "choice": "quest",
                        "probabilities": {"quest": 0.9, "wiki_general": 0.1},
                        "confidence": 0.8,
                    }
                },
            },
        )
    )
    answer, decision = await make(settings, http, retry).choose(
        {"user_message": "x"}, "Classify `user_message`", {"quest": "q", "wiki_general": "w"}
    )
    assert answer.choice == "quest" and decision.usage.input_tokens == 1
    assert list(json.loads(route.calls.last.request.content)["questions"]) == ["choice"]


def test_question_validation_follows_documented_limits() -> None:
    with pytest.raises(ValidationError, match='"true" and "false"'):
        NoulQuestion(instructions="q", criteria={"true": "y"})
    with pytest.raises(ValidationError, match=r"2\.\.255 options"):
        choice("q", {"only": None})
    with pytest.raises(ValidationError, match=r"2\.\.255 options"):
        choice("q", {f"o{i}": None for i in range(256)})
    choice("q", {f"o{i}": None for i in range(255)})  # the maximum is allowed
    with pytest.raises(ValidationError, match=r"2\.\.10 levels"):
        score("q", ["a"])
    with pytest.raises(ValidationError, match=r"2\.\.10 levels"):
        score("q", [str(i) for i in range(11)])
    score("q", [str(i) for i in range(10)])


async def test_oversized_state_is_rejected_before_any_http_call(
    respx_mock, settings, http, retry
) -> None:
    route = respx_mock.post(URL)
    huge = {"passage": "word " * 100_000}  # ~167K tokens by our conservative estimate
    with pytest.raises(StateTooLargeError, match="split into smaller batches"):
        await make(settings, http, retry).decide(huge, {"q": noul("q?", yes="y", no="n")})
    assert route.call_count == 0


def test_token_estimate_is_conservative_and_handles_json_and_unicode() -> None:
    assert estimate_tokens("a" * 300, 3.0) == 100
    assert estimate_tokens({"k": "ñ" * 30}) > 10  # JSON-serialised, no ascii escaping blow-up


@pytest.mark.parametrize(
    ("status", "exc"),
    [(401, InvalidKeyError), (402, InsufficientCreditsError)],
)
async def test_typed_errors_for_key_and_credits(
    respx_mock, settings, http, retry, status, exc
) -> None:
    respx_mock.post(URL).mock(
        return_value=httpx.Response(status, json={"error": {"code": status, "message": "denied"}})
    )
    with pytest.raises(exc, match="denied"):
        await make(settings, http, retry).decide({"a": 1}, {"q": noul("q?", yes="y", no="n")})


@pytest.mark.parametrize("status", [429, 529])
async def test_rate_limit_and_overload_are_retried_then_reported(
    respx_mock, settings, http, retry, status
) -> None:
    route = respx_mock.post(URL).mock(
        return_value=httpx.Response(status, json={"error": {"code": status, "message": "busy"}})
    )
    expected = RateLimitedError if status == 429 else UpstreamError
    with pytest.raises(expected):
        await make(settings, http, retry).decide({"a": 1}, {"q": noul("q?", yes="y", no="n")})
    assert route.call_count == 3


async def test_http_200_with_error_body_is_mapped(respx_mock, settings, http, retry) -> None:
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, json={"error": {"code": 402, "message": "no credits"}})
    )
    with pytest.raises(InsufficientCreditsError):
        await make(settings, http, retry).decide({"a": 1}, {"q": noul("q?", yes="y", no="n")})


async def test_malformed_answers_do_not_echo_payload_or_key(
    respx_mock, settings, http, retry
) -> None:
    secret_ish = f"state echo with {FAKE_KEY}"
    respx_mock.post(URL).mock(
        return_value=httpx.Response(
            200,
            json={"model": "m", "answers": {"q": {"type": "noul", "noul": 7, "note": secret_ish}}},
        )
    )
    with pytest.raises(UpstreamError, match="Unexpected Jev response shape") as err:
        await make(settings, http, retry).decide({"a": 1}, {"q": noul("q?", yes="y", no="n")})
    assert FAKE_KEY not in err.value.message and "state echo" not in err.value.message


def test_missing_key_and_repr_do_not_leak(settings, retry) -> None:
    with pytest.raises(InvalidKeyError):
        JevClient("  ", http=httpx.AsyncClient(), settings=settings, retry=retry)
    client = JevClient(FAKE_KEY, http=httpx.AsyncClient(), settings=settings, retry=retry)
    assert FAKE_KEY not in repr(client)
