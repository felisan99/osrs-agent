from __future__ import annotations

import json

import httpx
import pytest

from app.clients.http import RetryPolicy
from app.clients.openrouter import OpenRouterClient
from app.config import Settings
from app.errors import (
    InsufficientCreditsError,
    InvalidKeyError,
    RateLimitedError,
    UpstreamError,
)
from tests.conftest import FAKE_KEY, TEST_UA, RecordingSleep

URL = "https://openrouter.ai/api/v1/chat/completions"

OK_BODY = {
    "id": "gen-1",
    "model": "deepseek/deepseek-v4-flash",
    "choices": [
        {"finish_reason": "stop", "message": {"role": "assistant", "content": "Hola, aventurero."}}
    ],
    "usage": {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "cost": 0.0000421,
    },
}
TOOL_BODY = {
    "model": "deepseek/deepseek-v4-flash",
    "choices": [
        {
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_item_price",
                            "arguments": '{"item": "Abyssal whip"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "calc_flip", "arguments": "{not json"},
                    },
                ],
            },
        }
    ],
}
MESSAGES = [{"role": "user", "content": "hola"}]


def make(settings: Settings, http: httpx.AsyncClient, retry: RetryPolicy, key: str = FAKE_KEY):
    return OpenRouterClient(key, http=http, settings=settings, retry=retry)


async def test_sends_the_per_request_key_and_parses_content_usage_and_cost(
    respx_mock, settings, http, retry
) -> None:
    route = respx_mock.post(URL).mock(return_value=httpx.Response(200, json=OK_BODY))
    result = await make(settings, http, retry).chat(MESSAGES)

    request = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(request.content)
    assert body["model"] == settings.reasoning_model and body["messages"] == MESSAGES
    assert "provider" not in body and "tools" not in body  # plain chat: no routing constraints
    # Hidden "thinking" tokens are off and the budget is generous, so the visible answer can't be
    # starved (seen live: content="" with finish_reason=length at max_tokens=200).
    assert body["reasoning"] == {"effort": "none"} and body["max_tokens"] == settings.llm_max_tokens
    assert result.content == "Hola, aventurero."
    assert result.usage.prompt_tokens == 120 and result.usage.cost == pytest.approx(0.0000421)
    assert result.finish_reason == "stop"


async def test_shared_http_client_carries_no_credentials(respx_mock, settings, http, retry) -> None:
    respx_mock.post(URL).mock(return_value=httpx.Response(200, json=OK_BODY))
    await make(settings, http, retry).chat(MESSAGES)
    # The key must only be attached per request, never baked into the shared client.
    assert "authorization" not in {k.lower() for k in http.headers}


async def test_tools_require_provider_support_and_tool_calls_are_parsed(
    respx_mock, settings, http, retry
) -> None:
    route = respx_mock.post(URL).mock(return_value=httpx.Response(200, json=TOOL_BODY))
    tools = [
        {
            "type": "function",
            "function": {"name": "get_item_price", "parameters": {"type": "object"}},
        }
    ]
    result = await make(settings, http, retry).chat(MESSAGES, tools=tools)

    body = json.loads(route.calls.last.request.content)
    assert body["tools"] == tools and body["tool_choice"] == "auto"
    assert body["provider"] == {"require_parameters": True}

    good, bad = result.tool_calls
    assert (good.name, good.arguments, good.parse_error) == (
        "get_item_price",
        {"item": "Abyssal whip"},
        None,
    )
    assert bad.arguments == {} and bad.parse_error and "invalid JSON" in bad.parse_error
    # Ready to be appended to the conversation for the next round of a tool loop.
    assert result.assistant_message["tool_calls"][0]["id"] == "call_1"


@pytest.mark.parametrize(
    ("status", "exc"),
    [(401, InvalidKeyError), (402, InsufficientCreditsError)],
)
async def test_auth_and_credit_errors_are_typed_and_never_retried(
    respx_mock, settings, http, retry, sleeper: RecordingSleep, status, exc
) -> None:
    route = respx_mock.post(URL).mock(
        return_value=httpx.Response(status, json={"error": {"code": status, "message": "nope"}})
    )
    with pytest.raises(exc, match="nope") as err:
        await make(settings, http, retry).chat(MESSAGES)
    assert err.value.http_status == status
    assert route.call_count == 1 and sleeper.delays == []


async def test_429_with_long_retry_after_is_not_waited_out(
    respx_mock, settings, http, retry, sleeper: RecordingSleep
) -> None:
    route = respx_mock.post(URL).mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "60"},
            json={"error": {"code": 429, "message": "slow down"}},
        )
    )
    with pytest.raises(RateLimitedError) as err:
        await make(settings, http, retry).chat(MESSAGES)
    assert err.value.retry_after == 60 and err.value.http_status == 429
    assert route.call_count == 1  # 60 s > max_delay: surface it instead of blocking the chat


async def test_429_with_short_retry_after_is_retried(
    respx_mock, settings, http, retry, sleeper: RecordingSleep
) -> None:
    route = respx_mock.post(URL).mock(
        side_effect=[
            httpx.Response(
                429, headers={"Retry-After": "2"}, json={"error": {"code": 429, "message": "x"}}
            ),
            httpx.Response(200, json=OK_BODY),
        ]
    )
    result = await make(settings, http, retry).chat(MESSAGES)
    assert result.content and route.call_count == 2 and sleeper.delays == [2.0]


async def test_bad_gateway_is_retried_then_reported_as_upstream_error(
    respx_mock, settings, http, retry
) -> None:
    route = respx_mock.post(URL).mock(
        return_value=httpx.Response(
            502, json={"error": {"code": 502, "message": "Provider returned error"}}
        )
    )
    with pytest.raises(UpstreamError, match="Provider returned error") as err:
        await make(settings, http, retry).chat(MESSAGES)
    assert err.value.http_status == 502 and err.value.upstream_status == 502
    assert route.call_count == 3


@pytest.mark.parametrize(
    ("code", "exc"),
    [
        (402, InsufficientCreditsError),
        (401, InvalidKeyError),
        (429, RateLimitedError),
        (502, UpstreamError),
    ],
)
async def test_error_delivered_with_http_200_is_mapped_by_its_code(
    respx_mock, settings, http, retry, code, exc
) -> None:
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, json={"error": {"code": code, "message": "in body"}})
    )
    with pytest.raises(exc):
        await make(settings, http, retry).chat(MESSAGES)


async def test_generation_error_inside_a_choice_is_an_upstream_error(
    respx_mock, settings, http, retry
) -> None:
    body = {
        "choices": [
            {
                "finish_reason": "error",
                "message": {"content": "partial"},
                "error": {"code": 502, "message": "stream died"},
            }
        ]
    }
    respx_mock.post(URL).mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(UpstreamError, match="stream died"):
        await make(settings, http, retry).chat(MESSAGES)


async def test_network_failure_is_an_upstream_error(respx_mock, settings, http, retry) -> None:
    respx_mock.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(UpstreamError, match="network error"):
        await make(settings, http, retry).chat(MESSAGES)


async def test_key_never_leaks_through_errors_repr_or_str(
    respx_mock, settings, http, retry
) -> None:
    # Worst case: an upstream error message that echoes the credential.
    respx_mock.post(URL).mock(
        return_value=httpx.Response(
            401, json={"error": {"code": 401, "message": f"Invalid key {FAKE_KEY} for user"}}
        )
    )
    client = make(settings, http, retry)
    assert FAKE_KEY not in repr(client) and FAKE_KEY not in str(client)
    with pytest.raises(InvalidKeyError) as err:
        await client.chat(MESSAGES)
    assert FAKE_KEY not in str(err.value) and FAKE_KEY not in err.value.message
    assert "[REDACTED]" in err.value.message


async def test_missing_key_is_rejected_before_any_request(
    respx_mock, settings, http, retry
) -> None:
    route = respx_mock.post(URL)
    for blank in ("", "   "):
        with pytest.raises(InvalidKeyError):
            make(settings, http, retry, key=blank)
    assert route.call_count == 0


async def test_reasoning_control_is_configurable_and_overridable(respx_mock, http, retry) -> None:
    route = respx_mock.post(URL).mock(return_value=httpx.Response(200, json=OK_BODY))

    keep = Settings(_env_file=None, prices_user_agent=TEST_UA, reasoning_effort="")
    await make(keep, http, retry).chat(MESSAGES)
    assert "reasoning" not in json.loads(route.calls.last.request.content)  # provider default

    low = Settings(_env_file=None, prices_user_agent=TEST_UA, reasoning_effort="low")
    await make(low, http, retry).chat(MESSAGES, max_tokens=50, extra={"temperature": 0.9})
    body = json.loads(route.calls.last.request.content)
    assert body["reasoning"] == {"effort": "low"} and body["max_tokens"] == 50
    assert body["temperature"] == 0.9  # `extra` has the last word
