"""The HTTP contract (SPEC §4) and the key-handling rules (SPEC §1.5, §11), with a fake service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from app.api.schemas import PlayerState
from app.config import Settings
from app.errors import (
    InsufficientCreditsError,
    InvalidKeyError,
    RateLimitedError,
    UpstreamError,
)
from app.main import create_app
from app.service import TurnResult

KEY = "sk-or-v1-APITESTKEY0123456789abcdefghij"
THREAD = "3f2b8c1e-aaaa-bbbb-cccc-0123456789ab"


class FakeService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.result = TurnResult(
            answer="Necesitás 70 de Attack [Abyssal whip].",
            intent="wiki_general",
            sources=[
                {"title": "Abyssal whip", "url": "https://oldschool.runescape.wiki/w/Abyssal_whip"}
            ],
            route_probs={"wiki_general": 0.97},
            retrieval_iterations=1,
            tools_called=["calc_flip"],
        )
        self.healthy = True

    async def chat(self, *, thread_id, message, api_key, player_state) -> TurnResult:
        self.calls.append(
            {"thread_id": thread_id, "message": message, "key": api_key, "player": player_state}
        )
        if self.error is not None:
            raise self.error
        return self.result

    async def health(self) -> dict[str, Any]:
        if not self.healthy:
            raise RuntimeError("db down")
        return {"status": "ok", "documents": 31374, "guides": 637, "quests": 220}


def settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None, prices_user_agent="osrs-agent-tests - tests@example.com", **overrides
    )


@asynccontextmanager
async def client_for(service: FakeService, **overrides: Any):
    @asynccontextmanager
    async def factory(_: Settings):
        yield service

    app = create_app(settings=settings(**overrides), service_factory=factory)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


def body(**overrides: Any) -> dict[str, Any]:
    return {
        "thread_id": THREAD,
        "message": "¿Qué nivel de Attack necesito para la whip?",
    } | overrides


async def post(client: httpx.AsyncClient, payload: Any = None, *, key: str | None = KEY):
    headers = {"X-OpenRouter-Key": key} if key is not None else {}
    return await client.post(
        "/v1/chat", json=body() if payload is None else payload, headers=headers
    )


# --------------------------------------------------------------------------- the happy path
async def test_a_chat_turn_returns_the_contract_shape() -> None:
    service = FakeService()
    async with client_for(service) as client:
        response = await post(client)
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Necesitás 70 de Attack [Abyssal whip].",
        "intent": "wiki_general",
        "sources": [
            {"title": "Abyssal whip", "url": "https://oldschool.runescape.wiki/w/Abyssal_whip"}
        ],
    }  # no `debug` unless DEBUG_RESPONSES is on (SPEC §4)
    (call,) = service.calls
    assert call["key"] == KEY and call["thread_id"] == THREAD and call["player"] is None


async def test_debug_is_only_returned_when_enabled() -> None:
    async with client_for(FakeService(), debug_responses=True) as client:
        data = (await post(client)).json()
    assert data["debug"] == {
        "route_probs": {"wiki_general": 0.97},
        "retrieval_iterations": 1,
        "tools_called": ["calc_flip"],
    }


async def test_the_player_snapshot_reaches_the_service_validated() -> None:
    service = FakeService()
    snapshot = {
        "account_type": "ironman",
        "members": True,
        "skills": {"Attack": 70, "OVERALL": 1500},
        "quests": {"Cook's Assistant": "FINISHED"},
        "gp": 9_360_000_000,
        "unknown_field": "ignored",
    }
    async with client_for(service) as client:
        assert (await post(client, body(player_state=snapshot))).status_code == 200
    player = service.calls[0]["player"]
    assert isinstance(player, PlayerState) and player.is_ironman
    assert player.skills == {"attack": 70} and player.gp == 9_360_000_000


async def test_the_key_header_is_trimmed_and_the_body_cannot_carry_a_key() -> None:
    service = FakeService()
    async with client_for(service) as client:
        response = await client.post(
            "/v1/chat",
            json=body(openrouter_key="sk-or-v1-FROMTHEBODYxxxxxxxxxxxxxxxxx"),
            headers={"X-OpenRouter-Key": f"  {KEY}  "},
        )
    assert response.status_code == 200 and service.calls[0]["key"] == KEY


# --------------------------------------------------------------------------- typed errors
@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (InvalidKeyError("OpenRouter rejected the API key"), 401, "invalid_key"),
        (InsufficientCreditsError("Insufficient credits"), 402, "insufficient_credits"),
        (UpstreamError("OpenRouter chat returned no choices"), 502, "upstream_error"),
    ],
)
async def test_upstream_failures_become_the_typed_errors_of_the_contract(
    error: Exception, status: int, code: str
) -> None:
    service = FakeService()
    service.error = error
    async with client_for(service) as client:
        response = await post(client)
    assert response.status_code == status
    assert response.json() == {"error": code, "message": str(error)}


async def test_rate_limits_tell_the_plugin_when_to_come_back() -> None:
    service = FakeService()
    service.error = RateLimitedError("Too many requests", retry_after=12.0)
    async with client_for(service) as client:
        response = await post(client)
    assert response.status_code == 429 and response.headers["retry-after"] == "12"
    assert response.json() == {
        "error": "rate_limited",
        "message": "Too many requests",
        "retry_after_s": 12.0,
    }


async def test_a_missing_or_blank_key_is_an_invalid_key_not_a_server_error() -> None:
    service = FakeService()
    async with client_for(service) as client:
        missing = await post(client, key=None)
        blank = await post(client, key="   ")
    for response in (missing, blank):
        assert response.status_code == 401 and response.json()["error"] == "invalid_key"
    assert service.calls == []


async def test_unexpected_failures_are_a_generic_500_that_leaks_nothing() -> None:
    service = FakeService()
    service.error = RuntimeError(f"database exploded while using {KEY}")
    async with client_for(service) as client:
        response = await post(client)
    assert response.status_code == 500
    assert response.json() == {
        "error": "internal_error",
        "message": "Unexpected error in the assistant.",
    }


# --------------------------------------------------------------------------- validation and limits
@pytest.mark.parametrize(
    ("payload", "where"),
    [
        (body(thread_id="short"), "thread_id"),
        (body(thread_id="has spaces in it!"), "thread_id"),
        (body(message=""), "message"),
        ({"thread_id": THREAD}, "message"),
        (body(player_state={"skills": {"attack": 999}}), "player_state"),
        (body(player_state={"quests": {"X": "DONE"}}), "player_state"),
    ],
)
async def test_invalid_requests_get_a_typed_422_naming_the_field(payload: dict, where: str) -> None:
    async with client_for(FakeService()) as client:
        response = await post(client, payload)
    assert response.status_code == 422 and response.json()["error"] == "invalid_request"
    assert where in response.json()["message"]


async def test_malformed_json_is_a_422_too() -> None:
    async with client_for(FakeService()) as client:
        response = await client.post(
            "/v1/chat", content=b"{not json", headers={"X-OpenRouter-Key": KEY}
        )
    assert response.status_code == 422 and response.json()["error"] == "invalid_request"


async def test_a_message_over_the_limit_is_refused_before_the_model_sees_it() -> None:
    service = FakeService()
    async with client_for(service, max_message_chars=100) as client:
        response = await post(client, body(message="x" * 101))
        fine = await post(client, body(message="x" * 100))
    assert response.status_code == 422 and "100 characters" in response.json()["message"]
    assert fine.status_code == 200 and len(service.calls) == 1


async def test_an_oversized_body_is_refused_with_413_by_header_and_while_streaming() -> None:
    service = FakeService()
    async with client_for(service, max_message_chars=100, max_player_state_bytes=1_000) as client:
        huge = body(player_state={"quests": {f"quest {i}": "FINISHED" for i in range(2_000)}})
        by_header = await post(client, huge)

        async def chunks():
            for _ in range(40):
                yield b" " * 1_000  # no Content-Length: the limit must hold while streaming

        streamed = await client.post(
            "/v1/chat", content=chunks(), headers={"X-OpenRouter-Key": KEY}
        )
    assert by_header.status_code == 413 and by_header.json()["error"] == "payload_too_large"
    assert streamed.status_code == 413
    assert service.calls == []


# --------------------------------------------------------------------------- the key and the logs
async def test_the_key_appears_in_no_log_not_even_when_something_tries_to_print_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    class Leaky(FakeService):
        async def chat(self, *, thread_id, message, api_key, player_state) -> TurnResult:
            logging.getLogger("app.test").error("calling upstream with %s", api_key)
            logging.getLogger("app.test").info("headers: Authorization: Bearer %s", api_key)
            raise UpstreamError(f"upstream said: bad key {api_key}")

    async with client_for(Leaky()) as client:
        response = await post(client)
        await post(client, body(thread_id="short"))  # a validation error path as well
        await client.get("/healthz")
    assert response.status_code == 502
    assert KEY not in response.text and "APITESTKEY" not in response.text
    assert KEY not in caplog.text and "APITESTKEY" not in caplog.text
    assert "[REDACTED]" in caplog.text  # the leak attempts were scrubbed, not silently dropped
    assert any("POST /v1/chat -> 502" in r.getMessage() for r in caplog.records)  # access log works


async def test_the_unexpected_error_traceback_is_redacted_too(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    service = FakeService()
    service.error = RuntimeError(f"boom with {KEY} inside")
    async with client_for(service) as client:
        await post(client)
    assert KEY not in caplog.text and "APITESTKEY" not in caplog.text
    assert "unhandled error" in caplog.text


# --------------------------------------------------------------------------- surface
async def test_healthz_reports_the_corpus_or_a_503() -> None:
    service = FakeService()
    async with client_for(service) as client:
        ok = await client.get("/healthz")
        service.healthy = False
        down = await client.get("/healthz")
    assert ok.status_code == 200 and ok.json()["documents"] == 31374
    assert down.status_code == 503 and down.json()["error"] == "unavailable"


async def test_cors_is_closed_and_the_docs_are_off() -> None:
    async with client_for(FakeService()) as client:
        response = await client.options(
            "/v1/chat",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        simple = await client.get("/healthz", headers={"Origin": "https://evil.example"})
        docs = await client.get("/docs")
        schema = await client.get("/openapi.json")
    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-origin" not in simple.headers
    assert docs.status_code == 404 and schema.status_code == 404


async def test_cors_can_be_opened_for_listed_origins_only() -> None:
    async with client_for(FakeService(), cors_allow_origins=["http://localhost:3000"]) as client:
        allowed = await client.get("/healthz", headers={"Origin": "http://localhost:3000"})
        other = await client.get("/healthz", headers={"Origin": "https://evil.example"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in other.headers


async def test_the_server_defaults_to_loopback() -> None:
    assert settings().host == "127.0.0.1"
