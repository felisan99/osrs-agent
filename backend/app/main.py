"""The HTTP API the RuneLite plugin talks to (SPEC §4): ``POST /v1/chat`` and ``GET /healthz``.

Security posture (SPEC §1.5, §11):

* The user's OpenRouter key arrives in ``X-OpenRouter-Key`` on every request, goes to the two
  clients that need it for that request, and is never stored, never put in the graph state and
  never logged: the request is registered with the log redactor, access logging is our own (method,
  path, status, time: no headers, no bodies) and every error message is scrubbed on its way out.
* The server holds no key of its own (the development key of the scripts is not even a setting).
* Sizes are capped before anything is parsed; unknown body fields are ignored; CORS is closed
  (the plugin is not a browser) and the docs endpoints are off; the server listens on loopback.
* Errors are typed (``401 invalid_key``, ``402 insufficient_credits``, ``429 rate_limited``,
  ``502 upstream_error``) so the plugin can tell the player what to do.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api.schemas import ChatDebug, ChatRequest, ChatResponse, PlayerState
from app.config import Settings, get_settings
from app.errors import AgentError, InvalidKeyError, RateLimitedError
from app.security import install_log_redaction, redact, request_secrets
from app.service import AgentService, TurnResult

logger = logging.getLogger("app.api")

KEY_HEADER = "x-openrouter-key"
ENVELOPE_BYTES = 8_192  # room around message + player_state (thread id, JSON punctuation)


class Service(Protocol):
    async def chat(
        self, *, thread_id: str, message: str, api_key: str, player_state: PlayerState | None
    ) -> TurnResult: ...

    async def health(self) -> dict[str, Any]: ...


ServiceFactory = Callable[[Settings], AbstractAsyncContextManager[Service]]


def error_response(
    status: int, code: str, message: str, *, retry_after: float | None = None
) -> JSONResponse:
    body: dict[str, Any] = {"error": code, "message": message}
    headers: dict[str, str] = {}
    if retry_after is not None:
        body["retry_after_s"] = retry_after
        headers["Retry-After"] = str(max(1, round(retry_after)))
    return JSONResponse(body, status_code=status, headers=headers)


class BodyTooLarge(Exception):
    pass


async def read_limited_body(request: Request, limit: int) -> bytes:
    """The request body, refusing early (by header) or while streaming (chunked) past ``limit``."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise BodyTooLarge
    received = bytearray()
    async for chunk in request.stream():
        received += chunk
        if len(received) > limit:
            raise BodyTooLarge
    return bytes(received)


def create_app(
    *, settings: Settings | None = None, service_factory: ServiceFactory | None = None
) -> FastAPI:
    settings = settings or get_settings()
    install_log_redaction()
    logging.getLogger().setLevel(settings.log_level)
    factory: ServiceFactory = service_factory or AgentService.open  # type: ignore[assignment]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with factory(settings) as service:
            app.state.service = service
            yield

    app = FastAPI(
        title="OSRS Agent",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    if settings.cors_allow_origins:  # closed by default: the plugin is not a browser
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=["POST", "GET"],
            allow_headers=["Content-Type", "X-OpenRouter-Key"],
        )
    body_limit = settings.max_message_chars * 4 + settings.max_player_state_bytes + ENVELOPE_BYTES

    @app.middleware("http")
    async def access_log(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        response = await call_next(request)
        # Method, path, status, time. Never headers or bodies: the key is in a header.
        logger.info(
            "%s %s -> %d (%.0f ms)",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    def typed_error_response(exc: AgentError) -> JSONResponse:
        retry = exc.retry_after if isinstance(exc, RateLimitedError) else None
        message = redact(exc.message)
        logger.warning("typed error %s: %s", exc.code, message)
        return error_response(exc.http_status, exc.code, message, retry_after=retry)

    @app.exception_handler(AgentError)
    async def typed_error(request: Request, exc: AgentError) -> JSONResponse:
        return typed_error_response(exc)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error on %s", request.url.path, exc_info=exc)  # redacting filter
        return error_response(500, "internal_error", "Unexpected error in the assistant.")

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        try:
            return JSONResponse(await app.state.service.health())
        except Exception:
            logger.exception("health check failed")
            return error_response(503, "unavailable", "The assistant's database is not reachable.")

    @app.post("/v1/chat")
    async def chat(request: Request) -> JSONResponse:
        key = (request.headers.get(KEY_HEADER) or "").strip()
        if not key:
            raise InvalidKeyError("Send your OpenRouter key in the X-OpenRouter-Key header.")
        with request_secrets(key):
            try:
                raw = await read_limited_body(request, body_limit)
            except BodyTooLarge:
                return error_response(413, "payload_too_large", "The request is too large.")
            try:
                payload = ChatRequest.model_validate_json(raw)
            except ValidationError as exc:
                first = exc.errors()[0] if exc.errors() else {}
                where = ".".join(str(part) for part in first.get("loc", ()))
                return error_response(
                    422, "invalid_request", f"Invalid request at '{where}': {first.get('msg', '')}"
                )
            if len(payload.message) > settings.max_message_chars:
                return error_response(
                    422,
                    "invalid_request",
                    f"The message is longer than {settings.max_message_chars} characters.",
                )
            try:
                result = await app.state.service.chat(
                    thread_id=payload.thread_id,
                    message=payload.message,
                    api_key=key,
                    player_state=payload.player_state,
                )
            except AgentError as exc:  # converted while the key is still registered for redaction
                return typed_error_response(exc)
            except Exception as exc:
                logger.error("unhandled error in the chat turn", exc_info=exc)  # redacting filter
                return error_response(500, "internal_error", "Unexpected error in the assistant.")
        response = ChatResponse(
            answer=result.answer,
            intent=result.intent,  # type: ignore[arg-type]
            sources=result.sources,  # type: ignore[arg-type]
            debug=(
                ChatDebug(
                    route_probs=result.route_probs,
                    retrieval_iterations=result.retrieval_iterations,
                    tools_called=result.tools_called,
                )
                if settings.debug_responses
                else None
            ),
        )
        return JSONResponse(response.model_dump(mode="json", exclude_none=True))

    return app


def get_app() -> FastAPI:
    """The real app, built when the server starts: ``uvicorn app.main:get_app --factory``."""
    return create_app()


def run() -> None:  # pragma: no cover - thin wrapper over uvicorn
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:get_app",
        factory=True,
        host=settings.host,  # 127.0.0.1: the plugin and the backend run on the same machine
        port=settings.port,
        log_config=None,  # our own access log, without headers
        access_log=False,
    )


if __name__ == "__main__":  # pragma: no cover
    run()
