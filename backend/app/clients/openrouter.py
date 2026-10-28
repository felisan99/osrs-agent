"""OpenRouter chat completions with the user's key supplied per request (BYOK).

The ``httpx.AsyncClient`` is shared application-wide (connection pooling) and carries NO
credentials; an ``OpenRouterClient`` instance is created per request and holds the key only
for that request's lifetime. The key is never written to disk, never logged, and is scrubbed
from every upstream error message before it becomes one of our exceptions.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, Field, SecretStr

from app.clients.http import RetryPolicy, parse_retry_after, safe_json, send_with_retries
from app.config import Settings, get_settings
from app.errors import (
    AgentError,
    InsufficientCreditsError,
    InvalidKeyError,
    RateLimitedError,
    UpstreamError,
)
from app.security import redact


def error_message(payload: Any) -> str:
    """Best-effort human message from an OpenRouter error body ``{"error": {"message": ...}}``."""
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str) and msg:
                return msg
        elif isinstance(err, str) and err:
            return err
    return ""


def error_from_response(
    status: int,
    payload: Any,
    *,
    retry_after: float | None = None,
    key: str | None = None,
    what: str = "OpenRouter",
) -> AgentError:
    """Map an OpenRouter error (HTTP status or in-body ``error.code``) to our typed errors.

    401 -> invalid_key, 402 -> insufficient_credits, 429 -> rate_limited, anything else
    (400/403/404/5xx/524/529 and malformed bodies) -> upstream_error.
    """
    message = redact(error_message(payload), key)[:300]
    if status == 401:
        return InvalidKeyError(message or "OpenRouter rejected the API key")
    if status == 402:
        return InsufficientCreditsError(message or "Insufficient OpenRouter credits")
    if status == 429:
        return RateLimitedError(message or "Rate limited by OpenRouter", retry_after=retry_after)
    detail = f": {message}" if message else ""
    return UpstreamError(f"{what} returned HTTP {status}{detail}", upstream_status=status)


def error_status_from_body(payload: Any) -> int | None:
    """``error.code`` of an error delivered with HTTP 200 (OpenRouter can do that)."""
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        code = payload["error"].get("code")
        if isinstance(code, int) and 400 <= code <= 599:
            return code
        return 502
    return None


class ToolCall(BaseModel):
    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str = ""
    # Set when the model emitted arguments that are not valid JSON; callers may re-ask.
    parse_error: str | None = None


class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None  # USD, when OpenRouter reports it


class ChatResult(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    model: str | None = None
    usage: ChatUsage = Field(default_factory=ChatUsage)
    # The assistant message in OpenAI format, ready to be appended to `messages` in tool loops.
    assistant_message: dict[str, Any] = Field(default_factory=dict)


class OpenRouterClient:
    """Chat completions (``POST /chat/completions``) bound to one request's API key."""

    def __init__(
        self,
        api_key: str | SecretStr,
        *,
        http: httpx.AsyncClient,
        settings: Settings | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        self._key = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        if not self._key.get_secret_value().strip():
            raise InvalidKeyError("Missing OpenRouter API key")
        self._http = http
        self._settings = settings or get_settings()
        self._retry = retry or RetryPolicy.from_settings(self._settings)

    def __repr__(self) -> str:  # never leak the key through repr/logging of the client
        return "OpenRouterClient(api_key=**********)"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Title": self._settings.app_title,
        }

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": model or self._settings.reasoning_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self._settings.llm_max_tokens,
        }
        if self._settings.reasoning_effort:
            body["reasoning"] = {"effort": self._settings.reasoning_effort}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        if response_format:
            body["response_format"] = response_format
        if tools or response_format:
            # Route only to providers that actually honour tools / structured output.
            body["provider"] = {"require_parameters": True}
        if extra:
            body.update(extra)

        key = self._key.get_secret_value()
        response = await send_with_retries(
            self._http,
            "POST",
            f"{self._settings.openrouter_base_url.rstrip('/')}/chat/completions",
            policy=self._retry,
            what="OpenRouter chat",
            headers=self._headers(),
            json=body,
            timeout=self._settings.llm_timeout_s,
        )
        payload = safe_json(response)

        if response.status_code != 200:
            raise error_from_response(
                response.status_code,
                payload,
                retry_after=parse_retry_after(response),
                key=key,
                what="OpenRouter chat",
            )
        # OpenRouter may answer 200 with an `error` object and no `choices`.
        in_body = error_status_from_body(payload)
        if in_body is not None and not (payload or {}).get("choices"):
            raise error_from_response(in_body, payload, key=key, what="OpenRouter chat")
        if not isinstance(payload, dict):
            raise UpstreamError("OpenRouter chat returned a non-JSON body", upstream_status=200)
        return self._parse(payload, key=key)

    @staticmethod
    def _parse(payload: dict[str, Any], *, key: str) -> ChatResult:
        choices = payload.get("choices") or []
        if not choices:
            raise UpstreamError("OpenRouter chat returned no choices", upstream_status=200)
        choice = choices[0]
        if choice.get("error") or choice.get("finish_reason") == "error":
            status = error_status_from_body(choice) or 502
            raise error_from_response(
                status, choice, key=key, what="OpenRouter chat (generation failed)"
            )

        message = choice.get("message") or {}
        tool_calls: list[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            fn = raw_call.get("function") or {}
            raw_args = fn.get("arguments")
            parse_error: str | None = None
            arguments: dict[str, Any] = {}
            if isinstance(raw_args, dict):
                arguments, raw_text = raw_args, json.dumps(raw_args)
            else:
                raw_text = raw_args or "{}"
                try:
                    parsed = json.loads(raw_text)
                    if isinstance(parsed, dict):
                        arguments = parsed
                    else:
                        parse_error = "tool arguments are not a JSON object"
                except json.JSONDecodeError as exc:
                    parse_error = f"invalid JSON in tool arguments: {exc.msg}"
            tool_calls.append(
                ToolCall(
                    id=raw_call.get("id") or "",
                    name=fn.get("name") or "",
                    arguments=arguments,
                    raw_arguments=raw_text,
                    parse_error=parse_error,
                )
            )

        usage_raw = payload.get("usage") or {}
        cost = usage_raw.get("cost")
        usage = ChatUsage(
            prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
            completion_tokens=int(usage_raw.get("completion_tokens") or 0),
            total_tokens=int(usage_raw.get("total_tokens") or 0),
            cost=float(cost) if isinstance(cost, int | float) else None,
        )

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.get("content"),
        }
        if message.get("tool_calls"):
            assistant_message["tool_calls"] = message["tool_calls"]

        return ChatResult(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            model=payload.get("model"),
            usage=usage,
            assistant_message=assistant_message,
        )
