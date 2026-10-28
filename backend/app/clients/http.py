"""Retry/backoff shared by the upstream clients (OpenRouter chat, Jev, prices, wiki)."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings
from app.errors import UpstreamError

# 429 (rate limited), 5xx, and OpenRouter/Cloudflare-style 524/529. Never 401/402/4xx.
RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504, 524, 529})

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    retry_statuses: frozenset[int] = RETRYABLE_STATUSES
    jitter: bool = True
    sleep: Sleep = asyncio.sleep

    @classmethod
    def from_settings(cls, settings: Settings, *, sleep: Sleep = asyncio.sleep) -> RetryPolicy:
        return cls(
            max_retries=settings.http_max_retries,
            base_delay_s=settings.http_retry_base_delay_s,
            max_delay_s=settings.http_retry_max_delay_s,
            sleep=sleep,
        )

    def delay_for(self, attempt: int, retry_after: float | None) -> float | None:
        """Seconds to wait before retry number ``attempt`` (0-based); ``None`` means give up.

        If the server asks for a longer wait than ``max_delay_s`` we give up instead of
        blocking a chat request for minutes: the caller surfaces ``rate_limited`` instead.
        """
        delay = min(self.base_delay_s * (2**attempt), self.max_delay_s)
        if retry_after is not None:
            if retry_after > self.max_delay_s:
                return None
            delay = max(delay, retry_after)
        if self.jitter and delay > 0:
            delay *= 0.8 + 0.4 * random.random()
        return delay


def parse_retry_after(response: httpx.Response) -> float | None:
    """Numeric ``Retry-After`` (seconds). HTTP-date values are ignored."""
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def safe_json(response: httpx.Response) -> Any | None:
    try:
        return response.json()
    except ValueError:
        return None


async def send_with_retries(
    http: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    policy: RetryPolicy,
    what: str,
    **request_kwargs: Any,
) -> httpx.Response:
    """Send a request, retrying transport errors and retryable statuses.

    Returns the final response even when its status is an error: the caller owns the mapping
    to typed errors. Raises ``UpstreamError`` only when the network itself keeps failing.
    ``what`` labels error messages ("OpenRouter chat", "prices API", ...).
    """
    attempt = 0
    while True:
        try:
            response = await http.request(method, url, **request_kwargs)
        except httpx.RequestError as exc:
            if attempt >= policy.max_retries:
                raise UpstreamError(f"{what}: {type(exc).__name__} (network error)") from exc
            await policy.sleep(policy.delay_for(attempt, None) or 0.0)
            attempt += 1
            continue

        if response.status_code in policy.retry_statuses and attempt < policy.max_retries:
            delay = policy.delay_for(attempt, parse_retry_after(response))
            if delay is not None:
                await policy.sleep(delay)
                attempt += 1
                continue
        return response
