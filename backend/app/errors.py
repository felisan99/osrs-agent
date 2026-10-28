"""Typed errors shared by every upstream client.

The HTTP layer maps them onto the contract of SPEC §4:
``401 invalid_key``, ``402 insufficient_credits``, ``429 rate_limited``, ``502 upstream_error``.

Messages are built from upstream ``error.message`` text and always pass through
``app.security.redact`` before they get here, so they are safe to log and to return.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class: ``code`` and ``http_status`` feed the typed error body of the API."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.code)
        self.message = message or self.code


class InvalidKeyError(AgentError):
    code = "invalid_key"
    http_status = 401


class InsufficientCreditsError(AgentError):
    code = "insufficient_credits"
    http_status = 402


class RateLimitedError(AgentError):
    code = "rate_limited"
    http_status = 429

    def __init__(self, message: str = "", *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class UpstreamError(AgentError):
    """Any other upstream failure: bad gateway, timeouts, malformed bodies, unactionable 4xx."""

    code = "upstream_error"
    http_status = 502

    def __init__(self, message: str = "", *, upstream_status: int | None = None) -> None:
        super().__init__(message)
        self.upstream_status = upstream_status


class StateTooLargeError(AgentError):
    """A Jev ``state`` would exceed the token budget; the caller must batch or trim."""

    code = "state_too_large"
    http_status = 500
