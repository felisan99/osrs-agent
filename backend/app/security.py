"""Secret handling: the user's OpenRouter key is never persisted or logged (SPEC §1.5, §4, §11).

Two layers of defence:

1. Pattern-based scrubbing of anything that looks like an OpenRouter key or a bearer token.
2. Exact-match scrubbing of the secrets registered for the current request (a ``ContextVar``,
   so concurrent requests do not see each other's key and nothing is stored globally).

``RedactingFilter`` applies both to every log record; ``redact`` is used wherever an upstream
error message is turned into one of our exceptions.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

REDACTED = "[REDACTED]"

_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-or-[A-Za-z0-9_\-]{6,}"),
    re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=\-]{8,}"),
    re.compile(r"(?i)(x-openrouter-key['\"]?\s*[:=]\s*['\"]?)[^\s'\",}]+"),
    re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?)[^\s'\",}]+"),
)

_request_secrets: ContextVar[tuple[str, ...]] = ContextVar("request_secrets", default=())


def redact(text: str, *secrets: str | None) -> str:
    """Return ``text`` with known secrets and key-like tokens replaced by ``[REDACTED]``."""
    for secret in (*secrets, *_request_secrets.get()):
        if secret and len(secret) >= 4:
            text = text.replace(secret, REDACTED)
    for pattern in _PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


@contextmanager
def request_secrets(*secrets: str | None) -> Iterator[None]:
    """Register the secrets of the current request for the duration of the ``with`` block."""
    token = _request_secrets.set(tuple(s for s in secrets if s))
    try:
        yield
    finally:
        _request_secrets.reset(token)


class RedactingFilter(logging.Filter):
    """Scrub secrets from every record (message, args and exception text)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        if record.exc_info and record.exc_info[1] is not None and not record.exc_text:
            record.exc_text = redact("".join(logging.Formatter().formatException(record.exc_info)))
        elif record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


def install_log_redaction() -> None:
    """Attach the filter to the root logger's handlers and to the loggers that carry traffic.

    Called once at app start (M6). Handlers are filtered rather than loggers so records
    propagated from child loggers (httpx, uvicorn, langgraph, ...) are covered too.
    """
    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())
    # httpx logs full request lines at INFO; keep it quiet and never at DEBUG with headers.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
