from __future__ import annotations

import logging

from app.security import REDACTED, RedactingFilter, redact, request_secrets

KEY = "sk-or-v1-0123456789abcdef0123456789abcdef"


def test_openrouter_style_keys_are_scrubbed() -> None:
    assert KEY not in redact(f"upstream said: invalid key {KEY}!")
    assert REDACTED in redact(f"upstream said: invalid key {KEY}!")


def test_bearer_and_header_forms_are_scrubbed() -> None:
    assert "abcdef123456" not in redact("Authorization: Bearer abcdef123456")
    assert "abcdef123456" not in redact("Authorization: Bearer abcdef123456 trailing")
    assert "abcdef123456" not in redact("X-OpenRouter-Key: abcdef123456")
    assert "abcdef123456" not in redact("{'authorization': 'abcdef123456'}")


def test_registered_request_secret_is_scrubbed_even_if_it_has_no_known_shape() -> None:
    odd_key = "totally-custom-key-format-9999"
    with request_secrets(odd_key):
        assert odd_key not in redact(f"boom {odd_key}")
    # Outside the request the registration is gone (nothing global is kept).
    assert odd_key in redact(f"boom {odd_key}")


def test_explicit_secret_argument() -> None:
    assert "hunter2hunter2" not in redact("password is hunter2hunter2", "hunter2hunter2")


def test_normal_text_is_untouched() -> None:
    text = "Abyssal whip costs 1,200,000 gp; sk is not a key; bearer of bad news"
    assert redact(text) == text


def test_logging_filter_scrubs_message_args_and_exceptions() -> None:
    flt = RedactingFilter()
    try:
        raise RuntimeError(f"failed with {KEY}")
    except RuntimeError:
        import sys

        record = logging.LogRecord(
            "t", logging.ERROR, __file__, 1, "request used key %s", (KEY,), sys.exc_info()
        )
    assert flt.filter(record) is True
    assert KEY not in record.getMessage()
    assert record.exc_text is not None and KEY not in record.exc_text
