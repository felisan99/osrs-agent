"""Parsing what a chat model returns when asked for JSON."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str | None) -> dict[str, Any] | None:
    """The first JSON object in ``text`` (models add code fences or a sentence around it)."""
    if not text:
        return None
    cleaned = _FENCE.sub("", text.strip())
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for end in range(start, len(cleaned)):
            char = cleaned[end]
            depth += char == "{"
            depth -= char == "}"
            if depth == 0:
                try:
                    value = json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    break
                return value if isinstance(value, dict) else None
        start = cleaned.find("{", start + 1)
    return None
