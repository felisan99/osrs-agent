"""The price timestamp every answer with prices must carry (SPEC §7, prompt rules).

Added by code, not asked of the model: a rule that matters this much is not left to a prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.graph.prompts import fixed


def latest_price_time(results: Sequence[dict[str, Any]]) -> str | None:
    """The newest ``prices_fetched_at`` among the tool results (ISO strings sort correctly)."""
    stamps = [
        r["result"]["prices_fetched_at"]
        for r in results
        if isinstance(r.get("result"), dict) and r["result"].get("prices_fetched_at")
    ]
    return max(stamps) if stamps else None


def with_price_footer(answer: str, language: str, results: Sequence[dict[str, Any]]) -> str:
    stamp = latest_price_time(results)
    if stamp is None:
        return answer
    when = stamp.replace("T", " ")[:16]  # 2026-10-14T16:11:20.28Z -> 2026-10-14 16:11
    return f"{answer}\n\n{fixed('prices_footer', language, when=when)}"
