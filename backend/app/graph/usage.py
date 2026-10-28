"""Cost and call accounting kept in the graph state (for the eval report and ``debug``)."""

from __future__ import annotations

from app.clients.openrouter import ChatResult


def add_llm(usage: dict[str, float] | None, result: ChatResult) -> dict[str, float]:
    merged = dict(usage or {})
    merged["llm_calls"] = merged.get("llm_calls", 0) + 1
    merged["llm_cost"] = merged.get("llm_cost", 0.0) + (result.usage.cost or 0.0)
    merged["llm_tokens"] = merged.get("llm_tokens", 0) + result.usage.total_tokens
    return merged


def add_jev(usage: dict[str, float] | None, cost: float | None, calls: int = 1) -> dict[str, float]:
    merged = dict(usage or {})
    merged["jev_calls"] = merged.get("jev_calls", 0) + calls
    merged["jev_cost"] = merged.get("jev_cost", 0.0) + (cost or 0.0)
    return merged
