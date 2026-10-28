"""Helpers for developer scripts (smoke tests, ingestion, eval). Never imported by the API server.

The development OpenRouter key has a hard spending cap, so every script that makes paid calls
tracks the cost OpenRouter reports (``usage.cost``) and stops before it runs away.
"""

from __future__ import annotations


class BudgetExceeded(RuntimeError):
    pass


class SpendGuard:
    """Accumulates reported costs (USD) and raises once the limit is crossed."""

    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self.calls = 0

    def add(self, cost_usd: float | None) -> None:
        self.calls += 1
        self.spent_usd += cost_usd or 0.0
        if self.spent_usd > self.limit_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f} after {self.calls} calls, over the "
                f"${self.limit_usd:.2f} limit of this run"
            )

    def summary(self) -> str:
        return f"{self.calls} paid calls, ${self.spent_usd:.6f} spent (limit ${self.limit_usd:.2f})"
