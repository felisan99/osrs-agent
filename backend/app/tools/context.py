"""Shared state for the price tools: item index + tax policy (rebuilt when /mapping refreshes)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.clients.prices import PricesClient
from app.config import Settings
from app.tools.items import ItemIndex
from app.tools.tax import TaxPolicy

if TYPE_CHECKING:
    from app.tools.moneymakers import GuideStore

logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Bundle:
    mapping_fetched_at: datetime
    index: ItemIndex
    tax: TaxPolicy
    unresolved_exempt: tuple[str, ...]


class ToolContext:
    """Everything the tools need, minus the per-call arguments. One instance per process."""

    def __init__(
        self,
        settings: Settings,
        prices: PricesClient,
        *,
        now: Callable[[], datetime] = utcnow,
        guides: GuideStore | None = None,
    ) -> None:
        self.settings = settings
        self.prices = prices
        self.now = now
        self.guides = guides  # structured money-making guides (rank_moneymakers)
        self._bundle: Bundle | None = None

    async def bundle(self) -> Bundle:
        snapshot = await self.prices.mapping()
        current = self._bundle
        if current is not None and current.mapping_fetched_at == snapshot.fetched_at:
            return current
        items = list(snapshot.data.values())
        policy, unresolved = TaxPolicy.from_settings(self.settings, items)
        if unresolved:
            # A silently dropped exemption would understate profit for those items.
            logger.warning("GE tax exempt names not found in /mapping: %s", unresolved)
        self._bundle = Bundle(snapshot.fetched_at, ItemIndex(items), policy, tuple(unresolved))
        return self._bundle
