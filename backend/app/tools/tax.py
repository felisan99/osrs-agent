"""Grand Exchange tax (SPEC §6): parametrised, never hardcoded in the logic.

Rules (verified 2026-10-14 against the Wiki "Grand Exchange" page and the wiki's own
``Module:Mmgtable.deduct_tax``): the tax is charged on the SELL side, per item, rounded down
(so sales under 50 coins pay nothing at 2%), capped per item, and a list of items is exempt.

The rate is a ``Fraction`` and the arithmetic is integer-only, so results never depend on float
rounding, whatever rate is configured and however large the price (prices exceed int32).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

from app.clients.prices import ItemMeta
from app.config import Settings

_DOSE_SUFFIX = r"\(\d+\)"


def resolve_exempt_ids(
    names: Iterable[str], items: Iterable[ItemMeta]
) -> tuple[frozenset[int], list[str]]:
    """Map exempt item *names* to ids using ``/mapping``.

    A name with no exact match is treated as a dose family and expanded to ``name(1)``,
    ``name(2)``... (e.g. "Energy potion"). Returns ``(ids, unresolved_names)``; callers should
    surface the unresolved ones, since a silently dropped exemption would understate profit.
    """
    items = list(items)
    by_name = {item.name: item.id for item in items}
    ids: set[int] = set()
    unresolved: list[str] = []
    for name in names:
        if name in by_name:
            ids.add(by_name[name])
            continue
        family = re.compile(re.escape(name) + _DOSE_SUFFIX)
        variants = [item.id for item in items if family.fullmatch(item.name)]
        if variants:
            ids.update(variants)
        else:
            unresolved.append(name)
    return frozenset(ids), unresolved


@dataclass(frozen=True)
class TaxPolicy:
    rate: Fraction
    cap: int
    exempt_ids: frozenset[int] = frozenset()

    @classmethod
    def from_settings(
        cls, settings: Settings, items: Iterable[ItemMeta]
    ) -> tuple[TaxPolicy, list[str]]:
        """Build the policy from config + the current item table. Also returns unresolved names."""
        exempt, unresolved = resolve_exempt_ids(settings.ge_tax_exempt_items, items)
        policy = cls(
            rate=Fraction(settings.ge_tax_rate),  # Decimal -> exact Fraction (0.02 == 1/50)
            cap=settings.ge_tax_cap,
            exempt_ids=exempt,
        )
        return policy, unresolved

    def tax(self, item_id: int | None, sell_price: int) -> int:
        """Tax in coins for ONE unit sold at ``sell_price``."""
        if sell_price <= 0 or (item_id is not None and item_id in self.exempt_ids):
            return 0
        return min((sell_price * self.rate.numerator) // self.rate.denominator, self.cap)

    def net(self, item_id: int | None, sell_price: int) -> int:
        """What the seller actually receives for one unit."""
        return sell_price - self.tax(item_id, sell_price)

    def tax_estimate(self, item_id: int | None, sell_price: float) -> int:
        """Tax for an *average* price (``/1h`` averages are floats): floor to a whole coin first."""
        return self.tax(item_id, int(sell_price))

    def net_estimate(self, item_id: int | None, sell_price: float) -> float:
        return sell_price - self.tax_estimate(item_id, sell_price)
