"""GE tax maths, checked by hand (SPEC §13.2): 2%, rounded down per item, 5M cap, exemptions.

Reference points come from the Wiki's own Grand Exchange page: "the seller receives 49 coins if
they set the price per item to 49 or to 50 ... 98 coins at 99 or 100".
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from app.clients.prices import ItemMeta
from app.config import DEFAULT_GE_TAX_EXEMPT_ITEMS, Settings
from app.tools.tax import TaxPolicy, resolve_exempt_ids

BOND = 13190
POLICY = TaxPolicy(rate=Fraction(1, 50), cap=5_000_000, exempt_ids=frozenset({BOND}))


@pytest.mark.parametrize(
    ("price", "tax"),
    [
        (-5, 0),
        (0, 0),
        (1, 0),
        (49, 0),  # 0.98 -> rounds down: nothing to pay below 50 coins
        (50, 1),
        (99, 1),  # 1.98 -> 1
        (100, 2),
        (149, 2),
        (150, 3),
        (1_000, 20),
        (1_999, 39),  # 39.98 -> 39
        (2_000, 40),
        (123_456, 2_469),  # 2469.12 -> 2469
        (249_999_999, 4_999_999),  # just under the cap
        (250_000_000, 5_000_000),  # exactly the cap
        (250_000_050, 5_000_000),  # raw 5,000,001 -> capped
        (2_000_000_000, 5_000_000),
        (9_364_999_989, 5_000_000),  # > int32: the live maximum seen on 2026-10-14
    ],
)
def test_tax_per_unit(price: int, tax: int) -> None:
    assert POLICY.tax(None, price) == tax


@pytest.mark.parametrize(("price", "net"), [(49, 49), (50, 49), (99, 98), (100, 98), (1_000, 980)])
def test_net_matches_the_wikis_worked_examples(price: int, net: int) -> None:
    assert POLICY.net(None, price) == net


def test_exempt_items_pay_nothing_at_any_price() -> None:
    assert POLICY.tax(BOND, 11_700_000) == 0
    assert POLICY.net(BOND, 11_700_000) == 11_700_000
    assert POLICY.tax(999, 11_700_000) == 234_000  # same price, not exempt


def test_other_rates_and_cap_are_parametrised_not_hardcoded() -> None:
    three = TaxPolicy(rate=Fraction(Decimal("0.03")), cap=1_000)
    assert [three.tax(None, p) for p in (33, 34, 100, 10_000, 1_000_000)] == [0, 1, 3, 300, 1_000]
    assert TaxPolicy(rate=Fraction(0), cap=5_000_000).tax(None, 1_000_000) == 0


def test_average_prices_are_floored_to_a_whole_coin_before_taxing() -> None:
    assert POLICY.tax_estimate(None, 99.9) == 1  # floor(99.9) = 99 -> 1
    assert POLICY.tax_estimate(None, 100.4) == 2
    assert POLICY.net_estimate(None, 100.5) == pytest.approx(98.5)


ITEMS = [
    ItemMeta(id=BOND, name="Old school bond"),
    ItemMeta(id=3008, name="Energy potion(4)"),
    ItemMeta(id=3010, name="Energy potion(3)"),
    ItemMeta(id=3853, name="Games necklace(8)"),
    ItemMeta(id=3855, name="Games necklace(7)"),
    ItemMeta(id=379, name="Lobster"),
    ItemMeta(id=380, name="Lobster (burnt)"),
]


def test_exempt_names_resolve_exactly_or_as_dose_families() -> None:
    ids, unresolved = resolve_exempt_ids(
        ["Old school bond", "Energy potion", "Lobster", "Games necklace(8)", "Nonexistent item"],
        ITEMS,
    )
    # "Energy potion" has no exact entry -> expands to every dose; "Games necklace(8)" is exact,
    # so the 7-dose necklace is NOT exempt; a name that matches nothing is reported, not dropped.
    assert ids == {BOND, 3008, 3010, 379, 3853}
    assert unresolved == ["Nonexistent item"]


def test_policy_from_settings_uses_config_values() -> None:
    settings = Settings(_env_file=None, ge_tax_rate=Decimal("0.02"), ge_tax_cap=5_000_000)
    policy, unresolved = TaxPolicy.from_settings(settings, ITEMS)
    assert policy.rate == Fraction(1, 50) and policy.cap == 5_000_000
    assert BOND in policy.exempt_ids and 3008 in policy.exempt_ids
    # The fixture has only a few items, so most default names are unresolved: they must be listed.
    assert "Hammer" in unresolved and "Old school bond" not in unresolved


def test_default_exempt_list_has_no_duplicates() -> None:
    assert len(DEFAULT_GE_TAX_EXEMPT_ITEMS) == len(set(DEFAULT_GE_TAX_EXEMPT_ITEMS))


@pytest.mark.live
async def test_every_default_exempt_name_resolves_in_the_live_mapping() -> None:
    import httpx

    from app.clients.prices import PricesClient

    settings = Settings(_env_file=None, prices_user_agent="osrs-agent-tests - live check")
    async with httpx.AsyncClient() as http:
        mapping = await PricesClient(settings, http=http).mapping()
    _, unresolved = TaxPolicy.from_settings(settings, mapping.data.values())
    assert unresolved == [], (
        f"renamed or removed on the wiki? update DEFAULT_GE_TAX_EXEMPT_ITEMS: {unresolved}"
    )
