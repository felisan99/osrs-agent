"""Flip maths with hand-computed expectations (SPEC §13.2).

Convention used throughout: a flip BUYS at ``low`` (instant-sell price) and SELLS at ``high``
(instant-buy price); tax = floor(2% of the sale price), capped at 5,000,000, on the sale only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from fractions import Fraction

import pytest

from app.clients.prices import AveragePrice, ItemMeta, LatestPrice
from app.tools.flips import (
    FlipConfig,
    NoQuote,
    compute_flip,
    rank_candidates,
    unit_economics,
)
from app.tools.tax import TaxPolicy

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
NOW_TS = int(NOW.timestamp())
BOND = 13190
POLICY = TaxPolicy(rate=Fraction(1, 50), cap=5_000_000, exempt_ids=frozenset({BOND}))
NO_EXEMPTIONS = TaxPolicy(rate=Fraction(1, 50), cap=5_000_000)
CFG = FlipConfig(max_price_age_s=3_600, max_roi=0.50, low_volume_warn=50)


def quote(low: int, high: int, age: int = 60) -> LatestPrice:
    return LatestPrice(high=high, high_time=NOW_TS - age, low=low, low_time=NOW_TS - age)


def volume(low_volume: int, high_volume: int) -> AveragePrice:
    return AveragePrice(avg_high=1.0, high_volume=high_volume, avg_low=1.0, low_volume=low_volume)


def flip(meta: ItemMeta, latest: LatestPrice | None, hourly: AveragePrice | None = None, **kw):
    kw.setdefault("tax", POLICY)
    return compute_flip(meta, latest, hourly, now=NOW, cfg=CFG, **kw)


def has(result, fragment: str) -> bool:
    return any(fragment in w for w in result.warnings)


# --------------------------------------------------------------------------- unit economics
def test_unit_economics_by_hand() -> None:
    # sell 1,100 -> tax 22 -> net 1,078; bought at 1,000 -> margin 78
    assert unit_economics(1, 1_000, 1_100, POLICY) == (22, 1_078, 78)
    # sales under 50 coins pay no tax at all
    assert unit_economics(1, 30, 45, POLICY) == (0, 45, 15)


# --------------------------------------------------------------------------- calc_flip
WHIP = ItemMeta(id=1, name="Test whip", members=True, limit=70)


def test_basic_flip_numbers_and_buy_limit_window() -> None:
    r = flip(WHIP, quote(1_000, 1_100), volume(low_volume=400, high_volume=250))
    assert (r.buy_price, r.sell_price) == (1_000, 1_100)
    assert (r.tax_per_unit, r.net_sell_price, r.margin_per_unit) == (22, 1_078, 78)
    assert r.roi_pct == 7.8  # 78 / 1,000
    assert r.buy_limit_per_4h == 70
    assert r.capital_for_full_limit == 70_000
    assert r.max_profit_per_limit_window == 5_460  # 78 * 70
    assert r.qty_requested is None and r.profit_this_window is None  # no qty asked for
    assert r.warnings == []


def test_volume_bounds_liquidity_by_the_slower_side() -> None:
    r = flip(WHIP, quote(1_000, 1_100), volume(low_volume=400, high_volume=250))
    assert r.volume is not None
    assert r.volume.buy_side_units_per_hour == 400  # lowPriceVolume fills our buy offer
    assert r.volume.sell_side_units_per_hour == 250  # highPriceVolume fills our sell offer
    assert r.volume.bottleneck_units_per_hour == 250
    assert r.volume.typical_units_per_4h_window == 1_000


def test_quantity_within_the_limit() -> None:
    r = flip(WHIP, quote(1_000, 1_100), volume(400, 250), qty=30)
    assert (r.qty_this_window, r.windows_needed) == (30, 1)
    assert (r.cost_this_window, r.profit_this_window, r.profit_for_all_qty) == (
        30_000,
        2_340,
        2_340,
    )
    assert not has(r, "Buy limit")


def test_quantity_above_the_limit_needs_several_windows() -> None:
    r = flip(WHIP, quote(1_000, 1_100), volume(400, 250), qty=100)
    assert r.qty_requested == 100 and r.qty_this_window == 70
    assert r.windows_needed == 2  # ceil(100 / 70)
    assert (r.cost_this_window, r.profit_this_window, r.profit_for_all_qty) == (
        70_000,
        5_460,
        7_800,
    )
    assert has(r, "Buy limit") and has(r, "2 windows")


def test_large_quantity_vs_market_volume_warns_about_slow_fills() -> None:
    pot = ItemMeta(id=2, name="Slow pot", limit=5_000)
    r = flip(pot, quote(1_000, 1_100), volume(low_volume=250, high_volume=300), qty=500)
    assert r.volume is not None and r.volume.typical_units_per_4h_window == 1_000
    assert has(r, "50% of the typical 4-hour market volume")
    small = flip(pot, quote(1_000, 1_100), volume(250, 300), qty=200)  # 20% < 25%
    assert not has(small, "typical 4-hour market volume")


def test_tax_cap_binds_on_expensive_items_and_amounts_exceed_int32() -> None:
    rare = ItemMeta(id=3, name="Rare", limit=8)
    r = flip(rare, quote(300_000_000, 330_000_000), volume(5, 5))
    assert r.tax_per_unit == 5_000_000  # raw 2% would be 6,600,000
    assert r.net_sell_price == 325_000_000 and r.margin_per_unit == 25_000_000
    assert r.roi_pct == 8.33
    assert r.capital_for_full_limit == 2_400_000_000 > 2**31
    assert r.max_profit_per_limit_window == 200_000_000
    below_cap = flip(rare, quote(200_000_000, 210_000_000), volume(5, 5))
    assert (below_cap.tax_per_unit, below_cap.margin_per_unit) == (4_200_000, 5_800_000)
    assert below_cap.roi_pct == 2.9


def test_exempt_item_pays_no_tax_and_is_flagged() -> None:
    bond = ItemMeta(id=BOND, name="Old school bond", limit=100)
    r = flip(bond, quote(11_668_000, 11_700_000), volume(300, 300))
    assert r.tax_exempt and r.tax_per_unit == 0
    assert r.margin_per_unit == 32_000 and r.roi_pct == 0.27
    assert has(r, "exempt from the Grand Exchange tax")
    # The very same prices on a taxed item are a loss: 2% of 11.7M = 234,000.
    taxed = flip(ItemMeta(id=99, name="Taxed twin", limit=100), quote(11_668_000, 11_700_000))
    assert taxed.margin_per_unit == -202_000 and not taxed.tax_exempt
    assert has(taxed, "No profit after tax")


def test_negative_margin_is_reported_with_a_warning() -> None:
    r = flip(WHIP, quote(1_000, 1_010), volume(400, 400))
    assert (r.tax_per_unit, r.net_sell_price, r.margin_per_unit) == (20, 990, -10)
    assert r.roi_pct == -1.0 and has(r, "No profit after tax")


def test_unusually_high_roi_is_flagged() -> None:
    r = flip(WHIP, quote(100, 200), volume(400, 400))
    assert r.margin_per_unit == 96 and r.roi_pct == 96.0
    assert has(r, "unusually high")


def test_stale_quotes_are_flagged_with_their_age() -> None:
    r = flip(WHIP, quote(1_000, 1_100, age=7_200), volume(400, 400))
    assert r.buy_quote_age_s == r.sell_quote_age_s == 7_200
    assert has(r, "buy quote is 120 minutes old") and has(r, "sell quote is 120 minutes old")
    assert r.buy_quote_time == "2026-10-14T10:00:00Z"


def test_missing_volume_and_low_volume_warnings() -> None:
    assert has(flip(WHIP, quote(1_000, 1_100), None), "No hourly volume data")
    r = flip(WHIP, quote(1_000, 1_100), volume(low_volume=10, high_volume=500))
    assert has(r, "only 10 units/hour")


def test_unknown_buy_limit_leaves_limit_fields_empty() -> None:
    r = flip(ItemMeta(id=4, name="No limit"), quote(1_000, 1_100), volume(400, 400), qty=10)
    assert r.buy_limit_per_4h is None and r.capital_for_full_limit is None
    assert r.max_profit_per_limit_window is None and r.windows_needed is None
    assert r.qty_this_window == 10 and r.profit_this_window == 780
    assert has(r, "buy limit of this item is unknown")


@pytest.mark.parametrize(
    ("latest", "reason"),
    [
        (None, "no_quote"),
        (LatestPrice(high=1_100, high_time=NOW_TS, low=None, low_time=None), "one_sided"),
        (LatestPrice(high=None, high_time=None, low=1_000, low_time=NOW_TS), "one_sided"),
        (LatestPrice(high=1_100, high_time=NOW_TS, low=0, low_time=NOW_TS), "invalid_price"),
    ],
)
def test_items_without_a_usable_quote_raise_noquote(latest, reason) -> None:
    with pytest.raises(NoQuote) as err:
        flip(WHIP, latest)
    assert err.value.reason == reason


# --------------------------------------------------------------------------- rank_flips
def market() -> tuple[dict[int, ItemMeta], dict[int, LatestPrice], dict[int, AveragePrice]]:
    """12 items in the market; each one is built to fail exactly ONE rule (or to rank)."""
    metas = [
        ItemMeta(id=1, name="Alpha", limit=100),
        ItemMeta(id=2, name="Bravo", members=True, limit=50),
        ItemMeta(id=3, name="Charlie", limit=1_000),
        ItemMeta(id=4, name="Delta"),  # no buy limit
        ItemMeta(id=5, name="Echo", limit=10),
        ItemMeta(id=6, name="Foxtrot", limit=200),
        ItemMeta(id=7, name="Golf", limit=200),
        ItemMeta(id=8, name="Hotel", limit=200),
        ItemMeta(id=9, name="India", limit=200),
        ItemMeta(id=10, name="Juliet", limit=300),
        ItemMeta(id=12, name="Kilo", limit=200),
        ItemMeta(id=13, name="Lima", limit=200),
    ]
    latest = {
        1: quote(2_000, 2_200),
        2: quote(50_000, 53_000),
        3: quote(10, 14),
        4: quote(1_000, 1_200),
        5: quote(20_000_000, 21_000_000),  # unaffordable with 10M
        6: quote(1_000, 1_200),  # volume too low
        7: quote(1_000, 1_200, age=7_200),  # stale
        8: quote(1_000, 1_010),  # loses money after tax
        9: quote(1_000, 2_000),  # ROI 96%: implausible
        10: quote(100_000, 104_000),
        11: quote(1_000, 1_200),  # not in /mapping
        12: quote(1_000, 1_200),  # no hourly volume row
        13: LatestPrice(high=1_200, high_time=NOW_TS - 60, low=None, low_time=None),
    }
    hourly = {
        1: volume(500, 400),
        2: volume(300, 200),
        3: volume(5_000, 6_000),
        4: volume(500, 500),
        5: volume(500, 500),
        6: volume(30, 500),
        7: volume(500, 500),
        8: volume(500, 500),
        9: volume(500, 500),
        10: volume(150, 150),
        13: volume(500, 500),
    }
    return {m.id: m for m in metas}, latest, hourly


def rank(*, max_gp=10_000_000, members=True, min_volume=100, top_n=10, tax=POLICY, data=None):
    mapping, latest, hourly = data or market()
    return rank_candidates(
        mapping,
        latest,
        hourly,
        tax=tax,
        now=NOW,
        cfg=CFG,
        max_gp=max_gp,
        members=members,
        min_volume=min_volume,
        top_n=top_n,
    )


def test_ranking_order_and_every_number_by_hand() -> None:
    result = rank()
    assert [(r.rank, r.item) for r in result.rows] == [
        (1, "Juliet"),
        (2, "Bravo"),
        (3, "Alpha"),
        (4, "Charlie"),
    ]
    juliet, bravo, alpha, charlie = result.rows

    # Juliet: tax 2,080 -> margin 1,920; the 10M budget buys only 100 of the 300-unit limit.
    assert (juliet.margin_per_unit, juliet.roi_pct) == (1_920, 1.92)
    assert (juliet.units_this_window, juliet.limiting_factor) == (100, "budget")
    assert juliet.capital_needed == 10_000_000
    assert juliet.profit_per_limit_window == 192_000 and juliet.estimated_profit_per_hour == 48_000
    assert juliet.bottleneck_units_per_hour == 150 and juliet.warnings == []
    # Bravo: tax 1,060 -> margin 1,940; the 50-unit buy limit binds (budget and volume allow 200).
    assert (bravo.margin_per_unit, bravo.roi_pct) == (1_940, 3.88)
    assert (bravo.units_this_window, bravo.limiting_factor) == (50, "buy_limit")
    assert (bravo.profit_per_limit_window, bravo.capital_needed) == (97_000, 2_500_000)
    assert bravo.bottleneck_units_per_hour == 200 and bravo.warnings == []
    # Alpha: tax 44 -> margin 156 x 100.
    assert (
        alpha.margin_per_unit,
        alpha.profit_per_limit_window,
        alpha.estimated_profit_per_hour,
    ) == (
        156,
        15_600,
        3_900,
    )
    assert (alpha.units_this_window, alpha.limiting_factor) == (100, "buy_limit")
    # Charlie: sale price 14 < 50 so no tax at all; margin 4, ROI 40% (under the 50% cut-off).
    assert (charlie.margin_per_unit, charlie.roi_pct, charlie.profit_per_limit_window) == (
        4,
        40.0,
        4_000,
    )


def test_each_exclusion_rule_is_counted_once() -> None:
    result = rank()
    assert result.items_evaluated == 12  # item 11 is not in /mapping
    assert result.excluded == {
        "implausible_roi": 1,  # India
        "low_volume": 1,  # Foxtrot
        "no_buy_limit": 1,  # Delta
        "no_hourly_volume": 1,  # Kilo
        "no_profit_after_tax": 1,  # Hotel
        "no_two_sided_quote": 1,  # Lima
        "not_in_mapping": 1,  # 11
        "stale_quote": 1,  # Golf
        "unaffordable": 1,  # Echo
    }


def test_free_to_play_players_do_not_get_members_items() -> None:
    result = rank(members=False)
    assert [r.item for r in result.rows] == ["Juliet", "Alpha", "Charlie"]
    assert result.excluded["members_only"] == 1


def test_a_smaller_budget_reorders_the_ranking() -> None:
    # 150k: Alpha 75 units * 156 = 11,700; Bravo 3 * 1,940 = 5,820; Charlie 4,000; Juliet 1 * 1,920.
    result = rank(max_gp=150_000)
    assert [(r.item, r.units_this_window, r.profit_per_limit_window) for r in result.rows] == [
        ("Alpha", 75, 11_700),
        ("Bravo", 3, 5_820),
        ("Charlie", 1_000, 4_000),
        ("Juliet", 1, 1_920),
    ]
    assert result.excluded["unaffordable"] == 1  # only Echo; Juliet's 100k still fits


def test_min_volume_zero_admits_illiquid_items_with_a_warning() -> None:
    result = rank(min_volume=0)
    fox = next(r for r in result.rows if r.item == "Foxtrot")
    # margin 176 (tax 24); the market only trades 30 units/hour on its slow side, so a flipper
    # can realistically move 30 x 4h x 25% = 30 units, NOT the 200-unit buy limit.
    assert (fox.units_this_window, fox.limiting_factor) == (30, "market_volume")
    assert fox.profit_per_limit_window == 5_280
    assert any("Low liquidity (30 units/hour" in w for w in fox.warnings)
    assert [r.item for r in result.rows] == ["Juliet", "Bravo", "Alpha", "Foxtrot", "Charlie"]


def test_top_n_truncates_after_ranking() -> None:
    rows = rank(top_n=2).rows
    assert [r.item for r in rows] == ["Juliet", "Bravo"] and [r.rank for r in rows] == [1, 2]


def test_ties_break_on_roi_then_item_id() -> None:
    mapping, latest, hourly = {}, {}, {}
    for item_id, name, low, high, limit in [
        (21, "Tie low roi", 40, 46, 150),  # margin 6  -> profit 900, ROI 15%
        (20, "Tie high roi", 40, 49, 100),  # margin 9 -> profit 900, ROI 22.5%
        (22, "Tie same as 21", 40, 46, 150),  # identical to 21 -> id decides
    ]:
        mapping[item_id] = ItemMeta(id=item_id, name=name, limit=limit)
        latest[item_id] = quote(low, high)
        hourly[item_id] = volume(500, 500)
    result = rank(data=(mapping, latest, hourly))
    assert [r.item_id for r in result.rows] == [20, 21, 22]
    assert {r.profit_per_limit_window for r in result.rows} == {900}


def test_exemption_changes_the_ranking_outcome() -> None:
    bond = {BOND: ItemMeta(id=BOND, name="Old school bond", limit=100)}
    latest = {BOND: quote(11_668_000, 11_700_000)}
    hourly = {BOND: volume(300, 300)}
    exempt = rank(max_gp=20_000_000, data=(bond, latest, hourly))
    assert [(r.item, r.units_this_window, r.profit_per_limit_window) for r in exempt.rows] == [
        ("Old school bond", 1, 32_000)
    ]
    taxed = rank(max_gp=20_000_000, tax=NO_EXEMPTIONS, data=(bond, latest, hourly))
    assert taxed.rows == [] and taxed.excluded == {"no_profit_after_tax": 1}


def test_units_are_capped_by_the_share_of_market_volume_one_flipper_can_capture() -> None:
    # Huge buy limit and budget, but only 200 units/hour trade on the slow side:
    # 200 x 4 h x 25% = 200 units, so the 10,000-unit limit is irrelevant.
    data = (
        {70: ItemMeta(id=70, name="Thin market", limit=10_000)},
        {70: quote(1_000, 1_100)},
        {70: volume(low_volume=200, high_volume=900)},
    )
    row = rank(max_gp=1_000_000_000, data=data).rows[0]
    assert (row.units_this_window, row.limiting_factor) == (200, "market_volume")
    assert row.profit_per_limit_window == 78 * 200 and row.capital_needed == 200_000
    # The share is a parameter, not a constant: capture 50% and the cap doubles.
    greedy = rank_candidates(
        *data,
        tax=POLICY,
        now=NOW,
        cfg=FlipConfig(market_share=0.5),
        max_gp=1_000_000_000,
        members=True,
        min_volume=100,
        top_n=5,
    )
    assert greedy.rows[0].units_this_window == 400 and greedy.market_share_assumed == 0.5


def test_items_with_no_tradable_volume_are_excluded_even_without_a_volume_filter() -> None:
    data = (
        {71: ItemMeta(id=71, name="Dead market", limit=100)},
        {71: quote(1_000, 1_100)},
        {71: volume(low_volume=0, high_volume=500)},  # nobody sells to our buy offer
    )
    result = rank(min_volume=0, data=data)
    assert result.rows == [] and result.excluded == {"no_liquidity": 1}


def test_invalid_budget_is_rejected() -> None:
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_gp"):
            rank(max_gp=bad)
