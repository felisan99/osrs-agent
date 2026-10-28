"""Flip analytics (app/tools/flip_analysis.py): numbers checked against a hand-built history."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import pytest

from app.clients.prices import ItemMeta, LatestPrice, Timeseries, TimeseriesPoint
from app.tools.flip_analysis import (
    STRATEGIES,
    Insufficient,
    analyze,
    passes,
)
from app.tools.tax import TaxPolicy

END = 1_792_000_000
TAX = TaxPolicy(rate=Fraction(1, 50), cap=5_000_000)
ITEM = ItemMeta(id=1000, name="Test potion", members=True, limit=2_000)
Hour = tuple[float | None, float | None, int, int]  # avg_high, avg_low, high_volume, low_volume


def series(
    hours: Callable[[int], Hour], n: int = 168, step: int = 3_600, lookback: str = "7d"
) -> Timeseries:
    points = []
    for i in range(n):
        high, low, high_volume, low_volume = hours(i)
        points.append(
            TimeseriesPoint(
                timestamp=END - (n - 1 - i) * step,
                avg_high=high,
                avg_low=low,
                high_volume=high_volume,
                low_volume=low_volume,
            )
        )
    return Timeseries(
        itemId=1000, lookback=lookback, timestep=step, points=points, fetched_at=_dt()
    )


def _dt():
    from datetime import UTC, datetime

    return datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


def steady(i: int) -> Hour:
    return 1_050.0, 1_000.0, 600, 500  # tax 21 -> net 1,029 -> margin 29 every hour


def run(week, month=None, quote=None, strategy="volume", budget=None, item=ITEM):
    return analyze(
        item,
        week,
        month,
        quote or LatestPrice(high=1_050, low=1_000),
        tax=TAX,
        strategy=STRATEGIES[strategy],
        budget=budget,
        market_share=0.25,
    )


def test_a_steady_market_has_a_typical_margin_after_tax_and_full_repeatability() -> None:
    a = run(series(steady))
    # sell 1,050 -> tax 21 -> net 1,029; buy 1,000 -> 29 gp per unit, every hour
    assert (a.margin_typical_after_tax, a.margin_conservative_after_tax) == (29, 29)
    assert a.metrics["profitable_hours_pct"] == 100
    assert (a.buy_at, a.sell_at) == (1_000, 1_050)
    assert a.roi_typical_pct == 2.9
    assert a.confidence_pct == 100 and a.verdict == "solid"


def test_volume_is_the_slower_side_and_the_ratio_compares_buyers_with_sellers() -> None:
    a = run(series(steady))
    assert a.metrics["volume_typical_per_hour"] == 500  # min(600, 500)
    assert a.metrics["buy_sell_ratio"] == 1.2  # 600 instant buys vs 500 instant sells
    assert any("Buy/sell ratio 1.20" in line and "balanced" in line for line in a.evidence)


def test_units_are_capped_by_the_budget_the_buy_limit_and_the_market_share() -> None:
    # market: 500 x 4 h x 25% = 500; limit 2,000; budget 300,000 / 1,000 = 300
    assert (
        run(series(steady), budget=300_000).units,
        run(series(steady), budget=300_000).limiting_factor,
    ) == (300, "budget")
    assert run(series(steady), budget=10_000_000).limiting_factor == "market_volume"
    assert run(series(steady), budget=10_000_000).units == 500
    capped = run(series(steady), item=ITEM.model_copy(update={"limit": 120}))
    assert (capped.units, capped.limiting_factor) == (120, "buy_limit")
    overnight = run(series(steady), strategy="overnight", budget=10_000_000)
    assert overnight.units == 1_000  # eight hours of trading count


def test_the_worked_example_uses_the_offer_prices_and_checks_out_by_hand() -> None:
    a = run(series(steady), budget=300_000)
    w = a.worked_example
    assert (w.units, w.capital_gp, w.buy_at, w.sell_at, w.tax_per_unit) == (
        300,
        300_000,
        1_000,
        1_050,
        21,
    )
    assert w.net_profit_per_unit == 29 and w.profit_if_typical_margin == 8_700
    assert a.profit_per_window_conservative == 8_700 == a.profit_per_window_typical


def test_the_conservative_margin_is_a_bad_but_normal_hour_not_the_average() -> None:
    def uneven(i: int) -> Hour:
        # a quarter of the hours the spread is thin (net margin 4), the rest it is wide (net 79)
        return (1_100.0, 1_000.0, 600, 500) if i % 4 else (1_025.0, 1_000.0, 600, 500)

    a = run(series(uneven))
    # 1,100 -> tax 22 -> 1,078 -> 78 ; 1,025 -> tax 20 -> 1,005 -> 5
    assert a.margin_typical_after_tax == 78
    # 25th percentile: 42 of the 168 hours are at 5 and the rest at 78 -> 5 + 0.75 x 73 = 59.75
    assert a.margin_conservative_after_tax == 60


def test_how_often_the_margin_was_positive_counts_the_hours_it_was_not() -> None:
    def sometimes_underwater(i: int) -> Hour:
        return (1_010.0, 1_000.0, 600, 500) if i % 5 == 0 else (1_050.0, 1_000.0, 600, 500)

    a = run(series(sometimes_underwater))
    # 1,010 -> tax 20 -> net 990 < 1,000: negative one hour in five
    assert a.metrics["profitable_hours_pct"] == 80
    assert any("positive in 80% of the last 168" in line for line in a.evidence)


def test_a_price_far_above_its_usual_level_is_flagged_and_lowers_confidence() -> None:
    inflated = run(series(steady), quote=LatestPrice(high=1_300, low=1_250))
    assert inflated.metrics["price_vs_7d_median_pct"] == pytest.approx(24.4, abs=0.1)
    assert any("above its 7-day median" in f for f in inflated.flags)
    assert inflated.confidence_pct < 50 and inflated.verdict in ("risky", "avoid")
    assert run(series(steady)).flags == []


def test_hype_near_the_30_day_high_after_a_weekly_rise_is_flagged() -> None:
    def rising(i: int) -> Hour:
        base = 1_000 + i * 2  # +33% over the week
        return base + 50.0, float(base), 600, 500

    week = series(rising)
    month = series(
        lambda i: (1_100.0 + i, 1_050.0 + i, 600, 500), n=120, step=21_600, lookback="30d"
    )
    a = run(week, month, quote=LatestPrice(high=1_390, low=1_340))
    assert any("30-day high" in f for f in a.flags)


def test_an_item_with_a_one_sided_market_is_flagged() -> None:
    a = run(series(lambda i: (1_050.0, 1_000.0, 2_000, 400)))
    assert a.metrics["buy_sell_ratio"] == 5.0
    assert any("one-sided" in f for f in a.flags)


def test_a_thin_history_is_refused_not_guessed() -> None:
    sparse = series(lambda i: (1_050.0, 1_000.0, 5, 5) if i % 8 == 0 else (None, None, 0, 0))
    with pytest.raises(Insufficient) as err:
        run(sparse)
    assert err.value.reason == "thin_history"
    with pytest.raises(Insufficient) as no_quote:
        run(series(steady), quote=LatestPrice(high=1_050))
    assert no_quote.value.reason == "no_two_sided_quote"


def test_an_item_that_loses_money_after_tax_is_to_avoid() -> None:
    a = run(series(lambda i: (1_010.0, 1_000.0, 600, 500)))
    assert a.verdict == "avoid" and a.confidence_pct == 0


def test_strategies_apply_their_own_thresholds() -> None:
    liquid_thin_margin = run(series(steady))  # 2.9% ROI, 500 units/hour
    assert passes(liquid_thin_margin, STRATEGIES["volume"])
    assert not passes(liquid_thin_margin, STRATEGIES["margin"])  # needs >= 3% ROI
    illiquid = run(series(lambda i: (1_050.0, 1_000.0, 60, 40)))  # 40 units/hour
    assert not passes(illiquid, STRATEGIES["volume"])
    assert not passes(illiquid, STRATEGIES["overnight"])
    wide = run(series(lambda i: (1_120.0, 1_000.0, 60, 40)))  # ~9% ROI, 40 units/hour
    assert passes(wide, STRATEGIES["margin"])


def test_a_one_gp_margin_is_not_a_flip() -> None:
    penny = run(
        series(lambda i: (20.0, 18.0, 90_000, 80_000)),
        item=ITEM.model_copy(update={"limit": 300_000}),
    )
    # 20 -> no tax under 50 coins -> margin 2 on the median; the conservative margin must clear 2 gp
    assert penny.margin_conservative_after_tax == 2
    tick = run(
        series(lambda i: (19.0, 18.0, 90_000, 80_000)),
        item=ITEM.model_copy(update={"limit": 300_000}),
    )
    assert tick.margin_conservative_after_tax == 1 and not passes(tick, STRATEGIES["volume"])


def test_the_dip_strategy_buys_at_todays_low_and_sells_back_at_the_usual_price() -> None:
    a = run(
        series(steady),
        quote=LatestPrice(high=950, low=900),
        strategy="dip",
    )
    assert (a.buy_at, a.sell_at) == (900, 1_050)
    # sell 1,050 -> net 1,029; bought at 900 -> 129 gp if it recovers, half if it only gets halfway
    assert a.margin_typical_after_tax == 129 and a.margin_conservative_after_tax == 64
    assert passes(a, STRATEGIES["dip"])
    assert not passes(
        run(series(steady), strategy="dip"), STRATEGIES["dip"]
    )  # not below its usual price


def test_a_spread_that_has_tightened_since_the_week_is_flagged() -> None:
    a = run(series(steady), quote=LatestPrice(high=1_030, low=1_000))
    # 1,030 -> tax 20 -> 1,010: 10 gp now against a usual 29
    assert a.margin_now_after_tax == 10
    assert any("has tightened" in f for f in a.flags)
