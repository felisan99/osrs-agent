"""Price-history statistics against hand-computed series."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.clients.prices import Timeseries, TimeseriesPoint
from app.tools.history import summarize_timeseries

T0 = 1_791_000_000
HOUR = 3_600
FETCHED = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


def point(i: int, high: float | None, low: float | None, hv: int = 10, lv: int = 5):
    return TimeseriesPoint(
        timestamp=T0 + i * HOUR, avg_high=high, avg_low=low, high_volume=hv, low_volume=lv
    )


def series(points: list[TimeseriesPoint], timestep: int | None = HOUR) -> Timeseries:
    return Timeseries(
        itemId=4151, lookback="24h", timestep=timestep, points=points, fetched_at=FETCHED
    )


def rising() -> Timeseries:
    # high 100,102,...,118 ; low 90,92,...,108 ; mid 95..113 ; 10 hourly points
    return series([point(i, 100 + 2 * i, 90 + 2 * i) for i in range(10)])


def test_rising_series_statistics_by_hand() -> None:
    out = summarize_timeseries(rising(), "Abyssal whip")
    assert out["item"] == "Abyssal whip" and out["n_points"] == 10 and out["stale"] is False
    assert out["instant_buy_price_high"] == {
        "min": 100,
        "max": 118,
        "median": 109.0,
        "first": 100,
        "last": 118,
        "change_pct": 18.0,
    }
    assert out["instant_sell_price_low"] == {
        "min": 90,
        "max": 108,
        "median": 99.0,
        "first": 90,
        "last": 108,
        "change_pct": 20.0,
    }
    trend = out["trend"]
    # mid goes 95 -> 113: +18.95%; slope 2/hour = 48/day over a mean mid of 104 = 46.15%/day
    assert trend["direction"] == "up" and trend["change_pct"] == 18.95
    assert trend["slope_pct_per_day"] == 46.15
    assert out["range"] == {"start": "2026-10-03T04:00:00Z", "end": "2026-10-03T13:00:00Z"}


def test_volume_totals_and_per_hour_use_the_full_window() -> None:
    out = summarize_timeseries(rising(), "x")
    assert out["volume"] == {
        "units_bought_instantly_total": 100,  # 10 points x 10
        "units_sold_instantly_total": 50,  # 10 points x 5
        "avg_units_traded_per_hour": 15,  # 150 units over 10 hours (not 9: N points cover N steps)
    }


def test_timestep_is_inferred_when_the_api_omits_it() -> None:
    out = summarize_timeseries(series(rising().points, timestep=None), "x")
    assert out["volume"]["avg_units_traded_per_hour"] == 15


def test_downsampling_buckets_means_and_volume() -> None:
    out = summarize_timeseries(rising(), "x", max_points=5)
    assert len(out["points"]) == 5
    first, last = out["points"][0], out["points"][-1]
    # bucket 0 = points 0,1: highs 100,102 -> 101 ; lows 90,92 -> 91 ; 2 x 15 units
    assert (first["high"], first["low"], first["units_traded"]) == (101.0, 91.0, 30)
    assert (last["high"], last["low"]) == (117.0, 107.0)  # points 8,9
    assert first["t"] == "2026-10-03T05:00:00Z"  # timestamp of the bucket's middle point
    assert sum(p["units_traded"] for p in out["points"]) == 150  # nothing lost


def test_max_points_larger_than_the_series_returns_every_point() -> None:
    assert len(summarize_timeseries(rising(), "x", max_points=100)["points"]) == 10


def test_flat_and_falling_series() -> None:
    flat = summarize_timeseries(series([point(i, 100, 100) for i in range(6)]), "x")
    assert flat["trend"]["direction"] == "flat" and flat["trend"]["change_pct"] == 0.0
    assert flat["trend"]["slope_pct_per_day"] == 0.0 and flat["trend"]["volatility_pct"] == 0.0
    down = summarize_timeseries(
        series([point(i, 200 - 10 * i, 190 - 10 * i) for i in range(5)]), "x"
    )
    assert down["trend"]["direction"] == "down" and down["trend"]["change_pct"] < 0
    # a move under 1% over the whole window still counts as flat
    drift = summarize_timeseries(series([point(0, 1_000, 1_000), point(1, 1_005, 1_005)]), "x")
    assert drift["trend"]["direction"] == "flat"


def test_gaps_in_one_side_are_skipped_not_zeroed() -> None:
    pts = [point(0, 100, 90), point(1, None, 92), point(2, 104, None), point(3, 106, 96)]
    out = summarize_timeseries(series(pts), "x")
    assert (
        out["instant_buy_price_high"]["min"] == 100 and out["instant_buy_price_high"]["max"] == 106
    )
    assert out["instant_sell_price_low"]["median"] == 92  # median of 90, 92, 96
    assert out["trend"]["direction"] == "up"


def test_unsorted_input_is_sorted_by_time() -> None:
    shuffled = rising().points[::-1]
    out = summarize_timeseries(series(shuffled), "x")
    assert (
        out["instant_buy_price_high"]["first"] == 100
        and out["instant_buy_price_high"]["last"] == 118
    )


@pytest.mark.parametrize(
    "points",
    [[], [point(0, 100, 90)], [point(0, None, None), point(1, None, None)]],
)
def test_too_little_data_is_reported_explicitly(points) -> None:
    out = summarize_timeseries(series(points), "x")
    assert out["insufficient_data"] is True and "trend" not in out


def test_stale_flag_is_propagated() -> None:
    stale = series(rising().points).model_copy(update={"stale": True})
    assert summarize_timeseries(stale, "x")["stale"] is True
