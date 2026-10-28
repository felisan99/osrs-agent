"""Price-history summary (SPEC §6 ``get_price_history``): statistics computed in code.

Given a ``/timeseries`` response it returns min/max/median/first/last per side, a trend derived
from a least-squares fit of the mid price, volume totals, and a downsampled series the model can
quote. ``high`` is the instant-buy price and ``low`` the instant-sell price (Wiki semantics).
"""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

from app.clients.prices import Timeseries, TimeseriesPoint

FLAT_THRESHOLD_PCT = 1.0  # |change| below this over the whole window counts as "flat"


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")


def _round(value: float) -> float:
    return round(value, 2)


def _side_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    first, last = values[0], values[-1]
    return {
        "min": _round(min(values)),
        "max": _round(max(values)),
        "median": _round(statistics.median(values)),
        "first": _round(first),
        "last": _round(last),
        "change_pct": _round((last - first) / first * 100) if first else 0.0,
    }


def _mid(point: TimeseriesPoint) -> float | None:
    if point.avg_high is not None and point.avg_low is not None:
        return (point.avg_high + point.avg_low) / 2
    return point.avg_high if point.avg_high is not None else point.avg_low


def _downsample(points: list[TimeseriesPoint], max_points: int) -> list[dict[str, Any]]:
    """Equal-sized index buckets; each bucket reports mean prices and summed volume."""
    if not points:
        return []
    buckets = min(max_points, len(points))
    out: list[dict[str, Any]] = []
    for b in range(buckets):
        start = b * len(points) // buckets
        end = (b + 1) * len(points) // buckets
        chunk = points[start:end]
        highs = [p.avg_high for p in chunk if p.avg_high is not None]
        lows = [p.avg_low for p in chunk if p.avg_low is not None]
        out.append(
            {
                "t": _iso(chunk[len(chunk) // 2].timestamp),
                "high": _round(statistics.fmean(highs)) if highs else None,
                "low": _round(statistics.fmean(lows)) if lows else None,
                "units_traded": sum(p.high_volume + p.low_volume for p in chunk),
            }
        )
    return out


def summarize_timeseries(
    series: Timeseries, item_name: str, *, max_points: int = 24
) -> dict[str, Any]:
    points = sorted(series.points, key=lambda p: p.timestamp)
    base: dict[str, Any] = {
        "item": item_name,
        "item_id": series.item_id,
        "lookback": series.lookback,
        "timestep_seconds": series.timestep,
        "n_points": len(points),
        "fetched_at": series.fetched_at.isoformat().replace("+00:00", "Z"),
        "stale": series.stale,
    }
    mids = [(p.timestamp, m) for p in points if (m := _mid(p)) is not None]
    if len(mids) < 2:
        return {**base, "insufficient_data": True}

    highs = [p.avg_high for p in points if p.avg_high is not None]
    lows = [p.avg_low for p in points if p.avg_low is not None]
    mid_values = [m for _, m in mids]
    mean_mid = statistics.fmean(mid_values)
    days = [(t - mids[0][0]) / 86_400 for t, _ in mids]
    # Least-squares slope of mid price per day, expressed as % of the mean price.
    slope = statistics.linear_regression(days, mid_values).slope if days[-1] > 0 else 0.0
    change_pct = (mid_values[-1] - mid_values[0]) / mid_values[0] * 100 if mid_values[0] else 0.0
    if abs(change_pct) < FLAT_THRESHOLD_PCT:
        direction = "flat"
    else:
        direction = "up" if change_pct > 0 else "down"

    # N points of step S cover N*S seconds (not the first-to-last span, which drops one step).
    step = series.timestep
    if not step:
        gaps = sorted(b.timestamp - a.timestamp for a, b in pairwise(points))
        step = gaps[len(gaps) // 2] if gaps else 3_600
    hours = max((points[-1].timestamp - points[0].timestamp + step) / 3_600, 1.0)
    return {
        **base,
        "range": {"start": _iso(points[0].timestamp), "end": _iso(points[-1].timestamp)},
        "instant_buy_price_high": _side_stats(highs),
        "instant_sell_price_low": _side_stats(lows),
        "trend": {
            "direction": direction,
            "change_pct": _round(change_pct),
            "slope_pct_per_day": _round(slope / mean_mid * 100) if mean_mid else 0.0,
            "volatility_pct": _round(statistics.pstdev(mid_values) / mean_mid * 100)
            if mean_mid
            else 0.0,
            "flat_threshold_pct": FLAT_THRESHOLD_PCT,
        },
        "volume": {
            "units_bought_instantly_total": sum(p.high_volume for p in points),
            "units_sold_instantly_total": sum(p.low_volume for p in points),
            "avg_units_traded_per_hour": round(
                sum(p.high_volume + p.low_volume for p in points) / hours
            ),
        },
        "points": _downsample(points, max_points),
    }
