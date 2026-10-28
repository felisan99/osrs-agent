"""Flip analytics (SPEC §6, extended): is this margin real, repeatable and tradeable?

A single snapshot of prices says what the spread is *now*; it cannot say whether it is a spike,
whether the item trades enough to fill an offer, or whether its price is inflated. This module
reads an item's history (hourly points for 7 days, 6-hour points for 30 days) and computes, in
plain code, the facts a flipper looks at:

* **Margin after tax, hour by hour**: how often it was positive, the typical value (median) and a
  conservative one (25th percentile), and how the current quote compares with them.
* **Volume**: units per hour on the slower side of the book, typical and in a bad hour.
* **Buy/sell ratio**: instant buyers vs instant sellers; near 1 means a balanced market.
* **Price level**: against the 7- and 30-day median and range (a bubble or a crash), and the
  last-24-hour move.

Every conclusion is a sentence built here from those numbers (``evidence`` and ``flags``), so an
answer can quote a fact instead of an opinion. Pure functions, no I/O, no model.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.clients.prices import ItemMeta, LatestPrice, Timeseries, TimeseriesPoint
from app.tools.tax import TaxPolicy

StrategyName = Literal["volume", "margin", "overnight", "dip"]
MIN_PAIRED_READINGS = 36  # fewer hourly readings with both sides than this: too thin to judge
NEAR_EXTREME = 0.9  # position in the 30-day range that counts as "near its high" (or low: 0.1)


# --------------------------------------------------------------------------- strategies
@dataclass(frozen=True)
class Strategy:
    name: StrategyName
    description: str
    min_bottleneck: int  # typical units/hour on the slower side of the book
    min_roi_pct: float  # typical margin after tax, as % of the buy price
    min_profitable: float  # share of hourly readings with a positive margin after tax
    window_hours: int  # how long the offers may take to fill
    min_margin: int  # conservative margin per unit in gp: 1 gp is one tick, any move erases it
    sort: Literal["profit", "roi"]


STRATEGIES: dict[str, Strategy] = {
    "volume": Strategy(
        "volume",
        "High-volume items: small margin per unit, filled quickly, many repeats. The price is "
        "hard for one trader to move.",
        min_bottleneck=200,
        min_roi_pct=1.0,
        min_profitable=0.70,
        window_hours=4,
        min_margin=2,
        sort="profit",
    ),
    "margin": Strategy(
        "margin",
        "Bigger margin per unit on items that trade less often: slower to fill, less competition.",
        min_bottleneck=20,
        min_roi_pct=3.0,
        min_profitable=0.60,
        window_hours=4,
        min_margin=2,
        sort="roi",
    ),
    "overnight": Strategy(
        "overnight",
        "Long offers you leave for several hours (before logging off): up to 8 hours of "
        "trading counts toward the amount you can fill.",
        min_bottleneck=100,
        min_roi_pct=1.0,
        min_profitable=0.70,
        window_hours=8,
        min_margin=2,
        sort="profit",
    ),
    "dip": Strategy(
        "dip",
        "Speculative: a liquid item trading below its usual price, bought in the hope that it "
        "returns to its 7-day median. It may be a real drop, not a dip.",
        min_bottleneck=150,
        min_roi_pct=-100.0,
        min_profitable=0.60,
        window_hours=4,
        min_margin=2,
        sort="profit",
    ),
}


# --------------------------------------------------------------------------- result model
class WorkedExample(BaseModel):
    units: int
    capital_gp: int
    buy_at: int
    sell_at: int
    tax_per_unit: int
    net_profit_per_unit: int
    profit_if_typical_margin: int
    profit_if_conservative_margin: int


class FlipAnalysis(BaseModel):
    item: str
    item_id: int
    members: bool
    strategy: str
    verdict: Literal["solid", "reasonable", "risky", "avoid"]
    confidence_pct: int = Field(description="how repeatable the flip looks, 0-100")
    buy_at: int = Field(description="where to place the buy offer: its usual level this week")
    sell_at: int = Field(description="where to place the sell offer: its usual level this week")
    margin_now_after_tax: int = Field(description="spread after tax at the latest quotes")
    margin_typical_after_tax: int
    margin_conservative_after_tax: int
    roi_typical_pct: float
    buy_limit_per_4h: int | None
    units: int = Field(description="what can realistically be traded in one window")
    limiting_factor: Literal["budget", "buy_limit", "market_volume"]
    capital_needed: int
    profit_per_window_conservative: int
    profit_per_window_typical: int
    window_hours: int
    est_hours_to_fill: float
    evidence: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    worked_example: WorkedExample
    metrics: dict[str, float | int | None] = Field(default_factory=dict)
    history: dict[str, str | int] = Field(default_factory=dict)
    score: float = 0.0


class Insufficient(Exception):
    """The history is too thin to analyse; ``reason`` is a stable slug."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


# --------------------------------------------------------------------------- helpers
def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _mid(point: TimeseriesPoint) -> float | None:
    if point.avg_high is not None and point.avg_low is not None:
        return (point.avg_high + point.avg_low) / 2
    return None


def _mean_between(points: list[tuple[int, float]], start: int, end: int) -> float | None:
    chosen = [v for t, v in points if start < t <= end]
    return statistics.fmean(chosen) if chosen else None


def _pct(new: float, old: float) -> float:
    return (new - old) / old * 100 if old else 0.0


def _range_position(value: float, values: list[float]) -> float:
    low, high = min(values), max(values)
    if high <= low:
        return 0.5
    return min(max((value - low) / (high - low), 0.0), 1.0)


@dataclass(frozen=True)
class _Level:
    median_mid: float
    pct_vs_median: float
    position: float


def _level(series: Timeseries | None, current_mid: float) -> _Level | None:
    if series is None:
        return None
    mids = [m for p in series.points if (m := _mid(p)) is not None]
    if len(mids) < 8:
        return None
    median = statistics.median(mids)
    return _Level(median, _pct(current_mid, median), _range_position(current_mid, mids))


# --------------------------------------------------------------------------- the analysis
def analyze(
    meta: ItemMeta,
    week: Timeseries,
    month: Timeseries | None,
    quote: LatestPrice,
    *,
    tax: TaxPolicy,
    strategy: Strategy,
    budget: int | None,
    market_share: float,
) -> FlipAnalysis:
    """Everything a flipper checks about one item, from its history and current quote."""
    if quote.low is None or quote.high is None or quote.low <= 0 or quote.high <= 0:
        raise Insufficient("no_two_sided_quote", "The item has no current buy and sell quote.")
    points = sorted(week.points, key=lambda p: p.timestamp)
    paired = [p for p in points if p.avg_high is not None and p.avg_low is not None]
    if len(paired) < MIN_PAIRED_READINGS:
        raise Insufficient(
            "thin_history",
            f"Only {len(paired)} hourly readings with both sides in the last 7 days.",
        )
    item_id = meta.id

    # margin after tax, per hourly reading (typical = median, conservative = 25th percentile)
    margins = [tax.net_estimate(item_id, p.avg_high or 0.0) - (p.avg_low or 0.0) for p in paired]
    positive = sum(m > 0 for m in margins) / len(margins)
    m50, m25 = statistics.median(margins), _quantile(margins, 0.25)
    typical_buy = statistics.median(p.avg_low or 0.0 for p in paired)
    typical_sell = statistics.median(p.avg_high or 0.0 for p in paired)
    now_net = tax.net(item_id, quote.high) - quote.low
    # Prices to offer at: the usual levels of the week (patient offers, consistent with the typical
    # margin). The dip strategy buys at today's low and sells back at the usual price.
    sell_at = round(typical_sell)
    buy_at = quote.low if strategy.name == "dip" else round(typical_buy)
    if buy_at <= 0 or sell_at <= 0:
        raise Insufficient("no_two_sided_quote", "The item has no usable usual price.")
    tax_unit = tax.tax(item_id, sell_at)
    usual_net = tax.net(item_id, sell_at) - buy_at
    if strategy.name != "dip":
        # one definition of "typical" everywhere: the margin at the prices we tell the player to
        # offer at; the conservative margin can never be above it
        m50 = float(usual_net)
        m25 = min(m25, m50)
    roi_typical = m50 / buy_at * 100

    # volume on the slower side of the book, buy/sell balance
    bottlenecks = [min(p.low_volume, p.high_volume) for p in points]
    b50, b10 = statistics.median(bottlenecks), _quantile([float(b) for b in bottlenecks], 0.10)
    sold_low = sum(p.low_volume for p in points)
    bought_high = sum(p.high_volume for p in points)
    ratio = bought_high / sold_low if sold_low else None

    # price level and movement
    current_mid = (quote.low + quote.high) / 2
    mids_t = [(p.timestamp, m) for p in paired if (m := _mid(p)) is not None]
    mids = [m for _, m in mids_t]
    week_level = _Level(
        statistics.median(mids),
        _pct(current_mid, statistics.median(mids)),
        _range_position(current_mid, mids),
    )
    volatility = statistics.pstdev(mids) / statistics.fmean(mids) * 100
    end_ts = points[-1].timestamp
    recent = _mean_between(mids_t, end_ts - 3 * 3_600, end_ts)
    day_ago = _mean_between(mids_t, end_ts - 27 * 3_600, end_ts - 24 * 3_600)
    change_24h = _pct(recent, day_ago) if recent and day_ago else None
    first_6 = _mean_between(mids_t, mids_t[0][0] - 1, mids_t[0][0] + 6 * 3_600)
    change_7d = _pct(recent, first_6) if recent and first_6 else None
    month_level = _level(month, current_mid)

    # what can realistically be traded, and how long that takes
    limit = meta.limit if meta.limit and meta.limit > 0 else None
    caps: list[tuple[Literal["budget", "buy_limit", "market_volume"], int]] = [
        ("market_volume", int(b50 * strategy.window_hours * market_share))
    ]
    if budget is not None:
        caps.append(("budget", budget // buy_at))
    if limit is not None:
        caps.append(("buy_limit", limit))
    limiting, units = min(caps, key=lambda c: c[1])
    units = max(units, 0)
    capture_per_hour = b50 * market_share
    fill_hours = units / capture_per_hour if capture_per_hour > 0 else float("inf")

    # confidence: how repeatable the flip looks (each factor in 0..1, multiplied)
    balance = _balance_factor(ratio)
    stability = max(0.5, 1 - volatility / 20)
    spike = _spike_factor(week_level.pct_vs_median)
    hype = 1.0
    if month_level is not None and month_level.position >= NEAR_EXTREME and (change_7d or 0) > 10:
        hype = 0.7
    reliability = positive * balance * stability * spike * hype

    evidence = _evidence(
        n=len(paired),
        tax_pct=float(tax.rate * 100),
        positive=positive,
        m50=m50,
        m25=m25,
        b50=b50,
        b10=b10,
        ratio=ratio,
        week_level=week_level,
        month_level=month_level,
        units=units,
        buy_at=buy_at,
        latest=(quote.low, quote.high, now_net),
        window_hours=strategy.window_hours,
        b50_window=b50 * strategy.window_hours,
        fill_hours=fill_hours,
    )
    flags = _flags(
        now_net=now_net,
        m50=m50,
        positive=positive,
        ratio=ratio,
        week_level=week_level,
        month_level=month_level,
        change_24h=change_24h,
        change_7d=change_7d,
        b50=b50,
        limit=limit,
        coverage=len(paired) / len(points),
        hype=hype < 1.0,
    )

    use_margin = max(m25, 0.0)
    typical_margin = max(m50, 0.0)
    score_margin = use_margin
    if strategy.name == "dip":
        # buy at today's low, sell back at the usual price: the speculative gain; the conservative
        # figure assumes the price only recovers half of the way
        typical_margin = max(float(usual_net), 0.0)
        use_margin = typical_margin / 2
        score_margin = use_margin
    profit_conservative = int(use_margin * units)
    profit_typical = int(typical_margin * units)
    if strategy.sort == "roi":
        score = roi_typical * reliability
    else:
        score = score_margin * units * reliability

    verdict = _verdict(positive, reliability, b50, m25, roi_typical, flags, strategy)
    return FlipAnalysis(
        item=meta.name,
        item_id=item_id,
        members=meta.members,
        strategy=strategy.name,
        verdict=verdict,
        confidence_pct=round(reliability * 100),
        buy_at=buy_at,
        sell_at=sell_at,
        margin_now_after_tax=now_net,
        margin_typical_after_tax=round(typical_margin),
        margin_conservative_after_tax=round(use_margin),
        roi_typical_pct=round(roi_typical, 2),
        buy_limit_per_4h=limit,
        units=units,
        limiting_factor=limiting,
        capital_needed=units * buy_at,
        profit_per_window_conservative=profit_conservative,
        profit_per_window_typical=profit_typical,
        window_hours=strategy.window_hours,
        est_hours_to_fill=round(fill_hours, 1) if fill_hours != float("inf") else 99.0,
        evidence=evidence,
        flags=flags,
        worked_example=WorkedExample(
            units=units,
            capital_gp=units * buy_at,
            buy_at=buy_at,
            sell_at=sell_at,
            tax_per_unit=tax_unit,
            net_profit_per_unit=usual_net,
            profit_if_typical_margin=profit_typical,
            profit_if_conservative_margin=profit_conservative,
        ),
        metrics={
            "profitable_hours_pct": round(positive * 100),
            "volume_typical_per_hour": round(b50),
            "volume_bad_hour_per_hour": round(b10),
            "buy_sell_ratio": round(ratio, 2) if ratio is not None else None,
            "volatility_pct": round(volatility, 1),
            "price_vs_7d_median_pct": round(week_level.pct_vs_median, 1),
            "median_price_7d": round(week_level.median_mid),
            "price_vs_30d_median_pct": round(month_level.pct_vs_median, 1) if month_level else None,
            "position_in_30d_range_pct": round(month_level.position * 100) if month_level else None,
            "change_24h_pct": round(change_24h, 1) if change_24h is not None else None,
            "change_7d_pct": round(change_7d, 1) if change_7d is not None else None,
        },
        history={
            "readings": len(paired),
            "from": _iso(points[0].timestamp),
            "to": _iso(points[-1].timestamp),
        },
        score=score,
    )


def _balance_factor(ratio: float | None) -> float:
    if ratio is None or ratio <= 0:
        return 0.5
    skew = max(ratio, 1 / ratio)  # 1 = perfectly balanced, 2 = one side twice the other
    if skew <= 1.4:
        return 1.0
    return max(0.5, 1.0 - (skew - 1.4) / 2.2)


def _spike_factor(pct_vs_median: float) -> float:
    distance = abs(pct_vs_median)
    if distance <= 5:
        return 1.0
    return 0.7 if distance <= 10 else 0.4


def _evidence(
    *,
    n: int,
    tax_pct: float,
    positive: float,
    m50: float,
    m25: float,
    b50: float,
    b10: float,
    ratio: float | None,
    week_level: _Level,
    month_level: _Level | None,
    units: int,
    buy_at: int,
    latest: tuple[int, int, int],
    window_hours: int,
    b50_window: float,
    fill_hours: float,
) -> list[str]:
    lines = [
        f"The margin after the {tax_pct:g}% tax was positive in {positive * 100:.0f}% of the last "
        f"{n} hourly readings (typical {m50:,.0f} gp, conservative {m25:,.0f} gp per unit).",
        f"Volume: typically {b50:,.0f} units per hour trade on the slower side of the book, and "
        f"in 9 of 10 hours at least {b10:,.0f}.",
    ]
    if ratio is not None:
        label = "balanced" if 0.7 <= ratio <= 1.4 else "one-sided"
        lines.append(f"Buy/sell ratio {ratio:.2f} over the week ({label}).")
    side = "above" if week_level.pct_vs_median >= 0 else "below"
    level = (
        f"The price is {abs(week_level.pct_vs_median):.1f}% {side} its 7-day median "
        f"({week_level.median_mid:,.0f} gp)"
    )
    if month_level is not None:
        level += f" and at {month_level.position * 100:.0f}% of its 30-day range (0% = low)"
    lines.append(level + ".")
    low, high, now_net = latest
    lines.append(
        f"Latest quotes: buy {low:,}, sell {high:,} gp, a spread of {now_net:,} gp after tax "
        f"right now (usual {m50:,.0f} gp)."
    )
    if units > 0 and b50_window > 0:
        share = units / b50_window * 100
        lines.append(
            f"{units:,} units at {buy_at:,} gp is {share:.0f}% of the typical {window_hours}-hour "
            f"volume: expected to fill in about {fill_hours:.1f} hours."
        )
    return lines


def _flags(
    *,
    now_net: int,
    m50: float,
    positive: float,
    ratio: float | None,
    week_level: _Level,
    month_level: _Level | None,
    change_24h: float | None,
    change_7d: float | None,
    b50: float,
    limit: int | None,
    coverage: float,
    hype: bool,
) -> list[str]:
    flags: list[str] = []
    if m50 > 0 and 0 < now_net < 0.5 * m50:
        flags.append(
            f"The spread right now ({now_net:,} gp after tax) is well below its usual "
            f"{m50:,.0f} gp: the margin has tightened, so wait for it to widen or expect less."
        )
    if now_net <= 0:
        flags.append(f"No profit at the current quote ({now_net:,} gp per unit after tax).")
    if positive < 0.6:
        flags.append(f"The margin was positive in only {positive * 100:.0f}% of the readings.")
    if ratio is not None and (ratio > 1.8 or ratio < 1 / 1.8):
        flags.append(f"The market is one-sided (buy/sell ratio {ratio:.2f}).")
    if week_level.pct_vs_median > 8:
        flags.append(
            f"The price is {week_level.pct_vs_median:.1f}% above its 7-day median: it could "
            "fall back."
        )
    if hype:
        flags.append(
            "Near its 30-day high after rising more than 10% this week: typical of hype that fades."
        )
    if month_level is not None and month_level.position <= 1 - NEAR_EXTREME:
        flags.append("Near its 30-day low: cheap, but check whether something changed its demand.")
    if change_24h is not None and change_24h <= -5:
        flags.append(f"The price fell {abs(change_24h):.1f}% in the last 24 hours.")
    if b50 < 50:
        flags.append(f"Thin market: {b50:,.0f} units per hour on the slower side.")
    if limit is None:
        flags.append("The buy limit is unknown, so the amount per window is not capped.")
    if coverage < 0.8:
        flags.append(f"It traded on both sides in only {coverage * 100:.0f}% of the hours.")
    return flags


def _verdict(
    positive: float,
    reliability: float,
    b50: float,
    m25: float,
    roi_typical: float,
    flags: list[str],
    strategy: Strategy,
) -> Literal["solid", "reasonable", "risky", "avoid"]:
    if roi_typical <= 0 or positive < 0.4 or b50 < 10:
        return "avoid"
    meets = (
        b50 >= strategy.min_bottleneck
        and roi_typical >= strategy.min_roi_pct
        and positive >= strategy.min_profitable
    )
    if meets and reliability >= 0.75 and m25 > 0 and not flags:
        return "solid"
    if meets and reliability >= 0.5 and m25 > 0:
        return "reasonable"
    return "risky"


def passes(analysis: FlipAnalysis, strategy: Strategy) -> bool:
    """Does the item meet the strategy's own thresholds (volume, margin, repeatability)?"""
    metrics = analysis.metrics
    if strategy.name == "dip":
        return (
            (metrics["price_vs_7d_median_pct"] or 0) <= -4
            and (metrics["volume_typical_per_hour"] or 0) >= strategy.min_bottleneck
            and (metrics["profitable_hours_pct"] or 0) >= strategy.min_profitable * 100
            and analysis.profit_per_window_typical > 0
        )
    return (
        (metrics["volume_typical_per_hour"] or 0) >= strategy.min_bottleneck
        and analysis.roi_typical_pct >= strategy.min_roi_pct
        and (metrics["profitable_hours_pct"] or 0) >= strategy.min_profitable * 100
        and analysis.margin_conservative_after_tax >= strategy.min_margin
        and analysis.units > 0
    )
