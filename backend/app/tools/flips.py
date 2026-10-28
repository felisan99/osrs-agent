"""Flipping maths (SPEC §6): deterministic, no I/O, no LLM.

Terminology (Wiki prices API semantics):

* ``high``  = price at which an instant BUY happens   -> where a flip SELLS (patient sell offer).
* ``low``   = price at which an instant SELL happens  -> where a flip BUYS  (patient buy offer).
* A typical flip buys near ``low`` and sells near ``high``; the GE tax comes off the sale.
* ``/1h`` ``lowPriceVolume`` is what fills our buy offer; ``highPriceVolume`` fills our sell offer,
  so liquidity is bounded by the smaller of the two.
* The GE buy limit resets every 4 hours; it caps how many units can be bought per window.

Everything here takes plain data and returns pydantic models, so each number can be checked by
hand in tests. The I/O wrappers (fetching, item resolution) live in ``app/tools/prices.py``.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from app.clients.prices import AveragePrice, ItemMeta, LatestPrice, Timeseries
from app.config import Settings
from app.errors import UpstreamError
from app.tools.flip_analysis import (
    STRATEGIES,
    FlipAnalysis,
    Insufficient,
    Strategy,
    analyze,
    passes,
)
from app.tools.items import FACE_VALUE, resolution_error
from app.tools.tax import TaxPolicy

if TYPE_CHECKING:
    from app.tools.context import ToolContext

BUY_LIMIT_WINDOW_HOURS = 4


class NoQuote(ValueError):
    """The item has no usable instant buy/sell quote; ``reason`` is a stable slug."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class FlipConfig:
    max_price_age_s: int = 3600
    max_roi: float = 0.50
    low_volume_warn: int = 50
    market_share: float = 0.25  # realistic share of the 4h traded volume one flipper captures

    @classmethod
    def from_settings(cls, settings: Settings) -> FlipConfig:
        return cls(
            max_price_age_s=settings.flip_max_price_age_s,
            max_roi=settings.flip_max_roi,
            low_volume_warn=settings.flip_low_volume_warn,
            market_share=settings.flip_market_share,
        )


def iso(ts: int | None) -> str | None:
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z") if ts else None


def unit_economics(
    item_id: int | None, buy: int, sell: int, tax: TaxPolicy
) -> tuple[int, int, int]:
    """``(tax_per_unit, net_sell_price, margin_per_unit)`` for a buy/sell pair."""
    tax_unit = tax.tax(item_id, sell)
    net = sell - tax_unit
    return tax_unit, net, net - buy


# ------------------------------------------------------------------------------ calc_flip
class Volume(BaseModel):
    buy_side_units_per_hour: int = Field(
        description="units sold instantly at `low` (fills our buy)"
    )
    sell_side_units_per_hour: int = Field(
        description="units bought instantly at `high` (fills our sell)"
    )
    bottleneck_units_per_hour: int
    typical_units_per_4h_window: int
    window_start: str | None = None


class FlipResult(BaseModel):
    item_id: int
    item: str
    members: bool
    buy_price: int = Field(description="`low`: where the flip buys")
    sell_price: int = Field(description="`high`: where the flip sells")
    tax_per_unit: int
    tax_exempt: bool
    net_sell_price: int
    margin_per_unit: int
    roi_pct: float
    buy_limit_per_4h: int | None
    capital_for_full_limit: int | None
    max_profit_per_limit_window: int | None
    # Only when a quantity was asked for:
    qty_requested: int | None = None
    qty_this_window: int | None = None
    windows_needed: int | None = None
    cost_this_window: int | None = None
    profit_this_window: int | None = None
    profit_for_all_qty: int | None = None
    volume: Volume | None = None
    buy_quote_time: str | None = None
    sell_quote_time: str | None = None
    buy_quote_age_s: int | None = None
    sell_quote_age_s: int | None = None
    warnings: list[str] = Field(default_factory=list)


def _volume(hourly: AveragePrice | None, hourly_window_start: int | None) -> Volume | None:
    if hourly is None:
        return None
    bottleneck = min(hourly.low_volume, hourly.high_volume)
    return Volume(
        buy_side_units_per_hour=hourly.low_volume,
        sell_side_units_per_hour=hourly.high_volume,
        bottleneck_units_per_hour=bottleneck,
        typical_units_per_4h_window=bottleneck * BUY_LIMIT_WINDOW_HOURS,
        window_start=iso(hourly_window_start),
    )


def _quote(latest: LatestPrice | None) -> tuple[int, int]:
    if latest is None:
        raise NoQuote("no_quote", "no price data for this item")
    if latest.high is None or latest.low is None:
        side = "instant-buy (high)" if latest.high is None else "instant-sell (low)"
        raise NoQuote("one_sided", f"this item has no {side} price: it has not traded on that side")
    if latest.low <= 0 or latest.high <= 0:
        raise NoQuote("invalid_price", "non-positive price in the quote")
    return latest.low, latest.high


def compute_flip(
    meta: ItemMeta,
    latest: LatestPrice | None,
    hourly: AveragePrice | None,
    *,
    tax: TaxPolicy,
    now: datetime,
    cfg: FlipConfig,
    qty: int | None = None,
    hourly_window_start: int | None = None,
) -> FlipResult:
    buy, sell = _quote(latest)
    assert latest is not None  # narrowed by _quote
    tax_unit, net, margin = unit_economics(meta.id, buy, sell, tax)
    roi = margin / buy
    limit = meta.limit if meta.limit and meta.limit > 0 else None
    volume = _volume(hourly, hourly_window_start)

    now_ts = int(now.timestamp())
    buy_age = now_ts - latest.low_time if latest.low_time else None
    sell_age = now_ts - latest.high_time if latest.high_time else None

    warnings: list[str] = []
    if margin <= 0:
        warnings.append(
            f"No profit after tax: selling at {sell:,} nets {net:,} per unit vs {buy:,} to buy."
        )
    if tax_unit == 0 and tax.tax(None, sell) > 0:
        warnings.append("This item is exempt from the Grand Exchange tax.")
    if roi > cfg.max_roi:
        warnings.append(
            f"ROI of {roi * 100:.0f}% is unusually high: the quotes are probably stale, "
            "manipulated or too illiquid to trade at these prices."
        )
    for label, age in (("buy", buy_age), ("sell", sell_age)):
        if age is not None and age > cfg.max_price_age_s:
            warnings.append(f"The {label} quote is {age // 60} minutes old; prices may have moved.")
    if volume is None:
        warnings.append("No hourly volume data: this item barely traded in the last hour.")
    elif volume.bottleneck_units_per_hour < cfg.low_volume_warn:
        warnings.append(
            f"Low liquidity: only {volume.bottleneck_units_per_hour} units/hour trade on the "
            "slower side of the book, so offers may take long to fill."
        )
    if limit is None:
        warnings.append(
            "The buy limit of this item is unknown; quantity per window can't be capped."
        )

    result = FlipResult(
        item_id=meta.id,
        item=meta.name,
        members=meta.members,
        buy_price=buy,
        sell_price=sell,
        tax_per_unit=tax_unit,
        tax_exempt=meta.id in tax.exempt_ids,
        net_sell_price=net,
        margin_per_unit=margin,
        roi_pct=round(roi * 100, 2),
        buy_limit_per_4h=limit,
        capital_for_full_limit=buy * limit if limit else None,
        max_profit_per_limit_window=margin * limit if limit else None,
        volume=volume,
        buy_quote_time=iso(latest.low_time),
        sell_quote_time=iso(latest.high_time),
        buy_quote_age_s=buy_age,
        sell_quote_age_s=sell_age,
        warnings=warnings,
    )

    if qty is not None:
        this_window = min(qty, limit) if limit else qty
        result.qty_requested = qty
        result.qty_this_window = this_window
        result.windows_needed = math.ceil(qty / limit) if limit else None
        result.cost_this_window = buy * this_window
        result.profit_this_window = margin * this_window
        result.profit_for_all_qty = margin * qty
        if limit and qty > limit:
            result.warnings.append(
                f"Buy limit: only {limit} units can be bought per 4 hours, so {qty} units "
                f"take {result.windows_needed} windows."
            )
        if volume and volume.typical_units_per_4h_window > 0:
            share = this_window / volume.typical_units_per_4h_window
            if share > cfg.market_share:
                result.warnings.append(
                    f"{this_window} units is {share * 100:.0f}% of the typical 4-hour market "
                    "volume; expect slow fills."
                )
    return result


# ------------------------------------------------------------------------------ rank_flips
class RankedFlip(BaseModel):
    rank: int
    item_id: int
    item: str
    members: bool
    buy_price: int
    sell_price: int
    margin_per_unit: int
    roi_pct: float
    buy_limit_per_4h: int
    units_this_window: int = Field(
        description="min(budget, buy limit, market-share cap): what can realistically be traded"
    )
    limiting_factor: Literal["budget", "buy_limit", "market_volume"]
    capital_needed: int
    profit_per_limit_window: int
    estimated_profit_per_hour: int
    bottleneck_units_per_hour: int
    warnings: list[str] = Field(default_factory=list)


class RankResult(BaseModel):
    max_gp: int
    members: bool
    min_volume_per_hour: int
    market_share_assumed: float
    items_evaluated: int
    excluded: dict[str, int]
    rows: list[RankedFlip]
    hourly_window_start: str | None = None
    note: str = (
        "Ranked by profit per 4-hour window, buying near the instant-sell price (`low`) and "
        "selling near the instant-buy price (`high`) after tax. Units per window = min(budget, "
        "buy limit, `market_share_assumed` of the typical traded volume); `limiting_factor` says "
        "which one binds."
    )


def rank_candidates(
    mapping: Mapping[int, ItemMeta],
    latest: Mapping[int, LatestPrice],
    hourly: Mapping[int, AveragePrice],
    *,
    tax: TaxPolicy,
    now: datetime,
    cfg: FlipConfig,
    max_gp: int,
    members: bool,
    min_volume: int,
    top_n: int,
    hourly_window_start: int | None = None,
    min_roi_pct: float = 0.0,
) -> RankResult:
    """Rank every tradeable item by profit per buy-limit window, within the available GP."""
    if max_gp <= 0:
        raise ValueError("max_gp must be positive")
    now_ts = int(now.timestamp())
    excluded: dict[str, int] = {}

    def skip(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    evaluated = 0
    candidates: list[RankedFlip] = []
    for item_id, quote in latest.items():
        meta = mapping.get(item_id)
        if meta is None:
            skip("not_in_mapping")
            continue
        evaluated += 1
        if meta.members and not members:
            skip("members_only")
            continue
        if quote.high is None or quote.low is None or quote.low <= 0 or quote.high <= 0:
            skip("no_two_sided_quote")
            continue
        if quote.low > max_gp:
            skip("unaffordable")
            continue
        if not meta.limit or meta.limit <= 0:
            skip("no_buy_limit")
            continue
        row_volume = hourly.get(item_id)
        if row_volume is None:
            skip("no_hourly_volume")
            continue
        bottleneck = min(row_volume.low_volume, row_volume.high_volume)
        if bottleneck < min_volume:
            skip("low_volume")
            continue
        times = (quote.high_time, quote.low_time)
        if any(t is None or now_ts - t > cfg.max_price_age_s for t in times):
            skip("stale_quote")
            continue
        _, _, margin = unit_economics(item_id, quote.low, quote.high, tax)
        if margin <= 0:
            skip("no_profit_after_tax")
            continue
        roi = margin / quote.low
        if roi > cfg.max_roi:
            skip("implausible_roi")
            continue
        if roi * 100 < min_roi_pct:
            skip("margin_too_thin")
            continue

        # What can realistically be traded in one window: the budget, the buy limit, and the share
        # of the market's traded volume one flipper can capture. Ties resolve in this order.
        caps = (
            ("budget", max_gp // quote.low),
            ("buy_limit", meta.limit),
            ("market_volume", int(bottleneck * BUY_LIMIT_WINDOW_HOURS * cfg.market_share)),
        )
        limiting_factor, units = min(caps, key=lambda cap: cap[1])
        if units < 1:
            skip("no_liquidity")
            continue
        profit_window = margin * units
        warnings: list[str] = []
        if bottleneck < cfg.low_volume_warn:
            warnings.append(f"Low liquidity ({bottleneck} units/hour on the slower side).")
        candidates.append(
            RankedFlip(
                rank=0,
                item_id=item_id,
                item=meta.name,
                members=meta.members,
                buy_price=quote.low,
                sell_price=quote.high,
                margin_per_unit=margin,
                roi_pct=round(roi * 100, 2),
                buy_limit_per_4h=meta.limit,
                units_this_window=units,
                limiting_factor=limiting_factor,  # type: ignore[arg-type]
                capital_needed=quote.low * units,
                profit_per_limit_window=profit_window,
                estimated_profit_per_hour=profit_window // BUY_LIMIT_WINDOW_HOURS,
                bottleneck_units_per_hour=bottleneck,
                warnings=warnings,
            )
        )

    candidates.sort(key=lambda r: (-r.profit_per_limit_window, -r.roi_pct, r.item_id))
    rows = candidates[:top_n]
    for position, row in enumerate(rows, start=1):
        row.rank = position
    return RankResult(
        max_gp=max_gp,
        members=members,
        min_volume_per_hour=min_volume,
        market_share_assumed=cfg.market_share,
        items_evaluated=evaluated,
        excluded=dict(sorted(excluded.items())),
        rows=rows,
        hourly_window_start=iso(hourly_window_start),
    )


# ------------------------------------------------------------------------------ tool entry points
def _with_player_prices(
    quote: LatestPrice | None, buy_price: int | None, sell_price: int | None, now_ts: int
) -> LatestPrice:
    """The market quote with the sides the player specified replaced (as fresh as of now)."""
    base = quote or LatestPrice()
    return LatestPrice(
        high=sell_price if sell_price is not None else base.high,
        high_time=now_ts if sell_price is not None else base.high_time,
        low=buy_price if buy_price is not None else base.low,
        low_time=now_ts if buy_price is not None else base.low_time,
    )


def _tax_facts(tax: TaxPolicy) -> dict[str, Any]:
    """The tax rule in force, so an answer quotes it instead of remembering it (it changes)."""
    return {"tax_rate_pct": float(tax.rate * 100), "tax_cap_per_item": tax.cap}


def _stamp(fetched_at: datetime) -> str:
    return fetched_at.isoformat().replace("+00:00", "Z")


async def calc_flip(
    ctx: ToolContext,
    item: str,
    qty: int | None = None,
    buy_price: int | None = None,
    sell_price: int | None = None,
) -> dict[str, Any]:
    """Flip economics for one item (SPEC §6 ``calc_flip``).

    ``buy_price`` / ``sell_price`` replace the market quote of that side when the player states a
    price ("what tax would I pay selling at 1.5M?"); the result says which prices were theirs.
    """
    bundle = await ctx.bundle()
    resolution = bundle.index.resolve(item)
    if resolution.item is None:
        return resolution_error(resolution)
    meta = resolution.item
    if meta.id in FACE_VALUE:
        return {
            "error": "not_tradeable",
            "item": meta.name,
            "message": "Not tradeable on the Grand Exchange, so it cannot be flipped.",
        }

    (quote, fetched_at, stale), hourly = await asyncio.gather(
        ctx.prices.latest_item(meta.id), ctx.prices.one_hour()
    )
    if buy_price is not None or sell_price is not None:
        quote = _with_player_prices(quote, buy_price, sell_price, int(ctx.now().timestamp()))
    try:
        result = compute_flip(
            meta,
            quote,
            hourly.data.get(meta.id),
            tax=bundle.tax,
            now=ctx.now(),
            cfg=FlipConfig.from_settings(ctx.settings),
            qty=qty,
            hourly_window_start=hourly.api_timestamp,
        )
    except NoQuote as exc:
        return {"error": exc.reason, "item": meta.name, "item_id": meta.id, "message": str(exc)}

    out = result.model_dump(mode="json", exclude_none=True)
    match: dict[str, Any] = {"status": resolution.status}
    if resolution.note:
        match["note"] = resolution.note
    out["match"] = match
    out.update(_tax_facts(bundle.tax))
    if buy_price is not None or sell_price is not None:
        out["price_source"] = {
            "buy": "player" if buy_price is not None else "market",
            "sell": "player" if sell_price is not None else "market",
        }
        out["warnings"] = [
            *out.get("warnings", []),
            "Prices the player gave were used for: "
            + ", ".join(side for side, p in (("buy", buy_price), ("sell", sell_price)) if p),
        ]
    out["prices_fetched_at"] = _stamp(fetched_at)
    out["stale"] = stale or hourly.stale
    return out


STAGE_ONE_CANDIDATES = 40  # items whose history is fetched (7 days each)
STAGE_TWO_CANDIDATES = 12  # of those, items that also get the 30-day view
HISTORY_CONCURRENCY = 10
OFFER_SLOTS = {True: 8, False: 3}  # Grand Exchange offers at once: members / free-to-play


async def _fetch_history(
    ctx: ToolContext, item_ids: list[int], lookback: str
) -> dict[int, Timeseries]:
    """History for several items at once; an item whose request fails is simply absent."""
    gate = asyncio.Semaphore(HISTORY_CONCURRENCY)

    async def one(item_id: int) -> tuple[int, Timeseries | None]:
        async with gate:
            try:
                return item_id, await ctx.prices.timeseries(item_id, lookback)
            except UpstreamError:
                return item_id, None

    return {i: s for i, s in await asyncio.gather(*(one(i) for i in item_ids)) if s is not None}


def _analysis_notes(strategy: Strategy, tax: TaxPolicy) -> list[str]:
    return [
        f"Strategy `{strategy.name}`: {strategy.description}",
        "Each row is built from the item's last 7 days (hourly) and 30 days: `evidence` are "
        "facts computed from that history, `flags` are risks. Quote them, do not invent others.",
        "Prices to use: buy offer at `buy_at`, sell offer at `sell_at`. `profit_per_window_"
        "conservative` uses the 25th-percentile margin of the week (a bad-but-normal hour); "
        "`..._typical` uses the median. Neither is a guarantee.",
        f"Margins are after the {float(tax.rate * 100):g}% Grand Exchange tax (cap {tax.cap:,} "
        "per item). History cannot see manipulation or upcoming game updates.",
    ]


async def rank_flips(
    ctx: ToolContext,
    max_gp: int,
    members: bool = True,
    min_volume: int | None = None,
    top_n: int = 5,
    strategy: str = "volume",
) -> dict[str, Any]:
    """Rank flips over the whole market within a GP budget (SPEC §6 ``rank_flips``).

    Two stages: a snapshot of every item picks the candidates that could work now; then the
    history of each (7 days, and 30 days for the best) says whether the margin is real and
    repeatable. Rows are ordered by the strategy's own score, with facts that justify them.
    """
    chosen = STRATEGIES.get(strategy)
    if chosen is None:
        return {
            "error": "invalid_strategy",
            "message": f"strategy must be one of {', '.join(STRATEGIES)}",
        }
    bundle = await ctx.bundle()
    latest, hourly, mapping = await asyncio.gather(
        ctx.prices.latest(), ctx.prices.one_hour(), ctx.prices.mapping()
    )
    cfg = FlipConfig.from_settings(ctx.settings)
    stage_one = rank_candidates(
        mapping.data,
        latest.data,
        hourly.data,
        tax=bundle.tax,
        now=ctx.now(),
        cfg=cfg,
        max_gp=max_gp,
        members=members,
        min_volume=chosen.min_bottleneck if min_volume is None else min_volume,
        top_n=STAGE_ONE_CANDIDATES,
        hourly_window_start=hourly.api_timestamp,
        min_roi_pct=max(chosen.min_roi_pct, 0.0),
    )
    excluded = dict(stage_one.excluded)

    def skip(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    ids = [row.item_id for row in stage_one.rows]
    weeks = await _fetch_history(ctx, ids, "7d")
    if ids and not weeks:
        return {
            "error": "price_data_unavailable",
            "message": "The price history could not be fetched; do not give flipping figures.",
        }

    def run(item_id: int, month: Timeseries | None) -> FlipAnalysis | None:
        try:
            return analyze(
                mapping.data[item_id],
                weeks[item_id],
                month,
                latest.data[item_id],
                tax=bundle.tax,
                strategy=chosen,
                budget=max_gp,
                market_share=cfg.market_share,
            )
        except Insufficient as exc:
            skip(exc.reason)
            return None

    first = [a for i in ids if i in weeks and (a := run(i, None)) is not None]
    skipped_history = len(ids) - len(weeks)
    if skipped_history:
        excluded["history_unavailable"] = skipped_history
    shortlist = sorted(first, key=lambda a: -a.score)[:STAGE_TWO_CANDIDATES]
    months = await _fetch_history(ctx, [a.item_id for a in shortlist], "30d")
    final: list[FlipAnalysis] = []
    for a in shortlist:
        refined = run(a.item_id, months.get(a.item_id)) or a
        if passes(refined, chosen):
            final.append(refined)
        else:
            skip("below_strategy_thresholds")
    final.sort(key=lambda a: -a.score)
    rows = []
    for position, a in enumerate(final[:top_n], start=1):
        row = a.model_dump(mode="json", exclude={"score", "metrics", "history"})
        rows.append({"rank": position, **row})
    out: dict[str, Any] = {
        "strategy": chosen.name,
        "max_gp": max_gp,
        "members": members,
        "offer_slots": OFFER_SLOTS[members],
        "items_screened": stage_one.items_evaluated,
        "items_with_history_analysed": len(first),
        "excluded": dict(sorted(excluded.items())),
        "rows": rows,
        "notes": _analysis_notes(chosen, bundle.tax),
    }
    out.update(_tax_facts(bundle.tax))
    out["prices_fetched_at"] = _stamp(latest.fetched_at)
    out["stale"] = latest.stale or hourly.stale
    return out


async def analyze_flip(ctx: ToolContext, item: str, budget: int | None = None) -> dict[str, Any]:
    """Is ONE item a good flip? Its history says whether the margin is repeatable and tradeable."""
    bundle = await ctx.bundle()
    resolution = bundle.index.resolve(item)
    if resolution.item is None:
        return resolution_error(resolution)
    meta = resolution.item
    if meta.id in FACE_VALUE:
        return {
            "error": "not_tradeable",
            "item": meta.name,
            "message": "Not tradeable on the Grand Exchange, so it cannot be flipped.",
        }
    (quote, fetched_at, stale), week, month = await asyncio.gather(
        ctx.prices.latest_item(meta.id),
        ctx.prices.timeseries(meta.id, "7d"),
        ctx.prices.timeseries(meta.id, "30d"),
    )
    try:
        result = analyze(
            meta,
            week,
            month,
            quote or LatestPrice(),
            tax=bundle.tax,
            strategy=STRATEGIES["volume"],
            budget=budget,
            market_share=FlipConfig.from_settings(ctx.settings).market_share,
        )
    except Insufficient as exc:
        return {"error": exc.reason, "item": meta.name, "item_id": meta.id, "message": str(exc)}
    out = result.model_dump(mode="json", exclude={"score"})
    out["match"] = {
        "status": resolution.status,
        **({"note": resolution.note} if resolution.note else {}),
    }
    out["notes"] = _analysis_notes(STRATEGIES["volume"], bundle.tax)[1:]
    out.update(_tax_facts(bundle.tax))
    out["prices_fetched_at"] = _stamp(fetched_at)
    out["stale"] = stale or week.stale or month.stale
    return out
