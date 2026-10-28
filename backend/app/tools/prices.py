"""Price tools exposed to the LLM (SPEC §6): ``get_item_price`` and ``get_price_history``.

Thin I/O wrappers: resolve the item name, fetch (cached) data, and hand it to the pure code in
``flips.py`` / ``history.py``. They return plain JSON-able dicts; the LLM never computes numbers.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.clients.prices import LOOKBACKS
from app.tools.context import ToolContext
from app.tools.flips import iso
from app.tools.history import summarize_timeseries
from app.tools.items import FACE_VALUE, Resolution, resolution_error


def match_info(resolution: Resolution) -> dict[str, Any]:
    info: dict[str, Any] = {"status": resolution.status}
    if resolution.note:
        info["note"] = resolution.note
    return info


def _quote_side(price: int | None, ts: int | None, now_ts: int) -> dict[str, Any] | None:
    if price is None:
        return None
    return {
        "price": price,
        "as_of": iso(ts),
        "age_seconds": now_ts - ts if ts else None,
    }


async def get_item_price(ctx: ToolContext, item: str) -> dict[str, Any]:
    bundle = await ctx.bundle()
    resolution = bundle.index.resolve(item)
    if resolution.item is None:
        return resolution_error(resolution)
    meta = resolution.item
    base: dict[str, Any] = {"item": meta.name, "item_id": meta.id, "match": match_info(resolution)}

    if meta.id in FACE_VALUE:
        return {
            **base,
            "face_value_gp": FACE_VALUE[meta.id],
            "note": "Not tradeable on the Grand Exchange; it is worth its face value.",
        }

    (quote, fetched_at, stale), hourly = await asyncio.gather(
        ctx.prices.latest_item(meta.id), ctx.prices.one_hour()
    )
    now_ts = int(ctx.now().timestamp())
    result: dict[str, Any] = {
        **base,
        "members": meta.members,
        "buy_limit_per_4h": meta.limit,
        "high_alchemy_value": meta.highalch,
        "low_alchemy_value": meta.lowalch,
        "store_value": meta.value,
        "prices_fetched_at": fetched_at.isoformat().replace("+00:00", "Z"),
        "stale": stale or hourly.stale,
    }
    if quote is None or (quote.high is None and quote.low is None):
        return {**result, "no_price_data": True, "message": "This item has no recent trades."}

    instant_buy = _quote_side(quote.high, quote.high_time, now_ts)  # `high` in the Wiki API
    instant_sell = _quote_side(quote.low, quote.low_time, now_ts)  # `low` in the Wiki API
    result["instant_buy_price"] = instant_buy
    result["instant_sell_price"] = instant_sell
    if quote.high is not None and quote.low is not None:
        result["spread"] = quote.high - quote.low

    last_hour = hourly.data.get(meta.id)
    if last_hour is not None:
        result["last_hour"] = {
            "window_start": iso(hourly.api_timestamp),
            "avg_instant_buy_price": last_hour.avg_high,
            "avg_instant_sell_price": last_hour.avg_low,
            "units_bought_instantly": last_hour.high_volume,
            "units_sold_instantly": last_hour.low_volume,
        }
    else:
        result["last_hour"] = None
    return result


async def get_price_history(ctx: ToolContext, item: str, lookback: str = "7d") -> dict[str, Any]:
    if lookback not in LOOKBACKS:
        return {
            "error": "invalid_lookback",
            "message": f"lookback must be one of {', '.join(LOOKBACKS)}",
        }
    bundle = await ctx.bundle()
    resolution = bundle.index.resolve(item)
    if resolution.item is None:
        return resolution_error(resolution)
    meta = resolution.item
    if meta.id in FACE_VALUE:
        return {
            "error": "not_tradeable",
            "item": meta.name,
            "message": "Not tradeable on the Grand Exchange; it has no price history.",
        }
    series = await ctx.prices.timeseries(meta.id, lookback)
    summary = summarize_timeseries(series, meta.name)
    summary["match"] = match_info(resolution)
    return summary
