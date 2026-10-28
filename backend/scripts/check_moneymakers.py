"""Live checks of ``rank_moneymakers`` (SPEC §8) against the real prices and the ingested guides.

    uv run python -m scripts.check_moneymakers --crosscheck            # vs the wiki's own figures
    uv run python -m scripts.check_moneymakers --profile main_midgame  # a ranking for a profile

``--crosscheck`` values every guide for a maxed player and compares the result with the profit the
wiki itself lists (``source_hourly_profit``, refreshed daily by the wiki's own Lua module). With the
default ``mid`` price basis the two agree to within a few percent; a large gap points at a parser
or valuation bug, or at a price that moved since the wiki's last refresh.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys

import httpx
import yaml

from app.api.schemas import SKILL_NAMES, PlayerState
from app.clients.prices import PricesClient
from app.config import get_settings
from app.db import MissingDatabaseUrl, async_connection, require_db_url
from app.tools.context import ToolContext
from app.tools.moneymakers import GuideStore, Market, MoneyConfig, rank_guides
from app.tools.registry import run_tool
from eval.cases import QUESTIONS_PATH

MIN_REFERENCE = 50_000  # below this the wiki's figure is mostly noise from rounding


async def crosscheck(ctx: ToolContext, store: GuideStore, basis: str) -> int:
    settings = ctx.settings.model_copy(update={"mm_price_basis": basis})
    maxed = PlayerState(
        account_type="main",
        members=True,
        skills=dict.fromkeys(SKILL_NAMES, 99),
        quest_points=300,
        quests={},
    )
    bundle = await ctx.bundle()
    latest, hourly = await asyncio.gather(ctx.prices.latest(), ctx.prices.one_hour())
    market = Market(
        latest=latest.data,
        hourly=hourly.data,
        index=bundle.index,
        tax=bundle.tax,
        now=ctx.now(),
        cfg=MoneyConfig.from_settings(settings),
    )
    result = rank_guides(store.guides, maxed, market, top_n=10_000)
    rows = [
        r
        for r in result.ranked
        if r.reference_profit_per_hour
        and r.reference_profit_per_hour > MIN_REFERENCE
        and r.pace is None
    ]
    if not rows:
        print("no guide with a reference figure could be valued", file=sys.stderr)
        return 1
    ratios = sorted(r.profit_per_hour / r.reference_profit_per_hour for r in rows)  # type: ignore[operator]
    n = len(ratios)
    print(
        f"basis={basis}: {len(result.ranked)} continuous guides valued, {n} comparable "
        f"(wiki figure > {MIN_REFERENCE:,} gp/h, not market-limited)"
    )
    print(
        f"recomputed / wiki: median {statistics.median(ratios):.3f}, "
        f"p10 {ratios[n // 10]:.3f}, p90 {ratios[9 * n // 10]:.3f}"
    )
    print(
        f"within +-10%: {sum(0.9 <= x <= 1.1 for x in ratios)}/{n}; "
        f"within +-25%: {sum(0.75 <= x <= 1.25 for x in ratios)}/{n}"
    )
    worst = sorted(rows, key=lambda r: -abs(r.profit_per_hour / r.reference_profit_per_hour - 1))  # type: ignore[operator]
    print("largest gaps:")
    for r in worst[:8]:
        ratio = r.profit_per_hour / r.reference_profit_per_hour  # type: ignore[operator]
        print(
            f"  {r.title[:52]:52} mine {r.profit_per_hour:>11,}  wiki "
            f"{r.reference_profit_per_hour:>11,}  x{ratio:.2f}"
        )
    print("not valued:", dict(sorted(result.excluded.items())))
    return 0


async def show_profile(ctx: ToolContext, name: str) -> int:
    profiles = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))["profiles"]
    if name not in profiles:
        print(f"unknown profile {name!r}; choose from {', '.join(profiles)}", file=sys.stderr)
        return 2
    out = await run_tool(
        ctx, "rank_moneymakers", {"top_n": 8}, player=PlayerState.model_validate(profiles[name])
    )
    print(f"status: {out.get('status', out.get('error'))} | account: {out.get('account')}")
    print(f"prices fetched at {out.get('prices_fetched_at')} | basis: {out.get('price_basis')}")
    for section in ("ranking", "recurring", "almost_eligible", "needs_more_gp", "eligible_guides"):
        for row in out.get(section, []):
            figure = (
                f"{row['profit_per_hour']:,} gp/h"
                if "profit_per_hour" in row
                else f"{row['profit_per_run']:,} gp/run"
                if "profit_per_run" in row
                else "no profit figure"
            )
            print(f"[{section}] {row['title']} ({row.get('category')}): {figure}")
            for key in ("missing", "caveats", "warnings"):
                if row.get(key):
                    print(f"      {key}: {row[key]}")
    print("excluded:", out.get("excluded"), "| eligible:", out.get("eligible_total"))
    return 0


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        url = require_db_url(settings)
    except MissingDatabaseUrl as exc:
        print(exc, file=sys.stderr)
        return 2
    async with httpx.AsyncClient(timeout=settings.http_timeout_s) as http:
        store = await GuideStore.load(lambda: async_connection(url))
        ctx = ToolContext(settings, PricesClient(settings, http=http), guides=store)
        if args.crosscheck:
            return await crosscheck(ctx, store, args.basis)
        return await show_profile(ctx, args.profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--crosscheck", action="store_true")
    parser.add_argument("--basis", choices=["mid", "patient", "instant"], default="mid")
    parser.add_argument("--profile", default="main_midgame", help="a profile from questions.yaml")
    sys.exit(asyncio.run(main(parser.parse_args())))
