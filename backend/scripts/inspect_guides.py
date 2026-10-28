"""M3 acceptance check (SPEC §13.3): money-making guides, parsed fields next to their wikitext.

Prints N guides picked at random (``--seed``) so the structured data can be compared by hand with
the page it came from. ``--stratify`` forces a mix of per-kill, per-hour, recurring and
unsupported guides, which is where parsing bugs hide.

    uv run python -m scripts.inspect_guides [--n 10] [--seed 7] [--stratify]
    uv run python -m scripts.inspect_guides --title "Killing Zulrah"
"""

from __future__ import annotations

import argparse
import random
import re
import sys

from app.config import get_settings
from app.db import MissingDatabaseUrl, require_db_url
from app.ingest.dump import read_entry
from app.ingest.load import connect
from app.ingest.mmg import GuideItem, MoneyMakingGuide
from app.ingest.scope import SCOPES

SOURCE_CHARS = 1500


def item_line(item: GuideItem) -> str:
    qty = f"{item.qty:g}" + (" /h" if item.is_per_hour else " each")
    price = ""
    if item.fixed_value is not None:
        price = f"  fixed {item.fixed_value:g} gp"
    elif item.value_expr:
        price = f"  expr {item.value_expr}"
    elif item.cheapest_of:
        price = f"  cheapest of {', '.join(item.cheapest_of)}"
    ident = f"#{item.item_id}" if item.item_id is not None else "#?"
    return f"{item.name} ({ident}) x{qty}{price}"


def show(guide: MoneyMakingGuide, wikitext: str) -> None:
    print("=" * 110)
    print(f"{guide.page_title}   [{guide.kind}]   rankable={guide.rankable}")
    print(
        f"  category={guide.category} | members={guide.members_only} | "
        f"kph={guide.kph} ({guide.kph_label}) per_kill={guide.is_per_kill} | "
        f"reference profit/h={guide.source_hourly_profit}"
    )
    print(
        f"  skills={guide.skill_requirements} quests={guide.quest_requirements} "
        f"qp={guide.quest_points} combat={guide.combat_level} total={guide.total_level}"
    )
    for label, items in (("in", guide.items_in), ("out", guide.items_out)):
        for item in items[:8]:
            print(f"  {label:3} {item_line(item)}")
        if len(items) > 8:
            print(f"  {label:3} ... {len(items) - 8} more")
    if guide.unsupported:
        print(f"  UNSUPPORTED: {guide.unsupported}")
    if guide.issues:
        print(f"  issues: {guide.issues}")
    start = max(wikitext.find("{{Mmg"), 0)
    source = re.sub(r"\n{2,}", "\n", wikitext[start : start + SOURCE_CHARS])
    print("  --- wikitext ---")
    print("  " + source.replace("\n", "\n  "))


def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        url = require_db_url(settings)
    except MissingDatabaseUrl as exc:
        print(exc, file=sys.stderr)
        return 2
    with connect(url) as conn:
        rows = conn.execute("select data from moneymaking_guides order by page_title").fetchall()
    guides = [MoneyMakingGuide.model_validate(row[0]) for row in rows]
    sources = {
        p.title: p.wikitext
        for entry in SCOPES["v0"]
        if entry.page_type == "mmg"
        for p in read_entry(settings.data_dir, entry)
    }
    if args.title:
        picked = [g for g in guides if args.title.lower() in g.page_title.lower()]
    else:
        rng = random.Random(args.seed)
        if args.stratify:
            pools = [
                [g for g in guides if g.kind == "recurring" and g.rankable],
                [g for g in guides if g.kind == "standard" and g.is_per_kill and g.rankable],
                [g for g in guides if g.kind == "standard" and not g.is_per_kill and g.rankable],
                [g for g in guides if g.unsupported],
            ]
            picked = []
            for i in range(args.n):
                pool = pools[i % len(pools)]
                picked.append(rng.choice([g for g in pool if g not in picked]))
        else:
            picked = rng.sample(guides, args.n)
    for guide in picked:
        show(guide, sources.get(guide.page_title, ""))
    print("=" * 110)
    print(
        f"{len(picked)} of {len(guides)} guides shown; "
        f"{sum(g.rankable for g in guides)} rankable, "
        f"{sum(bool(g.unsupported) for g in guides)} unsupported"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--title", help="show every guide whose title contains this text")
    sys.exit(main(parser.parse_args()))
