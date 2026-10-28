"""Housekeeping of the conversation memory (docs/api-m6.md, app/graph/retention.py).

    uv run python -m scripts.maintenance stats
    uv run python -m scripts.maintenance prune                  # keep-latest on every thread
    uv run python -m scripts.maintenance prune --older-than-days 7   # ...and delete idle threads

The API prunes a thread after every turn and sweeps idle threads by itself (``THREAD_TTL_DAYS``);
this is for cleaning a database that grew before that, or on demand.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.config import get_settings
from app.db import MissingDatabaseUrl, async_connection, require_db_url
from app.graph.retention import delete_stale_threads, prune_thread, stats


def megabytes(value: int) -> str:
    return f"{value / 1_000_000:.1f} MB"


async def run(args: argparse.Namespace) -> int:
    try:
        url = require_db_url(get_settings())
    except MissingDatabaseUrl as exc:
        print(exc, file=sys.stderr)
        return 2

    def connect():  # a fresh short-lived connection per operation
        return async_connection(url)

    before = await stats(connect)
    print(
        f"memory: {before['threads']} threads, {before['checkpoints']} checkpoints, "
        f"{before['checkpoint_blobs']} blobs, {before['checkpoint_writes']} writes, "
        f"{megabytes(before['bytes'])}"
    )
    if args.command == "stats":
        return 0
    if args.older_than_days:
        deleted = await delete_stale_threads(connect, args.older_than_days)
        print(f"deleted {deleted} threads idle for more than {args.older_than_days} days")
    async with connect() as conn:
        rows = await (await conn.execute("select distinct thread_id from checkpoints")).fetchall()
    removed = 0
    for (thread_id,) in rows:
        removed += (await prune_thread(connect, thread_id)).rows
    after = await stats(connect)
    print(
        f"pruned {removed} rows; now {after['threads']} threads, {after['checkpoints']} "
        f"checkpoints, {megabytes(after['bytes'])}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stats", help="row counts and size of the memory tables")
    prune = sub.add_parser("prune", help="keep only the latest checkpoint of every thread")
    prune.add_argument("--older-than-days", type=int, default=0, help="also delete idle threads")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
