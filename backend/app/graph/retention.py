"""Retention of the conversation memory: LangGraph's checkpoint tables in Postgres.

LangGraph writes a checkpoint after every step of every turn and never deletes one, so the rows
of a thread (and the blobs of its replaced channel versions) pile up for as long as the database
lives. Nothing in this agent reads history, though: a turn only needs the thread's *latest* state.
Three limits keep the tables bounded:

* after every turn the thread keeps only its latest checkpoint (``prune_thread``);
* the messages a thread remembers are capped (``MAX_STORED_MESSAGES`` in ``graph/state.py``);
* threads nobody used for ``THREAD_TTL_DAYS`` are deleted (``delete_stale_threads``).

The SQL is written against LangGraph's documented tables (``checkpoints``, ``checkpoint_writes``,
``checkpoint_blobs``): ``AsyncPostgresSaver`` implements ``delete_thread`` but not ``prune``.
Only the thread's own rows are touched, and a failure here must never fail the answer.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass

import psycopg

logger = logging.getLogger(__name__)

ConnectionFactory = Callable[[], AbstractAsyncContextManager[psycopg.AsyncConnection]]

STALE_SWEEP_INTERVAL_S = 6 * 3600

# The newest checkpoint of each namespace of the thread (checkpoint ids sort by time).
_LATEST = (
    "select checkpoint_ns, max(checkpoint_id) as id from checkpoints "
    "where thread_id = %(thread)s group by checkpoint_ns"
)
_DELETE_OLD_WRITES = (
    f"delete from checkpoint_writes w using ({_LATEST}) latest "
    "where w.thread_id = %(thread)s and w.checkpoint_ns = latest.checkpoint_ns "
    "and w.checkpoint_id < latest.id"
)
_DELETE_OLD_CHECKPOINTS = (
    f"delete from checkpoints c using ({_LATEST}) latest "
    "where c.thread_id = %(thread)s and c.checkpoint_ns = latest.checkpoint_ns "
    "and c.checkpoint_id < latest.id"
)
# A blob is a channel's value at one version; keep exactly those the surviving checkpoints use.
_DELETE_UNUSED_BLOBS = (
    "delete from checkpoint_blobs b where b.thread_id = %(thread)s and not exists ("
    " select 1 from checkpoints c,"
    "  jsonb_each_text(c.checkpoint -> 'channel_versions') as v(channel, version)"
    " where c.thread_id = b.thread_id and c.checkpoint_ns = b.checkpoint_ns"
    "  and v.channel = b.channel and v.version = b.version)"
)
_STALE_THREADS = (
    "select thread_id from checkpoints group by thread_id "
    "having max((checkpoint ->> 'ts')::timestamptz) < now() - make_interval(days => %(days)s) "
    "limit %(limit)s"
)


@dataclass(frozen=True)
class PruneResult:
    checkpoints: int = 0
    writes: int = 0
    blobs: int = 0

    @property
    def rows(self) -> int:
        return self.checkpoints + self.writes + self.blobs


async def prune_thread(connect: ConnectionFactory, thread_id: str) -> PruneResult:
    """Keep only the latest checkpoint of ``thread_id`` (and what it points to)."""
    params = {"thread": thread_id}
    async with connect() as conn, conn.transaction():
        writes = (await conn.execute(_DELETE_OLD_WRITES, params)).rowcount
        checkpoints = (await conn.execute(_DELETE_OLD_CHECKPOINTS, params)).rowcount
        blobs = (await conn.execute(_DELETE_UNUSED_BLOBS, params)).rowcount
    return PruneResult(max(checkpoints, 0), max(writes, 0), max(blobs, 0))


async def delete_threads(connect: ConnectionFactory, thread_ids: Sequence[str]) -> int:
    """Delete whole threads; returns how many checkpoint rows went away."""
    if not thread_ids:
        return 0
    params = {"ids": list(thread_ids)}
    removed = 0
    async with connect() as conn, conn.transaction():
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            cursor = await conn.execute(
                f"delete from {table} where thread_id = any(%(ids)s)", params
            )
            removed += cursor.rowcount if table == "checkpoints" else 0
    return removed


async def delete_stale_threads(
    connect: ConnectionFactory, older_than_days: int, *, batch: int = 500
) -> int:
    """Delete the threads whose newest checkpoint is older than ``older_than_days`` days.

    ``0`` disables the sweep. Returns the number of threads deleted.
    """
    if older_than_days <= 0:
        return 0
    deleted = 0
    while True:
        async with connect() as conn:
            cursor = await conn.execute(_STALE_THREADS, {"days": older_than_days, "limit": batch})
            ids = [row[0] for row in await cursor.fetchall()]
        if not ids:
            return deleted
        await delete_threads(connect, ids)
        deleted += len(ids)


async def stats(connect: ConnectionFactory) -> dict[str, int]:
    """Row counts and size of the memory tables (what ``scripts.maintenance stats`` prints)."""
    async with connect() as conn:
        out: dict[str, int] = {}
        cursor = await conn.execute("select to_regclass('checkpoints') is not null")
        if not ((await cursor.fetchone()) or (False,))[0]:
            # a database nobody has chatted against yet: LangGraph creates its tables on first start
            return {
                "threads": 0,
                "checkpoints": 0,
                "checkpoint_blobs": 0,
                "checkpoint_writes": 0,
                "bytes": 0,
            }
        cursor = await conn.execute("select count(distinct thread_id), count(*) from checkpoints")
        threads, checkpoints = (await cursor.fetchone()) or (0, 0)
        out["threads"], out["checkpoints"] = threads, checkpoints
        for table in ("checkpoint_blobs", "checkpoint_writes"):
            cursor = await conn.execute(f"select count(*) from {table}")
            out[table] = ((await cursor.fetchone()) or (0,))[0]
        cursor = await conn.execute(
            "select pg_total_relation_size('checkpoints')"
            " + pg_total_relation_size('checkpoint_blobs')"
            " + pg_total_relation_size('checkpoint_writes')"
        )
        out["bytes"] = ((await cursor.fetchone()) or (0,))[0]
    return out


async def sweep_forever(
    connect: ConnectionFactory, older_than_days: int, interval_s: float = STALE_SWEEP_INTERVAL_S
) -> None:
    """Background task of the API process: delete stale threads now and then, never raise."""
    while True:
        try:
            deleted = await delete_stale_threads(connect, older_than_days)
            if deleted:
                logger.info(
                    "deleted %d conversation threads idle for %d days", deleted, older_than_days
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("stale-thread sweep failed", exc_info=True)
        await asyncio.sleep(interval_s)
