"""One-off wiki ingestion (SPEC §9 / §13.3).

    uv run python -m scripts.ingest fetch [--scope v0] [--limit-per-category N] [--refresh]
    uv run python -m scripts.ingest load  [--scope v0] [--limit-per-category N] [--force]
                                          [--page-type mmg ...] [--titles-file FILE]
    uv run python -m scripts.ingest stats

``fetch`` downloads the pages through the MediaWiki API into ``backend/data/wiki`` (resumable: a
finished category is skipped). ``load`` parses, chunks, embeds locally and writes Postgres; pages
whose revision is unchanged are skipped unless ``--force``. Embeddings are local, so ingestion
spends no OpenRouter credit.

A changed ``EMBEDDING_MODEL`` makes every stored vector meaningless, so ``load`` refuses to mix:
with another model or dimension already in the database it stops unless ``--force``, which empties
``documents``, re-creates the vector column for ``EMBEDDING_DIM`` and re-ingests everything.
``--titles-file`` (one page title per line) loads only those pages: experiment indexes.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import sys
import time
import traceback
from pathlib import Path

import httpx

from app.clients.prices import PricesClient
from app.clients.wiki import WikiClient
from app.config import Settings, get_settings
from app.db import (
    MissingDatabaseUrl,
    embedding_column_dim,
    get_index_meta,
    require_db_url,
    reset_embedding_column,
    set_index_meta,
)
from app.ingest.chunk import ChunkConfig, build_quest_matcher
from app.ingest.drops import build_index as build_drop_index
from app.ingest.dump import CachedPage, fetch_entry, read_entry
from app.ingest.load import (
    connect,
    counts_by_page_type,
    existing_revisions,
    replace_page,
    upsert_guide,
    upsert_quest,
)
from app.ingest.mmg import parse_reference_profits
from app.ingest.pipeline import PageResult, PipelineContext, process_page
from app.ingest.scope import SCOPES, ScopeEntry
from app.rag.embed import Embedder
from app.tools.items import ItemIndex

BATCH_CHUNKS = 192  # embed this many chunks at a time (whole pages only)
LIST_PAGE = "Money making guide"


def _entries(scope: str, page_types: list[str] | None = None) -> tuple[ScopeEntry, ...]:
    try:
        entries = SCOPES[scope]
    except KeyError:
        raise SystemExit(f"unknown scope {scope!r}; choose from {', '.join(SCOPES)}") from None
    if page_types:
        entries = tuple(e for e in entries if e.page_type in page_types)
    return entries


# --------------------------------------------------------------------------- fetch
async def cmd_fetch(args: argparse.Namespace) -> int:
    settings = get_settings()
    entries = _entries(args.scope)
    started = time.perf_counter()
    total = 0
    async with httpx.AsyncClient(timeout=settings.http_timeout_s * 3) as http:
        wiki = WikiClient(settings, http=http)
        for entry in entries:
            t0 = time.perf_counter()
            written, skipped = await fetch_entry(
                wiki,
                entry,
                settings.data_dir,
                limit=args.limit_per_category,
                refresh=args.refresh,
                progress=lambda n, c=entry.category: print(f"    {c}: {n} pages...", flush=True),
            )
            total += written
            note = "cached" if skipped else f"{time.perf_counter() - t0:.0f}s"
            print(
                f"  {entry.category:22} {entry.page_type:12} {written:6d} pages  ({note})",
                flush=True,
            )
        if any(e.page_type == "mmg" for e in entries):
            path = settings.data_dir / "wiki" / "mmg_list.html"
            path.write_text(await wiki.render_html(LIST_PAGE), encoding="utf-8")
            print(
                f"  rendered '{LIST_PAGE}' list saved for reference profits "
                f"({path.stat().st_size:,} bytes)"
            )
    print(
        f"fetched {total} pages in {time.perf_counter() - started:.0f}s "
        f"-> {settings.data_dir / 'wiki'}"
    )
    return 0


# --------------------------------------------------------------------------- load
async def _item_index(settings: Settings) -> ItemIndex:
    async with httpx.AsyncClient(timeout=settings.http_timeout_s) as http:
        mapping = await PricesClient(settings, http=http).mapping()
    return ItemIndex(mapping.data.values())


def cmd_load(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        url = require_db_url(settings)
    except MissingDatabaseUrl as exc:
        print(exc, file=sys.stderr)
        return 2
    entries = _entries(args.scope, args.page_type)

    def pages(entry: ScopeEntry):
        stream = read_entry(settings.data_dir, entry)
        return (
            itertools.islice(stream, args.limit_per_category) if args.limit_per_category else stream
        )

    print("loading embedding model...", flush=True)
    embedder = Embedder(settings)
    _ = embedder.model
    quest_titles = {p.title for e in entries if e.page_type == "quest" for p in pages(e)}
    profits_file = settings.data_dir / "wiki" / "mmg_list.html"
    profits = (
        parse_reference_profits(profits_file.read_text(encoding="utf-8"))
        if profits_file.exists()
        else {}
    )
    ctx = PipelineContext(
        chunk_config=ChunkConfig.from_settings(settings),
        count=embedder.count_tokens,
        quest_matcher=build_quest_matcher(quest_titles),
        quest_titles=quest_titles,
        item_index=asyncio.run(_item_index(settings)),
        reference_profits=profits,
    )
    print(f"  {len(quest_titles)} quest titles, {len(profits)} reference profits, item index ready")
    drop_index = build_drop_index(_unique_pages(_entries(args.scope), settings.data_dir))
    ctx.drop_index = drop_index
    rows = sum(map(len, drop_index.values()))
    print(f"  drop tables inverted: {rows} rows for {len(drop_index)} items")

    stats = {
        "pages": 0,
        "chunks": 0,
        "unchanged": 0,
        "empty": 0,
        "errors": 0,
        "guides": 0,
        "quests": 0,
    }
    errors: list[str] = []
    over_limit = 0
    buffer: list[PageResult] = []
    buffered_chunks = 0
    started = time.perf_counter()
    embedding_seconds = 0.0

    wanted = (
        {
            t.strip()
            for t in Path(args.titles_file).read_text(encoding="utf-8").splitlines()
            if t.strip()
        }
        if args.titles_file
        else None
    )

    with connect(url) as conn:
        if not _index_matches_model(conn, settings, force=args.force):
            return 2
        loaded = {} if args.force else existing_revisions(conn)

        def flush() -> None:
            nonlocal buffer, buffered_chunks, embedding_seconds
            if not buffer:
                return
            flat = [c for r in buffer for c in r.chunks]
            t0 = time.perf_counter()
            vectors = embedder.embed_passages([c.content for c in flat]) if flat else []
            embedding_seconds += time.perf_counter() - t0
            offset = 0
            with conn.transaction():
                for result in buffer:
                    n = len(result.chunks)
                    replace_page(conn, result.title, result.chunks, vectors[offset : offset + n])
                    offset += n
                    if result.guide is not None:
                        upsert_guide(conn, result.guide)
                        stats["guides"] += 1
                    if result.quest is not None:
                        upsert_quest(conn, result.quest)
                        stats["quests"] += 1
            stats["pages"] += len(buffer)
            stats["chunks"] += len(flat)
            elapsed = time.perf_counter() - started
            print(
                f"  {stats['pages']:6d} pages  {stats['chunks']:7d} chunks  "
                f"{stats['chunks'] / max(elapsed, 1e-9):6.1f} chunks/s  ({elapsed:.0f}s)",
                flush=True,
            )
            buffer, buffered_chunks = [], 0

        seen: set[str] = set()
        for entry in entries:
            for cached in pages(entry):
                if cached.title in seen or (wanted is not None and cached.title not in wanted):
                    continue
                seen.add(cached.title)
                if loaded.get(cached.title) == cached.revision_id:
                    stats["unchanged"] += 1
                    continue
                result = _safe_process(cached, ctx, errors)
                if result is None:
                    stats[
                        "errors" if errors and errors[-1].startswith(cached.title) else "empty"
                    ] += 1
                    continue
                over_limit += sum(1 for c in result.chunks if c.tokens > 512)
                buffer.append(result)
                buffered_chunks += len(result.chunks)
                if buffered_chunks >= BATCH_CHUNKS:
                    flush()
        flush()
        set_index_meta(
            conn,
            {
                "embedding_model": settings.embedding_model,
                "embedding_dim": str(settings.embedding_dim),
                "chunk_max_tokens": str(settings.chunk_target_max_tokens),
                "chunk_min_tokens": str(settings.chunk_target_min_tokens),
                "subset": "titles-file" if wanted is not None else "full",
            },
        )
        conn.commit()
        report = counts_by_page_type(conn)

    total_time = time.perf_counter() - started
    print(f"\nloaded in {total_time:.0f}s (embedding {embedding_seconds:.0f}s): {stats}")
    print(f"chunks over the 512-token embedding limit: {over_limit}")
    _print_report(report)
    for line in errors[:20]:
        print("  ERROR", line)
    return 0 if not errors else 1


def _index_matches_model(conn, settings: Settings, *, force: bool) -> bool:
    """Make sure new vectors will not be mixed with ones from another model or dimension."""
    dim = embedding_column_dim(conn)
    recorded = get_index_meta(conn).get("embedding_model")
    (rows,) = conn.execute("select count(*) from documents").fetchone()  # type: ignore[misc]
    different_model = recorded is not None and recorded != settings.embedding_model
    if dim == settings.embedding_dim and not different_model:
        return True
    if rows and not force:
        print(
            f"documents holds {rows} vectors of {recorded or 'an unknown model'} ({dim} dims) but "
            f"EMBEDDING_MODEL={settings.embedding_model} ({settings.embedding_dim} dims). Vectors "
            "of different models cannot be mixed: re-run with --force to rebuild the whole index.",
            file=sys.stderr,
        )
        return False
    print(f"re-creating documents.embedding for {settings.embedding_dim} dimensions", flush=True)
    reset_embedding_column(conn, settings.embedding_dim)
    conn.commit()
    return True


def _unique_pages(entries: tuple[ScopeEntry, ...], data_dir):
    """Every cached page of the scope once (the first category that lists a page wins)."""
    seen: set[str] = set()
    for entry in entries:
        for cached in read_entry(data_dir, entry):
            if cached.title not in seen:
                seen.add(cached.title)
                yield cached


def _safe_process(cached: CachedPage, ctx: PipelineContext, errors: list[str]) -> PageResult | None:
    try:
        return process_page(cached, ctx)
    except Exception as exc:  # one bad page must not abort a 17k-page run
        tail = traceback.extract_tb(exc.__traceback__)[-1]
        errors.append(
            f"{cached.title}: {type(exc).__name__}: {exc} "
            f"({tail.filename.split('/')[-1]}:{tail.lineno})"
        )
        return None


def _print_report(report: list[tuple[str, int, int]]) -> None:
    print("\ndocuments by page_type:")
    print(f"  {'page_type':12} {'chunks':>8} {'pages':>7}")
    for page_type, chunks, pages in report:
        print(f"  {page_type:12} {chunks:8d} {pages:7d}")
    print(f"  {'TOTAL':12} {sum(r[1] for r in report):8d} {sum(r[2] for r in report):7d}")


# --------------------------------------------------------------------------- stats
def cmd_stats(_: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        url = require_db_url(settings)
    except MissingDatabaseUrl as exc:
        print(exc, file=sys.stderr)
        return 2
    with connect(url) as conn:
        _print_report(counts_by_page_type(conn))
        guides, supported, rankable = conn.execute(
            "select count(*),"
            " count(*) filter (where jsonb_array_length(data->'unsupported') = 0"
            "  and jsonb_array_length(data->'items_out') > 0),"
            " count(*) filter (where jsonb_array_length(data->'unsupported') = 0"
            "  and jsonb_array_length(data->'items_out') > 0 and not (data->>'obsolete')::boolean)"
            " from moneymaking_guides"
        ).fetchone()  # type: ignore[misc]
        (quests,) = conn.execute("select count(*) from quests").fetchone()  # type: ignore[misc]
        print(
            f"\nmoney making guides: {guides} ({supported} fully supported, "
            f"{rankable} after dropping the ones the wiki marks obsolete) | quests: {quests}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "load"):
        p = sub.add_parser(name)
        p.add_argument("--scope", default="v0", choices=list(SCOPES))
        p.add_argument("--limit-per-category", type=int, default=None)
    sub.choices["fetch"].add_argument("--refresh", action="store_true")
    sub.choices["load"].add_argument("--force", action="store_true")
    sub.choices["load"].add_argument(
        "--page-type",
        nargs="+",
        choices=["quest", "mmg", "skill_guide", "item", "npc", "other"],
        help="only the scope entries of these page types (e.g. re-parse guides after a fix)",
    )
    sub.choices["load"].add_argument(
        "--titles-file", help="load only the pages whose titles are listed (one per line)"
    )
    sub.add_parser("stats")
    args = parser.parse_args()
    if args.command == "fetch":
        return asyncio.run(cmd_fetch(args))
    if args.command == "load":
        return cmd_load(args)
    return cmd_stats(args)


if __name__ == "__main__":
    sys.exit(main())
