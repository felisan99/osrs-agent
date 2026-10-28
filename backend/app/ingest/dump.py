"""Download and cache the wiki pages of a scope (SPEC §9).

No official dump exists (verified 2026-10-14), so pages are fetched through the MediaWiki API,
category by category, and cached as JSONL under ``data/wiki/``. A ``.done`` marker next to a file
means the category was fetched completely, so re-runs skip it unless ``refresh`` is set.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.clients.wiki import WikiClient, WikiPage
from app.ingest.scope import ScopeEntry


@dataclass(frozen=True)
class CachedPage:
    title: str
    page_id: int
    revision_id: int
    timestamp: str
    wikitext: str
    page_type: str
    category: str

    def to_wiki_page(self) -> WikiPage:
        return WikiPage(self.title, self.page_id, self.revision_id, self.timestamp, self.wikitext)


def cache_path(data_dir: Path, entry: ScopeEntry) -> Path:
    return data_dir / "wiki" / f"{entry.slug}.jsonl"


def done_path(data_dir: Path, entry: ScopeEntry) -> Path:
    return cache_path(data_dir, entry).with_suffix(".done")


async def fetch_entry(
    wiki: WikiClient,
    entry: ScopeEntry,
    data_dir: Path,
    *,
    limit: int | None = None,
    refresh: bool = False,
    progress: Callable[[int], None] | None = None,
) -> tuple[int, bool]:
    """Fetch one category into the cache; returns ``(pages written, skipped because cached)``."""
    path, marker = cache_path(data_dir, entry), done_path(data_dir, entry)
    if marker.exists() and not refresh and limit is None:
        return sum(1 for _ in path.open(encoding="utf-8")), True
    path.parent.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    temporary = path.with_suffix(".tmp")
    written = 0
    with temporary.open("w", encoding="utf-8") as out:
        async for page in wiki.iter_category_pages(
            entry.category, namespace=entry.namespace, limit=limit
        ):
            if page.is_redirect:
                continue
            record = CachedPage(
                page.title,
                page.page_id,
                page.revision_id,
                page.timestamp,
                page.wikitext,
                entry.page_type,
                entry.category,
            )
            out.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            written += 1
            if progress and written % 250 == 0:
                progress(written)
    temporary.replace(path)
    if limit is None:  # a partial (limited) fetch must not look complete
        marker.write_text(
            json.dumps({"pages": written, "fetched_at": datetime.now(UTC).isoformat()}) + "\n"
        )
    return written, False


def read_entry(data_dir: Path, entry: ScopeEntry) -> Iterator[CachedPage]:
    path = cache_path(data_dir, entry)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                yield CachedPage(**json.loads(line))
