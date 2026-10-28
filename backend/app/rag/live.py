"""``search_live`` (SPEC §7): candidate passages straight from the live wiki.

The ingested corpus is a snapshot. When retrieval cannot find enough, the CRAG correction also asks
the wiki itself: the MediaWiki search finds pages, their wikitext is parsed and chunked exactly like
ingested pages, and the chunks are ranked by embedding similarity to the query. The result is a
list of ``Passage`` that goes through the same Jev grading as everything else.

Live passages get negative ids (they have no row in ``documents``); the id is a stable hash of the
page and chunk so the same chunk is never pooled twice.
"""

from __future__ import annotations

import asyncio
import zlib
from collections.abc import Sequence

import numpy as np

from app.clients.wiki import WikiClient, WikiPage, page_url
from app.ingest.chunk import Chunk, ChunkConfig
from app.ingest.dump import CachedPage
from app.ingest.pipeline import PipelineContext, process_page
from app.rag.embed import Embedder
from app.rag.retrieve import Passage

SEARCH_HITS = 5


def synthetic_id(title: str, index: int) -> int:
    """A negative id derived from the page and chunk position."""
    return -(zlib.crc32(f"{title}|{index}".encode()) + 1)


class LiveWikiSearch:
    def __init__(
        self,
        wiki: WikiClient,
        embedder: Embedder,
        chunk_config: ChunkConfig,
        *,
        base_url: str = "https://oldschool.runescape.wiki/w/",
    ) -> None:
        self._wiki = wiki
        self._embedder = embedder
        self._chunk_config = chunk_config
        self._base_url = base_url

    def _chunks(self, pages: Sequence[WikiPage]) -> list[Chunk]:
        ctx = PipelineContext(chunk_config=self._chunk_config, count=self._embedder.count_tokens)
        chunks: list[Chunk] = []
        for page in pages:
            cached = CachedPage(
                page.title,
                page.page_id,
                page.revision_id,
                page.timestamp,
                page.wikitext,
                "other",
                "",
            )
            result = process_page(cached, ctx)
            if result is not None:
                chunks.extend(result.chunks)
        return chunks

    async def passages(self, query: str, *, limit: int) -> list[Passage]:
        hits = await self._wiki.search(query, limit=SEARCH_HITS)
        if not hits:
            return []
        pages = [
            p for p in await self._wiki.fetch_pages([h.title for h in hits]) if not p.is_redirect
        ]
        chunks = await asyncio.to_thread(self._chunks, pages)
        if not chunks:
            return []
        vectors = await asyncio.to_thread(
            self._embedder.embed_passages, [c.content for c in chunks]
        )
        query_vector = np.asarray(await asyncio.to_thread(self._embedder.embed_query, query))
        scores = np.asarray(vectors) @ query_vector
        best = np.argsort(-scores)[:limit]
        positions: dict[str, int] = {}
        out: list[Passage] = []
        order = {id(c): i for i, c in enumerate(chunks)}
        for index in best:
            chunk = chunks[int(index)]
            position = order[id(chunk)]
            positions[chunk.page_title] = positions.get(chunk.page_title, 0) + 1
            out.append(
                Passage(
                    id=synthetic_id(chunk.page_title, position),
                    page_title=chunk.page_title,
                    section=chunk.section,
                    page_type="other",
                    url=page_url(chunk.page_title, self._base_url),
                    content=chunk.content,
                    metadata={"live": True},
                    score=float(scores[int(index)]),
                )
            )
        return out
