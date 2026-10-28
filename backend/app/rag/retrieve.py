"""Hybrid retrieval (SPEC §7 ``retrieve``): pgvector cosine search + Postgres full-text search
(+ the chunks of the pages the question names), fused with Reciprocal Rank Fusion in one SQL
statement.

* Both sides search with the English standalone query that the graph's ``condense`` step produces
  (a Spanish question is translated there), so the embedding model only has to be good at English.
  The full-text side is English-only (``to_tsvector('english', ...)``) and gets ``keyword_text``
  when the caller has a different English wording. Chunk headers ("Page > Section") are indexed
  with a higher weight than the body.
* Title side (optional ``TitleIndex``): when the question names a wiki page ("Monkey Madness II"),
  that page's chunks (and its ``/Strategies`` page) form a third ranked list ordered by vector
  distance. On the eval set this lifted the answer chunk into the top 5 for 15/18 questions (9/18
  without; see docs/eval-m4.md).
* RRF score = sum over the lists of ``1 / (k + rank)``; a passage found by only one list still
  competes. ``k = 60`` is the usual constant (SPEC says RRF, not the constant).
* ``hnsw.iterative_scan`` (pgvector >= 0.8) keeps filtered vector searches from returning too few
  rows when a ``page_type`` filter is applied.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any

import psycopg

from app.config import Settings, get_settings
from app.rag.titles import TitleIndex

RRF_K = 60
HNSW_EF_SEARCH = 100

ConnectionFactory = Callable[[], AbstractAsyncContextManager[psycopg.AsyncConnection]]
QueryEmbedder = Callable[[str], Sequence[float]]

_WORD = re.compile(r"[^\W_][\w'+]*")  # Unicode-aware: "cuánto" is one word
# Two-letter English function words. Postgres' ``english`` configuration drops them anyway; they
# are removed here so the query string stays readable and "or" is never read as the OR operator.
_SHORT_STOP_WORDS = frozenset(
    {"am", "an", "as", "at", "be", "by", "do", "he", "if", "in", "is", "it", "me", "my", "no"}
    | {"of", "on", "or", "so", "to", "up", "us", "we"}
)

HYBRID_SQL = """
with vec as (
  select id, row_number() over (order by embedding <=> %(qvec)s::vector) as rank
  from documents
  where (%(types)s::text[] is null or page_type = any(%(types)s::text[]))
  order by embedding <=> %(qvec)s::vector
  limit %(pool)s
),
fts as (
  select id, row_number() over (order by ts_rank(tsv, q) desc, id) as rank
  from documents, websearch_to_tsquery('english', %(kw)s) as q
  where tsv @@ q and (%(types)s::text[] is null or page_type = any(%(types)s::text[]))
  order by ts_rank(tsv, q) desc, id
  limit %(pool)s
),
tit as (
  select id, row_number() over (order by embedding <=> %(qvec)s::vector) as rank
  from documents
  where page_title = any(%(titles)s::text[])
    and (%(types)s::text[] is null or page_type = any(%(types)s::text[]))
  order by embedding <=> %(qvec)s::vector
  limit %(pool)s
)
select d.id, d.page_title, d.section, d.page_type, d.url, d.content, d.metadata,
       coalesce(1.0 / (%(k)s + vec.rank), 0) + coalesce(1.0 / (%(k)s + fts.rank), 0)
         + coalesce(1.0 / (%(k)s + tit.rank), 0) as score,
       vec.rank as vector_rank, fts.rank as keyword_rank, tit.rank as title_rank
from vec full outer join fts using (id) full outer join tit using (id)
join documents d using (id)
order by score desc, d.id
limit %(limit)s
"""


@dataclass(frozen=True)
class Passage:
    id: int
    page_title: str
    section: str
    page_type: str
    url: str
    content: str
    metadata: dict[str, Any]
    score: float  # RRF
    vector_rank: int | None = None
    keyword_rank: int | None = None
    title_rank: int | None = None


def keyword_query(text: str) -> str:
    """OR-of-words query in websearch syntax: ranks by coverage instead of requiring every word.

    ``websearch_to_tsquery`` ANDs plain words, which makes natural-language questions match almost
    nothing; stop words are dropped by the ``english`` configuration itself. Two-letter words stay:
    "II" tells Monkey Madness II from Monkey Madness I, "XP" and "GP" are game vocabulary.
    """
    seen: dict[str, None] = {}
    for word in _WORD.findall(text):
        if len(word) >= 3 or (len(word) == 2 and word.lower() not in _SHORT_STOP_WORDS):
            seen.setdefault(word.lower(), None)
    return " or ".join(seen)


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


class Retriever:
    def __init__(
        self,
        connect: ConnectionFactory,
        embed_query: QueryEmbedder,
        settings: Settings | None = None,
        titles: TitleIndex | None = None,
    ) -> None:
        self._connect = connect
        self._embed_query = embed_query
        self._settings = settings or get_settings()
        self._titles = titles

    def _named_pages(
        self, query: str, keyword_text: str | None, explicit: Sequence[str] | None
    ) -> list[str]:
        """Pages the question names (found in the text) plus the ones the caller insists on, and
        the strategy pages of the bosses among them."""
        named = list(explicit or [])
        if self._titles is not None:
            named += [t for t in self._titles.match(query, keyword_text or "") if t not in named]
            named += [g for t in list(named) for g in self._titles.guide_pages(t) if g not in named]
        return named

    async def search(
        self,
        query: str,
        *,
        k: int | None = None,
        page_types: Sequence[str] | None = None,
        keyword_text: str | None = None,
        titles: Sequence[str] | None = None,
    ) -> list[Passage]:
        limit = k or self._settings.crag_topk
        vector = await asyncio.to_thread(self._embed_query, query)
        params = {
            "qvec": vector_literal(vector),
            "kw": keyword_query(keyword_text if keyword_text is not None else query),
            "types": list(page_types) if page_types else None,
            "titles": self._named_pages(query, keyword_text, titles),
            "pool": max(limit * 3, 60),
            "k": RRF_K,
            "limit": limit,
        }
        async with self._connect() as conn, conn.transaction():
            await conn.execute("set local hnsw.iterative_scan = relaxed_order")
            await conn.execute(f"set local hnsw.ef_search = {HNSW_EF_SEARCH}")
            cursor = await conn.execute(HYBRID_SQL, params)
            rows = await cursor.fetchall()
        return [
            Passage(
                id=row[0],
                page_title=row[1],
                section=row[2] or "",
                page_type=row[3],
                url=row[4],
                content=row[5],
                metadata=row[6],
                score=float(row[7]),
                vector_rank=row[8],
                keyword_rank=row[9],
                title_rank=row[10],
            )
            for row in rows
        ]
