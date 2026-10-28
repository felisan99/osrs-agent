"""The agent as one object: everything the HTTP API (and the dev scripts) need, wired once.

``AgentService.open`` builds the process-wide pieces (HTTP client, Postgres pools, the local
embedding model, the title/quest/guide stores, the price tools, the compiled graph with its
Postgres checkpointer). ``chat`` handles one turn: it builds the per-request clients from the
user's key (BYOK: both Jev and the reasoning model run on that key), runs the graph in the
thread's memory and returns a plain result.

The key exists only inside this call: it is registered with the log redactor for the duration of
the request, handed to the two clients that need it and never stored anywhere else (not in the
graph state, not in the checkpoints, not in the config).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import SecretStr

from app.api.schemas import PlayerState
from app.clients.jev import JevClient
from app.clients.openrouter import OpenRouterClient
from app.clients.prices import PricesClient
from app.clients.wiki import WikiClient
from app.config import Settings, get_settings
from app.db import aget_index_meta, require_db_url
from app.graph.build import build_graph
from app.graph.retention import prune_thread, sweep_forever
from app.graph.state import Deps, RunContext
from app.ingest.chunk import ChunkConfig
from app.rag.embed import Embedder
from app.rag.grade import Grader
from app.rag.live import LiveWikiSearch
from app.rag.retrieve import Retriever
from app.rag.titles import TitleIndex
from app.security import request_secrets
from app.tools.context import ToolContext
from app.tools.moneymakers import GuideStore
from app.tools.quests import QuestStore

logger = logging.getLogger(__name__)


async def check_index_model(pool: AsyncConnectionPool, settings: Settings) -> None:
    """Refuse to serve with a query embedder that differs from the one that built the index."""
    async with pool.connection() as conn:
        meta = await aget_index_meta(conn)
    built_with = meta.get("embedding_model")
    if built_with and built_with != settings.embedding_model:
        raise RuntimeError(
            f"the index was built with {built_with} but "
            f"EMBEDDING_MODEL={settings.embedding_model}: their vectors cannot be compared. Set "
            "EMBEDDING_MODEL back, or rebuild the index with "
            "`uv run python -m scripts.ingest load --force`."
        )


@dataclass(frozen=True)
class TurnResult:
    answer: str
    intent: str
    sources: list[dict[str, str]]
    route_probs: dict[str, float] = field(default_factory=dict)
    retrieval_iterations: int = 0
    tools_called: list[str] = field(default_factory=list)
    usage: dict[str, float] = field(default_factory=dict)


@dataclass
class AgentService:
    settings: Settings
    http: httpx.AsyncClient
    deps: Deps
    graph: Any
    embedder: Embedder
    pool: AsyncConnectionPool

    @classmethod
    @asynccontextmanager
    async def open(
        cls, settings: Settings | None = None, *, http: httpx.AsyncClient | None = None
    ) -> AsyncIterator[AgentService]:
        settings = settings or get_settings()
        url = require_db_url(settings)
        owns_http = http is None
        client = http or httpx.AsyncClient(timeout=settings.http_timeout_s)
        pool = AsyncConnectionPool(url, min_size=1, max_size=8, open=False)
        checkpoint_pool = AsyncConnectionPool(
            url,
            min_size=1,
            max_size=4,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        await pool.open()
        await checkpoint_pool.open()
        sweeper: asyncio.Task[None] | None = None
        try:
            await check_index_model(pool, settings)
            embedder = Embedder(settings)
            await asyncio.to_thread(lambda: embedder.model)  # load now, not on the first question
            titles = await TitleIndex.load(pool.connection)
            guides = await GuideStore.load(pool.connection)
            quests = await QuestStore.load(pool.connection)
            logger.info(
                "loaded %d titles, %d money-making guides, %d quests",
                len(titles),
                len(guides),
                len(quests),
            )
            tools = ToolContext(settings, PricesClient(settings, http=client), guides=guides)
            deps = Deps(
                settings=settings,
                retriever=Retriever(pool.connection, embedder.embed_query, settings, titles),
                tools=tools,
                titles=titles,
                quests=quests,
                live=LiveWikiSearch(
                    WikiClient(settings, http=client),
                    embedder,
                    ChunkConfig.from_settings(settings),
                    base_url=settings.wiki_page_base_url,
                ),
            )
            saver = AsyncPostgresSaver(checkpoint_pool)  # type: ignore[arg-type]
            await saver.setup()
            if settings.thread_ttl_days > 0:
                sweeper = asyncio.create_task(
                    sweep_forever(pool.connection, settings.thread_ttl_days)
                )
            yield cls(settings, client, deps, build_graph(settings, saver), embedder, pool)
        finally:
            if sweeper is not None:
                sweeper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sweeper
            await checkpoint_pool.close()
            await pool.close()
            if owns_http:
                await client.aclose()

    def context(self, api_key: str | SecretStr, player: PlayerState | None) -> RunContext:
        """The run-scoped context of one request: the only place the key is held."""
        key = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        jev = JevClient(key, http=self.http, settings=self.settings)
        return RunContext(
            deps=self.deps,
            llm=OpenRouterClient(key, http=self.http, settings=self.settings),
            jev=jev,
            grader=Grader(jev, self.settings),
            player=player,
        )

    async def chat(
        self,
        *,
        thread_id: str,
        message: str,
        api_key: str,
        player_state: PlayerState | None = None,
    ) -> TurnResult:
        context = self.context(api_key, player_state)
        with request_secrets(api_key):
            state = await self.graph.ainvoke(
                {"messages": [{"role": "user", "content": message}]},
                config={"configurable": {"thread_id": thread_id}},
                context=context,
            )
        await self._prune(thread_id)
        return TurnResult(
            answer=state.get("answer", ""),
            intent=state.get("intent", "wiki_general"),
            sources=list(state.get("sources", [])),
            route_probs=dict(state.get("route_probs", {})),
            retrieval_iterations=state.get("retrieval_iterations", 0),
            tools_called=list(state.get("tools_called", [])),
            usage=dict(state.get("usage", {})),
        )

    async def _prune(self, thread_id: str) -> None:
        """Drop the thread's superseded checkpoints; the answer never depends on it."""
        try:
            await prune_thread(self.pool.connection, thread_id)
        except Exception:
            logger.warning("could not prune the thread's old checkpoints", exc_info=True)

    async def health(self) -> dict[str, Any]:
        """What ``/healthz`` reports: can we reach Postgres and is the corpus there."""
        async with self.pool.connection() as conn:
            cursor = await conn.execute("select count(*) from documents")
            row = await cursor.fetchone()
        return {
            "status": "ok",
            "documents": row[0] if row else 0,
            "guides": len(self.deps.tools.guides) if self.deps.tools.guides else 0,
            "quests": len(self.deps.quests),
        }
