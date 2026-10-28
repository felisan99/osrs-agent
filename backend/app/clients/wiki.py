"""MediaWiki API client for the OSRS Wiki: bulk ingestion and live search (SPEC §3, §7, §9).

No official XML dump is published (verified 2026-10-14), so ingestion pages through the API:
``generator=categorymembers`` + ``prop=revisions`` returns up to 50 pages with their wikitext per
request. Politeness rules: descriptive ``User-Agent``, sequential requests with a small delay,
``maxlag`` so the server can tell us to back off (answered as HTTP 200 with ``error.code=maxlag``).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from app.clients.http import RetryPolicy, parse_retry_after, safe_json, send_with_retries
from app.config import Settings, get_settings
from app.errors import UpstreamError

Sleep = Callable[[float], Awaitable[None]]
MAXLAG_RETRIES = 4
URL_SAFE = "/:()',!*$@~-._"


@dataclass(frozen=True)
class WikiPage:
    title: str
    page_id: int
    revision_id: int
    timestamp: str
    wikitext: str

    @property
    def is_redirect(self) -> bool:
        return self.wikitext.lstrip()[:9].lower() == "#redirect"


@dataclass(frozen=True)
class SearchHit:
    title: str
    snippet: str
    word_count: int | None = None


def page_url(title: str, base: str = "https://oldschool.runescape.wiki/w/") -> str:
    """Canonical page URL: spaces become underscores, MediaWiki-safe characters stay literal."""
    return base + quote(title.replace(" ", "_"), safe=URL_SAFE)


class WikiClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http: httpx.AsyncClient | None = None,
        retry: RetryPolicy | None = None,
        sleep: Sleep = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings or get_settings()
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(timeout=self._settings.http_timeout_s)
        self._retry = retry or RetryPolicy.from_settings(self._settings)
        self._sleep = sleep
        self._clock = clock
        self._last_request = 0.0
        self._gate = asyncio.Lock()

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    # ---- transport
    async def _throttle(self) -> None:
        delay = self._settings.wiki_request_delay_s
        if delay <= 0:
            return
        async with self._gate:
            wait = self._last_request + delay - self._clock()
            if wait > 0:
                await self._sleep(wait)
            self._last_request = self._clock()

    async def _api(self, params: dict[str, Any]) -> dict[str, Any]:
        query = {
            "format": "json",
            "formatversion": "2",
            "maxlag": str(self._settings.wiki_maxlag_s),
            **params,
        }
        for attempt in range(MAXLAG_RETRIES + 1):
            await self._throttle()
            response = await send_with_retries(
                self._http,
                "GET",
                self._settings.wiki_api_url,
                policy=self._retry,
                what="wiki API",
                headers={
                    "User-Agent": self._settings.effective_wiki_user_agent,
                    "Accept": "application/json",
                },
                params=query,
                timeout=self._settings.http_timeout_s,
            )
            if response.status_code != 200:
                raise UpstreamError(
                    f"wiki API returned HTTP {response.status_code}",
                    upstream_status=response.status_code,
                )
            payload = safe_json(response)
            if not isinstance(payload, dict):
                raise UpstreamError("wiki API returned a non-JSON body")
            error = payload.get("error")
            if isinstance(error, dict):
                if error.get("code") == "maxlag" and attempt < MAXLAG_RETRIES:
                    # The server is asking clients to back off: honour Retry-After (capped).
                    await self._sleep(min(parse_retry_after(response) or 5.0, 30.0))
                    continue
                raise UpstreamError(
                    f"wiki API error {error.get('code', '?')}: {str(error.get('info', ''))[:200]}"
                )
            return payload
        raise UpstreamError("wiki API stayed lagged after retries")  # pragma: no cover

    # ---- reads
    async def category_members(
        self, category: str, *, namespace: int = 0, limit: int | None = None
    ) -> list[str]:
        """Titles of the pages directly in ``Category:<category>`` (no recursion)."""
        titles: list[str] = []
        cont: dict[str, Any] = {}
        while limit is None or len(titles) < limit:
            payload = await self._api(
                {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": f"Category:{category}",
                    "cmnamespace": str(namespace),
                    "cmlimit": "500",
                    **cont,
                }
            )
            titles += [m["title"] for m in payload.get("query", {}).get("categorymembers", [])]
            if "continue" not in payload:
                break
            cont = payload["continue"]
        return titles if limit is None else titles[:limit]

    async def subcategories(self, category: str) -> list[str]:
        payload = await self._api(
            {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmtype": "subcat",
                "cmlimit": "500",
            }
        )
        return [
            m["title"].removeprefix("Category:")
            for m in payload.get("query", {}).get("categorymembers", [])
        ]

    @staticmethod
    def _pages_from(payload: dict[str, Any]) -> list[WikiPage]:
        pages: list[WikiPage] = []
        for raw in payload.get("query", {}).get("pages", []):
            if raw.get("missing") or raw.get("invalid"):
                continue
            revisions = raw.get("revisions") or []
            if not revisions:
                continue
            revision = revisions[0]
            content = (revision.get("slots", {}).get("main", {}) or {}).get("content")
            if content is None:
                continue
            pages.append(
                WikiPage(
                    title=raw["title"],
                    page_id=int(raw.get("pageid", 0)),
                    revision_id=int(revision.get("revid", 0)),
                    timestamp=str(revision.get("timestamp", "")),
                    wikitext=content,
                )
            )
        return pages

    async def fetch_pages(self, titles: Sequence[str]) -> list[WikiPage]:
        """Wikitext of the given titles (batched, order not guaranteed, missing pages skipped)."""
        out: list[WikiPage] = []
        size = self._settings.wiki_batch_size
        for start in range(0, len(titles), size):
            payload = await self._api(
                {
                    "action": "query",
                    "prop": "revisions",
                    "rvprop": "ids|timestamp|content",
                    "rvslots": "main",
                    "titles": "|".join(titles[start : start + size]),
                }
            )
            out += self._pages_from(payload)
        return out

    async def iter_category_pages(
        self, category: str, *, namespace: int = 0, limit: int | None = None
    ) -> AsyncIterator[WikiPage]:
        """Stream the wikitext of every page in a category, one batch per request."""
        yielded = 0
        cont: dict[str, Any] = {}
        while True:
            payload = await self._api(
                {
                    "action": "query",
                    "generator": "categorymembers",
                    "gcmtitle": f"Category:{category}",
                    "gcmnamespace": str(namespace),
                    "gcmlimit": str(self._settings.wiki_batch_size),
                    "prop": "revisions",
                    "rvprop": "ids|timestamp|content",
                    "rvslots": "main",
                    **cont,
                }
            )
            for page in self._pages_from(payload):
                yield page
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
            if "continue" not in payload:
                return
            cont = payload["continue"]

    async def search(self, query: str, *, limit: int = 5, namespace: int = 0) -> list[SearchHit]:
        """Full-text search (``list=search``), best matches first."""
        payload = await self._api(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": str(limit),
                "srnamespace": str(namespace),
                "srprop": "snippet|wordcount",
            }
        )
        return [
            SearchHit(
                title=hit["title"],
                snippet=hit.get("snippet", ""),
                word_count=hit.get("wordcount"),
            )
            for hit in payload.get("query", {}).get("search", [])
        ]

    async def render_html(self, title: str) -> str:
        """Rendered HTML of a page (``action=parse``), e.g. the Money making guide list table."""
        payload = await self._api({"action": "parse", "page": title, "prop": "text"})
        text = (payload.get("parse") or {}).get("text")
        if not isinstance(text, str):
            raise UpstreamError(f"wiki API returned no rendered text for {title!r}")
        return text
