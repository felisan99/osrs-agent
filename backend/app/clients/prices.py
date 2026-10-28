"""Client for the OSRS Wiki real-time prices API: https://prices.runescape.wiki/api/v2/osrs

Rules this client enforces (SPEC §6, verified live on 2026-10-14):

* Every request carries a descriptive ``User-Agent`` (generic ones such as ``python-requests``,
  ``curl/x``, ``Java/x`` are blocked by the API; the setting is validated in ``config.py``).
* Never loop ``/latest?id=`` item by item: callers use the cached full snapshot; ``latest_item``
  only falls back to ``?id=`` for a one-off lookup when no fresh snapshot is cached.
* Short TTL for ``/latest``, ``/5m`` and ``/1h``; long TTL for ``/mapping``; concurrent callers
  share one in-flight request (single flight). If the API is down a snapshot up to
  ``PRICES_MAX_STALE_S`` old is served with ``stale=True`` so answers can cite the real timestamp.
* Prices are plain Python ints (they exceed int32: max ``high`` seen live was 9.36e9) and the
  ``/5m`` ``/1h`` averages are floats with up to two decimals.
* ``/timeseries`` requires ``lookback`` (``6h|24h|7d|30d|6m|1y``); ``timestep`` is rejected (400).
* Semantics: ``high`` = price at which an instant buy happens, ``low`` = price at which an instant
  sell happens. Either side can be ``null`` when that side never traded.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.clients.http import RetryPolicy, send_with_retries
from app.config import Settings, get_settings
from app.errors import UpstreamError

LOOKBACKS: tuple[str, ...] = ("6h", "24h", "7d", "30d", "6m", "1y")


# --------------------------------------------------------------------------- models
class LatestPrice(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    high: int | None = None
    high_time: int | None = Field(None, alias="highTime")
    low: int | None = None
    low_time: int | None = Field(None, alias="lowTime")


class AveragePrice(BaseModel):
    """One item's row in ``/5m`` or ``/1h``."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    avg_high: float | None = Field(None, alias="avgHighPrice")
    high_volume: int = Field(0, alias="highPriceVolume")
    avg_low: float | None = Field(None, alias="avgLowPrice")
    low_volume: int = Field(0, alias="lowPriceVolume")


class ItemMeta(BaseModel):
    """One row of ``/mapping`` (``limit`` is the 4-hour GE buy limit; absent for ~11% of items)."""

    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    members: bool = False
    limit: int | None = None
    highalch: int | None = None
    lowalch: int | None = None
    value: int | None = None
    examine: str | None = None
    icon: str | None = None


class TimeseriesPoint(AveragePrice):
    timestamp: int


class Timeseries(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    item_id: int = Field(alias="itemId")
    lookback: str
    timestep: int | None = None  # seconds; "not guaranteed" per the docs
    start_timestamp: int | None = Field(None, alias="startTimestamp")
    end_timestamp: int | None = Field(None, alias="endTimestamp")
    points: list[TimeseriesPoint] = Field(default_factory=list)
    fetched_at: datetime
    stale: bool = False


@dataclass(frozen=True)
class Snapshot[T]:
    """A full-table response plus provenance, so callers can always state *when* prices are from."""

    data: dict[int, T]
    fetched_at: datetime  # when this process fetched it (UTC)
    api_timestamp: int | None = None  # start of the averaging window (/5m, /1h only)
    stale: bool = False


# --------------------------------------------------------------------------- caching
class _Cell[T]:
    """One cached value with TTL, single-flight loading and stale-on-error."""

    def __init__(self, ttl_s: float, max_stale_s: float, clock: Callable[[], float]) -> None:
        self._ttl = ttl_s
        self._max_stale = max_stale_s
        self._clock = clock
        self._value: T | None = None
        self._loaded_at = 0.0
        self._lock = asyncio.Lock()

    def peek_fresh(self) -> T | None:
        if self._value is not None and self._clock() - self._loaded_at < self._ttl:
            return self._value
        return None

    async def get(
        self, loader: Callable[[], Awaitable[T]], mark_stale: Callable[[T], T] | None = None
    ) -> T:
        fresh = self.peek_fresh()
        if fresh is not None:
            return fresh
        async with self._lock:
            fresh = self.peek_fresh()  # someone else may have loaded it while we waited
            if fresh is not None:
                return fresh
            try:
                value = await loader()
            except UpstreamError:
                age = self._clock() - self._loaded_at
                if self._value is not None and age <= self._max_stale:
                    return mark_stale(self._value) if mark_stale else self._value
                raise
            self._value = value
            self._loaded_at = self._clock()
            return value


# --------------------------------------------------------------------------- client
class PricesClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http: httpx.AsyncClient | None = None,
        retry: RetryPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings or get_settings()
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(timeout=self._settings.http_timeout_s)
        self._retry = retry or RetryPolicy.from_settings(self._settings)
        self._clock = clock

        s = self._settings
        stale = s.prices_max_stale_s
        self._mapping = _Cell[Snapshot[ItemMeta]](s.prices_mapping_ttl_s, stale, clock)
        self._latest = _Cell[Snapshot[LatestPrice]](s.prices_latest_ttl_s, stale, clock)
        self._five_min = _Cell[Snapshot[AveragePrice]](s.prices_latest_ttl_s, stale, clock)
        self._one_hour = _Cell[Snapshot[AveragePrice]](s.prices_hourly_ttl_s, stale, clock)
        self._timeseries: dict[tuple[int, str], tuple[float, Timeseries]] = {}
        self._timeseries_lock = asyncio.Lock()
        self._single: dict[int, tuple[float, LatestPrice | None, datetime]] = {}

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    # ---- transport
    async def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = await send_with_retries(
            self._http,
            "GET",
            f"{self._settings.prices_base_url.rstrip('/')}{path}",
            policy=self._retry,
            what="prices API",
            headers={
                "User-Agent": self._settings.prices_user_agent,
                "Accept": "application/json",
            },
            params=params,
            timeout=self._settings.http_timeout_s,
        )
        if response.status_code != 200:
            snippet = response.text[:200].replace("\n", " ")
            raise UpstreamError(
                f"prices API returned HTTP {response.status_code} for {path}: {snippet}",
                upstream_status=response.status_code,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise UpstreamError(f"prices API returned invalid JSON for {path}") from exc

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _mark_stale[T](snapshot: Snapshot[T]) -> Snapshot[T]:
        return replace(snapshot, stale=True)

    # ---- endpoints
    async def mapping(self) -> Snapshot[ItemMeta]:
        async def load() -> Snapshot[ItemMeta]:
            raw = await self._get_json("/mapping")
            try:
                items = [ItemMeta.model_validate(row) for row in raw]
            except (ValidationError, TypeError) as exc:
                raise UpstreamError("unexpected /mapping response shape") from exc
            return Snapshot({i.id: i for i in items}, self._now())

        return await self._mapping.get(load, self._mark_stale)

    async def latest(self) -> Snapshot[LatestPrice]:
        async def load() -> Snapshot[LatestPrice]:
            raw = await self._get_json("/latest")
            return Snapshot(self._parse_table(raw, LatestPrice, "/latest"), self._now())

        return await self._latest.get(load, self._mark_stale)

    async def latest_item(self, item_id: int) -> tuple[LatestPrice | None, datetime, bool]:
        """One-off lookup: ``(price | None, fetched_at, stale)``.

        Uses the cached full snapshot when it is fresh; otherwise ``/latest?id=`` (small) and a
        short per-id cache. Never call this in a loop over many items; use ``latest()``.
        """
        fresh = self._latest.peek_fresh()
        if fresh is not None:
            return fresh.data.get(item_id), fresh.fetched_at, False
        cached = self._single.get(item_id)
        if cached and self._clock() - cached[0] < self._settings.prices_latest_ttl_s:
            return cached[1], cached[2], False
        raw = await self._get_json("/latest", {"id": item_id})
        table = self._parse_table(raw, LatestPrice, "/latest?id=")
        price, fetched = table.get(item_id), self._now()
        if len(self._single) > 512:
            self._single.clear()
        self._single[item_id] = (self._clock(), price, fetched)
        return price, fetched, False

    async def five_minute(self) -> Snapshot[AveragePrice]:
        async def load() -> Snapshot[AveragePrice]:
            raw = await self._get_json("/5m")
            return Snapshot(
                self._parse_table(raw, AveragePrice, "/5m"), self._now(), self._api_ts(raw)
            )

        return await self._five_min.get(load, self._mark_stale)

    async def one_hour(self) -> Snapshot[AveragePrice]:
        async def load() -> Snapshot[AveragePrice]:
            raw = await self._get_json("/1h")
            return Snapshot(
                self._parse_table(raw, AveragePrice, "/1h"), self._now(), self._api_ts(raw)
            )

        return await self._one_hour.get(load, self._mark_stale)

    async def timeseries(self, item_id: int, lookback: str) -> Timeseries:
        if lookback not in LOOKBACKS:
            raise ValueError(f"lookback must be one of {', '.join(LOOKBACKS)}; got {lookback!r}")
        key = (item_id, lookback)
        ttl = self._settings.prices_timeseries_ttl_s
        hit = self._timeseries.get(key)
        if hit and self._clock() - hit[0] < ttl:
            return hit[1]
        async with self._timeseries_lock:
            hit = self._timeseries.get(key)
            if hit and self._clock() - hit[0] < ttl:
                return hit[1]
            try:
                raw = await self._get_json("/timeseries", {"id": item_id, "lookback": lookback})
                series = self._parse_timeseries(raw, item_id, lookback)
            except UpstreamError:
                if hit and self._clock() - hit[0] <= self._settings.prices_max_stale_s:
                    return hit[1].model_copy(update={"stale": True})
                raise
            if len(self._timeseries) > 256:
                self._timeseries.clear()
            self._timeseries[key] = (self._clock(), series)
            return series

    # ---- parsing
    @staticmethod
    def _api_ts(raw: Any) -> int | None:
        ts = raw.get("timestamp") if isinstance(raw, dict) else None
        return ts if isinstance(ts, int) else None

    @staticmethod
    def _parse_table(raw: Any, model: type[BaseModel], path: str) -> dict[int, Any]:
        try:
            return {int(k): model.model_validate(v) for k, v in raw["data"].items()}
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise UpstreamError(f"unexpected {path} response shape") from exc

    def _parse_timeseries(self, raw: Any, item_id: int, lookback: str) -> Timeseries:
        try:
            return Timeseries(
                itemId=raw.get("itemId", item_id),
                lookback=lookback,
                timestep=raw.get("timestep"),
                startTimestamp=raw.get("startTimestamp"),
                endTimestamp=raw.get("endTimestamp"),
                points=[TimeseriesPoint.model_validate(p) for p in raw["data"]],
                fetched_at=self._now(),
            )
        except (KeyError, TypeError, AttributeError, ValidationError) as exc:
            raise UpstreamError("unexpected /timeseries response shape") from exc
