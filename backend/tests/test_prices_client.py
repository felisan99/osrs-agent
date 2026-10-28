from __future__ import annotations

import asyncio

import httpx
import pytest

from app.clients.http import RetryPolicy
from app.clients.prices import LOOKBACKS, PricesClient
from app.config import Settings
from app.errors import UpstreamError
from tests.conftest import TEST_UA, FakeClock, RecordingSleep

BASE = "https://prices.runescape.wiki/api/v2/osrs"

# Shapes copied from live responses (2026-10-14).
LATEST = {
    "data": {
        "2": {"high": 267, "highTime": 1791985616, "low": 256, "lowTime": 1791985615},
        # above int32 (2_147_483_647): real items already trade here
        "20997": {
            "high": 9_364_999_989,
            "highTime": 1791985000,
            "low": 9_000_000_000,
            "lowTime": 1791984900,
        },
        # one side never traded
        "13190": {"high": 1_900_000, "highTime": 1791985100, "low": None, "lowTime": None},
    }
}
MAPPING = [
    {
        "examine": "Cannonball",
        "id": 2,
        "members": False,
        "lowalch": 3,
        "limit": 11000,
        "value": 5,
        "highalch": 5,
        "icon": "Cannonball.png",
        "name": "Cannonball",
    },
    {
        "examine": "A bond",
        "id": 13190,
        "members": False,
        "limit": 100,
        "value": 2000000,
        "icon": "Old school bond.png",
        "name": "Old school bond",
    },  # no alch values at all
]
ONE_HOUR = {
    "timestamp": 1791975600,
    "data": {
        "2": {
            "avgHighPrice": 266.18,
            "highPriceVolume": 582016,
            "avgLowPrice": 259.55,
            "lowPriceVolume": 287831,
        },
        "13190": {
            "avgHighPrice": None,
            "highPriceVolume": 0,
            "avgLowPrice": 1_850_000.5,
            "lowPriceVolume": 12,
        },
    },
}
TIMESERIES = {
    "data": [
        {
            "timestamp": 1791378000,
            "avgHighPrice": 831813.97,
            "avgLowPrice": 813230.45,
            "highPriceVolume": 62,
            "lowPriceVolume": 42,
        }
    ],
    "itemId": 4151,
    "startTimestamp": 1791378000,
    "endTimestamp": 1791982800,
    "timestep": 3600,
}


def make_client(settings: Settings, http: httpx.AsyncClient, retry: RetryPolicy, clock: FakeClock):
    return PricesClient(settings, http=http, retry=retry, clock=clock)


async def test_latest_parses_big_ints_nulls_and_sends_descriptive_user_agent(
    respx_mock, settings, http, retry, clock
) -> None:
    route = respx_mock.get(f"{BASE}/latest").mock(return_value=httpx.Response(200, json=LATEST))
    client = make_client(settings, http, retry, clock)

    snap = await client.latest()

    assert route.calls.last.request.headers["user-agent"] == TEST_UA
    assert snap.data[20997].high == 9_364_999_989 > 2**31
    assert snap.data[13190].low is None and snap.data[13190].low_time is None
    assert snap.data[2].high_time == 1791985616
    assert snap.stale is False and snap.fetched_at.tzinfo is not None


async def test_latest_is_cached_and_concurrent_callers_share_one_request(
    respx_mock, settings, http, retry, clock
) -> None:
    route = respx_mock.get(f"{BASE}/latest").mock(return_value=httpx.Response(200, json=LATEST))
    client = make_client(settings, http, retry, clock)

    await asyncio.gather(*(client.latest() for _ in range(8)))
    await client.latest()
    assert route.call_count == 1  # single flight + TTL cache

    clock.advance(settings.prices_latest_ttl_s + 1)
    await client.latest()
    assert route.call_count == 2  # TTL expired


async def test_mapping_uses_long_ttl_and_tolerates_missing_fields(
    respx_mock, settings, http, retry, clock
) -> None:
    route = respx_mock.get(f"{BASE}/mapping").mock(return_value=httpx.Response(200, json=MAPPING))
    client = make_client(settings, http, retry, clock)

    snap = await client.mapping()
    assert snap.data[13190].highalch is None and snap.data[13190].limit == 100

    clock.advance(settings.prices_latest_ttl_s + 1)  # past the short TTL, still within mapping's
    await client.mapping()
    assert route.call_count == 1
    clock.advance(settings.prices_mapping_ttl_s)
    await client.mapping()
    assert route.call_count == 2


async def test_one_hour_keeps_float_averages_and_api_timestamp(
    respx_mock, settings, http, retry, clock
) -> None:
    respx_mock.get(f"{BASE}/1h").mock(return_value=httpx.Response(200, json=ONE_HOUR))
    snap = await make_client(settings, http, retry, clock).one_hour()
    assert snap.api_timestamp == 1791975600
    assert snap.data[2].avg_high == pytest.approx(266.18)
    assert snap.data[13190].avg_high is None and snap.data[13190].low_volume == 12


async def test_stale_snapshot_is_served_with_flag_when_api_is_down(
    respx_mock, settings, http, retry, clock
) -> None:
    route = respx_mock.get(f"{BASE}/latest")
    route.mock(return_value=httpx.Response(200, json=LATEST))
    client = make_client(settings, http, retry, clock)
    first = await client.latest()

    route.mock(return_value=httpx.Response(503, text="down"))
    clock.advance(settings.prices_latest_ttl_s + 1)
    served = await client.latest()
    assert served.stale is True
    assert served.fetched_at == first.fetched_at  # the caller can cite the REAL timestamp

    clock.advance(settings.prices_max_stale_s + 1)  # too old to serve
    with pytest.raises(UpstreamError, match="HTTP 503"):
        await client.latest()


async def test_timeseries_uses_lookback_not_timestep_and_validates_it(
    respx_mock, settings, http, retry, clock
) -> None:
    route = respx_mock.get(f"{BASE}/timeseries").mock(
        return_value=httpx.Response(200, json=TIMESERIES)
    )
    client = make_client(settings, http, retry, clock)

    series = await client.timeseries(4151, "7d")
    params = dict(route.calls.last.request.url.params)
    assert params == {"id": "4151", "lookback": "7d"}  # `timestep` is rejected by the API (400)
    assert series.timestep == 3600 and series.points[0].avg_high == pytest.approx(831813.97)

    await client.timeseries(4151, "7d")
    assert route.call_count == 1  # cached

    for bad in ("1h", "7days", "", "5m"):
        with pytest.raises(ValueError, match="lookback"):
            await client.timeseries(4151, bad)
    assert set(LOOKBACKS) == {"6h", "24h", "7d", "30d", "6m", "1y"}


async def test_latest_item_prefers_fresh_snapshot_else_single_id_query(
    respx_mock, settings, http, retry, clock
) -> None:
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        seen.append(params)
        if "id" in params:
            item = LATEST["data"].get(params["id"])
            return httpx.Response(200, json={"data": {params["id"]: item} if item else {}})
        return httpx.Response(200, json=LATEST)

    respx_mock.get(f"{BASE}/latest").mock(side_effect=handler)
    client = make_client(settings, http, retry, clock)

    # No fresh snapshot yet -> one small `?id=` request, then cached per id.
    price, fetched_at, stale = await client.latest_item(2)
    assert seen == [{"id": "2"}]
    assert price is not None and price.high == 267 and stale is False
    assert fetched_at.tzinfo is not None
    await client.latest_item(2)
    assert len(seen) == 1

    # A fresh full snapshot serves any other id without another request (never loop ?id=).
    await client.latest()
    assert seen[-1] == {}
    count = len(seen)
    price, _, _ = await client.latest_item(13190)
    assert price is not None and price.low is None
    assert len(seen) == count


async def test_http_errors_become_upstream_errors_with_the_api_message(
    respx_mock, settings, http, retry, clock
) -> None:
    respx_mock.get(f"{BASE}/timeseries").mock(
        return_value=httpx.Response(400, json={"error": "lookback must be a valid value"})
    )
    with pytest.raises(UpstreamError, match="lookback must be a valid value") as err:
        await make_client(settings, http, retry, clock).timeseries(4151, "7d")
    assert err.value.upstream_status == 400


async def test_transient_5xx_is_retried_with_backoff_then_succeeds(
    respx_mock, settings, http, retry, clock, sleeper: RecordingSleep
) -> None:
    route = respx_mock.get(f"{BASE}/latest").mock(
        side_effect=[
            httpx.Response(502),
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(200, json=LATEST),
        ]
    )
    snap = await make_client(settings, http, retry, clock).latest()
    assert route.call_count == 3 and len(snap.data) == 3
    assert sleeper.delays == [0.5, 1.0]  # 0.5*2^0 ; max(0.5*2^1, Retry-After=1)


async def test_network_errors_surface_as_upstream_error(
    respx_mock, settings, http, retry, clock
) -> None:
    route = respx_mock.get(f"{BASE}/latest").mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(UpstreamError, match="network error"):
        await make_client(settings, http, retry, clock).latest()
    assert route.call_count == 3  # 1 + 2 retries


async def test_malformed_payload_is_an_upstream_error(
    respx_mock, settings, http, retry, clock
) -> None:
    respx_mock.get(f"{BASE}/latest").mock(return_value=httpx.Response(200, json={"nope": 1}))
    with pytest.raises(UpstreamError, match="unexpected /latest response shape"):
        await make_client(settings, http, retry, clock).latest()
