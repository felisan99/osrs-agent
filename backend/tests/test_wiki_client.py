from __future__ import annotations

import httpx
import pytest

from app.clients.http import RetryPolicy
from app.clients.wiki import WikiClient, WikiPage, page_url
from app.config import Settings
from app.errors import UpstreamError
from tests.conftest import TEST_UA, FakeClock, RecordingSleep

API = "https://oldschool.runescape.wiki/api.php"


@pytest.fixture
def wiki_settings() -> Settings:
    return Settings(_env_file=None, prices_user_agent=TEST_UA, wiki_request_delay_s=0)


def make(settings, http, retry, sleeper, clock=None) -> WikiClient:
    return WikiClient(settings, http=http, retry=retry, sleep=sleeper, clock=clock or FakeClock())


def raw_page(title: str, text: str, revid: int = 1, pageid: int = 10) -> dict:
    return {
        "pageid": pageid,
        "title": title,
        "revisions": [
            {
                "revid": revid,
                "timestamp": "2026-10-01T00:00:00Z",
                "slots": {"main": {"content": text}},
            }
        ],
    }


# --------------------------------------------------------------------------- URLs
@pytest.mark.parametrize(
    ("title", "url"),
    [
        ("Dragon Slayer II", "https://oldschool.runescape.wiki/w/Dragon_Slayer_II"),
        ("Cook's Assistant", "https://oldschool.runescape.wiki/w/Cook's_Assistant"),
        (
            "Money making guide/Aerial fishing",
            "https://oldschool.runescape.wiki/w/Money_making_guide/Aerial_fishing",
        ),
        (
            "Ardougne teleport (tablet)",
            "https://oldschool.runescape.wiki/w/Ardougne_teleport_(tablet)",
        ),
        ("What is this?", "https://oldschool.runescape.wiki/w/What_is_this%3F"),
        ("Café #1", "https://oldschool.runescape.wiki/w/Caf%C3%A9_%231"),
    ],
)
def test_page_url(title: str, url: str) -> None:
    assert page_url(title) == url


def test_redirect_detection() -> None:
    page = WikiPage("A", 1, 1, "t", "  #REDIRECT [[B]]")
    assert page.is_redirect and not WikiPage("A", 1, 1, "t", "Real text").is_redirect


# --------------------------------------------------------------------------- category members
async def test_category_members_follow_continue_and_send_polite_headers(
    respx_mock, wiki_settings, http, retry, sleeper
) -> None:
    pages = [
        {
            "query": {"categorymembers": [{"title": "A"}, {"title": "B"}]},
            "continue": {"cmcontinue": "page|C"},
        },
        {"query": {"categorymembers": [{"title": "C"}]}},
    ]
    route = respx_mock.get(API).mock(side_effect=[httpx.Response(200, json=p) for p in pages])

    titles = await make(wiki_settings, http, retry, sleeper).category_members("Quests")

    assert titles == ["A", "B", "C"]
    first, second = (c.request for c in route.calls)
    assert first.headers["user-agent"] == TEST_UA
    assert (
        dict(first.url.params).items()
        >= {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:Quests",
            "cmnamespace": "0",
            "maxlag": "5",
            "format": "json",
            "formatversion": "2",
        }.items()
    )
    assert second.url.params["cmcontinue"] == "page|C"


async def test_category_members_limit_truncates(
    respx_mock, wiki_settings, http, retry, sleeper
) -> None:
    respx_mock.get(API).mock(
        return_value=httpx.Response(
            200, json={"query": {"categorymembers": [{"title": t} for t in "ABCDE"]}}
        )
    )
    assert await make(wiki_settings, http, retry, sleeper).category_members("X", limit=3) == [
        "A",
        "B",
        "C",
    ]


async def test_subcategories_strip_the_namespace_prefix(
    respx_mock, wiki_settings, http, retry, sleeper
) -> None:
    respx_mock.get(API).mock(
        return_value=httpx.Response(
            200, json={"query": {"categorymembers": [{"title": "Category:Free-to-play quests"}]}}
        )
    )
    assert await make(wiki_settings, http, retry, sleeper).subcategories("Quests") == [
        "Free-to-play quests"
    ]


# --------------------------------------------------------------------------- page content
async def test_iter_category_pages_streams_batches_and_skips_missing(
    respx_mock, wiki_settings, http, retry, sleeper
) -> None:
    first = {
        "query": {
            "pages": [
                raw_page("Cook's Assistant", "== Walkthrough ==\nText", revid=7),
                {"title": "Gone", "missing": True},
            ]
        },
        "continue": {"gcmcontinue": "page|X", "continue": "gcmcontinue||"},
    }
    second = {"query": {"pages": [raw_page("Rune Mysteries", "More text", revid=9, pageid=11)]}}
    route = respx_mock.get(API).mock(
        side_effect=[httpx.Response(200, json=first), httpx.Response(200, json=second)]
    )

    pages = [
        p async for p in make(wiki_settings, http, retry, sleeper).iter_category_pages("Quests")
    ]

    assert [(p.title, p.revision_id, p.page_id) for p in pages] == [
        ("Cook's Assistant", 7, 10),
        ("Rune Mysteries", 9, 11),
    ]
    assert pages[0].wikitext.startswith("== Walkthrough ==") and pages[0].timestamp.startswith(
        "2026-10-01"
    )
    assert route.calls[0].request.url.params["gcmlimit"] == "50"
    assert route.calls[1].request.url.params["gcmcontinue"] == "page|X"


async def test_iter_category_pages_limit_stops_early(
    respx_mock, wiki_settings, http, retry, sleeper
) -> None:
    body = {
        "query": {"pages": [raw_page(f"P{i}", "t", pageid=i) for i in range(5)]},
        "continue": {"gcmcontinue": "z"},
    }
    route = respx_mock.get(API).mock(return_value=httpx.Response(200, json=body))
    pages = [
        p async for p in make(wiki_settings, http, retry, sleeper).iter_category_pages("X", limit=2)
    ]
    assert len(pages) == 2 and route.call_count == 1  # did not ask for the next batch


async def test_fetch_pages_batches_titles_in_fifties(
    respx_mock, wiki_settings, http, retry, sleeper
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        titles = request.url.params["titles"].split("|")
        return httpx.Response(
            200,
            json={"query": {"pages": [raw_page(t, "x", pageid=i) for i, t in enumerate(titles)]}},
        )

    route = respx_mock.get(API).mock(side_effect=handler)
    titles = [f"Page {i}" for i in range(120)]
    pages = await make(wiki_settings, http, retry, sleeper).fetch_pages(titles)
    assert route.call_count == 3 and len(pages) == 120
    assert [len(c.request.url.params["titles"].split("|")) for c in route.calls] == [50, 50, 20]


# --------------------------------------------------------------------------- search / render
async def test_search_returns_hits_in_order(
    respx_mock, wiki_settings, http, retry, sleeper
) -> None:
    respx_mock.get(API).mock(
        return_value=httpx.Response(
            200,
            json={
                "query": {
                    "search": [
                        {
                            "title": "Zulrah",
                            "snippet": "Zulrah is a <span>boss</span>",
                            "wordcount": 4200,
                        },
                        {"title": "Zulrah's scales", "snippet": "scales"},
                    ]
                }
            },
        )
    )
    hits = await make(wiki_settings, http, retry, sleeper).search("zulrah drops", limit=2)
    assert [(h.title, h.word_count) for h in hits] == [("Zulrah", 4200), ("Zulrah's scales", None)]


async def test_render_html(respx_mock, wiki_settings, http, retry, sleeper) -> None:
    respx_mock.get(API).mock(
        return_value=httpx.Response(200, json={"parse": {"text": "<table></table>"}})
    )
    assert (
        await make(wiki_settings, http, retry, sleeper).render_html("Money making guide")
        == "<table></table>"
    )
    respx_mock.get(API).mock(return_value=httpx.Response(200, json={"parse": {}}))
    with pytest.raises(UpstreamError, match="no rendered text"):
        await make(wiki_settings, http, retry, sleeper).render_html("Nope")


# --------------------------------------------------------------------------- politeness & errors
async def test_maxlag_is_honoured_with_retry_after_then_succeeds(
    respx_mock, wiki_settings, http, retry, sleeper: RecordingSleep
) -> None:
    lagged = httpx.Response(
        200, headers={"Retry-After": "2"}, json={"error": {"code": "maxlag", "info": "waiting"}}
    )
    ok = httpx.Response(200, json={"query": {"categorymembers": [{"title": "A"}]}})
    route = respx_mock.get(API).mock(side_effect=[lagged, ok])
    assert await make(wiki_settings, http, retry, sleeper).category_members("X") == ["A"]
    assert route.call_count == 2 and sleeper.delays == [2.0]


async def test_persistent_maxlag_gives_up(respx_mock, wiki_settings, http, retry, sleeper) -> None:
    respx_mock.get(API).mock(
        return_value=httpx.Response(200, json={"error": {"code": "maxlag", "info": "lagged"}})
    )
    with pytest.raises(UpstreamError, match=r"lagged|maxlag"):
        await make(wiki_settings, http, retry, sleeper).category_members("X")


async def test_api_errors_and_bad_responses_are_upstream_errors(
    respx_mock, wiki_settings, http, retry, sleeper
) -> None:
    client = make(wiki_settings, http, retry, sleeper)
    respx_mock.get(API).mock(
        return_value=httpx.Response(200, json={"error": {"code": "badtitle", "info": "Bad title"}})
    )
    with pytest.raises(UpstreamError, match="badtitle"):
        await client.category_members("X")
    respx_mock.get(API).mock(return_value=httpx.Response(200, text="<html>not json</html>"))
    with pytest.raises(UpstreamError, match="non-JSON"):
        await client.category_members("X")
    respx_mock.get(API).mock(return_value=httpx.Response(404, text="nope"))
    with pytest.raises(UpstreamError, match="HTTP 404"):
        await client.category_members("X")


async def test_requests_are_spaced_by_the_configured_delay(
    respx_mock, http, retry, sleeper
) -> None:
    settings = Settings(_env_file=None, prices_user_agent=TEST_UA, wiki_request_delay_s=0.3)
    clock = FakeClock(start=1_000.0)
    respx_mock.get(API).mock(
        return_value=httpx.Response(200, json={"query": {"categorymembers": []}})
    )
    client = WikiClient(settings, http=http, retry=retry, sleep=sleeper, clock=clock)
    await client.category_members("A")  # first request: nothing to wait for
    await client.category_members("B")  # immediately after: must wait the full delay
    assert sleeper.delays == [pytest.approx(0.3)]


def test_wiki_user_agent_falls_back_to_the_prices_one_and_is_validated() -> None:
    assert Settings(_env_file=None, prices_user_agent=TEST_UA).effective_wiki_user_agent == TEST_UA
    dedicated = Settings(
        _env_file=None,
        prices_user_agent=TEST_UA,
        wiki_user_agent="osrs-agent-ingest - me@example.com",
    )
    assert dedicated.effective_wiki_user_agent == "osrs-agent-ingest - me@example.com"
    with pytest.raises(ValueError, match="generic client string"):
        Settings(_env_file=None, prices_user_agent=TEST_UA, wiki_user_agent="python-requests/2.0")


def test_retry_policy_fixture_is_the_deterministic_one(retry: RetryPolicy) -> None:
    assert retry.jitter is False and retry.max_retries == 2
