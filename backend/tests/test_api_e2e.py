"""The real service behind the real API, with only OpenRouter and Jev mocked at the network edge.

Everything else is real: the embedding model (so this is a ``live`` test), pgvector retrieval, the
LangGraph graph and its Postgres checkpointer. It proves where the user's key travels: to the two
upstream calls that need it (Jev and the chat model) and nowhere else.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import psycopg
import pytest

from app.config import Settings
from app.main import create_app
from app.rag.embed import Embedder
from app.rag.retrieve import vector_literal
from tests.conftest import TEST_UA
from tests.test_graph import checkpoint_dump

pytestmark = pytest.mark.live

KEY = "sk-or-v1-E2EKEYSHOULDNEVERLEAK0123456789"
JEV = "https://openrouter.ai/api/alpha/decisions"
CHAT = "https://openrouter.ai/api/v1/chat/completions"

DOCS = [
    (
        "Abyssal whip",
        "item",
        "The abyssal whip is a weapon requiring an Attack level of 70 to wield.",
    ),
    ("Dragon bones", "item", "Dragon bones give 72 Prayer experience when buried."),
    ("Cake", "item", "A cake heals three hitpoints over three bites."),
]


def seed(db_url: str, embedder: Embedder) -> None:
    texts = [f"{title}\n{body}" for title, _, body in DOCS]
    vectors = embedder.embed_passages(texts)
    with psycopg.connect(db_url) as conn:
        for (title, page_type, _), text, vector in zip(DOCS, texts, vectors, strict=True):
            conn.execute(
                "insert into documents (page_title, section, page_type, url, content, metadata, "
                "embedding) values (%s, '', %s, %s, %s, '{}', %s::vector)",
                (
                    title,
                    page_type,
                    f"https://oldschool.runescape.wiki/w/{title.replace(' ', '_')}",
                    text,
                    vector_literal([float(x) for x in vector]),
                ),
            )
        conn.commit()


def jev_handler(request: httpx.Request) -> httpx.Response:
    questions = json.loads(request.content)["questions"]
    answers: dict[str, Any] = {}
    if "choice" in questions:  # the router
        answers["choice"] = {
            "type": "choice",
            "choice": "wiki_general",
            "probabilities": {"wiki_general": 0.96, "quest": 0.02},
            "confidence": 0.95,
        }
    for qid in questions:  # the grader: relevance per passage + sufficiency
        if qid.startswith("passage_") or qid == "sufficient":
            answers[qid] = {"type": "noul", "noul": 0.95}
    return httpx.Response(
        200,
        json={
            "model": "typesafe/jev-1.13-test",
            "usage": {"input_tokens": 100},
            "answers": answers,
        },
    )


def chat_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "deepseek/deepseek-v4-flash",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "You need 70 Attack [Abyssal whip].",
                    },
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        },
    )


async def test_the_key_goes_to_the_two_upstream_calls_and_nowhere_else(
    db: str, respx_mock, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    # Loading the embedding model may ask the Hugging Face hub whether it is up to date; that is
    # the only call allowed through, everything else must hit a mock.
    respx_mock.route(host__regex=r"(.+\.)?huggingface\.co").pass_through()
    seed(db, Embedder(Settings(_env_file=None)))
    jev = respx_mock.post(JEV).mock(side_effect=jev_handler)
    chat = respx_mock.post(CHAT).mock(side_effect=chat_handler)

    settings = Settings(_env_file=None, supabase_db_url=db, prices_user_agent=TEST_UA)
    app = create_app(settings=settings)  # the real AgentService
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/healthz")
            response = await client.post(
                "/v1/chat",
                headers={"X-OpenRouter-Key": KEY},
                json={
                    "thread_id": "e2e-thread-0001",
                    "message": "What Attack level do I need to wield an abyssal whip?",
                    "player_state": {"account_type": "main", "members": True, "gp": 7_654_321},
                },
            )
    assert health.status_code == 200 and health.json()["documents"] == 3
    assert response.status_code == 200, response.text
    data = response.json()
    assert (
        data["answer"] == "You need 70 Attack [Abyssal whip]." and data["intent"] == "wiki_general"
    )
    assert next(s["title"] for s in data["sources"]) == "Abyssal whip"

    # both upstream services were called, each with the user's key and nothing of ours
    assert jev.call_count >= 2 and chat.call_count == 1  # router + grading; one answer
    for route in (jev, chat):
        for call in route.calls:
            assert call.request.headers["authorization"] == f"Bearer {KEY}"
    # ...and the key is nowhere else: not in the response, the logs, the checkpoints
    assert KEY not in response.text and KEY not in caplog.text
    dump = checkpoint_dump(db)
    assert b"E2EKEYSHOULDNEVERLEAK" not in dump and b"7654321" not in dump
    assert b"abyssal whip" in dump.lower()  # the conversation itself is what the thread remembers
