"""Test doubles and a harness for the graph: fake LLM and Jev, real Postgres, real checkpointer.

The graph is exercised end to end (router -> condense -> retrieval in Postgres -> grading ->
generation -> checkpointed memory) with nothing but the two network services faked:

* ``FakeLLM`` answers by role: the condense/rewrite JSON calls and the answer generation are told
  apart by their system prompts, and every call is recorded so tests can read the prompts.
* ``FakeJev`` answers the router's Choice and the grader's Noul batch from small rules, and records
  what it was asked.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.api.schemas import PlayerState
from app.clients.jev import (
    ChoiceAnswer,
    DecisionsResponse,
    JevUsage,
    NoulAnswer,
)
from app.clients.openrouter import ChatResult, ChatUsage, ToolCall
from app.config import Settings
from app.graph.build import build_graph
from app.graph.state import Deps, RunContext
from app.rag.grade import Grader
from app.rag.retrieve import Passage, Retriever, vector_literal
from app.rag.titles import TitleIndex
from app.tools.context import ToolContext
from app.tools.quests import QuestStore

DIM = 384


def unit(axis: int) -> list[float]:
    vector = [0.0] * DIM
    vector[axis] = 1.0
    return vector


def text_result(content: str, cost: float = 0.0001) -> ChatResult:
    return ChatResult(
        content=content,
        finish_reason="stop",
        usage=ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=cost),
        assistant_message={"role": "assistant", "content": content},
    )


def tool_result(*calls: tuple[str, dict[str, Any]]) -> ChatResult:
    tool_calls = [
        ToolCall(id=f"call_{i}", name=name, arguments=args, raw_arguments="{}")
        for i, (name, args) in enumerate(calls)
    ]
    return ChatResult(
        content=None,
        tool_calls=tool_calls,
        finish_reason="tool_calls",
        usage=ChatUsage(total_tokens=15, cost=0.0001),
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": "{}"},
                }
                for c in tool_calls
            ],
        },
    )


# --------------------------------------------------------------------------- fake LLM
@dataclass
class Call:
    messages: list[dict[str, Any]]
    kwargs: dict[str, Any]

    @property
    def system(self) -> str:
        return self.messages[0]["content"]

    @property
    def last_user(self) -> str:
        return next(m["content"] for m in reversed(self.messages) if m["role"] == "user")


class FakeLLM:
    """Scripted chat model. ``condense``/``rewrite`` are JSON-producing callables, ``answer`` makes
    the final text, ``tool_script`` (if given) drives tool-calling workflows round by round."""

    def __init__(
        self,
        *,
        condense: Callable[[Call], dict[str, str]] | None = None,
        rewrite: Callable[[Call], dict[str, str]] | None = None,
        answer: Callable[[Call], str] | None = None,
        tool_script: Callable[[Call, int], ChatResult] | None = None,
    ) -> None:
        self.calls: list[Call] = []
        self._condense = condense
        self._rewrite = rewrite
        self._answer = answer or (lambda call: "Respuesta de prueba [Abyssal whip].")
        self._tool_script = tool_script
        self.tool_rounds = 0

    def __repr__(self) -> str:
        return "FakeLLM()"

    def of_kind(self, marker: str) -> list[Call]:
        return [c for c in self.calls if marker in c.system]

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        call = Call(list(messages), kwargs)  # a copy: workflows keep appending to the list
        self.calls.append(call)
        system = call.system
        if "You prepare a player's chat message" in system:
            assert self._condense is not None, "unexpected condense call"
            return text_result(_json(self._condense(call)))
        if "did not find enough information" in system:
            assert self._rewrite is not None, "unexpected rewrite call"
            return text_result(_json(self._rewrite(call)))
        if kwargs.get("tools") and self._tool_script is not None:
            reply = self._tool_script(call, self.tool_rounds)
            self.tool_rounds += 1
            return reply
        return text_result(self._answer(call))


def _json(data: dict[str, str]) -> str:
    import json

    return json.dumps(data)


# --------------------------------------------------------------------------- fake Jev
class FakeJev:
    """Router: ``route(message) -> (intent, probabilities, confidence)``. Grader: ``relevance``
    scores a passage text for the query, ``sufficiency`` scores the set."""

    def __init__(
        self,
        *,
        route: Callable[[str], tuple[str, dict[str, float], float]] | None = None,
        relevance: Callable[[str, str], float] | None = None,
        sufficiency: Callable[[str, list[str]], float] | None = None,
    ) -> None:
        self.route = route or (lambda message: ("wiki_general", {"wiki_general": 0.95}, 0.9))
        self.relevance = relevance or (lambda query, text: 0.9 if "whip" in text.lower() else 0.1)
        self.sufficiency = sufficiency or (lambda query, texts: 0.9 if texts else 0.0)
        self.route_states: list[dict[str, Any]] = []
        self.grade_calls: list[dict[str, Any]] = []

    def __repr__(self) -> str:
        return "FakeJev()"

    async def choose(self, state: Any, instructions: Any, options: Any):
        self.route_states.append(state)
        choice, probabilities, confidence = self.route(state["user_message"])
        answer = ChoiceAnswer(
            type="choice", choice=choice, probabilities=probabilities, confidence=confidence
        )
        return answer, DecisionsResponse(
            model="fake-jev", usage=JevUsage(cost=0.00002), answers={"choice": answer}
        )

    async def decide(self, state: dict[str, Any], questions: dict[str, Any]) -> DecisionsResponse:
        self.grade_calls.append({"state": state, "questions": list(questions)})
        query = state["query"]
        texts = [v for k, v in state.items() if k.startswith("passage_")]
        answers: dict[str, Any] = {}
        for key, text in state.items():
            if key.startswith("passage_"):
                answers[key] = NoulAnswer(type="noul", noul=self.relevance(query, text))
        if "sufficient" in questions:
            answers["sufficient"] = NoulAnswer(type="noul", noul=self.sufficiency(query, texts))
        return DecisionsResponse(model="fake-jev", usage=JevUsage(cost=0.00001), answers=answers)


# --------------------------------------------------------------------------- live search
class FakeLive:
    def __init__(self, passages: Sequence[Passage] = ()) -> None:
        self.passages_to_return = list(passages)
        self.queries: list[str] = []

    async def passages(self, query: str, *, limit: int) -> list[Passage]:
        self.queries.append(query)
        return self.passages_to_return[:limit]


# --------------------------------------------------------------------------- harness
def add_chunk(
    conn: psycopg.Connection,
    title: str,
    section: str,
    body: str,
    axis: int,
    page_type: str = "other",
) -> None:
    header = title + (f" > {section}" if section else "")
    conn.execute(
        "insert into documents (page_title, section, page_type, url, content, metadata, embedding)"
        " values (%s, %s, %s, %s, %s, '{}', %s::vector)",
        (
            title,
            section,
            page_type,
            f"https://oldschool.runescape.wiki/w/{title.replace(' ', '_')}",
            f"{header}\n{body}",
            vector_literal(unit(axis)),
        ),
    )


def seed_corpus(db_url: str) -> None:
    with psycopg.connect(db_url) as conn:
        add_chunk(
            conn, "Abyssal whip", "", "The abyssal whip requires an Attack level of 70.", 0, "item"
        )
        add_chunk(conn, "Abyssal whip", "Combat stats", "Slash +82, speed 4 ticks.", 3, "item")
        add_chunk(
            conn,
            "Monkey Madness II",
            "Details",
            "Requirements: 69 Slayer, 70 Crafting. Items required: A m'speak amulet.",
            1,
            "quest",
        )
        add_chunk(
            conn,
            "M'speak amulet",
            "",
            "The m'speak amulet can be bought from the Ape Atoll gift shop for 1,000 coins.",
            2,
            "item",
        )
        add_chunk(conn, "Cake", "", "A cake heals three hitpoints.", 4, "item")
        conn.commit()


def embed_by_keyword(text: str) -> list[float]:
    lowered = text.lower()
    if "amulet" in lowered:
        return unit(2)
    if "whip" in lowered:
        return unit(0)
    if "monkey" in lowered:
        return unit(1)
    return unit(60)


@dataclass
class Harness:
    graph: Any
    llm: FakeLLM
    jev: FakeJev
    deps: Deps
    live: FakeLive | None = None
    settings: Settings = field(default_factory=lambda: Settings(_env_file=None))
    # A fresh conversation per harness: the checkpoint tables outlive a test's data cleanup.
    thread: str = field(default_factory=lambda: f"thread-{uuid.uuid4().hex[:12]}")

    def context(self, player: PlayerState | None = None) -> RunContext:
        return RunContext(
            deps=self.deps,
            llm=self.llm,  # type: ignore[arg-type]
            jev=self.jev,
            grader=Grader(self.jev, self.settings),  # type: ignore[arg-type]
            player=player,
        )

    async def ask(
        self,
        message: str,
        *,
        thread: str | None = None,
        player: PlayerState | None = None,
    ) -> dict[str, Any]:
        return await self.graph.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread or self.thread}},
            context=self.context(player),
        )


@asynccontextmanager
async def harness(
    db_url: str,
    *,
    llm: FakeLLM | None = None,
    jev: FakeJev | None = None,
    live: FakeLive | None = None,
    tools: ToolContext | None = None,
    quests: QuestStore | None = None,
    **settings: Any,
):
    """A compiled graph over a seeded Postgres, with a real ``AsyncPostgresSaver``."""
    cfg = Settings(
        _env_file=None, prices_user_agent="osrs-agent-tests - tests@example.com", **settings
    )
    seed_corpus(db_url)

    @asynccontextmanager
    async def connect():
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            yield conn

    titles = await TitleIndex.load(connect)
    retriever = Retriever(connect, embed_by_keyword, cfg, titles)
    if tools is None:
        from tests.test_moneymakers import tool_ctx

        tools = tool_ctx([])
    deps = Deps(
        settings=cfg,
        retriever=retriever,
        tools=tools,
        titles=titles,
        quests=quests or QuestStore([]),
        live=live,
    )
    async with AsyncPostgresSaver.from_conn_string(db_url) as saver:
        await saver.setup()
        yield Harness(
            graph=build_graph(cfg, saver),
            llm=llm or FakeLLM(),
            jev=jev or FakeJev(),
            deps=deps,
            live=live,
            settings=cfg,
        )
