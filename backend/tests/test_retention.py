"""Retention of the conversation memory over a real Postgres and the real checkpointer."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace

import psycopg

from app.graph.retention import PruneResult, delete_stale_threads, prune_thread, stats
from app.service import AgentService
from tests.graph_helpers import FakeJev, FakeLLM, harness


def connector(db_url: str):
    @asynccontextmanager
    async def connect():
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            yield conn

    return connect


def condense(call) -> dict[str, str]:
    return {"query": "abyssal whip", "language": "en"}


def counts(db_url: str, thread: str) -> dict[str, int]:
    with psycopg.connect(db_url) as conn:
        return {
            table: conn.execute(
                f"select count(*) from {table} where thread_id = %s", (thread,)
            ).fetchone()[0]
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes")
        }


async def test_pruning_keeps_the_latest_checkpoint_and_the_memory_still_works(db) -> None:
    llm = FakeLLM(condense=condense, answer=lambda call: "You need 70 Attack [Abyssal whip].")
    jev = FakeJev()
    async with harness(db, llm=llm, jev=jev) as h:
        for question in ("abyssal whip one", "abyssal whip two", "abyssal whip three"):
            await h.ask(question, thread="busy")
        await h.ask("abyssal whip elsewhere", thread="other")
        before, other_before = counts(db, "busy"), counts(db, "other")
        assert before["checkpoints"] > 6 and before["checkpoint_writes"] > 0

        result = await prune_thread(connector(db), "busy")

        after = counts(db, "busy")
        assert after["checkpoints"] == 1  # only the latest
        assert (
            after["checkpoint_writes"] == 0
            or after["checkpoint_writes"] < before["checkpoint_writes"]
        )
        assert after["checkpoint_blobs"] < before["checkpoint_blobs"]
        assert result == PruneResult(
            before["checkpoints"] - 1,
            before["checkpoint_writes"] - after["checkpoint_writes"],
            before["checkpoint_blobs"] - after["checkpoint_blobs"],
        )
        assert counts(db, "other") == other_before  # another thread is not touched

        # the memory survived: the next turn still sees the earlier ones
        state = await h.ask("abyssal whip four", thread="busy")
        assert [m["role"] for m in state["messages"]] == ["user", "assistant"] * 4
        assert state["messages"][0]["content"] == "abyssal whip one"
        assert "abyssal whip three" in jev.route_states[-1]["conversation_summary"]


async def test_pruning_an_unknown_thread_is_a_no_op(db) -> None:
    async with harness(db):
        assert (await prune_thread(connector(db), "never-existed")).rows == 0


async def test_idle_threads_are_deleted_and_recent_ones_stay(db) -> None:
    llm = FakeLLM(condense=condense, answer=lambda call: "ok [Abyssal whip]")
    async with harness(db, llm=llm) as h:
        await h.ask("abyssal whip", thread="fresh")
        await h.ask("abyssal whip", thread="idle")
        with psycopg.connect(db) as conn:
            conn.execute(
                "update checkpoints set checkpoint = jsonb_set(checkpoint, '{ts}', "
                "to_jsonb((now() - interval '60 days')::text)) where thread_id = 'idle'"
            )
        assert await delete_stale_threads(connector(db), 0) == 0  # 0 = never delete
        assert await delete_stale_threads(connector(db), 30) == 1
        assert sum(counts(db, "idle").values()) == 0
        assert counts(db, "fresh")["checkpoints"] > 0
        assert await delete_stale_threads(connector(db), 30) == 0  # nothing else is idle

        summary = await stats(connector(db))
        assert summary["threads"] >= 1 and summary["bytes"] > 0


async def test_a_failing_prune_never_fails_the_turn(caplog) -> None:
    class Broken:
        def connection(self):
            raise RuntimeError("database went away")

    service = SimpleNamespace(pool=Broken())
    with caplog.at_level(logging.WARNING):
        await AgentService._prune(service, "any-thread")  # type: ignore[arg-type]
    assert "could not prune" in caplog.text


async def test_a_turn_prunes_its_own_thread_after_the_graph_ran(monkeypatch, settings) -> None:
    import httpx

    order: list[str] = []

    class StubGraph:
        async def ainvoke(self, *args, **kwargs):
            order.append("graph")
            assert kwargs["config"] == {"configurable": {"thread_id": "thread-42"}}
            return {"answer": "ok", "intent": "wiki_general"}

    async def fake_prune(connect, thread_id):
        order.append(f"prune:{thread_id}")
        return PruneResult()

    monkeypatch.setattr("app.service.prune_thread", fake_prune)
    async with httpx.AsyncClient() as http:
        service = AgentService(
            settings,
            http,
            SimpleNamespace(),
            StubGraph(),
            SimpleNamespace(),
            SimpleNamespace(connection=None),
        )
        result = await service.chat(
            thread_id="thread-42", message="hi", api_key="sk-or-v1-0123456789abcdef"
        )
    assert result.answer == "ok"
    assert order == ["graph", "prune:thread-42"]  # pruned after the turn, only that thread


async def test_stats_of_a_database_without_conversations_is_all_zeros(scratch_db) -> None:
    # migrations only: LangGraph creates its tables the first time the API starts
    assert await stats(connector(scratch_db)) == {
        "threads": 0,
        "checkpoints": 0,
        "checkpoint_blobs": 0,
        "checkpoint_writes": 0,
        "bytes": 0,
    }
