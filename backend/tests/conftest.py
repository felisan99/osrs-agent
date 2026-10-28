from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator

import httpx
import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app.clients.http import RetryPolicy
from app.config import Settings
from app.db import apply_migrations

TEST_UA = "osrs-agent-tests - tests@example.com"
FAKE_KEY = "sk-or-v1-0123456789abcdef0123456789abcdef"


@pytest.fixture
def settings() -> Settings:
    # _env_file=None: never let a developer's local .env change test behaviour.
    return Settings(_env_file=None, prices_user_agent=TEST_UA)


class RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


@pytest.fixture
def sleeper() -> RecordingSleep:
    return RecordingSleep()


@pytest.fixture
def retry(sleeper: RecordingSleep) -> RetryPolicy:
    """2 retries, no real waiting, no jitter (deterministic delays)."""
    return RetryPolicy(
        max_retries=2, base_delay_s=0.5, max_delay_s=8.0, jitter=False, sleep=sleeper
    )


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture(scope="session")
def pg_url() -> Iterator[str]:
    """A throwaway database with every migration applied.

    Needs a reachable Postgres with pgvector (default: local Homebrew Postgres, trust auth;
    override with TEST_DB_ADMIN_URL). Skips, instead of failing, when there is none.
    """
    admin_url = os.environ.get("TEST_DB_ADMIN_URL", "postgresql://127.0.0.1:5432/postgres")
    try:
        admin = psycopg.connect(admin_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as exc:
        pytest.skip(f"no Postgres reachable at TEST_DB_ADMIN_URL: {exc.__class__.__name__}")
    name = f"osrs_agent_test_{uuid.uuid4().hex[:10]}"
    with admin:
        admin.execute(sql.SQL("create database {}").format(sql.Identifier(name)))
        try:
            url = make_conninfo(admin_url, dbname=name)
            apply_migrations(url)
            yield url
        finally:
            admin.execute(
                sql.SQL("drop database if exists {} with (force)").format(sql.Identifier(name))
            )


@pytest.fixture
def db(pg_url: str) -> str:
    """The throwaway database with every data table emptied (code may commit: no rollback)."""
    with psycopg.connect(pg_url) as connection:
        connection.execute(
            "truncate documents, moneymaking_guides, quests, jev_judgments restart identity"
        )
    return pg_url


@pytest.fixture
def scratch_db(pg_url: str) -> Iterator[str]:
    """Its own database: changing the vector dimension must not disturb the shared test one."""
    admin_url = os.environ.get("TEST_DB_ADMIN_URL", "postgresql://127.0.0.1:5432/postgres")
    name = f"osrs_agent_dim_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(sql.SQL("create database {}").format(sql.Identifier(name)))
        try:
            url = make_conninfo(admin_url, dbname=name)
            apply_migrations(url)
            yield url
        finally:
            admin.execute(
                sql.SQL("drop database if exists {} with (force)").format(sql.Identifier(name))
            )
