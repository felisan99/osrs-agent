"""Raw Jev probabilities in Postgres (SPEC §9 ``jev_judgments``), for threshold tuning.

Written ONLY in eval/debug runs, never with player data or the user's key: the rows hold a run id,
a kind (route | relevance | sufficiency), a reference to the input and the probabilities.
"""

from __future__ import annotations

from psycopg.types.json import Jsonb

from app.rag.retrieve import ConnectionFactory

KINDS = ("route", "relevance", "sufficiency")


class PostgresJudgmentLog:
    def __init__(self, connect: ConnectionFactory, run_id: str) -> None:
        self._connect = connect
        self.run_id = run_id

    async def record(self, kind: str, input_ref: str, probs: dict[str, float]) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        async with self._connect() as conn:
            await conn.execute(
                "insert into jev_judgments (run_id, kind, input_ref, probs) "
                "values (%s, %s, %s, %s)",
                (self.run_id, kind, input_ref, Jsonb(probs)),
            )
            await conn.commit()
