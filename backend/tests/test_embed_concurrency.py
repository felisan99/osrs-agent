"""Concurrent ``encode`` calls crash the process on Apple's MPS backend (SIGSEGV, found while
running the M4 check: two queries embedded from two worker threads). The embedder must therefore
run one encode at a time, whatever calls it. A fake model detects overlapping calls without needing
the real weights (the real model's behaviour is covered by the ``live`` tests in test_embed.py).
"""

from __future__ import annotations

import asyncio
import threading
import time

from app.config import Settings
from app.rag.embed import Embedder


class OverlapDetector:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self._guard = threading.Lock()

    def encode(self, texts, **kwargs):
        with self._guard:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01)  # long enough for another thread to barge in if nothing stops it
        with self._guard:
            self.active -= 1
        return [0.0] * 384 if isinstance(texts, str) else [[0.0] * 384 for _ in texts]


def embedder_with(model: OverlapDetector) -> Embedder:
    embedder = Embedder(Settings(_env_file=None))
    embedder._model = model  # skip loading the real weights
    return embedder


async def test_queries_embedded_from_many_threads_never_overlap() -> None:
    model = OverlapDetector()
    embedder = embedder_with(model)
    await asyncio.gather(
        *(asyncio.to_thread(embedder.embed_query, f"question {i}") for i in range(12))
    )
    assert model.calls == 12 and model.max_active == 1


async def test_queries_and_passage_batches_share_the_same_lock() -> None:
    model = OverlapDetector()
    embedder = embedder_with(model)
    jobs = [asyncio.to_thread(embedder.embed_query, "q") for _ in range(6)]
    jobs += [asyncio.to_thread(embedder.embed_passages, ["a", "b"]) for _ in range(6)]
    await asyncio.gather(*jobs)
    assert model.calls == 12 and model.max_active == 1
