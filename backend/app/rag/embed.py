"""Local embeddings (SPEC §2) with sentence-transformers.

* Local on purpose: embedding a Spanish query against an English wiki must not spend the user's
  BYOK credits (and ingestion embeds ~40k chunks).
* Every model family has its own prefix convention (E5: ``"passage: "`` / ``"query: "``; BGE and
  Arctic: an instruction in front of the *query* only; Nomic: ``search_query: `` /
  ``search_document: ``). Using the wrong one quietly costs retrieval quality, so the prefixes
  come from ``profile_for`` (overridable with ``EMBEDDING_QUERY_PREFIX`` /
  ``EMBEDDING_PASSAGE_PREFIX``) and are applied here, in one place.
* Vectors are L2-normalised, so cosine distance (``<=>`` in pgvector) is the right metric.
* ``torch`` and the model are imported/loaded lazily: importing this module is free.
* One encode at a time: concurrent ``encode`` calls from several threads crash the process
  (SIGSEGV on Apple's MPS backend, verified 2026-10-14), and the API embeds queries from worker
  threads. A lock serialises them; a query embedding takes a few milliseconds.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings

MAX_SEQ_LENGTH = 512  # chunks are budgeted below this (``Settings.chunk_target_max_tokens``)
BGE_QUERY = "Represent this sentence for searching relevant passages: "
QWEN_QUERY = (
    "Instruct: Given a question about Old School RuneScape, retrieve the wiki passages that "
    "answer it\nQuery: "
)


@dataclass(frozen=True)
class ModelProfile:
    query_prefix: str = ""
    passage_prefix: str = ""
    left_padding: bool = False  # decoder-based embedders pool the last token


# First match wins; matched against the lower-cased model name.
_PROFILES: tuple[tuple[str, ModelProfile], ...] = (
    ("e5", ModelProfile("query: ", "passage: ")),
    ("bge-m3", ModelProfile()),
    ("bge-", ModelProfile(BGE_QUERY)),
    ("arctic-embed-l-v2", ModelProfile("query: ")),
    ("arctic-embed", ModelProfile(BGE_QUERY)),
    ("mxbai-embed", ModelProfile(BGE_QUERY)),
    ("nomic-embed", ModelProfile("search_query: ", "search_document: ")),
    ("qwen3-embedding", ModelProfile(QWEN_QUERY, left_padding=True)),
)


def profile_for(settings: Settings) -> ModelProfile:
    """The prefix convention of the configured model, with the settings' overrides applied."""
    name = settings.embedding_model.lower()
    base = next((p for key, p in _PROFILES if key in name), ModelProfile())
    return ModelProfile(
        query_prefix=(
            base.query_prefix
            if settings.embedding_query_prefix is None
            else settings.embedding_query_prefix
        ),
        passage_prefix=(
            base.passage_prefix
            if settings.embedding_passage_prefix is None
            else settings.embedding_passage_prefix
        ),
        left_padding=base.left_padding,
    )


def best_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    def __init__(self, settings: Settings | None = None, *, device: str | None = None) -> None:
        self._settings = settings or get_settings()
        self._profile = profile_for(self._settings)
        self._device = device
        self._model: Any = None
        self._encode_lock = threading.Lock()

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {}
            if self._profile.left_padding:
                kwargs["processor_kwargs"] = {"padding_side": "left"}
            model = SentenceTransformer(
                self._settings.embedding_model, device=self._device or best_device(), **kwargs
            )
            # The limit encode() truncates at. (sentence-transformers keeps it in the tokenizer's
            # model_max_length: raising that to silence a warning, as this code once did, turned
            # truncation off and made any text over 512 tokens crash the encode.)
            model.max_seq_length = MAX_SEQ_LENGTH
            dimension_of = getattr(model, "get_embedding_dimension", None) or (
                model.get_sentence_embedding_dimension
            )
            dim = dimension_of()
            if dim != self._settings.embedding_dim:
                raise RuntimeError(
                    f"{self._settings.embedding_model} produces {dim}-dim vectors but "
                    f"EMBEDDING_DIM={self._settings.embedding_dim}; set EMBEDDING_DIM={dim} and "
                    "rebuild the index (uv run python -m scripts.ingest load --force)"
                )
            self._model = model
        return self._model

    def count_tokens(self, text: str) -> int:
        """Token count with the model's own tokenizer (what the 512-token limit applies to)."""
        tokenizer = self.model.tokenizer
        # verbose=False: counting an oversized block before it is split is not a mistake
        return len(
            tokenizer.encode(
                self._profile.passage_prefix + text, add_special_tokens=True, verbose=False
            )
        )

    def embed_passages(self, texts: Sequence[str], *, show_progress: bool = False) -> Any:
        """``(n, dim)`` float32 array of normalised vectors for stored text."""
        with self._encode_lock:
            return self.model.encode(
                [self._profile.passage_prefix + t for t in texts],
                batch_size=self._settings.embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

    def embed_query(self, text: str) -> list[float]:
        with self._encode_lock:
            vector = self.model.encode(
                self._profile.query_prefix + text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,  # one query per request: a progress bar is just log noise
            )
        return [float(x) for x in vector]
