"""Which prefix convention each embedding model family gets (no model is loaded)."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.rag.embed import BGE_QUERY, ModelProfile, profile_for


def profile(model: str, **settings: str) -> ModelProfile:
    return profile_for(Settings(_env_file=None, embedding_model=model, **settings))


@pytest.mark.parametrize(
    ("model", "query", "passage"),
    [
        ("intfloat/multilingual-e5-small", "query: ", "passage: "),
        ("intfloat/e5-base-v2", "query: ", "passage: "),
        ("BAAI/bge-base-en-v1.5", BGE_QUERY, ""),
        ("BAAI/bge-large-en-v1.5", BGE_QUERY, ""),
        ("BAAI/bge-m3", "", ""),  # bge-m3 takes no instruction, unlike the English bge models
        ("Snowflake/snowflake-arctic-embed-m-v1.5", BGE_QUERY, ""),
        ("Snowflake/snowflake-arctic-embed-l-v2.0", "query: ", ""),
        ("mixedbread-ai/mxbai-embed-large-v1", BGE_QUERY, ""),
        ("nomic-ai/nomic-embed-text-v1.5", "search_query: ", "search_document: "),
        ("some/unknown-model", "", ""),
    ],
)
def test_each_family_gets_its_own_prefixes(model: str, query: str, passage: str) -> None:
    found = profile(model)
    assert (found.query_prefix, found.passage_prefix) == (query, passage)


def test_decoder_based_models_pad_on_the_left_and_instruct_the_query() -> None:
    found = profile("Qwen/Qwen3-Embedding-0.6B")
    assert found.left_padding and found.query_prefix.startswith("Instruct:")
    assert found.passage_prefix == ""
    assert not profile("BAAI/bge-base-en-v1.5").left_padding


def test_settings_override_the_convention_and_an_empty_string_means_no_prefix() -> None:
    forced = profile("intfloat/multilingual-e5-small", embedding_query_prefix="q: ")
    assert (forced.query_prefix, forced.passage_prefix) == ("q: ", "passage: ")
    none = profile("intfloat/multilingual-e5-small", embedding_passage_prefix="")
    assert (none.query_prefix, none.passage_prefix) == ("query: ", "")
