"""The real embedding model (~130 MB download once; excluded from the default run: `-m live`)."""

from __future__ import annotations

import math

import pytest

from app.config import Settings
from app.rag.embed import Embedder, profile_for

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder(Settings(_env_file=None))


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_vectors_are_384_dim_and_normalised(embedder: Embedder) -> None:
    vectors = embedder.embed_passages(
        ["Dragon bones give 72 Prayer experience.", "Cake heals 3 hitpoints."]
    )
    assert vectors.shape == (2, 384)
    for row in vectors:
        assert math.isclose(sum(float(x) ** 2 for x in row), 1.0, rel_tol=1e-3)
    assert len(embedder.embed_query("how much prayer xp do dragon bones give")) == 384


def test_an_english_question_retrieves_the_passage_that_answers_it(embedder: Embedder) -> None:
    # Spanish never reaches the embedder: the graph translates the question into an English
    # standalone query first (graph/nodes/condense.py), so the model only has to be good at English.
    passages = [
        "Dragon bones give 72 Prayer experience when buried, or 252 at a gilded altar.",
        "The abyssal whip is a one-handed melee weapon requiring 70 Attack.",
        "Cooked lobster heals 12 hitpoints and is a common food.",
    ]
    stored = embedder.embed_passages(passages)
    query = embedder.embed_query("How much Prayer experience do dragon bones give when buried?")
    scores = [cosine(query, row) for row in stored]
    assert scores.index(max(scores)) == 0, scores
    whip = embedder.embed_query("What Attack level do I need to wield the abyssal whip?")
    assert max(range(3), key=lambda i: cosine(whip, stored[i])) == 1


def test_token_count_uses_the_model_tokenizer(embedder: Embedder) -> None:
    short = embedder.count_tokens("abyssal whip")
    assert 3 <= short <= 12
    assert embedder.count_tokens("abyssal whip " * 50) > 5 * short


def test_the_default_model_is_the_one_the_prefix_profile_knows() -> None:
    settings = Settings(_env_file=None)
    assert settings.embedding_model == "Snowflake/snowflake-arctic-embed-s"
    profile = profile_for(settings)
    assert (
        profile.query_prefix.startswith("Represent this sentence") and profile.passage_prefix == ""
    )


def test_a_dimension_mismatch_fails_loudly() -> None:
    wrong = Embedder(Settings(_env_file=None, embedding_dim=768))
    with pytest.raises(RuntimeError, match="rebuild the index"):
        _ = wrong.model


def test_text_over_the_model_limit_is_truncated_instead_of_crashing(embedder: Embedder) -> None:
    long_text = "abyssal whip " * 600  # ~1,800 tokens, far over the 512 the model accepts
    assert embedder.count_tokens(long_text) > 1_500
    vectors = embedder.embed_passages([long_text, "short"])
    assert vectors.shape == (2, 384)
    # a pasted wall of text as a question must not take the API down either
    assert len(embedder.embed_query(long_text)) == 384
