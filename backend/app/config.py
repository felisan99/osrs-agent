"""Application settings (SPEC §11).

Every tunable lives here; business logic never hardcodes thresholds, tax values or model slugs.
Values are read from the environment / ``backend/.env`` (see ``.env.example``).

The user's OpenRouter key is deliberately NOT a setting: it is a per-request secret that
only exists in memory for the duration of a request (SPEC §1, principle 5).
"""

from __future__ import annotations

import os
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# `.env` is looked up by absolute path, so it works from any working directory: backend/.env first,
# then the repository root's (later files override earlier ones).
BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_FILES: tuple[Path, ...] = (BACKEND_DIR / ".env", BACKEND_DIR.parent / ".env")

# User-Agents the Wiki prices API rejects ("python-requests", "curl/...", "Java/...", ...),
# plus httpx's own default which would be rejected for the same reason.
BLOCKED_USER_AGENT_PREFIXES: tuple[str, ...] = (
    "python-requests",
    "python-urllib",
    "python-httpx",
    "apache-httpclient",
    "restsharp",
    "java/",
    "curl/",
)

# Verified 2026-10-14 against https://oldschool.runescape.wiki/w/Grand_Exchange
# ("Exempt from tax") and cross-checked so every name resolves in the prices API /mapping
# (tablets are named "<City> teleport (tablet)" there; "Energy potion" is a dose family and is
# expanded to "(1)".."(4)" by app.tools.items). Override with GE_TAX_EXEMPT_ITEMS (JSON list).
DEFAULT_GE_TAX_EXEMPT_ITEMS: tuple[str, ...] = (
    "Old school bond",
    "Energy potion",
    # low-level combat consumables
    "Bronze arrow",
    "Bronze dart",
    "Iron arrow",
    "Iron dart",
    "Mind rune",
    "Steel arrow",
    "Steel dart",
    # low-level food
    "Bass",
    "Bread",
    "Cake",
    "Cooked chicken",
    "Cooked meat",
    "Herring",
    "Lobster",
    "Mackerel",
    "Meat pie",
    "Pike",
    "Salmon",
    "Shrimps",
    "Tuna",
    # teleport items
    "Ardougne teleport (tablet)",
    "Camelot teleport (tablet)",
    "Civitas illa fortis teleport",
    "Falador teleport (tablet)",
    "Games necklace(8)",
    "Kourend castle teleport (tablet)",
    "Lumbridge teleport (tablet)",
    "Ring of dueling(8)",
    "Teleport to house (tablet)",
    "Varrock teleport (tablet)",
    # tools
    "Chisel",
    "Gardening trowel",
    "Glassblowing pipe",
    "Hammer",
    "Needle",
    "Pestle and mortar",
    "Rake",
    "Saw",
    "Secateurs",
    "Seed dibber",
    "Shears",
    "Spade",
    "Watering can",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,  # `GE_TAX_CAP=` in .env means "use the default"
    )

    # --- Database (Supabase Postgres + pgvector) -------------------------------------------
    # Default: the local Homebrew Postgres 17 (trust auth, connects as your macOS user).
    supabase_db_url: str | None = "postgresql://127.0.0.1:5432/osrs_agent"

    # --- Models & endpoints ----------------------------------------------------------------
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Alpha endpoint (SPEC §2). Kept configurable: if the shape drifts, the adapter in
    # app/clients/jev.py is the only place to touch.
    jev_endpoint: str = "https://openrouter.ai/api/alpha/decisions"
    # Pinned version on purpose: `~typesafe/jev-latest` moves and would silently shift the
    # probabilities the thresholds below were tuned against.
    jev_model: str = "typesafe/jev-1.13"
    reasoning_model: str = "deepseek/deepseek-v4-flash"
    # English retrieval model, 33M parameters, 384 dims, local. Retrieval embeds the English rewrite
    # of the question (graph/nodes/condense.py), so Spanish questions need no multilingual model.
    # Chosen on the retrieval benchmark (docs/rag-quality.md): beat multilingual-e5-small on both
    # the dev and the held-out set, and base/large models did not beat it.
    embedding_model: str = "Snowflake/snowflake-arctic-embed-s"
    embedding_dim: int = 384
    # Prefix conventions come from the model name (app.rag.embed.profile_for); set these only to
    # override them ("" = no prefix). None = use the model's own convention.
    embedding_query_prefix: str | None = None
    embedding_passage_prefix: str | None = None
    app_title: str = "OSRS Agent (dev)"  # sent as X-Title for OpenRouter attribution
    # DeepSeek V4 "thinks" by default and bills those hidden tokens; with a small max_tokens that
    # can leave the visible answer EMPTY (seen live: finish_reason=length, content=""). Direct
    # answers are what this assistant needs. Set to "" to leave the provider default alone.
    reasoning_effort: str = "none"
    llm_max_tokens: int = Field(1_500, ge=64)

    # --- Chunking ---------------------------------------------------------------------------
    # SPEC §9 asks for 300-600 tokens, but the embedding models used here accept 512 tokens at most,
    # so the default maximum leaves room for the model's prefix and the title/section header.
    chunk_target_min_tokens: int = Field(300, ge=50)
    chunk_target_max_tokens: int = Field(480, ge=100)
    chunk_overlap_tokens: int = Field(40, ge=0)
    embedding_batch_size: int = Field(32, ge=1)

    # --- Jev batching ----------------------------------------------------------------------
    # Docs: 32K tokens for `state` + the longest question (64K total). Assume the smaller one
    # and leave headroom; batch sizing in app.rag.grade derives from this.
    jev_max_state_tokens: int = Field(24_000, gt=1_000)
    # Conservative chars-per-token for budgeting without a tokenizer (English ~4, JSON/wikitext ~3).
    jev_chars_per_token: float = Field(3.0, gt=0.5)

    # --- Wiki prices API -------------------------------------------------------------------
    prices_base_url: str = "https://prices.runescape.wiki/api/v2/osrs"
    prices_user_agent: str = "osrs-agent-dev/0.1 (BYOK OSRS assistant; local development)"
    prices_latest_ttl_s: float = Field(45.0, ge=0)  # /latest and /5m: 30-60 s (SPEC §6)
    prices_hourly_ttl_s: float = Field(300.0, ge=0)  # /1h
    prices_timeseries_ttl_s: float = Field(300.0, ge=0)
    prices_mapping_ttl_s: float = Field(6 * 3600.0, ge=0)  # /mapping changes rarely
    # If the API is down, serve a cached snapshot up to this old (the caller sees its timestamp).
    prices_max_stale_s: float = Field(900.0, ge=0)

    # --- Ingestion cache ----------------------------------------------------------------------
    data_dir: Path = (
        BACKEND_DIR / "data"
    )  # gitignored: raw wiki pages as JSONL, one file per category

    # --- Wiki MediaWiki API (ingestion + live search) ------------------------------------------
    wiki_api_url: str = "https://oldschool.runescape.wiki/api.php"
    wiki_page_base_url: str = "https://oldschool.runescape.wiki/w/"
    wiki_user_agent: str | None = None  # defaults to PRICES_USER_AGENT
    wiki_request_delay_s: float = Field(0.3, ge=0)  # politeness between sequential API calls
    wiki_batch_size: int = Field(50, ge=1, le=50)  # MediaWiki caps content queries at 50 titles
    wiki_maxlag_s: int = Field(5, ge=0)  # tell the server to refuse us when replicas lag

    # --- Grand Exchange tax (SPEC §6) --------------------------------------------------------
    # Verified 2026-10-14 (Wiki "Grand Exchange"): 2% since 29 May 2025, rounded down per item,
    # capped at 5,000,000 coins per item, applied on the sell side, exempt list below.
    ge_tax_rate: Decimal = Field(Decimal("0.02"), ge=0, le=1)
    ge_tax_cap: int = Field(5_000_000, ge=0)
    ge_tax_exempt_items: tuple[str, ...] = DEFAULT_GE_TAX_EXEMPT_ITEMS

    # --- Flipping (calc_flip / rank_flips) ---------------------------------------------------
    flip_max_price_age_s: int = Field(3600, ge=0)  # older quotes are flagged / excluded
    flip_max_roi: float = Field(
        0.50, gt=0
    )  # rank_flips excludes implausibly high ROI (stale/illiquid)
    flip_default_min_volume: int = Field(100, ge=0)  # units/hour on BOTH sides of the book
    flip_low_volume_warn: int = Field(50, ge=0)
    # Share of the typical traded volume one flipper can realistically capture with patient offers.
    flip_market_share: float = Field(0.25, gt=0, le=1)

    # --- Money making (rank_moneymakers, SPEC §8) ----------------------------------------------
    # How outputs and inputs are priced. "mid" (default): the average of the high and low prices,
    # which reproduces the wiki's own figures (measured 2026-10-14 over 102 guides: median ratio
    # 1.000, 101 within +-10%; docs/moneymakers.md). "patient": sell at the average instant-buy
    # (high) price and buy at the average instant-sell (low) price, like a flip (SPEC §6), which
    # is optimistic by the spread; "instant": the pessimistic opposite.
    mm_price_basis: Literal["patient", "mid", "instant"] = "mid"
    mm_max_price_age_s: int = Field(3600, ge=0)  # /latest quotes older than this are not used
    # An item that makes up at least this share of a guide's revenue (outputs) or cost (inputs)
    # is checked against the market's hourly volume; a bigger share than `mm_market_share` of
    # the hourly volume is warned about, more than 100% of it excludes the guide.
    mm_liquidity_min_share: float = Field(0.05, ge=0, le=1)
    mm_market_share: float = Field(0.25, gt=0, le=1)
    # "Almost eligible": at most this many skills short by at most this many levels, and at most
    # this many quests left (SPEC §8: "a pocos niveles").
    mm_almost_max_level_gap: int = Field(5, ge=1)
    mm_almost_max_skills: int = Field(2, ge=1)
    mm_almost_max_quests: int = Field(1, ge=0)

    # --- CRAG --------------------------------------------------------------------------------
    # Tuned with the end-to-end eval (docs/eval-m8.md). Relevance keeps the passage with the answer
    # anywhere in 0.3-0.8 (0.9 loses 4 of 21), so the SPEC's 0.70 stays. Sufficiency separates
    # perfectly in 0.5-0.6 and loses a case at 0.7: it moves from the SPEC's 0.50, the edge of that
    # plateau where a false "sufficient" would let an unanswerable question through, to its centre.
    rel_threshold: float = Field(0.70, ge=0, le=1)
    sufficiency_threshold: float = Field(0.60, ge=0, le=1)
    crag_topk: int = Field(20, ge=1)
    max_crag_iterations: int = Field(2, ge=0)
    # Below this Jev confidence the router falls back to wiki_general (SPEC §7). Starting point,
    # tuned with the eval set.
    route_min_confidence: float = Field(0.45, ge=0, le=1)
    # A false "out of domain" rejects a legitimate question; a false "in domain" just yields a
    # "no evidence" answer. So rejecting needs real conviction (live: a Spanish Dragon Slayer II
    # question got out_of_domain=0.35). Below this probability it routes to wiki_general instead.
    out_of_domain_min_prob: float = Field(0.80, ge=0, le=1)

    # --- HTTP API / limits -------------------------------------------------------------------
    host: str = "127.0.0.1"  # v0 only listens on loopback (SPEC §11)
    port: int = 8000
    cors_allow_origins: list[str] = Field(default_factory=list)  # closed by default
    max_message_chars: int = Field(4_000, ge=1)
    max_player_state_bytes: int = Field(256_000, ge=1)
    history_max_messages: int = Field(12, ge=1)
    # Conversation memory (app/graph/retention.py): a thread keeps one checkpoint and at most
    # MAX_STORED_MESSAGES messages; threads nobody used for this many days are deleted (at start-up
    # and every 6 hours). 0 keeps every thread forever.
    thread_ttl_days: int = Field(30, ge=0)
    debug_responses: bool = False
    log_level: str = "INFO"

    # --- Outbound HTTP ---------------------------------------------------------------------
    http_timeout_s: float = Field(20.0, gt=0)
    jev_timeout_s: float = Field(30.0, gt=0)
    llm_timeout_s: float = Field(90.0, gt=0)
    http_max_retries: int = Field(2, ge=0)
    http_retry_base_delay_s: float = Field(0.5, ge=0)
    http_retry_max_delay_s: float = Field(8.0, ge=0)

    @field_validator("prices_user_agent", "wiki_user_agent")
    @classmethod
    def _user_agent_must_be_descriptive(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("PRICES_USER_AGENT must not be empty")
        lowered = value.lower()
        if lowered.startswith(BLOCKED_USER_AGENT_PREFIXES):
            raise ValueError(
                "PRICES_USER_AGENT is a generic client string that the Wiki prices API blocks; "
                "describe your tool and add a contact, e.g. 'osrs-agent-dev - you@example.com'"
            )
        return value

    @property
    def effective_wiki_user_agent(self) -> str:
        return self.wiki_user_agent or self.prices_user_agent


@lru_cache
def get_settings() -> Settings:
    return Settings()


class ScriptSettings(BaseSettings):
    """Credentials for developer scripts ONLY (m1_smoke, ingestion LLM extraction, eval).

    SPEC §11: the backend never uses a developer key of its own; the API server only ever sees
    the user's key through ``X-OpenRouter-Key``. Nothing in ``app/`` except the scripts' entry
    points may instantiate this class. ``Settings`` ignores this variable even if it sits in the
    same ``.env``, so it can never reach a request handler through configuration.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
        populate_by_name=True,
    )

    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPEN_ROUTER_KEY", "OPENROUTER_API_KEY")
    )


def dev_api_key(*, env_files: tuple[Path, ...] = ENV_FILES) -> str | None:
    """The developer's OpenRouter key for scripts: process environment first, then ``.env``."""
    for name in ("OPEN_ROUTER_KEY", "OPENROUTER_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    secret = ScriptSettings(_env_file=env_files).openrouter_api_key
    return secret.get_secret_value() if secret else None
