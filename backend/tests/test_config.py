from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.config import (
    DEFAULT_GE_TAX_EXEMPT_ITEMS,
    ENV_FILES,
    ScriptSettings,
    Settings,
    dev_api_key,
)


def make(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


@pytest.mark.parametrize(
    "ua",
    [
        "python-requests/2.32.3",
        "curl/8.7.1",
        "Java/21.0.1",
        "python-httpx/0.28.1",
        "Python-urllib/3.12",
    ],
)
def test_generic_user_agents_are_rejected(ua: str) -> None:
    with pytest.raises(ValidationError, match="generic client string"):
        make(prices_user_agent=ua)


def test_descriptive_user_agent_is_accepted_and_stripped() -> None:
    assert make(prices_user_agent="  osrs-agent-dev - me@example.com ").prices_user_agent == (
        "osrs-agent-dev - me@example.com"
    )


def test_default_user_agent_is_descriptive_not_generic() -> None:
    ua = make().prices_user_agent
    assert "osrs-agent" in ua


def test_verified_ge_tax_defaults() -> None:
    s = make()
    assert s.ge_tax_rate == Decimal("0.02")
    assert s.ge_tax_cap == 5_000_000
    assert "Old school bond" in s.ge_tax_exempt_items
    assert s.ge_tax_exempt_items == DEFAULT_GE_TAX_EXEMPT_ITEMS


def test_empty_env_value_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # `.env.example` ships `GE_TAX_CAP=` style empties in the spec; they must mean "default".
    monkeypatch.setenv("GE_TAX_CAP", "")
    assert make().ge_tax_cap == 5_000_000


def test_env_overrides_including_exempt_list_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GE_TAX_RATE", "0.03")
    monkeypatch.setenv("GE_TAX_EXEMPT_ITEMS", '["Hammer", "Spade"]')
    monkeypatch.setenv("REL_THRESHOLD", "0.8")
    s = make()
    assert s.ge_tax_rate == Decimal("0.03")
    assert s.ge_tax_exempt_items == ("Hammer", "Spade")
    assert s.rel_threshold == 0.8


@pytest.mark.parametrize("name", ["rel_threshold", "sufficiency_threshold", "route_min_confidence"])
def test_thresholds_are_probabilities(name: str) -> None:
    with pytest.raises(ValidationError):
        make(**{name: 1.5})


def test_spec_defaults() -> None:
    s = make()
    # The SPEC starts at 0.70 / 0.50; the eval (docs/eval-m8.md) moved sufficiency to 0.60 and the
    # router's confidence floor (no starting value in the SPEC) to 0.45.
    assert (s.rel_threshold, s.sufficiency_threshold, s.route_min_confidence) == (0.70, 0.60, 0.45)
    assert (s.crag_topk, s.max_crag_iterations) == (20, 2)
    assert s.out_of_domain_min_prob == 0.80 and s.reasoning_effort == "none"
    assert s.debug_responses is False
    assert s.host == "127.0.0.1"  # v0 only listens on loopback
    assert s.cors_allow_origins == []  # CORS closed


# ------------------------------------------------------------------ developer key (scripts only)
FAKE = "sk-or-v1-" + "ab" * 32


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in ("OPEN_ROUTER_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def write_env(directory, text: str):
    path = directory / ".env"
    path.write_text(text)
    return path


@pytest.mark.parametrize("name", ["OPEN_ROUTER_KEY", "OPENROUTER_API_KEY"])
def test_dev_key_is_read_under_either_name(clean_env, tmp_path, name: str) -> None:
    env = write_env(tmp_path, f"{name}={FAKE}\n")
    assert dev_api_key(env_files=(env,)) == FAKE
    assert ScriptSettings(_env_file=env).openrouter_api_key is not None


def test_dev_key_tolerates_quotes_spaces_and_export(clean_env, tmp_path) -> None:
    assert dev_api_key(env_files=(write_env(tmp_path, f'OPEN_ROUTER_KEY = "{FAKE}"\n'),)) == FAKE


def test_process_environment_wins_over_the_file(clean_env, tmp_path) -> None:
    env = write_env(tmp_path, f"OPEN_ROUTER_KEY={FAKE}\n")
    clean_env.setenv("OPEN_ROUTER_KEY", "sk-or-v1-from-environment")
    assert dev_api_key(env_files=(env,)) == "sk-or-v1-from-environment"


def test_later_env_file_overrides_earlier_and_missing_files_are_fine(clean_env, tmp_path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    one = write_env(first, "OPEN_ROUTER_KEY=sk-or-v1-first\n")
    two = write_env(second, f"OPEN_ROUTER_KEY={FAKE}\n")
    assert dev_api_key(env_files=(one, two)) == FAKE
    assert dev_api_key(env_files=(tmp_path / "missing" / ".env", two)) == FAKE


def test_no_key_anywhere_is_none(clean_env, tmp_path) -> None:
    assert dev_api_key(env_files=(write_env(tmp_path, "OTHER=1\nOPEN_ROUTER_KEY=\n"),)) is None
    assert dev_api_key(env_files=(tmp_path / "nope",)) is None


def test_the_key_never_reaches_server_settings_or_reprs(clean_env, tmp_path) -> None:
    env = write_env(tmp_path, f"OPEN_ROUTER_KEY={FAKE}\nGE_TAX_CAP=123\n")
    server = Settings(_env_file=env)
    assert server.ge_tax_cap == 123  # the same .env is read for real settings...
    assert FAKE not in repr(server) and FAKE not in str(server.model_dump())  # ...but not the key
    assert not any("key" in field and "openrouter" in field for field in Settings.model_fields)
    script = ScriptSettings(_env_file=env)
    assert FAKE not in repr(script) and FAKE not in str(script.model_dump())  # SecretStr masks it


def test_env_files_are_absolute_and_include_the_repo_root() -> None:
    assert all(path.is_absolute() for path in ENV_FILES)
    assert (
        ENV_FILES[0].parent.name == "backend" and ENV_FILES[1].parent == ENV_FILES[0].parent.parent
    )
