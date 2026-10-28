"""Adapter for the OpenRouter Decisions API running Jev (TypeSafe's decision model).

ALL knowledge of the wire format lives in this module (SPEC §3) because the endpoint is alpha.

Wire format, verified 2026-10-14 against OpenRouter's API reference
(``POST https://openrouter.ai/api/alpha/decisions``) and TypeSafe's docs:

* Request: ``{"model", "state", "questions": {<id>: {"type", "instructions", "criteria"}}}``.
  ``state`` may be a string, an object or an array. Every question sees the same ``state`` and is
  evaluated independently. Question ids are ours; they are not sent to the model and come back as
  keys of ``answers``. Instructions reference state fields by name in backticks.
* ``noul``   -> criteria ``{"true": ..., "false": ...}`` (OpenRouter marks both required).
  Answer: ``{"type": "noul", "noul": P(yes)}`` (no confidence field).
* ``choice`` -> criteria ``{option: description | null}``, at most 255 options.
  Answer: ``{"choice", "probabilities", "confidence"}``.
* ``score``  -> criteria ``[level descriptions]``, 2..10 levels.
  Answer: ``{"score", "legend", "probabilities", "confidence"}``.
* Response envelope: ``{id, model, provider, usage{input_tokens, output_tokens, cost}, answers}``.
* Errors: ``{"error": {"code", "message", "metadata"}}`` with the usual OpenRouter statuses.
* Context: 32K tokens for ``state`` + the longest question (64K for state + all questions).
  We budget against the smaller figure (``JEV_MAX_STATE_TOKENS``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from app.clients.http import RetryPolicy, parse_retry_after, safe_json, send_with_retries
from app.clients.openrouter import error_from_response, error_status_from_body
from app.config import Settings, get_settings
from app.errors import InvalidKeyError, StateTooLargeError, UpstreamError
from app.security import redact

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10

Instructions = str | dict[str, Any] | list[Any]


# --------------------------------------------------------------------------- questions
class NoulQuestion(BaseModel):
    """Yes/no proposition -> probability of yes."""

    type: Literal["noul"] = "noul"
    instructions: Instructions
    criteria: dict[str, str]

    @field_validator("criteria")
    @classmethod
    def _both_sides(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != {"true", "false"}:
            raise ValueError('noul criteria must have exactly the keys "true" and "false"')
        return value


class ChoiceQuestion(BaseModel):
    """Pick one option -> probability distribution + confidence."""

    type: Literal["choice"] = "choice"
    instructions: Instructions
    criteria: dict[str, str | None]

    @field_validator("criteria")
    @classmethod
    def _option_count(cls, value: dict[str, str | None]) -> dict[str, str | None]:
        if not 2 <= len(value) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice needs 2..{MAX_CHOICE_OPTIONS} options, got {len(value)}")
        return value


class ScoreQuestion(BaseModel):
    """Ordered rubric -> probability-weighted level."""

    type: Literal["score"] = "score"
    instructions: Instructions
    criteria: list[str]

    @field_validator("criteria")
    @classmethod
    def _level_count(cls, value: list[str]) -> list[str]:
        if not MIN_SCORE_LEVELS <= len(value) <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"score needs {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels, got {len(value)}"
            )
        return value


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


def noul(instructions: Instructions, *, yes: str, no: str) -> NoulQuestion:
    return NoulQuestion(instructions=instructions, criteria={"true": yes, "false": no})


def choice(instructions: Instructions, options: Mapping[str, str | None]) -> ChoiceQuestion:
    return ChoiceQuestion(instructions=instructions, criteria=dict(options))


def score(instructions: Instructions, levels: Sequence[str]) -> ScoreQuestion:
    return ScoreQuestion(instructions=instructions, criteria=list(levels))


# --------------------------------------------------------------------------- answers
class NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["noul"]
    noul: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["score"]
    score: float
    legend: dict[str, Any] = Field(default_factory=dict)
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]
A = TypeVar("A", NoulAnswer, ChoiceAnswer, ScoreAnswer)


class JevUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None  # USD; output tokens are free, input is ~$0.042/Mtok


class DecisionsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    model: str = ""  # versioned id that actually answered, e.g. "typesafe/jev-1.13-20260917"
    provider: str | None = None
    usage: JevUsage = Field(default_factory=JevUsage)
    answers: dict[str, Answer]

    def get(self, question_id: str, kind: type[A]) -> A:
        answer = self.answers.get(question_id)
        if not isinstance(answer, kind):
            raise UpstreamError(
                f"Jev answer {question_id!r} missing or not a {kind.__name__} "
                f"(got {type(answer).__name__})"
            )
        return answer


# --------------------------------------------------------------------------- budgeting
def estimate_tokens(obj: Any, chars_per_token: float = 3.0) -> int:
    """Cheap, conservative token estimate (no tokenizer dependency)."""
    text = (
        obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    )
    return math.ceil(len(text) / chars_per_token)


# --------------------------------------------------------------------------- client
class JevClient:
    """``POST /api/alpha/decisions`` bound to one request's OpenRouter key."""

    def __init__(
        self,
        api_key: str | SecretStr,
        *,
        http: httpx.AsyncClient,
        settings: Settings | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        self._key = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        if not self._key.get_secret_value().strip():
            raise InvalidKeyError("Missing OpenRouter API key")
        self._http = http
        self._settings = settings or get_settings()
        self._retry = retry or RetryPolicy.from_settings(self._settings)

    def __repr__(self) -> str:
        return "JevClient(api_key=**********)"

    def estimated_tokens(
        self, state: Any, questions: Mapping[str, Question]
    ) -> tuple[int, int, int]:
        """Return ``(state_tokens, longest_question_tokens, all_questions_tokens)``."""
        cpt = self._settings.jev_chars_per_token
        per_question = [
            estimate_tokens(q.model_dump(exclude_none=True), cpt) for q in questions.values()
        ]
        return (
            estimate_tokens(state, cpt),
            max(per_question, default=0),
            sum(per_question),
        )

    def check_budget(self, state: Any, questions: Mapping[str, Question]) -> None:
        state_tokens, longest, total = self.estimated_tokens(state, questions)
        budget = self._settings.jev_max_state_tokens
        # Documented limits: 32K for state + longest question, 64K for state + all questions.
        if state_tokens + longest > budget or state_tokens + total > 2 * budget:
            raise StateTooLargeError(
                f"Jev request too large (state~{state_tokens}, longest question~{longest}, "
                f"all questions~{total}, budget {budget}); split into smaller batches"
            )

    async def decide(self, state: Any, questions: Mapping[str, Question]) -> DecisionsResponse:
        if not questions:
            raise ValueError("at least one question is required")
        self.check_budget(state, questions)

        body = {
            "model": self._settings.jev_model,
            "state": state,
            "questions": {qid: q.model_dump(exclude_none=True) for qid, q in questions.items()},
        }
        key = self._key.get_secret_value()
        response = await send_with_retries(
            self._http,
            "POST",
            self._settings.jev_endpoint,
            policy=self._retry,
            what="Jev decisions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "X-Title": self._settings.app_title,
            },
            json=body,
            timeout=self._settings.jev_timeout_s,
        )
        payload = safe_json(response)

        if response.status_code != 200:
            raise error_from_response(
                response.status_code,
                payload,
                retry_after=parse_retry_after(response),
                key=key,
                what="Jev decisions",
            )
        in_body = error_status_from_body(payload)
        if in_body is not None and "answers" not in (payload or {}):
            raise error_from_response(in_body, payload, key=key, what="Jev decisions")

        try:
            return DecisionsResponse.model_validate(payload)
        except ValidationError as exc:
            # Keep the message short and free of payload content.
            first = exc.errors()[0] if exc.errors() else {}
            where = ".".join(str(p) for p in first.get("loc", ()))
            raise UpstreamError(
                redact(f"Unexpected Jev response shape at '{where}': {first.get('type', '?')}", key)
            ) from exc

    async def choose(
        self,
        state: Any,
        instructions: Instructions,
        options: Mapping[str, str | None],
    ) -> tuple[ChoiceAnswer, DecisionsResponse]:
        """Single Choice question (the router)."""
        qid = "choice"
        decision = await self.decide(state, {qid: choice(instructions, options)})
        return decision.get(qid, ChoiceAnswer), decision
