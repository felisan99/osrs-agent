"""The graph's state (SPEC §7 ``AgentState``) and its run-scoped context.

Two kinds of data, kept apart on purpose:

* **State** is what LangGraph checkpoints into Postgres after every step, so it holds only what a
  later turn needs (the conversation) and the working data of the current turn. It never holds the
  user's OpenRouter key or the player's snapshot.
* **Context** (``RunContext``) lives for one request: the clients (which hold the user's key), the
  player's snapshot and the shared singletons. LangGraph passes it to nodes without persisting it.

Passages are stored in the state as plain dicts (msgpack-friendly, and what lands in the
checkpoint is readable); ``passage_to_dict`` / ``passage_from_dict`` convert at the node boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, TypedDict

from app.api.schemas import PlayerState
from app.clients.jev import ChoiceAnswer, DecisionsResponse
from app.clients.openrouter import ChatResult
from app.config import Settings
from app.rag.grade import Grader
from app.rag.retrieve import Passage, Retriever
from app.rag.titles import TitleIndex
from app.tools.context import ToolContext
from app.tools.quests import QuestStore

MAX_STORED_MESSAGES = 40  # what a thread remembers; prompts use the last HISTORY_MAX_MESSAGES

Language = Literal["es", "en"]


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


def append_messages(left: list[Message] | None, right: list[Message] | Message) -> list[Message]:
    """Reducer for ``messages``: append, keeping the most recent ``MAX_STORED_MESSAGES``."""
    added = [right] if isinstance(right, dict) else list(right)
    return [*(left or []), *added][-MAX_STORED_MESSAGES:]


class AgentState(TypedDict, total=False):
    # --- remembered across turns (the thread's memory)
    messages: Annotated[list[Message], append_messages]
    # --- working data of the current turn (reset by the first node)
    intent: str
    routed: bool  # True once classify_intent has run this turn
    route_probs: dict[str, float]
    language: str
    query: str  # standalone English version of the current question (what Jev judges against)
    underspecified: bool  # a first message that only says "where do I get it?": ask, do not search
    queries: list[str]  # the searches already run (the rewrite must not repeat them)
    search_types: list[str] | None  # restrict retrieval to these page types
    force_titles: list[str]  # pages retrieval must include (quest being asked about, ...)
    candidates: list[dict[str, Any]]
    graded: list[dict[str, Any]]
    sufficiency: float | None
    sufficient: bool
    retrieval_iterations: int
    tool_results: list[dict[str, Any]]
    tools_called: list[str]
    answer: str
    sources: list[dict[str, str]]
    usage: dict[str, float]


TURN_RESET: dict[str, Any] = {
    "intent": "wiki_general",
    "routed": False,
    "route_probs": {},
    "language": "es",
    "query": "",
    "underspecified": False,
    "queries": [],
    "search_types": None,
    "force_titles": [],
    "candidates": [],
    "graded": [],
    "sufficiency": None,
    "sufficient": False,
    "retrieval_iterations": 0,
    "tool_results": [],
    "tools_called": [],
    "answer": "",
    "sources": [],
    "usage": {},
}


# --------------------------------------------------------------------------- passages <-> state
def passage_to_dict(passage: Passage) -> dict[str, Any]:
    return {
        "id": passage.id,
        "page_title": passage.page_title,
        "section": passage.section,
        "page_type": passage.page_type,
        "url": passage.url,
        "content": passage.content,
        "score": passage.score,
    }


def passage_from_dict(data: dict[str, Any]) -> Passage:
    return Passage(
        id=data["id"],
        page_title=data["page_title"],
        section=data.get("section", ""),
        page_type=data.get("page_type", "other"),
        url=data["url"],
        content=data["content"],
        metadata={},
        score=data.get("score", 0.0),
    )


# --------------------------------------------------------------------------- run context
class ChatLLM(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = ...,
        tools: list[dict[str, Any]] | None = ...,
        tool_choice: str | dict[str, Any] | None = ...,
        temperature: float = ...,
        max_tokens: int | None = ...,
        response_format: dict[str, Any] | None = ...,
        extra: dict[str, Any] | None = ...,
    ) -> ChatResult: ...


class Chooser(Protocol):
    async def choose(
        self, state: Any, instructions: Any, options: Any
    ) -> tuple[ChoiceAnswer, DecisionsResponse]: ...


class LiveSearch(Protocol):
    """``search_live`` (SPEC §7): candidate passages from the live wiki for a query."""

    async def passages(self, query: str, *, limit: int) -> list[Passage]: ...


@dataclass
class Deps:
    """Process-wide singletons: nothing here knows a user's key."""

    settings: Settings
    retriever: Retriever
    tools: ToolContext
    titles: TitleIndex
    quests: QuestStore
    live: LiveSearch | None = None


@dataclass
class RunContext:
    """Everything one request's nodes need. Holds the user's key (inside ``llm`` and ``jev``,
    whose reprs hide it) and the player's snapshot, and is never checkpointed."""

    deps: Deps
    llm: ChatLLM
    jev: Chooser
    grader: Grader
    player: PlayerState | None = None

    @property
    def settings(self) -> Settings:
        return self.deps.settings
