"""The whole graph over a real Postgres (corpus + LangGraph checkpoints) with fake LLM and Jev.

Only the two network services are faked; routing, condensing, hybrid retrieval in pgvector,
grading, the CRAG correction loop, the workflows, the tools and the checkpointed thread memory all
run for real.
"""

from __future__ import annotations

import json

import psycopg

from app.api.schemas import PlayerState
from app.config import Settings
from app.graph.nodes.tools_loop import PRICES_SOURCE
from app.graph.prompts import fixed
from app.tools.context import ToolContext
from app.tools.quests import QuestStore
from tests.conftest import TEST_UA
from tests.graph_helpers import (
    Call,
    FakeJev,
    FakeLive,
    FakeLLM,
    harness,
    text_result,
    tool_result,
)
from tests.test_moneymakers import catalogue, guide, item
from tests.test_moneymakers import player as money_player
from tests.test_moneymakers import tool_ctx as money_tools
from tests.test_quests_tool import MM2
from tests.test_quests_tool import player as quest_player
from tests.test_tools_registry import NOW as PRICES_NOW
from tests.test_tools_registry import FakePrices as RegistryPrices


def price_tools(prices: RegistryPrices | None = None) -> ToolContext:
    settings = Settings(_env_file=None, prices_user_agent=TEST_UA)
    return ToolContext(settings, prices or RegistryPrices(), now=lambda: PRICES_NOW)  # type: ignore[arg-type]


def route_to(intent: str):
    return lambda message: (intent, {intent: 0.95}, 0.9)


def wiki_relevance(query: str, text: str) -> float:
    lowered = text.lower()
    return 0.9 if any(w in lowered for w in ("whip", "monkey", "amulet")) else 0.1


# =========================================================================== wiki path
async def test_a_plain_wiki_question_runs_retrieve_grade_generate(db) -> None:
    llm = FakeLLM(answer=lambda call: "You need 70 Attack [Abyssal whip].")
    async with harness(db, llm=llm) as h:
        out = await h.ask("What Attack level do I need to wield an abyssal whip?")
    assert out["intent"] == "wiki_general" and out["language"] == "en"
    assert out["answer"] == "You need 70 Attack [Abyssal whip]."
    assert [s["title"] for s in out["sources"]] == ["Abyssal whip"]
    assert out["sufficient"] is True and out["retrieval_iterations"] == 1
    assert [m["role"] for m in out["messages"]] == ["user", "assistant"]
    assert llm.of_kind("You prepare") == []  # a clear English first message needs no condensing


async def test_a_spanish_question_is_condensed_to_english_and_answered_in_spanish(db) -> None:
    llm = FakeLLM(
        condense=lambda call: {"query": "abyssal whip Attack level requirement", "language": "es"},
        answer=lambda call: "Necesitás 70 de Attack [Abyssal whip].",
    )
    async with harness(db, llm=llm) as h:
        out = await h.ask("¿Qué nivel de Attack necesito para usar la abyssal whip?")
    assert out["query"] == "abyssal whip Attack level requirement" and out["language"] == "es"
    assert out["answer"] == "Necesitás 70 de Attack [Abyssal whip]."
    generation = llm.calls[-1]
    assert "Answer in Spanish" in generation.system and "Abyssal whip" in generation.system
    assert generation.last_user == "¿Qué nivel de Attack necesito para usar la abyssal whip?"


async def test_a_broken_condense_answer_falls_back_to_the_players_own_words(db) -> None:
    class Broken(FakeLLM):
        async def chat(self, messages, **kwargs):
            if "You prepare a player's chat message" in messages[0]["content"]:
                self.calls.append(Call(messages, kwargs))
                return text_result("sorry, I cannot produce JSON today")
            return await super().chat(messages, **kwargs)

    async with harness(db, llm=Broken()) as h:
        out = await h.ask("¿Qué nivel de Attack necesito para la abyssal whip?")
    assert out["query"] == "¿Qué nivel de Attack necesito para la abyssal whip?"
    assert out["language"] == "es" and out["answer"]


async def test_an_empty_model_answer_is_retried_once_then_gives_up_honestly(db) -> None:
    replies = iter(["", "Second try [Abyssal whip]."])
    llm = FakeLLM(answer=lambda call: next(replies))
    async with harness(db, llm=llm) as h:
        out = await h.ask("What Attack level do I need to wield an abyssal whip?")
    assert out["answer"] == "Second try [Abyssal whip]."
    assert len(llm.calls) == 2

    llm = FakeLLM(answer=lambda call: "")
    async with harness(db, llm=llm) as h:
        out = await h.ask(
            "What Attack level do I need to wield an abyssal whip?", thread="thread-empty-answer-2"
        )
    assert out["answer"] == fixed("no_evidence", "en") and out["sources"] == []


# =========================================================================== routing
async def test_out_of_domain_gets_a_fixed_refusal_without_calling_the_model(db) -> None:
    jev = FakeJev(
        route=lambda m: ("out_of_domain", {"out_of_domain": 0.96, "wiki_general": 0.03}, 0.95)
    )
    llm = FakeLLM()
    async with harness(db, llm=llm, jev=jev) as h:
        es = await h.ask("Escribime una función en Python que ordene una lista", thread="thread-es")
        en = await h.ask("What's the best pizza topping?", thread="thread-en")
    assert es["intent"] == "out_of_domain" and "asistente informativo" in es["answer"]
    assert "informational assistant" in en["answer"]
    assert llm.calls == [] and es["sources"] == [] and jev.grade_calls == []
    assert [m["role"] for m in es["messages"]] == ["user", "assistant"]  # the refusal is remembered


async def test_a_doubtful_out_of_domain_goes_to_the_wiki_instead_of_being_rejected(db) -> None:
    jev = FakeJev(route=lambda m: ("out_of_domain", {"out_of_domain": 0.6, "quest": 0.35}, 0.6))
    async with harness(db, jev=jev) as h:
        out = await h.ask("What Attack level do I need to wield an abyssal whip?")
    assert out["intent"] == "wiki_general" and out["route_probs"]["out_of_domain"] == 0.6
    assert "whip" in out["answer"].lower() or out["answer"]


async def test_a_low_confidence_route_falls_back_to_wiki_general(db) -> None:
    jev = FakeJev(route=lambda m: ("quest", {"quest": 0.5, "wiki_general": 0.4}, 0.4))
    async with harness(db, jev=jev) as h:
        out = await h.ask("What Attack level do I need to wield an abyssal whip?")
    assert out["intent"] == "wiki_general"


# =========================================================================== CRAG correction
async def test_insufficient_evidence_triggers_a_rewrite_and_a_second_search(db) -> None:
    rounds = {"n": 0}

    def sufficiency(query: str, texts: list[str]) -> float:
        rounds["n"] += 1
        return 0.1 if rounds["n"] == 1 else 0.9

    llm = FakeLLM(
        rewrite=lambda call: {"query": "abyssal whip attack level"},
        answer=lambda call: "You need 70 Attack [Abyssal whip].",
    )
    jev = FakeJev(sufficiency=sufficiency)
    async with harness(db, llm=llm, jev=jev) as h:
        out = await h.ask("tell me about that weapon from abyssal demons")
    assert out["retrieval_iterations"] == 2 and out["sufficient"] is True
    assert out["queries"] == [
        "tell me about that weapon from abyssal demons",
        "abyssal whip attack level",
    ]
    (rewrite,) = llm.of_kind("did not find enough information")
    assert "tell me about that weapon from abyssal demons" in rewrite.last_user
    assert out["answer"] == "You need 70 Attack [Abyssal whip]."
    assert len(jev.grade_calls) == 2  # one batch call per grading round


async def test_the_correction_also_searches_the_live_wiki_and_pools_what_it_finds(db) -> None:
    from app.rag.retrieve import Passage

    live_passage = Passage(
        id=-7,
        page_title="Live page",
        section="",
        page_type="other",
        url="https://oldschool.runescape.wiki/w/Live_page",
        content="Live page\nThe whip is also described here, live.",
        metadata={"live": True},
        score=0.5,
    )
    live = FakeLive([live_passage])
    rounds = {"n": 0}

    def sufficiency(query: str, texts: list[str]) -> float:
        rounds["n"] += 1
        return 0.1 if rounds["n"] == 1 else 0.9

    llm = FakeLLM(rewrite=lambda call: {"query": "abyssal whip"})
    async with harness(db, llm=llm, jev=FakeJev(sufficiency=sufficiency), live=live) as h:
        out = await h.ask("tell me about that weapon from abyssal demons")
    assert live.queries == ["abyssal whip"]
    assert "Live page" in {g["page_title"] for g in out["graded"]}
    assert "Live page" in {s["title"] for s in out["sources"]}


async def test_a_dead_live_wiki_never_breaks_the_correction(db) -> None:
    from app.errors import UpstreamError

    class Down:
        async def passages(self, query: str, *, limit: int):
            raise UpstreamError("wiki down")

    rounds = {"n": 0}

    def sufficiency(query: str, texts: list[str]) -> float:
        rounds["n"] += 1
        return 0.1 if rounds["n"] == 1 else 0.9

    llm = FakeLLM(rewrite=lambda call: {"query": "abyssal whip"})
    async with harness(db, llm=llm, jev=FakeJev(sufficiency=sufficiency), live=Down()) as h:  # type: ignore[arg-type]
        out = await h.ask("tell me about that weapon from abyssal demons")
    assert out["sufficient"] is True and out["answer"]


async def test_without_enough_evidence_after_the_corrections_it_says_so_and_invents_nothing(
    db,
) -> None:
    rewrites = iter(["attempt one", "attempt two", "attempt three"])
    llm = FakeLLM(rewrite=lambda call: {"query": next(rewrites)})
    jev = FakeJev(sufficiency=lambda q, t: 0.1)
    async with harness(db, llm=llm, jev=jev) as h:
        out = await h.ask("tell me about that weapon from abyssal demons")
    assert out["sufficient"] is False and out["retrieval_iterations"] == 3  # 1 + 2 corrections
    assert out["answer"] == fixed("no_evidence", "en") and out["sources"] == []
    assert len(llm.of_kind("did not find enough information")) == 2
    assert llm.of_kind("Rules, in order of importance") == []  # generation never ran
    assert out["messages"][-1]["content"] == out["answer"]


async def test_a_rewrite_that_repeats_a_failed_query_still_searches_with_the_question(db) -> None:
    llm = FakeLLM(rewrite=lambda call: {"query": "tell me about that weapon from abyssal demons"})
    async with harness(db, llm=llm, jev=FakeJev(sufficiency=lambda q, t: 0.1)) as h:
        out = await h.ask("tell me about that weapon from abyssal demons")
    assert out["queries"][1] == "tell me about that weapon from abyssal demons"  # no empty search


# =========================================================================== memory
async def test_a_follow_up_is_resolved_with_the_threads_memory(db) -> None:
    def condense(call: Call) -> dict[str, str]:
        assert "How do I start Monkey Madness II?" in call.last_user  # the earlier turn is shown
        return {
            "query": "what if I don't have the m'speak amulet for Monkey Madness II",
            "language": "es",
        }

    llm = FakeLLM(condense=condense, answer=lambda call: "Respuesta [M'speak amulet].")
    jev = FakeJev(route=route_to("quest"), relevance=wiki_relevance)
    async with harness(db, llm=llm, jev=jev, quests=QuestStore([MM2])) as h:
        first = await h.ask("How do I start Monkey Madness II?", player=quest_player())
        second = await h.ask("¿y si no tengo el m'speak amulet?", player=quest_player())
        generation = [c for c in llm.calls if "Rules, in order of importance" in c.system][-1]
        other = await h.ask(
            "What Attack level do I need to wield an abyssal whip?", thread="x-0002"
        )
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user", "assistant"]
    assert second["messages"][0]["content"] == "How do I start Monkey Madness II?"
    assert first["messages"][1]["content"] == second["messages"][1]["content"]
    # the router saw the first turn, a different thread did not
    assert "Monkey Madness II" in jev.route_states[1]["conversation_summary"]
    assert jev.route_states[0]["conversation_summary"] == "(none: first message)"
    assert jev.route_states[2]["conversation_summary"] == "(none: first message)"
    assert len(other["messages"]) == 2
    # the follow-up retrieved the item page the first answer never needed
    titles = {g["page_title"] for g in second["graded"] if g["kept"]}
    assert "M'speak amulet" in titles and "Monkey Madness II" in titles
    # and generation saw the conversation so far
    assert [m["role"] for m in generation.messages] == ["system", "user", "assistant", "user"]


async def test_a_first_message_that_is_only_a_pronoun_question_asks_instead_of_searching(
    db,
) -> None:
    llm = FakeLLM(answer=lambda call: "must never be generated")
    jev = FakeJev()
    async with harness(db, llm=llm, jev=jev) as h:
        out = await h.ask("Where do I get it?")
    assert out["answer"] == fixed("no_evidence", "en")
    assert out["retrieval_iterations"] == 0 and out["candidates"] == []
    assert llm.calls == []  # nothing was searched, graded or generated


async def test_the_same_words_after_a_question_about_something_are_a_normal_follow_up(db) -> None:
    llm = FakeLLM(
        condense=lambda call: {"query": "where do I get the abyssal whip", "language": "en"},
        answer=lambda call: "From abyssal demons [Abyssal whip].",
    )
    async with harness(db, llm=llm, jev=FakeJev(relevance=wiki_relevance)) as h:
        await h.ask("What Attack level do I need to wield an abyssal whip?")
        out = await h.ask("And where do I get it?")
    assert out["retrieval_iterations"] >= 1 and out["answer"] != fixed("no_evidence", "en")


async def test_a_follow_up_is_condensed_before_it_is_routed_and_the_router_sees_what_it_means(
    db,
) -> None:
    llm = FakeLLM(
        condense=lambda call: {
            "query": "what is the buy limit of the abyssal whip",
            "language": "es",
        },
        tool_script=lambda call, n: (
            tool_result(("get_item_price", {"item": "abyssal whip"}))
            if n == 0
            else text_result("El límite es 70.")
        ),
    )
    jev = FakeJev(route=route_to("price_lookup"))
    async with harness(db, llm=llm, jev=jev, tools=price_tools()) as h:
        await h.ask("¿Cuánto cuesta la abyssal whip?")  # Spanish: condensed to translate it
        before = len(llm.of_kind("You prepare"))
        out = await h.ask("¿y cuál es su límite de compra?")
    first, second = jev.route_states
    assert first["standalone_question"] == "(same as user_message)"  # a first message: routed raw
    assert second["standalone_question"] == "what is the buy limit of the abyssal whip"
    assert "abyssal whip" in second["conversation_summary"]
    assert out["intent"] == "price_lookup" and out["answer"].startswith("El límite es 70.")
    assert len(llm.of_kind("You prepare")) - before == 1  # once, before routing: not twice


async def test_a_first_message_is_routed_before_condensing_so_refusals_stay_free(db) -> None:
    llm = FakeLLM(condense=lambda call: {"query": "x", "language": "es"})
    jev = FakeJev(route=lambda m: ("out_of_domain", {"out_of_domain": 0.97}, 0.95))
    async with harness(db, llm=llm, jev=jev) as h:
        out = await h.ask("¿Cómo derroto a Malenia en Elden Ring?")
    assert out["intent"] == "out_of_domain" and llm.calls == []


async def test_an_out_of_domain_follow_up_still_gets_the_fixed_refusal(db) -> None:
    llm = FakeLLM(condense=lambda call: {"query": "write me a poem", "language": "en"})
    jev = FakeJev(
        route=lambda m: (
            ("out_of_domain", {"out_of_domain": 0.97}, 0.95)
            if "poem" in m
            else ("wiki_general", {"wiki_general": 0.95}, 0.9)
        )
    )
    async with harness(db, llm=llm, jev=jev) as h:
        await h.ask("What Attack level do I need to wield an abyssal whip?")
        out = await h.ask("now write me a poem about it")
    assert out["intent"] == "out_of_domain" and "informational assistant" in out["answer"]
    assert len(llm.of_kind("You prepare")) == 1  # the only model call of that turn


async def test_the_working_data_of_a_turn_does_not_leak_into_the_next_one(db) -> None:
    llm = FakeLLM(condense=lambda call: {"query": "abyssal whip attack level", "language": "es"})
    jev = FakeJev(
        route=lambda m: ("quest" if "monkey" in m.lower() else "wiki_general", {"x": 1.0}, 0.9)
    )
    jev.route = lambda m: (
        ("quest", {"quest": 0.95}, 0.9)
        if "monkey" in m.lower()
        else ("wiki_general", {"wiki_general": 0.95}, 0.9)
    )
    async with harness(db, llm=llm, jev=jev, quests=QuestStore([MM2])) as h:
        first = await h.ask("How do I start Monkey Madness II?", player=quest_player())
        second = await h.ask("¿y la abyssal whip?", player=quest_player())
    assert first["tool_results"] and first["force_titles"]
    assert second["intent"] == "wiki_general"
    assert second["tool_results"] == [] and second["force_titles"] == []
    assert second["retrieval_iterations"] == 1


# =========================================================================== quest workflow
async def test_the_quest_workflow_hands_the_deterministic_check_to_the_answer(db) -> None:
    seen: dict[str, str] = {}

    def answer(call: Call) -> str:
        seen["system"] = call.system
        return "Te falta Slayer [Monkey Madness II]."

    jev = FakeJev(route=route_to("quest"), relevance=wiki_relevance)
    llm = FakeLLM(answer=answer)
    async with harness(db, llm=llm, jev=jev, quests=QuestStore([MM2])) as h:
        out = await h.ask("What do I need to start Monkey Madness II?", player=quest_player())
    (entry,) = out["tool_results"]
    assert entry["tool"] == "check_quest_requirements"
    assert entry["result"]["skills"]["missing"][0]["skill"] == "Slayer"
    assert entry["result"]["suggested_order"] == [
        "Monkey Madness I",
        "Recipe for Disaster",
        "Monkey Madness II",
    ]
    assert out["tools_called"] == ["check_quest_requirements"]
    assert "Monkey Madness II" in out["force_titles"]
    assert "check_quest_requirements" in seen["system"] and "levels_short" in seen["system"]
    assert "slayer 66" in seen["system"]  # the player's summary is in the prompt too


async def test_without_a_snapshot_the_quest_answer_is_told_nothing_is_known_about_the_player(
    db,
) -> None:
    seen: dict[str, str] = {}

    def answer(call: Call) -> str:
        seen["system"] = call.system
        return "ok"

    jev = FakeJev(route=route_to("quest"), relevance=wiki_relevance)
    async with harness(db, llm=FakeLLM(answer=answer), jev=jev, quests=QuestStore([MM2])) as h:
        out = await h.ask("What do I need to start Monkey Madness II?")
    assert out["tool_results"] == [] and out["force_titles"]  # the pages are still forced
    assert "UNKNOWN: the plugin sent no snapshot" in seen["system"]


async def test_a_quest_question_that_names_no_known_quest_is_plain_crag(db) -> None:
    jev = FakeJev(route=route_to("quest"), relevance=wiki_relevance)
    async with harness(db, jev=jev, quests=QuestStore([MM2])) as h:
        out = await h.ask("Which quest gives the most Slayer experience?", player=quest_player())
    assert out["intent"] == "quest" and out["tool_results"] == [] and out["force_titles"] == []


# =========================================================================== moneymaking workflow
async def test_the_moneymaking_workflow_ranks_with_the_tool_and_stamps_the_prices(db) -> None:
    seen: dict[str, str] = {}

    def answer(call: Call) -> str:
        seen["system"] = call.system
        return "Looting pays 600,000 gp/h [Money making guide/Looting]."

    jev = FakeJev(
        route=route_to("moneymaking"), relevance=lambda q, t: 0.1
    )  # no guide page indexed
    async with harness(
        db, llm=FakeLLM(answer=answer), jev=jev, tools=money_tools(catalogue())
    ) as h:
        out = await h.ask("What's a good way to make money with my stats?", player=money_player())
    (entry,) = out["tool_results"]
    assert entry["tool"] == "rank_moneymakers" and entry["result"]["status"] == "ok"
    assert [r["title"] for r in entry["result"]["ranking"]] == ["Looting", "Yew cutting", "Planks"]
    assert out["tools_called"] == ["rank_moneymakers"]
    assert out["force_titles"] == ["Money making guide/Looting", "Money making guide/Yew cutting"]
    assert "rank_moneymakers" in seen["system"] and "profit_per_hour" in seen["system"]
    assert out["answer"].endswith(
        "_Prices from the OSRS Wiki prices API, fetched 2026-10-14 11:59 UTC._"
    )
    # the recommended guides are the cited pages (all three of the ranking, best first)
    assert [s["title"] for s in out["sources"]] == [
        "Money making guide/Looting",
        "Money making guide/Yew cutting",
        "Money making guide/Planks",
    ]


async def test_moneymaking_figures_the_model_makes_up_are_rewritten_then_dropped(db) -> None:
    def answer(call: Call) -> str:
        if "not in the tool results" in call.last_user:
            return "Looting pays 600,000 gp/h."
        return "Looting pays 600,000 gp/h. Over a week that is 99,999,999 gp."

    jev = FakeJev(route=route_to("moneymaking"))
    llm = FakeLLM(answer=answer)
    async with harness(db, llm=llm, jev=jev, tools=money_tools(catalogue())) as h:
        out = await h.ask("What's a good way to make money with my stats?", player=money_player())
    assert "99,999,999" not in out["answer"] and "600,000 gp/h" in out["answer"]
    assert len([c for c in llm.calls if "Rules, in order of importance" in c.system]) == 2

    stubborn = FakeLLM(answer=lambda call: "Looting pays 600,000 gp/h. A week is 99,999,999 gp.")
    async with harness(db, llm=stubborn, jev=jev, tools=money_tools(catalogue())) as h:
        out = await h.ask(
            "What's a good way to make money with my stats?",
            thread="thread-stubborn-moneymaking",
            player=money_player(),
        )
    assert "99,999,999" not in out["answer"] and out["answer"].startswith(
        "Looting pays 600,000 gp/h."
    )


async def test_an_ironman_gets_no_gp_per_hour_and_the_ironman_pages(db) -> None:
    jev = FakeJev(route=route_to("moneymaking"))
    player = money_player(account_type="ironman")
    async with harness(db, jev=jev, tools=money_tools(catalogue())) as h:
        out = await h.ask("What's a good way to make money with my stats?", player=player)
    result = out["tool_results"][0]["result"]
    assert result["status"] == "not_applicable_ironman" and "ranking" not in result
    assert "Ironman money making guide" in out["force_titles"]
    assert "Prices from" not in out["answer"]  # no prices were used


async def test_a_ranking_is_reported_even_if_no_guide_page_is_found_and_costs_no_corrections(
    db,
) -> None:
    llm = FakeLLM(answer=lambda call: "Looting pays 600,000 gp/h [Money making guide/Looting].")
    jev = FakeJev(route=route_to("moneymaking"), relevance=lambda q, t: 0.1)
    async with harness(db, llm=llm, jev=jev, tools=money_tools(catalogue())) as h:
        out = await h.ask("What's a good way to make money with my stats?", player=money_player())
    assert out["sufficient"] is False and out["retrieval_iterations"] == 1  # no rewrite spent
    assert llm.of_kind("did not find enough information") == []
    assert out["answer"].startswith("Looting pays 600,000 gp/h")


async def test_a_quest_check_is_reported_even_when_the_walkthrough_is_not_found(db) -> None:
    rewrites = iter(["first rewrite", "second rewrite"])
    llm = FakeLLM(
        rewrite=lambda call: {"query": next(rewrites)},
        answer=lambda call: "Te falta Slayer; no encontré el paso a paso.",
    )
    jev = FakeJev(route=route_to("quest"), relevance=lambda q, t: 0.1)
    async with harness(db, llm=llm, jev=jev, quests=QuestStore([MM2])) as h:
        out = await h.ask("What do I need to start Monkey Madness II?", player=quest_player())
    assert out["retrieval_iterations"] == 3 and out["sufficient"] is False  # corrections tried
    assert out["answer"] == "Te falta Slayer; no encontré el paso a paso."  # not "no evidence"
    assert out["tool_results"][0]["tool"] == "check_quest_requirements"


async def test_a_failed_tool_result_does_not_count_as_something_to_answer_with(db) -> None:
    rewrites = iter(["first", "second"])
    llm = FakeLLM(rewrite=lambda call: {"query": next(rewrites)})
    jev = FakeJev(route=route_to("moneymaking"), relevance=lambda q, t: 0.1)
    async with harness(db, llm=llm, jev=jev, tools=money_tools(None)) as h:  # no guides loaded
        out = await h.ask("What's a good way to make money with my stats?", player=money_player())
    assert out["tool_results"][0]["result"]["error"] == "guides_unavailable"
    assert out["answer"] == fixed("no_evidence", "en")


async def test_an_ironman_answer_never_states_a_gp_per_hour_even_from_a_wiki_page(db) -> None:
    def answer(call: Call) -> str:
        if "GP-per-hour figures" in call.last_user:
            return "Blackjacking needs 45 Thieving. Wintertodt needs 50 Woodcutting."
        return (
            "Blackjacking pays 127,000 gp/hr. Wintertodt gives 30,000 gp/hr and needs 50 "
            "Woodcutting."
        )

    jev = FakeJev(route=route_to("moneymaking"))
    llm = FakeLLM(answer=answer)
    player = money_player(account_type="ironman")
    async with harness(db, llm=llm, jev=jev, tools=money_tools(catalogue())) as h:
        out = await h.ask("What's a good way to make money with my stats?", player=player)
    assert "gp/hr" not in out["answer"] and "Blackjacking needs 45 Thieving" in out["answer"]
    assert len([c for c in llm.calls if "Rules, in order of importance" in c.system]) == 2

    stubborn = FakeLLM(
        answer=lambda call: "Needs 45 Thieving. Pays 127,000 gp/hr. Needs 50 Woodcutting."
    )
    async with harness(db, llm=stubborn, jev=jev, tools=money_tools(catalogue())) as h:
        out = await h.ask(
            "What's a good way to make money?", thread="thread-ironman-2", player=player
        )
    assert "gp/hr" not in out["answer"] and "Needs 45 Thieving." in out["answer"]


async def test_without_a_snapshot_the_ranking_lists_what_each_guide_needs(db) -> None:
    seen: dict[str, str] = {}

    def answer(call: Call) -> str:
        seen["system"] = call.system
        return "Looting pays 600,000 gp/h [Money making guide/Looting]."

    jev = FakeJev(route=route_to("moneymaking"), relevance=lambda q, t: 0.1)
    async with harness(
        db, llm=FakeLLM(answer=answer), jev=jev, tools=money_tools(catalogue())
    ) as h:
        out = await h.ask("What's a good way to make money with my stats?")
    (entry,) = out["tool_results"]
    result = entry["result"]
    assert result["status"] == "ok" and result["requirements_checked"] is False
    assert out["tools_called"] == ["rank_moneymakers"]
    assert [r["title"] for r in result["ranking"]] == ["Looting", "Yew cutting", "Planks"]
    assert all("caveats" not in r or not r["caveats"] for r in result["ranking"])
    assert "requirements_checked" in seen["system"]
    assert any("No character data" in note for note in result["notes"])


async def test_without_a_snapshot_the_players_words_narrow_the_ranking(db) -> None:
    jev = FakeJev(route=route_to("moneymaking"), relevance=lambda q, t: 0.1)
    guides = [
        *catalogue(),
        guide(
            "Boss",
            [item("Coins", 900_000, 995, fixed_value=1)],
            category="Combat/High",
            combat_level=100,
        ),
    ]
    async with harness(db, jev=jev, tools=money_tools(guides)) as h:
        out = await h.ask("Best combat money maker at level 80 combat?")
    (entry,) = out["tool_results"]
    ranking = entry["result"]["ranking"]
    assert [r["title"] for r in ranking] == ["Looting"]  # Combat only; the level-100 boss is out
    assert "requirements" in ranking[0]


async def test_a_snapshot_is_never_overridden_by_what_the_player_says(db) -> None:
    jev = FakeJev(route=route_to("moneymaking"), relevance=lambda q, t: 0.1)
    async with harness(db, jev=jev, tools=money_tools(catalogue())) as h:
        out = await h.ask("Best combat money maker at level 80 combat?", player=money_player())
    result = out["tool_results"][0]["result"]
    assert result["requirements_checked"] is True
    assert all("requirements" not in r for r in result["ranking"])


# =========================================================================== small talk
async def test_a_greeting_gets_a_friendly_fixed_answer_and_costs_nothing(db) -> None:
    llm, jev = FakeLLM(), FakeJev()
    async with harness(db, llm=llm, jev=jev) as h:
        en = await h.ask("Hello!", thread="thread-hi-en")
        es = await h.ask("Hola", thread="thread-hi-es")
    assert en["intent"] == "smalltalk" and "Old School RuneScape assistant" in en["answer"]
    assert "asistente de Old School RuneScape" in es["answer"]
    assert "outside what I can do" not in en["answer"]
    assert llm.calls == [] and jev.grade_calls == [] and en["sources"] == []


async def test_thanks_after_an_answer_is_acknowledged_without_searching(db) -> None:
    jev = FakeJev()
    async with harness(db, jev=jev) as h:
        await h.ask("What Attack level do I need to wield an abyssal whip?", thread="thread-ty")
        calls_before = len(jev.grade_calls)
        out = await h.ask("gracias!", thread="thread-ty")
    assert out["answer"] == fixed("thanks", "es") and len(jev.grade_calls) == calls_before


async def test_a_greeting_with_a_question_is_a_normal_question(db) -> None:
    async with harness(db) as h:
        out = await h.ask("Hi! What Attack level do I need to wield an abyssal whip?")
    assert out["intent"] != "smalltalk"


# =========================================================================== flipping and prices
def whip_script(final: str):
    def script(call: Call, round_number: int):
        if round_number == 0:
            return tool_result(("calc_flip", {"item": "abyssal whip"}))
        return text_result(final)

    return script


async def test_flipping_runs_a_tool_loop_and_the_answer_carries_the_price_time(db) -> None:
    llm = FakeLLM(
        tool_script=whip_script("Buy at 775,464 and sell at 791,060: no profit after tax.")
    )
    jev = FakeJev(route=route_to("flipping"))
    async with harness(db, llm=llm, jev=jev, tools=price_tools()) as h:
        out = await h.ask("Is it worth flipping the abyssal whip?")
    assert out["tools_called"] == ["calc_flip"] and out["sources"] == [PRICES_SOURCE]
    assert out["answer"].startswith("Buy at 775,464 and sell at 791,060")
    assert out["answer"].endswith("fetched 2026-10-14 11:59 UTC._")
    tool_message = llm.calls[1].messages[-1]
    assert tool_message["role"] == "tool" and "tax_per_unit" in tool_message["content"]
    assert llm.calls[0].kwargs["tools"][0]["function"]["name"] in {"calc_flip", "rank_flips"}


async def test_flipping_figures_the_model_computed_itself_are_sent_back_for_a_rewrite(db) -> None:
    llm = FakeLLM(
        tool_script=whip_script("You would make 9,999,999 gp profit on 70 whips."),
        answer=lambda call: (
            "Buy at 775,464, sell at 791,060."
            if "not in the tool results" in call.last_user
            else "unexpected"
        ),
    )
    jev = FakeJev(route=route_to("flipping"))
    async with harness(db, llm=llm, jev=jev, tools=price_tools()) as h:
        out = await h.ask("Is it worth flipping the abyssal whip?")
    assert "9,999,999" not in out["answer"] and out["answer"].startswith("Buy at 775,464")
    retry = llm.calls[-1].messages[-1]["content"]
    assert "9,999,999" in retry and "tool results" in retry


async def test_flipping_figures_that_survive_the_rewrite_are_dropped(db) -> None:
    llm = FakeLLM(
        tool_script=whip_script("Buy at 775,464. You would make 9,999,999 gp profit."),
        answer=lambda call: "Buy at 775,464. You would make 9,999,999 gp profit.",
    )
    jev = FakeJev(route=route_to("flipping"))
    async with harness(db, llm=llm, jev=jev, tools=price_tools()) as h:
        out = await h.ask("Is it worth flipping the abyssal whip?")
    assert "9,999,999" not in out["answer"] and "Buy at 775,464." in out["answer"]


async def test_the_model_cannot_answer_prices_from_memory_without_calling_a_tool(db) -> None:
    def script(call: Call, round_number: int):
        if round_number == 0:
            return text_result("The whip costs about 1,200,000 gp.")
        if round_number == 1:
            return tool_result(("get_item_price", {"item": "abyssal whip"}))
        return text_result("Instant buy 791,060, instant sell 775,464.")

    llm = FakeLLM(tool_script=script)
    jev = FakeJev(route=route_to("price_lookup"))
    async with harness(db, llm=llm, jev=jev, tools=price_tools()) as h:
        out = await h.ask("How much is an abyssal whip?")
    assert llm.tool_rounds == 3
    assert llm.calls[1].messages[-1]["content"].startswith("Call the tools")
    assert out["tools_called"] == ["get_item_price"] and "1,200,000" not in out["answer"]
    assert out["intent"] == "price_lookup" and "Instant buy 791,060" in out["answer"]


async def test_an_ironman_is_told_flipping_does_not_apply_without_any_model_call(db) -> None:
    llm = FakeLLM()
    jev = FakeJev(route=route_to("flipping"))
    player = PlayerState(account_type="hardcore_ironman")  # type: ignore[arg-type]
    async with harness(db, llm=llm, jev=jev, tools=price_tools()) as h:
        out = await h.ask("Is it worth flipping the abyssal whip?", player=player)
    assert out["answer"] == fixed("flip_ironman", "en") and llm.calls == []


async def test_when_the_price_api_is_down_the_answer_has_no_numbers_at_all(db) -> None:
    prices = RegistryPrices()
    prices.fail = True
    llm = FakeLLM(tool_script=whip_script("It costs 1,000,000 gp."))
    jev = FakeJev(route=route_to("flipping"))
    async with harness(db, llm=llm, jev=jev, tools=price_tools(prices)) as h:
        out = await h.ask("Is it worth flipping the abyssal whip?")
    assert out["answer"] == fixed("tool_failed", "en") and out["sources"] == []


async def test_a_model_that_calls_an_unknown_tool_gets_an_error_not_a_crash(db) -> None:
    def script(call: Call, round_number: int):
        if round_number == 0:
            return tool_result(("rank_moneymakers", {}))  # not a flipping tool
        return text_result("I could not do that.")

    llm = FakeLLM(tool_script=script)
    jev = FakeJev(route=route_to("flipping"))
    async with harness(db, llm=llm, jev=jev, tools=price_tools()) as h:
        out = await h.ask("Is it worth flipping the abyssal whip?")
    assert out["tool_results"][0]["result"]["error"] == "unknown_tool"
    assert out["answer"] == "I could not do that."


# =========================================================================== what is persisted
LEAK_KEY = "sk-or-v1-LEAKTESTKEY0123456789abcdef"
MARKER_QUEST = "ZZ-LEAK-MARKER-QUEST"
USER_MARKER = "ZZ-USER-MESSAGE-MARKER"


def checkpoint_dump(db_url: str) -> bytes:
    """Every cell of every LangGraph table, as bytes."""
    out = bytearray()
    with psycopg.connect(db_url) as conn:
        for table in (
            "checkpoints",
            "checkpoint_blobs",
            "checkpoint_writes",
            "checkpoint_migrations",
        ):
            for row in conn.execute(f"select * from {table}").fetchall():
                for cell in row:
                    if isinstance(cell, bytes | memoryview):
                        out += bytes(cell)
                    elif isinstance(cell, dict | list):
                        out += json.dumps(cell, ensure_ascii=False).encode()
                    else:
                        out += str(cell).encode()
                    out += b"\n"
    return bytes(out)


async def test_neither_the_key_nor_the_players_snapshot_ever_reaches_the_checkpoints(db) -> None:
    llm = FakeLLM(
        condense=lambda call: {"query": "Monkey Madness II second part", "language": "en"},
        answer=lambda call: "ok [Monkey Madness II]",
    )
    llm.secret = LEAK_KEY  # type: ignore[attr-defined]  # stands for the client that holds the key
    jev = FakeJev(route=route_to("quest"), relevance=wiki_relevance)
    player = quest_player(
        gp=7_654_321, quests={"Monkey Madness I": "FINISHED", MARKER_QUEST: "FINISHED"}
    )
    async with harness(db, llm=llm, jev=jev, quests=QuestStore([MM2])) as h:
        await h.ask(f"{USER_MARKER} How do I start Monkey Madness II?", player=player)
        await h.ask("And what about the second part?", player=player)
    dump = checkpoint_dump(db)
    assert USER_MARKER.encode() in dump  # the scan does see what is stored: the conversation
    assert LEAK_KEY.encode() not in dump and b"LEAKTESTKEY" not in dump
    assert MARKER_QUEST.encode() not in dump  # the snapshot itself is not persisted
    assert b"7654321" not in dump and b"7,654,321" not in dump
