"""Turn one cached wiki page into everything we store for it (no I/O, no embedding)."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from app.ingest.chunk import Chunk, ChunkConfig, QuestMatcher, TokenCounter, chunk_page
from app.ingest.chunk import estimate_tokens as default_count
from app.ingest.drops import DropRow, sources_text
from app.ingest.dump import CachedPage
from app.ingest.mmg import MoneyMakingGuide, attach_reference_profits, resolve_item_ids
from app.ingest.parse import Section, parse_page
from app.ingest.quests import QuestRecord, parse_quest
from app.ingest.render import Renderer
from app.tools.items import ItemIndex


@dataclass
class PipelineContext:
    chunk_config: ChunkConfig
    count: TokenCounter = default_count
    renderer: Renderer = field(default_factory=Renderer)
    quest_matcher: QuestMatcher | None = None
    quest_titles: Collection[str] = ()
    item_index: ItemIndex | None = None
    reference_profits: Mapping[str, int] = field(default_factory=dict)
    drop_index: Mapping[str, list[DropRow]] = field(
        default_factory=dict
    )  # item (lower case) -> rows


@dataclass(frozen=True)
class PageResult:
    title: str
    revision_id: int
    page_type: str
    chunks: list[Chunk]
    guide: MoneyMakingGuide | None = None
    quest: QuestRecord | None = None


def keep_only_real_quests(guide: MoneyMakingGuide, quest_titles: Collection[str]) -> None:
    """A guide's Quest field also links strategy pages ("Nightmare Zone/Strategies"); only real
    quest titles are requirements the player's quest list can be checked against."""
    real: list[str] = []
    for name in guide.quest_requirements:
        if name in quest_titles:
            real.append(name)
        else:
            guide.issues.append(f"'{name}' is linked under Quest but is not a quest page")
    guide.quest_requirements = real


def process_page(page: CachedPage, ctx: PipelineContext) -> PageResult | None:
    """Parse, chunk and structure one page; ``None`` when nothing useful remains."""
    parsed = parse_page(page.to_wiki_page(), page.page_type, renderer=ctx.renderer)
    if parsed is None:
        return None
    sources = ctx.drop_index.get(page.title.lower()) if page.page_type == "item" else None
    if sources:  # the wiki generates this table at view time, so it is not in the page source
        parsed.sections.append(Section(("Drop sources",), sources_text(page.title, sources)))
    chunks = chunk_page(parsed, ctx.chunk_config, count=ctx.count, quest_matcher=ctx.quest_matcher)
    guide = parsed.guide
    if guide is not None:
        if ctx.item_index is not None:
            resolve_item_ids(guide, ctx.item_index)
        attach_reference_profits([guide], ctx.reference_profits)
        if ctx.quest_titles:
            keep_only_real_quests(guide, set(ctx.quest_titles))
    quest = None
    if page.page_type == "quest":
        quest = parse_quest(page.title, page.wikitext, quest_titles=ctx.quest_titles)
    return PageResult(page.title, page.revision_id, page.page_type, chunks, guide, quest)
