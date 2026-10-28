package com.osrsagent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.List;
import org.junit.Test;

public class MarkdownLiteTest
{
	private static MarkdownLite.Block only(String markdown)
	{
		List<MarkdownLite.Block> blocks = MarkdownLite.blocks(markdown);
		assertEquals(1, blocks.size());
		return blocks.get(0);
	}

	@Test
	public void boldSegmentsAreMarkedAndTheMarkersDropped()
	{
		MarkdownLite.Block block = only("You need **70 Attack** to wield it");
		assertEquals(MarkdownLite.Kind.PARAGRAPH, block.kind);
		assertEquals(3, block.runs.size());
		assertEquals("You need ", block.runs.get(0).text);
		assertFalse(block.runs.get(0).bold);
		assertEquals("70 Attack", block.runs.get(1).text);
		assertTrue(block.runs.get(1).bold);
		assertEquals(" to wield it", block.runs.get(2).text);
		assertEquals("You need 70 Attack to wield it", block.text());
	}

	@Test
	public void bulletsKeepTheirNestingLevel()
	{
		List<MarkdownLite.Block> blocks = MarkdownLite.blocks("- first\n  * nested\n    - deeper\n      - capped");
		assertEquals(4, blocks.size());
		assertEquals(MarkdownLite.Kind.BULLET, blocks.get(0).kind);
		assertEquals(0, blocks.get(0).level);
		assertEquals(1, blocks.get(1).level);
		assertEquals(2, blocks.get(2).level);
		assertEquals(2, blocks.get(3).level);
		assertEquals("nested", blocks.get(1).text());
	}

	@Test
	public void numberedItemsKeepTheirNumber()
	{
		List<MarkdownLite.Block> blocks = MarkdownLite.blocks("1. **Ruby bolts** — best bet\n2) Steam rune");
		assertEquals(MarkdownLite.Kind.NUMBERED, blocks.get(0).kind);
		assertEquals("1.", blocks.get(0).marker);
		assertEquals("2.", blocks.get(1).marker);
		assertEquals("Ruby bolts — best bet", blocks.get(0).text());
		assertTrue(blocks.get(0).runs.get(0).bold);
	}

	@Test
	public void hashesAndAWholeBoldLineAreHeadings()
	{
		MarkdownLite.Block hash = only("## What you can do now");
		assertEquals(MarkdownLite.Kind.HEADING, hash.kind);
		assertEquals(2, hash.level);
		assertEquals("What you can do now", hash.text());
		MarkdownLite.Block bold = only("**What you're close to**");
		assertEquals(MarkdownLite.Kind.HEADING, bold.kind);
		assertTrue(bold.runs.get(0).bold);
		MarkdownLite.Block titled = only("**1. Ruby bolts (e)** — best bet");
		assertEquals(MarkdownLite.Kind.HEADING, titled.kind);
		assertEquals("1. Ruby bolts (e) — best bet", titled.text());
		// bold at the start of a sentence is not a heading
		assertEquals(MarkdownLite.Kind.PARAGRAPH, only("**Note:** this is a sentence").kind);
		assertEquals(MarkdownLite.Kind.PARAGRAPH, only("**a** and **b**").kind);
	}

	@Test
	public void aLineWrappedInUnderscoresIsTheItalicFooter()
	{
		List<MarkdownLite.Block> blocks = MarkdownLite.blocks("Answer\n\n_Prices fetched 2026-10-14 11:59 UTC._");
		assertEquals(2, blocks.size());
		MarkdownLite.Block footer = blocks.get(1);
		assertEquals(MarkdownLite.Kind.FOOTER, footer.kind);
		assertTrue(footer.runs.get(0).italic);
		assertEquals("Prices fetched 2026-10-14 11:59 UTC.", footer.text());
		assertFalse(blocks.get(0).runs.get(0).italic);
	}

	@Test
	public void citationsAreMutedAndCodeIsMarked()
	{
		MarkdownLite.Block block = only("Pays 1.9M GP/h [Money making guide/Wilderness Agility Course] with `rank_flips`");
		assertEquals(4, block.runs.size());
		assertTrue(block.runs.get(1).muted);
		assertEquals("[Money making guide/Wilderness Agility Course]", block.runs.get(1).text);
		assertTrue(block.runs.get(3).code);
		assertEquals("rank_flips", block.runs.get(3).text);
		// a markdown link and a tiny bracket are plain text
		assertFalse(only("see [the wiki](https://x.org)").runs.get(0).muted);
		assertEquals("a [x] b", only("a [x] b").text());
	}

	@Test
	public void rulesBlankLinesAndOrderAreHandled()
	{
		List<MarkdownLite.Block> blocks = MarkdownLite.blocks("first\n\n\n---\n\nsecond\n");
		assertEquals(3, blocks.size());
		assertEquals(MarkdownLite.Kind.PARAGRAPH, blocks.get(0).kind);
		assertEquals(MarkdownLite.Kind.RULE, blocks.get(1).kind);
		assertEquals("second", blocks.get(2).text());
		assertTrue(MarkdownLite.blocks("").isEmpty());
	}

	@Test
	public void anUnclosedBoldMarkerDoesNotLoseText()
	{
		assertEquals("a b", only("a **b").text());
	}

	@Test
	public void plainTextIsLeftAlone()
	{
		MarkdownLite.Block block = only("2 * 3 = 6 and snake_case_name");
		assertEquals(1, block.runs.size());
		assertEquals("2 * 3 = 6 and snake_case_name", block.text());
	}
}
