package com.osrsagent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import java.awt.Component;
import java.awt.Container;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.swing.JButton;
import javax.swing.SwingUtilities;
import net.runelite.client.ui.FontManager;
import org.junit.Test;

public class AgentPanelTest
{
	private static final class Look implements AgentConfig
	{
		int size;
		FontStyle style;

		Look(int size, FontStyle style)
		{
			this.size = size;
			this.style = style;
		}

		@Override
		public int textSize()
		{
			return size;
		}

		@Override
		public FontStyle fontStyle()
		{
			return style;
		}
	}

	private static void onSwing(Runnable work) throws Exception
	{
		SwingUtilities.invokeAndWait(work);
	}

	private static void buttons(Container container, List<JButton> into)
	{
		for (Component child : container.getComponents())
		{
			if (child instanceof JButton)
			{
				into.add((JButton) child);
			}
			if (child instanceof Container)
			{
				buttons((Container) child, into);
			}
		}
	}

	private static ChatReply reply(String answer)
	{
		return new ChatReply(answer, "wiki_general", Collections.emptyList());
	}

	@Test
	public void theTextIsLargerThanTheGamesSmallFontByDefaultAndFollowsTheConfig() throws Exception
	{
		Look look = new Look(0, null); // a config that reports nothing: the defaults apply
		AgentPanel[] panel = new AgentPanel[1];
		onSwing(() -> panel[0] = new AgentPanel(look, text -> { }, () -> { }));
		assertEquals(AgentConfig.DEFAULT_TEXT_SIZE, panel[0].historyView().getFont().getSize());
		assertTrue(AgentConfig.DEFAULT_TEXT_SIZE > 12);
		assertEquals(AgentConfig.DEFAULT_TEXT_SIZE, panel[0].historyView().getFont().getSize());

		look.size = 22;
		onSwing(panel[0]::restyle);
		assertEquals(22, panel[0].historyView().getFont().getSize());
	}

	@Test
	public void theFontStyleSwitchesBetweenCleanAndTheGamesFont() throws Exception
	{
		Look look = new Look(16, AgentConfig.FontStyle.CLEAN);
		AgentPanel[] panel = new AgentPanel[1];
		onSwing(() -> panel[0] = new AgentPanel(look, text -> { }, () -> { }));
		String clean = panel[0].historyView().getFont().getFamily();
		look.style = AgentConfig.FontStyle.RUNESCAPE;
		onSwing(panel[0]::restyle);
		String game = panel[0].historyView().getFont().getFamily();
		assertEquals(FontManager.getRunescapeFont().getFamily(), game);
		assertNotEquals(clean, game);
	}

	@Test
	public void anAnswerIsLaidOutWithoutMarkdownSymbolsAndKeepsEveryWord() throws Exception
	{
		AgentPanel[] panel = new AgentPanel[1];
		onSwing(() -> panel[0] = new AgentPanel(new Look(15, AgentConfig.FontStyle.CLEAN), text -> { }, () -> { }));
		onSwing(() -> panel[0].appendAssistant(reply(
			"Intro line.\n\n**What you can do now**\n- **Ruby bolts** – 7 gp [Ruby bolts]\n"
				+ "1. First step\n\n_Prices fetched 2026-10-15 UTC._")));
		String text = panel[0].historyText();
		assertTrue(text, text.contains("What you can do now"));
		assertTrue(text, text.contains("• Ruby bolts – 7 gp [Ruby bolts]"));
		assertTrue(text, text.contains("1. First step"));
		assertTrue(text, text.contains("Prices fetched 2026-10-15 UTC."));
		assertFalse(text, text.contains("**"));
		assertFalse(text, text.contains("_Prices"));
	}

	@Test
	public void changingTheFontKeepsTheConversation() throws Exception
	{
		Look look = new Look(15, AgentConfig.FontStyle.CLEAN);
		AgentPanel[] panel = new AgentPanel[1];
		onSwing(() -> panel[0] = new AgentPanel(look, text -> { }, () -> { }));
		onSwing(() ->
		{
			panel[0].appendUser("hello there");
			panel[0].appendAssistant(reply("an answer"));
			panel[0].appendError("an error");
		});
		String before = panel[0].historyText();
		look.size = 24;
		look.style = AgentConfig.FontStyle.RUNESCAPE;
		onSwing(panel[0]::restyle);
		assertEquals(before, panel[0].historyText());
	}

	@Test
	public void theQuestionShortcutsShowOnlyWhileTheConversationIsEmpty() throws Exception
	{
		List<String> sent = new ArrayList<>();
		AgentPanel[] panel = new AgentPanel[1];
		onSwing(() -> panel[0] = new AgentPanel(new Look(15, AgentConfig.FontStyle.CLEAN), sent::add, () -> { }));
		List<JButton> found = new ArrayList<>();
		onSwing(() -> buttons(panel[0], found));
		JButton flip = found.stream().filter(b -> b.getText().equals("Find me a flip")).findFirst().get();
		assertTrue(flip.isVisible() && flip.getParent().isVisible());
		onSwing(flip::doClick);
		assertEquals(Collections.singletonList("Is there a good item to flip right now?"), sent);

		onSwing(() -> panel[0].appendUser("a question"));
		assertFalse(flip.getParent().isVisible());
		onSwing(panel[0]::clear);
		assertTrue(flip.getParent().isVisible());
		assertFalse(panel[0].historyText().contains("a question"));
	}
}
