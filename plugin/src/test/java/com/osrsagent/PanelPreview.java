package com.osrsagent;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Collections;
import javax.imageio.ImageIO;
import javax.swing.SwingUtilities;

/**
 * Renders the chat panel to PNG files without opening RuneLite, so a change to its look can be seen:
 * ``./gradlew panelPreview`` writes build/panel-preview/*.png (one per font setting, at 2x).
 */
public final class PanelPreview
{
	static final String FLIP = "Here are two solid picks for your 2.5M GP using the **volume** strategy "
		+ "(liquid items you can flip repeatedly):\n\n"
		+ "**1. Ruby bolts (e)** — best bet\n"
		+ "- **Buy at 189 gp**, sell at **196 gp** (margin 7 gp after 2% tax right now)\n"
		+ "- **11,000 units** = 2,079,000 GP capital\n"
		+ "- Conservative profit per 4h window: ~36,602 GP (typical ~44,000 GP)\n"
		+ "- The margin was positive in **99% of the last 168 hours**. Volume is strong at ~28k/hr "
		+ "on the slower side, and the price sits just 0.1% below its 7-day median.\n\n"
		+ "**2. Steam rune** — slightly higher ROI\n"
		+ "- **Buy at 95 gp**, sell at **99 gp** (margin 4 gp after tax)\n"
		+ "- **18,000 units** = 1,710,000 GP capital\n"
		+ "- Margin positive **97% of hours**, volume ~39k/hr, buy/sell ratio 1.30 (balanced).\n\n"
		+ "Practical steps: put a buy offer in for 11k ruby bolts (e) at 189 gp each. It should fill in "
		+ "about 1.6 hours. Once filled, list them at 196 gp. Don't put your whole bank into one item.\n\n"
		+ "_Prices from the OSRS Wiki prices API, fetched 2026-10-15 10:30 UTC._";

	static final String MONEY = "With your 75 attack, 81 strength and 74 defence you have a few solid options, "
		+ "though the best ones need expensive gear.\n\n"
		+ "**What you can do now**\n"
		+ "- **Wilderness Agility Course** – ~1,947,630 GP/h. You meet the 52 Agility requirement (you have 56). "
		+ "Needs 150,000 coins for the dispenser fee. [Wilderness Agility Course]\n"
		+ "- **Smelting steel bars at Blast Furnace** – ~1,719,079 GP/h. You have 45 Smithing (needs 30) and "
		+ "have finished The Giant Dwarf. [Smelting steel bars at Blast Furnace]\n"
		+ "  - You'll want a coal bag and weight-reducing clothing.\n\n"
		+ "**What you're close to**\n"
		+ "1. **Killing Blood Moon** (~1,801,768 GP/h) once you hit Defence 75 (you have 74).\n"
		+ "2. **Smelting mithril bars** (~1,686,220 GP/h) once you get Smithing 50 (you have 45).\n\n"
		+ "---\n"
		+ "Use `rank_flips` for ideas.\n\n"
		+ "_Prices from the OSRS Wiki prices API, fetched 2026-10-15 09:57 UTC._";

	private PanelPreview()
	{
	}

	static BufferedImage render(AgentConfig config, int width, int height) throws Exception
	{
		BufferedImage[] image = new BufferedImage[1];
		SwingUtilities.invokeAndWait(() ->
		{
			AgentPanel panel = new AgentPanel(config, text -> { }, () -> { });
			panel.setSize(new Dimension(width, height));
			panel.doLayout();
			layoutAll(panel);
			image[0] = new BufferedImage(width * 2, height * 2, BufferedImage.TYPE_INT_RGB);
			Graphics2D g = image[0].createGraphics();
			g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
			g.scale(2, 2);
			g.setColor(Color.BLACK);
			g.fillRect(0, 0, width, height);
			panel.paint(g);
			g.dispose();
		});
		return image[0];
	}

	/** The whole conversation at its full height (the panel itself scrolls). */
	static BufferedImage renderConversation(AgentConfig config, int width) throws Exception
	{
		BufferedImage[] image = new BufferedImage[1];
		SwingUtilities.invokeAndWait(() ->
		{
			AgentPanel panel = new AgentPanel(config, text -> { }, () -> { });
			panel.appendUser("What can I do for moneymaking given my combat stats?");
			panel.appendAssistant(new ChatReply(MONEY, "moneymaking", Collections.emptyList()));
			panel.appendUser("Any good flipping item right now?");
			panel.appendAssistant(new ChatReply(FLIP, "flipping", java.util.Arrays.asList(
				new ChatReply.Source("OSRS Wiki real-time prices", "https://prices.runescape.wiki/"))));
			panel.appendError("The backend did not answer within 120 seconds.");
			javax.swing.JTextPane view = panel.historyView();
			java.awt.Insets in = view.getInsets();
			javax.swing.text.View root = view.getUI().getRootView(view);
			root.setSize(width - in.left - in.right, Integer.MAX_VALUE);
			int height = (int) root.getPreferredSpan(javax.swing.text.View.Y_AXIS) + in.top + in.bottom;
			view.setSize(width, height);
			image[0] = new BufferedImage(width * 2, height * 2, BufferedImage.TYPE_INT_RGB);
			Graphics2D g = image[0].createGraphics();
			g.scale(2, 2);
			view.paint(g);
			g.dispose();
		});
		return image[0];
	}

	private static void layoutAll(java.awt.Container container)
	{
		container.doLayout();
		for (java.awt.Component child : container.getComponents())
		{
			if (child instanceof java.awt.Container)
			{
				layoutAll((java.awt.Container) child);
			}
		}
	}

	private static AgentConfig config(int size, AgentConfig.FontStyle style)
	{
		return new AgentConfig()
		{
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
		};
	}

	public static void main(String[] args) throws Exception
	{
		System.setProperty("java.awt.headless", "true");
		File out = new File("build/panel-preview");
		out.mkdirs();
		int width = 240;
		int height = 760;
		ImageIO.write(render(config(15, AgentConfig.FontStyle.CLEAN), width, 560), "png",
			new File(out, "empty-state.png"));
		ImageIO.write(renderConversation(config(15, AgentConfig.FontStyle.CLEAN), 215), "png",
			new File(out, "clean-15.png"));
		ImageIO.write(renderConversation(config(18, AgentConfig.FontStyle.CLEAN), 215), "png",
			new File(out, "clean-18.png"));
		ImageIO.write(renderConversation(config(16, AgentConfig.FontStyle.RUNESCAPE), 215), "png",
			new File(out, "runescape-16.png"));
		System.out.println("wrote " + out.getAbsolutePath());
	}
}
