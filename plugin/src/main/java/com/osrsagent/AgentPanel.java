package com.osrsagent;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.event.ActionEvent;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import javax.swing.AbstractAction;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextPane;
import javax.swing.KeyStroke;
import javax.swing.ScrollPaneConstants;
import javax.swing.SwingUtilities;
import javax.swing.text.BadLocationException;
import javax.swing.text.SimpleAttributeSet;
import javax.swing.text.StyleConstants;
import javax.swing.text.StyledDocument;
import net.runelite.client.ui.ColorScheme;
import net.runelite.client.ui.FontManager;
import net.runelite.client.ui.PluginPanel;

/**
 * The side panel (SPEC §10): the conversation, a box to type in and a few buttons. The answers are
 * laid out from a little markdown (headings, indented lists, muted citations and footer) so a long
 * one reads as sections instead of a block of text. No overlays, no highlighting in the game.
 * Every method must be called on the Swing thread (the plugin marshals results back with
 * ``SwingUtilities.invokeLater``).
 */
public class AgentPanel extends PluginPanel
{
	private static final Color USER = ColorScheme.BRAND_ORANGE;
	private static final Color ASSISTANT = new Color(0x7F, 0xC4, 0xE8);
	private static final Color TEXT = new Color(0xEE, 0xEE, 0xEE);
	private static final Color HEADING = new Color(0xFF, 0xC1, 0x6A);
	private static final Color META = new Color(0x9A, 0x9A, 0x9A);
	private static final Color CODE = new Color(0xE6, 0xC7, 0x8B);
	private static final Color ERROR = new Color(0xE0, 0x6C, 0x6C);
	private static final float LINE_SPACING = 0.18f;

	private static final String[][] SUGGESTIONS = {
		{"Make money with my stats", "What can I do to make money with my stats?"},
		{"Find me a flip", "Is there a good item to flip right now?"},
		{"Start Monkey Madness II", "What do I need to start Monkey Madness II?"},
	};

	private enum Type
	{
		USER, ASSISTANT, NOTE, ERROR
	}

	private static final class Entry
	{
		final Type type;
		final String text;
		final ChatReply reply;

		Entry(Type type, String text, ChatReply reply)
		{
			this.type = type;
			this.text = text;
			this.reply = reply;
		}
	}

	/** A text pane that smooths its text: the clean font looks rough without it. */
	private static final class TextPane extends JTextPane
	{
		boolean smooth = true;

		@Override
		protected void paintComponent(Graphics g)
		{
			if (smooth && g instanceof Graphics2D)
			{
				Graphics2D g2 = (Graphics2D) g;
				g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
				g2.setRenderingHint(RenderingHints.KEY_FRACTIONALMETRICS, RenderingHints.VALUE_FRACTIONALMETRICS_ON);
			}
			super.paintComponent(g);
		}
	}

	private final AgentConfig config;
	private final TextPane history = new TextPane();
	private final JScrollPane scroll = new JScrollPane(history);
	private final JScrollPane inputScroll;
	private final JTextArea input = new JTextArea(3, 20);
	private final JButton send = new JButton("Send");
	private final JButton fresh = new JButton("New conversation");
	private final JLabel status = new JLabel(" ");
	private final JPanel suggestions = new JPanel(new GridLayout(0, 1, 0, 4));
	private final Consumer<String> onSend;
	private final List<Entry> entries = new ArrayList<>();
	private Font base;

	public AgentPanel(AgentConfig config, Consumer<String> onSend, Runnable onNewConversation)
	{
		super(false);
		this.config = config;
		this.onSend = onSend;
		setLayout(new BorderLayout(0, 6));
		setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));
		setBackground(ColorScheme.DARK_GRAY_COLOR);

		fresh.setFocusable(false);
		fresh.addActionListener(e -> onNewConversation.run());
		add(fresh, BorderLayout.NORTH);

		history.setEditable(false);
		history.setBackground(ColorScheme.DARKER_GRAY_COLOR);
		history.setBorder(BorderFactory.createEmptyBorder(8, 10, 8, 10));
		scroll.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
		scroll.setBorder(BorderFactory.createEmptyBorder());
		add(scroll, BorderLayout.CENTER);

		input.setLineWrap(true);
		input.setWrapStyleWord(true);
		input.setBackground(ColorScheme.DARKER_GRAY_COLOR);
		input.setForeground(Color.WHITE);
		input.setCaretColor(Color.WHITE);
		input.setBorder(BorderFactory.createEmptyBorder(6, 8, 6, 8));
		input.getInputMap().put(KeyStroke.getKeyStroke("ENTER"), "send");
		input.getActionMap().put("send", new AbstractAction()
		{
			@Override
			public void actionPerformed(ActionEvent e)
			{
				submit();
			}
		});
		input.getInputMap().put(KeyStroke.getKeyStroke("shift ENTER"), "insert-break");
		send.addActionListener(e -> submit());
		send.setFocusable(false);
		status.setForeground(META);

		suggestions.setOpaque(false);
		for (String[] suggestion : SUGGESTIONS)
		{
			JButton chip = new JButton(suggestion[0]);
			chip.setFocusable(false);
			chip.addActionListener(e -> onSend.accept(suggestion[1]));
			suggestions.add(chip);
		}

		JPanel bottom = new JPanel(new BorderLayout(0, 4));
		bottom.setOpaque(false);
		bottom.add(suggestions, BorderLayout.NORTH);
		inputScroll = new JScrollPane(input);
		inputScroll.setBorder(BorderFactory.createEmptyBorder());
		bottom.add(inputScroll, BorderLayout.CENTER);
		JPanel row = new JPanel(new BorderLayout(6, 0));
		row.setOpaque(false);
		row.add(status, BorderLayout.CENTER);
		row.add(send, BorderLayout.EAST);
		bottom.add(row, BorderLayout.SOUTH);
		add(bottom, BorderLayout.SOUTH);
		restyle();
	}

	// ------------------------------------------------------------------ fonts

	private int textSize()
	{
		int size = config == null ? 0 : config.textSize();
		return size > 0 ? size : AgentConfig.DEFAULT_TEXT_SIZE;
	}

	private boolean cleanFont()
	{
		AgentConfig.FontStyle style = config == null ? null : config.fontStyle();
		return style != AgentConfig.FontStyle.RUNESCAPE;
	}

	/** Applies the configured font and size to everything and lays the conversation out again. */
	public void restyle()
	{
		boolean clean = cleanFont();
		base = clean
			? new Font(Font.SANS_SERIF, Font.PLAIN, textSize())
			: FontManager.getRunescapeFont().deriveFont((float) textSize());
		history.smooth = clean;
		history.setFont(base);
		input.setFont(base);
		status.setFont(base.deriveFont(Math.max(textSize() - 2, 10) * 1f));
		inputScroll.setPreferredSize(new Dimension(0, input.getFontMetrics(base).getHeight() * 3 + 16));
		render();
	}

	// ------------------------------------------------------------------ input

	private void submit()
	{
		String text = input.getText().trim();
		if (text.isEmpty() || !send.isEnabled())
		{
			return;
		}
		input.setText("");
		onSend.accept(text);
	}

	// ------------------------------------------------------------------ writing the conversation

	private SimpleAttributeSet attributes(Color color, boolean bold, boolean italic, boolean mono, int size)
	{
		SimpleAttributeSet style = new SimpleAttributeSet();
		StyleConstants.setForeground(style, color);
		StyleConstants.setBold(style, bold);
		StyleConstants.setItalic(style, italic);
		StyleConstants.setFontFamily(style, mono ? Font.MONOSPACED : base.getFamily());
		StyleConstants.setFontSize(style, size);
		return style;
	}

	private int write(String text, SimpleAttributeSet style)
	{
		StyledDocument doc = history.getStyledDocument();
		try
		{
			doc.insertString(doc.getLength(), text, style);
		}
		catch (BadLocationException e)
		{
			throw new IllegalStateException(e);
		}
		return doc.getLength();
	}

	private void layout(int start, float above, float below, float left, float firstLine, int alignment)
	{
		SimpleAttributeSet paragraph = new SimpleAttributeSet();
		StyleConstants.setSpaceAbove(paragraph, above);
		StyleConstants.setSpaceBelow(paragraph, below);
		StyleConstants.setLeftIndent(paragraph, left);
		StyleConstants.setFirstLineIndent(paragraph, firstLine);
		StyleConstants.setLineSpacing(paragraph, LINE_SPACING);
		StyleConstants.setAlignment(paragraph, alignment);
		StyledDocument doc = history.getStyledDocument();
		doc.setParagraphAttributes(start, doc.getLength() - start, paragraph, false);
	}

	private void writeSpeaker(String name, Color color)
	{
		int start = history.getStyledDocument().getLength();
		write(name + "\n", attributes(color, true, false, false, Math.max(textSize() - 1, 10)));
		layout(start, 14, 2, 0, 0, StyleConstants.ALIGN_LEFT);
	}

	private void writeRuns(List<MarkdownLite.Run> runs, Color color, int size, boolean baseBold, boolean baseItalic)
	{
		for (MarkdownLite.Run run : runs)
		{
			Color runColor = run.muted ? META : run.code ? CODE : color;
			int runSize = run.muted ? Math.max(size - 2, 10) : size;
			write(run.text, attributes(runColor, baseBold || run.bold, baseItalic || run.italic, run.code, runSize));
		}
	}

	private void writeBlock(MarkdownLite.Block block, boolean first)
	{
		StyledDocument doc = history.getStyledDocument();
		int start = doc.getLength();
		int size = textSize();
		switch (block.kind)
		{
			case HEADING:
				writeRuns(block.runs, HEADING, size + (block.level == 1 ? 4 : block.level == 2 ? 2 : 1), true, false);
				write("\n", attributes(HEADING, true, false, false, size));
				layout(start, first ? 2 : 12, 3, 0, 0, StyleConstants.ALIGN_LEFT);
				break;
			case BULLET:
			case NUMBERED:
				String marker = block.kind == MarkdownLite.Kind.NUMBERED ? block.marker : block.level == 0 ? "•" : "◦";
				write(marker + " ", attributes(HEADING, true, false, false, size));
				writeRuns(block.runs, TEXT, size, false, false);
				write("\n", attributes(TEXT, false, false, false, size));
				float hang = size * (block.kind == MarkdownLite.Kind.NUMBERED ? 1.5f : 1.0f);
				layout(start, 4, 1, 6 + hang + block.level * size * 1.2f, -hang, StyleConstants.ALIGN_LEFT);
				break;
			case RULE:
				write("· · ·\n", attributes(META, false, false, false, size));
				layout(start, 6, 6, 0, 0, StyleConstants.ALIGN_CENTER);
				break;
			case FOOTER:
				writeRuns(block.runs, META, Math.max(size - 2, 10), false, true);
				write("\n", attributes(META, false, true, false, size));
				layout(start, 10, 0, 0, 0, StyleConstants.ALIGN_LEFT);
				break;
			default:
				writeRuns(block.runs, TEXT, size, false, false);
				write("\n", attributes(TEXT, false, false, false, size));
				layout(start, 6, 2, 0, 0, StyleConstants.ALIGN_LEFT);
				break;
		}
	}

	/** Writes one entry; returns where it starts, so the view can scroll there. */
	private int writeEntry(Entry entry)
	{
		int start = history.getStyledDocument().getLength();
		int size = textSize();
		switch (entry.type)
		{
			case USER:
				writeSpeaker("You", USER);
				int at = history.getStyledDocument().getLength();
				write(entry.text + "\n", attributes(Color.WHITE, false, false, false, size));
				layout(at, 0, 4, 0, 0, StyleConstants.ALIGN_LEFT);
				break;
			case ASSISTANT:
				writeSpeaker("Assistant", ASSISTANT);
				List<MarkdownLite.Block> blocks = MarkdownLite.blocks(entry.reply.getAnswer());
				for (int i = 0; i < blocks.size(); i++)
				{
					writeBlock(blocks.get(i), i == 0);
				}
				List<ChatReply.Source> sources = entry.reply.getSources();
				if (!sources.isEmpty())
				{
					StringBuilder line = new StringBuilder("Sources: ");
					for (int i = 0; i < sources.size(); i++)
					{
						line.append(i == 0 ? "" : " | ").append(sources.get(i).getTitle());
					}
					int sourcesAt = history.getStyledDocument().getLength();
					write(line + "\n", attributes(META, false, true, false, Math.max(size - 2, 10)));
					layout(sourcesAt, 8, 0, 0, 0, StyleConstants.ALIGN_LEFT);
				}
				break;
			case NOTE:
				int noteAt = history.getStyledDocument().getLength();
				write(entry.text + "\n", attributes(META, false, true, false, Math.max(size - 1, 10)));
				layout(noteAt, 10, 2, 0, 0, StyleConstants.ALIGN_LEFT);
				break;
			default:
				int errorAt = history.getStyledDocument().getLength();
				write(entry.text + "\n", attributes(ERROR, false, false, false, size));
				layout(errorAt, 10, 2, 0, 0, StyleConstants.ALIGN_LEFT);
				break;
		}
		return start;
	}

	private void writeWelcome()
	{
		int start = history.getStyledDocument().getLength();
		write("Ask about quests, ways to make GP, flipping, prices or game mechanics. "
			+ "Answers use the OSRS Wiki and your character's stats.\n",
			attributes(META, false, true, false, Math.max(textSize() - 1, 10)));
		layout(start, 0, 4, 0, 0, StyleConstants.ALIGN_LEFT);
	}

	private void render()
	{
		history.setText("");
		writeWelcome();
		for (Entry entry : entries)
		{
			writeEntry(entry);
		}
		suggestions.setVisible(entries.isEmpty());
		scrollToEnd();
	}

	private void add(Entry entry, boolean scrollToStart)
	{
		entries.add(entry);
		suggestions.setVisible(false);
		int start = writeEntry(entry);
		if (scrollToStart)
		{
			scrollTo(start);
		}
		else
		{
			scrollToEnd();
		}
	}

	private void scrollToEnd()
	{
		history.setCaretPosition(history.getStyledDocument().getLength());
	}

	/** A long answer is read from its first line: bring its start to the top of the view. */
	private void scrollTo(int offset)
	{
		SwingUtilities.invokeLater(() ->
		{
			try
			{
				Rectangle2D at = history.modelToView2D(offset);
				if (at != null)
				{
					int height = Math.max(scroll.getViewport().getHeight(), 1);
					history.scrollRectToVisible(new Rectangle(0, (int) at.getY(), 1, height));
				}
			}
			catch (BadLocationException e)
			{
				scrollToEnd();
			}
		});
	}

	// ------------------------------------------------------------------ what the plugin calls

	public void appendUser(String text)
	{
		add(new Entry(Type.USER, text, null), false);
	}

	public void appendAssistant(ChatReply reply)
	{
		add(new Entry(Type.ASSISTANT, null, reply), true);
	}

	public void appendNote(String text)
	{
		add(new Entry(Type.NOTE, text, null), false);
	}

	public void appendError(String text)
	{
		add(new Entry(Type.ERROR, text, null), false);
	}

	public void setBusy(boolean busy)
	{
		send.setEnabled(!busy);
		input.setEnabled(!busy);
		status.setText(busy ? "Thinking..." : " ");
		if (!busy)
		{
			input.requestFocusInWindow();
		}
	}

	public void clear()
	{
		entries.clear();
		render();
	}

	/** The conversation view (used by the tests and the preview). */
	JTextPane historyView()
	{
		return history;
	}

	/** The conversation as plain text (used by the tests). */
	String historyText()
	{
		return history.getText();
	}
}
