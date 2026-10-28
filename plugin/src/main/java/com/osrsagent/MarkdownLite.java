package com.osrsagent;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Just enough markdown for the assistant's answers, split into blocks the panel can lay out: headings
 * (``#`` lines and lines that are entirely bold), bullets and numbered items with their nesting,
 * rules, the italic price footer, and paragraphs. Inside a block: ``**bold**``, ``code`` and a
 * muted style for ``[citations]``. Everything else is shown as written.
 */
final class MarkdownLite
{
	enum Kind
	{
		PARAGRAPH, HEADING, BULLET, NUMBERED, RULE, FOOTER
	}

	static final class Run
	{
		final String text;
		final boolean bold;
		final boolean italic;
		final boolean code;
		final boolean muted;

		Run(String text, boolean bold, boolean italic, boolean code, boolean muted)
		{
			this.text = text;
			this.bold = bold;
			this.italic = italic;
			this.code = code;
			this.muted = muted;
		}
	}

	static final class Block
	{
		final Kind kind;
		/** Heading level 1-3, or the nesting depth (0-2) of a list item. */
		final int level;
		/** The number of a numbered item ("2."), empty otherwise. */
		final String marker;
		final List<Run> runs;

		Block(Kind kind, int level, String marker, List<Run> runs)
		{
			this.kind = kind;
			this.level = level;
			this.marker = marker;
			this.runs = runs;
		}

		String text()
		{
			StringBuilder text = new StringBuilder();
			runs.forEach(run -> text.append(run.text));
			return text.toString();
		}
	}

	private static final Pattern RULE = Pattern.compile("^(-{3,}|\\*{3,}|_{3,})$");
	private static final Pattern HEADING = Pattern.compile("^(#{1,6})\\s+(.+)$");
	private static final Pattern BULLET = Pattern.compile("^(\\s*)[-*•]\\s+(.+)$");
	private static final Pattern NUMBERED = Pattern.compile("^(\\s*)(\\d{1,2})[.)]\\s+(.+)$");
	private static final Pattern TITLED_ITEM = Pattern.compile("^\\*\\*\\d{1,2}[.)]\\s[^*]+\\*\\*.{0,80}$");
	private static final int MAX_CITATION_CHARS = 90;
	private static final int MAX_NESTING = 2;

	private MarkdownLite()
	{
	}

	static List<Block> blocks(String markdown)
	{
		List<Block> blocks = new ArrayList<>();
		for (String line : markdown.split("\n", -1))
		{
			String trimmed = line.strip();
			if (trimmed.isEmpty())
			{
				continue; // blocks carry their own spacing; blank lines only separate them
			}
			Matcher heading = HEADING.matcher(trimmed);
			Matcher bullet = BULLET.matcher(line);
			Matcher numbered = NUMBERED.matcher(line);
			if (RULE.matcher(trimmed).matches())
			{
				blocks.add(new Block(Kind.RULE, 0, "", new ArrayList<>()));
			}
			else if (heading.matches())
			{
				int level = Math.min(heading.group(1).length(), 3);
				blocks.add(new Block(Kind.HEADING, level, "", inline(heading.group(2).strip(), true, false)));
			}
			else if (isWholeBold(trimmed))
			{
				String inner = trimmed.substring(2, trimmed.length() - 2).strip();
				blocks.add(new Block(Kind.HEADING, 3, "", inline(inner, true, false)));
			}
			else if (TITLED_ITEM.matcher(trimmed).matches())
			{
				// "**1. Ruby bolts (e)** — best bet": the name of one recommendation, with a tag after it
				blocks.add(new Block(Kind.HEADING, 3, "", inline(trimmed, true, false)));
			}
			else if (bullet.matches())
			{
				blocks.add(new Block(Kind.BULLET, nesting(bullet.group(1)), "", inline(bullet.group(2), false, false)));
			}
			else if (numbered.matches())
			{
				blocks.add(new Block(Kind.NUMBERED, nesting(numbered.group(1)), numbered.group(2) + ".",
					inline(numbered.group(3), false, false)));
			}
			else if (trimmed.length() > 2 && trimmed.startsWith("_") && trimmed.endsWith("_"))
			{
				blocks.add(new Block(Kind.FOOTER, 0, "", inline(trimmed.substring(1, trimmed.length() - 1), false, true)));
			}
			else
			{
				blocks.add(new Block(Kind.PARAGRAPH, 0, "", inline(trimmed, false, false)));
			}
		}
		return blocks;
	}

	/** ``**Title**`` and nothing else: the model's way of writing a heading. */
	private static boolean isWholeBold(String line)
	{
		return line.length() > 4 && line.startsWith("**") && line.endsWith("**")
			&& line.indexOf("**", 2) == line.length() - 2;
	}

	private static int nesting(String indent)
	{
		return Math.min(indent.replace("\t", "  ").length() / 2, MAX_NESTING);
	}

	/** ``**bold**`` toggles, backticks mark code, ``[a citation]`` is muted; the markers are dropped. */
	static List<Run> inline(String text, boolean bold, boolean italic)
	{
		List<Run> runs = new ArrayList<>();
		StringBuilder current = new StringBuilder();
		boolean isBold = bold;
		boolean isCode = false;
		int i = 0;
		while (i < text.length())
		{
			if (text.startsWith("**", i))
			{
				flush(runs, current, isBold, italic, isCode, false);
				isBold = !isBold;
				i += 2;
			}
			else if (text.charAt(i) == '`')
			{
				flush(runs, current, isBold, italic, isCode, false);
				isCode = !isCode;
				i++;
			}
			else if (text.charAt(i) == '[' && !isCode && citationEnd(text, i) > 0)
			{
				int end = citationEnd(text, i);
				flush(runs, current, isBold, italic, isCode, false);
				current.append(text, i, end + 1);
				flush(runs, current, false, italic, false, true);
				i = end + 1;
			}
			else
			{
				current.append(text.charAt(i));
				i++;
			}
		}
		flush(runs, current, isBold, italic, isCode, false);
		return runs;
	}

	/** Index of the ``]`` closing a citation that starts at ``from``, or -1 (links ``[a](b)`` are not). */
	private static int citationEnd(String text, int from)
	{
		int end = text.indexOf(']', from + 1);
		if (end < 0 || end - from - 1 < 3 || end - from > MAX_CITATION_CHARS)
		{
			return -1;
		}
		boolean link = end + 1 < text.length() && text.charAt(end + 1) == '(';
		return link ? -1 : end;
	}

	private static void flush(List<Run> runs, StringBuilder text, boolean bold, boolean italic, boolean code,
		boolean muted)
	{
		if (text.length() > 0)
		{
			runs.add(new Run(text.toString(), bold, italic, code, muted));
			text.setLength(0);
		}
	}
}
