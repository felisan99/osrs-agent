package com.osrsagent;

import net.runelite.client.config.Config;
import net.runelite.client.config.ConfigGroup;
import net.runelite.client.config.ConfigItem;
import net.runelite.client.config.Range;

@ConfigGroup(AgentConfig.GROUP)
public interface AgentConfig extends Config
{
	String GROUP = "osrsagent";
	int DEFAULT_TEXT_SIZE = 15;

	enum FontStyle
	{
		CLEAN("Clean (easier to read)"),
		RUNESCAPE("RuneScape");

		private final String label;

		FontStyle(String label)
		{
			this.label = label;
		}

		@Override
		public String toString()
		{
			return label;
		}
	}

	@ConfigItem(
		keyName = "backendUrl",
		name = "Backend URL",
		description = "Where the OSRS Agent backend runs. Questions and your character snapshot are sent here.",
		position = 1
	)
	default String backendUrl()
	{
		return "http://localhost:8000";
	}

	@ConfigItem(
		keyName = "openRouterKey",
		name = "OpenRouter API key",
		description = "Your own OpenRouter key. It is sent to the backend with every question, which "
			+ "uses it for that request only and never stores it. RuneLite keeps it in your local "
			+ "profile file.",
		secret = true,
		position = 2
	)
	default String openRouterKey()
	{
		return "";
	}

	@ConfigItem(
		keyName = "timeoutSeconds",
		name = "Answer timeout (seconds)",
		description = "How long to wait for an answer before giving up.",
		position = 3
	)
	@Range(min = 10, max = 300)
	default int timeoutSeconds()
	{
		return 120;
	}

	@ConfigItem(
		keyName = "textSize",
		name = "Text size",
		description = "Size of the text in the chat. Larger is easier to read.",
		position = 4
	)
	@Range(min = 11, max = 28)
	default int textSize()
	{
		return DEFAULT_TEXT_SIZE;
	}

	@ConfigItem(
		keyName = "fontStyle",
		name = "Font",
		description = "Clean is a plain sans-serif font that is easier to read in long answers; "
			+ "RuneScape is the game's own font.",
		position = 5
	)
	default FontStyle fontStyle()
	{
		return FontStyle.CLEAN;
	}
}
