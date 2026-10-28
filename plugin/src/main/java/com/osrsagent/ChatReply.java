package com.osrsagent;

import java.util.Collections;
import java.util.List;

/** A parsed ``POST /v1/chat`` answer. */
public final class ChatReply
{
	/** A wiki page the answer used. */
	public static final class Source
	{
		private final String title;
		private final String url;

		public Source(String title, String url)
		{
			this.title = title;
			this.url = url;
		}

		public String getTitle()
		{
			return title;
		}

		public String getUrl()
		{
			return url;
		}
	}

	private final String answer;
	private final String intent;
	private final List<Source> sources;

	public ChatReply(String answer, String intent, List<Source> sources)
	{
		this.answer = answer;
		this.intent = intent;
		this.sources = Collections.unmodifiableList(sources);
	}

	public String getAnswer()
	{
		return answer;
	}

	public String getIntent()
	{
		return intent;
	}

	public List<Source> getSources()
	{
		return sources;
	}
}
