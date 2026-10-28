package com.osrsagent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import com.google.gson.Gson;
import java.time.Instant;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import okhttp3.OkHttpClient;
import org.junit.Test;

/**
 * Talks to a REAL backend with a real OpenRouter key, sending a snapshot built like the plugin's.
 * Skipped unless ``OSRS_AGENT_KEY`` is set (it spends a fraction of a cent):
 *
 * <pre>OSRS_AGENT_KEY=sk-or-... [OSRS_AGENT_URL=http://localhost:8000] ./gradlew test --tests '*Live*'</pre>
 */
public class LiveBackendTest
{
	private static final Gson GSON = new Gson();

	private static PlayerSnapshot midGameMain()
	{
		Map<String, Integer> skills = new LinkedHashMap<>();
		for (String name : Arrays.asList("attack", "defence", "strength", "hitpoints", "ranged", "prayer", "magic",
			"cooking", "woodcutting", "fletching", "fishing", "firemaking", "crafting", "smithing", "mining",
			"herblore", "agility", "thieving", "slayer", "farming", "runecraft", "hunter", "construction", "sailing"))
		{
			skills.put(name, 60);
		}
		skills.put("slayer", 66);
		skills.put("crafting", 62);
		Map<String, String> quests = new LinkedHashMap<>();
		quests.put("Monkey Madness I", "FINISHED");
		quests.put("Tree Gnome Village", "FINISHED");
		quests.put("The Grand Tree", "FINISHED");
		quests.put("Recipe for Disaster", "NOT_STARTED");
		return new PlayerSnapshot("main", Boolean.TRUE, skills, quests, 150, 6_500_000L,
			Collections.singletonList(new ItemStack(995, 6_500_000)), null, Instant.now());
	}

	private static String env(String name, String fallback)
	{
		String value = System.getenv(name);
		return value == null || value.trim().isEmpty() ? fallback : value.trim();
	}

	@Test
	public void aRealQuestionWithARealSnapshotGetsARealAnswer() throws Exception
	{
		String key = env("OSRS_AGENT_KEY", "");
		assumeTrue("set OSRS_AGENT_KEY to run against a real backend", !key.isEmpty());
		AgentApi api = new AgentApi(new OkHttpClient(), GSON);
		CompletableFuture<Object> result = new CompletableFuture<>();
		api.chat(env("OSRS_AGENT_URL", "http://localhost:8000"), key, "live-" + System.nanoTime(),
			"What do I need to start Monkey Madness II?", midGameMain().toJson(), 120,
			new AgentApi.Listener()
			{
				@Override
				public void onReply(ChatReply reply)
				{
					result.complete(reply);
				}

				@Override
				public void onError(String message)
				{
					result.complete(message);
				}
			});
		Object outcome = result.get(130, TimeUnit.SECONDS);
		assertTrue(String.valueOf(outcome), outcome instanceof ChatReply);
		ChatReply reply = (ChatReply) outcome;
		assertEquals("quest", reply.getIntent());
		assertFalse(reply.getAnswer().trim().isEmpty());
		assertFalse(reply.getSources().isEmpty());
		System.out.println("LIVE ANSWER (" + reply.getIntent() + "): " + reply.getAnswer());
	}

	@Test
	public void aWrongKeyIsReportedAsAReadableMessage() throws Exception
	{
		assumeTrue("set OSRS_AGENT_KEY to run against a real backend", !env("OSRS_AGENT_KEY", "").isEmpty());
		AgentApi api = new AgentApi(new OkHttpClient(), GSON);
		CompletableFuture<Object> result = new CompletableFuture<>();
		api.chat(env("OSRS_AGENT_URL", "http://localhost:8000"), "sk-or-v1-notarealkey000000000000000000",
			"live-bad-" + System.nanoTime(), "What is a whip?", null, 60, new AgentApi.Listener()
			{
				@Override
				public void onReply(ChatReply reply)
				{
					result.complete(reply);
				}

				@Override
				public void onError(String message)
				{
					result.complete(message);
				}
			});
		Object outcome = result.get(70, TimeUnit.SECONDS);
		assertTrue(String.valueOf(outcome), outcome instanceof String);
		assertTrue((String) outcome, ((String) outcome).contains("API key"));
	}
}
