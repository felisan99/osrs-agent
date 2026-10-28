package com.osrsagent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import okhttp3.OkHttpClient;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import okhttp3.mockwebserver.SocketPolicy;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/** The HTTP client against a local server that plays the backend's part of SPEC §4. */
public class AgentApiTest
{
	private static final String KEY = "sk-or-v1-PLUGINTESTKEY0123456789";
	private static final Gson GSON = new Gson();

	private MockWebServer server;
	private AgentApi api;
	private String base;

	@Before
	public void setUp() throws IOException
	{
		server = new MockWebServer();
		server.start();
		base = server.url("/").toString();
		api = new AgentApi(new OkHttpClient(), GSON);
	}

	@After
	public void tearDown() throws IOException
	{
		server.shutdown();
	}

	/** Runs one question and returns what the listener received: a reply or an error message. */
	private Object ask(String baseUrl, JsonObject state, int timeoutSeconds) throws Exception
	{
		CompletableFuture<Object> result = new CompletableFuture<>();
		api.chat(baseUrl, KEY, "thread-123456789", "¿Qué nivel de Attack necesito?", state, timeoutSeconds,
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
		return result.get(15, TimeUnit.SECONDS);
	}

	private static MockResponse json(int status, String body)
	{
		return new MockResponse().setResponseCode(status).setHeader("Content-Type", "application/json").setBody(body);
	}

	@Test
	public void aQuestionIsPostedWithTheKeyInTheHeaderOnly() throws Exception
	{
		server.enqueue(json(200, "{\"answer\":\"Necesitás **70** de Attack [Abyssal whip].\","
			+ "\"intent\":\"wiki_general\",\"sources\":[{\"title\":\"Abyssal whip\",\"url\":\"https://x/w\"}]}"));
		JsonObject state = new JsonObject();
		state.addProperty("account_type", "main");
		Object result = ask("  " + base.replaceAll("/$", "/ ") .trim() + "/", state, 20);

		assertTrue(result instanceof ChatReply);
		ChatReply reply = (ChatReply) result;
		assertEquals("Necesitás **70** de Attack [Abyssal whip].", reply.getAnswer());
		assertEquals("wiki_general", reply.getIntent());
		assertEquals("Abyssal whip", reply.getSources().get(0).getTitle());

		RecordedRequest request = server.takeRequest();
		assertEquals("POST", request.getMethod());
		assertEquals("/v1/chat", request.getPath());
		assertEquals(KEY, request.getHeader("X-OpenRouter-Key"));
		assertTrue(request.getHeader("Content-Type").startsWith("application/json"));
		String body = request.getBody().readUtf8();
		JsonObject sent = GSON.fromJson(body, JsonObject.class);
		assertEquals("thread-123456789", sent.get("thread_id").getAsString());
		assertEquals("¿Qué nivel de Attack necesito?", sent.get("message").getAsString());
		assertEquals("main", sent.getAsJsonObject("player_state").get("account_type").getAsString());
		assertFalse("the key never goes in the body", body.contains("sk-or"));
		assertNull("no Authorization header: the backend has no key of its own to check", request.getHeader("Authorization"));
	}

	@Test
	public void withoutASnapshotThePlayerStateIsOmitted() throws Exception
	{
		server.enqueue(json(200, "{\"answer\":\"ok\",\"intent\":\"wiki_general\",\"sources\":[]}"));
		ask(base, null, 20);
		JsonObject sent = GSON.fromJson(server.takeRequest().getBody().readUtf8(), JsonObject.class);
		assertFalse(sent.has("player_state"));
	}

	@Test
	public void trailingSlashesInTheBackendUrlAreHarmless() throws Exception
	{
		server.enqueue(json(200, "{\"answer\":\"ok\",\"intent\":\"wiki_general\",\"sources\":[]}"));
		ask(base + "///", null, 20);
		assertEquals("/v1/chat", server.takeRequest().getPath());
	}

	private String error(int status, String body, String... headers) throws Exception
	{
		MockResponse response = json(status, body);
		for (int i = 0; i < headers.length; i += 2)
		{
			response.setHeader(headers[i], headers[i + 1]);
		}
		server.enqueue(response);
		Object result = ask(base, null, 20);
		assertTrue("expected an error message, got " + result, result instanceof String);
		return (String) result;
	}

	@Test
	public void theTypedErrorsOfTheContractBecomeReadableMessages() throws Exception
	{
		assertTrue(error(401, "{\"error\":\"invalid_key\",\"message\":\"User not found.\"}").contains("API key"));
		assertTrue(error(402, "{\"error\":\"insufficient_credits\",\"message\":\"x\"}").contains("credits"));
		assertTrue(error(502, "{\"error\":\"upstream_error\",\"message\":\"x\"}").contains("Try again"));
		assertTrue(error(413, "{\"error\":\"payload_too_large\",\"message\":\"x\"}").contains("too large"));
		assertTrue(error(422, "{\"error\":\"invalid_request\",\"message\":\"Invalid request at 'message'\"}")
			.contains("message"));
	}

	@Test
	public void rateLimitsUseTheBodyOrTheRetryAfterHeader() throws Exception
	{
		assertTrue(error(429, "{\"error\":\"rate_limited\",\"message\":\"x\",\"retry_after_s\":12.0}",
			"Retry-After", "12").contains("12 seconds"));
		assertTrue(error(429, "{\"error\":\"rate_limited\",\"message\":\"x\"}", "Retry-After", "30")
			.contains("30 seconds"));
	}

	@Test
	public void responsesThatAreNotOurTypedErrorsStillGetAMessage() throws Exception
	{
		assertTrue(error(404, "<html>nope</html>").contains("Backend URL"));
		assertTrue(error(500, "{\"error\":\"internal_error\",\"message\":\"Unexpected\"}").contains("HTTP 500"));
		assertTrue(error(502, "").contains("HTTP 502"));
	}

	@Test
	public void anAnswerThePluginCannotReadIsReportedNotThrown() throws Exception
	{
		server.enqueue(json(200, "this is not json"));
		Object result = ask(base, null, 20);
		assertEquals("The backend sent an answer the plugin could not read.", result);
	}

	@Test
	public void aBackendThatIsNotRunningIsExplained() throws Exception
	{
		String dead = server.url("/").toString();
		server.shutdown();
		Object result = ask(dead, null, 20);
		assertTrue(String.valueOf(result), result instanceof String && ((String) result).contains("Could not reach"));
		assertTrue(((String) result).contains(AgentApi.normalise(dead)));
	}

	@Test
	public void aBackendThatNeverAnswersTimesOut() throws Exception
	{
		server.enqueue(new MockResponse().setSocketPolicy(SocketPolicy.NO_RESPONSE));
		long started = System.nanoTime();
		Object result = ask(base, null, 1);
		assertEquals("The backend did not answer within 1 seconds.", result);
		assertTrue(TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - started) < 10);
	}

	@Test
	public void anInvalidUrlFailsAtOnceWithoutARequest() throws Exception
	{
		Object result = ask("not a url", null, 20);
		assertTrue(((String) result).contains("not valid"));
		assertEquals(0, server.getRequestCount());
	}

	@Test
	public void aSlowAnswerIsNotCutByTheSharedClientsShortReadTimeout() throws Exception
	{
		// RuneLite's injected client keeps OkHttp's default 10 s read timeout; stand in for it with 1 s.
		api = new AgentApi(new OkHttpClient.Builder().readTimeout(1, TimeUnit.SECONDS).build(), GSON);
		server.enqueue(json(200, "{\"answer\":\"slow but fine\",\"intent\":\"flipping\",\"sources\":[]}")
			.setHeadersDelay(2500, TimeUnit.MILLISECONDS));
		Object result = ask(base, null, 30);
		assertTrue(String.valueOf(result), result instanceof ChatReply);
		assertEquals("slow but fine", ((ChatReply) result).getAnswer());
	}

	@Test
	public void theCallReturnedCanBeCancelledWithoutAFalseError() throws Exception
	{
		server.enqueue(new MockResponse().setSocketPolicy(SocketPolicy.NO_RESPONSE));
		CompletableFuture<Object> result = new CompletableFuture<>();
		okhttp3.Call call = api.chat(base, KEY, "thread-123456789", "hola", null, 20, new AgentApi.Listener()
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
		assertNotNull(call);
		call.cancel();
		try
		{
			result.get(2, TimeUnit.SECONDS);
			throw new AssertionError("a cancelled question must stay silent");
		}
		catch (java.util.concurrent.TimeoutException expected)
		{
			// silence is the correct outcome
		}
	}
}
