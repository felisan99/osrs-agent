package com.osrsagent;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonSyntaxException;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.HttpUrl;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;

/**
 * The HTTP side of the plugin: one asynchronous ``POST /v1/chat`` per question (SPEC §4, §10).
 *
 * Requests run on OkHttp's own threads, never the client thread. The key travels only in the
 * ``X-OpenRouter-Key`` header; nothing here logs headers or bodies, and there is no logging
 * interceptor on the client.
 */
public final class AgentApi
{
	public interface Listener
	{
		void onReply(ChatReply reply);

		void onError(String message);
	}

	private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
	private static final int CONNECT_TIMEOUT_SECONDS = 5;

	private final OkHttpClient http;
	private final Gson gson;

	public AgentApi(OkHttpClient http, Gson gson)
	{
		this.http = http;
		this.gson = gson;
	}

	static String normalise(String baseUrl)
	{
		String root = baseUrl == null ? "" : baseUrl.trim();
		while (root.endsWith("/"))
		{
			root = root.substring(0, root.length() - 1);
		}
		return root;
	}

	/** Starts the request and returns at once; the listener is called from an OkHttp thread. */
	public Call chat(
		String baseUrl,
		String key,
		String threadId,
		String message,
		JsonObject playerState,
		int timeoutSeconds,
		Listener listener)
	{
		String root = normalise(baseUrl);
		HttpUrl url = HttpUrl.parse(root + "/v1/chat");
		if (url == null)
		{
			listener.onError(ErrorMessages.badUrl(root));
			return null;
		}
		JsonObject body = new JsonObject();
		body.addProperty("thread_id", threadId);
		body.addProperty("message", message);
		if (playerState != null)
		{
			body.add("player_state", playerState);
		}
		Request request = new Request.Builder()
			.url(url)
			.header("X-OpenRouter-Key", key.trim())
			.post(RequestBody.create(JSON, gson.toJson(body)))
			.build();
		OkHttpClient client = http.newBuilder()
			.connectTimeout(CONNECT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
			// The injected client keeps OkHttp's 10 s read timeout, far below the answer timeout the
			// player configured: an answer that takes 12 s would be cut at 10 and reported as if the
			// whole timeout had passed. The whole call is still capped by callTimeout.
			.readTimeout(timeoutSeconds, TimeUnit.SECONDS)
			.writeTimeout(timeoutSeconds, TimeUnit.SECONDS)
			.callTimeout(timeoutSeconds, TimeUnit.SECONDS)
			.build();
		Call call = client.newCall(request);
		call.enqueue(new Callback()
		{
			@Override
			public void onFailure(Call call, IOException e)
			{
				// A player's cancel is silent, but OkHttp also cancels a call that times out: that one
				// surfaces as an InterruptedIOException and must still be reported.
				if (call.isCanceled() && !(e instanceof InterruptedIOException))
				{
					return;
				}
				listener.onError(e instanceof InterruptedIOException
					? ErrorMessages.timeout(timeoutSeconds)
					: ErrorMessages.unreachable(root));
			}

			@Override
			public void onResponse(Call call, Response response)
			{
				try (ResponseBody responseBody = response.body())
				{
					String text = responseBody == null ? "" : responseBody.string();
					if (response.isSuccessful())
					{
						listener.onReply(parseReply(text));
					}
					else
					{
						listener.onError(parseError(response.code(), response.header("Retry-After"), text));
					}
				}
				catch (IOException e)
				{
					listener.onError(e instanceof InterruptedIOException
						? ErrorMessages.timeout(timeoutSeconds)
						: ErrorMessages.unreachable(root));
				}
				catch (RuntimeException e)
				{
					listener.onError("The backend sent an answer the plugin could not read.");
				}
			}
		});
		return call;
	}

	ChatReply parseReply(String text)
	{
		JsonObject json = gson.fromJson(text, JsonObject.class);
		List<ChatReply.Source> sources = new ArrayList<>();
		JsonElement array = json.get("sources");
		if (array != null && array.isJsonArray())
		{
			for (JsonElement element : (JsonArray) array)
			{
				JsonObject source = element.getAsJsonObject();
				sources.add(new ChatReply.Source(text(source, "title"), text(source, "url")));
			}
		}
		return new ChatReply(text(json, "answer"), text(json, "intent"), sources);
	}

	String parseError(int status, String retryAfterHeader, String text)
	{
		String code = null;
		String message = null;
		Double retryAfter = null;
		try
		{
			JsonObject json = gson.fromJson(text, JsonObject.class);
			if (json != null)
			{
				code = text(json, "error");
				message = text(json, "message");
				if (json.has("retry_after_s") && !json.get("retry_after_s").isJsonNull())
				{
					retryAfter = json.get("retry_after_s").getAsDouble();
				}
			}
		}
		catch (JsonSyntaxException | IllegalStateException e)
		{
			// not our typed error (a proxy page, for instance): the status alone has to do
		}
		if (retryAfter == null && retryAfterHeader != null)
		{
			try
			{
				retryAfter = Double.parseDouble(retryAfterHeader.trim());
			}
			catch (NumberFormatException e)
			{
				retryAfter = null;
			}
		}
		return ErrorMessages.http(status, code, message, retryAfter);
	}

	private static String text(JsonObject json, String name)
	{
		JsonElement element = json.get(name);
		return element == null || element.isJsonNull() ? null : element.getAsString();
	}
}
