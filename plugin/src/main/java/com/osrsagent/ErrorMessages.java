package com.osrsagent;

/**
 * What the player reads when something goes wrong. The backend's typed errors (SPEC §4) and the
 * ways a request can fail before it gets one are turned into a sentence that says what to do.
 */
final class ErrorMessages
{
	static final String MISSING_KEY =
		"Set your OpenRouter API key in the plugin settings (wrench icon), then ask again.";

	private ErrorMessages()
	{
	}

	static String http(int status, String code, String message, Double retryAfterSeconds)
	{
		String detail = message == null || message.isEmpty() ? "" : " (" + message + ")";
		switch (code == null ? "" : code)
		{
			case "invalid_key":
				return "OpenRouter did not accept your API key. Check it in the plugin settings." + detail;
			case "insufficient_credits":
				return "Your OpenRouter account has no credits left for this request. Add credits at "
					+ "openrouter.ai and ask again." + detail;
			case "rate_limited":
				return "OpenRouter is rate limiting you. Try again in "
					+ (retryAfterSeconds == null ? "a moment" : Math.max(1, Math.round(retryAfterSeconds)) + " seconds")
					+ ".";
			case "upstream_error":
				return "OpenRouter or a data service failed while answering. Try again in a moment." + detail;
			case "payload_too_large":
				return "The question or your character data was too large to send.";
			case "invalid_request":
				return "The backend rejected the request: " + message;
			case "unavailable":
				return "The backend is running but its database is not reachable.";
			default:
				break;
		}
		if (status == 404)
		{
			return "The backend URL answered 404. Check the Backend URL in the plugin settings.";
		}
		return "The backend returned an unexpected error (HTTP " + status + ")" + detail + ".";
	}

	static String unreachable(String baseUrl)
	{
		return "Could not reach the backend at " + baseUrl + ". Is it running? "
			+ "Check the Backend URL in the plugin settings.";
	}

	static String timeout(int seconds)
	{
		return "The backend did not answer within " + seconds + " seconds.";
	}

	static String badUrl(String baseUrl)
	{
		return "The Backend URL in the plugin settings is not valid: " + baseUrl;
	}
}
