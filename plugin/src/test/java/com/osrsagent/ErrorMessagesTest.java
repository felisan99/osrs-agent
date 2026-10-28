package com.osrsagent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class ErrorMessagesTest
{
	@Test
	public void eachTypedErrorOfTheContractSaysWhatToDo()
	{
		assertTrue(ErrorMessages.http(401, "invalid_key", "User not found.", null).contains("API key"));
		assertTrue(ErrorMessages.http(402, "insufficient_credits", "", null).contains("credits"));
		assertTrue(ErrorMessages.http(502, "upstream_error", "boom", null).contains("Try again"));
		assertTrue(ErrorMessages.http(413, "payload_too_large", "", null).contains("too large"));
		assertTrue(ErrorMessages.http(422, "invalid_request", "bad field", null).contains("bad field"));
		assertTrue(ErrorMessages.http(503, "unavailable", "", null).contains("database"));
	}

	@Test
	public void rateLimitsTellWhenToComeBack()
	{
		assertTrue(ErrorMessages.http(429, "rate_limited", "", 12.0).contains("12 seconds"));
		assertTrue(ErrorMessages.http(429, "rate_limited", "", 0.2).contains("1 seconds"));
		assertTrue(ErrorMessages.http(429, "rate_limited", "", null).contains("a moment"));
	}

	@Test
	public void aWrongUrlAndUnknownStatusesAreExplained()
	{
		assertTrue(ErrorMessages.http(404, null, null, null).contains("Backend URL"));
		String other = ErrorMessages.http(500, "internal_error", "Unexpected", null);
		assertTrue(other.contains("HTTP 500"));
	}

	@Test
	public void transportFailuresNameTheUrlAndTheTimeout()
	{
		assertTrue(ErrorMessages.unreachable("http://localhost:8000").contains("http://localhost:8000"));
		assertEquals("The backend did not answer within 30 seconds.", ErrorMessages.timeout(30));
		assertTrue(ErrorMessages.badUrl("nonsense").contains("nonsense"));
	}

	@Test
	public void noMessageEverEchoesAKey()
	{
		// The messages are built from the backend's own text, which never contains the key.
		assertFalse(ErrorMessages.MISSING_KEY.contains("sk-or"));
	}
}
