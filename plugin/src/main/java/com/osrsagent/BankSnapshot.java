package com.osrsagent;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import net.runelite.api.Item;

/**
 * The bank as it was the last time the player had it open. The game only sends the bank's contents
 * while it is open, so the plugin keeps the last copy together with when it was taken (SPEC §5).
 */
public final class BankSnapshot
{
	private final Instant capturedAt;
	private final List<ItemStack> items;

	public BankSnapshot(Instant capturedAt, List<ItemStack> items)
	{
		this.capturedAt = capturedAt;
		this.items = Collections.unmodifiableList(new ArrayList<>(items));
	}

	public static BankSnapshot of(Item[] raw, Instant capturedAt)
	{
		return new BankSnapshot(capturedAt, Stacks.from(raw));
	}

	public Instant getCapturedAt()
	{
		return capturedAt;
	}

	public List<ItemStack> getItems()
	{
		return items;
	}

	JsonObject toJson()
	{
		JsonArray array = new JsonArray();
		items.forEach(stack -> array.add(stack.toJson()));
		JsonObject json = new JsonObject();
		json.addProperty("captured_at", capturedAt.toString());
		json.add("items", array);
		return json;
	}
}
