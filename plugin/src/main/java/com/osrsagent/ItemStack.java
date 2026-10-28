package com.osrsagent;

import com.google.gson.JsonObject;

/** One stack of items: the backend resolves ids to names with the Wiki prices mapping. */
public final class ItemStack
{
	private final int id;
	private final long quantity;

	public ItemStack(int id, long quantity)
	{
		this.id = id;
		this.quantity = quantity;
	}

	public int getId()
	{
		return id;
	}

	public long getQuantity()
	{
		return quantity;
	}

	JsonObject toJson()
	{
		JsonObject json = new JsonObject();
		json.addProperty("id", id);
		json.addProperty("qty", quantity);
		return json;
	}
}
