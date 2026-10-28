package com.osrsagent;

import java.util.ArrayList;
import java.util.List;
import net.runelite.api.Item;

/** Turns the game's item arrays (which include empty slots) into plain stacks. */
final class Stacks
{
	static final int COINS = 995;
	static final int PLATINUM_TOKEN = 13204;
	static final long PLATINUM_TOKEN_VALUE = 1000;

	private Stacks()
	{
	}

	static List<ItemStack> from(Item[] raw)
	{
		List<ItemStack> stacks = new ArrayList<>();
		if (raw == null)
		{
			return stacks;
		}
		for (Item item : raw)
		{
			if (item != null && item.getId() > 0 && item.getQuantity() > 0)
			{
				stacks.add(new ItemStack(item.getId(), item.getQuantity()));
			}
		}
		return stacks;
	}

	/** Coins plus platinum tokens at face value (GE trades above 2.1B settle in tokens). */
	static long coins(List<ItemStack> stacks)
	{
		long total = 0;
		for (ItemStack stack : stacks)
		{
			if (stack.getId() == COINS)
			{
				total += stack.getQuantity();
			}
			else if (stack.getId() == PLATINUM_TOKEN)
			{
				total += stack.getQuantity() * PLATINUM_TOKEN_VALUE;
			}
		}
		return total;
	}
}
