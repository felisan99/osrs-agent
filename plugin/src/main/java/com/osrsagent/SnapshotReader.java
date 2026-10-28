package com.osrsagent;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import net.runelite.api.Client;
import net.runelite.api.Quest;
import net.runelite.api.Skill;
import net.runelite.api.WorldType;
import net.runelite.api.gameval.InventoryID;
import net.runelite.api.gameval.VarPlayerID;
import net.runelite.api.ItemContainer;
import net.runelite.api.vars.AccountType;

/**
 * Reads the character from the game client. Every method here touches the client API, so it must
 * only be called on the client thread (``ClientThread#invokeLater``): the plugin never reads the
 * client from the Swing or the HTTP threads.
 */
public final class SnapshotReader
{
	private final Client client;

	public SnapshotReader(Client client)
	{
		this.client = client;
	}

	@SuppressWarnings("deprecation")
	static String accountTypeName(AccountType type)
	{
		if (type == null)
		{
			return "unknown";
		}
		switch (type)
		{
			case NORMAL:
				return "main";
			case IRONMAN:
				return "ironman";
			case ULTIMATE_IRONMAN:
				return "ultimate_ironman";
			case HARDCORE_IRONMAN:
				return "hardcore_ironman";
			case GROUP_IRONMAN:
				return "group_ironman";
			case HARDCORE_GROUP_IRONMAN:
				return "hardcore_group_ironman";
			default:
				return "unknown";
		}
	}

	/**
	 * Membership cannot be read directly. A members world proves the account is a member; a free
	 * world proves nothing (a member can play there), so the answer is "unknown" and the backend
	 * asks the player when it matters (docs/verification.md §6).
	 */
	Boolean members()
	{
		return client.getWorldType().contains(WorldType.MEMBERS) ? Boolean.TRUE : null;
	}

	Map<String, Integer> skills()
	{
		Map<String, Integer> skills = new LinkedHashMap<>();
		for (Skill skill : Skill.values())
		{
			if (skill != Skill.OVERALL)
			{
				skills.put(skill.getName().toLowerCase(Locale.ROOT), client.getRealSkillLevel(skill));
			}
		}
		return skills;
	}

	Map<String, String> quests()
	{
		Map<String, String> quests = new LinkedHashMap<>();
		for (Quest quest : Quest.values())
		{
			try
			{
				quests.put(quest.getName(), quest.getState(client).name());
			}
			catch (RuntimeException e)
			{
				// One quest whose state cannot be read must not lose the whole snapshot: it is
				// left out, and the backend treats an unreported quest as unknown, never as not done.
			}
		}
		return quests;
	}

	List<ItemStack> inventory()
	{
		ItemContainer container = client.getItemContainer(InventoryID.INV);
		return container == null ? new ArrayList<>() : Stacks.from(container.getItems());
	}

	/** ``bank`` is what the plugin cached the last time the bank was open (or ``null``). */
	@SuppressWarnings("deprecation")
	public PlayerSnapshot read(BankSnapshot bank, Instant now)
	{
		List<ItemStack> inventory = inventory();
		long coins = Stacks.coins(inventory);
		if (bank != null)
		{
			coins += Stacks.coins(bank.getItems());
		}
		return new PlayerSnapshot(
			accountTypeName(client.getAccountType()),
			members(),
			skills(),
			quests(),
			client.getVarpValue(VarPlayerID.QP),
			coins,
			inventory,
			bank,
			now);
	}
}
