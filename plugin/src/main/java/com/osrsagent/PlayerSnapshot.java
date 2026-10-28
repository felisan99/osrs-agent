package com.osrsagent;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.time.Instant;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A compact snapshot of the character, sent with every question (SPEC §5). The backend keeps
 * nothing about the player between requests, so everything it may need travels here.
 */
public final class PlayerSnapshot
{
	private final String accountType;
	private final Boolean members;
	private final Map<String, Integer> skills;
	private final Map<String, String> quests;
	private final Integer questPoints;
	private final Long gp;
	private final List<ItemStack> inventory;
	private final BankSnapshot bank;
	private final Instant capturedAt;

	public PlayerSnapshot(
		String accountType,
		Boolean members,
		Map<String, Integer> skills,
		Map<String, String> quests,
		Integer questPoints,
		Long gp,
		List<ItemStack> inventory,
		BankSnapshot bank,
		Instant capturedAt)
	{
		this.accountType = accountType;
		this.members = members;
		this.skills = Collections.unmodifiableMap(new LinkedHashMap<>(skills));
		this.quests = Collections.unmodifiableMap(new LinkedHashMap<>(quests));
		this.questPoints = questPoints;
		this.gp = gp;
		this.inventory = inventory;
		this.bank = bank;
		this.capturedAt = capturedAt;
	}

	public String getAccountType()
	{
		return accountType;
	}

	public Boolean getMembers()
	{
		return members;
	}

	public Map<String, Integer> getSkills()
	{
		return skills;
	}

	public Map<String, String> getQuests()
	{
		return quests;
	}

	public Long getGp()
	{
		return gp;
	}

	public BankSnapshot getBank()
	{
		return bank;
	}

	/** The JSON of SPEC §5; ``members`` is omitted when unknown (the backend then asks). */
	public JsonObject toJson()
	{
		JsonObject json = new JsonObject();
		json.addProperty("account_type", accountType);
		if (members != null)
		{
			json.addProperty("members", members);
		}
		JsonObject skillLevels = new JsonObject();
		skills.forEach(skillLevels::addProperty);
		json.add("skills", skillLevels);
		JsonObject questStates = new JsonObject();
		quests.forEach(questStates::addProperty);
		json.add("quests", questStates);
		if (questPoints != null)
		{
			json.addProperty("quest_points", questPoints);
		}
		if (gp != null)
		{
			json.addProperty("gp", gp);
		}
		JsonArray items = new JsonArray();
		inventory.forEach(stack -> items.add(stack.toJson()));
		json.add("inventory", items);
		if (bank != null)
		{
			json.add("bank", bank.toJson());
		}
		json.addProperty("captured_at", capturedAt.toString());
		return json;
	}
}
