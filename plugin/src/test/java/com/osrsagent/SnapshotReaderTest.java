package com.osrsagent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.gson.JsonObject;
import java.time.Instant;
import java.util.Collections;
import java.util.EnumSet;
import net.runelite.api.Client;
import net.runelite.api.Item;
import net.runelite.api.ItemContainer;
import net.runelite.api.Skill;
import net.runelite.api.WorldType;
import net.runelite.api.gameval.InventoryID;
import net.runelite.api.gameval.VarPlayerID;
import net.runelite.api.vars.AccountType;
import org.junit.Before;
import org.junit.Test;

/** The reader against a mocked client: what the plugin sends, per SPEC §5. */
public class SnapshotReaderTest
{
	private static final Instant NOW = Instant.parse("2026-10-14T12:00:00Z");

	private Client client;
	private SnapshotReader reader;

	@Before
	public void setUp()
	{
		client = mock(Client.class);
		reader = new SnapshotReader(client);
		when(client.getRealSkillLevel(any())).thenAnswer(
			invocation -> invocation.getArgument(0) == Skill.ATTACK ? 75 : 1);
		when(client.getAccountType()).thenReturn(AccountType.NORMAL);
		when(client.getWorldType()).thenReturn(EnumSet.noneOf(WorldType.class));
		when(client.getVarpValue(VarPlayerID.QP)).thenReturn(150);
	}

	private void inventory(Item... items)
	{
		ItemContainer container = mock(ItemContainer.class);
		when(container.getItems()).thenReturn(items);
		when(client.getItemContainer(InventoryID.INV)).thenReturn(container);
	}

	@Test
	public void skillsAreRealLevelsKeyedByLowercaseName()
	{
		inventory();
		PlayerSnapshot snapshot = reader.read(null, NOW);
		assertEquals(Integer.valueOf(75), snapshot.getSkills().get("attack"));
		assertEquals(Integer.valueOf(1), snapshot.getSkills().get("runecraft"));
		assertTrue(snapshot.getSkills().containsKey("sailing"));
		assertFalse("overall is not a skill", snapshot.getSkills().containsKey("overall"));
		long real = java.util.Arrays.stream(Skill.values()).filter(skill -> skill != Skill.OVERALL).count();
		assertEquals(real, snapshot.getSkills().size());
	}

	@Test
	public void everyAccountTypeHasTheBackendsName()
	{
		assertEquals("main", SnapshotReader.accountTypeName(AccountType.NORMAL));
		assertEquals("ironman", SnapshotReader.accountTypeName(AccountType.IRONMAN));
		assertEquals("ultimate_ironman", SnapshotReader.accountTypeName(AccountType.ULTIMATE_IRONMAN));
		assertEquals("hardcore_ironman", SnapshotReader.accountTypeName(AccountType.HARDCORE_IRONMAN));
		assertEquals("group_ironman", SnapshotReader.accountTypeName(AccountType.GROUP_IRONMAN));
		assertEquals(
			"hardcore_group_ironman", SnapshotReader.accountTypeName(AccountType.HARDCORE_GROUP_IRONMAN));
		assertEquals("unknown", SnapshotReader.accountTypeName(null));
		assertEquals(AccountType.values().length, 6);
	}

	@Test
	public void membershipIsOnlyKnownOnAMembersWorld()
	{
		inventory();
		when(client.getWorldType()).thenReturn(EnumSet.of(WorldType.MEMBERS));
		assertEquals(Boolean.TRUE, reader.read(null, NOW).getMembers());
		when(client.getWorldType()).thenReturn(EnumSet.noneOf(WorldType.class));
		assertNull("a free world proves nothing: members can play there", reader.read(null, NOW).getMembers());
	}

	@Test
	public void gpIsCoinsPlusPlatinumTokensAtFaceValueInTheInventoryAndTheKnownBank()
	{
		inventory(new Item(995, 1_000), new Item(-1, 0), new Item(13204, 2), new Item(4151, 1));
		PlayerSnapshot withoutBank = reader.read(null, NOW);
		assertEquals(Long.valueOf(1_000 + 2 * 1_000), withoutBank.getGp());
		assertEquals("empty slots are not stacks", 3, withoutBank.toJson().getAsJsonArray("inventory").size());

		BankSnapshot bank = new BankSnapshot(
			NOW.minusSeconds(600),
			java.util.Arrays.asList(new ItemStack(995, 5_000), new ItemStack(13204, 1), new ItemStack(560, 99)));
		PlayerSnapshot withBank = reader.read(bank, NOW);
		assertEquals(Long.valueOf(3_000 + 5_000 + 1_000), withBank.getGp());
	}

	@Test
	public void bigStacksDoNotOverflow()
	{
		inventory(new Item(995, Integer.MAX_VALUE), new Item(13204, Integer.MAX_VALUE));
		long expected = (long) Integer.MAX_VALUE + (long) Integer.MAX_VALUE * 1_000L;
		assertEquals(Long.valueOf(expected), reader.read(null, NOW).getGp());
	}

	@Test
	public void anInventoryThatIsNotLoadedYetIsEmpty()
	{
		when(client.getItemContainer(InventoryID.INV)).thenReturn(null);
		assertEquals(0, reader.read(null, NOW).toJson().getAsJsonArray("inventory").size());
	}

	@Test
	public void theJsonFollowsTheBackendContract()
	{
		inventory(new Item(4151, 1));
		when(client.getWorldType()).thenReturn(EnumSet.of(WorldType.MEMBERS));
		BankSnapshot bank = new BankSnapshot(NOW.minusSeconds(60), Collections.singletonList(new ItemStack(995, 10)));
		JsonObject json = reader.read(bank, NOW).toJson();
		assertEquals("main", json.get("account_type").getAsString());
		assertTrue(json.get("members").getAsBoolean());
		assertEquals(75, json.getAsJsonObject("skills").get("attack").getAsInt());
		assertEquals(150, json.get("quest_points").getAsInt());
		assertEquals("2026-10-14T12:00:00Z", json.get("captured_at").getAsString());
		JsonObject stack = json.getAsJsonArray("inventory").get(0).getAsJsonObject();
		assertEquals(4151, stack.get("id").getAsInt());
		assertEquals(1, stack.get("qty").getAsInt());
		JsonObject jsonBank = json.getAsJsonObject("bank");
		assertEquals("2026-10-14T11:59:00Z", jsonBank.get("captured_at").getAsString());
		assertEquals(10, jsonBank.getAsJsonArray("items").get(0).getAsJsonObject().get("qty").getAsInt());
		assertTrue(json.has("quests"));
	}

	@Test
	public void unknownMembershipAndNoBankAreOmittedNotGuessed()
	{
		inventory();
		JsonObject json = reader.read(null, NOW).toJson();
		assertFalse(json.has("members"));
		assertFalse(json.has("bank"));
	}
}
