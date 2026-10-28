package com.osrsagent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.inject.AbstractModule;
import com.google.inject.Guice;
import java.io.IOException;
import java.util.EnumSet;
import java.util.concurrent.TimeUnit;
import javax.swing.SwingUtilities;
import net.runelite.api.Client;
import net.runelite.api.GameState;
import net.runelite.api.Item;
import net.runelite.api.ItemContainer;
import net.runelite.api.Skill;
import net.runelite.api.WorldType;
import net.runelite.api.events.GameStateChanged;
import net.runelite.api.events.ItemContainerChanged;
import net.runelite.api.gameval.InventoryID;
import net.runelite.api.gameval.VarPlayerID;
import net.runelite.api.vars.AccountType;
import net.runelite.client.callback.ClientThread;
import net.runelite.client.config.ConfigManager;
import net.runelite.client.events.ConfigChanged;
import net.runelite.client.ui.ClientToolbar;
import net.runelite.client.ui.NavigationButton;
import okhttp3.OkHttpClient;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * The plugin as RuneLite would run it, minus the game: Guice wires it, the panel takes a question,
 * the snapshot is read on the (immediate) client thread, and the HTTP request reaches a local
 * server. It catches what only shows at load time: missing injections, threading slips, and what
 * exactly goes over the wire.
 */
public class AgentPluginFlowTest
{
	private static final String KEY = "sk-or-v1-FLOWTESTKEY0123456789";
	private static final Gson GSON = new Gson();
	private static final String REPLY =
		"{\"answer\":\"Te falta **Slayer**.\",\"intent\":\"quest\",\"sources\":[{\"title\":\"Monkey Madness II\",\"url\":\"https://x\"}]}";

	private MockWebServer server;
	private Client client;
	private AgentConfig config;
	private ClientToolbar toolbar;
	private AgentPlugin plugin;

	@Before
	public void setUp() throws IOException
	{
		server = new MockWebServer();
		server.start();

		client = mock(Client.class);
		when(client.getGameState()).thenReturn(GameState.LOGGED_IN);
		when(client.getRealSkillLevel(any())).thenAnswer(i -> i.getArgument(0) == Skill.SLAYER ? 66 : 70);
		when(client.getAccountType()).thenReturn(AccountType.GROUP_IRONMAN);
		when(client.getWorldType()).thenReturn(EnumSet.of(WorldType.MEMBERS));
		when(client.getVarpValue(VarPlayerID.QP)).thenReturn(150);
		container(InventoryID.INV, new Item(995, 2_000), new Item(4151, 1));

		ClientThread clientThread = mock(ClientThread.class);
		doAnswer(invocation ->
		{
			((Runnable) invocation.getArgument(0)).run(); // the "client thread" runs at once
			return null;
		}).when(clientThread).invokeLater(any(Runnable.class));

		config = mock(AgentConfig.class);
		when(config.openRouterKey()).thenReturn("  " + KEY + "  ");
		when(config.backendUrl()).thenReturn(server.url("/").toString());
		when(config.timeoutSeconds()).thenReturn(20);
		toolbar = mock(ClientToolbar.class);

		plugin = Guice.createInjector(new AbstractModule()
		{
			@Override
			protected void configure()
			{
				bind(Client.class).toInstance(client);
				bind(ClientThread.class).toInstance(clientThread);
				bind(ClientToolbar.class).toInstance(toolbar);
				bind(AgentConfig.class).toInstance(config);
				bind(OkHttpClient.class).toInstance(new OkHttpClient());
				bind(Gson.class).toInstance(GSON);
			}
		}).getInstance(AgentPlugin.class);
		plugin.startUp();
	}

	@After
	public void tearDown() throws IOException
	{
		plugin.shutDown();
		server.shutdown();
	}

	private ItemContainer container(int id, Item... items)
	{
		ItemContainer container = mock(ItemContainer.class);
		when(container.getItems()).thenReturn(items);
		when(client.getItemContainer(id)).thenReturn(container);
		return container;
	}

	private AgentPanel panel()
	{
		org.mockito.ArgumentCaptor<NavigationButton> captor = org.mockito.ArgumentCaptor.forClass(NavigationButton.class);
		verify(toolbar).addNavigation(captor.capture());
		return (AgentPanel) captor.getValue().getPanel();
	}

	private String text(AgentPanel panel) throws Exception
	{
		String[] text = new String[1];
		SwingUtilities.invokeAndWait(() -> text[0] = panel.historyText());
		return text[0];
	}

	private void waitFor(AgentPanel panel, String needle) throws Exception
	{
		long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15);
		while (System.nanoTime() < deadline)
		{
			if (text(panel).contains(needle))
			{
				return;
			}
			Thread.sleep(25);
		}
		throw new AssertionError("never saw '" + needle + "' in: " + text(panel));
	}

	private JsonObject lastRequestBody() throws Exception
	{
		RecordedRequest request = server.takeRequest(10, TimeUnit.SECONDS);
		assertNotNull("no request reached the backend", request);
		assertEquals("/v1/chat", request.getPath());
		assertEquals("the key is trimmed and sent in the header", KEY, request.getHeader("X-OpenRouter-Key"));
		String body = request.getBody().readUtf8();
		assertFalse("the key never goes in the body", body.contains("sk-or"));
		return GSON.fromJson(body, JsonObject.class);
	}

	@Test
	public void changingTheTextSizeInTheSettingsRestylesThePanelAtOnce() throws Exception
	{
		AgentPanel panel = panel();
		when(config.textSize()).thenReturn(26);
		ConfigChanged other = new ConfigChanged();
		other.setGroup("somethingelse");
		other.setKey("textSize");
		plugin.onConfigChanged(other); // another plugin's setting: ignored
		ConfigChanged ours = new ConfigChanged();
		ours.setGroup(AgentConfig.GROUP);
		ours.setKey("textSize");
		plugin.onConfigChanged(ours);
		SwingUtilities.invokeAndWait(() ->
		{
		});
		assertEquals(26, panel.historyView().getFont().getSize());
	}

	@Test
	public void aQuestionCarriesTheCharactersSnapshotAndTheAnswerAppearsInThePanel() throws Exception
	{
		server.enqueue(new MockResponse().setBody(REPLY).setHeader("Content-Type", "application/json"));
		AgentPanel panel = panel();
		plugin.ask("What do I need to start Monkey Madness II?");

		JsonObject body = lastRequestBody();
		assertEquals("What do I need to start Monkey Madness II?", body.get("message").getAsString());
		JsonObject state = body.getAsJsonObject("player_state");
		assertEquals("group_ironman", state.get("account_type").getAsString());
		assertTrue(state.get("members").getAsBoolean());
		assertEquals(66, state.getAsJsonObject("skills").get("slayer").getAsInt());
		assertEquals(150, state.get("quest_points").getAsInt());
		assertEquals(2_000, state.get("gp").getAsInt());
		assertEquals(2, state.getAsJsonArray("inventory").size());
		assertFalse("no bank was ever opened", state.has("bank"));

		waitFor(panel, "Te falta Slayer.");
		String shown = text(panel);
		assertTrue(shown.contains("What do I need to start Monkey Madness II?"));
		assertTrue(shown.contains("Sources: Monkey Madness II"));
		assertFalse("markdown markers are rendered, not shown", shown.contains("**"));
	}

	@Test
	public void withoutAKeyNothingIsSentAndThePlayerIsToldWhatToDo() throws Exception
	{
		when(config.openRouterKey()).thenReturn("   ");
		AgentPanel panel = panel();
		plugin.ask("hola");
		assertTrue(text(panel).contains("Set your OpenRouter API key"));
		assertEquals(0, server.getRequestCount());
	}

	@Test
	public void beforeLoginTheQuestionStillGoesButWithoutPlayerData() throws Exception
	{
		when(client.getGameState()).thenReturn(GameState.LOGIN_SCREEN);
		server.enqueue(new MockResponse().setBody(REPLY).setHeader("Content-Type", "application/json"));
		AgentPanel panel = panel();
		plugin.ask("What is a whip?");
		JsonObject body = lastRequestBody();
		assertFalse(body.has("player_state"));
		waitFor(panel, "Te falta Slayer.");
		assertTrue(text(panel).contains("Not logged in"));
	}

	@Test
	public void theBankIsCachedWhenItOpensAndForgottenAtTheLoginScreen() throws Exception
	{
		server.enqueue(new MockResponse().setBody(REPLY).setHeader("Content-Type", "application/json"));
		server.enqueue(new MockResponse().setBody(REPLY).setHeader("Content-Type", "application/json"));
		ItemContainer bank = container(InventoryID.BANK, new Item(995, 5_000), new Item(560, 300), new Item(-1, 0));
		plugin.onItemContainerChanged(new ItemContainerChanged(InventoryID.BANK, bank));

		AgentPanel panel = panel();
		plugin.ask("first");
		JsonObject state = lastRequestBody().getAsJsonObject("player_state");
		assertEquals("inventory + bank coins", 7_000, state.get("gp").getAsInt());
		JsonObject sentBank = state.getAsJsonObject("bank");
		assertNotNull(sentBank.get("captured_at"));
		assertEquals("empty slots are dropped", 2, sentBank.getAsJsonArray("items").size());
		waitFor(panel, "Te falta Slayer.");

		plugin.onGameStateChanged(new GameStateChanged());
		GameStateChanged logout = new GameStateChanged();
		logout.setGameState(GameState.LOGIN_SCREEN);
		plugin.onGameStateChanged(logout);
		plugin.ask("second");
		JsonObject next = lastRequestBody().getAsJsonObject("player_state");
		assertFalse("another account must never inherit this bank", next.has("bank"));
		assertEquals(2_000, next.get("gp").getAsInt());
	}

	@Test
	public void aConversationKeepsItsThreadUntilANewOneIsStarted() throws Exception
	{
		for (int i = 0; i < 3; i++)
		{
			server.enqueue(new MockResponse().setBody(REPLY).setHeader("Content-Type", "application/json"));
		}
		AgentPanel panel = panel();
		plugin.ask("one");
		String first = lastRequestBody().get("thread_id").getAsString();
		waitFor(panel, "Te falta Slayer.");
		plugin.ask("two");
		String second = lastRequestBody().get("thread_id").getAsString();
		assertEquals(first, second);

		long answers = 0;
		while (answers < 2)
		{
			Thread.sleep(25);
			String shown = text(panel);
			answers = shown.split("Te falta Slayer\\.", -1).length - 1;
		}
		SwingUtilities.invokeAndWait(plugin::newConversation);
		assertFalse("the panel starts clean", text(panel).contains("Te falta Slayer."));
		plugin.ask("three");
		assertNotEquals(first, lastRequestBody().get("thread_id").getAsString());
	}

	@Test
	public void backendErrorsAreShownAsReadableMessages() throws Exception
	{
		server.enqueue(new MockResponse().setResponseCode(402)
			.setBody("{\"error\":\"insufficient_credits\",\"message\":\"Insufficient credits\"}"));
		AgentPanel panel = panel();
		plugin.ask("hola");
		lastRequestBody();
		waitFor(panel, "no credits left");
	}

	@Test
	public void theConfigProviderAsksRuneLiteForTheConfig()
	{
		ConfigManager manager = mock(ConfigManager.class);
		when(manager.getConfig(AgentConfig.class)).thenReturn(config);
		assertEquals(config, plugin.provideConfig(manager));
	}

	@Test
	public void theKeyFieldIsASecretInTheSettings() throws Exception
	{
		net.runelite.client.config.ConfigItem item = AgentConfig.class.getMethod("openRouterKey")
			.getAnnotation(net.runelite.client.config.ConfigItem.class);
		assertTrue("RuneLite must render the key as a password field", item.secret());
		assertEquals("http://localhost:8000", new AgentConfig() { }.backendUrl());
	}
}
