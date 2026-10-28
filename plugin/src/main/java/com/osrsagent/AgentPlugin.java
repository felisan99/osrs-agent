package com.osrsagent;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.inject.Provides;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.time.Instant;
import java.util.UUID;
import javax.inject.Inject;
import javax.swing.SwingUtilities;
import lombok.extern.slf4j.Slf4j;
import net.runelite.api.Client;
import net.runelite.api.GameState;
import net.runelite.api.events.GameStateChanged;
import net.runelite.api.events.ItemContainerChanged;
import net.runelite.api.gameval.InventoryID;
import net.runelite.client.config.ConfigManager;
import net.runelite.client.eventbus.Subscribe;
import net.runelite.client.events.ConfigChanged;
import net.runelite.client.plugins.Plugin;
import net.runelite.client.plugins.PluginDescriptor;
import net.runelite.client.callback.ClientThread;
import net.runelite.client.ui.ClientToolbar;
import net.runelite.client.ui.NavigationButton;
import okhttp3.OkHttpClient;

/**
 * OSRS Agent: a chat panel that answers Old School RuneScape questions using the player's own
 * character data. It only READS the game state and only when the player asks; there are no
 * overlays, no menu entries and nothing is ever sent to the game (SPEC §1, §10).
 *
 * Threads: the Swing panel runs on the EDT, every access to the game client happens on the client
 * thread (``ClientThread#invokeLater``), and the HTTP request runs on OkHttp's threads.
 */
@Slf4j
@PluginDescriptor(
	name = "OSRS Agent",
	description = "Ask an assistant about quests, money making, flipping and game mechanics, "
		+ "answered with your own character's data",
	tags = {"assistant", "ai", "wiki", "prices", "quests", "flipping"}
)
public class AgentPlugin extends Plugin
{
	@Inject
	private Client client;

	@Inject
	private ClientThread clientThread;

	@Inject
	private ClientToolbar clientToolbar;

	@Inject
	private AgentConfig config;

	@Inject
	private OkHttpClient okHttpClient;

	@Inject
	private Gson gson;

	private AgentApi api;
	private SnapshotReader reader;
	private AgentPanel panel;
	private NavigationButton navigationButton;

	/** A conversation id: the backend's memory of this chat is keyed by it. */
	private volatile String threadId = newThreadId();

	/** The bank as last seen open; written on the client thread, read from it too. */
	private BankSnapshot bank;

	@Provides
	AgentConfig provideConfig(ConfigManager configManager)
	{
		return configManager.getConfig(AgentConfig.class);
	}

	@Override
	protected void startUp()
	{
		api = new AgentApi(okHttpClient, gson);
		reader = new SnapshotReader(client);
		panel = new AgentPanel(config, this::ask, this::newConversation);
		navigationButton = NavigationButton.builder()
			.tooltip("OSRS Agent")
			.icon(icon())
			.priority(7)
			.panel(panel)
			.build();
		clientToolbar.addNavigation(navigationButton);
		log.debug("OSRS Agent started");
	}

	@Override
	protected void shutDown()
	{
		clientToolbar.removeNavigation(navigationButton);
		bank = null;
		log.debug("OSRS Agent stopped");
	}

	private static String newThreadId()
	{
		return UUID.randomUUID().toString();
	}

	private static BufferedImage icon()
	{
		BufferedImage image = new BufferedImage(16, 16, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g = image.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g.setColor(new Color(0xFF, 0x98, 0x1F));
		g.fillRoundRect(1, 1, 14, 12, 4, 4);
		g.fillPolygon(new int[] {4, 4, 9}, new int[] {12, 15, 12}, 3);
		g.setColor(new Color(0x2B, 0x2B, 0x2B));
		g.fillOval(4, 5, 2, 2);
		g.fillOval(10, 5, 2, 2);
		g.dispose();
		return image;
	}

	@Subscribe
	public void onConfigChanged(ConfigChanged event)
	{
		boolean look = "textSize".equals(event.getKey()) || "fontStyle".equals(event.getKey());
		if (AgentConfig.GROUP.equals(event.getGroup()) && look && panel != null)
		{
			SwingUtilities.invokeLater(panel::restyle);
		}
	}

	@Subscribe
	public void onItemContainerChanged(ItemContainerChanged event)
	{
		if (event.getContainerId() == InventoryID.BANK)
		{
			bank = BankSnapshot.of(event.getItemContainer().getItems(), Instant.now());
		}
	}

	@Subscribe
	public void onGameStateChanged(GameStateChanged event)
	{
		if (event.getGameState() == GameState.LOGIN_SCREEN)
		{
			bank = null; // another account may log in: never carry one player's bank to another
		}
	}

	/** Called on the Swing thread when the player sends a message. */
	void ask(String text)
	{
		String key = config.openRouterKey() == null ? "" : config.openRouterKey().trim();
		if (key.isEmpty())
		{
			panel.appendError(ErrorMessages.MISSING_KEY);
			return;
		}
		String baseUrl = config.backendUrl();
		int timeout = config.timeoutSeconds();
		String thread = threadId;
		panel.appendUser(text);
		panel.setBusy(true);
		// The snapshot needs the game client, so it is taken on the client thread.
		clientThread.invokeLater(() ->
		{
			JsonObject snapshot = null;
			if (client.getGameState() == GameState.LOGGED_IN)
			{
				try
				{
					snapshot = reader.read(bank, Instant.now()).toJson();
				}
				catch (RuntimeException e)
				{
					log.warn("Could not read the character snapshot", e);
				}
			}
			if (snapshot == null)
			{
				SwingUtilities.invokeLater(() -> panel.appendNote(
					"Not logged in (or the character could not be read): answering without your character data."));
			}
			api.chat(baseUrl, key, thread, text, snapshot, timeout, new AgentApi.Listener()
			{
				@Override
				public void onReply(ChatReply reply)
				{
					SwingUtilities.invokeLater(() ->
					{
						panel.appendAssistant(reply);
						panel.setBusy(false);
					});
				}

				@Override
				public void onError(String message)
				{
					SwingUtilities.invokeLater(() ->
					{
						panel.appendError(message);
						panel.setBusy(false);
					});
				}
			});
		});
	}

	void newConversation()
	{
		threadId = newThreadId();
		panel.clear();
	}
}
