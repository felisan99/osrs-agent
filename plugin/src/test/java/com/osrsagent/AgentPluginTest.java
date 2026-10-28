package com.osrsagent;

import net.runelite.client.RuneLite;
import net.runelite.client.externalplugins.ExternalPluginManager;

/** ``./gradlew run`` starts the RuneLite client in developer mode with this plugin loaded. */
public class AgentPluginTest
{
	public static void main(String[] args) throws Exception
	{
		ExternalPluginManager.loadBuiltin(AgentPlugin.class);
		RuneLite.main(args);
	}
}
