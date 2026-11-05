"""QUMO agents in Mojo: native executables that run the qumo-operator and execution
agents without Python, talk to QUMO's MCP server as a client, and serve themselves
over MCP so other agents (or any MCP client) can call them like a library.

Modules: proc (child processes over pipes), mcp_client, llm (OpenRouter through curl,
plus an offline replay provider), config (agents.json, prompts, specs, tool guides),
agent (the tool-calling loop), server (the agent as an MCP stdio server).
"""
