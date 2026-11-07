"""QUMO agents in Mojo: a native port of the minimal deepagents harness (virtual file system,
large-result eviction, subagents, summarization, tool-call patching), the QUMO agent tools,
a Mojo toolsmith that builds new tools, and the `qumo-agent` command line. No Python
or external helpers at run time. The design is in agents/mojo/DESIGN.md.

Subpackages: native (libc, spawn, fs, sandbox), util, core (messages, replay, errors), net
(HTTP and TLS), llm (OpenAI-compatible client), mcp (stdio and HTTP clients, stdio server),
harness (the agent loop and its middleware), tools (QUMO, repository and research tools),
toolsmith (generated tools: manifest, build, review) and app (CLI, serve, call, doctor).
`tool_sdk` is the library compiled into every generated tool.
"""
