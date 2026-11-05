# QUMO agents in Mojo (experiment)

`qumo-operator` and `execution` as one native executable, `build/qumo-agent`, with no
Python at run time. The other seven agents stay in Python (`bin/qumo-agents`): they
need alphaXiv, web research, planning and long contexts.

```bash
make agents-mojo                      # build/qumo-agent (about 400 KB; needs build/qumo)
build/qumo-agent list
build/qumo-agent run qumo-operator "Build a 3-qubit GHZ, simulate 1000 shots with seed 7, check the counts"
build/qumo-agent serve qumo-operator  # MCP server on stdio with one tool, run {task}
build/qumo-agent call qumo-operator "..."   # start `serve` as a child and call it like a library
make test-agents-mojo                 # 6 offline tests
```

## How it works

* **One source of truth.** Prompts, per-agent tool subsets and MCP tool guides are read at
  run time from `agents/prompts/`, which `python -m qumo_agents export-prompts` writes
  from the Python code; a Python test fails when they are stale. Models come from
  `agents/agents.json`.
* **Processes and MCP.** Mojo 1.2 has no subprocess pipes in `std.subprocess`, but
  `std.os.process` has `Pipe` and `Process`. A child gets its stdin/stdout through
  `/bin/sh -c 'exec "$0" "$@" 0<&R 1>&W ...'`, where only the two fd numbers are in the
  script and the program and its arguments are positional parameters. This keeps a
  persistent MCP session with `build/qumo mcp`, and lets one agent call another.
* **HTTPS.** There is no TLS in Mojo's standard library. Requests go through
  `/usr/bin/curl` started without a shell; the key is read by curl from the
  environment (`--variable %OPENROUTER_API_KEY --expand-header ...`), so it is never in
  argv, logs or files. Request and response bodies live in a 0700 temp directory that
  is removed after each call.

## Same task, three ways

Task: find and fix QM-E521 in a broken OpenQASM program, simulate a GHZ state with 2000
shots and seed 11, and check the counts (`anthropic/claude-sonnet-5.5`, 2026-11-05).

| | Mojo `run` | Mojo `call` (child over MCP) | Python deepagents |
|---|---|---|---|
| Wall time | 9.9 s | 10.9 s | 11.8 to 13.4 s |
| CPU (user + sys) | 0.08 s | 0.10 s | 1.90 s |
| Model turns / tool calls | 4 / 4 | 5 / 4 | 5 / 4 |
| Prompt tokens | 66,384 | 82,739 | 100,594 |
| Result | QM-E521 fixed, only 000/111 | same | same |

The model dominates wall time. The gains are CPU, start-up and fewer prompt tokens
(deepagents adds its own base prompt and built-in tools). One run each: not a benchmark.

## Limits

* **Mojo 1.2 argv bug.** `Process.run` loses short or non-NUL-terminated argument
  strings (`["ab", "lit"]` reached the child as `lit lit`). `proc.spawn()` copies each
  argument into a NUL-terminated heap buffer. The fix relies on how copies share that
  buffer, an implementation detail; `test_proc` catches a regression.
* **Not fully self-contained:** the binary needs the Mojo runtime libraries (like
  `build/qumo`), not Python.
* **No read timeout on MCP children:** a hung child blocks the caller.
* **Not ported:** deepagents' planning, virtual files and context summarisation; the
  Claude Code and Codex back ends; `write_report` was not exercised in a live run.
