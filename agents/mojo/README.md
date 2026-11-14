# QUMO agents in Mojo (native harness)

All ten QUMO agents as one native executable, `build/qumo-agent`, with no Python, no
`curl` and no shell at run time. It is a minimal port of [deepagents](https://github.com/langchain-ai/deepagents)
to Mojo (the virtual file system, large-result eviction, subagents, summarization and
tool-call patching), plus QUMO's own tools, an HTTPS and MCP client written on libc and
OpenSSL, and a tenth agent, `mojo-toolsmith`, that builds new tools in Mojo for the other
agents. The design is in [DESIGN.md](DESIGN.md); the Python agents (`bin/qumo-agents`)
stay as they are.

```bash
make cli                              # build/qumo: the MCP server the agents call
make agents-mojo                      # build/qumo-agent
export OPENROUTER_API_KEY=...         # the default back end; any OpenAI-compatible endpoint works
build/qumo-agent doctor               # what is ready, what is missing
build/qumo-agent list
build/qumo-agent run qumo-operator "Build a 3-qubit GHZ, simulate 1000 shots with seed 7, check the counts"
build/qumo-agent serve qumo-operator  # MCP server on stdio with one tool, run {task}
build/qumo-agent call qumo-operator "..."   # start `serve` as a child and call it like a library
make test-agents-mojo                 # offline tests (replayed model, real build/qumo)
make agents-mojo-live                 # live checks: QUMO_LIVE=1 plus keys; each test SKIPs without them
```

## Commands

| Command | Does |
|---|---|
| `list [--json]` | The agents, their model, delegation, modules, alphaXiv status, generated tools |
| `run AGENT TASK` | Run one agent; the answer goes to stdout, statistics and logs to stderr |
| `serve AGENT` | The agent as an MCP stdio server with one tool, `run {task}` |
| `call AGENT TASK` | Start `serve` as a child process and call it over MCP: executable to executable |
| `tools list / show / review / revoke / remove / verify` | Manage generated tools (see below) |
| `doctor [--offline]` | One line per check: `[ok]` or `[--]` with a hint. Key names, never values |
| `reports [--show TEXT]` | Saved reports, newest first |

`TASK` `-` reads the task from stdin. Options of `run`: `--module M`, `--model ID`,
`--replay FILE` (an offline model: no network, no key), `--recursion-limit N` (200),
`--parallel N` (4 concurrent subagent children; 1 runs them inline), `--quiet`,
`--stats-json FILE`, `--dry-run` (prints the model, wire tool list and subagents; calls
nothing). Exit codes: 0 ok, 1 the run failed, 2 usage error, 3 configuration error
(unknown agent, missing key, `build/qumo` not built, OpenSSL not found).

The model is any OpenAI-compatible chat-completions endpoint listed in `agents/agents.json`
(OpenRouter by default); the key is read from the environment per request and redacted from
every log and error. Prompts, tool subsets and MCP tool guides come from `agents/prompts/`,
which `python -m qumo_agents export-prompts` writes from the Python code (a Python test fails
when they are stale), so both implementations share one source of truth.

## How it works

* **No outside programs.** Processes start with `posix_spawn`, HTTP and TLS are written on
  sockets and OpenSSL 3 (loaded at run time with `dlopen`; the macOS Keychain is not used, set
  `SSL_CERT_FILE` for a corporate CA), MCP is a client over stdio and streamable HTTP. A lint
  test (`test_native_lint`) fails when `qumo_agent/` mentions `std.subprocess`, a shell or
  `curl`.
* **Parallel subagents.** When one turn has several `task` calls, each subagent runs in its
  own `qumo-agent subagent` child process with a minimal environment and the results come back
  in call order. Other tools of a turn run one after the other.
* **Same wire format as deepagents.** The request bodies are compared with captures of the
  Python agent for six scenarios (tool results, parallel tools, subagents, eviction,
  summarization). Texts ported from deepagents are generated from the installed packages
  (`tests/golden/gen_texts.py`), never retyped. Licences: [../../NOTICE](../../NOTICE) and
  [LICENSES/langchain-MIT.txt](LICENSES/langchain-MIT.txt).

## The toolsmith: agents that build tools

`mojo-toolsmith` writes small native tools in Mojo, compiles them, tests them and publishes
them for other agents. A tool is one program with one tool, built on
`qumo_agent/tool_sdk.mojo`; [tools/README.md](tools/README.md) shows how to write one and
`tools/gen_bitstring_stats/` is the reference.

```bash
build/qumo-agent run mojo-toolsmith "Build gen_bitstring_entropy: Shannon entropy in bits of a counts JSON"
# open_sandbox -> toolsmith_new -> toolsmith_build -> toolsmith_test -> toolsmith_publish
build/qumo-agent tools list                       # the new tool is listed as NOT reviewed
git diff <the sandbox branch the agent reports>   # read the code like any other agent change
build/qumo-agent tools review gen_bitstring_entropy
build/qumo-agent run qumo-operator "Use gen_bitstring_entropy on {\"00\": 500, \"11\": 500}"
```

**The review gate.** A published tool is invisible to every agent until a human runs
`tools review NAME`. The command checks the stored hashes, prints the provenance, the full
source, the cases and the test result, and asks you to type the tool name (a script pipes the
name on stdin; `--yes` works only at a terminal). It then rebuilds the binary from exactly the
source you just read, re-runs the stored cases against it, replaces the agent-built binary and
marks the entry reviewed. At every load and every call the binary is hashed again against the
manifest. `tools revoke` takes a tool back, `tools verify` re-checks the store. The store is
`build/agent-tools/` (or `$QUMO_AGENT_TOOLS_DIR`), outside git. `--allow-unreviewed-tools`
exists for experiments and marks the tool description "NOT reviewed".

**Confinement.** Model-written code is never run unconfined. On macOS the build, the tests, the
`run_check` commands inside a sandbox and every call of a generated tool run under
`sandbox-exec` (no network, writes only to their own temporary directory, no reads of the
home directory beyond the Mojo runtime, no fork); generated tools also get an empty
environment, a timeout and capped output. Linux uses Landlock and is
documented in DESIGN.md 6.8. On a machine with no OS sandbox, building and testing are
refused unless a human passes `--allow-unsandboxed-tool-tests`. `doctor` reports which
sandbox applies.

## Limits

* **Linux is untested.** The Linux paths (socket and `addrinfo` layouts, glibc spawn helpers,
  CA paths, Landlock) are compile-checked and written from headers, not run. The first Linux
  run is a live check (DESIGN.md 8.3). glibc 2.29 or newer is assumed; older glibc takes a
  fallback for the working-directory action.
* **Not self-contained.** The binary needs the Mojo runtime libraries (like `build/qumo`) and
  an OpenSSL 3 library at run time. The toolsmith needs `.venv/bin/mojo` and `git`.
* **No proxy support** (`HTTPS_PROXY` is ignored) and no macOS Keychain trust.
* **Regular expressions** in `search_code` and `grep` are POSIX ERE with the Python syntax
  translated; lookarounds and backreferences are refused with a message.
* **Only `task` fans out.** The other tools of one turn run sequentially, so a turn with several
  slow network tools can take longer than in Python.
* **Back ends.** Only OpenAI-compatible HTTP endpoints. The Claude Code and Codex CLI back ends
  of `bin/qumo-agents` are not ported.
* **Cost.** Subagents have a recursion limit of 9999 (as in Python); watch the statistics line.
* **Tool trust.** A reviewed tool is trusted code of your own review: the confinement limits
  what a mistake can do, not what you decide to approve. On Linux the confined build and
  `run_check` profiles can still read non-sensitive files under `$HOME`.
* **Mojo 1.2 is pinned** (`docs/MOJO_GUIDE.md`); a new Mojo release needs a pass over the FFI
  layer (`qumo_agent/native/libc.mojo`).

## Layout

```
main.mojo               exit(main_entry(args))
qumo_agent/native/      libc bindings, spawn, files, regex, sandbox (the only place with external_call)
qumo_agent/net/         URL, sockets, HTTP/1.1, TLS (OpenSSL), SSE
qumo_agent/llm/         the OpenAI-compatible model client
qumo_agent/mcp/         MCP over stdio and HTTP; the stdio server
qumo_agent/harness/     the deepagents port: loop, file tools, eviction, summarization, subagents
qumo_agent/tools/       QUMO MCP, repository, research and alphaXiv tools; configuration
qumo_agent/toolsmith/   manifest, templates, build/test/publish, generated tools
qumo_agent/tool_sdk.mojo   compiled into every generated tool
qumo_agent/app/         the command line, serve, call, doctor, parallel subagents
specs/, prompts/        the mojo-toolsmith agent
tools/                  generated tools and the reference tool
tests/                  offline tests (+ golden data from Python); tests/test_live_*.mojo are the live checks
```

Errors the harness raises have stable codes, `QA-E101` to `QA-E170`:
[docs/ERRORS.md](../../docs/ERRORS.md#qa-e1xx--agent-harness-agentsmojo).
