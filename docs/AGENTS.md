# QUMO agents: how to use them

QUMO ships nine AI agents, built on [deepagents](https://www.langchain.com/deep-agents),
that keep the project current and let you use it in plain language. They watch papers
and SDK releases, review the code, improve a module in a sandbox branch, drive QUMO
through its MCP server, research algorithms, error correction and compilation, run
circuits, and orchestrate hybrid workflows.

A person starts every run. No agent runs on a schedule, commits, pushes or submits
to quantum hardware.

## 1. Quick start

```bash
make cli                         # build/qumo: the MCP server the agents call
make setup-agents                # agents/.venv with deepagents and the MCP adapters
export OPENROUTER_API_KEY=...    # default back end
bin/qumo-agents doctor           # what is ready, what is missing
bin/qumo-agents run qumo-operator "Build a 3-qubit GHZ, simulate 1000 shots with seed 7, check the counts"
```

`bin/qumo-agents` works from any directory. On its first run it creates
`agents/.venv` by itself if `make setup-agents` was not run.

## 2. Commands

| Command | Does |
|---|---|
| `bin/qumo-agents list` | Agents, their model, what they delegate to, modules, back ends, alphaXiv status |
| `bin/qumo-agents run AGENT "TASK"` | Run one agent on a task |
| `bin/qumo-agents doctor` | Check build/qumo, API keys (by name, never their values) and the Claude Code / Codex CLIs. Exit code 1 when nothing can run |
| `bin/qumo-agents reports` | List saved reports, newest first; `--show watch-ir` prints one |

Options of `run`:

| Option | Meaning |
|---|---|
| `--module NAME` | Focus on one QUMO module: `ir`, `sim`, `gpu`, `compiler`, `algorithms`, `qec`, `ft`, `exec`, `ai` |
| `--backend NAME` | `openrouter` (default), `claude-code` or `codex` |
| `--model ID` | Model for this run: an OpenRouter id (`anthropic/claude-sonnet-5.5`) or the CLI's own model name |
| `--dry-run` | Show the model, tools and subagents (or the CLI command line) and call nothing |
| `TASK` = `-` | Read the task from stdin: `cat task.md \| bin/qumo-agents run orchestrator -` |

## 3. The agents

| Agent | Use it to | Delegates to |
|---|---|---|
| `research-watch` | Find new papers and SDK releases for one module and rank what QUMO should adopt | |
| `code-review` | Get a performance and security review of a module from a second, independent model | |
| `autoresearch` | Improve a module step by step in a sandbox branch, with measurements | research-watch, code-review, algorithm-research, qec-research, compiler-research |
| `qumo-operator` | Use QUMO in plain language: algorithms, simulation, compilation, QEC, error codes | |
| `algorithm-research` | Explore new quantum algorithms and prototype toy instances on QUMO | |
| `qec-research` | Explore new error-correcting codes and decoders against QUMO's QEC baseline | |
| `compiler-research` | Explore new optimisation and compilation methods against QUMO's compiler | |
| `execution` | Run circuits the best way available: simulator, GPU, noisy emulation of a target | |
| `orchestrator` | Split a problem into classical, AI and quantum stages and run them | all of the above |

### Example tasks

```bash
# What changed for the compiler in the last 90 days, in papers and in Qiskit / TKET / BQSKit?
bin/qumo-agents run research-watch "Last 90 days, focus on routing and synthesis" --module compiler

# Independent review of the OpenQASM parser
bin/qumo-agents run code-review "Review qumo/ir/qasm.mojo for untrusted-input risks and hot loops" --module ir

# Try to improve the router; changes land on a branch agents/<topic>-<time>
bin/qumo-agents run autoresearch "Reduce SWAP count on heavy-hex without slowing compile time" --module compiler

# Use QUMO, explain and fix an error code
bin/qumo-agents run qumo-operator "Why does this fail? OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; cx q[0],q[2];"

# Research
bin/qumo-agents run algorithm-research "Recent quantum linear-system algorithms worth a toy reproduction"
bin/qumo-agents run qec-research "Compare bivariate bicycle codes with QUMO's surface code at d=6"
bin/qumo-agents run compiler-research "ZX-based T-count reduction methods since 2025"

# Run circuits
(echo "Run this circuit on a 5x5 grid with sc_cz basis and device noise:"; cat circuit.qasm) \
  | bin/qumo-agents run execution -

# Hybrid workflow
bin/qumo-agents run orchestrator "MaxCut on a 12-node 3-regular graph: classical warm start, QAOA on QUMO, compare with brute force"
```

## 4. Outputs

* **Answer**: printed on stdout when the run ends.
* **Reports**: research, review, autoresearch and orchestration runs save
  `agents/reports/<date>-<name>.md`. Reports are ignored by git; copy the ones you want to keep.
* **Sandbox branches** (`autoresearch` only): each run opens a worktree in
  `.worktrees/agent-<topic>-<time>` on branch `agents/<topic>-<time>`. Review it like any
  contribution:

```bash
git -C .worktrees/agent-<topic>-<time> diff
make -C .worktrees/agent-<topic>-<time> test
git worktree remove .worktrees/agent-<topic>-<time>   # when you are done with it
```

## 5. Back ends and models

| Back end | How it runs | Needs |
|---|---|---|
| `openrouter` (default) | A deepagents graph; delegation nests (orchestrator, then autoresearch, then research-watch) | `OPENROUTER_API_KEY` |
| `claude-code` | `claude -p` with the agent's prompt appended, QUMO's MCP server (and alphaXiv's) attached, and an allow-list of tools; `git commit`, `git push` and `gh` are blocked; a budget cap from `agents.json` | Claude Code installed and signed in |
| `codex` | `codex exec`, read-only sandbox, workspace-write only for `autoresearch` inside its worktree | Codex CLI. Not tested yet: it was not installed when this layer was written |

Models live in `agents/agents.json`, one OpenRouter id per agent. `code-review` uses a
different vendor from the agents that write code, so the review is independent. Change a
model there, per run with `--model`, or for every run with `QUMO_AGENTS_MODEL`. Set
`QUMO_AGENTS_BACKEND` to change the default back end.

## 6. Keys and environment

| Variable | Needed for |
|---|---|
| `OPENROUTER_API_KEY` | The `openrouter` back end |
| `ALPHAXIV_API_KEY` | Paper research through [alphaXiv](https://www.alphaxiv.org)'s MCP server (alphaxiv.org, Settings > API Keys). Without it, agents use the arXiv API |
| `GITHUB_TOKEN` | Optional: higher rate limit for release notes |
| `QUMO_BIN` | Optional: another QUMO binary for the MCP server (default `build/qumo`) |
| `QUMO_AGENTS_CONFIG` | Optional: another `agents.json` |

Keys are read from the environment only. They are never written to files, logs, reports
or command lines; the Claude Code MCP config refers to `${ALPHAXIV_API_KEY}` instead of
the value.

## 7. What the agents may and may not do

* Read the repository, except `.venv`, `build`, `.git`, `.worktrees` and secret-looking files.
* Run only allow-listed commands, without a shell: one Mojo test or example, one example
  through `examples/run_examples.py`, the hardware plugin tests, `build/qumo`, `git diff`
  and `git status`.
* Write only `agents/reports/`, and (autoresearch) files inside its own sandbox worktree.
* Use only alphaXiv's read tools. Its library, folder and follow tools are never loaded.
* Treat fetched papers, release notes and pages as data. The prompts tell every agent to
  ignore instructions found in them and to report them.
* Never commit, push, open pull requests or submit jobs to real hardware. The execution
  agent describes the hardware plugin command for a person to run.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `build/qumo not found` | `make cli` |
| `set OPENROUTER_API_KEY ...` | Export the key, or use `--backend claude-code` |
| `'codex' is not on PATH` | Install the Codex CLI, or choose another back end |
| A research report says the paper search timed out | Network or arXiv rate limit; rerun later, or set `ALPHAXIV_API_KEY` |
| The run stops with a recursion limit | The task is too broad; split it, or name a module with `--module` |

## 9. Extending

* **A module**: add a `Module` to `agents/qumo_agents/modules.py` (paths, docs, tests,
  SDKs, paper queries, MCP tools). `make test-agents` checks that its paths exist.
* **An agent**: write its prompt in `prompts.py`, add an `AgentSpec` to `registry.py`
  (tool groups, MCP tools, subagents) and a model in `agents.json`.
* **An MCP tool guide**: add an entry to `TOOL_GUIDE` in `tools/qumo_mcp.py` when the
  QUMO MCP server gains a tool.

## 10. Tests

```bash
make test-agents
```

The tests run offline. They check the module table, the safety limits of the tools, the
Claude Code and Codex command lines (keys never appear), the arXiv parser, network
failures, and that every agent compiles. A scripted model also drives the operator through
QUMO's real MCP server.
