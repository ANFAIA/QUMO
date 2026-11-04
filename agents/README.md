# QUMO agent layer

Nine agents, built on [deepagents](https://www.langchain.com/deep-agents), keep QUMO
moving: they watch new papers and SDK releases, review the code, improve modules in a
sandbox, use QUMO through its MCP server, and orchestrate hybrid workflows. Each run is
started by a person; nothing runs on a schedule, commits or pushes.

| Agent | Does | Delegates to |
|---|---|---|
| `research-watch` | New papers (alphaXiv) and SDK releases (PyPI, GitHub) for one module; ranks what QUMO should adopt | |
| `code-review` | Performance and security review on an independent model; file:line findings with failure scenarios | |
| `autoresearch` | Improves one module in a sandbox branch: hypothesis, baseline, change, measurement, review | research-watch, code-review, the three research agents |
| `qumo-operator` | Uses QUMO through MCP: algorithms, simulation, compilation, QEC, error codes | |
| `algorithm-research` | New quantum algorithms, compared with QUMO's library, toy prototypes | |
| `qec-research` | New error-correcting codes and decoders against QUMO's QEC baseline | |
| `compiler-research` | New optimisation and compilation methods against QUMO's compiler baseline | |
| `execution` | Runs circuits the best available way (simulator, GPU, noisy emulation) | |
| `orchestrator` | Splits a problem into classical, AI and quantum stages | every agent above |

Modules the agents know (`qumo_agents/modules.py`): ir, sim, gpu, compiler, algorithms,
qec, ft, exec, ai. Each lists its code, docs, tests, alternative SDKs and paper queries.

## Setup

```bash
uv venv agents/.venv --python 3.12
uv pip install --python agents/.venv/bin/python -e agents
make cli   # build/qumo, the MCP server the agents call
```

Environment (keys are read from the environment only, never written):

| Variable | For |
|---|---|
| `OPENROUTER_API_KEY` | default back end (OpenRouter) |
| `ALPHAXIV_API_KEY` | paper research through alphaXiv's MCP server (alphaxiv.org, Settings > API Keys). Without it, the agents fall back to the arXiv API |
| `GITHUB_TOKEN` | optional, higher GitHub rate limit for release notes |
| `QUMO_AGENTS_MODEL`, `QUMO_AGENTS_BACKEND` | optional overrides |

## Use

```bash
cd agents
.venv/bin/python -m qumo_agents list
.venv/bin/python -m qumo_agents run research-watch "What changed in the last 90 days?" --module compiler
.venv/bin/python -m qumo_agents run qumo-operator "Build a 4-qubit GHZ, simulate 1000 shots, check the counts"
.venv/bin/python -m qumo_agents run code-review "Review the OpenQASM parser" --module ir --backend claude-code
.venv/bin/python -m qumo_agents run autoresearch "Reduce SWAP count on heavy-hex" --module compiler --dry-run
```

Reports land in `agents/reports/<date>-<name>.md`.

## Back ends

* **openrouter** (default): a deepagents graph on an OpenRouter model. Per-agent models are in
  `agents.json`; `code-review` uses a different vendor than the others on purpose.
* **claude-code**: `claude -p` with the agent prompt appended, QUMO's MCP server (and
  alphaXiv's) attached, and an allow-list of tools. `git commit`, `git push` and `gh` are
  disallowed.
* **codex**: `codex exec`, read-only sandbox for every agent but `autoresearch`
  (workspace-write, inside its own worktree). Not tested here: the Codex CLI was not installed.

## Safety

* Read tools stay inside the repository and skip `.venv`, `build`, `.git` and secret-looking files.
* Only `autoresearch` writes code, and only inside a sandbox git worktree on a new
  `agents/<topic>-<time>` branch. The main checkout is never written. A person reviews the branch.
* Commands run from an allow-list, without a shell.
* Papers, release notes and pages are data: the prompts tell the agents never to follow
  instructions found in them.
* alphaXiv: only read tools are loaded; library, folder and follow tools are never exposed.

## Tests

```bash
cd agents && .venv/bin/python -m pytest -q tests
```
