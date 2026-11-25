# QUMO agent layer

Nine agents, built on [deepagents](https://www.langchain.com/deep-agents), keep QUMO
moving: they watch new papers and SDK releases, review the code, improve modules in a
sandbox, use QUMO through its MCP server, and orchestrate hybrid workflows. No agent
merges or pushes to `main`. A person starts each run, or a schedule the owner set up does: the
weekly job that researches the nine modules and opens pull requests for at most two small changes
([docs/AGENTS.md](../docs/AGENTS.md), section 12).

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

Full guide: [docs/AGENTS.md](../docs/AGENTS.md). Entry point: `bin/qumo-agents` (any directory).

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
| `QUMO_AGENTS_BACKEND`, `QUMO_AGENTS_MODEL` | optional: harness and model for every run in this shell |
| `QUMO_AGENTS_LOCAL_URL` | optional: base URL of the `local` back end (default `http://localhost:11434/v1`) |
| `QUMO_AGENTS_LOCAL_API_KEY` | optional: key for a local server that asks for one; never read from `agents.json` |

What a check command can do: `run_check` (and the same rule in the native harness) runs only
allow-listed commands with path-like, relative arguments. `build/qumo` takes the `run`,
`statevector`, `expect`, `transpile`, `emulate`, `validate`, `info`, `draw`, `convert`, `vqe` and
`describe` subcommands; file arguments follow the `read_file` rules, and `-o` / `--output` works
only inside an open sandbox. On the `claude-code` back end `convert` and `transpile` are not
allowed (use the MCP tools). Checks on the API back ends and the Mojo harness do not see
variables named like a token, key or secret; the Claude Code and Codex CLIs pass your whole
environment unless `agents.local.json` sets `"claude_code": {"env": "scrub"}` or
`"codex": {"env": "scrub"}` (see [docs/AGENTS.md](../docs/AGENTS.md), section 7).

## Use

```bash
bin/qumo-agents doctor
bin/qumo-agents list
bin/qumo-agents run research-watch "What changed in the last 90 days?" --module compiler
bin/qumo-agents run qumo-operator "Build a 4-qubit GHZ, simulate 1000 shots, check the counts"
bin/qumo-agents run code-review "Review the OpenQASM parser" --module ir --backend claude-code
bin/qumo-agents run autoresearch "Reduce SWAP count on heavy-hex" --module compiler --dry-run
```

Reports land in `agents/reports/<date>-<name>.md`.

## Back ends

The shared `agents.json` defaults to OpenRouter. For your own machine, copy
`agents.local.example.json` to `agents.local.json` (git-ignored) and set what you want there,
for example Claude Code on your subscription.

Pick the harness with `--backend` and the model with `--model`, per run; or per agent in
`agents.json` (`"backends"`, and `"models"` inside each back end). `bin/qumo-agents list`
shows what each agent would run on. Details: [docs/AGENTS.md](../docs/AGENTS.md), section 5.

* **openrouter** (default): a deepagents graph on an OpenRouter model. Per-agent models are in
  `agents.json`; `code-review` uses a different vendor than the others on purpose.
* **mojo**: QUMO's own harness (`build/qumo-agent`, `make agents-mojo`) on an OpenRouter model:
  `--backend mojo`, or per agent in `agents.json`. No Python at run time. Same model ids as `openrouter`.
* **local**: the same graph on your own OpenAI-compatible server (Ollama, LM Studio,
  llama.cpp, vLLM): `--backend local --model <name>`. Not run end to end here.
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
