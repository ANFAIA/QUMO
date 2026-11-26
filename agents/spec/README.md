# Guard rules: one specification for both agent harnesses

QUMO has two agent harnesses: Python (`agents/qumo_agents`, deepagents) and Mojo
(`agents/mojo`, the native `build/qumo-agent`). Both give a model tools that read the
repository, run commands and fetch pages, and both must refuse the same things.
`guard_rules.json` in this directory is the one place where those rules are written.
Each harness loads its lists from it and runs every test vector in it through its own
functions. A rule changed on one side only fails the other side's test.

## The file

| Key | Content |
|---|---|
| `lists.repo` | skip directories, secret-looking file names, the `run_check` allow-list, the characters of an argument, git flags, `build/qumo` subcommands and options |
| `lists.env` | substrings of a secret environment name, the extra ones the Python check environment drops, the git-configuration group, the prefixes each CLI back end keeps, the names a Mojo child inherits |
| `lists.research` | scheme, port, API hosts, redirect limit, and the networks refused or judged by their embedded IPv4 address on top of "not globally routable" |
| `lists.model_key` | which hosts may receive a key over plain `http` (this machine only) |
| `lists.delegation` | the tool groups that fetch from the web and that run code |
| `lists.untrusted` | the marker and size limits of the untrusted-data wrapper |
| `reasons` | for each rule, the reason classes of a refusal and the text that identifies each one in a harness's message |
| `path_fixture` | a small file tree (with symlinks) both tests build in a temp directory |
| `vectors` | `{id, rule, input, expect, reason}`: `expect` is `allowed` or `refused`; `reason` is a class of `reasons[rule]` |

Rules of the vectors:

| `rule` | Input | Python function | Mojo function |
|---|---|---|---|
| `path` | a path, relative to the fixture root | `repo._resolve` | `resolve_repo_path` |
| `command` | an argv, sandbox open or not | `repo._run_check_refusal` | `run_check_refusal` |
| `url` | a list of hops (the URL, then each redirect target), the host list (`api` or `any`), the addresses each host name resolves to | `research._check_url`, `_CheckedRedirects.redirect_request` | `parse_url`, `resolve_reference`, `UrlGuard.check`, `literal_is_global` |
| `check_env` | a variable name: does a `run_check` child inherit it | `repo._check_env` | `check_env_keeps` |
| `secret_env_name` | a variable name: is it a secret (`refused` = yes) | `backends.CLI_SECRET_ENV` | `is_secret_env_name` |
| `cli_scrub_env` | a back end and a variable name | `backends.scrub_env` | none (no CLI back ends) |
| `model_key` | a base URL, with or without a key | `config.check_key_url` | `LlmClient.__init__` |
| `delegation` | a set of agents (tool groups, subagents, declared untrusted subagents) | `registry.check_web_exec` | `check_web_exec` |
| `delegation_shipped` | none: the agents the harness really loads | `registry.check_web_exec()` | `Registry(".")` |
| `untrusted_wrap` | text, source, limit; `output` is the wrapped text with the random token written `TOKEN` | `untrusted.wrap` | `untrusted_wrap` |

## How each harness uses it

- **Python** loads the file at import (`qumo_agents/guard_spec.py`). `tools/repo.py`,
  `tools/research.py`, `registry.py`, `backends.py`, `config.py` and `untrusted.py` build
  their tables from `lists`. `agents/tests/test_guard_spec.py` runs the vectors.
- **Mojo** reads `lists` through `agents/mojo/qumo_agent/core/guard_rules.mojo`, which
  `gen_mojo_rules.py` writes from the file. It is generated and not read at start-up because
  the low layers (`native/`, `net/`) have no JSON reader, and a rule file next to the binary
  could be edited by whatever the agent runs. `agents/mojo/tests/test_guard_spec.mojo` runs
  the vectors and compares the generated module with the file value by value. The Mojo
  harness still reads skip directories and the `run_check` allow-list from
  `agents/prompts/tools.json` at start-up; the Python export writes them there from this
  file, and the Mojo test checks they are equal.

To change a rule:

```bash
$EDITOR agents/spec/guard_rules.json            # the list, and a vector that shows the change
python3 agents/spec/gen_mojo_rules.py           # rewrite core/guard_rules.mojo
agents/.venv/bin/python -m qumo_agents export-prompts   # if skip dirs, allow-list or arg chars changed
make test-agents                                 # every vector, Python
.venv/bin/mojo run -I . -I agents/mojo agents/mojo/tests/test_guard_spec.mojo   # every vector, Mojo
```

A stale generated module fails both tests. A vector whose `rule` a harness has no runner
for fails that harness's test, so a new rule cannot be added to one side only.

## Differences that remain, and why

A vector may say that the harnesses differ: `harnesses` limits it to one of them,
`by_harness` gives another expectation for one of them, and both need a `difference` text
(the Python test refuses a vector without it). Today:

| Difference | Why |
|---|---|
| A `run_check` child of the Mojo harness inherits ten named variables (`lists.env.minimal_child_env`); the Python harness passes everything except names matching the secret pattern or the git-configuration group. Vectors `env-21` to `env-24`. | By design: an allow-list on the Mojo side. Python stays a deny-list because the checks it runs (virtualenv Python, pytest) depend on variables that are not known in advance. |
| `cli_scrub_env` vectors run in Python only. | The Mojo harness has no Claude Code / Codex back ends. |
| A keyless model URL (`model-key-18`) runs in Python only. | The Mojo harness always needs a key variable; the Python `local` back end may run without one, and then nothing secret is sent. |

Not expressed as vectors:

| Difference | Why |
|---|---|
| The Mojo harness runs a `run_check` inside a sandbox under OS confinement (macOS `sandbox-exec`, Linux Landlock: writes only to the sandbox and temp directories, no network, `$HOME` denied except named paths). The Python harness does not confine it. | By design of the Mojo harness (`agents/mojo/DESIGN.md` 6.8); the Python tool runs ordinary Python and Mojo code in the user's account, as `docs/AGENTS.md` section 7 says. |
| A host name is resolved once by the Mojo harness and the connection goes to the addresses that were checked. The Python fetcher checks the addresses and then lets `urllib` resolve the name again, so a name whose answer changes between the two lookups is not caught. | Open: closing it needs a connection class that connects to the checked address. The vectors give each host fixed addresses, so they cannot show it. |
| An IPv4 literal with a leading zero in dotted form (`0177.0.0.1`) is octal for glibc and decimal for macOS. Each harness judges the address it would connect to (Mojo also refuses the octal reading), so there is no vector for these spellings. | Platform behaviour of `getaddrinfo`. |
| The URL-with-credentials and malformed-URL refusals have different texts (`reasons.url` lists both markers of a class). | The Mojo URL parser refuses them before the guard runs. |
| The hop limit of a redirect chain is enforced inside `HttpClient` on the Mojo side; the vector test passes each hop through the real URL rules and compares the count with the same constant, and `test_net_redirects.mojo` drives the client itself with a fixture server. | The real client cannot be pointed at public host names without a network. |
| The Python harness also wraps reports (a report read from `agents/reports/` with `read_file`, and the copies `open_sandbox` puts in a sandbox). The Mojo harness wraps subagent answers only; its `open_sandbox` copies no reports. | Open: the Mojo `read_file` returns a report as it is. Follow-up. |
| The Claude Code and Codex back ends of the Python harness do not apply the untrusted-data wrapper, and their command rules are Claude Code permission patterns, not this file. | Those CLIs run their own tools. |
