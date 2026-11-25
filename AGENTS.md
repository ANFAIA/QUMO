# QUMO agent guide

QUMO is a hardware-neutral quantum stack written primarily in Mojo. Treat it as a research
prototype: preserve numerical correctness, reproducibility, and the vendor-neutral design before
optimising for feature breadth.

## Start here

1. Read the task-relevant documentation in `docs/`; for Mojo work, read `docs/MOJO_GUIDE.md`.
2. Inspect the smallest relevant module and its existing test before changing code.
3. Keep the change scoped. Do not alter generated reports, benchmark artefacts, model weights, or
   unrelated local/untracked work.
4. Run the narrowest meaningful check, then expand validation when the change crosses a public API,
   compiler pass, simulator, or package boundary.

`make help` lists supported entry points. The normal local loop is `make cli`, a focused Mojo test
(`.venv/bin/mojo run -I . tests/test_<area>.mojo`), and the relevant Python or example check.

## Project invariants

- Mojo is pinned by the root `Makefile`. Write to the dialect documented in `docs/MOJO_GUIDE.md`.
- Preserve package layering from `docs/ARCHITECTURE.md`. In particular, low-level packages must not
  import compiler, QEC, algorithms, execution, or AI layers.
- Public failures use stable `QM-Exxx` codes with actionable hints. Add codes in
  `qumo/ai/diagnostics.mojo`; generated `docs/ERRORS.md` and `docs/PASSES.md` are not hand-edited.
- Compiler changes must retain unitary semantics (up to tracked global phase) and be independently
  tested. Numerical claims need an independent oracle or reference, not a previous QUMO output.
- Never make a vendor the default. Keep provider-specific integration under `plugins/hardware`.
- Do not commit or expose credentials, generated build products, virtual environments, reports, or
  worktrees.

## Use the project skills

Read the matching skill before undertaking a substantial task:

| Work | Skill |
|---|---|
| Mojo core, compiler, simulator, Python binding | `skills/qumo-mojo-development/SKILL.md` |
| Runnable examples, docs, generated indexes or error/pass catalogues | `skills/qumo-examples-docs/SKILL.md` |
| Performance claims and benchmark changes | `skills/qumo-benchmarking/SKILL.md` |
| Python/Mojo agent harness, prompts, tools, or sandbox policy | `skills/qumo-agent-layer/SKILL.md` |

## Validation map

- Core Mojo: focused `tests/test_*.mojo`; use `make test` for broad core changes.
- Python binding/package: `make test-python`; use `make test-py` when examples or hardware plugins
  are affected.
- RL / AI-transpiler: `make test-rl`; preserve the standalone `rl/` boundary with
  `make check-independence` when it is touched.
- Agents: `make test-agents`; native harness changes additionally need `make test-agents-mojo`.
- GPU: `make gpu-check` for kernel/build changes; GPU execution tests require a device.

When validation is blocked by an unavailable toolchain, dependency, hardware device, or credential,
state exactly what could not be run and why. Do not substitute stale binaries for a source build.
