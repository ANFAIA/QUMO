# Verified discovery loop

A small, runnable version of the loop in Shirobokov et al.,
[Discovering formally verified quantum programs with AI](https://arxiv.org/abs/2611.09822) (v3, Sec. 1.2 and
Appendices B and C): teams propose programs, a verifier they cannot edit checks them against a frozen
specification, and only verified programs enter a shared wiki.

The task is the paper's parallel Find-Any search. The straight (non-learning) form of the winning program is
`examples/algorithms/parallel_grover_find_any.mojo` (added by a separate pull request, #99).

```bash
make py
```
```bash
PYTHONPATH=build:plugins/hardware bench/.venv/bin/python examples/heterogeneous/discovery_loop/discovery_loop.py --seed 7
```

## Read this first

| | Paper | This example |
|---|---|---|
| Verifier | Lean kernel and Comparator, all sizes | QUMO simulator, 7 to 18 qubits: numerical testing, **not a proof** |
| Proposers | teams of LLM agents writing programs and proofs | REINFORCE over five fields, plus offline rules or a model choosing the same five fields |
| Search space | open | designed by us, and it **contains the paper's answer**: the loop selects and tunes, it does not invent circuits |
| Learning | agentic search, not reinforcement learning | the RL is this example's own choice: a one-step bandit |
| Conferences | scheduled cross-team forums | none |
| Merge threshold | 5% by default | 1% by default (`--min-gain`) |

## Who does what

| File | Role | Can the teams change what it decides? |
|---|---|---|
| `spec.json` | cost, correctness, grid sizes, the five fields. Its SHA-256 is pinned in `verifier.py` | no |
| `screening.py` | closed-form cost in the two-dimensional symmetric subspace; what teams optimise | it is their own estimate, never evidence |
| `verifier.py` | builds each candidate as an OpenQASM 3 program with a `while` loop, draws marked items from a private seed, runs it in QUMO | no: it receives data only |
| `proposers.py` | `RLTeam` (REINFORCE) and `ScribeTeam` (rules, or any OpenAI-compatible model through `../llm_client.py`) | |
| `discovery_loop.py` | rounds, wiki, devices, report | |

The verifier requires, on every verification instance: no wrong answer in any shot, the loop ends within its
budget, the mean number of query rounds is within 5 standard errors of the screening model, and for the weak
loop the exact `P(hit)` of the first check (from the state vector) matches the model to 1e-6. A candidate with
extra fields, such as a claimed cost, is refused before anything runs.

A refusal for wrong answers is written to the wiki as a dead end and that combination is not proposed again.
Zero error is tested with 800 shots per instance: that catches a shortcut that is wrong often, not one that is
wrong once in a million runs.

## What a run finds

Cost is expected parallel query rounds over `(pi/4) sqrt(N/mk)`, in the screening model at n = 30, averaged
over four (m, k) cells.

| Candidate | Cost |
|---|---|
| m independent searches (the published idea, best `a` by grid search; this is the wiki's starting record) | 0.967 |
| one joint reflection over all slots, with restarts (grid search; not part of the default run) | 0.879 |
| joint reflection with a weakly measured loop | 0.8183 at the optimum, 1/2 + 1/pi = 0.8183 |

Seeds 7, 1, 2, 3 and 4 (macOS, CPU, 2026-11-27) reached 0.8185 to 0.8210, with `a` from 0.36 to 0.43 (pi/8 is
0.39) and `c` from 3.98 to 4.40. In each, the learning team's first submission was already the final program:
later rounds found nothing 1% better. The cost surface is flat near the optimum, so `a` is only loosely fixed.

**The planted shortcut** is the same loop without the confirming query. The screening model scores it at least
as well, and the verifier refuses it: with seed 7, 417 of 800 answers were wrong at m = 2. It is planted: at
n = 30 one query is too small a saving for the learning team to be drawn to it by itself.

**Does the learning matter?** Best screened cost over 20 seeds, same number of evaluations, shortcuts excluded
(ad hoc measurement, not part of the run):

| Evaluations | REINFORCE median (worst) | Random search median (worst) |
|---|---|---|
| 256 | 0.828 (0.859) | 0.830 (0.856) |
| 512 | 0.821 (0.837) | 0.826 (0.850) |
| 960 | 0.819 (0.823) | 0.825 (0.839) |
| 3840 | 0.8186 (0.8194) | 0.8192 (0.8235) |

At 256 they are equal; between 512 and 960 the policy is clearly better; at the default budget both get close.
The space is small, so this says little about harder searches.

## Devices

| Stage | Device |
|---|---|
| policy updates, screening | CPU |
| verifier shots (dynamic circuits) | CPU for these widths |
| widest exact check (18 qubits) | GPU through `expectation_device` with a `make py-gpu` build; otherwise CPU. **The GPU path of this example has not been run** (no Metal toolchain on the test machine) |
| one attempt of the winner, N = 8 | vendor-neutral noisy QPU emulator; remote targets are refused |

On the emulator (seed 7, 89 two-qubit gates), confirmed-but-wrong answers went from 0 without noise to 4.9%
with it. The confirming query is itself noisy, so on hardware the returned address must be checked
classically. The device stage runs one unrolled attempt, not the `while` loop.

## A model as the second team

Default: no model. `--provider ollama|lmstudio|llamacpp|vllm|openrouter [--model ID]` uses the profiles of the
[heterogeneous example](../README.md#choosing-the-model). The model sees the specification, the wiki's best
program, dead ends and history, and returns one candidate under a strict schema; it never sees or sets a cost,
and nothing it writes is executed. An unusable answer falls back to the rules.

Tried on 2026-11-27 with LM Studio: `qwen/qwen3.8-27b` did not load (not enough memory) and `lfm3-1.7b`
returned empty output, so both runs used the rules. **No real model has produced an accepted candidate here**;
the model path is tested with a fake client.

## The authors' repository

[google-deepmind/formal-quantum-computation](https://github.com/google-deepmind/formal-quantum-computation)
(Apache-2.0) holds the Lean library and, on its `discoveries` branch, the solutions. With a local clone:

```bash
git clone https://github.com/google-deepmind/formal-quantum-computation ../formal-quantum-computation
```
```bash
git -C ../formal-quantum-computation fetch origin discoveries:discoveries
```

add `--formal-repo ../formal-quantum-computation`. The run reads the constant (`relCost`) claimed by their
Find-Any solution (commit `a88de8ff65` on 2026-11-27: within 2e-21 of 1/2 + 1/pi by its own docstring) and
compares it with 1/2 + 1/pi and with the loop's best. It **only reads the text**: no Lean proof is built or
checked, no code is copied, and this loop's candidates are not Lean programs, so their Comparator cannot judge
them. To check the authors' own proof, follow their `QuantumComputationDiscoveries/README.md` (they report
about 3 minutes for Find-Any with `lake comparator`; not run here).

## Output

`build/hetero/discovery_loop/seed<N>/`: `wiki/wiki.json`, `wiki/history.md` (every submission and verdict) and
`report.json` (stages, devices, learning curve, model log).

Tests: `bench/.venv/bin/python -m pytest -q examples/heterogeneous/discovery_loop`.
