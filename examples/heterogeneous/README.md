# Heterogeneous CPU + GPU + QPU, with an AI choosing the algorithm

`ai_gpu_qpu.py` takes a problem description (MaxCut graph, molecule, spin
model, search, classification), asks a language model **which algorithm to
run and how**, builds that algorithm from QUMO's library, and runs it in three
stages: a **GPU** stage (landscape scan, Gram matrix, exact verification), a
**CPU** stage (optimiser, SVM solve, post-processing) and a **QPU** stage
(final sampling or estimation on a noisy emulator or, only with explicit
human flags, real hardware). Every run writes a report and prints `CHECK`
lines.

The split of responsibilities is the point of the example:

| Who | Decides |
|---|---|
| The model (OpenRouter, a local server, or the offline rules) | *what* to run: algorithm, depth, optimiser, gradient, iterations, shots, what each stage does |
| QUMO (deterministic code) | *where* each stage runs (GPU status, qubit thresholds), size caps, whether the answer is valid |
| A human (command-line flags) | anything remote or billable: `--allow-remote`, `--submit`, `--yes`, `--max-cost` |

```bash
make py                                   # build/qumo_py.so (CPU); `make py-gpu` for build/gpu/qumo_py.so
export PYTHONPATH=build:plugins/hardware  # or build/gpu:plugins/hardware

# offline, deterministic (no model, no network): what CI runs
bench/.venv/bin/python examples/heterogeneous/ai_gpu_qpu.py --offline --small
bench/.venv/bin/python examples/heterogeneous/ai_gpu_qpu.py --offline          # 16-node 3-regular MaxCut

# a model picks the algorithm (OpenRouter, default model anthropic/claude-opus-5.5)
export OPENROUTER_API_KEY=...             # read from the environment only; never logged or stored
bench/.venv/bin/python examples/heterogeneous/ai_gpu_qpu.py --problem examples/heterogeneous/problems/h2_molecule.json
```

Output: `build/hetero/<problem>-<seed>/report.json`, `report.md`, `plan.json`
(replayable with `--replay`) and `payload.qasm` (the bound circuit sent to
the QPU stage).

## Architecture

```
problem.json --+--> (1) context pack: `qumo describe capabilities`, qumo_gpu_info, every
               |        compatible algorithm built + qumo_estimate_resources for the QPU's
               |        modality, QPU backends with cost class                (deterministic)
               v
          (2) planner.select_plan
               |  model (OpenAI-compatible chat completions, JSON schema)  --or--  offline rules
               |  -> validate: JSON schema (all enums, no extra fields) + algorithm allow-list
               |     + cross-field rules (algorithm vs problem kind, stage tasks, caps, fp64 on Apple)
               |  -> bounded corrective retries, then fall back to the rules (reason logged)
               v
          (3) build: qumo_build_circuit (build/qumo tool) or the Python twin builders (kernels.py)
               v
          (4) GPU stage   qumo_py.sample_device(..., "gpu") when GPU support is compiled in and
               |          n >= the GPU threshold (or --force-gpu); otherwise CPU, with a note
          (5) CPU stage   scipy optimiser on exact qumo_py.expectation; brute-force optimum; SVM
          (6) QPU stage   qumo_hardware.runner.run(...) on the chosen backend (local = runs;
               |          remote = dry run with the cost estimate unless --submit --yes --max-cost)
               v
          (7) report.json / report.md / plan.json + CHECK lines (examples/_checks.py)
```

Algorithms dispatch through a dict (`stages.PIPELINES`); nothing the model
writes is executed, evaluated or used as a path. The plan schema has no field
for confirmation, cost limits, credentials, files or commands.

| Problem file | Kind | Offline choice | Stages (GPU / CPU / QPU) |
|---|---|---|---|
| `maxcut_16_3reg.json` (default) | maxcut, 16 qubits | `qaoa_maxcut` p=2, COBYLA | landscape scan / gradient-free refine / final sampling |
| `maxcut_8_ring.json` (`--small`) | maxcut, 8 qubits | `qaoa_maxcut` p=2 | same |
| `maxcut_20_3reg.json` (`--size large`) | maxcut, 20 qubits | `qaoa_maxcut` p=1 | same, 5x5 grid emulator |
| `h2_molecule.json` | molecule, 4 qubits | `vqe_uccsd`, BFGS | exact verification / refine / one estimation job per qubit-wise-commuting group |
| `heisenberg_16.json` | spin model, 16 qubits | `vqe_hea`, SPSA | exact training / post-process / estimation |
| `grover_5.json` | search, 5 qubits | `grover_search` | exact verification / post-process / sampling |
| `moons_classification.json` | classification | `quantum_kernel_svm` | Gram matrix / SVM solve / 20 kernel entries on the QPU |
| `injection_test.json` | maxcut | `qaoa_maxcut` | a problem whose text tries to make the model submit to real hardware: it never can |

## Choosing the model

One standard-library client (`llm_client.py`, `urllib`) speaks the
OpenAI-compatible chat-completions protocol, so one code path reaches hosted
models through OpenRouter and local models through Ollama, LM Studio,
llama.cpp or vLLM. Providers are profiles in `llm_profiles.json`; flags and
environment variables override them.

| Flag | Environment | Meaning |
|---|---|---|
| `--provider` | `QUMO_LLM_PROVIDER` | `openrouter` (default), `ollama`, `lmstudio`, `llamacpp`, `vllm`, `local` (= `ollama`, or any server with `--base-url`), `rules` |
| `--model` | `QUMO_LLM_MODEL` | model id; `auto` (local profiles) = the server's first listed model |
| `--base-url` | `QUMO_LLM_BASE_URL` | any OpenAI-compatible endpoint, e.g. `http://gpu-box:8000/v1` |
| `--profiles` | | another profiles file |
| `--llm-retries` | | corrective retries after an invalid answer (default 2) |
| `--llm-timeout` | | seconds per request |
| `--transport openai` | | use the `openai` package instead of `urllib` (only if installed) |
| `--offline` | | never call a model |

### OpenRouter (default)

```bash
export OPENROUTER_API_KEY=sk-or-...                         # https://openrouter.ai/keys
export OPENROUTER_HTTP_REFERER=https://github.com/you/app   # optional attribution headers
export OPENROUTER_X_TITLE="my QUMO run"
bench/.venv/bin/python examples/heterogeneous/ai_gpu_qpu.py                       # anthropic/claude-opus-5.5
bench/.venv/bin/python examples/heterogeneous/ai_gpu_qpu.py --model openai/gpt-6.1-sol
bench/.venv/bin/python examples/heterogeneous/ai_gpu_qpu.py --model google/gemini-3.8-flash
```

Model ids were checked against `https://openrouter.ai/api/v1/models` on
2026-10-13. The profile lists alternatives from several vendors
(`anthropic/claude-sonnet-5.5`, `anthropic/claude-haiku-5.5`,
`openai/gpt-6.1-sol`, `google/gemini-3.8-flash`, `qwen/qwen3.8-max-prime`,
`deepseek/deepseek-v4-pro`, `mistralai/mistral-large-4-0`); any id from
<https://openrouter.ai/models> works. The client requests
`response_format: json_schema` with `provider.require_parameters: true`, so
OpenRouter routes only to providers that honour structured output, and
`usage: {include: true}`, so the report records tokens and the billed cost
(`decision_log.attempts[].cost_usd`, `cost.llm_usd`).

Without `OPENROUTER_API_KEY` nothing is sent: the run logs
`OPENROUTER_API_KEY is not set` and uses the offline rules.

### Ollama

```bash
ollama pull qwen3:8b && ollama serve          # http://localhost:11434/v1
bench/.venv/bin/python examples/heterogeneous/ai_gpu_qpu.py --provider ollama --model qwen3:8b --small
```

### LM Studio

Start the local server (Developer tab, or `lms server start`) and load a model:

```bash
bench/.venv/bin/python examples/heterogeneous/ai_gpu_qpu.py --provider lmstudio --model google/gemma-4-e4b --small
bench/.venv/bin/python examples/heterogeneous/ai_gpu_qpu.py --provider lmstudio --small     # model auto: first loaded
```

Checked on 2026-10-13 against LM Studio on an M5 with `google/gemma-4-e4b`:
a valid structured plan in 35 s (2 552 tokens), all checks passed.

### llama.cpp, vLLM and other servers

```bash
llama-server -m model.gguf --port 8080        # then --provider llamacpp
vllm serve Qwen/Qwen3-8B --port 8000          # then --provider vllm (key, if any, from LOCAL_LLM_API_KEY)
bench/.venv/bin/python examples/heterogeneous/ai_gpu_qpu.py --provider local --base-url http://host:port/v1 --model NAME
```

Local profiles need no key. If the server is not reachable within a few
seconds, the run falls back to the rules and says why.

### What happens to the model's answer

1. Structured output: `response_format` `json_schema` (strict) when the
   endpoint accepts it; if it answers 400/404/415/422/501, the schema moves
   into the prompt and the JSON is extracted from the text (fenced or bare).
2. Validation, always: the JSON schema (every bound an enum, no extra
   properties), the algorithm allow-list, compatibility with the problem kind,
   stage tasks, qubit and shot caps, fp64 on Apple GPUs.
3. An invalid answer gets the validator's messages back for up to
   `--llm-retries` corrections. Refusals, truncation (`finish_reason`
   `length`), content filtering, 401/403, exhausted retries for 429/5xx or
   timeouts, and connection errors go straight to the offline rules. The
   reason is in `decision_log.fallback_reason`.
4. A remote QPU chosen by the model without `--allow-remote` is replaced by
   the default local emulator (logged in `decision_log.overrides`); `--qpu`
   given by a human always wins over the model.

Logged per attempt: provider, profile, base URL, requested and served model,
response id, latency, structured-output mode, token usage, OpenRouter cost,
finish reason, the raw text (first 8 000 characters) and validation errors.
Keys are read only from the variable named in the profile (`api_key_env`),
are sent only in the `Authorization` header, and every error message passes
through `llm_client.redact()`.

## The QPU stage and its safety

| `--qpu` | What it is | Cost |
|---|---|---|
| `local:qumo-sim:grid:4x4/sc_cz+noise:superconducting` (default, n <= 16) | QUMO compiles for a 4x4 grid with a CZ basis, then a depolarizing + readout model with the superconducting modality rates | free, local |
| `local:qumo-sim:grid:5x5/sc_cz+noise:superconducting` (default, n > 16) | same on 5x5 | free, local |
| `local:qumo-sim:all:16/ion_zz+noise:trapped_ion` | all-to-all, ZZ basis, trapped-ion modality rates | free, local |
| `local:qumo-sim:grid:5x5/atom_cz+noise:neutral_atom` | neutral-atom modality rates | free, local |
| `ibm:fake_torino` | IBM calibration snapshot + Qiskit Aer | free, local |
| `quantinuum:H2-1LE` | Quantinuum local emulator (pytket-pecos) | free, local |
| `ibm:ibm_fez`, `quantinuum:H2-1`, any `qumo_hardware` entry-point provider | real hardware or cloud emulators | billable: needs `--allow-remote` to plan, dry run by default |

The modality noise model is vendor-neutral and documented in
[docs/HARDWARE_PLUGINS.md](../../docs/HARDWARE_PLUGINS.md#vendor-neutral-noisy-emulator-noisemodality).
Real submission needs all of `--allow-remote --submit --yes --max-cost X`,
and `qumo_hardware.runner.run` re-checks the cost cap and the confirmation
(QM-E956 / QM-E957) whatever this script does. Without `--submit` a remote
backend gets a dry run that reports the provider's cost estimate (for example
`57.8 HQC` for the 8-node ring on `quantinuum:H2-1`) and submits nothing.

## Where things run: honest GPU notes

* The GPU stage runs on the GPU only when `qumo_py` was built with GPU support
  (`make py-gpu`, then `PYTHONPATH=build/gpu:plugins/hardware`), a device is
  usable, and the circuit has at least `min_qubits_auto` qubits (16 on Apple,
  20 elsewhere) or `--force-gpu` is given. Otherwise it runs on the CPU and
  the report says why; the run never fails over placement.
* Below about 12-16 qubits the CPU wins: kernel launches and transfers cost
  more than the arithmetic. The GPU wins at width (26 qubits in 0.42 s on an
  Apple M5, 7.8x QUMO's own CPU simulator) and on batched scans. The GPU
  default precision is fp32; Apple GPUs have no fp64, so a plan asking for
  fp64 there is rejected by the validator.
* Phase 1 limits: the GPU stage uses `qumo_py.sample_device` (sampled
  energies with standard errors); exact expectations, adjoint gradients and
  kernel matrices run on the CPU. Phase 2 adds `expectation_device`,
  `batch_value_and_gradient` and `kernel_matrix` to `qumo_py`, and job pinning
  to the orchestrator; the report's `phase` field says which applies.
* QAOA's angles are shared across gates, so per-gate parameter shift is not a
  valid gradient; the CPU stage uses gradient-free optimisers (or adjoint in
  phase 2).
* Run times depend on the machine and on the `qumo_py` build. Measured on an
  Apple M5 CPU (2026-10-14, current `build/qumo_py.so`): the default 16-node
  run takes 30-40 s (an earlier build took up to 2.5 minutes), almost all in
  the noisy QPU emulation: after routing on the 4x4 grid the circuit measures
  some qubits early and reuses them, so the emulator falls back to per-shot
  trajectories (2 000 shots of a 16-qubit state vector). `--small` finishes in
  under a second; `--size large` (20 qubits, 5x5 grid) takes 3.5-6 minutes,
  most of it in the noisy emulator. With a GPU build
  (`PYTHONPATH=build/gpu:plugins/hardware`) the 16-node landscape scan runs
  as `gpu:statevector:fp32` (257 sampled evaluations in about 2 s on an M5).
* The QAOA checks follow the plan the selector returned, not a fixed setting.
  The ideal approximation ratio is compared with the known guarantee for the
  chosen depth p (rings: (2p+1)/(2p+2); 3-regular graphs: 0.6924 at p=1,
  0.7559 at p=2; deeper plans use the deepest tabulated bound, and SPSA plans
  use the p=1 bound because a stochastic optimiser with a small budget does
  not reach optimal angles). The QPU must sample the brute-force optimum only
  when the ideal distribution puts at least 20 expected shots on it;
  otherwise its best sample must reach the ideal mean cut. Every schema-valid
  QAOA plan (depth 1-6 x 5 optimisers x 4 budgets) passes on the bundled
  8-ring and 16-node problems (`--size large` caps the depth at 2).

## How this compares

| Framework | Heterogeneous pattern | Who picks the device | Who picks the algorithm |
|---|---|---|---|
| CUDA-Q (`mqpu`, `nvidia-mqpu`) | asynchronous kernels over several GPU-simulated QPUs | the programmer (`qpu_id`) | the programmer |
| Qiskit Runtime sessions / batches | classical loop next to a QPU, jobs grouped for queueing | the programmer (backend) | the programmer |
| Amazon Braket Hybrid Jobs | containerised classical code with priority QPU access | the programmer (`device`) | the programmer |
| PennyLane | swap `default.qubit` / `lightning.gpu` / hardware plugins | the programmer (`qml.device`) | the programmer |
| This example | GPU, CPU and QPU stages in one run | QUMO, from GPU status and thresholds, and it explains the placement | a model (or the offline rules), validated against an allow-list |

Sources: <https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/mqpusims.html>,
<https://quantum.cloud.ibm.com/docs/en/guides/execution-modes>,
<https://docs.aws.amazon.com/braket/latest/developerguide/braket-jobs.html>,
<https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html>.

## Mojo twin

`orchestrate_cpu_gpu_qpu.mojo` runs the same pattern natively with QUMO's
orchestrator: rule-based selection, 64 estimate jobs for an 8x8 QAOA grid
scheduled by HEFT over `[cpu, gpu]` (the schedule explains every placement),
Nelder-Mead refinement on the CPU, then one sampling job on a QPU-only pool
and the bound payload written to `build/hetero/payload.qasm`.

```bash
.venv/bin/mojo run -I . examples/heterogeneous/orchestrate_cpu_gpu_qpu.mojo             # ~7 s
.venv/bin/mojo run -I . -D QUMO_GPU examples/heterogeneous/orchestrate_cpu_gpu_qpu.mojo # GPU jobs: gpu:statevector:fp32
python -m qumo_hardware run build/hetero/payload.qasm -p local -b 'qumo-sim:grid:4x4/sc_cz+noise:superconducting'
```

## Driving it from an agent

[AGENT_PLAYBOOK.md](AGENT_PLAYBOOK.md) runs the same flow from Claude Code
(or any MCP client) over `build/qumo mcp`.

## Files

| File | Role |
|---|---|
| `ai_gpu_qpu.py` | command line, context pack, checks, report |
| `planner.py` | catalogue, plan schema, validation, offline rules, model loop |
| `llm_client.py` | OpenAI-compatible client (urllib), retries, redaction, JSON extraction |
| `llm_profiles.json` | providers and presets |
| `kernels.py` | problem loading, circuit builders (tool first, Python twins), Hamiltonians |
| `stages.py` | GPU / CPU / QPU stages per algorithm, placement rules, QPU gate |
| `problems/*.json` | problem descriptions |
| `test_ai_gpu_qpu.py` | offline tests with a fake OpenAI-compatible server (stdlib `http.server`) |
| `orchestrate_cpu_gpu_qpu.mojo` | Mojo twin |

Tests: `bench/.venv/bin/python -m pytest -q examples/heterogeneous` (no
network; the fake server covers valid answers, invalid JSON, schema
violations, non-allow-listed algorithms, refusals, truncation, timeouts,
401/429/500, structured-output downgrade, remote downgrade, the injection
problem, replay, and blocked `openai` / `anthropic` imports).
