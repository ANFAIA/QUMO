# QUMO build & test entry points.
MOJO ?= .venv/bin/mojo
PY ?= bench/.venv/bin/python
# the RL tests live in rl/ (make test runs the fast ones once, via make test-rl; make test-rl-slow runs the rest)
TESTS := $(wildcard tests/test_*.mojo)

.PHONY: all help setup setup-py setup-py-locked test cli py bench-sv bench-neutral bench-charts clean examples examples-all examples-gpu test-py test-python install-py

all: cli py

# The targets a new user needs, in the order to run them; `make help` marks them with *.
ESSENTIAL := setup cli py test examples

help:  ## list the documented targets (* = the ones a new user needs)
	@echo "QUMO make targets (* = start here: make setup, then make cli; docs/GETTING_STARTED.md)"; echo
	@awk -v ess="$(ESSENTIAL)" 'BEGIN { n = split(ess, e, " "); for (i = 1; i <= n; i++) mark[e[i]] = 1; FS = ":.*## " } \
		/^[a-zA-Z0-9_-]+:.*## / { printf "  %s %-20s %s\n", (($$1 in mark) ? "*" : " "), $$1, $$2 }' $(MAKEFILE_LIST)

# Mojo changes dialect between releases and QUMO pins the one it is written in
# (docs/MOJO_GUIDE.md); bump these together with the guide.
MOJO_VERSION ?= 1.2.0
MAX_VERSION ?= 26.7.0

setup:  ## create .venv, or update an existing one, to the pinned Mojo toolchain + MAX runtime (parallelize)
	[ -x .venv/bin/python ] || uv venv .venv --python 3.12
	uv pip install --python .venv/bin/python "mojo==$(MOJO_VERSION)" "max-mojo-libs==$(MAX_VERSION)" "max-core==$(MAX_VERSION)"

setup-py:  ## bench/.venv: Python 3.12 + latest Qiskit, Aer, TKET, Cirq, Stim, PyMatching (examples, benchmarks, plugin tests); setup-py-locked pins them
	[ -x bench/.venv/bin/python ] || uv venv bench/.venv --python 3.12
	uv pip install --python bench/.venv/bin/python -U qiskit qiskit-aer qiskit-ibm-runtime pytket pytket-qiskit \
		cirq-core ply stim pymatching numpy scipy matplotlib altair vl-convert-python psutil pytest nbformat "openqasm3[parser]"
	uv pip install --python bench/.venv/bin/python -U "./plugins/hardware[all]"

setup-py-locked:  ## bench/.venv from bench/requirements.lock: the exact versions of the last measured run (reproducible; setup-py takes the latest)
	[ -x bench/.venv/bin/python ] || uv venv bench/.venv --python 3.12
	uv pip install --python bench/.venv/bin/python -r bench/requirements.lock
	uv pip install --python bench/.venv/bin/python "./plugins/hardware[all]"

test:  ## run every Mojo test file (tests/ and, once, the fast RL tests of rl/ through make test-rl)
	@fail=0; for t in $(TESTS); do \
		log=$$(mktemp); $(MOJO) run -I . $$t > $$log 2>&1; \
		out=$$(grep -v Crashpad $$log | tail -1); \
		case "$$out" in *OK*) echo "PASS $$t";; *) echo "FAIL $$t: $$out"; grep -v Crashpad $$log | tail -25; fail=1;; esac; \
		rm -f $$log; \
	done; $(MAKE) --no-print-directory test-rl || fail=1; exit $$fail

cli:  ## native CLI + MCP server (build/qumo)
	@mkdir -p build
	$(MOJO) build -I . -o build/qumo cli/main.mojo

py:  ## CPython extension (build/qumo_py.so)
	@mkdir -p build
	$(MOJO) build -I . --emit shared-lib -o build/qumo_py.so python/qumo_py.mojo

bench-sv:  ## state-vector throughput vs Qiskit Aer
	@mkdir -p build
	$(MOJO) build -I . -o build/bench_sv bench/bench_sv.mojo && ./build/bench_sv 26
	$(PY) bench/bench_sv_aer.py 26

bench-neutral: py  ## hardware-neutral compiler benchmark (QUMO, Qiskit, TKET, Cirq)
	$(PY) bench/neutral/run_neutral.py --circuits bench/neutral/circuits --out bench/results/neutral.jsonl
	$(PY) bench/neutral/report.py bench/results/neutral.jsonl > docs/NEUTRAL_BENCHMARK_REPORT.md

bench-charts:  ## charts and summary of compile time and memory per SDK (bench/results/neutral_rss.jsonl)
	$(PY) bench/charts/make_charts.py --data bench/results/neutral_rss.jsonl --out bench/results/charts

# ---------------------------------------------------------------- AI agents (docs/AGENTS.md)
AGENTS_PY ?= agents/.venv/bin/python
.PHONY: setup-agents test-agents agents-doctor

setup-agents:  ## agents/.venv: deepagents, LangChain OpenAI + MCP adapters (the QUMO agent layer)
	[ -x $(AGENTS_PY) ] || uv venv agents/.venv --python 3.12
	uv pip install --python $(AGENTS_PY) -e "agents[test]"

test-agents:  ## agent layer tests (offline; the MCP tests need build/qumo from `make cli`)
	cd agents && .venv/bin/python -m pytest -q tests

agents-doctor:  ## check keys, the MCP server and the back-end CLIs the agents need
	bin/qumo-agents doctor

# ---------------------------------------------------------------- AI transpiler (docs/AI_TRANSPILER.md)
# Models live in models/ai_transpiler/{synth,route}/ with an INDEX.json registry (sha256 per file); the Mojo
# loader re-checks an FNV-1a checksum. Training is the independent native Mojo package in rl/ (qrl, plus the
# qumo_bridge glue; rl/README.md, rl/Makefile): the training targets run no Python, no Rust and no torch
# (`make check-pure-mojo` enforces it). The targets below delegate to rl/Makefile. `make test` runs the fast RL
# tests once, through `make test-rl` (rl/tests/test_rl_*.mojo and rl/qumo_bridge/tests/test_bridge_*.mojo).
AI_BUILD ?= build/ai_train
AI_CORES ?= 4
# two lanes share the same pool threads and each runs about 0.69x slower (rl/docs/MOJO_PORT.md section 3, rule 8)
AI_LANES ?= 1
# hard wall-clock cap of the whole synthesis campaign (the shipped budgets total 22,200 s)
AI_SYNTH_BUDGET_S ?= 28800
# behaviour-cloning wall-clock budget per round (train_route stops after the epoch that crosses it)
AI_ROUTE_BC_BUDGET_S ?= 1800
AI_ROUTE_WORK ?= $(AI_BUILD)/route
AI_ROUTE_ID ?= route_mlp_h64_v1
AI_BENCH_ARGS ?= --sdks qumo,qumo-ai,qiskit,tket --levels default,max --max-qubits 14
AI_TOOLS := route_tool train_route export_route train_synth export_synth eval_synth eval_blocks train_all models_verify train_route_ppo make_fixtures

# MOJO, AI_BUILD and the work dir as rl/Makefile (which runs in rl/) must see them
RL_ARGS = --no-print-directory -C rl QUMO_ROOT=$(CURDIR) \
	MOJO=$(if $(findstring /,$(MOJO)),$(abspath $(MOJO)),$(MOJO)) AI_BUILD=$(abspath $(AI_BUILD)) \
	MODELS=$(CURDIR)/models/ai_transpiler AI_ROUTE_WORK=$(abspath $(AI_ROUTE_WORK)) AI_CORES=$(AI_CORES) \
	AI_LANES=$(AI_LANES) AI_SYNTH_BUDGET_S=$(AI_SYNTH_BUDGET_S) AI_ROUTE_BC_BUDGET_S=$(AI_ROUTE_BC_BUDGET_S) \
	AI_ROUTE_ID=$(AI_ROUTE_ID)

.PHONY: models-verify train-ai train-ai-synth train-ai-route train-ai-smoke train-ai-route-ppo ai-tools bench-ai \
	test-rl test-rl-slow check-pure-mojo check-independence bench-rl

# one tool per main (core mains in rl/drivers, QUMO glue mains in rl/qumo_bridge/drivers); kept here so that
# `make $(AI_BUILD)/bin/<tool>` works from the QUMO root (the e2e test builds missing tools that way)
vpath %.mojo rl/drivers/synth rl/drivers/route rl/drivers/common rl/qumo_bridge/drivers
AI_SRC := $(shell find qumo rl/qrl rl/qumo_bridge -name '*.mojo' ! -path '*/tests/*' 2>/dev/null)

$(AI_BUILD)/bin/%: %.mojo $(AI_SRC)
	@mkdir -p $(dir $@)
	$(MOJO) build -I . -I rl -o $@ $<

ai-tools: $(addprefix $(AI_BUILD)/bin/,$(AI_TOOLS))  ## build every training tool into $(AI_BUILD)/bin

models-verify:  ## sha256 of every model file under models/ai_transpiler (QUMO_MODEL_DIR) against its INDEX.json
	@$(MAKE) $(RL_ARGS) models-verify

train-ai:  ## retrain the AI-transpiler models (hours; no Python, no Rust, no torch; not part of make test)
	@$(MAKE) $(RL_ARGS) train-ai

train-ai-synth:  ## synthesis models of rl/drivers/synth/campaign.json (cap AI_SYNTH_BUDGET_S)
	@$(MAKE) $(RL_ARGS) train-ai-synth

train-ai-route:  ## learned routing policy: teacher data + BC + one DAgger round (rl/drivers/route/README.md)
	@$(MAKE) $(RL_ARGS) train-ai-route

train-ai-smoke:  ## ~5 min end-to-end into $(AI_BUILD)/smoke (synth learns and beats greedy, router BC, export + verify); never writes models/
	@$(MAKE) $(RL_ARGS) train-ai-smoke

train-ai-route-ppo:  ## optional router PPO fine-tune: make train-ai-route-ppo INIT=<router dir> BUDGET_S=<s>; never part of train-ai
	@$(MAKE) $(RL_ARGS) train-ai-route-ppo INIT=$(if $(INIT),$(abspath $(INIT))) BUDGET_S=$(BUDGET_S)

test-rl:  ## fast RL tests: rl/ independence check, core tests (-I rl only) and bridge tests; wall time per file, each under 30 s
	@$(MAKE) $(RL_ARGS) test test-bridge

test-rl-slow:  ## statistical and end-to-end RL tests (core *_slow, smoke; bridge e2e) and test_rl_tensor with -D ASSERT=all
	@$(MAKE) $(RL_ARGS) test-slow test-bridge-slow

check-independence:  ## fails if anything under rl/qrl, rl/drivers or rl/tests imports qumo.* (rl/check_independence.sh)
	@$(MAKE) $(RL_ARGS) check-independence

check-pure-mojo:  ## fails if the training path invokes or names python/pip/uv/torch/numpy/qiskit/twisterl/rustworkx/safetensors/cargo/rustc (CHECK_PURE_SMOKE=1 also runs train-ai-smoke in a clean environment)
	rl/check_pure_mojo.sh

bench-rl:  ## performance gates G1-G10 of rl/docs/MOJO_PORT.md section 15, one JSON line each (rl/bench/bench_rl.mojo; run on an otherwise idle machine)
	@$(MAKE) $(RL_ARGS) bench-rl

bench-ai: py  ## hardware-neutral benchmark of the AI transpiler (qumo-ai vs qumo, Qiskit, TKET; statevector-verified)
	QUMO_MODEL_DIR=$(CURDIR)/models/ai_transpiler $(PY) bench/neutral/run_neutral.py --circuits bench/neutral/circuits $(AI_BENCH_ARGS) --out bench/results/ai_neutral.jsonl
	$(PY) bench/neutral/report.py bench/results/ai_neutral.jsonl --ai > bench/results/ai_neutral.md

# ---------------------------------------------------------------- GPU (qumo.gpu, docs/GPU.md)
# Default targets never compile GPU code. The *-gpu targets build with
# -D QUMO_GPU for the host's accelerator: CUDA (NVIDIA) / ROCm (AMD) drivers on
# Linux, or Xcode + `xcodebuild -downloadComponent MetalToolchain` on macOS.
GPU_TARGETS ?= sm_75 sm_80 sm_90 gfx942 metal:4
GPU_CHECK_SRCS := tests/test_gpu.mojo tests/test_gpu_frame.mojo bench/bench_gpu.mojo
# extra -D flags for gpu-check, e.g. GPU_CHECK_DEFINES="-D QUMO_GPU_FP64" (fp64 as the build default;
# accepted and ignored on metal:4). GPU precision: fp32 by default on every GPU (docs/GPU.md).
GPU_CHECK_DEFINES ?=
GPU_TESTS := $(wildcard tests/test_gpu*.mojo)

.PHONY: gpu-check test-gpu test-gpu-regbits cli-gpu py-gpu bench-gpu

gpu-check:  ## cross-compile every GPU kernel for NVIDIA sm_75 (T4) / sm_80 / sm_90, AMD gfx942, Apple metal:4 (no GPU needed; GPU_CHECK_DEFINES adds -D flags)
	@fail=0; for a in $(GPU_TARGETS); do \
		d=build/gpucheck/$$(echo $$a | tr ':' '_'); rm -rf $$d; mkdir -p $$d; \
		for f in $(GPU_CHECK_SRCS); do \
			b=$$(basename $$f .mojo); \
			if $(MOJO) build -I . -D QUMO_GPU $(GPU_CHECK_DEFINES) --emit asm --target-accelerator=$$a -o $$d/$$b.s $$f > $$d/$$b.log 2>&1; then \
				k=$$(ls $$d | grep "^$${b}_" | grep -c "_gk_"); \
				if [ $$k -gt 0 ]; then echo "PASS $$a $$f ($$k kernels)"; \
				else echo "FAIL $$a $$f: no kernels emitted (gating broken?)"; fail=1; fi; \
			else echo "FAIL $$a $$f (log: $$d/$$b.log)"; grep -v Crashpad $$d/$$b.log | head -8; fail=1; fi; \
		done; \
	done; exit $$fail

# The device suites must run: a machine whose GPU is not usable fails
# test-gpu instead of passing on the emulation suites alone
# (QUMO_GPU_REQUIRE_DEVICE=0 to allow the skip).
QUMO_GPU_REQUIRE_DEVICE ?= 1

test-gpu:  ## GPU test suites on the real device (+ emulation suites, both register-blocking widths)
	@fail=0; for t in $(GPU_TESTS); do \
		out=$$(QUMO_GPU_REQUIRE_DEVICE=$(QUMO_GPU_REQUIRE_DEVICE) $(MOJO) run -I . -D QUMO_GPU $$t 2>&1 | grep -v Crashpad | tail -1); \
		case "$$out" in *OK*) echo "PASS $$t";; *) echo "FAIL $$t: $$out"; fail=1;; esac; \
	done; $(MAKE) --no-print-directory test-gpu-regbits || fail=1; \
	if [ "$$(uname -s)" = Darwin ]; then \
		out=$$(QUMO_GPU_REQUIRE_DEVICE=$(QUMO_GPU_REQUIRE_DEVICE) $(MOJO) run -I . -D QUMO_GPU -D QUMO_GPU_TILE_SMEM_DYNAMIC=1 tests/test_gpu.mojo 2>&1 | grep -v Crashpad | tail -1); \
		case "$$out" in *OK*) echo "PASS tests/test_gpu.mojo (device, dynamic shared memory as on NVIDIA / AMD)";; \
		*) echo "FAIL tests/test_gpu.mojo (device, QUMO_GPU_TILE_SMEM_DYNAMIC=1): $$out"; fail=1;; esac; \
	fi; exit $$fail

# The default register-blocking width is per target (QUMO_GPU_REG_BITS: 4 on
# Apple, 3 elsewhere): run the emulated engine with both, on every machine.
test-gpu-regbits:  ## tests/test_gpu.mojo in emulation with 8 and 16 amplitudes per thread
	@fail=0; for rb in 3 4; do \
		out=$$($(MOJO) run -I . -D QUMO_GPU_REG_BITS=$$rb tests/test_gpu.mojo 2>&1 | grep -v Crashpad | tail -1); \
		case "$$out" in *OK*) echo "PASS tests/test_gpu.mojo (emulation, QUMO_GPU_REG_BITS=$$rb)";; \
		*) echo "FAIL tests/test_gpu.mojo (emulation, QUMO_GPU_REG_BITS=$$rb): $$out"; fail=1;; esac; \
	done; exit $$fail

cli-gpu:  ## CLI + MCP server with GPU dispatch (build/gpu/qumo)
	@mkdir -p build/gpu
	$(MOJO) build -I . -D QUMO_GPU -o build/gpu/qumo cli/main.mojo

py-gpu:  ## CPython extension with GPU dispatch (build/gpu/qumo_py.so)
	@mkdir -p build/gpu
	$(MOJO) build -I . -D QUMO_GPU --emit shared-lib -o build/gpu/qumo_py.so python/qumo_py.mojo

# BENCH_GPU_ARGS: "nmax reps [families nmin cpu_max | crossover nmin | frames shots]" (bench/bench_gpu.mojo)
BENCH_GPU_ARGS ?= 26

bench-gpu:  ## GPU vs CPU benchmarks (state-vector families, dispatch crossover, QEC Pauli frames)
	@mkdir -p build/gpu
	$(MOJO) build -I . -D QUMO_GPU -o build/gpu/bench_gpu bench/bench_gpu.mojo && ./build/gpu/bench_gpu $(BENCH_GPU_ARGS)

# ---------------------------------------------------------------- examples (examples/README.md)
examples:  ## fast-tier examples, verified by their CHECK lines
	$(PY) examples/run_examples.py --tier fast

examples-all:  ## every example tier
	$(PY) examples/run_examples.py --tier all

examples-gpu: cli-gpu py-gpu  ## fast tier with GPU builds
	$(PY) examples/run_examples.py --tier fast --gpu

install-py: py  ## install the Python package `qumo` (editable) into the environment of $(PY); docs/PYTHON.md
	uv pip install --python $(PY) -e .

test-python: py  ## tests of the Python package `qumo` (python/tests; needs numpy and pytest in $(PY))
	PYTHONPATH=python $(PY) -m pytest -q python/tests

# two pytest runs: python/tests and plugins/hardware/tests both have a conftest.py and a test_compile.py
test-py:  ## every Python test: the `qumo` package, examples runner, heterogeneous example, hardware plugins
	PYTHONPATH=python $(PY) -m pytest -q python/tests
	PYTHONPATH=build:plugins/hardware $(PY) -m pytest -q examples/heterogeneous examples/test_run_examples.py plugins/hardware/tests

clean:
	rm -rf build
