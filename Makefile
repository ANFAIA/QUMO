# QUMO build & test entry points.
MOJO ?= .venv/bin/mojo
PY ?= bench/.venv/bin/python
# make test skips the slow statistical and end-to-end RL files (make test-rl-slow runs them)
TESTS := $(filter-out %_slow.mojo tests/test_rl_smoke.mojo tests/test_rl_e2e.mojo,$(wildcard tests/test_*.mojo))

.PHONY: all setup setup-py test cli py bench-sv bench-neutral bench-charts benchpress clean examples examples-all examples-gpu test-py

all: cli py

# Mojo changes dialect between releases and QUMO pins the one it is written in
# (docs/MOJO_GUIDE.md); bump these together with the guide.
MOJO_VERSION ?= 1.2.0
MAX_VERSION ?= 26.7.0

setup:  ## create .venv, or update an existing one, to the pinned Mojo toolchain + MAX runtime (parallelize)
	[ -x .venv/bin/python ] || uv venv .venv --python 3.12
	uv pip install --python .venv/bin/python "mojo==$(MOJO_VERSION)" "max-mojo-libs==$(MAX_VERSION)" "max-core==$(MAX_VERSION)"

setup-py:  ## bench/.venv: Python 3.12 + latest Qiskit, Aer, TKET, Cirq, Stim, PyMatching (examples, benchmarks, plugin tests)
	[ -x bench/.venv/bin/python ] || uv venv bench/.venv --python 3.12
	uv pip install --python bench/.venv/bin/python -U qiskit qiskit-aer qiskit-ibm-runtime pytket pytket-qiskit \
		cirq-core ply stim pymatching numpy scipy matplotlib altair vl-convert-python psutil pytest nbformat "openqasm3[parser]"
	uv pip install --python bench/.venv/bin/python -U "./plugins/hardware[all]"

test:  ## run every Mojo test file
	@fail=0; for t in $(TESTS); do \
		out=$$($(MOJO) run -I . $$t 2>&1 | grep -v Crashpad | tail -1); \
		case "$$out" in *OK*) echo "PASS $$t";; *) echo "FAIL $$t: $$out"; fail=1;; esac; \
	done; exit $$fail

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
# loader re-checks an FNV-1a checksum. Training is native Mojo (qumo/rl, training/ai_transpiler): the training
# targets run no Python, no Rust and no torch (`make check-pure-mojo` enforces it). None of this is part of
# `make test` except the fast `tests/test_rl_*.mojo` files.
AI_BUILD ?= build/ai_train
AI_CORES ?= 4
# two lanes share the same pool threads and each runs about 0.69x slower (MOJO_PORT.md section 3, rule 8)
AI_LANES ?= 1
# hard wall-clock cap of the whole synthesis campaign (the shipped budgets total 22,200 s)
AI_SYNTH_BUDGET_S ?= 28800
# behaviour-cloning wall-clock budget per round (train_route stops after the epoch that crosses it)
AI_ROUTE_BC_BUDGET_S ?= 1800
AI_ROUTE_WORK ?= $(AI_BUILD)/route
AI_ROUTE_ID ?= route_mlp_h64_v1
AI_BENCH_ARGS ?= --sdks qumo,qumo-ai,qiskit,tket --levels default,max --max-qubits 14
AI_TOOLS := route_tool train_route export_route train_synth export_synth eval_synth eval_blocks train_all models_verify train_route_ppo make_fixtures

.PHONY: models-verify train-ai train-ai-synth train-ai-route train-ai-smoke train-ai-route-ppo ai-tools bench-ai \
	test-rl test-rl-slow check-pure-mojo bench-rl

vpath %.mojo training/ai_transpiler/synth training/ai_transpiler/route training/ai_transpiler/common
AI_SRC := $(shell find qumo -name '*.mojo' 2>/dev/null)

# one tool per main file; vpath finds the main in synth/, route/ or common/
$(AI_BUILD)/bin/%: %.mojo $(AI_SRC)
	@mkdir -p $(dir $@)
	$(MOJO) build -I . -o $@ $<

ai-tools: $(addprefix $(AI_BUILD)/bin/,$(AI_TOOLS))  ## build every training tool into $(AI_BUILD)/bin

models-verify: $(AI_BUILD)/bin/models_verify  ## sha256 of every model file under models/ai_transpiler (QUMO_MODEL_DIR) against its INDEX.json
	@$<

train-ai: train-ai-synth train-ai-route  ## retrain the AI-transpiler models (hours; no Python, no Rust, no torch; not part of make test)
	@$(MAKE) --no-print-directory models-verify

train-ai-synth: $(addprefix $(AI_BUILD)/bin/,train_all train_synth export_synth eval_synth)  ## synthesis models of training/ai_transpiler/synth/campaign.json (cap AI_SYNTH_BUDGET_S)
	@pids=; for k in $$(seq 1 $(AI_LANES)); do \
	  $(AI_BUILD)/bin/train_all --campaign training/ai_transpiler/synth/campaign.json --lane $$k/$(AI_LANES) \
	      --cores $(AI_CORES) --budget-cap-s $(AI_SYNTH_BUDGET_S) --work $(AI_BUILD)/synth & pids="$$pids $$!"; done; \
	  rc=0; for p in $$pids; do wait $$p || rc=1; done; exit $$rc

train-ai-route: $(addprefix $(AI_BUILD)/bin/,route_tool train_route export_route)  ## learned routing policy: teacher data + BC + one DAgger round (training/ai_transpiler/route/README.md)
	@mkdir -p $(AI_ROUTE_WORK)/stage
	$(AI_BUILD)/bin/route_tool gen-jobs 31 1000 30 0.5 $(AI_ROUTE_WORK)/jobs0.json
	$(AI_BUILD)/bin/route_tool gen-jobs 32 1000 30 0.5 $(AI_ROUTE_WORK)/jobs1.json
	ROUTE_TOOL_CAP=96 $(AI_BUILD)/bin/route_tool gen $(AI_ROUTE_WORK)/jobs0.json $(AI_ROUTE_WORK)/d0.bin - 1.0 401 heuristic 16
	$(AI_BUILD)/bin/train_route --data $(AI_ROUTE_WORK)/d0.bin --out $(AI_ROUTE_WORK)/run0 --hidden 64 --tau 0.5 --epochs 30 --seed 0 --budget-s $(AI_ROUTE_BC_BUDGET_S)
	$(AI_BUILD)/bin/export_route $(AI_ROUTE_WORK)/run0 --root $(AI_ROUTE_WORK)/stage --id round0 --default
	ROUTE_TOOL_CAP=96 $(AI_BUILD)/bin/route_tool gen $(AI_ROUTE_WORK)/jobs1.json $(AI_ROUTE_WORK)/d1.bin $(AI_ROUTE_WORK)/stage/round0 0.3 402 heuristic 16
	$(AI_BUILD)/bin/train_route --data $(AI_ROUTE_WORK)/d0.bin $(AI_ROUTE_WORK)/d1.bin --out $(AI_ROUTE_WORK)/run1 --hidden 64 --tau 0.5 --epochs 30 --seed 0 --budget-s $(AI_ROUTE_BC_BUDGET_S)
	$(AI_BUILD)/bin/export_route $(AI_ROUTE_WORK)/run1 --root models/ai_transpiler/route --id $(AI_ROUTE_ID) --default

train-ai-smoke: $(addprefix $(AI_BUILD)/bin/,train_synth export_synth eval_synth route_tool train_route export_route models_verify)  ## ~5 min end-to-end into $(AI_BUILD)/smoke (synth learns and beats greedy, router BC, export + verify); never writes models/
	training/ai_transpiler/smoke.sh $(AI_BUILD)

train-ai-route-ppo: $(addprefix $(AI_BUILD)/bin/,train_route_ppo export_route)  ## optional router PPO fine-tune: make train-ai-route-ppo INIT=<router dir> BUDGET_S=<s>; never part of train-ai
	@test -n "$(INIT)" -a -n "$(BUDGET_S)" || { echo "usage: make train-ai-route-ppo INIT=<router dir> BUDGET_S=<seconds>"; exit 2; }
	$(AI_BUILD)/bin/train_route_ppo --init $(INIT) --out $(AI_ROUTE_WORK)/ppo_run --budget-s $(BUDGET_S)

test-rl:  ## fast RL tests only (tests/test_rl_*.mojo except *_slow, smoke, e2e); prints the wall time per file, each must stay under 30 s
	@fail=0; for t in $(filter-out %_slow.mojo tests/test_rl_smoke.mojo tests/test_rl_e2e.mojo,$(wildcard tests/test_rl_*.mojo)); do \
	  s=$$(date +%s); out=$$($(MOJO) run -I . $$t 2>&1 | grep -v Crashpad | tail -1); e=$$(( $$(date +%s) - s )); \
	  case "$$out" in *OK*) echo "PASS $$t ($${e}s)";; *) echo "FAIL $$t: $$out"; fail=1;; esac; \
	done; exit $$fail

test-rl-slow:  ## statistical and end-to-end RL tests (*_slow, smoke, e2e) and test_rl_tensor with -D ASSERT=all
	@fail=0; for t in $(wildcard tests/test_rl_*_slow.mojo) tests/test_rl_smoke.mojo tests/test_rl_e2e.mojo; do \
	  s=$$(date +%s); out=$$($(MOJO) run -I . $$t 2>&1 | grep -v Crashpad | tail -1); e=$$(( $$(date +%s) - s )); \
	  case "$$out" in *OK*) echo "PASS $$t ($${e}s)";; *) echo "FAIL $$t: $$out"; fail=1;; esac; \
	done; \
	out=$$($(MOJO) run -D ASSERT=all -I . tests/test_rl_tensor.mojo 2>&1 | grep -v Crashpad | tail -1); \
	case "$$out" in *OK*) echo "PASS tests/test_rl_tensor.mojo (-D ASSERT=all)";; *) echo "FAIL tests/test_rl_tensor.mojo: $$out"; fail=1;; esac; exit $$fail

check-pure-mojo:  ## fails if the training path invokes or names python/pip/uv/torch/numpy/qiskit/twisterl/rustworkx/safetensors/cargo/rustc (CHECK_PURE_SMOKE=1 also runs train-ai-smoke in a clean environment)
	training/ai_transpiler/check_pure_mojo.sh

bench-rl:  ## performance gates G1-G10 of MOJO_PORT.md section 15, one JSON line each (bench/bench_rl.mojo; run on an otherwise idle machine)
	@mkdir -p build
	$(MOJO) build -I . -o build/bench_rl bench/bench_rl.mojo && ./build/bench_rl

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

test-py:  ## Python tests: examples runner, heterogeneous example, hardware plugins
	PYTHONPATH=build:plugins/hardware $(PY) -m pytest -q examples/heterogeneous examples/test_run_examples.py plugins/hardware/tests

clean:
	rm -rf build
