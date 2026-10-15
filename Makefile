# QUMO build & test entry points.
MOJO ?= .venv/bin/mojo
PY ?= bench/.venv/bin/python
TESTS := $(wildcard tests/test_*.mojo)

.PHONY: all setup setup-py test cli py bench-sv bench-neutral benchpress clean examples examples-all examples-gpu test-py

all: cli py

setup:  ## create .venv with the Mojo toolchain + MAX runtime (parallelize)
	uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python mojo max-mojo-libs max-core

setup-py:  ## bench/.venv: Python 3.12 + latest Qiskit, Aer, TKET, Cirq, Stim, PyMatching (examples, benchmarks, plugin tests)
	[ -x bench/.venv/bin/python ] || uv venv bench/.venv --python 3.12
	uv pip install --python bench/.venv/bin/python -U qiskit qiskit-aer qiskit-ibm-runtime pytket pytket-qiskit \
		cirq-core stim pymatching numpy scipy matplotlib psutil pytest nbformat
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

# ---------------------------------------------------------------- AI transpiler (docs/AI_TRANSPILER.md)
# Models live in models/ai_transpiler/{synth,route}/ with an INDEX.json
# registry (sha256 per file); the Mojo loader re-checks an FNV-1a checksum.
# Training needs a separate venv with qiskit-gym 0.4.1 / twisterl 0.5.1 / torch
# (AI_TRAIN_PY); it is never part of `make test`.
AI_TRAIN_PY ?= training/ai_transpiler/.venv/bin/python
# hard wall-clock caps: all synthesis models 8 h, the router 3 h
AI_SYNTH_BUDGET_S ?= 28800
AI_ROUTE_BUDGET_S ?= 10800
AI_BENCH_ARGS ?= --sdks qumo,qumo-ai,qiskit,tket --levels default,max --max-qubits 14

.PHONY: models-verify train-ai train-ai-synth train-ai-route bench-ai

define MODELS_VERIFY_PY
import hashlib, json, os, sys
root = os.environ.get("QUMO_MODEL_DIR") or "models/ai_transpiler"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
def pairs(node, base, out):
    # collect (path, sha256) pairs from any INDEX.json layout:
    # {"dir": d, "sha256": h} (h of d/weights.bin, the qumo registry layout),
    # {"path"|"file": p, "sha256": h}, {"dir": d, "sha256": {name: h}},
    # {"files": {p: h}} or {"files": [{"path": p, "sha256": h}]}
    if isinstance(node, list):
        for x in node:
            pairs(x, base, out)
        return
    if not isinstance(node, dict):
        return
    d = node.get("dir") or node.get("path_dir") or ""
    h = node.get("sha256")
    p = node.get("path") or node.get("file")
    if isinstance(h, str) and isinstance(p, str):
        out.append((os.path.join(base, p), h))
    elif isinstance(h, str) and isinstance(d, str) and d:
        # qumo INDEX.json entry {id, kind, ..., dir, sha256 (of dir/weights.bin), bytes}
        out.append((os.path.join(base, d, node.get("weights_file", "weights.bin")), h))
    elif isinstance(h, dict):
        for k, v in h.items():
            out.append((os.path.join(base, d, k), v))
    f = node.get("files")
    if isinstance(f, dict):
        for k, v in f.items():
            if isinstance(v, str):
                out.append((os.path.join(base, d, k), v))
            elif isinstance(v, dict) and isinstance(v.get("sha256"), str):
                out.append((os.path.join(base, d, v.get("path", k)), v["sha256"]))
    elif isinstance(f, list):
        for x in f:
            if isinstance(x, dict) and isinstance(x.get("sha256"), str):
                out.append((os.path.join(base, d, x.get("path") or x.get("file")), x["sha256"]))
    for k, v in node.items():
        if k not in ("sha256", "files") and isinstance(v, (dict, list)):
            pairs(v, base, out)
bad, ok, idx = 0, 0, 0
if not os.path.isdir(root):
    print("models-verify: no model directory '%s' (nothing to verify)" % root)
    sys.exit(0)
listed = set()
for dp, _, fs in os.walk(root):
    if "INDEX.json" not in fs:
        continue
    idx += 1
    ip = os.path.join(dp, "INDEX.json")
    try:
        index = json.load(open(ip))
    except Exception as e:
        print("FAIL %s: unreadable (%s)" % (ip, e)); bad += 1; continue
    out = []
    pairs(index, dp, out)
    for p, h in out:
        p = os.path.normpath(p)
        listed.add(p)
        if not os.path.isfile(p):
            print("FAIL %s: missing (listed in %s)" % (p, ip)); bad += 1
        elif sha(p) != h.lower():
            print("FAIL %s: sha256 mismatch (QM-E617)" % p); bad += 1
        else:
            ok += 1
unlisted = []
for dp, _, fs in os.walk(root):
    for f in fs:
        p = os.path.normpath(os.path.join(dp, f))
        if f.endswith(".bin") and p not in listed:
            unlisted.append(p)
for p in unlisted:
    print("FAIL %s: weights not listed in any INDEX.json" % p); bad += 1
size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(root) for f in fs)
print("models-verify: %d INDEX.json, %d files ok, %d problems, %.2f MB total" % (idx, ok, bad, size / 1e6))
sys.exit(1 if bad else 0)
endef
export MODELS_VERIFY_PY

models-verify:  ## sha256 of every model file under models/ai_transpiler against its INDEX.json
	@python3 -c "$$MODELS_VERIFY_PY"

train-ai: train-ai-synth train-ai-route  ## retrain the AI-transpiler models (hours; budgets above; not part of make test)
	@$(MAKE) --no-print-directory models-verify

train-ai-synth:  ## qiskit-gym synthesis models (budget AI_SYNTH_BUDGET_S)
	@if [ -x training/ai_transpiler/synth/train_all.sh ]; then \
		AI_TRAIN_PY=$(AI_TRAIN_PY) BUDGET_S=$(AI_SYNTH_BUDGET_S) training/ai_transpiler/synth/train_all.sh; \
	else echo "training/ai_transpiler/synth/train_all.sh not found (see training/ai_transpiler/README.md)"; exit 1; fi

AI_ROUTE_WORK ?= build/ai_train/route
AI_ROUTE_ID ?= route_mlp_h64_v1

train-ai-route:  ## learned routing policy: teacher data + BC + DAgger round (~45 min on an M5; training/ai_transpiler/route/README.md)
	@mkdir -p $(AI_ROUTE_WORK)/stage
	$(MOJO) build -I . -o build/route_tool training/ai_transpiler/route/route_tool.mojo
	$(AI_TRAIN_PY) training/ai_transpiler/route/parity.py $(AI_ROUTE_WORK)/parity build/route_tool
	$(AI_TRAIN_PY) -c "import sys,json; sys.path.insert(0,'training/ai_transpiler/route'); import circuits as C; json.dump(C.build_jobs(31, 1000, max_n=30, scale=0.5), open('$(AI_ROUTE_WORK)/jobs0.json','w')); json.dump(C.build_jobs(32, 1000, max_n=30, scale=0.5), open('$(AI_ROUTE_WORK)/jobs1.json','w'))"
	ROUTE_TOOL_CAP=96 build/route_tool gen $(AI_ROUTE_WORK)/jobs0.json $(AI_ROUTE_WORK)/d0.bin - 1.0 401 heuristic 16
	$(AI_TRAIN_PY) training/ai_transpiler/route/train_route.py --data $(AI_ROUTE_WORK)/d0.bin --out $(AI_ROUTE_WORK)/run0 --hidden 64 --tau 0.5 --epochs 30 --seed 0
	$(AI_TRAIN_PY) training/ai_transpiler/route/export_npz.py $(AI_ROUTE_WORK)/run0 $(AI_ROUTE_WORK)/stage round0 --default
	ROUTE_TOOL_CAP=96 build/route_tool gen $(AI_ROUTE_WORK)/jobs1.json $(AI_ROUTE_WORK)/d1.bin $(AI_ROUTE_WORK)/stage/round0 0.3 402 heuristic 16
	$(AI_TRAIN_PY) training/ai_transpiler/route/train_route.py --data $(AI_ROUTE_WORK)/d0.bin $(AI_ROUTE_WORK)/d1.bin --out $(AI_ROUTE_WORK)/run1 --hidden 64 --tau 0.5 --epochs 30 --seed 0
	$(AI_TRAIN_PY) training/ai_transpiler/route/export_npz.py $(AI_ROUTE_WORK)/run1 models/ai_transpiler/route $(AI_ROUTE_ID) --default

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
