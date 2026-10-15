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
