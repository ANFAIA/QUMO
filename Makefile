# QUMO build & test entry points.
MOJO ?= .venv/bin/mojo
PY ?= bench/.venv/bin/python
TESTS := $(wildcard tests/test_*.mojo)

.PHONY: all setup test cli py bench-sv bench-neutral benchpress clean

all: cli py

setup:  ## create .venv with the Mojo toolchain + MAX runtime (parallelize)
	uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python mojo max-mojo-libs max-core

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
GPU_TARGETS ?= sm_90 gfx942 metal:4
GPU_CHECK_SRCS := tests/test_gpu.mojo tests/test_gpu_frame.mojo bench/bench_gpu.mojo
GPU_TESTS := $(wildcard tests/test_gpu*.mojo)

.PHONY: gpu-check test-gpu cli-gpu py-gpu bench-gpu

gpu-check:  ## cross-compile every GPU kernel for NVIDIA sm_90, AMD gfx942, Apple metal:4 (no GPU needed)
	@fail=0; for a in $(GPU_TARGETS); do \
		d=build/gpucheck/$$(echo $$a | tr ':' '_'); rm -rf $$d; mkdir -p $$d; \
		for f in $(GPU_CHECK_SRCS); do \
			b=$$(basename $$f .mojo); \
			if $(MOJO) build -I . -D QUMO_GPU --emit asm --target-accelerator=$$a -o $$d/$$b.s $$f > $$d/$$b.log 2>&1; then \
				k=$$(ls $$d | grep "^$${b}_" | grep -c "_gk_"); \
				if [ $$k -gt 0 ]; then echo "PASS $$a $$f ($$k kernels)"; \
				else echo "FAIL $$a $$f: no kernels emitted (gating broken?)"; fail=1; fi; \
			else echo "FAIL $$a $$f (log: $$d/$$b.log)"; grep -v Crashpad $$d/$$b.log | head -8; fail=1; fi; \
		done; \
	done; exit $$fail

test-gpu:  ## GPU test suites on the real device (+ emulation suites)
	@fail=0; for t in $(GPU_TESTS); do \
		out=$$($(MOJO) run -I . -D QUMO_GPU $$t 2>&1 | grep -v Crashpad | tail -1); \
		case "$$out" in *OK*) echo "PASS $$t";; *) echo "FAIL $$t: $$out"; fail=1;; esac; \
	done; exit $$fail

cli-gpu:  ## CLI + MCP server with GPU dispatch (build/gpu/qumo)
	@mkdir -p build/gpu
	$(MOJO) build -I . -D QUMO_GPU -o build/gpu/qumo cli/main.mojo

py-gpu:  ## CPython extension with GPU dispatch (build/gpu/qumo_py.so)
	@mkdir -p build/gpu
	$(MOJO) build -I . -D QUMO_GPU --emit shared-lib -o build/gpu/qumo_py.so python/qumo_py.mojo

bench-gpu:  ## GPU vs CPU state-vector throughput (layered circuit of bench_sv)
	@mkdir -p build/gpu
	$(MOJO) build -I . -D QUMO_GPU -o build/gpu/bench_gpu bench/bench_gpu.mojo && ./build/gpu/bench_gpu 26

clean:
	rm -rf build
