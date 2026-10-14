# CUDA-Q signature examples, reproduced with QUMO

Four NVIDIA CUDA-Q examples, rebuilt from QUMO's own library and verified with
machine-readable `CHECK` lines. CUDA-Q removed these notebooks from its
repository at 0.15.0, so the sources cite the 0.14.0 documentation. CUDA-Q is
not installed: the numbers come from the pages or from analytic values.

```bash
.venv/bin/mojo run -I . examples/frameworks/cudaq/qaoa_nelder_mead.mojo
bench/.venv/bin/python examples/run_examples.py --only examples/frameworks/cudaq --tier all
# GPU (Apple / NVIDIA / AMD through MAX): GHZ up to 25 qubits on the GPU state vector
.venv/bin/mojo run -I . -D QUMO_GPU examples/frameworks/cudaq/ghz_cpu_gpu.mojo --max 25
```

| Original | URL | QUMO file | What is checked | Reference numbers (provenance) |
|---|---|---|---|---|
| Multi-GPU workflows: from CPU to GPU | https://nvidia.github.io/cuda-quantum/0.14.0/using/examples/multi_gpu_workflows.html | `ghz_cpu_gpu.mojo` (3 checks; fast, `--max 25` in GPU mode) | `sample_on(..., "cpu" \| "gpu")` for n = 2..20: only 0^n / 1^n; P(0^n) = 0.5 within 4 sigma; CPU and GPU frequencies agree (two-sample z <= 4); timings and method strings printed, not asserted | {00: 475, 11: 525} and 510 / 490 at n = 25 (page). Verified with -D QUMO_GPU to n = 25: `gpu:pauli_frame` (Clifford GHZ) and `gpu:statevector:fp32` (phase-GHZ) |
| Max-Cut with QAOA | https://nvidia.github.io/cuda-quantum/0.14.0/applications/python/qaoa.html | `qaoa_nelder_mead.mojo` (7) | <H> at the page optimum == page value (1e-9; measured 9e-15); the page's 1000-shot counts within 4 sigma; NelderMead + 5 seeded restarts: best <H> in [-5, -4.49]; most frequent of 10k samples is an optimal cut | -4.495973826282007 and counts 10101: 165, 01011: 156, 10100: 155, 01010: 153 (page) |
| Multi-GPU workflows (mqpu `observe_async` batching) + VQE parallel gradients | https://nvidia.github.io/cuda-quantum/0.14.0/using/examples/multi_gpu_workflows.html and https://nvidia.github.io/cuda-quantum/0.14.0/applications/python/vqe_advanced.html | `parallel_observe_gradients.mojo` (7; slow tier) | one EstimatorPub with 10,000 parameter sets == sequential loop == cos(params[0]) (1e-12); `Workflow` over [Device.cpu(), Device.gpu()] places every job on a capable device; PUB parameter shift == `batch_gradient` parameter shift == adjoint on a 12-qubit HEA (1e-10); speedups printed | 3.185 s -> 1.175 s on 4 GPUs (page, quoted). The page's (E+ - E-)/(2 eps) with eps = pi/4 is the exact derivative times sin(eps)/eps; QUMO uses the exact two-term rule |
| Hamiltonian simulation (Heisenberg) + Hadamard test | https://nvidia.github.io/cuda-quantum/0.14.0/applications/python/hamiltonian_simulation.html and https://nvidia.github.io/cuda-quantum/0.14.0/applications/python/hadamard_test.html | `hamiltonian_simulation.mojo` (14) | page instance (4 spins, \|1010>, t = pi, K = 100): exact evolution conserves <H> (1e-10), order-1 survival probability within 0.02 of exact; 10 spins: log-log error slopes -1 and -2 (+/- 0.2); Hadamard tests within 4 sigma (real and imaginary parts) | exact 0.7071067690849304, sampled 0.71018 +- 0.00157 (page); multi-QPU instance Re = 0, Im = -sin(angle/2) sin(theta/2) (analytic) |

## Library gaps met

- None blocking. The CUDA-Q "target" is global state; QUMO selects the device
  per call (`sample_on`) or per job (`Workflow`), so the examples print the
  method that actually ran instead of setting a target.
- Without -D QUMO_GPU (or with no usable device) the GPU rows run on the CPU
  and print why; every check still applies.
- Phase 2 (other workflow): the orchestrator's device pinning and in-process
  GPU expectations will let `parallel_observe_gradients.mojo` pin jobs
  explicitly; the example uses the public `Workflow` API as it is today.
