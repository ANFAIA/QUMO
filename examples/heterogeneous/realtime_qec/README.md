# Decoding while sampling: streaming surface-code decoding under a real-time budget

`stream_decode.py` streams a distance-5 surface-code memory experiment from a
QPU stand-in to a pool of CPU decoder processes, paced to a 1 us syndrome
cycle, and checks the decoders keep up.

```bash
PYTHONPATH=build:plugins/hardware bench/.venv/bin/python examples/heterogeneous/realtime_qec/stream_decode.py
bench/.venv/bin/python examples/heterogeneous/ai_gpu_qpu.py --offline --problem examples/heterogeneous/problems/surface_code_stream.json
```

## Pipeline: stage, device, why

| Stage | Device | Why this device |
|---|---|---|
| Syndrome stream | QPU stand-in: QUMO Pauli-frame sampler, CPU or GPU frames (emulated) | Produces detection events of the noisy memory circuit (`qumo_py.QecMemory.sample`). The stand-in is paced to the hardware clock: a batch is released when its last round would arrive (`--round-us`, `--time-scale`). `auto` measures CPU and GPU frames and uses the faster one. |
| Decoder pool | CPU, `--workers` processes | Each process owns a QUMO sparse-blossom MWPM decoder (`QecMemory.decode`, one thread) and decodes batches while later batches are still being generated. |
| Real-time budget | FPGA / controller, **emulated as a budget model** | Not run on an FPGA: the requirement is checked - decode capacity >= stream rate, bounded backlog, each batch decoded within `--latency-budget-ms` of its last round. |
| Reference decoder | CPU | Stim derives its own DEM from QUMO's circuit (Stim text export) and PyMatching 2 decodes the same syndromes. |

## Measured (Apple M5, 2026-10-17, seed 7, d = 5, 5 rounds, p = 0.004, 200 000 shots)

| Quantity | Value |
|---|---|
| Stream | 1.0 M rounds/s (1000-shot batches every 5 ms) |
| One QUMO decoder | 0.16 - 0.35 us per round (depends on load) |
| Pool capacity, 4 processes | 11 - 25 M rounds/s: 11-25x the stream |
| Batch latency (budget 10 ms) | p50 1.9 ms, p99 4.0 ms, 100 % within budget |
| Logical error rate | 0.00749 (QUMO) = 0.00749 (PyMatching), agreement 100 % of shots |
| PyMatching, one core | 0.21 us per round |
| QPU stand-in placement | CPU frames 0.28 ms vs GPU frames 11.5 ms per 1000-shot batch (launch, transfer and packing dominate): auto picked the CPU |
| Wall time | 1.6 s |

With 50 000-shot batches the per-batch latency is about 60 ms and the
10 ms budget fails: batching trades latency for throughput, which is why
the default batch is small.

## Checks

Pool capacity >= stream rate; final backlog <= 2 batches; >= 90 % of
batches within the budget; decoding overlapped generation; QUMO and
PyMatching agree on >= 99.9 % of shots and their logical error rates within
4 sigma; logical error rate below p x rounds.

## Limits

* Shots are decoded whole (memory experiment), not with sliding windows;
  "1 us per round" is a throughput budget per stream, not per-round reaction
  time on an FPGA.
* Packed syndromes cross processes as hex strings (pickled); a real system
  uses shared memory or an FPGA link (NVQLink).
* PyMatching built from QUMO's own `dem_text()` disagreed on 0.2 % of shots
  at p = 0.004 (QUMO's internal graph agrees with Stim's DEM); the reference
  therefore uses Stim's DEM. Tracked as a separate issue.
* No GPU-trained decoder stage.

Sources: [CUDA-Q QEC](https://nvidia.github.io/cudaqx/components/qec/introduction.html);
[NVQLink](https://developer.nvidia.com/blog/nvidia-nvqlink-architecture-integrates-accelerated-computing-with-quantum-processors/);
Riverlane real-time decoding, [arXiv:2410.05202](https://arxiv.org/abs/2410.05202);
Higgott & Gidney, sparse blossom, [arXiv:2303.15933](https://arxiv.org/abs/2303.15933).
