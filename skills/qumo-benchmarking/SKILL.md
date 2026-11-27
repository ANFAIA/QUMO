---
name: qumo-benchmarking
description: Design, run, review, or report QUMO performance benchmarks with reproducible inputs, fair comparisons, and complete disclosure.
---

# QUMO benchmarking

Use this skill for changes to `bench/`, benchmark reports, charts, or performance claims. It is not
for ordinary correctness tests.

## Measurement standard

Read `CONTRIBUTING.md#reporting-benchmark-results` before claiming a speed, memory, fidelity, or
gate-count improvement. Record the exact command, commit and tree state, toolchain/SDK versions,
machine and GPU details, precision, thread count, repetitions, aggregation method, and raw output.

Compare equal work: circuit set, target/basis, precision, shots, and correctness criterion must be
the same. Prefer median results from at least three repetitions and alternate binaries in A/B runs.
Report losses and failures as well as wins; disclose benchmark tuning.

## Repository conventions

- `make setup-py-locked` is the reproducible environment for an existing published result;
  `make setup-py` intentionally follows latest dependencies.
- Keep raw benchmark data separate from derived reports and charts. Do not overwrite published
  numbers merely because a local machine differs.
- Use the established harnesses (`make bench-sv`, `make bench-neutral`, `make bench-ai`, or the
  matching scripts) unless the task specifically requires a new harness.

Before publishing a report, verify referenced raw artifacts exist and update the applicable bias or
methodology disclosure when the comparison changes.
