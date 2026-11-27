---
name: qumo-optimize
description: Improve QUMO runtime, memory use, compiler output, or resource estimates while preserving semantics and accepting only reproducible measured gains.
---

# QUMO optimisation

Use this skill for a performance or output-quality improvement: simulation throughput, allocation
behaviour, compiler depth/gate count, routing quality, synthesis cost, decoder throughput, or an
algorithmic resource estimate. It is not for adding a new research capability; use `qumo-research`
when the method itself is new to QUMO.

## Define the objective and baseline

Name one primary objective and guardrails before changing code: for example, lower two-qubit count
without increasing depth beyond a stated bound, or faster state-vector execution without changing
fp64 results. Capture the baseline command, commit, representative inputs, target/basis, seed,
precision, thread count, and machine conditions.

Read nearby implementation and tests to identify the actual bottleneck or suboptimal decision. Do
not optimise a proxy unless it has a documented relationship to the user-visible objective.

## Preserve semantics

- Compiler rewrites must preserve the unitary up to tracked global phase and respect target and
  program/control-flow constraints.
- Simulator and numerical changes need a correctness oracle at small scale and stated tolerance.
- Keep determinism for a fixed seed unless the public contract explicitly changes.
- Bound new caches, searches, and allocations. A speedup that makes ordinary inputs unbounded is
  not an acceptable default.

## Evaluate fairly

Use representative small, median, and stress cases, including a case where the change is expected
to lose. Validate correctness before timing. For measured claims, follow
`skills/qumo-benchmarking/SKILL.md`: alternate A/B runs, use repeated measurements, retain raw
data, and report regressions as well as wins.

Add a regression test for the decision quality or semantic bug being improved. Add a benchmark or
performance gate only when it is stable enough to be meaningful across the supported hardware.
