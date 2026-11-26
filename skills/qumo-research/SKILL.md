---
name: qumo-research
description: Investigate an approved quantum-software research direction and turn it into a narrow, reproducible QUMO prototype or implementation with independent validation.
---

# QUMO research

Use this skill after a paper, SOTA scan, or maintainer decision identifies a concrete research
direction. It applies to new algorithms, compiler passes, simulation methods, QEC/FT techniques,
and execution features; use the dedicated Mojo and benchmarking skills alongside it when editing
those areas.

## Scope the smallest scientific claim

State the claim before coding: the supported inputs, expected output or resource property, the
reference baseline, and an explicit non-goal. Start with a toy instance or narrow API rather than
an implied general implementation. Preserve QUMO's hardware-neutral core; provider-specific work
belongs in `plugins/hardware`.

Read the source material, relevant module docs, nearby tests, and the corresponding row in
`docs/FRAMEWORK_UPDATES.md`. If the proposal contradicts an existing deliberate constraint, surface
the trade-off instead of silently bypassing it.

## Evidence-driven implementation

- Choose an independent oracle before implementation: analytic derivation, exhaustive small-case
  enumeration, state-vector equivalence, Stim/PyMatching, or a documented reference SDK.
- Add boundary, invalid-input, determinism, and regression tests. A result matching the previous
  QUMO implementation is not independent validation.
- For an optimisation, report correctness and the full measurement conditions; use the benchmark
  skill for fair comparisons. No isolated best-case claim is sufficient.
- Keep resource bounds explicit. Reject unsupported scale or semantics with a stable error and hint
  rather than silently approximating.

## Completion standard

The work is ready for review when it has a focused test, a runnable example or reproducible driver
when user-facing, module documentation that states scope and limits, and a concise entry in
`docs/FRAMEWORK_UPDATES.md` with status and source. Record deferred generalisations separately;
do not label a narrow prototype as a complete adoption.
