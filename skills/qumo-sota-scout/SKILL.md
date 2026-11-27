---
name: qumo-sota-scout
description: Research quantum-software state of the art, compare papers and framework changes against QUMO, and produce a ranked, evidence-backed adoption backlog without implementing it.
---

# QUMO SOTA scouting

Use this skill to investigate a QUMO area (`ir`, `sim`, `gpu`, `compiler`, `algorithms`, `qec`,
`ft`, `exec`, or `ai`), a research question, or changes in peer frameworks. The outcome is a
decision-ready research brief, not a speculative implementation.

## Establish the baseline first

Read the relevant section of `docs/FRAMEWORK_UPDATES.md`, the module document in `docs/`, and the
nearest code/tests. Use `docs/ANALYSIS.md` to understand the original design choice. Distinguish:

- capabilities QUMO already has;
- deferred ideas with a stated reason;
- an upstream API change or experimental result that does not yet justify adoption.

Do not mistake an unpublished preprint, vendor benchmark, or upstream feature announcement for a
validated result. Prefer primary papers, official release notes, and source repositories.

## Evaluate candidates

For each candidate, record the source and date, what it contributes beyond QUMO, its assumptions,
and the smallest meaningful QUMO scope. Assess semantic fit, dependency/layering impact, hardware
neutrality, numerical risk, expected cost, and an independent validation route.

Rank candidates as **adopt now**, **prototype**, **defer**, or **skip**. A useful recommendation
names the baseline, one measurable success condition, one reason it could fail, and the exact test
or benchmark that would discriminate it. Do not manufacture performance claims.

## Deliverable

Produce a concise brief with:

1. question and QUMO baseline;
2. sources and key evidence;
3. ranked table of candidates and trade-offs;
4. recommended next experiment or implementation slice;
5. explicit unknowns and non-goals.

When the finding is accepted by a maintainer, add or update its row in
`docs/FRAMEWORK_UPDATES.md`, including source, status, destination, and rationale. Do not mark a
feature `done` until code, tests, user documentation, and independent validation exist.
