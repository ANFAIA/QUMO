---
name: qumo-examples-docs
description: Add or update QUMO documentation, runnable examples, generated example indexes, or public error/pass catalogues with reproducible checks.
---

# QUMO examples and documentation

Use this skill for changes under `docs/`, `examples/`, `README.md`, `CHANGELOG.md`, and generated
documentation catalogues.

## Runnable examples

An example is a machine-checked deliverable, not a code snippet. Follow the header and
`CheckLog` conventions in `CONTRIBUTING.md#adding-an-example`. Use an independent expected value
or a clearly linked source. Put slow work in the slow tier rather than weakening a fast example.

After changing example metadata or adding/removing an example, regenerate and check both indexes:

```bash
bench/.venv/bin/python examples/run_examples.py --write-index
bench/.venv/bin/python examples/run_examples.py --check-index
```

Use `python3` only when the selected examples do not require Python packages from `bench/.venv`.

## Public documentation

- Put user-facing capability details in the relevant module document, not only in a changelog.
- Describe scope, limits, reproducibility conditions, and unsupported cases next to claims that
  need them.
- Update `CHANGELOG.md` under Unreleased for a material user-facing change.
- Cite the original paper or SDK when a feature reproduces or adapts one; do not copy restricted
  source text or code.

## Generated catalogues

Error codes and pass definitions have one source of truth in Mojo. If they change, run:

```bash
QUMO_UPDATE_DOCS=1 .venv/bin/mojo run -I . tests/test_mcp.mojo
```

Never repair `docs/ERRORS.md` or `docs/PASSES.md` by hand. Finish with the narrow example or doc
check plus `git diff --check`.
