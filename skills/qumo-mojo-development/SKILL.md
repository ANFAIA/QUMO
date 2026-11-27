---
name: qumo-mojo-development
description: Implement or review QUMO's Mojo core, compiler, simulators, GPU code, or Python binding while preserving layering, semantics, and focused validation.
---

# QUMO Mojo development

Use this skill for changes under `qumo/`, `cli/`, `python/qumo_py.mojo`, or their direct Mojo tests.
It does not cover benchmark methodology or agent-harness policy; use their dedicated skills for
those tasks.

## Before editing

- Read `docs/MOJO_GUIDE.md` and the relevant module documentation in `docs/`.
- Find the nearest existing test in `tests/` before designing a new test.
- Check `docs/ARCHITECTURE.md#package-layering` before introducing an import. Keep dependencies
  downward; move agent-facing handlers into `qumo.ai` rather than importing AI from lower layers.

## Implementation constraints

- Preserve the public error-code contract. New user-visible errors need a `QM-Exxx` code and an
  actionable hint in `qumo/ai/diagnostics.mojo`.
- A compiler pass must be exact up to tracked global phase. Test its result against an independent
  state-vector, algebraic, or published oracle.
- Treat seed handling, measurement ordering, endianness, numerical tolerances, and resource bounds
  as API behaviour. Add a regression test whenever changing one.
- Put hardware-provider behavior in `plugins/hardware`; keep core targets modality/topology based.

## Validate

Run the focused test first:

```bash
.venv/bin/mojo run -I . tests/test_<area>.mojo
```

Then choose the relevant wider check: `make test`, `make test-python`, `make test-rl`, or
`make gpu-check`. If errors, passes, or their arguments changed, regenerate catalogues with:

```bash
QUMO_UPDATE_DOCS=1 .venv/bin/mojo run -I . tests/test_mcp.mojo
```

Do not hand-edit the generated error or pass catalogues.
