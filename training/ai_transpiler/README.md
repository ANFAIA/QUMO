# Moved: AI-transpiler training now lives in `rl/`

The native Mojo training stack that used to live here (and in `qumo/rl`) is now the independent package in
[`rl/`](../../rl/README.md):

| was | is |
|---|---|
| `qumo/rl/*.mojo` | `rl/qrl/*.mojo` (package `qrl`, imports nothing from QUMO) and `rl/qumo_bridge/*.mojo` (the QUMO glue) |
| `training/ai_transpiler/{synth,route,common}/*.mojo` | `rl/drivers/{synth,route,common}/` and `rl/qumo_bridge/drivers/` |
| `training/ai_transpiler/synth/campaign.json` | `rl/drivers/synth/campaign.json` |
| `training/ai_transpiler/README.md` | `rl/docs/TRAINING.md` |
| `training/ai_transpiler/MOJO_PORT.md`, `spec/` | `rl/docs/MOJO_PORT.md`, `rl/spec/` |
| `training/ai_transpiler/smoke.sh`, `check_pure_mojo.sh` | `rl/smoke.sh`, `rl/check_pure_mojo.sh` |
| `tests/test_rl_*.mojo`, `tests/fixtures/rl/` | `rl/tests/`, `rl/tests/fixtures/`, `rl/qumo_bridge/tests/` |

The make targets are unchanged (`make train-ai`, `make train-ai-smoke`, `make models-verify`, `make test-rl`,
`make test-rl-slow`); they delegate to `rl/Makefile`.
