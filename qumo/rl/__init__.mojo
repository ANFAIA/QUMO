"""QUMO's native Mojo training stack for the AI-transpiler models (pure Mojo, no foreign runtime or numerical framework).

Design of record: `training/ai_transpiler/MOJO_PORT.md`. Training-only code: nothing under
`qumo/compiler/` imports from here.

* `testutil`      - fixture loader, tolerance and statistics helpers, reference GEMM, guard regions (tests only).
* `digest`, `f16` - SHA-256 / CRC-32 and the f32 -> f16 encoder.
* `rlrng`, `tensor`, `optim`, `nn`, `loss` - seeded sampling, f32 kernels, Adam, the synthesis network and its PPO loss.
* `topology`, `symmetry`, `gateset`, `synth_env` - the four synthesis environments.
* `ppo`, `checkpoint`, `export_synth`, `eval_blocks`, `synth_train` - PPO trainer, checkpoints, export, evaluation, driver.
* `container` - model-directory writer, INDEX and verification.
* `teacher_data`, `router_net`, `router_loss`, `bc`, `route_ppo`, `targets`, `jobs`, `export_route` - the router side.
"""
