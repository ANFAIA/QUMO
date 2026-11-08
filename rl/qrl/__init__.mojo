"""Package qrl: a native Mojo reinforcement-learning stack for quantum-circuit synthesis and routing policies.

Pure Mojo, no foreign runtime or numerical framework, and no dependency on QUMO: nothing in this package imports
`qumo.*` (`rl/check_independence.sh` enforces it). The optional QUMO glue lives in the sibling package `qumo_bridge`.
Design of record: `rl/docs/MOJO_PORT.md`; usage, formats and layout: `rl/README.md`.

* `json`, `rng`, `gates`  - JSON reader / writer, the xoshiro256** generator and the gate opcodes (copied from QUMO).
* `model`, `env_core`     - the `qumo.rlsynth.v0` / `qumo.rlroute.v0` reader with the reference MLP forward, and the
                            packed-state synthesis dynamics (`PolicyEnv`, `greedy_trace`).
* `testutil`              - fixture loader, tolerance and statistics helpers, reference GEMM, guard regions (tests only).
* `digest`, `f16`         - SHA-256 / CRC-32 and the f32 -> f16 encoder.
* `rlrng`, `tensor`, `optim`, `nn`, `loss` - seeded sampling, f32 kernels, Adam, the synthesis network and its PPO loss.
* `topology`, `symmetry`, `gateset`, `synth_env` - the four synthesis training environments.
* `ppo`, `checkpoint`, `export_synth`, `synth_train`, `campaign` - PPO trainer, checkpoints, export, driver, campaigns.
* `container`             - model-directory writer, INDEX and verification (owner of both container formats).
* `teacher_data`, `router_net`, `router_loss`, `bc`, `targets`, `jobs` - the router side (data, net, loss, behaviour
                            cloning, neutral topologies, circuit-family job sets).
"""
