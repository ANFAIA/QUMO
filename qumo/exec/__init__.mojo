"""qumo.exec: orchestration.

  primitives    EstimatorPub / SamplerPub, run_estimator, run_sampler (PUBs with parameter broadcasting)
  orchestrator  Workflow: a DAG of jobs scheduled with HEFT over a pool of CPU / GPU / QPU devices

`dispatch`, `emulate` and `resources` are re-exports kept for one release: simulator selection is
`qumo.sim.dispatch` and `qumo.gpu.dispatch`, the device emulator `qumo.noise.emulate`, resource
estimation `qumo.compiler.resources`. Hardware clients are not in `qumo/` (docs/ARCHITECTURE.md).
"""
