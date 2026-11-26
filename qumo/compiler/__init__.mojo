"""QUMO compiler: targets, text-spec pass manager and exact optimisation passes.

Modules:
  target        re-export of qumo.device.target (Target, presets, parse_target), kept for one release
  pass_manager  PassManager / text pipelines / preset levels O0-O3 / transpile
  optimize      remove_identities, cancel, merge, commute, fuse1q, eject_z, elide_swaps
  phase_fold    phase-polynomial T-count reduction
  pauli_simp    Pauli-gadget extraction + ladder-sharing resynthesis (Trotter circuits)
  kak           2q block consolidation + Weyl/KAK resynthesis
  basis         basis translation (any target gate family) + unroll3q
  routing       VF2 exact layout + LightSABRE routing
  synth         basis-aware single-qubit synthesis
  common        gate sets, commutation rules, wire lists
  resources     estimate_resources / estimate_hardware / estimate_fault_tolerant (static analysis)

`Target` itself is defined in qumo.device.target.

See docs/COMPILER.md.
"""
