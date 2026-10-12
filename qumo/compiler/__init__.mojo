"""QUMO compiler: targets, text-spec pass manager and exact optimisation passes.

Modules:
  target        Target (coupling graph, basis, errors, durations), presets, parse_target
  pass_manager  PassManager / text pipelines / preset levels O0-O3 / transpile
  optimize      remove_identities, cancel, merge, commute, fuse1q, eject_z, elide_swaps
  phase_fold    phase-polynomial T-count reduction
  pauli_simp    Pauli-gadget extraction + ladder-sharing resynthesis (Trotter circuits)
  kak           2q block consolidation + Weyl/KAK resynthesis
  basis         basis translation (any target gate family) + unroll3q
  routing       VF2 exact layout + LightSABRE routing
  synth         basis-aware single-qubit synthesis
  common        gate sets, commutation rules, wire lists

See docs/COMPILER.md.
"""
