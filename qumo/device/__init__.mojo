"""Device description: what a quantum device is, with nothing about compiling for it.

Modules:
  target   Target (coupling graph, native basis, per-gate errors and durations, T1/T2), the modality
           and topology presets, and `parse_target` / `parse_basis` for one-line specs
  format   `basis_str` and `fmt_num`, the text helpers `Target.describe()` and the compiler reports share

`qumo.device` imports only `qumo.ir`. The compiler, the simulators, the noise model, the execution layer
and the AI layer import it; it imports none of them. `qumo.compiler.target` re-exports it for one release.
"""
