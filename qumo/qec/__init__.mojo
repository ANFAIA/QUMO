"""QUMO quantum error correction + experimentalist toolkit.

    qumo.qec.codes            repetition / rotated surface code memory circuits, to_stim()
    qumo.qec.dem              detector error model extraction (Stim-exact, with decomposition)
    qumo.qec.decoders         MatchingGraph, UnionFindDecoder, MwpmDecoder, LookupTableDecoder
    qumo.qec.experiments      Sinter-style memory experiments and threshold sweeps
    qumo.qec.characterization re-export of qumo.noise.characterization (kept for one release)

The noise model, RB and readout mitigation live in qumo.noise. Simulation back-ends live in qumo.sim.stabilizer (tableau) and qumo.sim.frame
(bit-packed Pauli frames). See docs/QEC.md.
"""
