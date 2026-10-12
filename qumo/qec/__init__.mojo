"""QUMO quantum error correction + experimentalist toolkit.

    qumo.qec.codes            repetition / rotated surface code memory circuits, to_stim()
    qumo.qec.dem              detector error model extraction (Stim-exact, with decomposition)
    qumo.qec.decoders         MatchingGraph, UnionFindDecoder, MwpmDecoder, LookupTableDecoder
    qumo.qec.experiments      Sinter-style memory experiments and threshold sweeps
    qumo.qec.characterization noise models from calibration, RB, T1/Ramsey, readout mitigation

Simulation back-ends live in qumo.sim.stabilizer (tableau) and qumo.sim.frame
(bit-packed Pauli frames). See docs/QEC.md.
"""
