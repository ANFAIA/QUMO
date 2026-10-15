"""QUMO fault-tolerant programming layer (native re-implementation of Guppy FT ideas).

    qumo.ft.pauli      symplectic Pauli operators, stabilizer groups with phases
    qumo.ft.code       StabilizerCode + library (Steane, Iceberg, [[4,2,2]], [[5,1,3]], surface/repetition adapters)
    qumo.ft.verify     valid_clifford_implementation / valid_stabilizer_state_preparation (tableau)
    qumo.ft.gadgets    physical primitives of the Steane and Iceberg architectures (+ magic-state check)
    qumo.ft.arch       Architecture, QECPolicy, FactoryConf, SteaneBuilder, IcebergBuilder
    qumo.ft.logical    LogicalProgram: the logical-level IR and API
    qumo.ft.classical  classical expressions (deferred reads, Pauli frame, classical lane)
    qumo.ft.encode     compile / encode / check_may_encode
    qumo.ft.implement  implement_ops: allocator, factories, QEC policy, frame, decoding
    qumo.ft.rz         Clifford+T Rz synthesis and the comparator RUS Rz (run-time angles)
    qumo.ft.emulate    emulation with noise, logical semantics, error-rate sweeps
    qumo.ft.tools      JSON + MCP tools (qumo_ft_codes, qumo_ft_encode, qumo_ft_verify)

See docs/FT.md. Guppy FT: https://github.com/Quantinuum/guppyft (Apache-2.0).
"""
