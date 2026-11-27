"""QUMO's shared, dependency-free utilities (below the IR and every other layer).

* `json` - streaming JSON writer + arena JSON parser, used by the IR, compiler, backends and AI layer.
* `svd` - complex singular value decomposition by one-sided Jacobi (`svd`, `CMatrix`, `SVDResult`); the matrix product state simulator truncates with it.
* `rng` - the deterministic, splittable PRNG (`Rng`, xoshiro256** seeded by SplitMix64); `qumo.sim.rng` re-exports it for one release.

Nothing in qumo/core imports another QUMO package.
"""
