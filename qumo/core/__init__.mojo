"""QUMO's shared, dependency-free utilities (below the IR and every other layer).

* `json` - streaming JSON writer + arena JSON parser, used by the IR, compiler, backends and AI layer.
* `rng` - the deterministic, splittable PRNG (`Rng`, xoshiro256** seeded by SplitMix64); `qumo.sim.rng` re-exports it for one release.

Nothing in qumo/core imports another QUMO package.
"""
