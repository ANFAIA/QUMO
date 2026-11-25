"""Compatibility path: `Rng` now lives in `qumo/core/rng.mojo` (a low layer every package may import).

`from qumo.sim.rng import Rng` keeps working and is the same type. This path is kept for one release;
new code imports `qumo.core.rng`.
"""

from qumo.core.rng import Rng, log1p_neg
