"""Compatibility path: M3 readout mitigation now lives in `qumo/noise/mthree.mojo`.

`from qumo.qec.mthree import mitigate_counts_m3` keeps working and is the same function. This path is
kept for one release; new code imports `qumo.noise.mthree`.
"""

from qumo.noise.mthree import mitigate_counts_m3
