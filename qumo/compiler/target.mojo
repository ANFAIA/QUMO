"""Compatibility path: `Target` and its parsers now live in `qumo/device/target.mojo`.

`from qumo.compiler.target import Target, parse_target` keeps working and gives the same objects.
This path is kept for one release; new code imports `qumo.device.target`, which the compiler,
the simulators and the execution layer can use without importing the compiler.
"""

from qumo.device.target import Target, basis_presets, parse_basis, parse_target, parse_angle_bound
