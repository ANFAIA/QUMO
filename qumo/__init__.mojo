"""QUMO: the names a first program needs, importable from the package root.

    from qumo import Circuit, sample_auto, transpile, parse_target

`from qumo import *` works too. The names are the re-exports of `qumo.prelude`
(see its docstring for the list and where each one lives). Importing a
submodule (`from qumo.ir.circuit import Circuit`) is unchanged and builds as
fast as before.
"""

from qumo.prelude import *
