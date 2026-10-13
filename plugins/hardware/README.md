# qumo-hardware

Run QUMO-compiled circuits on real quantum hardware through one vendor-neutral
provider interface. First providers: **IBM Quantum** (qiskit-ibm-runtime,
client-side Sampler) and **Quantinuum** (Nexus via qnexus; H2 and Helios), plus
offline emulators for both.

```bash
make py                                            # build/qumo_py.so (the compiler) at the repo root
pip install -e 'plugins/hardware[ibm,quantinuum,test]'   # or: PYTHONPATH=plugins/hardware
python -m qumo_hardware run examples/bell.qasm                                     # local QUMO simulator
python -m qumo_hardware run examples/ghz.qasm -p ibm -b fake_torino                # IBM snapshot + Aer noise
python -m qumo_hardware run examples/ghz.qasm -p quantinuum -b H2-1 --dry-run      # plan + HQC estimate
python -m qumo_hardware run examples/ghz.qasm -p ibm -b ibm_fez --yes              # real QPU (credentials needed)
python -m pytest plugins/hardware/tests -q                                         # no accounts needed
```

Nothing is submitted to a remote backend without `--yes` (CLI) or
`confirm=True` (Python, both `runner.run` and `Provider.submit`). The compiler
extension is loaded only from `QUMO_PY_PATH` or from `build/qumo_py.so` of the
checkout that contains this package (never from the current directory), so a
non-editable install needs `QUMO_PY_PATH=/path/to/qumo_py.so`. Full documentation:
[docs/HARDWARE_PLUGINS.md](../../docs/HARDWARE_PLUGINS.md).
