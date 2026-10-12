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
`confirm=True` (Python). Full documentation:
[docs/HARDWARE_PLUGINS.md](../../docs/HARDWARE_PLUGINS.md).
