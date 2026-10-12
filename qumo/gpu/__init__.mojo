"""qumo.gpu: vendor-neutral GPU state vector (NVIDIA, AMD, Apple) on MAX.

Every kernel launch is compiled only with `-D QUMO_GPU`; default builds keep
working on machines without a GPU toolchain and run the same kernel bodies
through the host emulator (see docs/GPU.md).
"""
