"""qumo.gpu: vendor-neutral GPU back ends (NVIDIA, AMD, Apple) on MAX.

  * `statevector_gpu.GpuStateVector[dtype]`: the state-vector simulator on a
    GPU (fp32 on Apple, fp64 default on NVIDIA/AMD), executing the CPU
    runner's fused plan one kernel per op.
  * `frame_gpu.GpuFrameSampler`: QEC Pauli-frame sampling, one thread per
    64-shot word, integer-only kernel (bit-identical on every vendor).
  * `device`: availability, device info, precision policy; `emulate`: host
    emulation of the kernel bodies for tests.

Every kernel launch is compiled only with `-D QUMO_GPU`; default builds keep
working on machines without a GPU toolchain and run the same kernel bodies
through the host emulator (see docs/GPU.md).
"""
