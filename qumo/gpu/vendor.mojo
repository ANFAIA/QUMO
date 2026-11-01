"""Compile-time vendor of the target accelerator.

Mojo 1.2 removed `has_{nvidia,amd,apple}_gpu_accelerator()` from `std.sys`; the
replacement, `default_accelerator().is_*_gpu()`, stops the build with "GPU
architecture '' is not supported" on a host with no accelerator. Default QUMO
builds must compile on such hosts (no GPU toolchain needed), so every query is
guarded by `has_accelerator()`: the three functions below return False there,
exactly what the removed predicates did. Import these instead of the std
predicates.
"""

from std.sys import default_accelerator, has_accelerator


def gpu_target_is_apple() -> Bool:
    """The default accelerator of this build is an Apple GPU (Metal)."""
    comptime if has_accelerator():
        return default_accelerator().is_apple_gpu()
    else:
        return False


def gpu_target_is_nvidia() -> Bool:
    """The default accelerator of this build is an NVIDIA GPU."""
    comptime if has_accelerator():
        return default_accelerator().is_nvidia_gpu()
    else:
        return False


def gpu_target_is_amd() -> Bool:
    """The default accelerator of this build is an AMD GPU."""
    comptime if has_accelerator():
        return default_accelerator().is_amd_gpu()
    else:
        return False
