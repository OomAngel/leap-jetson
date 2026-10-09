"""Minimal libcudart bindings through ctypes: device buffers, copies and a stream.

The Jetson's system Python has TensorRT but no pycuda, cupy or cuda-python; these few calls are
all the TensorRT runner needs.
"""

from __future__ import annotations

import ctypes

H2D, D2H = 1, 2  # cudaMemcpyKind
_cudart: ctypes.CDLL | None = None


def cudart() -> ctypes.CDLL:
    """Load libcudart once (CUDA 11.4 on JetPack 5; 12/13 elsewhere) and declare the calls used."""
    global _cudart
    if _cudart is not None:
        return _cudart
    for name in ("libcudart.so.11.0", "libcudart.so.12", "libcudart.so.13", "libcudart.so"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        vp, sz = ctypes.c_void_p, ctypes.c_size_t
        signatures = {
            "cudaMalloc": [ctypes.POINTER(vp), sz],
            "cudaFree": [vp],
            "cudaMemcpyAsync": [vp, vp, sz, ctypes.c_int, vp],
            "cudaStreamCreate": [ctypes.POINTER(vp)],
            "cudaStreamSynchronize": [vp],
            "cudaStreamDestroy": [vp],
            "cudaSetDeviceFlags": [ctypes.c_uint],
        }
        for fn, args in signatures.items():
            getattr(lib, fn).argtypes = args
            getattr(lib, fn).restype = ctypes.c_int
        _cudart = lib
        return lib
    raise RuntimeError("libcudart not found; cannot allocate device memory")


def check(ret: int, what: str) -> None:
    if ret != 0:
        msg = f"{what} failed with CUDA error {ret}"
        raise RuntimeError(msg)


def spin_wait() -> None:
    """Busy-wait stream syncs (cudaDeviceScheduleSpin): ~0.4 ms less per call on a Xavier.

    Must run before anything creates the CUDA context (TensorRT runtime, engines).
    """
    check(cudart().cudaSetDeviceFlags(1), "cudaSetDeviceFlags")
