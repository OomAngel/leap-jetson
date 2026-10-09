"""TensorRT runner for engines with several named inputs and outputs (the Mercury hand models).

Device memory comes from libcudart through ctypes (`cuda_rt`), so the env needs no pycuda, cupy
or cuda-python. Shapes are static (the Mercury ONNX models have fixed shapes), so every buffer
is allocated once at load.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import tensorrt as trt

from cuda_rt import D2H, H2D, check, cudart

LOGGER = trt.Logger(trt.Logger.WARNING)
# trt.nptype() in TensorRT 8.5 refers to np.bool, which numpy >= 1.24 removed.
_NP_DTYPE = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT32: np.int32,
    trt.DataType.INT8: np.int8,
    trt.DataType.BOOL: np.bool_,
}


class MultiEngine:
    """One engine, one context, one host and one device buffer per I/O tensor."""

    def __init__(self, engine_path: str | Path) -> None:
        self.path = Path(engine_path)
        self._runtime = trt.Runtime(LOGGER)
        self.engine = self._runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            msg = f"could not deserialize {self.path} (TensorRT version or GPU mismatch?)"
            raise RuntimeError(msg)
        self.context = self.engine.create_execution_context()
        lib = cudart()
        self._stream = ctypes.c_void_p()
        check(lib.cudaStreamCreate(ctypes.byref(self._stream)), "cudaStreamCreate")
        self.inputs: dict[str, tuple[np.ndarray, ctypes.c_void_p]] = {}
        self.outputs: dict[str, tuple[np.ndarray, ctypes.c_void_p]] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                msg = f"{name} has a dynamic shape {shape}; this runner needs static shapes"
                raise RuntimeError(msg)
            host = np.empty(shape, dtype=_NP_DTYPE[self.engine.get_tensor_dtype(name)])
            dev = ctypes.c_void_p()
            check(lib.cudaMalloc(ctypes.byref(dev), max(host.nbytes, 1)), f"cudaMalloc({name})")
            self.context.set_tensor_address(name, dev.value)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            (self.inputs if is_input else self.outputs)[name] = (host, dev)

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Copy the inputs in, execute, copy every output back; returns copies keyed by name."""
        lib = cudart()
        for name, (host, dev) in self.inputs.items():
            x = np.ascontiguousarray(feeds[name], dtype=host.dtype).reshape(host.shape)
            check(lib.cudaMemcpyAsync(dev, x.ctypes.data, x.nbytes, H2D, self._stream), f"H2D {name}")
        if not self.context.execute_async_v3(self._stream.value):
            raise RuntimeError("execute_async_v3 returned False")
        for name, (host, dev) in self.outputs.items():
            check(lib.cudaMemcpyAsync(host.ctypes.data, dev, host.nbytes, D2H, self._stream), f"D2H {name}")
        check(lib.cudaStreamSynchronize(self._stream), "sync")
        return {name: host.copy() for name, (host, _) in self.outputs.items()}

    def close(self) -> None:
        lib = cudart()
        for _, dev in (*self.inputs.values(), *self.outputs.values()):
            if dev.value:
                lib.cudaFree(dev)
                dev.value = None
        if self._stream.value:
            lib.cudaStreamDestroy(self._stream)
            self._stream.value = None

    def __enter__(self) -> MultiEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
