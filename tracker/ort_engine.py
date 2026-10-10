"""ONNX Runtime backend with the same `run(feeds) -> outputs` interface as `trt_engines.MultiEngine`.

Lets the tracker run on any PC (CPU, or CUDA if onnxruntime-gpu is installed) straight from
Monado's ONNX files: replay recordings without a Jetson or TensorRT, and serve as the FP32
reference in `check_precision.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort


class OrtModel:
    def __init__(self, path: str | Path, providers: Optional[List[str]] = None) -> None:
        available = ort.get_available_providers()
        wanted = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=[p for p in wanted if p in available])
        self.outputs = [o.name for o in self.session.get_outputs()]

    def run(self, feeds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return dict(zip(self.outputs, self.session.run(self.outputs, feeds)))
