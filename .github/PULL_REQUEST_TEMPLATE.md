## What this changes

<!-- One or two sentences. Link the issue for anything larger than a small fix. -->

## How it was tested

- Jetson board:
- JetPack / L4T:
- TensorRT and Python versions:
- What you ran and what you measured (fps, frames with a 3D hand, latency):

## Checklist

- [ ] `ruff check .`, `ruff format --check .`, `mypy tracker` and `pytest` pass
- [ ] Still runs on Python 3.8 (JetPack 5's system Python)
- [ ] No model files (`.onnx`), TensorRT engines, raw recordings, calibration files or Ultraleap software
- [ ] No images that show faces
