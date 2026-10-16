# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Live 3D hand tracking for the original Leap Motion Controller on a Jetson AGX Xavier
  (JetPack 5.1 / L4T 35.6.5, TensorRT 8.5.2, Python 3.8): UVC stereo capture at 115 fps, calibration
  read from the controller over libusb, Monado Mercury models (commit `37a8a81`) in TensorRT FP16,
  keypoint triangulation.
- 98 fps with a hand in view and 14 ms median camera-to-3D latency with the GPU clock pinned;
  FP16 within 0.04 px of ONNX FP32 (`docs/RESULTS.md`).
- Checking tools: precision (`check_precision.py`), detector geometry (`probe_detector.py`) and
  lens-model fit (`fit_lens_model.py`).
- Fallback notes for Ultraleap V2 under box64, with a local box64 patch.
- Replay mode (`--replay`, `--every-frame`) and an ONNX Runtime backend (`--backend ort`): the tracker
  runs on recorded frames on any PC, without a Jetson or TensorRT.
- CPU-only unit tests for decoding, geometry, triangulation, calibration parsing and replay.
- Stereo-consistency gate: a frame whose median stereo mismatch is above 3 px gets no 3D
  (`--max-epipolar`); the summary counts those frames.
- Plumb-line test of the lens model (`plumb_line.py`).

### Changed
- The stored radial lens correction is applied at exponent 0.1 instead of 1.0, from the plumb-line
  test; 3D near the centre changes by about 1%, further out by about 4%.

### Known issues
- 3D accuracy falls with distance (bones too long beyond ~30 cm).
