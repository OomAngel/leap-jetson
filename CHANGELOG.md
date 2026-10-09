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

### Known issues
- Edge-of-view 3D accuracy: the stored lens model's radial exponent is not settled.
