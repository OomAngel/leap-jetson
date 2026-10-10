# Contributing

Thanks for your interest. leap-jetson is maintained by one person, on a best-effort basis.

## Reporting a bug

Use the bug report form. It asks for what is needed to reproduce a problem on this hardware: the Jetson
board, JetPack/L4T, TensorRT and Python versions, the controller's firmware, and the SHA-256 of your model
files (`sha256sum -c models.sha256`). Logs from `tracker/live_track.py` help. Please do not attach images
that show faces.

## Changes

- Open an issue first for anything larger than a small fix, so we can agree on the approach.
- Code must stay compatible with Python 3.8 (JetPack 5's system Python).
- Run `ruff check .`, `ruff format --check .`, `mypy tracker` and `pytest` before opening a pull request.
- Never commit model files (`.onnx`), TensorRT engines, raw recordings, calibration files or any
  Ultraleap software.
- Describe how you tested on hardware: board, JetPack, and what you measured.

## Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
