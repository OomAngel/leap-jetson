"""CPU-only tests of the tracker's geometry and decoding (no GPU, camera or models needed)."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from live_track import FileSource
from mercury import HEATMAP, Crop, decode_heatmaps, eye_to_raw, letterbox, prepare_eye
from pipeline import prepared_to_480
from stereo import STRUCT, Calibration, LensModel, cayley, stereo_consistent, triangulate


def gaussian_heatmaps(peaks: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """[21,2] peak positions in heatmap cells (continuous) -> [1,21,22,22] heatmaps."""
    ys, xs = np.mgrid[0:HEATMAP, 0:HEATMAP] + 0.5
    hm = np.exp(
        -((xs[None] - peaks[:, 0, None, None]) ** 2 + (ys[None] - peaks[:, 1, None, None]) ** 2) / (2 * sigma**2)
    )
    return hm[None].astype(np.float32)


def test_decode_recovers_known_peaks() -> None:
    rng = np.random.default_rng(1)
    peaks = rng.uniform(6.0, 16.0, size=(21, 2))  # away from the border, so the window is not clipped
    crop = Crop(x0=100, y0=50, width=220)
    pts, conf = decode_heatmaps(gaussian_heatmaps(peaks), crop, flip=False)
    expected = np.stack(
        [crop.x0 + peaks[:, 0] / HEATMAP * crop.width, crop.y0 + peaks[:, 1] / HEATMAP * crop.width], axis=1
    )
    assert np.abs(pts - expected).max() < 0.05 * crop.width / HEATMAP  # within 5 % of a heatmap cell
    assert (conf > 0.9).all()


def test_decode_flip_mirrors_x() -> None:
    peaks = np.full((21, 2), 8.3)
    crop = Crop(x0=0, y0=0, width=HEATMAP)
    straight, _ = decode_heatmaps(gaussian_heatmaps(peaks), crop, flip=False)
    mirrored, _ = decode_heatmaps(gaussian_heatmaps(peaks), crop, flip=True)
    assert np.allclose(mirrored[:, 0], HEATMAP - straight[:, 0], atol=1e-4)
    assert np.allclose(mirrored[:, 1], straight[:, 1], atol=1e-4)


def test_letterbox_maps_back_to_eye_pixels() -> None:
    img = np.zeros((480, 640), np.uint8)
    canvas, lb = letterbox(img)
    assert canvas.shape == (160, 160)
    ex, ey = lb.to_eye(lb.off_x + 320 * lb.scale, lb.off_y + 240 * lb.scale)
    assert abs(ex - 320) < 1e-6 and abs(ey - 240) < 1e-6


def test_prepare_eye_and_back() -> None:
    raw = np.zeros((240, 640), np.uint8)
    raw[60, 100] = 255  # one bright pixel in the raw 640x240 eye image
    prepared = prepare_eye(raw)
    assert prepared.shape == (480, 640)
    y, x = np.unravel_index(int(np.argmax(prepared)), prepared.shape)
    rx, ry = eye_to_raw(np.array([float(x)]), np.array([float(y)]))
    assert abs(rx[0] - 100) <= 1.0 and abs(ry[0] - 60) <= 1.0
    p480 = prepared_to_480(np.array([[float(x), float(y)]]))
    assert abs(p480[0, 0] - 100) <= 1.0 and abs(p480[0, 1] - 120) <= 2.0  # unrotated 640x480 row = 2 x raw row


def calibration_bytes(baseline: float = 40.0, focal: float = 134.0, signature: bytes = b"CA") -> bytes:
    cam = [focal, 0.0, 0.0, 0.0, 0.0] + [0.0] * 6 + [focal, 0.0, 0.0] + [0.0, 0.0, 0.0]
    return struct.pack(STRUCT, signature[0], signature[1], 2, 80, 0, baseline, 0.0, *cam, *cam, 0)


def test_calibration_parse_and_bad_signature() -> None:
    cal = Calibration.from_bytes(calibration_bytes())
    assert cal.baseline_mm == pytest.approx(40.0)
    assert cal.left.focal == pytest.approx(134.0) and cal.left.cx == pytest.approx(320.0)
    with pytest.raises(ValueError):
        Calibration.from_bytes(calibration_bytes(signature=b"\x4c\x4c"))  # the cached-control read gives 76,76


def test_triangulate_recovers_synthetic_points() -> None:
    cal = Calibration.from_bytes(calibration_bytes())  # no distortion, no rotation
    model = LensModel(power=1.0, rotation_sign=-1.0)
    rng = np.random.default_rng(2)
    pts3 = np.stack([rng.uniform(-60, 60, 21), rng.uniform(-40, 40, 21), rng.uniform(150, 400, 21)], axis=1)
    f, c = 134.0, np.array([320.0, 240.0])
    left = pts3[:, :2] / pts3[:, 2:] * f + c
    right = (pts3[:, :2] - [cal.baseline_mm, 0.0]) / pts3[:, 2:] * f + c  # right camera 40 mm along +x
    est, epi = triangulate(cal, left, right, model)
    assert np.abs(est - pts3).max() < 1e-3
    assert epi.max() < 1e-6


def test_cayley_is_a_rotation() -> None:
    r = cayley(np.array([0.1, -0.2, 0.05]))
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12) and np.linalg.det(r) == pytest.approx(1.0)


def test_file_source_every_frame(tmp_path) -> None:
    frames = np.arange(3, dtype=np.uint8)[:, None, None, None] * np.ones((3, 240, 640, 2), np.uint8)
    path = tmp_path / "rec.raw"
    frames.tofile(path)
    src = FileSource(str(path), every_frame=True)
    src.start()
    seen, values = 0, []
    while True:
        frame, count, _ = src.latest(seen)
        if frame is None:
            break
        seen = count
        values.append(int(frame[0, 0, 0]))
    assert values == [0, 1, 2] and src.done


def test_stereo_gate_rejects_mismatched_eyes() -> None:
    ok = np.ones(21, bool)
    assert stereo_consistent(np.full(21, 1.5), ok)
    assert not stereo_consistent(np.full(21, 20.0), ok)
    err = np.full(21, 1.0)
    err[:5] = 50.0  # a few outliers do not move the median
    assert stereo_consistent(err, ok)
    assert not stereo_consistent(err, np.zeros(21, bool))
