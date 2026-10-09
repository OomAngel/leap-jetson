"""Pre- and post-processing for Monado's Mercury hand models on Leap Motion IR frames.

Conventions follow Monado's Mercury (`src/xrt/tracking/hand/mercury/hg_model.cpp`) as
documented by OpenLeap (github.com/julianjc84/OpenLeap, BSL-1.0, `tracking/skeleton.py`):

- detection model `grayscale_detection_160x160`: input `inputImg [1,1,160,160]`, the eye
  image letterboxed to 160x160, scaled to 0..1 and shifted so its mean is 0.5; outputs two
  hand slots: `hand_exists [1,2]`, centre `cx, cy [1,2]` in -1..1 of the canvas, `size [1,2]`
  as a fraction of the canvas.
- keypoint model `grayscale_keypoint_jan18`: input a 128x128 crop normalised the same way,
  `lastKeypoints [1,42]` (previous joints in -1..1 of this crop) and `useLastKeypoints [1]`;
  output `heatmap_xy [1,21,22,22]`. A joint is the heatmap argmax refined by the centre of
  mass in a +-4 cell window; its confidence is the heatmap maximum. The model knows one
  handedness, so the other hand is run on a mirrored crop.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DET_SIZE = 160
KP_SIZE = 128
HEATMAP = 22
N_JOINTS = 21
REFINE = 4  # centre-of-mass window half-width, heatmap cells


def prepare_eye(eye_640x240: np.ndarray) -> np.ndarray:
    """Leap eye image -> the geometry the detector expects.

    The 640x240 mode bins rows 2:1, so the image is stretched back to 640x480; a controller
    lying face up sees hands upside down relative to Mercury's training views, so it is
    rotated by 180 degrees. Measured on 2026-10-09 (probe_detector.py, 86 eye frames with a
    hand): median `hand_exists` 0.74 for this variant against 0.006 for the raw image.
    """
    return cv2.rotate(cv2.resize(eye_640x240, (640, 480), interpolation=cv2.INTER_LINEAR), cv2.ROTATE_180)


def eye_to_raw(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates in a `prepare_eye` image -> the raw 640x240 eye image."""
    return 639.0 - x, (479.0 - y) * 0.5


def normalise(img: np.ndarray) -> np.ndarray:
    """uint8 image -> float32 in 0..1 with its mean moved to 0.5."""
    f = img.astype(np.float32) * (1.0 / 255.0)
    return f + (0.5 - float(f.mean()))


@dataclass
class Letterbox:
    scale: float
    off_x: int
    off_y: int

    def to_eye(self, canvas_x: float, canvas_y: float) -> tuple[float, float]:
        return (canvas_x - self.off_x) / self.scale, (canvas_y - self.off_y) / self.scale


def letterbox(img: np.ndarray, size: int = DET_SIZE) -> tuple[np.ndarray, Letterbox]:
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    canvas = np.zeros((size, size), np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, Letterbox(scale, ox, oy)


@dataclass
class Detection:
    exists: float
    x: float  # eye pixels
    y: float
    size: float  # eye pixels


def decode_detections(out: dict[str, np.ndarray], lb: Letterbox) -> list[Detection]:
    """Both detector slots in eye coordinates, best first."""
    dets = []
    for slot in range(out["hand_exists"].shape[1]):
        px = (float(out["cx"][0, slot]) * 0.5 + 0.5) * DET_SIZE
        py = (float(out["cy"][0, slot]) * 0.5 + 0.5) * DET_SIZE
        ex, ey = lb.to_eye(px, py)
        size = abs(float(out["size"][0, slot])) * DET_SIZE / lb.scale
        dets.append(Detection(float(out["hand_exists"][0, slot]), ex, ey, size))
    return sorted(dets, key=lambda d: -d.exists)


@dataclass
class Crop:
    x0: int
    y0: int
    width: int  # square crop side in eye pixels


def crop_square(img: np.ndarray, cx: float, cy: float, half: float) -> tuple[np.ndarray, Crop]:
    """Square crop centred on (cx, cy), zero-padded where it leaves the image, resized to 128."""
    half = max(8.0, half)
    x0, y0 = round(cx - half), round(cy - half)
    x1, y1 = round(cx + half), round(cy + half)
    h, w = img.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (KP_SIZE, KP_SIZE), interpolation=cv2.INTER_AREA), Crop(x0, y0, x1 - x0)


def keypoint_feeds(
    crop128: np.ndarray, crop: Crop, flip: bool, last_pts: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Model inputs for one crop; `last_pts` are the previous frame's joints in eye pixels."""
    img = cv2.flip(crop128, 1) if flip else crop128
    last = np.zeros((1, 2 * N_JOINTS), np.float32)
    use = np.zeros((1,), np.float32)
    if last_pts is not None:
        xn = (last_pts[:, 0] - crop.x0) / crop.width * 2.0 - 1.0
        yn = (last_pts[:, 1] - crop.y0) / crop.width * 2.0 - 1.0
        last[0, 0::2] = -xn if flip else xn
        last[0, 1::2] = yn
        use[0] = 1.0
    return {"inputImg": normalise(img)[None, None], "lastKeypoints": last, "useLastKeypoints": use}


def decode_heatmaps(heatmap_xy: np.ndarray, crop: Crop, flip: bool) -> tuple[np.ndarray, np.ndarray]:
    """[1,21,22,22] heatmaps -> joints [21,2] in eye pixels and confidences [21].

    All joints at once: the argmax cell, then the centre of mass of the +-REFINE window around
    it (zero padding stands in for the window being clipped at the heatmap border).
    """
    hm = heatmap_xy.reshape(N_JOINTS, HEATMAP, HEATMAP).astype(np.float32)
    flat = hm.reshape(N_JOINTS, -1)
    conf = flat.max(axis=1)
    cy, cx = np.divmod(flat.argmax(axis=1), HEATMAP)
    padded = np.pad(hm, ((0, 0), (REFINE, REFINE), (REFINE, REFINE)))
    off = np.arange(-REFINE, REFINE + 1)
    ys, xs = cy[:, None] + off, cx[:, None] + off  # [21, 2R+1] heatmap cells
    win = padded[np.arange(N_JOINTS)[:, None, None], (ys + REFINE)[:, :, None], (xs + REFINE)[:, None, :]]
    total = win.sum(axis=(1, 2))
    ok = total > 0
    safe = np.where(ok, total, 1.0)
    hx = np.where(ok, (win.sum(axis=1) * (xs + 0.5)).sum(axis=1) / safe, cx + 0.5)
    hy = np.where(ok, (win.sum(axis=2) * (ys + 0.5)).sum(axis=1) / safe, cy + 0.5)
    if flip:
        hx = HEATMAP - hx
    pts = np.stack([crop.x0 + hx / HEATMAP * crop.width, crop.y0 + hy / HEATMAP * crop.width], axis=1)
    return pts.astype(np.float32), conf
