"""Per-eye hand keypoints on the GPU: detector + keypoint model (TensorRT) on prepared Leap images.

`EyeTracker.update(img)` takes one eye image as returned by `mercury.prepare_eye` and returns
the best hand's 21 joints (pixels in that image) with confidences, or None. It keeps a
temporal prior: the previous joints go back into the keypoint model, and while the detector
misses a hand (it fails first on small, far hands) the crop follows the previous joints for
as long as the keypoint confidence holds ("coasting", as Mercury does). Handedness is tried
both ways on acquisition and re-checked every `FLIP_RECHECK` frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np

from mercury import (
    Crop,
    crop_square,
    decode_detections,
    decode_heatmaps,
    keypoint_feeds,
    letterbox,
    normalise,
)


class Model(Protocol):
    """A detector or keypoint model: TensorRT (`trt_engines.MultiEngine`) or ONNX Runtime (`ort_engine.OrtModel`)."""

    def run(self, feeds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]: ...


HAND_MIN = 0.40  # detector score that acquires a hand
COAST_MIN = 0.32  # mean keypoint confidence that keeps a hand without the detector
CROP_SCALE = 1.3  # crop side = detector size x this
FLIP_RECHECK = 60  # frames between handedness re-checks


@dataclass
class Hand:
    pts: np.ndarray  # [21,2] pixels in the prepared eye image
    conf: np.ndarray  # [21]
    crop: Crop
    flip: bool
    detected: bool  # False while coasting
    score: float  # detector score, or mean keypoint confidence while coasting


class EyeTracker:
    def __init__(self, det: Model, kp: Model) -> None:
        self.det, self.kp = det, kp
        self.last: Hand | None = None
        self.flip_age = 0

    def _keypoints(
        self, img: np.ndarray, cx: float, cy: float, half: float, flip: bool, prior: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, Crop]:
        crop128, crop = crop_square(img, cx, cy, half)
        out = self.kp.run(keypoint_feeds(crop128, crop, flip, prior))
        pts, conf = decode_heatmaps(out["heatmap_xy"], crop, flip)
        return pts, conf, crop

    def _best_flip(
        self, img: np.ndarray, cx: float, cy: float, half: float, prior: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, Crop, bool]:
        if self.last is not None and self.flip_age < FLIP_RECHECK:
            pts, conf, crop = self._keypoints(img, cx, cy, half, self.last.flip, prior)
            if float(conf.mean()) >= 0.25:
                self.flip_age += 1
                return pts, conf, crop, self.last.flip
        a = self._keypoints(img, cx, cy, half, False, prior)
        b = self._keypoints(img, cx, cy, half, True, prior)
        self.flip_age = 0
        if float(b[1].mean()) > float(a[1].mean()):
            return (*b, True)
        return (*a, False)

    def track_at(
        self, img: np.ndarray, cx: float, cy: float, half: float, flip: bool, prior: np.ndarray | None
    ) -> Hand:
        """Keypoints in a crop placed from outside (the other eye, through the 3D hand)."""
        pts, conf, crop = self._keypoints(img, cx, cy, half, flip, prior)
        return Hand(pts, conf, crop, flip, False, float(conf.mean()))

    def update(self, img: np.ndarray) -> Hand | None:
        canvas, lb = letterbox(img)
        best = decode_detections(self.det.run({"inputImg": normalise(canvas)[None, None]}), lb)[0]
        prior = self.last.pts if self.last is not None else None
        if best.exists >= HAND_MIN:
            if (
                self.last is not None
                and np.hypot(best.x - self.last.pts[:, 0].mean(), best.y - self.last.pts[:, 1].mean())
                > self.last.crop.width
            ):
                prior = None  # a different place: do not feed the old joints
            pts, conf, crop, flip = self._best_flip(img, best.x, best.y, 0.5 * best.size * CROP_SCALE, prior)
            self.last = Hand(pts, conf, crop, flip, True, best.exists)
            return self.last
        if self.last is not None:  # coast on the previous joints
            c = self.last.pts.mean(axis=0)
            ext = float(np.ptp(self.last.pts, axis=0).max())
            half = max(20.0, 0.7 * 0.5 * self.last.crop.width + 0.3 * 0.5 * ext * CROP_SCALE * 1.15)
            pts, conf, crop, flip = self._best_flip(img, float(c[0]), float(c[1]), half, self.last.pts)
            if float(conf.mean()) >= COAST_MIN:
                self.last = Hand(pts, conf, crop, flip, False, float(conf.mean()))
                return self.last
        self.last = None
        return None


def prepared_to_480(pts: np.ndarray) -> np.ndarray:
    """Joints in a prepared (stretched, 180-degree rotated) eye image -> unrotated 640x480 pixels."""
    return np.stack([639.0 - pts[:, 0], 479.0 - pts[:, 1]], axis=1)
