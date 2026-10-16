"""Both eyes tracked together: when they disagree, the confident eye places the other eye's crop.

`EyeTracker` follows a hand in one eye from that eye's own previous joints. Once one eye drifts (onto
the forearm, say) nothing brings it back, and the two eyes report different joints. `StereoTracker`
runs both eye trackers, triangulates, and if the eyes disagree (or one lost the hand) it takes the eye
with the stronger hand as reference, places the reference joints at a plausible depth, projects them
into the other eye and runs the keypoint model in a crop placed there, with the reference eye's
handedness and no joint prior (so the agreement that follows comes from the image). A repair counts only if the eyes then agree (`stereo_consistent`); the
repaired eye continues from it, so the eyes stay coupled. Monado's Mercury couples its views the same
way, through the 3D hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from pipeline import COAST_MIN, EyeTracker, Hand, Model, prepared_to_480
from stereo import MAX_EPIPOLAR_PX, Calibration, LensModel, project, rays, stereo_consistent, triangulate

MIN_JOINTS = 10  # confident joints (both eyes) needed for a 3D hand
JOINT_CONF = 0.3
DEPTHS_MM = (200.0, 300.0, 400.0)  # tried after the last good depth when repairing
SAME_PLACE = 0.15  # crop centres closer than this x crop width count as the same placement


@dataclass
class StereoResult:
    hands: Tuple[Optional[Hand], Optional[Hand]]
    pts: Optional[np.ndarray] = None  # [21,3] mm, left rectified frame
    err: Optional[np.ndarray] = None  # [21] epipolar error, px
    ok: Optional[np.ndarray] = None  # [21] joint confident in both eyes
    consistent: bool = False
    repaired: Optional[int] = None  # index of the eye that was repaired, if any


class StereoTracker:
    def __init__(
        self, det: Model, kp: Model, cal: Calibration, model: LensModel, max_epipolar: float = MAX_EPIPOLAR_PX
    ) -> None:
        self.eyes = (EyeTracker(det, kp), EyeTracker(det, kp))
        self.cal, self.model, self.max_epipolar = cal, model, max_epipolar
        self.depth = 250.0  # last good median hand depth, mm
        self.repair = True

    def _cams(self, eye: int):
        return self.cal.left if eye == 0 else self.cal.right

    def _offset(self, eye: int) -> np.ndarray:
        """Camera centre in the left rectified frame: the right camera sits one baseline along +x."""
        return np.array([0.0 if eye == 0 else self.cal.baseline_mm, 0.0, 0.0])

    def _solve(self, hands: Sequence[Optional[Hand]]) -> StereoResult:
        res = StereoResult((hands[0], hands[1]))
        if hands[0] is None or hands[1] is None:
            return res
        pts, err = triangulate(self.cal, prepared_to_480(hands[0].pts), prepared_to_480(hands[1].pts), self.model)
        ok = np.minimum(hands[0].conf, hands[1].conf) > JOINT_CONF
        res.pts, res.err, res.ok = pts, err, ok
        res.consistent = int(ok.sum()) >= MIN_JOINTS and stereo_consistent(err, ok, self.max_epipolar)
        return res

    def _centre(self, ref: int, hand: Hand, depth: float) -> np.ndarray:
        """Where the reference eye's hand lands in the other eye if it is `depth` mm away (prepared pixels)."""
        other = 1 - ref
        r = rays(self._cams(ref), prepared_to_480(hand.pts), self.model)
        pts3 = r * (depth / r[:, 2:3]) + self._offset(ref)
        px = prepared_to_480(project(self._cams(other), pts3 - self._offset(other), self.model))
        return np.median(px, axis=0)

    def update(self, imgs: Sequence[np.ndarray]) -> StereoResult:
        hands: List[Optional[Hand]] = [self.eyes[0].update(imgs[0]), self.eyes[1].update(imgs[1])]
        res = self._solve(hands)
        if not res.consistent and self.repair:
            refs = [i for i in (0, 1) if hands[i] is not None]
            refs.sort(key=lambda i: (hands[i].detected, float(hands[i].conf.mean())), reverse=True)  # type: ignore[union-attr]
            for ref in refs:
                href = hands[ref]
                assert href is not None
                tried: List[np.ndarray] = []
                for depth in dict.fromkeys((self.depth,) + DEPTHS_MM):
                    c = self._centre(ref, href, depth)
                    if any(np.hypot(*(c - t)) < SAME_PLACE * href.crop.width for t in tried):
                        continue  # this depth puts the crop where one was already tried
                    tried.append(c)
                    trial = list(hands)
                    # no joint prior: agreement must come from the image, not from the projection
                    trial[1 - ref] = self.eyes[1 - ref].track_at(
                        imgs[1 - ref], float(c[0]), float(c[1]), 0.5 * href.crop.width, href.flip, None
                    )
                    t = self._solve(trial)
                    if t.consistent and float(trial[1 - ref].conf.mean()) >= COAST_MIN:  # type: ignore[union-attr]
                        self.eyes[1 - ref].last = trial[1 - ref]
                        t.repaired = 1 - ref
                        res = t
                        break
                if res.consistent:
                    break
        if res.consistent and res.pts is not None and res.ok is not None:
            self.depth = float(np.median(res.pts[res.ok, 2]))
        return res
