#!/usr/bin/env python3
"""Decide how to read the controller's stored lens model, from recorded hand frames.

Runs the GPU keypoint pipeline on both eyes of a raw recording, keeps frames where both eyes
track a confident hand, and scores each reading of the calibration (`stereo.LensModel`:
exponent of the radial function, sign of the Cayley rotations, tangential on/off):

- epipolar error: after rectification the same joint must sit on the same row in both eyes;
- bone-length spread: a rigid finger bone keeps its length from frame to frame, so the
  coefficient of variation of the triangulated lengths should be small;
- hand depth and bone lengths, for plausibility (the hand was held ~20 cm above the device).

Usage:
  fit_lens_model.py <recording.raw> --calib calibration.bin --models <engine dir> [--every 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mercury import prepare_eye
from pipeline import EyeTracker, prepared_to_480
from stereo import Calibration, LensModel, triangulate
from trt_engines import MultiEngine

DET, KP = "grayscale_detection_160x160", "grayscale_keypoint_jan18"
MIN_JOINT_CONF = 0.30
# finger bones by joint index (Mercury order): proximal, intermediate, distal phalanges
BONES = {
    f"{name}_{seg}": (b + i, b + i + 1)
    for name, b in (("index", 5), ("middle", 9), ("ring", 13), ("little", 17))
    for i, seg in enumerate(("prox", "int", "dist"))
}
BONES["palm_wrist_middle"] = (0, 9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--json")
    args = ap.parse_args()

    cal = Calibration.from_bytes(Path(args.calib).read_bytes())
    m = Path(args.models)
    det, kp = MultiEngine(m / f"{DET}.fp16.engine"), MultiEngine(m / f"{KP}.fp16.engine")
    left, right = EyeTracker(det, kp), EyeTracker(det, kp)
    frames = np.memmap(args.recording, dtype=np.uint8, mode="r").reshape(-1, 240, 640, 2)
    pairs = []
    for k in range(0, len(frames), args.every):
        raw = np.asarray(frames[k])
        hl = left.update(prepare_eye(np.ascontiguousarray(raw[:, :, 0])))
        hr = right.update(prepare_eye(np.ascontiguousarray(raw[:, :, 1])))
        if hl is None or hr is None:
            continue
        ok = (hl.conf > MIN_JOINT_CONF) & (hr.conf > MIN_JOINT_CONF)
        if ok.sum() >= 15:
            pairs.append((prepared_to_480(hl.pts), prepared_to_480(hr.pts), ok))

    results: List[Dict[str, Any]] = []
    for power in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        for rot in (-1.0, 1.0):
            for tang in (False, True):
                model = LensModel(power=power, rotation_sign=rot, tangential=tang)
                epi: List[float] = []
                depth: List[float] = []
                lengths: Dict[str, List[float]] = {b: [] for b in BONES}
                for pl, pr, ok in pairs:
                    pts, err = triangulate(cal, pl, pr, model)
                    epi += list(err[ok])
                    depth.append(float(np.median(pts[ok, 2])))
                    for b, (i, j) in BONES.items():
                        if ok[i] and ok[j]:
                            lengths[b].append(float(np.linalg.norm(pts[i] - pts[j])))
                cv = [np.std(v) / np.mean(v) for v in lengths.values() if len(v) > 5]
                results.append(
                    {
                        "power": power,
                        "rotation_sign": rot,
                        "tangential": tang,
                        "epipolar_px_median": round(float(np.median(epi)), 3),
                        "epipolar_px_p90": round(float(np.percentile(epi, 90)), 3),
                        "bone_length_cv_median": round(float(np.median(cv)), 4),
                        "hand_depth_mm_median": round(float(np.median(depth)), 1),
                        "bone_mm_median": {b: round(float(np.median(v)), 1) for b, v in lengths.items() if v},
                    }
                )
    results.sort(key=lambda r: (r["epipolar_px_median"], r["bone_length_cv_median"]))
    print(f"stereo frames with a confident hand in both eyes: {len(pairs)}")
    print(
        f"{'power':>5s} {'rot':>4s} {'tang':>5s} {'epi_med':>8s} {'epi_p90':>8s} {'bone_cv':>8s} {'depth':>7s}  index prox/int/dist  wrist-middle"
    )
    for r in results:
        b = r["bone_mm_median"]
        print(
            f"{r['power']:5.1f} {r['rotation_sign']:4.0f} {str(r['tangential']):>5s} {r['epipolar_px_median']:8.3f} {r['epipolar_px_p90']:8.3f} "
            f"{r['bone_length_cv_median']:8.4f} {r['hand_depth_mm_median']:7.1f}  "
            f"{b.get('index_prox', 0):.0f}/{b.get('index_int', 0):.0f}/{b.get('index_dist', 0):.0f}  {b.get('palm_wrist_middle', 0):.0f}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps({"pairs": len(pairs), "results": results}, indent=1))


if __name__ == "__main__":
    main()
