#!/usr/bin/env python3
"""Which image geometry does the Mercury detector want for Leap frames?

For frames of a raw recording, prints the best `hand_exists` score per variant: the
640x240 eye image as delivered (rows binned 2:1), stretched back to 640x480, and each of
those rotated by 90/180/270 degrees and mirrored.

Usage:
  probe_detector.py <recording.raw> --engine <detector .engine> [--every 50]
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from mercury import letterbox, normalise
from trt_engines import MultiEngine

ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def variants(eye: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for stretch, img in (("240", eye), ("480", cv2.resize(eye, (640, 480), interpolation=cv2.INTER_LINEAR))):
        for deg, code in ROTATIONS.items():
            r = img if code is None else cv2.rotate(img, code)
            out[f"{stretch}/rot{deg}"] = r
            out[f"{stretch}/rot{deg}/mirror"] = cv2.flip(r, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--every", type=int, default=50)
    args = ap.parse_args()
    det = MultiEngine(args.engine)
    frames = np.memmap(args.recording, dtype=np.uint8, mode="r").reshape(-1, 240, 640, 2)
    scores: dict[str, list[float]] = {}
    for k in range(0, len(frames), args.every):
        for e in (0, 1):
            for name, img in variants(np.ascontiguousarray(frames[k][:, :, e])).items():
                canvas, _ = letterbox(img)
                out = det.run({"inputImg": normalise(canvas)[None, None]})
                scores.setdefault(name, []).append(float(out["hand_exists"].max()))
    print(f"{'variant':24s} {'median':>7s} {'p90':>7s} {'max':>7s}  frames>=0.40")
    for name, s in sorted(scores.items(), key=lambda kv: -np.median(kv[1])):
        a = np.asarray(s)
        print(
            f"{name:24s} {np.median(a):7.3f} {np.percentile(a, 90):7.3f} {a.max():7.3f}  {int((a >= 0.4).sum())}/{a.size}"
        )


if __name__ == "__main__":
    main()
