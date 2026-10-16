#!/usr/bin/env python3
"""Plumb-line test of the lens model: straight edges must stay straight after undistortion.

Record a rigid sheet with straight edges (paper taped to a book) moved slowly through the view, then:

  plumb_line.py <recording.raw> --calib calibration.bin [--every 10]

For each frame and eye the brightest large region (the sheet) is outlined; its long sides are refined
to sub-pixel edge points along their normals (middle 70% of each side, profiles with exactly one clean
step only, so fingers and corners drop out). For each exponent of the stored radial warp
(`stereo.LensModel.power`) the points are undistorted and a line is fitted per edge; the score is the
RMS distance from that line in pixels. Edges that are not straight to 0.6 px without any correction
(a bent sheet, a finger on the edge) are left out. A bootstrap over edges gives the spread of the best
exponent. Rotation and tangential terms map lines to lines, so only the radial warp is tested.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

from stereo import Calibration, Camera

POWERS = np.round(np.arange(-0.25, 1.51, 0.05), 2)
MIN_SIDE_PX = 120  # long sides only: sheet edges, not fingers
CLEAN_RMS_PX = 0.6


def edge_points(img480: np.ndarray) -> List[np.ndarray]:
    """Sub-pixel points [N,2] (x, y in the 640x480 eye image) on each long side of the sheet."""
    _, mask = cv2.threshold(img480, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return []
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < 8000:
        return []
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True).reshape(-1, 2).astype(float)
    imgf = img480.astype(np.float32)
    steps = np.arange(-10, 10.01, 0.25)
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        d = b - a
        length = float(np.hypot(*d))
        if length < MIN_SIDE_PX:
            continue
        normal = np.array([-d[1], d[0]]) / length
        pts = []
        for t in np.linspace(0.15, 0.85, int(length * 0.35)):
            p = a + t * d
            xs = (p[0] + steps * normal[0]).astype(np.float32)[None, :]
            ys = (p[1] + steps * normal[1]).astype(np.float32)[None, :]
            prof = cv2.GaussianBlur(cv2.remap(imgf, xs, ys, cv2.INTER_LINEAR), (1, 5), 0)[0]
            lo, hi = np.percentile(prof, 10), np.percentile(prof, 90)
            if hi - lo < 40:
                continue
            level = 0.5 * (lo + hi)
            above = prof > level
            flips = np.nonzero(above[1:] != above[:-1])[0]
            if len(flips) != 1:  # a finger or clutter: not one clean step
                continue
            j = int(flips[0])
            s = steps[j] + (level - prof[j]) / (prof[j + 1] - prof[j]) * (steps[j + 1] - steps[j])
            pts.append(p + s * normal)
        if len(pts) >= 20:
            out.append(np.array(pts))
    return out


def undistort(cam: Camera, px: np.ndarray, power: float) -> np.ndarray:
    """Pixels -> normalised undistorted coordinates, with the stored warp raised to `power`."""
    xd = (px[:, 0] - cam.cx) / cam.focal
    yd = (px[:, 1] - cam.cy) / cam.focal
    s = cam.warp(xd * xd + yd * yd) ** power
    return np.stack([xd * s, yd * s], axis=1)


def straightness(p: np.ndarray) -> float:
    """RMS distance from the best-fit line, dropping points beyond 3 sigma (twice)."""
    keep = np.ones(len(p), bool)
    d = np.zeros(len(p))
    for _ in range(3):
        centre = p[keep].mean(axis=0)
        _, _, vt = np.linalg.svd(p[keep] - centre, full_matrices=False)
        d = (p - centre) @ vt[1]
        keep = np.abs(d) < max(3 * float(np.std(d[keep])), 1e-12)
    return float(np.sqrt(np.mean(d[keep] ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--every", type=int, default=10)
    args = ap.parse_args()

    cal = Calibration.from_bytes(Path(args.calib).read_bytes())
    frames = np.memmap(args.recording, dtype=np.uint8, mode="r").reshape(-1, 240, 640, 2)
    rng = np.random.default_rng(0)
    for eye, cam in ((0, cal.left), (1, cal.right)):
        edges: List[np.ndarray] = []
        for k in range(0, len(frames), args.every):
            img = cv2.resize(np.ascontiguousarray(frames[k, :, :, eye]), (640, 480))
            edges += edge_points(img)
        if not edges:
            print(f"eye {eye}: no edges found")
            continue
        rms = np.array([[straightness(undistort(cam, e, p)) * cam.focal for p in POWERS] for e in edges])
        clean = rms[:, list(POWERS).index(0.0)] < CLEAN_RMS_PX
        rc = rms[clean]
        if len(rc) == 0:
            print(f"eye {eye}: no edge straighter than {CLEAN_RMS_PX} px without correction")
            continue
        med = np.median(rc, axis=0)
        boots = [POWERS[np.argmin(np.median(rc[rng.integers(0, len(rc), len(rc))], axis=0))] for _ in range(500)]
        radius = np.concatenate([np.hypot(*undistort(cam, e, 0.0).T) for e in edges])
        print(
            f"eye {eye}: {len(rc)} clean edges of {len(edges)}, normalised radius p50 {np.median(radius):.2f} "
            f"p95 {np.percentile(radius, 95):.2f}; best exponent {POWERS[np.argmin(med)]} "
            f"(bootstrap 5-95% {np.percentile(boots, 5):.2f}..{np.percentile(boots, 95):.2f})"
        )
        for p in (0.0, 0.1, 0.5, 1.0):
            print(f"   exponent {p:4.1f}: median RMS from straight {med[list(POWERS).index(p)]:.3f} px")


if __name__ == "__main__":
    main()
