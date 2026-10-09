"""Leap Motion stereo calibration (read from the device) and keypoint triangulation.

The 156-byte block (`leap_usb_calib.py`) holds, little endian: 4 bytes (signature 'C','A',
version, score), a u32 timestamp, 36 floats and a u32 checksum. Floats: baseline (mm),
one unused value, then per camera (left, right) 17 values: focal length, principal-point
offset x/y (pixels from the 640x480 centre), 2 tangential and 6 radial coefficients, a
second focal length and centre (unused here) and a 3-parameter Cayley rotation.

The radial coefficients describe a rational function w(r2) = (1 + r2(k1 + r2(k2 + r2 k3))) /
(1 + r2(k4 + r2(k5 + r2 k6))) in Leap's inverse form (it maps distorted to undistorted
coordinates). Keypoints are therefore undistorted directly, without image remapping: only the
21 joints per eye are corrected. Which exponent and rotation sign apply was settled on
recorded hand frames by the epipolar check (`fit_lens_model.py`), not assumed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

STRUCT = "<BBBBI" + "f" * 36 + "I"


def cayley(p: np.ndarray) -> np.ndarray:
    """Rotation matrix from Cayley (Gibbs) parameters."""
    x, y, z = (float(v) for v in p)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    i = np.eye(3)
    return (i + k) @ np.linalg.inv(i - k)


@dataclass
class Camera:
    focal: float
    cx: float  # pixels in the 640x480 eye image
    cy: float
    tangential: np.ndarray  # (2,)
    radial: np.ndarray  # (6,)
    rotation: np.ndarray  # (3,) Cayley parameters

    def warp(self, r2: np.ndarray) -> np.ndarray:
        k1, k2, k3, k4, k5, k6 = self.radial
        return (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6)))


@dataclass
class LensModel:
    """How to read the stored coefficients; chosen by fit_lens_model.py."""

    power: float = 1.0  # undistorted = distorted * w(r2)**power
    rotation_sign: float = -1.0  # R = cayley(rotation_sign * rotation)
    tangential: bool = False


@dataclass
class Calibration:
    baseline_mm: float
    left: Camera
    right: Camera
    signature: tuple[int, int, int, int]

    @classmethod
    def from_bytes(cls, raw: bytes) -> Calibration:
        a = struct.unpack(STRUCT, raw[: struct.calcsize(STRUCT)])
        if (a[0], a[1]) != (ord("C"), ord("A")):
            msg = f"calibration signature {a[0]},{a[1]} is not 'CA'; read failed?"
            raise ValueError(msg)
        cams = []
        for off in (7, 24):
            cams.append(
                Camera(
                    focal=a[off],
                    cx=320.0 + a[off + 1],
                    cy=240.0 + a[off + 2],
                    tangential=np.array(a[off + 3 : off + 5], np.float64),
                    radial=np.array(a[off + 5 : off + 11], np.float64),
                    rotation=np.array(a[off + 14 : off + 17], np.float64),
                )
            )
        return cls(baseline_mm=a[5], left=cams[0], right=cams[1], signature=(a[0], a[1], a[2], a[3]))


def rays(cam: Camera, px480: np.ndarray, model: LensModel) -> np.ndarray:
    """Pixels [N,2] in the unrotated 640x480 eye image -> unit-free rays [N,3] in the rectified frame."""
    xn = (px480[:, 0] - cam.cx) / cam.focal
    yn = (px480[:, 1] - cam.cy) / cam.focal
    if model.tangential:
        p1, p2 = cam.tangential
        r2 = xn * xn + yn * yn
        xn, yn = xn + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn), yn + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    s = cam.warp(xn * xn + yn * yn) ** model.power
    v = np.stack([xn * s, yn * s, np.ones_like(xn)], axis=1)
    return v @ cayley(model.rotation_sign * cam.rotation).T


def triangulate(
    cal: Calibration, left480: np.ndarray, right480: np.ndarray, model: LensModel
) -> tuple[np.ndarray, np.ndarray]:
    """Joints [N,2] in both eyes -> points [N,3] in mm (left-camera rectified frame) and the
    vertical mismatch [N] in pixels (epipolar error; 0 for a perfect match)."""
    rl, rr = rays(cal.left, left480, model), rays(cal.right, right480, model)
    xl, yl = rl[:, 0] / rl[:, 2], rl[:, 1] / rl[:, 2]
    xr, yr = rr[:, 0] / rr[:, 2], rr[:, 1] / rr[:, 2]
    disparity = xl - xr
    sign = 1.0 if np.median(disparity) > 0 else -1.0
    z = cal.baseline_mm / np.maximum(sign * disparity, 1e-6)
    pts = np.stack([xl * z, 0.5 * (yl + yr) * z, z], axis=1)
    focal = 0.5 * (cal.left.focal + cal.right.focal)
    return pts, np.abs(yl - yr) * focal
