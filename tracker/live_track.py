#!/usr/bin/env python3
"""3D hand tracking from the original Leap Motion Controller: live on a Jetson GPU, or replayed anywhere.

Capture: the controller as a plain UVC camera (uvcvideo, YUYV 640x240 at 115 fps; even bytes
left eye, odd bytes right), read by a thread that always keeps the newest frame. Per frame:
both eyes prepared (`mercury.prepare_eye`), detector + keypoint model (`pipeline.EyeTracker`),
the 21 joints triangulated with the controller's own calibration (`stereo`). No Ultraleap
software is involved.

Backends: `trt` (TensorRT FP16 engines, Jetson) or `ort` (ONNX Runtime on the .onnx files; any
PC, CPU or CUDA). `--replay` reads a raw recording instead of the camera (paced at 115 fps, or
`--every-frame` to process all frames), so the tracker can be tried without the hardware.

Usage:
  live_track.py --calib calibration.bin --models <dir> [--device 6] [--seconds 20]
                [--backend trt|ort] [--replay rec.raw [--every-frame]]
                [--out frames.jsonl] [--snapshots DIR]
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from mercury import prepare_eye
from pipeline import EyeTracker, Model, prepared_to_480
from stereo import MAX_EPIPOLAR_PX, Calibration, LensModel, stereo_consistent, triangulate

DET, KP = "grayscale_detection_160x160", "grayscale_keypoint_jan18"
MODEL = LensModel(power=1.0, rotation_sign=-1.0, tangential=False)  # fit_lens_model.py, 2026-10-09
BONES = [(0, b) for b in (1, 5, 9, 13, 17)] + [(b + i, b + i + 1) for b in (1, 5, 9, 13, 17) for i in range(3)]


class Grabber(threading.Thread):
    def __init__(self, device: int) -> None:
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 115)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        if not self.cap.isOpened():
            msg = f"cannot open /dev/video{device} (is another program using the Leap?)"
            raise RuntimeError(msg)
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.stamp = 0.0
        self.count = 0
        self.running = True

    def run(self) -> None:
        while self.running:
            ok, f = self.cap.read()
            if ok and f.size == 640 * 240 * 2:
                with self.lock:
                    self.frame, self.stamp = f.reshape(240, 640, 2), time.perf_counter()
                    self.count += 1

    def latest(self, after: int) -> tuple[np.ndarray | None, int, float]:
        with self.lock:
            if self.count == after:
                return None, after, 0.0
            return self.frame, self.count, self.stamp


class FileSource:
    """A raw recording (YUYV 640x240 frames back to back) with the same `latest()` interface as `Grabber`.

    Paced: frames are published at the camera's 115 fps and the tracker takes the newest, as live.
    `every_frame`: each frame is handed over in turn, so none is dropped (for offline analysis).
    """

    FPS = 115.0

    def __init__(self, path: str, every_frame: bool) -> None:
        self.frames = np.memmap(path, dtype=np.uint8, mode="r").reshape(-1, 240, 640, 2)
        self.every_frame = every_frame
        self.t0 = time.perf_counter()
        self.done = False

    def start(self) -> None:
        self.t0 = time.perf_counter()

    def latest(self, after: int) -> tuple[np.ndarray | None, int, float]:
        n = len(self.frames)
        if self.every_frame:
            idx = after  # frame counts are 1-based, like the camera grabber's
        else:
            idx = min(int((time.perf_counter() - self.t0) * self.FPS), n - 1)
            if idx + 1 <= after:
                return None, after, 0.0
        if idx >= n:
            self.done = True
            return None, after, 0.0
        if not self.every_frame and idx == n - 1:
            self.done = True
        stamp = self.t0 + idx / self.FPS if not self.every_frame else time.perf_counter()
        return np.asarray(self.frames[idx]), idx + 1, stamp


def load_models(backend: str, models: Path) -> tuple[Model, Model]:
    """Detector and keypoint model for the chosen backend."""
    if backend == "trt":
        from cuda_rt import spin_wait  # before anything creates the CUDA context

        spin_wait()
        from trt_engines import MultiEngine

        return MultiEngine(models / f"{DET}.fp16.engine"), MultiEngine(models / f"{KP}.fp16.engine")
    from ort_engine import OrtModel

    return OrtModel(models / f"{DET}.onnx"), OrtModel(models / f"{KP}.onnx")


def snapshot(path: Path, eyes: tuple[np.ndarray, np.ndarray], hands: tuple) -> None:
    views = []
    for img, hand in zip(eyes, hands):
        vis = cv2.cvtColor(cv2.convertScaleAbs(img, alpha=3.0), cv2.COLOR_GRAY2BGR)
        if hand is not None:
            col = (0, 200, 0) if hand.detected else (0, 180, 220)
            for a, b in BONES:
                cv2.line(
                    vis, tuple(int(v) for v in hand.pts[a]), tuple(int(v) for v in hand.pts[b]), col, 2, cv2.LINE_AA
                )
            for p in hand.pts:
                cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1, cv2.LINE_AA)
        views.append(vis)
    cv2.imwrite(str(path), np.hstack(views))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--device", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", help="JSON lines, one per processed frame")
    ap.add_argument("--snapshots", help="directory for a few annotated frames")
    ap.add_argument("--backend", choices=("trt", "ort"), default="trt", help="TensorRT engines or ONNX Runtime")
    ap.add_argument("--replay", help="raw recording to play instead of the camera")
    ap.add_argument(
        "--max-epipolar",
        type=float,
        default=MAX_EPIPOLAR_PX,
        help="drop the 3D of a frame whose median stereo mismatch is above this (px)",
    )
    ap.add_argument("--every-frame", action="store_true", help="with --replay: process every frame, drop none")
    args = ap.parse_args()

    cal = Calibration.from_bytes(Path(args.calib).read_bytes())
    m = Path(args.models)
    det, kp = load_models(args.backend, m)
    trackers = (EyeTracker(det, kp), EyeTracker(det, kp))
    grab: Grabber | FileSource = FileSource(args.replay, args.every_frame) if args.replay else Grabber(args.device)
    grab.start()
    out = open(args.out, "w") if args.out else None  # noqa: SIM115
    snapdir = Path(args.snapshots) if args.snapshots else None
    if snapdir:
        snapdir.mkdir(parents=True, exist_ok=True)

    seen, processed, with_hand, rejected, last_snap = 0, 0, 0, 0, -1e9
    proc_ms, latency_ms, epi_px, depth_mm = [], [], [], []
    bone_mm: dict[str, list[float]] = {"index_prox": [], "index_int": [], "index_dist": [], "wrist_middle": []}
    t_start = time.perf_counter()
    t_end = t_start + args.seconds
    start_count = None
    while time.perf_counter() < t_end:
        frame, count, stamp = grab.latest(seen)
        if frame is None:
            if isinstance(grab, FileSource) and grab.done:
                break
            time.sleep(0.0005)
            continue
        if start_count is None:
            start_count = count
        seen = count
        t0 = time.perf_counter()
        eyes = (prepare_eye(np.ascontiguousarray(frame[:, :, 0])), prepare_eye(np.ascontiguousarray(frame[:, :, 1])))
        hands = (trackers[0].update(eyes[0]), trackers[1].update(eyes[1]))
        record: dict = {"t": round(stamp, 4), "frame": count}
        if hands[0] is not None and hands[1] is not None:
            pts, err = triangulate(cal, prepared_to_480(hands[0].pts), prepared_to_480(hands[1].pts), MODEL)
            conf = np.minimum(hands[0].conf, hands[1].conf)
            ok = conf > 0.3
            consistent = stereo_consistent(err, ok, args.max_epipolar)
            if ok.sum() >= 10 and not consistent:
                rejected += 1
            if ok.sum() >= 10 and consistent:
                with_hand += 1
                epi_px += list(err[ok])
                depth_mm.append(float(np.median(pts[ok, 2])))
                for name, (i, j) in {
                    "index_prox": (5, 6),
                    "index_int": (6, 7),
                    "index_dist": (7, 8),
                    "wrist_middle": (0, 9),
                }.items():
                    if ok[i] and ok[j]:
                        bone_mm[name].append(float(np.linalg.norm(pts[i] - pts[j])))
            record.update(
                left_px480=np.round(prepared_to_480(hands[0].pts), 2).tolist(),
                right_px480=np.round(prepared_to_480(hands[1].pts), 2).tolist(),
                joints_mm=np.round(pts, 1).tolist() if consistent else None,
                conf=np.round(conf, 3).tolist(),
                epipolar_px=np.round(err, 2).tolist(),
                detected=[hands[0].detected, hands[1].detected],
            )
        t1 = time.perf_counter()
        processed += 1
        proc_ms.append(1000 * (t1 - t0))
        latency_ms.append(1000 * (t1 - stamp))
        if out:
            out.write(json.dumps(record) + "\n")
        if snapdir and t1 - last_snap > 2.0 and any(h is not None for h in hands):
            snapshot(snapdir / f"frame_{count:06d}.png", eyes, hands)
            last_snap = t1
    elapsed = time.perf_counter() - t_start
    if isinstance(grab, Grabber):
        grab.running = False
    captured = seen - (start_count or seen) + 1
    if out:
        out.close()

    def med(v: list[float]) -> float | None:
        return round(float(np.median(v)), 2) if v else None

    summary = {
        "source": args.replay or f"/dev/video{args.device}",
        "backend": args.backend,
        "seconds": round(elapsed, 2),
        "camera_frames": captured,
        "processed_frames": processed,
        "processed_fps": round(processed / elapsed, 1) if elapsed > 0 else None,
        "frames_with_3d_hand": with_hand,
        "frames_rejected_stereo_mismatch": rejected,
        "processing_ms_median": med(proc_ms),
        "processing_ms_p95": round(float(np.percentile(proc_ms, 95)), 2) if proc_ms else None,
        "camera_to_result_ms_median": med(latency_ms),
        "epipolar_px_median": med(epi_px),
        "hand_depth_mm_median": med(depth_mm),
        "bone_mm_median": {k: med(v) for k, v in bone_mm.items()},
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
