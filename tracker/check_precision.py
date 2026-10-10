#!/usr/bin/env python3
"""Step 1 of the GPU plan: do the TensorRT engines give the same hands as the ONNX models?

For frames of a raw Leap recording (YUYV 640x240: even bytes left eye, odd bytes right),
prepared as the detector expects (`mercury.prepare_eye`), both eyes are run through the detector and the keypoint model with three backends:
ONNX Runtime FP32 on the CPU (reference), TensorRT FP32 and TensorRT FP16. The keypoint
model gets the crop from the reference detection, in both handedness orientations, so all
backends see identical inputs. Reports detection agreement and joint differences in eye
pixels.

Usage:
  check_precision.py <recording.raw> --models <dir with .onnx and .engine> [--every 25]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mercury import (
    crop_square,
    decode_detections,
    decode_heatmaps,
    keypoint_feeds,
    letterbox,
    normalise,
    prepare_eye,
)
from ort_engine import OrtModel
from trt_engines import MultiEngine

DET, KP = "grayscale_detection_160x160", "grayscale_keypoint_jan18"
HAND_MIN = 0.40  # detection threshold used by OpenLeap
CROP_SCALE = 1.3


def eyes(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return prepare_eye(np.ascontiguousarray(raw[:, :, 0])), prepare_eye(np.ascontiguousarray(raw[:, :, 1]))


def stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, np.float64)
    if a.size == 0:
        return {}
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "max": round(float(a.max()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--models", required=True)
    ap.add_argument("--every", type=int, default=25, help="use every n-th frame")
    ap.add_argument("--json", help="write the summary here too")
    args = ap.parse_args()

    m = Path(args.models)
    det: Dict[str, Any] = {
        "ort": OrtModel(m / f"{DET}.onnx", ["CPUExecutionProvider"]),
        "trt32": MultiEngine(m / f"{DET}.fp32.engine"),
        "trt16": MultiEngine(m / f"{DET}.fp16.engine"),
    }
    kp: Dict[str, Any] = {
        "ort": OrtModel(m / f"{KP}.onnx", ["CPUExecutionProvider"]),
        "trt32": MultiEngine(m / f"{KP}.fp32.engine"),
        "trt16": MultiEngine(m / f"{KP}.fp16.engine"),
    }
    frames = np.memmap(args.recording, dtype=np.uint8, mode="r").reshape(-1, 240, 640, 2)

    det_diff: Dict[str, List[float]] = {b: [] for b in ("trt32", "trt16")}  # max |output - reference| per call
    exists_flip = {b: 0 for b in ("trt32", "trt16")}  # threshold decisions that differ
    centre_px: Dict[str, List[float]] = {b: [] for b in ("trt32", "trt16")}
    joint_px: Dict[str, List[float]] = {b: [] for b in ("trt32", "trt16")}
    conf_diff: Dict[str, List[float]] = {b: [] for b in ("trt32", "trt16")}
    hm_diff: Dict[str, List[float]] = {b: [] for b in ("trt32", "trt16")}
    timing: Dict[str, Dict[str, List[float]]] = {b: {"det": [], "kp": []} for b in det}
    n_eye_frames = n_hands = 0

    for k in range(0, len(frames), args.every):
        for img in eyes(np.asarray(frames[k])):
            n_eye_frames += 1
            canvas, lb = letterbox(img)
            feed = {"inputImg": normalise(canvas)[None, None]}
            out = {}
            for b, model in det.items():
                t = time.perf_counter()
                out[b] = model.run(feed)
                timing[b]["det"].append(time.perf_counter() - t)
            ref = decode_detections(out["ort"], lb)
            for b in ("trt32", "trt16"):
                det_diff[b].append(max(float(np.abs(out[b][n] - out["ort"][n]).max()) for n in out["ort"]))
                dets = decode_detections(out[b], lb)
                exists_flip[b] += sum((r.exists >= HAND_MIN) != (d.exists >= HAND_MIN) for r, d in zip(ref, dets))
                if ref[0].exists >= HAND_MIN:
                    centre_px[b].append(float(np.hypot(dets[0].x - ref[0].x, dets[0].y - ref[0].y)))
            if ref[0].exists < HAND_MIN:
                continue
            n_hands += 1
            crop128, crop = crop_square(img, ref[0].x, ref[0].y, 0.5 * ref[0].size * CROP_SCALE)
            for flip in (False, True):
                feeds = keypoint_feeds(crop128, crop, flip)
                kout = {}
                for b, model in kp.items():
                    t = time.perf_counter()
                    kout[b] = model.run(feeds)
                    timing[b]["kp"].append(time.perf_counter() - t)
                pts_ref, conf_ref = decode_heatmaps(kout["ort"]["heatmap_xy"], crop, flip)
                for b in ("trt32", "trt16"):
                    pts, conf = decode_heatmaps(kout[b]["heatmap_xy"], crop, flip)
                    joint_px[b] += list(np.hypot(*(pts - pts_ref).T))
                    conf_diff[b] += list(np.abs(conf - conf_ref))
                    hm_diff[b].append(float(np.abs(kout[b]["heatmap_xy"] - kout["ort"]["heatmap_xy"]).max()))

    summary = {
        "recording": args.recording,
        "eye_frames": n_eye_frames,
        "eye_frames_with_hand": n_hands,
        "detector_max_abs_output_diff": {b: stats(v) for b, v in det_diff.items()},
        "detector_threshold_decisions_changed": exists_flip,
        "detector_centre_shift_px": {b: stats(v) for b, v in centre_px.items()},
        "keypoint_heatmap_max_abs_diff": {b: stats(v) for b, v in hm_diff.items()},
        "joint_shift_px": {b: stats(v) for b, v in joint_px.items()},
        "joint_confidence_abs_diff": {b: stats(v) for b, v in conf_diff.items()},
        "time_per_call_ms (incl. copies)": {
            b: {s: round(1000 * float(np.median(v)), 3) for s, v in t.items() if v} for b, t in timing.items()
        },
    }
    text = json.dumps(summary, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text)


if __name__ == "__main__":
    main()
