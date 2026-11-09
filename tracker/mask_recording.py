#!/usr/bin/env python3
"""Cut a section of a raw recording and black out everything but the hand, in both eyes, frame by frame.

A recording of the controller can show the person's head, other arm or the room. For each frame and
eye this keeps only the convex outline of the 21 joints (from the `--out` file of `live_track.py`)
plus a margin. Everything else becomes 0, so what can leak is only what touches the hand: still look
at every frame, brightened, before publishing the result.

Usage:
  mask_recording.py rec.raw tracks.jsonl --start 1800 --frames 460 [--margin 16] -o sample.raw
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

HEIGHT = 240  # rows in the raw frame; keypoints are in a 640x480 image, so raw rows are every second row
WIDTH = 640


def hand_mask(joints_480: list[list[float]], margin: int) -> np.ndarray:
    """A boolean (240, 640) mask: the convex outline of the joints, grown by `margin` 640x480 pixels."""
    canvas = np.zeros((HEIGHT * 2, WIDTH), np.uint8)
    points = np.round(np.asarray(joints_480, dtype=np.float32)).astype(np.int32)
    cv2.fillConvexPoly(canvas, cv2.convexHull(points), 1)
    size = 2 * margin + 1
    canvas = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return canvas[::2].astype(bool)


def mask_frame(frame: np.ndarray, left: list[list[float]], right: list[list[float]], margin: int) -> np.ndarray:
    """A copy of one raw frame (240, 640, 2) with each eye kept only inside its own hand outline."""
    out = np.zeros_like(frame)
    for channel, joints in enumerate((left, right)):
        keep = hand_mask(joints, margin)
        out[..., channel][keep] = frame[..., channel][keep]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("tracks", help="JSON lines from live_track.py --out, one per frame of the recording")
    ap.add_argument("--start", type=int, required=True, help="first frame, 0-based")
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--margin", type=int, default=16, help="around the joints, in 640x480 pixels")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    with open(args.tracks) as handle:
        rows = [json.loads(line) for line in handle]
    section = rows[args.start : args.start + args.frames]
    if len(section) != args.frames or not all(r.get("left_px480") and r.get("right_px480") for r in section):
        raise SystemExit("the section must lie in the file, with the hand tracked in both eyes in every frame")

    frames = np.memmap(args.recording, dtype=np.uint8, mode="r").reshape(-1, HEIGHT, WIDTH, 2)
    with open(args.out, "wb") as out:
        for index, row in enumerate(section):
            frame = np.asarray(frames[args.start + index])
            mask_frame(frame, row["left_px480"], row["right_px480"], args.margin).tofile(out)
    print(f"{args.frames} frames masked to the hand, margin {args.margin}")


if __name__ == "__main__":
    main()
