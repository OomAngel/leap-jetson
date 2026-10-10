# leap-jetson

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Platform: Jetson AGX Xavier · JetPack 5.1 · TensorRT 8.5](https://img.shields.io/badge/platform-Jetson%20AGX%20Xavier%20%C2%B7%20JetPack%205.1%20%C2%B7%20TensorRT%208.5-76b900.svg)

3D hand tracking for the **original Leap Motion Controller** (LMC1, USB `f182:0003`) on **NVIDIA Jetson
GPUs**: TensorRT FP16, no Ultraleap software.

Ultraleap never shipped hand tracking for this controller on ARM Linux, and its download servers are
gone. leap-jetson reads the controller as a plain USB camera, runs Monado's open "Mercury" hand models
in TensorRT, and triangulates 21 joints per hand with the calibration stored on the controller.

> **Status: working prototype.** Last validated 2026-10-10 on a Jetson AGX Xavier (JetPack 5.1,
> L4T 35.6.5, MAXN, GPU clock pinned). Live tracking runs at camera rate. 3D is correct near the
> centre of the view; towards the edges the lens model is not settled yet (see [Limits](#limits)).

*Not affiliated with, endorsed by or supported by Ultraleap. Leap Motion is a trademark of its owner.*

## Results

| | |
|---|---|
| Camera | 640×240 per eye at 115 fps (UVC, `uvcvideo`) |
| Tracking rate, hand in view | **98 fps** (27 fps with the default GPU governor → 67.5 fps with the GPU clock pinned → 98 fps with vectorised heatmap decoding) |
| Latency, camera frame → 3D joints | 14 ms median (same run as the 98 fps) |
| FP16 TensorRT vs ONNX FP32 | joints differ by 0.04 px on average (max 1.6 px) |
| Stereo consistency, centre of view | 1.4–1.6 px median between the eyes |
| Bone lengths, centre of view | index finger 38 / 26 / 24 mm, wrist to middle knuckle 90–96 mm |

Each figure, the run it comes from and the method are in [docs/RESULTS.md](docs/RESULTS.md). The 98 fps run
had the hand nearer the edges of the view, where the open lens-model question shows (2.56 px mismatch).

## Tested on

| Board | JetPack / L4T | TensorRT | Python | Controller firmware | Status |
|---|---|---|---|---|---|
| Jetson AGX Xavier (32 GB) | 5.1 / 35.6.5 | 8.5.2 | 3.8 | 1.7.1 | working |
| Jetson Orin family | 6.x | — | — | — | untested |

## How it works

```text
uvcvideo YUYV 640x240 @115 fps  (even bytes = left eye, odd bytes = right eye)
  -> each eye: stretch to 640x480, rotate 180°                       mercury.prepare_eye
  -> hand detector 160x160          TensorRT FP16                     pipeline.EyeTracker
  -> 128x128 crop -> 21 keypoint heatmaps (FP16), both handedness orientations,
     previous joints as a temporal prior, coasting when the detector loses a far hand
  -> both eyes: undistort the 21 joints with the controller's own lens model,
     rectify, triangulate (baseline 40 mm)                            stereo.triangulate
```

- **Calibration** lives on the controller (156 bytes). It is read with two standard UVC control requests
  (`tracker/leap_usb_calib.py`). Linux's `uvcvideo` caches control values, so the read goes through libusb
  and detaches the driver for about a second.
- **Orientation:** a controller lying face up sees hands upside down compared with Mercury's training
  views. The detector needs the binned 640×240 image stretched back to 640×480 and rotated by 180°:
  median detector score 0.74 against 0.006 for the raw image.
- **Only keypoints are undistorted** (21 per eye), never whole images.

## Requirements

- An original Leap Motion Controller. Tested only with firmware **1.7.1**, which streams as a plain UVC
  camera. Older firmware may stay silent until a bring-up sequence is sent (OpenLeap documents one for
  1.7.0) or need a firmware update. This project never flashes firmware.
- A Jetson with JetPack 5: system Python 3.8 with TensorRT 8.5 and OpenCV, plus numpy and libusb-1.0.
- Write access to the controller's USB device node, e.g. a udev rule:
  `SUBSYSTEM=="usb", ATTRS{idVendor}=="f182", ATTRS{idProduct}=="0003", GROUP="plugdev", MODE="0660"`.
- No Ultraleap service (`leapd`, `libtrack_server`) running: they detach `uvcvideo` from the controller.
  Replug the controller after one has run.

## Quick start

1. **Models (not included).** Download Monado's models yourself and check them against the
   checksums this project was tested with:
   ```bash
   git clone https://gitlab.freedesktop.org/monado/utilities/hand-tracking-models.git
   git -C hand-tracking-models checkout 37a8a81
   (cd hand-tracking-models && sha256sum -c ../models.sha256)
   ```
2. **Engines.** Build the TensorRT engines on the Jetson itself (they are specific to the GPU and the
   TensorRT version):
   ```bash
   bash tracker/build_engines.sh hand-tracking-models
   ```
3. **GPU clock.** The Jetson's GPU governor keeps the clock low for this short, bursty load (about
   217 MHz on a Xavier, about 3× slower). Raise the minimum for the session (root, until reboot):
   ```bash
   G=/sys/devices/17000000.gv11b/devfreq/17000000.gv11b   # AGX Xavier; other boards differ
   cat $G/max_freq | sudo tee $G/min_freq
   ```
4. **Calibration.** Read it from the controller:
   ```bash
   python3 tracker/leap_usb_calib.py -o calibration.bin
   ```
5. **Run.** Find the video node with `v4l2-ctl --list-devices`, then:
   ```bash
   python3 tracker/live_track.py --calib calibration.bin --models hand-tracking-models \
       --device 6 --seconds 20 --out frames.jsonl --snapshots snapshots/
   ```
   `frames.jsonl` holds one line per frame: 3D joints in mm (left-camera frame), per-eye 2D joints,
   confidences and the stereo mismatch. TensorRT engines and recordings stay on your device; this
   project never distributes engines.

## Tools

| Script | What |
|---|---|
| `tracker/live_track.py` | live tracking; JSON lines, stats, annotated snapshots |
| `tracker/leap_usb_calib.py` | reads the controller's calibration over libusb |
| `tracker/build_engines.sh` | builds TensorRT engines from the ONNX models with `trtexec` |
| `tracker/check_precision.py` | ONNX Runtime FP32 vs TensorRT FP32 vs FP16 on a recording |
| `tracker/probe_detector.py` | detector score for each image geometry |
| `tracker/fit_lens_model.py` | chooses how to read the stored lens model (stereo mismatch, bone-length spread) |

Record raw frames for the offline tools with
`v4l2-ctl -d /dev/video6 --set-fmt-video=width=640,height=240,pixelformat=YUYV --stream-mmap --stream-count=1150 --stream-to=rec.raw`.

## Limits

- **Edge of view:** the radial exponent of the stored lens model is not settled. With hands near the
  edges, the stereo mismatch rises to about 2.6 px and bones come out up to ~45% too long. Fix in
  progress: refit on edge recordings with the 2D joints now logged in `frames.jsonl`.
- One hand is tracked. The detector has two slots, but the 3D path takes the best hand per eye.
- Fists and heavily occluded poses are the weakest, as in any keypoint model.
- No smoothing in 3D yet.

## Roadmap

1. Settle the lens model at the edges of the view (refit on edge recordings).
2. Replay mode: run the tracker on recorded frames, so it can be tried without the hardware.
3. 3D smoothing (One-Euro filter) and two hands.
4. ROS 2 node publishing `/hand_tracking/raw`.
5. Test on Jetson Orin (JetPack 6).

## Fallback: Ultraleap V2 under box64

If you need Ultraleap's own tracking on a Jetson, the 2015 V2 service (x64) runs under the box64 emulator
at 80–90 fps once a box64 race is fixed. Newer Ultraleap services do not run there. See
[fallback_v2_box64/](fallback_v2_box64/README.md). This route uses Ultraleap's software and is governed
by its licence terms; you obtain that software yourself.

## Licence and credits

- Code in this repository: **Apache-2.0** (see `LICENSE` and `NOTICE`).
- **Hand models:** Monado's Mercury models (`grayscale_detection_160x160.onnx`,
  `grayscale_keypoint_jan18.onnx`, commit `37a8a81`) from the Monado project,
  https://gitlab.freedesktop.org/monado/utilities/hand-tracking-models. **Not included;** you download
  them yourself. The upstream repository carries no licence file; OpenLeap describes the models as
  CC BY-SA 4.0. Confirmation was asked of the Monado maintainers on 2026-10-10
  ([issue #1](https://gitlab.freedesktop.org/monado/utilities/hand-tracking-models/-/work_items/1)).
- Model conventions follow Monado's Mercury (BSL-1.0) as documented by
  [OpenLeap](https://github.com/julianjc84/OpenLeap) (BSL-1.0).
- The calibration read follows the controller behaviour described by Leap Motion's LeapUVC; no LeapUVC
  code is used, and frames come from the kernel's `uvcvideo` driver, never from LeapUVC.
- `fallback_v2_box64/box64-cleanup-lock.patch` modifies box64 (MIT, see `fallback_v2_box64/LICENSE.box64`).

Details in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [docs/LEGAL.md](docs/LEGAL.md) (not legal
advice).

## Citation and support

Use GitHub's "Cite this repository" (from `CITATION.cff`). One maintainer, best effort: report bugs with
the issue form, which asks for your board, JetPack, TensorRT and model versions.
