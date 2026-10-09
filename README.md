# leap-jetson

3D hand tracking from the **original Leap Motion Controller** (LMC1, USB `f182:0003`) running
natively on an **NVIDIA Jetson**, on the GPU, with **no Ultraleap software**.

Ultraleap never shipped hand tracking for this controller on ARM Linux, and its servers are gone.
This project reads the controller as a plain USB camera, runs Monado's open "Mercury" hand models
in TensorRT, and triangulates 21 joints per hand with the calibration stored on the controller.

> **Status: working prototype (October 2026).** It runs live at camera rate, and 3D is correct near
> the middle of the view. Towards the edges of the view, the lens model is not settled yet and bone
> lengths come out too long (see "Limits").

![Stereo IR pair with the tracked skeleton](docs/images/stereo_skeleton.png)

## Results (Jetson AGX Xavier, JetPack 5.1 / L4T 35.6.5, MAXN, GPU clock pinned)

| | |
|---|---|
| Camera | 640×240 per eye at 115 fps (UVC, `uvcvideo`) |
| Tracking rate | 98 fps with a hand in view, 114 fps without |
| Latency, camera → 3D joints | 14 ms median |
| GPU time per model call | ~1.0 ms (FP16 TensorRT; 0.6 ms in `trtexec`) |
| FP16 vs ONNX FP32 | joints differ by 0.04 px on average (max 1.6 px) |
| Stereo consistency, centre of view | median 1.4–1.6 px between the eyes |
| Bone lengths, centre of view | index finger 38 / 26 / 24 mm, wrist to middle knuckle 90–96 mm |

Measurement scripts are in `tracker/`; the full record of how these were measured is in
[docs/RESULTS.md](docs/RESULTS.md).

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

- **Calibration** lives on the controller (156 bytes). It is read with two standard UVC
  control requests (`tracker/leap_usb_calib.py`). Linux's `uvcvideo` caches control values, so the
  read goes through libusb and detaches the driver for about a second.
- **Orientation:** a controller lying face up sees hands upside down compared with Mercury's
  training views. The detector needs the binned 640×240 image stretched back to 640×480 and
  rotated by 180°: median detector score 0.74 against 0.006 for the raw image.
- **Only keypoints are undistorted** (21 per eye), never whole images.

## Requirements

- An original Leap Motion Controller. Tested only with firmware **1.7.1**, which streams as a plain
  UVC camera. Older firmware may stay silent until a bring-up sequence is sent (OpenLeap documents
  one for 1.7.0) or need a firmware update. This project never flashes firmware.
- A Jetson with JetPack 5 (tested: AGX Xavier). It uses the system Python 3.8 with TensorRT 8.5
  and OpenCV, plus numpy and libusb-1.0.
- Write access to the controller's USB device node, e.g. a udev rule:
  `SUBSYSTEM=="usb", ATTRS{idVendor}=="f182", ATTRS{idProduct}=="0003", GROUP="plugdev", MODE="0660"`.
- No Ultraleap service (`leapd`, `libtrack_server`) running: they detach `uvcvideo` from the
  controller. Replug the controller after one has run.

## Quick start

```bash
git clone https://gitlab.freedesktop.org/monado/utilities/hand-tracking-models.git
bash tracker/build_engines.sh hand-tracking-models          # FP32 + FP16 engines, on the Jetson itself
python3 tracker/leap_usb_calib.py -o calibration.bin        # read the controller's calibration
python3 tracker/live_track.py --calib calibration.bin --models hand-tracking-models \
    --device 6 --seconds 20 --out frames.jsonl --snapshots snapshots/
```

Find the video node with `v4l2-ctl --list-devices`. `frames.jsonl` holds one line per frame:
3D joints in mm (left-camera frame), per-eye 2D joints, confidences and the stereo mismatch.

**GPU clock (important on Jetson).** The GPU governor keeps the clock low for this short, bursty
load: about 217 MHz on a Xavier, which costs about 3× in speed. Raise the GPU minimum frequency for
the session (root, until reboot):

```bash
G=/sys/devices/17000000.gv11b/devfreq/17000000.gv11b   # AGX Xavier; other boards differ
cat $G/max_freq | sudo tee $G/min_freq
```

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

- **Edge of view:** the radial exponent of the stored lens model is not settled. With hands near
  the edges, the stereo mismatch rises to about 2.6 px and bones come out up to ~45% too long. Fix in
  progress: refit on edge recordings with the 2D joints now logged in `frames.jsonl`.
- One hand is tracked. The detector has two slots, but the 3D path takes the best hand per eye.
- Fists and heavily occluded poses are the weakest, as in any keypoint model.
- No smoothing in 3D yet, and no ROS 2 node yet (planned: `/hand_tracking/raw`).

## Fallback: Ultraleap V2 under box64

If you need Ultraleap's own tracking on a Jetson, the 2015 V2 service (x64) runs under the box64
emulator at 80–90 fps once a box64 race is fixed. Newer Ultraleap services do not run there. See
[fallback_v2_box64/](fallback_v2_box64/README.md). This route uses Ultraleap's software and is
governed by its licence terms; you obtain that software yourself.

## Licence and credits

Apache-2.0 for the code in this repository (see `LICENSE` and `NOTICE`). The Mercury models are
not included; they come from Monado and carry their own licence. Notes on licensing and on what
this project does and does not use: [docs/LEGAL.md](docs/LEGAL.md) (not legal advice).
