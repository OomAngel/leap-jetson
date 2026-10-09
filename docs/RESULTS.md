# Results and how they were measured

All on one Jetson AGX Xavier (JetPack 5.1, L4T 35.6.5, MAXN, TensorRT 8.5.2, Python 3.8) with
one original Leap Motion Controller (firmware 1.7.1) lying face up, between 9 and 10 October 2026.

## Calibration read

`tracker/leap_usb_calib.py`, two passes, identical bytes:
- signature `CA`, version 2, score 80;
- baseline 40.000 mm;
- focal length 133.8 / 133.9 px per eye;
- Cayley rotations mirror each other (±0.00176, ±0.00081, ±0.00057).

Reading the same addresses through V4L2 (`v4l2-ctl -c sharpness=… -C saturation`) returned 76 for
every byte: `uvcvideo` serves the cached saturation value, hence the libusb reader.

## Model precision (`tracker/check_precision.py`)

Recording: 1,700 frames, 15 s of a moving hand. Every 20th frame, both eyes: 170 eye images, 135
with a hand. The keypoint model got the reference crop in both handedness orientations (270
heatmap sets, 5,670 joints).

| Against ONNX Runtime FP32 (CPU) | TensorRT FP32 | TensorRT FP16 |
|---|---|---|
| detector outputs, max abs difference | 0.0 | 0.021 |
| hand/no-hand decisions changed (threshold 0.40) | 0 | 1 of 340 |
| detected hand centre shift | 0.0 px | mean 0.35, max 1.33 px |
| heatmaps, max abs difference | 0.0 | 0.014 |
| joint position shift | 0.0 px | mean 0.036, p95 0.093, max 1.59 px |
| joint confidence difference | 0.0 | mean 0.0012, max 0.013 |

## Detector geometry (`tracker/probe_detector.py`)

86 eye images with a hand; best `hand_exists` per variant:

| Variant | Median | Detected (≥ 0.40) |
|---|---|---|
| stretched to 640×480, rotated 180°, mirrored | 0.795 | 77/86 |
| stretched to 640×480, rotated 180° (used) | 0.737 | 70/86 |
| stretched, rotated 270°, mirrored | 0.723 | 71/86 |
| raw 640×240 | 0.006 | 0/86 |

The mirrored variant scores slightly higher, but mirroring swaps handedness and the stereo geometry
for no gain in keypoints, so the pipeline uses the unmirrored one.

## Lens model reading (`tracker/fit_lens_model.py`)

555 stereo frames with a confident hand in both eyes, near the centre of the view:

| Reading | Median stereo mismatch |
|---|---|
| Cayley rotation sign −1 (used) | 1.51 px |
| Cayley rotation sign +1 | 1.95 px |

The radial exponent (0 to 3) changed the mismatch by less than 0.02 px here, because a hand near
the centre barely sees the lens distortion. It is undetermined (1.0 used) until edge data are fitted.

## Live runs (`tracker/live_track.py`, 20 s each, hand moving)

| Run | GPU clock | Processed fps (hand in view) | ms/frame | Camera → 3D latency, median | Frames with 3D hand | Stereo mismatch | Index bones (mm) | Depth |
|---|---|---|---|---|---|---|---|---|
| 1 | governor (~217 MHz) | 27 | 35.1 | 41.0 ms | 530/539 | 1.58 px | 38 / 26 / 24 | 224 mm |
| 2 | pinned 1377 MHz | 67.5 | 14.8 | 19.1 ms | 940/1350 | 1.40 px | 42 / 27 / 26 | 313 mm |
| 3 | pinned, vectorised heatmap decoding | **98.2** | 9.1 | **14.2 ms** | 1512/1965 | 2.56 px | 55 / 33 / 37 | 291 mm |

The camera delivers 115 fps; with no hand in view the tracker keeps up (112–114 fps, 3.8–8.6 ms per
frame; 12.0 ms camera-to-result before the decoder change). Latency is `camera_to_result_ms_median` from
`live_track.py`: the time from the capture thread receiving a frame to its 3D joints being ready (USB
transfer and sensor exposure come on top). Run 3 moved further towards the edges of the view. Its larger mismatch and bone lengths are
the open lens-model question, not a speed effect: the vectorised decoder returns exactly the same
joints as the loop it replaced on 464 heatmap sets from the recording.

## Speed

- **GPU clock:** Jetson's devfreq governor held the GPU at 216.75 MHz during tracking. One
  detector call took 3.5 ms. With the minimum frequency raised to 1377 MHz it took 1.04 ms.
- **Spin-wait** CUDA syncs (`cudaDeviceScheduleSpin`) save about 0.4 ms per call.
- **Heatmap decoding:** a per-joint Python loop took half the frame time. Vectorised, it takes about
  0.75 ms per pair of calls.
