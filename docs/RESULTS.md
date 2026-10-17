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
the centre barely sees the lens distortion. Hand data could not decide it, not even from edge
recordings (see "Distance, disagreeing eyes and the stereo gate" below). It was settled with a
plumb-line test on 16 October: **0.1** (see "Plumb-line test" below; 1.0 was used until then).

## Live runs (`tracker/live_track.py`, 20 s each, hand moving)

| Run | GPU clock | Processed fps (hand in view) | ms/frame | Camera → 3D latency, median | Frames with 3D hand | Stereo mismatch | Index bones (mm) | Depth |
|---|---|---|---|---|---|---|---|---|
| 1 | governor (~217 MHz) | 27 | 35.1 | 41.0 ms | 530/539 | 1.58 px | 38 / 26 / 24 | 224 mm |
| 2 | pinned 1377 MHz | 67.5 | 14.8 | 19.1 ms | 940/1350 | 1.40 px | 42 / 27 / 26 | 313 mm |
| 3 | pinned, vectorised heatmap decoding | **98.2** | 9.1 | **14.2 ms** | 1512/1965 | 2.56 px | 55 / 33 / 37 | 291 mm |

The camera delivers 115 fps; with no hand in view the tracker keeps up (112–114 fps, 3.8–8.6 ms per
frame; 12.0 ms camera-to-result before the decoder change). Latency is `camera_to_result_ms_median` from
`live_track.py`: the time from the capture thread receiving a frame to its 3D joints being ready (USB
transfer and sensor exposure come on top). Run 3 had the hand further away and nearer the edges. Its larger mismatch and bone
lengths come from distance and from frames where the two eyes disagree (16 October analysis below; first
attributed to the lens model, which the edge recordings did not support), not from a speed effect: the vectorised decoder returns exactly the same
joints as the loop it replaced on 464 heatmap sets from the recording.

## Replay and backends (`live_track.py --replay`, `--backend`; 15 October 2026)

The 15 s hand recording used above (1,700 frames), replayed after the replay mode was added:

| Machine | Backend | Mode | Processed | fps | Frames with 3D hand | Stereo mismatch | Index proximal / wrist to middle knuckle |
|---|---|---|---|---|---|---|---|
| AGX Xavier, GPU pinned | TensorRT FP16 | every frame | 1,700 / 1,700 | 111 | 1,699 | 1.5 px | 37.8 / 90.5 mm |
| AGX Xavier, CPU | ONNX Runtime | paced at 115 fps | 168 (rest dropped, as live) | 11.3 | 164 | 1.3 px | — |
| Desktop PC (Ryzen 9 5900X), CPU | ONNX Runtime 1.23 | every frame, 40-frame excerpt, approximate calibration | 40 / 40 | 43 | 40 | 0.9 px | 37.2 / 91.3 mm |

Live on the Xavier after the change (no hand in view): 115 fps, 3.8 ms per frame, 4.0 ms camera to
result. The replay numbers match the live centre-of-view results above, so the change (TensorRT loaded
only for `--backend trt`, any model with `run()` accepted) did not alter the tracking.

## Distance, disagreeing eyes and the stereo gate (16 October 2026)

Two 20 s recordings with the hand moved along the edges of the view (left/right, then front/back) at
about 15 and 35 cm, compared with the 15 s centre recording. Analysis offline from the per-frame 2D
joints in `frames.jsonl`.

- **The eyes disagreed, the lens did not.** In the left/right edge recording the median stereo mismatch
  was 14–45 px at every image radius, centre included (centre recording: 1.4 px). Even with the
  detector finding the hand in both eyes, 343 of 508 frames disagreed by more than 10 px (centre
  recording: 1 of 1,099). Looking at those frames: the hand was far (30–45 cm), turned side-on, against
  the arm and a bright lamp, and the keypoints fell on the forearm in one eye and on the fingers in
  the other. The front/back recording had no frame with a hand in both eyes.
- **No reading of the lens model fixes it.** On the frames where the eyes agree, exponents 0–3 and an
  inverted warp change the median mismatch by under 0.1 px, and none brings the bone lengths back
  (wrist to middle knuckle 114–141 mm against 89–93 mm in the centre recording).
- **Distance does.** Focal length 134 px (at 640×480) and baseline 40 mm: one pixel of disparity is
  11.7 mm of depth at 25 cm, 16.8 mm at 30 cm, 22.9 mm at 35 cm, 29.9 mm at 40 cm (z² / (baseline × focal)). Depth noise makes a bone longer on
  average. Centre recording, frames where the eyes agree, wrist to middle knuckle (exponent 1.0;
  with 0.1: 87.6, 91.3 and 100.8 mm): 87.8 mm at
  200–250 mm, 91.2 mm at 250–300 mm, 102.2 mm at 300–350 mm.

Stereo gate (`stereo.stereo_consistent`, 3 px), replayed on the Xavier with every frame:

| Recording | Frames with 3D | Dropped by the gate | Mismatch | Depth | Index proximal | Wrist to middle knuckle |
|---|---|---|---|---|---|---|
| centre, 1,700 frames | 1,405 | 294 | 1.29 px | 252 mm | 37.6 mm | 90.2 mm |
| edges left/right, 2,300 frames | 262 | 221 | 1.10 px | 346 mm | 43.8 mm | 113.2 mm |
| edges front/back, 2,300 frames | 0 | 8 | — | — | — | — |

Offline, on the centre recording, the gate narrows wrist-to-middle-knuckle from 76.9–120.8 mm to
78.6–113.8 mm (10th–90th percentile); on the edge recording it removes triangulations of up to
kilometres (90th percentile 208 mm with the gate). Thresholds of 2–5 px give similar medians; 3 px keeps
82% of the centre frames.

The table above used exponent 1.0. With the exponent from the plumb-line test (0.1), the same replays
give: centre 1,403 frames with 3D, 1.29 px, depth 249 mm, index proximal 37.5 mm, wrist to middle
knuckle 89.4 mm; edges left/right 260 frames, 1.14 px, depth 326 mm, 41.3 mm, 109.0 mm.

## Coupled eyes (`tracker/stereo_tracker.py`, 16 October 2026)

The disagreeing frames above came from each eye tracking on its own: a crop follows that eye's own
previous joints, so an eye that drifts onto the forearm stays there. The stereo tracker couples them:
when the eyes disagree (or one has lost the hand), the eye with the stronger hand (detected before
coasting, then higher keypoint confidence) places its joints at the last good depth (then 200, 300,
400 mm; placements within 0.15 crop widths of one already tried are skipped), projects them into the
other eye (`stereo.project`, the inverse of `rays`), and runs the keypoint model in a crop there, with
the same handedness and **no joint prior**. A repair counts only if the eyes then agree (3 px gate); the
repaired eye continues from it.

Replays on the Xavier (TensorRT, every frame, exponent 0.1):

| Recording | Eyes | Frames with 3D | Dropped (> 3 px) | Repaired | Mismatch | Depth | Index proximal | Wrist to middle knuckle | Replay fps | ms/frame median / p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| centre, 1,700 frames | independent | 1,403 | 296 | — | 1.29 px | 249 mm | 37.5 mm | 89.4 mm | 91 | — |
| centre | coupled | **1,657** | 43 | 52 | 1.28 px | 248 mm | 37.3 mm | 90.1 mm | 89 | 9.0 / 21.0 |
| edges left/right, 2,300 frames | independent | 260 | 223 | — | 1.14 px | 326 mm | 41.3 mm | 109.0 mm | 99 | — |
| edges left/right | coupled | **1,114** | 56 | 334 | 1.05 px | 359 mm | 43.6 mm | 102.6 mm | 54 | 16.6 / 35.5 |

- **The agreement is not an echo of the projection.** A first version gave the repaired eye the
  projected joints as prior; without that prior the gain stays almost whole (centre 1,696 → 1,661,
  edges 1,266 → 1,161 frames with 3D), so it comes from placing the crop, not from the model repeating
  the prior. The prior is not used.
- **Visual check:** six repaired frames spread over the edge recording, joints drawn in both eyes: on
  the hand in both, same pose.
- **Cost:** a repair is one keypoint call per placement tried. Hard recordings replay at 54 fps; live,
  the tracker takes the newest camera frame, so it would skip frames rather than lag. Not yet measured
  live.

## Following from 3D (17 October 2026)

Live, the coupled tracker ran at 55 fps (camera 115 fps; 98 fps before coupling) with 28% of frames
needing a repair, 87% of processed frames with 3D and 19 ms camera-to-3D latency: the coupling only
stepped in after the eyes had drifted apart. Now, after a frame where the eyes agree, the next frame
skips the detector: both crops are placed by projecting the last 3D hand into each eye (crop centre the
median of the projected joints, size blended from the last crop and the joints' extent), each eye keeps
its own previous joints as prior, and the detector path runs when following fails and at least every
8 frames.

Replays on the Xavier, every frame. The GPU clock was **not** pinned (the board had rebooted: 115 MHz),
so only the comparison between modes at the same clock is meaningful, not the absolute speed:

| Recording | Mode | Frames with 3D | Dropped | Repaired | Followed | ms/frame median | Mismatch |
|---|---|---|---|---|---|---|---|
| hand-only (new), 2,300 | detector every frame | 2,006 | 52 | 526 | — | 21.9 | 1.18 px |
| hand-only | following | 2,016 | 40 | 66 | 1,783 | **15.6** | 0.95 px |
| centre, 1,700 | detector every frame | 1,657 | 43 | 52 | — | 19.0 | 1.28 px |
| centre | following | 1,621 | 79 | 20 | 1,416 | **15.5** | 0.99 px |
| edges left/right, 2,300 | detector every frame | 1,114 | 56 | 334 | — | 23.7 | 1.05 px |
| edges left/right | following | **1,449** | 22 | 67 | 1,259 | **12.7** | 0.86 px |

**Following is not a shrunken or echoed hand.** On the same frames the two modes' 2D joints differ by
1.3–2.1 px (median) and following gives 3–7% shorter wrist-to-middle-knuckle distances. Bones are rigid,
so the length should not change from frame to frame: the robust spread (MAD/median, mean of five bones)
on 1,986 hand-only frames is 13.8% with the detector on every frame and **7.5%** with following (centre
recording 15.8% vs 11.5%). The detector path is the noisier one, and noise lengthens bones on average.
Two variants were tried and not kept: the detector's crop size while following (8.3% spread, more
frames) and the crop centred on the joints' bounding box (11.0%).

## Plumb-line test of the lens model (`tracker/plumb_line.py`, 16 October 2026)

A sheet of paper taped to a hardcover book, held 12–20 cm above the controller and moved through the
view for 20 s (2,300 frames; every 10th used). The sheet's edges are straight, so after a correct
undistortion they must be straight lines. Edge points were found to sub-pixel precision along the
normals of each long side (middle 70%, one clean step per profile, so fingers and corners drop out);
edges not straight to 0.6 px without any correction (bent paper, a finger) were left out. The edges
reached a normalised radius of 0.74 (median) to 1.1–1.3 (95th percentile), i.e. well into the outer
part of the 135° view.

| Eye | Clean edges | Best exponent (bootstrap 5–95%) | RMS from straight: none | 0.1 | 0.5 | 1.0 (old) |
|---|---|---|---|---|---|---|
| left | 311 of 413 | 0.05 (0.05–0.15) | 0.213 px | 0.208 px | 0.240 px | 0.327 px |
| right | 285 of 408 | 0.10 (0.10–0.25) | 0.208 px | 0.197 px | 0.229 px | 0.319 px |

The controller's images are already nearly rectilinear: the stored radial function shifts points by
up to 11 px at full strength, which bends straight edges by half again as much as no correction.
Exponent 0.1 is used from now on (`live_track.py`, `stereo.LensModel`). Rotation and tangential terms
map lines to lines, so this test does not check them; the rotation sign stays as fitted above. The
synthetic check in `tests/` confirms the scoring recovers a known exponent.

## Speed

- **GPU clock:** Jetson's devfreq governor held the GPU at 216.75 MHz during tracking. One
  detector call took 3.5 ms. With the minimum frequency raised to 1377 MHz it took 1.04 ms.
- **Spin-wait** CUDA syncs (`cudaDeviceScheduleSpin`) save about 0.4 ms per call.
- **Heatmap decoding:** a per-joint Python loop took half the frame time. Vectorised, it takes about
  0.75 ms per pair of calls.
