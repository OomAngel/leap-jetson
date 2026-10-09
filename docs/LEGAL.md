# What this project uses, and licensing notes

**Not legal advice.** This records what the code does and does not touch, so that anyone can judge
their own use.

## What the tracker uses

| Part | Source | Status |
|---|---|---|
| Stereo images | the controller's standard USB Video Class interface, read by Linux's `uvcvideo` | no vendor software |
| Calibration | two standard UVC control requests (set sharpness, read saturation), the behaviour described by Leap Motion's LeapUVC | independent implementation; no LeapUVC code |
| Hand models | Monado's Mercury ONNX models | downloaded by the user, not redistributed; stated CC-BY-SA 4.0, see below |
| Model conventions | Monado `hg_model.cpp` (BSL-1.0) as documented by OpenLeap (BSL-1.0) | credited in `NOTICE` |
| Everything else | this repository | Apache-2.0 |

The tracker uses **no** Ultraleap / Leap Motion software, libraries, services, model weights or
code: no `leapd`, `libtrack_server`, LeapC, SDK headers or `.ldat` files.

## Points to be aware of

- **Mercury model licence.** OpenLeap states CC-BY-SA 4.0. The model repository
  (`monado/utilities/hand-tracking-models`, commit `37a8a81`) carries no licence file. Confirmation
  from the Monado maintainers has been requested. TensorRT engines built from the models stay on
  the device that built them and are never distributed.
- **LeapUVC.** Its examples are published under the Leap Motion developer SDK agreement. This
  project uses only the documented device behaviour (which control holds which byte), not that
  code.
- **Fallback V2 under box64.** That route runs Ultraleap's own 2015 V2 service and is governed by
  Ultraleap's licence terms, including restrictions on controlling industrial or commercial
  equipment. Obtain that software yourself; this repository ships none of it, only a patch to box64.
- **Firmware** is never written.

## Related open work

[OpenLeap](https://github.com/julianjc84/OpenLeap) (BSL-1.0, x86) is an independent open driver and
tracker for the same controller. Its `LICENSING.md` sets out a similar posture in detail.
