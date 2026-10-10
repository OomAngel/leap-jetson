# Third-party notices

leap-jetson's own code is Apache-2.0. It builds on the following work.

## Monado "Mercury" hand-tracking models (not included)

| | |
|---|---|
| Files | `grayscale_detection_160x160.onnx`, `grayscale_keypoint_jan18.onnx` |
| Source | https://gitlab.freedesktop.org/monado/utilities/hand-tracking-models, commit `37a8a81` (2023-05-20) |
| Checksums | `models.sha256` |
| Licence | **Unconfirmed.** The upstream repository has no licence file. OpenLeap describes the models as CC BY-SA 4.0. Confirmation requested from the Monado maintainers on 2026-10-10: [issue #1](https://gitlab.freedesktop.org/monado/utilities/hand-tracking-models/-/work_items/1). |
| Use here | downloaded by the user; converted to TensorRT engines on the user's own device; engines are never distributed by this project |
| Intended use | 2D hand detection and 21-keypoint estimation on grayscale (infrared) images |

## Monado Mercury conventions (Boost Software License 1.0)

Model pre- and post-processing (`tracker/mercury.py`) follows Monado's
`src/xrt/tracking/hand/mercury/hg_model.cpp`, https://gitlab.freedesktop.org/monado/monado, as
documented by OpenLeap.

## OpenLeap (Boost Software License 1.0)

https://github.com/julianjc84/OpenLeap: its notes on Mercury's conventions and on the controller's
calibration block informed this work. No OpenLeap code is included.

## Leap Motion LeapUVC (method only)

https://github.com/leapmotion/leapuvc describes the controller's UVC "property knocking"
(set sharpness to an address, read saturation). `tracker/leap_usb_calib.py` implements that read
independently over libusb. No LeapUVC code is included, and frames come from Linux's `uvcvideo`
driver, never from LeapUVC.

## box64 (MIT)

`fallback_v2_box64/box64-cleanup-lock.patch` modifies box64, https://github.com/ptitSeb/box64,
Copyright (c) 2020 ptitSeb. The patch is offered under box64's MIT licence, reproduced in
`fallback_v2_box64/LICENSE.box64`.
