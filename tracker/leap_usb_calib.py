#!/usr/bin/env python3
"""Read the original Leap Motion Controller's on-device stereo calibration (156 bytes).

The controller (USB f182:0003, firmware >= 1.7.1) exposes its calibration through
"property knocking" (LeapUVC, github.com/leapmotion/leapuvc): set the UVC sharpness
control to an address 100..255, then read the saturation control, which returns the byte
at that address. Linux's uvcvideo caches control values and never re-reads saturation, so
going through V4L2 returns the same cached byte 156 times. This tool sends the two UVC
class requests directly with libusb instead: it detaches uvcvideo from the VideoControl
interface for about a second, reads the bytes, and reattaches the driver (the video node
comes back). Needs write access to the USB device node (the `plugdev` udev rule) and no
other program streaming from the controller.

Usage:
  leap_usb_calib.py [-o calibration.bin] [--reads 2]
"""

from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

VID, PID = 0xF182, 0x0003
VC_INTERFACE = 0  # VideoControl interface
PU_UNIT = 5  # Processing Unit id from the descriptors (lsusb -v)
PU_SATURATION = 0x07  # UVC 1.1 PU control selectors
PU_SHARPNESS = 0x08
SET_CUR, GET_CUR = 0x01, 0x81
TO_INTERFACE_OUT, TO_INTERFACE_IN = 0x21, 0xA1  # class request, recipient interface
CALIB_FIRST, CALIB_LAST = 100, 255  # 156 addresses
TIMEOUT_MS = 1000


def _libusb() -> ctypes.CDLL:
    lib = ctypes.CDLL("libusb-1.0.so.0")
    lib.libusb_open_device_with_vid_pid.restype = ctypes.c_void_p
    lib.libusb_open_device_with_vid_pid.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16]
    for name in (
        "libusb_kernel_driver_active",
        "libusb_detach_kernel_driver",
        "libusb_attach_kernel_driver",
        "libusb_claim_interface",
        "libusb_release_interface",
    ):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.libusb_control_transfer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_uint16,
        ctypes.c_uint16,
        ctypes.c_void_p,
        ctypes.c_uint16,
        ctypes.c_uint,
    ]
    lib.libusb_close.argtypes = [ctypes.c_void_p]
    lib.libusb_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.libusb_exit.argtypes = [ctypes.c_void_p]
    return lib


def _control(lib: ctypes.CDLL, handle: int, request_type: int, request: int, selector: int, buf: ctypes.Array) -> int:
    n = lib.libusb_control_transfer(
        handle, request_type, request, selector << 8, (PU_UNIT << 8) | VC_INTERFACE, buf, len(buf), TIMEOUT_MS
    )
    if n != len(buf):
        msg = f"control transfer request {request:#x} selector {selector:#x} returned {n}"
        raise OSError(msg)
    return n


def read_calibration(reads: int = 2) -> bytes:
    """Return the 156 calibration bytes; read `reads` times and require identical results."""
    lib = _libusb()
    ctx = ctypes.c_void_p()
    if lib.libusb_init(ctypes.byref(ctx)) != 0:
        raise OSError("libusb_init failed")
    handle = lib.libusb_open_device_with_vid_pid(ctx, VID, PID)
    if not handle:
        lib.libusb_exit(ctx)
        raise OSError("Leap Motion Controller f182:0003 not found or not accessible (plugdev rule?)")
    detached = False
    claimed = False
    try:
        if lib.libusb_kernel_driver_active(handle, VC_INTERFACE) == 1:
            if lib.libusb_detach_kernel_driver(handle, VC_INTERFACE) != 0:
                raise OSError("could not detach uvcvideo from the VideoControl interface")
            detached = True
        if lib.libusb_claim_interface(handle, VC_INTERFACE) != 0:
            raise OSError("could not claim the VideoControl interface (another program streaming?)")
        claimed = True
        results = []
        for _ in range(reads):
            data = bytearray()
            value = (ctypes.c_uint8 * 2)()
            for address in range(CALIB_FIRST, CALIB_LAST + 1):
                value[0], value[1] = address & 0xFF, 0
                _control(lib, handle, TO_INTERFACE_OUT, SET_CUR, PU_SHARPNESS, value)
                time.sleep(0.004)  # LeapUVC waits 4 ms between the write and the read
                out = (ctypes.c_uint8 * 2)()
                _control(lib, handle, TO_INTERFACE_IN, GET_CUR, PU_SATURATION, out)
                data.append(out[0])
            results.append(bytes(data))
        if any(r != results[0] for r in results[1:]):
            raise OSError("calibration reads differ between passes")
        return results[0]
    finally:
        if claimed:
            lib.libusb_release_interface(handle, VC_INTERFACE)
        if detached:
            lib.libusb_attach_kernel_driver(handle, VC_INTERFACE)
        lib.libusb_close(handle)
        lib.libusb_exit(ctx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="calibration.bin")
    ap.add_argument("--reads", type=int, default=2, help="read passes that must agree")
    args = ap.parse_args()
    raw = read_calibration(args.reads)
    Path(args.out).write_bytes(raw)
    distinct = len(set(raw))
    print(f"wrote {len(raw)} bytes to {args.out} ({distinct} distinct values; first bytes {list(raw[:6])})")


if __name__ == "__main__":
    main()
