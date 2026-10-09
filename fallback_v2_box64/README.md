# Fallback: Ultraleap V2 under box64 on a Jetson

Ultraleap's own tracking for the original controller exists only for x86/x64 (the ARM build has no
model for it). On a Jetson AGX Xavier, the 2015 **V2 2.3.1** service (`leapd`, x64) runs under the
[box64](https://github.com/ptitSeb/box64) emulator **once a box64 race is fixed**. Hand frames then
come from its WebSocket server (`ws://127.0.0.1:6437/v6.json`) at 80–90 fps with a hand in view, read
by native ARM code. This repository ships no Ultraleap software; obtain V2 2.3.1 yourself, and note
that its licence terms apply.

## The box64 fix

`box64-cleanup-lock.patch` (against box64 v0.4.4, `src/tools/cleanup.c`):
- **The bug:** box64's `__cxa_atexit` implementation (`AddCleanup1Arg`, `AddCleanup`) grows a shared
  array with `realloc` and takes no lock.
- **The trigger:** two emulated threads that initialise function-local statics at the same time
  corrupt the heap (`free(): invalid next size`). V2's `leapd` hits this within seconds.
- **The fix:** a mutex around both writers.

Build box64 for the Xavier as its docs describe (`-DTEGRA_T194=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo`).
Apply the patch first with `patch -p1 < box64-cleanup-lock.patch`.

## Running it

- Unpack `Leap-2.3.1+31549-x64.deb` (do not install it):
  `dpkg-deb -x Leap-2.3.1+31549-x64.deb ~/leap-v2`.
- **Before the first run, move V2's firmware images** (`usr/share/Leap/*.ldat` except `trparams*`) out
  of that tree. V2 then has nothing to flash, and controllers on newer firmware keep it. With 1.7.1 it
  reports "Firmware is up to date".
- `v2_run.sh` starts `leapd` under box64 as your user, with a memory watchdog (`HOLD=1` keeps it
  running). `ws_frames.py` counts frames and hands from the WebSocket.

## Not working

- **Hyperion 6.2.0** (x64) under box64 v0.4.4: as soon as any TCP client connects, `leapd` allocates
  without bound (about 200 MB/s) in a user-space loop. It loops the same way with the box64
  recompiler off, so the fault is not in the recompiler. Root cause open.
- **Gemini 5.17.1** for Raspberry Pi (arm64) runs natively but has no model for the original
  controller ("No compatible models").
- box64's native `libatomic` wrapper lacks the 16-byte functions (`__atomic_compare_exchange_16`
  and others). Use `BOX64_EMULATED_LIBS=libatomic.so.1` with an x86-64 `libatomic.so.1` when an
  emulated program needs them.
