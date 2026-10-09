"""Minimal WebSocket client for the Leap V2 service: counts frames and hands for N seconds."""

import base64
import json
import os
import socket
import struct
import sys
import time


def recv_exact(s, n):
    b = b""
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise ConnectionError("closed")
        b += c
    return b


def read_msg(s):
    h = recv_exact(s, 2)
    ln = h[1] & 0x7F
    if ln == 126:
        ln = struct.unpack(">H", recv_exact(s, 2))[0]
    elif ln == 127:
        ln = struct.unpack(">Q", recv_exact(s, 8))[0]
    return h[0] & 0x0F, recv_exact(s, ln)


def send_text(s, txt):
    data = txt.encode()
    mask = os.urandom(4)
    hdr = bytes([0x81, 0x80 | len(data)]) + mask
    s.sendall(hdr + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))


dur = float(sys.argv[1]) if len(sys.argv) > 1 else 10
s = socket.create_connection(("127.0.0.1", 6437), timeout=5)
key = base64.b64encode(os.urandom(16)).decode()
s.sendall(
    (
        "GET /v6.json HTTP/1.1\r\nHost: 127.0.0.1:6437\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
    ).encode()
)
resp = b""
while b"\r\n\r\n" not in resp:
    resp += s.recv(1)
print(resp.split(b"\r\n")[0].decode())
send_text(s, json.dumps({"background": True}))
send_text(s, json.dumps({"focused": True}))
t0 = time.time()
frames = 0
with_hands = 0
first = None
last = None
while time.time() - t0 < dur:
    op, payload = read_msg(s)
    if op != 1:
        continue
    m = json.loads(payload)
    if "version" in m and frames == 0 and first is None:
        print("server:", m)
        first = m
        continue
    if "hands" in m:
        frames += 1
        last = m
        if m["hands"]:
            with_hands += 1
el = time.time() - t0
print(f"frames {frames} in {el:.1f}s = {frames / el:.1f} fps; frames with hands {with_hands}")
if last:
    print(
        f"last frame id {last.get('id')}, currentFrameRate {last.get('currentFrameRate')}, hands {len(last['hands'])}"
    )
    if last["hands"]:
        h = last["hands"][0]
        print(
            "hand:",
            h.get("type"),
            "palm(mm)",
            h.get("palmPosition"),
            "confidence",
            h.get("confidence"),
        )
