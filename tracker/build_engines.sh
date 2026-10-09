#!/usr/bin/env bash
# Build TensorRT engines (FP32 and FP16) for Monado's Mercury hand models on this machine.
# Models: gitlab.freedesktop.org/monado/utilities/hand-tracking-models (commit 37a8a81, 2023-05-20;
# the repo carries no licence file, OpenLeap states CC-BY-SA 4.0). Engines are specific to this GPU
# and TensorRT version, so build them on the device that runs them.
#   build_engines.sh <models_dir> [out_dir]
set -euo pipefail
MODELS=${1:?usage: build_engines.sh <models_dir> [out_dir]}
OUT=${2:-$MODELS}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
mkdir -p "$OUT"
for m in grayscale_detection_160x160 grayscale_keypoint_jan18; do
  for p in fp32 fp16; do
    flag=""
    [ "$p" = fp16 ] && flag="--fp16"
    echo "== $m $p"
    "$TRTEXEC" --onnx="$MODELS/$m.onnx" --saveEngine="$OUT/$m.$p.engine" $flag \
      --useSpinWait --iterations=500 --warmUp=500 > "$OUT/$m.$p.trtexec.log" 2>&1
    grep -E "GPU Compute Time: min" "$OUT/$m.$p.trtexec.log" | sed -E 's/.*(mean = [0-9.]+ ms).*/  \1/'
  done
done
