#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY="./.venv311/Scripts/python.exe"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Cannot find venv python at: $PY" >&2
  exit 1
fi

CAMERA="${1:-1}"
BACKEND="${BACKEND:-msmf}"
DEVICE_NAME="${DEVICE_NAME:-}"

echo "=== Realtime compare ==="
echo "Camera: $CAMERA"
echo

echo "[1/2] Running v1 (best_model.h5)"
echo "Close the webcam window (press Q) to continue."
if [[ -n "$DEVICE_NAME" ]]; then
  echo "Using DirectShow device name: $DEVICE_NAME"
  "$PY" integrate_cnn.py --device-name "$DEVICE_NAME" --backend dshow --model best_model.h5 --class-json class_indices.json
else
  "$PY" integrate_cnn.py --camera "$CAMERA" --backend "$BACKEND" --model best_model.h5 --class-json class_indices.json
fi

echo
read -r -p "Press Enter to run v2..." _

echo "[2/2] Running v2 (best_model_v2.h5)"
echo "Close the webcam window (press Q) to finish."
if [[ -n "$DEVICE_NAME" ]]; then
  echo "Using DirectShow device name: $DEVICE_NAME"
  "$PY" integrate_cnn.py --device-name "$DEVICE_NAME" --backend dshow --model best_model_v2.h5 --class-json class_indices_v2.json
else
  "$PY" integrate_cnn.py --camera "$CAMERA" --backend "$BACKEND" --model best_model_v2.h5 --class-json class_indices_v2.json
fi
