#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY="./.venv311/Scripts/python.exe"
if [[ ! -f "$PY" ]]; then
  echo "ERROR: Cannot find venv python at: $PY" >&2
  echo "Hint: create venv at .venv311 or edit this script." >&2
  exit 1
fi

CAMERA="${1:-1}"

echo "Running realtime v1 on camera $CAMERA..."
"$PY" integrate_cnn.py --camera "$CAMERA" --model best_model.h5 --class-json class_indices.json
