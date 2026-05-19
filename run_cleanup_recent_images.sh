#!/usr/bin/env bash
set -euo pipefail

# Run cleanup_recent_images.py using the project's virtual environment (Git Bash / Windows).

cd "$(dirname "$0")"

PY=""
for candidate in \
  "./.venv311/Scripts/python.exe" \
  "./.venv/Scripts/python.exe" \
  "./venv/Scripts/python.exe" \
  ; do
  if [[ -f "$candidate" ]]; then
    PY="$candidate"
    break
  fi
done

if [[ -z "$PY" ]]; then
  echo "[ERR] Không tìm thấy Python trong venv (.venv311/.venv/venv)." >&2
  echo "      Hãy chạy: ./.venv311/Scripts/python.exe cleanup_recent_images.py ..." >&2
  exit 1
fi

exec "$PY" cleanup_recent_images.py "$@"
