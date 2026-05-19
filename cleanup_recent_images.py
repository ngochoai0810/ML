"""cleanup_recent_images.py — Xóa ảnh vừa thu thập theo thời gian.

Mục đích: lỡ thu nhầm/thu lỗi thì xóa nhanh các ảnh .jpg mới tạo trong `dataset/<label>`.

Ví dụ (xem trước, KHÔNG xóa):
    ./.venv311/Scripts/python.exe cleanup_recent_images.py --label open --minutes 10

Ví dụ (xóa thật):
    ./.venv311/Scripts/python.exe cleanup_recent_images.py --label open --minutes 10 --apply

Tuỳ chọn thư mục dataset:
    ./.venv311/Scripts/python.exe cleanup_recent_images.py --out-dir dataset --label open --minutes 30 --apply

Lưu ý: Script chỉ động tới file `.jpg` trong đúng thư mục `out-dir/label`.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    path: str
    mtime: float
    size: int


def _iter_jpg_files(folder: str) -> list[Candidate]:
    items: list[Candidate] = []
    if not os.path.isdir(folder):
        return items

    for name in os.listdir(folder):
        if not name.lower().endswith(".jpg"):
            continue
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue
        items.append(Candidate(path=path, mtime=float(st.st_mtime), size=int(st.st_size)))

    # newest first
    items.sort(key=lambda c: c.mtime, reverse=True)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete recently created .jpg images in dataset/<label>.")
    parser.add_argument("--out-dir", default="dataset", help="Dataset root folder (default: dataset).")
    parser.add_argument("--label", default="open", choices=["open", "closed", "yawn"], help="Label subfolder.")
    parser.add_argument("--minutes", type=int, default=10, help="Delete files modified within last N minutes (default: 10).")
    parser.add_argument("--apply", action="store_true", help="Actually delete files. Without this flag, runs in dry-run mode.")

    args = parser.parse_args()

    target_dir = os.path.join(args.out_dir, args.label)
    minutes = max(int(args.minutes), 0)
    cutoff = time.time() - minutes * 60

    files = _iter_jpg_files(target_dir)
    recent = [c for c in files if c.mtime >= cutoff]

    mode = "APPLY (DELETE)" if args.apply else "DRY-RUN (NO DELETE)"
    print(f"Target: {target_dir}")
    print(f"Window: last {minutes} minute(s)")
    print(f"Mode:   {mode}")

    if not os.path.isdir(target_dir):
        print("[ERR] Folder does not exist.")
        return 1

    if not recent:
        print("No matching recent .jpg files.")
        return 0

    total_bytes = sum(c.size for c in recent)
    print(f"Found {len(recent)} file(s), total ~{total_bytes / (1024 * 1024):.2f} MB")

    # Print first 20 files for visibility
    for c in recent[:20]:
        age_s = int(time.time() - c.mtime)
        print(f"- {os.path.basename(c.path)}  ({age_s}s ago)")
    if len(recent) > 20:
        print(f"... and {len(recent) - 20} more")

    if not args.apply:
        print("\nNothing deleted. Re-run with --apply to delete.")
        return 0

    deleted = 0
    failed = 0

    for c in recent:
        try:
            os.remove(c.path)
            deleted += 1
        except OSError:
            failed += 1

    print(f"\nDeleted: {deleted}")
    if failed:
        print(f"Failed:  {failed}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
