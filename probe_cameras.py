import argparse
import os
import time

import cv2


def _open(index: int, backend: str) -> cv2.VideoCapture:
    if backend == "dshow":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if backend == "msmf":
        return cv2.VideoCapture(index, cv2.CAP_MSMF)
    return cv2.VideoCapture(index)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe available camera indexes.")
    parser.add_argument("--backend", choices=["any", "dshow", "msmf"], default="dshow")
    parser.add_argument("--max", type=int, default=10, help="Probe indexes [0..max-1]")
    parser.add_argument("--frames", type=int, default=3, help="Frames to try reading per camera")
    args = parser.parse_args()

    print(f"Probing cameras: backend={args.backend} max={args.max}")
    if os.name != "nt" and args.backend != "any":
        print("Note: backend forcing is mainly for Windows.")

    found_any = False

    for idx in range(args.max):
        cap = _open(idx, args.backend)
        try:
            opened = cap.isOpened()
            ok = False
            w = h = 0
            if opened:
                # Let camera warm up a bit
                time.sleep(0.05)
                for _ in range(max(1, args.frames)):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        ok = True
                        h, w = frame.shape[:2]
                        break
            status = "OK" if (opened and ok) else ("OPEN" if opened else "NO")
            if opened and ok:
                found_any = True
            print(f"index {idx}: {status}  {w}x{h}")
        finally:
            cap.release()

    if not found_any:
        print("No working camera indexes found. Try a different backend: --backend msmf or --backend any")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
