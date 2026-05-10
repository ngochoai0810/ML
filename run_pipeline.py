import os
import subprocess
import sys


def _select_python() -> str:
    # Prefer a project venv interpreter on Windows.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, ".venv311", "Scripts", "python.exe"),
        os.path.join(base_dir, ".venv", "Scripts", "python.exe"),
        os.path.join(base_dir, "venv", "Scripts", "python.exe"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # Fall back to the interpreter used to start this launcher.
    return sys.executable


def _run(cmd: list[str]) -> int:
    print("\n$", " ".join(cmd))
    try:
        completed = subprocess.run(cmd)
        return int(completed.returncode or 0)
    except KeyboardInterrupt:
        return 130


def main() -> int:
    py = _select_python()
    print(f"Using Python: {py}")

    while True:
        print("\n=== Driver Drowsiness Detector — Launcher ===")
        print("1) Collect dataset (webcam crop eye)")
        print("2) Run realtime (integrate_cnn.py)")
        print("Q) Quit")

        choice = input("Chọn: ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            return 0

        if choice == "1":
            label = input("Label (open/closed/yawn) [open]: ").strip().lower() or "open"
            if label not in {"open", "closed", "yawn"}:
                print("Label không hợp lệ.")
                continue

            camera_raw = input("Camera index [0]: ").strip()
            camera = int(camera_raw) if camera_raw else 0

            max_raw = input("Max images (0 = no limit) [0]: ").strip()
            max_images = int(max_raw) if max_raw else 0

            return_code = _run(
                [
                    py,
                    "collect_data.py",
                    "--label",
                    label,
                    "--camera",
                    str(camera),
                    "--max",
                    str(max_images),
                ]
            )
            print(f"Exit code: {return_code}")

        elif choice == "2":
            return_code = _run([py, "integrate_cnn.py"])
            print(f"Exit code: {return_code}")

        else:
            print("Lựa chọn không hợp lệ.")


if __name__ == "__main__":
    raise SystemExit(main())
