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

            camera_raw = input("Camera index [1]: ").strip()
            camera = int(camera_raw) if camera_raw else 1

            max_raw = input("Max images (0 = no limit) [0]: ").strip()
            max_images = int(max_raw) if max_raw else 0

            auto_raw = input("Auto-save? (y/n) [y]: ").strip().lower() or "y"
            auto = auto_raw in {"y", "yes", "1"}

            interval_raw = input("Auto interval seconds [0.2]: ").strip()
            interval = float(interval_raw) if interval_raw else 0.2

            both_raw = input("Save both eyes? (y/n) [y]: ").strip().lower() or "y"
            both_eyes = both_raw in {"y", "yes", "1"}

            sharp_raw = input("Min sharpness (0 disable) [20.0]: ").strip()
            min_sharpness = float(sharp_raw) if sharp_raw else 20.0

            dedupe_raw = input("Dedupe threshold (0 disable) [2.0]: ").strip()
            dedupe_threshold = float(dedupe_raw) if dedupe_raw else 2.0

            cmd = [
                py,
                "collect_data.py",
                "--label",
                label,
                "--camera",
                str(camera),
                "--max",
                str(max_images),
                "--interval",
                str(interval),
                "--min-sharpness",
                str(min_sharpness),
                "--dedupe-threshold",
                str(dedupe_threshold),
            ]

            if auto:
                cmd.append("--auto")
            if both_eyes:
                cmd.append("--both-eyes")

            return_code = _run(cmd)
            print(f"Exit code: {return_code}")

        elif choice == "2":
            return_code = _run([py, "integrate_cnn.py"])
            print(f"Exit code: {return_code}")

        else:
            print("Lựa chọn không hợp lệ.")


if __name__ == "__main__":
    raise SystemExit(main())
