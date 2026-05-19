import argparse
import os
import time

import cv2
import dlib
from imutils import face_utils


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Tạo thư mục
os.makedirs("dataset/open", exist_ok=True)
os.makedirs("dataset/closed", exist_ok=True)
os.makedirs("dataset/yawn", exist_ok=True)

parser = argparse.ArgumentParser(description="Collect eye dataset crops from webcam.")
parser.add_argument(
    "--label",
    default="open",
    choices=["open", "closed", "yawn"],
    help="Class label to save into (open/closed/yawn).",
)
parser.add_argument(
    "--camera",
    type=int,
    default=1,
    help="Camera index for cv2.VideoCapture (default: 1).",
)
parser.add_argument(
    "--out-dir",
    default="dataset",
    help="Output dataset directory (default: dataset).",
)
parser.add_argument(
    "--max",
    type=int,
    default=0,
    help="Stop after saving N images (0 = no limit).",
)

parser.add_argument(
    "--auto",
    action="store_true",
    help="Auto-save crops without pressing S (uses --interval).",
)
parser.add_argument(
    "--interval",
    type=float,
    default=0.2,
    help="Minimum seconds between auto-saves (default: 0.2).",
)
parser.add_argument(
    "--both-eyes",
    action="store_true",
    help="Save both left and right eye crops (label still applies).",
)
parser.add_argument(
    "--min-sharpness",
    type=float,
    default=20.0,
    help="Skip blurry crops using Laplacian variance threshold (default: 20.0).",
)
parser.add_argument(
    "--dedupe-threshold",
    type=float,
    default=2.0,
    help=(
        "Skip near-duplicate crops using mean-abs-diff threshold (0 disables). "
        "Default: 2.0"
    ),
)

parser.add_argument(
    "--head-angle-hint",
    default="",
    help=(
        "Gợi ý tư thế/góc đầu khi thu thập (ví dụ: 'nghieng trai 20-30', "
        "'nghieng phai', 'cui xuong'). Chỉ để nhắc, không ảnh hưởng logic."
    ),
)

args = parser.parse_args()

# Ensure output directories exist
os.makedirs(os.path.join(args.out_dir, "open"), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, "closed"), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, "yawn"), exist_ok=True)

cap = cv2.VideoCapture(args.camera)
label = args.label
out_label_dir = os.path.join(args.out_dir, label)
os.makedirs(out_label_dir, exist_ok=True)

# Start counting from existing files to avoid overwriting.
existing = [
    f
    for f in os.listdir(out_label_dir)
    if os.path.isfile(os.path.join(out_label_dir, f))
]
count = len(existing)

if args.auto:
    print(
        f"Thu thập class: '{label}' — AUTO mode (interval={args.interval}s), nhấn Q để thoát"
    )
else:
    print(f"Thu thập class: '{label}' — nhấn S để lưu, Q để thoát")

if args.head_angle_hint:
    print(f"🛈 Gợi ý góc đầu: {args.head_angle_hint}")

last_eye_img = None
_last_saved_img = None
_last_save_time = 0.0


def _sharpness_laplacian(img_gray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def _mean_abs_diff(a, b) -> float:
    if a is None or b is None:
        return 999.0
    if a.shape != b.shape:
        return 999.0
    return float(abs(a.astype("float32") - b.astype("float32")).mean())


def _save_crop(crop_gray, suffix="") -> bool:
    global count, _last_saved_img, _last_save_time

    if crop_gray is None or crop_gray.size == 0:
        return False

    sharp = _sharpness_laplacian(crop_gray)
    if args.min_sharpness and sharp < args.min_sharpness:
        return False

    if args.dedupe_threshold and args.dedupe_threshold > 0:
        if _mean_abs_diff(crop_gray, _last_saved_img) < args.dedupe_threshold:
            return False

    out_path = os.path.join(out_label_dir, f"{count:04d}{suffix}.jpg")
    ok = cv2.imwrite(out_path, crop_gray)
    if ok:
        count += 1
        _last_saved_img = crop_gray.copy()
        _last_save_time = time.time()
        print(f"Đã lưu {count} ảnh -> {out_path} (sharp={sharp:.1f})")
        if args.max and count >= args.max:
            print(f"Đã đủ {args.max} ảnh, dừng.")
            return True
    return False

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        eye_img = None
        eye_img_r = None
        mouth_img = None

        for face in faces:
            shape = face_utils.shape_to_np(predictor(gray, face))

            pad = 5

            if label == "yawn":
                # Crop miệng (48-67)
                (mx, my, mw, mh) = cv2.boundingRect(shape[48:68])
                x1 = max(mx - pad, 0)
                y1 = max(my - pad, 0)
                x2 = min(mx + mw + pad, gray.shape[1])
                y2 = min(my + mh + pad, gray.shape[0])
                mouth_img = gray[y1:y2, x1:x2]
                if mouth_img.size > 0:
                    mouth_img = cv2.resize(mouth_img, (64, 64))
                    cv2.imshow("Preview", mouth_img)
                    last_eye_img = mouth_img
            else:
                # Crop mắt trái (36-41)
                (lx, ly, lw, lh) = cv2.boundingRect(shape[36:42])
                x1 = max(lx - pad, 0)
                y1 = max(ly - pad, 0)
                x2 = min(lx + lw + pad, gray.shape[1])
                y2 = min(ly + lh + pad, gray.shape[0])
                eye_img = gray[y1:y2, x1:x2]
                if eye_img.size > 0:
                    eye_img = cv2.resize(eye_img, (64, 64))
                    cv2.imshow("Preview", eye_img)
                    last_eye_img = eye_img

                if args.both_eyes:
                    # Crop mắt phải (42-47)
                    (rx, ry, rw, rh) = cv2.boundingRect(shape[42:48])
                    x1 = max(rx - pad, 0)
                    y1 = max(ry - pad, 0)
                    x2 = min(rx + rw + pad, gray.shape[1])
                    y2 = min(ry + rh + pad, gray.shape[0])
                    eye_img_r = gray[y1:y2, x1:x2]
                    if eye_img_r.size > 0:
                        eye_img_r = cv2.resize(eye_img_r, (64, 64))

            # Chỉ lấy 1 face đầu tiên để preview/saving ổn định
            break

        if args.head_angle_hint:
            cv2.putText(
                frame,
                f"Goi y: {args.head_angle_hint}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Video", frame)

        # Auto-save
        if args.auto and last_eye_img is not None:
            now = time.time()
            if now - _last_save_time >= args.interval:
                stop_now = False
                if label == "yawn":
                    stop_now = _save_crop(last_eye_img)
                else:
                    # Save left eye crop
                    stop_now = _save_crop(last_eye_img, suffix="_L" if args.both_eyes else "")
                    # Save right eye crop if available
                    if not stop_now and args.both_eyes and eye_img_r is not None and eye_img_r.size > 0:
                        stop_now = _save_crop(eye_img_r, suffix="_R")
                if stop_now:
                    break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            if last_eye_img is None or last_eye_img.size == 0:
                print("Chưa có ảnh mắt để lưu (không detect được mắt).")
                continue
            stop_now = False
            if label == "yawn":
                stop_now = _save_crop(last_eye_img)
            else:
                stop_now = _save_crop(last_eye_img, suffix="_L" if args.both_eyes else "")
                if not stop_now and args.both_eyes and eye_img_r is not None and eye_img_r.size > 0:
                    stop_now = _save_crop(eye_img_r, suffix="_R")
            if stop_now:
                break
        elif key == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
