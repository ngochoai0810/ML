import argparse
import os

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
    default=0,
    help="Camera index for cv2.VideoCapture (default: 0).",
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

args = parser.parse_args()

# Ensure output directories exist
os.makedirs(os.path.join(args.out_dir, "open"), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, "closed"), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, "yawn"), exist_ok=True)

cap = cv2.VideoCapture(args.camera)
label = args.label
count = 0

print(f"Thu thập class: '{label}' — nhấn S để lưu, Q để thoát")

last_eye_img = None

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        eye_img = None

        for face in faces:
            shape = face_utils.shape_to_np(predictor(gray, face))

            # Crop mắt trái (36-41)
            (lx, ly, lw, lh) = cv2.boundingRect(shape[36:42])

            pad = 5
            x1 = max(lx - pad, 0)
            y1 = max(ly - pad, 0)
            x2 = min(lx + lw + pad, gray.shape[1])
            y2 = min(ly + lh + pad, gray.shape[0])

            eye_img = gray[y1:y2, x1:x2]

            if eye_img.size > 0:
                eye_img = cv2.resize(eye_img, (64, 64))
                cv2.imshow("Preview", eye_img)
                last_eye_img = eye_img

            # Chỉ lấy 1 face đầu tiên để preview/saving ổn định
            break

        cv2.imshow("Video", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            if last_eye_img is None or last_eye_img.size == 0:
                print("Chưa có ảnh mắt để lưu (không detect được mắt).")
                continue
            out_path = os.path.join(args.out_dir, label, f"{count:04d}.jpg")
            cv2.imwrite(out_path, last_eye_img)
            count += 1
            print(f"Đã lưu {count} ảnh -> {out_path}")
            if args.max and count >= args.max:
                print(f"Đã đủ {args.max} ảnh, dừng.")
                break
        elif key == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
