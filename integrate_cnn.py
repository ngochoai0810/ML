"""
integrate_cnn.py
================
Phiên bản chạy realtime từ webcam đã tích hợp:
  - CNN phân loại mắt (thay thế EAR threshold cứng), có EAR fallback nếu thiếu model
  - MAR phát hiện ngáp
  - Head Pose phát hiện gật đầu
  - Dashboard trạng thái tổng hợp

Yêu cầu file (cùng thư mục project):
  - best_model.h5         (train từ train_cnn.py)
  - class_indices.json    (tự động tạo sau khi train)
  - shape_predictor_68_face_landmarks.dat
  - audio/alert.wav

Phím tắt:
  - Q để thoát
"""

import json
import os
import time

import cv2
import dlib
import numpy as np
import pygame
from imutils import face_utils
from scipy.spatial import distance

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

# ─────────────────────────────────────────────────────────────
# 1) LOAD MODEL CNN (nếu có)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "best_model.h5"
CLASS_JSON = "class_indices.json"

USE_CNN = False
cnn_model = None
idx_to_class = None
CLOSED_IDX = 1

if not os.path.exists(MODEL_PATH):
    print(f"⚠️  Không tìm thấy '{MODEL_PATH}'.")
    print("   Dùng EAR fallback.")
else:
    if not os.path.exists(CLASS_JSON):
        print(f"⚠️  Không tìm thấy '{CLASS_JSON}'.")
        print("   Dùng EAR fallback.")
    else:
        if tf is None:
            print("⚠️  Chưa cài tensorflow. Dùng EAR fallback.")
        else:
            cnn_model = tf.keras.models.load_model(MODEL_PATH)
            with open(CLASS_JSON, encoding="utf-8") as f:
                idx_to_class = json.load(f)

            # idx_to_class = {"0": "closed", "1": "open", ...}
            closed_idx = [k for k, v in idx_to_class.items() if v == "closed"]
            CLOSED_IDX = int(closed_idx[0]) if closed_idx else 1
            USE_CNN = True
            print(f"✅ Đã load CNN model. Classes: {idx_to_class}")

# ─────────────────────────────────────────────────────────────
# 2) NGƯỠNG & HẰNG SỐ
# ─────────────────────────────────────────────────────────────
# EAR (fallback)
EAR_THRESHOLD = 0.28
EAR_CONSEC_FRAMES = 40

# MAR (ngáp)
MAR_THRESHOLD = 0.55
MAR_CONSEC_FRAMES = 12

# Head pose (gật đầu)
PITCH_THRESHOLD = -15  # độ
POSE_CONSEC_FRAMES = 18

# CNN confidence
CNN_CLOSED_THRESHOLD = 0.55

# Face-miss grace period
MAX_FACE_MISS = 12  # ~0.5s ở 25fps

# Smoothing CNN confidence
EMA_ALPHA = 0.3

# Debug head pose để calibrate PITCH_THRESHOLD
DEBUG_POSE = False

# ─────────────────────────────────────────────────────────────
# 3) KHỞI TẠO
# ─────────────────────────────────────────────────────────────
pygame.mixer.init()
pygame.mixer.music.load("audio/alert.wav")

_detector = dlib.get_frontal_face_detector()
_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (48, 68)

EAR_COUNTER = 0
MAR_COUNTER = 0
POSE_COUNTER = 0
_alarm_on = False

FACE_MISS_FRAMES = 0

CALIBRATION_FRAMES = 75  # ~3s ở 25fps
ear_samples = []
calibrated = False
EAR_PERSONAL_THRESHOLD = EAR_THRESHOLD

_last_valid_ear = 0.0
_last_valid_mar = 0.0
_last_valid_pitch = 0.0
_last_valid_yaw = 0.0
_last_valid_roll = 0.0

_ema_cnn_conf = 0.0

MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # nose 30
        (0.0, -330.0, -65.0),  # chin 8
        (-225.0, 170.0, -135.0),  # left eye outer 36
        (225.0, 170.0, -135.0),  # right eye outer 45
        (-150.0, -150.0, -125.0),  # mouth left 48
        (150.0, -150.0, -125.0),  # mouth right 54
    ],
    dtype=np.float64,
)


# ─────────────────────────────────────────────────────────────
# 4) HÀM TIỆN ÍCH
# ─────────────────────────────────────────────────────────────

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)


def rotate_landmarks_2d(pts, angle_deg, center=None):
    """Xoay 2D landmarks ngược chiều roll để normalize về thẳng."""

    if pts is None or len(pts) == 0:
        return pts

    pts = np.asarray(pts, dtype=np.float64)
    if center is None:
        center = pts.mean(axis=0)
    else:
        center = np.asarray(center, dtype=np.float64)

    angle_rad = np.radians(-float(angle_deg))
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    return ((pts - center) @ rot.T) + center


def calibrate_ear(ear_val: float):
    global calibrated, EAR_PERSONAL_THRESHOLD, ear_samples

    if calibrated:
        return

    if ear_val <= 0:
        return

    ear_samples.append(float(ear_val))
    if len(ear_samples) >= CALIBRATION_FRAMES:
        baseline = float(np.mean(ear_samples))
        EAR_PERSONAL_THRESHOLD = baseline * 0.75
        calibrated = True
        print(
            f"✅ Calibrated EAR threshold: {EAR_PERSONAL_THRESHOLD:.3f} (baseline={baseline:.3f})"
        )


def predict_eye_cnn(eye_region_gray):
    """Trả về (is_closed: bool, confidence: float)."""

    if eye_region_gray is None or eye_region_gray.size == 0:
        return False, 0.0

    eye = cv2.resize(eye_region_gray, (64, 64))
    eye = eye.astype("float32") / 255.0
    eye = eye.reshape(1, 64, 64, 1)

    probs = cnn_model.predict(eye, verbose=0)[0]
    closed_prob = float(probs[CLOSED_IDX])
    return closed_prob > CNN_CLOSED_THRESHOLD, closed_prob


def get_head_pose(shape, frame_shape):
    """Trả về (pitch, yaw, roll) theo độ."""

    h, w = frame_shape[:2]
    focal = float(w)

    cam_matrix = np.array(
        [[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64
    )

    img_points = np.array(
        [shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]],
        dtype=np.float64,
    )

    success, rvec, _tvec = cv2.solvePnP(
        MODEL_POINTS,
        img_points,
        cam_matrix,
        np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        raise RuntimeError("solvePnP failed")

    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return float(angles[0]), float(angles[1]), float(angles[2])


def play_alarm():
    global _alarm_on
    if not _alarm_on:
        pygame.mixer.music.play(-1)
        _alarm_on = True


def stop_alarm():
    global _alarm_on
    if _alarm_on:
        pygame.mixer.music.stop()
        _alarm_on = False


def draw_dashboard(frame, ear, mar, pitch, ear_alert, mar_alert, pose_alert):
    """Vẽ dashboard góc dưới trái."""

    h, _w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, h - 130), (320, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = h - 110

    cv2.putText(
        frame,
        "=== TRANG THAI ===",
        (10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    cv2.putText(
        frame,
        f"{'[CNN]' if USE_CNN else '[EAR]'} MAT: {'NHAM' if ear_alert else 'MO  '}",
        (10, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255) if ear_alert else (60, 220, 60),
        1,
    )

    cv2.putText(
        frame,
        f"MIENG: {'NGAP' if mar_alert else 'THUONG'} (MAR={mar:.2f})",
        (10, y + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255) if mar_alert else (60, 220, 60),
        1,
    )

    cv2.putText(
        frame,
        f"DAU: {'GAT' if pose_alert else 'BINH THUONG'} (Pitch={pitch:.1f}°)",
        (10, y + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 165, 255) if pose_alert else (60, 220, 60),
        1,
    )

    danger = int(ear_alert) * 50 + int(mar_alert) * 30 + int(pose_alert) * 20

    bar_color = (
        (60, 220, 60)
        if danger < 40
        else (0, 165, 255)
        if danger < 70
        else (0, 0, 255)
    )

    cv2.rectangle(frame, (10, y + 90), (10 + danger * 3, y + 105), bar_color, -1)
    cv2.putText(
        frame,
        f"BUON NGU: {danger}%",
        (10, y + 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        bar_color,
        1,
    )


# ─────────────────────────────────────────────────────────────
# 5) VÒNG LẶP CHÍNH
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Không mở được camera!")
    raise SystemExit(1)

prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️  Mất tín hiệu camera")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detector(gray, 0)

        ear = _last_valid_ear
        mar = _last_valid_mar
        pitch = _last_valid_pitch
        _yaw = _last_valid_yaw
        _roll = _last_valid_roll

        ear_alert = False
        mar_alert = False
        pose_alert = False

        if len(faces) == 0:
            FACE_MISS_FRAMES += 1
            if FACE_MISS_FRAMES > MAX_FACE_MISS:
                EAR_COUNTER = 0
                MAR_COUNTER = 0
                POSE_COUNTER = 0
                stop_alarm()

            # Giữ nguyên trạng thái hiển thị khi mất face tạm thời
            ear_alert = EAR_COUNTER >= EAR_CONSEC_FRAMES
            mar_alert = MAR_COUNTER >= MAR_CONSEC_FRAMES
            pose_alert = POSE_COUNTER >= POSE_CONSEC_FRAMES
        else:
            FACE_MISS_FRAMES = 0
            for face in faces:
                shape = face_utils.shape_to_np(_predictor(gray, face))

                # ── Eyes ────────────────────────────────────────
                left_eye = shape[lStart:lEnd]
                right_eye = shape[rStart:rEnd]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                calibrate_ear(ear)
                ear_threshold = EAR_PERSONAL_THRESHOLD if calibrated else EAR_THRESHOLD

                if USE_CNN:
                    results = []
                    for eye_pts in (left_eye, right_eye):
                        (ex, ey, ew, eh) = cv2.boundingRect(eye_pts)
                        pad = 6
                        crop = gray[
                            max(0, ey - pad) : min(gray.shape[0], ey + eh + pad),
                            max(0, ex - pad) : min(gray.shape[1], ex + ew + pad),
                        ]
                        if crop.size > 0:
                            _is_closed, conf = predict_eye_cnn(crop)
                            results.append(conf)

                    avg_conf = float(np.mean(results)) if results else 0.0
                    _ema_cnn_conf = EMA_ALPHA * avg_conf + (1 - EMA_ALPHA) * _ema_cnn_conf
                    eye_trigger = _ema_cnn_conf > CNN_CLOSED_THRESHOLD

                    cv2.putText(
                        frame,
                        f"CNN: {_ema_cnn_conf:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 200, 0),
                        1,
                    )
                else:
                    eye_trigger = ear < ear_threshold

                if eye_trigger:
                    EAR_COUNTER += 1
                    if EAR_COUNTER >= EAR_CONSEC_FRAMES:
                        ear_alert = True
                        cv2.putText(
                            frame,
                            "!!! BUON NGU !!!",
                            (frame.shape[1] // 2 - 120, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 0, 255),
                            3,
                        )
                        play_alarm()
                else:
                    EAR_COUNTER = 0
                    stop_alarm()

                # ── Mouth (yawn) ────────────────────────────────
                mouth = shape[mStart:mEnd]

                # ── Head pose (nod) ────────────────────────────
                try:
                    pitch, _yaw, _roll = get_head_pose(shape, frame.shape)
                    _last_valid_pitch = pitch
                    _last_valid_yaw = _yaw
                    _last_valid_roll = _roll
                except Exception as e:
                    pitch = _last_valid_pitch
                    _yaw = _last_valid_yaw
                    _roll = _last_valid_roll
                    # Uncomment để debug:
                    # print(f"[HeadPose ERR] {e}")

                face_center = shape.mean(axis=0)
                mouth_pts_norm = rotate_landmarks_2d(
                    mouth.astype(np.float64), _roll, center=face_center
                )
                mar = mouth_aspect_ratio(mouth_pts_norm)
                mouth_hull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [mouth_hull], -1, (0, 100, 255), 1)

                if mar > MAR_THRESHOLD:
                    MAR_COUNTER += 1
                    if MAR_COUNTER >= MAR_CONSEC_FRAMES:
                        mar_alert = True
                        cv2.putText(
                            frame,
                            "NGAP!",
                            (frame.shape[1] - 140, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2,
                        )
                else:
                    MAR_COUNTER = 0

                # Trigger khi pitch vượt ngưỡng (dùng |pitch| để tránh phụ thuộc dấu)
                if abs(pitch) > abs(PITCH_THRESHOLD):
                    POSE_COUNTER += 1
                    if POSE_COUNTER >= POSE_CONSEC_FRAMES:
                        pose_alert = True
                        cv2.putText(
                            frame,
                            "GAT DAU!",
                            (frame.shape[1] // 2 - 80, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 165, 255),
                            2,
                        )
                else:
                    POSE_COUNTER = 0

                if DEBUG_POSE:
                    cv2.putText(
                        frame,
                        f"P:{pitch:.1f} Y:{_yaw:.1f} R:{_roll:.1f}",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )

                # Draw eye contours
                left_hull = cv2.convexHull(left_eye)
                right_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_hull], -1, (0, 255, 0), 1)

                # only process first detected face
                _last_valid_ear = ear
                _last_valid_mar = mar
                break

        # FPS
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (frame.shape[1] - 110, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (150, 150, 150),
            1,
        )

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        draw_dashboard(frame, ear, mar, pitch, ear_alert, mar_alert, pose_alert)

        cv2.imshow("Driver Drowsiness Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except KeyboardInterrupt:
    # Exit gracefully on Ctrl+C
    pass

finally:
    cap.release()
    pygame.mixer.music.stop()
    cv2.destroyAllWindows()
    print("👋 Đã thoát.")
