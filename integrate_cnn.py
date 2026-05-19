"""
integrate_cnn.py  —  FIXED VERSION
====================================
Các fix so với phiên bản cũ:
  [FIX-1] HEAD POSE: Pitch 174° bug → normalize về [-90,90], chỉ báo GAT DAU khi pitch thực sự cúi
  [FIX-2] NHAM MAT: EMA_ALPHA 0.3→0.6, EAR_CONSEC_FRAMES 40→20, CNN_CLOSED_THRESHOLD 0.55→0.50
  [FIX-3] HIEU NANG: Batch CNN 2 mắt 1 lần predict, resize frame 480p trước pipeline
  [FIX-4] NGAP CHAM: MAR_CONSEC_FRAMES 12→8
  [FIX-5] XOA frame.copy() thừa trong draw_dashboard

Yêu cầu file (cùng thư mục project):
  - best_model.h5
  - class_indices.json
  - shape_predictor_68_face_landmarks.dat
  - audio/alert.wav

Phím tắt:
  - Q để thoát
  - D để bật/tắt debug head pose (in Pitch/Yaw/Roll)
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
CLOSED_IDX = 0  # default theo class_indices.json: {"0":"closed","1":"open","2":"yawn"}

if not os.path.exists(MODEL_PATH):
    print(f"⚠️  Không tìm thấy '{MODEL_PATH}'. Dùng EAR fallback.")
elif not os.path.exists(CLASS_JSON):
    print(f"⚠️  Không tìm thấy '{CLASS_JSON}'. Dùng EAR fallback.")
elif tf is None:
    print("⚠️  Chưa cài tensorflow. Dùng EAR fallback.")
else:
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_JSON, encoding="utf-8") as f:
        idx_to_class = json.load(f)

    # idx_to_class = {"0": "closed", "1": "open", "2": "yawn"}
    closed_idx_list = [k for k, v in idx_to_class.items() if v == "closed"]
    CLOSED_IDX = int(closed_idx_list[0]) if closed_idx_list else 0
    USE_CNN = True
    print(f"✅ Đã load CNN model. Classes: {idx_to_class}  CLOSED_IDX={CLOSED_IDX}")

# ─────────────────────────────────────────────────────────────
# 2) NGƯỠNG & HẰNG SỐ
# ─────────────────────────────────────────────────────────────
# EAR (fallback khi không có CNN)
EAR_THRESHOLD = 0.28
EAR_CONSEC_FRAMES = 20          # [FIX-2] cũ: 40 → quá cao, mắt nhắm lâu mới báo

# MAR (ngáp)
MAR_THRESHOLD = 0.55
MAR_CONSEC_FRAMES = 8           # [FIX-4] cũ: 12 → chậm

# Head pose — chỉ báo khi THỰC SỰ cúi đầu (pitch âm = cúi)
# [FIX-1] Dùng pitch đã normalize, không dùng abs()
PITCH_NOD_THRESHOLD = -6        # fallback khi chưa calibrate; do pitch normalize đang có biên độ nhỏ
POSE_CONSEC_FRAMES = 20

# [NEW] Calibrate pitch baseline (raw) để tránh phụ thuộc góc camera
POSE_CALIBRATION_SECONDS = 1.5  # thời gian lấy mẫu khi ngồi thẳng
PITCH_DELTA_THRESHOLD = -12.0   # độ; delta(pitch_norm - baseline) < -12 => cúi đáng kể

# CNN
CNN_CLOSED_THRESHOLD = 0.50     # [FIX-2] cũ: 0.55 → khó vượt ngưỡng khi EMA bị smooth
EMA_ALPHA = 0.6                 # [FIX-2] cũ: 0.3 → quá smooth, phản ứng chậm

# Face-miss grace period (~0.5s ở 25fps)
MAX_FACE_MISS = 12

# Calibration EAR cá nhân (chỉ dùng khi fallback EAR)
CALIBRATION_FRAMES = 75

# [FIX-3] Resize frame xuống 480p để giảm tải dlib + solvePnP
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480

# ─────────────────────────────────────────────────────────────
# 3) KHỞI TẠO
# ─────────────────────────────────────────────────────────────
pygame.mixer.init()

# Âm báo ngắn (beep) thay vì phát lặp vô hạn
ALERT_BEEP_MS = 700          # độ dài beep (ms)
ALERT_COOLDOWN_SEC = 1.6     # tối thiểu bao lâu mới beep lại
_alert_last_play_ts = 0.0

_alert_sound = pygame.mixer.Sound("audio/alert.wav")
_alert_channel = pygame.mixer.Channel(0)

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

ear_samples = []
calibrated = False
EAR_PERSONAL_THRESHOLD = EAR_THRESHOLD

_last_valid_ear = 0.35
_last_valid_mar = 0.0
_last_valid_pitch = 0.0

_ema_cnn_conf = 0.0

# Debug flag — bật bằng phím D
DEBUG_POSE = False

# [NEW] Pose calibration state (dùng pitch đã normalize để ổn định, khớp overlay)
_pose_calib_start = None
_pose_pitch_samples = []
_pose_calibrated = False
_pose_baseline_pitch = 0.0

MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
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


def calibrate_ear(ear_val: float):
    global calibrated, EAR_PERSONAL_THRESHOLD, ear_samples
    if calibrated or ear_val <= 0:
        return
    ear_samples.append(float(ear_val))
    if len(ear_samples) >= CALIBRATION_FRAMES:
        baseline = float(np.mean(ear_samples))
        EAR_PERSONAL_THRESHOLD = baseline * 0.75
        calibrated = True
        print(f"✅ Calibrated EAR: {EAR_PERSONAL_THRESHOLD:.3f} (baseline={baseline:.3f})")


def predict_eye_cnn_batch(crops_gray: list) -> float:
    """
    [FIX-3] Predict CẢ HAI mắt trong một lần gọi predict() thay vì 2 lần riêng.
    Trả về confidence trung bình rằng mắt đang NHẮM.
    """
    if not crops_gray:
        return 0.0

    batch = []
    for crop in crops_gray:
        if crop is None or crop.size == 0:
            continue
        eye = cv2.resize(crop, (64, 64)).astype("float32") / 255.0
        batch.append(eye.reshape(64, 64, 1))

    if not batch:
        return 0.0

    batch_arr = np.array(batch)                        # shape: (N, 64, 64, 1)
    probs = cnn_model.predict(batch_arr, verbose=0)    # shape: (N, num_classes)
    closed_probs = probs[:, CLOSED_IDX]
    return float(np.mean(closed_probs))


def normalize_pitch(pitch_raw: float) -> float:
    """
    [FIX-1] cv2.RQDecomp3x3 trả về pitch ~174° khi đầu thẳng thay vì 0°.
    Map về khoảng [-90, 90]:
      - pitch_raw gần 180 (hoặc -180) = đầu thẳng → map về ~0
      - pitch_raw nhỏ dần (170, 160...) = cúi đầu → map về âm
      - pitch_raw lớn dần (190...) = ngẩng đầu → map về dương
    """
    # Đưa về [-180, 180]
    p = pitch_raw % 360
    if p > 180:
        p -= 360
    # Lúc này: đầu thẳng ≈ 180° hoặc -180°
    # Flip: cúi → âm, ngẩng → dương
    if p > 90:
        p = 180 - p     # 174 → 6  (ngẩng nhẹ)
    elif p < -90:
        p = -180 - p    # -174 → -6
    # p bây giờ: 0=thẳng, âm=cúi, dương=ngẩng
    return p


def _angle_diff_deg(a: float, b: float) -> float:
    """Return signed difference a-b in degrees, wrapped to [-180, 180]."""

    diff = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return diff


def calibrate_pose_baseline(pitch_norm: float, now_ts: float):
    """Calibrate normalized pitch baseline over a short window while face is detected."""

    global _pose_calib_start, _pose_pitch_samples, _pose_calibrated, _pose_baseline_pitch

    if _pose_calibrated:
        return

    if _pose_calib_start is None:
        _pose_calib_start = float(now_ts)

    _pose_pitch_samples.append(float(pitch_norm))

    if (now_ts - _pose_calib_start) >= POSE_CALIBRATION_SECONDS and len(_pose_pitch_samples) >= 8:
        _pose_baseline_pitch = float(np.median(_pose_pitch_samples))
        _pose_calibrated = True
        print(f"✅ Pose baseline calibrated (pitch): {_pose_baseline_pitch:.1f}°")


def get_head_pose(shape, frame_shape):
    """Trả về (pitch_normalized, yaw, roll) theo độ."""
    h, w = frame_shape[:2]
    focal = float(w)
    cam_matrix = np.array(
        [[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64
    )
    img_points = np.array(
        [shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]],
        dtype=np.float64,
    )
    success, rvec, _ = cv2.solvePnP(
        MODEL_POINTS, img_points, cam_matrix, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("solvePnP failed")

    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch_raw = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    pitch = normalize_pitch(pitch_raw)   # [FIX-1]
    return pitch, yaw, roll, pitch_raw   # trả thêm raw để debug


def play_alarm():
    global _alarm_on, _alert_last_play_ts

    _alarm_on = True
    now_ts = time.time()
    if (now_ts - _alert_last_play_ts) < ALERT_COOLDOWN_SEC:
        return

    _alert_last_play_ts = now_ts
    try:
        # maxtime tính bằng ms; không loop
        _alert_channel.play(_alert_sound, loops=0, maxtime=ALERT_BEEP_MS)
    except Exception:
        # Nếu audio device lỗi thì bỏ qua, tránh crash realtime
        pass


def stop_alarm():
    global _alarm_on
    if _alarm_on:
        _alarm_on = False
        try:
            _alert_channel.stop()
        except Exception:
            pass


def put_text_outline(
    frame,
    text: str,
    org,
    font,
    font_scale: float,
    color,
    thickness: int = 1,
    outline_color=(0, 0, 0),
    outline_thickness: int = 3,
):
    """Draw readable text without any background box (outline + foreground)."""

    cv2.putText(
        frame,
        text,
        org,
        font,
        font_scale,
        outline_color,
        outline_thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        org,
        font,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_dashboard(frame, ear, mar, pitch, ear_alert, mar_alert, pose_alert):
    """Vẽ chữ trạng thái (không nền), có outline để dễ đọc."""

    h, _w = frame.shape[:2]
    y = h - 110

    put_text_outline(
        frame,
        "=== TRANG THAI ===",
        (10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        thickness=1,
    )

    eye_label = "[CNN]" if USE_CNN else "[EAR]"
    put_text_outline(
        frame,
        f"{eye_label} MAT: {'NHAM' if ear_alert else 'MO  '}",
        (10, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255) if ear_alert else (60, 220, 60),
        thickness=2,
    )
    put_text_outline(
        frame,
        f"MIENG: {'NGAP' if mar_alert else 'THUONG'} (MAR={mar:.2f})",
        (10, y + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255) if mar_alert else (60, 220, 60),
        thickness=2,
    )
    put_text_outline(
        frame,
        f"DAU: {'GAT' if pose_alert else 'BINH THUONG'} (Pitch={pitch:.1f})",
        (10, y + 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 165, 255) if pose_alert else (60, 220, 60),
        thickness=2,
    )

    danger = int(ear_alert) * 50 + int(mar_alert) * 30 + int(pose_alert) * 20
    put_text_outline(
        frame,
        f"BUON NGU: {danger}%",
        (10, y + 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255) if danger >= 70 else (0, 165, 255) if danger >= 40 else (60, 220, 60),
        thickness=2,
    )


# ─────────────────────────────────────────────────────────────
# 5) VÒNG LẶP CHÍNH
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ Không mở được camera!")
    raise SystemExit(1)

# [FIX-3] Ép resolution về 640×480 để giảm tải xử lý
cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESS_HEIGHT)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📷 Camera resolution: {actual_w}×{actual_h}")

prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️  Mất tín hiệu camera")
            break

        frame = cv2.flip(frame, 1)

        # [FIX-3] Nếu camera vẫn trả về > 480p, resize xuống để dlib chạy nhanh hơn
        fh, fw = frame.shape[:2]
        if fw > PROCESS_WIDTH:
            scale = PROCESS_WIDTH / fw
            frame = cv2.resize(frame, (PROCESS_WIDTH, int(fh * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detector(gray, 0)

        ear = _last_valid_ear
        mar = _last_valid_mar
        pitch = _last_valid_pitch

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
            # Giữ trạng thái hiển thị khi mất face tạm thời
            ear_alert = EAR_COUNTER >= EAR_CONSEC_FRAMES
            mar_alert = MAR_COUNTER >= MAR_CONSEC_FRAMES
            pose_alert = POSE_COUNTER >= POSE_CONSEC_FRAMES
        else:
            FACE_MISS_FRAMES = 0
            for face in faces:
                shape = face_utils.shape_to_np(_predictor(gray, face))

                # ── Mắt ─────────────────────────────────────────
                left_eye = shape[lStart:lEnd]
                right_eye = shape[rStart:rEnd]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                _last_valid_ear = ear

                if USE_CNN:
                    # [FIX-3] Crop cả 2 mắt rồi predict 1 lần batch
                    crops = []
                    for eye_pts in (left_eye, right_eye):
                        (ex, ey, ew, eh) = cv2.boundingRect(eye_pts)
                        pad = 6
                        crop = gray[
                            max(0, ey - pad): min(gray.shape[0], ey + eh + pad),
                            max(0, ex - pad): min(gray.shape[1], ex + ew + pad),
                        ]
                        crops.append(crop if crop.size > 0 else None)

                    avg_conf = predict_eye_cnn_batch(crops)
                    # [FIX-2] EMA_ALPHA cao hơn → phản ứng nhanh hơn
                    _ema_cnn_conf = EMA_ALPHA * avg_conf + (1 - EMA_ALPHA) * _ema_cnn_conf
                    eye_trigger = _ema_cnn_conf > CNN_CLOSED_THRESHOLD

                    put_text_outline(
                        frame,
                        f"CNN: {_ema_cnn_conf:.2f} (raw:{avg_conf:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 200, 0),
                        thickness=2,
                    )
                else:
                    calibrate_ear(ear)
                    ear_threshold = EAR_PERSONAL_THRESHOLD if calibrated else EAR_THRESHOLD
                    eye_trigger = ear < ear_threshold

                if eye_trigger:
                    EAR_COUNTER += 1
                    if EAR_COUNTER >= EAR_CONSEC_FRAMES:
                        ear_alert = True
                        put_text_outline(
                            frame,
                            "!!! BUON NGU !!!",
                            (frame.shape[1] // 2 - 140, 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 0, 255),
                            thickness=3,
                            outline_thickness=5,
                        )
                        play_alarm()
                else:
                    EAR_COUNTER = max(0, EAR_COUNTER - 1)   # giảm dần thay vì reset ngay → ổn định hơn
                    if EAR_COUNTER == 0:
                        stop_alarm()

                # ── Miệng (ngáp) ─────────────────────────────────
                mouth = shape[mStart:mEnd]
                mar = mouth_aspect_ratio(mouth)
                _last_valid_mar = mar

                mouth_hull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [mouth_hull], -1, (0, 100, 255), 1)

                if mar > MAR_THRESHOLD:
                    MAR_COUNTER += 1
                    if MAR_COUNTER >= MAR_CONSEC_FRAMES:
                        mar_alert = True
                        put_text_outline(
                            frame,
                            "NGAP!",
                            (frame.shape[1] - 140, 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            thickness=2,
                            outline_thickness=4,
                        )
                else:
                    MAR_COUNTER = max(0, MAR_COUNTER - 1)

                # ── Head pose (gật đầu) ──────────────────────────
                try:
                    pitch, _yaw, _roll, pitch_raw = get_head_pose(shape, frame.shape)
                    _last_valid_pitch = pitch
                except Exception:
                    pitch = _last_valid_pitch
                    pitch_raw = pitch
                    _yaw = 0.0
                    _roll = 0.0

                # [NEW] Calibrate baseline (pitch) và dùng delta để detect cúi đầu ổn định hơn
                now_ts = time.time()
                calibrate_pose_baseline(pitch, now_ts)
                if _pose_calibrated:
                    pitch_delta = float(pitch) - float(_pose_baseline_pitch)
                    pose_trigger = pitch_delta < PITCH_DELTA_THRESHOLD
                else:
                    pitch_delta = 0.0
                    pose_trigger = pitch < PITCH_NOD_THRESHOLD

                # CHỈ báo khi thực sự cúi (delta âm đủ lớn). KHÔNG dùng abs().
                if pose_trigger:
                    POSE_COUNTER += 1
                    if POSE_COUNTER >= POSE_CONSEC_FRAMES:
                        pose_alert = True
                        put_text_outline(
                            frame,
                            "GAT DAU!",
                            (frame.shape[1] // 2 - 90, 115),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 165, 255),
                            thickness=2,
                            outline_thickness=4,
                        )
                else:
                    POSE_COUNTER = max(0, POSE_COUNTER - 1)

                if DEBUG_POSE:
                    put_text_outline(
                        frame,
                        (
                            f"P:{pitch:.1f} raw:{pitch_raw:.1f} "
                            + (
                                f"base:{_pose_baseline_pitch:.1f} d:{pitch_delta:.1f} "
                                if _pose_calibrated
                                else "calib... "
                            )
                            + f"Y:{_yaw:.1f} R:{_roll:.1f}"
                        ),
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        thickness=2,
                    )

                # Draw eye contours
                cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

                break  # chỉ xử lý face đầu tiên

        # ── Overlay thông tin ─────────────────────────────────────
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time

        put_text_outline(
            frame,
            f"FPS: {fps:.1f}",
            (frame.shape[1] - 130, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            thickness=2,
        )
        put_text_outline(
            frame,
            f"EAR: {ear:.3f}  CNT: {EAR_COUNTER}/{EAR_CONSEC_FRAMES}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            thickness=2,
        )

        # Calibration progress (chỉ hiện khi dùng EAR fallback)
        if not USE_CNN and not calibrated:
            pct = len(ear_samples) / CALIBRATION_FRAMES
            put_text_outline(
                frame,
                f"Calibrating... {int(pct*100)}%",
                (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                thickness=2,
            )

        draw_dashboard(frame, ear, mar, pitch, ear_alert, mar_alert, pose_alert)

        cv2.imshow("Driver Drowsiness Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            DEBUG_POSE = not DEBUG_POSE
            print(f"Debug pose: {'ON' if DEBUG_POSE else 'OFF'}")

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    try:
        _alert_channel.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print("👋 Đã thoát.")