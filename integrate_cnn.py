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

import argparse
import json
import os
import sys
import threading
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Driver drowsiness detector (CNN + landmarks)")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (default: 1)")
    parser.add_argument(
        "--backend",
        choices=["any", "dshow", "msmf"],
        default=("dshow" if os.name == "nt" else "any"),
        help=(
            "OpenCV VideoCapture backend. On Windows, 'dshow' often maps camera indexes more reliably "
            "when you have multiple cameras (default: any)."
        ),
    )
    parser.add_argument(
        "--device-name",
        default="",
        help=(
            "DirectShow device name to open (Windows only). Example: --device-name 'USB2.0 Camera'. "
            "When provided, '--backend' is forced to 'dshow'."
        ),
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help=(
            "List available DirectShow camera device names (requires 'pygrabber' on Windows), then exit."
        ),
    )
    parser.add_argument(
        "--process-width",
        type=int,
        default=640,
        help="Processing width (resizes frames down to this width). Lower = faster (default: 640).",
    )
    parser.add_argument(
        "--process-height",
        type=int,
        default=480,
        help="Requested capture height hint (default: 480).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1,
        help=(
            "Camera buffer size hint (OpenCV CAP_PROP_BUFFERSIZE). Lower reduces latency (default: 1)."
        ),
    )
    parser.add_argument(
        "--drop-frames",
        type=int,
        default=0,
        help=(
            "Drop N grabbed frames each loop to reduce lag when CPU is slow (default: 0). Example: 2."
        ),
    )

    parser.add_argument(
        "--threaded-capture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a background capture thread and always process the newest frame (lower latency). "
            "Recommended for USB webcams on Windows (default: true)."
        ),
    )
    parser.add_argument("--model", default="best_model.h5", help="Keras .h5 model path")
    parser.add_argument(
        "--class-json",
        default="class_indices.json",
        help="JSON mapping index->class name (default: class_indices.json)",
    )
    return parser.parse_args()


ARGS = _parse_args()

# Avoid UnicodeEncodeError on some Windows terminals (e.g., Git Bash piping).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

if ARGS.list_cameras:
    if os.name != "nt":
        print("--list-cameras is supported on Windows only.")
        raise SystemExit(0)
    try:
        # Optional dependency. If missing, we print instructions.
        from pygrabber.dshow_graph import FilterGraph  # type: ignore

        devices = FilterGraph().get_input_devices()
        if not devices:
            print("No DirectShow camera devices found.")
        else:
            print("DirectShow camera devices:")
            for i, name in enumerate(devices):
                print(f"  {i}: {name}")
    except ModuleNotFoundError:
        print("Missing optional dependency: pygrabber")
        print("Install: ./.venv311/Scripts/python.exe -m pip install pygrabber")
    raise SystemExit(0)

# ─────────────────────────────────────────────────────────────
# 1) LOAD MODEL CNN (nếu có)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = ARGS.model
CLASS_JSON = ARGS.class_json

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
# [FIX-1] Dùng pitch đã normalize, không dùng abs().
#
# NOTE: Pitch thường âm khi cúi. Khi đã calibrate baseline, dùng:
#   pitch_drop = baseline_pitch - pitch
# (dương khi pitch cúi xuống so với baseline). Ngưỡng cũng là số dương.
POSE_CONSEC_FRAMES = 20

# [NEW] Calibrate pitch baseline (normalized) để tránh phụ thuộc góc camera
POSE_CALIBRATION_SECONDS = 1.5  # thời gian lấy mẫu khi ngồi thẳng
PITCH_DELTA_THRESHOLD = 12.0    # độ; baseline - pitch > 12 => cúi đáng kể
PITCH_ABS_NOD_LIMIT = -18.0     # độ; fallback tuyệt đối: pitch <= -18 => coi là cúi mạnh

# [NEW] Person switch detection via face signature (helps when 2 users sit similarly)
FACE_SIG_SIZE = 24
FACE_SIG_SIM_THRESHOLD = 0.72
FACE_SIG_EMA = 0.12
MIN_FACE_AREA_FOR_SIG = 80 * 80

# CNN
CNN_CLOSED_THRESHOLD = 0.50     # [FIX-2] cũ: 0.55 → khó vượt ngưỡng khi EMA bị smooth
EMA_ALPHA = 0.6                 # [FIX-2] cũ: 0.3 → quá smooth, phản ứng chậm

# Face-miss grace period (~0.5s ở 25fps)
MAX_FACE_MISS = 12

# Face selection / person switch handling
DETECTOR_UPSAMPLE = 0          # tăng lên 1 nếu hay miss face (chậm hơn)
FACE_CHANGE_CENTER_FRAC = 0.35 # đổi người nếu tâm mặt dịch > 35% bề rộng mặt trước
FACE_CHANGE_AREA_RATIO = 0.55  # hoặc diện tích mặt mới <55% hoặc > (1/0.55)x mặt trước

# Calibration EAR cá nhân (chỉ dùng khi fallback EAR)
CALIBRATION_FRAMES = 75

# [FIX-3] Resize frame xuống 480p để giảm tải dlib + solvePnP
PROCESS_WIDTH = int(ARGS.process_width)
PROCESS_HEIGHT = int(ARGS.process_height)

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
_pose_delta_threshold = PITCH_DELTA_THRESHOLD

# [NEW] Track last face to reset per-person state
_last_face_box = None  # (l, t, r, b)
_last_face_sig = None

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
    global _pose_delta_threshold

    if _pose_calibrated:
        return

    if _pose_calib_start is None:
        _pose_calib_start = float(now_ts)

    _pose_pitch_samples.append(float(pitch_norm))

    if (now_ts - _pose_calib_start) >= POSE_CALIBRATION_SECONDS and len(_pose_pitch_samples) >= 8:
        samples = np.array(_pose_pitch_samples, dtype=np.float32)
        baseline = float(np.median(samples))

        # Robust noise estimate (MAD) to adjust nod threshold per person/camera.
        mad = float(np.median(np.abs(samples - baseline)))
        robust_sigma = 1.4826 * mad
        # More noise => require a stronger downward drop to trigger nod.
        adaptive = 12.0 + 2.0 * robust_sigma
        _pose_delta_threshold = float(min(25.0, max(10.0, adaptive)))

        _pose_baseline_pitch = baseline
        _pose_calibrated = True
        print(
            f"✅ Pose baseline calibrated (pitch): {_pose_baseline_pitch:.1f}°  "
            f"drop_th={_pose_delta_threshold:.1f}° (noise≈{robust_sigma:.1f})"
        )


def _compute_face_signature(gray_img: np.ndarray, box: tuple):
    """Compute a small, illumination-robust face signature vector for person switch detection."""

    l, t, r, b = box
    h, w = gray_img.shape[:2]
    bw = max(1, r - l)
    bh = max(1, b - t)
    pad = int(0.12 * max(bw, bh))

    l2 = max(0, l - pad)
    t2 = max(0, t - pad)
    r2 = min(w, r + pad)
    b2 = min(h, b + pad)
    if (r2 - l2) <= 8 or (b2 - t2) <= 8:
        return None

    roi = gray_img[t2:b2, l2:r2]
    try:
        roi = cv2.resize(roi, (FACE_SIG_SIZE, FACE_SIG_SIZE), interpolation=cv2.INTER_AREA)
        roi = cv2.equalizeHist(roi)
    except Exception:
        return None

    vec = roi.astype(np.float32).reshape(-1)
    vec -= float(vec.mean())
    std = float(vec.std())
    if std > 1e-6:
        vec /= std
    # L2 normalize for cosine similarity
    norm = float(np.linalg.norm(vec))
    if norm > 1e-6:
        vec /= norm
    return vec


def _rect_to_box(rect) -> tuple:
    return (int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom()))


def _box_area(box: tuple) -> float:
    l, t, r, b = box
    return float(max(0, r - l) * max(0, b - t))


def _box_center(box: tuple) -> tuple:
    l, t, r, b = box
    return ((l + r) / 2.0, (t + b) / 2.0)


def reset_person_state(reason: str):
    """Reset counters + calibration when switching to a different person."""

    global EAR_COUNTER, MAR_COUNTER, POSE_COUNTER
    global _ema_cnn_conf
    global _pose_calib_start, _pose_pitch_samples, _pose_calibrated, _pose_baseline_pitch
    global _pose_delta_threshold
    global _last_face_sig
    global calibrated, EAR_PERSONAL_THRESHOLD, ear_samples

    EAR_COUNTER = 0
    MAR_COUNTER = 0
    POSE_COUNTER = 0
    _ema_cnn_conf = 0.0

    _pose_calib_start = None
    _pose_pitch_samples = []
    _pose_calibrated = False
    _pose_baseline_pitch = 0.0
    _pose_delta_threshold = PITCH_DELTA_THRESHOLD
    _last_face_sig = None

    if not USE_CNN:
        calibrated = False
        EAR_PERSONAL_THRESHOLD = EAR_THRESHOLD
        ear_samples = []

    stop_alarm()
    print(f"🔄 Reset state: {reason}")


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
def _open_camera(index: int, backend: str) -> cv2.VideoCapture:
    if backend == "dshow":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        # Some OpenCV builds on Windows can't capture by index with CAP_DSHOW.
        # Fallback to MSMF for index-based capture.
        if (not cap.isOpened()) and os.name == "nt":
            cap.release()
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        return cap
    if backend == "msmf":
        return cv2.VideoCapture(index, cv2.CAP_MSMF)
    return cv2.VideoCapture(index)


def _open_camera_by_name_dshow(device_name: str) -> cv2.VideoCapture:
    # OpenCV on Windows (CAP_DSHOW) supports "video=<friendly name>".
    return cv2.VideoCapture(f"video={device_name}", cv2.CAP_DSHOW)


class _LatestFrameGrabber:
    """Continuously grabs frames so processing loop always receives the newest frame."""

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = False
        self._thread: threading.Thread | None = None

        self._frame = None
        self._seq = 0
        self._last_ok = False

    def start(self) -> "_LatestFrameGrabber":
        self._thread = threading.Thread(target=self._run, name="LatestFrameGrabber", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def read(self, timeout: float = 1.0):
        """Return (ok, frame). Waits until at least one new frame arrives or timeout."""
        end = time.time() + float(timeout)
        with self._cond:
            start_seq = self._seq
            while (not self._stop) and (self._seq == start_seq) and (time.time() < end):
                self._cond.wait(timeout=0.05)
            return bool(self._last_ok), self._frame

    def _run(self):
        # Warmup: some backends need a few grabs before retrieve() succeeds.
        while True:
            with self._lock:
                if self._stop:
                    return

            ok = self._cap.grab()
            if not ok:
                with self._cond:
                    self._last_ok = False
                    self._cond.notify_all()
                time.sleep(0.005)
                continue

            ok2, frame = self._cap.retrieve()
            with self._cond:
                self._last_ok = bool(ok2 and frame is not None)
                if self._last_ok:
                    self._frame = frame
                    self._seq += 1
                self._cond.notify_all()


requested_backend = ARGS.backend
cap = None
grabber = None

if ARGS.device_name:
    if os.name != "nt":
        print("❌ --device-name is supported on Windows only.")
        raise SystemExit(2)
    if requested_backend != "dshow":
        requested_backend = "dshow"
    print(f"🎥 Opening camera device-name='{ARGS.device_name}' backend={requested_backend}")
    cap = _open_camera_by_name_dshow(ARGS.device_name)
else:
    print(f"🎥 Opening camera index={ARGS.camera} backend={requested_backend}")
    cap = _open_camera(ARGS.camera, requested_backend)

if not cap.isOpened():
    print("❌ Không mở được camera!")
    if os.name == "nt":
        if not ARGS.device_name:
            print("💡 Gợi ý: chọn theo tên thiết bị (ổn định nhất với USB webcam):")
            print("   --device-name 'USB Camera' --backend dshow")
            print("💡 Nếu vẫn muốn chọn theo index, thử backend: --backend msmf")
        else:
            print("💡 Gợi ý: đảm bảo đúng tên thiết bị (xem --list-cameras)")
        print("💡 Có thể liệt kê tên camera (cần pygrabber): --list-cameras")
    raise SystemExit(1)

# Try to reduce capture latency (may be ignored by backend/driver)
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, max(0, int(ARGS.buffer_size)))
except Exception:
    pass

# [FIX-3] Ép resolution để giảm tải xử lý
cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESS_HEIGHT)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📷 Camera resolution: {actual_w}×{actual_h}")

if bool(ARGS.threaded_capture):
    grabber = _LatestFrameGrabber(cap).start()
    if int(ARGS.drop_frames) > 0:
        print("ℹ️  --drop-frames is ignored when --threaded-capture is enabled.")

prev_time = time.time()

try:
    while True:
        if grabber is not None:
            ret, frame = grabber.read(timeout=1.0)
        else:
            # Optional: drop frames to reduce latency when processing can't keep up.
            if int(ARGS.drop_frames) > 0:
                for _ in range(int(ARGS.drop_frames)):
                    cap.grab()

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
        faces = _detector(gray, DETECTOR_UPSAMPLE)

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
            miss_long = FACE_MISS_FRAMES > MAX_FACE_MISS
            FACE_MISS_FRAMES = 0

            # Chọn khuôn mặt lớn nhất (tránh pick nhầm mặt nhỏ/xa)
            face = max(faces, key=lambda r: r.width() * r.height())

            # Nếu đổi người (face box thay đổi mạnh) thì reset + calibrate lại
            curr_box = _rect_to_box(face)
            curr_area = _box_area(curr_box)
            curr_sig = None
            if curr_area >= MIN_FACE_AREA_FOR_SIG:
                curr_sig = _compute_face_signature(gray, curr_box)

            if _last_face_box is None:
                _last_face_box = curr_box
                _last_face_sig = curr_sig
                reset_person_state("first face")
            else:
                prev_box = _last_face_box
                prev_area = _box_area(prev_box)
                (pcx, pcy) = _box_center(prev_box)
                (ccx, ccy) = _box_center(curr_box)
                prev_w = max(1.0, float(prev_box[2] - prev_box[0]))
                center_shift = ((ccx - pcx) ** 2 + (ccy - pcy) ** 2) ** 0.5
                area_ratio = (curr_area / prev_area) if prev_area > 1 else 1.0

                if miss_long:
                    _last_face_box = curr_box
                    _last_face_sig = curr_sig
                    reset_person_state("face reacquired")
                elif center_shift > (FACE_CHANGE_CENTER_FRAC * prev_w) or (
                    area_ratio < FACE_CHANGE_AREA_RATIO or area_ratio > (1.0 / FACE_CHANGE_AREA_RATIO)
                ):
                    _last_face_box = curr_box
                    _last_face_sig = curr_sig
                    reset_person_state("face changed")
                elif curr_sig is not None and _last_face_sig is not None:
                    # Signature-based switch detection: catches different people in similar pose/position.
                    sim = float(np.dot(_last_face_sig, curr_sig))
                    if sim < FACE_SIG_SIM_THRESHOLD:
                        _last_face_box = curr_box
                        _last_face_sig = curr_sig
                        reset_person_state(f"face signature changed (sim={sim:.2f})")
                    else:
                        # Update signature slowly to adapt to small lighting changes.
                        updated = (1.0 - FACE_SIG_EMA) * _last_face_sig + FACE_SIG_EMA * curr_sig
                        n = float(np.linalg.norm(updated))
                        if n > 1e-6:
                            updated /= n
                        _last_face_sig = updated
                else:
                    _last_face_box = curr_box

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
                EAR_COUNTER = max(0, EAR_COUNTER - 1)
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

            now_ts = time.time()
            calibrate_pose_baseline(pitch, now_ts)
            if _pose_calibrated:
                pitch_drop = float(_pose_baseline_pitch) - float(pitch)
                # Trigger if drop vs baseline is large OR pitch is strongly downward in absolute terms.
                pose_trigger = (pitch_drop > float(_pose_delta_threshold)) or (float(pitch) <= float(PITCH_ABS_NOD_LIMIT))
            else:
                pitch_drop = 0.0
                pose_trigger = float(pitch) <= float(PITCH_ABS_NOD_LIMIT)

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
                            f"base:{_pose_baseline_pitch:.1f} drop:{pitch_drop:.1f} "
                            if _pose_calibrated
                            else "calib... "
                        )
                        + (f"th:{_pose_delta_threshold:.1f} " if _pose_calibrated else "")
                        + f"abs:{PITCH_ABS_NOD_LIMIT:.0f} "
                        + f"tr:{int(pose_trigger)} cnt:{POSE_COUNTER}/{POSE_CONSEC_FRAMES} "
                        + f"Y:{_yaw:.1f} R:{_roll:.1f}"
                    ),
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    thickness=2,
                )

            cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

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
    if grabber is not None:
        grabber.stop()
    cap.release()
    try:
        _alert_channel.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print("👋 Đã thoát.")