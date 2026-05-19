"""smart_collect.py — Thu thập dataset thông minh, có hướng dẫn góc đầu tự động.

Dùng thay thế `collect_data.py` để tiết kiệm thời gian.

Cách dùng:
    python smart_collect.py --label closed --camera 1
    python smart_collect.py --label open   --camera 1
    python smart_collect.py --label yawn   --camera 1

Nếu file landmark nằm chỗ khác, chỉ định thêm:
    python smart_collect.py --label open --camera 1 --predictor path/to/shape_predictor_68_face_landmarks.dat

Tính năng:
    - Tự động xoay qua 7 tư thế đầu (thẳng, trái, phải, cúi, ngẩng, xoay trái, xoay phải)
    - Countdown + overlay hướng dẫn trực tiếp lên webcam
    - Auto-save với bộ lọc sharpness + dedupe
    - Augment ngay lúc thu thập (rotate/flip/brightness/noise) → 1 ảnh gốc = 7 ảnh saved
    - Progress bar realtime + ETA

Phím tắt:
    - Q: thoát
"""

from __future__ import annotations

import argparse
import math
import os
import time

import cv2
import dlib
import numpy as np
from imutils import face_utils


# ─── Config ──────────────────────────────────────────────────────────────────
POSE_SESSIONS = [
    {
        "name": "THANG",
        "hint": "Giu dau thang, nhin thang vao camera",
        "icon": "⬆",
        "seconds": 8,
    },
    {
        "name": "NGHIENG_TRAI",
        "hint": "Nghieng dau sang TRAI ~25 do",
        "icon": "↖",
        "seconds": 6,
    },
    {
        "name": "NGHIENG_PHAI",
        "hint": "Nghieng dau sang PHAI ~25 do",
        "icon": "↗",
        "seconds": 6,
    },
    {
        "name": "CUI_XUONG",
        "hint": "Cui dau xuong ~20 do (nhu nhin xuong duong)",
        "icon": "⬇",
        "seconds": 6,
    },
    {
        "name": "NGANG_LEN",
        "hint": "Ngang dau len ~15 do",
        "icon": "⬆",
        "seconds": 5,
    },
    {
        "name": "XOAY_TRAI",
        "hint": "Xoay mat sang TRAI (yaw ~20 do)",
        "icon": "←",
        "seconds": 5,
    },
    {
        "name": "XOAY_PHAI",
        "hint": "Xoay mat sang PHAI (yaw ~20 do)",
        "icon": "→",
        "seconds": 5,
    },
]

TARGET_PER_SESSION_DEFAULT = 60  # ảnh gốc mỗi pose
SAVE_INTERVAL = 0.12  # giây giữa 2 lần lưu
DEDUPE_THRESH = 2.5
CROP_SIZE = (64, 64)
FACE_GRACE_FRAMES = 30  # giữ bbox cũ tối đa N frame khi mất detect

# MIN_SHARPNESS thấp hơn cho closed/yawn vì ít texture hơn
MIN_SHARPNESS_BY_LABEL = {
    "open": 18.0,
    "closed": 8.0,
    "yawn": 10.0,
}

# Will be set at runtime based on --label
MIN_SHARPNESS = MIN_SHARPNESS_BY_LABEL["open"]


# ─── Dlib setup ──────────────────────────────────────────────────────────────
_DETECTOR = dlib.get_frontal_face_detector()


def _load_predictor(path: str) -> dlib.shape_predictor:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy '{path}'. Hãy tải file shape_predictor_68_face_landmarks.dat và đặt cùng thư mục project."
        )
    return dlib.shape_predictor(path)


# ─── Paths ───────────────────────────────────────────────────────────────────

def _count_existing(out_dir: str) -> int:
    try:
        return len([f for f in os.listdir(out_dir) if f.lower().endswith(".jpg")])
    except FileNotFoundError:
        return 0


# ─── Augmentation ────────────────────────────────────────────────────────────

def augment(img: np.ndarray) -> list[np.ndarray]:
    """Trả về list ảnh augmented từ 1 crop gốc (grayscale)."""
    variants: list[np.ndarray] = [img]

    h, w = img.shape
    cx, cy = w // 2, h // 2

    # Rotations: ±15°
    for angle in (-15, 15):
        M = cv2.getRotationMatrix2D((cx, cy), float(angle), 1.0)
        variants.append(cv2.warpAffine(img, M, (w, h)))

    # Flip ngang
    variants.append(cv2.flip(img, 1))

    # Brightness: tối & sáng
    for delta in (-35, 35):
        variants.append(np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8))

    # Gaussian noise
    noise = np.random.normal(0, 6, img.shape).astype(np.int16)
    variants.append(
        np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    )

    return variants  # 7 ảnh


# ─── Quality filters ─────────────────────────────────────────────────────────

def is_sharp(img: np.ndarray) -> bool:
    return float(cv2.Laplacian(img, cv2.CV_64F).var()) >= MIN_SHARPNESS


_last_saved: np.ndarray | None = None


def is_duplicate(img: np.ndarray) -> bool:
    global _last_saved
    if _last_saved is None:
        return False
    if img.shape != _last_saved.shape:
        return False
    return (
        float(abs(img.astype(np.float32) - _last_saved.astype(np.float32)).mean())
        < DEDUPE_THRESH
    )


# ─── Crop helpers ────────────────────────────────────────────────────────────

def crop_eye(gray: np.ndarray, pts: np.ndarray, pad: int = 6) -> np.ndarray | None:
    x, y, w, h = cv2.boundingRect(pts)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray.shape[1], x + w + pad)
    y2 = min(gray.shape[0], y + h + pad)
    c = gray[y1:y2, x1:x2]
    return cv2.resize(c, CROP_SIZE) if c.size > 0 else None


def crop_mouth(gray: np.ndarray, pts: np.ndarray, pad: int = 8) -> np.ndarray | None:
    x, y, w, h = cv2.boundingRect(pts)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray.shape[1], x + w + pad)
    y2 = min(gray.shape[0], y + h + pad)
    c = gray[y1:y2, x1:x2]
    return cv2.resize(c, CROP_SIZE) if c.size > 0 else None


# ─── Save helpers ─────────────────────────────────────────────────────────────

def save_crop(
    crop: np.ndarray | None,
    out_dir: str,
    base_idx: int,
    augment_per_crop: bool,
    suffix: str = "",
) -> int:
    global _last_saved

    if crop is None or (not is_sharp(crop)) or is_duplicate(crop):
        return 0

    variants = augment(crop) if augment_per_crop else [crop]
    saved = 0

    for i, v in enumerate(variants):
        fname = f"{base_idx:06d}{suffix}_a{i}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname), v)
        saved += 1

    _last_saved = crop.copy()
    return saved


# ─── Overlay drawing ─────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(
    frame: np.ndarray,
    label: str,
    session: dict,
    session_idx: int,
    raw_count: int,
    total_saved: int,
    target_per_session: int,
    countdown: int,
    pose_done: bool,
    overall_done: bool,
    eta_seconds: int | None,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # Top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 40), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Session progress (top bar)
    pct = (session_idx + 1) / max(len(POSE_SESSIONS), 1)
    bar_w = int(w * pct)
    cv2.rectangle(frame, (0, 55), (bar_w, 60), (80, 220, 100), -1)

    label_str = (
        f"[{label.upper()}]  Pose {session_idx + 1}/{len(POSE_SESSIONS)}: {session['name']}"
    )
    cv2.putText(frame, label_str, (10, 22), FONT, 0.6, (200, 220, 255), 1)

    saved_str = f"Da luu: {total_saved} anh  (goc: {raw_count})"
    cv2.putText(frame, saved_str, (10, 44), FONT, 0.55, (120, 255, 180), 1)

    # Hint box (bottom)
    box_y = h - 110
    cv2.rectangle(frame, (0, box_y), (w, h), (15, 15, 35), -1)

    hint = session["hint"]
    cv2.putText(frame, hint, (12, box_y + 24), FONT, 0.58, (255, 255, 120), 1)

    # Countdown / session progress
    if countdown > 0:
        cd_str = f"Chuan bi: {countdown}"
        cv2.putText(frame, cd_str, (w // 2 - 120, h // 2), FONT, 1.4, (0, 200, 255), 3)
    elif not pose_done:
        prog = min(raw_count / max(target_per_session, 1), 1.0)
        bar_x1, bar_y1 = 12, box_y + 40
        bar_x2 = w - 12
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y1 + 16), (50, 50, 70), -1)
        cv2.rectangle(
            frame,
            (bar_x1, bar_y1),
            (int(bar_x1 + (bar_x2 - bar_x1) * prog), bar_y1 + 16),
            (80, 220, 100),
            -1,
        )
        prog_str = f"{int(prog * 100)}%  ({raw_count}/{target_per_session})"
        cv2.putText(frame, prog_str, (bar_x1 + 4, bar_y1 + 13), FONT, 0.45, (255, 255, 255), 1)

        if eta_seconds is not None and eta_seconds < 900:
            eta_str = f"ETA: {int(eta_seconds)}s"
            cv2.putText(frame, eta_str, (w - 110, box_y + 24), FONT, 0.5, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "XONG! Chuyen pose tiep theo...", (12, box_y + 55), FONT, 0.65, (100, 255, 100), 2)

    if overall_done:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 180, 0), 4)
        cv2.putText(frame, "HOAN THANH! Nhan Q de thoat.", (w // 2 - 240, h // 2), FONT, 0.9, (0, 255, 120), 2)

    return frame


# ─── Main loop ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="open", choices=["open", "closed", "yawn"])
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument(
        "--predictor",
        default="shape_predictor_68_face_landmarks.dat",
        help="Path to dlib shape predictor .dat file (default: shape_predictor_68_face_landmarks.dat).",
    )
    parser.add_argument("--out-dir", default="dataset")
    parser.add_argument("--no-aug", action="store_true", help="Tắt augmentation ngay lúc thu thập")
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_PER_SESSION_DEFAULT,
        help="Số ảnh GỐC mỗi pose session",
    )
    args = parser.parse_args()

    augment_per_crop = not args.no_aug
    target_per_session = int(args.target)

    global MIN_SHARPNESS
    MIN_SHARPNESS = float(MIN_SHARPNESS_BY_LABEL.get(args.label, 18.0))

    out_dir = os.path.join(args.out_dir, args.label)
    os.makedirs(out_dir, exist_ok=True)

    predictor = _load_predictor(args.predictor)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERR] Khong mo duoc camera {args.camera}")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    file_idx = _count_existing(out_dir) * 10  # offset để tránh ghi đè file cũ
    total_saved = 0

    print(
        f"\n=== Smart Collect v2: label='{args.label}', aug={'ON' if augment_per_crop else 'OFF'} ==="
    )
    print(
        "    Moi pose: "
        f"{target_per_session} anh goc  x  {7 if augment_per_crop else 1} aug  =  "
        f"{target_per_session * (7 if augment_per_crop else 1)} anh/pose"
    )
    print(
        "    Tong du kien: "
        f"{target_per_session * (7 if augment_per_crop else 1) * len(POSE_SESSIONS)} anh"
    )
    print("    Nhan Q bat cu luc nao de thoat som.\n")

    try:
        for s_idx, session in enumerate(POSE_SESSIONS):
            print(
                f"\n→ Pose [{s_idx + 1}/{len(POSE_SESSIONS)}]: {session['name']}  —  {session['hint']}"
            )

            # ── Countdown 3 giây ─────────────────────────────────────────
            countdown_end = time.time() + 3
            while time.time() < countdown_end:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame = cv2.flip(frame, 1)
                cd = int(math.ceil(countdown_end - time.time()))
                frame = draw_overlay(
                    frame,
                    args.label,
                    session,
                    s_idx,
                    0,
                    total_saved,
                    target_per_session,
                    cd,
                    False,
                    False,
                    None,
                )
                cv2.imshow("Smart Collect", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return 0

            # ── Thu thập ───────────────────────────────────────────────
            raw_count = 0
            last_save_t = 0.0
            session_start = time.time()

            # Face bbox cache: dlib HOG detector thường khó detect khi mắt nhắm.
            cached_face = None  # dlib.rectangle | None
            face_miss_n = 0

            while raw_count < target_per_session:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect face with upsample=1 (nhạy hơn với mặt nhỏ/xa)
                faces = _DETECTOR(gray, 1)
                if len(faces) > 0:
                    cached_face = faces[0]
                    face_miss_n = 0
                else:
                    face_miss_n += 1

                use_face = None
                if len(faces) > 0:
                    use_face = faces[0]
                elif cached_face is not None and face_miss_n <= FACE_GRACE_FRAMES:
                    use_face = cached_face

                crops: list[np.ndarray] = []
                shape_pts = None
                if use_face is not None:
                    try:
                        shape_pts = face_utils.shape_to_np(predictor(gray, use_face))
                    except Exception:
                        shape_pts = None

                if shape_pts is not None:
                    if args.label == "yawn":
                        c = crop_mouth(gray, shape_pts[48:68])
                        if c is not None:
                            crops.append(c)
                            for pt in shape_pts[48:68]:
                                cv2.circle(frame, tuple(pt), 2, (0, 200, 255), -1)
                    else:
                        cl = crop_eye(gray, shape_pts[36:42])
                        cr = crop_eye(gray, shape_pts[42:48])
                        if cl is not None:
                            crops.append(cl)
                            for pt in shape_pts[36:42]:
                                cv2.circle(frame, tuple(pt), 2, (0, 255, 100), -1)
                        if cr is not None:
                            crops.append(cr)
                            for pt in shape_pts[42:48]:
                                cv2.circle(frame, tuple(pt), 2, (0, 255, 100), -1)

                    # Draw bbox: green = fresh detect, yellow = cache
                    x1 = int(use_face.left())
                    y1 = int(use_face.top())
                    x2 = int(use_face.right())
                    y2 = int(use_face.bottom())
                    color = (0, 255, 0) if len(faces) > 0 else (0, 200, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    if len(faces) == 0 and cached_face is not None:
                        cv2.putText(
                            frame,
                            f"Cache ({face_miss_n}f)",
                            (x1, max(0, y1 - 6)),
                            FONT,
                            0.4,
                            (0, 200, 255),
                            1,
                        )

                # Preview crop nhỏ góc phải
                if crops:
                    preview = cv2.resize(crops[0], (80, 80))
                    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                    frame[8:88, frame.shape[1] - 88 : frame.shape[1] - 8] = preview_rgb
                    cv2.rectangle(
                        frame,
                        (frame.shape[1] - 89, 7),
                        (frame.shape[1] - 7, 89),
                        (200, 200, 0),
                        1,
                    )

                # Auto-save theo interval
                now = time.time()
                if crops and (now - last_save_t) >= SAVE_INTERVAL:
                    for crop in crops:
                        saved_n = save_crop(
                            crop,
                            out_dir,
                            file_idx,
                            augment_per_crop,
                            suffix=f"_p{s_idx}",
                        )
                        if saved_n > 0:
                            total_saved += saved_n
                            file_idx += 1
                            raw_count += 1
                            last_save_t = now
                            if raw_count >= target_per_session:
                                break

                # ETA
                elapsed = time.time() - session_start + 0.001
                rate = raw_count / elapsed
                eta = int((target_per_session - raw_count) / rate) if rate > 0 else 999

                frame = draw_overlay(
                    frame,
                    args.label,
                    session,
                    s_idx,
                    raw_count,
                    total_saved,
                    target_per_session,
                    0,
                    False,
                    False,
                    eta,
                )

                if use_face is None:
                    cv2.putText(
                        frame,
                        "!! Mat khoi vung camera !!",
                        (10, frame.shape[0] - 75),
                        FONT,
                        0.6,
                        (0, 60, 255),
                        2,
                    )
                elif not crops:
                    cv2.putText(
                        frame,
                        "Khong crop duoc - di chuyen gan hon",
                        (10, frame.shape[0] - 75),
                        FONT,
                        0.5,
                        (0, 150, 255),
                        1,
                    )

                cv2.imshow("Smart Collect", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print(f"\nDung som. Tong da luu: {total_saved} anh.")
                    return 0

            # ── 'XONG' flash 1 giây ────────────────────────────────────
            done_end = time.time() + 1.0
            while time.time() < done_end:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame = cv2.flip(frame, 1)
                frame = draw_overlay(
                    frame,
                    args.label,
                    session,
                    s_idx,
                    raw_count,
                    total_saved,
                    target_per_session,
                    0,
                    True,
                    False,
                    None,
                )
                cv2.imshow("Smart Collect", frame)
                cv2.waitKey(1)

            print(
                f"   ✓ {raw_count} goc  →  {raw_count * (7 if augment_per_crop else 1)} da luu"
            )

        # ── Hoàn thành ────────────────────────────────────────────────
        print(f"\n✅ XONG! Tong tat ca: {total_saved} anh trong '{out_dir}/'")
        print("   Tip: chay 'python train_cnn.py' de retrain voi dataset moi.")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame = cv2.flip(frame, 1)
            frame = draw_overlay(
                frame,
                args.label,
                POSE_SESSIONS[-1],
                len(POSE_SESSIONS) - 1,
                target_per_session,
                total_saved,
                target_per_session,
                0,
                True,
                True,
                None,
            )
            cv2.imshow("Smart Collect", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
