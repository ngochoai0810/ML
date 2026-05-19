"""augment_dataset.py

Tạo dataset augmented từ dataset/ sang dataset_aug/.

- Rotate ±10°, ±20°
- Flip ngang
- Brightness ±30%
- Gaussian noise (sigma=5)
- Horizontal shift ±5px

Mục tiêu: mỗi ảnh gốc sinh ra ~10 ảnh augmented.
"""

from pathlib import Path
from typing import List

import cv2
import numpy as np


def _ensure_gray_64(img_gray: np.ndarray) -> np.ndarray:
    if img_gray is None or img_gray.size == 0:
        raise ValueError("Ảnh rỗng")
    if len(img_gray.shape) != 2:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img_gray, (64, 64), interpolation=cv2.INTER_AREA)


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _shift_x(img: np.ndarray, dx: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _brightness(img: np.ndarray, factor: float) -> np.ndarray:
    # factor: 0.7 (tối hơn), 1.3 (sáng hơn)
    out = np.clip(img.astype(np.float32) * factor, 0, 255)
    return out.astype(np.uint8)


def _gaussian_noise(img: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255)
    return out.astype(np.uint8)


def augment_image(img_gray_64: np.ndarray) -> List[np.ndarray]:
    """Return list of augmented variants (không bao gồm ảnh gốc)."""

    results: List[np.ndarray] = []

    # Rotations để mô phỏng đầu nghiêng
    for angle in (-20, -10, 10, 20):
        results.append(_rotate(img_gray_64, angle))

    # Flip
    results.append(cv2.flip(img_gray_64, 1))

    # Brightness ±30%
    results.append(_brightness(img_gray_64, 0.7))
    results.append(_brightness(img_gray_64, 1.3))

    # Gaussian noise
    results.append(_gaussian_noise(img_gray_64, sigma=5.0))

    # Horizontal shift ±5px
    results.append(_shift_x(img_gray_64, -5))
    results.append(_shift_x(img_gray_64, 5))

    # Tổng: 4 + 1 + 2 + 1 + 2 = 10
    return results


def main() -> None:
    src_root = Path("dataset")
    dst_root = Path("dataset_aug")

    labels = ["open", "closed", "yawn"]
    total_written = 0

    for label in labels:
        src_dir = src_root / label
        dst_dir = dst_root / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"⚠️  Không thấy thư mục: {src_dir}")
            continue

        idx = 0
        img_paths = [
            *src_dir.glob("*.jpg"),
            *src_dir.glob("*.jpeg"),
            *src_dir.glob("*.png"),
            *src_dir.glob("*.bmp"),
        ]

        if not img_paths:
            print(f"⚠️  Không có ảnh trong: {src_dir}")
            continue

        for img_path in img_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            try:
                img = _ensure_gray_64(img)
            except Exception:
                continue

            for aug in augment_image(img):
                out_path = dst_dir / f"{idx:06d}.jpg"
                if cv2.imwrite(str(out_path), aug):
                    idx += 1
                    total_written += 1

        print(f"✅ {label}: đã tạo {idx} ảnh -> {dst_dir}")

    print(f"\nAugmentation done! Tổng ảnh tạo mới: {total_written}")


if __name__ == "__main__":
    main()
