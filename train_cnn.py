"""
train_cnn.py — Colab-ready CNN training script
Driver Drowsiness Detector — Eye State Classification

Cách dùng trên Google Colab:
  1. Upload file này lên Colab
  2. Upload dataset.zip (hoặc thư mục dataset/), giải nén
  3. Runtime > Change runtime type > T4 GPU
  4. Chạy: !python train_cnn.py
  5. Tải về: best_model.h5, eye_model.tflite, class_indices.json,
             training_history.png, confusion_matrix.png

Cấu trúc dataset mong đợi:
  dataset/
    open/     ← ảnh mắt mở   (300-500 ảnh .jpg)
    closed/   ← ảnh mắt nhắm (300-500 ảnh .jpg)
    yawn/     ← miệng ngáp   (200-300 ảnh .jpg) [tuỳ chọn]
"""

# ─────────────────────────────────────────────────────────────
# BƯỚC 0 — Cài thư viện (bỏ comment nếu chạy trên Colab)
# ─────────────────────────────────────────────────────────────
# !pip install tensorflow matplotlib seaborn scikit-learn

import json
import os
import sys
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image

print(f"✅ TensorFlow version: {tf.__version__}")
print(f"✅ GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ─────────────────────────────────────────────────────────────
# BƯỚC 1 — Giải nén dataset (nếu upload file .zip lên Colab)
# ─────────────────────────────────────────────────────────────
DATASET_ZIP = "dataset.zip"  # tên file zip bạn upload
DATASET_DIR = "dataset"  # thư mục sau khi giải nén

if not os.path.isdir(DATASET_DIR):
    if os.path.exists(DATASET_ZIP):
        print(f"📦 Giải nén {DATASET_ZIP}...")
        with zipfile.ZipFile(DATASET_ZIP, "r") as z:
            z.extractall(".")
        print("✅ Giải nén xong")
    else:
        print(f"❌ Không tìm thấy '{DATASET_DIR}/' hoặc '{DATASET_ZIP}'")
        print("   Hãy upload dataset.zip hoặc thư mục dataset/ lên Colab trước")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────
# BƯỚC 2 — Kiểm tra & thống kê dataset
# ─────────────────────────────────────────────────────────────
classes = sorted(
    [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
)
print(f"\n📁 Classes tìm thấy: {classes}")

total = 0
for cls in classes:
    n = len(os.listdir(os.path.join(DATASET_DIR, cls)))
    print(f"   {cls:10s}: {n} ảnh")
    total += n
print(f"   {'TOTAL':10s}: {total} ảnh\n")

if total < 100:
    print("⚠️  Dataset quá nhỏ (<100 ảnh). Nên thu thêm để model học tốt hơn.")

# ─────────────────────────────────────────────────────────────
# Kiểm tra ảnh corrupt (tránh crash giữa chừng khi training)
# ─────────────────────────────────────────────────────────────
bad_files = []
for cls in classes:
    cls_path = os.path.join(DATASET_DIR, cls)
    for fname in os.listdir(cls_path):
        fpath = os.path.join(cls_path, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with Image.open(fpath) as img:
                img.verify()
        except Exception:
            bad_files.append(fpath)

if bad_files:
    print(f"⚠️  {len(bad_files)} file ảnh lỗi — đang xóa...")
    for f in bad_files:
        print(f"   {f}")
        try:
            os.remove(f)
        except OSError:
            pass
else:
    print("✅ Tất cả ảnh hợp lệ")

NUM_CLASSES = len(classes)

# ─────────────────────────────────────────────────────────────
# BƯỚC 3 — Cấu hình
# ─────────────────────────────────────────────────────────────
IMG_SIZE = 64  # ảnh 64x64 (crop mắt từ collect_data.py)
BATCH_SIZE = 32
EPOCHS = 40  # EarlyStopping sẽ dừng sớm nếu không cải thiện
VALIDATION_SPLIT = 0.2  # 80% train, 20% validation
SEED = 42

# ─────────────────────────────────────────────────────────────
# BƯỚC 4 — Data Augmentation & Generator
# ─────────────────────────────────────────────────────────────
# Augmentation chỉ áp dụng cho train, val chỉ rescale
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    preprocessing_function=lambda x: x,  # tường minh: không có xử lý ẩn
    rotation_range=20,  # match góc đầu thực tế (thường ~25°)
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,  # lật ngang — mắt trái/phải giống nhau
    brightness_range=[0.6, 1.4],  # giả lập điều kiện ánh sáng khác nhau
    fill_mode="nearest",
    validation_split=VALIDATION_SPLIT,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=VALIDATION_SPLIT)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",  # ảnh grayscale — nhẹ hơn RGB
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=SEED,
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=SEED,
    shuffle=False,
)

print(f"\n📊 Train: {train_gen.samples} ảnh | Val: {val_gen.samples} ảnh")
print(f"📌 Class indices: {train_gen.class_indices}")

# ─────────────────────────────────────────────────────────────
# BƯỚC 5 — Xây dựng model CNN
# ─────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> tf.keras.Model:
    """CNN nhỏ gọn, phù hợp ảnh mắt 64x64 grayscale."""

    model = models.Sequential(
        [
            # Block 1
            layers.Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="relu",
                input_shape=(IMG_SIZE, IMG_SIZE, 1),
            ),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            # Block 2
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            # Block 3
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            # Fully connected
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="EyeStateCNN",
    )

    return model


model = build_model(NUM_CLASSES)
model.summary()

# ─────────────────────────────────────────────────────────────
# BƯỚC 6 — Compile
# ─────────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ─────────────────────────────────────────────────────────────
# BƯỚC 7 — Callbacks
# ─────────────────────────────────────────────────────────────
cb_list = [
    callbacks.EarlyStopping(
        monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),
    callbacks.ModelCheckpoint(
        "best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1
    ),
]

# ─────────────────────────────────────────────────────────────
# BƯỚC 8 — Train
# ─────────────────────────────────────────────────────────────
print("\n🚀 Bắt đầu training...\n")

class_ids = np.unique(train_gen.classes)
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=class_ids,
    y=train_gen.classes,
)
class_weight_dict = {
    int(class_id): float(weight) for class_id, weight in zip(class_ids, class_weights_arr)
}
print(f"⚖️  Class weights: {class_weight_dict}")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=cb_list,
    class_weight=class_weight_dict,
    verbose=1,
)

print("\n✅ Training hoàn tất!")

# ─────────────────────────────────────────────────────────────
# BƯỚC 9 — Đánh giá & vẽ biểu đồ
# ─────────────────────────────────────────────────────────────
# Tải lại model tốt nhất
model = tf.keras.models.load_model("best_model.h5")
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f"\n📈 Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"📉 Val Loss:     {val_loss:.4f}")

# Vẽ accuracy & loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history["accuracy"], label="Train Acc")
ax1.plot(history.history["val_accuracy"], label="Val Acc")
ax1.set_title("Accuracy qua các epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)

ax2.plot(history.history["loss"], label="Train Loss")
ax2.plot(history.history["val_loss"], label="Val Loss")
ax2.set_title("Loss qua các epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
plt.show()
print("💾 Đã lưu: training_history.png")

# Confusion matrix
val_gen.reset()
y_pred_probs = model.predict(val_gen, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_gen.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.ylabel("Thực tế")
plt.xlabel("Dự đoán")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("💾 Đã lưu: confusion_matrix.png")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

# ─────────────────────────────────────────────────────────────
# BƯỚC 10 — Export sang TFLite
# ─────────────────────────────────────────────────────────────
print("\n🔄 Đang export sang TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantization nhẹ

tflite_model = converter.convert()
with open("eye_model.tflite", "wb") as f:
    f.write(tflite_model)

size_h5 = os.path.getsize("best_model.h5") / 1024
size_tflite = os.path.getsize("eye_model.tflite") / 1024
print(f"✅ best_model.h5    : {size_h5:.1f} KB")
print(f"✅ eye_model.tflite : {size_tflite:.1f} KB")

# ─────────────────────────────────────────────────────────────
# BƯỚC 11 — Lưu class_indices
# ─────────────────────────────────────────────────────────────
class_indices = train_gen.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}

with open("class_indices.json", "w", encoding="utf-8") as f:
    json.dump(idx_to_class, f, indent=2, ensure_ascii=False)

print(f"\n💾 Đã lưu class_indices.json: {idx_to_class}")

# ─────────────────────────────────────────────────────────────
# TỔNG KẾT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("🎉 HOÀN THÀNH! Các file đã tạo:")
print("   best_model.h5          ← dùng trong integrate_cnn.py")
print("   eye_model.tflite       ← phiên bản nhỏ hơn")
print("   class_indices.json     ← mapping index → tên class")
print("   training_history.png   ← biểu đồ accuracy/loss")
print("   confusion_matrix.png   ← ma trận nhầm lẫn")
print("=" * 50)
print("\n📌 Bước tiếp theo:")
print("   1. Tải best_model.h5 + class_indices.json về máy")
print("   2. Đặt vào thư mục project")
print("   3. Chạy integrate_cnn.py")
