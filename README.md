# 🚗 Driver Drowsiness Detection System

> **Hệ Thống Phát Hiện Tình Trạng Buồn Ngủ Của Lái Xe**
> 
> Sử dụng kỹ thuật Computer Vision, Deep Learning, và 68 Face Landmarks để phát hiện và cảnh báo tình trạng buồn ngủ của lái xe trong thời gian thực.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

##  Mục Đích Dự Án

Dự án này nhằm **phát hiện tình trạng buồn ngủ/mắt nhắm của lái xe** thông qua:
- ✅ Phân tích ảnh video từ webcam
- ✅ Tính toán Eye Aspect Ratio (EAR) và Mouth Aspect Ratio (MAR)
- ✅ Sử dụng Convolutional Neural Network (CNN) để phân loại trạng thái mắt
- ✅ Phát cảnh báo (âm thanh) khi phát hiện lái xe buồn ngủ
- ✅ Ghi lại lịch sử buồn ngủ và thống kê

**Ứng dụng thực tế**: Giảm tai nạn giao thông do ngủ gật, tăng an toàn cho lái xe.

---

##  Kết Quả Chính

| Chỉ Số | Giá Trị |
|--------|--------|
| **Độ Chính Xác (Accuracy)** | 96.49% |
| **Precision (Mắt Đóng)** | 97.64% |
| **Recall (Mắt Đóng)** | 91.67% |
| **Recall (Miệng Ngáp)** | 100% |
| **FPS (Desktop)** | 20-30 FPS |
| **Model Size** | 12-15 MB |
| **Dataset** | 8,464 ảnh (3 lớp) |

---

##  Kiến Trúc Hệ Thống

### 2 Phiên Bản:

#### **V1: Phiên Bản Cơ Bản (dlib + EAR)**
```
Webcam → dlib Detector → 68 Landmarks → EAR Calculation → Alert
```
- Tính toán Eye Aspect Ratio (EAR)
- Không sử dụng CNN
- Tốc độ nhanh, nhẹ
- Kém chính xác trong điều kiện phức tạp

#### **V2: Phiên Bản Nâng Cao (CNN)**
```
Webcam → dlib Detector → 68 Landmarks → CNN Classifier → Alert + Analytics
```
- Sử dụng CNN (250K parameters)
- Phân loại 3 lớp: closed, open, yawn
- Accuracy cao hơn 96.49%
- Hỗ trợ các điều kiện phức tạp

### Chi Tiết Kiến Trúc CNN

```
Input (64×64×1 Grayscale)
    ↓
Conv2D(32) → ReLU → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → ReLU → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → ReLU → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(256, ReLU) → Dropout(0.5)
    ↓
Output: Dense(3, Softmax) → [P_closed, P_open, P_yawn]
```

---

## 📁 Cấu Trúc Dự Án

```
ML/
├── README.md                          # File này
├── requirements.txt                   # Thư viện cần thiết
├── 
├── # ============ TRAINING ============
├── train_cnn.py                       # Huấn luyện mô hình CNN
├── augment_dataset.py                 # Tăng cường dữ liệu (augmentation)
├── smart_collect.py                   # Thu thập dữ liệu thông minh
├── 
├── # ============ DEMO ============
├── face_and_eye_detector_single_image.py    # Detect mắt từ 1 ảnh
├── face_and_eye_detector_webcam_video.py    # Detect mắt từ webcam
├── drowsiness_detect.py                     # Detect buồn ngủ (v1)
├── 
├── # ============ PRODUCTION ============
├── integrate_cnn.py                   # Hệ thống production (v2)
├── probe_cameras.py                   # Kiểm tra camera
├── run_pipeline.py                    # Pipeline chính
├── 
├── # ============ MODELS ============
├── best_model.h5                      # Model CNN tốt nhất (v1)
├── best_model_v2.h5                   # Model CNN v2
├── eye_model.tflite                   # Model tối ưu cho mobile
├── class_indices.json                 # Ánh xạ lớp
├── class_indices_v2.json              # Ánh xạ lớp v2
├── 
├── # ============ DATA ============
├── dataset/                           # Dataset gốc
│   ├── closed/                        # Ảnh mắt đóng
│   ├── open/                          # Ảnh mắt mở
│   └── yawn/                          # Ảnh miệng ngáp
├── 
├── # ============ CASCADE ============
├── haarcascades/
│   ├── haarcascade_frontalface_default.xml
│   └── haarcascade_eye.xml
├── 
├── # ============ AUDIO ============
├── audio/
│   └── alert.wav                      # Âm thanh cảnh báo
├── 
├── # ============ REPORTS ============
├── compare_models_report.txt          # So sánh các model
├── eval_report.txt                    # Báo cáo đánh giá
├── eval_report_after_clean.txt        # Báo cáo sau làm sạch
├── 
├── # ============ SHELL SCRIPTS ============
├── run_v1.sh                          # Chạy v1 (Linux/Git Bash)
├── run_v2.sh                          # Chạy v2 (Linux/Git Bash)
├── run_realtime_compare.sh            # So sánh v1 & v2 (Linux/Git Bash)
├── run_v1.bat                         # Chạy v1 (Windows)
├── run_v2.bat                         # Chạy v2 (Windows)
└── run_ab.bat                         # So sánh A/B (Windows)
```

---

##  Yêu Cầu Hệ Thống

### Hardware
- **CPU**: Intel i5 trở lên hoặc tương đương
- **RAM**: 4GB tối thiểu (8GB khuyến nghị)
- **GPU**: Tuỳ chọn (CUDA/cuDNN nếu có)
- **Camera**: Webcam hoặc camera tích hợp

### Software
- **Python**: 3.8+ 
- **OS**: Windows 10+, macOS, Linux

### Dependencies
```
numpy>=1.20.0
scipy>=1.5.0
opencv-python>=4.5.0
imutils>=0.5.4
dlib>=19.20.0
pygame>=2.0.0
scikit-learn>=0.24.0
tensorflow>=2.6.0
Pillow>=8.0.0
```

---

##  Cài Đặt & Cấu Hình

### 1. Clone hoặc tải dự án
```bash
git clone https://github.com/ngochoai0810/ML
cd /path/to/ML
```

### 2. Cài thư viện
```bash
pip install -r requirements.txt
```
```bash
 `./.venv311/Scripts/python.exe integrate_cnn.py --camera 1 --model best_model_v2.h5 --class-json class_indices_v2.json`
```
### 3. Download Shape Predictor (Important!)
```bash
# Download từ dlib
# Link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Extract (Linux/Git Bash)
bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2

# Hoặc Extract (Windows)
# Dùng 7-Zip hoặc WinRAR để giải nén file .bz2
```

### 4. Kiểm tra camera
```bash
python probe_cameras.py
```

---

##  Cách Sử Dụng

### **Option 1: Demo - Detect Mắt Từ Ảnh**
```bash
# Đặt ảnh vào thư mục images/ với tên "test.jpeg"
python face_and_eye_detector_single_image.py
```

### **Option 2: Demo - Detect Mắt Từ Webcam**
```bash
python face_and_eye_detector_webcam_video.py
```

### **Option 3: V1 - Phát Hiện Buồn Ngủ (dlib + EAR)**
```bash
# Windows
run_v1.bat
# Hoặc
run_v1.bat 0  # Chỉ định camera index

# Linux/Git Bash
./run_v1.sh
./run_v1.sh 0
```

Hoặc chạy trực tiếp:
```bash
python drowsiness_detect.py
```

### **Option 4: V2 - Phát Hiện Buồn Ngủ (CNN - Tốt Nhất)**
```bash
# Windows
run_v2.bat
# Hoặc
run_v2.bat 0  # Chỉ định camera index

# Linux/Git Bash
./run_v2.sh
./run_v2.sh 0
```

Hoặc chạy trực tiếp:
```bash
python integrate_cnn.py --camera 1 --model best_model.h5 --class-json class_indices.json
```

### **Option 5: So Sánh V1 & V2 (Real-time)**
```bash
# Windows
run_ab.bat

# Linux/Git Bash
./run_realtime_compare.sh
```

### **Option 6: Thu Thập Dữ Liệu**
```bash
# Thu thập 7 ảnh mắt mở không augment
python smart_collect.py --label open --camera 1 --target 7 --no-aug

# Thu thập 7 ảnh mắt đóng
python smart_collect.py --label closed --camera 1 --target 7 --no-aug

# Hoặc với augmentation (tạo 7 ảnh gốc × ~10 biến đổi)
python smart_collect.py --label yawn --camera 1 --target 7
```

### **Option 7: Huấn Luyện Mô Hình (Advanced)**
```bash
# Train mô hình mới từ dataset/
python train_cnn.py --epochs 40 --batch-size 32

# Fine-tune từ model cũ
python train_cnn.py --resume best_model.h5 --lr 1e-4 --epochs 15

# Train trên GPU
python train_cnn.py --epochs 100 --batch-size 64
```

---

##  Phím Tắt Trong Ứng Dụng

| Phím | Chức Năng |
|------|----------|
| **q** | Thoát chương trình |
| **D** | Bật/tắt debug mode (hiển thị head pose) |
| **P** | Pause/Resume video |
| **S** | Save screenshot |

---

##  Kết Quả & Metrics

### Validation Results
```
Classes: ['closed', 'open', 'yawn']
Validation Accuracy: 96.49%

Confusion Matrix:
              closed  open  yawn
closed         539    49    0
open            13   576    0
yawn             0    0   588

Classification Report:
              precision    recall  f1-score
closed         0.9764     0.9167    0.9456
open           0.9216     0.9779    0.9489
yawn           1.0000     1.0000    1.0000

Accuracy: 96.49%
```

### Performance
- **Inference Time**: ~10-20ms per frame (Desktop CPU)
- **FPS**: 20-30 FPS (Desktop)
- **Memory Usage**: ~200-300MB
- **Model Size**: 12-15MB (H5 format), 4-5MB (TFLite)

---

##  Cơ Sở Lý Thuyết

### 1. OpenCV (Open Source Computer Vision)
- Thư viện xử lý ảnh và video
- Đọc frame từ webcam, chuyển đổi màu, vẽ đồ họa

### 2. Haar Cascade
- Phương pháp phát hiện vật thể (trong demo)
- Dùng để phát hiện khuôn mặt và mắt ban đầu
- Nhẹ nhưng kém chính xác

### 3. dlib + 68 Face Landmarks
- Xác định 68 điểm đặc trưng trên khuôn mặt
- Dùng để tính Eye Aspect Ratio (EAR)
- Độ chính xác cao

### 4. Eye Aspect Ratio (EAR)
```
EAR = ||p2 - p6|| + ||p3 - p5|| / (2 × ||p1 - p4||)

Nếu EAR < threshold (0.3) → Mắt đóng
```

### 5. CNN (Convolutional Neural Network)
- 3 Conv Blocks + Fully Connected
- Phân loại 3 lớp: closed, open, yawn
- Accuracy 96.49%

### 6. Data Augmentation
- Rotate, Flip, Brightness, Noise, Shift
- Mỗi ảnh gốc → ~10 ảnh augmented
- Giảm overfitting

---

##  References & Credits

- **Eye Aspect Ratio Algorithm**: Adrian RoseBrock, PyImageSearch
- **dlib Library**: Davis King (http://dlib.net)
- **TensorFlow/Keras**: Google
- **OpenCV**: OpenCV Community

