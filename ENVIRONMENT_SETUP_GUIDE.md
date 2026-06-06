# Environment Setup Guide

Hướng dẫn thiết lập môi trường để chạy project HRT-Project trên máy local.

> **Quan trọng**: Đọc kỹ toàn bộ hướng dẫn trước khi thực hiện. Thứ tự các bước rất quan trọng.

---

## 1. Actual Hardware

Thông tin phần cứng thực tế từ `nvidia-smi`:

```text
GPU:                    NVIDIA GeForce RTX 5060 Ti
VRAM:                   16 GB (16311 MiB)
Driver Version:         581.80
nvidia-smi CUDA Version: 13.0
Kiến trúc GPU:          Blackwell
```

### Giải thích CUDA Version

- Giá trị `CUDA Version: 13.0` hiển thị bởi `nvidia-smi` là **phiên bản CUDA tối đa mà driver hỗ trợ**.
- Đây **KHÔNG** phải phiên bản CUDA Toolkit đã cài trên hệ thống.
- PyTorch CUDA wheels **đóng gói sẵn CUDA runtime riêng** bên trong package → không cần cài CUDA Toolkit system-wide.
- Ta chỉ cần chọn PyTorch wheel có CUDA version **≤ 13.0** (ví dụ: cu126, cu128).
- Driver 581.80 hỗ trợ backward compatibility với các phiên bản CUDA thấp hơn.

---

## 2. Why Not Use Existing Environment

Máy hiện có environment `D:\IntroAI_env`. **KHÔNG khuyến nghị dùng** vì:

- Có thể chứa PyTorch nightly/dev build, không phải stable release.
- Có thể bị mismatch version giữa `torch`, `torchvision`, `torchaudio` (CUDA runtime conflict).
- Môi trường đã được cài nhiều package khác → dễ xung đột dependency.
- Đang dùng Python 3.14 — quá mới, nhiều package chưa hỗ trợ chính thức.
- PyTorch hiện tại trong env là **CPU-only** (`torch.version.cuda = None`, `torch.cuda.is_available() = False`).

**Giải pháp**: Tạo conda environment mới, sạch, với Python 3.11 ổn định.

---

## 3. Create Conda Environment on Drive C

> **Ràng buộc**: Drive D gần đầy → tạo env trên drive C.

```bat
REM Bước 1: Tạo thư mục chứa env trên drive C
mkdir C:\conda_envs

REM Bước 2: Tạo conda environment với Python 3.11
conda create --prefix C:\conda_envs\hrt python=3.11 -y

REM Bước 3: Activate environment
conda activate C:\conda_envs\hrt

REM Bước 4: Kiểm tra
python --version
where python
python -m pip --version
```

Kết quả mong đợi:
- `python --version` → `Python 3.11.x`
- `where python` → `C:\conda_envs\hrt\python.exe`
- `python -m pip --version` → pip version ... (python 3.11)

---

## 4. Install PyTorch Separately

PyTorch **PHẢI** được cài riêng, không qua `requirements.txt`.

### Cách xác định command đúng

1. Truy cập **[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)**
2. Chọn cấu hình:
   - **PyTorch Build**: Stable (ưu tiên stable trước)
   - **Your OS**: Windows
   - **Package**: Pip
   - **Language**: Python
   - **Compute Platform**: CUDA (chọn phiên bản cao nhất có sẵn)
3. Copy command mà trang web đề xuất.

### Khuyến nghị cho RTX 5060 Ti (Blackwell)

Vì đây là GPU thế hệ mới (RTX 50-series, Blackwell architecture):

- **Ưu tiên CUDA 12.8 (`cu128`)** nếu có ở stable release — đây thường là phiên bản tốt nhất cho GPU mới.
- **Nếu stable chỉ có `cu126`**, dùng `cu126` — vẫn hoạt động tốt vì driver hỗ trợ backward compatibility.
- **Chỉ xem xét PyTorch nightly** nếu stable PyTorch không thể detect hoặc sử dụng RTX 5060 Ti (khả năng thấp).

### Command mẫu

```bat
REM Đảm bảo đã activate đúng env
conda activate C:\conda_envs\hrt

REM Ví dụ — thay bằng command chính thức từ pytorch.org nếu khác:
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Nếu `cu128` chưa có ở stable, thử cu126:

```bat
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> **Cảnh báo**: KHÔNG chạy `pip install torch` đơn thuần (không có `--index-url`) — sẽ cài bản CPU-only.

### Nếu cần dùng nightly (chỉ khi stable thất bại)

```bat
REM CHỈ dùng khi stable PyTorch không detect được RTX 5060 Ti
REM torch, torchvision, torchaudio PHẢI cùng từ một nightly CUDA index
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## 5. Install Project Dependencies

Sau khi PyTorch CUDA đã cài thành công:

```bat
REM Activate env và di chuyển đến project
conda activate C:\conda_envs\hrt
cd /d D:\Downloads\HRT-Project

REM Cài dependencies từ requirements.txt (không chứa torch)
python -m pip install -r requirements.txt
```

File `requirements.txt` chỉ chứa:
```
opencv-python
pillow
numpy
pandas
scikit-learn
tqdm
matplotlib
albumentations
editdistance
tensorboard
torchinfo
```

---

## 6. Verify GPU and PyTorch

Chạy lần lượt các command sau để kiểm tra:

### Test 1: nvidia-smi

```bat
nvidia-smi
```

### Test 2: PyTorch CUDA detection

```bat
python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Test 3: CUDA tensor computation

```bat
python -c "import torch; x=torch.randn(1024,1024,device='cuda'); y=x@x; print(y.shape); print('CUDA tensor test passed')"
```

### Test 4: ResNet18 on CUDA

```bat
python -c "import torch, torchvision; print('torchvision:', torchvision.__version__); m=torchvision.models.resnet18(weights=None).cuda(); x=torch.randn(2,3,64,256).cuda(); y=m(x); print('resnet output shape:', y.shape); print('ResNet CUDA test passed')"
```

---

## 7. Expected Result

Nếu mọi thứ hoạt động đúng, kết quả mong đợi:

| Test | Kết quả mong đợi |
|---|---|
| `torch.cuda.is_available()` | `True` |
| `torch.version.cuda` | `12.8` hoặc `12.6` (tùy wheel đã cài) |
| GPU name | `NVIDIA GeForce RTX 5060 Ti` |
| CUDA tensor test | `torch.Size([1024, 1024])` + `CUDA tensor test passed` |
| ResNet CUDA test | `resnet output shape: torch.Size([2, 1000])` + `ResNet CUDA test passed` |

Khi tất cả test passed → **sẵn sàng bắt đầu implement project**.

---

## 8. If It Fails

### `torch.cuda.is_available()` trả về `False`

**KHÔNG tiếp tục implement nếu CUDA chưa hoạt động.** Kiểm tra theo thứ tự:

1. **Kiểm tra active Python**:
   ```bat
   where python
   ```
   Phải trỏ đến `C:\conda_envs\hrt\python.exe`. Nếu không → chưa activate đúng env.

2. **Confirm env đang active**:
   ```bat
   conda activate C:\conda_envs\hrt
   ```

3. **Kiểm tra PyTorch CUDA version**:
   ```bat
   python -c "import torch; print(torch.version.cuda)"
   ```
   Nếu kết quả là `None` → đã cài bản CPU-only. Cần cài lại với đúng `--index-url`.

4. **Cài lại PyTorch** (trong cùng env):
   ```bat
   python -m pip uninstall torch torchvision torchaudio -y
   python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

### Lỗi version mismatch

- **KHÔNG** trộn stable torch với nightly torchvision (hoặc ngược lại).
- Torch, torchvision, torchaudio **PHẢI** cùng từ một index URL và compatible version.
- Nếu dùng nightly → tất cả 3 package phải từ cùng nightly CUDA index.

### Lỗi "CUDA error: no kernel image is available"

- Có thể do PyTorch wheel chưa hỗ trợ kiến trúc Blackwell (sm_120).
- Thử nâng lên PyTorch nightly hoặc phiên bản mới hơn.
- Kiểm tra [PyTorch GitHub Issues](https://github.com/pytorch/pytorch/issues) cho RTX 5060 Ti support.

### Nguyên tắc chung

- **KHÔNG** thử ngẫu nhiên nhiều CUDA indexes trong cùng một environment.
- Nếu cần thử index khác → uninstall torch hoàn toàn trước, rồi cài lại.
- Mỗi lần thử, chỉ thử **một** CUDA index duy nhất.

---

## Tóm Tắt Flow Setup

```bat
REM 1. Tạo env
mkdir C:\conda_envs
conda create --prefix C:\conda_envs\hrt python=3.11 -y
conda activate C:\conda_envs\hrt

REM 2. Cài PyTorch CUDA (thay command nếu cần)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

REM 3. Cài dependencies
cd /d D:\Downloads\HRT-Project
python -m pip install -r requirements.txt

REM 4. Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Nếu bước 4 hiện `True NVIDIA GeForce RTX 5060 Ti` → **DONE** ✅
