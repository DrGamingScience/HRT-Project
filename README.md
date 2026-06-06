# Word-Level Handwritten Text Recognition (HTR)

Hệ thống nhận dạng chữ viết tay mức độ từ đơn lẻ sử dụng PyTorch. Dự án so sánh giữa mô hình chính (ResNet18 + Bahdanau Attention + GRU) và mô hình đối chứng (CNN + BiLSTM + CTC).

## Cấu trúc dự án
- `dataset/`: Chứa dữ liệu ảnh `words/` và file nhãn gốc `label.txt`.
- `src/data/`: Module từ vựng (`vocab.py`), transform (`transforms.py`) và dataset (`dataset.py`).
- `src/models/`: Định nghĩa mô hình `AttentionHTR` và `CTCBaseline`.
- `src/inference/`: Giải mã greedy, beam search và giải mã ctc.
- `src/utils/`: Lưu các hàm tiện ích về seed, checkpoint, metrics và vẽ attention map.

## Hướng dẫn sử dụng

Trước khi chạy, hãy đảm bảo đã kích hoạt môi trường Conda phù hợp:
```bash
conda activate C:\conda_envs\hrt
```

### 1. Chuẩn bị dữ liệu
Làm sạch nhãn tiếng Anh, lọc ký tự, kiểm tra ảnh tồn tại và chia tập dữ liệu (80/10/10):
```bash
python -m src.prepare_data
```

### 2. Mô hình Attention (Mô hình chính)
- **Huấn luyện** (Tự động chạy 2 phase: đóng băng và fine-tune encoder):
  ```bash
  python -m src.train_attention
  ```
- **Đánh giá trên tập test** (Xuất kết quả ra `outputs/predictions.csv`):
  ```bash
  python -m src.evaluate_attention
  ```
- **Nhận dạng ảnh đơn lẻ & Trực quan hóa bản đồ Attention** (Xuất ảnh overlay vào `outputs/attention_maps/`):
  ```bash
  python -m src.predict_attention --image đường_dẫn_ảnh.png --save_attention
  ```

### 3. Mô hình CTC (Mô hình đối chứng)
- **Huấn luyện**:
  ```bash
  python -m src.train_ctc_baseline
  ```
- **Đánh giá trên tập test** (Xuất kết quả ra `outputs/predictions_ctc.csv`):
  ```bash
  python -m src.evaluate_ctc_baseline
  ```

## Cấu hình hệ thống
Toàn bộ siêu tham số (Epochs, Batch size, LR, Hidden dim) và đường dẫn thư mục được tập trung cấu hình tại [src/config.py](file:///d:/Downloads/HRT-Project/src/config.py).
