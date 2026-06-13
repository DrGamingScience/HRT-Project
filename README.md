# Word-Level Handwritten Text Recognition (HTR)

Hệ thống nhận dạng chữ viết tay mức độ từ đơn lẻ sử dụng PyTorch chạy local. Dự án thực hiện so sánh chi tiết hiệu năng giữa mô hình đề xuất cải tiến (**ResNet18 + BiLSTM + Bahdanau Attention + GRU Decoder**) và mô hình đối chứng (**ResNet18 + BiLSTM + CTC Baseline**).

---

## 📁 Cấu trúc thư mục dự án

```text
HRT-Project/
├── src/                            # Mã nguồn chính của dự án
│   ├── data/                       # Dataloader, transforms và vocab
│   ├── models/                     # Các mô hình Attention và CTC Baseline
│   ├── inference/                  # Các giải thuật giải mã (Greedy, Beam Search)
│   └── utils/                      # Tiện ích (metrics, checkpoint, seed, visualization)
├── report/                         # Báo cáo kỹ thuật chi tiết phục vụ viết báo cáo/LaTeX
│   └── project_report.md           # [BÁO CÁO TỔNG HỢP DUY NHẤT]
├── powerpoint/                     # Tài liệu phục vụ thiết kế slide thuyết trình
│   └── slides_outline.md           # Dàn ý chi tiết 10 slide thuyết trình & lời thoại gợi ý
├── checkpoints/                    # Thư mục chứa trọng số mô hình tốt nhất (.pth)
│   ├── attention_bilstm/
│   │   └── best_attention_model.pth
│   └── best_ctc_baseline.pth
├── dataset/                        # Dữ liệu ảnh words/ và nhãn label.txt (Tải riêng)
├── requirements.txt                # Danh sách thư viện phụ thuộc
└── README.md                       # Hướng dẫn nhanh này
```

---

## ⚡ Hướng dẫn chạy nhanh (Quickstart)

### 1. Cài đặt môi trường
Xem chi tiết hướng dẫn cài đặt Python 3.11, PyTorch CUDA tương thích với card đồ họa tại [report/project_report.md (Mục 3)](file:///d:/Downloads/HRT-Project/report/project_report.md).
Sau khi kích hoạt môi trường:
```bash
python -m pip install -r requirements.txt
```

### 2. Tải checkpoints mô hình
Do file checkpoints nặng, vui lòng tải trọng số đã train sẵn về và đặt đúng cấu trúc thư mục ở trên:
*   [Tải best_ctc_baseline.pth (Google Drive)](#) *(Chèn link của bạn tại đây)*
*   [Tải best_attention_model.pth (Google Drive)](#) *(Chèn link của bạn tại đây)*

### 3. Chuẩn bị dữ liệu
Đặt bộ dữ liệu IAM vào thư mục `dataset/` (gồm file `label.txt` và thư mục `words/`). Sau đó chạy lệnh chia tập dữ liệu:
```bash
python -m src.prepare_data
```

### 4. Đánh giá mô hình trên tập kiểm thử (Test Set)
```bash
# Đánh giá mô hình CTC Baseline
python -m src.evaluate_ctc_baseline

# Đánh giá mô hình Attention
python -m src.evaluate_attention
```

### 5. Chạy dự đoán thực tế trên một ảnh bất kỳ
```bash
# Nhận dạng ảnh đơn lẻ bằng Attention và trực quan hóa bản đồ chú ý (lưu tại outputs/):
python -m src.predict_attention --image duong_dan_anh.png --save_attention

# Nhận dạng bằng mô hình CTC Baseline:
python -m src.predict_ctc --image duong_dan_anh.png
```

---

## 📖 Hướng dẫn chạy chi tiết từng Tiến trình

<details>
<summary><b>Xem chi tiết các câu lệnh và tham số nâng cao (Click để mở rộng)</b></summary>

### A. Tiền xử lý dữ liệu (`src.prepare_data`)
*   **Chức năng:** Làm sạch nhãn (lowercase, bỏ ký tự đặc biệt), lọc ảnh lỗi, chia dữ liệu thành 3 tập `train.csv`, `val.csv`, `test.csv` lưu trong `data_processed/`.
*   **Lệnh chạy:**
    ```bash
    python -m src.prepare_data
    ```

### B. Huấn luyện mô hình từ đầu (Training From Scratch)
Nếu bạn muốn tự train lại mô hình thay vì dùng checkpoint có sẵn:
*   **CTC Baseline:** 
    ```bash
    python -m src.train_ctc_baseline
    ```
    *Mô hình tốt nhất tự động lưu tại `checkpoints/best_ctc_baseline.pth`.*
*   **Attention Model:**
    ```bash
    python -m src.train_attention
    ```
    *Huấn luyện tự động qua 2 phase: Đóng băng Encoder 5 epoch đầu; mở khóa từ epoch 5 để fine-tune. Checkpoint tốt nhất tự động lưu tại `checkpoints/attention_bilstm/best_attention_model.pth`.*

### C. Đánh giá mô hình trên tập kiểm thử (`src.evaluate_...`)
Đánh giá độ chính xác (Word Accuracy, CER, NED) trên 9,684 mẫu test:
*   **CTC Baseline:**
    ```bash
    python -m src.evaluate_ctc_baseline --checkpoint checkpoints/best_ctc_baseline.pth
    ```
    *(Dự đoán chi tiết từng ảnh được lưu ra file `outputs/predictions_ctc.csv`)*
*   **Attention Model:**
    ```bash
    python -m src.evaluate_attention --checkpoint checkpoints/attention_bilstm/best_attention_model.pth
    ```
    *(Dự đoán chi tiết từng ảnh được lưu ra file `outputs/predictions.csv`)*

### D. Nhận dạng ảnh thực tế (`src.predict_...`)
*   **CTC Baseline:**
    ```bash
    python -m src.predict_ctc --image duong_dan_anh.png
    ```
*   **Attention Model (Greedy Decode & Lưu Attention Map):**
    ```bash
    python -m src.predict_attention --image duong_dan_anh.png --save_attention
    ```
    *(Ảnh trực quan hóa attention heatmap được lưu tại `outputs/attention_maps/`)*
*   **Attention Model (Giải mã Beam Search nâng cao):**
    ```bash
    python -m src.predict_attention --image duong_dan_anh.png --beam
    ```

</details>

---

## 📄 Tài liệu chi tiết nộp bài
*   **Dành cho viết báo cáo (Word/LaTeX):** [report/project_report.md](file:///d:/Downloads/HRT-Project/report/project_report.md)
*   **Dành cho thiết kế slide thuyết trình:** [powerpoint/slides_outline.md](file:///d:/Downloads/HRT-Project/powerpoint/slides_outline.md)

