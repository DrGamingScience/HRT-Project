# Word-Level Handwritten Text Recognition (HTR)

Hệ thống nhận dạng chữ viết tay mức độ từ đơn lẻ sử dụng PyTorch chạy local. Mô hình được xây dựng theo kiến trúc **ResNet18 + BiLSTM + Bahdanau Attention + GRU Decoder**, đạt **83.09% Word Accuracy** và **7.06% CER** trên tập kiểm thử IAM.

---

## 📁 Cấu trúc thư mục dự án

```text
HRT-Project/
├── src/                            # Mã nguồn chính của dự án
│   ├── data/                       # Dataloader, transforms và vocab
│   ├── models/                     # Kiến trúc mô hình Attention
│   ├── inference/                  # Các giải thuật giải mã (Greedy, Beam Search)
│   └── utils/                      # Tiện ích (metrics, checkpoint, seed, visualization)
├── report/                         # Báo cáo kỹ thuật chi tiết phục vụ viết báo cáo/LaTeX
│   └── project_report.md           # [BÁO CÁO TỔNG HỢP DUY NHẤT]
├── powerpoint/                     # Tài liệu phục vụ thiết kế slide thuyết trình
│   └── slides_outline.md           # Dàn ý chi tiết 10 slide thuyết trình & lời thoại gợi ý
├── checkpoints/                    # Thư mục chứa trọng số mô hình tốt nhất (.pth)
│   └── attention_bilstm/
│       └── best_attention_model.pth
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

### 2. Tải checkpoint mô hình
Do file checkpoint nặng, vui lòng tải trọng số đã train sẵn về và đặt đúng cấu trúc thư mục ở trên:
*   [Tải best_attention_model.pth (Google Drive)](#) *(Chèn link của bạn tại đây)*

### 3. Chuẩn bị dữ liệu
Đặt bộ dữ liệu IAM vào thư mục `dataset/` (gồm file `label.txt` và thư mục `words/`). Sau đó chạy lệnh chia tập dữ liệu:
```bash
python -m src.prepare_data
```

### 4. Đánh giá mô hình trên tập kiểm thử (Test Set)
```bash
python -m src.evaluate_attention
```

### 5. Chạy dự đoán thực tế
```bash
# Nhận dạng một ảnh đơn lẻ và trực quan hóa bản đồ chú ý (lưu tại outputs/):
python -m src.predict_attention --image duong_dan_anh.png --save_attention

# Nhận dạng toàn bộ ảnh trong một thư mục (tên file = nhãn chuẩn):
python -m src.predict_attention --image duong_dan_thu_muc/
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

### B. Huấn luyện lại mô hình từ đầu (Training From Scratch)
Nếu bạn muốn tự train lại thay vì dùng checkpoint có sẵn:
```bash
python -m src.train_attention
```
*Huấn luyện tự động qua 2 phase: Đóng băng Encoder 5 epoch đầu; mở khóa từ epoch 5 để fine-tune. Checkpoint tốt nhất tự động lưu tại `checkpoints/attention_bilstm/best_attention_model.pth`.*

### C. Đánh giá mô hình trên tập kiểm thử (`src.evaluate_attention`)
Đánh giá độ chính xác (Word Accuracy, CER, NED) trên 9,684 mẫu test:
```bash
python -m src.evaluate_attention --checkpoint checkpoints/attention_bilstm/best_attention_model.pth
```
*(Dự đoán chi tiết từng ảnh được lưu ra file `outputs/predictions.csv`)*

### D. Nhận dạng ảnh thực tế (`src.predict_attention`)
*   **Ảnh đơn lẻ (Greedy Decode & Lưu Attention Map):**
    ```bash
    python -m src.predict_attention --image duong_dan_anh.png --save_attention
    ```
    *(Ảnh trực quan hóa attention heatmap được lưu tại `outputs/attention_maps/`)*
*   **Ảnh đơn lẻ (Beam Search nâng cao):**
    ```bash
    python -m src.predict_attention --image duong_dan_anh.png --beam
    ```
*   **Toàn bộ thư mục ảnh (tên file = nhãn chuẩn):**
    ```bash
    python -m src.predict_attention --image duong_dan_thu_muc/
    ```

</details>

---

## 📄 Tài liệu chi tiết nộp bài
*   **Dành cho viết báo cáo (Word/LaTeX):** [report/project_report.md](file:///d:/Downloads/HRT-Project/report/project_report.md)
*   **Dành cho thiết kế slide thuyết trình:** [powerpoint/slides_outline.md](file:///d:/Downloads/HRT-Project/powerpoint/slides_outline.md)
