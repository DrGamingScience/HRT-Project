# Word-Level Handwritten Text Recognition (HTR)

Dự án xây dựng hệ thống nhận dạng chữ viết tay mức độ từ đơn lẻ (Word-level HTR) chạy local bằng PyTorch. Hệ thống được phát triển và đánh giá trên bộ dữ liệu IAM Handwriting Dataset mức độ word-level, so sánh hai hướng tiếp cận: CTC Baseline (ResNet18 + BiLSTM + CTC Loss) và Attention Model (ResNet18 + Context BiLSTM + Bahdanau Attention + GRU Decoder).

## Phiên bản mô hình (Model Versions)

Dự án trải qua hai giai đoạn phát triển chính:
- **Phase 1 (v1 - Thử nghiệm ban đầu):** Mô hình chỉ hỗ trợ từ vựng gồm 26 chữ cái tiếng Anh viết thường (a-z). Tổng số mẫu sau làm sạch là 96,835. Đây là phiên bản dùng để đánh giá tính khả thi và thiết lập pipeline ban đầu.
- **Phase 2 (v2 - Phiên bản chính thức/Final):** Bộ từ vựng mở rộng lên 76 ký tự bao gồm chữ thường, chữ hoa, chữ số và các ký tự đặc biệt (`abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#&()*+,-./:;?`). Dữ liệu giữ nguyên định dạng viết tay thực tế của IAM với 111,706 mẫu. Các checkpoint huấn luyện và lệnh chạy dưới đây mặc định áp dụng cho phiên bản v2 này.

### Bảng kết quả trên tập kiểm thử (Test Set)

| Phiên bản | Mô hình | Giải mã | Word Accuracy | CER | NED |
|---|---|---|---|---|---|
| **Phase 2 (v2 - Final)** | CTC Baseline | CTC Greedy | 79.95% | 8.25% | 93.13% |
| | Attention Model | Greedy Decode | 82.67% | 7.79% | 93.32% |
| | Attention Model | Beam Search (K=2) | 82.84% | 7.74% | 93.35% |
| *Phase 1 (v1)* | CTC Baseline | CTC Greedy | 80.75% | 7.27% | 94.16% |
| | Attention Model | Greedy Decode | 83.09% | 7.06% | 94.21% |

*Lưu ý: Kết quả của Phase 1 cao hơn một chút do không gian từ vựng nhỏ hơn (26 ký tự so với 76 ký tự của Phase 2).*

---

## Cấu trúc thư mục

```text
HRT-Project/
├── src/                            # Mã nguồn chính
│   ├── data/                       # Dataloader, transforms và vocab
│   ├── models/                     # Kiến trúc mô hình (ResNet, Attention, GRU)
│   ├── inference/                  # Thuật toán giải mã (Greedy, Beam Search)
│   ├── utils/                      # Tiện ích tính metric, save/load checkpoint, visualization
│   ├── ctc_baseline/               # Pipeline huấn luyện & suy diễn cho CTC
│   ├── attention_bilstm/           # Pipeline huấn luyện & suy diễn cho Attention
│   └── prepare_data.py            # Tiền xử lý & chia dữ liệu
├── checkpoints/                    # Thư mục lưu trọng số mô hình (.pth)
│   ├── ctc_baseline_v2/            # Checkpoint v2 cho CTC Baseline
│   └── attention_bilstm_v2/        # Checkpoint v2 cho Attention Model
├── dataset/                        # Thư mục chứa ảnh và nhãn gốc (tải riêng)
├── requirements.txt                # Thư viện phụ thuộc
└── README.md                       # File hướng dẫn này
```

---

## Hướng dẫn cài đặt và chạy nhanh

### 1. Môi trường và thư viện
Kích hoạt môi trường Python 3.11 của bạn và cài đặt các thư viện cần thiết:
```bash
python -m pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu và checkpoint
1. Tải bộ dữ liệu IAM (mức word-level) gồm thư mục `words/` và file `label.txt`. Đặt chúng vào thư mục `dataset/`.
2. Tải các checkpoint v2 và đặt vào cấu trúc sau:
   - `checkpoints/ctc_baseline_v2/best_ctc_baseline.pth`
   - `checkpoints/attention_bilstm_v2/best_attention_model.pth`

Chạy script để làm sạch nhãn, lọc ảnh lỗi và chia tập train/val/test (tỉ lệ 80/10/10):
```bash
python -m src.prepare_data
```

*Lưu ý: Mặc định script sẽ chạy theo cấu hình Phase 2 (giữ nguyên chữ hoa, chữ số và dấu câu). Để chuyển đổi cấu hình hoặc huấn luyện lại phiên bản Phase 1, xem mục Hướng dẫn chuyển cấu hình ở cuối.*

### 3. Đánh giá trên tập kiểm thử (Test Set)

Đánh giá mô hình v2 bằng các lệnh sau:

```bash
# Đánh giá CTC Baseline v2
python -m src.ctc_baseline.evaluate --checkpoint checkpoints/ctc_baseline_v2/best_ctc_baseline.pth

# Đánh giá Attention Model v2 (Greedy Decode)
python -m src.attention_bilstm.evaluate --checkpoint checkpoints/attention_bilstm_v2/best_attention_model.pth

# Đánh giá Attention Model v2 (Beam Search)
python -m src.attention_bilstm.evaluate --checkpoint checkpoints/attention_bilstm_v2/best_attention_model.pth --beam
```

### 4. Suy diễn trên ảnh thực tế
Nhận dạng một ảnh hoặc cả thư mục ảnh (tên file ảnh dùng làm nhãn so sánh nếu có):

```bash
# Sử dụng CTC Baseline v2
python -m src.ctc_baseline.predict --image path/to/image.png --checkpoint checkpoints/ctc_baseline_v2/best_ctc_baseline.pth

# Sử dụng Attention Model v2 (sinh attention map)
python -m src.attention_bilstm.predict --image path/to/image.png --checkpoint checkpoints/attention_bilstm_v2/best_attention_model.pth --save_attention

# Sử dụng Attention Model v2 với giải mã Beam Search
python -m src.attention_bilstm.predict --image path/to/image.png --checkpoint checkpoints/attention_bilstm_v2/best_attention_model.pth --beam
```

---

## Chi tiết các bước huấn luyện (Training)

Nếu bạn muốn huấn luyện lại các mô hình từ đầu:

### CTC Baseline
```bash
python -m src.ctc_baseline.train
```
Trọng số tốt nhất được lưu tại `checkpoints/ctc_baseline_v2/best_ctc_baseline.pth`. Quá trình train sẽ freeze ResNet18 trong 5 epoch đầu và unfreeze để fine-tune ở các epoch tiếp theo.

### Attention Model
```bash
python -m src.attention_bilstm.train
```
Trọng số tốt nhất được lưu tại `checkpoints/attention_bilstm_v2/best_attention_model.pth`. Mô hình sử dụng cơ chế giảm dần tỉ lệ Teacher Forcing từ 0.5 xuống 0.01 để hạn chế Exposure Bias.

---

## Hướng dẫn chuyển đổi cấu hình (Phase 1 vs Phase 2)

Hệ thống hỗ trợ chuyển đổi linh hoạt thông qua cấu hình mã nguồn. Nếu muốn chuyển từ mặc định (Phase 2) về Phase 1 (chữ cái thường):

1. Trong file `src/config.py`: Đặt `VOCAB_CHARS = "abcdefghijklmnopqrstuvwxyz"`.
2. Trong file `src/prepare_data.py`: Bỏ comment dòng code `clean_label = clean_label.lower()` để chuyển toàn bộ nhãn về chữ thường.
3. Chạy lại lệnh xử lý dữ liệu: `python -m src.prepare_data`.
4. Khi chạy các script train/evaluate/predict, chỉ định checkpoint tương ứng của Phase 1 (đặt trong `checkpoints/ctc_baseline/` và `checkpoints/attention_bilstm/`).

---

## Phân công nhiệm vụ thành viên

| STT | Thành viên | Vai trò và nhiệm vụ chính | Đóng góp |
| :---: | :--- | :--- | :---: |
| 1 | **Nguyễn Trí Hiếu** | - Chuẩn bị và làm sạch dữ liệu (`prepare_data.py`).<br>- Thiết kế, triển khai mô hình chính **Attention Model** (Bahdanau Attention + GRU Decoder).<br>- Xây dựng pipeline huấn luyện, tinh chỉnh hyperparameter và Teacher Forcing decay.<br>- Biên soạn nội dung báo cáo kỹ thuật và chuyển đổi sang LaTeX. | 100% |
| 2 | **Đỗ Hải Đăng** | - Triển khai cấu trúc Encoder-Decoder dùng chung và trích xuất đặc trưng với ResNet18.<br>- Triển khai mô hình đối chứng **CTC Baseline** (ResNet18 + BiLSTM + CTC Loss).<br>- Xây dựng pipeline huấn luyện và đánh giá cho CTC Baseline.<br>- Thiết kế cấu trúc slide thuyết trình. | 100% |
| 3 | **Đinh Thái Sơn** | - Triển khai các thuật toán giải mã tại thư mục `inference/` (CTC Greedy, Attention Greedy, Beam Search).<br>- Viết hàm tính toán chỉ số đánh giá (Word Accuracy, CER, NED).<br>- Phát triển công cụ trực quan hóa Attention Map và các script dự đoán ảnh thực tế (`predict.py`). | 100% |

