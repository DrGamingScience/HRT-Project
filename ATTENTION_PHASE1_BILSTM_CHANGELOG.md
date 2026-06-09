# NHẬT KÝ THAY ĐỔI: ATTENTION PHASE 1 - TÍCH HỢP BiLSTM (CHANGELOG)

Tệp này ghi nhận các thay đổi mã nguồn trong đợt cập nhật Phase 1, chuyển đổi cấu trúc của mô hình Attention từ dạng truyền thống (Raw CNN Features) sang cấu hình có thêm lớp BiLSTM ngữ cảnh trước khi tính Attention.

---

## 1. Các file đã thay đổi (Files changed)
*   [src/config.py](file:///D:/Downloads/HRT-Project/src/config.py): Bổ sung các hằng số cấu hình lớp BiLSTM (`ATTENTION_LSTM_...`).
*   [src/models/attention_model.py](file:///D:/Downloads/HRT-Project/src/models/attention_model.py): Khởi tạo lớp BiLSTM `context_lstm`, chuyển đổi kích thước đầu vào decoder từ 1024 xuống 512, định nghĩa hàm `extract_features(images, return_debug=True)` và cập nhật `forward`.
*   [src/train_attention.py](file:///D:/Downloads/HRT-Project/src/train_attention.py): Chia tham số an toàn trong optimizer (`encoder_params` và `decoder_params`) và chuyển thư mục lưu checkpoint sang `checkpoints/attention_bilstm/`.
*   [src/inference/greedy_decode.py](file:///D:/Downloads/HRT-Project/src/inference/greedy_decode.py): Chuyển đổi phần trích xuất thủ công sang sử dụng `model.extract_features()`.
*   [src/inference/beam_search.py](file:///D:/Downloads/HRT-Project/src/inference/beam_search.py): Chuyển đổi phần trích xuất thủ công sang sử dụng `model.extract_features()`.
*   [src/evaluate_attention.py](file:///D:/Downloads/HRT-Project/src/evaluate_attention.py): Đổi đường dẫn mặc định của checkpoint sang `checkpoints/attention_bilstm/best_attention_model.pth`.
*   [src/predict_attention.py](file:///D:/Downloads/HRT-Project/src/predict_attention.py): Đổi đường dẫn mặc định của checkpoint sang `checkpoints/attention_bilstm/best_attention_model.pth`.

---

## 2. Lý do sửa đổi
Cơ chế Attention cũ lấy trực tiếp đầu ra từ CNN của ResNet18 (layer3). Điều này làm cho mô hình Attention chỉ nhận được đặc trưng cục bộ (local features) của từng cột dọc mà thiếu đi ngữ cảnh xung quanh, khiến việc hội tụ khó khăn. Việc chèn 2 lớp BiLSTM trước Attention (tương tự như mô hình CTC Baseline) giúp đặc trưng chuỗi có ngữ cảnh toàn cục hai chiều trước khi bước giải mã chú ý diễn ra, tạo điều kiện so sánh công bằng hiệu năng với CTC Baseline.

---

## 3. Quy trình Attention cũ (Attention Flow)
```text
Input image:              (B, 3, 64, 512)
ResNet18 layer3 output:   (B, 256, 4, 32)
Reshape to sequence:      (B, 32, 1024)
Attention input:          (B, 32, 1024)
```

---

## 4. Quy trình Attention mới (Attention Flow)
```text
Input image:                    (B, 3, 64, 512)
ResNet18 layer3 output:         (B, 256, 4, 32)
Reshape to raw sequence:        (B, 32, 1024)
2-layer BiLSTM output:          (B, 32, 512)
Bahdanau Attention input:       (B, 32, 512)
GRU Decoder output logits:      (B, target_len, vocab_size)
Attention weights:              (B, target_len, 32)
```

---

## 5. Các kích thước hình dạng quan trọng (Important shapes)
*   **Images:** `(B, 3, 64, 512)`
*   **ResNet features:** `(B, 256, 4, 32)`
*   **Raw sequence:** `(B, 32, 1024)`
*   **BiLSTM contextual sequence:** `(B, 32, 512)`
*   **Output logits:** `(B, target_len, 29)`
*   **Attention weights:** `(B, target_len, 32)`

---

## 6. Tính tương thích của Checkpoint cũ
*   **KHÔNG TƯƠNG THÍCH.** Mô hình mới có cấu trúc khác hoàn toàn (thêm lớp BiLSTM và thay đổi số chiều đầu vào của Decoder từ 1024 thành 512). 
*   **Giải pháp:** Đã chuyển hướng lưu trữ checkpoint mới sang thư mục riêng `checkpoints/attention_bilstm/` để không ghi đè hoặc làm hỏng checkpoint cũ. Quá trình huấn luyện mới sẽ bắt đầu lại từ đầu (start fresh).

---

## 7. Script kiểm tra kích thước đã tạo
Đã tạo script gỡ lỗi và kiểm thử kích thước gọn nhẹ tại:
*   [src/debug_attention_bilstm_shapes.py](file:///D:/Downloads/HRT-Project/src/debug_attention_bilstm_shapes.py)

---

## 8. Lệnh bạn cần tự chạy để kiểm tra kích thước
*   Nếu terminal của bạn đã kích hoạt sẵn môi trường:
    ```bash
    python -m src.debug_attention_bilstm_shapes
    ```
*   Nếu terminal chưa kích hoạt môi trường:
    ```bash
    C:\conda_envs\hrt\python.exe -m src.debug_attention_bilstm_shapes
    ```

---

## 9. Lệnh huấn luyện mô hình Attention mới (sau khi kiểm tra shape thành công)
*   Để chạy huấn luyện mô hình Attention mới có BiLSTM:
    ```bash
    python -m src.train_attention
    ```
*   *(Lệnh này sẽ tự động lưu các checkpoints mới tại thư mục `checkpoints/attention_bilstm/`)*

---

## 10. Rủi ro còn lại
Do chưa có mặt nạ đệm ảnh (attention padding mask), mô hình Attention vẫn có thể hướng sự chú ý vào các vùng đệm màu trắng ở các bước cuối chuỗi nếu ảnh từ có độ dài viết tay ngắn.

---

## 11. Việc chưa làm ở Phase này (Out of scope)
*   Chưa thêm padding mask.
*   Chưa chỉnh sửa lịch trình teacher forcing.
*   Chưa tune lại learning rate.
*   Chưa chạy train lại thực tế.
*   Chưa sửa đổi bất kỳ tệp tin nào của pipeline CTC baseline.
