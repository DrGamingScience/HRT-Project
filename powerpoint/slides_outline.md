# DÀN Ý & NỘI DUNG SLIDE THUYẾT TRÌNH (SLIDES OUTLINE)
## ĐỒ ÁN: NHẬN DẠNG CHỮ VIẾT TAY MỨC ĐỘ TỪ ĐƠN

Dưới đây là thiết kế chi tiết gồm **10 slide** thuyết trình chuẩn học thuật. Bạn chỉ cần copy nội dung vào PowerPoint và xem phần "Lời thoại gợi ý" để tập thuyết trình trước hội đồng.

---

### SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
*   **Tiêu đề chính:** Hệ Thống Nhận Dạng Chữ Viết Tay Mức Độ Từ Đơn Sử Dụng Học Sâu (Word-Level Handwritten Text Recognition)
*   **Người thực hiện:** [Tên của bạn]
*   **Giáo viên hướng dẫn:** [Tên Thầy/Cô]
*   **Công nghệ:** PyTorch | Python | ResNet18 | Attention | CTC
*   **Lời thoại gợi ý:** *"Kính chào thầy cô trong hội đồng. Hôm nay em xin phép trình bày báo cáo đồ án nghiên cứu về đề tài 'Nhận dạng chữ viết tay mức độ từ đơn lẻ sử dụng mạng tích chập kết hợp cơ chế Attention và đối chứng với mô hình CTC Baseline'."*

---

### SLIDE 2: ĐẶT VẤN ĐỀ & MỤC TIÊU ĐỀ TÀI (PROBLEM & GOAL)
*   **Đặt vấn đề:**
    *   Chữ viết tay có độ biến dạng cao, nét chữ dính nhau, nghiêng, mờ.
    *   Hệ thống OCR truyền thống (nhận dạng ký tự quang học) hoạt động kém với chữ viết tay tự nhiên.
*   **Mục tiêu nghiên cứu:**
    *   Xây dựng hệ thống local tự động nhận dạng ảnh chứa duy nhất một từ (word-level).
    *   Không sử dụng API bên thứ ba để đảm bảo tính độc lập và bảo mật.
    *   Nghiên cứu sâu và so sánh giữa hai hướng tiếp cận chính: **Attention** (Mô hình chính) và **CTC** (Mô hình đối chứng).
*   **Lời thoại gợi ý:** *"Nhận dạng chữ viết tay là một bài toán khó do phong cách viết của mỗi người là khác nhau, nét chữ có thể nghiêng, mờ hoặc dính nét. Mục tiêu của đồ án này là xây dựng một hệ thống chạy local nhận dạng ảnh từ đơn lẻ từ bộ dữ liệu chuẩn IAM và so sánh hai giải pháp kinh điển là Attention và CTC."*

---

### SLIDE 3: BỘ DỮ LIỆU & TIỀN XỬ LÝ (DATASET & PREPROCESSING)
*   **Bộ dữ liệu:**
    *   IAM Handwriting Dataset (English).
    *   Sau khi tiền xử lý và lọc ký tự lạ: **96,835 mẫu ảnh sạch**.
    *   Phân chia tập dữ liệu: **80% Train | 10% Validation | 10% Test**.
*   **Tiền xử lý & Tăng cường:**
    *   Chuyển sang ảnh xám (Grayscale), resize giữ tỉ lệ về chiều cao $64$ px và pad trắng về chiều rộng $512$ px.
    *   Tăng cường dữ liệu ngẫu nhiên (Augmentation): Xoay ($\pm 15^\circ$), biến dạng nghiêng chữ (Shear), nhiễu ảnh (Noise), làm mờ (Blur) và **biến dạng đàn hồi (Elastic Transform)** để mô phỏng nét viết tay tự nhiên.
*   **Lời thoại gợi ý:** *"Về dữ liệu, em sử dụng bộ dữ liệu IAM nổi tiếng với hơn 96 nghìn mẫu sạch sau khi lọc. Tất cả các ảnh được chuẩn hóa về kích thước 64x512 và được áp dụng tăng cường dữ liệu mạnh mẽ, đặc biệt là phép biến dạng đàn hồi Elastic Transform để giúp mô hình học được các biến thể nét chữ viết tay tốt hơn."*

---

### SLIDE 4: MÔ HÌNH ĐỐI CHỨNG: CTC BASELINE (CTC PIPELINE)
*   **Sơ đồ kiến trúc:**
    `Ảnh (64x512) -> CNN (ResNet18 Layer3) -> Chuỗi đặc trưng (32x1024) -> 2-layer BiLSTM -> Phân loại Tuyến tính -> CTCLoss`
*   **Đặc điểm:**
    *   **CNN Encoder:** Trích xuất đặc trưng không gian.
    *   **2-layer BiLSTM:** Tích hợp ngữ cảnh hai chiều dọc theo chiều rộng ảnh để hiểu sự liên kết giữa các ký tự.
    *   **CTC Loss:** Giúp huấn luyện mô hình mà không cần gán nhãn từng vị trí cụ thể (non-alignment), mô hình tự căn chỉnh ký tự.
*   **Lời thoại gợi ý:** *"Mô hình đối chứng đầu tiên là CTC Baseline. Kiến trúc gồm ResNet18 trích xuất đặc trưng, đi qua 2 lớp BiLSTM để học ngữ cảnh chuỗi ký tự theo 2 chiều trái-phải và phải-trái, và tính hàm mất mát CTC Loss. Ưu điểm của CTC là huấn luyện rất nhanh và không cần dữ liệu căn chỉnh vị trí ký tự."*

---

### SLIDE 5: MÔ HÌNH ĐỀ XUẤT: ATTENTION MODEL (ATTENTION PIPELINE)
*   **Sơ đồ kiến trúc:**
    `Ảnh -> CNN Encoder -> Lớp ngữ cảnh BiLSTM -> Cơ chế Bahdanau Attention -> GRU Decoder -> Cross Entropy Loss`
*   **Đặc điểm:**
    *   **Bahdanau Attention (Additive):** Tính toán phân phối trọng số chú ý để tại mỗi bước giải mã, mô hình biết cần "tập trung nhìn" vào vùng đặc trưng nào trên ảnh.
    *   **GRU Decoder:** Giải mã tuần tự tự hồi quy (Autoregressive), sinh ra từng ký tự một bằng cách kết hợp đặc trưng ảnh và ký tự vừa sinh ra trước đó.
*   **Lời thoại gợi ý:** *"Mô hình chính đề xuất là Attention Model. Điểm khác biệt lớn là thay vì phân loại song song như CTC, mô hình này sử dụng cơ chế Attention để tạo vector ngữ cảnh động tại mỗi bước, và dùng Decoder GRU để giải mã tuần tự từng ký tự một, giúp tận dụng mối quan hệ giữa các ký tự đứng trước và đứng sau."*

---

### SLIDE 6: CÁC CẢI TIẾN QUAN TRỌNG TRONG PHASE 1 (KEY IMPROVEMENTS)
*   **1. Tích hợp lớp ngữ cảnh BiLSTM vào Encoder:**
    *   *Trước cải tiến:* Gửi trực tiếp đặc trưng cục bộ của CNN tới Attention Decoder $\rightarrow$ Mô hình khó hội tụ, Attention bị phân mảnh.
    *   *Sau cải tiến:* Thêm 2 lớp BiLSTM làm khối ngữ cảnh bổ trợ giúp làm mịn và tích hợp thông tin chuỗi trước khi tính toán Attention.
*   **2. Tối ưu hóa Learning Rate của Encoder:**
    *   Cấu hình tốc độ học của Encoder nhỏ hơn Decoder 10-20 lần và thực hiện mở đóng băng (unfreeze) từ epoch 5 để fine-tune nhẹ nhàng, tránh phá hỏng trọng số ImageNet tốt có sẵn.
*   **Lời thoại gợi ý:** *"Trong quá trình thực nghiệm ban đầu, mô hình Attention thuần túy rất khó hội tụ do đặc trưng CNN cục bộ thiếu thông tin ngữ cảnh. Trong Phase 1, em đã cải tiến bằng cách chèn thêm 2 lớp BiLSTM làm bộ ngữ cảnh cho Encoder và tối ưu hóa tốc độ học từng phần, giúp mô hình Attention đạt được sự ổn định cao."*

---

### SLIDE 7: KẾT QUẢ THỰC NGHIỆM (EXPERIMENTAL RESULTS)
*   **Bảng so sánh hiệu năng trên tập Test Set độc lập:**

| Mô hình (Model) | Độ chính xác từ (WA) | Tỉ lệ lỗi ký tự (CER) | Khoảng cách NED | Trọng lượng file |
|---|---|---|---|---|
| **CTC Baseline** | **80.89%** | **7.25%** | **94.26%** | ~78.6 MB |
| **Attention Model (Phase 1)** | **83.09%** | **7.06%** | **94.21%** | ~91.0 MB |

*   **Nhận xét:**
    *   Cả hai mô hình đều đạt hiệu năng xuất sắc trên tập dữ liệu kiểm thử (WA > 80%).
    *   Mô hình Attention cải tiến (Phase 1) có tích hợp lớp BiLSTM đã **vượt qua** mô hình đối chứng CTC Baseline về Word Accuracy (83.09%) và CER (7.06%).
*   **Lời thoại gợi ý:** *"Đây là kết quả đánh giá thực tế thu được trên tập kiểm thử Test Set độc lập của hai mô hình. Mô hình CTC Baseline đạt kết quả tốt với 80.89% độ chính xác từ và CER là 7.25%. Đặc biệt, mô hình chính Attention cải tiến có thêm 2 lớp BiLSTM đã xuất sắc đạt độ chính xác từ lên tới 83.09% và CER giảm xuống còn 7.06%, chính thức vượt trội hơn mô hình đối chứng. Chỉ số NED của cả hai mô hình đều đạt rất cao trên 94%."*

---

### SLIDE 8: LẬP LUẬN & PHÂN TÍCH CHUYÊN SÂU (ANALYSIS & DISCUSSION)
*   **Tại sao CTC Baseline tốt hơn Attention?**
    1.  **Hạn chế Exposure Bias:** Decoder GRU của Attention dễ tích lũy sai số từ các ký tự đầu tiên khi giải mã tự hồi quy. CTC giải mã song song độc lập nên không bị lỗi dây chuyền này.
    2.  **Vấn đề trôi Attention (Attention Drift):** Ảnh đệm khoảng trắng lớn ở phía bên phải. Softmax tính trên toàn bộ 32 bước làm Attention đôi khi tập trung vào các vùng đệm trắng ở cuối chuỗi.
*   **Ý nghĩa thực tiễn:**
    *   CTC rất tối ưu cho các bài toán nhận dạng từ đơn ngắn, tốc độ nhanh.
    *   Attention có khả năng giải thích tốt nhờ vẽ được bản đồ chú ý.
*   **Lời thoại gợi ý:** *"Lý do CTC hoạt động tốt hơn là do nó tránh được lỗi tích lũy sai số dây chuyền khi giải mã tuần tự. Đồng thời, mô hình Attention hiện tại chưa có mặt nạ đệm ảnh (Padding Masking) nên khi gặp vùng đệm trắng lớn ở góc phải ảnh, trọng số Attention đôi khi bị phân tán vào vùng vô nghĩa này."*

---

### SLIDE 9: TRỰC QUAN HÓA BẢN ĐỒ ATTENTION (ATTENTION MAP VISUALIZATION)
*   *Chèn hình ảnh bản đồ Attention của dự án tại đây (tải từ thư mục `outputs/yabi_preprocessed.png` hoặc các file tương tự)*
*   **Mô tả hình ảnh:**
    *   Heatmap hiển thị vùng sáng tương ứng với chữ cái đang được dịch.
    *   Chứng minh mô hình thực sự học được cách căn chỉnh (alignment) nét viết tay mà không cần gán nhãn thủ công tọa độ ký tự.
*   **Lời thoại gợi ý:** *"Một ưu thế lớn của mô hình Attention là tính tường minh. Ở đây thầy cô có thể thấy bản đồ Attention Map của mô hình. Khi mô hình dịch ra từng ký tự, vùng màu sáng di chuyển chính xác theo từng nét chữ trên ảnh gốc. Điều này chứng minh mô hình đã tự học được quy luật căn chỉnh một cách thông minh."*

---

### SLIDE 10: KẾT LUẬN & HƯỚNG PHÁT TRIỂN (CONCLUSIONS)
*   **Kết luận:**
    *   Đã xây dựng thành công 2 pipeline nhận dạng chữ viết tay local đạt độ chính xác cao trên IAM Dataset.
    *   Đã cải tiến thành công Attention Model thông qua khối BiLSTM bổ trợ ngữ cảnh.
*   **Hướng phát triển tương lai:**
    1.  Tích hợp cơ chế **Attention Padding Masking** để loại bỏ hoàn toàn vùng đệm trắng.
    2.  Kết hợp thêm **Language Model (Mô hình ngôn ngữ)** ở Decoder để sửa lỗi chính tả tự động.
    3.  Thử nghiệm kiến trúc Transformer hoàn toàn tự xây dựng khi có thêm dữ liệu.
*   **Lời thoại gợi ý:** *"Tóm lại, đồ án đã hoàn thành các mục tiêu đặt ra. Hướng đi tiếp theo của em là bổ sung cơ chế Masking để loại bỏ ảnh hưởng của vùng đệm và tích hợp mô hình ngôn ngữ để tự động sửa lỗi chính tả sau khi nhận dạng. Em xin chân thành cảm ơn thầy cô và mong nhận được ý kiến đóng góp."*
