# BÁO CÁO KỸ THUẬT: HỆ THỐNG NHẬN DẠNG CHỮ VIẾT TAY MỨC ĐỘ TỪ ĐƠN
## (TECHNICAL REPORT: WORD-LEVEL HANDWRITTEN TEXT RECOGNITION)

Tài liệu này tổng hợp toàn bộ thông tin kỹ thuật, phân tích kiến trúc, hướng dẫn thiết lập môi trường và kết quả thử nghiệm thực tế của dự án **HRT-Project** phục vụ việc chuyển đổi sang báo cáo LaTeX.

---

## 1. MỞ ĐẦU & MỤC TIÊU ĐỀ TÀI (INTRODUCTION & GOALS)

### 1.1. Bài toán
*   **Đầu vào (Input):** Ảnh grayscale chứa duy nhất một từ tiếng Anh viết tay (Word-level image).
*   **Đầu ra (Output):** Chuỗi văn bản dự đoán tương ứng với từ viết tay trong ảnh (ví dụ: `"hello"`, `"charming"`).
*   **Không gian bài toán:** Nhận dạng từ đơn lẻ từ bộ dữ liệu IAM (IAM Handwriting Dataset). Không xử lý mức độ dòng hay đoạn văn.

### 1.2. Ràng buộc & Công nghệ sử dụng
*   Huấn luyện và suy diễn hoàn toàn local, không sử dụng các dịch vụ OCR API bên thứ ba (Google Cloud Vision, Azure OCR, OpenAI).
*   **Framework chính:** PyTorch.
*   **Phương pháp tiếp cận:** Xây dựng và so sánh **hai kiến trúc mô hình** trên cùng bộ dữ liệu:
    1.  **CTC Baseline:** CNN (ResNet18) + BiLSTM + CTC Loss — mô hình cơ sở (baseline) sử dụng hàm mất mát CTC để căn chỉnh chuỗi đầu vào–đầu ra.
    2.  **Attention Model (Đề xuất):** CNN (ResNet18) + BiLSTM Context Layer + Bahdanau Attention + GRU Decoder — mô hình chính sử dụng cơ chế chú ý để căn chỉnh động.

---

## 2. KIẾN TRÚC MÔ HÌNH (MODEL ARCHITECTURE)

### 2.1. Mô hình CTC Baseline

Mô hình CTC Baseline là kiến trúc cơ sở được xây dựng đầu tiên để thiết lập mức đánh giá ban đầu (baseline). Kiến trúc gồm 3 khối chính:

$$\text{Image} \rightarrow \text{CNN (ResNet18)} \rightarrow \text{Sequence Reshape} \rightarrow \text{2-layer BiLSTM} \rightarrow \text{Linear Projection} \rightarrow \text{CTC Loss}$$

*   **CNN Encoder (ResNet18):** Dùng chung với mô hình Attention, ResNet18 pretrained trên ImageNet dừng lại ở `layer3` để trích xuất đặc trưng không gian. Đầu ra: $(B, 256, 4, 16)$ với ảnh đầu vào $64 \times 256$.
*   **Sequence Reshape:** Ghép chiều cao vào chiều kênh để biến đổi đặc trưng 2D thành chuỗi thời gian: $(B, 256, 4, 16) \rightarrow (B, 16, 1024)$. Chuỗi có độ dài thời gian $T = 16$ bước.
*   **BiLSTM:** 2 lớp BiLSTM với $hidden\_dim = 256$ (tổng $512$ chiều ở đầu ra), dropout $0.3$. Mạng hồi quy hai chiều giúp nắm bắt ngữ cảnh toàn cục của từ.
*   **Linear Projection & CTC Loss:** Lớp chiếu tuyến tính sang số lượng từ vựng, sau đó áp dụng `log_softmax`. Hàm mất mát `nn.CTCLoss` tự động căn chỉnh (alignment) giữa chuỗi đầu ra có độ dài cố định $T = 16$ và nhãn đích có độ dài thay đổi, mà không cần xác định vị trí tương ứng từng ký tự trên ảnh.
*   **CTC Greedy Decoding:** Khi suy diễn, giải mã bằng cách lấy argmax tại mỗi bước thời gian, sau đó loại bỏ ký tự lặp liên tiếp (collapse) và loại bỏ blank token (index 0).

### 2.2. Mô hình Attention (Đề xuất)

Kiến trúc mô hình chính được xây dựng theo hướng Encoder–Decoder kết hợp cơ chế chú ý Bahdanau để căn chỉnh động đặc trưng ảnh và chuỗi ký tự đầu ra:

$$\text{Image} \rightarrow \text{CNN (ResNet18)} \rightarrow \text{Sequence Reshape} \rightarrow \text{2-layer BiLSTM (Context)} \rightarrow \text{Bahdanau Attention} \rightarrow \text{GRU Decoder} \rightarrow \text{Cross Entropy Loss}$$

*   **CNN Encoder:** Sử dụng ResNet18 dừng lại ở `layer3` để trích xuất đặc trưng không gian dạng 2D. Đầu ra có kích thước $(B, 256, H', W')$, trong đó với kích thước ảnh đầu vào $64 \times 512$, đầu ra có dạng $(B, 256, 4, 32)$.
*   **Sequence Reshape:** Biến đổi đặc trưng 2D thành chuỗi thời gian dọc theo chiều ngang ảnh: $(B, 256, 4, 32) \rightarrow (B, 32, 1024)$. Chuỗi có độ dài thời gian $T = 32$ bước và chiều đặc trưng là $1024$.
*   **BiLSTM Context Layer:** 2 lớp BiLSTM với số chiều ẩn $256$ (mỗi chiều, tổng cộng $512$ chiều ở đầu ra) có dropout $0.3$. Lớp này học thông tin ngữ cảnh hai chiều toàn cục của từ, biến đặc trưng CNN cục bộ thành chuỗi đặc trưng ngữ cảnh chất lượng cao trước khi đưa vào Attention.
*   **Bahdanau Attention (Additive Attention):**
    Tính toán phân phối trọng số chú ý $\alpha_{t, s}$ tại bước giải mã $t$ trên toàn bộ các vị trí đặc trưng $s \in [1, 32]$ của ảnh:
    $$e_{t, s} = v_a^T \tanh(W_a s_{t-1} + U_a h_s)$$
    $$\alpha_{t, s} = \frac{\exp(e_{t, s})}{\sum_{k=1}^T \exp(e_{t, k})}$$
    $$c_t = \sum_{s=1}^T \alpha_{t, s} h_s$$
    Trong đó $s_{t-1}$ là trạng thái ẩn của Decoder GRU ở bước trước, và $h_s$ là đặc trưng ngữ cảnh từ BiLSTM.
*   **GRU Decoder:** Một lớp `GRUCell` với kích thước ẩn $256$. Đầu vào ở mỗi bước giải mã là sự kết hợp của Embedding ký tự trước đó ($128$ chiều) và Vector ngữ cảnh $c_t$ ($512$ chiều).
*   **Teacher Forcing:** Khi train áp dụng Teacher Forcing tỉ lệ $0.5$ (giảm dần $0.02$ sau mỗi epoch) để hỗ trợ quá trình hội tụ nhanh. Khi đánh giá (validation/test), tỉ lệ này bằng $0.0$.

### 2.3. So sánh thiết kế hai mô hình

| Tiêu chí | CTC Baseline | Attention Model |
|---|---|---|
| **Phương pháp căn chỉnh** | Tự động qua hàm CTC Loss (marginalize tất cả alignment hợp lệ) | Động qua cơ chế Bahdanau Attention (học soft alignment) |
| **Decoder** | Không có (projection trực tiếp) | GRU Decoder autoregressive |
| **Hàm mất mát** | `nn.CTCLoss` (blank = 0) | `nn.CrossEntropyLoss` (ignore_index = pad) |
| **Giải mã** | CTC Greedy: collapse repeated → remove blank | Greedy hoặc Beam Search |
| **Ưu điểm** | Huấn luyện đơn giản, không cần teacher forcing | Căn chỉnh chính xác hơn, có thể giải thích qua attention map |
| **Hạn chế** | Chiều dài đầu ra bị giới hạn bởi $T$ (16 bước); giả định conditional independence giữa các ký tự | Phức tạp hơn, cần teacher forcing, huấn luyện chậm hơn |

### 2.4. Sự cần thiết của bộ giải mã độc lập (inference/)
Một câu hỏi lý luận quan trọng thường gặp khi bảo vệ đồ án là: *Tại sao khi đã có các file dự đoán (`predict_attention.py`) và đánh giá (`evaluate_attention.py`) nhưng hệ thống vẫn cần một thư mục riêng biệt là `inference/`?*

Lý do nằm ở sự phân tách trách nhiệm (Separation of Concerns) trong thiết kế hệ thống và bản chất của mô hình học sâu:
1.  **Bản chất đầu ra của mô hình:** Các mạng nơ-ron chỉ làm nhiệm vụ tính toán và trả về các tensor chứa phân phối xác suất thô của các ký tự. Bản thân mô hình AI không tự sinh ra văn bản mà chỉ đưa ra xác suất.
2.  **Độc lập về giải thuật giải mã (Decoding Algorithms):** Việc chuyển đổi từ ma trận xác suất sang chuỗi chữ viết có nghĩa (ví dụ: CTC Greedy Decoding, tìm kiếm tham lam Greedy hoặc tìm kiếm chùm Beam Search) là các thuật toán logic độc lập với mô hình học sâu. Cả hai mô hình CTC và Attention đều cần bộ giải mã riêng biệt, được tổ chức gọn gàng trong thư mục `inference/`.
3.  **Tái sử dụng mã nguồn và dễ bảo trì (Modular Design):** Cả tiến trình đánh giá tập test (`evaluate_*.py`) và tiến trình dự đoán ảnh (`predict_*.py`) đều cần dịch ma trận xác suất thành chữ. Việc tách các giải thuật này vào thư mục `inference/` giúp tránh trùng lặp code, đồng thời cho phép dễ dàng nâng cấp hoặc thay thế thuật toán giải mã mà không cần sửa đổi cấu trúc mô hình.

---

## 3. THIẾT LẬP MÔI TRƯỜNG HUẤN LUYỆN (ENVIRONMENT & HARDWARE SETUP)

Để đảm bảo tính tái lập của nghiên cứu, môi trường huấn luyện được cấu hình trên máy local chạy Windows với các thông số phần cứng và phần mềm như sau:

### 3.1. Phần cứng thực tế (Hardware Specification)
*   **GPU:** NVIDIA GeForce RTX 5060 Ti (Kiến trúc Blackwell)
*   **VRAM:** 16 GB
*   **Driver Version:** 581.80
*   **CUDA Support tối đa:** 13.0

### 3.2. Phần mềm & Thư viện (Software Environment)
*   **Python Interpreter:** `3.11.15` (Cài đặt trong môi trường ảo Conda tại `C:\conda_envs\hrt`)
*   **PyTorch Framework:** `2.12.0+cu130` (Kèm CUDA runtime 13.0 đóng gói sẵn để tương thích tối đa với kiến trúc GPU mới)
*   **Các thư viện bổ trợ chính:**
    *   `opencv-python` (Đọc và xử lý ảnh)
    *   `albumentations` (Tăng cường dữ liệu mạnh mẽ)
    *   `editdistance` (Tính toán khoảng cách Levenshtein)
    *   `pandas`, `numpy` (Xử lý tập nhãn và dữ liệu dạng bảng)

---

## 4. TIỀN XỬ LÝ & TĂNG CƯỜNG DỮ LIỆU (PREPROCESSING & AUGMENTATION)

### 4.1. Tiền xử lý hình ảnh (Image Preprocessing)
*   **Grayscale Conversion:** Chuyển ảnh RGB sang 1 kênh grayscale để loại bỏ nhiễu màu sắc. Sau đó nhân bản thành 3 kênh ở bước cuối để tương thích với ResNet18.
*   **Resize & Padding:** Ảnh được resize theo tỉ lệ khía cạnh (aspect ratio) đưa về chiều cao cố định $64$ px. Nếu chiều rộng thực tế nhỏ hơn chiều rộng mục tiêu, ảnh sẽ được chèn đệm trắng (pixel 255) về bên phải. Nếu lớn hơn, ảnh sẽ bị crop.
*   **Chuẩn hóa (Normalization):** Dữ liệu ảnh được chuẩn hóa theo phân phối ImageNet: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.

> **Lưu ý về kích thước ảnh đầu vào:** Mô hình CTC Baseline sử dụng ảnh $64 \times 256$ (IMAGE_WIDTH trong config) để đạt chuỗi thời gian $T = 16$ bước. Mô hình Attention sử dụng ảnh $64 \times 512$ để đạt $T = 32$ bước, phù hợp với cơ chế attention cần nhiều vị trí hơn.

### 4.2. Tăng cường dữ liệu (Data Augmentation)
Sử dụng thư viện `albumentations` thiết lập quy trình tăng cường ngẫu nhiên khi huấn luyện:
1.  **Xoay ảnh (Random Rotation):** Góc xoay trong khoảng $[-15^\circ, 15^\circ]$.
2.  **Biến dạng Affine/Shear:** Biến dạng góc nghiêng chữ viết trong khoảng $[-15^\circ, 15^\circ]$ để mô phỏng kiểu chữ nghiêng.
3.  **Elastic Transform:** Áp dụng biến dạng đàn hồi ($\alpha=34, \sigma=4$) để mô phỏng sự co giãn nét chữ viết tay tự nhiên.
4.  **Làm mờ & Nhiễu:** `GaussianBlur` và `GaussNoise` để mô phỏng ảnh chất lượng thấp hoặc bị nhòe mực.
5.  **Thay đổi độ sáng/tương phản:** `RandomBrightnessContrast` mô phỏng ánh sáng không đồng đều khi quét ảnh.

---

## 5. KẾT QUẢ THỰC NGHIỆM (EXPERIMENTAL RESULTS)

### 5.1. Bộ dữ liệu huấn luyện (Dataset Split)
Sử dụng tập dữ liệu **IAM Handwriting Dataset** ở cấp độ từ (word level). Sau khi lọc bỏ các nhãn chứa ký tự lạ (chỉ giữ lại $a-z$), tổng số mẫu sạch là **96,835 mẫu**. Dữ liệu được chia theo tỉ lệ $80/10/10$:
*   **Tập huấn luyện (Train Set):** 77,468 mẫu
*   **Tập kiểm định (Val Set):** 9,683 mẫu
*   **Tập kiểm thử (Test Set):** 9,684 mẫu

### 5.2. Các chỉ số đánh giá (Evaluation Metrics)
1.  **Word Accuracy (WA - Độ chính xác từ):**
    $$WA = \frac{\text{Số từ dự đoán khớp 100%}}{\text{Tổng số từ thử nghiệm}} \times 100\%$$
2.  **Character Error Rate (CER - Tỉ lệ lỗi ký tự):**
    $$CER = \frac{\sum \text{edit\_distance}(\text{pred}, \text{target})}{\sum \text{len}(\text{target})} \times 100\%$$
3.  **Normalized Edit Distance (NED - Khoảng cách chỉnh sửa chuẩn hóa):**
    $$NED = 1 - \frac{1}{N} \sum_{i=1}^N \frac{\text{edit\_distance}(\text{pred}_i, \text{target}_i)}{\max(\text{len}(\text{pred}_i), \text{len}(\text{target}_i))}$$

### 5.3. Kết quả trên tập Test Set độc lập
Kết quả thực tế thu được từ checkpoint tốt nhất của cả hai mô hình trên **9,684 mẫu kiểm thử**:

| Chỉ số (Metrics) | CTC Baseline | Attention Model (Đề xuất) |
|---|---|---|
| **Word Accuracy (WA)** | *(chạy evaluate để cập nhật)* | **83.09%** |
| **Character Error Rate (CER)** | *(chạy evaluate để cập nhật)* | **7.06%** |
| **Normalized Edit Distance (NED)** | *(chạy evaluate để cập nhật)* | **94.21%** |
| **Epoch đạt kết quả tốt nhất** | — | Epoch 62 / 80 |
| **Dung lượng file checkpoint** | ~75 MB | ~91 MB |

> **Ghi chú:** Kết quả CTC Baseline sẽ được cập nhật sau khi chạy lệnh: `python -m src.evaluate_ctc_baseline`

### 5.4. Phân tích & Lập luận khoa học

#### 5.4.1. So sánh CTC Baseline vs Attention Model
Mô hình CTC Baseline đóng vai trò là **mức chuẩn** (baseline) để đo lường mức cải thiện mà kiến trúc Attention mang lại. Sự khác biệt cơ bản nằm ở cách hai mô hình xử lý **bài toán căn chỉnh (alignment)** giữa chuỗi đặc trưng ảnh và chuỗi ký tự đầu ra:

*   **CTC** giải quyết alignment bằng cách marginalize trên tất cả các đường căn chỉnh hợp lệ thông qua thuật toán forward-backward. Tuy hiệu quả nhưng CTC giả định **conditional independence** giữa các ký tự tại từng bước thời gian, dẫn đến việc mô hình không thể nắm bắt mối quan hệ phụ thuộc cục bộ giữa các ký tự liền kề (ví dụ: cặp ký tự `th`, `ch`, `qu` thường đi cùng nhau).
*   **Attention** học một soft alignment rõ ràng tại mỗi bước giải mã, cho phép decoder "nhìn" vào đúng vùng ảnh tương ứng. Kết hợp với decoder autoregressive (GRU), mô hình có thể khai thác **language model implicitly** thông qua việc điều kiện hóa ký tự hiện tại dựa trên các ký tự đã sinh trước đó.

#### 5.4.2. Hiệu năng mô hình Attention
Mô hình Attention được cải tiến bằng cách bổ sung **2 lớp BiLSTM làm khối ngữ cảnh bổ trợ** cho Encoder đã đạt kết quả ấn tượng trên tập Test độc lập: **83.09% Word Accuracy** và **7.06% CER**. Điều này chứng minh sự kết hợp giữa đặc trưng ngữ cảnh hai chiều mạnh mẽ từ BiLSTM và khả năng căn chỉnh động của cơ chế Attention là một hướng thiết kế hiệu quả cho bài toán nhận dạng chữ viết tay.

#### 5.4.3. Tầm quan trọng của BiLSTM Context Layer
Ở phiên bản Attention gốc ban đầu (không có lớp BiLSTM), mô hình chỉ đạt **73.24% WA** và **12.05% CER** — cực kỳ khó hội tụ do đặc trưng CNN cục bộ thiếu thông tin liên kết chuỗi. Việc tích hợp thêm khối BiLSTM đã giúp Word Accuracy tăng thêm gần **10%** và CER giảm gần **5%** trên tập kiểm thử. Điều này khẳng định tầm quan trọng của việc thu nhận đặc trưng ngữ cảnh toàn cục trước khi thực hiện cơ chế tập trung chú ý.

#### 5.4.4. Hạn chế và Hướng phát triển
*   **Hạn chế CTC:** Chiều dài đầu ra bị giới hạn bởi $T = 16$ bước thời gian, dẫn đến mô hình không thể nhận dạng các từ có nhiều hơn 16 ký tự. Ngoài ra, giả định conditional independence khiến CTC dễ nhầm lẫn với các cặp ký tự tương tự.
*   **Hạn chế Attention:** Hiện tượng Attention bị trôi nhẹ vào các vùng đệm trắng ở cuối ảnh vẫn thỉnh thoảng xảy ra do chưa áp dụng **mặt nạ đệm (Attention Padding Masking)** khi tính toán softmax. Đây là điểm hạn chế sẽ được khắc phục ở nghiên cứu tiếp theo.
*   **Hướng phát triển:** Tích hợp Transformer Decoder thay thế GRU, áp dụng CTC–Attention joint decoding, hoặc kết hợp với language model ngoài để cải thiện thêm.

---

## 6. TRỰC QUAN HÓA BẢN ĐỒ ATTENTION (ATTENTION VISUALIZATION)

Một trong những ưu điểm lớn nhất của mô hình Attention so với CTC là khả năng giải thích được (Explainability). Bằng cách lưu lại trọng số Attention $\alpha_{t, s}$ thu được từ bước giải mã, hệ thống vẽ ra heatmap thể hiện vùng ảnh mà mô hình đang "nhìn" vào khi dự đoán ký tự tương ứng:

*   **Đầu vào:** Một ảnh từ viết tay đơn lẻ.
*   **Xử lý:** Trích xuất mảng numpy kích thước $(T_{\text{target}}, T_{\text{source}})$, với $T_{\text{source}} = 32$.
*   **Trực quan hóa:** Sử dụng thư viện `matplotlib` vẽ lớp màu nhiệt (heatmap overlay) chồng lên ảnh gốc. Khi mô hình dự đoán đúng ký tự thứ $i$, vùng màu sáng nhất sẽ nằm chính xác tại vị trí nét chữ của ký tự đó trên ảnh. Điều này giúp tăng tính thuyết phục khi bảo vệ đồ án trước hội đồng chấm thi.
*   **Lệnh chạy:**
    ```bash
    python -m src.predict_attention --image duong_dan_anh.png --save_attention
    ```
    *(Kết quả heatmap được lưu tại `outputs/attention_maps/`)*

> **Lưu ý:** Mô hình CTC Baseline **không có** attention map do kiến trúc không sử dụng cơ chế chú ý. Đây là một trong những lý do khiến mô hình Attention được ưu tiên trong nghiên cứu.
