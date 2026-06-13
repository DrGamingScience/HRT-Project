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
*   **Mô hình chính đề xuất:** CNN Encoder (ResNet18) + Cơ chế Bahdanau Attention + RNN Decoder (GRU).
*   **Mô hình đối chứng (Baseline):** CNN Encoder (ResNet18) + Chuỗi tuần tự BiLSTM + Hàm mất mát CTC (Connectionist Temporal Classification).

---

## 2. KIẾN TRÚC MÔ HÌNH (MODEL ARCHITECTURES)

### 2.1. Mô hình Đối chứng (CTC Baseline)
Kiến trúc baseline sử dụng mạng nơ-ron tích chập kết hợp mạng hồi quy hai chiều và hàm mất mát CTC để học căn chỉnh tự động mà không cần định nghĩa nhãn ở từng frame:

$$\text{Image} \rightarrow \text{CNN (ResNet18)} \rightarrow \text{Sequence Reshape} \rightarrow \text{2-layer BiLSTM} \rightarrow \text{Linear Head} \rightarrow \text{CTC Loss}$$

*   **CNN Encoder:** Sử dụng ResNet18 dừng lại ở `layer3` để trích xuất đặc trưng không gian dạng 2D. Đầu ra có kích thước $(B, 256, H', W')$, trong đó với kích thước ảnh đầu vào $64 \times 512$, đầu ra có dạng $(B, 256, 4, 32)$.
*   **Sequence Reshape:** Biến đổi đặc trưng 2D thành chuỗi thời gian dọc theo chiều ngang ảnh: $(B, 256, 4, 32) \rightarrow (B, 32, 1024)$. Chuỗi có độ dài thời gian $T = 32$ bước và chiều đặc trưng là $1024$.
*   **BiLSTM Context Layer:** 2 lớp BiLSTM với số chiều ẩn $256$ (mỗi chiều, tổng cộng $512$ chiều ở đầu ra) có dropout $0.3$. Nó học thông tin ngữ cảnh hai chiều toàn cục của từ.
*   **Linear Classifier & CTCLoss:** Chiếu đặc trưng từ $512 \rightarrow V$ (với $V = 29$ là kích thước bộ từ vựng). Sử dụng hàm mất mát `nn.CTCLoss` cấu hình `blank=0` (token `<pad>`).

### 2.2. Mô hình Đề xuất Cải tiến (Attention Model - Phase 1)
Để khắc phục điểm yếu của CTC (giả định độc lập giữa các ký tự đầu ra), mô hình Attention được cải tiến bằng cách bổ sung lớp ngữ cảnh tuần tự trước khi tính cơ chế chú ý:

$$\text{Image} \rightarrow \text{CNN (ResNet18)} \rightarrow \text{Sequence Reshape} \rightarrow \text{2-layer BiLSTM (Context)} \rightarrow \text{Bahdanau Attention} \rightarrow \text{GRU Decoder} \rightarrow \text{Cross Entropy Loss}$$

*   **BiLSTM Context Layer:** Đã được tích hợp thêm vào trước cơ chế Attention trong đợt cập nhật **Phase 1** để chuyển đổi đặc trưng CNN cục bộ thành chuỗi đặc trưng ngữ cảnh hai chiều chất lượng cao. Chiều đầu ra của lớp này giảm xuống còn $512$.
*   **Bahdanau Attention (Additive Attention):**
    Tính toán phân phối trọng số chú ý $\alpha_{t, s}$ tại bước giải mã $t$ trên toàn bộ các vị trí đặc trưng $s \in [1, 32]$ của ảnh:
    $$e_{t, s} = v_a^T \tanh(W_a s_{t-1} + U_a h_s)$$
    $$\alpha_{t, s} = \frac{\exp(e_{t, s})}{\sum_{k=1}^T \exp(e_{t, k})}$$
    $$c_t = \sum_{s=1}^T \alpha_{t, s} h_s$$
    Trong đó $s_{t-1}$ là trạng thái ẩn của Decoder GRU ở bước trước, và $h_s$ là đặc trưng ngữ cảnh từ BiLSTM.
*   **GRU Decoder:** Một lớp `GRUCell` với kích thước ẩn $256$. Đầu vào ở mỗi bước giải mã là sự kết hợp của Embedding ký tự trước đó ($128$ chiều) và Vector ngữ cảnh $c_t$ ($512$ chiều).
*   **Teacher Forcing:** Khi train áp dụng Teacher Forcing tỉ lệ $0.5$ (giảm dần $0.02$ sau mỗi epoch) để hỗ trợ quá trình hội tụ nhanh. Khi đánh giá (validation/test), tỉ lệ này bằng $0.0$.

### 2.3. Sự cần thiết của bộ giải mã độc lập (inference/)
Một câu hỏi lý luận quan trọng thường gặp khi bảo vệ đồ án là: *Tại sao khi đã có các file dự đoán (`predict_...`) và đánh giá (`evaluate_...`) nhưng hệ thống vẫn cần một thư mục riêng biệt là `inference/`?*

Lý do nằm ở sự phân tách trách nhiệm (Separation of Concerns) trong thiết kế hệ thống và bản chất của mô hình học sâu:
1.  **Bản chất đầu ra của mô hình:** Các mạng nơ-ron chỉ làm nhiệm vụ tính toán và trả về các tensor chứa phân phối xác suất thô của các ký tự. Bản thân mô hình AI không tự sinh ra văn bản mà chỉ đưa ra xác suất.
2.  **Độc lập về giải thuật giải mã (Decoding Algorithms):** Việc chuyển đổi từ ma trận xác suất sang chuỗi chữ viết có nghĩa (ví dụ: gộp các ký tự trùng nhau trong CTC, hoặc tìm kiếm chùm Beam Search) là các thuật toán logic độc lập với mô hình học sâu.
3.  **Tái sử dụng mã nguồn và dễ bảo trì (Modular Design):** Cả tiến trình đánh giá tập test (`evaluate_...`) và tiến trình dự đoán ảnh đơn lẻ (`predict_...`) đều cần dịch ma trận xác suất thành chữ. Việc tách các giải thuật này vào thư mục `inference/` giúp tránh trùng lặp code, đồng thời cho phép dễ dàng nâng cấp hoặc thay thế thuật toán giải mã (ví dụ: chuyển từ Greedy sang Beam Search) mà không cần sửa đổi cấu trúc của mô hình hay các file chạy chính.

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
*   **Resize & Padding:** Ảnh được resize theo tỉ lệ khía cạnh (aspect ratio) đưa về chiều cao cố định $64$ px. Nếu chiều rộng thực tế nhỏ hơn $512$ px, ảnh sẽ được chèn đệm trắng (pixel 255) về bên phải. Nếu lớn hơn, ảnh sẽ bị crop về $512$ px.
*   **Chuẩn hóa (Normalization):** Dữ liệu ảnh được chuẩn hóa theo phân phối ImageNet: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.

### 4.2. Tăng cường dữ liệu (Data Augmentation)
Sử dụng thư viện `albumentations` thiết lập quy trình tăng cường ngẫu nhiên khi huấn luyện:
1.  **Xoay ảnh (Random Rotation):** Góc xoay trong khoảng $[-15^\circ, 15^\circ]$.
2.  **Biến dạng Affine/Shear:** Biến dạng góc nghiêng chữ viết trong khoảng $[-15^\circ, 15^\circ]$ để mô phỏng kiểu chữ nghiêng.
3.  **Elastic Transform:** Áp dụng biến dạng đàn hồi ($\alpha=34, \sigma=4$) để mô phỏng sự co giãn nét chữ viết tay tự nhiên.
4.  **Làm mờ & Nhiễu:** `GaussianBlur` và `GaussNoise` để mô phỏng ảnh chất lượng thấp hoặc bị nhòe mực.
5.  **Thay đổi độ sáng/tương phản:** `RandomBrightnessContrast` mô phỏng ánh sáng không đồng đều khi quét ảnh.

---

## 5. KẾT QUẢ THỰC NGHIỆM & PHÂN TÍCH (EXPERIMENTAL RESULTS)

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

### 5.3. Kết quả so sánh trên tập Test Set độc lập
Dưới đây là bảng so sánh hiệu năng thực tế thu được từ các checkpoints của mô hình Baseline và mô hình Attention cải tiến (có lớp BiLSTM):

| Chỉ số (Metrics) | Mô hình CTC Baseline | Mô hình Attention (Phase 1) |
|---|---|---|
| **Word Accuracy (WA)** | **80.89%** | **83.09%** |
| **Character Error Rate (CER)** | **7.25%** | **7.06%** |
| **Normalized Edit Distance (NED)** | **94.26%** | **94.21%** |
| **Epoch đạt kết quả tốt nhất** | Epoch 49 / 50 | Epoch 62 / 80 |
| **Dung lượng file checkpoint** | ~78.6 MB | ~91.0 MB |

### 5.4. Đánh giá và Lập luận khoa học
1.  **Sự cải tiến vượt bậc và tính ưu việt của Mô hình Attention (Phase 1):**
    Sau khi được cải tiến bằng cách bổ sung thêm **2 lớp BiLSTM làm khối ngữ cảnh bổ trợ** cho Encoder ở Phase 1, mô hình Attention đã đạt bước nhảy vọt về hiệu năng. Kết quả thực nghiệm trên tập Test độc lập cho thấy mô hình Attention đạt độ chính xác từ (WA) lên tới **83.09%** và tỷ lệ lỗi ký tự (CER) giảm xuống chỉ còn **7.06%**, chính thức **vượt qua mô hình đối chứng CTC Baseline** (WA 80.89%, CER 7.25%). Điều này chứng minh sự kết hợp giữa đặc trưng ngữ cảnh hai chiều mạnh mẽ từ BiLSTM và khả năng căn chỉnh động của cơ chế Attention đã tối ưu hơn so với việc phân loại độc lập của CTC.
2.  **So sánh với phiên bản Attention cũ:**
    Ở phiên bản gốc ban đầu (không có lớp BiLSTM ở Encoder), Attention Model chỉ đạt kết quả rất thấp (**73.24% WA** và **12.05% CER**) và cực kỳ khó hội tụ do đặc trưng CNN cục bộ thiếu thông tin liên kết chuỗi. Việc tích hợp thêm khối BiLSTM đã giúp Word Accuracy tăng thêm gần **10%** và CER giảm gần **5%** trên tập kiểm thử. Điều này khẳng định tầm quan trọng của việc thu nhận đặc trưng ngữ cảnh trước khi thực hiện cơ chế tập trung chú ý (Attention).
3.  **Hạn chế và Hướng phát triển:**
    Mặc dù mô hình đã đạt hiệu năng rất cao, hiện tượng Attention bị trôi nhẹ vào các vùng đệm trắng ở cuối ảnh vẫn thỉnh thoảng xảy ra do chưa áp dụng **mặt nạ đệm (Attention Padding Masking)** khi tính toán softmax. Đây là điểm hạn chế sẽ được khắc phục ở nghiên cứu tiếp theo bằng cách truyền mặt nạ độ dài thực của ảnh từ dataloader.

---

## 6. TRỰC QUAN HÓA BẢN ĐỒ ATTENTION (ATTENTION VISUALIZATION)

Một trong những ưu điểm lớn nhất của mô hình Attention so với CTC là khả năng giải thích được (Explainability). Bằng cách lưu lại trọng số Attention $\alpha_{t, s}$ thu được từ bước giải mã, hệ thống vẽ ra heatmap thể hiện vùng ảnh mà mô hình đang "nhìn" vào khi dự đoán ký tự tương ứng:

*   **Đầu vào:** Một ảnh từ viết tay đơn lẻ.
*   **Xử lý:** Trích xuất mảng numpy kích thước $(T_{\text{target}}, T_{\text{source}})$, với $T_{\text{source}} = 32$.
*   **Trực quan hóa:** Sử dụng thư viện `matplotlib` vẽ lớp màu nhiệt (heatmap overlay) chồng lên ảnh gốc. Khi mô hình dự đoán đúng ký tự thứ $i$, vùng màu sáng nhất sẽ nằm chính xác tại vị trí nét chữ của ký tự đó trên ảnh. Điều này giúp tăng tính thuyết phục khi bảo vệ đồ án trước hội đồng chấm thi.
