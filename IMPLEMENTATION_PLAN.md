# IMPLEMENTATION PLAN — Word-Level Handwritten Text Recognition

> **Project**: HRT-Project
> **Location**: `D:\Downloads\HRT-Project`
> **GPU**: NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)
> **Framework**: PyTorch (local, không dùng API OCR bên thứ ba)
> **Ngày tạo**: 2026-06-06

---

## Mục Lục

1. [Project Goal](#1-project-goal)
2. [Dataset Assumption](#2-dataset-assumption)
3. [Recommended Project Structure](#3-recommended-project-structure)
4. [Main Model Design](#4-main-model-design)
5. [Vocabulary Design](#5-vocabulary-design)
6. [Preprocessing Pipeline](#6-preprocessing-pipeline)
7. [Data Augmentation](#7-data-augmentation)
8. [Data Preparation](#8-data-preparation)
9. [Training Plan For Attention Model](#9-training-plan-for-attention-model)
10. [Inference Plan](#10-inference-plan)
11. [Evaluation Plan](#11-evaluation-plan)
12. [Attention Visualization](#12-attention-visualization)
13. [CTC Baseline Plan](#13-ctc-baseline-plan)
14. [RTX 5060 Ti Local Setup](#14-rtx-5060-ti-local-setup)
15. [Requirements](#15-requirements)
16. [Implementation Order](#16-implementation-order)
17. [Risk Management](#17-risk-management)
18. [Final Deliverable Checklist](#18-final-deliverable-checklist)

---

## 1. Project Goal

### Mục tiêu

Xây dựng hệ thống nhận dạng chữ viết tay ở mức **một từ (word-level)** sử dụng deep learning.

### Đặc tả

| Mục | Chi tiết |
|---|---|
| **Input** | Ảnh chứa duy nhất một từ viết tay (word image) |
| **Output** | Chuỗi ký tự dự đoán cho từ đó (ví dụ: `"hello"`, `"world"`) |
| **Phạm vi** | Chỉ nhận dạng từ đơn lẻ, **không phải** nhận dạng cả câu hay cả dòng |
| **Ràng buộc** | Không dùng OCR API bên thứ ba (Google Vision, Azure OCR, OpenAI API, v.v.) |
| **Phần cứng** | Chạy local trên GPU NVIDIA GeForce RTX 5060 Ti (16 GB VRAM) |
| **Framework** | PyTorch |
| **Pretrained** | Có thể dùng CNN backbone pretrained (ResNet18) chạy local, không gọi API |

### Kiến trúc đã chốt

- **Hướng chính**: CNN Encoder (ResNet18 pretrained) + Bahdanau Attention + GRU Decoder + CrossEntropyLoss
- **Baseline**: CNN + BiLSTM + CTC Loss
- **Metrics**: Word Accuracy, Character Error Rate, Edit Distance / Normalized Edit Distance

### Lý do chọn hướng này

1. Attention Decoder giải quyết tốt chữ ngoáy, mờ, dính nét — vấn đề mà CTC đơn thuần xử lý kém.
2. Attention map có thể visualize → điểm mạnh khi bảo vệ đồ án.
3. GRU Decoder nhẹ hơn LSTM, đủ mạnh cho word-level (từ ngắn, max ~32 ký tự).
4. ResNet18 pretrained giúp hội tụ nhanh hơn train from scratch.
5. Baseline CTC cần có để chứng minh Attention Decoder thực sự cải thiện.

---

## 2. Dataset Assumption

### Cấu trúc thư mục dataset

```
dataset/
├── words/           ← Chứa các subfolder ảnh word-level
│   ├── a01/
│   │   ├── a01-000u/
│   │   │   ├── a01-000u-00-00.png
│   │   │   ├── a01-000u-00-01.png
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── label.txt        ← File label dạng TSV
```

### Format `label.txt`

Mỗi dòng là một cặp `image_path<TAB>label`:

```
words/a01/a01-000u/a01-000u-00-00.png	A
words/a01/a01-000u/a01-000u-00-01.png	MOVE
words/a01/a01-000u/a01-000u-00-02.png	to
words/a01/a01-000u/a01-000u-00-03.png	stop
```

### Thống kê dataset (từ file thực tế)

| Thông số | Giá trị |
|---|---|
| Tổng dòng trong `label.txt` | ~115,319 |
| Separator | TAB (`\t`) |
| Path format | Relative path tính từ `dataset/` |
| Ngôn ngữ | Tiếng Anh |
| Nguồn gốc | IAM Handwriting Dataset |

### Yêu cầu validation khi đọc dataset

Implementation **phải** kiểm tra các điều kiện sau cho mỗi sample:

1. **File ảnh tồn tại**: Kiểm tra `os.path.exists(full_image_path)`. Nếu không tồn tại → bỏ qua sample, ghi log.
2. **Label hợp lệ**: Label không rỗng, không chỉ chứa whitespace. Nếu không hợp lệ → bỏ qua, ghi log.
3. **Label trong phạm vi vocab**: Ở phase đầu, chỉ giữ label chứa ký tự `[a-z]` (sau khi lowercase). Label có ký tự ngoài phạm vi → bỏ qua, ghi log.
4. **Ảnh đọc được**: Thử load ảnh bằng PIL/OpenCV. Nếu ảnh bị corrupt, không đọc được → bỏ qua, ghi log.
5. **Không crash pipeline**: Dù có bao nhiêu sample lỗi, pipeline phải tiếp tục xử lý các sample còn lại. Cuối cùng in summary.

### Xử lý HTML entities trong label

Dataset IAM gốc có thể chứa HTML entities trong label, ví dụ:
- `&apos;` → `'`
- `&quot;` → `"`

Implementation cần decode HTML entities trước khi xử lý label.

---

## 3. Recommended Project Structure

```
HRT-Project/
├── dataset/                          ← Dataset gốc (KHÔNG chỉnh sửa)
│   ├── words/                        ← Folder ảnh word images
│   └── label.txt                     ← File label TSV gốc
│
├── data_processed/                   ← Output của prepare_data.py
│   ├── train.csv                     ← Training split
│   ├── val.csv                       ← Validation split
│   └── test.csv                      ← Test split
│
├── checkpoints/                      ← Model checkpoints
│   ├── best_attention_model.pth      ← Best attention model (theo val CER)
│   ├── last_attention_model.pth      ← Checkpoint cuối cùng
│   └── best_ctc_baseline.pth        ← Best CTC baseline
│
├── outputs/                          ← Kết quả inference & evaluation
│   ├── predictions.csv               ← Predictions trên test set
│   ├── attention_maps/               ← Attention heatmaps cho bảo vệ
│   └── logs/                         ← Training logs
│
├── src/                              ← Source code chính
│   ├── config.py                     ← Tất cả hyperparameters & paths
│   ├── prepare_data.py               ← Script tạo train/val/test split
│   ├── train_attention.py            ← Training loop cho attention model
│   ├── evaluate_attention.py         ← Evaluate attention model trên test set
│   ├── predict_attention.py          ← Predict một ảnh đơn lẻ
│   ├── train_ctc_baseline.py         ← Training loop cho CTC baseline
│   ├── evaluate_ctc_baseline.py      ← Evaluate CTC baseline trên test set
│   │
│   ├── data/                         ← Data loading & preprocessing
│   │   ├── dataset.py                ← PyTorch Dataset class
│   │   ├── transforms.py            ← Preprocessing & augmentation
│   │   └── vocab.py                  ← Vocabulary (char↔index)
│   │
│   ├── models/                       ← Model definitions
│   │   ├── resnet_encoder.py         ← ResNet18 CNN encoder
│   │   ├── attention.py              ← Bahdanau Attention module
│   │   ├── gru_decoder.py            ← GRU Decoder module
│   │   ├── attention_model.py        ← Full Encoder-Attention-Decoder model
│   │   └── ctc_baseline.py           ← CNN + BiLSTM + CTC model
│   │
│   ├── inference/                    ← Decoding strategies
│   │   ├── greedy_decode.py          ← Greedy decoding
│   │   └── beam_search.py           ← Beam search decoding
│   │
│   └── utils/                        ← Utility functions
│       ├── metrics.py                ← WA, CER, Edit Distance
│       ├── checkpoint.py             ← Save/load checkpoint
│       ├── image_utils.py            ← Ảnh debug/visualize helpers
│       └── seed.py                   ← Set random seed reproducibility
│
├── requirements.txt                  ← Dependencies
├── README.md                         ← Hướng dẫn sử dụng
├── ANALYSIS_RESULTS.md               ← Phân tích kỹ thuật lựa chọn mô hình
└── IMPLEMENTATION_PLAN.md            ← File này
```

### Mô tả chi tiết từng file

---

#### `src/config.py`

**Vai trò**: Tập trung **toàn bộ** hyperparameters, đường dẫn, cấu hình vào một file duy nhất. Tất cả các file khác import config từ đây.

**Input**: Không có (file constant).

**Output**: Các biến config được import bởi toàn bộ project.

**Nội dung chính cần có**:

```python
# --- Paths ---
PROJECT_ROOT = ...         # Auto-detect hoặc hardcode
DATASET_DIR = ...          # dataset/
LABEL_FILE = ...           # dataset/label.txt
WORDS_DIR = ...            # dataset/words/
DATA_PROCESSED_DIR = ...   # data_processed/
CHECKPOINT_DIR = ...       # checkpoints/
OUTPUT_DIR = ...           # outputs/
ATTENTION_MAP_DIR = ...    # outputs/attention_maps/
LOG_DIR = ...              # outputs/logs/

# --- Image ---
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3         # 3 vì ResNet18 pretrained cần RGB

# --- Vocabulary ---
MAX_LABEL_LENGTH = 32
MIN_LABEL_LENGTH = 1

# --- Model ---
ENCODER_MODEL = "resnet18"
EMBED_DIM = 128
DECODER_HIDDEN_DIM = 256
ATTENTION_DIM = 256
DECODER_NUM_LAYERS = 1
DROPOUT = 0.3

# --- Training ---
BATCH_SIZE = 64
NUM_EPOCHS = 80
LEARNING_RATE_DECODER = 1e-3
LEARNING_RATE_ENCODER = 1e-5
WEIGHT_DECAY = 1e-5
TEACHER_FORCING_RATIO = 0.5
TEACHER_FORCING_DECAY = 0.01    # Giảm dần mỗi epoch
GRAD_CLIP = 5.0
EARLY_STOPPING_PATIENCE = 10
USE_AMP = True                  # Mixed precision

# --- Data ---
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
NUM_WORKERS = 0                 # 0 cho Windows an toàn
PIN_MEMORY = True
RANDOM_SEED = 42

# --- Inference ---
BEAM_WIDTH = 5

# --- CTC Baseline ---
CTC_HIDDEN_DIM = 256
CTC_NUM_LAYERS = 2
CTC_LEARNING_RATE = 1e-3
CTC_BATCH_SIZE = 64
CTC_EPOCHS = 50

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

**Phụ thuộc**: Không phụ thuộc file nào (root config).

**Thứ tự implement**: **#2** (sau requirements.txt).

---

#### `src/data/vocab.py`

**Vai trò**: Quản lý mapping giữa ký tự và index. Cung cấp encode/decode cho label.

**Input**: Danh sách ký tự cho phép (từ config).

**Output**: Object `Vocabulary` với các method encode/decode.

**Class/Function chính**:

```python
class Vocabulary:
    def __init__(self, chars: str):
        """
        Khởi tạo vocab từ chuỗi ký tự.
        Tự động thêm <pad>, <sos>, <eos>.
        """

    PAD_TOKEN = "<pad>"    # index 0
    SOS_TOKEN = "<sos>"    # index 1
    EOS_TOKEN = "<eos>"    # index 2

    @property
    def pad_idx(self) -> int
    @property
    def sos_idx(self) -> int
    @property
    def eos_idx(self) -> int
    @property
    def size(self) -> int       # Tổng số entries trong vocab

    def char_to_idx(self, char: str) -> int
    def idx_to_char(self, idx: int) -> str

    def encode(self, text: str) -> List[int]:
        """Chuyển text → list of indices. KHÔNG thêm <sos>/<eos>."""

    def decode(self, indices: List[int], stop_at_eos: bool = True) -> str:
        """Chuyển list of indices → text. Dừng tại <eos> nếu có."""

    def is_valid_label(self, text: str) -> bool:
        """Kiểm tra text chỉ chứa ký tự trong vocab (không tính special tokens)."""
```

**Phụ thuộc**: `config.py`.

**Thứ tự implement**: **#3**.

---

#### `src/prepare_data.py`

**Vai trò**: Script chạy một lần để đọc `label.txt`, validate, lọc, shuffle, split → tạo `train.csv`, `val.csv`, `test.csv`.

**Input**: `dataset/label.txt` + `dataset/words/`.

**Output**: `data_processed/train.csv`, `data_processed/val.csv`, `data_processed/test.csv`.

**Function chính**:

```python
def load_label_file(label_path: str) -> List[Tuple[str, str]]:
    """Đọc label.txt, parse TSV, trả về list of (image_path, label)."""

def validate_sample(image_path: str, label: str, vocab: Vocabulary) -> Tuple[bool, str]:
    """
    Kiểm tra:
    - File ảnh tồn tại
    - Label hợp lệ (chỉ chứa ký tự trong vocab)
    - Độ dài label trong range [MIN_LABEL_LENGTH, MAX_LABEL_LENGTH]
    Trả về (is_valid, reason_if_invalid).
    """

def prepare_data():
    """
    Main function:
    1. Load label file
    2. Decode HTML entities trong label
    3. Lowercase label
    4. Validate từng sample
    5. Log số lượng valid/invalid + lý do
    6. Shuffle
    7. Split train/val/test
    8. Ghi CSV
    9. In summary
    """
```

**Format output CSV**:

```csv
image_path,label
dataset/words/a01/a01-000u/a01-000u-00-02.png,to
dataset/words/a01/a01-000u/a01-000u-00-03.png,stop
```

**Log output mẫu**:

```
=== Data Preparation Summary ===
Total lines read:              115,319
Valid samples:                  82,456
Skipped - image not found:      1,203
Skipped - label empty:             47
Skipped - label invalid chars:  28,412
Skipped - label too long:        3,201
---
Train samples:                 65,965 (80.0%)
Val samples:                    8,245 (10.0%)
Test samples:                   8,246 (10.0%)
---
Files written:
  data_processed/train.csv
  data_processed/val.csv
  data_processed/test.csv
```

**Phụ thuộc**: `config.py`, `data/vocab.py`.

**Thứ tự implement**: **#4**.

---

#### `src/data/transforms.py`

**Vai trò**: Định nghĩa tất cả preprocessing và augmentation transforms.

**Input**: Ảnh PIL/numpy.

**Output**: Tensor chuẩn hóa sẵn sàng cho model.

**Function/Class chính**:

```python
def get_train_transform() -> albumentations.Compose:
    """
    Trả về augmentation pipeline cho training:
    - Random rotation ±8°
    - Random affine/shear nhẹ
    - Gaussian blur (kernel 3, sigma 0.1-1.0)
    - Gaussian noise (var_limit 10-50)
    - Brightness/Contrast jitter (±0.2)
    - Erosion/Dilation nhẹ (optional)
    """

def get_val_transform() -> albumentations.Compose:
    """Trả về transform cho val/test (chỉ resize + normalize, KHÔNG augment)."""

def resize_and_pad(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    1. Resize ảnh giữ aspect ratio (scale theo height)
    2. Pad width bằng pixel trắng (255) nếu ngắn hơn target_w
    3. Crop width nếu dài hơn target_w
    """

def to_tensor_and_normalize(image: np.ndarray) -> torch.Tensor:
    """
    1. Convert grayscale → 3 channels (repeat)
    2. Scale pixel [0, 255] → [0, 1]
    3. Normalize theo ImageNet mean/std
    4. Trả về tensor (C, H, W)
    """
```

**Phụ thuộc**: `config.py`, `albumentations`, `torchvision.transforms`.

**Thứ tự implement**: **#5**.

---

#### `src/data/dataset.py`

**Vai trò**: PyTorch Dataset class cho cả training và validation/test.

**Input**: CSV file (từ `data_processed/`), `Vocabulary`, transform.

**Output**: `(image_tensor, target_tensor, target_length)` per sample.

**Class chính**:

```python
class HTRDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, vocab: Vocabulary, transform=None, is_train: bool = False):
        """
        - Đọc CSV
        - Lưu list of (image_path, label)
        - Lưu vocab reference
        - Lưu transform
        """

    def __len__(self) -> int

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        """
        1. Load ảnh từ disk
        2. Apply transform (resize, augment nếu train, normalize)
        3. Encode label → tensor (thêm <eos> ở cuối)
        4. Trả về (image_tensor, target_tensor, target_length)
        
        Nếu ảnh load lỗi → trả về sample khác (random fallback).
        """

def collate_fn(batch):
    """
    Custom collate function:
    1. Stack images thành batch tensor
    2. Pad targets về cùng max_length trong batch
    3. Trả về (images, targets, target_lengths)
    """
```

**Phụ thuộc**: `config.py`, `data/vocab.py`, `data/transforms.py`.

**Thứ tự implement**: **#6**.

---

#### `src/models/resnet_encoder.py`

**Vai trò**: CNN Encoder sử dụng ResNet18 pretrained ImageNet. Trích xuất feature map 2D từ ảnh.

**Input**: Batch ảnh tensor `(B, 3, H, W)`.

**Output**: Feature map `(B, C_enc, H', W')` — sau đó sẽ reshape thành sequence.

**Class chính**:

```python
class ResNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        """
        1. Load ResNet18 pretrained
        2. Bỏ layer avgpool và fc cuối
        3. Giữ: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        4. Nếu freeze=True → đóng băng toàn bộ params
        """

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, 3, 64, 256)
        → features: (B, 512, H', W')  # ResNet18 layer4 output channels = 512
        """

    def unfreeze_from(self, layer_name: str):
        """Mở đóng băng từ layer_name trở đi. Dùng cho fine-tune phase 2."""

    def get_output_size(self, input_h: int, input_w: int) -> Tuple[int, int, int]:
        """Tính (channels, out_h, out_w) khi input có kích thước (input_h, input_w)."""
```

**Chi tiết feature map**:

Với input `(B, 3, 64, 256)` qua ResNet18 (bỏ avgpool + fc):
- conv1 + maxpool: `(B, 64, 16, 64)`
- layer1: `(B, 64, 16, 64)`
- layer2: `(B, 128, 8, 32)`
- layer3: `(B, 256, 4, 16)`
- layer4: `(B, 512, 2, 8)`

Feature map output = `(B, 512, 2, 8)`.

Reshape cho decoder: `(B, 512, 2, 8)` → collapse H' vào channels → `(B, W', 512*H')` = `(B, 8, 1024)`.

Hoặc tốt hơn: `permute(0, 3, 1, 2).reshape(B, W', C*H')` = `(B, 8, 1024)`.

→ Sequence length = 8, feature dim = 1024.

> **Lưu ý**: Sequence length = 8 có thể quá ngắn. Xem xét bỏ layer4 hoặc giảm stride ở layer3/layer4 để tăng spatial resolution. Hoặc chỉ dùng đến layer3: output = `(B, 256, 4, 16)` → sequence length = 16, feature dim = 256*4 = 1024. **Khuyến nghị: thử cả hai, bắt đầu với layer3.**

**Phụ thuộc**: `config.py`, `torchvision.models`.

**Thứ tự implement**: **#7**.

---

#### `src/models/attention.py`

**Vai trò**: Module Bahdanau Attention (Additive Attention).

**Input**: Decoder hidden state `(B, decoder_hidden_dim)` + Encoder outputs `(B, seq_len, encoder_dim)`.

**Output**: Context vector `(B, encoder_dim)` + Attention weights `(B, seq_len)`.

**Class chính**:

```python
class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        """
        W_enc: Linear(encoder_dim, attention_dim)    — project encoder outputs
        W_dec: Linear(decoder_dim, attention_dim)     — project decoder hidden
        v_att: Linear(attention_dim, 1)               — score
        """

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        1. Project encoder outputs:  enc_proj = W_enc(encoder_outputs)  → (B, seq_len, att_dim)
        2. Project decoder hidden:   dec_proj = W_dec(decoder_hidden)   → (B, att_dim)
        3. Sum + tanh:               energy = tanh(enc_proj + dec_proj.unsqueeze(1))
        4. Score:                    scores = v_att(energy).squeeze(-1)  → (B, seq_len)
        5. Softmax:                  attn_weights = softmax(scores, dim=1)
        6. Context:                  context = sum(attn_weights * encoder_outputs)

        Return: (context_vector, attn_weights)
        """
```

**Phụ thuộc**: Không phụ thuộc file nào trong project (pure PyTorch module).

**Thứ tự implement**: **#8**.

---

#### `src/models/gru_decoder.py`

**Vai trò**: GRU Decoder sinh ký tự từng bước, sử dụng attention.

**Input**: Target sequence (khi train, teacher forcing) hoặc ký tự trước đó (khi inference).

**Output**: Sequence of logits `(B, max_len, vocab_size)` + attention weights per step.

**Class chính**:

```python
class GRUDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder_dim: int,
                 decoder_hidden_dim: int, attention_dim: int, dropout: float):
        """
        - embedding: Embedding(vocab_size, embed_dim)
        - attention: BahdanauAttention(encoder_dim, decoder_hidden_dim, attention_dim)
        - gru: GRUCell(embed_dim + encoder_dim, decoder_hidden_dim)
        - fc_out: Linear(decoder_hidden_dim + encoder_dim + embed_dim, vocab_size)
        - dropout: Dropout(dropout)
        - init_h: Linear(encoder_dim, decoder_hidden_dim)  — khởi tạo hidden state
        """

    def init_hidden(self, encoder_outputs: Tensor) -> Tensor:
        """
        Khởi tạo hidden state cho GRU từ mean của encoder outputs.
        mean_enc = encoder_outputs.mean(dim=1)  → (B, encoder_dim)
        h0 = tanh(init_h(mean_enc))             → (B, decoder_hidden_dim)
        """

    def forward_step(self, input_char: Tensor, hidden: Tensor, encoder_outputs: Tensor):
        """
        Một bước decode:
        1. Embed input char: emb = embedding(input_char)     → (B, embed_dim)
        2. Attention: context, attn_w = attention(hidden, encoder_outputs)
        3. GRU input: concat(emb, context)                   → (B, embed_dim + encoder_dim)
        4. GRU: hidden_new = gru(gru_input, hidden)
        5. Output: logits = fc_out(concat(hidden_new, context, emb))
        6. Return: logits, hidden_new, attn_w
        """

    def forward(self, encoder_outputs: Tensor, targets: Tensor = None,
                teacher_forcing_ratio: float = 0.5, max_len: int = 32):
        """
        Full forward pass:
        1. Init hidden state
        2. First input = <sos> token
        3. For each step t in [0, max_len):
            a. forward_step(input_t, hidden, encoder_outputs)
            b. If teacher forcing (random < ratio): input_{t+1} = targets[:, t]
            c. Else: input_{t+1} = argmax(logits)
        4. Collect all logits and attention weights
        5. Return: all_logits (B, T, vocab_size), all_attn_weights (B, T, seq_len)
        """
```

**Phụ thuộc**: `models/attention.py`, `config.py`.

**Thứ tự implement**: **#9**.

---

#### `src/models/attention_model.py`

**Vai trò**: Gộp Encoder + Decoder thành model hoàn chỉnh.

**Input**: Batch ảnh + targets (khi train).

**Output**: Logits + attention weights.

**Class chính**:

```python
class AttentionHTR(nn.Module):
    def __init__(self, vocab_size: int, ...):
        """
        - encoder: ResNetEncoder(pretrained=True, freeze=True)
        - decoder: GRUDecoder(...)
        """

    def forward(self, images: Tensor, targets: Tensor = None,
                teacher_forcing_ratio: float = 0.5, max_len: int = 32):
        """
        1. encoder_outputs = encoder(images)           → (B, C, H', W')
        2. Reshape: (B, C, H', W') → (B, W', C*H')    ← sequence cho attention
        3. logits, attn_weights = decoder(encoder_outputs, targets, tf_ratio, max_len)
        4. Return: logits, attn_weights
        """

    def freeze_encoder(self):
        """Đóng băng toàn bộ encoder."""

    def unfreeze_encoder(self, from_layer: str = "layer4"):
        """Mở đóng băng encoder từ layer chỉ định."""
```

**Phụ thuộc**: `models/resnet_encoder.py`, `models/gru_decoder.py`.

**Thứ tự implement**: **#10**.

---

#### `src/models/ctc_baseline.py`

**Vai trò**: Model baseline CNN + BiLSTM + CTC.

**Input**: Batch ảnh tensor.

**Output**: Log-probabilities `(T, B, vocab_size)` cho CTC loss.

**Class chính**:

```python
class CTCBaseline(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int):
        """
        - cnn: CNN đơn giản (hoặc ResNet18 bỏ layers sau) trích xuất features
        - bilstm: BiLSTM(input_size, hidden_dim, num_layers, bidirectional=True)
        - fc: Linear(hidden_dim * 2, vocab_size)  # *2 vì bidirectional
        """

    def forward(self, images: Tensor) -> Tensor:
        """
        1. features = cnn(images)                    → (B, C, H', W')
        2. Reshape: collapse H → (B, W', C*H')      ← sequence theo width
        3. Permute: (W', B, C*H') cho LSTM
        4. lstm_out = bilstm(features)               → (W', B, hidden*2)
        5. logits = fc(lstm_out)                      → (W', B, vocab_size)
        6. log_probs = log_softmax(logits, dim=2)
        7. Return: log_probs
        """
```

**Phụ thuộc**: `config.py`.

**Thứ tự implement**: **#17**.

---

#### `src/utils/metrics.py`

**Vai trò**: Tính tất cả evaluation metrics.

**Input**: Lists of predicted strings + ground truth strings.

**Output**: Dictionary of metric values.

**Function chính**:

```python
def word_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Tỷ lệ từ predict đúng hoàn toàn (exact match)."""

def character_error_rate(predictions: List[str], targets: List[str]) -> float:
    """CER = tổng edit_distance / tổng len(target)."""

def edit_distance(pred: str, target: str) -> int:
    """Levenshtein distance giữa 2 strings. Dùng thư viện `editdistance`."""

def normalized_edit_distance(predictions: List[str], targets: List[str]) -> float:
    """NED = 1 - mean(edit_distance / max(len(pred), len(target)))."""

def compute_all_metrics(predictions: List[str], targets: List[str]) -> dict:
    """Tính và trả về dict: {'word_accuracy': ..., 'cer': ..., 'ned': ...}."""
```

**Phụ thuộc**: Thư viện `editdistance`.

**Thứ tự implement**: **#11**.

---

#### `src/utils/checkpoint.py`

**Vai trò**: Save/load model checkpoints.

**Function chính**:

```python
def save_checkpoint(model, optimizer, epoch, metrics, path, is_best=False):
    """
    Save dict:
    - model_state_dict
    - optimizer_state_dict
    - epoch
    - metrics (val_loss, val_cer, val_wa)
    Nếu is_best → copy sang best_model.pth
    """

def load_checkpoint(model, optimizer, path, device):
    """Load checkpoint, return epoch và metrics."""
```

**Phụ thuộc**: `config.py`.

**Thứ tự implement**: **#12**.

---

#### `src/utils/seed.py`

**Vai trò**: Set random seed cho reproducibility.

**Function chính**:

```python
def set_seed(seed: int = 42):
    """
    Set seed cho:
    - random
    - numpy
    - torch (CPU + CUDA)
    - torch.backends.cudnn (deterministic + benchmark)
    """
```

**Thứ tự implement**: Có thể implement cùng lúc với `config.py`.

---

#### `src/utils/image_utils.py`

**Vai trò**: Helper functions cho xử lý ảnh, debug, visualize.

**Function chính**:

```python
def load_image(path: str) -> np.ndarray:
    """Load ảnh, handle lỗi gracefully."""

def show_image_with_label(image: np.ndarray, label: str):
    """Hiển thị ảnh + label bằng matplotlib. Dùng để debug."""

def save_attention_map(image: np.ndarray, attention_weights: np.ndarray,
                        predicted_chars: List[str], save_path: str):
    """Tạo và lưu attention heatmap overlay lên ảnh gốc."""
```

**Thứ tự implement**: Implement khi cần (parallel với các bước khác).

---

#### `src/train_attention.py`

**Vai trò**: Script chính để train Attention model.

**Input**: `data_processed/train.csv`, `data_processed/val.csv`.

**Output**: Checkpoints trong `checkpoints/`, logs trong `outputs/logs/`.

**Function chính**:

```python
def train_one_epoch(model, dataloader, optimizer, criterion, device, 
                     teacher_forcing_ratio, scaler=None):
    """
    1. model.train()
    2. For each batch:
       a. Forward pass (with AMP if enabled)
       b. Compute CrossEntropyLoss (ignore padding)
       c. Backward + clip grad + optimizer step
    3. Return average loss
    """

def validate(model, dataloader, vocab, device, max_len):
    """
    1. model.eval()
    2. For each batch:
       a. Forward pass (no teacher forcing)
       b. Greedy decode → predicted strings
       c. Collect predictions + targets
    3. Compute metrics (WA, CER)
    4. Return metrics dict + average loss
    """

def main():
    """
    1. Set seed
    2. Load vocab
    3. Create datasets + dataloaders
    4. Create model
    5. Setup optimizer (2 param groups: encoder LR, decoder LR)
    6. Setup scheduler (optional: ReduceLROnPlateau)
    7. Setup AMP scaler
    8. Phase 1: Freeze encoder, train decoder (5-10 epochs)
    9. Phase 2: Unfreeze encoder layer4, fine-tune (remaining epochs)
    10. Each epoch: train → validate → log → save checkpoint
    11. Early stopping if val CER doesn't improve
    12. Print final results
    """
```

**Phụ thuộc**: `config.py`, `data/dataset.py`, `data/vocab.py`, `data/transforms.py`, `models/attention_model.py`, `utils/metrics.py`, `utils/checkpoint.py`, `utils/seed.py`, `inference/greedy_decode.py`.

**Thứ tự implement**: **#13**.

---

#### `src/inference/greedy_decode.py`

**Vai trò**: Greedy decoding cho Attention model.

**Function chính**:

```python
def greedy_decode(model, images, vocab, device, max_len=32):
    """
    1. model.eval()
    2. Encode images
    3. Start với <sos>
    4. For each step:
       a. Decoder forward_step
       b. argmax → next char
       c. Nếu next char == <eos> → stop
    5. Decode indices → text
    6. Return: list of predicted strings, attention weights
    """
```

**Phụ thuộc**: `data/vocab.py`.

**Thứ tự implement**: **#14**.

---

#### `src/inference/beam_search.py`

**Vai trò**: Beam search decoding cho Attention model (cải thiện chất lượng so với greedy).

**Function chính**:

```python
def beam_search_decode(model, image, vocab, device, beam_width=5, max_len=32):
    """
    1. Encode single image
    2. Init beam: [(score=0.0, sequence=[<sos>], hidden=h0)]
    3. For each step:
       a. Expand mỗi beam → vocab_size candidates
       b. Giữ top-k beams theo score
       c. Nếu beam kết thúc (<eos>) → chuyển sang completed
    4. Chọn completed beam có score cao nhất
    5. Return: predicted string, score, attention weights
    """
```

**Phụ thuộc**: `data/vocab.py`.

**Thứ tự implement**: **#15** (sau greedy decode).

---

#### `src/predict_attention.py`

**Vai trò**: Script predict một ảnh đơn lẻ.

**Input**: Path đến ảnh + path đến checkpoint.

**Output**: In ra predicted text, (optional) save attention map.

```python
def predict_single_image(image_path: str, checkpoint_path: str, 
                          save_attention: bool = False):
    """
    1. Load model từ checkpoint
    2. Load + preprocess ảnh
    3. Greedy decode (hoặc beam search)
    4. In kết quả
    5. Save attention map nếu cần
    """

# CLI:
# python src/predict_attention.py --image path/to/image.png --checkpoint checkpoints/best_attention_model.pth
```

**Phụ thuộc**: `config.py`, `models/attention_model.py`, `data/vocab.py`, `data/transforms.py`, `inference/greedy_decode.py`, `inference/beam_search.py`, `utils/image_utils.py`.

**Thứ tự implement**: **#15**.

---

#### `src/evaluate_attention.py`

**Vai trò**: Evaluate Attention model trên toàn bộ test set.

**Input**: `data_processed/test.csv`, checkpoint.

**Output**: `outputs/predictions.csv`, in metrics.

```python
def evaluate_test_set(checkpoint_path: str):
    """
    1. Load model
    2. Load test dataset
    3. Predict toàn bộ test set
    4. Tính metrics (WA, CER, NED)
    5. Ghi predictions.csv:
       image_path, ground_truth, prediction, edit_distance, correct
    6. In summary
    """
```

**Phụ thuộc**: `config.py`, `models/attention_model.py`, `data/dataset.py`, `data/vocab.py`, `utils/metrics.py`, `inference/greedy_decode.py`.

**Thứ tự implement**: **#16**.

---

#### `src/train_ctc_baseline.py`

**Vai trò**: Training loop cho CTC baseline model.

**Input**: `data_processed/train.csv`, `data_processed/val.csv`.

**Output**: `checkpoints/best_ctc_baseline.pth`.

**Cấu trúc tương tự** `train_attention.py` nhưng:
- Dùng `CTCBaseline` model
- Dùng `torch.nn.CTCLoss(blank=0, zero_infinity=True)`
- Decode bằng CTC greedy (argmax + remove blanks + merge repeats)
- Không có teacher forcing

**Phụ thuộc**: `config.py`, `data/dataset.py`, `models/ctc_baseline.py`, `utils/metrics.py`.

**Thứ tự implement**: **#18**.

---

#### `src/evaluate_ctc_baseline.py`

**Vai trò**: Evaluate CTC baseline trên test set.

**Tương tự** `evaluate_attention.py` nhưng dùng CTC decode.

**Thứ tự implement**: **#19**.

---

## 4. Main Model Design

### Kiến trúc tổng thể

```
Input Image (1 × 64 × 256, grayscale)
    │
    ▼
[Repeat channel → 3 × 64 × 256]
    │
    ▼
┌──────────────────────────────────────┐
│  ResNet18 CNN Encoder (pretrained)   │
│  Bỏ avgpool + fc                     │
│  Output: (B, 256, 4, 16)            │
│          nếu dùng đến layer3        │
└──────────────────────────────────────┘
    │
    ▼
[Reshape: (B, 16, 1024)]              ← collapse H' vào features
    │                                     seq_len=16, feat_dim=256×4=1024
    ▼
┌──────────────────────────────────────┐
│  Bahdanau Attention                  │
│  Decoder hidden → attention weights  │
│  → context vector                    │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  GRU Decoder (autoregressive)        │
│  Step t:                             │
│    input = embed(char_{t-1})         │
│    context = attention(hidden, enc)  │
│    hidden = GRU(concat(input, ctx))  │
│    logits = FC(hidden + ctx + input) │
└──────────────────────────────────────┘
    │
    ▼
[Softmax over vocabulary]
    │
    ▼
Output: ký tự từng bước → ghép thành từ
```

### Giải thích các quyết định thiết kế

#### Vì sao dùng ResNet18 làm encoder?

1. **Pretrained trên ImageNet** → đã học feature extraction tốt (edges, textures, shapes). Dù ImageNet là ảnh tự nhiên, các low-level features vẫn hữu ích cho chữ viết tay.
2. **Nhẹ** (11M params) → train nhanh trên RTX 5060 Ti, không lo OOM.
3. **Residual connections** → giúp gradient flow tốt, tránh vanishing gradient.
4. **Phổ biến** → dễ tìm tài liệu, dễ giải thích khi bảo vệ.
5. Không cần backbone nặng hơn (ResNet50, EfficientNet-B4) vì input chỉ là ảnh word nhỏ (64×256).

#### Vì sao dùng Attention Decoder thay vì CTC?

| Khía cạnh | CTC | Attention Decoder |
|---|---|---|
| Context giữa ký tự | ❌ Conditional independence — mỗi time step predict độc lập | ✅ Autoregressive — biết ký tự trước khi predict ký tự sau |
| Xử lý chữ ngoáy | ❌ Feature columns bị nhiễu → sai | ✅ Attention nhìn đúng vùng cần thiết |
| Implicit LM | ❌ Không có | ✅ GRU decoder tự học transition probability |
| Visualize | ❌ Không có gì để show | ✅ Attention heatmap → ấn tượng khi bảo vệ |
| Decode strategy | Greedy/beam search phẳng | Greedy/beam search có context |

#### Vì sao dùng GRU thay vì LSTM?

1. **GRU ít tham số hơn LSTM** (2 gates vs 3 gates) → train nhanh hơn ~20%.
2. **Word-level sequence ngắn** (max 32 chars) → không cần LSTM's cell state cho long-range dependency.
3. **Hiệu quả tương đương LSTM** trên sequence ngắn (empirical evidence từ nhiều paper).
4. **Dễ giải thích hơn** — ít gates hơn, flow đơn giản hơn.
5. Có thể upgrade lên LSTM sau nếu GRU không đủ tốt.

#### Vì sao vẫn cần baseline CTC?

1. **So sánh khách quan**: Chứng minh Attention thực sự tốt hơn, không chỉ "cảm giác".
2. **Yêu cầu bảo vệ**: Hội đồng thường hỏi "So với phương pháp khác thì sao?"
3. **Validation**: Nếu cả Attention và CTC đều kém → vấn đề nằm ở data/preprocessing, không phải model.
4. **Chi phí thấp**: CTC baseline code ít, train nhanh.

#### Vì sao output là character-level thay vì word classification?

1. **Open vocabulary**: IAM có >10,000 unique words, nhiều từ chỉ xuất hiện 1-2 lần. Word classifier không thể predict unseen words.
2. **Generalization**: Character decoder có thể predict bất kỳ tổ hợp ký tự nào, kể cả từ chưa thấy trong training.
3. **Bản chất bài toán**: HTR là sequence prediction (ảnh → chuỗi ký tự), không phải classification.

#### Vì sao transfer learning chỉ ở CNN encoder?

1. **CNN encoder**: Feature extraction là task-agnostic — edges, textures, shapes hữu ích cho mọi loại ảnh. Pretrained ImageNet weights là starting point tốt.
2. **Decoder**: Sinh chuỗi ký tự là task-specific cho HTR. Không có pretrained decoder phù hợp. Decoder nhẹ (chỉ GRU + Attention + FC) → train from scratch với 115K samples là đủ.

---

## 5. Vocabulary Design

### Vocabulary ban đầu (Phase 1)

```
Index 0:  <pad>    ← Padding token
Index 1:  <sos>    ← Start of sequence
Index 2:  <eos>    ← End of sequence
Index 3:  a
Index 4:  b
...
Index 28: z
```

**Tổng vocab size = 29** (3 special + 26 chữ cái).

### Mở rộng sau (Phase 2, optional)

```
Index 29: A-Z   (26 ký tự)
Index 55: 0-9   (10 ký tự)
Index 65: '     (apostrophe)
Index 66: -     (hyphen)
Index 67: .     (period)
Index 68: ,     (comma)
```

> **Khuyến nghị**: Phase 1 chỉ dùng lowercase `a-z` để giảm complexity. Sau khi model chạy ổn, mở rộng thêm. Khi mở rộng, cần retrain hoặc fine-tune.

### Chi tiết implementation

#### `char_to_idx` và `idx_to_char`

```python
# Ví dụ logic (KHÔNG phải code implementation):
chars = "abcdefghijklmnopqrstuvwxyz"
special = ["<pad>", "<sos>", "<eos>"]

char_to_idx = {token: i for i, token in enumerate(special)}
for i, c in enumerate(chars):
    char_to_idx[c] = len(special) + i

idx_to_char = {v: k for k, v in char_to_idx.items()}
```

#### Encode label thành tensor

```
Input text: "hello"
→ encode("hello") = [10, 7, 14, 14, 17]     ← indices cho h, e, l, l, o
→ Khi tạo target cho decoder: [1, 10, 7, 14, 14, 17, 2]   ← thêm <sos> đầu, <eos> cuối

Nhưng thực tế trong implementation:
- Target input cho decoder (teacher forcing): [<sos>, h, e, l, l, o]
- Target output để tính loss:                 [h, e, l, l, o, <eos>]
```

#### Decode tensor thành text

```
Input indices: [10, 7, 14, 14, 17, 2, 0, 0, 0]
→ decode: "hello" (dừng tại index 2 = <eos>, bỏ qua padding)
```

#### Padding label trong batch

Trong một batch, các label có độ dài khác nhau. Pad bằng `<pad>` (index 0) đến max length trong batch:

```
Batch:
  "hello"  → [10, 7, 14, 14, 17, 2, 0, 0]
  "cat"    → [5,  3, 22, 2,  0,  0, 0, 0]
  "world"  → [25, 17, 20, 14, 6, 2, 0, 0]
```

CrossEntropyLoss phải dùng `ignore_index=0` để bỏ qua padding.

#### Xử lý sample không hợp lệ

Ở Phase 1, chỉ giữ sample mà label (sau lowercase) chỉ chứa `[a-z]`:
- `"hello"` → ✅ giữ
- `"Mr."` → ❌ bỏ (có `.` và uppercase, nhưng lowercase "mr." vẫn có `.`)
- `"can't"` → ❌ bỏ (có `'`)
- `"123"` → ❌ bỏ

#### Max label length

- `MAX_LABEL_LENGTH = 32` — đủ cho hầu hết từ tiếng Anh.
- Từ dài hơn 32 chars rất hiếm trong IAM → bỏ qua.
- `MIN_LABEL_LENGTH = 1` — giữ cả từ 1 ký tự ("a", "I").

---

## 6. Preprocessing Pipeline

### Quy trình xử lý ảnh

```
1. Load ảnh (PIL hoặc OpenCV)
2. Convert sang grayscale
3. Resize theo chiều cao cố định (64px), giữ aspect ratio
4. Pad hoặc crop width về 256px
5. Convert grayscale → 3 channels (repeat)
6. Scale pixel [0,255] → [0,1]
7. Normalize theo ImageNet mean/std
8. Output: tensor (3, 64, 256)
```

### Thông số

```
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3     ← Vì ResNet18 pretrained cần 3 channels
```

### Chi tiết từng bước

#### Bước 1-2: Load và grayscale

```
Load ảnh bằng PIL: Image.open(path).convert('L')
Hoặc OpenCV: cv2.imread(path, cv2.IMREAD_GRAYSCALE)
```

#### Bước 3: Resize giữ aspect ratio

```
Ví dụ: ảnh gốc 48×200
- Scale factor = 64 / 48 = 1.333
- Width mới = 200 × 1.333 = 267
- → Resize thành 64×267
```

#### Bước 4: Pad/Crop width

```
Nếu width < 256: pad bên phải bằng pixel trắng (255 cho grayscale)
Nếu width > 256: resize (shrink) để width = 256, hoặc crop
→ Khuyến nghị: resize để không mất thông tin
```

#### Bước 5: Grayscale → 3 channels

```
Cách đơn giản nhất (dùng cho Phase 1):
  image_3ch = np.stack([image_gray, image_gray, image_gray], axis=-1)
  # Kết quả: (64, 256, 3)
```

> **Tại sao không sửa conv1 của ResNet?** Cách repeat channels đơn giản hơn nhiều, pretrained weights vẫn hoạt động vì input 3 channels giống nhau → mean/std normalize vẫn đúng. Sửa conv1 phức tạp hơn và mất pretrained weights của layer đầu.

#### Bước 6-7: Normalize

```
# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# Scale [0,255] → [0,1] rồi normalize:
pixel_normalized = (pixel / 255.0 - mean) / std
```

### Lưu ý quan trọng

- **Không dùng hard thresholding** (binarization cứng) ở phase đầu — dễ mất nét chữ mảnh.
- **Không aggressive denoise** — có thể làm mất chi tiết chữ ngoáy.
- Nếu sau này muốn thêm preprocessing, thêm dưới dạng augmentation (chỉ khi train).

---

## 7. Data Augmentation

### Augmentation cho Training

| # | Augmentation | Tham số | Mô phỏng |
|---|---|---|---|
| 1 | **Random Rotation** | angle ±8° | Chữ viết nghiêng |
| 2 | **Affine (Shear)** | shear_x ±10° | Chữ ngoáy, italic |
| 3 | **Gaussian Blur** | kernel 3, sigma 0.1–1.5 | Ảnh mờ |
| 4 | **Gaussian Noise** | var_limit (10, 50) | Noise từ scanner/camera |
| 5 | **Brightness + Contrast** | ±20% | Điều kiện ánh sáng khác nhau |
| 6 | **Erosion nhẹ** | kernel 2×2, iterations=1 | Nét mực đậm hơn |
| 7 | **Dilation nhẹ** | kernel 2×2, iterations=1 | Nét mực nhạt hơn |
| 8 | **Random Scale** | scale 0.9–1.1 | Ảnh thu/phóng nhẹ |

Mỗi augmentation áp dụng với **probability ~0.3–0.5** (không áp dụng tất cả mọi lúc).

### Augmentation KHÔNG nên dùng

| Augmentation | Lý do tránh |
|---|---|
| **Horizontal Flip** | Chữ bị ngược → vô nghĩa ("hello" → "olleh" visual) |
| **Vertical Flip** | Chữ bị lộn ngược → vô nghĩa |
| **Rotation > 15°** | Chữ quá nghiêng, không thực tế |
| **Large Random Crop** | Cắt mất ký tự đầu/cuối → label sai |
| **Color Jitter mạnh** | Chữ viết tay thường grayscale, không cần |
| **Cutout / Random Erasing lớn** | Xóa mất phần chữ → label sai |

### Pipeline augmentation bằng Albumentations

```python
# Pseudo-code cho training transform:
train_aug = A.Compose([
    A.Rotate(limit=8, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255),
    A.Affine(shear=(-10, 10), p=0.3, border_mode=cv2.BORDER_CONSTANT, cval=255),
    A.GaussianBlur(blur_limit=3, sigma_limit=(0.1, 1.5), p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    # Erosion/Dilation qua Morphological transforms
    A.OneOf([
        A.Morphological(scale=(2, 3), operation='erosion', p=1.0),
        A.Morphological(scale=(2, 3), operation='dilation', p=1.0),
    ], p=0.2),
])
```

> **Lưu ý**: Augmentation chỉ áp dụng khi **training**. Validation và test dùng transform chuẩn (chỉ resize + normalize).

---

## 8. Data Preparation

### Script `src/prepare_data.py`

#### Workflow chi tiết

```
1. Đọc dataset/label.txt
2. Parse từng dòng → (image_path, raw_label)
3. Xây dựng full path: dataset/ + image_path (hoặc dataset/words/... tùy format)
4. Decode HTML entities: &apos; → ', &quot; → "
5. Lowercase label
6. Validate:
   a. File ảnh tồn tại?
   b. Label chỉ chứa [a-z]? (Phase 1)
   c. 1 ≤ len(label) ≤ 32?
7. Nếu invalid → log lý do, skip
8. Nếu valid → thêm vào list
9. Shuffle list (với seed cố định)
10. Split 80/10/10
11. Ghi train.csv, val.csv, test.csv
12. In summary
```

#### Format output CSV

```csv
image_path,label
dataset/words/a01/a01-000u/a01-000u-00-02.png,to
dataset/words/a01/a01-000u/a01-000u-00-03.png,stop
dataset/words/a01/a01-000u/a01-000u-01-00.png,nominating
```

- `image_path`: Relative path từ project root `D:\Downloads\HRT-Project\`.
- `label`: Đã lowercase, đã validate.

#### Xử lý path

Label file chứa path dạng `words/a01/a01-000u/a01-000u-00-00.png`. Full path cần build:

```python
full_path = os.path.join(DATASET_DIR, image_relative_path)
# = "dataset/words/a01/a01-000u/a01-000u-00-00.png"
```

Trong CSV, lưu path relative từ project root:

```python
csv_path = os.path.join("dataset", image_relative_path)
```

#### Log output mong muốn

```
============================================================
               DATA PREPARATION SUMMARY
============================================================
Total lines in label.txt:          115,319
------------------------------------------------------------
Valid samples:                      XX,XXX
Skipped - image not found:          X,XXX
Skipped - label empty/whitespace:       XX
Skipped - label has invalid chars:  XX,XXX
Skipped - label too short (<1):          X
Skipped - label too long (>32):        XXX
Skipped - image cannot be opened:       XX
------------------------------------------------------------
Train set:                          XX,XXX  (80.0%)
Validation set:                      X,XXX  (10.0%)
Test set:                            X,XXX  (10.0%)
------------------------------------------------------------
Files saved:
  → data_processed/train.csv
  → data_processed/val.csv
  → data_processed/test.csv
============================================================
```

---

## 9. Training Plan For Attention Model

### Training Loop Overview

```
For each epoch:
    1. Train one epoch (forward + backward + update)
    2. Validate (forward only, compute metrics)
    3. Log metrics (loss, WA, CER)
    4. Save checkpoint
    5. Check early stopping
    6. Adjust teacher forcing ratio
    7. (Phase transition: unfreeze encoder at epoch X)
```

### Loss Function

```python
criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
```

- Input: `logits (B, T, vocab_size)` → reshape thành `(B*T, vocab_size)`
- Target: `targets (B, T)` → reshape thành `(B*T,)`
- Padding positions (index 0) được ignore.

### Optimizer

```python
optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-3},
    {'params': model.encoder.parameters(), 'lr': 1e-5},  # Phase 2 mới enable
], weight_decay=1e-5)
```

Phase 1: Chỉ encoder params có `requires_grad=False` → optimizer không update chúng.

### Scheduler (optional)

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)
# Giảm LR khi val_loss không giảm sau 5 epochs
```

### Mixed Precision (AMP)

RTX 5060 Ti hỗ trợ FP16 (và FP8 trên Blackwell arch) → giảm memory usage, tăng tốc training.

```python
scaler = torch.cuda.amp.GradScaler()

# Trong training loop:
with torch.cuda.amp.autocast():
    logits, attn_weights = model(images, targets, teacher_forcing_ratio)
    loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
scaler.step(optimizer)
scaler.update()
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

Ngăn exploding gradients, đặc biệt quan trọng cho RNN-based decoder.

### Teacher Forcing Schedule

```
Epoch 1:  tf_ratio = 0.5
Epoch 2:  tf_ratio = 0.49
...
Epoch 50: tf_ratio = 0.01

Formula: tf_ratio = max(0.01, 0.5 - epoch * 0.01)
```

Giảm dần teacher forcing giúp model quen với việc tự sinh ký tự (giảm exposure bias).

### Training Phases

#### Phase 1: Freeze Encoder (Epochs 1–10)

```
- Freeze toàn bộ ResNet18 encoder
- Chỉ train: Embedding + Attention + GRU + FC layers
- Learning rate: 1e-3 (chỉ decoder)
- Mục tiêu: Decoder học cách decode feature map → characters
```

#### Phase 2: Fine-tune Encoder (Epochs 11–80)

```
- Unfreeze ResNet18 từ layer3 trở đi (layer3 + layer4)
- Train toàn bộ model
- Encoder LR: 1e-5 (rất thấp để không phá pretrained weights)
- Decoder LR: giữ nguyên hoặc giảm nhẹ
- Mục tiêu: CNN encoder adapt features cho handwriting domain
```

#### Phase 3 (Optional): Full fine-tune

```
- Unfreeze toàn bộ ResNet18
- LR rất thấp: 1e-6 cho encoder, 1e-5 cho decoder
- Chỉ chạy nếu Phase 2 chưa đủ tốt
```

### Early Stopping

```
patience = 10 epochs
monitor = val_cer (Character Error Rate)
mode = min (lower is better)

Nếu val_cer không giảm sau 10 epochs liên tiếp → dừng training.
```

### Checkpoint Strategy

Mỗi epoch save:
- `last_attention_model.pth` — always overwrite
- `best_attention_model.pth` — chỉ overwrite khi val_cer cải thiện

Checkpoint content:
```python
{
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    'val_loss': val_loss,
    'val_cer': val_cer,
    'val_wa': val_wa,
    'vocab_chars': vocab.chars,  # Để reconstruct vocab khi load
}
```

### Hyperparameters tổng hợp

| Parameter | Giá trị | Ghi chú |
|---|---|---|
| `BATCH_SIZE` | 64 | Có thể tăng lên 128 với 16GB VRAM; giảm xuống 32 nếu OOM |
| `NUM_EPOCHS` | 80 | Phase 1: 10, Phase 2: 70 |
| `LR_DECODER` | 1e-3 | |
| `LR_ENCODER` | 1e-5 | Chỉ dùng từ Phase 2 |
| `EMBED_DIM` | 128 | |
| `DECODER_HIDDEN_DIM` | 256 | |
| `ATTENTION_DIM` | 256 | |
| `MAX_DECODE_LEN` | 32 | |
| `TEACHER_FORCING_RATIO` | 0.5 → giảm dần | |
| `GRAD_CLIP` | 5.0 | |
| `DROPOUT` | 0.3 | |
| `WEIGHT_DECAY` | 1e-5 | |
| `EARLY_STOPPING_PATIENCE` | 10 | |
| `USE_AMP` | True | Mixed precision |

---

## 10. Inference Plan

### Greedy Decoding

```
1. Load checkpoint tốt nhất (best_attention_model.pth)
2. model.eval() + torch.no_grad()
3. Load ảnh → preprocess (giống val transform, KHÔNG augment)
4. images → device
5. encoder_outputs = model.encoder(images)
6. Reshape encoder_outputs
7. hidden = decoder.init_hidden(encoder_outputs)
8. input_token = <sos>
9. predicted_chars = []
10. For step in range(MAX_DECODE_LEN):
      a. logits, hidden, attn_weights = decoder.forward_step(input_token, hidden, encoder_outputs)
      b. next_char_idx = argmax(logits)
      c. Nếu next_char_idx == <eos> → break
      d. predicted_chars.append(idx_to_char[next_char_idx])
      e. input_token = next_char_idx
11. predicted_word = "".join(predicted_chars)
12. Return predicted_word, all_attention_weights
```

### Beam Search Decoding

```
1. Tương tự bước 1-8 của greedy
2. Init beams: [(score=0.0, sequence=[<sos>], hidden=h0, attn_history=[])]
3. For step in range(MAX_DECODE_LEN):
     For each active beam:
       a. forward_step → logits (vocab_size,)
       b. log_probs = log_softmax(logits)
       c. Top-k candidates = beam.score + log_probs[k]
     Collect all candidates from all beams
     Keep top beam_width candidates
     Move finished beams (với <eos>) sang completed list
4. Chọn completed beam có best score / length_normalized_score
5. Return best prediction
```

**Beam width khuyến nghị**: 3 hoặc 5.

**Length normalization** (optional): `score / len(sequence)^alpha` với alpha ≈ 0.6–0.7.

### So sánh Greedy vs Beam Search

| Aspect | Greedy | Beam Search |
|---|---|---|
| Tốc độ | Nhanh | Chậm hơn ~beam_width lần |
| Chất lượng | Tốt cho từ ngắn | Tốt hơn cho từ dài |
| Implement | Đơn giản | Phức tạp hơn |
| Khuyến nghị | Implement trước, dùng làm default | Implement sau, so sánh với greedy |

---

## 11. Evaluation Plan

### Metrics chi tiết

#### Word Accuracy (WA)

```
WA = (Số từ predict đúng hoàn toàn) / (Tổng số từ) × 100%

Ví dụ:
  GT:   ["hello", "world", "test"]
  Pred: ["hello", "worid", "test"]
  WA = 2/3 = 66.67%
```

#### Character Error Rate (CER)

```
CER = Σ edit_distance(pred_i, gt_i) / Σ len(gt_i) × 100%

Ví dụ:
  GT:   ["hello", "world"]        → tổng len = 5 + 5 = 10
  Pred: ["helo",  "worid"]        → ED = 1 + 1 = 2
  CER = 2/10 = 20%
```

#### Normalized Edit Distance (NED)

```
NED = 1 - (1/N) × Σ edit_distance(pred_i, gt_i) / max(len(pred_i), len(gt_i))

Ví dụ:
  ("helo", "hello"):  ED=1, max_len=5, normalized=0.2
  ("worid", "world"): ED=1, max_len=5, normalized=0.2
  NED = 1 - (0.2 + 0.2)/2 = 0.8 = 80%
```

### Output file: `outputs/predictions.csv`

```csv
image_path,ground_truth,prediction,edit_distance,correct
dataset/words/a01/a01-003/a01-003-00-00.png,though,though,0,True
dataset/words/a01/a01-003/a01-003-00-01.png,they,thay,1,False
dataset/words/a01/a01-003/a01-003-00-02.png,may,may,0,True
```

### Evaluation workflow (`evaluate_attention.py`)

```
1. Load best checkpoint
2. Load test.csv → Dataset → DataLoader
3. For each batch:
     a. Greedy decode → predictions
     b. Collect (image_path, ground_truth, prediction)
4. Compute metrics: WA, CER, NED
5. Ghi predictions.csv
6. In summary:

============================================================
          EVALUATION RESULTS - ATTENTION MODEL
============================================================
Test samples:                    8,246
Word Accuracy:                   XX.XX%
Character Error Rate:            XX.XX%
Normalized Edit Distance:        XX.XX%
------------------------------------------------------------
Predictions saved to: outputs/predictions.csv
============================================================
```

---

## 12. Attention Visualization

### Mục tiêu

Tạo hình ảnh trực quan cho thấy model "nhìn vào đâu" trên ảnh gốc khi sinh từng ký tự. Đây là **điểm mạnh nhất** khi bảo vệ đồ án.

### Workflow

```
1. Predict một ảnh bằng attention model
2. Thu thập attention weights: (num_chars, seq_len) 
   → mỗi hàng là distribution attention cho 1 ký tự output
3. Upsample attention weights về kích thước ảnh gốc (width dimension)
4. Tạo heatmap cho mỗi ký tự:
   a. Lấy attention row cho ký tự t
   b. Reshape thành 1D vector theo width
   c. Expand thành 2D heatmap (repeat theo height)
   d. Overlay lên ảnh gốc bằng colormap (jet/viridis)
5. Tạo grid: ảnh gốc ở trên, các heatmap ở dưới (mỗi ký tự 1 panel)
6. Annotate: ghi ký tự predicted bên trên mỗi panel
7. Save vào outputs/attention_maps/
```

### Ví dụ output

```
Ảnh: "hello"

Panel 1 (char 'h'): Attention tập trung ở bên trái ảnh
Panel 2 (char 'e'): Attention dịch sang phải một chút
Panel 3 (char 'l'): Attention ở giữa ảnh
Panel 4 (char 'l'): Attention vẫn ở giữa (hơi dịch phải)
Panel 5 (char 'o'): Attention ở bên phải ảnh
```

### File output

```
outputs/attention_maps/
├── sample_001_hello.png
├── sample_002_world.png
├── sample_003_test.png
└── ...
```

### Số lượng cần tạo

- **Tối thiểu 10-20 ảnh** cho báo cáo.
- Chọn cả samples đúng và sai để phân tích.
- Chọn từ ngắn (2-3 chars), trung bình (5-6 chars), dài (8+ chars).

---

## 13. CTC Baseline Plan

### Kiến trúc

```
Input Image (1 × 64 × 256)
    │
    ▼
[Simple CNN hoặc ResNet18 modified]
    │ Output: (B, C, H', W')
    ▼
[Reshape: collapse H' → (B, W', C×H')]     ← sequence theo width
    │
    ▼
[BiLSTM (2 layers, hidden=256)]
    │ Output: (B, W', 512)                   ← 256*2 vì bidirectional
    ▼
[Linear(512, vocab_size_ctc)]
    │
    ▼
[Log Softmax]
    │
    ▼
[CTC Loss]
```

### Lưu ý về vocabulary cho CTC

CTC cần **blank token** (thường index 0):

```
CTC vocab:
  Index 0: <blank>
  Index 1: a
  ...
  Index 26: z
```

Khác với Attention vocab (có `<pad>`, `<sos>`, `<eos>`).

→ Implementation cần xử lý 2 vocab mapping riêng, hoặc thiết kế `Vocabulary` class đủ flexible.

### CTC Decoding

```python
def ctc_greedy_decode(log_probs):
    """
    1. argmax mỗi time step → sequence of indices
    2. Remove consecutive duplicates
    3. Remove blank tokens
    4. Map indices → characters
    """
```

### Training CTC Baseline

```python
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

# CTC loss input:
# - log_probs: (T, B, vocab_size) — T là sequence length (chiều width)
# - targets: (B, S) — S là target length (flatten hoặc packed)
# - input_lengths: (B,) — T cho mỗi sample
# - target_lengths: (B,) — len(label) cho mỗi sample
```

### Yêu cầu CTC baseline

- Dùng **cùng dataset split** (train.csv, val.csv, test.csv) với attention model.
- Dùng **cùng preprocessing** (resize, normalize).
- Dùng **cùng augmentation** (nếu có).
- Dùng **cùng metrics** (WA, CER, NED).
- **Không cần attention visualization** (CTC không có attention).

### Kết quả mong đợi

CTC baseline sẽ có WA và CER **kém hơn** Attention model. Sự khác biệt này chính là contribution chính của đồ án:

```
============================================================
          MODEL COMPARISON
============================================================
                        CTC Baseline    Attention Model
Word Accuracy:          XX.XX%          YY.YY% (↑)
Character Error Rate:   XX.XX%          YY.YY% (↓)
NED:                    XX.XX%          YY.YY% (↑)
============================================================
```

---

## 14. RTX 5060 Ti Local Setup

### Phần cứng thực tế

| Component | Giá trị |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| VRAM | 16 GB (16311 MiB) |
| Driver Version | 581.80 |
| nvidia-smi CUDA Version | 13.0 |
| Kiến trúc GPU | Blackwell |
| Python | 3.11 (khuyến nghị, dùng conda) |
| OS | Windows |

> **Giải thích về CUDA Version 13.0**: Giá trị `CUDA Version: 13.0` hiển thị bởi `nvidia-smi` là **phiên bản CUDA tối đa mà driver hỗ trợ**, không phải phiên bản CUDA Toolkit đã cài. PyTorch CUDA wheels đóng gói sẵn CUDA runtime riêng, nên không cần cài CUDA Toolkit hệ thống. Ta chỉ cần chọn PyTorch wheel có CUDA version ≤ 13.0.

### Ràng buộc ổ đĩa

- **Drive D** gần đầy → **KHÔNG** tạo conda env hoặc venv trên drive D.
- Conda environment sẽ được tạo trên **drive C**: `C:\conda_envs\hrt`
- Project folder vẫn nằm trên drive D: `D:\Downloads\HRT-Project`

### Setup conda environment

```bat
REM Tạo thư mục chứa env trên drive C
mkdir C:\conda_envs

REM Tạo conda env với Python 3.11
conda create --prefix C:\conda_envs\hrt python=3.11 -y

REM Activate env
conda activate C:\conda_envs\hrt

REM Kiểm tra
python --version
where python
python -m pip --version
```

### Cài PyTorch với CUDA

PyTorch phải được cài **riêng**, không qua `requirements.txt`.

Cách xác định command cài đặt đúng:
1. Truy cập [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
2. Chọn: Stable → Windows → Pip → Python → CUDA
3. Vì đây là GPU RTX 50-series (Blackwell), ưu tiên **CUDA 12.8 (cu128)** nếu có
4. Nếu stable chỉ có cu126, dùng cu126
5. Chỉ xem xét nightly nếu stable PyTorch không detect được RTX 5060 Ti

```bat
REM Ví dụ — thay bằng command chính thức từ pytorch.org nếu khác:
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

REM Nếu cu128 chưa có ở stable, thử cu126:
REM python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> **Quan trọng**: KHÔNG `pip install torch` đơn thuần — sẽ cài bản CPU-only.

### Cài dependencies còn lại

```bat
cd /d D:\Downloads\HRT-Project
conda activate C:\conda_envs\hrt
python -m pip install -r requirements.txt
```

### Kiểm tra GPU

```bat
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import torch; x=torch.randn(1024,1024,device='cuda'); y=x@x; print(y.shape); print('CUDA tensor test passed')"
python -c "import torch, torchvision; print('torchvision:', torchvision.__version__); m=torchvision.models.resnet18(weights=None).cuda(); x=torch.randn(2,3,64,256).cuda(); y=m(x); print('resnet output shape:', y.shape); print('ResNet CUDA test passed')"
```

Kết quả mong đợi:
- `torch.cuda.is_available()` → `True`
- GPU name → `NVIDIA GeForce RTX 5060 Ti`
- CUDA tensor test → passed
- ResNet CUDA test → passed

### Yêu cầu trong code

#### Device management

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tất cả tensors phải đưa lên device:
images = images.to(device)
targets = targets.to(device)
model = model.to(device)
```

#### DataLoader Windows caveats

```python
# Trên Windows, num_workers > 0 CÓ THỂ gây lỗi multiprocessing.
# Khuyến nghị ban đầu: num_workers=0
# Nếu muốn tăng tốc: thử num_workers=2 với if __name__ == '__main__': guard

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,          # An toàn cho Windows
    pin_memory=True,        # Tăng tốc CPU→GPU transfer
    collate_fn=collate_fn,
)
```

#### Mixed Precision Training

```python
# RTX 5060 Ti hỗ trợ FP16 (và FP8 trên Blackwell arch)
# → giảm ~30-40% memory, tăng tốc ~20-30%
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### Ước tính memory usage

```
ResNet18 encoder: ~44 MB
GRU Decoder + Attention: ~5 MB
Batch 64 images (3×64×256, FP32): ~12 MB
Feature maps + gradients: ~200-500 MB
---
Total estimate: ~500 MB - 1 GB (FP32), ~300-600 MB (FP16)
→ RTX 5060 Ti 16GB: RẤT THOẢI MÁI — có thể tăng batch_size lên 128 nếu cần
```

Nếu OOM (out of memory) — rất ít khả năng với 16GB:
1. Giảm `BATCH_SIZE` từ 64 → 32 → 16
2. Bật AMP (mixed precision)
3. Giảm `DECODER_HIDDEN_DIM` từ 256 → 128

---

## 15. Requirements

### File `requirements.txt` (chỉ chứa non-PyTorch dependencies)

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

> **Quan trọng**: `torch`, `torchvision`, `torchaudio` **KHÔNG** nằm trong `requirements.txt`. Lý do: PyTorch CUDA phải được cài riêng bằng command chuyên dụng từ pytorch.org với đúng CUDA index URL. Nếu để trong requirements.txt, pip sẽ cài bản CPU-only.

### Cài đặt

```bat
REM Bước 1: Activate conda env
conda activate C:\conda_envs\hrt

REM Bước 2: Cài PyTorch CUDA riêng (xem Section 14 để chọn đúng command)
REM Ví dụ:
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

REM Bước 3: Cài các dependencies còn lại
cd /d D:\Downloads\HRT-Project
python -m pip install -r requirements.txt
```

### Giải thích từng package

| Package | Vai trò |
|---|---|
| `torch` | Framework deep learning chính (**cài riêng, không qua requirements.txt**) |
| `torchvision` | Pretrained models (ResNet18), transforms (**cài riêng**) |
| `torchaudio` | Dependency của torch (**cài riêng**) |
| `opencv-python` | Đọc/xử lý ảnh, morphological operations |
| `pillow` | Đọc ảnh (backup cho OpenCV) |
| `numpy` | Array operations |
| `pandas` | Đọc/ghi CSV |
| `scikit-learn` | Train/test split |
| `tqdm` | Progress bar cho training |
| `matplotlib` | Visualize attention maps, training curves |
| `albumentations` | Data augmentation pipeline |
| `editdistance` | Tính Levenshtein distance cho CER |
| `tensorboard` | Logging training metrics, loss curves |
| `torchinfo` | Model summary (params, layers, FLOPS) |

---

## 16. Implementation Order

### Thứ tự implement và điều kiện hoàn thành

| # | File | Điều kiện hoàn thành |
|---|---|---|
| **1** | `requirements.txt` | File tồn tại. Chạy `pip install -r requirements.txt` thành công. `import torch; print(torch.cuda.is_available())` → `True`. |
| **2** | `src/config.py` | File tồn tại. Import được từ các file khác. Tất cả paths, hyperparameters có giá trị hợp lệ. Tự tạo thư mục output nếu chưa có. |
| **3** | `src/data/vocab.py` | `Vocabulary` class hoạt động. `encode("hello")` trả về list of ints. `decode([...])` trả về string. `is_valid_label()` hoạt động đúng. Roundtrip test: `decode(encode(text)) == text`. |
| **4** | `src/prepare_data.py` | Chạy script thành công. Tạo ra `data_processed/train.csv`, `val.csv`, `test.csv`. In summary log. Mọi image_path trong CSV đều tồn tại. Mọi label đều valid theo vocab. |
| **5** | `src/data/transforms.py` | `get_train_transform()` trả về augmentation pipeline. `get_val_transform()` trả về transform không augment. Ảnh output có shape `(3, 64, 256)` đúng. |
| **6** | `src/data/dataset.py` | `HTRDataset` load được từ CSV. `__getitem__` trả về `(image_tensor, target_tensor, target_length)`. `collate_fn` tạo batch đúng shape. DataLoader iterate được không lỗi. |
| **7** | `src/models/resnet_encoder.py` | `ResNetEncoder` forward pass: input `(B,3,64,256)` → output `(B,C,H',W')`. `freeze()`/`unfreeze_from()` hoạt động. Pretrained weights load thành công. |
| **8** | `src/models/attention.py` | `BahdanauAttention` forward: `(B,dec_dim)` + `(B,seq,enc_dim)` → `(B,enc_dim)` + `(B,seq)`. Attention weights sum = 1 per sample. |
| **9** | `src/models/gru_decoder.py` | `GRUDecoder` forward: input `(B,seq,enc_dim)` + targets → `(B,T,vocab)` + `(B,T,seq)`. Teacher forcing hoạt động. Greedy mode (no target) hoạt động. |
| **10** | `src/models/attention_model.py` | `AttentionHTR` forward: `(B,3,64,256)` → logits + attn_weights. `freeze_encoder()` / `unfreeze_encoder()` hoạt động. End-to-end forward pass không lỗi. |
| **11** | `src/utils/metrics.py` | `word_accuracy()` tính đúng. `character_error_rate()` tính đúng. `compute_all_metrics()` trả về dict đúng. Unit test vài ví dụ. |
| **12** | `src/utils/checkpoint.py` | `save_checkpoint()` tạo file `.pth`. `load_checkpoint()` load lại model đúng. Model sau load predict giống trước save. |
| **13** | `src/train_attention.py` | Training loop chạy được ít nhất 1 epoch không lỗi. Loss giảm. Validation metrics tính được. Checkpoint save được. AMP hoạt động (nếu enable). GPU utilization > 0%. |
| **14** | `src/inference/greedy_decode.py` | `greedy_decode()` trả về list of strings + attention weights. Decode dừng tại `<eos>`. Output strings hợp lệ (chỉ chứa chars trong vocab). |
| **15** | `src/predict_attention.py` | Chạy `python src/predict_attention.py --image path.png --checkpoint ckpt.pth` → in predicted text. |
| **16** | `src/evaluate_attention.py` | Chạy evaluate trên test set. In metrics (WA, CER, NED). Ghi `outputs/predictions.csv` đúng format. |
| **17** | `src/models/ctc_baseline.py` | `CTCBaseline` forward: `(B,3,64,256)` → `(T,B,vocab)` log-probs. CTC loss tính được. |
| **18** | `src/train_ctc_baseline.py` | CTC baseline train được. Loss giảm. Checkpoint save được. |
| **19** | `src/evaluate_ctc_baseline.py` | CTC baseline evaluate trên test set. Metrics tính được. So sánh với Attention model. |
| **20** | `README.md` | Có hướng dẫn setup, train, evaluate, predict. Có bảng kết quả. Có ví dụ command. |

### Dependency Graph

```
config.py
    ↓
vocab.py ← prepare_data.py → [train.csv, val.csv, test.csv]
    ↓
transforms.py
    ↓
dataset.py
    ↓
resnet_encoder.py ─┐
attention.py ──────┤
gru_decoder.py ────┘
    ↓
attention_model.py
    ↓
metrics.py ─────┐
checkpoint.py ──┤
seed.py ────────┤
greedy_decode.py┘
    ↓
train_attention.py
    ↓
evaluate_attention.py
predict_attention.py
beam_search.py
    ↓
ctc_baseline.py
train_ctc_baseline.py
evaluate_ctc_baseline.py
    ↓
README.md
```

---

## 17. Risk Management

| # | Rủi ro | Xác suất | Tác động | Giải pháp |
|---|---|---|---|---|
| 1 | **Label sai format** — dòng không có TAB, có nhiều TAB, encoding lỗi | Trung bình | Crash `prepare_data.py` | Try-except mỗi dòng, skip + log nếu lỗi. Không crash cả pipeline. |
| 2 | **Path ảnh không đúng** — path trong label.txt không match file thật | Trung bình | Thiếu data | `os.path.exists()` check. Log missing files. |
| 3 | **Ảnh kích thước quá lạ** — quá nhỏ (5×5), quá lớn (4000×4000), hoặc corrupt | Thấp | Crash DataLoader | Check min size sau load. Try-except trong `__getitem__`, fallback sang sample khác. |
| 4 | **Label chứa ký tự ngoài vocab** — số, dấu câu, ký tự đặc biệt | Cao | Giảm dataset size | Phase 1: lọc bỏ. Phase 2: mở rộng vocab. Đã handle trong `prepare_data.py`. |
| 5 | **Overfitting** — train loss rất thấp nhưng val loss tăng | Trung bình | Model không generalize | Augmentation mạnh hơn. Dropout 0.3-0.5. Early stopping. Reduce model size. |
| 6 | **Exposure bias** (Attention) — model rely teacher forcing, inference khác train | Trung bình | Inference kém hơn validation | Scheduled sampling (giảm tf_ratio dần). Fine-tune với tf_ratio=0 cuối training. |
| 7 | **CTC decode sai nhiều** — baseline rất kém | Cao | Không phải rủi ro — thể hiện lợi thế Attention | Đây là kết quả mong đợi. Đảm bảo CTC code đúng để comparison fair. |
| 8 | **GPU OOM** — batch quá lớn hoặc model quá nặng | Rất thấp | Crash training | Giảm batch_size. Bật AMP. Giảm hidden_dim. RTX 5060 Ti 16GB rất thoải mái cho model này. |
| 9 | **DataLoader Windows lỗi multiprocessing** | Trung bình | Crash hoặc freeze | Dùng `num_workers=0`. Thêm `if __name__ == '__main__':` guard. |
| 10 | **Training loss giảm nhưng WA không tăng** | Trung bình | Metric misleading | Monitor CER song song. WA khắt khe (exact match), CER thực tế hơn. Kiểm tra decode logic. |
| 11 | **Test ảnh thật khác domain IAM** — chữ viết tay ngoài đời vs scan sạch IAM | Cao | Model kém trên ảnh thật | Augmentation mạnh (blur, noise, rotation). Sau này có thể fine-tune trên data riêng. Ghi nhận trong báo cáo là limitation. |
| 12 | **ResNet18 feature map resolution quá thấp** — sequence length quá ngắn cho Attention | Trung bình | Attention không đủ resolution | Dùng đến layer3 thay vì layer4. Hoặc giảm stride. Test cả 2 config. |
| 13 | **HTML entities trong label** — `&apos;`, `&quot;` không được decode | Trung bình | Label bị sai | Dùng `html.unescape()` trước khi process label. |

---

## 18. Final Deliverable Checklist

### Checklist bắt buộc

- [ ] `data_processed/train.csv`, `val.csv`, `test.csv` — đã tạo, valid
- [ ] Attention model train được — loss giảm qua các epoch
- [ ] Checkpoint save được — `best_attention_model.pth` tồn tại
- [ ] Predict một ảnh đơn lẻ — `predict_attention.py` chạy thành công
- [ ] Evaluate test set — `evaluate_attention.py` chạy, in metrics
- [ ] Có Word Accuracy (WA) trên test set
- [ ] Có Character Error Rate (CER) trên test set
- [ ] Có Normalized Edit Distance (NED) trên test set
- [ ] CTC baseline train được — loss giảm
- [ ] CTC baseline evaluate được — có metrics
- [ ] Bảng so sánh Attention vs CTC — WA, CER, NED
- [ ] Ít nhất 10 ảnh attention map — saved trong `outputs/attention_maps/`
- [ ] `outputs/predictions.csv` — predictions trên test set
- [ ] `README.md` — hướng dẫn setup, train, evaluate, predict
- [ ] Không dùng OCR API bên thứ ba
- [ ] Tất cả code chạy trên RTX 5060 Ti không OOM

### Checklist nâng cao (nice-to-have)

- [ ] Beam search decode hoạt động
- [ ] So sánh greedy vs beam search results
- [ ] Training curves (loss, CER theo epoch) plotted
- [ ] Confusion analysis — từ nào hay sai nhất
- [ ] Augmentation ablation — so sánh có/không augmentation
- [ ] Phase 1 vs Phase 2 comparison — freeze vs fine-tune encoder
- [ ] Thử predict trên ảnh chữ viết tay thật (ngoài IAM)

---

## Kết Thúc

File này là blueprint đầy đủ để implement toàn bộ project HRT-Project. Mọi quyết định thiết kế, hyperparameter, và thứ tự implement đã được mô tả chi tiết.

**Bước tiếp theo**: Bắt đầu implement theo thứ tự trong [Section 16](#16-implementation-order), từ `requirements.txt` → `config.py` → ... → `README.md`.
