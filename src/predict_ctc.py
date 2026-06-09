import os
import sys
import io
import argparse
import cv2
import torch
import numpy as np

# Ép kiểu stdout sử dụng UTF-8 trên Windows để tránh crash khi in tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import src.config as config
from src.data.vocab import Vocabulary
from src.data.transforms import resize_and_pad, to_tensor_and_normalize
from src.models.ctc_baseline import CTCBaseline
from src.inference.ctc_decode import ctc_decode
from src.utils.checkpoint import load_checkpoint

def predict_single_image(image_path: str, checkpoint_path: str):
    """
    Dự đoán nhãn cho một ảnh viết tay đơn lẻ sử dụng mô hình CTC Baseline.
    """
    device = torch.device(config.DEVICE)
    print(f"Chạy inference CTC trên thiết bị: {device}")
    
    # 1. Khởi tạo vocab và tải mô hình
    vocab = Vocabulary(config.VOCAB_CHARS)
    
    model = CTCBaseline(
        vocab_size=vocab.size,
        hidden_dim=config.CTC_HIDDEN_DIM,
        num_layers=config.CTC_NUM_LAYERS
    ).to(device)
    
    # Nạp trọng số từ checkpoint
    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    
    # 2. Đọc và tiền xử lý hình ảnh
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {image_path}")
        
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise ValueError(f"Không thể giải mã ảnh: {image_path}")
        
    # Resize giữ aspect ratio và pad về kích thước trong config
    image_processed = resize_and_pad(image_gray, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    
    # Chuyển thành tensor và thêm chiều batch: (1, 3, H, W)
    image_tensor = to_tensor_and_normalize(image_processed).unsqueeze(0).to(device)
    
    # 3. Giải mã nhãn (Inference)
    print("Đang giải mã...")
    with torch.no_grad():
        # log_probs shape: (T, B, vocab_size)
        log_probs = model(image_tensor)
        
        # Giải mã bằng CTC Greedy Decoding
        decoded_words = ctc_decode(log_probs, vocab)
        
    predicted_word = decoded_words[0]
    
    print("-" * 40)
    print(f"Ảnh: {image_path}")
    print(f"Từ dự đoán (CTC): '{predicted_word}'")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict single handwriting image using CTC Baseline Model.")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến file ảnh cần nhận dạng.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_ctc_baseline.pth", 
                        help="Đường dẫn đến file checkpoint (.pth).")
    
    args = parser.parse_args()
    
    img_path = os.path.abspath(args.image)
    ckpt_path = os.path.abspath(args.checkpoint)
    
    predict_single_image(img_path, ckpt_path)
