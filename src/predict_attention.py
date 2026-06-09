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
from src.models.attention_model import AttentionHTR
from src.inference.greedy_decode import greedy_decode
from src.utils.image_utils import save_attention_map
from src.utils.checkpoint import load_checkpoint

def predict_single_image(image_path: str, checkpoint_path: str, save_attention: bool = False):
    """
    Dự đoán nhãn cho một ảnh viết tay đơn lẻ sử dụng mô hình Attention.
    """
    device = torch.device(config.DEVICE)
    print(f"Chạy inference trên thiết bị: {device}")
    
    # 1. Khởi tạo vocab và tải mô hình
    vocab = Vocabulary(config.VOCAB_CHARS)
    
    # Khởi tạo mô hình ở chế độ freeze để an toàn
    model = AttentionHTR(
        vocab_size=vocab.size,
        pretrained=False,  # Không tải trọng số ImageNet vì ta sẽ nạp checkpoint
        freeze=True,
        out_layer="layer3"
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
        
    # Resize giữ aspect ratio và pad về 64x256
    image_processed = resize_and_pad(image_gray, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    
    # Chuyển thành tensor và thêm chiều batch: (1, 3, 64, 256)
    image_tensor = to_tensor_and_normalize(image_processed).unsqueeze(0).to(device)
    
    # 3. Giải mã nhãn (Inference)
    print("Đang giải mã...")
    decoded_words, attn_weights_list = greedy_decode(
        model=model,
        images=image_tensor,
        vocab=vocab,
        device=device,
        max_len=config.MAX_LABEL_LENGTH
    )
    
    predicted_word = decoded_words[0]
    attn_weights = attn_weights_list[0]  # (T, seq_len)
    
    print("-" * 40)
    print(f"Ảnh: {image_path}")
    print(f"Từ dự đoán: '{predicted_word}'")
    print("-" * 40)
    
    # 4. Trực quan hóa bản đồ attention nếu được kích hoạt
    if save_attention:
        if len(predicted_word) == 0:
            print("Không thể lưu bản đồ attention do không dự đoán được ký tự nào (từ rỗng).")
            return
            
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_name = f"predict_{base_name}_attn.png"
        save_path = os.path.join(config.ATTENTION_MAP_DIR, save_name)
        
        # Chuyển đổi ký tự sang danh sách đơn lẻ
        predicted_chars = list(predicted_word)
        
        print(f"Đang tạo và lưu bản đồ attention tại: {save_path}...")
        save_attention_map(
            image=image_processed,
            attention_weights=attn_weights,
            predicted_chars=predicted_chars,
            save_path=save_path
        )
        print("Đã lưu bản đồ attention thành công!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict single handwriting image using Attention HTR Model.")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến file ảnh cần nhận dạng.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/attention_bilstm/best_attention_model.pth", 
                        help="Đường dẫn đến file checkpoint (.pth).")
    parser.add_argument("--save_attention", action="store_true", help="Cờ kích hoạt trực quan hóa và lưu bản đồ Attention.")
    
    args = parser.parse_args()
    
    # Giải quyết đường dẫn tuyệt đối hoặc tương đối
    img_path = os.path.abspath(args.image)
    ckpt_path = os.path.abspath(args.checkpoint)
    
    predict_single_image(img_path, ckpt_path, args.save_attention)
