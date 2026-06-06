import torch
import torch.nn as nn
from typing import Tuple, Optional

import src.config as config
from src.models.resnet_encoder import ResNetEncoder
from src.models.gru_decoder import GRUDecoder

class AttentionHTR(nn.Module):
    def __init__(self, vocab_size: int, pretrained: bool = True, freeze: bool = True,
                 out_layer: str = "layer3", embed_dim: int = 128, 
                 decoder_hidden_dim: int = 256, attention_dim: int = 256, 
                 dropout: float = 0.3):
        """
        Mô hình nhận dạng chữ viết tay Attention-HTR tích hợp Encoder và Decoder.
        """
        super().__init__()
        
        # 1. Khởi tạo CNN Encoder
        self.encoder = ResNetEncoder(pretrained=pretrained, freeze=freeze, out_layer=out_layer)
        
        # Lấy kích thước đầu ra thực tế của Encoder dựa trên cấu hình ảnh
        enc_channels, enc_h, enc_w = self.encoder.get_output_size(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        
        # Gộp chiều cao H' vào số channels C làm đặc trưng cho mỗi cột sequence chiều rộng W'
        encoder_dim = enc_channels * enc_h
        
        # 2. Khởi tạo GRU Decoder
        self.decoder = GRUDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            encoder_dim=encoder_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout
        )

    def forward(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5, max_len: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        images: (B, 3, 64, 256)
        targets: (B, target_len)
        
        Trả về:
        - logits: (B, T, vocab_size)
        - attn_weights: (B, T, seq_len)
        """
        # 1. Trích xuất đặc trưng hình ảnh: (B, C, H', W')
        enc_features = self.encoder(images)
        
        # 2. Reshape đặc trưng: (B, C, H', W') -> (B, W', C, H') -> (B, W', C * H')
        # Chiều rộng W' đóng vai trò là sequence length đầu vào cho Attention Decoder
        B, C, H_prime, W_prime = enc_features.shape
        enc_seq = enc_features.permute(0, 3, 1, 2).contiguous().view(B, W_prime, C * H_prime)
        
        # 3. Đi qua Attention Decoder để sinh chuỗi logits
        logits, attn_weights = self.decoder(
            encoder_outputs=enc_seq,
            targets=targets,
            teacher_forcing_ratio=teacher_forcing_ratio,
            max_len=max_len
        )
        
        return logits, attn_weights

    def freeze_encoder(self):
        """Đóng băng toàn bộ tham số của encoder."""
        self.encoder.freeze_all()

    def unfreeze_encoder(self, from_layer: str = "layer3"):
        """Mở đóng băng các tham số encoder từ layer chỉ định."""
        self.encoder.unfreeze_from(from_layer)
