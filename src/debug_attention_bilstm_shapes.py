import torch
import src.config as config
from src.data.vocab import Vocabulary
from src.models.attention_model import AttentionHTR

def debug_shapes():
    # 1. Setup device and vocab
    device = torch.device(config.DEVICE)
    vocab = Vocabulary(config.VOCAB_CHARS)
    vocab_size = vocab.size
    
    print("=" * 60)
    print("          DEBUG ATTENTION-BILSTM SHAPE CHECK")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Vocab size: {vocab_size}")
    
    # 2. Instantiate new model
    print("\nInstantiating AttentionHTR with BiLSTM context...")
    model = AttentionHTR(vocab_size=vocab_size, pretrained=False, freeze=True).to(device)
    model.eval()
    
    # 3. Create dummy batch
    B = 2
    dummy_images = torch.randn(B, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH).to(device)
    dummy_targets = torch.randint(low=3, high=vocab_size, size=(B, 9), dtype=torch.long).to(device)
    
    print("\nRunning feature extraction flow...")
    with torch.no_grad():
        # Call extract_features with debug info
        enc_seq, enc_features, enc_seq_raw = model.extract_features(dummy_images, return_debug=True)
        
        # Call full forward pass
        logits, attn_weights = model(dummy_images, dummy_targets)
        
    print("-" * 60)
    print(f"1. Input images shape:               {dummy_images.shape}")
    print(f"2. ResNet features shape:            {enc_features.shape}")
    print(f"3. Raw sequence shape:               {enc_seq_raw.shape}")
    print(f"4. BiLSTM contextual sequence shape: {enc_seq.shape}")
    print(f"5. Output logits shape:              {logits.shape}")
    print(f"6. Attention weights shape:          {attn_weights.shape}")
    print("=" * 60)
    print("Shape check completed successfully!")

if __name__ == "__main__":
    debug_shapes()
