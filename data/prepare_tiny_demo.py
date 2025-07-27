#!/usr/bin/env python3
"""
Create a tiny custom dataset for ultra-fast demo training.
This creates a very small dataset with repeated patterns for quick experimentation.
"""

import os
import pickle
import numpy as np
import sys
sys.path.append('..')  # Add parent directory to path
from model_config import ModelConfig

def create_tiny_demo_data():
    """Create a tiny demo dataset with simple patterns."""
    
    # Create data directory
    os.makedirs('tiny_demo', exist_ok=True)
    
    # Create a very simple dataset with patterns
    # This will help the model learn quickly for demonstration
    patterns = [
        "hello world ",
        "the quick brown fox ",
        "to be or not to be ",
        "once upon a time ",
        "machine learning is fun ",
        "artificial intelligence ",
        "neural networks learn ",
        "transformers are amazing "
    ]
    
    # Repeat patterns to create training data
    data = ""
    for _ in range(50):  # Repeat 50 times
        for pattern in patterns:
            data += pattern
    
    print(f"Created dataset with {len(data)} characters")
    
    # Create character vocabulary
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    
    # Get target vocab size from model config
    target_vocab_size = ModelConfig.vocab_size
    
    # Pad vocabulary to exactly target_vocab_size characters (our model's vocab_size)
    while len(chars) < target_vocab_size:
        chars.append(f'<PAD{len(chars)}>')
    
    chars = chars[:target_vocab_size]  # Ensure exactly target_vocab_size
    vocab_size = len(chars)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters: {chars[:20]}...")  # Show first 20 chars
    
    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(s):
        return [stoi.get(c, 0) for c in s]  # Use 0 for unknown chars
    
    def decode(l):
        return ''.join([itos.get(i, '') for i in l])
    
    # Encode the data
    data_encoded = encode(data)
    
    # Create train/val split (80/20)
    n = len(data_encoded)
    train_data = data_encoded[:int(0.8 * n)]
    val_data = data_encoded[int(0.8 * n):]
    
    print(f"Train set: {len(train_data)} tokens")
    print(f"Val set: {len(val_data)} tokens")
    
    # Save as binary files
    train_ids = np.array(train_data, dtype=np.uint16)
    val_ids = np.array(val_data, dtype=np.uint16)
    
    train_ids.tofile('tiny_demo/train.bin')
    val_ids.tofile('tiny_demo/val.bin')
    
    # Save metadata
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'chars': chars
    }
    
    with open('tiny_demo/meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    
    print("Tiny demo data preparation complete!")
    print("Files created:")
    print(f"  - tiny_demo/train.bin ({len(train_ids)} tokens)")
    print(f"  - tiny_demo/val.bin ({len(val_ids)} tokens)")
    print(f"  - tiny_demo/meta.pkl (vocab_size: {vocab_size})")
    
    # Show some example sequences
    print("\nExample training sequences:")
    for i in range(0, min(50, len(train_data)), 10):
        seq = train_data[i:i+8]  # Show sequences of length 8 (our seq_length)
        decoded = decode(seq)
        print(f"  {seq} -> '{decoded}'")

if __name__ == "__main__":
    create_tiny_demo_data()
