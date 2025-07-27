#!/usr/bin/env python3
"""
Prepare the Shakespeare dataset for character-level language modeling.
This script downloads the tiny shakespeare dataset and creates train/val splits.
"""

import os
import pickle
import requests
import numpy as np

# Download the tiny shakespeare dataset
data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

def download_data():
    """Download the tiny shakespeare dataset."""
    print("Downloading Shakespeare dataset...")
    
    # Create data directory
    os.makedirs('shakespeare_char', exist_ok=True)
    
    # Download the data
    response = requests.get(data_url)
    with open('shakespeare_char/input.txt', 'w', encoding='utf-8') as f:
        f.write(response.text)
    
    print(f"Downloaded {len(response.text)} characters")
    return response.text

def prepare_shakespeare_data():
    """Prepare the Shakespeare data for training."""
    
    # Download data if it doesn't exist
    input_file = 'shakespeare_char/input.txt'
    if not os.path.exists(input_file):
        data = download_data()
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = f.read()
    
    print(f"Length of dataset in characters: {len(data):,}")
    
    # Get all unique characters
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Unique characters: {''.join(chars)}")
    
    # For demo purposes, we'll limit the vocabulary to make it even smaller
    # Take only the most common characters to match our small vocab_size of 32
    from collections import Counter
    char_counts = Counter(data)
    most_common_chars = [char for char, count in char_counts.most_common(31)]  # 31 + 1 unknown token = 32
    
    # Add a special token for unknown characters
    chars = most_common_chars + ['<UNK>']
    vocab_size = len(chars)
    
    print(f"Reduced vocabulary size for demo: {vocab_size}")
    print(f"Most common characters: {most_common_chars}")
    
    # Create character mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(s):
        """Encode string to list of integers."""
        return [stoi.get(c, stoi['<UNK>']) for c in s]
    
    def decode(l):
        """Decode list of integers to string."""
        return ''.join([itos[i] for i in l])
    
    # Test the encoding/decoding
    test_string = "Hello World!"
    encoded = encode(test_string)
    decoded = decode(encoded)
    print(f"Test - Original: '{test_string}'")
    print(f"Test - Encoded: {encoded}")
    print(f"Test - Decoded: '{decoded}'")
    
    # Encode the entire dataset
    data_encoded = encode(data)
    print(f"Dataset encoded to {len(data_encoded):,} tokens")
    
    # Split into train/validation (90/10 split)
    n = len(data_encoded)
    train_data = data_encoded[:int(0.9 * n)]
    val_data = data_encoded[int(0.9 * n):]
    
    print(f"Train set: {len(train_data):,} tokens")
    print(f"Val set: {len(val_data):,} tokens")
    
    # Save as binary files
    train_ids = np.array(train_data, dtype=np.uint16)
    val_ids = np.array(val_data, dtype=np.uint16)
    
    train_ids.tofile('shakespeare_char/train.bin')
    val_ids.tofile('shakespeare_char/val.bin')
    
    # Save the metadata
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'chars': chars
    }
    
    with open('shakespeare_char/meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    
    print("Data preparation complete!")
    print(f"Files created:")
    print(f"  - shakespeare_char/train.bin ({len(train_ids)} tokens)")
    print(f"  - shakespeare_char/val.bin ({len(val_ids)} tokens)")
    print(f"  - shakespeare_char/meta.pkl (vocab_size: {vocab_size})")

if __name__ == "__main__":
    prepare_shakespeare_data()
