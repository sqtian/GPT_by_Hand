#!/usr/bin/env python3
"""
Create a tiny custom dataset for ultra-fast demo training using WORD-LEVEL tokenization.
This creates a very small dataset with repeated patterns for quick experimentation.
"""

import os
import pickle
import numpy as np
import sys
import re
import random
sys.path.append('..')  # Add parent directory to path
from model_config import ModelConfig


def create_tiny_demo_word_data():
  """Create a tiny demo dataset with simple patterns using word-level tokenization."""

  # Create data directory
  os.makedirs('tiny_demo_words', exist_ok=True)

  # Create a very simple dataset with patterns
  # This will help the model learn quickly for demonstration
  patterns = [
      "hello world",
      "to be or not to be",
      "once upon a time",
      "machine learning is fun",
      "artificial intelligence rocks",
      "neural networks learn patterns",
      "transformers are amazing models",
      "I love GPT by hand project",
  ]

  # Repeat patterns to create training data
  data_text = ""
  total_repetitions = 200  # Total number of pattern insertions

  # Set random seed for reproducible results
  random.seed(42)

  for _ in range(total_repetitions):
    # Randomly select a pattern
    pattern = random.choice(patterns)
    data_text += pattern + " "

  print(f"Created dataset with {len(data_text)} characters")

  # Word-level tokenization
  def tokenize_words(text):
    """Simple word tokenization - split by whitespace and handle punctuation."""
    # Convert to lowercase and extract words, punctuation, and spaces
    # This approach keeps spaces as literal ' ' characters
    words = re.findall(r'\b\w+\b|[.,!?;\s]', text.lower())
    # Filter out empty strings
    words = [w for w in words if w != '']
    return words

  # Get all words
  words = tokenize_words(data_text)
  print(f"Total words/tokens: {len(words)}")

  # Create vocabulary from unique words
  vocab = sorted(list(set(words)))
  vocab_size = len(vocab)

  print(f"Unique vocabulary size: {vocab_size}")
  print(f"Vocabulary: {vocab}")

  # Get target vocab size from model config
  target_vocab_size = ModelConfig.vocab_size

  # Handle vocabulary size mismatch
  if vocab_size > target_vocab_size:
    print(
      f"Warning: Vocabulary size ({vocab_size}) exceeds model config ({target_vocab_size})")
    print("Consider increasing ModelConfig.vocab_size or reducing vocabulary")
    # Keep only the most frequent words
    from collections import Counter
    word_counts = Counter(words)
    most_common = word_counts.most_common(
      target_vocab_size - 1)  # -1 for <UNK>
    vocab = ['<UNK>'] + [word for word, count in most_common]
    vocab_size = len(vocab)
  elif vocab_size < target_vocab_size:
    # Pad vocabulary to exactly target_vocab_size (add special tokens)
    special_tokens = ['<UNK>', '<PAD>', '<START>', '<END>']
    for token in special_tokens:
      if token not in vocab and len(vocab) < target_vocab_size:
        vocab.append(token)

    # Add more padding tokens if needed
    while len(vocab) < target_vocab_size:
      vocab.append(f'<PAD{len(vocab)}>')

  vocab = vocab[:target_vocab_size]  # Ensure exactly target_vocab_size
  vocab_size = len(vocab)

  print(f"Final vocabulary size: {vocab_size}")
  print(f"Final vocabulary: {vocab}")

  # Create mappings
  stoi = {word: i for i, word in enumerate(vocab)}
  itos = {i: word for i, word in enumerate(vocab)}

  def encode(text):
    """Encode text to token IDs."""
    words = tokenize_words(text)
    return [stoi.get(word, stoi.get('<UNK>', 0)) for word in words]

  def decode(token_ids):
    """Decode token IDs back to text."""
    words = [itos.get(i, '<UNK>') for i in token_ids]
    return ' '.join(words)

  # Encode the data
  data_encoded = encode(data_text)

  # Create train/val split (80/20)
  n = len(data_encoded)
  train_data = data_encoded[:int(0.8 * n)]
  val_data = data_encoded[int(0.8 * n):]

  print(f"Train set: {len(train_data)} tokens")
  print(f"Val set: {len(val_data)} tokens")

  # Save as binary files
  train_ids = np.array(train_data, dtype=np.uint16)
  val_ids = np.array(val_data, dtype=np.uint16)

  train_ids.tofile('tiny_demo_words/train.bin')
  val_ids.tofile('tiny_demo_words/val.bin')

  # Save metadata
  meta = {
      'vocab_size': vocab_size,
      'itos': itos,
      'stoi': stoi,
      'vocab': vocab,
      'tokenization': 'word'
  }

  with open('tiny_demo_words/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

  print("Tiny demo word-level data preparation complete!")
  print("Files created:")
  print(f"  - tiny_demo_words/train.bin ({len(train_ids)} tokens)")
  print(f"  - tiny_demo_words/val.bin ({len(val_ids)} tokens)")
  print(f"  - tiny_demo_words/meta.pkl (vocab_size: {vocab_size})")

  # Show some example sequences
  print("\nExample training sequences:")
  seq_length = 8  # From ModelConfig
  for i in range(0, min(50, len(train_data)), seq_length):
    seq = train_data[i:i + seq_length]
    decoded = decode(seq)
    print(f"  {seq} -> '{decoded}'")

  # Show vocabulary mappings
  print(f"\nWord-to-token mappings:")
  for i, word in enumerate(vocab):
    print(f"  '{word}' -> {i}")


if __name__ == "__main__":
  create_tiny_demo_word_data()
