#!/usr/bin/env python3
"""
Generate text using a trained GPT model.
This script loads a trained model and generates text samples.
"""

import argparse
import torch
import pickle
import os
import re
from model import GPT
from model_config import ModelConfig


def load_model(checkpoint_path, device='cpu'):
  """Load a trained model from checkpoint."""

  # Load the checkpoint
  checkpoint = torch.load(checkpoint_path, map_location=device)

  # Create model with same config
  model_config = ModelConfig()
  model = GPT(model_config)

  # Load the state dict
  model.load_state_dict(checkpoint['model'])
  model.eval()
  model.to(device)

  return model


def load_tokenizer(data_dir):
  """Load the tokenizer from meta.pkl."""
  meta_path = os.path.join(data_dir, 'meta.pkl')

  if not os.path.exists(meta_path):
    raise FileNotFoundError(f"No meta.pkl found in {data_dir}")

  with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

  # Check if this is word-level or character-level tokenization
  tokenization_type = meta.get('tokenization', 'char')  # Default to char for backward compatibility
  
  return meta['stoi'], meta['itos'], tokenization_type


def tokenize_words(text):
  """Tokenize text into words (same function as in prepare_tiny_demo_words.py)."""
  # Convert to lowercase and extract words, punctuation, and spaces
  words = re.findall(r'\b\w+\b|[.,!?;\s]', text.lower())
  # Filter out empty strings
  words = [w for w in words if w != '']
  return words


def encode(text, stoi, tokenization_type='char'):
  """Encode text to token ids."""
  if tokenization_type == 'word':
    # Word-level tokenization
    words = tokenize_words(text)
    return [stoi.get(word, stoi.get('<UNK>', 0)) for word in words]
  else:
    # Character-level tokenization (original behavior)
    return [stoi.get(c, 0) for c in text]


def decode(tokens, itos, tokenization_type='char'):
  """Decode token ids to text."""
  if tokenization_type == 'word':
    # Word-level tokenization - join with spaces
    words = [itos.get(token, '<UNK>') for token in tokens]
    return ' '.join(words)
  else:
    # Character-level tokenization (original behavior)
    return ''.join([itos.get(token, '') for token in tokens])


@torch.no_grad()
def generate_text(model, prompt, stoi, itos, tokenization_type='char', max_new_tokens=20, temperature=1.0, device='cpu', seed=None):
  """Generate text given a prompt."""

  # Set random seed for reproducible generation
  if seed is not None:
    torch.manual_seed(seed)
    if device.startswith('cuda'):
      torch.cuda.manual_seed(seed)

  # Encode the prompt
  prompt_tokens = encode(prompt, stoi, tokenization_type)

  # Convert to tensor
  x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

  print(f"Prompt: '{prompt}'")
  print(f"Prompt tokens: {prompt_tokens}")
  if tokenization_type == 'word':
    # Show word-to-token mapping for the prompt
    prompt_words = tokenize_words(prompt)
    print(f"Prompt words: {prompt_words}")
    for word, token in zip(prompt_words, prompt_tokens):
      print(f"  '{word}' -> {token}")
  print("Generating...")

  # Generate tokens
  generated_tokens = []

  for _ in range(max_new_tokens):
    # Get the last seq_length tokens (or all if shorter)
    seq_length = model.config.seq_length
    x_cond = x if x.size(1) <= seq_length else x[:, -seq_length:]

    # Forward pass
    logits, _ = model(x_cond)

    # Get logits for the last position and apply temperature
    logits = logits[:, -1, :] / temperature

    # Sample from the distribution
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    # Append to sequence
    x = torch.cat((x, next_token), dim=1)
    generated_tokens.append(next_token.item())

    # Decode and print progress
    generated_text = decode(generated_tokens, itos, tokenization_type)
    if tokenization_type == 'word':
      print(f"Generated token {next_token.item()} -> '{itos.get(next_token.item(), '<UNK>')}'")
      print(f"Generated so far: '{prompt} {generated_text}'")
    else:
      print(f"Generated so far: '{prompt + generated_text}'")

  # Decode final result
  if tokenization_type == 'word':
    final_text = prompt + " " + decode(generated_tokens, itos, tokenization_type)
  else:
    final_text = prompt + decode(generated_tokens, itos, tokenization_type)
  return final_text


def main():
  """Main function for text generation."""

  # Set up argument parser
  parser = argparse.ArgumentParser(
    description='Generate text using a trained GPT model')

  # Add arguments
  parser.add_argument('--device', type=str, default=None,
                      help='Device to use (e.g., "cpu", "cuda", "cuda:0"). If not specified, auto-detects.')
  parser.add_argument('--checkpoint', type=str, default='out/ckpt.pt',
                      help='Path to the model checkpoint file.')
  parser.add_argument('--data_dir', type=str, default='data/tiny_demo_words',
                      help='Directory containing the dataset and meta.pkl file.')
  parser.add_argument('--prompt', type=str, default=None,
                      help='Custom prompt for text generation. If not specified, uses default prompts.')
  parser.add_argument('--max_tokens', type=int, default=10,
                      help='Maximum number of new tokens to generate.')
  parser.add_argument('--temperature', type=float, default=0.2,
                      help='Temperature for sampling (higher = more random).')
  parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducible generation across devices.')

  args = parser.parse_args()

  # Configuration from arguments or defaults
  device = args.device if args.device else (
    'cuda' if torch.cuda.is_available() else 'cpu')
  checkpoint_path = args.checkpoint
  data_dir = args.data_dir

  print(f"Using device: {device}")

  # Check if checkpoint exists
  if not os.path.exists(checkpoint_path):
    print(f"No checkpoint found at {checkpoint_path}")
    print("Please train the model first by running: python train.py")
    return

  # Load tokenizer
  try:
    stoi, itos, tokenization_type = load_tokenizer(data_dir)
    print(f"Loaded tokenizer with vocab size: {len(stoi)}")
    print(f"Tokenization type: {tokenization_type}")
    
    # Print word-to-token mappings for better understanding
    print(f"\nWord-to-token mappings:")
    if tokenization_type == 'word':
      # For word-level, show all mappings
      for word, token_id in sorted(stoi.items(), key=lambda x: x[1]):
        print(f"  '{word}' -> {token_id}")
    else:
      # For character-level, show first 20 mappings
      for i, (char, token_id) in enumerate(sorted(stoi.items(), key=lambda x: x[1])):
        if i >= 20:
          print(f"  ... and {len(stoi) - 20} more characters")
          break
        print(f"  '{char}' -> {token_id}")
    print()
    
  except FileNotFoundError as e:
    print(f"Error loading tokenizer: {e}")
    print("Please prepare the data first by running the data preparation script.")
    return

  # Load model
  try:
    model = load_model(checkpoint_path, device)
    print("Model loaded successfully!")
    print(f"Model parameters: {model.get_num_params():,}")
  except Exception as e:
    print(f"Error loading model: {e}")
    return

  # Generate text with different prompts
  if args.prompt:
    # Use custom prompt
    prompts = [args.prompt]
  else:
    # Use default prompts based on tokenization type
    if tokenization_type == 'word':
      prompts = [
        "hello",
        "GPT by",
        "I love",
      ]
    else:
      prompts = [
        "hello ",
        "GPT ",
        "I love",
      ]

  for prompt in prompts:
    print("\n" + "=" * 50)
    try:
      generated = generate_text(
          model=model,
          prompt=prompt,
          stoi=stoi,
          itos=itos,
          tokenization_type=tokenization_type,
          max_new_tokens=args.max_tokens,
          temperature=args.temperature,
          device=device,
          seed=args.seed
      )
      print(f"\nFinal result: '{generated}'")
    except Exception as e:
      print(f"Error generating text: {e}")
    print("=" * 50)


if __name__ == "__main__":
  main()
