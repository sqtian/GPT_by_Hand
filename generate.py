#!/usr/bin/env python3
"""
Generate text using a trained GPT model.
This script loads a trained model and generates text samples.
"""

import argparse
import torch
import pickle
import os
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
    
    return meta['stoi'], meta['itos']

def encode(text, stoi):
    """Encode text to token ids."""
    return [stoi.get(c, 0) for c in text]

def decode(tokens, itos):
    """Decode token ids to text."""
    return ''.join([itos.get(token, '') for token in tokens])

@torch.no_grad()
def generate_text(model, prompt, stoi, itos, max_new_tokens=20, temperature=1.0, device='cpu', seed=None):
    """Generate text given a prompt."""
    
    # Set random seed for reproducible generation
    if seed is not None:
        torch.manual_seed(seed)
        if device.startswith('cuda'):
            torch.cuda.manual_seed(seed)
    
    # Encode the prompt
    prompt_tokens = encode(prompt, stoi)
    
    # Convert to tensor
    x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"Prompt: '{prompt}'")
    print(f"Prompt tokens: {prompt_tokens}")
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
        generated_text = decode(generated_tokens, itos)
        print(f"Generated so far: '{prompt + generated_text}'")
    
    # Decode final result
    final_text = prompt + decode(generated_tokens, itos)
    return final_text

def main():
    """Main function for text generation."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate text using a trained GPT model')
    
    # Add arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cpu", "cuda", "cuda:0"). If not specified, auto-detects.')
    parser.add_argument('--checkpoint', type=str, default='out/ckpt.pt',
                        help='Path to the model checkpoint file.')
    parser.add_argument('--data_dir', type=str, default='data/tiny_demo',
                        help='Directory containing the dataset and meta.pkl file.')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom prompt for text generation. If not specified, uses default prompts.')
    parser.add_argument('--max_tokens', type=int, default=10,
                        help='Maximum number of new tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature for sampling (higher = more random).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible generation across devices.')
    
    args = parser.parse_args()
    
    # Configuration from arguments or defaults
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
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
        stoi, itos = load_tokenizer(data_dir)
        print(f"Loaded tokenizer with vocab size: {len(stoi)}")
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
        # Use default prompts
        prompts = [
            "to be or ",
            "I am",
        ]
    
    for prompt in prompts:
        print("\n" + "="*50)
        try:
            generated = generate_text(
                model=model,
                prompt=prompt,
                stoi=stoi,
                itos=itos,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                device=device,
                seed=args.seed
            )
            print(f"\nFinal result: '{generated}'")
        except Exception as e:
            print(f"Error generating text: {e}")
        print("="*50)

if __name__ == "__main__":
    main()
