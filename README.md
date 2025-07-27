# GPT-by-Hand

This repository contains a tested, documented, and modular implementation of a GPT language model designed for personal interests. The project uses **extremely small model sizes** that make it possible to manually trace through calculations, making it an ideal tool for learning and understanding transformer architectures.

## Small Model Sizes

Our demo model uses tiny parameters perfect for educational purposes:

- **Vocabulary Size**: 32 tokens (vs GPT-2's 50,257)
- **Sequence Length**: 8 tokens (vs GPT-2's 1,024)  
- **Layers**: 2 (vs GPT-2's 12)
- **Attention Heads**: 2 (vs GPT-2's 12)
- **Embedding Dimension**: 16 (vs GPT-2's 768)
- **Total Parameters**: ~1,000 (vs GPT-2's 124M)

This allows you to manually calculate attention + FFN activations, and trace forward passes by hand.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Training Data

For ultra-fast demo (recommended for first try):
```bash
cd data
python prepare_tiny_demo.py
cd ..
```

For Shakespeare dataset:
```bash
cd data  
python prepare_shakespeare.py
cd ..
```

### 3. Train the Model

Train on tiny demo data (very fast, <1 minute):
```bash
python train.py --device cpu --max_iters 200
```

Or for longer training:
```bash
python train.py --device cuda --max_iters 2000  # Use GPU if available
```

### 4. Generate Text
```bash
python generate.py
```

## Understanding the Architecture

The model is small enough that you can manually trace through calculations:

### Example Forward Pass

TODO



With sequence "hello" (tokens: [7, 4, 11, 11, 14]):
1. **Token Embeddings**: Each token → 16-dimensional vector
2. **Position Embeddings**: Position 0,1,2,3,4 → 16-dimensional vectors  
3. **Attention**: 2 heads, each seeing 8-dimensional query/key/value
4. **Feed Forward**: 16 → 64 → 16 dimensions
5. **Output**: Logits over 32 vocabulary tokens

### Manual Calculation Exercise
Try calculating attention scores by hand:


TODO



## Appendix


### Code Structure

```
gpt-by-hand/
├── README.md                     # Project documentation
├── model.py                      # GPT model architecture
├── model_config.py               # Model configuration class
├── model_test.py                 # Tests for model components
├── train.py                      # Training loop and logic
├── train_config.py               # Training configuration class
├── train_test.py                 # Tests for training components
├── generate.py                   # Text generation script (inference)
├── data/                         # Data preparation scripts
│   ├── prepare_tiny_demo.py      # Create tiny demo dataset (Primary choice)
│   └── prepare_shakespeare.py    # Prepare Shakespeare dataset
└── out/                          # The model checkpoints
```

### Training Configuration

Key training parameters optimized for small models:

```python
# Small batch sizes and iterations for demo
batch_size = 4
max_iters = 2000
eval_interval = 100
gradient_accumulation_steps = 4

# Learning rate schedule
learning_rate = 3e-4
warmup_iters = 100
```

## Dataset Preparation

### Tiny Demo Dataset
- **Size**: ~3,200 characters
- **Patterns**: Repeated phrases like "hello world", "the quick brown fox"
- **Purpose**: Quick experimentation and testing

### Shakespeare Dataset  
- **Size**: ~1MB of Shakespeare text
- **Vocabulary**: Most common 31 characters + `<UNK>`
- **Purpose**: More realistic text generation

### Custom Dataset
To use your own text data:

1. Create a new preparation script in `data/` folder
2. Process your text into train.bin and val.bin (numpy uint16 arrays)
3. Create meta.pkl with vocabulary mappings
4. Update `dataset` in `train_config.py`


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
* Based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
* Inspired by the architecture of the original GPT models from OpenAI
