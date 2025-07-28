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




### Example Iterations

Here's a breakdown of what happens during actual training iterations when you use the `--gpt_by_hand` flag, which outputs detailed tensor information.

#### Iteration 0 - Initial Training Step

This iteration uses the first batch of data.

**Input Sequence:** `[0, 8, 0, 6, 0, 9, 0, 25]` which decodes to `"' gpt ' by ' hand ' project"`.

**Target Sequence:** `[8, 0, 6, 0, 9, 0, 25, 0]` which decodes to `"gpt ' by ' hand ' project '"`. The model aims to predict these tokens given the input.

##### Model Architecture in Action

1. **Token Embeddings:** Each input token is converted into a 16-dimensional vector.

2. **Position Embeddings:** These are added to the token embeddings to provide positional information within the sequence.

3. **Layer 1 (Transformer Block):** The combined embeddings pass through the first transformer block, consisting of a 2-head self-attention mechanism followed by a feed-forward network.

4. **Layer 2 (Transformer Block):** The output from Layer 1 then passes through a second identical transformer block.

5. **Output Layer:** The final output from the transformer blocks is transformed into "logits," which are raw predictions for each of the 32 vocabulary tokens at each position in the sequence.

##### Key Tensors (and their shapes for this iteration)

**Combined Embeddings:** `[1, 8, 16]` (Batch Size: 1, Sequence Length: 8, Embedding Dimension: 16)

```
[[ 0.0025648 -0.00763123 ... 0.01054394 0.00168534]
[ 0.02813374 0.00895928 ... 0.05949021 0.00570134]
...
[-0.01078424 -0.02195856 ... 0.04654259 0.03406077]
[ 0.00170384 -0.00331098 ... -0.01766913 -0.01369504]]
```

**Attention Output:** `[1, 8, 16]` (Output after the self-attention mechanism)

```
[[-3.3116e-04 3.0899e-03 ... 2.2488e-03 -1.8942e-04]
[ 9.9182e-04 1.5125e-03 ... -8.4543e-04 2.4433e-03]
...
[ 6.1154e-05 1.7376e-03 ... 3.3627e-03 -1.0290e-03]
[ 5.4646e-04 1.8024e-03 ... 3.2291e-03 -7.6866e-04]]
```

**Final Logits (first sequence, first 8 vocab items):** `[1, 8, 32]`

```
[[ 0.2225 0.129 ... -0.06494 -0.001535]
[ 0.1207 0.04 ... -0.04004 0.09106 ]
...
[ 0.2181 0.1218 ... 0.0318 0.03198 ]
[-0.03055 -0.009766 ... 0.1681 0.05457 ]]
```

##### Training Progress at Iteration 0

- **Training Loss:** 3.4518
- **Validation Loss:** 3.4536
- **Overall Loss:** 3.4873 (This is the loss calculated for the specific batch shown, which contributes to the overall training loss.)
- **Learning Rate:** 0.0 (The model is in a "warmup" phase, where the learning rate is initially zero and gradually increases.)

**Top Token Predictions (for the last position in the sequence):**
- `'networks'`: 0.0382
- `'project'`: 0.0368
- `'by'`: 0.0360

#### Iteration 1 - Learning Begins

This iteration processes a different batch of data and shows the initial effects of training.

**Input Sequence:** `[17, 0, 15, 0, 13, 0, 7, 0]` which decodes to `"machine ' learning ' is ' fun '"`.

**Target Sequence:** `[0, 15, 0, 13, 0, 7, 0, 4]` which decodes to `"' learning ' is ' fun ' artificial"`.

##### Observable Changes

- **Loss:** 3.4309 (This is slightly lower than the loss from the previous batch, indicating that the model is starting to learn.)
- **Learning Rate:** 2.9999999999999997e−06 (The learning rate has started to increase from zero, allowing the model's weights to be updated.)

##### Weight Updates (Example: Attention c_attn weights)

You'll notice that the "Attention c_attn weights after update" are slightly different from their "Before" values, indicating that the model's parameters are beginning to adjust based on the gradients. These are very small, incremental changes, as expected early in training.

**Attention c_attn weights before update (first 4x4):**

```
[[ 0.019159678 -0.024671469 ... -0.0022882789 0.0053827376]
[-0.020287437 -0.016469905 ... -0.0079479981 0.027898392]
...
[ 0.021043016 0.010694436 ... 9.2175527e-05 0.0038459431]]
```

**Attention c_attn weight changes (first 4x4):**

```
[[-1.5515834e-06 2.1010637e-06 ... 2.0395964e-06 -2.2901222e-06]
[-1.5124679e-06 -2.6337802e-06 ... -1.2312084e-06 2.5294721e-06]
...
[-2.1774322e-06 -2.6402995e-06 ... -2.0524967e-06 2.5054906e-06]]
```

**Attention c_attn weights after update (first 4x4):**

```
[[ 0.0191581268 -0.0246693678 ... -0.00228623929 0.00538044749]
[-0.0202889498 -0.0164725389 ... -0.00794922933 0.0279009212]
...
[ 0.0210408382 0.0106917955 ... 9.01230305e-05 0.00384844863]]
```

##### Gradient Flow

Gradients are calculated (though not fully displayed for all parameters in the log, "No gradients computed yet" refers to the end of the iteration when the log is printed, but they were computed during the backward pass).

These gradients flow backward from the output layer (logits to vocabulary) through Layer 2 (the second transformer block), then Layer 1 (the first transformer block), and finally to the token and position embeddings. This backpropagation process updates all the model's parameters.

Each iteration processes different word sequences, gradually teaching the model patterns like "machine learning", "artificial intelligence", and "to be or not", leading to improved predictions over time.

### Manual Calculation Exercise
Try calculating attention scores by hand:

TODO



## Tokenization

The model uses word-level tokenization instead of character-level, making it easier to understand token mappings:

### Vocabulary
- **Size**: 32 unique words/tokens
- **Examples**: `'hello' -> 10`, `'world' -> 31`, `'machine' -> 17`
- **Special token**: `' '` (space) -> `0` for word separation

### Example Tokenization
```
"hello world" -> [10, 0, 31, 0]  # hello + space + world + space
"machine learning" -> [17, 0, 15, 0]  # machine + space + learning + space
```

### Training Sequences
The model sees sequences like:
- `[28, 0, 5, 0, 23, 0, 21, 0]` → `"to be or not "`
- `[4, 0, 12, 0, 26, 0, 17, 0]` → `"artificial intelligence rocks machine "`

This word-level approach makes it easy to trace which tokens correspond to which words during manual calculations.

## Appendix


### Code Structure

```
gpt-by-hand/
├── README.md                      # Project documentation
├── model.py                       # GPT model architecture
├── model_config.py                # Model configuration class
├── model_test.py                  # Tests for model components
├── train.py                       # Training loop and logic
├── train_config.py                # Training configuration class
├── train_test.py                  # Tests for training components
├── generate.py                    # Text generation script (inference)
├── data/                          # Data preparation scripts
│   ├── prepare_tiny_demo_words.py # Create tiny demo dataset (Primary choice)
└── out/                           # The model checkpoints
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
