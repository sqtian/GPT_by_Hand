import math
import inspect
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from model_config import ModelConfig


class LayerNorm(nn.Module):
  def __init__(self, normalized_shape, bias=True, eps=1e-5):
    """Init function of LayerNorm. The argument `ndim` is usually the size of embeddings."""
    super().__init__()
    # Initializes gamma to be all ones.
    self.gamma = nn.Parameter(torch.ones(normalized_shape))
    # Initializes beta to be all zeros.
    self.beta = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
    # Initializes epsilon.
    self.eps = eps

  def forward(self, input_tensor: torch.Tensor):
    """Feed forward to apply LayerNorm on the input tensor."""
    # Mean of the token embeddings
    mean = input_tensor.mean(-1, keepdim=True)
    # Variance of the token embeddings
    var = input_tensor.var(-1, keepdim=True, unbiased=False)
    # Normalize
    x_norm = (input_tensor - mean) / torch.sqrt(var + self.eps)
    # Calculates the unbiased normalized value
    normalized_unbiased = self.gamma * x_norm

    if self.beta is not None:
      return normalized_unbiased + self.beta
    return normalized_unbiased


class Attention(nn.Module):
  """The self attention block used in the GPT model."""

  def __init__(self, config: ModelConfig):
    super().__init__()
    # The embedding dimension must be divisible by the number of heads.
    assert config.dim_embedding % config.num_heads == 0

    # The query, key, value projections for all heads, but in a batch
    # For the same input tensor of size config.dim_embedding, we will get 3 tensors of size config.dim_embedding.
    self.c_attn = nn.Linear(config.dim_embedding, 3 *
                            config.dim_embedding, bias=config.use_bias)
    # The output projection
    self.c_proj = nn.Linear(config.dim_embedding,
                            config.dim_embedding, bias=config.use_bias)
    # The regularization
    # During training, randomly zeroes some of the elements of the input tensor with probability p.
    self.attn_dropout = nn.Dropout(config.dropout_rate)
    self.resid_dropout = nn.Dropout(config.dropout_rate)
    # Stores some configuration parameters.
    self.num_heads = config.num_heads
    # The embedding dimensionality (dim_embedding).
    self.dim_embedding = config.dim_embedding
    self.dropout_rate = config.dropout_rate

    # Uses the flash attention which makes the calculation faster.
    self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    if not self.flash:
      print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
      # Creates a causal mask to ensure that attention is only applied to the left in the input sequence.
      # It is equivalent to the creation of self.bias of shape (1, 1, seq_length, seq_length).
      # For example, the initialization of self.bias can be
      # tensor([[[[1., 0., 0., 0., 0.],
      #           [1., 1., 0., 0., 0.],
      #           [1., 1., 1., 0., 0.],
      #           [1., 1., 1., 1., 0.],
      #           [1., 1., 1., 1., 1.]]]])
      self.register_buffer("bias", torch.tril(torch.ones(config.seq_length, config.seq_length))
                           .view(1, 1, config.seq_length, config.seq_length))

  def forward(self, x):
    """Feed forward for the self attention block.

    Args:
        x: The input tensor of size  (B, T, D) == (batch_size, sequence_length, embedding_dimensionality dim_embedding).

    Returns:
       y: The output tensor.
    """
    B, T, D = x.size()  # batch size, sequence length, embedding dimensionality (dim_embedding)

    # Calculates query, key, values for all heads in batch and move head forward to be the batch dim.
    # The output of self.c_attn() is a tensor of size (B, T, 3 * D).
    q, k, v = self.c_attn(x).split(self.dim_embedding, dim=2)
    # Gets the transpose of the tensor, of size (B, number of heads, T, head size)
    # Note that D == embedding dimensionality (dim_embedding) == nh x hs
    q = q.view(B, T, self.num_heads, D //
               self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
    k = k.view(B, T, self.num_heads, D //
               self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, self.num_heads, D //
               self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

    # Causal self-attention: Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    if self.flash:
      # Uses efficient attention with Flash Attention CUDA kernels.
      y = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=self.dropout_rate if self.training else 0, is_causal=True)
    else:
      # In summary, the purpose of self attention is
      #     A query looks for keys, and retrieves values.
      # Query (Q): What this token is "asking" for.
      # Key (K): What this token "offers".
      # Value (V): The information to return if this token is selected.

      # Uses the manual implementation of self attention.
      scale_factor = 1.0 / math.sqrt(k.size(-1))
      # Calculates the attention
      # q: (B, nh, T, hs), k.transpose(-2, -1):  (B, nh, hs, T)
      # => q @ k.transpose(-2, -1): (B, nh, T, T)
      # In PyTorch, the @ operator is shorthand for matrix multiplication.
      # Examples:
      #  1. 2D matrix multiplication:
      #     a.shape == (m, n)
      #     b.shape == (n, p)
      #     c = a @ b  # shape: (m, p)
      #  2. 2D @ 1D (Matrix @ Vector)
      #     a.shape == (m, n)
      #     b.shape == (n,)
      #     c = a @ b  # shape: (m,)
      # 3. Batch Matrix Multiplication (nD tensors ≥ 3D).
      #   If a and b are more than 2D (e.g., batch of matrices),
      #   then @ does batch matrix multiplication, broadcasting as needed.
      #     For tensors A and B of shape:
      #     A: (..., m, k)
      #     B: (..., k, n)
      #     Then:
      #     C = A @ B has shape (..., m, n)
      # attn shape is (B, nh, T, T)
      # The understanding can be (batch, head, query_pos, key_pos).
      attn = q @ k.transpose(-2, -1) * scale_factor
      # The attention bias is applied to -2,-1 dims of (B, nh, T, T)
      # For example, if T=3, the self.bias[:, :, :T, :T] can be
      # Position →      0   1   2
      #            ┌─────────────
      # Token 0 │   1   0   0   ← query token 0 can only attend to 0
      # Token 1 │   1   1   0   ← query token 1 can attend to 0, 1
      # Token 2 │   1   1   1   ← query token 2 can attend to 0, 1, 2
      # Then, after masked_fill, the attention matrix will be
      # tensor([[[[-0.4370,    -inf,    -inf],
      #           [-1.8460, -0.8049,    -inf],
      #           [-0.0721,  0.3787, -0.6573]]]])
      attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

      # The masked positions (with -inf) become zero probability.
      # Example:
      # tensor([[[[1.0000, 0.0000, 0.0000],
      #           [0.2609, 0.7391, 0.0000],
      #           [0.3198, 0.5020, 0.1781]]]])
      # The understanding can be (batch, head, query_pos, key_pos).
      # The purpose is: The query looks for keys.
      attn = torch.softmax(attn, dim=-1)
      # Randomly zeroing out some of those attention weights,
      # and scaling the rest so the expectation stays the same.
      # The dropout is used to help prevent overfitting.
      attn = self.attn_dropout(attn)
      # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
      # The understanding can be (batch, head, query_pos, value_pos).
      # The purpose is: The query retrieves the values based the keys.
      y = attn @ v

    # Re-assembles all head outputs side by side.
    # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, nh * hs)
    y = y.transpose(1, 2).contiguous().view(B, T, D)

    # Gets the output projection
    y = self.c_proj(y)
    y = self.resid_dropout(y)
    return y


class FFN(nn.Module):
  """FFN block for the GPT model."""

  def __init__(self, config: ModelConfig):
    super().__init__()
    # First linear layer, FFI.
    self.c_fc = nn.Linear(config.dim_embedding, 4 *
                          config.dim_embedding, bias=config.use_bias)
    self.gelu = nn.GELU()
    # Second linear layer, FFO.
    self.c_proj = nn.Linear(4 * config.dim_embedding,
                            config.dim_embedding, bias=config.use_bias)
    self.dropout = nn.Dropout(config.dropout_rate)

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    x = self.dropout(x)
    return x


class TransformerBlock(nn.Module):
  """One transformer block."""

  def __init__(self, config):
    super().__init__()
    self.ln_1 = LayerNorm(config.dim_embedding, bias=config.use_bias)
    self.attn = Attention(config)
    self.ln_2 = LayerNorm(config.dim_embedding, bias=config.use_bias)
    self.mlp = FFN(config)

  def forward(self, x):
    # Applies layer normalization to the input tensor x.
    x_norm_1 = self.ln_1(x)
    # Applies the attention to the normalized input tensor.
    x_attn = self.attn(x_norm_1)
    # Adds the attention output to the original input tensor x,
    # which is the residual connection.
    x = x + x_attn
    # Applies layer normalization to the input tensor x.
    x_norm_2 = self.ln_2(x)
    # Applies the feed forward network to the normalized input tensor.
    x_mlp = self.mlp(x_norm_2)
    # Adds the feed forward network output to the original input tensor x.
    # This is the second residual connection.
    x = x + x_mlp

    return x


class GPT(nn.Module):
  """The GPT2 model."""

  def __init__(self, config):
    super().__init__()
    assert config.vocab_size is not None
    assert config.seq_length is not None
    self.config = config

    self.transformer = nn.ModuleDict(dict(
      # The token embeddings. For example, we have vocab_size words, each word is a vector of size dim_embedding.
      # It tells us the meaning of the word.
      wte=nn.Embedding(config.vocab_size, config.dim_embedding),
      # The position embeddings. For example, in a sentence we have seq_length positions, each position is a vector of size dim_embedding.
      # It tells us the position of the word in the sentence.
      wpe=nn.Embedding(config.seq_length, config.dim_embedding),
      # Drop some values to avoid overfitting.
      drop=nn.Dropout(config.dropout_rate),
      # The list of layers. Here it is a list of num_layers decoders as we use masked self-attention.
      h=nn.ModuleList([TransformerBlock(config)
                      for _ in range(config.num_layers)]),
      # The layer normalization layer at the end.
      ln_f=LayerNorm(config.dim_embedding, bias=config.use_bias),
    ))
    # For each word of size dim_embedding, we use the linear layer to get the probability of
    # each word in the vocabulary.
    self.lm_head = nn.Linear(
      config.dim_embedding, config.vocab_size, bias=False)

    # Ensures the transformer and the linear layer share the same vocabulary.
    # https://paperswithcode.com/method/weight-tying
    self.transformer.wte.weight = self.lm_head.weight

    # Inits all weights
    self.apply(self._init_weights)

    # Applies special scaled init to the residual projections, per GPT-2 paper
    for pn, p in self.named_parameters():
      if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02 /
                              math.sqrt(2 * config.num_layers))

    # Report number of parameters
    print("Number of parameters: %.2fM" % (self.get_num_params() / 1e6))

  def forward(self, setup, targets=None):
    device = setup.device
    b, t = setup.size()
    assert t <= self.config.seq_length, f"Cannot forward sequence of length {t}, block size is only {self.config.seq_length}"
    pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

    # forward the GPT model itself
    # token embeddings of shape (b, t, dim_embedding)
    tok_emb = self.transformer.wte(setup)
    # position embeddings of shape (t, dim_embedding)
    pos_emb = self.transformer.wpe(pos)
    x = self.transformer.drop(tok_emb + pos_emb)
    for block in self.transformer.h:
      x = block(x)
    x = self.transformer.ln_f(x)

    if targets is not None:
      # Calculates the loss if we are given some desired targets
      logits = self.lm_head(x)
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                             targets.view(-1), ignore_index=-1)
    else:
      # Inference-time mini-optimization: only forward the lm_head on the very last position
      # note: using list [-1] to preserve the time dim
      logits = self.lm_head(x[:, [-1], :])
      loss = None

    return logits, loss

  def get_num_params(self, non_embedding=True):
    """Return the number of parameters in the model, without the consideration of precision.
      For non-embedding count (default), the position embeddings get subtracted.
      The token embeddings would too, except due to the parameter sharing these
      params are actually used as weights in the final layer, so we include them.
    """
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
      n_params -= self.transformer.wpe.weight.numel()
    return n_params

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str):
    """Sets up an AdamW optimizer for training a GPT model.

    Args:
        weight_decay (float): The weight decay factor.
        learning_rate (float): The learning rate for the optimizer.
        betas (Tuple[float, float]): The beta parameters for the AdamW optimizer.
        device_type (str): The type of device ('cuda' or 'cpu').
    """
    # Get all parameters that require gradients.
    param_dict = {pn: p for pn, p in self.named_parameters()
                  if p.requires_grad}

    # Create optim groups. Any parameters that is at least 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    # Gets and prints the number of parameters in the optimizer groups.
    # p.numel() returns the number of elements in tensor p.
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
      f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(
      f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    # Some versions of PyTorch support a fused kernel implementation of AdamW (much faster on GPU).
    # This code checks dynamically if it's available.
    # If so, it passes fused=True to enable it.
    fused_available = 'fused' in inspect.signature(
      torch.optim.AdamW).parameters

    # Create AdamW optimizer and use the fused version if it is available.
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
      optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"Whether to use fused AdamW: {use_fused}")

    return optimizer

  def estimate_mfu(self, fwdbwd_per_iter, dt) -> float:
    """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
    # First estimate the number of flops we do per iteration.
    # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    N = self.get_num_params()
    cfg = self.config
    L, H, Q, T = cfg.num_layers, cfg.num_heads, cfg.dim_embedding // cfg.num_heads, cfg.seq_length
    flops_per_token = 6 * N + 12 * L * H * Q * T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    # express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0 / dt)  # per second
    flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu
