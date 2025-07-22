import dataclasses
import jax.numpy as jnp


@dataclasses.dataclass
class ModelConfig:
  """ModelConfig is a dataclass that holds the configuration parameters for a GPT-like model.

  Attributes:
    seq_length (int): The size of the blocks used in the model. Default is 1024.
    vocab_size (int): The size of the vocabulary. Default is 50304, which is the GPT-2 vocab size of 50257 padded up to the nearest multiple of 64 for efficiency.
    num_layers (int): The number of layers in the model. Default is 12.
    num_heads (int): The number of attention heads in the model. Default is 12.
    dim_embedding (int): The dimensionality of the embeddings. Default is 768.
    dropout_rate (float): The dropout rate. Default is 0.0.
    use_bias (bool): Whether to use bias in Linear and LayerNorm layers. Default is True. Setting this to False can make the model a bit better and faster.
  """
  # The token embeddings. For example, we have vocab_size words, each word is a vector of size dim_embedding.
  # GPT-2 vocab_size is 50257, padded up to nearest multiple of 64 for efficiency.]
  # In GPT_BY_HAND, we use 64 as the vocabulary size.
  vocab_size: int = 64  # 50304
  # The position embeddings. For example, in a sentence we have block_size / seq_length positions,
  # each position is a vector of size dim_embedding.
  # It tells us the position of the word in the sentence.
  seq_length: int = 16  # GPT2 uses 1024
  num_layers: int = 2  # GPT2 uses 12
  num_heads: int = 2  # GPT2 uses 12
  # The embedding dimensionality.
  dim_embedding: int = 32  # GPT2 uses 768
  dropout_rate: float = 0.0
  # True: bias in Linears and LayerNorms, like GPT-2. 
  # False: a bit better and faster
  use_bias: bool = False
