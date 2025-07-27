import dataclasses
import torch
import model_config


# Constants for `init_from` options
INIT_FROM_SCRATCH = "scratch"
INIT_FROM_RESUME = "resume"
INIT_FROM_GPT2 = "gpt2"
INIT_FROM_GPT2_MEDIUM = "gpt2-medium"
INIT_FROM_GPT2_LARGE = "gpt2-large"
INIT_FROM_GPT2_XL = "gpt2-xl"

# Constants for `device` options
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_CUDA_0 = "cuda:0"
DEVICE_CUDA_1 = "cuda:1"
DEVICE_MPS = "mps"  # Apple Silicon GPU


@dataclasses.dataclass
class TrainConfig:
  """The configs for training the model."""
  # Default config values designed to train a gpt2 (124M) on OpenWebText
  # I/O
  out_dir = 'out'
  eval_interval = 100  # Reduced for demo - evaluate more frequently; Was 2000.
  log_interval = 50
  eval_iters = 20  # Reduced for demo; Was 200
  eval_only = False  # if True, script exits right after the first eval
  always_save_checkpoint = True  # if True, always save a checkpoint after each eval
  init_from = INIT_FROM_SCRATCH  # 'scratch' or 'resume' or 'gpt2*'
  # wandb logging
  wandb_log = False  # disabled by default
  wandb_project = 'gpt-by-hand'  # Updated project name
  wandb_run_name = 'demo-gpt'  # Updated run name
  # data
  dataset = 'tiny_demo'  # 'tiny_demo' or 'shakespeare_char'
  data_dir = 'data/'  # The path to the folder that contains dataset.
  # Used to simulate a larger batch size.
  # If you can only fit batch size = 8 on your GPU
  # But you want to train as if using batch size = 64
  # Then you can set batch_size = 8, gradient_accumulation_steps = 8
  # This will simulate a batch size of 64.
  # Normally, training looks like:
  #   for x in data:
  #     loss = model(x)
  #     loss.backward()
  #     optimizer.step()
  #     optimizer.zero_grad()
  # But with gradient accumulation:
  #   for i, x in enumerate(data):
  #     loss = model(x)
  #     loss.backward()
  #     if (i + 1) % gradient_accumulation_steps == 0:
  #       optimizer.step()
  #       optimizer.zero_grad()
  gradient_accumulation_steps = 4  # Reduced for demo; Was 5*8
  batch_size = 4  # Reduced for demo - small batches for small model; Was 12
  seq_length = model_config.ModelConfig.seq_length  # Aka block_size.
  # The number of tokens per iteration, assuming we only have 1 process on 1 GPU.
  # If ddp_world_size > 1, we will divide gradient_accumulation_steps by ddp_world_size,
  # and the number of tokens per iteration will be
  # gradient_accumulation_steps * ddp_world_size * batch_size * seq_length.
  tokens_per_iter = gradient_accumulation_steps * batch_size * seq_length
  # model configs are defined in model_config.py
  # adamw optimizer
  learning_rate = 3e-4  # Slightly reduced learning rate; Max learning rate, Was 6e-4
  # Much reduced for demo - 2000 iterations instead of 600k; Total number of training iterations
  max_iters: int = 2000
  weight_decay = 1e-1
  beta1 = 0.9
  beta2 = 0.95
  grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
  # learning rate decay settings
  decay_lr = True  # whether to decay the learning rate
  warmup_iters = 100  # Reduced warmup for demo; Was 2000
  lr_decay_iters = 2000  # should be ~= max_iters per Chinchilla; Was 600k
  min_lr = 3e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla; Was 6e-5
  # system
  device: str = DEVICE_CPU  # Will be auto-detected or overridden
  # for later use in torch.autocast
  device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
  dtype = torch.bfloat16 if torch.cuda.is_available(
      # The latter will auto implement a GradScaler
  ) and torch.cuda.is_bf16_supported() and (device == DEVICE_CUDA) else torch.float16
  compile = False  # If true, use PyTorch 2.0 to compile the model to be faster

  # Distributed Data Parallel (DDP) settings
  backend = 'nccl'  # 'nccl', 'gloo', etc.
  # If True, this process will be the master process for DDP.
  master_process = True
  # Different seed per process, used to ensure each process sees different data orders (e.g., in shuffling).
  seed_offset = 0
  ddp_world_size = 1  # Total number of processes across all nodes.
  ddp_rank = 0  # Unique ID of this process (across all nodes).
  # GPU index on this machine (used to set correct CUDA device), e.g. cuda:0
  ddp_local_rank = 0
