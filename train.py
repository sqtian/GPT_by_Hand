import argparse
import os
import time
import math
import pickle
from contextlib import nullcontext
from typing import Tuple
import dataclasses

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

from model import GPT
from model_config import ModelConfig
from train_config import *
import wandb


class GPTTrainer:
  """GPTTrainer is a class that handles the training of the GPT model."""

  def __init__(self, train_config: TrainConfig, model_config: ModelConfig):
    """Initialize the GPTTrainer with the given configuration and model.

    Args:
      config (TrainConfig): The training configuration.
      model (GPT): The GPT model to be trained.
    """
    # It ensures us to use either automatic mixed precision (AMP) on GPU,
    # or nothing special on CPU (where AMP doesn’t help).
    self.is_ddp = self.is_ddp_environment()
    self.ctx = nullcontext()
    self.model_config = model_config
    self.train_config = self.setup_train_config(train_config)
    # Configures PyTorch settings based on the training configuration.
    self.configure_torch()

  def is_ddp_environment(self) -> bool:
    """Check if the environment is a distributed environment."""
    return int(os.environ.get('RANK', -1)) != -1  # If true, we are in a distributed environment.

  def get_global_env(self) -> dict:
    """Get the global environment variables."""
    config_keys = [k for k, v in globals().items() if not k.startswith(
      '_') and isinstance(v, (int, float, bool, str))]
    # Returns a dict for all global variables that are not private (not starting with '_')
    # and are of type int, float, bool or str.
    # This is useful for debugging or logging purposes.
    self.env_configs = {k: globals()[k] for k in config_keys}
    return self.env_configs

  def setup_train_config(self, train_config: TrainConfig) -> TrainConfig:
    """Setup training configs for distributed training and returns the updated TrainConfig."""
    if self.is_ddp:
      # Initialize the backend for PyTorch.
      init_process_group(backend=train_config.backend)
      # Sets up the DDP environment in TrainConfig.
      # Unique ID of this process across all nodes.
      train_config.ddp_rank = int(os.environ['RANK'])
      # The GPU index on one node.
      train_config.ddp_local_rank = int(os.environ['LOCAL_RANK'])
      # The total number of processes across all nodes.
      train_config.ddp_world_size = int(os.environ['WORLD_SIZE'])
      train_config.device = f'cuda:{train_config.ddp_local_rank}'
      # Set the device for this process.
      torch.cuda.set_device(train_config.device)
      train_config.master_process = (train_config.ddp_rank == 0)
      # Each process gets a different seed
      train_config.seed_offset = train_config.ddp_rank

      # In Distributed Data Parallel (DDP) training, each GPU runs in its own process.
      # `world_size` number of processes will be training simultaneously, so we can scale
      # down the desired gradient accumulation iterations per process proportionally
      # We want to accumulate gradients across multiple forward/backward passes
      # before updating weights — especially if your batch size is too large to fit in memory.
      # That’s what gradient_accumulation_steps is for.
      # Checks to ensure we can evenly split the accumulation steps across all DDP processes.
      assert train_config.gradient_accumulation_steps % train_config.ddp_world_size == 0
      # Divides the global number of accumulation steps by the number of processes.
      train_config.gradient_accumulation_steps //= train_config.ddp_world_size
    else:
      train_config.master_process = True
      train_config.seed_offset = 0
      train_config.ddp_world_size = 1

    # Updates the value of `tokens_per_iter` in the TrainConfig.
    # This is the number of tokens processed per iteration across all processes.
    train_config.tokens_per_iter = train_config.gradient_accumulation_steps * \
        train_config.ddp_world_size * train_config.batch_size * train_config.seq_length
    print(
      f"The number of tokens per iteration will be: {train_config.tokens_per_iter}")

    # Sets the device type based on the device string, for later use in torch.autocast.
    if 'cuda' in train_config.device:
      train_config.device_type = 'cuda'
    elif 'mps' in train_config.device:
      train_config.device_type = 'mps'
    else:
      train_config.device_type = 'cpu'

    # Sets the path to the data file.
    train_config.data_dir = os.path.join(
      train_config.data_dir, train_config.dataset)

    ### After updating the train_config, we need to set up some necessary objects. ###
    # Sets up the output directory for saving checkpoints and logs on master node.
    if train_config.master_process:
      os.makedirs(train_config.out_dir, exist_ok=True)

    # Sets the data type for the device.
    # It ensures us to use either automatic mixed precision (AMP) on GPU,
    # or nothing special on CPU (where AMP doesn’t help).
    self.ctx = nullcontext() if train_config.device_type == 'cpu' else torch.amp.autocast(
      device_type=train_config.device_type, dtype=train_config.dtype)

    return train_config

  def configure_torch(self):
    """Configure PyTorch settings based on the training configuration."""
    # Sets the random seed for reproducibility.
    torch.manual_seed(1337 + self.train_config.seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

  def get_batch(self, use_val: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gets the very first batch of data from either train or val datasets.

    Args:
        split: String indicating which split to use ('train' or 'val')

    Returns:
        tuple: (x, y) where x are the input tokens and y are the target tokens
    """
    # We recreate np.memmap every batch to avoid a memory leak.
    # See https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if not use_val:
      # Uses the training set.
      data = np.memmap(os.path.join(self.train_config.data_dir, 'train.bin'),
                       dtype=np.uint16, mode='r')
    else:
      # Uses the validation set.
      data = np.memmap(os.path.join(self.train_config.data_dir, 'val.bin'),
                       dtype=np.uint16, mode='r')

    # Randomly selects a starting index for the batch.
    # Sample `batch_size`` random starting indices such that:
    #   * Each sample is a slice like data[i : i + seq_length]
    #   * Never run off the end of the tensor.
    # E.g. ix = torch.randint(15, (4,)) => tensor([3, 7, 0, 11])
    ix = torch.randint(len(data) - self.train_config.seq_length,
                       (self.train_config.batch_size,))
    # Creates the input and target tensors.
    # For example,
    #     data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    #     seq_length = 4
    #     ix = [2, 4]
    # We got
    #    x = data[2:6], data[4:8] = [[2, 3, 4, 5], [4, 5, 6, 7]]
    #    y = data[3:7], data[5:9] = [[3, 4, 5, 6], [5, 6, 7, 8]]
    # And it's learning to predict 3 given 2, etc.
    x = torch.stack(
      [torch.from_numpy((data[i:i + self.train_config.seq_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(
      (data[i + 1:i + 1 + self.train_config.seq_length]).astype(np.int64)) for i in ix])

    if self.train_config.device_type == 'cuda':
      # Pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True).
      # 1. `pin_memory()`
      #    Puts the tensor in pinned (page-locked) memory.
      #    This is a special memory allocation that the GPU can access directly via DMA.
      #    Result: much faster transfer of data from CPU to GPU.
      # 2. `.to(self.train_config.device, non_blocking=True)`
      #    Moves the tensor to the target device.
      #    non_blocking=True means the CPU does not wait for the copy to finish.
      #    This allows asynchronous data transfer, letting you overlap data loading with computation.
      # So: ✔️ pin memory + ✔️ non-blocking transfer = maximum data pipeline speed.
      x = x.pin_memory().to(self.train_config.device, non_blocking=True)
      y = y.pin_memory().to(self.train_config.device, non_blocking=True)
    else:
      # Move the tensors to device.
      # Note: MPS doesn't support pin_memory(), so we move directly
      x = x.to(self.train_config.device)
      y = y.to(self.train_config.device)

    return x, y

  def setup_model(self) -> None:
    """Set up and create the GPT model from scratch."""
    # Attempts to derive vocab_size from the dataset.
    meta_path = os.path.join(self.train_config.data_dir, 'meta.pkl')
    meta_vocab_size = ModelConfig.vocab_size
    if os.path.exists(meta_path):
      with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")
    else:
      print(
        f"Warning: {meta_path} not found. Using default vocab_size = {meta_vocab_size}.")
      print(
        f"Default the vocab_size of GPT-2 to {meta_vocab_size} (50257 rounded up for efficiency).")

    print("Initializing a new model from scratch")

    # Updates the vocab_size and seq_length in the ModelConfig.
    self.model = GPT(ModelConfig(
      vocab_size=meta_vocab_size,
      seq_length=self.train_config.seq_length,
    ))

    # Check the sequence length of the model.
    assert self.model.config.seq_length == self.train_config.seq_length, \
        f"Model seq_length (block size) {self.model.config.seq_length} does not match training seq_length {self.train_config.seq_length}"

    # Move the model to the device.
    self.model.to(self.train_config.device)

    # Initialize a GradScaler. If enabled=False scaler is a no-op
    self.scaler = torch.cuda.amp.GradScaler(
      enabled=(self.train_config.dtype == torch.float16))
    # Initializes the optimizer.
    self.optimizer = self.model.configure_optimizers(self.train_config.weight_decay,
                                                     self.train_config.learning_rate,
                                                     (self.train_config.beta1,
                                                      self.train_config.beta2),
                                                     self.train_config.device_type)
    if self.train_config.compile:
      # Compiles the model for faster performance.
      print("Compiling the model...")
      self.model = torch.compile(self.model)

    # Wrap model into DDP container
    if self.is_ddp:
      self.model = DistributedDataParallel(self.model, device_ids=[
          self.train_config.ddp_local_rank])

    # Store initial weights for comparison in gpt_by_hand mode
    if hasattr(self.train_config, 'gpt_by_hand') and self.train_config.gpt_by_hand:
      self.store_initial_weights()

  @torch.no_grad()
  def estimate_loss(self):
    """Helps estimate an arbitrarily accurate loss over either split using many batches.
      @torch.no_grad() is used to tell PyTorch not to track gradients inside this function.
      Saves memory and speeds up inference — since we're just evaluating, not training.
      Why is this useful?
      You can call this periodically during training (e.g., every N iterations) to monitor progress
      Helps detect overfitting: if validation loss starts increasing while training loss decreases
      Easy to plug into a training dashboard or logger
    """
    # Store the final loss values.
    self.out = {}
    # self.model.eval() switches the model to evaluation mode,
    # disabling dropout, batchnorm updates, etc.
    self.model.eval()
    # Gets the loss for the training set and saves it in the output dict.
    losses = torch.zeros(self.train_config.eval_iters)
    for k in range(self.train_config.eval_iters):
      X, Y = self.get_batch(use_val=False)
      with self.ctx:
        # Run the model under the mixed precision or null context (self.ctx)
        logits, loss = self.model(X, Y)
      losses[k] = loss.item()
    self.out['train'] = losses.mean()

    # Gets the loss for the validation set and saves it in the output dict.
    losses = torch.zeros(self.train_config.eval_iters)
    for k in range(self.train_config.eval_iters):
      X, Y = self.get_batch(use_val=True)
      with self.ctx:
        # Run the model under the mixed precision or null context (self.ctx)
        logits, loss = self.model(X, Y)
      losses[k] = loss.item()
    self.out['val'] = losses.mean()

    # Switches model back to training mode (in case training resumes after this call).
    self.model.train()
    # Returns the dictionary like: {'train': 1.234, 'val': 1.567}.
    return self.out

  def get_lr(self, it: int) -> float:
    """Calculates learning rate based on current training iteration.

    This function gives you a learning rate based on the current training iteration (it).
    It is designed to:
        Start small (warmup)
        Ramp up linearly to the base LR
        Then decay smoothly using a cosine curve
        Eventually settle at a minimum LR for fine-tuning.

    Implements cosine learning rate decay with warmup:
    1) Linear warmup for warmup_iters steps
    2) If it > lr_decay_iters, return min learning rate
    3) In between, use cosine decay down to min learning rate

    Args:
        it: Current training iteration

    Returns:
        float: The calculated learning rate for this iteration
    """
    # 1) Linear warmup for warmup_iters steps.
    if it < self.train_config.warmup_iters:
      return self.train_config.learning_rate * it / self.train_config.warmup_iters
    # 2) If it > lr_decay_iters, return min learning rate
    if it > self.train_config.lr_decay_iters:
      return self.train_config.min_lr
    # 3) Use cosine decay down to min learning rate for iterations in between.
    decay_ratio = (it - self.train_config.warmup_iters) / (
        self.train_config.lr_decay_iters - self.train_config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return self.train_config.min_lr + coeff * (
        self.train_config.learning_rate - self.train_config.min_lr)

  def train(self) -> None:
    """Main training loop.
    This function is responsible for the entire training process.
    """
    # Inits the wandb logging.
    if self.train_config.wandb_log and self.train_config.master_process:
      wandb.init(project=self.train_config.wandb_project,
                 name=self.train_config.wandb_run_name, config=self.get_global_env())
    # Gets the very first batch.
    X, Y = self.get_batch(use_val=False)
    t0 = time.time()
    # The number of iterations in the lifetime of this process on local device.
    local_iter_num = 0
    running_mfu = -1.0
    # Unwrap DDP container if needed
    raw_model = self.model.module if self.is_ddp else self.model
    # Init the iteration counters.
    # `iter_num` denotes the number of iterations from scatch. Please note that it
    # can be overridden by resume from a checkpoint.
    # `local_iter_num` denotes the number of iterations in the lifetime of
    # this process on local device. It is the number of iterations since we start current run.
    iter_num = 0
    local_iter_num = 0
    best_val_loss = 1e9

    # Main training loop.
    while True:
      # Determine and set the learning rate for this iteration.
      lr = self.get_lr(
        iter_num) if self.train_config.decay_lr else self.train_config.learning_rate
      for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr

      # Every eval_interval (e.g. 2000) iterations, evaluate the loss on train/val
      # sets and write checkpoints. Please note that this is not the training.
      if iter_num % self.train_config.eval_interval == 0 and self.train_config.master_process:
        # The master process will evaluate the loss and write checkpoints every eval_interval iterations.
        losses = self.estimate_loss()
        print(
          f"Evaluating iteration {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # Logs the information.
        if self.train_config.wandb_log:
          wandb.log({
              "iter": iter_num,
              "train/loss": losses['train'],
              "val/loss": losses['val'],
              "lr": lr,
              "mfu": running_mfu * 100,  # convert to percentage
          })
        # Stores the best validation loss and saves the model checkpoint.
        if (not self.train_config.skip_save_checkpoint) and (losses['val'] < best_val_loss or self.train_config.always_save_checkpoint):
          best_val_loss = losses['val']
          if iter_num > 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'model_args': dataclasses.asdict(raw_model.config),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': self.get_global_env(),
            }
            print(f"Saving checkpoint to {self.train_config.out_dir}")
            torch.save(checkpoint, os.path.join(
              self.train_config.out_dir, 'ckpt.pt'))

      # For eval_only, returns immediately after the first iteration.
      if self.train_config.eval_only and iter_num == 0:
        break

      # Executes the forward backward pass, with optional gradient accumulation to simulate
      # larger batch size and using the GradScaler if data type is float16.
      for micro_step in range(self.train_config.gradient_accumulation_steps):
        # This for loop inside the big while loop is for the sub-iterations of the gradient accumulation.
        # It allows us to simulate a larger batch size by accumulating gradients over multiple iterations.
        # For example, if you can only fit batch size = 8 on your GPU, but you want to train as if
        # using batch size = 64,
        # then you can set batch_size = 8, gradient_accumulation_steps = 8
        # This will simulate a batch size of 64.

        # Sets up gradient sync for Distributed Data Parallel (DDP) training.
        # In DDP training, we only need to sync gradients at the last micro step.
        # the official way to do this is with model.no_sync() context manager, but
        # the author really dislike that as it bloats the code and forces us to repeat code.
        # Looking at the source of that context manager, it just toggles this variable.
        if self.is_ddp:
          self.model.require_backward_grad_sync = (
            micro_step == self.train_config.gradient_accumulation_steps - 1)

        with self.ctx:
          logits, loss = self.model(X, Y)
          # Scales the loss to account for gradient accumulation
          loss = loss / self.train_config.gradient_accumulation_steps

          # Print detailed info for first 2 iterations if gpt_by_hand is enabled
          if hasattr(self.train_config, 'gpt_by_hand') and self.train_config.gpt_by_hand and iter_num < 2:
            self.print_gpt_by_hand_info(
              iter_num, X, Y, logits, loss * self.train_config.gradient_accumulation_steps)

        # Prefetch next batch immediately asynchronously, while model is doing the forward pass on the GPU.
        X, Y = self.get_batch(use_val=False)

        # Conducts the backward pass, with gradient scaling if training in fp16.
        self.scaler.scale(loss).backward()

      # clip the gradient
      if self.train_config.grad_clip != 0.0:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
          self.model.parameters(), self.train_config.grad_clip)

      # Step the optimizer and scaler if training in fp16.
      self.scaler.step(self.optimizer)
      self.scaler.update()

      # Print weight updates for first 2 iterations if gpt_by_hand is enabled
      if hasattr(self.train_config, 'gpt_by_hand') and self.train_config.gpt_by_hand and iter_num < 2:
        self.print_weight_updates(iter_num)

      # Flush the gradients to be 0 as soon as we can, no need for this memory anymore.
      self.optimizer.zero_grad(set_to_none=True)

      # timing and logging
      t1 = time.time()
      dt = t1 - t0
      t0 = t1
      if iter_num % self.train_config.log_interval == 0 and self.train_config.master_process:
        # Get loss as float. note: this is a CPU-GPU sync point.
        # Scales it up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * self.train_config.gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
          # MFU (Model FLOPs Utilization)
          # MFU = (Actual FLOPs / Theoretical peak FLOPs) × 100%
          mfu = raw_model.estimate_mfu(
            self.train_config.batch_size * self.train_config.gradient_accumulation_steps, dt)
          running_mfu = mfu if running_mfu == - \
              1.0 else (0.9 * running_mfu + 0.1 * mfu)
        print(
            f"Iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

      iter_num += 1
      local_iter_num += 1

      # Termination the training loop.
      if iter_num > self.train_config.max_iters:
        break

    if self.is_ddp:
      destroy_process_group()

  def print_gpt_by_hand_info(self, iter_num: int, X: torch.Tensor, Y: torch.Tensor, logits: torch.Tensor, loss: torch.Tensor) -> None:
    """Print detailed information for manual calculation tracing.

    Args:
        iter_num: Current iteration number
        X: Input tokens
        Y: Target tokens  
        logits: Model output logits
        loss: Computed loss
    """
    print(f"\n{'=' * 80}")
    print(f"GPT BY HAND - ITERATION {iter_num}")
    print(f"{'=' * 80}")

    # Load vocabulary for token-to-word conversion
    meta_path = os.path.join(self.train_config.data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
      with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
        itos = meta.get('itos', {i: str(i)
                        for i in range(self.model.config.vocab_size)})
    else:
      itos = {i: str(i) for i in range(self.model.config.vocab_size)}

    print(f"\n--- INPUT DATA ---")
    print(f"Input shape (X): {X.shape}")
    print(f"Target shape (Y): {Y.shape}")

    # Print tokens and their corresponding words
    for batch_idx in range(X.shape[0]):
      input_tokens = X[batch_idx].cpu().numpy()
      target_tokens = Y[batch_idx].cpu().numpy()

      input_words = [
        itos.get(int(token), f"UNK_{token}") for token in input_tokens]
      target_words = [
        itos.get(int(token), f"UNK_{token}") for token in target_tokens]

      print(f"\nBatch {batch_idx}:")
      print(f"  Input tokens:  {input_tokens}")
      print(f"  Input words:   {input_words}")
      print(f"  Target tokens: {target_tokens}")
      print(f"  Target words:  {target_words}")

    # Get model components for detailed inspection
    raw_model = self.model.module if self.is_ddp else self.model

    print(f"\n--- MODEL ARCHITECTURE ---")
    print(f"Vocab size: {raw_model.config.vocab_size}")
    print(f"Sequence length: {raw_model.config.seq_length}")
    print(f"Embedding dim: {raw_model.config.dim_embedding}")
    print(f"Number of layers: {raw_model.config.num_layers}")
    print(f"Number of heads: {raw_model.config.num_heads}")

    # Print embeddings and activations for the first batch
    print(f"\n--- EMBEDDINGS & ACTIVATIONS (Batch 0) ---")
    with torch.no_grad():
      device = X.device
      b, t = X[0:1].size()  # First batch only

      # Token embeddings
      tok_emb = raw_model.transformer.wte(X[0:1])
      print(f"Token embeddings shape: {tok_emb.shape}")
      print(f"Token embeddings:\n{tok_emb[0].cpu().numpy()}")

      # Position embeddings
      pos = torch.arange(0, t, dtype=torch.long, device=device)
      pos_emb = raw_model.transformer.wpe(pos)
      print(f"\nPosition embeddings shape: {pos_emb.shape}")
      print(f"Position embeddings:\n{pos_emb.cpu().numpy()}")

      # Combined embeddings (after dropout)
      x = raw_model.transformer.drop(tok_emb + pos_emb)
      print(f"\nCombined embeddings (after dropout) shape: {x.shape}")
      print(f"Combined embeddings:\n{x[0].cpu().numpy()}")

      # First transformer block activations
      first_block = raw_model.transformer.h[0]

      # Layer norm 1 output
      x_norm1 = first_block.ln_1(x)
      print(f"\nAfter first LayerNorm shape: {x_norm1.shape}")
      print(f"After first LayerNorm:\n{x_norm1[0].cpu().numpy()}")

      # Attention output (just the output, not intermediate Q,K,V for simplicity)
      attn_out = first_block.attn(x_norm1)
      print(f"\nAttention output shape: {attn_out.shape}")
      print(f"Attention output:\n{attn_out[0].cpu().numpy()}")

      # After first residual connection
      x_after_attn = x + attn_out
      print(
        f"\nAfter attention residual connection shape: {x_after_attn.shape}")
      print(f"After attention residual:\n{x_after_attn[0].cpu().numpy()}")

      # Layer norm 2 output
      x_norm2 = first_block.ln_2(x_after_attn)
      print(f"\nAfter second LayerNorm shape: {x_norm2.shape}")
      print(f"After second LayerNorm:\n{x_norm2[0].cpu().numpy()}")

      # MLP/FFN output
      mlp_out = first_block.mlp(x_norm2)
      print(f"\nMLP output shape: {mlp_out.shape}")
      print(f"MLP output:\n{mlp_out[0].cpu().numpy()}")

      # After second residual connection (output of first transformer block)
      x_after_mlp = x_after_attn + mlp_out
      print(f"\nAfter MLP residual connection shape: {x_after_mlp.shape}")
      print(
        f"After MLP residual (first block output):\n{x_after_mlp[0].cpu().numpy()}")

    print(f"\n--- MODEL WEIGHTS (First Layer) ---")
    # Print some key weights for manual calculation
    first_attn = raw_model.transformer.h[0].attn
    print(f"Attention c_attn weight shape: {first_attn.c_attn.weight.shape}")
    print(
      f"Attention c_attn weights (first 8x8):\n{first_attn.c_attn.weight[:8, :8].cpu().detach().numpy()}")

    if first_attn.c_attn.bias is not None:
      print(f"Attention c_attn bias shape: {first_attn.c_attn.bias.shape}")
      print(
        f"Attention c_attn bias (first 8):\n{first_attn.c_attn.bias[:8].cpu().detach().numpy()}")

    print(f"\n--- FORWARD PASS OUTPUTS ---")
    print(f"Logits shape: {logits.shape}")
    print(
      f"Logits (first sequence, first 8 vocab items):\n{logits[0, :, :8].cpu().detach().numpy()}")

    print(f"\nLoss: {loss.item():.6f}")

    # Print softmax probabilities for the last position (next token prediction)
    with torch.no_grad():
      last_logits = logits[0, -1, :]  # Last position of first batch
      probs = torch.softmax(last_logits, dim=-1)

      print(f"\nNext token probabilities (last position, batch 0):")
      # Show top 5 most likely tokens
      top_probs, top_indices = torch.topk(probs, min(5, len(probs)))
      for i, (prob, idx) in enumerate(zip(top_probs.cpu(), top_indices.cpu())):
        word = itos.get(int(idx), f"UNK_{idx}")
        print(f"  {i + 1}. Token {int(idx)} ('{word}'): {prob.item():.4f}")

    print(f"\n--- GRADIENTS ---")
    # Print gradients for key parameters
    if first_attn.c_attn.weight.grad is not None:
      print(f"Attention c_attn weight gradient (first 4x4):")
      print(f"{first_attn.c_attn.weight.grad[:4, :4].cpu().numpy()}")
    else:
      print("No gradients computed yet")

    # Print embedding gradients
    if raw_model.transformer.wte.weight.grad is not None:
      print(f"\nToken embedding gradients (first 4x4):")
      print(f"{raw_model.transformer.wte.weight.grad[:4, :4].cpu().numpy()}")

    print(f"\n--- OPTIMIZER STATE ---")
    print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
    print(f"Weight decay: {self.optimizer.param_groups[0]['weight_decay']}")

    print(f"\n{'=' * 80}")
    print(f"END ITERATION {iter_num}")
    print(f"{'=' * 80}\n")

  def print_weight_updates(self, iter_num: int) -> None:
    """Print weight updates after optimizer step for manual calculation tracing.

    Args:
        iter_num: Current iteration number
    """
    print(f"\n--- WEIGHT UPDATES (Iteration {iter_num}) ---")

    raw_model = self.model.module if self.is_ddp else self.model

    # Print updates for first attention layer
    first_attn = raw_model.transformer.h[0].attn
    lr = self.optimizer.param_groups[0]['lr']

    print(f"Learning rate: {lr}")

    # Show weight changes if we have initial weights stored
    if hasattr(self, 'initial_weights'):
      current_attn_weight = first_attn.c_attn.weight[:4, :4].cpu().detach()
      initial_attn_weight = self.initial_weights['attn_c_attn'][:4, :4].cpu()
      weight_change = current_attn_weight - initial_attn_weight

      print(f"\nAttention c_attn weight changes (first 4x4):")
      print(f"{weight_change.numpy()}")

      print(f"Attention c_attn weights after update (first 4x4):")
      print(f"{current_attn_weight.numpy()}")

      # Show embedding weight changes
      current_emb_weight = raw_model.transformer.wte.weight[:4, :4].cpu(
      ).detach()
      initial_emb_weight = self.initial_weights['token_emb'][:4, :4].cpu()
      emb_weight_change = current_emb_weight - initial_emb_weight

      print(f"\nToken embedding weight changes (first 4x4):")
      print(f"{emb_weight_change.numpy()}")

      print(f"Token embedding weights after update (first 4x4):")
      print(f"{current_emb_weight.numpy()}")
    else:
      # Fallback if no initial weights stored
      print(f"Attention c_attn weights (first 4x4) after update:")
      print(f"{first_attn.c_attn.weight[:4, :4].cpu().detach().numpy()}")

      print(f"\nToken embedding weights (first 4x4) after update:")
      print(
        f"{raw_model.transformer.wte.weight[:4, :4].cpu().detach().numpy()}")

    print(f"--- END WEIGHT UPDATES ---\n")

  def store_initial_weights(self) -> None:
    """Store initial weights for comparison in gpt_by_hand mode."""
    raw_model = self.model.module if self.is_ddp else self.model

    # Store some key initial weights for comparison
    self.initial_weights = {}

    # Store attention weights
    first_attn = raw_model.transformer.h[0].attn
    self.initial_weights['attn_c_attn'] = first_attn.c_attn.weight.clone(
    ).detach()

    # Store embedding weights
    self.initial_weights['token_emb'] = raw_model.transformer.wte.weight.clone(
    ).detach()

    print(f"\n--- INITIAL WEIGHTS STORED ---")
    print(f"Initial attention c_attn weights (first 4x4):")
    print(f"{self.initial_weights['attn_c_attn'][:4, :4].cpu().numpy()}")

    print(f"\nInitial token embedding weights (first 4x4):")
    print(f"{self.initial_weights['token_emb'][:4, :4].cpu().numpy()}")
    print(f"--- END INITIAL WEIGHTS ---\n")


if __name__ == "__main__":
  # Set up argument parser
  parser = argparse.ArgumentParser(description='Train a GPT model')

  # Add device argument with default value
  parser.add_argument('--device', type=str, default=None,
                      help='Device to train on (e.g., "cpu", "cuda", "cuda:0", "mps"). If not specified, auto-detects.')
  parser.add_argument('--max_iters', type=int, default=None,
                      help='The maximum number of training iterations. Overrides config setting.')
  parser.add_argument('--dataset', type=str, default=None,
                      help='Dataset to use for training (e.g., "shakespeare_char", "tiny_demo", "tiny_demo_words"). Overrides config setting.')
  parser.add_argument('--gpt_by_hand', action='store_true',
                      help='Enable detailed printing for first 2 iterations to trace calculations by hand.')
  parser.add_argument('--batch_size', type=int, default=None,
                      help='Batch size for training. Overrides config setting.')
  parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                      help='Number of gradient accumulation steps. Overrides config setting.')
  args = parser.parse_args()

  # Auto-detect device if not specified
  if args.device is None:
    if torch.cuda.is_available():
      args.device = 'cuda'
    elif torch.backends.mps.is_available():
      args.device = 'mps'
    else:
      args.device = 'cpu'

  # Create the model and training configurations.
  model_config = ModelConfig()

  # Build TrainConfig with all command line arguments
  train_config_kwargs = {
    'device': args.device,
  }

  # Add optional arguments if provided
  if args.dataset:
    train_config_kwargs['dataset'] = args.dataset
  if args.max_iters:
    train_config_kwargs['max_iters'] = args.max_iters
  if args.batch_size:
    train_config_kwargs['batch_size'] = args.batch_size
  if args.gradient_accumulation_steps:
    train_config_kwargs['gradient_accumulation_steps'] = args.gradient_accumulation_steps
  if args.gpt_by_hand:
    train_config_kwargs['gpt_by_hand'] = True
    train_config_kwargs['skip_save_checkpoint'] = True
    # Limit to 2 iterations for gpt_by_hand mode
    train_config_kwargs['max_iters'] = 2

  train_config = TrainConfig(**train_config_kwargs)

  print(f"Using device: {train_config.device}")
  print(f"Using dataset: {train_config.dataset}")
  print(f"Using batch size: {train_config.batch_size}")
  print(
    f"Using gradient accumulation steps: {train_config.gradient_accumulation_steps}")
  print(f"Using max iterations: {train_config.max_iters}")

  # Initialize the trainer and start training.
  trainer = GPTTrainer(train_config, model_config)
  trainer.setup_model()
  trainer.train()

  print("Training completed.")
