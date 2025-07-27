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
    train_config.device_type = 'cuda' if 'cuda' in train_config.device else 'cpu'

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
        if losses['val'] < best_val_loss or self.train_config.always_save_checkpoint:
          best_val_loss = losses['val']
          if iter_num > 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'model_args': dataclasses.asdict(self.model.config),
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


if __name__ == "__main__":
  # Set up argument parser
  parser = argparse.ArgumentParser(description='Train a GPT model')

  # Add device argument with default value
  parser.add_argument('--device', type=str, default="cuda",
                      help='Device to train on (e.g., "cpu", "cuda", "cuda:0", "mps"). Overrides config setting.')
  parser.add_argument('--max_iters', type=int, default=None,
                      help='The maximum number of training iterations. Overrides config setting.')
  parser.add_argument('--dataset', type=str, default=None,
                      help='Dataset to use for training (e.g., "shakespeare_char", "tiny_demo"). Overrides config setting.')
  args = parser.parse_args()

  # Create the model and training configurations.
  model_config = ModelConfig()
  train_config = TrainConfig(
    device=args.device,
  )

  print(f"Using device: {train_config.device}")

  # Override train_config with command line arguments.
  if args.dataset:
    train_config.dataset = args.dataset

  if args.max_iters:
    train_config.max_iters = args.max_iters

  # Initialize the trainer and start training.
  trainer = GPTTrainer(train_config, model_config)
  trainer.setup_model()
  trainer.train()

  print("Training completed.")
