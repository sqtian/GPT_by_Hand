import unittest
import os
from unittest.mock import PropertyMock, patch, MagicMock

import torch
import torch.nn as nn
import numpy as np

from train import GPTTrainer
from model_config import ModelConfig
from train_config import TrainConfig, DEVICE_CPU, DEVICE_CUDA
import tempfile
import model_config

class TestGPTTrainer(unittest.TestCase):
  def setUp(self):
    """Set up test fixtures"""
    self.model_config = ModelConfig()
    self.train_config = TrainConfig()
    # Create a temporary directory for get_batch tests
    self.temp_dir = None

  def tearDown(self):
    """Clean up temporary files."""
    if self.temp_dir:
      self.temp_dir.cleanup()

  @patch.object(GPTTrainer, 'setup_train_config')
  def test_init(self, mock_setup_train_config):
    """Test trainer initialization with mocked setup_train_config"""
    # Configure the mock to return a modified config
    modified_config = TrainConfig()
    modified_config.device_type = 'mocked_device'
    mock_setup_train_config.return_value = modified_config

    # Initialize the trainer
    trainer = GPTTrainer(self.train_config, self.model_config)

    # Verify setup_train_config was called with the right arguments
    mock_setup_train_config.assert_called_once_with(self.train_config)

    # Verify the trainer is using the modified config returned by setup_train_config
    self.assertEqual(trainer.train_config, modified_config)
    self.assertEqual(trainer.model_config, self.model_config)
    self.assertEqual(trainer.train_config.device_type, 'mocked_device')

  def test_is_ddp_environment_false(self):
    """Test DDP environment detection when not in DDP mode"""
    # Need to patch setup_train_config since it's called in __init__
    with patch.object(GPTTrainer, 'setup_train_config', return_value=self.train_config):
      trainer = GPTTrainer(self.train_config, self.model_config)

      # Ensure RANK is not set in environment
      with patch.dict(os.environ, {}, clear=True):
        self.assertFalse(trainer.is_ddp_environment())

  def test_is_ddp_environment_true(self):
    """Test DDP environment detection when in DDP mode"""
    # Need to patch setup_train_config since it's called in __init__
    with patch.object(GPTTrainer, 'setup_train_config', return_value=self.train_config):
      trainer = GPTTrainer(self.train_config, self.model_config)

      # Mock RANK environment variable
      with patch.dict(os.environ, {'RANK': '0'}):
        self.assertTrue(trainer.is_ddp_environment())

  @patch('train.init_process_group')
  @patch('torch.cuda.set_device')
  def test_setup_ddp_in_distributed_mode(self, mock_set_device, mock_init_process_group):
    """Test DDP setup in distributed environment"""
    # Mock DDP environment variables
    env_vars = {
        'RANK': '1',
        'LOCAL_RANK': '0',
        'WORLD_SIZE': '2'
    }

    with patch.dict(os.environ, env_vars):
      # Set gradient_accumulation_steps to a value divisible by world_size
      self.train_config.gradient_accumulation_steps = 10
      self.train_config.batch_size = 4
      self.train_config.seq_length = 1024
      trainer = GPTTrainer(self.train_config, self.model_config)

      # `setup_train_config()` is called inside __init__ of GPTTrainer.
      updated_config = trainer.train_config

      # Verify init_process_group was called
      mock_init_process_group.assert_called_once_with(
        backend=self.train_config.backend)

      # Verify device was set correctly
      mock_set_device.assert_called_once_with(f'cuda:0')

      # Check updated config values
      self.assertEqual(updated_config.ddp_rank, 1)
      self.assertEqual(updated_config.ddp_local_rank, 0)
      self.assertEqual(updated_config.ddp_world_size, 2)
      self.assertEqual(updated_config.device, 'cuda:0')
      self.assertFalse(updated_config.master_process)
      self.assertEqual(updated_config.seed_offset, 1)

      # Check gradient accumulation steps
      self.assertEqual(updated_config.gradient_accumulation_steps, 5)

      # Check tokens_per_iter calculation
      expected_tokens = 5 * 2 * 4 * 1024  # gas * world_size * batch_size * seq_length
      self.assertEqual(updated_config.tokens_per_iter, expected_tokens)

  def test_setup_ddp_in_non_distributed_mode(self):
    """Test DDP setup in non-distributed environment"""
    trainer = GPTTrainer(self.train_config, self.model_config)

    # Ensure no DDP environment variables are set
    with patch.dict(os.environ, {}, clear=True):
      self.train_config.batch_size = 4
      self.train_config.seq_length = 1024
      self.train_config.gradient_accumulation_steps = 8

      # Call the setup function
      updated_config = trainer.setup_train_config(self.train_config)

      # Check updated config values for non-DDP
      self.assertTrue(updated_config.master_process)
      self.assertEqual(updated_config.seed_offset, 0)
      self.assertEqual(updated_config.ddp_world_size, 1)
      self.assertEqual(
        updated_config.gradient_accumulation_steps, 8)  # Unchanged

      # Check tokens_per_iter calculation
      expected_tokens = 8 * 1 * 4 * 1024  # gas * world_size * batch_size * seq_length
      self.assertEqual(updated_config.tokens_per_iter, expected_tokens)

  def _setup_get_batch_test(self):
    """Helper method to set up get_batch tests."""
    # Create a small temporary directory for test data
    self.temp_dir = tempfile.TemporaryDirectory()

    # Set up a custom TrainConfig for testing
    test_config = TrainConfig()
    test_config.data_dir = self.temp_dir.name
    test_config.seq_length = 10
    test_config.batch_size = 4
    test_config.device = DEVICE_CPU
    test_config.device_type = "cpu"

    # These values need to be saved for assertions
    self.seq_length = test_config.seq_length
    self.batch_size = test_config.batch_size

    # Create small test datasets as memmap files
    train_data = np.arange(50, dtype=np.uint16)
    val_data = np.arange(50, 100, dtype=np.uint16)

    train_path = os.path.join(self.temp_dir.name, 'train.bin')
    val_path = os.path.join(self.temp_dir.name, 'val.bin')

    # Write data to memmap files
    train_memmap = np.memmap(train_path, dtype=np.uint16,
                             mode='w+', shape=train_data.shape)
    train_memmap[:] = train_data[:]
    train_memmap.flush()

    val_memmap = np.memmap(val_path, dtype=np.uint16,
                           mode='w+', shape=val_data.shape)
    val_memmap[:] = val_data[:]
    val_memmap.flush()

    # Need to patch setup_train_config to return our test config
    with patch.object(GPTTrainer, 'setup_train_config', return_value=test_config):
      trainer = GPTTrainer(test_config, self.model_config)
      return trainer

  def test_get_train_batch(self):
    """Test getting a batch from the training set."""
    trainer = self._setup_get_batch_test()

    x, y = trainer.get_batch(use_val=False)

    # Check shapes
    self.assertEqual(x.shape, (self.batch_size, self.seq_length))
    self.assertEqual(y.shape, (self.batch_size, self.seq_length))

    # Check device
    self.assertEqual(x.device.type, "cpu")
    self.assertEqual(y.device.type, "cpu")

    # Check data type
    self.assertEqual(x.dtype, torch.int64)
    self.assertEqual(y.dtype, torch.int64)

    # Check target is shifted by 1 from input
    for i in range(self.batch_size):
      self.assertTrue(torch.all(x[i, 1:] == y[i, :-1]))

  def test_get_val_batch(self):
    """Test getting a batch from the validation set."""
    trainer = self._setup_get_batch_test()

    x, y = trainer.get_batch(use_val=True)

    # Check shapes
    self.assertEqual(x.shape, (self.batch_size, self.seq_length))
    self.assertEqual(y.shape, (self.batch_size, self.seq_length))

    # Check device
    self.assertEqual(x.device.type, "cpu")
    self.assertEqual(y.device.type, "cpu")

    # Check all values are from the validation set (i.e., >= 50)
    self.assertTrue((x >= 50).all())
    self.assertTrue((y >= 50).all())

  @patch('torch.Tensor.pin_memory')
  @patch('torch.Tensor.to')
  def test_cuda_path(self, mock_to, mock_pin_memory):
    """Test CUDA path with mocks for pin_memory and to."""
    # Create a test config with CUDA device
    test_config = TrainConfig()
    test_config.data_dir = tempfile.mkdtemp()
    test_config.seq_length = 10
    test_config.batch_size = 4
    test_config.device = DEVICE_CUDA
    test_config.device_type = "cuda"

    # Need to ensure temporary directory exists with test files
    self.temp_dir = tempfile.TemporaryDirectory()
    test_config.data_dir = self.temp_dir.name

    # Create test files
    train_path = os.path.join(self.temp_dir.name, 'train.bin')
    train_data = np.arange(50, dtype=np.uint16)
    train_memmap = np.memmap(train_path, dtype=np.uint16,
                             mode='w+', shape=train_data.shape)
    train_memmap[:] = train_data[:]
    train_memmap.flush()

    # Setup mocks
    mock_pin_memory.return_value = torch.zeros(1)
    mock_to.return_value = torch.zeros(1)

    # Use patched setup_train_config to avoid calling the real one during initialization
    with patch.object(GPTTrainer, 'setup_train_config', return_value=test_config):
      trainer = GPTTrainer(test_config, self.model_config)

      # Call get_batch
      trainer.get_batch(use_val=False)

      # Verify pin_memory was called
      self.assertEqual(mock_pin_memory.call_count, 2)  # Once for x, once for y

      # Verify to() was called with non_blocking=True
      mock_to.assert_called_with(DEVICE_CUDA, non_blocking=True)

  def test_random_indices(self):
    """Test that random indices are generated correctly."""
    trainer = self._setup_get_batch_test()

    # Create a trainer with a fixed seed
    torch.manual_seed(42)
    x1, y1 = trainer.get_batch(use_val=False)

    # Create another trainer with a different seed
    torch.manual_seed(43)
    x2, y2 = trainer.get_batch(use_val=False)

    # Batches should be different due to different random indices
    self.assertFalse(torch.all(x1 == x2))

  @patch('numpy.memmap')
  def test_memmap_creation(self, mock_memmap):
    """Test that memmap is created with correct parameters."""
    # Need to patch setup_train_config to return a clean config
    test_config = TrainConfig()
    test_config.data_dir = "/test/path"

    with patch.object(GPTTrainer, 'setup_train_config', return_value=test_config):
      trainer = GPTTrainer(test_config, self.model_config)

      # Set up mock to return usable data
      mock_memmap.return_value = np.arange(100000, dtype=np.uint16)

      # Call get_batch
      trainer.get_batch(use_val=False)

      # Check memmap was created with correct parameters
      mock_memmap.assert_called_with(
          os.path.join(test_config.data_dir, 'train.bin'),
          dtype=np.uint16,
          mode='r'
      )

  @patch('torch.manual_seed')
  def test_configure_torch(self, mock_manual_seed):
    """Test configure_torch method correctly sets PyTorch settings"""
    # Set up trainer with patched setup_train_config
    test_config = TrainConfig()
    test_config.seed_offset = 42

    with patch.object(GPTTrainer, 'setup_train_config', return_value=test_config):
      trainer = GPTTrainer(test_config, self.model_config)

      # Verify random seed is set with the correct offset
      mock_manual_seed.assert_called_once_with(1337 + 42)

      # Verify TF32 settings are enabled
      self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
      self.assertTrue(torch.backends.cudnn.allow_tf32)

  @patch('torch.cuda.amp.GradScaler')
  @patch('train.GPT')
  @patch('os.path.exists')
  def test_setup_model_from_scratch(self, mock_exists, mock_gpt_class, mock_gradscaler):
    """Test setup_model when initializing model from scratch"""
    # Setup mocks
    mock_exists.return_value = False  # No meta.pkl exists.
    mock_model = MagicMock()
    # Return our mock when GPT is instantiated
    mock_gpt_class.return_value = mock_model
    # Mock GradScaler instance
    mock_gradscaler.return_value = MagicMock()

    # Configure test parameters
    test_config = TrainConfig()
    test_config.init_from = "scratch"
    test_config.device = DEVICE_CPU
    test_config.dtype = torch.float32

    # Set the sequence length attribute directly on the mock model
    # We'll set it to a different value than test_config.seq_length to trigger the assertion
    mock_model.config = MagicMock()
    mock_model.config.seq_length = model_config.ModelConfig.seq_length

    with patch.object(GPTTrainer, 'setup_train_config', return_value=test_config):
      trainer = GPTTrainer(test_config, self.model_config)

      # Call setup_model
      model = trainer.setup_model()

      # Verify GPT class was instantiated with the correct config
      mock_gpt_class.assert_called_once_with(self.model_config)

      # Verify model was correctly moved to device
      mock_model.to.assert_called_once_with(test_config.device)

      # Verify GradScaler was created with the correct parameters
      mock_gradscaler.assert_called_once_with(enabled=False)

      # Verify the created model is our mock.
      self.assertEqual(trainer.model, mock_model)


if __name__ == '__main__':
  unittest.main()
