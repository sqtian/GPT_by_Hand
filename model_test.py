import unittest
import torch
from model_config import ModelConfig
from model import *


class TestLayerNorm(unittest.TestCase):

  def test_layernorm_output_shape(self):
    """The shapes of output and input tensors are the same."""
    input_tensor = torch.randn(10, 20)
    layernorm = LayerNorm(20)
    output_tensor = layernorm(input_tensor)
    self.assertEqual(output_tensor.shape, input_tensor.shape)

  def test_layernorm_mean_and_var(self):
    """After normalization, the mean of output tensor should be close to 0, the var should be close to 1."""
    input_tensor = torch.randn(10, 20)
    layernorm = LayerNorm(20)
    output_tensor = layernorm(input_tensor)
    mean = output_tensor.mean(-1)
    var = output_tensor.var(-1, unbiased=False)
    self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-2))
    self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-2))

  def test_layernorm_init_parameters(self):
    """Test the initialization of gamma, beta, and eps."""
    ndim = 20
    layernorm = LayerNorm(ndim)
    self.assertTrue(torch.equal(layernorm.gamma, torch.ones(ndim)))
    self.assertTrue(torch.equal(layernorm.beta, torch.zeros(ndim)))
    self.assertEqual(layernorm.eps, 1e-5)

  def test_layernorm_init_parameters_no_bias(self):
    """Test the initialization of gamma, beta (None when bias=False), and eps."""
    ndim = 20
    layernorm = LayerNorm(ndim, bias=False)
    self.assertTrue(torch.equal(layernorm.gamma, torch.ones(ndim)))
    self.assertIsNone(layernorm.beta)
    self.assertEqual(layernorm.eps, 1e-5)


class TestAttention(unittest.TestCase):

  def setUp(self):
    config = ModelConfig(
      dim_embedding=32,
      num_heads=4,
      dropout_rate=0.1,
      use_bias=True,
      seq_length=128
    )
    self.attention = Attention(config)

  def test_attention_output_shape(self):
    """The shapes of output and input tensors are the same."""
    input_tensor = torch.randn(2, 10, 32)
    output_tensor = self.attention(input_tensor)
    self.assertEqual(output_tensor.shape, input_tensor.shape)

  def test_attention_no_nan(self):
    """The output tensor should not contain NaN values."""
    input_tensor = torch.randn(2, 10, 32)
    output_tensor = self.attention(input_tensor)
    self.assertFalse(torch.isnan(output_tensor).any())


class TestMLP(unittest.TestCase):

  def setUp(self):
    config = ModelConfig(
      dim_embedding=32,
      dropout_rate=0.1,
      use_bias=True
    )
    self.mlp = FFN(config)

  def test_mlp_output_shape(self):
    """The shapes of output and input tensors are the same."""
    input_tensor = torch.randn(2, 10, 32)
    output_tensor = self.mlp(input_tensor)
    self.assertEqual(output_tensor.shape, input_tensor.shape)

  def test_mlp_no_nan(self):
    """The output tensor should not contain NaN values."""
    input_tensor = torch.randn(2, 10, 32)
    output_tensor = self.mlp(input_tensor)
    self.assertFalse(torch.isnan(output_tensor).any())

  def test_mlp_forward_pass(self):
    """Test a forward pass through the FFN."""
    input_tensor = torch.randn(2, 10, 32)
    output_tensor = self.mlp(input_tensor)
    self.assertIsNotNone(output_tensor)


class TestBlock(unittest.TestCase):

  def setUp(self):
    config = ModelConfig(
      dim_embedding=32,
      num_heads=4,
      dropout_rate=0.1,
      use_bias=True,
      seq_length=128
    )
    self.encoder = TransformerBlock(config)

  def test_encoder_output_shape(self):
    """The shapes of output and input tensors are the same."""
    input_tensor = torch.randn(2, 10, 32)
    output_tensor = self.encoder(input_tensor)
    self.assertEqual(output_tensor.shape, input_tensor.shape)

  def test_encoder_no_nan(self):
    """The output tensor should not contain NaN values."""
    input_tensor = torch.randn(2, 10, 32)
    output_tensor = self.encoder(input_tensor)
    self.assertFalse(torch.isnan(output_tensor).any())

  def test_encoder_forward_pass(self):
    """Test a forward pass through the TransformerBlock."""
    input_tensor = torch.randn(2, 10, 32)
    output_tensor = self.encoder(input_tensor)
    self.assertIsNotNone(output_tensor)


class TestGPT(unittest.TestCase):

  def setUp(self):
    """Set up a GPT model instance for testing."""
    config = ModelConfig(
      num_layers=2,
      num_heads=4,
      dim_embedding=32,
      dropout_rate=0.1,
      use_bias=True,
      vocab_size=100,
      seq_length=128
    )
    self.model = GPT(config)

  def test_gpt_forward_pass(self):
    """Test a basic forward pass through the GPT model."""
    batch_size = 2
    seq_len = 10
    idx = torch.randint(0, 100, (batch_size, seq_len))  # Random token indices

    # Get logits and loss with targets = None
    logits, _ = self.model(idx)

    # Check output shape: [batch_size, 1, vocab_size]
    # Inference-time mini-optimization: only forward the lm_head on the very last position
    # note: using list [-1] to preserve the time dim
    self.assertEqual(logits.shape, (batch_size, 1, 100))

    # No NaN values
    self.assertFalse(torch.isnan(logits).any())

  def test_gpt_loss_calculation(self):
    """Test loss calculation when targets are provided."""
    batch_size = 2
    seq_len = 10
    idx = torch.randint(0, 100, (batch_size, seq_len))  # Random token indices
    # Random target indices
    targets = torch.randint(0, 100, (batch_size, seq_len))

    # Get logits and loss with targets
    _, loss = self.model(idx, targets)

    # Loss should be a scalar
    self.assertEqual(loss.dim(), 0)

    # Loss should not be NaN
    self.assertFalse(torch.isnan(loss).any())

  def test_configure_optimizers_basic(self):
    """Test if configure_optimizers creates an optimizer object."""
    optimizer = self.model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-4,
        betas=(0.9, 0.999),
        device_type='cpu'
    )

    # Check if we get back an AdamW optimizer
    self.assertIsInstance(optimizer, torch.optim.AdamW)

    # Check if the learning rate is set correctly
    self.assertEqual(optimizer.defaults['lr'], 1e-4)

    # Check if betas are set correctly
    self.assertEqual(optimizer.defaults['betas'], (0.9, 0.999))

  def test_configure_optimizers_parameter_groups(self):
    """Test that configure_optimizers correctly separates parameters into decay and no-decay groups."""
    optimizer = self.model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-4,
        betas=(0.9, 0.95),
        device_type='cpu'
    )

    # Should have exactly 2 parameter groups (decay and no-decay)
    self.assertEqual(len(optimizer.param_groups), 2)

    # One group should have weight decay of 0.1
    decay_group = [
      g for g in optimizer.param_groups if g['weight_decay'] > 0][0]
    self.assertEqual(decay_group['weight_decay'], 0.1)

    # One group should have weight decay of 0.0
    no_decay_group = [
      g for g in optimizer.param_groups if g['weight_decay'] == 0][0]
    self.assertEqual(no_decay_group['weight_decay'], 0.0)

    # Instead of trying to check specific parameters, let's verify the logic more generally
    # Get parameter names in each group
    decay_param_names = set()
    no_decay_param_names = set()

    # Map parameters to their names for the decay group
    for param in decay_group['params']:
      for name, model_param in self.model.named_parameters():
        if param is model_param:
          decay_param_names.add(name)

    # Map parameters to their names for the no-decay group
    for param in no_decay_group['params']:
      for name, model_param in self.model.named_parameters():
        if param is model_param:
          no_decay_param_names.add(name)

    # Check that parameters are correctly categorized
    for name in decay_param_names:
      # Parameters in decay group should not contain bias, LayerNorm weight, or be 1D
      self.assertFalse(
        'bias' in name or 'ln' in name or '.gamma' in name or '.beta' in name)

    for name in no_decay_param_names:
      # Parameters in no-decay group should contain bias, LayerNorm weight, or be 1D
      self.assertTrue(
        'bias' in name or 'ln' in name or '.gamma' in name or '.beta' in name)

    # Verify all parameters are accounted for
    all_param_names = set(
      name for name, _ in self.model.named_parameters() if _.requires_grad)
    self.assertEqual(
      all_param_names, decay_param_names.union(no_decay_param_names))


if __name__ == "__main__":
  unittest.main()
