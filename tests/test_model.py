"""
Unit tests for the OpenMind Transformer model.
"""

import os
import sys
import tempfile
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.config_openmind import OpenMindConfig
from models.modeling_openmind import (
    RMSNorm,
    RotaryEmbedding,
    GroupedQueryAttention,
    SwiGLU,
    TransformerBlock,
    OpenMindModel,
)


# Use a small config for fast testing
TEST_CONFIG = OpenMindConfig(
    vocab_size=1000,
    max_seq_len=128,
    dim=64,
    n_layers=2,
    n_heads=4,
    n_kv_heads=4,
    intermediate_dim=128,
    dropout=0.0,
    tie_embeddings=True,
)


class TestRMSNorm:
    """Test RMSNorm layer."""

    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        # RMS should be approximately 1 after normalization
        rms = out.pow(2).mean(-1).sqrt()
        assert rms.mean().item() < 5.0  # Reasonable range


class TestRotaryEmbedding:
    """Test Rotary Positional Embeddings."""

    def test_output_shapes(self):
        rope = RotaryEmbedding(64, max_seq_len=128)
        x = torch.randn(2, 4, 10, 64)
        cos, sin = rope(x, seq_len=10)
        assert cos.shape == (10, 64)
        assert sin.shape == (10, 64)

    def test_dynamic_extension(self):
        rope = RotaryEmbedding(64, max_seq_len=32)
        x = torch.randn(2, 4, 64, 64)
        # Should auto-extend cache
        cos, sin = rope(x, seq_len=64)
        assert cos.shape == (64, 64)


class TestGroupedQueryAttention:
    """Test GQA module."""

    def test_mha_output_shape(self):
        config = OpenMindConfig(dim=64, n_heads=4, n_kv_heads=4, max_seq_len=128,
                                vocab_size=100, n_layers=1, intermediate_dim=128)
        attn = GroupedQueryAttention(config)
        x = torch.randn(2, 10, 64)
        out, cache = attn(x)
        assert out.shape == (2, 10, 64)
        assert cache is None

    def test_gqa_output_shape(self):
        config = OpenMindConfig(dim=64, n_heads=4, n_kv_heads=2, max_seq_len=128,
                                vocab_size=100, n_layers=1, intermediate_dim=128)
        attn = GroupedQueryAttention(config)
        x = torch.randn(2, 10, 64)
        out, cache = attn(x)
        assert out.shape == (2, 10, 64)

    def test_kv_cache(self):
        config = OpenMindConfig(dim=64, n_heads=4, n_kv_heads=4, max_seq_len=128,
                                vocab_size=100, n_layers=1, intermediate_dim=128)
        attn = GroupedQueryAttention(config)

        # First pass
        x = torch.randn(1, 5, 64)
        out1, cache = attn(x, use_cache=True)
        assert cache is not None
        assert cache[0].shape[2] == 5  # K cache seq_len

        # Second pass with cache
        x2 = torch.randn(1, 1, 64)
        out2, cache2 = attn(x2, past_key_value=cache, use_cache=True)
        assert out2.shape == (1, 1, 64)
        assert cache2[0].shape[2] == 6  # Extended cache


class TestSwiGLU:
    """Test SwiGLU feed-forward network."""

    def test_output_shape(self):
        config = OpenMindConfig(dim=64, intermediate_dim=128, vocab_size=100,
                                n_heads=4, n_kv_heads=4, n_layers=1, max_seq_len=128)
        ffn = SwiGLU(config)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)


class TestTransformerBlock:
    """Test full transformer block."""

    def test_output_shape(self):
        block = TransformerBlock(TEST_CONFIG)
        x = torch.randn(2, 10, 64)
        out, cache = block(x)
        assert out.shape == (2, 10, 64)

    def test_residual_connection(self):
        block = TransformerBlock(TEST_CONFIG)
        x = torch.randn(2, 10, 64)
        out, _ = block(x)
        # Output should be different from input (not zero residual)
        assert not torch.allclose(out, x, atol=1e-6)


class TestOpenMindModel:
    """Test the full OpenMind model."""

    def test_forward_pass(self):
        model = OpenMindModel(TEST_CONFIG)
        input_ids = torch.randint(0, 1000, (2, 10))
        outputs = model(input_ids)
        assert outputs["logits"].shape == (2, 10, 1000)
        assert outputs["loss"] is None

    def test_forward_with_labels(self):
        model = OpenMindModel(TEST_CONFIG)
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))
        outputs = model(input_ids, labels=labels)
        assert outputs["loss"] is not None
        assert outputs["loss"].dim() == 0  # Scalar

    def test_gradient_flow(self):
        model = OpenMindModel(TEST_CONFIG)
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))

        outputs = model(input_ids, labels=labels)
        outputs["loss"].backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_generate(self):
        model = OpenMindModel(TEST_CONFIG)
        input_ids = torch.randint(0, 1000, (1, 5))
        generated = model.generate(input_ids, max_new_tokens=10)
        assert generated.shape[1] > 5
        assert generated.shape[1] <= 15

    def test_kv_cache_generation(self):
        model = OpenMindModel(TEST_CONFIG)
        input_ids = torch.randint(0, 1000, (1, 3))
        outputs = model(input_ids, use_cache=True)
        assert outputs["past_key_values"] is not None
        assert len(outputs["past_key_values"]) == TEST_CONFIG.n_layers

    def test_count_parameters(self):
        model = OpenMindModel(TEST_CONFIG)
        counts = model.count_parameters()
        assert "total" in counts
        assert counts["total"] > 0
        assert "embedding" in counts
        assert "attention" in counts

    def test_save_and_load(self):
        model = OpenMindModel(TEST_CONFIG)
        input_ids = torch.randint(0, 1000, (1, 5))

        with torch.no_grad():
            original_output = model(input_ids)["logits"]

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = OpenMindModel.from_pretrained(tmpdir)

            with torch.no_grad():
                loaded_output = loaded(input_ids)["logits"]

            assert torch.allclose(original_output, loaded_output, atol=1e-5)


class TestOpenMindConfig:
    """Test model configuration."""

    def test_default_config(self):
        config = OpenMindConfig()
        assert config.dim == 768
        assert config.n_heads == 12
        assert config.head_dim == 64

    def test_head_dim_computation(self):
        config = OpenMindConfig(dim=128, n_heads=8, n_kv_heads=8, n_layers=1,
                                vocab_size=100, intermediate_dim=256, max_seq_len=128)
        assert config.head_dim == 16

    def test_invalid_config(self):
        with pytest.raises(AssertionError):
            OpenMindConfig(dim=100, n_heads=3, n_kv_heads=3, n_layers=1,
                           vocab_size=100, intermediate_dim=200, max_seq_len=128)

    def test_save_and_load_config(self):
        config = OpenMindConfig(dim=256, n_heads=8, n_kv_heads=4, n_layers=6,
                                vocab_size=5000, intermediate_dim=512, max_seq_len=512)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = OpenMindConfig.from_pretrained(tmpdir)
            assert loaded.dim == 256
            assert loaded.n_heads == 8
            assert loaded.n_kv_heads == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
