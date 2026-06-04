"""
Unit tests for the BPE Tokenizer.
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.tokenizer import BPETokenizer


class TestBPETokenizer:
    """Test suite for BPETokenizer."""

    def setup_method(self):
        """Create a small tokenizer for testing."""
        self.tokenizer = BPETokenizer(vocab_size=300)

    def test_initialization(self):
        """Test tokenizer initializes with correct vocab size."""
        tok = BPETokenizer(vocab_size=32000)
        assert tok.vocab_size == 32000
        assert len(tok.SPECIAL_TOKENS) == 6
        assert tok.eos_token_id == 0
        assert tok.pad_token_id == 1
        assert tok.unk_token_id == 2

    def test_byte_level_encoding(self):
        """Test encoding without training (byte-level fallback)."""
        tok = BPETokenizer()
        text = "hi"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_encode_decode_roundtrip_ascii(self):
        """Test encode->decode preserves ASCII text."""
        tok = BPETokenizer()
        texts = [
            "Hello, world!",
            "The quick brown fox",
            "Testing 123",
            "Special chars: @#$%",
        ]
        for text in texts:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text, f"Roundtrip failed for: {text}"

    def test_encode_decode_roundtrip_unicode(self):
        """Test encode->decode preserves Unicode text."""
        tok = BPETokenizer()
        texts = [
            "café résumé",
            "日本語テスト",
            "emoji 🎉🚀",
            "über straße",
        ]
        for text in texts:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text, f"Unicode roundtrip failed for: {text}"

    def test_special_tokens(self):
        """Test special token encoding/decoding."""
        tok = BPETokenizer()
        text = "Hello<|endoftext|>World"

        # Without allowing special tokens
        ids_no_special = tok.encode(text)
        # The special token should be encoded as regular bytes
        assert tok.eos_token_id not in ids_no_special

        # With allowing special tokens
        ids_with_special = tok.encode(text, allowed_special={"<|endoftext|>"})
        assert tok.eos_token_id in ids_with_special

    def test_special_tokens_all(self):
        """Test allowing all special tokens."""
        tok = BPETokenizer()
        text = "<|system|>Hello<|endoftext|>"
        ids = tok.encode(text, allowed_special={"all"})
        assert 3 in ids  # system token
        assert 0 in ids  # endoftext token

    def test_training(self):
        """Test tokenizer training on a small corpus."""
        tok = BPETokenizer(vocab_size=300)
        corpus = "the cat sat on the mat. the cat sat on the hat. " * 100
        tok.train(corpus, verbose=False)

        # Should have learned some merges
        assert len(tok.merge_list) > 0
        assert len(tok.vocab) > 262  # 6 special + 256 bytes + merges

        # Encoding should produce fewer tokens than byte-level
        text = "the cat sat"
        ids = tok.encode(text)
        assert len(ids) < len(text.encode("utf-8"))

    def test_save_and_load(self):
        """Test saving and loading tokenizer."""
        tok = BPETokenizer(vocab_size=300)
        corpus = "hello world " * 100
        tok.train(corpus, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            tok.save(tmpdir, "test_tok")

            # Check files exist
            assert os.path.exists(os.path.join(tmpdir, "test_tok_vocab.json"))
            assert os.path.exists(os.path.join(tmpdir, "test_tok_merges.txt"))
            assert os.path.exists(os.path.join(tmpdir, "tokenizer.json"))

            # Load and verify
            loaded = BPETokenizer.load(tmpdir, "test_tok")
            assert len(loaded.vocab) == len(tok.vocab)
            assert len(loaded.merge_list) == len(tok.merge_list)

            # Verify encoding consistency
            test_text = "hello world"
            assert tok.encode(test_text) == loaded.encode(test_text)

    def test_empty_input(self):
        """Test handling of empty input."""
        tok = BPETokenizer()
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    def test_repr(self):
        """Test string representation."""
        tok = BPETokenizer(vocab_size=32000)
        repr_str = repr(tok)
        assert "BPETokenizer" in repr_str
        assert "vocab_size" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
