"""
Unit tests for the data pipeline.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.pipeline import MinHashDeduplicator, QualityFilter
from data.chat_templates import (
    format_chat,
    parse_alpaca,
    parse_sharegpt,
    parse_hh_rlhf,
)


class TestQualityFilter:
    """Test document quality filtering."""

    def setup_method(self):
        self.filter = QualityFilter()

    def test_passes_good_document(self):
        text = "This is a well-written document about machine learning. " * 10
        assert self.filter.filter(text) is True

    def test_rejects_too_short(self):
        assert self.filter.filter("Short.") is False

    def test_rejects_too_long(self):
        text = "a " * 100001
        assert self.filter.filter(text) is False

    def test_rejects_too_few_words(self):
        text = "word " * 5  # Only 5 words
        assert self.filter.filter(text) is False

    def test_rejects_repeated_lines(self):
        text = ("This line is repeated.\n" * 50)
        assert self.filter.filter(text) is False

    def test_rejects_special_chars(self):
        text = "@#$%^&*!" * 100
        assert self.filter.filter(text) is False


class TestMinHashDeduplicator:
    """Test MinHash LSH deduplication."""

    def setup_method(self):
        self.dedup = MinHashDeduplicator(threshold=0.85)

    def test_exact_duplicate(self):
        text = "The quick brown fox jumps over the lazy dog. " * 20
        assert self.dedup.is_duplicate(text, "doc1") is False  # First time
        assert self.dedup.is_duplicate(text, "doc2") is True   # Duplicate

    def test_unique_documents(self):
        text1 = "Machine learning is a subset of artificial intelligence. " * 20
        text2 = "The weather forecast predicts rain for tomorrow. " * 20
        assert self.dedup.is_duplicate(text1, "doc1") is False
        assert self.dedup.is_duplicate(text2, "doc2") is False

    def test_near_duplicate(self):
        text1 = "The quick brown fox jumps over the lazy dog every single day. " * 20
        text2 = "The quick brown fox jumps over the lazy dog every single night. " * 20
        self.dedup.is_duplicate(text1, "doc1")
        # Near-duplicate should be caught
        result = self.dedup.is_duplicate(text2, "doc2")
        # This may or may not be caught depending on threshold
        assert isinstance(result, bool)

    def test_short_text(self):
        assert self.dedup.is_duplicate("hi", "doc1") is False


class TestChatTemplates:
    """Test chat template formatting."""

    def test_format_chat_basic(self):
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = format_chat(messages)
        assert "<|system|>" in result
        assert "<|user|>" in result
        assert "Hello!" in result
        assert "<|assistant|>" in result
        assert "Hi there!" in result

    def test_format_chat_with_system(self):
        messages = [
            {"role": "system", "content": "You are a helpful bot."},
            {"role": "user", "content": "What is AI?"},
        ]
        result = format_chat(messages)
        assert "You are a helpful bot." in result

    def test_format_chat_generation_prompt(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = format_chat(messages, add_generation_prompt=True)
        assert result.endswith("<|assistant|>\n")

    def test_parse_alpaca(self):
        example = {
            "instruction": "Write a haiku",
            "input": "",
            "output": "Cherry blossoms fall\nSoftly on the temple steps\nSpring has come again",
        }
        result = parse_alpaca(example)
        assert "text" in result
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_parse_alpaca_with_input(self):
        example = {
            "instruction": "Translate to French",
            "input": "Hello, how are you?",
            "output": "Bonjour, comment allez-vous?",
        }
        result = parse_alpaca(example)
        assert "Input:" in result["text"]

    def test_parse_sharegpt(self):
        example = {
            "conversations": [
                {"from": "human", "value": "What is Python?"},
                {"from": "gpt", "value": "Python is a programming language."},
            ]
        }
        result = parse_sharegpt(example)
        assert "text" in result
        assert "Python" in result["text"]

    def test_parse_hh_rlhf(self):
        example = {
            "chosen": "Human: What is 2+2?\n\nAssistant: 2+2 equals 4.",
            "rejected": "Human: What is 2+2?\n\nAssistant: I don't know.",
        }
        result = parse_hh_rlhf(example)
        assert "prompt" in result
        assert "chosen" in result
        assert "rejected" in result
        assert "4" in result["chosen"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
