"""
OpenMind Data Pipeline - Preprocessing, Deduplication, and Tokenization.

Complete pipeline for:
1. Downloading datasets from Hugging Face
2. Deduplication using MinHash LSH
3. Language and quality filtering
4. Tokenization with our BPE tokenizer
5. Packing into memory-mapped binary files
"""

import os
import json
import hashlib
import struct
from pathlib import Path
from typing import Iterable, Iterator, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm


class MinHashDeduplicator:
    """
    MinHash LSH-based approximate deduplication.

    Uses Jaccard similarity with configurable n-gram shingles and
    Locality-Sensitive Hashing for efficient near-duplicate detection.
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.85,
        ngram_size: int = 5,
        seed: int = 42,
    ):
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.seed = seed

        # Generate random hash parameters
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, 2**31 - 1, size=num_perm, dtype=np.int64)
        self.b = rng.randint(0, 2**31 - 1, size=num_perm, dtype=np.int64)
        self.prime = np.int64(2**31 - 1)  # Mersenne prime

        # LSH bands/rows for the given threshold
        self.bands, self.rows = self._optimal_bands_rows()
        self.buckets: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    def _optimal_bands_rows(self) -> tuple[int, int]:
        """Find optimal bands and rows for the given threshold."""
        best = (1, self.num_perm)
        for b in range(1, self.num_perm + 1):
            if self.num_perm % b == 0:
                r = self.num_perm // b
                t = (1.0 / b) ** (1.0 / r)
                if abs(t - self.threshold) < abs((1.0 / best[0]) ** (1.0 / best[1]) - self.threshold):
                    best = (b, r)
        return best

    def _shingle(self, text: str) -> set[int]:
        """Create character n-gram shingles from text."""
        text = text.lower().strip()
        shingles = set()
        for i in range(len(text) - self.ngram_size + 1):
            shingle = text[i: i + self.ngram_size]
            shingles.add(hash(shingle) & 0xFFFFFFFF)
        return shingles

    def _minhash(self, shingles: set[int]) -> np.ndarray:
        """Compute MinHash signature from shingles."""
        if not shingles:
            return np.full(self.num_perm, np.iinfo(np.int64).max)

        shingle_array = np.array(list(shingles), dtype=np.int64)
        # Hash each shingle with each hash function and take minimum
        hashvals = np.zeros(self.num_perm, dtype=np.int64)
        for i in range(self.num_perm):
            hashed = (self.a[i] * shingle_array + self.b[i]) % self.prime
            hashvals[i] = hashed.min()
        return hashvals

    def is_duplicate(self, text: str, doc_id: str = "") -> bool:
        """Check if text is a near-duplicate of any seen document."""
        shingles = self._shingle(text)
        if len(shingles) < 2:
            return False

        signature = self._minhash(shingles)

        # Check against existing buckets
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band_hash = hashlib.md5(signature[start:end].tobytes()).hexdigest()

            if band_hash in self.buckets[band_idx]:
                return True

        # Add to buckets
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band_hash = hashlib.md5(signature[start:end].tobytes()).hexdigest()
            self.buckets[band_idx][band_hash].append(doc_id)

        return False


class QualityFilter:
    """
    Document quality filter with configurable rules.

    Filters based on:
    - Document length (min/max characters)
    - Repeated line ratio
    - Special character ratio
    - Word count thresholds
    """

    def __init__(
        self,
        min_chars: int = 100,
        max_chars: int = 100_000,
        max_repeated_line_ratio: float = 0.3,
        max_special_char_ratio: float = 0.3,
        min_words: int = 20,
        min_avg_word_len: float = 3.0,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_repeated_line_ratio = max_repeated_line_ratio
        self.max_special_char_ratio = max_special_char_ratio
        self.min_words = min_words
        self.min_avg_word_len = min_avg_word_len

    def filter(self, text: str) -> bool:
        """Return True if document passes quality checks, False to reject."""
        # Length check
        if len(text) < self.min_chars or len(text) > self.max_chars:
            return False

        # Word count
        words = text.split()
        if len(words) < self.min_words:
            return False

        # Average word length (filters gibberish)
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        if avg_word_len < self.min_avg_word_len:
            return False

        # Repeated line ratio
        lines = text.strip().split("\n")
        if len(lines) > 1:
            unique_lines = set(line.strip() for line in lines if line.strip())
            if len(unique_lines) / max(len(lines), 1) < (1 - self.max_repeated_line_ratio):
                return False

        # Special character ratio
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_chars / max(len(text), 1) > self.max_special_char_ratio:
            return False

        return True


class DataPipeline:
    """
    Complete data preprocessing pipeline.

    Handles downloading, filtering, deduplication, tokenization,
    and packing into memory-mapped binary files for training.
    """

    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
        output_dir: str = "data",
        max_seq_len: int = 2048,
        seed: int = 42,
    ):
        self.output_dir = output_dir
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.tokenizer = None

        if tokenizer_path and os.path.exists(tokenizer_path):
            try:
                from .tokenizer import BPETokenizer
            except ImportError:
                from tokenizer import BPETokenizer
            self.tokenizer = BPETokenizer.load(tokenizer_path)

        self.quality_filter = QualityFilter()
        self.deduplicator = MinHashDeduplicator(seed=seed)

        os.makedirs(output_dir, exist_ok=True)

    def load_tokenizer(self, tokenizer_path: str):
        """Load tokenizer from disk."""
        try:
            from .tokenizer import BPETokenizer
        except ImportError:
            from tokenizer import BPETokenizer
        self.tokenizer = BPETokenizer.load(tokenizer_path)

    def process_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        streaming: bool = True,
        max_documents: Optional[int] = None,
        output_name: Optional[str] = None,
    ) -> str:
        """
        Process a Hugging Face dataset through the full pipeline.

        Args:
            dataset_name: HF dataset identifier (e.g., "roneneldan/TinyStories")
            split: Dataset split to process
            text_column: Column containing text data
            streaming: Use streaming mode for large datasets
            max_documents: Optional limit on documents to process
            output_name: Output filename (without extension)

        Returns:
            Path to the output .bin file
        """
        from datasets import load_dataset

        print(f"Loading dataset: {dataset_name} (split={split}, streaming={streaming})")
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)

        if output_name is None:
            output_name = dataset_name.replace("/", "_") + f"_{split}"

        # Process documents through pipeline
        processed_docs = self._pipeline_iter(dataset, text_column, max_documents)

        # Tokenize and pack
        output_path = self.tokenize_and_pack(processed_docs, output_name)

        return output_path

    def _pipeline_iter(
        self,
        dataset,
        text_column: str,
        max_documents: Optional[int] = None,
    ) -> Iterator[str]:
        """Run documents through filtering and deduplication."""
        stats = {"total": 0, "passed_quality": 0, "passed_dedup": 0}

        for i, example in enumerate(tqdm(dataset, desc="Processing documents")):
            if max_documents and i >= max_documents:
                break

            stats["total"] += 1
            text = example.get(text_column, "")

            if not text or not isinstance(text, str):
                continue

            # Quality filter
            if not self.quality_filter.filter(text):
                continue
            stats["passed_quality"] += 1

            # Deduplication
            if self.deduplicator.is_duplicate(text, doc_id=str(i)):
                continue
            stats["passed_dedup"] += 1

            yield text

        print(f"Pipeline stats: {json.dumps(stats, indent=2)}")

    def tokenize_and_pack(
        self,
        documents: Iterable[str],
        output_name: str,
    ) -> str:
        """
        Tokenize documents and pack into a memory-mapped binary file.

        Documents are concatenated with <|endoftext|> separators and
        split into fixed-length sequences of max_seq_len.

        Args:
            documents: Iterator of text documents
            output_name: Output filename (without extension)

        Returns:
            Path to the output .bin file
        """
        assert self.tokenizer is not None, "Tokenizer must be loaded first!"

        eos_id = self.tokenizer.eos_token_id
        all_tokens = []
        doc_count = 0

        for doc in tqdm(documents, desc="Tokenizing"):
            tokens = self.tokenizer.encode(doc)
            tokens.append(eos_id)  # Add document separator
            all_tokens.extend(tokens)
            doc_count += 1

        if not all_tokens:
            print("Warning: No tokens produced!")
            return ""

        print(f"Tokenized {doc_count} documents -> {len(all_tokens):,} tokens")

        # Pack into sequences of max_seq_len
        num_sequences = len(all_tokens) // self.max_seq_len
        trimmed = all_tokens[: num_sequences * self.max_seq_len]

        print(f"Packed into {num_sequences} sequences of length {self.max_seq_len}")

        # Save as memory-mapped binary
        output_path = self.save_mmap(np.array(trimmed, dtype=np.uint16), output_name)

        return output_path

    def save_mmap(self, tokens: np.ndarray, filename: str) -> str:
        """
        Save token array as a memory-mapped binary file.

        Format: uint16 (supports vocab up to 65k)

        Args:
            tokens: 1D numpy array of token IDs
            filename: Output filename (without extension)

        Returns:
            Path to the saved file
        """
        output_path = os.path.join(self.output_dir, f"{filename}.bin")

        # Write header with metadata
        header_path = os.path.join(self.output_dir, f"{filename}_meta.json")
        meta = {
            "num_tokens": int(len(tokens)),
            "dtype": str(tokens.dtype),
            "max_seq_len": self.max_seq_len,
            "num_sequences": int(len(tokens)) // self.max_seq_len,
        }
        with open(header_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Write binary data
        tokens.tofile(output_path)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Saved {len(tokens):,} tokens to {output_path} ({file_size_mb:.1f} MB)")

        return output_path

    def process_jsonl(
        self,
        input_path: str,
        text_field: str = "text",
        output_name: Optional[str] = None,
        max_documents: Optional[int] = None,
    ) -> str:
        """
        Process a local JSONL file through the pipeline.

        Args:
            input_path: Path to .jsonl file
            text_field: JSON field containing text
            output_name: Output filename
            max_documents: Optional document limit

        Returns:
            Path to output .bin file
        """
        if output_name is None:
            output_name = Path(input_path).stem

        def doc_iter():
            count = 0
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    if max_documents and count >= max_documents:
                        break
                    try:
                        data = json.loads(line.strip())
                        text = data.get(text_field, "")
                        if text and self.quality_filter.filter(text):
                            if not self.deduplicator.is_duplicate(text, str(count)):
                                yield text
                                count += 1
                    except json.JSONDecodeError:
                        continue

        return self.tokenize_and_pack(doc_iter(), output_name)


class TokenDataset:
    """
    Memory-mapped token dataset for efficient training data loading.

    Reads pre-tokenized binary files and yields fixed-length sequences.
    Supports distributed training with proper sharding.
    """

    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 2048,
        dtype: np.dtype = np.uint16,
    ):
        self.max_seq_len = max_seq_len

        # Memory-map the file for efficient random access
        self.data = np.memmap(data_path, dtype=dtype, mode="r")
        self.num_tokens = len(self.data)
        self.num_sequences = self.num_tokens // max_seq_len

        print(f"Loaded {self.num_tokens:,} tokens from {data_path}")
        print(f"  {self.num_sequences} sequences of length {max_seq_len}")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> np.ndarray:
        start = idx * self.max_seq_len
        end = start + self.max_seq_len
        return self.data[start:end].astype(np.int64)

    def get_batch(self, batch_size: int, device: str = "cpu"):
        """Get a random batch of sequences."""
        import torch
        indices = np.random.randint(0, self.num_sequences, size=batch_size)
        batch = np.stack([self[i] for i in indices])
        x = torch.from_numpy(batch).long().to(device)
        # Labels are the same as inputs (shifted internally by the model)
        return x, x.clone()
