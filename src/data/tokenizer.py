"""
OpenMind BPE Tokenizer - Built from scratch.

A complete Byte-Pair Encoding tokenizer implementation with:
- GPT-2 style pre-tokenization
- Unicode/UTF-8 support
- Special tokens handling
- Save/Load to JSON format (Hugging Face compatible)
"""

import json
import os
import regex
from collections import Counter, OrderedDict
from typing import Optional


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer trained from scratch.

    Supports:
    - Training on arbitrary text corpora
    - GPT-2 style regex pre-tokenization
    - Special tokens: <|endoftext|>, <|padding|>, <|unk|>
    - Encode/decode with full Unicode roundtrip
    - Save/load vocabulary and merges to disk
    """

    # GPT-2 style pre-tokenization regex
    PAT = regex.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    SPECIAL_TOKENS = OrderedDict([
        ("<|endoftext|>", 0),
        ("<|padding|>", 1),
        ("<|unk|>", 2),
        ("<|system|>", 3),
        ("<|user|>", 4),
        ("<|assistant|>", 5),
    ])

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size

        # Initialize with byte-level tokens (256 bytes) + special tokens
        self.num_special = len(self.SPECIAL_TOKENS)
        self.num_base = 256  # One token per byte value

        # Vocab: special tokens (0..5) + byte tokens (6..261) + merges (262..)
        self.vocab: dict[int, bytes] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self.merge_list: list[tuple[int, int]] = []

        # Inverse mappings
        self.token_to_id: dict[bytes, int] = {}
        self.special_token_ids: dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.id_to_special: dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}

        self._build_base_vocab()

    def _build_base_vocab(self):
        """Initialize vocab with special tokens and 256 byte-level tokens."""
        # Add special tokens
        for token_str, token_id in self.SPECIAL_TOKENS.items():
            self.vocab[token_id] = token_str.encode("utf-8")

        # Add byte-level tokens (0x00 to 0xFF)
        for i in range(256):
            token_id = self.num_special + i
            byte_val = bytes([i])
            self.vocab[token_id] = byte_val
            self.token_to_id[byte_val] = token_id

    def _get_pair_counts(self, token_sequences: list[list[int]]) -> Counter:
        """Count frequency of adjacent token pairs across all sequences."""
        counts = Counter()
        for seq in token_sequences:
            for i in range(len(seq) - 1):
                counts[(seq[i], seq[i + 1])] += 1
        return counts

    def _merge_pair(
        self, token_sequences: list[list[int]], pair: tuple[int, int], new_id: int
    ) -> list[list[int]]:
        """Replace all occurrences of pair with new_id in all sequences."""
        result = []
        for seq in token_sequences:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            result.append(new_seq)
        return result

    def train(self, corpus: str, vocab_size: Optional[int] = None, verbose: bool = True) -> None:
        """
        Train the BPE tokenizer on a text corpus.

        Args:
            corpus: Training text
            vocab_size: Target vocabulary size (default: self.vocab_size)
            verbose: Print progress during training
        """
        if vocab_size is not None:
            self.vocab_size = vocab_size

        num_merges = self.vocab_size - self.num_special - self.num_base

        if verbose:
            print(f"Training BPE tokenizer with target vocab size {self.vocab_size}")
            print(f"  Special tokens: {self.num_special}")
            print(f"  Byte tokens: {self.num_base}")
            print(f"  Merges to learn: {num_merges}")

        # Pre-tokenize: split corpus into words using GPT-2 regex
        words = regex.findall(self.PAT, corpus)

        # Convert each word to a sequence of byte-level token IDs
        word_freqs: dict[tuple[int, ...], int] = Counter()
        for word in words:
            byte_ids = tuple(self.num_special + b for b in word.encode("utf-8"))
            word_freqs[byte_ids] += 1

        # Expand into weighted token sequences
        token_sequences = []
        weights = []
        for seq, count in word_freqs.items():
            token_sequences.append(list(seq))
            weights.append(count)

        # Iteratively find and merge the most frequent pair
        for merge_idx in range(num_merges):
            # Count pairs (weighted by word frequency)
            pair_counts = Counter()
            for seq, w in zip(token_sequences, weights):
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += w

            if not pair_counts:
                if verbose:
                    print(f"No more pairs to merge at step {merge_idx}")
                break

            # Find most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            new_id = self.num_special + self.num_base + merge_idx

            # Record the merge
            self.merges[best_pair] = new_id
            self.merge_list.append(best_pair)

            # Create the new token (concatenation of the two tokens' bytes)
            new_token = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = new_token
            self.token_to_id[new_token] = new_id

            # Apply merge to all sequences
            token_sequences = self._merge_pair(token_sequences, best_pair, new_id)

            if verbose and (merge_idx + 1) % 1000 == 0:
                pair_str = (
                    self.vocab[best_pair[0]].decode("utf-8", errors="replace")
                    + " + "
                    + self.vocab[best_pair[1]].decode("utf-8", errors="replace")
                )
                print(
                    f"  Merge {merge_idx + 1}/{num_merges}: "
                    f"{pair_str} -> id {new_id} "
                    f"(freq={pair_counts[best_pair]})"
                )

        if verbose:
            print(f"Training complete. Final vocab size: {len(self.vocab)}")

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        """Encode a chunk of bytes into token IDs using learned merges."""
        # Start with byte-level tokens
        ids = [self.num_special + b for b in text_bytes]

        # Apply merges in order of learning
        for pair, new_id in self.merges.items():
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

        return ids

    def encode(
        self,
        text: str,
        allowed_special: Optional[set[str]] = None,
    ) -> list[int]:
        """
        Encode text into a list of token IDs.

        Args:
            text: Input text to encode
            allowed_special: Set of special token strings to recognize.
                             If None, no special tokens are processed.
                             Use {"all"} to allow all special tokens.

        Returns:
            List of integer token IDs
        """
        if allowed_special is None:
            allowed_special = set()

        if "all" in allowed_special:
            allowed_special = set(self.SPECIAL_TOKENS.keys())

        # Handle special tokens by splitting on them
        if allowed_special:
            # Build regex pattern for special tokens
            special_pattern = "(" + "|".join(
                regex.escape(s) for s in sorted(allowed_special, key=len, reverse=True)
            ) + ")"
            parts = regex.split(special_pattern, text)
        else:
            parts = [text]

        ids = []
        for part in parts:
            if part in self.special_token_ids and part in allowed_special:
                ids.append(self.special_token_ids[part])
            elif part:
                # Pre-tokenize with GPT-2 regex
                chunks = regex.findall(self.PAT, part)
                for chunk in chunks:
                    chunk_bytes = chunk.encode("utf-8")
                    ids.extend(self._encode_chunk(chunk_bytes))

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back into text.

        Args:
            ids: List of integer token IDs

        Returns:
            Decoded text string
        """
        byte_parts = []
        for token_id in ids:
            if token_id in self.id_to_special:
                byte_parts.append(self.id_to_special[token_id].encode("utf-8"))
            elif token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
            else:
                # Unknown token
                byte_parts.append(self.id_to_special.get(2, "<|unk|>").encode("utf-8"))

        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def save(self, directory: str, name: str = "tokenizer") -> None:
        """
        Save tokenizer vocab and merges to disk.

        Creates two files:
        - {name}_vocab.json: Token ID -> token string mapping
        - {name}_merges.txt: Merge rules in order

        Args:
            directory: Output directory
            name: File name prefix
        """
        os.makedirs(directory, exist_ok=True)

        # Save vocab as JSON
        vocab_data = {
            "vocab_size": self.vocab_size,
            "num_special": self.num_special,
            "num_base": self.num_base,
            "special_tokens": dict(self.SPECIAL_TOKENS),
            "vocab": {},
        }

        for token_id, token_bytes in self.vocab.items():
            # Store bytes as list of ints for JSON serialization
            vocab_data["vocab"][str(token_id)] = list(token_bytes)

        vocab_path = os.path.join(directory, f"{name}_vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2)

        # Save merges
        merges_path = os.path.join(directory, f"{name}_merges.txt")
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write(f"# OpenMind BPE Merges - {len(self.merge_list)} merges\n")
            for pair in self.merge_list:
                f.write(f"{pair[0]} {pair[1]}\n")

        # Save Hugging Face compatible tokenizer.json
        hf_vocab = {}
        for token_id, token_bytes in self.vocab.items():
            try:
                token_str = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                token_str = "".join(f"<0x{b:02X}>" for b in token_bytes)
            hf_vocab[token_str] = token_id

        hf_data = {
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": hf_vocab,
                "merges": [
                    f"{p[0]} {p[1]}" for p in self.merge_list
                ],
            },
            "added_tokens": [
                {"id": v, "content": k, "single_word": False, "lstrip": False,
                 "rstrip": False, "normalized": False, "special": True}
                for k, v in self.SPECIAL_TOKENS.items()
            ],
        }

        hf_path = os.path.join(directory, "tokenizer.json")
        with open(hf_path, "w", encoding="utf-8") as f:
            json.dump(hf_data, f, indent=2)

        print(f"Tokenizer saved to {directory}/")

    @classmethod
    def load(cls, directory: str, name: str = "tokenizer") -> "BPETokenizer":
        """
        Load a trained tokenizer from disk.

        Args:
            directory: Directory containing tokenizer files
            name: File name prefix

        Returns:
            Loaded BPETokenizer instance
        """
        vocab_path = os.path.join(directory, f"{name}_vocab.json")
        merges_path = os.path.join(directory, f"{name}_merges.txt")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        tokenizer = cls(vocab_size=vocab_data["vocab_size"])

        # Rebuild vocab
        tokenizer.vocab = {}
        tokenizer.token_to_id = {}
        for token_id_str, byte_list in vocab_data["vocab"].items():
            token_id = int(token_id_str)
            token_bytes = bytes(byte_list)
            tokenizer.vocab[token_id] = token_bytes
            if token_id >= tokenizer.num_special:
                tokenizer.token_to_id[token_bytes] = token_id

        # Rebuild merges
        tokenizer.merges = {}
        tokenizer.merge_list = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                pair = (int(parts[0]), int(parts[1]))
                merge_id = tokenizer.num_special + tokenizer.num_base + len(tokenizer.merge_list)
                tokenizer.merges[pair] = merge_id
                tokenizer.merge_list.append(pair)

        print(f"Tokenizer loaded from {directory}/ (vocab size: {len(tokenizer.vocab)})")
        return tokenizer

    @property
    def eos_token_id(self) -> int:
        return self.SPECIAL_TOKENS["<|endoftext|>"]

    @property
    def pad_token_id(self) -> int:
        return self.SPECIAL_TOKENS["<|padding|>"]

    @property
    def unk_token_id(self) -> int:
        return self.SPECIAL_TOKENS["<|unk|>"]

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={len(self.vocab)}, "
            f"merges={len(self.merge_list)}, "
            f"special_tokens={list(self.SPECIAL_TOKENS.keys())})"
        )
