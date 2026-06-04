"""
OpenMind Model Configuration.

Hugging Face compatible configuration class for the OpenMind transformer model.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class OpenMindConfig:
    """Configuration for the OpenMind decoder-only transformer model."""

    # Model architecture
    vocab_size: int = 32000
    max_seq_len: int = 2048
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12           # Set < n_heads for Grouped Query Attention
    intermediate_dim: int = 2048    # SwiGLU intermediate dimension
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_theta: float = 10000.0

    # Derived
    head_dim: int = 0  # Will be computed as dim // n_heads

    # Model metadata
    model_type: str = "openmind"
    architectures: list = field(default_factory=lambda: ["OpenMindForCausalLM"])

    def __post_init__(self):
        self.head_dim = self.dim // self.n_heads
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    def save_pretrained(self, output_dir: str) -> None:
        """Save config to directory as config.json."""
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"Config saved to {config_path}")

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "OpenMindConfig":
        """Load config from a directory containing config.json."""
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        # Remove derived fields that will be recomputed
        config_dict.pop("head_dim", None)
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OpenMindConfig":
        """Load config from a YAML training config file."""
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        model_cfg = data.get("model", {})
        # Map YAML keys to dataclass fields
        return cls(
            vocab_size=model_cfg.get("vocab_size", 32000),
            max_seq_len=model_cfg.get("max_seq_len", 2048),
            dim=model_cfg.get("dim", 768),
            n_layers=model_cfg.get("n_layers", 12),
            n_heads=model_cfg.get("n_heads", 12),
            n_kv_heads=model_cfg.get("n_kv_heads", 12),
            intermediate_dim=model_cfg.get("intermediate_dim", 2048),
            dropout=model_cfg.get("dropout", 0.0),
            tie_embeddings=model_cfg.get("tie_embeddings", True),
            rope_theta=model_cfg.get("rope_theta", 10000.0),
            model_type=model_cfg.get("name", "openmind"),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in asdict(self).items())
        return f"OpenMindConfig({params})"


# Predefined model sizes
CONFIGS = {
    "openmind-125m": OpenMindConfig(
        dim=768, n_layers=12, n_heads=12, n_kv_heads=12, intermediate_dim=2048,
    ),
    "openmind-350m": OpenMindConfig(
        dim=1024, n_layers=24, n_heads=16, n_kv_heads=4, intermediate_dim=2730,
    ),
    "openmind-760m": OpenMindConfig(
        dim=1536, n_layers=24, n_heads=16, n_kv_heads=4, intermediate_dim=4096,
    ),
    "openmind-1.3b": OpenMindConfig(
        dim=2048, n_layers=24, n_heads=16, n_kv_heads=4, intermediate_dim=5460,
    ),
}
