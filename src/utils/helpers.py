"""
OpenMind Utility Functions.

Common helpers used across the project.
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    name: str = "openmind",
) -> logging.Logger:
    """Configure logging for the project."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> dict:
    """Count model parameters by trainability."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_mb": total * 4 / (1024 * 1024),  # Assuming float32
    }


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    elif prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_number(n: int) -> str:
    """Format large numbers with suffixes (e.g., 125M, 1.3B)."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: str, indent: int = 2):
    """Save dictionary as JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: str) -> dict:
    """Load JSON file as dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def estimate_model_memory(config) -> dict:
    """Estimate memory requirements for a model configuration."""
    # Parameters (approximate)
    embed_params = config.vocab_size * config.dim
    attn_params = config.n_layers * (
        config.dim * config.dim * 3  # Q, K, V
        + config.dim * config.dim    # O
    )
    ffn_params = config.n_layers * (
        config.dim * config.intermediate_dim * 3  # gate, up, down
    )
    norm_params = config.n_layers * config.dim * 2 + config.dim  # per-layer + final
    total_params = embed_params + attn_params + ffn_params + norm_params

    # Memory estimates (bytes)
    fp32_mem = total_params * 4
    fp16_mem = total_params * 2
    training_mem = fp16_mem * 4  # Rough: params + grads + optimizer states

    return {
        "parameters": total_params,
        "parameters_formatted": format_number(total_params),
        "fp32_gb": fp32_mem / (1024**3),
        "fp16_gb": fp16_mem / (1024**3),
        "training_estimate_gb": training_mem / (1024**3),
    }
