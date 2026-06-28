"""
download_weights.py — Download OpenMind model weights from Hugging Face Hub.

Usage:
    python scripts/download_weights.py

Environment variables:
    HF_MODEL_REPO  — HF Hub repo ID  (default: Rachit17-12/openmind-125m)
    HF_TOKEN       — Optional HF token for private repos
    MODEL_PATH     — Target file path (default: ./weights/model.pt)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (development convenience)
load_dotenv()

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print(
        "[ERROR] huggingface_hub is not installed. "
        "Run: pip install huggingface_hub"
    )
    sys.exit(1)


def download_weights() -> None:
    """Download model weights from HF Hub, skipping if already cached."""
    repo_id = os.getenv("HF_MODEL_REPO", "Rachit17-12/openmind-125m")
    hf_token = os.getenv("HF_TOKEN") or None  # None → anonymous access

    # Resolve target path from MODEL_PATH (default: ./weights/model.pt)
    model_path = Path(os.getenv("MODEL_PATH", "./weights/model.pt"))
    weights_dir = model_path.parent
    filename = model_path.name

    # Skip download if weights already exist
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1_000_000
        print(
            f"[INFO] Weights already present at {model_path} ({size_mb:.1f} MB). "
            "Skipping download."
        )
        return

    # Create weights directory
    weights_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading '{filename}' from HF Hub repo: {repo_id} ...")

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(weights_dir),
            token=hf_token,
        )
        print(f"[OK] Weights saved to: {downloaded_path}")
    except Exception as exc:
        print(f"[ERROR] Failed to download weights: {exc}")
        print(
            "\nTips:\n"
            "  1. Make sure HF_MODEL_REPO is correct.\n"
            "  2. For private repos, set HF_TOKEN in your environment.\n"
            "  3. Verify the filename matches the one on the Hub.\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    download_weights()
