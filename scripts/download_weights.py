"""
download_weights.py — Download OpenMind model weights from Hugging Face Hub.

Usage:
    python scripts/download_weights.py

Environment variables:
    HF_MODEL_REPO  — HF Hub repo ID  (default: Rachit17-12/openmind-125m)
    HF_TOKEN       — Optional HF token for private repos
    MODEL_PATH     — Used ONLY to derive the weights directory
                     (default: ./weights/model.pt → dir: ./weights/)

Files downloaded (always hardcoded):
    ./weights/model.pt
    ./weights/config.json
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


# Filenames to download from HF Hub — always hardcoded, never from env vars
HF_FILES = ["model.pt", "config.json"]


def _download_file(repo_id: str, filename: str, weights_dir: Path, hf_token) -> None:
    """Download one file from HF Hub into weights_dir, skip if already a file."""
    target_file = weights_dir / filename

    if target_file.exists() and target_file.is_file():
        size_mb = target_file.stat().st_size / 1_000_000
        print(f"[INFO] '{filename}' already present ({size_mb:.1f} MB). Skipping.")
        return

    print(f"[INFO] Downloading '{filename}' from {repo_id} into {weights_dir} ...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(weights_dir),
            token=hf_token,
        )
        print(f"[OK]   Saved to: {downloaded_path}")
    except Exception as exc:
        print(f"[ERROR] Failed to download '{filename}': {exc}")
        print(
            "\nTips:\n"
            "  1. Make sure HF_MODEL_REPO is correct.\n"
            "  2. For private repos, set HF_TOKEN in your environment.\n"
            "  3. Verify the filename exists in the Hub repo.\n"
        )
        sys.exit(1)


def download_weights() -> None:
    """Download model.pt and config.json from HF Hub into the weights directory."""
    repo_id = os.getenv("HF_MODEL_REPO", "Rachit17-12/openmind-125m")
    hf_token = os.getenv("HF_TOKEN") or None  # None → anonymous access

    # Derive the weights DIRECTORY from MODEL_PATH — never use it as a filename
    model_path = Path(os.getenv("MODEL_PATH", "./weights/model.pt"))
    weights_dir = model_path.parent.resolve()  # absolute: e.g. /app/weights

    # Create weights directory if it doesn't exist
    weights_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Weights directory: {weights_dir}")
    print(f"[INFO] HF repo:           {repo_id}")

    # Download each hardcoded file individually
    for filename in HF_FILES:
        _download_file(repo_id, filename, weights_dir, hf_token)

    # Final summary
    print(f"\n[OK] All files ready in: {weights_dir}")
    for filename in HF_FILES:
        fpath = weights_dir / filename
        status = f"{fpath.stat().st_size / 1_000_000:.1f} MB" if fpath.exists() else "MISSING"
        print(f"     {fpath}  [{status}]")


if __name__ == "__main__":
    download_weights()
