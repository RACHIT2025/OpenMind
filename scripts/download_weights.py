"""
download_weights.py — Download OpenMind model weights from Hugging Face Hub.

Usage:
    python scripts/download_weights.py

Environment variables:
    HF_MODEL_REPO  — HF Hub repo ID  (default: Rachit17-12/openmind-125m)
    HF_TOKEN       — Optional HF token for private repos
    MODEL_PATH     — Target weights file path (default: ./weights/model.pt)

Downloads into the same directory as MODEL_PATH:
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


def _download_file(repo_id: str, filename: str, weights_dir: Path, hf_token) -> None:
    """Download a single file from HF Hub into weights_dir, skipping if cached.

    BUG 1 fix: skip check uses the absolute resolved *file* path so it can
    never accidentally match a directory of the same name.
    BUG 2 fix: weights_dir is always an absolute resolved path, guaranteeing
    hf_hub_download places every file inside ./weights/ and not the cwd root.
    """
    # Always work with an absolute path to avoid any ambiguity
    abs_weights_dir = weights_dir.resolve()
    # BUG 1 fix: check the specific file, not the directory
    target_file = abs_weights_dir / filename

    if target_file.exists() and target_file.is_file():
        size_mb = target_file.stat().st_size / 1_000_000
        print(f"[INFO] '{filename}' already present at {target_file} ({size_mb:.1f} MB). Skipping.")
        return

    print(f"[INFO] Downloading '{filename}' from {repo_id} into {abs_weights_dir} ...")
    try:
        # BUG 2 fix: pass the absolute weights dir so hf_hub_download never
        # writes relative to the cwd (which would put config.json at /app/)
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(abs_weights_dir),
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

    # Resolve MODEL_PATH to a concrete *file* path (not a directory)
    # e.g. ./weights/model.pt  →  weights_dir = ./weights/
    model_path = Path(os.getenv("MODEL_PATH", "./weights/model.pt"))
    weights_dir = model_path.parent

    # Ensure the weights directory exists before any download attempt
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Files to download — both land in weights_dir
    #   ./weights/model.pt
    #   ./weights/config.json
    files_to_download = [
        model_path.name,   # e.g. "model.pt"
        "config.json",
    ]

    for filename in files_to_download:
        _download_file(repo_id, filename, weights_dir, hf_token)

    abs_weights_dir = weights_dir.resolve()
    print(f"\n[OK] All required files are present in: {abs_weights_dir}")
    for fname in files_to_download:
        fpath = abs_weights_dir / fname
        status = f"{fpath.stat().st_size / 1_000_000:.1f} MB" if fpath.exists() else "MISSING"
        print(f"     {fpath}  [{status}]")


if __name__ == "__main__":
    download_weights()
