"""
OpenMind Training Script.

Supports:
- Single GPU and multi-GPU training (PyTorch DDP/FSDP)
- Mixed precision (bf16/fp16)
- Gradient accumulation and clipping
- Cosine learning rate schedule with warmup
- Checkpointing and resume
- WandB logging (optional)
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from contextlib import nullcontext

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.config_openmind import OpenMindConfig
from src.models.modeling_openmind import OpenMindModel
from src.data.pipeline import TokenDataset


# ─── Dataset Wrapper ──────────────────────────────────────────────────────────

class TrainDataset(Dataset):
    """PyTorch Dataset wrapper for memory-mapped token files."""

    def __init__(self, data_path: str, max_seq_len: int = 2048):
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.max_seq_len = max_seq_len
        self.num_sequences = len(self.data) // max_seq_len
        print(f"TrainDataset: {self.num_sequences} sequences from {data_path}")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.max_seq_len
        end = start + self.max_seq_len
        tokens = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(tokens)
        return x, x.clone()  # input_ids, labels


# ─── Learning Rate Scheduler ──────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Cosine decay
    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─── Checkpoint Management ────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: dict,
    output_dir: str,
    keep_last_n: int = 3,
):
    """Save training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save model state
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), os.path.join(checkpoint_path, "model.pt"))

    # Save optimizer state
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))

    # Save training state
    state = {
        "step": step,
        "loss": loss,
        "config": config,
        "rng_state": torch.random.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state().tolist()

    with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2)

    # Save model config
    model_to_save.config.save_pretrained(checkpoint_path)

    print(f"Checkpoint saved at step {step} -> {checkpoint_path}")

    # Cleanup old checkpoints
    if keep_last_n > 0:
        checkpoints = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        for old_ckpt in checkpoints[:-keep_last_n]:
            old_path = os.path.join(output_dir, old_ckpt)
            import shutil
            shutil.rmtree(old_path)
            print(f"Removed old checkpoint: {old_ckpt}")


def load_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> int:
    """Load checkpoint and return the step number."""
    model_to_load = model.module if hasattr(model, "module") else model

    model_path = os.path.join(checkpoint_dir, "model.pt")
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
    state_path = os.path.join(checkpoint_dir, "training_state.json")

    # Load model weights
    state_dict = torch.load(model_path, map_location=device)
    model_to_load.load_state_dict(state_dict)

    # Load optimizer state
    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

    # Load training state
    step = 0
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        step = state["step"]

        # Restore RNG state
        if "rng_state" in state:
            torch.random.set_rng_state(torch.ByteTensor(state["rng_state"]))
        if "cuda_rng_state" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state(torch.ByteTensor(state["cuda_rng_state"]))

    print(f"Resumed from checkpoint at step {step}")
    return step


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint in output directory."""
    if not os.path.exists(output_dir):
        return None

    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]

    if not checkpoints:
        return None

    latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, latest)


# ─── Main Training Function ──────────────────────────────────────────────────

def main(config_path: str):
    """Main training loop."""

    # ── Load config ────────────────────────────────────────
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    ckpt_cfg = config["checkpoint"]
    log_cfg = config["logging"]

    # ── Setup distributed training ─────────────────────────
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        is_master = rank == 0
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        is_master = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Seed everything ────────────────────────────────────
    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Build model ────────────────────────────────────────
    model_config = OpenMindConfig(
        vocab_size=model_cfg["vocab_size"],
        max_seq_len=model_cfg["max_seq_len"],
        dim=model_cfg["dim"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        n_kv_heads=model_cfg.get("n_kv_heads", model_cfg["n_heads"]),
        intermediate_dim=model_cfg.get("intermediate_dim", int(model_cfg["dim"] * 2.67)),
        dropout=model_cfg.get("dropout", 0.0),
        tie_embeddings=model_cfg.get("tie_embeddings", True),
        rope_theta=model_cfg.get("rope_theta", 10000.0),
    )

    if is_master:
        print(f"Model config: {model_config}")

    model = OpenMindModel(model_config).to(device)

    # Compile if supported
    if train_cfg.get("compile", False) and hasattr(torch, "compile"):
        if is_master:
            print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Wrap with DDP if distributed
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # ── Optimizer ──────────────────────────────────────────
    # Separate weight decay groups (no decay for biases and norms)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        eps=train_cfg["eps"],
    )

    # ── Data loading ───────────────────────────────────────
    train_dataset = TrainDataset(data_cfg["train_path"], model_cfg["max_seq_len"])

    if ddp:
        sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=data_cfg.get("shuffle", True)
        )
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["micro_batch"],
        shuffle=(sampler is None and data_cfg.get("shuffle", True)),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ── Gradient accumulation ──────────────────────────────
    grad_accum = train_cfg.get("gradient_accumulation_steps", "auto")
    if grad_accum == "auto":
        grad_accum = max(1, train_cfg["batch_size"] // (train_cfg["micro_batch"] * world_size))
    if is_master:
        print(f"Gradient accumulation steps: {grad_accum}")
        print(f"Effective batch size: {train_cfg['micro_batch'] * world_size * grad_accum}")

    # ── Mixed precision ────────────────────────────────────
    dtype_str = train_cfg.get("dtype", "float32")
    if dtype_str == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif dtype_str == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    amp_ctx = torch.autocast(device_type="cuda", dtype=dtype) if device.startswith("cuda") else nullcontext()
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    # ── Resume from checkpoint ─────────────────────────────
    start_step = 0
    output_dir = ckpt_cfg["output_dir"]
    latest_ckpt = find_latest_checkpoint(output_dir)
    if latest_ckpt:
        if is_master:
            print(f"Found checkpoint: {latest_ckpt}")
        raw_model = model.module if ddp else model
        start_step = load_checkpoint(latest_ckpt, raw_model, optimizer, device)

    # ── WandB ──────────────────────────────────────────────
    if log_cfg.get("use_wandb", False) and is_master:
        import wandb
        wandb.init(project=log_cfg["project_name"], config=config)

    # ── Training loop ──────────────────────────────────────
    max_steps = train_cfg["max_steps"]
    warmup_steps = train_cfg["warmup_steps"]
    max_lr = train_cfg["lr"]
    min_lr = train_cfg["min_lr"]
    grad_clip = train_cfg["grad_clip"]
    log_every = log_cfg.get("log_every", 10)
    save_every = ckpt_cfg.get("save_every", 5000)

    if is_master:
        print(f"\n{'='*60}")
        print(f"Starting training from step {start_step} to {max_steps}")
        print(f"Device: {device}, DDP: {ddp}, World size: {world_size}")
        print(f"{'='*60}\n")

    model.train()
    data_iter = iter(train_loader)
    running_loss = 0.0
    tokens_processed = 0
    t0 = time.time()

    for step in range(start_step, max_steps):
        # Update learning rate
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for micro_step in range(grad_accum):
            # Get batch (restart iterator if exhausted)
            try:
                x, y = next(data_iter)
            except StopIteration:
                if ddp:
                    sampler.set_epoch(step)
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            # Forward pass
            with amp_ctx:
                outputs = model(x, labels=y)
                loss = outputs["loss"] / grad_accum

            # Backward pass
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

        # Gradient clipping
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Tracking
        running_loss += accumulated_loss
        tokens_processed += train_cfg["micro_batch"] * model_cfg["max_seq_len"] * grad_accum * world_size

        # Logging
        if is_master and (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / log_every
            tokens_per_sec = tokens_processed / elapsed

            gpu_mem = ""
            if torch.cuda.is_available():
                mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                gpu_mem = f" | GPU mem: {mem_gb:.1f}GB"

            print(
                f"Step {step + 1}/{max_steps} | "
                f"loss: {avg_loss:.4f} | "
                f"lr: {lr:.2e} | "
                f"tok/s: {tokens_per_sec:.0f}{gpu_mem}"
            )

            if log_cfg.get("use_wandb", False):
                import wandb
                wandb.log({
                    "loss": avg_loss,
                    "lr": lr,
                    "tokens_per_sec": tokens_per_sec,
                    "step": step + 1,
                })

            running_loss = 0.0
            tokens_processed = 0
            t0 = time.time()

        # Save checkpoint
        if is_master and (step + 1) % save_every == 0:
            raw_model = model.module if ddp else model
            save_checkpoint(
                model, optimizer, step + 1, accumulated_loss,
                config, output_dir, ckpt_cfg.get("keep_last_n", 3)
            )

    # ── Final save ─────────────────────────────────────────
    if is_master:
        raw_model = model.module if ddp else model
        final_dir = os.path.join(output_dir, f"openmind-{model_cfg['name']}-final")
        raw_model.save_pretrained(final_dir)
        print(f"\nTraining complete! Final model saved to {final_dir}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenMind Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
