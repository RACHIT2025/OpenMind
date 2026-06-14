"""
OpenMind Supervised Fine-Tuning (SFT) Script.

Fine-tunes a pretrained OpenMind model on instruction-following data
using LoRA (Low-Rank Adaptation) for parameter efficiency.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.config_openmind import OpenMindConfig
from src.models.modeling_openmind import OpenMindModel
from src.data.chat_templates import get_parser, format_chat


# ─── LoRA Implementation ──────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """LoRA adapter for a linear layer."""

    def __init__(
        self,
        original: nn.Linear,
        r: int = 64,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original.in_features
        out_features = original.out_features

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_out * self.scaling

    def merge(self):
        """Merge LoRA weights into the original linear layer."""
        self.original.weight.data += (self.lora_B @ self.lora_A) * self.scaling


def apply_lora(model: nn.Module, lora_config: dict) -> nn.Module:
    """Apply LoRA adapters to specified modules in the model."""
    target_modules = lora_config.get("target_modules", ["q_proj", "v_proj"])
    r = lora_config.get("r", 64)
    alpha = lora_config.get("lora_alpha", 16)
    dropout = lora_config.get("lora_dropout", 0.05)

    lora_count = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Find parent module
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, child_name = parts
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                    child_name = name

                lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora_layer)
                lora_count += 1

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied to {lora_count} layers")
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model


# ─── SFT Dataset ──────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 2048,
        format_name: str = "alpaca",
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.parser = get_parser(format_name)

        # Load data
        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        print(f"Loaded {len(self.examples)} SFT examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        parsed = self.parser(example)

        # Tokenize
        text = parsed["text"]
        token_ids = self.tokenizer.encode(text, allowed_special={"all"})

        # Get format template name (default to chat)
        prompt_messages = parsed["messages"][:-1]
        prompt_text = format_chat(prompt_messages, add_generation_prompt=True)
        prompt_token_ids = self.tokenizer.encode(prompt_text, allowed_special={"all"})
        prompt_len = len(prompt_token_ids)

        # Truncate or pad
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(token_ids)
            token_ids = token_ids + [self.tokenizer.pad_token_id] * pad_len

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = input_ids.clone()
        
        # Mask prompt (instruction/system) tokens in labels so loss is response-only
        labels[:min(prompt_len, self.max_seq_len)] = -100
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, labels



# ─── Main SFT Training ───────────────────────────────────────────────────────

def main(config_path: str):
    """Run supervised fine-tuning."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    sft_cfg = config["sft"]
    lora_cfg = config.get("lora", {})
    data_cfg = config["data"]
    ckpt_cfg = config["checkpoint"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained model
    print(f"Loading base model from {model_cfg['base_model_path']}...")
    model = OpenMindModel.from_pretrained(model_cfg["base_model_path"], device=device)

    # Apply LoRA
    if lora_cfg.get("enabled", True):
        model = apply_lora(model, lora_cfg)

    model = model.to(device)

    # Load tokenizer
    from src.data.tokenizer import BPETokenizer
    tokenizer_path = os.path.join(model_cfg["base_model_path"], "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        tokenizer = BPETokenizer(vocab_size=32000)
        print("Warning: Using untrained tokenizer!")

    # Create dataset
    dataset = SFTDataset(
        data_cfg["sft_dataset"],
        tokenizer,
        max_seq_len=sft_cfg.get("max_seq_len", 2048),
        format_name=data_cfg.get("format", "alpaca"),
    )

    loader = DataLoader(
        dataset,
        batch_size=sft_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Optimizer (only train LoRA params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=sft_cfg["lr"], weight_decay=sft_cfg.get("weight_decay", 0.0))

    # Training loop
    epochs = sft_cfg.get("epochs", 3)
    grad_accum = sft_cfg.get("gradient_accumulation_steps", 1)
    save_every = ckpt_cfg.get("save_every", 500)
    output_dir = ckpt_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    dtype = torch.bfloat16 if sft_cfg.get("dtype") == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float32

    global_step = 0
    model.train()

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        epoch_loss = 0.0

        for batch_idx, (input_ids, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=device == "cuda"):
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"] / grad_accum

            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    print(f"  Step {global_step} | Loss: {loss.item() * grad_accum:.4f}")

                if global_step % save_every == 0:
                    save_path = os.path.join(output_dir, f"sft-checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    # Save only LoRA weights
                    lora_state = {
                        k: v for k, v in model.state_dict().items()
                        if "lora_" in k
                    }
                    torch.save(lora_state, os.path.join(save_path, "lora_weights.pt"))
                    print(f"  LoRA checkpoint saved: {save_path}")

            epoch_loss += loss.item() * grad_accum

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

    # Final save
    final_path = os.path.join(output_dir, "sft-final")
    os.makedirs(final_path, exist_ok=True)

    # Merge LoRA weights and save full model
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.merge()

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.save_pretrained(final_path)
    print(f"\nSFT complete! Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenMind SFT Training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
