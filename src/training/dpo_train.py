"""
OpenMind Direct Preference Optimization (DPO) Training.

Implements DPO (Rafailov et al., 2023) for alignment training.
Uses paired (chosen, rejected) responses to learn human preferences
without a separate reward model.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.modeling_openmind import OpenMindModel
from src.data.chat_templates import get_parser
from src.training.sft_train import apply_lora


class DPODataset(Dataset):
    """Dataset for DPO training with chosen/rejected pairs."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        max_prompt_length: int = 1024,
        format_name: str = "hh_rlhf",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.parser = get_parser(format_name)

        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        print(f"Loaded {len(self.examples)} DPO examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def _tokenize_and_pad(self, text: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text and return padded input_ids and attention_mask."""
        ids = self.tokenizer.encode(text, allowed_special={"all"})
        if len(ids) > max_len:
            ids = ids[:max_len]

        attention_mask = [1] * len(ids)
        pad_len = max_len - len(ids)
        ids = ids + [self.tokenizer.pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len

        return torch.tensor(ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)

    def __getitem__(self, idx):
        example = self.examples[idx]
        parsed = self.parser(example)

        prompt = parsed["prompt"]
        chosen = parsed["chosen"]
        rejected = parsed["rejected"]

        # Tokenize prompt + chosen
        chosen_text = prompt + chosen + "<|endoftext|>"
        rejected_text = prompt + rejected + "<|endoftext|>"

        chosen_ids, chosen_mask = self._tokenize_and_pad(chosen_text, self.max_length)
        rejected_ids, rejected_mask = self._tokenize_and_pad(rejected_text, self.max_length)

        # Get prompt length for label masking
        prompt_ids = self.tokenizer.encode(prompt, allowed_special={"all"})
        prompt_len = min(len(prompt_ids), self.max_prompt_length)

        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask,
            "prompt_len": prompt_len,
        }


def compute_dpo_loss(
    model: nn.Module,
    ref_model: nn.Module,
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
    prompt_len: int,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
) -> tuple[torch.Tensor, dict]:
    """
    Compute DPO loss.

    DPO loss = -log σ(β * (log π(chosen) - log π_ref(chosen) - log π(rejected) + log π_ref(rejected)))
    """
    def get_log_probs(model_out: dict, labels: torch.Tensor, mask: torch.Tensor, p_len: int) -> torch.Tensor:
        """Extract per-token log probabilities for the response portion."""
        logits = model_out["logits"]
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_mask = mask[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Mask prompt tokens (only score response)
        response_mask = shift_mask.clone()
        response_mask[:, :p_len - 1] = 0

        return (token_log_probs * response_mask).sum(-1)

    # Policy model log probs
    with torch.no_grad():
        ref_chosen_out = ref_model(chosen_ids)
        ref_rejected_out = ref_model(rejected_ids)

    policy_chosen_out = model(chosen_ids)
    policy_rejected_out = model(rejected_ids)

    policy_chosen_logps = get_log_probs(policy_chosen_out, chosen_ids, chosen_mask, prompt_len)
    policy_rejected_logps = get_log_probs(policy_rejected_out, rejected_ids, rejected_mask, prompt_len)
    ref_chosen_logps = get_log_probs(ref_chosen_out, chosen_ids, chosen_mask, prompt_len)
    ref_rejected_logps = get_log_probs(ref_rejected_out, rejected_ids, rejected_mask, prompt_len)

    # DPO implicit reward
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # Loss
    logits_diff = chosen_rewards - rejected_rewards

    if loss_type == "sigmoid":
        loss = -F.logsigmoid(logits_diff).mean()
    elif loss_type == "hinge":
        loss = torch.relu(1 - logits_diff).mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Label smoothing
    if label_smoothing > 0:
        loss = (1 - label_smoothing) * loss + label_smoothing * (-F.logsigmoid(-logits_diff).mean())

    metrics = {
        "loss": loss.item(),
        "chosen_rewards": chosen_rewards.mean().item(),
        "rejected_rewards": rejected_rewards.mean().item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
    }

    return loss, metrics


def main(config_path: str):
    """Run DPO training."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    dpo_cfg = config["dpo"]
    lora_cfg = config.get("lora", {})
    data_cfg = config["data"]
    ckpt_cfg = config["checkpoint"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load base model
    print("Loading policy model...")
    model = OpenMindModel.from_pretrained(model_cfg["base_model_path"], device=device)

    # Load reference model (frozen copy)
    print("Loading reference model...")
    ref_model = OpenMindModel.from_pretrained(model_cfg["base_model_path"], device=device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Apply LoRA to policy model only
    if lora_cfg.get("enabled", True):
        model = apply_lora(model, lora_cfg)

    # Load tokenizer
    from src.data.tokenizer import BPETokenizer
    tokenizer_path = os.path.join(model_cfg["base_model_path"], "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        tokenizer = BPETokenizer(vocab_size=32000)

    # Create dataset
    dataset = DPODataset(
        data_cfg["dpo_dataset"],
        tokenizer,
        max_length=dpo_cfg.get("max_length", 2048),
        max_prompt_length=dpo_cfg.get("max_prompt_length", 1024),
        format_name=data_cfg.get("format", "hh_rlhf"),
    )

    loader = DataLoader(dataset, batch_size=dpo_cfg["batch_size"], shuffle=True, num_workers=2)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=dpo_cfg["lr"])

    # Training
    epochs = dpo_cfg.get("epochs", 1)
    grad_accum = dpo_cfg.get("gradient_accumulation_steps", 1)
    beta = dpo_cfg.get("beta", 0.1)
    label_smoothing = dpo_cfg.get("label_smoothing", 0.0)
    loss_type = dpo_cfg.get("loss_type", "sigmoid")
    output_dir = ckpt_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    global_step = 0
    model.train()

    for epoch in range(epochs):
        print(f"\n--- DPO Epoch {epoch + 1}/{epochs} ---")

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"DPO Epoch {epoch+1}")):
            loss, metrics = compute_dpo_loss(
                model, ref_model,
                batch["chosen_ids"].to(device),
                batch["chosen_mask"].to(device),
                batch["rejected_ids"].to(device),
                batch["rejected_mask"].to(device),
                batch["prompt_len"][0].item(),
                beta=beta,
                label_smoothing=label_smoothing,
                loss_type=loss_type,
            )

            (loss / grad_accum).backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    print(
                        f"  Step {global_step} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Acc: {metrics['accuracy']:.2%} | "
                        f"Margin: {metrics['reward_margin']:.3f}"
                    )

    # Save final model
    final_path = os.path.join(output_dir, "dpo-final")
    os.makedirs(final_path, exist_ok=True)

    # Merge LoRA and save
    from src.training.sft_train import LoRALinear
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.merge()

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.save_pretrained(final_path)
    print(f"\nDPO training complete! Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenMind DPO Training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
