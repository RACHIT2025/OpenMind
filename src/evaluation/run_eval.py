"""
OpenMind Evaluation Suite.

Evaluates language models on standard benchmarks:
- Perplexity on held-out data
- HellaSwag (commonsense reasoning)
- ARC-Easy / ARC-Challenge (science QA)
- TruthfulQA (truthfulness)
- MMLU subset (multitask knowledge)

Results are exported as JSON for comparison.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.modeling_openmind import OpenMindModel
from src.data.tokenizer import BPETokenizer


# ─── Perplexity Evaluation ────────────────────────────────────────────────────

def compute_perplexity(
    model: OpenMindModel,
    tokenizer: BPETokenizer,
    eval_data_path: str = None,
    eval_text: str = None,
    max_samples: int = 1000,
    max_seq_len: int = 2048,
    device: str = "cpu",
) -> float:
    """
    Compute perplexity on evaluation data.

    Args:
        model: The language model
        tokenizer: BPE tokenizer
        eval_data_path: Path to .bin file or text file
        eval_text: Raw text to evaluate on (alternative to file)
        max_samples: Maximum number of sequences to evaluate
        max_seq_len: Sequence length
        device: Device to run on

    Returns:
        Perplexity score (lower is better)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    if eval_data_path and eval_data_path.endswith(".bin"):
        # Memory-mapped binary
        data = np.memmap(eval_data_path, dtype=np.uint16, mode="r")
        num_sequences = min(len(data) // max_seq_len, max_samples)

        with torch.no_grad():
            for i in tqdm(range(num_sequences), desc="Computing perplexity"):
                start = i * max_seq_len
                end = start + max_seq_len
                tokens = torch.tensor(data[start:end].astype(np.int64), dtype=torch.long)
                tokens = tokens.unsqueeze(0).to(device)

                outputs = model(tokens, labels=tokens)
                total_loss += outputs["loss"].item() * (max_seq_len - 1)
                total_tokens += max_seq_len - 1

    elif eval_text:
        # Tokenize raw text
        token_ids = tokenizer.encode(eval_text)
        num_sequences = min(len(token_ids) // max_seq_len, max_samples)

        with torch.no_grad():
            for i in tqdm(range(num_sequences), desc="Computing perplexity"):
                start = i * max_seq_len
                end = start + max_seq_len
                tokens = torch.tensor(token_ids[start:end], dtype=torch.long)
                tokens = tokens.unsqueeze(0).to(device)

                outputs = model(tokens, labels=tokens)
                total_loss += outputs["loss"].item() * (max_seq_len - 1)
                total_tokens += max_seq_len - 1

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity


# ─── Multiple Choice Evaluation ──────────────────────────────────────────────

def evaluate_multiple_choice(
    model: OpenMindModel,
    tokenizer: BPETokenizer,
    examples: list[dict],
    num_fewshot: int = 0,
    device: str = "cpu",
) -> dict:
    """
    Evaluate model on multiple-choice questions.

    Each example should have:
    - "context": The question/context text
    - "choices": List of possible completions
    - "answer": Index of correct answer (0-based)

    Args:
        model: Language model
        tokenizer: Tokenizer
        examples: List of MC examples
        num_fewshot: Number of few-shot examples to prepend
        device: Device

    Returns:
        Dictionary with accuracy and per-example results
    """
    model.eval()
    correct = 0
    total = 0
    results = []

    for example in tqdm(examples, desc="Evaluating"):
        context = example["context"]
        choices = example["choices"]
        answer = example["answer"]

        # Score each choice by computing log-likelihood
        scores = []
        for choice in choices:
            full_text = context + " " + choice
            token_ids = tokenizer.encode(full_text)
            context_ids = tokenizer.encode(context)

            input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs["logits"]

            # Compute log probability of the choice tokens only
            choice_start = len(context_ids)
            log_probs = F.log_softmax(logits[0, choice_start - 1: -1], dim=-1)

            choice_token_ids = token_ids[choice_start:]
            score = sum(
                log_probs[i, tid].item()
                for i, tid in enumerate(choice_token_ids)
                if i < len(log_probs)
            )
            # Length-normalize
            score /= max(len(choice_token_ids), 1)
            scores.append(score)

        predicted = int(np.argmax(scores))
        is_correct = predicted == answer
        correct += int(is_correct)
        total += 1

        results.append({
            "context": context[:100] + "...",
            "predicted": predicted,
            "answer": answer,
            "correct": is_correct,
            "scores": scores,
        })

    accuracy = correct / max(total, 1)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


# ─── Benchmark Loaders ───────────────────────────────────────────────────────

def load_hellaswag(max_examples: int = 1000) -> list[dict]:
    """Load HellaSwag benchmark from Hugging Face."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Rowan/hellaswag", split="validation")

        examples = []
        for i, item in enumerate(ds):
            if i >= max_examples:
                break
            examples.append({
                "context": item["ctx"],
                "choices": item["endings"],
                "answer": int(item["label"]),
            })
        return examples
    except Exception as e:
        print(f"Could not load HellaSwag: {e}")
        return []


def load_arc(difficulty: str = "easy", max_examples: int = 1000) -> list[dict]:
    """Load ARC benchmark from Hugging Face."""
    try:
        from datasets import load_dataset
        subset = "ARC-Easy" if difficulty == "easy" else "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", subset, split="test")

        examples = []
        for i, item in enumerate(ds):
            if i >= max_examples:
                break

            choices = item["choices"]
            choice_texts = choices["text"]
            answer_key = item["answerKey"]

            # Convert answer key to index
            labels = choices["label"]
            answer_idx = labels.index(answer_key) if answer_key in labels else 0

            examples.append({
                "context": item["question"],
                "choices": choice_texts,
                "answer": answer_idx,
            })
        return examples
    except Exception as e:
        print(f"Could not load ARC: {e}")
        return []


def load_truthfulqa(max_examples: int = 500) -> list[dict]:
    """Load TruthfulQA benchmark."""
    try:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")

        examples = []
        for i, item in enumerate(ds):
            if i >= max_examples:
                break

            mc = item["mc1_targets"]
            choices = mc["choices"]
            labels = mc["labels"]
            answer_idx = labels.index(1) if 1 in labels else 0

            examples.append({
                "context": item["question"],
                "choices": choices,
                "answer": answer_idx,
            })
        return examples
    except Exception as e:
        print(f"Could not load TruthfulQA: {e}")
        return []


# ─── Full Benchmark Suite ─────────────────────────────────────────────────────

BENCHMARK_LOADERS = {
    "hellaswag": lambda n: load_hellaswag(n),
    "arc_easy": lambda n: load_arc("easy", n),
    "arc_challenge": lambda n: load_arc("challenge", n),
    "truthfulqa": lambda n: load_truthfulqa(n),
}


def run_benchmark_suite(
    model_path: str,
    tasks: list[str] = None,
    num_fewshot: int = 0,
    max_examples: int = 500,
    output_dir: str = "results",
    device: str = None,
) -> dict:
    """
    Run the full evaluation benchmark suite.

    Args:
        model_path: Path to model directory
        tasks: List of benchmark names to run
        num_fewshot: Number of few-shot examples
        max_examples: Max examples per benchmark
        output_dir: Directory to save results
        device: Device to use

    Returns:
        Dictionary with all results
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if tasks is None:
        tasks = list(BENCHMARK_LOADERS.keys())

    print(f"Loading model from {model_path}...")
    model = OpenMindModel.from_pretrained(model_path, device=device)
    model.eval()

    # Try to load tokenizer
    tokenizer_path = os.path.join(model_path, "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        tokenizer = BPETokenizer(vocab_size=32000)
        print("Warning: Using untrained tokenizer!")

    all_results = {
        "model": model_path,
        "timestamp": datetime.now().isoformat(),
        "num_fewshot": num_fewshot,
        "device": device,
        "tasks": {},
    }

    for task_name in tasks:
        if task_name not in BENCHMARK_LOADERS:
            print(f"Unknown task: {task_name}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Running: {task_name}")
        print(f"{'='*60}")

        examples = BENCHMARK_LOADERS[task_name](max_examples)
        if not examples:
            print(f"No examples loaded for {task_name}")
            continue

        result = evaluate_multiple_choice(
            model, tokenizer, examples, num_fewshot, device
        )

        all_results["tasks"][task_name] = {
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
        }

        print(f"  Accuracy: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    model_name = Path(model_path).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(output_dir, f"eval_{model_name}_{timestamp}.json")

    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for task, res in all_results["tasks"].items():
        print(f"  {task:20s}: {res['accuracy']:.2%}")
    print(f"\nResults saved to: {result_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenMind Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--fewshot", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_benchmark_suite(
        args.model, args.tasks, args.fewshot, args.max_examples, args.output, args.device
    )
