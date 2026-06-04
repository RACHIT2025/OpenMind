"""
OpenMind CLI - Command Line Interface.

Commands:
  openmind download <model_id>           Download from HuggingFace hub
  openmind chat <model_dir>              Interactive terminal chat
  openmind serve <model_dir> --port 8000 Launch API server
  openmind eval <model_dir> --tasks ...  Run evaluation benchmarks
  openmind train --config <path>         Start training
  openmind convert <checkpoint_dir>      Convert checkpoint to HF format
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich.table import Table
except ImportError:
    print("CLI dependencies not installed. Run: pip install typer rich")
    sys.exit(1)

app = typer.Typer(
    name="openmind",
    help="🧠 OpenMind - Build, train, and serve your own LLM",
    add_completion=False,
)
console = Console()


@app.command()
def download(
    model_id: str = typer.Argument(..., help="HuggingFace model ID to download"),
    output_dir: str = typer.Option("models/", help="Output directory"),
):
    """Download a pretrained model from HuggingFace Hub."""
    console.print(Panel(f"Downloading [bold cyan]{model_id}[/]", title="OpenMind Download"))

    try:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=os.path.join(output_dir, model_id.split("/")[-1]),
            local_dir_use_symlinks=False,
        )
        console.print(f"\n✅ Model downloaded to: [bold green]{local_path}[/]")
    except Exception as e:
        console.print(f"\n❌ Download failed: [bold red]{e}[/]")
        raise typer.Exit(1)


@app.command()
def chat(
    model_dir: str = typer.Argument(..., help="Path to model directory"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    max_tokens: int = typer.Option(256, help="Maximum tokens to generate"),
    top_k: int = typer.Option(50, help="Top-k sampling"),
    top_p: float = typer.Option(0.9, help="Nucleus sampling"),
):
    """Interactive terminal chat with the model."""
    import torch
    from src.models.modeling_openmind import OpenMindModel
    from src.data.tokenizer import BPETokenizer
    from src.data.chat_templates import format_chat

    console.print(Panel("🧠 OpenMind Interactive Chat", subtitle="Type 'quit' to exit"))

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Loading model...")) as progress:
        task = progress.add_task("Loading", total=None)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = OpenMindModel.from_pretrained(model_dir, device=device)
        model.eval()

        tokenizer_path = os.path.join(model_dir, "tokenizer")
        if os.path.exists(tokenizer_path):
            tokenizer = BPETokenizer.load(tokenizer_path)
        else:
            tokenizer = BPETokenizer(vocab_size=32000)

    console.print(f"[dim]Model loaded on {device}. Ready to chat![/dim]\n")

    messages = []

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/] ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break

        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        prompt = format_chat(messages, add_generation_prompt=True)

        # Tokenize and generate
        input_ids = tokenizer.encode(prompt, allowed_special={"all"})
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0, len(input_ids):].tolist()
        response = tokenizer.decode(generated)

        # Clean up special tokens from response
        for special in ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>"]:
            response = response.replace(special, "")
        response = response.strip()

        messages.append({"role": "assistant", "content": response})

        console.print(f"\n[bold green]OpenMind:[/] ", end="")
        try:
            console.print(Markdown(response))
        except Exception:
            console.print(response)
        console.print()

    console.print("\n[dim]Goodbye! 👋[/dim]")


@app.command()
def serve(
    model_dir: str = typer.Argument(..., help="Path to model directory"),
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    device: str = typer.Option(None, help="Device (cuda/cpu/auto)"),
):
    """Launch the OpenAI-compatible API server."""
    console.print(Panel(
        f"Starting API server\n"
        f"Model: [bold]{model_dir}[/]\n"
        f"Endpoint: [bold cyan]http://{host}:{port}[/]",
        title="🌐 OpenMind Server",
    ))

    from src.inference.api_server import start_server
    start_server(model_dir, host, port, device)


@app.command(name="eval")
def evaluate(
    model_dir: str = typer.Argument(..., help="Path to model directory"),
    tasks: list[str] = typer.Option(None, help="Benchmark tasks to run"),
    fewshot: int = typer.Option(0, help="Number of few-shot examples"),
    max_examples: int = typer.Option(500, help="Max examples per benchmark"),
    output: str = typer.Option("results", help="Output directory"),
):
    """Run evaluation benchmarks on the model."""
    console.print(Panel("Running evaluation suite", title="🔬 OpenMind Eval"))

    from src.evaluation.run_eval import run_benchmark_suite

    if tasks is None:
        tasks = ["hellaswag", "arc_easy", "arc_challenge", "truthfulqa"]

    results = run_benchmark_suite(
        model_dir, tasks, fewshot, max_examples, output
    )

    # Display results table
    table = Table(title="Evaluation Results")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Accuracy", style="green", justify="right")

    for task, res in results.get("tasks", {}).items():
        table.add_row(task, f"{res['accuracy']:.2%}")

    console.print(table)


@app.command()
def train(
    config: str = typer.Option("configs/base_config.yaml", help="Path to training config"),
):
    """Start model training."""
    console.print(Panel(f"Starting training with config: [bold]{config}[/]", title="🚀 OpenMind Train"))

    from src.training.train import main as train_main
    train_main(config)


@app.command()
def convert(
    checkpoint_dir: str = typer.Argument(..., help="Path to training checkpoint"),
    output_dir: str = typer.Option(None, help="Output directory (default: same dir)"),
):
    """Convert a training checkpoint to HuggingFace format."""
    import torch
    from src.models.modeling_openmind import OpenMindModel
    from src.models.config_openmind import OpenMindConfig

    console.print(Panel(f"Converting checkpoint: [bold]{checkpoint_dir}[/]", title="🔄 Convert"))

    if output_dir is None:
        output_dir = checkpoint_dir + "-hf"

    config = OpenMindConfig.from_pretrained(checkpoint_dir)
    model = OpenMindModel(config)

    model_path = os.path.join(checkpoint_dir, "model.pt")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        console.print(f"[red]No model.pt found in {checkpoint_dir}[/]")
        raise typer.Exit(1)

    model.save_pretrained(output_dir)
    console.print(f"\n✅ Converted model saved to: [bold green]{output_dir}[/]")


@app.command()
def info():
    """Display system and project information."""
    import torch

    table = Table(title="🧠 OpenMind System Info")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Python", sys.version.split()[0])
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda or "N/A")
        table.add_row("GPU", torch.cuda.get_device_name(0))
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        table.add_row("GPU Memory", f"{mem:.1f} GB")
    table.add_row("BF16 Supported", str(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ))

    console.print(table)


if __name__ == "__main__":
    app()
