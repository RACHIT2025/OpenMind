# 🧠 OpenMind

**Build, Train, and Serve Your Own Language Model**

OpenMind is a complete, from-scratch implementation of a GPT-style language model. From tokenizer to training to deployment — everything is built transparently so you can learn, modify, and own your AI.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔤 **BPE Tokenizer** | Custom byte-pair encoding tokenizer with GPT-2 style pre-tokenization |
| 🧠 **Transformer Model** | Decoder-only GPT with RoPE, RMSNorm, SwiGLU, and Grouped Query Attention |
| 📊 **Data Pipeline** | MinHash deduplication, quality filtering, memory-mapped binary output |
| 🚀 **Training** | Distributed training with DDP, mixed precision, gradient accumulation |
| ⚙️ **Fine-Tuning** | LoRA + DPO for instruction following and alignment |
| 🔬 **Evaluation** | HellaSwag, ARC, TruthfulQA benchmarks |
| 🌐 **API Server** | OpenAI-compatible `/v1/chat/completions` with streaming |
| 🖥️ **Chat UI** | Beautiful ChatGPT-like web interface |
| 📦 **CLI** | Full command-line tool for download, train, chat, serve, eval |
| 🐳 **Docker** | Complete containerization for training and serving |

## 🏗️ Architecture

```
125M Parameter Model (openmind-125m)
├── Embedding: 32,000 × 768
├── 12 × Transformer Block
│   ├── RMSNorm → Grouped Query Attention (12 heads)
│   │   ├── Q/K/V projections (RoPE applied)
│   │   ├── Scaled dot-product attention
│   │   └── Output projection
│   └── RMSNorm → SwiGLU FFN
│       ├── Gate projection (768 → 2048)
│       ├── Up projection (768 → 2048)
│       └── Down projection (2048 → 768)
├── RMSNorm
└── LM Head (tied with embeddings)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd openmind
pip install -r requirements.txt
pip install -e .
```

### 2. Preview the Chat UI
Open `frontend/index.html` in your browser to see the ChatGPT-like interface (works in demo mode without a model).

### 3. Train on Google Colab
See the [Google Colab Training Guide](#-google-colab-training-guide) below for step-by-step instructions to train the 125M model for free.

### 4. Serve Your Model
```bash
python src/inference/api_server.py --model models/checkpoints/openmind-125m --port 8000
```

### 5. Chat!
Open `http://localhost:8000` or use the CLI:
```bash
python src/cli/main.py chat models/checkpoints/openmind-125m
```

## 📁 Project Structure

```
openmind/
├── configs/               # YAML training configurations
├── data/                  # Datasets (gitignored)
├── docker/                # Dockerfiles
├── frontend/              # Chat web UI
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── models/                # Model checkpoints (gitignored)
├── scripts/               # Launch scripts
├── src/
│   ├── cli/               # Command-line interface
│   ├── data/              # Tokenizer & data pipeline
│   ├── evaluation/        # Benchmark evaluation
│   ├── inference/         # API server
│   ├── models/            # Transformer architecture
│   ├── training/          # Training & fine-tuning
│   └── utils/             # Helpers
├── tests/                 # Unit tests
├── docker-compose.yml
├── requirements.txt
├── setup.py
└── README.md
```

## 🎓 Google Colab Training Guide

Since training requires a GPU, we recommend using **Google Colab** (free T4 GPU). Here's the complete process:

### Step 1: Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com) → New Notebook → Runtime → Change runtime type → **T4 GPU**

### Step 2: Setup (Run these cells)

```python
# Cell 1: Clone and setup
!git clone https://github.com/YOUR_USERNAME/openmind.git
%cd openmind
!pip install -q torch transformers datasets regex numpy tqdm pyyaml

# Cell 2: Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### Step 3: Train Tokenizer

```python
# Cell 3: Train tokenizer on TinyStories (small, high quality)
from src.data.tokenizer import BPETokenizer
from datasets import load_dataset

# Load a small dataset for tokenizer training
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
corpus = ""
for i, example in enumerate(ds):
    if i >= 10000:  # 10K documents for tokenizer
        break
    corpus += example["text"] + "\n"

# Train tokenizer
tokenizer = BPETokenizer(vocab_size=32000)
tokenizer.train(corpus, verbose=True)
tokenizer.save("models/tokenizer")
print("Tokenizer trained and saved!")
```

### Step 4: Prepare Training Data

```python
# Cell 4: Process dataset into binary format
from src.data.pipeline import DataPipeline

pipeline = DataPipeline(output_dir="data", max_seq_len=512)  # Shorter for Colab
pipeline.load_tokenizer("models/tokenizer")

# Process TinyStories (small, fits in Colab)
train_path = pipeline.process_dataset(
    "roneneldan/TinyStories",
    split="train",
    max_documents=100000,  # 100K stories
    output_name="train",
)

val_path = pipeline.process_dataset(
    "roneneldan/TinyStories",
    split="validation",
    max_documents=5000,
    output_name="val",
)
print(f"Train: {train_path}, Val: {val_path}")
```

### Step 5: Train the Model

```python
# Cell 5: Train! (Takes ~2-4 hours on T4)
import yaml
from src.training.train import main as train_main

# Create Colab-optimized config
config = {
    "model": {
        "name": "openmind-125m",
        "vocab_size": 32000,
        "max_seq_len": 512,
        "dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "n_kv_heads": 12,
        "intermediate_dim": 2048,
        "dropout": 0.0,
        "tie_embeddings": True,
        "rope_theta": 10000.0,
    },
    "training": {
        "batch_size": 64,
        "micro_batch": 8,
        "max_steps": 10000,
        "warmup_steps": 500,
        "lr": 6e-4,
        "min_lr": 6e-5,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "scheduler": "cosine",
        "dtype": "float16",  # T4 doesn't support bf16
        "compile": False,
        "gradient_accumulation_steps": 8,
        "seed": 42,
    },
    "data": {
        "train_path": "data/train.bin",
        "val_path": "data/val.bin",
        "val_tokens": 1000000,
        "shuffle": True,
    },
    "checkpoint": {
        "save_every": 2000,
        "keep_last_n": 2,
        "output_dir": "models/checkpoints",
    },
    "logging": {
        "use_wandb": False,
        "log_every": 50,
        "project_name": "openmind-colab",
    },
}

# Save config
with open("configs/colab_config.yaml", "w") as f:
    yaml.dump(config, f)

# Train!
train_main("configs/colab_config.yaml")
```

### Step 6: Test Your Model

```python
# Cell 6: Generate text!
from src.models.modeling_openmind import OpenMindModel

model = OpenMindModel.from_pretrained("models/checkpoints/openmind-125m-final", device="cuda")
tokenizer = BPETokenizer.load("models/tokenizer")

prompt = "Once upon a time"
input_ids = torch.tensor([tokenizer.encode(prompt)]).cuda()
output = model.generate(input_ids, max_new_tokens=200, temperature=0.8)
print(tokenizer.decode(output[0].tolist()))
```

### Step 7: Download Your Model

```python
# Cell 7: Zip and download
!zip -r openmind-125m.zip models/checkpoints/ models/tokenizer/
from google.colab import files
files.download("openmind-125m.zip")
```

## 🔧 CLI Reference

```bash
# System info
python src/cli/main.py info

# Interactive chat
python src/cli/main.py chat models/checkpoints/openmind-125m

# Start API server
python src/cli/main.py serve models/checkpoints/openmind-125m --port 8000

# Run evaluations
python src/cli/main.py eval models/checkpoints/openmind-125m --tasks hellaswag arc_easy

# Start training
python src/cli/main.py train --config configs/base_config.yaml
```

## 🐳 Docker Deployment

```bash
# Build and run everything
docker-compose up --build

# Access:
# - Chat UI: http://localhost:80
# - API: http://localhost:8000
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v
```

## 📊 Model Sizes

| Model | Params | Layers | Dim | Heads | Training Time (T4) |
|-------|--------|--------|-----|-------|---------------------|
| openmind-125m | 125M | 12 | 768 | 12 | ~4 hours |
| openmind-350m | 350M | 24 | 1024 | 16 | ~12 hours |
| openmind-760m | 760M | 24 | 1536 | 16 | ~24 hours |

## 📜 License

MIT License - Build freely, share openly.

---

**Made with ❤️ by the OpenMind community**
