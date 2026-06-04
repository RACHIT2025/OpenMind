"""
OpenMind 125M - Google Colab Training Notebook
================================================
Copy this file to Google Colab and run each section as a cell.
Runtime -> Change runtime type -> T4 GPU

Sections are separated by: # %%
"""

# %% [markdown]
# # 🧠 OpenMind 125M - Train Your Own LLM
# **Runtime → Change runtime type → T4 GPU**

# %% Cell 1: Setup
# !git clone https://github.com/YOUR_USERNAME/openmind.git
# %cd openmind
# !pip install -q torch transformers datasets regex numpy tqdm pyyaml

import torch
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% Cell 2: Train Tokenizer
import sys; sys.path.insert(0, "src")
from data.tokenizer import BPETokenizer
from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
corpus = ""
for i, ex in enumerate(ds):
    if i >= 10000: break
    corpus += ex["text"] + "\n"

tokenizer = BPETokenizer(vocab_size=32000)
tokenizer.train(corpus, verbose=True)
tokenizer.save("models/tokenizer")
print("✅ Tokenizer trained!")

# %% Cell 3: Prepare Data
from data.pipeline import DataPipeline

pipeline = DataPipeline(output_dir="data", max_seq_len=512)
pipeline.load_tokenizer("models/tokenizer")

pipeline.process_dataset("roneneldan/TinyStories", split="train",
                         max_documents=100000, output_name="train")
pipeline.process_dataset("roneneldan/TinyStories", split="validation",
                         max_documents=5000, output_name="val")
print("✅ Data prepared!")

# %% Cell 4: Train Model (~2-4 hours)
from training.train import main as train_main
train_main("configs/colab_config.yaml")

# %% Cell 5: Test Generation
from models.modeling_openmind import OpenMindModel

model = OpenMindModel.from_pretrained("models/checkpoints/openmind-openmind-125m-final", device="cuda")
model.eval()
tokenizer = BPETokenizer.load("models/tokenizer")

prompt = "Once upon a time"
ids = torch.tensor([tokenizer.encode(prompt)]).cuda()
out = model.generate(ids, max_new_tokens=200, temperature=0.8)
print(tokenizer.decode(out[0].tolist()))

# %% Cell 6: Download Model
# !zip -r openmind-125m.zip models/checkpoints/ models/tokenizer/
# from google.colab import files
# files.download("openmind-125m.zip")
