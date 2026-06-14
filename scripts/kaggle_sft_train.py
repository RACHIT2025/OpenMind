"""
KAGLLE SFT FINE-TUNING NOTEBOOK SCRIPT - OpenMind 125M (Fast LoRA Version)
========================================================================
Instructions:
  1. Kaggle -> Create -> New Notebook -> Settings -> GPU T4 x1 (or x2)
  2. Copy CELL 1 below into first cell, configure your base model link, and run it (~2 min)
  3. Copy CELL 2 below into second cell, and run it (~3-5 mins)
  4. Once finished, click the generated link or use the direct link to download 'openmind-sft-final.zip'.
"""

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1: SETUP + DOWNLOAD BASE MODEL (paste this first)   ║
# ╚══════════════════════════════════════════════════════════╝

"""
import subprocess, os, sys, shutil

# 1. Clone repository
print("=== CLONING REPOSITORY ===")
subprocess.run(["git", "clone", "https://github.com/RACHIT2025/OpenMind.git"], cwd="/kaggle/working")
os.chdir("/kaggle/working/OpenMind")

# 2. Install dependencies
print("\n=== INSTALLING DEPENDENCIES ===")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "datasets", "regex", "tqdm", "pyyaml"])

# 3. Download and unpack your base model
# PASTE YOUR BASE MODEL DIRECT LINK HERE (e.g. transfer.sh or Google Drive direct link)
# If you uploaded openmind-125m.zip as a Kaggle input dataset, set BASE_MODEL_ZIP = "/kaggle/input/.../openmind-125m.zip"
BASE_MODEL_URL = "YOUR_BASE_MODEL_ZIP_DIRECT_URL" 

os.makedirs("models/checkpoints", exist_ok=True)
dest_dir = "models/checkpoints/openmind-125m-final"

if BASE_MODEL_URL != "YOUR_BASE_MODEL_ZIP_DIRECT_URL":
    print("\n=== DOWNLOADING BASE MODEL ===")
    import urllib.request
    urllib.request.urlretrieve(BASE_MODEL_URL, "/kaggle/working/base_model.zip")
    print("Unpacking base model...")
    shutil.unpack_archive("/kaggle/working/base_model.zip", dest_dir)
elif os.path.exists("/kaggle/input"):
    # Try to search for zip in Kaggle input datasets
    found_zip = None
    for root, dirs, files in os.walk("/kaggle/input"):
        for file in files:
            if file.endswith(".zip"):
                found_zip = os.path.join(root, file)
                break
    if found_zip:
        print(f"\n=== FOUND KAGGLE DATASET ZIP: {found_zip} ===")
        shutil.unpack_archive(found_zip, dest_dir)
        
if not os.path.exists(dest_dir) or not os.listdir(dest_dir):
    print("\n⚠️ WARNING: No base model found in models/checkpoints/openmind-125m-final!")
    print("Please upload your base model zip to Kaggle or paste a download link in BASE_MODEL_URL.")
else:
    print(f"\n✅ Base model successfully loaded into {dest_dir}!")
    print(os.listdir(dest_dir))
    
# 4. Optional: Customize your SFT Q&A dataset inline (runs if you want to overwrite data/sft_train.jsonl)
# Add your custom conversations below:
sft_examples = [
    {"instruction": "Hi", "input": "", "output": "Hello! I am OpenMind, your AI assistant. How can I help you today?"},
    {"instruction": "What is the capital of France?", "input": "", "output": "The capital of France is Paris."},
    {"instruction": "Tell me a joke", "input": "", "output": "Why don't scientists trust atoms? Because they make up everything!"},
    {"instruction": "Who are you?", "input": "", "output": "I am OpenMind, an open-source AI assistant built from scratch."},
    {"instruction": "What is 2+2?", "input": "", "output": "2 + 2 equals 4."},
]

import json
os.makedirs("data", exist_ok=True)
with open("data/sft_train.jsonl", "w") as f:
    for ex in sft_examples:
        f.write(json.dumps(ex) + "\\n")
print(f"✅ sft_train.jsonl created with {len(sft_examples)} conversations.")
"""

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2: RUN SFT FINE-TUNING + DOWNLOAD (paste this second)║
# ╚══════════════════════════════════════════════════════════╝

"""
import os, sys, shutil, subprocess
os.chdir("/kaggle/working/OpenMind")

# 1. Start Fine-Tuning using our LoRA configuration
print("=== STARTING SFT FINE-TUNING ===")
# Adjust config to run fast (3 epochs on small dataset is extremely quick)
subprocess.run([sys.executable, "src/training/sft_train.py", "--config", "configs/finetune_config.yaml"])

final_sft_dir = "models/checkpoints/sft/sft-final"
if os.path.exists(final_sft_dir):
    print("\n=== SFT COMPLETE! PACKAGING MODEL ===")
    # Copy tokenizer from base model since it is required for serving
    base_tokenizer = "models/checkpoints/openmind-125m-final/tokenizer"
    if os.path.exists(base_tokenizer):
        shutil.copytree(base_tokenizer, os.path.join(final_sft_dir, "tokenizer"), dirs_exist_ok=True)
        print("Copied tokenizer to fine-tuned folder.")
        
    shutil.make_archive("/kaggle/working/openmind-sft-final", "zip", final_sft_dir)
    print(f"✅ SFT Zip created: {os.path.getsize('/kaggle/working/openmind-sft-final.zip')/1e6:.1f}MB", flush=True)

    # Display local download link
    from IPython.display import FileLink, HTML
    display(FileLink("/kaggle/working/openmind-sft-final.zip"))
    display(HTML('<a href="/kaggle/working/openmind-sft-final.zip" download>📥 Click to download openmind-sft-final.zip</a>'))

    # Upload to transfer.sh for a direct remote download link
    try:
        print("\n📤 Uploading to transfer.sh for direct download link...", flush=True)
        res = subprocess.run(["curl", "--upload-file", "/kaggle/working/openmind-sft-final.zip", "https://transfer.sh/openmind-sft-final.zip"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            print(f"\n📥 Direct Download Link: {res.stdout.strip()}\n", flush=True)
        else:
            print("Upload to transfer.sh failed, please use local Kaggle sidebar downloads.", flush=True)
    except Exception as e:
         print(f"Upload failed: {e}", flush=True)
else:
    print("\n❌ SFT Training failed! Check logs above for details.")
"""
