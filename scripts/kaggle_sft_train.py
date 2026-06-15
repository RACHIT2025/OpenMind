"""
KAGLLE SFT FINE-TUNING NOTEBOOK SCRIPT - OpenMind 125M (Fast Cleaned Alpaca Version)
=================================================================================
Instructions:
  1. Kaggle -> Create -> New Notebook -> Settings -> GPU T4 x1 (or T4 x2)
  2. In Kaggle, click "Add Input" -> Upload your pre-trained base model folder or zip (openmind-125m-final).
  3. Copy CELL 1 below into first cell and run it (~2 mins). It will auto-detect your uploaded base model
     and download 2,000 cleaned instructions from the Alpaca dataset.
  4. Copy CELL 2 below into second cell and run it (~3 mins on T4 GPU).
  5. Once finished, click the generated link or use the direct transfer.sh link to download 'openmind-sft-final.zip'.
"""

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1: SETUP + AUTOMATED BASE MODEL + DATASET DOWNLOAD ║
# ╚══════════════════════════════════════════════════════════╝

"""
import subprocess, os, sys, shutil

# 1. Clone repository
print("=== CLONING REPOSITORY ===")
if os.path.exists("/kaggle/working/OpenMind"):
    shutil.rmtree("/kaggle/working/OpenMind")
subprocess.run(["git", "clone", "https://github.com/RACHIT2025/OpenMind.git"], cwd="/kaggle/working")
os.chdir("/kaggle/working/OpenMind")

# 2. Install dependencies
print("\n=== INSTALLING DEPENDENCIES ===")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "datasets", "regex", "tqdm", "pyyaml"])

# 3. Locate and Link/Unpack Base Model
dest_dir = "models/checkpoints/openmind-125m-final"
os.makedirs(dest_dir, exist_ok=True)

print("\n=== LOCATING BASE MODEL ===")
found_base = False

# Search in Kaggle inputs
if os.path.exists("/kaggle/input"):
    # Look for config.json first
    for root, dirs, files in os.walk("/kaggle/input"):
        if "config.json" in files and any(f in files for f in ["model.pt", "model.safetensors", "pytorch_model.bin"]):
            print(f"Found base model files in directory: {root}")
            for f in files:
                if f in ["config.json", "model.pt", "model.safetensors", "pytorch_model.bin"]:
                    shutil.copy(os.path.join(root, f), os.path.join(dest_dir, f))
            found_base = True
            break
            
    # If not found, look for zip files in Kaggle inputs
    if not found_base:
        for root, dirs, files in os.walk("/kaggle/input"):
            for file in files:
                if file.endswith(".zip"):
                    zip_path = os.path.join(root, file)
                    print(f"Found zip archive: {zip_path}. Attempting to unpack...")
                    try:
                        shutil.unpack_archive(zip_path, dest_dir)
                        # Check if unpacked directory has another nested folder or contains files directly
                        # If nested, move them up
                        unpacked_files = os.listdir(dest_dir)
                        if len(unpacked_files) == 1 and os.path.isdir(os.path.join(dest_dir, unpacked_files[0])):
                            nested = os.path.join(dest_dir, unpacked_files[0])
                            for f in os.listdir(nested):
                                shutil.move(os.path.join(nested, f), os.path.join(dest_dir, f))
                            shutil.rmtree(nested)
                        found_base = True
                        break
                    except Exception as e:
                        print(f"Error unpacking zip: {e}")

if not found_base:
    # Option to download from direct link if not found in Kaggle inputs
    # PASTE YOUR BASE MODEL DIRECT LINK HERE (e.g. transfer.sh or Google Drive direct link)
    BASE_MODEL_URL = "YOUR_BASE_MODEL_ZIP_DIRECT_URL" 
    if BASE_MODEL_URL != "YOUR_BASE_MODEL_ZIP_DIRECT_URL":
        print(f"Downloading base model from URL: {BASE_MODEL_URL}")
        import urllib.request
        urllib.request.urlretrieve(BASE_MODEL_URL, "/kaggle/working/base_model.zip")
        print("Unpacking base model...")
        shutil.unpack_archive("/kaggle/working/base_model.zip", dest_dir)
        found_base = True

if found_base and os.path.exists(dest_dir) and os.listdir(dest_dir):
    print(f"\n✅ Base model successfully set up in {dest_dir}!")
    print(os.listdir(dest_dir))
else:
    print("\n⚠️ WARNING: Base model weights (model.pt/model.safetensors/pytorch_model.bin) or config.json not found!")
    print("Please upload your base model as a Kaggle dataset and attach it to this notebook, or paste a download link.")

# 4. Download Cleaned Alpaca Dataset from HF
print("\n=== DOWNLOADING CLEANED ALPACA DATASET ===")
import json
try:
    from datasets import load_dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset_slice = list(dataset)[:2000]
    
    os.makedirs("data", exist_ok=True)
    with open("data/sft_train.jsonl", "w", encoding="utf-8") as f:
        for ex in dataset_slice:
            f.write(json.dumps({
                "instruction": ex["instruction"],
                "input": ex["input"],
                "output": ex["output"]
            }) + "\n")
    print(f"✅ Downloaded and formatted {len(dataset_slice)} cleaned Alpaca instructions into data/sft_train.jsonl!")
except Exception as e:
    print(f"❌ Failed to download from HuggingFace ({e}). Using fallback mock dataset.")
    # Fallback to local examples
    sft_examples = [
        {"instruction": "Hi", "input": "", "output": "Hello! I am OpenMind, your AI assistant. How can I help you today?"},
        {"instruction": "What is the capital of France?", "input": "", "output": "The capital of France is Paris."},
        {"instruction": "Tell me a joke", "input": "", "output": "Why don't scientists trust atoms? Because they make up everything!"},
        {"instruction": "Who are you?", "input": "", "output": "I am OpenMind, an open-source AI assistant built from scratch."},
        {"instruction": "What is 2+2?", "input": "", "output": "2 + 2 equals 4."},
    ]
    os.makedirs("data", exist_ok=True)
    with open("data/sft_train.jsonl", "w", encoding="utf-8") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")
    print("✅ Fallback dataset prepared.")
"""

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2: RUN SFT FINE-TUNING + DOWNLOAD (paste this second)║
# ╚══════════════════════════════════════════════════════════╝

"""
import os, sys, shutil, subprocess
os.chdir("/kaggle/working/OpenMind")

# 1. Start Fine-Tuning
print("=== RUNNING SUPERVISED FINE-TUNING ===")
# Adjust config to run fast (3 epochs on 2000 instructions)
subprocess.run([sys.executable, "src/training/sft_train.py", "--config", "configs/finetune_config.yaml"])

final_sft_dir = "models/checkpoints/sft/sft-final"
if os.path.exists(final_sft_dir) and any(f in os.listdir(final_sft_dir) for f in ["model.pt", "model.safetensors", "pytorch_model.bin"]):
    print("\n=== FINE-TUNING COMPLETE! PACKAGING MODEL ===")
    
    # Pack the model files into zip
    zip_path = "/kaggle/working/openmind-sft-final"
    shutil.make_archive(zip_path, "zip", final_sft_dir)
    print(f"✅ SFT Zip created at: {zip_path}.zip ({os.path.getsize(zip_path + '.zip')/1e6:.1f}MB)", flush=True)

    # Display local download link
    from IPython.display import FileLink, HTML
    display(FileLink(zip_path + ".zip"))
    display(HTML(f'<a href="{zip_path}.zip" download>📥 Click here to download openmind-sft-final.zip</a>'))

    # Upload to transfer.sh for a direct remote download link
    try:
        print("\n📤 Uploading to transfer.sh for a direct download link...", flush=True)
        res = subprocess.run(["curl", "--upload-file", zip_path + ".zip", "https://transfer.sh/openmind-sft-final.zip"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            print(f"\n📥 Direct Download Link: {res.stdout.strip()}\n", flush=True)
        else:
            print("Upload to transfer.sh failed, please use local Kaggle sidebar downloads.", flush=True)
    except Exception as e:
         print(f"Upload failed: {e}", flush=True)
else:
    print("\n❌ Fine-Tuning failed! Check logs above for errors.")
"""
