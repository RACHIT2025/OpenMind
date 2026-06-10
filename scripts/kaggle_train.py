"""
KAGGLE TRAINING SCRIPT - OpenMind 125M (Fast Version)
======================================================
Instructions:
  1. Kaggle -> Create -> New Notebook -> Settings -> GPU T4 x2 (or T4 x1)
  2. Copy CELL 1 below into first cell, run it (~2 min)
  3. Copy CELL 2 below into second cell, run it (~1.5 hours)
  4. Once finished, click the generated link to download the model zip.
"""

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1: SETUP + DATA PREPARATION (paste this first)      ║
# ╚══════════════════════════════════════════════════════════╝

"""
import subprocess, os, sys
subprocess.run(["git", "clone", "https://github.com/RACHIT2025/OpenMind.git"], cwd="/kaggle/working")
os.chdir("/kaggle/working/OpenMind")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "datasets", "regex", "tqdm", "pyyaml", "sentencepiece"])

import numpy as np, torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# Use GPT-2 tokenizer (Rust-based, super fast)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
VOCAB_SIZE = tokenizer.vocab_size
EOS_ID = tokenizer.eos_token_id
SEQ_LEN = 512
print(f"Tokenizer vocab={VOCAB_SIZE}", flush=True)

# Prepare training data (100K docs)
print("\\n=== PREPARING TRAINING DATA ===", flush=True)
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
tokens = []
ct = 0
for ex in tqdm(ds, total=100000, desc="Train"):
    if ct >= 100000: break
    t = ex.get("text", "")
    if len(t) < 50: continue
    tokens.extend(tokenizer.encode(t))
    tokens.append(EOS_ID)
    ct += 1

ns = len(tokens) // SEQ_LEN
os.makedirs("data", exist_ok=True)
np.array(tokens[:ns*SEQ_LEN], dtype=np.uint16).tofile("data/train.bin")
print(f"Train: {ns:,} seqs, {os.path.getsize('data/train.bin')/1e6:.1f}MB", flush=True)

# Validation data
print("\\n=== PREPARING VAL DATA ===", flush=True)
ds2 = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
vt = []
vc = 0
for ex in tqdm(ds2, total=5000, desc="Val"):
    if vc >= 5000: break
    t = ex.get("text", "")
    if len(t) < 50: continue
    vt.extend(tokenizer.encode(t))
    vt.append(EOS_ID)
    vc += 1

nv = len(vt) // SEQ_LEN
np.array(vt[:nv*SEQ_LEN], dtype=np.uint16).tofile("data/val.bin")
print(f"Val: {nv:,} seqs", flush=True)
print(f"\\n=== CELL 1 DONE === VOCAB={VOCAB_SIZE}", flush=True)
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2: TRAIN + TEST + DOWNLOAD (paste this second)       ║
# ╚══════════════════════════════════════════════════════════╝

"""
import os, math, time, shutil, importlib.util
import numpy as np, torch, torch.nn as nn
os.chdir("/kaggle/working/OpenMind")
torch.cuda.empty_cache()

# Load modules by file path
for name, path in [("cfg","/kaggle/working/OpenMind/src/models/config_openmind.py"),("mdl","/kaggle/working/OpenMind/src/models/modeling_openmind.py")]:
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    globals()[name] = m

V = 50257
model_config = cfg.OpenMindConfig(vocab_size=V, max_seq_len=512, dim=768, n_layers=12, n_heads=12, n_kv_heads=12, intermediate_dim=2048, dropout=0.0, tie_embeddings=True)
model = mdl.OpenMindModel(model_config).cuda()
print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB used", flush=True)

data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
NS = len(data) // 512

def batch(bs):
    ix = np.random.randint(0, NS, size=bs)
    b = np.stack([data[i*512:(i+1)*512].astype(np.int64) for i in ix])
    x = torch.from_numpy(b).long().cuda()
    return x, x.clone()

decay_p = [p for n,p in model.named_parameters() if p.ndim>=2 and "norm" not in n]
other_p = [p for n,p in model.named_parameters() if p.ndim<2 or "norm" in n]
opt = torch.optim.AdamW([{"params":decay_p,"weight_decay":0.1},{"params":other_p,"weight_decay":0.0}], lr=6e-4, betas=(0.9,0.95), eps=1e-8)

STEPS, WARM, MB, GA = 3000, 300, 4, 8
LR_MAX, LR_MIN = 6e-4, 6e-5

def lr(step):
    if step<WARM: return LR_MAX*(step+1)/WARM
    return LR_MIN+0.5*(LR_MAX-LR_MIN)*(1+math.cos(math.pi*(step-WARM)/(STEPS-WARM)))

ckdir = "models/checkpoints"
os.makedirs(ckdir, exist_ok=True)
start = 0
ex = [d for d in os.listdir(ckdir) if d.startswith("step-")]
if ex:
    lt = max(ex, key=lambda x:int(x.split("-")[1]))
    model.load_state_dict(torch.load(f"{ckdir}/{lt}/model.pt", map_location="cuda"))
    start = int(lt.split("-")[1])
    print(f"Resumed step {start}", flush=True)

scaler = torch.amp.GradScaler(enabled=True)
model.train()
losses = []
t0 = time.time()
print(f"Training {start}->{STEPS}, batch={MB}x{GA}={MB*GA}", flush=True)

for step in range(start, STEPS):
    clr = lr(step)
    for pg in opt.param_groups: pg["lr"] = clr
    opt.zero_grad(set_to_none=True)
    al = 0.0
    for _ in range(GA):
        x,y = batch(MB)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            lo = model(x, labels=y)["loss"] / GA
        scaler.scale(lo).backward()
        al += lo.item()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    losses.append(al)
    if (step+1)%50==0:
        avg=sum(losses[-50:])/50; el=time.time()-t0; eta=(STEPS-step-1)/(50/el)/3600
        print(f"Step {step+1:>6}/{STEPS} | loss {avg:.4f} | lr {clr:.2e} | ETA {eta:.1f}h", flush=True)
        t0=time.time()
    if (step+1)%1000==0:
        sp=f"{ckdir}/step-{step+1}"; os.makedirs(sp, exist_ok=True)
        torch.save(model.state_dict(), f"{sp}/model.pt"); model_config.save_pretrained(sp)
        print(f"Saved {sp}", flush=True)

final=f"{ckdir}/openmind-125m-final"
os.makedirs(final, exist_ok=True)
torch.save(model.state_dict(), f"{final}/model.pt")
model_config.save_pretrained(final)
print(f"\\nDONE! Saved to {final}", flush=True)

# ── TEST GENERATION ──
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()
print("="*60)
print("🧠 OpenMind 125M - Generation Test")
print("="*60)
prompts = [
    "Once upon a time",
    "The little dog went to",
    "There was a beautiful princess who"
]
for p in prompts:
    ids = tokenizer.encode(p)
    inp = torch.tensor([ids]).cuda()
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=150, temperature=0.8, top_k=50, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    print(f"\\nPrompt: {p}")
    print(f"Output: {text}")
    print("-"*60)

# ── PACKAGE FOR DOWNLOAD ──
shutil.make_archive("/kaggle/working/openmind-125m", "zip", final)
print(f"\\n✅ Zip created: {os.path.getsize('/kaggle/working/openmind-125m.zip')/1e6:.1f}MB", flush=True)

from IPython.display import FileLink, HTML
display(FileLink("/kaggle/working/openmind-125m.zip"))
display(HTML('<a href="/kaggle/working/openmind-125m.zip" download>📥 Click to download openmind-125m.zip</a>'))
"""
