# ============================================================
# OpenMind 125M — Hugging Face Docker Space
# ============================================================
# Port 7860 is REQUIRED by HF Spaces.
# Model weights are downloaded at container startup via
# scripts/download_weights.py using the HF_MODEL_REPO env var.
# ============================================================

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────
# Copy requirements first for layer caching
COPY requirements.txt .

# Install CPU-only PyTorch first (smaller image, HF Spaces default)
# then install remaining requirements
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────
COPY . .

# ── Install package in editable mode ─────────────────────────
RUN pip install --no-cache-dir -e .

# ── Weights directory ─────────────────────────────────────────
RUN mkdir -p weights

# ── HF Spaces requires port 7860 ─────────────────────────────
EXPOSE 7860

# ── Environment defaults (override via HF Space secrets) ─────
ENV MODEL_PATH=./weights/model.pt \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=7860

# ── Startup: download weights (if needed) then serve ─────────
CMD ["sh", "-c", "python scripts/download_weights.py && python src/inference/api_server.py"]
