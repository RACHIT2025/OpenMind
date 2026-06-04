#!/bin/bash
# ============================================================
# OpenMind - Launch Training Script
# ============================================================
# Usage:
#   Single GPU:  bash scripts/launch_train.sh
#   Multi-GPU:   bash scripts/launch_train.sh --nproc 4
# ============================================================

set -e

NPROC=${1:-1}
CONFIG=${2:-configs/base_config.yaml}

echo "============================================"
echo "  OpenMind Training Launcher"
echo "============================================"
echo "  GPUs: $NPROC"
echo "  Config: $CONFIG"
echo "============================================"

if [ "$NPROC" -gt 1 ]; then
    torchrun \
        --nnodes=1 \
        --nproc_per_node=$NPROC \
        --rdzv_id=openmind \
        --rdzv_backend=c10d \
        src/training/train.py \
        --config $CONFIG
else
    python src/training/train.py --config $CONFIG
fi
