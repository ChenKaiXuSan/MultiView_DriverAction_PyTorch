#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N trid_full
#PBS -o /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/trid_full.out
#PBS -e /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/trid_full.err

set -euo pipefail

# =============================================================================
# TriPoseFusion 完整模型实验：full
# =============================================================================
# 目的：
#   训练完整 TriPoseFusion，作为最终方法。
#
# 消融设置：
#   - 打开 dilated temporal refiner
#   - 打开 multi-scale velocity
#   - 打开 gate entropy regularization，lambda=0.01
#   - 打开 robust canonicalization
#
# Fold：
#   固定只跑 fold 0。
# =============================================================================

PROJECT_DIR=/work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/pegasus"

export EXPERIMENT_NAME=full
export FOLD=0

export MAX_EPOCHS=${MAX_EPOCHS:-50}
export NUM_WORKERS=${NUM_WORKERS:-32}
export BATCH_SIZE=${BATCH_SIZE:-32}
export NUM_FRAMES=${NUM_FRAMES:-16}
export DEVICES=${DEVICES:-1}

bash "${PROJECT_DIR}/pegasus/train_trid_pose_fusion_ablation.sh"
