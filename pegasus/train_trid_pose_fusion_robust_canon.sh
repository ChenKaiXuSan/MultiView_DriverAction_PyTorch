#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N trid_robust_canon
#PBS -o /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/trid_robust_canon.out
#PBS -e /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/trid_robust_canon.err

set -euo pipefail

# =============================================================================
# TriPoseFusion 单模块消融：robust_canon
# =============================================================================
# 目的：
#   单独验证 robust canonicalization 的贡献。
#
# 消融设置：
#   - 关闭 dilated temporal refiner
#   - 关闭 multi-scale velocity
#   - 关闭 gate entropy regularization
#   - 打开 robust canonicalization
#
# Fold：
#   固定只跑 fold 0。
# =============================================================================

PROJECT_DIR=/work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/pegasus"

export EXPERIMENT_NAME=robust_canon
export FOLD=0

export MAX_EPOCHS=${MAX_EPOCHS:-50}
export NUM_WORKERS=${NUM_WORKERS:-32}
export BATCH_SIZE=${BATCH_SIZE:-32}
export NUM_FRAMES=${NUM_FRAMES:-16}
export DEVICES=${DEVICES:-1}

bash "${PROJECT_DIR}/pegasus/train_trid_pose_fusion_ablation.sh"
