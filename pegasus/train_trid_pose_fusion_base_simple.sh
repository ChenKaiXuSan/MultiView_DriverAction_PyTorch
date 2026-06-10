#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N trid_base_simple
#PBS -o /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/trid_base_simple.out
#PBS -e /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/trid_base_simple.err

set -euo pipefail

# =============================================================================
# TriPoseFusion baseline 实验：base_simple
# =============================================================================
# 目的：
#   作为最基础的三视角姿态融合 baseline。
#
# 消融设置：
#   - 关闭 dilated temporal refiner
#   - 关闭 multi-scale velocity
#   - 关闭 gate entropy regularization
#   - 关闭 robust canonicalization
#
# Fold：
#   固定只跑 fold 0。
# =============================================================================

PROJECT_DIR=/work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/pegasus"

export EXPERIMENT_NAME=base_simple
export FOLD=0

# 可通过 qsub -v MAX_EPOCHS=5 覆盖这些默认值做快速测试。
export MAX_EPOCHS=${MAX_EPOCHS:-50}
export NUM_WORKERS=${NUM_WORKERS:-32}
export BATCH_SIZE=${BATCH_SIZE:-32}
export NUM_FRAMES=${NUM_FRAMES:-16}
export DEVICES=${DEVICES:-1}

bash "${PROJECT_DIR}/pegasus/train_trid_pose_fusion_ablation.sh"
