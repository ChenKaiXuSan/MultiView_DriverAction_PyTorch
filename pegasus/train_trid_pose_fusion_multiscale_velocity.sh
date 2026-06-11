#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N trid_multiscale_vel
#PBS -o /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/trid_multiscale_velocity.out
#PBS -e /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/trid_multiscale_velocity.err

set -euo pipefail

# =============================================================================
# TriPoseFusion 单模块消融：multiscale_velocity
# =============================================================================
# 目的：
#   单独验证 multi-scale velocity 动态特征的贡献。
#
# 消融设置：
#   - 关闭 dilated temporal refiner
#   - 打开 multi-scale velocity
#   - 关闭 gate entropy regularization
#   - 关闭 robust canonicalization
#
# Fold：
#   固定只跑 fold 0。
# =============================================================================

PROJECT_DIR=/work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/pegasus"

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/direction
set -u

echo "============================================================"
echo "TriPoseFusion ablation job: multiscale_velocity"
echo "Project dir: ${PROJECT_DIR}"
echo "Python: $(python --version)"
echo "Python path: $(which python)"
echo "Start time: $(date)"
echo "============================================================"
nvidia-smi
conda env list

export PYTHONPATH="${PROJECT_DIR}/TriPoseFusion:${PROJECT_DIR}:${PYTHONPATH:-}"

root_path=/work/SKIING/chenkaixu/data/drive
index_mapping=${root_path}/index_mapping
sam3d_results_path=/work/SKIING/chenkaixu/data/drive/sam3d_body_results_right

num_workers=${NUM_WORKERS:-32}
batch_size=${BATCH_SIZE:-32}
uniform_temporal_subsample_num=${NUM_FRAMES:-16}
max_epochs=${MAX_EPOCHS:-50}
devices=${DEVICES:-1}
fold=0
experiment_name=multiscale_velocity
view_names=${VIEW_NAMES:-'["front","left","right"]'}
view_tag=${view_names//[\"\[\] ]/}
view_tag=${view_tag//,/_}

use_dilated_refiner=false
use_multiscale_velocity=true
gate_entropy_lambda=0.0
use_robust_canonicalization=false
run_name="trifusion_${experiment_name}_views${view_tag}_${uniform_temporal_subsample_num}f_fold${fold}_dilated${use_dilated_refiner}_msvel${use_multiscale_velocity}_gate${gate_entropy_lambda}_robust${use_robust_canonicalization}"

echo "Experiment: ${run_name}"
echo "Fold: ${fold}"
echo "Views: ${view_names}"
echo "Index mapping: ${index_mapping}"
echo "SAM3D path: ${sam3d_results_path}"
echo "Dilated refiner: ${use_dilated_refiner}"
echo "Multiscale velocity: ${use_multiscale_velocity}"
echo "Gate entropy lambda: ${gate_entropy_lambda}"
echo "Robust canonicalization: ${use_robust_canonicalization}"

python TriPoseFusion/train.py \
  paths.root_path="${root_path}" \
  paths.index_mapping="${index_mapping}" \
  paths.sam3d_results_path="${sam3d_results_path}" \
  data.num_workers="${num_workers}" \
  data.batch_size="${batch_size}" \
  data.uniform_temporal_subsample_num="${uniform_temporal_subsample_num}" \
  model.backbone=triple_fusion \
  model.geofusion_use_dilated_refiner="${use_dilated_refiner}" \
  model.geofusion_use_multiscale_velocity="${use_multiscale_velocity}" \
  model.geofusion_gate_entropy_reg_lambda="${gate_entropy_lambda}" \
  model.geofusion_use_robust_canonicalization="${use_robust_canonicalization}" \
  train.view=multi \
  train.view_name="${view_names}" \
  train.fold="${fold}" \
  train.max_epochs="${max_epochs}" \
  train.devices="${devices}" \
  experiment="${run_name}"

echo "============================================================"
echo "Finished ${run_name}"
echo "End time: $(date)"
echo "============================================================"
