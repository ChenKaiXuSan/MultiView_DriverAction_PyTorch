#!/bin/bash
#PBS -A SSR                        # ✅ 项目名（必须修改）
#PBS -q gpu                        # ✅ 队列名（gpu / debug / gen_S）
#PBS -l elapstim_req=24:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N run_single_train_3dcnn                     # 🏷 作业名
#PBS -t 0-2                       # 多任务作业 ID 范围（根据需要修改）
#PBS -o logs/pegasus/run_single_train_3dcnn.log
#PBS -e logs/pegasus/run_single_train_3dcnn_err.log

# === 切换到作业提交目录 ===
cd /work/SSR/share/code/MultiView_DriverAction_PyTorch

mkdir -p logs/pegasus/

# === 加载 Python + 激活 Conda 环境 ===
source activate /home/SSR/luoxi/miniconda3/envs/multiview-video-cls
conda env list # 列出所有 Conda 环境以供参考

# === 可选：打印 GPU 状态 ===
nvidia-smi

# 输出当前环境信息
echo "Current working directory: $(pwd)"
echo "Current Python version: $(python --version)"
echo "Current virtual environment: $(which python)"

# === 从 config.yaml 读取配置参数 ===
root_path=/work/SSR/share/data/drive/multi_view_driver_action
num_workers=16
batch_size=16
backbone=3dcnn
max_video_frames=30

# mapping view 
# 声明关联数组
declare -A VIEW_NAME_MAP
VIEW_NAME_MAP['0']='front'
VIEW_NAME_MAP['1']='left'
VIEW_NAME_MAP['2']='right'

echo "Training with view: ${VIEW_NAME_MAP[$PBS_SUBREQNO]}"

# === 运行训练脚本（使用配置中的参数）===
python -m project.main \
  paths.root_path=${root_path} \
  paths.video_path=/work/SSR/share/data/drive/videos_split \
  paths.sam3d_results_path=/work/SSR/share/data/drive/sam3d_body_results_right \
  data.num_workers=${num_workers} \
  data.batch_size=${batch_size} \
  model.backbone=${backbone} \
  train.view=single \
  train.view_name=[${VIEW_NAME_MAP[$PBS_SUBREQNO]}] \
  data.max_video_frames=${max_video_frames} \