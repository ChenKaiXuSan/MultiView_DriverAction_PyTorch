# RAM优化速查表

## 🚀 最快开始（推荐配置）

### 8GB VRAM + 16GB RAM
```bash
python project/main.py \
    trainer.precision=16 \
    train.accumulate_grad_batches=4 \
    data.batch_size=1 \
    data.max_video_frames=500
```

### 6GB VRAM + 8GB RAM
```bash
python project/main.py \
    trainer.precision=16 \
    train.accumulate_grad_batches=8 \
    data.batch_size=1 \
    data.max_video_frames=300 \
    data.img_size=160 \
    data.num_workers=2
```

### 4GB VRAM + <8GB RAM
```bash
python project/main.py \
    trainer.precision=16 \
    train.accumulate_grad_batches=16 \
    data.batch_size=1 \
    data.max_video_frames=200 \
    data.img_size=112 \
    data.num_workers=0 \
    model.use_gradient_checkpointing=true
```

---

## 📋 优化选项速查

### 内存优化参数

| 参数 | 推荐值 | 内存节省 | 说明 |
|-----|-------|---------|------|
| `trainer.precision` | 16 | 50% | 混合精度 |
| `train.accumulate_grad_batches` | 4-8 | 75-87% | 梯度累积 |
| `data.batch_size` | 1 | - | 减小batch |
| `data.max_video_frames` | 300-500 | 大 | 视频分块 |
| `data.img_size` | 112-160 | 中 | 降低分辨率 |
| `data.load_kpt` | false | 中 | 跳过关键点 |
| `data.num_workers` | 2-4 | 小 | 减少worker |
| `model.use_gradient_checkpointing` | true | 30% | 梯度检查点 |

---

## 🎯 常见场景

### 场景1：训练时OOM
```yaml
train.accumulate_grad_batches: 8
data.batch_size: 1
trainer.precision: 16
```

### 场景2：数据加载OOM
```yaml
data.max_video_frames: 300
data.num_workers: 2
data.load_kpt: false  # 如果不需要
```

### 场景3：想要大batch效果
```yaml
data.batch_size: 1
train.accumulate_grad_batches: 16  # 等效batch=16
```

### 场景4：Transformer模型太大
```yaml
model.use_gradient_checkpointing: true
trainer.precision: 16
```

---

## ⚡ 配置文件模板

### 标准配置 (configs/memory_optimized.yaml)
```yaml
data:
  batch_size: 2
  num_workers: 4
  img_size: 224
  max_video_frames: 500
  load_rgb: true
  load_kpt: false
  num_io_threads: 4

train:
  accumulate_grad_batches: 2

trainer:
  precision: 16

model:
  use_gradient_checkpointing: false
```

### 低内存配置 (configs/low_memory.yaml)
```yaml
data:
  batch_size: 1
  num_workers: 2
  img_size: 160
  max_video_frames: 300
  load_rgb: true
  load_kpt: false
  num_io_threads: 2

train:
  accumulate_grad_batches: 8

trainer:
  precision: 16

model:
  use_gradient_checkpointing: true
```

---

## 💡 快速诊断

### 问题：训练时OOM
```bash
# 解决方案（按顺序尝试）
1. 设置 trainer.precision=16
2. 增加 train.accumulate_grad_batches
3. 减小 data.batch_size
4. 启用 model.use_gradient_checkpointing
```

### 问题：数据加载慢/OOM
```bash
# 解决方案
1. 减小 data.max_video_frames
2. 设置 data.load_kpt=false（如果不需要）
3. 减少 data.num_workers
```

### 问题：系统RAM不足
```bash
# 解决方案
1. 减少 data.num_workers
2. 减小 data.batch_size
3. 降低 data.img_size
```

---

## 📊 性能参考

| 配置 | VRAM | RAM | 速度 | 精度 |
|-----|------|-----|------|------|
| 基线 | 14GB | 20GB | 1.0x | 100% |
| +FP16 | 7GB | 20GB | 1.4x | ~100% |
| +累积(4) | 4GB | 20GB | 1.0x | 100% |
| +组合 | 2GB | 12GB | 1.3x | ~100% |

---

## 🔍 监控命令

```bash
# GPU内存
nvidia-smi

# 系统RAM
free -h

# 实时监控
watch -n 1 nvidia-smi
```

---

## 📖 详细文档

完整说明请参考：[MEMORY_OPTIMIZATION_GUIDE_CN.md](./MEMORY_OPTIMIZATION_GUIDE_CN.md)
