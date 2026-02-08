# 🚨 训练OOM快速修复指南

## 问题
训练时出现 `CUDA out of memory` 错误

## 立即解决（2步）

### 1️⃣ 修改配置文件

```yaml
# configs/config.yaml
train:
  accumulate_grad_batches: 8  # 添加这行

trainer:
  precision: 16  # 添加这行

data:
  batch_size: 1  # 改小
  load_kpt: false  # 只用RGB
```

### 2️⃣ 运行训练

```bash
python project/main.py --config configs/config.yaml
```

**效果：内存降低~80%** ✅

---

## 如果还是OOM

### 方案A：降低分辨率
```yaml
data:
  img_size: [112, 112]  # 从224改到112
```

### 方案B：减少帧数
```yaml
data:
  num_frames: 8  # 从16改到8
```

### 方案C：更多梯度累积
```yaml
train:
  accumulate_grad_batches: 16  # 从8改到16
```

---

## 命令行快速测试

```bash
# 快速测试（只跑10个batch）
python project/main.py \
    --config configs/config.yaml \
    --trainer.precision=16 \
    --trainer.limit_train_batches=10

# 正式训练
python project/main.py \
    --config configs/config_low_memory.yaml
```

---

## 配置对比

| 配置项 | OOM前 | 修复后 | 效果 |
|--------|-------|--------|------|
| **precision** | 32 | 16 | 内存↓50% |
| **batch_size** | 4 | 1 | 内存↓75% |
| **accumulate** | 1 | 8 | 等效batch=8 |
| **load_kpt** | true | false | 内存↓10% |
| **综合** | 100% | **~15%** | ✅ |

---

## 监控内存

```bash
# 实时监控GPU
watch -n 0.5 nvidia-smi

# 或
nvidia-smi dmon -s mu
```

---

## ⚠️ 重要说明

**为什么训练时chunking不管用？**

- ❌ **训练**：需要保存所有激活用于梯度计算
- ✅ **推理**：不需要保存，chunking有效

**正确做法：**
- 训练：使用**梯度累积** + **混合精度**
- 推理：使用**batch chunking**（已实现）

---

## 完整示例配置

已提供：`configs/config_low_memory.yaml`

```bash
# 直接使用
python project/main.py --config configs/config_low_memory.yaml
```

---

## 帮助文档

详细说明见：`doc/TRAINING_OOM_SOLUTIONS.md`

---

**最快5分钟解决OOM问题！** ⚡
