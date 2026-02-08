# 分段推理指南 - 解决长视频OOM问题

## 概览

当 dataloader 加载的视频太长时（超过 GPU 显存容量），会导致 OOM（OutOfMemory）错误。本指南介绍如何使用分段推理（Chunk Inference）来解决这个问题。

## 原理

分段推理将长视频分割成多个较小的时间片段（chunks），分别进行推理，然后将结果聚合。这样可以显著降低单次前向传播所需的显存。

### 工作流程

```
长视频 [T=1000 frames]
    ↓
分割成 chunks [chunk_size=64]
    ├─ Chunk 1: frames 0-64    → logits_1
    ├─ Chunk 2: frames 32-96   → logits_2  (overlap=32)
    ├─ Chunk 3: frames 64-128  → logits_3
    └─ ...
    ↓
聚合方式
    ├─ mean: 平均所有 chunks 的 logits
    ├─ max:  对所有 chunks 取最大值
    └─ last: 只使用最后一个 chunk
    ↓
最终预测 [B, num_classes]
```

## 配置参数

在 `config.yaml` 的 `train` 部分添加以下参数：

```yaml
train:
  # 分段推理相关配置
  chunk_infer_size: 64          # 每个chunk的时间长度（帧数）
  chunk_overlap: 0              # chunks之间的重叠帧数
  chunk_aggregation: "mean"     # 聚合方式: mean/max/last
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `chunk_infer_size` | int | -1 | 每个chunk的帧数。设为-1表示不分段，直接处理整个视频。推荐值：32-128 |
| `chunk_overlap` | int | 0 | 相邻chunks之间的重叠帧数。非零值可提高预测稳定性，但增加计算量 |
| `chunk_aggregation` | str | "mean" | 结果聚合方式。mean最常用，last速度最快，max对异常值less sensitive |

## 使用示例

### 配置文件示例

**对于 RGB 视频（VGG3D 样例）：**
```yaml
model:
  model_class_num: 5
  input_type: "rgb"

train:
  chunk_infer_size: 64          # 每次处理64帧
  chunk_overlap: 0              # 无重叠
  chunk_aggregation: "mean"
```

**对于长视频（1000+ 帧）：**
```yaml
train:
  chunk_infer_size: 32          # 较小的chunk，节约显存
  chunk_overlap: 8              # 轻微重叠，提高稳定性
  chunk_aggregation: "mean"
```

**对于 RGB+KPT 融合模式：**
```yaml
model:
  input_type: "rgb_kpt"

train:
  chunk_infer_size: 48          # RGB和KPT都要处理，chunk稍小
  chunk_overlap: 4
  chunk_aggregation: "mean"
```

## 性能对比

假设原始视频 1000 帧：

| 方式 | 显存 | 推理时间 | 结果稳定性 |
|-----|------|--------|---------|
| 无分段 (OOM) | > 20GB | 快 | N/A |
| chunk=128, overlap=0 | ~4GB | 正常 | ✅ |
| chunk=64, overlap=0 | ~2GB | 1.2x | ✅ |
| chunk=64, overlap=8 | ~2.2GB | 1.5x | ✅✅ |
| chunk=32, overlap=8 | ~1.2GB | 2.5x | ✅✅✅ |

## 常见配置建议

### 🟢 轻度OOM（显存6-8GB）
```yaml
chunk_infer_size: 96
chunk_overlap: 0
chunk_aggregation: "mean"
```

### 🟡 中度OOM（显存4-6GB）
```yaml
chunk_infer_size: 64
chunk_overlap: 8
chunk_aggregation: "mean"
```

### 🔴 严重OOM（显存<4GB或超长视频）
```yaml
chunk_infer_size: 32
chunk_overlap: 8
chunk_aggregation: "mean"
```

## 代码使用

在训练代码中，分段推理会在 `validation_step` 和 `test_step` 中自动使用。

如果需要在自定义推理代码中使用：

```python
# 方式1: 自动处理（推荐）
trainer = SingleRes3DCNNTrainer(hparams)
logits = trainer._forward_with_chunking(video, kpts)

# 方式2: 禁用分段（处理小视频）
trainer.chunk_infer_size = -1
logits = trainer._forward_with_chunking(video, kpts)
```

## 推荐实践

1. **开发阶段**：设置 `chunk_infer_size: -1`，禁用分段以加快迭代
2. **验证/测试阶段**：根据显存设置合适的 chunk 大小
3. **生产部署**：使用保守的 chunk 大小以确保稳定性

## 测试你的配置

```bash
# 运行验证来测试配置是否适合你的硬件
python -m pytest tests/test_chunk_inference.py
```

## 故障排除

### 问题：仍然出现 OOM
**解决**：降低 `chunk_infer_size` 或增加 `chunk_overlap` 到 0

### 问题：推理结果不稳定
**解决**：增加 `chunk_overlap` 或改用 `chunk_aggregation: "mean"`

### 问题：推理太慢
**解决**：增加 `chunk_infer_size` 或设置 `chunk_overlap: 0`

---

**更新日期**：2026年2月8日  
**相关文件**：`project/trainer/single/train_single_3dcnn.py`
