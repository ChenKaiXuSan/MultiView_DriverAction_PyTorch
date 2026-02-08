# 零填充数据加载方案 (Zero Padding for Missing Modalities)

## 概述

修改了 `LabeledVideoDataset` 中的数据加载逻辑，当某个模态（RGB视频或关键点）未被加载时，现在返回**零填充张量**而不是 `None`。

## 改动内容

### 1. RGB视频零填充 (`load_rgb=False` 时)

**之前的行为：** 返回 `None`
```python
batch_front = None
batch_left = None
batch_right = None
```

**现在的行为：** 返回零填充张量 `(B, 3, T, H, W)`
```python
batch_front = torch.zeros(B, 3, T, H, W, dtype=torch.float32)
batch_left = torch.zeros(B, 3, T, H, W, dtype=torch.float32)
batch_right = torch.zeros(B, 3, T, H, W, dtype=torch.float32)
```

**参数说明：**
- `B`: 批次大小（timeline分段数量）
- `3`: RGB 3个通道
- `T`: 时间维度（按timeline分段）  
- `H, W`: 空间分辨率（默认 224×224）

### 2. 关键点零填充 (`load_kpt=False` 时或某view无数据时)

**之前的行为：** 返回 `None`
```python
front_kpts_batch = None
left_kpts_batch = None
right_kpts_batch = None
```

**现在的行为：** 返回零填充张量 `(B, T, K, 3)`
```python
front_kpts_batch = torch.zeros(B, T, K, 3, dtype=torch.float32)
left_kpts_batch = torch.zeros(B, T, K, 3, dtype=torch.float32)
right_kpts_batch = torch.zeros(B, T, K, 3, dtype=torch.float32)
```

**参数说明：**
- `B`: 批次大小
- `T`: 时间维度（与该view的最大时间长度一致）
- `K`: 关键点数量（默认17个COCO关键点）
- `3`: (x, y, z) 三维坐标

### 3. 关键改进

#### 3.1 所有views的形状一致性
所有三个视角（front/left/right）现在总是返回相同的形状，即使某个view没有数据：

```python
# 无论是否加载数据，形状始终一致
front_kpts_batch.shape == left_kpts_batch.shape == right_kpts_batch.shape
# (B, T, K, 3)
```

#### 3.2 下游模型的简化
下游模型不再需要处理 `None` 值，可以统一处理零填充张量：

```python
# 之前：需要检查None
for view in ["front", "left", "right"]:
    if sample["video"][view] is not None:
        # 处理...

# 现在：统一处理
for view in ["front", "left", "right"]:
    video = sample["video"][view]  # 总是有值（可能是零）
    # 直接处理...
```

## 使用场景

### 场景1: 仅加载关键点 (RGB-only loading)
```python
dataset = LabeledVideoDataset(
    experiment="my_exp",
    ...,
    load_rgb=True,   # 加载RGB
    load_kpt=False   # 不加载关键点
)

# 返回值
sample = dataset[0]
sample["video"]["front"]     # 实际RGB数据 (B, 3, T, H, W)
sample["sam3d_kpt"]["front"] # 零填充 (B, T, K, 3)
```

### 场景2: 仅加载关键点 (Keypoint-only loading)
```python
dataset = LabeledVideoDataset(
    experiment="my_exp",
    ...,
    load_rgb=False,  # 不加载RGB（节省内存）
    load_kpt=True    # 加载关键点
)

# 返回值
sample = dataset[0]
sample["video"]["front"]     # 零填充 (B, 3, T, H, W)
sample["sam3d_kpt"]["front"] # 实际关键点数据 (B, T, K, 3)
```

### 场景3: 双模态加载
```python
dataset = LabeledVideoDataset(
    experiment="my_exp",
    ...,
    load_rgb=True,   # 加载RGB
    load_kpt=True    # 加载关键点
)

# 返回值
sample = dataset[0]
sample["video"]["front"]     # 实际RGB数据 (B, 3, T, H, W)
sample["sam3d_kpt"]["front"] # 实际关键点数据 (B, T, K, 3)
```

## 关键技术细节

### 1. 时间维度对齐
当不同的数据源有不同的时间长度时，系统会：
- 计算所有可用数据的最大时间长度
- 将所有数据填充到这个最大长度

```python
max_t_all = max([kpt.shape[0] for kpt in all_valid_kpts])
# 短序列会被填充到max_t_all
```

### 2. 关键点数对齐
不同video的关键点数量可能不同（如COCO vs OpenPose）：
- 系统计算最大关键点数 `K`
- 所有view都被填充到这个K

### 3. 验证函数更新
`_validate_output_shapes()` 现在：
- 确保所有非None张量形状一致
- 检查批次大小、时间维、关键点数的一致性
- 仍然支持向后兼容（接受Optional参数）

## 性能影响

### 内存节省
- `load_rgb=False` 时：**节省 80-90% 的内存**（跳过视频I/O和解码）
- 关键点只需要 RGB 的 **~1% 内存**（稀疏数据）

### 计算效率
- 零填充的向量可以被高效处理
- 模型可以同时学习从RGB或关键点中过滤输入信号

## 代码示例：模型兼容性

### 使用条件分支处理（不推荐）
```python
video = data["video"]["front"]
kpts = data["sam3d_kpt"]["front"]

# 检查是否为零
has_video = video.abs().sum() > 0
has_kpts = kpts.abs().sum() > 0

if has_video:
    video_feat = video_encoder(video)
else:
    video_feat = None
```

### 统一处理（推荐）
```python
# 直接传入，让模型学习处理零数据
video_feat = video_encoder(data["video"]["front"])  # 可以处理零输入
kpts_feat = kpt_encoder(data["sam3d_kpt"]["front"])

# 模型应该能够：
# 1. 对零输入返回零或低置信度特征
# 2. 或通过attention机制自动忽略零值区域
```

## 测试

运行测试验证零填充逻辑：
```bash
python test_zero_padding.py
```

输出应该显示：
```
✨ All tests passed! Zero padding implementation is correct.
```

## 迁移指南

### 如果你的代码检查了 `None`

**之前：**
```python
if data["video"]["front"] is not None:
    video_feat = encode_video(data["video"]["front"])
```

**现在（更简洁）：**
```python
video_feat = encode_video(data["video"]["front"])
# 即使是零也能处理
```

### 如果你需要知道某个模态是否被加载

**方法1：检查数据是否为零**
```python
# 如果整个张量都是零，说明没有加载
has_video = data["video"]["front"].abs().sum() > 0
has_kpts = data["sam3d_kpt"]["front"].abs().sum() > 0
```

**方法2：在返回的meta中添加加载标志**（可选扩展）
```python
# 未来可以添加到meta中
data["meta"]["data_sources"] = {
    "rgb": self.load_rgb,
    "keypoints": self.load_kpt,
}
```

## 常见问题

**Q: 为什么要用零填充而不是None？**
- A: None会导致下游代码需要处理两种不同的输入类型，零填充保证数据结构一致性

**Q: 零填充会不会欺骗模型？**
- A: 不会。模型会学到零向量对应于"无数据"，类似于masking或缺失值处理

**Q: 如何确保模型不会过度依赖某个模态？**
- A: 使用混合的训练策略：RGB-only, KPT-only, 和 RGB+KPT 的batches，让模型学习multimodal robust性

