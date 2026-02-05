# 改进的多视角视频数据集使用指南

## 概述
`whole_video_dataset.py` 已完善，现在支持：
1. **多视角视频读取**：front、left、right 三个视角
2. **SAM 3D Body 关键点加载**：自动从 NPZ 文件读取 3D 关键点
3. **帧范围索引**：从 annotation dict 中读取 start/end 帧，准确索引视频片段

## 使用示例

### 基础使用

```python
from pathlib import Path
from project.dataloader.whole_video_dataset import whole_video_dataset
from project.map_config import VideoSample

# 1. 准备数据列表
dataset_idx = [
    VideoSample(
        person_id="person_01",
        env_folder="夜多い",
        env_key="night_high",
        label_path=Path("path/to/annotation.json"),
        videos={
            "front": Path("path/to/front_video.mp4"),
            "left": Path("path/to/left_video.mp4"),
            "right": Path("path/to/right_video.mp4"),
        }
    ),
    # 更多样本...
]

# 2. 配置 SAM 3D Body 结果目录（可选）
sam3d_body_dirs = {
    "front": Path("/workspace/data/sam3d_body_results_right/01/夜多い/front/"),
    "left": Path("/workspace/data/sam3d_body_results_right/01/夜多い/left/"),
    "right": Path("/workspace/data/sam3d_body_results_right/01/夜多い/right/"),
}

# 3. 创建数据集
annotation_file = "path/to/annotation_dict.json"
dataset = whole_video_dataset(
    experiment="test_experiment",
    dataset_idx=dataset_idx,
    annotation_file=annotation_file,
    sam3d_body_dirs=sam3d_body_dirs,
    transform=None  # 可选：添加转换函数
)

# 4. 使用数据集
for i in range(len(dataset)):
    sample = dataset[i]
    
    # 视频数据：每个视角 (B, C, T, H, W)
    front_video = sample["video"]["front"]
    left_video = sample["video"]["left"]
    right_video = sample["video"]["right"]
    
    # SAM 3D Body 关键点：每个视角 (B, T, K, 3) 或 None
    front_kpts = sample["sam3d_kpt"]["front"]
    left_kpts = sample["sam3d_kpt"]["left"]
    right_kpts = sample["sam3d_kpt"]["right"]
    
    # 标签
    labels = sample["label"]  # (B,)
    label_names = sample["label_info"]  # List[str]
    
    # 元数据
    meta = sample["meta"]
    print(f"Frames: {meta['start_frame']} - {meta['end_frame']}")
    print(f"FPS: {meta['fps']}")
```

## 输出格式详解

### sample["video"]
- **结构**：`{"front": Tensor, "left": Tensor, "right": Tensor}`
- **形状**：`(B, C, T, H, W)`
  - B: 按标签分割的片段数
  - C: 通道数（3 for RGB）
  - T: 时间步长（可变）
  - H, W: 高度和宽度

### sample["sam3d_kpt"]
- **结构**：`{"front": Tensor, "left": Tensor, "right": Tensor}`
- **形状**：`(B, T, K, 3)` 或 `None`
  - B: 片段数
  - T: 时间步长
  - K: 关键点数（从 NPZ 文件自动推断）
  - 3: (x, y, z) 坐标

### sample["label"] 
- **形状**：`(B,)`
- **内容**：Label ID（0-8，对应不同动作）

### sample["label_info"]
- **形状**：`List[str]`
- **内容**：Label 名称（如 "left", "right", "front"）

### sample["meta"]
- **字段**：
  - `experiment`: 实验名称
  - `index`: 数据集索引
  - `person_id`: 人员 ID
  - `env_folder`: 环境文件夹（如 "夜多い"）
  - `env_key`: 环境 Key
  - `start_frame`: 视频起始帧索引
  - `end_frame`: 视频结束帧索引
  - `fps`: 视频帧率

## 注意事项

### 1. 帧索引原理
数据集会自动从 annotation dict 中查找每个视频的 start/end 帧：
```
annotation_dict = {
    "person_01": {
        "夜多い": {"start": 100, "mid": 2000, "end": 4000}
    }
}
```
数据集将自动切取 frames[start:end]

### 2. SAM 3D Body 关键点格式
NPZ 文件中应包含 'output' 字段，其中包含以下之一：
- `keypoints_3d`: Shape (K, 3) 或 (K, 3, ...)
- `poses`: SMPL 格式的关键点

### 3. 缺失数据处理
- 如果 SAM 3D Body 目录不存在：不加载关键点（返回 None）
- 如果某些帧的 NPZ 文件不存在：使用零向量填充
- 如果关键点数量不一致：自动对齐到最大尺寸

### 4. 性能优化建议
```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,  # 多进程数据加载
    pin_memory=True,  # 加速 GPU 传输
)
```

## 故障排查

### 问题：FPS 无效
```
ValueError: Invalid fps=0 for video: ...
```
解决：确保视频文件格式正确且包含元数据

### 问题：SAM 3D Body NPZ 文件找不到
```
SAM 3D body directory not found: ...
```
解决：检查 `sam3d_body_dirs` 的路径是否正确

### 问题：关键点维度不匹配
增加日志级别查看详细信息：
```python
logging.getLogger("project.dataloader.whole_video_dataset").setLevel(logging.DEBUG)
```

## 相关文件

- [annotiation_dict.py](project/dataloader/annotation_dict.py) - Annotation 字典加载
- [prepare_label_dict.py](project/dataloader/prepare_label_dict.py) - Label 时间线处理
- [map_config.py](project/map_config.py) - 数据配置定义
