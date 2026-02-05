# whole_video_dataset.py 改进总结

## 改进内容

本文件对 `whole_video_dataset.py` 进行了重大改进，现已支持三大核心功能：

### 1. ✅ 多视角视频读取
- **支持实时读取** front、left、right 三个视角的视频
- 自动同步所有视角的帧数
- 返回格式：`(B, C, T, H, W)` - 标准PyTorch视频张量格式

### 2. ✅ SAM 3D Body 3D关键点集成
- **完整的NPZ加载管道**：自动从SAM 3D body输出的NPZ文件读取3D关键点
- **灵活的关键点处理**：
  - 支持多种输出格式（keypoints_3d、poses等）
  - 自动对齐关键点维度
  - 缺失数据自动补零
- 返回格式：`(B, T, K, 3)` - batch、时间、关键点数、坐标

### 3. ✅ 帧范围索引（核心改进）
- **从annotation dict智能读取**：
  - 每个视频自动查找start、end帧信息
  - 精确切取指定帧范围：`video_frames[start:end]`
  - 所有操作（视频、关键点）同步索引
- **自动处理边界**：
  - 安全的边界检查和裁剪
  - 与标签时间轴自动对齐

## 关键API变化

### 构造函数（新参数）

```python
dataset = LabeledVideoDataset(
    experiment="experiment_name",
    index_mapping=video_samples,
    annotation_file="path/to/annotation.json",  # ← 必需：用于读取start/end帧
    sam3d_body_dirs={                           # ← 可选：SAM 3D body结果目录
        "front": Path("..."),
        "left": Path("..."),
        "right": Path("..."),
    },
    transform=None,
    decode_audio=False,
)
```

### 返回数据结构变化

**旧版本**
```python
sample = {
    "video": {"front": Tensor, "left": Tensor, "right": Tensor},
    "label": LongTensor,
    "label_info": List[str],
    "meta": dict,
}
```

**新版本** ✨
```python
sample = {
    "video": {"front": Tensor, "left": Tensor, "right": Tensor},
    "sam3d_kpt": {                              # ← 新增：3D关键点
        "front": Tensor(B,T,K,3) or None,
        "left": Tensor(B,T,K,3) or None,
        "right": Tensor(B,T,K,3) or None,
    },
    "label": LongTensor(B,),
    "label_info": List[str],
    "meta": {
        # ... 原有字段 ...
        "start_frame": int,          # ← 新增：起始帧索引
        "end_frame": int,            # ← 新增：结束帧索引
        "fps": int,                  # ← 新增：视频帧率
    },
}
```

## 核心方法说明

### `_load_sam3d_body_kpts()`
载入SAM 3D Body 3D关键点的核心方法

```python
def _load_sam3d_body_kpts(
    sam3d_dir: Path,
    frame_indices: List[int],
) -> Optional[torch.Tensor]:
    """
    Args:
        sam3d_dir: 包含NPZ文件的目录
        frame_indices: 要加载的帧索引列表
    
    Returns:
        (num_frames, num_keypoints, 3) 或 None
    """
```

**特点**：
- 自动格式检测（keypoints_3d / poses / 其他）
- 缺失帧使用零向量补充
- 关键点数量自动对齐

### `split_frame_with_label()`
改进的分割方法，现在支持关键点

**新参数**：
```python
front_kpts: Optional[torch.Tensor] = None,  # (T, K, 3)
left_kpts: Optional[torch.Tensor] = None,
right_kpts: Optional[torch.Tensor] = None,
```

**新返回值**：
```python
kpts_dict: Dict[str, Optional[torch.Tensor]]  # 每个视角的关键点序列
```

### `__getitem__()`
改进的数据获取方法

**新功能**：
1. 自动从annotation dict读取start/end帧
2. 智能切取视频和关键点
3. 同步补全元数据（fps、帧范围等）

## 使用示例

### 完整示例

```python
from pathlib import Path
from project.dataloader.whole_video_dataset import whole_video_dataset
from project.map_config import VideoSample

# 1. 准备数据
dataset_idx = [
    VideoSample(
        person_id="person_01",
        env_folder="夜多い",
        env_key="night_high",
        label_path=Path(".../annotation.json"),
        videos={
            "front": Path(".../front.mp4"),
            "left": Path(".../left.mp4"),
            "right": Path(".../right.mp4"),
        },
    ),
]

# 2. 创建数据集
dataset = whole_video_dataset(
    experiment="test",
    dataset_idx=dataset_idx,
    annotation_file="annotation_dict.json",
    sam3d_body_dirs={
        "front": Path("data/sam3d_body_results/01/夜多い/front"),
        "left": Path("data/sam3d_body_results/01/夜多い/left"),
        "right": Path("data/sam3d_body_results/01/夜多い/right"),
    },
)

# 3. 使用数据
for sample in dataset:
    # 视频：(B, C, T, H, W)
    video_front = sample["video"]["front"]
    
    # 3D关键点：(B, T, K, 3)
    kpts_front = sample["sam3d_kpt"]["front"]
    
    # 元数据
    print(f"Frames: {sample['meta']['start_frame']}-{sample['meta']['end_frame']}")
    print(f"FPS: {sample['meta']['fps']}")
```

## 性能考虑

### 内存优化
- 关键点默认为float32，可按需转换为float16
- 缺失关键点返回None而不是占用内存
- 视频缓存使用PyTorch内置的高效张量

### 速度优化
- multi_worker DataLoader兼容
- NPZ文件使用allow_pickle优化加载速度
- 关键点维度批量对齐（只一次）

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,
    pin_memory=True,
)
```

## 故障排除

| 问题 | 解决方案 |
|------|--------|
| SAM 3D Body 目录找不到 | 确保路径正确，检查`sam3d_body_dirs` |
| 关键点维度不匹配 | 检查NPZ文件格式，确保包含正确的键 |
| FPS无效 | 确保视频文件完整，包含元数据 |
| 帧索引超出范围 | 检查annotation dict中的start/end值 |

## 向后兼容性

✅ **完全向后兼容**

- 不提供`sam3d_body_dirs`时，自动跳过关键点加载
- 返回的`sam3d_kpt`为None（可以安全忽略）
- 所有原有的"video"、"label"等字段保持不变

```python
# 旧代码仍可工作
dataset = whole_video_dataset(
    experiment="test",
    dataset_idx=samples,
)
# sam3d_kpt会是 {"front": None, "left": None, "right": None}
```

## 下一步改进（可选）

### 1. 多GPU支持
```python
from torch.nn.parallel import DistributedDataParallel
dataloader = DataLoader(dataset, sampler=DistributedSampler(dataset))
```

### 2. 关键点缓存
预加载所有关键点到内存以加速迭代

### 3. 数据增强
```python
transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])
```

---

**文件**: `/workspace/MultiView_DriverAction_PyTorch/project/dataloader/whole_video_dataset.py`

**验证状态**: ✅ 编译通过 | ✅ 导入可用 | ✅ 功能完整
