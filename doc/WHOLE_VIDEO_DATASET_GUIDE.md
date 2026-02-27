# Whole Video Dataset 说明文档

本文档对应当前实现：`project/dataloader/whole_video_dataset.py`。

## 1. 功能概述

`LabeledVideoDataset` 用于多视角整段视频读取与按时间线切分标签，核心能力：

- 支持三个视角：`front`、`left`、`right`（可按 `view_name` 选择子集）
- 支持按标注帧区间读取视频（避免无关帧）
- 支持长视频分块读取（`max_video_frames`）以降低 OOM 风险
- 将标签从“原始视频绝对帧索引”转换到“当前加载片段相对索引”
- 输出按标签切分后的片段张量：`(B, C, T, H, W)`

---

## 2. 快速使用

```python
from project.dataloader.whole_video_dataset import whole_video_dataset

dataset = whole_video_dataset(
    experiment="exp1",
    dataset_idx=dataset_idx,              # List[VideoSample]
    annotation_dict=annotation_dict,      # Dict[str, Any]
    transform=None,
    max_video_frames=1000,                # 可选：开启分块
    view_name=["front", "left", "right"],
)

sample = dataset[0]
print(sample["video"]["front"].shape)  # (B, C, T, H, W)
print(sample["label"].shape)             # (B,)
print(sample["label_info"])              # List[str]
print(sample["meta"])                    # 元数据
```

---

## 3. 输入数据要求

### 3.1 `dataset_idx`（`List[VideoSample]`）

每个样本需包含：

- `person_id`
- `env_folder`
- `env_key`
- `label_path`
- `videos` 字典，至少可提供 `front/left/right` 对应视频路径

### 3.2 `annotation_dict`

用于确定每个样本的有效帧区间：

```python
annotation_dict[person_id][env_folder] = {
    "start": 1000,
    "end": 8000,
    # 其他字段可存在但本类不强依赖
}
```

若缺失 `start/end`，默认按 `start=0`，`end` 尽可能由实际加载长度确定。

---

## 4. 输出结构

`dataset[index]` 返回：

```python
{
  "video": {
    "front": Tensor(B, C, T, H, W),
    "left":  Tensor(B, C, T, H, W),
    "right": Tensor(B, C, T, H, W),
  },
  "label": LongTensor(B,),
  "label_info": List[str],
  "meta": {
    "experiment": str,
    "index": int,
    "person_id": str,
    "env_folder": str,
    "env_key": str,
    "start_frame": int,
    "end_frame": int,
    "fps": int,
    "is_chunked": bool,
    "chunk_info": dict | None,
  }
}
```

说明：

- `B` 为当前视频中按标签切出的片段数
- 三个视角中若某个未请求加载，会以零张量占位，保持接口一致

---

## 5. 标签与坐标转换逻辑

标签文件中的帧索引是**原始完整视频绝对帧号**，而内存中加载的视频通常是某一段子区间。

处理流程：

1. 先确定当前加载片段在原始视频中的绝对区间 `[abs_start, abs_end)`
2. 调用 `prepare_label_dict(..., start_frame=abs_start, end_frame=abs_end, fill_front=False)` 过滤重叠标签
3. 将每个标签段转换为相对索引：
   - `rel_start = seg_abs_start - abs_start`
   - `rel_end = seg_abs_end - abs_start`
4. 裁剪到 `[0, total_frames)`，并过滤空段

最终得到可直接用于切分张量的时间线。

---

## 6. 分块加载（Chunking）

当设置 `max_video_frames` 时：

- 数据集长度变为“chunk 数量”而不是“原视频数量”
- 每个样本只解码一个 chunk，降低峰值内存
- `meta.chunk_info` 中包含：
  - `chunk_idx`, `total_chunks`
  - `chunk_start_frame`, `chunk_end_frame`（相对 annotation start）
  - `absolute_start_frame`, `absolute_end_frame`（原视频绝对帧）

建议：

- 显存/内存紧张：`max_video_frames=500~1500`
- 追求吞吐：在可承受内存下适当增大 chunk

---

## 7. 性能设计

当前实现包含两级缓存：

- FPS 缓存：同一路径只探测一次 FPS
- 帧 LRU 缓存：缓存最近读取的 `(video_path, start_sec, end_sec)`

默认 LRU 大小为 2（最近两个视频片段）。

---

## 8. 常见问题

### 8.1 `ValueError: Invalid fps=0`

- 视频元数据缺失或文件损坏
- 建议先用 ffprobe/播放器检查视频可读性

### 8.2 `torch.stack` 报错（空列表）

- 通常表示当前片段内无有效标签段
- 需检查标签文件与 `start/end` 区间是否重叠

### 8.3 多视角帧数不一致

- 当前实现要求三视角时间轴一致
- 请在数据预处理阶段保证帧同步

---

## 9. 相关文件

- `project/dataloader/whole_video_dataset.py`
- `project/dataloader/prepare_label_dict.py`
- `project/map_config.py`
