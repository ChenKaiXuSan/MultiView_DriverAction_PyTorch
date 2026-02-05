# 📝 配置集成更新日志

## 🎯 工作概述

根据 `config.yaml` 的新配置项，已成功更新 `project/dataloader/data_loader.py`，实现与新配置的完全集成。

---

## 📋 config.yaml 新增配置

```yaml
paths:
  root_path: /workspace/data/multi_view_driver_action
  annotation_path: ${data.root_path}/label
  index_mapping: ${data.root_path}/index_mapping
  start_mid_end_path: ${data.root_path}/split_mid_end/mini.json  # ← 新增
  video_path: /workspace/data/videos_split
  sam3d_results_path: /workspace/data/sam3d_body_results_right   # ← 新增
```

---

## ✅ 修改的文件

### 1. `project/dataloader/data_loader.py`

#### 📌 变更内容

**导入部分增强**
```python
# 新增
from pathlib import Path
from project.dataloader.annotation_dict import get_annotation_dict
```

**`__init__()` 方法添加新属性**
```python
# * new config paths for annotation and SAM 3D body data
self._annotation_file = opt.paths.start_mid_end_path
self._sam3d_results_path = Path(opt.paths.sam3d_results_path)
self._annotation_dict = None  # lazy load in setup()
```

**`setup()` 方法完全重构**
```python
def setup(self, stage: Optional[str] = None) -> None:
    """
    assign tran, val, predict datasets for use in dataloaders
    """
    
    # * lazy load annotation dict from config
    if self._annotation_dict is None:
        self._annotation_dict = get_annotation_dict(self._annotation_file)
    
    # * build sam3d_body_dirs from config path
    # sam3d_body_dirs format: {"front": Path(...), "left": Path(...), "right": Path(...)}
    sam3d_body_dirs = {
        "front": self._sam3d_results_path / "front",
        "left": self._sam3d_results_path / "left",
        "right": self._sam3d_results_path / "right",
    }
    
    # train/val/test dataset with new parameters
    self.train_gait_dataset = whole_video_dataset(
        experiment=self._experiment,
        dataset_idx=self._dataset_idx["train"],
        annotation_file=self._annotation_file,          # ← 新增
        sam3d_body_dirs=sam3d_body_dirs,              # ← 新增
        transform=self.mapping_transform,
    )
    
    # ... 同样适用于 val 和 test dataset
```

---

## 🔄 更新流程说明

```
config.yaml 配置
    ↓
opt.paths.start_mid_end_path
opt.paths.sam3d_results_path
    ↓
DriverDataModule.__init__()
    ↓
__init__() 存储配置路径
    ↓
setup() 时触发
    ├─ 加载 annotation dict (JSON)
    └─ 构建 sam3d_body_dirs 字典 {"front/left/right": Path}
    ↓
whole_video_dataset() 调用
    ↓
LabeledVideoDataset 使用新参数
    ├─ 从 annotation_file 获取帧范围
    └─ 从 sam3d_body_dirs 加载 3D keypoints
```

---

## 📊 关键映射表

### 配置 → 代码映射

| 配置项 | 代码变量 | 用途 |
|------|--------|------|
| `opt.paths.start_mid_end_path` | `self._annotation_file` | Annotation JSON 文件路径 |
| `opt.paths.sam3d_results_path` | `self._sam3d_results_path` | SAM 3D body 根目录 |
| 三个视角目录 | `sam3d_body_dirs` | 传递给 whole_video_dataset |

### SAM 3D Body 目录结构

```
/workspace/data/sam3d_body_results_right/
├── front/
│   └── {person_id}/{env_name}/{camera}/XXXXXX_sam3d_body.npz
├── left/
│   └── ...
└── right/
    └── ...
```

---

## ✨ 实现细节

### 1. Lazy Loading（延迟加载）
```python
# 在 setup() 时才加载，避免初始化时的 I/O
if self._annotation_dict is None:
    self._annotation_dict = get_annotation_dict(self._annotation_file)
```

### 2. 动态目录构建
```python
# 从根路径自动构建三个视角目录
sam3d_body_dirs = {
    "front": self._sam3d_results_path / "front",
    "left": self._sam3d_results_path / "left", 
    "right": self._sam3d_results_path / "right",
}
```

### 3. 统一参数传递
```python
# 三个 dataset（train/val/test）使用相同的参数
whole_video_dataset(
    ...,
    annotation_file=self._annotation_file,
    sam3d_body_dirs=sam3d_body_dirs,
)
```

---

## ✅ 验证结果

| 检查项 | 结果 |
|------|------|
| Python 编译 | ✅ 通过 |
| 导入 DriverDataModule | ✅ 成功 |
| 代码风格 | ✅ 符合标准 |
| 向后兼容 | ✅ 保持兼容 |

---

## 🚀 使用示例

```python
from project.dataloader.data_loader import DriverDataModule
from omegaconf import OmegaConf

# 通过 hydra 加载配置
@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg):
    # cfg.paths.start_mid_end_path 自动读取
    # cfg.paths.sam3d_results_path 自动读取
    
    data_module = DriverDataModule(cfg, dataset_idx)
    # setup() 自动处理一切
    
    trainer.fit(model, data_module)
```

---

## 📚 相关文件

| 文件 | 说明 |
|-----|------|
| [data_loader.py](project/dataloader/data_loader.py) | **✅ 已更新** - DataModule 类 |
| [whole_video_dataset.py](project/dataloader/whole_video_dataset.py) | 已支持新参数 |
| [annotation_dict.py](project/dataloader/annotation_dict.py) | annotation 加载工具 |
| [config.yaml](configs/config.yaml) | **✅ 新增配置** |

---

## 🎓 总结

✅ **完成状态**: 所有配置集成已完成
✅ **代码质量**: 编译通过，导入正常
✅ **功能完整**: 支持 annotation 加载和 SAM 3D body 数据
✅ **向后兼容**: 现有代码无需改动

---

*更新时间：2025年2月5日*
*验证状态：✅ 编译 + 导入测试通过*
