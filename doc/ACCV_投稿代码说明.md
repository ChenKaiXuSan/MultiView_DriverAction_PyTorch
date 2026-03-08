# ACCV 投稿代码说明（MultiView_DriverAction_PyTorch）

## 1. 项目目标与任务定义

本项目用于**多视角驾驶员头部动作识别**。输入来自同步的三路外部摄像头（`front / left / right`），输出为动作类别。

当前主任务为 **4 类分类**：
- `left`（合并 `left_up / left_down`）
- `right`（合并 `right_up / right_down`）
- `up`
- `down`

任务形式是**视频片段级分类**（由时间线标签切分后得到若干 segment）。

---

## 2. 代码总体流程

训练入口：`project/main.py`

主流程如下：
1. 读取 Hydra 配置（`configs/config.yaml`）
2. 调用 `DefineCrossValidation` 生成 K 折数据划分
3. 按 `train.view` 选择单视角或多视角 trainer
4. 构建 `DriverDataModule`
5. 使用 PyTorch Lightning 进行 `fit`
6. 每折训练后执行 `test(ckpt_path="best")`

这意味着本仓库默认是**K 折训练 + 每折最佳模型测试**的实验模式。

---

## 3. 数据与划分策略

### 3.1 样本构建

样本由 `label/*.json` 与视频目录配对得到。每个样本包含：
- `person_id`
- 环境（昼/夜 + high/low）
- 三视角视频路径
- 对应标签文件路径

核心代码：`project/cross_validation.py`、`project/map_config.py`

### 3.2 交叉验证

采用 `GroupKFold`，按 `person_id` 分组，确保同一被试不同时出现在训练集与验证集，避免身份泄漏。

### 3.3 标签处理

原始标签先映射/合并到 4 类（见 `map_config.py` 的 `normalize_label_to_4_class`）。

### 3.4 长视频内存优化

`whole_video_dataset` 支持 `max_video_frames` 分块读取：
- 长视频自动切 chunk
- 仅加载当前 chunk，降低显存/内存峰值
- 同时做 fps 与 frame 的缓存优化

---

## 4. 模型与融合方法

## 4.1 Backbone

支持三种主干：
- `3dcnn`（基于 PyTorchVideo Slow-R50）
- `transformer`
- `mamba`（时序建模变体）

入口：`project/models/make_model.py`

## 4.2 训练设置

### 单视角（Single-view）
- `train.view=single`
- 只用一个视角（如 `front`）
- Trainer：`project/trainer/single/*`

### 多视角（Multi-view）
- `train.view=multi`
- 融合策略在 `project/trainer/multi_selector.py` 路由

多视角支持：
1. **Early fusion**：`add / mul / concat / avg`
2. **Late fusion**：`late`（支持 `logit_mean / prob_mean / feature_mean / feature_concat`）
3. **Mid fusion（当前核心）**：`mid`，对应 TS-CVA trainer

---

## 5. TS-CVA（核心方法）

核心实现：`project/models/ts_cva_model.py`

TS-CVA 结构可概括为：
1. 对每个视角提取时空特征
2. 对每个时间步执行跨视角 MHSA（3 个 view token）
3. 用可学习 gate 进行动态视角加权聚合
4. 用 TCN 做时序建模
5. 分类头输出 logits

可选开关（用于消融）：
- 是否共享 backbone（`ts_cva_shared_backbone`）
- 是否使用 view embedding（`ts_cva_use_view_embedding`）
- 是否使用 gated aggregation（`ts_cva_use_gated_aggregation`）
- attention head 数与 temporal 维度层数

训练器：`project/trainer/multi/mid/train_multi_ts_cva.py`

训练器会记录：
- Accuracy / Precision / Recall / F1
- Confusion Matrix
- Gate 权重统计（便于解释性分析）

---

## 6. 关键配置（建议在论文中明确）

配置文件：`configs/config.yaml`

建议在论文实验设置中明确以下字段：
- `model.model_class_num`（当前为 4）
- `data.fold`（当前为 10）
- `data.uniform_temporal_subsample_num`（当前为 8）
- `data.max_video_frames`（当前为 1000）
- `train.view`（single / multi）
- `train.view_name`（单视角时使用）
- `model.backbone`（3dcnn / transformer / mamba）
- `model.fuse_method`（early / late / mid 路由）
- TS-CVA 各 ablation 开关参数

> 注意：当前代码中 TS-CVA 的路由键是 `model.fuse_method=mid`。

---

## 7. 复现实验命令示例

### 7.1 单视角基线（front）
```bash
python project/main.py train.view=single train.view_name="['front']" model.backbone=3dcnn
```

### 7.2 多视角晚融合（logit mean）
```bash
python project/main.py train.view=multi model.backbone=3dcnn model.fuse_method=late model.fusion_mode=logit_mean
```

### 7.3 多视角 TS-CVA（mid）
```bash
python project/main.py train.view=multi model.backbone=3dcnn model.fuse_method=mid train.view_name="['front','left','right']"
```

### 7.4 TS-CVA 消融：去掉 gated aggregation
```bash
python project/main.py train.view=multi model.backbone=3dcnn model.fuse_method=mid model.ts_cva_use_gated_aggregation=false
```

---

## 8. 论文可直接使用的实验对比设置

建议最少包含以下对比：
1. **Single-view**：front / left / right
2. **Early fusion**：`avg`、`concat`（可选 add/mul）
3. **Late fusion**：`logit_mean`、`feature_concat`
4. **TS-CVA（mid）**：完整模型
5. **TS-CVA 消融**：
   - 无 gate
   - 无 view embedding
   - 非共享 backbone

并报告：
- Acc、Macro-F1、Macro-Recall
- 每类结果与混淆矩阵
- Gate/Attention 的可解释性可视化

---

## 9. 当前仓库中的一致性说明（投稿前建议）

- 主训练流程以 `project/main.py` + Lightning trainer 为准。
- `project/eval.py` 中存在旧路径/命名痕迹（例如模块命名不一致），建议投稿实验统一使用 `main.py` 的训练测试日志与结果文件。
- 建议固定随机种子、记录配置文件快照和每折 checkpoint，保证复现性。

---

## 10. 一段可放论文方法章节的简述（可直接改写）

我们提出一个多视角驾驶员动作识别框架，输入来自同步的 front/left/right 视频流。系统首先通过共享或非共享的视角编码器提取每个视角的时空特征；随后在每个时间步执行跨视角多头自注意力，建模视角间互补关系；再通过可学习门控机制自适应分配视角权重，抑制遮挡或低质量视角；最后对融合后的时序表示进行时序卷积建模并输出动作类别。该框架在统一的数据划分与训练协议下，与单视角、早融合和晚融合基线进行公平对比，并提供门控权重与注意力矩阵用于模型可解释性分析。
