# TS-CVA 模型结构与 Trainer 详细说明

本文档针对以下实现文件，做“模型结构 + 训练流程 + 评估对接”的工程化说明：

- [project/trainer/multi/mid/train_multi_ts_cva.py](project/trainer/multi/mid/train_multi_ts_cva.py)
- [project/models/ts_cva_model.py](project/models/ts_cva_model.py)

---

## 1. 目标与整体思路

TS-CVA（Temporal-Synchronous Cross-View Attention）用于多视角驾驶行为识别，核心思想是：

1. 每个时刻先做跨视角交互（front/left/right 之间注意力）；
2. 再做动态视角加权融合（gating）；
3. 最后沿时间建模（TCN）并分类。

区别于仅在末端做 late fusion 的方法，TS-CVA 在时间轴上的每个 step 都显式建模视角互补关系。

---

## 2. 输入/输出与张量约定

### 2.1 输入 batch 格式

- `batch["video"]["front"]`: `(B, C, T, H, W)`
- `batch["video"]["left"]`: `(B, C, T, H, W)`
- `batch["video"]["right"]`: `(B, C, T, H, W)`
- `batch["label"]`: `(B,)`

### 2.2 模型最终输出

- `logits`: `(B, K)`，其中 `K = num_classes`
- 训练/测试预测：`probs = softmax(logits)`，`preds = argmax(probs)`

---

## 3. TS-CVA 结构分解（`TSCVAModel`）

## 3.1 Backbone 特征提取

每个视角视频先经 3D CNN 编码，得到特征：

$$
F^v \in \mathbb{R}^{B \times C' \times T' \times H' \times W'},\quad v \in \{front,left,right\}
$$

- 支持共享 backbone（默认）或每视角独立 backbone（ablation）。

## 3.2 空间池化为时序 token

对每个视角特征做空间维 `H',W'` 的全局平均池化：

$$
S^v \in \mathbb{R}^{B \times T' \times C'}
$$

三个视角堆叠后：

$$
S \in \mathbb{R}^{B \times T' \times 3 \times C'}
$$

可选地加入 `view embedding`（用于显式区分前/左/右角色）。

## 3.3 同步跨视角注意力（每个时间步）

对每个时刻 `t`，取 3 个 view token 做多头自注意力：

$$
X_t \in \mathbb{R}^{B \times 3 \times C'},\quad X'_t = MHSA(X_t)
$$

注意力权重形状：`(B, num_heads, 3, 3)`，表示视角间相互关注关系。

## 3.4 可学习门控聚合（Gated Aggregation）

对 `X'_t` 每个视角打分并 softmax 归一化：

$$
w_t = softmax(MLP(X'_t)) \in \mathbb{R}^{B \times 3}
$$

融合 token：

$$
z_t = \sum_{v=1}^{3} w_t^{(v)} X_t'^{(v)} \in \mathbb{R}^{B \times C'}
$$

若关闭门控，则退化为 3 个视角简单均值。

## 3.5 时间建模（TCN）

将 `z_t` 组成序列：

$$
Z \in \mathbb{R}^{B \times T' \times C'}
$$

经 `TemporalConvNet`（若干层 Conv1d + BN + ReLU + Dropout），最后自适应池化到长度 1，得到：

$$
h \in \mathbb{R}^{B \times D_t}
$$

其中 `D_t = ts_cva_temporal_dim`。

## 3.6 分类头

线性层输出类别 logits：

$$
logits = W h + b \in \mathbb{R}^{B \times K}
$$

---

## 4. Trainer 流程（`MultiTSCVATrainer`）

## 4.1 `training_step`

- 前向：`logits = model(videos, return_attention=False)`
- 损失：`cross_entropy(logits, labels)`
- 指标：accuracy / precision(macro) / recall(macro) / f1(macro)
- 日志：`train/loss`（step+epoch）

## 4.2 `validation_step`

- 前向：`return_attention=True`（便于可视化分析）
- 更新验证指标与混淆矩阵
- 前 10 个 batch 记录注意力与 gate 权重到内存（用于统计/可视化）

## 4.3 `on_validation_epoch_end`

- 汇总验证指标并写日志
- 输出验证混淆矩阵（logger 允许时）
- 统计 gate 的均值权重：front/left/right
- 清空本 epoch 的缓存与 metric state

## 4.4 `test_step`

- 前向：`return_attention=True`
- 计算并累计测试指标
- 关键：保存 `probs`（不是 `argmax`）到 `test_pred_list`，以便后续 AUROC 等分析
- 保存 `labels` 到 `test_label_list`

## 4.5 `on_test_epoch_end`

- 汇总并记录测试指标
- 调用 `save_helper(...)` 落盘预测与标签
- 目前命名规则与评估脚本对齐：
  - `best_preds/fold_x_pred.pt`
  - `best_preds/fold_x_label.pt`

---

## 5. 与评估脚本（eval）对接关系

当前评估脚本 [analysis/eval/evaluate_experiments.py](analysis/eval/evaluate_experiments.py) 的读取逻辑是：

1. 扫描 run 目录下 `best_preds/*_pred.pt`；
2. 找同名 `*_label.pt` 组成有效 fold；
3. 若 `pred` 是二维张量，按类别维 `argmax` 得到预测类；
4. 计算总体指标、per-fold、per-class、confusion matrix；
5. 进行基于共同 fold 的 paired permutation 显著性检验。

因此，Trainer 侧只要稳定产出上述命名格式，即可无缝进入后续 eval 与显著性检验。

---

## 6. 关键配置项（来自 `configs/config.yaml`）

- `model.ts_cva_shared_backbone`：是否共享 3D CNN 主干
- `model.ts_cva_use_view_embedding`：是否加入视角嵌入
- `model.ts_cva_use_gated_aggregation`：是否使用门控融合（否则均值）
- `model.ts_cva_num_heads`：跨视角注意力 head 数
- `model.ts_cva_temporal_dim`：TCN 隐层维度
- `model.ts_cva_temporal_layers`：TCN 层数

训练日志根目录：

- `log_path: logs/train/${experiment}/${date}/${time}`

这也是 `save_helper` 最终保存 `best_preds` 的上层路径来源。

---

## 7. 结构优缺点（工程视角）

## 7.1 优点

- 跨视角关系显式可建模（不是简单拼接）
- 门控权重有可解释性，可看到每个时间步依赖哪个视角
- 与现有训练/评估链路兼容（尤其是 fold 级 best_preds）

## 7.2 代价

- 比单视角或简单 late fusion 计算更重
- 需严格的多视角时序同步
- 注意力与门控缓存若过多，可能增加验证显存/内存占用

---

## 8. 快速检查清单（训练后）

1. run 目录下是否生成 `best_preds/fold_x_pred.pt` 与 `fold_x_label.pt`。
2. `pred.pt` 的形状是否为 `(N, K)`（概率）或 `(N,)`（类别索引）。
3. `label.pt` 是否为 `(N,)` 且与 `pred` 一一对应。
4. eval 输出是否生成 `per_fold_metrics.csv`、`summary.json`、`significance_vs_reference.csv`。

满足以上条件，说明 TS-CVA 的训练与后续统计分析链路是闭环的。
