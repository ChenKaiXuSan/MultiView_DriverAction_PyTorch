# Multi-View Driver Head Action Analysis

**Person Multi-View Sensing for Driver Action Understanding under Visual Field Impairment**

This repository provides a **third-person, multi-view video framework** that analyzes driver head/actions from **three external cameras (left/front/right)**.
Unlike typical in-cabin or driver-centric perception, we study **human–system cooperative perception**: the **driver is perceptually constrained**, while the **system has a richer multi-view observation**.

Our central goal is not merely higher classification accuracy, but to understand:

> **How can a multi-view sensing system infer driver actions robustly and provide compensation signals when the driver’s perceptual access is limited?**

This framing targets **SIGCHI** audiences interested in **assistive interfaces, cooperative perception, human-centered sensing, and reliability under human constraints**.

---

## Why SIGCHI? Human-Centered Motivation

Drivers with **visual field impairments** may struggle to perceive (or confirm) their own head movements and surrounding cues.
However, an external sensing system can observe the same behavior from multiple viewpoints and potentially provide:

- **Awareness support** (e.g., “you checked left mirror” confirmation)
- **Safety feedback** (e.g., missing checks / delayed checks)
- **Robust monitoring** under partial sensor failure

**Key premise:** the limitation lies in the human’s perception—not in camera sensing.

---

## Contributions

This project contributes:

1. **A third-person multi-view driver action analysis pipeline** for studying human–system cooperative perception under visual constraints.
2. **View contribution and complementarity analysis**, quantifying how each viewpoint helps (or fails) across actions.
3. **Robustness evaluation under view dropout**, approximating real-world sensing degradation.
4. _(Optional)_ A modular baseline suite that enables **fair comparison** of fusion strategies with a shared backbone.

---

## System Overview

**Left / Front / Right videos** are synchronized and processed with a **shared backbone** (weight-tied across views) for fairness.

```
Left  ┐
Front ├──► Shared Backbone (per-view) ─► Feature Fusion ─► Action Prediction
Right ┘
```

Fusion is performed at the **feature level**:

- Average fusion
- Concatenation + MLP
- Optional view-weighting (attention / reliability scoring)

---

## Key Features

### Multi-View Inputs

- 🎥 Synchronized **Left / Front / Right** third-person cameras

### Driver Action Understanding

- 🧠 8 predefined action classes (extensible)
- Frame-level or clip-level labeling supported

### Fusion & Analysis

- 🔀 Single-view baselines
- 🔀 Multi-view fusion strategies (avg / concat / weighting)
- 📊 View contribution analysis: **Single-view**, **LOVO**, **pairwise complementarity**
- 🧪 Robustness tests: **view drop at inference**

---

## 当前可做的对比实验（基于 config 选项）

### 1) 单视角输入（train.view=single）
- **RGB 单视角**：`model.input_type=rgb` + `model.backbone=3dcnn|transformer|mamba`

### 2) 三视角输入（train.view=multi）
- **RGB 三视角**：`model.input_type=rgb` + `model.backbone=3dcnn|transformer|mamba` + `model.fuse_method=late`

### 3) 多视角融合方式（late fusion）
- **logit/prob 融合**：`model.fusion_mode=logit_mean|prob_mean`
- **特征级融合**：`model.fusion_mode=feature_mean|feature_concat`

### 4) 多视角融合方式（early fusion）
- **加权/拼接融合**：`model.fuse_method=add|mul|concat|avg`

### 5) 多视角融合方式（mid fusion）
- **SE 注意力融合**：`model.fuse_method=se_attn`（旧配置可用 `se_atn`）

---

## Dataset Structure

```
data/
├── subject_01/
│   ├── left/video.mp4
│   ├── front/video.mp4
│   ├── right/video.mp4
│   └── labels.csv
```

- Views are **temporally synchronized**
- Experiments should be split **by subject/session** to avoid leakage

---

## Driver Action Classes (Example)

1. Looking forward
2. Checking left mirror
3. Checking right mirror
4. Operating dashboard
5. Steering adjustment
6. Lane checking
7. Idle driving
8. Other driver actions

---

## Evaluation

### Metrics (for performance + human-centered reliability)

- Accuracy
- Macro F1
- Per-class recall
- Confusion matrix

### View Contribution

- **Single-view**: L / F / R
- **Leave-One-View-Out (LOVO)**: quantify marginal utility
- **Pairwise fusion**: (L+F), (F+R), (L+R)

### Robustness (system reliability)

- Random view removal at inference
- Performance degradation curves under partial sensing

These analyses measure:

- **Independent view quality**
- **Marginal contribution**
- **Complementarity**
- **Graceful degradation**

---

## Reproducibility

- Deterministic seed setting
- Subject/session split config
- Modular fusion components
- Logging of confusion matrices and per-class recall

> SIGCHI 会很看重：你能不能把实验跑得出来、能不能解释系统什么时候会失败。

---

## Ethics & Intended Use

This repository is intended for **research on assistive sensing** and **human–system cooperative perception**, not for surveillance or punitive monitoring.
If you plan to deploy in real settings, consider:

- consent + transparency
- data minimization
- privacy protection
- bias across impairment types and driving contexts

---

## Citation

```bibtex
@inproceedings{Chen202XCHI,
  title     = {Third-Person Multi-View Driver Action Analysis for Cooperative Perception under Visual Field Impairment},
  author    = {Chen, Kaixu},
  booktitle = {CHI Conference on Human Factors in Computing Systems},
  year      = {202X}
}
```
