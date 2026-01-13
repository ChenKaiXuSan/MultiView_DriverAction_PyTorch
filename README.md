# Third-Person Multi-View Driver Action Analysis

This repository presents a **third-person multi-view video framework for driver action analysis**, designed to study **system-level perception** under **driver visual field impairments**.

We observe drivers using **three external cameras (left, front, right)** and analyze driver actions from a **system perspective**, where the sensing system has access to richer visual information than the driver.

This project targets research in **IEEE Systems, Man, and Cybernetics (SMC)**, focusing on **human–system cooperative perception** rather than pure action classification.

---

## 🔍 Research Motivation

Drivers with **visual field impairments** may have limited perceptual access to their own actions and surrounding cues.  
In contrast, external sensing systems can observe the same driver from multiple viewpoints.

This work investigates the following question:

> **How can a third-person multi-view sensing system robustly analyze driver actions and compensate for human perceptual limitations?**

Key characteristics:
- The **visual limitation lies in the human**, not in the cameras.
- The system observes the driver from **multiple external viewpoints**.
- The goal is **system-level compensation**, not driver replacement.

---

## ✨ Key Features

- 🎥 **Third-person multi-view input**
  - Synchronized Left / Front / Right cameras
- 🚗 **Driver action analysis**
  - 8 predefined driver action classes
- 🧠 **Multi-view fusion strategies**
  - Single-view baselines
  - Multi-view average fusion
  - Multi-view concatenation + MLP
  - *(Optional)* attention-based view weighting
- 📊 **View contribution analysis**
  - Leave-One-View-Out (LOVO)
  - Pairwise view complementarity
- 🧪 **Robustness evaluation**
  - View-drop experiments simulating partial sensing failures

---

## 🗂 Dataset Format (Example)

data/
├── subject_01/
│   ├── left/video.mp4
│   ├── front/video.mp4
│   ├── right/video.mp4
│   └── labels.csv

- All three views are temporally synchronized.
- Labels are provided at frame-level or clip-level.
- Experiments are split **by subject or recording session** to avoid data leakage.

---

## 🏗 System Overview

Left View  ┐
Front View ├──► Shared Video Backbone ──► View Fusion ──► Action Prediction
Right View ┘

The backbone network is shared across views to ensure fair comparison.  
View fusion is performed at the feature level.

---

## 📊 Evaluation Protocol

### Metrics
- Accuracy
- Macro F1-score
- Per-class Recall
- Confusion Matrix

### View Contribution Analysis
- **Single-view performance** (L / F / R)
- **Leave-One-View-Out (LOVO)**:
  - Performance drop when removing one view
- **Pairwise view fusion**:
  - (L+F), (F+R), (L+R)

### Robustness Tests
- Random view removal at inference time
- Performance degradation under partial sensing

These evaluations quantify **independent view quality**, **marginal contribution**, and **view complementarity**.

---

## 🧩 Driver Action Classes (Example)

1. Looking forward
2. Checking left mirror
3. Checking right mirror
4. Operating dashboard
5. Steering adjustment
6. Lane checking
7. Idle driving
8. Other driver actions

---

## 📄 Citation

If you use this code in academic research, please cite:

@inproceedings{Chen202XSMC,
title     = {Third-Person Multi-View Driver Action Analysis for Compensating Visual Field Impairments},
author    = {Chen, Kaixu},
booktitle = {IEEE International Conference on Systems, Man, and Cybernetics},
year      = {202X}
}
