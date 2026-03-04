#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/helper.py
Project: /workspace/skeleton/project
Created Date: Tuesday May 14th 2024
Author: Kaixu Chen
-----
Comment:
This is a helper script to save the results of the training.
The saved items include:
1. the prediction and label for the further analysis.
2. the metrics for the model evaluation.
3. the confusion matrix for the model evaluation.

This script is executed at the end of each training in main.py file.

Have a good code time :)
-----
Last Modified: Tuesday May 14th 2024 3:23:52 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

04-12-2024	Kaixu Chen	refactor the code, add the save_inference method.

14-05-2024	Kaixu Chen	add save_CAM method, now it can save the CAM for the model evaluation.
"""

import logging
from pathlib import Path
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from project.map_config import label_mapping_Dict

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
    MulticlassAUROC,
)

logger = logging.getLogger(__name__)


def _get_class_axis_labels(num_class: int) -> list[str]:
    """Return stable class labels for outputs (metrics text and confusion matrix)."""
    if num_class == 4:
        return ["left", "right", "down", "up"]

    labels = [
        label_mapping_Dict[idx]
        for idx in sorted(label_mapping_Dict.keys())
        if idx < num_class
    ]
    if len(labels) == num_class:
        return labels

    return [str(i) for i in range(num_class)]


def load_class_order_from_metrics(save_root: str | Path) -> list[str] | None:
    """Load class order from metrics.txt if present.

    Args:
        save_root: training log root directory (the same root passed to save_helper).

    Returns:
        Parsed class order list if found, otherwise None.
    """
    metrics_path = Path(save_root) / "metrics.txt"
    if not metrics_path.exists():
        return None

    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        if line.startswith("class_order:"):
            value = line.split(":", 1)[1].strip()
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list) and all(
                    isinstance(item, str) for item in parsed
                ):
                    return parsed
            except Exception:
                logger.warning("Failed to parse class_order from metrics.txt: %s", value)
                return None

    return None


def load_saved_inference_with_class_order(
    save_root: str | Path,
    fold: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str] | None]:
    """Load saved prediction/label tensors together with class order.

    Args:
        save_root: training log root directory (the same root passed to save_helper).
        fold: fold name used in save_helper (e.g., "0", "fold0", etc.).

    Returns:
        (pred_tensor, label_tensor, class_order)
    """
    pred_path = Path(save_root) / "best_preds" / f"{fold}_pred.pt"
    label_path = Path(save_root) / "best_preds" / f"{fold}_label.pt"

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    pred_tensor = torch.load(pred_path, map_location="cpu")
    label_tensor = torch.load(label_path, map_location="cpu")
    class_order = load_class_order_from_metrics(save_root)

    return pred_tensor, label_tensor, class_order


def save_helper(
    all_pred: list[torch.Tensor],
    all_label: list[torch.Tensor],
    fold: str,
    save_path: str | Path,
    num_class: int,
):
    """save the inference results and metrics.

    Args:
        all_pred (list): predict result.
        all_label (list): label result.
        fold (str): fold number.
        save_path (str): save path.
        num_class (int): number of class.
    """

    # check device 
    pred_list = [pred.cpu() if pred.is_cuda else pred for pred in all_pred]
    label_list = [label.cpu() if label.is_cuda else label for label in all_label]

    all_pred_tensor = torch.cat(pred_list, dim=0)
    all_label_tensor = torch.cat(label_list, dim=0)

    save_inference(all_pred_tensor, all_label_tensor, fold, save_path)
    save_metrics(all_pred_tensor, all_label_tensor, fold, save_path, num_class)
    save_CM(all_pred_tensor, all_label_tensor, save_path, num_class, fold)


def save_inference(
    all_pred: torch.Tensor, all_label: torch.Tensor, fold: str, save_path: str | Path
):
    """save the inference results to .pt file.

    Args:
        all_pred (list): predict result.
        all_label (list): label result.
        fold (str): fold number.
        save_path (str): save path.
    """

    # save the results
    save_dir = Path(save_path) / "best_preds"

    if save_dir.exists() is False:
        save_dir.mkdir(parents=True)

    torch.save(
        all_pred,
        save_dir / f"{fold}_pred.pt",
    )
    torch.save(
        all_label,
        save_dir / f"{fold}_label.pt",
    )

    logger.info(f"save the pred and label into {save_dir} / {fold}")


def save_metrics(
    all_pred: torch.Tensor,
    all_label: torch.Tensor,
    fold: str,
    save_path: str | Path,
    num_class: int,
):
    """save the metrics to .txt file.

    Args:
        all_pred (list): all the predict result.
        all_label (list): all the label result.
        fold (str): the fold number.
        save_path (str): the path to save the metrics.
        num_class (int): number of class.
    """

    metrics_path = Path(save_path) / "metrics.txt"

    _accuracy = MulticlassAccuracy(num_class)
    _precision = MulticlassPrecision(num_class)
    _recall = MulticlassRecall(num_class)
    _f1_score = MulticlassF1Score(num_class)
    _auroc = MulticlassAUROC(num_class)
    _confusion_matrix = MulticlassConfusionMatrix(num_class, normalize="true")
    class_labels = _get_class_axis_labels(num_class)

    # For AUROC, use probabilities (all_pred)
    # For other metrics, use class indices (argmax of all_pred)
    pred_classes = torch.argmax(all_pred, dim=1) if all_pred.dim() > 1 else all_pred

    logger.info("*" * 100)
    logger.info("accuracy: %s" % _accuracy(pred_classes, all_label))
    logger.info("precision: %s" % _precision(pred_classes, all_label))
    logger.info("recall: %s" % _recall(pred_classes, all_label))
    logger.info("f1_score: %s" % _f1_score(pred_classes, all_label))
    logger.info("aurroc: %s" % _auroc(all_pred, all_label.long()))
    logger.info("confusion_matrix: %s" % _confusion_matrix(pred_classes, all_label))
    logger.info("#" * 100)

    with open(metrics_path, "a") as f:
        f.writelines(f"Fold {fold}\n")
        f.writelines(f"class_order: {class_labels}\n")
        f.writelines(f"accuracy: {_accuracy(pred_classes, all_label)}\n")
        f.writelines(f"precision: {_precision(pred_classes, all_label)}\n")
        f.writelines(f"recall: {_recall(pred_classes, all_label)}\n")
        f.writelines(f"f1_score: {_f1_score(pred_classes, all_label)}\n")
        f.writelines(f"aurroc: {_auroc(all_pred, all_label.long())}\n")
        f.writelines(f"confusion_matrix: {_confusion_matrix(pred_classes, all_label)}\n")
        f.writelines("#" * 100)
        f.writelines("\n")


def save_CM(
    all_pred: torch.Tensor,
    all_label: torch.Tensor,
    save_path: str | Path,
    num_class: int,
    fold: str,
):
    """save the confusion matrix to file.

    Args:
        all_pred (list): predict result.
        all_label (list): label result.
        save_path (Path): the path to save the confusion matrix.
        num_class (int): the number of class.
        fold (str): the fold number.
    """

    save_dir = Path(save_path) / "CM"

    if save_dir.exists() is False:
        save_dir.mkdir(parents=True)

    # Convert probabilities to class indices if needed
    pred_classes = torch.argmax(all_pred, dim=1) if all_pred.dim() > 1 else all_pred

    _confusion_matrix = MulticlassConfusionMatrix(num_class, normalize="true")

    logger.info("_confusion_matrix: %s" % _confusion_matrix(pred_classes, all_label))

    # set the font and title
    plt.rcParams.update({"font.size": 30, "font.family": "sans-serif"})

    confusion_matrix_data = _confusion_matrix(pred_classes, all_label).cpu().numpy() * 100

    axis_labels = _get_class_axis_labels(num_class)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix_data,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=axis_labels,
        yticklabels=axis_labels,
        vmin=0,
        vmax=100,
    )
    plt.title(f"Fold {fold} (%)", fontsize=30)
    plt.ylabel("Actual Label", fontsize=30)
    plt.xlabel("Predicted Label", fontsize=30)

    plt.savefig(
        save_dir / f"fold{fold}_confusion_matrix.png", dpi=300, bbox_inches="tight"
    )

    logger.info(
        f"save the confusion matrix into {save_dir}/fold{fold}_confusion_matrix.png"
    )

