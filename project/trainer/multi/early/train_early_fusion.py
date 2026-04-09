#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/train_late_fusion.py
Project: /workspace/skeleton/project
Created Date: Monday May 13th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday October 28th 2025 9:49:19 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

from project.models.make_model import select_model


class EarlyFusion3DCNNTrainer(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.img_size = hparams.data.img_size
        self.lr = hparams.optimizer.lr
        self.num_classes = hparams.model.model_class_num

        # define model
        self.stance_cnn = select_model(hparams)
        self.swing_cnn = select_model(hparams)

        # OOM guard configuration (preserves original behavior: trim at batch_size >= 15)
        self.batch_size_threshold = getattr(hparams.data, 'batch_size_threshold', 15)
        self.trim_size = getattr(hparams.data, 'trim_size', 14)

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

    def forward(self, x, swing_video=None):
        """
        Forward pass for early fusion.
        
        Args:
            x: Either a dict/batch with "video" key, or stance_video tensor directly
            swing_video: Optional swing_video tensor (used when x is stance_video)
            
        Returns:
            Fused predictions from both views
        """
        if swing_video is not None:
            # Called with two separate tensors
            stance_video = x
        elif isinstance(x, dict) and "video" in x:
            # Called with batch dict
            stance_video = x["video"][..., 0]
            swing_video = x["video"][..., 1]
        else:
            # Assume x contains both views in last dimension
            stance_video = x[..., 0]
            swing_video = x[..., 1]
        
        stance_preds = self.stance_cnn(stance_video)
        swing_preds = self.swing_cnn(swing_video)
        return (stance_preds + swing_preds) / 2

    def _shared_step(self, batch: torch.Tensor, stage: str):
        """Shared step for training, validation, and testing."""
        stance_video = batch["video"][..., 0].detach()  # b, c, t, h, w
        swing_video = batch["video"][..., 1].detach()  # b, c, t, h, w
        label = batch["label"]

        # OOM guard: trim batch if too large
        # Note: stance and swing videos have the same batch size
        if stance_video.size()[0] >= self.batch_size_threshold:
            stance_video = stance_video[:self.trim_size]
            swing_video = swing_video[:self.trim_size]
            label = label[:self.trim_size]

        stance_preds = self.stance_cnn(stance_video)
        swing_preds = self.swing_cnn(swing_video)

        # compute losses
        stance_loss = F.cross_entropy(stance_preds, label.long())
        swing_loss = F.cross_entropy(swing_preds, label.long())
        loss = (stance_loss + swing_loss) / 2

        # compute fused prediction
        predict = (stance_preds + swing_preds) / 2
        predict_softmax = torch.softmax(predict, dim=1)

        # log loss
        self.log(
            f"{stage}/loss", loss, on_epoch=True, on_step=True, batch_size=label.size()[0]
        )

        # compute and log metrics
        video_acc = self._accuracy(predict_softmax, label)
        video_precision = self._precision(predict_softmax, label)
        video_recall = self._recall(predict_softmax, label)
        video_f1_score = self._f1_score(predict_softmax, label)

        self.log_dict(
            {
                f"{stage}/video_acc": video_acc,
                f"{stage}/video_precision": video_precision,
                f"{stage}/video_recall": video_recall,
                f"{stage}/video_f1_score": video_f1_score,
            },
            on_epoch=True,
            on_step=True,
            batch_size=label.size()[0],
        )

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        """
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.estimated_stepping_batches,
                    verbose=True,
                ),
                "monitor": "train/loss",
            },
        }


class EarlyFusionTransformerTrainer(EarlyFusion3DCNNTrainer):
    """Early fusion trainer alias for transformer backbone routing."""


class EarlyFusionMambaTrainer(EarlyFusion3DCNNTrainer):
    """Early fusion trainer alias for mamba backbone routing."""


class EarlyFusionSTGCNTrainer(EarlyFusion3DCNNTrainer):
    """Early fusion trainer alias for ST-GCN backbone routing."""


class EarlyFusionRGBKeypointTrainer(EarlyFusion3DCNNTrainer):
    """Early fusion trainer alias for RGB+KPT fusion backbone."""
