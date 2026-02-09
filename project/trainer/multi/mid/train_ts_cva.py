#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: train_ts_cva.py
Project: project/trainer/multi/mid
Created Date: 2026-02-09
Author: Kaixu Chen
-----
Comment:
Trainer for Temporal-Synchronous Cross-View Attention (TS-CVA) model.

This trainer implements the training and validation loop for TS-CVA,
including metrics logging and visualization support.

Have a good code time :)
-----
Copyright (c) 2026 The University of Tsukuba
-----
"""

from typing import Any, Dict, Optional
import logging

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

from project.models.ts_cva_model import TSCVAModel

logger = logging.getLogger(__name__)


class TSCVATrainer(LightningModule):
    """
    Trainer for Temporal-Synchronous Cross-View Attention (TS-CVA) model.
    
    Expected batch format:
        batch["video"]["front"] : (B, C, T, H, W)
        batch["video"]["left"]  : (B, C, T, H, W)
        batch["video"]["right"] : (B, C, T, H, W)
        batch["label"]          : (B,)
    """
    
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        
        self.img_size = hparams.data.img_size
        self.lr = hparams.loss.lr
        self.num_classes = hparams.model.model_class_num
        
        # Initialize TS-CVA model
        self.model = TSCVAModel(hparams)
        
        # Metrics
        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)
        
        # Store attention and gate weights for visualization
        self.val_attention_weights = []
        self.val_gate_weights = []
        self.val_labels = []
        
    def forward(
        self, 
        videos: Dict[str, torch.Tensor],
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            videos: dict with keys 'front', 'left', 'right', each (B, C, T, H, W)
            return_attention: whether to return attention weights
            
        Returns:
            logits: (B, num_classes)
        """
        return self.model(videos, return_attention=return_attention)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: batch data containing videos and labels
            batch_idx: batch index
            
        Returns:
            loss: training loss
        """
        videos = batch["video"]
        labels = batch["label"].long()
        B = labels.size(0)
        
        # Forward pass
        logits = self.forward(videos, return_attention=False)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Compute metrics
        probs = torch.softmax(logits, dim=1)
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=B, prog_bar=True)
        self.log_dict(
            {
                "train/acc": self._accuracy(probs, labels),
                "train/precision": self._precision(probs, labels),
                "train/recall": self._recall(probs, labels),
                "train/f1": self._f1_score(probs, labels),
            },
            on_step=False,
            on_epoch=True,
            batch_size=B,
        )
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: batch data containing videos and labels
            batch_idx: batch index
            
        Returns:
            dict containing loss and predictions
        """
        videos = batch["video"]
        labels = batch["label"].long()
        B = labels.size(0)
        
        # Forward pass with attention weights
        logits = self.forward(videos, return_attention=True)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Compute metrics
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=B, prog_bar=True)
        self.log_dict(
            {
                "val/video_acc": self._accuracy(probs, labels),
                "val/precision": self._precision(probs, labels),
                "val/recall": self._recall(probs, labels),
                "val/f1": self._f1_score(probs, labels),
            },
            on_step=False,
            on_epoch=True,
            batch_size=B,
        )
        
        # Store attention and gate weights for visualization
        # Only store first few batches to avoid memory issues
        if batch_idx < 10:
            attn_weights = self.model.get_attention_weights()
            gate_weights = self.model.get_gate_weights()
            
            if attn_weights is not None:
                self.val_attention_weights.append(attn_weights.detach().cpu())
            if gate_weights is not None:
                self.val_gate_weights.append(gate_weights.detach().cpu())
            self.val_labels.append(labels.detach().cpu())
        
        return {
            "val_loss": loss,
            "val_preds": preds,
            "val_labels": labels,
        }
    
    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of validation epoch.
        Clear stored visualization data.
        """
        # Clear visualization data to avoid memory accumulation
        self.val_attention_weights = []
        self.val_gate_weights = []
        self.val_labels = []
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        
        Args:
            batch: batch data containing videos and labels
            batch_idx: batch index
            
        Returns:
            dict containing loss and predictions
        """
        videos = batch["video"]
        labels = batch["label"].long()
        B = labels.size(0)
        
        # Forward pass with attention weights
        logits = self.forward(videos, return_attention=True)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Compute metrics
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=B)
        self.log_dict(
            {
                "test/video_acc": self._accuracy(probs, labels),
                "test/precision": self._precision(probs, labels),
                "test/recall": self._recall(probs, labels),
                "test/f1": self._f1_score(probs, labels),
            },
            on_step=False,
            on_epoch=True,
            batch_size=B,
        )
        
        return {
            "test_loss": loss,
            "test_preds": preds,
            "test_labels": labels,
        }
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            optimizer configuration
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/video_acc",
                "interval": "epoch",
                "frequency": 1,
            },
        }
