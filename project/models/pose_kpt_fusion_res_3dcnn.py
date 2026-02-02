#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Fuse 3D RGB backbone features with 3D keypoint sequences.

Inputs:
    video: (B, 3, T, H, W)
    kpt:   (B, T, J, C)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from project.models.base_model import BaseModel


class PoseKptFusionRes3DCNN(BaseModel):
    def __init__(self, hparams) -> None:
        super().__init__(hparams)

        model_cfg = hparams.model
        self.num_classes = int(model_cfg.model_class_num)
        self.ckpt = getattr(model_cfg, "ckpt_path", "")
        self.kpt_hidden_dim = int(getattr(model_cfg, "kpt_hidden_dim", 128))
        self.kpt_dropout = float(getattr(model_cfg, "kpt_dropout", 0.1))

        self.model = self.init_resnet(self.num_classes, self.ckpt)

        self.kpt_in_dim = int(getattr(model_cfg, "kpt_in_dim", 3))
        self.kpt_mlp = nn.Sequential(
            nn.Linear(self.kpt_in_dim, self.kpt_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.kpt_dropout),
            nn.Linear(self.kpt_hidden_dim, self.kpt_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.kpt_head = nn.Linear(self.kpt_hidden_dim, self.num_classes)

    def forward(self, video: torch.Tensor, kpt: torch.Tensor) -> torch.Tensor:
        video_logits = self.model(video)

        if kpt.ndim != 4:
            raise ValueError(f"Expected kpt shape (B,T,J,C) but got {kpt.shape}")
        kpt = kpt.to(dtype=video.dtype, device=video.device)

        B, T, J, C = kpt.shape
        if C != self.kpt_in_dim:
            raise ValueError(f"kpt last dim {C} does not match kpt_in_dim={self.kpt_in_dim}")
        kpt_flat = kpt.reshape(B * T * J, C)
        kpt_feat = self.kpt_mlp(kpt_flat)
        kpt_feat = kpt_feat.view(B, T, J, -1).mean(dim=(1, 2))
        kpt_logits = self.kpt_head(kpt_feat)

        return (video_logits + kpt_logits) / 2.0
