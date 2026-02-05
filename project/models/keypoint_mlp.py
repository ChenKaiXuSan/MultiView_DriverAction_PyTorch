#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Simple keypoint backbone for (B, T, K, 3) inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointMLP(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()

        model_cfg = hparams.model
        self.model_class_num = int(model_cfg.model_class_num)
        hidden_dim = int(getattr(model_cfg, "kpt_hidden_dim", 256))
        self.feature_dim = int(getattr(model_cfg, "kpt_feature_dim", hidden_dim))
        dropout_p = float(getattr(model_cfg, "kpt_dropout", 0.1))

        self.fc1 = nn.LazyLinear(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.feature_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(self.feature_dim, self.model_class_num)

    @staticmethod
    def _flatten_kpts(kpts: torch.Tensor) -> torch.Tensor:
        if kpts.dim() == 3:
            kpts = kpts.unsqueeze(1)
        if kpts.dim() != 4:
            raise ValueError(f"Expected kpts with shape (B,T,K,3), got {kpts.shape}")
        bsz, timesteps, joints, coords = kpts.shape
        if coords != 3:
            raise ValueError(f"Expected kpts coord dim=3, got {coords}")
        return kpts.reshape(bsz, timesteps, joints * coords)

    def forward_features(self, kpts: torch.Tensor) -> torch.Tensor:
        kpts = kpts.float()
        x = self._flatten_kpts(kpts).mean(dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, kpts: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(kpts)
        return self.classifier(features)
