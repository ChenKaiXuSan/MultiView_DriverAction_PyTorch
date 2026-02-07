#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Mamba-style temporal backbone using GRU as a lightweight baseline implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VideoMamba(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        model_cfg = hparams.model
        self.model_class_num = int(model_cfg.model_class_num)
        embed_dim = int(getattr(model_cfg, "mamba_dim", 256))
        num_layers = int(getattr(model_cfg, "mamba_layers", 2))
        dropout = float(getattr(model_cfg, "mamba_dropout", 0.1))

        self.feature_dim = embed_dim
        self.stem = nn.Conv3d(3, embed_dim, kernel_size=3, padding=1, bias=False)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(self.feature_dim, self.model_class_num)

    def forward_features(self, video: torch.Tensor) -> torch.Tensor:
        x = self.stem(video)
        # Pool spatial dimensions before temporal GRU processing.
        x = x.mean(dim=(3, 4))
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        feat = out[:, -1]
        return self.norm(feat)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(video)
        return self.classifier(features)
