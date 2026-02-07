#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Lightweight video transformer backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VideoTransformer(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        model_cfg = hparams.model
        self.model_class_num = int(model_cfg.model_class_num)
        embed_dim = int(getattr(model_cfg, "transformer_dim", 256))
        num_layers = int(getattr(model_cfg, "transformer_layers", 4))
        num_heads = int(getattr(model_cfg, "transformer_heads", 4))
        ff_dim = int(getattr(model_cfg, "transformer_ff_dim", embed_dim * 4))
        dropout = float(getattr(model_cfg, "transformer_dropout", 0.1))

        self.feature_dim = embed_dim
        self.stem = nn.Conv3d(3, embed_dim, kernel_size=3, padding=1, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(self.feature_dim, self.model_class_num)

    def forward_features(self, video: torch.Tensor) -> torch.Tensor:
        x = self.stem(video)
        x = x.mean(dim=(3, 4))
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(video)
        return self.classifier(features)
