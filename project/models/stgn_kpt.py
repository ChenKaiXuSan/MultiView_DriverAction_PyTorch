#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Spatio-temporal graph network backbone for 3D keypoints.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class STGNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = (temporal_kernel // 2, 0)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.tcn = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel, 1),
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("bctk,kv->bctv", x, adj)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.drop(x)


class STGCNKeypoint(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()

        model_cfg = hparams.model
        self.model_class_num = int(model_cfg.model_class_num)
        hidden_dim = int(getattr(model_cfg, "stgn_hidden_dim", 64))
        num_layers = int(getattr(model_cfg, "stgn_layers", 3))
        temporal_kernel = int(getattr(model_cfg, "stgn_temporal_kernel", 3))
        dropout = float(getattr(model_cfg, "stgn_dropout", 0.1))
        self.feature_dim = hidden_dim

        num_kpts = getattr(model_cfg, "stgn_num_kpts", None)
        self.register_buffer("adj", self._build_adj(num_kpts))

        layers = []
        in_channels = 3
        for _ in range(num_layers):
            layers.append(
                STGNBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    temporal_kernel=temporal_kernel,
                    dropout=dropout,
                )
            )
            in_channels = hidden_dim
        self.blocks = nn.ModuleList(layers)
        self.classifier = nn.Linear(self.feature_dim, self.model_class_num)

    @staticmethod
    def _build_adj(num_kpts: Optional[int]) -> Optional[torch.Tensor]:
        if num_kpts is None:
            return None
        return torch.eye(int(num_kpts), dtype=torch.float32)

    @staticmethod
    def _to_bctk(kpts: torch.Tensor) -> torch.Tensor:
        if kpts.dim() != 4:
            raise ValueError(f"Expected kpts with shape (B,T,K,3), got {kpts.shape}")
        return kpts.permute(0, 3, 1, 2).contiguous()

    def _get_adj(self, kpts: torch.Tensor) -> torch.Tensor:
        if self.adj is None:
            num_kpts = kpts.size(2)
            return torch.eye(num_kpts, device=kpts.device, dtype=kpts.dtype)
        return self.adj.to(device=kpts.device, dtype=kpts.dtype)

    def forward_features(self, kpts: torch.Tensor) -> torch.Tensor:
        x = self._to_bctk(kpts.float())
        adj = self._get_adj(kpts)
        for block in self.blocks:
            x = block(x, adj)
        return x.mean(dim=(2, 3))

    def forward(self, kpts: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(kpts)
        return self.classifier(features)


STGNKeypoint = STGCNKeypoint
