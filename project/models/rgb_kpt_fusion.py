#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Fuse RGB and keypoint features for single-view inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from project.models.res_3dcnn import Res3DCNN
from project.models.keypoint_mlp import KeypointMLP


class RGBKeypointFusion(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.model_class_num = int(hparams.model.model_class_num)
        self.modality_fusion = getattr(hparams.model, "modality_fusion", "concat")
        self.fusion_feature_dim = int(
            getattr(hparams.model, "fusion_feature_dim", 0)
        )

        self.rgb_backbone = Res3DCNN(hparams)
        self.kpt_backbone = KeypointMLP(hparams)

        rgb_dim = int(getattr(self.rgb_backbone, "feature_dim", 0))
        kpt_dim = int(getattr(self.kpt_backbone, "feature_dim", 0))

        if self.modality_fusion == "concat":
            fused_dim = rgb_dim + kpt_dim
            self.rgb_proj = None
            self.kpt_proj = None
        elif self.modality_fusion == "mean":
            if self.fusion_feature_dim <= 0:
                self.fusion_feature_dim = rgb_dim
            fused_dim = self.fusion_feature_dim
            self.rgb_proj = nn.Linear(rgb_dim, fused_dim)
            self.kpt_proj = nn.Linear(kpt_dim, fused_dim)
        else:
            raise ValueError(f"Unknown modality_fusion: {self.modality_fusion}")

        self.feature_dim = fused_dim
        self.classifier = nn.Linear(self.feature_dim, self.model_class_num)

    def forward_features(self, video: torch.Tensor, kpts: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_backbone.forward_features(video)
        kpt_feat = self.kpt_backbone.forward_features(kpts)

        if self.modality_fusion == "concat":
            return torch.cat([rgb_feat, kpt_feat], dim=1)

        rgb_feat = self.rgb_proj(rgb_feat)
        kpt_feat = self.kpt_proj(kpt_feat)
        return (rgb_feat + kpt_feat) / 2.0

    def forward(self, video: torch.Tensor, kpts: torch.Tensor) -> torch.Tensor:
        fused = self.forward_features(video, kpts)
        return self.classifier(fused)
