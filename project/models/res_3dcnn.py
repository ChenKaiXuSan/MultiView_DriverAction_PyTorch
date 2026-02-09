#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/models/make_model copy.py
Project: /workspace/code/project/models
Created Date: Thursday May 8th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday May 8th 2025 1:23:28 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import torch
import torch.nn as nn
from project.models.base_model import BaseModel


logger = logging.getLogger(__name__)


class Res3DCNN(BaseModel):
    """
    make 3D CNN model from the PytorchVideo lib.
    
    Supports two modes:
    1. Standard mode: unified model with blocks
    2. Separable mode: stem, body, head as separate modules
    """

    def __init__(self, hparams, use_separable: bool = False) -> None:
        super().__init__(hparams=hparams)

        self.model_class_num = hparams.model.model_class_num
        self.use_separable = use_separable

        if self.use_separable:
            # 使用可分离的结构
            self.stem, self.body, self.head, self.feature_dim = self.init_resnet_separable(
                self.model_class_num,
                return_feature_dim=True
            )
            self.model = None  # 在可分离模式下不使用统一的 model
        else:
            # 使用传统的统一结构
            self.model = self.init_resnet(
                self.model_class_num,
            )
            self.feature_dim = self.model.blocks[-1].proj.in_features
            self.stem = None
            self.body = None
            self.head = None

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W)

        Returns:
            torch.Tensor: (B, num_classes)
        """
        if self.use_separable:
            x = self.stem(video)
            x = self.body(x)
            x = self.head(x)
            return x
        else:
            return self.model(video)

    def forward_features(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract pooled features before the classification head.

        Args:
            video: (B, C, T, H, W)

        Returns:
            torch.Tensor: (B, feature_dim)

        Note:
            In standard mode, assumes the final element in model.blocks is the classification head.
            In separable mode, returns features after stem and body.
        """
        if self.use_separable:
            x = self.stem(video)
            x = self.body(x)
            # Apply pooling similar to head but without projection
            if x.dim() == 5:
                x = x.mean(dim=(2, 3, 4))  # (B, C)
            else:
                x = x.view(x.size(0), -1)
            return x
        else:
            x = video
            for idx in range(len(self.model.blocks) - 1):
                x = self.model.blocks[idx](x)

            head = self.model.blocks[-1]
            if hasattr(head, "pool"):
                x = head.pool(x)
            else:
                if x.dim() != 5:
                    raise ValueError(f"Expected 5D features, got shape {x.shape}")
                x = x.mean(dim=(2, 3, 4), keepdim=True)

            x = x.view(x.size(0), -1)

            dropout = getattr(head, "dropout", None)
            if dropout is not None:
                x = dropout(x)

            return x
    
    def get_stem(self):
        """Get the stem module (initial layers)."""
        if self.use_separable:
            return self.stem
        else:
            return self.model.blocks[0]
    
    def get_body(self):
        """Get the body module (middle ResNet stages)."""
        if self.use_separable:
            return self.body
        else:
            return nn.Sequential(*self.model.blocks[1:-1])
    
    def get_head(self):
        """Get the head module (classification layer)."""
        if self.use_separable:
            return self.head
        else:
            return self.model.blocks[-1]
