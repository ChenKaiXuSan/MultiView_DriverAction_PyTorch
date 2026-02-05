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
from project.models.base_model import BaseModel


logger = logging.getLogger(__name__)


class Res3DCNN(BaseModel):
    """
    make 3D CNN model from the PytorchVideo lib.

    """

    def __init__(self, hparams) -> None:
        super().__init__(hparams=hparams)

        self.model_class_num = hparams.model.model_class_num

        self.model = self.init_resnet(
            self.model_class_num,
        )
        self.feature_dim = self.model.blocks[-1].proj.in_features

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W)
            attn_map: (B, 1, T, H, W)

        Returns:
            torch.Tensor: (B, C, T, H, W)
        """

        output = self.model(video)

        return output

    def forward_features(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract pooled features before the classification head.

        Args:
            video: (B, C, T, H, W)

        Returns:
            torch.Tensor: (B, feature_dim)
        """
        x = video
        for idx in range(5):
            x = self.model.blocks[idx](x)

        head = self.model.blocks[5]
        if hasattr(head, "pool"):
            x = head.pool(x)
        else:
            x = x.mean(dim=(2, 3, 4), keepdim=True)

        x = x.view(x.size(0), -1)

        dropout = getattr(head, "dropout", None)
        if dropout is not None:
            x = dropout(x)

        return x
