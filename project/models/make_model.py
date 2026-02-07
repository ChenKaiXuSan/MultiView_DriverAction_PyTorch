#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/models/make_model.py
Project: /workspace/skeleton/project/models
Created Date: Thursday October 19th 2023
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday April 19th 2025 7:58:58 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2023 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

26-11-2024	Kaixu Chen	remove x3d network.
"""

# ! prepare not used
import torch.nn as nn

from project.models.se_attn_res_3dcnn import SEFusionRes3DCNN
from project.models.res_3dcnn import Res3DCNN
from project.models.rgb_kpt_fusion import RGBKeypointFusion
from project.models.stgcn_kpt import STGCNKeypoint
from project.models.video_transformer import VideoTransformer
from project.models.video_mamba import VideoMamba


def select_model(hparams) -> nn.Module:
    """
    Select the model based on the hparams.

    Args:
        hparams: the hyperparameters of the model.

    Returns:
        nn.Module: the selected model.
    """

    model_backbone = hparams.model.backbone
    fuse_method = hparams.model.fuse_method
    input_type = getattr(hparams.model, "input_type", "rgb")

    if input_type == "kpt":
        return STGCNKeypoint(hparams)
    if input_type == "rgb_kpt":
        return RGBKeypointFusion(hparams)

    if model_backbone == "rgb_kpt":
        return RGBKeypointFusion(hparams)

    if model_backbone == "3dcnn":
        model = Res3DCNN(hparams)
    elif model_backbone == "stgcn":
        model = STGCNKeypoint(hparams)
    elif model_backbone == "transformer":
        model = VideoTransformer(hparams)
    elif model_backbone == "mamba":
        model = VideoMamba(hparams)
    else:
        raise ValueError(f"Unknown model backbone: {model_backbone}")

    return model
