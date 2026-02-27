#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /home/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch/project/models/base.model.py
Project: /home/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch/project/models
Created Date: Thursday June 26th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday June 26th 2025 11:09:31 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import requests
from pathlib import Path

import torch
import torch.nn as nn
from pytorchvideo.models.hub.resnet import slow_r50
import requests, shutil, tempfile
import socket, errno


logger = logging.getLogger(__name__)

root_dir = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo"
checkpoint_paths = {
    "slow_r50": f"{root_dir}/kinetics/SLOW_8x8_R50.pyth",
    "slow_r50_detection": f"{root_dir}/ava/SLOW_4x16_R50_DETECTION.pyth",
    "c2d_r50": f"{root_dir}/kinetics/C2D_8x8_R50.pyth",
    "i3d_r50": f"{root_dir}/kinetics/I3D_8x8_R50.pyth",
}

# 默认权重缓存目录
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pytorchvideo"


# ---------------- 辅助函数 ---------------- #
def has_internet(host="dl.fbaipublicfiles.com", timeout=3) -> bool:
    """检查是否有网络连接"""

    try:
        socket.create_connection((host, 443), timeout=timeout)
        return True
    except OSError as e:
        if e.errno == errno.EHOSTUNREACH:
            return False
        return False


def download_file(url: str, save_path: Path):
    """Download file from URL and save directly to `save_path`."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading weights from %s...", url)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # 避免 keep-alive 空包
                    f.write(chunk)

    logger.info("Weights downloaded to %s", save_path.resolve())


def get_or_download_weights(model_name: str = "slow_r50") -> Path | None:
    """
    获取或下载预训练权重

    Args:
        model_name: 模型名称，用于查找对应的 checkpoint URL

    Returns:
        权重文件路径 (如果成功下载或已存在)，否则返回 None
    """
    # 1) 检查缓存目录中是否已有权重
    cache_path = DEFAULT_CACHE_DIR / f"{model_name}.pyth"

    if cache_path.exists():
        logger.info("Found cached weights at %s", cache_path)
        return cache_path

    # 2) 检查网络连接
    if not has_internet():
        logger.warning(
            "No internet connection and no cached weights — model will be randomly initialized."
        )
        return None

    # 3) 下载权重
    try:
        url = checkpoint_paths.get(model_name)
        if not url:
            logger.error("Unknown model name: %s", model_name)
            return None

        download_file(url, cache_path)
        return cache_path
    except Exception as e:
        logger.error("Failed to download weights: %s", e)
        logger.warning("Model will be randomly initialized.")
        return None


class BaseModel(nn.Module):
    """
    Base class for all models.
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = None

    def forward(self, video: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        raise NotImplementedError("Forward method not implemented.")

    def load_state_dict(self, state_dict):
        """
        Load the state dict into the model.
        """
        self.model.load_state_dict(state_dict)

    def save_model(self, path):
        """
        Save the model to the specified path.
        """
        torch.save(self.model.state_dict(), path)

    @staticmethod
    def init_resnet(class_num: int = 3) -> nn.Module:
        """
        初始化 ResNet 3D CNN

        Args:
            class_num: 输出类别数

        Returns:
            初始化好的模型（如果有网络连接，会自动下载并加载预训练权重）
        """
        # 1) 初始化模型结构
        model = slow_r50(pretrained=False)

        # 2) 获取或下载权重
        weight_path = get_or_download_weights("slow_r50")

        if weight_path and weight_path.exists():
            logger.info("Loading pretrained weights from %s", weight_path)
            try:
                state = torch.load(weight_path, map_location="cpu")
                model_state = state.get("model_state", state)
                model.load_state_dict(model_state)
                logger.info("Pretrained weights loaded successfully.")
            except Exception as e:
                logger.error("Failed to load weights: %s", e)
                logger.warning("Using randomly initialized model.")
        else:
            logger.info("Using randomly initialized model.")

        # 3) 修改首层和最后输出层
        model.blocks[0].conv = nn.Conv3d(
            3,
            model.blocks[0].conv.out_channels,
            kernel_size=(7, 7, 7),
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, class_num)
        return model

    @staticmethod
    def init_resnet_separable(class_num: int = 3, return_feature_dim: bool = False):
        """
        Initialize ResNet 3D CNN with separable stem, body, and head.

        Args:
            class_num: 输出类别数
            return_feature_dim: 是否返回特征维度

        Returns:
            If return_feature_dim is False:
                (stem, body, head): 三个独立的模块
            If return_feature_dim is True:
                (stem, body, head, feature_dim): 三个独立模块和特征维度
        """
        # 1) 初始化完整模型结构
        model = slow_r50(pretrained=False)

        # 2) 获取或下载权重
        weight_path = get_or_download_weights("slow_r50")

        if weight_path and weight_path.exists():
            logger.info("Loading pretrained weights from %s", weight_path)
            try:
                state = torch.load(weight_path, map_location="cpu")
                model_state = state.get("model_state", state)
                model.load_state_dict(model_state)
                logger.info("Pretrained weights loaded successfully.")
            except Exception as e:
                logger.error("Failed to load weights: %s", e)
                logger.warning("Using randomly initialized model.")
        else:
            logger.info("Using randomly initialized model.")

        # 3) 修改首层 (stem)
        model.blocks[0].conv = nn.Conv3d(
            3,
            model.blocks[0].conv.out_channels,
            kernel_size=(7, 7, 7),
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )

        # 4) 分离 stem, body, head
        # Stem: 第一个 block (conv1 + bn + relu + pool)
        stem = model.blocks[0]

        # Body: 中间的 ResNet stages (通常是 blocks[1:-1])
        body = nn.Sequential(*model.blocks[1:-1])

        # Head: 最后的分类层
        head_block = model.blocks[-1]
        feature_dim = head_block.proj.in_features

        # 重新创建 head，使其可以灵活使用
        class ResNetHead(nn.Module):
            def __init__(self, pool, dropout, proj):
                super().__init__()
                self.pool = pool
                self.dropout = dropout
                self.proj = proj

            def forward(self, x):
                # x: (B, C, T, H, W)
                # 始终使用自适应池化以确保输出为 (B, C, 1, 1, 1)
                x = nn.functional.adaptive_avg_pool3d(x, (1, 1, 1))

                # Flatten: (B, C, 1, 1, 1) -> (B, C)
                x = x.view(x.size(0), -1)

                # Dropout
                if hasattr(self, "dropout") and self.dropout is not None:
                    x = self.dropout(x)

                # Projection
                x = self.proj(x)
                return x

        # 修改分类层的输出维度
        new_proj = nn.Linear(feature_dim, class_num)
        head = ResNetHead(
            pool=getattr(head_block, "pool", None),
            dropout=getattr(head_block, "dropout", None),
            proj=new_proj,
        )

        if return_feature_dim:
            return stem, body, head, feature_dim
        return stem, body, head
