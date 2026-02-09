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

import requests
from pathlib import Path

import torch
import torch.nn as nn
from pytorchvideo.models.hub.resnet import slow_r50
import requests, shutil, tempfile

root_dir = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo"
checkpoint_paths = {
    "slow_r50": f"{root_dir}/kinetics/SLOW_8x8_R50.pyth",
    "slow_r50_detection": f"{root_dir}/ava/SLOW_4x16_R50_DETECTION.pyth",
    "c2d_r50": f"{root_dir}/kinetics/C2D_8x8_R50.pyth",
    "i3d_r50": f"{root_dir}/kinetics/I3D_8x8_R50.pyth",
}


# ---------------- 辅助函数 ---------------- #
def has_internet(host="github.com", timeout=3) -> bool:
    import socket, errno

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

    with requests.get(url, stream=True, timeout=10) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # 避免 keep-alive 空包
                    f.write(chunk)

    print(f"[INFO] Weights downloaded to {save_path.resolve()}")

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
    def init_resnet(
        class_num: int = 3, weight_path: str = "") -> nn.Module:
        """
        Args:
            class_num: 输出类别数
            weight_path: 预训练权重(.pth/.pyth)保存路径；若为空则跳过下载/加载
        """
        weight_path = Path(weight_path) if weight_path else None

        # 1) 初始化模型结构
        model = slow_r50(pretrained=False)

        # 2) 加载权重
        if weight_path:
            if weight_path.exists():
                print(f"[INFO] Loading local weights: {weight_path}")
                state = torch.load(weight_path, map_location="cpu")
                model_state = state["model_state"]
                model.load_state_dict(model_state)
            elif has_internet():
                print("[INFO] No local weights, downloading …")
                url = checkpoint_paths["slow_r50"]
                download_file(url, weight_path)
                state = torch.load(weight_path, map_location="cpu")
                model_state = state["model_state"]
                model.load_state_dict(model_state)
            else:
                print("[WARN] No internet and no local weights — model will be random.")
        else:
            print("[INFO] No weight_path specified — model will be random.")

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
    def init_resnet_separable(
        class_num: int = 3, weight_path: str = "", return_feature_dim: bool = False
    ):
        """
        Initialize ResNet 3D CNN with separable stem, body, and head.
        
        Args:
            class_num: 输出类别数
            weight_path: 预训练权重(.pth/.pyth)保存路径；若为空则跳过下载/加载
            return_feature_dim: 是否返回特征维度
            
        Returns:
            If return_feature_dim is False:
                (stem, body, head): 三个独立的模块
            If return_feature_dim is True:
                (stem, body, head, feature_dim): 三个独立模块和特征维度
        """
        weight_path = Path(weight_path) if weight_path else None

        # 1) 初始化完整模型结构
        model = slow_r50(pretrained=False)

        # 2) 加载权重
        if weight_path:
            if weight_path.exists():
                print(f"[INFO] Loading local weights: {weight_path}")
                state = torch.load(weight_path, map_location="cpu")
                model_state = state["model_state"]
                model.load_state_dict(model_state)
            elif has_internet():
                print("[INFO] No local weights, downloading …")
                url = checkpoint_paths["slow_r50"]
                download_file(url, weight_path)
                state = torch.load(weight_path, map_location="cpu")
                model_state = state["model_state"]
                model.load_state_dict(model_state)
            else:
                print("[WARN] No internet and no local weights — model will be random.")
        else:
            print("[INFO] No weight_path specified — model will be random.")

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
                if hasattr(self, 'dropout') and self.dropout is not None:
                    x = self.dropout(x)
                
                # Projection
                x = self.proj(x)
                return x
        
        # 修改分类层的输出维度
        new_proj = nn.Linear(feature_dim, class_num)
        head = ResNetHead(
            pool=getattr(head_block, 'pool', None),
            dropout=getattr(head_block, 'dropout', None),
            proj=new_proj
        )
        
        if return_feature_dim:
            return stem, body, head, feature_dim
        return stem, body, head
