#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Spatio-temporal graph network backbone for 3D keypoints.
Enhanced with residual connections, flexible adjacency matrices, and multiple pooling strategies.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


def _get_config_value_with_legacy_fallback(
    model_cfg, primary: str, legacy: str, default: Any
) -> Any:
    """Return config value with primary -> legacy -> default fallback."""
    return getattr(model_cfg, primary, getattr(model_cfg, legacy, default))


class GraphConvolution(nn.Module):
    """Graph Convolution Layer with adjacency matrix."""
    
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T, K] - feature tensor
            adj: [K, K] - adjacency matrix
        
        Returns:
            [B, C_out, T, K] - output tensor
        """
        # Graph convolution: aggregate features from neighbors
        x = torch.einsum("bctk,kv->bctv", x, adj)
        x = torch.einsum("bctk,kc->bctc", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x


class TemporalConvolution(nn.Module):
    """Temporal 1D Convolution Layer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int = 3,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        padding = ((temporal_kernel - 1) * dilation) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(temporal_kernel, 1),
            padding=(padding, 0),
            dilation=(dilation, 1),
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class STGNBlock(nn.Module):
    """ST-GCN Block: Spatial (Graph) + Temporal Convolution with residual connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)
        
        # Spatial: Graph Convolution
        self.gcn = GraphConvolution(in_channels, out_channels, bias=False)
        
        # Temporal: 1D Convolution
        self.tcn = TemporalConvolution(
            out_channels,
            out_channels,
            temporal_kernel=temporal_kernel,
            dilation=dilation,
            bias=False,
        )
        
        # Normalization and activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, K] - input features
            adj: [K, K] - adjacency matrix
        
        Returns:
            [B, C_out, T, K] - output features
        """
        residual = x if self.use_residual else None
        
        # Graph convolution (spatial)
        x = self.gcn(x, adj)
        
        # Temporal convolution
        x = self.tcn(x)
        
        # Normalization and activation
        x = self.bn(x)
        x = self.relu(x)
        
        # Residual connection
        if residual is not None:
            x = x + residual
        
        x = self.drop(x)
        return x


class STGCNKeypoint(nn.Module):
    """
    ST-GCN for 3D Keypoint Action Recognition.
    
    Features:
    - Spatio-temporal graph convolution
    - Configurable adjacency matrix (identity, skeleton, or custom)
    - Residual connections for deep networks
    - Multiple temporal pooling strategies
    - Flexible channel dimensions
    - Support for variable number of keypoints
    """
    
    def __init__(self, hparams) -> None:
        super().__init__()

        model_cfg = hparams.model
        self.model_class_num = int(model_cfg.model_class_num)
        
        # Dimensions
        hidden_dim = int(
            _get_config_value_with_legacy_fallback(
                model_cfg, "stgcn_hidden_dim", "stgn_hidden_dim", 64
            )
        )
        num_layers = int(
            _get_config_value_with_legacy_fallback(
                model_cfg, "stgcn_layers", "stgn_layers", 3
            )
        )
        
        # Temporal configuration
        temporal_kernel = int(
            _get_config_value_with_legacy_fallback(
                model_cfg, "stgcn_temporal_kernel", "stgn_temporal_kernel", 3
            )
        )
        dropout = float(
            _get_config_value_with_legacy_fallback(
                model_cfg, "stgcn_dropout", "stgn_dropout", 0.1
            )
        )
        
        # Advanced options
        self.use_residual = bool(getattr(model_cfg, "stgcn_use_residual", True))
        self.temporal_pool = getattr(model_cfg, "stgcn_temporal_pool", "mean")  # mean, max, last
        self.adj_type = getattr(model_cfg, "stgcn_adj_type", "identity")  # identity, skeleton, custom
        
        self.feature_dim = hidden_dim
        
        # Number of keypoints
        num_kpts = _get_config_value_with_legacy_fallback(
            model_cfg, "stgcn_num_kpts", "stgn_num_kpts", None
        )
        self.num_kpts = num_kpts
        
        # Build and register adjacency matrix
        adj = self._build_adj(num_kpts)
        if adj is not None:
            self.register_buffer("adj", adj)
        else:
            self.adj = None

        # Build ST-GCN blocks
        layers = []
        in_channels = 3  # x, y, z coordinates
        for i in range(num_layers):
            layers.append(
                STGNBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    temporal_kernel=temporal_kernel,
                    dilation=1 + (i // 2),  # Increasing dilation for larger receptive field
                    dropout=dropout,
                    use_residual=self.use_residual,
                )
            )
            in_channels = hidden_dim
        
        self.blocks = nn.ModuleList(layers)
        
        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.model_class_num)
        )

    def _build_adj(self, num_kpts: Optional[int]) -> Optional[torch.Tensor]:
        """
        Build adjacency matrix for the skeleton graph.
        
        Args:
            num_kpts: Number of keypoints
        
        Returns:
            Adjacency matrix or None
        """
        if num_kpts is None:
            return None
        
        num_kpts = int(num_kpts)
        
        if self.adj_type == "identity":
            # Identity matrix: each node only connects to itself
            return torch.eye(num_kpts, dtype=torch.float32)
        
        elif self.adj_type == "skeleton":
            # Skeleton connectivity for common keypoint formats
            adj = self._get_skeleton_adjacency(num_kpts)
            return adj
        
        elif self.adj_type == "custom":
            # Custom adjacency matrix from config
            adj_matrix = getattr(self, "custom_adj_matrix", None)
            if adj_matrix is None:
                return torch.eye(num_kpts, dtype=torch.float32)
            return torch.tensor(adj_matrix, dtype=torch.float32)
        
        else:
            return torch.eye(num_kpts, dtype=torch.float32)

    @staticmethod
    def _get_skeleton_adjacency(num_kpts: int) -> torch.Tensor:
        """
        Get standard skeleton adjacency matrix.
        Supports:
        - 17 keypoints (COCO format): head, shoulders, elbows, wrists, hips, knees, ankles
        - 25 keypoints (OpenPose format)
        - 33 keypoints (MediaPipe format)
        """
        adj = torch.eye(num_kpts, dtype=torch.float32)
        
        # Standard skeleton edges (parent-child relationships)
        skeleton_edges = []
        
        if num_kpts == 17:  # COCO format
            skeleton_edges = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head and arms
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Body and upper limbs
                (5, 11), (6, 12), (11, 13), (13, 15),  # Lower body
                (12, 14), (14, 16),
            ]
        elif num_kpts == 25:  # OpenPose format
            skeleton_edges = [
                (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
                (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
                (1, 0), (0, 15), (15, 17), (0, 16), (16, 18), (2, 17), (5, 18),
            ]
        elif num_kpts == 33:  # MediaPipe format
            skeleton_edges = [
                (0, 1), (1, 2), (2, 3), (3, 7),
                (0, 4), (4, 5), (5, 6), (6, 8),
                (9, 10), (10, 11), (11, 12), (12, 14),
                (9, 13), (13, 15), (15, 17),
                (18, 19), (19, 20), (20, 21), (21, 22),
                (23, 24), (24, 25), (25, 26), (26, 27),
                (28, 29), (29, 30), (30, 31), (31, 32),
            ]
        
        # Add bidirectional edges
        for (i, j) in skeleton_edges:
            if i < num_kpts and j < num_kpts:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        
        # Normalize adjacency matrix
        # Add identity matrix for self-loops
        adj = adj + torch.eye(num_kpts)
        
        # Row-wise normalization: D^-1 * A
        degree = adj.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # Avoid division by zero
        adj = adj / degree
        
        return adj

    @staticmethod
    def _to_bctk(kpts: torch.Tensor) -> torch.Tensor:
        """
        Convert keypoint tensor from (B, T, K, 3) to (B, C, T, K).
        
        Args:
            kpts: Tensor of shape [B, T, K, 3]
        
        Returns:
            Tensor of shape [B, 3, T, K]
        """
        if kpts.dim() != 4:
            raise ValueError(f"Expected kpts with shape (B,T,K,3), got {kpts.shape}")
        return kpts.permute(0, 3, 1, 2).contiguous()

    def _get_adj(self, kpts: torch.Tensor) -> torch.Tensor:
        """Get adjacency matrix on the correct device."""
        if self.adj is None:
            num_kpts = kpts.size(2)
            return torch.eye(num_kpts, device=kpts.device, dtype=kpts.dtype)
        return self.adj.to(device=kpts.device, dtype=kpts.dtype)

    def forward_features(self, kpts: torch.Tensor) -> torch.Tensor:
        """
        Extract features from keypoints.
        
        Args:
            kpts: Tensor of shape [B, T, K, 3]
        
        Returns:
            features: Tensor of shape [B, hidden_dim]
        """
        # Convert to [B, 3, T, K] format
        x = self._to_bctk(kpts.float())
        
        # Get adjacency matrix
        adj = self._get_adj(kpts)
        
        # Process through ST-GCN blocks
        for block in self.blocks:
            x = block(x, adj)
        
        # Temporal and spatial pooling
        if self.temporal_pool == "mean":
            features = x.mean(dim=(2, 3))  # [B, C]
        elif self.temporal_pool == "max":
            features = x.amax(dim=(2, 3))  # [B, C]
        elif self.temporal_pool == "last":
            features = x[:, :, -1, :].mean(dim=2)  # [B, C]
        else:
            raise ValueError(f"Unknown temporal_pool: {self.temporal_pool}")
        
        # Final normalization
        features = self.final_norm(features)
        
        return features

    def forward(self, kpts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            kpts: Tensor of shape [B, T, K, 3]
        
        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        features = self.forward_features(kpts)
        return self.classifier(features)
