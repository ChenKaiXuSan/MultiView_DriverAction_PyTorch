#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Mamba-style temporal backbone using GRU with configurable input/output dimensions.
Optimized for efficient temporal modeling of video sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """3D Convolutional block with batch normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TemporalMambaBlock(nn.Module):
    """
    Temporal Mamba-style block using GRU with gating mechanism.
    Supports bidirectional processing and residual connections.
    """
    
    def __init__(self, hidden_size: int, num_layers: int, dropout: float,
                 bidirectional: bool = False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Gating mechanism for adaptive temporal modeling
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.gate = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Projection layer to match dimensions if bidirectional
        if bidirectional:
            self.proj = nn.Linear(gru_output_size, hidden_size, bias=False)
        else:
            self.proj = nn.Identity()
        
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, T, hidden_size]
            residual: Optional residual connection
        
        Returns:
            output: Tensor of shape [B, T, hidden_size]
        """
        # GRU temporal encoding
        gru_out, _ = self.gru(x)
        
        # Project to hidden_size if bidirectional
        gru_out = self.proj(gru_out)
        
        # Apply gating for adaptive modeling
        gate = self.gate(gru_out)
        out = gru_out * gate
        
        # Add residual connection and normalize
        if residual is not None:
            out = out + residual
        out = self.norm(out)
        
        return out


class VideoMamba(nn.Module):
    """
    Video Mamba model for temporal action recognition.
    
    Features:
    - Configurable input channels and embedding dimensions
    - Multi-layer temporal modeling with gating
    - Flexible pooling strategies (mean, max, last, weighted)
    - Optional bidirectional temporal processing
    - Skip connections and layer normalization
    """
    
    def __init__(self, hparams) -> None:
        super().__init__()
        model_cfg = hparams.model
        self.model_class_num = int(model_cfg.model_class_num)
        
        # Input configuration
        self.input_channels = int(getattr(model_cfg, "input_channels", 3))
        
        # Mamba configuration
        embed_dim = int(getattr(model_cfg, "mamba_dim", 256))
        num_layers = int(getattr(model_cfg, "mamba_layers", 2))
        dropout = float(getattr(model_cfg, "mamba_dropout", 0.1))
        
        # Advanced features
        self.bidirectional = bool(getattr(model_cfg, "mamba_bidirectional", False))
        self.temporal_pool = getattr(model_cfg, "mamba_temporal_pool", "last")  # last, mean, max, weighted
        self.use_residual = bool(getattr(model_cfg, "mamba_use_residual", True))
        self.spatial_pool = getattr(model_cfg, "spatial_pool", "mean")  # mean or max

        self.feature_dim = embed_dim
        
        # Stem: Multi-stage 3D convolution for feature extraction
        # Gradually reduce spatial dimensions and increase channel depth
        self.stem = nn.Sequential(
            ConvBlock3D(self.input_channels, embed_dim // 2, kernel_size=3, 
                       stride=(1, 2, 2), padding=(1, 1, 1)),
            ConvBlock3D(embed_dim // 2, embed_dim, kernel_size=3,
                       stride=(1, 2, 2), padding=(1, 1, 1)),
        )
        
        # Temporal Mamba block for sequence modeling
        self.mamba_block = TemporalMambaBlock(
            hidden_size=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=self.bidirectional
        )
        
        # Optional learnable weights for weighted temporal pooling
        if self.temporal_pool == "weighted":
            self.temporal_weights = nn.Parameter(torch.ones(1, 1, 1) / 1.0)
        
        # Pre-final normalization
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Classifier head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.model_class_num)
        )

    def forward_features(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video input.
        
        Args:
            video: Tensor of shape [B, C, T, H, W]
                B: batch size
                C: input channels (configurable)
                T: temporal frames
                H, W: spatial height and width
        
        Returns:
            features: Tensor of shape [B, embed_dim]
        """
        # Spatial feature extraction with 3D convolutions
        x = self.stem(video)  # [B, embed_dim, T, H', W']
        
        # Spatial pooling to reduce spatial dimensions
        if self.spatial_pool == "mean":
            x = x.mean(dim=(3, 4))  # [B, embed_dim, T]
        elif self.spatial_pool == "max":
            x = x.amax(dim=(3, 4))  # [B, embed_dim, T]
        else:
            raise ValueError(f"Unknown spatial_pool: {self.spatial_pool}")
        
        # Transpose for temporal processing: [B, T, embed_dim]
        x = x.permute(0, 2, 1)
        
        # Temporal modeling with Mamba block
        if self.use_residual:
            temporal_out = self.mamba_block(x, residual=x)
        else:
            temporal_out = self.mamba_block(x)
        
        # Apply final normalization
        temporal_out = self.final_norm(temporal_out)
        
        # Temporal pooling to obtain final features
        if self.temporal_pool == "last":
            # Take the last frame output
            features = temporal_out[:, -1]  # [B, embed_dim]
        elif self.temporal_pool == "mean":
            # Average pooling across time
            features = temporal_out.mean(dim=1)  # [B, embed_dim]
        elif self.temporal_pool == "max":
            # Max pooling across time
            features = temporal_out.amax(dim=1)  # [B, embed_dim]
        elif self.temporal_pool == "weighted":
            # Learnable weighted pooling
            weights = torch.softmax(self.temporal_weights.expand_as(temporal_out), dim=1)
            features = (temporal_out * weights).sum(dim=1)  # [B, embed_dim]
        else:
            raise ValueError(f"Unknown temporal_pool: {self.temporal_pool}")
        
        return features

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            video: Tensor of shape [B, C, T, H, W]
        
        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        features = self.forward_features(video)
        return self.classifier(features)
