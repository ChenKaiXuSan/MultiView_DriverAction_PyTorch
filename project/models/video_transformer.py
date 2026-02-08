#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Lightweight video transformer backbone with configurable input/output dimensions.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal dimension."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VideoTransformer(nn.Module):
    """
    Video Transformer with configurable dimensions.
    
    Supports:
    - Configurable input channels
    - Configurable embedding dimension and output dimension
    - Positional encoding
    - Multiple pooling strategies (spatial and temporal)
    - CLS token for classification
    """
    
    def __init__(self, hparams) -> None:
        super().__init__()
        model_cfg = hparams.model
        self.model_class_num = int(model_cfg.model_class_num)
        
        # Input channels - can be modified (e.g., 3 for RGB, 4 for RGBD, etc.)
        self.input_channels = int(getattr(model_cfg, "input_channels", 3))
        
        # Transformer configuration
        embed_dim = int(getattr(model_cfg, "transformer_dim", 256))
        num_layers = int(getattr(model_cfg, "transformer_layers", 4))
        num_heads = int(getattr(model_cfg, "transformer_heads", 4))
        ff_dim = int(getattr(model_cfg, "transformer_ff_dim", embed_dim * 4))
        dropout = float(getattr(model_cfg, "transformer_dropout", 0.1))
        
        # Positional encoding
        self.use_pos_encoding = bool(getattr(model_cfg, "use_pos_encoding", True))
        
        # Pooling strategies
        self.spatial_pool = getattr(model_cfg, "spatial_pool", "mean")  # "mean" or "max"
        self.temporal_pool = getattr(model_cfg, "temporal_pool", "mean")  # "mean", "max", or "cls"
        
        self.feature_dim = embed_dim
        
        # Stem: 3D convolution to extract initial spatial-temporal features
        # Two-stage design: gradually reduce spatial size and increase channels
        self.stem = nn.Sequential(
            nn.Conv3d(self.input_channels, embed_dim // 2, kernel_size=(3, 7, 7), 
                     stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=(1, 3, 3),
                     stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
        # Positional encoding for temporal dimension
        if self.use_pos_encoding:
            self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        
        # CLS token for classification (if using cls pooling)
        if self.temporal_pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classifier head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.model_class_num)
        )
        
        # Gradient checkpointing flag
        self.use_gradient_checkpointing = bool(
            getattr(model_cfg, "use_gradient_checkpointing", False)
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
        # Stem: extract spatial-temporal features
        x = self.stem(video)  # [B, embed_dim, T, H', W']
        
        # Spatial pooling to reduce spatial dimensions
        if self.spatial_pool == "mean":
            x = x.mean(dim=(3, 4))  # [B, embed_dim, T]
        elif self.spatial_pool == "max":
            x = x.amax(dim=(3, 4))  # [B, embed_dim, T]
        else:
            raise ValueError(f"Unknown spatial_pool: {self.spatial_pool}")
        
        # Reshape for transformer: [B, T, embed_dim]
        x = x.permute(0, 2, 1)
        
        # Add CLS token if using cls pooling
        if self.temporal_pool == "cls":
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, embed_dim]
        
        # Add positional encoding to capture temporal order
        if self.use_pos_encoding:
            x = self.pos_encoder(x)
        
        # Transformer encoding for temporal modeling
        # Use gradient checkpointing if enabled (saves memory at cost of compute)
        # Note: use_reentrant=False is the new recommended way in PyTorch 2.0+
        # It provides better compatibility with autograd and is more stable.
        # Set to False to use the more robust non-reentrant implementation.
        if self.use_gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self.encoder, x, use_reentrant=False)
        else:
            x = self.encoder(x)
        x = self.norm(x)
        
        # Temporal pooling to get final feature vector
        if self.temporal_pool == "mean":
            features = x.mean(dim=1)  # [B, embed_dim]
        elif self.temporal_pool == "max":
            features = x.amax(dim=1)  # [B, embed_dim]
        elif self.temporal_pool == "cls":
            features = x[:, 0]  # [B, embed_dim] - take CLS token
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
