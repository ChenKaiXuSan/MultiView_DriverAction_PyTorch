#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: ts_cva_model.py
Project: project/models
Created Date: 2026-02-09
Author: Kaixu Chen
-----
Comment:
Temporal-Synchronous Cross-View Attention (TS-CVA) Model

This module implements TS-CVA for multi-view driver action recognition.
Key features:
1. Frame-synchronous cross-view attention at each timestep
2. Learnable gated view aggregation for dynamic view selection
3. View embeddings for view distinction
4. Temporal modeling with TCN

Have a good code time :)
-----
Copyright (c) 2026 The University of Tsukuba
-----
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from project.models.base_model import BaseModel


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention for cross-view interaction.
    
    For each timestep t, applies MHSA on the set of three view tokens
    to model complementary relationships between views.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, num_views, C) - view tokens at a single timestep
            
        Returns:
            output: (B, num_views, C) - attended view tokens
            attn_weights: (B, num_heads, num_views, num_views) - attention weights
        """
        B, N, C = x.shape  # N = num_views (typically 3)
        
        # Linear projections and reshape for multi-head
        Q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        K = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        V = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, N, N)
        attn_weights_dropped = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights_dropped, V)  # (B, H, N, D)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class LearnableGatedAggregation(nn.Module):
    """
    Learnable gated aggregation for dynamic view selection.
    
    Produces per-timestep weights for each view, allowing the model
    to adaptively weight views based on their reliability (e.g., 
    downweighting occluded views).
    """
    
    def __init__(self, embed_dim: int, num_views: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_views = num_views
        
        # MLP to produce gating scores
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, num_views, C) - view tokens at a single timestep
            
        Returns:
            fused: (B, C) - weighted aggregation of view tokens
            weights: (B, num_views) - gating weights (softmax normalized)
        """
        B, N, C = x.shape
        
        # Compute gating scores for each view
        scores = self.gate_mlp(x).squeeze(-1)  # (B, N)
        weights = F.softmax(scores, dim=-1)  # (B, N)
        
        # Weighted sum
        weights_expanded = weights.unsqueeze(-1)  # (B, N, 1)
        fused = (x * weights_expanded).sum(dim=1)  # (B, C)
        
        return fused, weights


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) for temporal modeling.
    
    Applies 1D convolutions on the temporal dimension to model
    temporal dependencies in the fused representation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size, 
                         padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) - temporal sequence of fused features
            
        Returns:
            output: (B, C) - temporally aggregated features
        """
        # Transpose for Conv1d: (B, C, T)
        x = x.transpose(1, 2)
        
        # Apply temporal convolutions
        x = self.conv_layers(x)  # (B, hidden_dim, T)
        
        # Global temporal pooling
        x = self.pool(x).squeeze(-1)  # (B, hidden_dim)
        
        return x


class TSCVAModel(BaseModel):
    """
    Temporal-Synchronous Cross-View Attention (TS-CVA) Model.
    
    Architecture:
    1. Per-view 3D CNN encoding -> temporal feature sequences
    2. Spatial pooling to get view tokens per timestep
    3. Cross-view attention at each timestep (MHSA)
    4. Gated view aggregation to produce fused tokens
    5. Temporal modeling (TCN) over fused sequence
    6. Classification head
    
    Args:
        hparams: Hydra configuration with model parameters
    """
    
    def __init__(self, hparams):
        super().__init__(hparams)
        
        self.num_classes = hparams.model.model_class_num
        self.num_views = 3  # front, left, right
        
        # Configuration options for ablation studies
        self.use_shared_backbone = getattr(hparams.model, 'ts_cva_shared_backbone', True)
        self.use_view_embedding = getattr(hparams.model, 'ts_cva_use_view_embedding', True)
        self.use_gated_aggregation = getattr(hparams.model, 'ts_cva_use_gated_aggregation', True)
        self.num_attention_heads = getattr(hparams.model, 'ts_cva_num_heads', 4)
        self.temporal_hidden_dim = getattr(hparams.model, 'ts_cva_temporal_dim', 512)
        self.temporal_layers = getattr(hparams.model, 'ts_cva_temporal_layers', 2)
        
        # Initialize 3D CNN backbones
        if self.use_shared_backbone:
            # Shared backbone across all views
            self.backbone = self.init_resnet(
                class_num=self.num_classes
            )
            self.feature_dim = self.backbone.blocks[-1].proj.in_features
        else:
            # Separate backbones for each view
            self.backbone_front = self.init_resnet(class_num=self.num_classes)
            self.backbone_left = self.init_resnet(class_num=self.num_classes)
            self.backbone_right = self.init_resnet(class_num=self.num_classes)
            self.feature_dim = self.backbone_front.blocks[-1].proj.in_features
        
        # View embeddings for view distinction
        if self.use_view_embedding:
            self.view_embeddings = nn.Parameter(
                torch.randn(self.num_views, self.feature_dim) * 0.02
            )
        
        # Cross-view attention module
        self.cross_view_attn = MultiHeadSelfAttention(
            embed_dim=self.feature_dim,
            num_heads=self.num_attention_heads,
            dropout=0.1
        )
        
        # View aggregation module
        if self.use_gated_aggregation:
            self.view_aggregation = LearnableGatedAggregation(
                embed_dim=self.feature_dim,
                num_views=self.num_views
            )
        
        # Temporal modeling
        self.temporal_model = TemporalConvNet(
            input_dim=self.feature_dim,
            hidden_dim=self.temporal_hidden_dim,
            num_layers=self.temporal_layers,
            kernel_size=3,
            dropout=0.1
        )
        
        # Classification head
        self.classifier = nn.Linear(self.temporal_hidden_dim, self.num_classes)
        
        # Store attention weights and gate weights for visualization
        self.attention_weights = None
        self.gate_weights = None
        
    def extract_view_features(self, video: torch.Tensor, view_idx: int) -> torch.Tensor:
        """
        Extract features from a single view using 3D CNN backbone.
        
        Args:
            video: (B, C, T, H, W) - input video for one view
            view_idx: index of the view (0=front, 1=left, 2=right)
            
        Returns:
            features: (B, C', T', H', W') - 3D CNN features
        """
        if self.use_shared_backbone:
            backbone = self.backbone
        else:
            backbone = [self.backbone_front, self.backbone_left, self.backbone_right][view_idx]
        
        # Forward through all blocks except the final classification layer
        x = video
        for idx in range(len(backbone.blocks) - 1):
            x = backbone.blocks[idx](x)
        
        return x
    
    def forward(
        self, 
        videos: Dict[str, torch.Tensor],
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of TS-CVA model.
        
        Args:
            videos: dict with keys 'front', 'left', 'right', each (B, C, T, H, W)
            return_attention: whether to return attention weights for visualization
            
        Returns:
            logits: (B, num_classes) - classification logits
        """
        B = videos['front'].size(0)
        
        # Step 1: Extract per-view 3D CNN features
        view_features = []
        for idx, view_name in enumerate(['front', 'left', 'right']):
            feat = self.extract_view_features(videos[view_name], view_idx=idx)
            view_features.append(feat)
        
        # Step 2: Spatial pooling to get view tokens per timestep
        # F^v shape: (B, C, T', H', W')
        # After GAP: (B, C, T')
        view_tokens = []
        for feat in view_features:
            # Global average pooling over spatial dimensions
            token = F.adaptive_avg_pool3d(feat, (feat.size(2), 1, 1))  # (B, C, T', 1, 1)
            token = token.squeeze(-1).squeeze(-1)  # (B, C, T')
            token = token.transpose(1, 2)  # (B, T', C)
            view_tokens.append(token)
        
        # Stack view tokens: (B, T', num_views, C)
        view_tokens = torch.stack(view_tokens, dim=2)  # (B, T', 3, C)
        B, T_prime, num_views, C = view_tokens.shape
        
        # Add view embeddings if enabled
        if self.use_view_embedding:
            view_tokens = view_tokens + self.view_embeddings.unsqueeze(0).unsqueeze(0)
        
        # Step 3: Cross-view attention at each timestep
        # Reshape to process all timesteps in parallel: (B*T', num_views, C)
        view_tokens_flat = view_tokens.view(B * T_prime, num_views, C)
        
        attended_tokens, attn_weights = self.cross_view_attn(view_tokens_flat)
        # attended_tokens: (B*T', num_views, C)
        # attn_weights: (B*T', num_heads, num_views, num_views)
        
        # Reshape back: (B, T', num_views, C)
        attended_tokens = attended_tokens.view(B, T_prime, num_views, C)
        
        # Store attention weights for visualization
        if return_attention:
            self.attention_weights = attn_weights.view(B, T_prime, self.num_attention_heads, 
                                                       num_views, num_views)
        
        # Step 4: View aggregation
        fused_tokens = []
        gate_weights_list = []
        
        for t in range(T_prime):
            tokens_t = attended_tokens[:, t, :, :]  # (B, num_views, C)
            
            if self.use_gated_aggregation:
                fused_t, weights_t = self.view_aggregation(tokens_t)
                gate_weights_list.append(weights_t)
            else:
                # Simple mean pooling as baseline
                fused_t = tokens_t.mean(dim=1)  # (B, C)
                weights_t = torch.ones(B, num_views, device=tokens_t.device) / num_views
                gate_weights_list.append(weights_t)
            
            fused_tokens.append(fused_t)
        
        # Stack fused tokens: (B, T', C)
        fused_sequence = torch.stack(fused_tokens, dim=1)
        
        # Store gate weights for visualization
        if return_attention:
            self.gate_weights = torch.stack(gate_weights_list, dim=1)  # (B, T', num_views)
        
        # Step 5: Temporal modeling
        temporal_features = self.temporal_model(fused_sequence)  # (B, temporal_hidden_dim)
        
        # Step 6: Classification
        logits = self.classifier(temporal_features)  # (B, num_classes)
        
        return logits
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return stored attention weights for visualization."""
        return self.attention_weights
    
    def get_gate_weights(self) -> Optional[torch.Tensor]:
        """Return stored gate weights for visualization."""
        return self.gate_weights
