#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS-CVA: Temporal-Synchronous Cross-View Attention for 3-view RGB (front/left/right) with 3D CNN backbones.

Core idea:
1) Each view -> 3D CNN -> feature map [B,C,T',H',W']
2) Spatial GAP -> per-time token [B,T',C]
3) For each time t, build a 3-token set (front/left/right) and run self-attention
4) Gate (optional) to aggregate 3 view tokens -> fused token z_t
5) Temporal head (TCN/Transformer/GRU) -> classification

This file provides:
- TemporalSyncCrossViewAttn (vectorized, no python for-loop over time)
- SimpleTemporalHead (2-layer TCN)
- MultiViewTS_CVA_Classifier (full model wrapper)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ViewName = Literal["front", "left", "right"]


# ---------------------------
# 1) TS-CVA fusion block
# ---------------------------
class TemporalSyncCrossViewAttn(nn.Module):
    """
    Temporal-Synchronous Cross-View Attention (TS-CVA).

    Inputs:
        F_front, F_left, F_right: [B, C, T, H, W]  (T = T' after backbone)
    Output:
        Z: [B, T, C]  fused per-time token sequence

    Options:
        - view embedding: add learnable (front/left/right) embeddings to tokens
        - MHSA across views (3 tokens) for each time step
        - gated pooling across views (per time) to get fused token
    """

    def __init__(
        self,
        c: int,
        n_heads: int = 4,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_view_embed: bool = True,
        use_gate: bool = True,
        ln: bool = True,
        residual: bool = True,
        gate_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert c % n_heads == 0, f"embed dim c={c} must be divisible by n_heads={n_heads}"

        self.c = c
        self.use_view_embed = use_view_embed
        self.use_gate = use_gate
        self.residual = residual

        self.attn = nn.MultiheadAttention(
            embed_dim=c,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_out_drop = nn.Dropout(proj_dropout)

        self.ln1 = nn.LayerNorm(c) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(c) if ln else nn.Identity()

        # Lightweight FFN to stabilize (optional but usually helps)
        self.ffn = nn.Sequential(
            nn.Linear(c, 4 * c),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(4 * c, c),
            nn.Dropout(proj_dropout),
        )

        if use_view_embed:
            self.view_embed = nn.Parameter(torch.zeros(3, c))
            nn.init.trunc_normal_(self.view_embed, std=0.02)

        if use_gate:
            h = gate_hidden if gate_hidden is not None else max(64, c // 2)
            self.gate = nn.Sequential(
                nn.Linear(c, h),
                nn.ReLU(inplace=True),
                nn.Linear(h, 1),
            )

    @staticmethod
    def _to_time_tokens(Fv: torch.Tensor) -> torch.Tensor:
        """
        Convert feature map [B,C,T,H,W] -> per-time tokens [B,T,C]
        by spatial GAP.
        """
        # Adaptive pool only over H,W keeping T
        # -> [B,C,T,1,1] -> [B,C,T]
        Sv = F.adaptive_avg_pool3d(Fv, output_size=(Fv.shape[2], 1, 1)).squeeze(-1).squeeze(-1)
        # [B,C,T] -> [B,T,C]
        return Sv.transpose(1, 2).contiguous()

    def forward(
        self,
        F_front: torch.Tensor,
        F_left: torch.Tensor,
        F_right: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            F_*: [B,C,T,H,W]
            return_weights: if True, also return gating weights and attention weights.

        Returns:
            Z: [B,T,C]
            aux (optional): dict with
                - "gate_w": [B,T,3]
                - "attn_w": [B,T,3,3]   (averaged over heads)
        """
        Sf = self._to_time_tokens(F_front)  # [B,T,C]
        Sl = self._to_time_tokens(F_left)
        Sr = self._to_time_tokens(F_right)

        B, T, C = Sf.shape
        # Stack views -> [B,T,3,C]
        X = torch.stack([Sf, Sl, Sr], dim=2)

        # Add view embedding (broadcast over B,T)
        if self.use_view_embed:
            X = X + self.view_embed.view(1, 1, 3, C)

        # Vectorize: merge (B,T) -> BT; treat 3 views as sequence length
        X_bt = X.view(B * T, 3, C)  # [BT,3,C]

        # Pre-LN
        X_norm = self.ln1(X_bt)

        # Self-attention across views (length=3)
        # attn_w returned as [BT, 3, 3] when average_attn_weights=True (default)
        X_attn, attn_w = self.attn(X_norm, X_norm, X_norm, need_weights=return_weights)
        X_attn = self.attn_out_drop(X_attn)

        if self.residual:
            X_bt2 = X_bt + X_attn
        else:
            X_bt2 = X_attn

        # FFN
        X_bt2_norm = self.ln2(X_bt2)
        X_ffn = self.ffn(X_bt2_norm)

        if self.residual:
            X_bt3 = X_bt2 + X_ffn
        else:
            X_bt3 = X_ffn

        # Restore [B,T,3,C]
        X_out = X_bt3.view(B, T, 3, C)

        # Gate / pooling over views -> [B,T,C]
        aux: Optional[Dict[str, torch.Tensor]] = None
        if self.use_gate:
            # score: [B,T,3,1] -> [B,T,3]
            score = self.gate(X_out).squeeze(-1)
            gate_w = torch.softmax(score, dim=2)  # [B,T,3]
            Z = (X_out * gate_w.unsqueeze(-1)).sum(dim=2)  # [B,T,C]
        else:
            gate_w = None
            Z = X_out.mean(dim=2)

        if return_weights:
            aux = {}
            if gate_w is not None:
                aux["gate_w"] = gate_w.detach()
            if attn_w is not None:
                # attn_w: [BT,3,3] -> [B,T,3,3]
                aux["attn_w"] = attn_w.view(B, T, 3, 3).detach()

        return Z, aux


# ---------------------------
# 2) Simple temporal head (TCN)
# ---------------------------
class SimpleTCNHead(nn.Module):
    """
    A small temporal conv head:
        [B,T,C] -> [B,C,T] -> Conv1d -> Conv1d -> GlobalPool -> [B,C]
    """

    def __init__(self, c: int, hidden: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        h = hidden if hidden is not None else c
        self.net = nn.Sequential(
            nn.Conv1d(c, h, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(h, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: [B,T,C] -> [B,C,T]
        x = Z.transpose(1, 2).contiguous()
        x = self.net(x)
        # Global average over time -> [B,C]
        return x.mean(dim=2)


# ---------------------------
# 3) Full multi-view model wrapper
# ---------------------------
@dataclass
class TS_CVA_Config:
    num_classes: int
    feat_dim: int  # C
    n_heads: int = 4
    use_view_embed: bool = True
    use_gate: bool = True
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    temporal_dropout: float = 0.1


class MultiViewTS_CVA_Classifier(nn.Module):
    """
    Full model:
      3x (3D CNN backbone) -> feature maps -> TS-CVA -> temporal head -> classifier

    You can pass:
      - one shared backbone for all views, OR
      - three separate backbones (front/left/right)

    Expected backbone output: [B,C,T',H',W']
    """

    def __init__(
        self,
        cfg: TS_CVA_Config,
        backbone_shared: Optional[nn.Module] = None,
        backbone_by_view: Optional[Dict[ViewName, nn.Module]] = None,
        backbone_out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert (backbone_shared is not None) ^ (backbone_by_view is not None), \
            "Provide either backbone_shared OR backbone_by_view (exactly one)."

        self.cfg = cfg
        self.backbone_shared = backbone_shared
        self.backbone_by_view = backbone_by_view

        # If your backbone already outputs cfg.feat_dim, set backbone_out_channels=cfg.feat_dim
        # Otherwise you can add a 1x1x1 projection.
        c_in = backbone_out_channels if backbone_out_channels is not None else cfg.feat_dim
        self.proj = nn.Identity() if c_in == cfg.feat_dim else nn.Conv3d(c_in, cfg.feat_dim, kernel_size=1)

        self.fuser = TemporalSyncCrossViewAttn(
            c=cfg.feat_dim,
            n_heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout,
            proj_dropout=cfg.proj_dropout,
            use_view_embed=cfg.use_view_embed,
            use_gate=cfg.use_gate,
            ln=True,
            residual=True,
        )
        self.temporal = SimpleTCNHead(cfg.feat_dim, dropout=cfg.temporal_dropout)
        self.cls = nn.Linear(cfg.feat_dim, cfg.num_classes)

    def _encode_view(self, x: torch.Tensor, view: ViewName) -> torch.Tensor:
        # x: [B,3,T,H,W]
        if self.backbone_shared is not None:
            Fv = self.backbone_shared(x)
        else:
            assert self.backbone_by_view is not None
            Fv = self.backbone_by_view[view](x)

        # Expect Fv: [B,C,T',H',W']; project if needed
        Fv = self.proj(Fv)
        return Fv

    def forward(
        self,
        x_front: torch.Tensor,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        Ff = self._encode_view(x_front, "front")
        Fl = self._encode_view(x_left, "left")
        Fr = self._encode_view(x_right, "right")

        Z, aux = self.fuser(Ff, Fl, Fr, return_weights=return_aux)  # Z: [B,T',C]
        u = self.temporal(Z)  # [B,C]
        logits = self.cls(u)  # [B,K]
        return logits, aux


# ---------------------------
# 4) Minimal backbone example (toy)
#    Replace this with your Res3DCNN / torchvision video model outputting [B,C,T',H',W']
# ---------------------------
class Tiny3DBackbone(nn.Module):
    """A tiny 3D CNN that outputs [B,C,T',H',W'] for quick wiring tests."""
    def __init__(self, out_c: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, out_c, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,T,H,W] -> [B,C,T',H',W']
        return self.net(x)


# ---------------------------
# 5) Quick sanity check
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = TS_CVA_Config(num_classes=6, feat_dim=256, n_heads=4, use_view_embed=True, use_gate=True)

    backbone = Tiny3DBackbone(out_c=256)
    model = MultiViewTS_CVA_Classifier(
        cfg=cfg,
        backbone_shared=backbone,
        backbone_out_channels=256,
    )

    B, T, H, W = 2, 32, 224, 224
    x_f = torch.randn(B, 3, T, H, W)
    x_l = torch.randn(B, 3, T, H, W)
    x_r = torch.randn(B, 3, T, H, W)

    logits, aux = model(x_f, x_l, x_r, return_aux=True)
    print("logits:", logits.shape)  # [B,K]
    if aux is not None:
        print("gate_w:", aux["gate_w"].shape)  # [B,T',3]
        print("attn_w:", aux["attn_w"].shape)  # [B,T',3,3]