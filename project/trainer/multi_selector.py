#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Multi-view training selection helpers (early/mid/late fusion).

This module provides utilities to select and instantiate the appropriate
trainer class based on the fusion method and backbone configuration.

Fusion Strategy:
    - Early Fusion: Concatenate/fuse video features before encoding
    - Mid Fusion: Fuse features after per-view encoding (TS-CVA)
    - Late Fusion: Training separate models per view and fusing predictions

Examples:
    Early Fusion with 3D CNN:
        fuse_method: "add" | "mul" | "concat" | "avg"
        backbone: "3dcnn"
    
    Mid Fusion with TS-CVA:
        fuse_method: "ts_cva"
        backbone: "3dcnn"
    
    Late Fusion:
        fuse_method: "late"
        backbone: "3dcnn" | "transformer" | "mamba"
"""

import logging
from typing import Type

from project.trainer.multi.early.train_early_fusion import (
    EarlyFusion3DCNNTrainer,
)
from project.trainer.multi.late.train_late_fusion import (
    LateFusion3DCNNTrainer,
    LateFusionTransformerTrainer,
    LateFusionMambaTrainer,
)
from project.trainer.multi.mid.train_multi_ts_cva import MultiTSCVATrainer

logger = logging.getLogger(__name__)


# ============================================================================
# Fusion Method Definitions
# ============================================================================

EARLY_FUSION_METHODS = {"add", "mul", "concat", "avg"}
MID_FUSION_METHODS = {"mid"}
LATE_FUSION_METHODS = {"late"}


# ============================================================================
# Trainer Class Mappings
# ============================================================================

EARLY_FUSION_TRAINERS = {
    "3dcnn": EarlyFusion3DCNNTrainer,
}

MID_FUSION_TRAINERS = {
    "mid": MultiTSCVATrainer,
}

LATE_FUSION_TRAINERS = {
    "3dcnn": LateFusion3DCNNTrainer,
    "transformer": LateFusionTransformerTrainer,
    "mamba": LateFusionMambaTrainer,
}

# Comprehensive trainers mapping for quick lookup
ALL_TRAINERS = {
    **{(strategy, "early"): trainer 
       for strategy, trainer in EARLY_FUSION_TRAINERS.items()},
    **{(strategy, "mid"): trainer 
       for strategy, trainer in MID_FUSION_TRAINERS.items()},
    **{(strategy, "late"): trainer 
       for strategy, trainer in LATE_FUSION_TRAINERS.items()},
}


# ============================================================================
# Trainer Selection Functions
# ============================================================================

def _validate_hparams(hparams) -> tuple:
    """
    Validate and extract key hyperparameters.
    
    Args:
        hparams: Hydra configuration object
        
    Returns:
        Tuple of (view_type, input_type, fuse_method, backbone)
        
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    view_type = getattr(hparams.train, "view", None)
    input_type = getattr(hparams.model, "input_type", "rgb")
    fuse_method = getattr(hparams.model, "fuse_method", None)
    backbone = getattr(hparams.model, "backbone", None)
    
    # Validate view type
    if view_type != "multi":
        raise ValueError(
            f"Multi-view trainer requires train.view='multi', got '{view_type}'. "
            f"Please use single_selector.build_single_trainer() for single-view training."
        )
    
    # Validate input type
    if input_type != "rgb":
        raise ValueError(
            f"Multi-view trainer only supports model.input_type='rgb', got '{input_type}'."
        )
    
    # Validate fusion method
    if not fuse_method:
        raise ValueError(
            f"model.fuse_method is required. "
            f"Supported: {EARLY_FUSION_METHODS | MID_FUSION_METHODS | LATE_FUSION_METHODS}"
        )
    
    # Validate backbone
    if not backbone:
        raise ValueError(
            f"model.backbone is required. "
            f"Supported: 3dcnn, transformer, mamba"
        )
    
    return view_type, input_type, fuse_method, backbone


def _determine_fusion_strategy(fuse_method: str) -> str:
    """
    Determine which fusion strategy (early/mid/late) to use.
    
    Args:
        fuse_method: Fusion method name
        
    Returns:
        Strategy name: "early", "mid", or "late"
        
    Raises:
        ValueError: If fusion method is not recognized
    """
    if fuse_method in EARLY_FUSION_METHODS:
        return "early"
    elif fuse_method in MID_FUSION_METHODS:
        return "mid"
    elif fuse_method in LATE_FUSION_METHODS:
        return "late"
    else:
        raise ValueError(
            f"Unknown fuse_method: '{fuse_method}'. "
            f"Supported methods: {EARLY_FUSION_METHODS | MID_FUSION_METHODS | LATE_FUSION_METHODS}"
        )


def select_multi_trainer_cls(hparams) -> Type:
    """
    Select the appropriate trainer class based on configuration.
    
    This function implements a routing logic to select the correct trainer
    based on the fusion strategy (early/mid/late) and backbone architecture.
    
    Args:
        hparams: Hydra configuration containing:
            - train.view: Must be "multi"
            - model.input_type: Must be "rgb"
            - model.fuse_method: Fusion strategy (e.g., "ts_cva", "add", "late")
            - model.backbone: Model architecture (e.g., "3dcnn", "transformer")
    
    Returns:
        Trainer class corresponding to the configuration
        
    Raises:
        ValueError: If configuration is invalid or unsupported
    
    Examples:
        >>> from omegaconf import OmegaConf
        >>> config = OmegaConf.create({
        ...     "train": {"view": "multi"},
        ...     "model": {"input_type": "rgb", "fuse_method": "ts_cva", "backbone": "3dcnn"}
        ... })
        >>> trainer_cls = select_multi_trainer_cls(config)
        >>> # trainer_cls is MultiTSCVATrainer
    """
    # Validate hyperparameters
    view_type, input_type, fuse_method, backbone = _validate_hparams(hparams)
    
    # Determine fusion strategy
    strategy = _determine_fusion_strategy(fuse_method)
    
    # Get trainer class based on strategy
    if strategy == "early":
        trainer_cls = EARLY_FUSION_TRAINERS.get(backbone)
        if trainer_cls is None:
            supported = ", ".join(EARLY_FUSION_TRAINERS.keys())
            raise ValueError(
                f"Backbone '{backbone}' not supported for early fusion. "
                f"Supported backbones: {supported}"
            )
    
    elif strategy == "mid":
        # Mid-fusion only supports 3D CNN backbone
        if backbone != "3dcnn":
            raise ValueError(
                f"Mid-fusion (fuse_method='{fuse_method}') requires backbone='3dcnn', "
                f"got backbone='{backbone}'. "
                f"Mid-fusion methods only work with 3D CNN architecture."
            )
        trainer_cls = MID_FUSION_TRAINERS.get(fuse_method)
        if trainer_cls is None:
            supported = ", ".join(MID_FUSION_TRAINERS.keys())
            raise ValueError(
                f"Fusion method '{fuse_method}' not supported for mid-fusion. "
                f"Supported methods: {supported}"
            )
    
    elif strategy == "late":
        trainer_cls = LATE_FUSION_TRAINERS.get(backbone)
        if trainer_cls is None:
            supported = ", ".join(LATE_FUSION_TRAINERS.keys())
            raise ValueError(
                f"Backbone '{backbone}' not supported for late fusion. "
                f"Supported backbones: {supported}"
            )
    
    else:
        # Should not happen due to _determine_fusion_strategy
        raise RuntimeError(f"Unknown fusion strategy: {strategy}")
    
    # Log selection
    logger.info(
        f"Selected trainer: {trainer_cls.__name__} "
        f"(fusion={strategy}, method={fuse_method}, backbone={backbone})"
    )
    
    return trainer_cls


def build_multi_trainer(hparams):
    """
    Build a trainer instance for multi-view training.
    
    This is a convenience function that combines trainer class selection
    and instantiation in a single call.
    
    Args:
        hparams: Hydra configuration object
        
    Returns:
        Instantiated trainer object
        
    Raises:
        ValueError: If configuration is invalid
        
    Examples:
        >>> trainer = build_multi_trainer(config)
        >>> # Use trainer in training loop
    """
    trainer_cls = select_multi_trainer_cls(hparams)
    logger.info(f"Instantiating {trainer_cls.__name__}...")
    return trainer_cls(hparams)
