#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Multi-view training selection helpers (early/mid/late fusion).
"""

import logging

from project.trainer.multi.early.train_early_fusion import (
    EarlyFusion3DCNNTrainer,
    EarlyFusionTransformerTrainer,
    EarlyFusionMambaTrainer,
)
from project.trainer.multi.late.train_late_fusion import (
    LateFusion3DCNNTrainer,
    LateFusionTransformerTrainer,
    LateFusionMambaTrainer,
)
from project.trainer.multi.mid.train_se_attn import SEAttnTrainer
from project.trainer.multi.mid.train_ts_cva import TSCVATrainer

logger = logging.getLogger(__name__)


EARLY_FUSION_METHODS = {"add", "mul", "concat", "avg"}
# Prefer "se_attn"; accept legacy "se_atn" spelling used in older configs (TODO: remove after migration).
LEGACY_FUSE_METHOD_ALIASES = {"se_atn": "se_attn"}
MID_FUSION_METHODS = {"se_attn", "ts_cva"}
LATE_FUSION_METHODS = {"late"}

EARLY_FUSION_TRAINERS = {
    "3dcnn": EarlyFusion3DCNNTrainer,
    "transformer": EarlyFusionTransformerTrainer,
    "mamba": EarlyFusionMambaTrainer,
}
LATE_FUSION_TRAINERS = {
    "3dcnn": LateFusion3DCNNTrainer,
    "transformer": LateFusionTransformerTrainer,
    "mamba": LateFusionMambaTrainer,
}
# Mid fusion only supports a single backbone (3dcnn), so it is keyed by fuse method.
MID_FUSION_TRAINERS = {"se_attn": SEAttnTrainer, "ts_cva": TSCVATrainer}


def select_multi_trainer_cls(hparams):
    """Select the trainer class for RGB multi-view fusion experiments."""
    if getattr(hparams.train, "view", None) != "multi":
        raise ValueError("Multi-view trainer only supports train.view=multi.")
    if getattr(hparams.model, "input_type", "rgb") != "rgb":
        raise ValueError("Multi-view trainer only supports model.input_type=rgb.")

    fuse_method = getattr(hparams.model, "fuse_method", None)
    backbone = getattr(hparams.model, "backbone", None)

    if fuse_method in LEGACY_FUSE_METHOD_ALIASES:
        logger.warning(
            "fuse_method 'se_atn' is deprecated and will be removed in a future "
            "version; use 'se_attn'."
        )
        fuse_method = LEGACY_FUSE_METHOD_ALIASES[fuse_method]

    if fuse_method in EARLY_FUSION_METHODS:
        trainer_cls = EARLY_FUSION_TRAINERS.get(backbone)
        if trainer_cls is None:
            raise ValueError(f"backbone {backbone} is not supported for early fusion.")
        return trainer_cls

    if fuse_method in MID_FUSION_METHODS:
        if backbone != "3dcnn":
            raise ValueError(
                f"backbone {backbone} is not supported for mid fusion (requires 3dcnn)."
            )
        return MID_FUSION_TRAINERS[fuse_method]
    if fuse_method in LATE_FUSION_METHODS:
        trainer_cls = LATE_FUSION_TRAINERS.get(backbone)
        if trainer_cls is None:
            raise ValueError(f"backbone {backbone} is not supported for late fusion.")
        return trainer_cls

    raise ValueError(f"fuse_method {fuse_method} is not supported.")


def build_multi_trainer(hparams):
    trainer_cls = select_multi_trainer_cls(hparams)
    return trainer_cls(hparams)
