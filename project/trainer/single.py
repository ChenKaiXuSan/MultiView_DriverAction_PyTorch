#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Single-view training selection helpers.
"""

import logging

from project.trainer.baseline.train_3dcnn import (
    Res3DCNNTrainer,
    TransformerTrainer,
    MambaTrainer,
)

logger = logging.getLogger(__name__)


RGB_SINGLE_VIEW_TRAINERS = {
    "3dcnn": Res3DCNNTrainer,
    "transformer": TransformerTrainer,
    "mamba": MambaTrainer,
}


def select_single_trainer_cls(hparams):
    """Select the trainer class for RGB single-view experiments."""
    if getattr(hparams.train, "view", None) != "single":
        raise ValueError("Single-view trainer only supports train.view=single.")
    if getattr(hparams.model, "input_type", "rgb") != "rgb":
        raise ValueError("Single-view trainer only supports model.input_type=rgb.")

    backbone = getattr(hparams.model, "backbone", None)
    trainer_cls = RGB_SINGLE_VIEW_TRAINERS.get(backbone)
    if trainer_cls is None:
        raise ValueError(
            f"backbone {backbone} is not supported for single-view RGB training."
        )
    return trainer_cls


def build_single_trainer(hparams):
    trainer_cls = select_single_trainer_cls(hparams)
    return trainer_cls(hparams)
