#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

from project.models.make_model import select_model


class LateFusion3DCNNTrainer(LightningModule):
    """
    Late-fusion multi-view video classifier.

    Expected batch format:
        batch["video"]["front"] : (B, C, T, H, W)
        batch["video"]["left"]  : (B, C, T, H, W)
        batch["video"]["right"] : (B, C, T, H, W)
        batch["sam3d_kpt"][view] : (B, T, K, 3) when input_type uses keypoints
        batch["label"]          : (B,)
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        self.img_size = hparams.data.img_size
        self.lr = hparams.loss.lr
        self.num_classes = hparams.model.model_class_num
        self.input_type = getattr(hparams.model, "input_type", "rgb")

        # three independent backbones
        self.front_cnn = select_model(hparams)
        self.left_cnn = select_model(hparams)
        self.right_cnn = select_model(hparams)

        # fusion config (optional)
        self.fusion_mode = getattr(hparams.model, "fusion_mode", "logit_mean")
        # OOM guard (optional)
        self.batch_size = int(getattr(hparams.data, "batch_size", 1))
        self.video_batch_size = int(getattr(hparams.data, "video_batch_size", 8))

        self._feature_dim = None
        if self.fusion_mode in {"feature_mean", "feature_concat"}:
            self._feature_dim = self._infer_feature_dim(self.front_cnn)
            fusion_dim = (
                self._feature_dim * 3
                if self.fusion_mode == "feature_concat"
                else self._feature_dim
            )
            self.view_fusion_head = torch.nn.Linear(fusion_dim, self.num_classes)
        else:
            self.view_fusion_head = None

        # metrics
        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

    # ---- core ----
    def forward(
        self,
        videos: Dict[str, torch.Tensor],
        kpts: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        videos: dict with keys: front/left/right, each (B,C,T,H,W)
        returns: fused logits (B,num_classes)
        """
        if self.fusion_mode in {"logit_mean", "prob_mean"}:
            front_logits = self._forward_view(
                self.front_cnn, videos["front"], kpts, "front"
            )
            left_logits = self._forward_view(
                self.left_cnn, videos["left"], kpts, "left"
            )
            right_logits = self._forward_view(
                self.right_cnn, videos["right"], kpts, "right"
            )

            if self.fusion_mode == "logit_mean":
                return (front_logits + left_logits + right_logits) / 3.0

            front_p = torch.softmax(front_logits, dim=1)
            left_p = torch.softmax(left_logits, dim=1)
            right_p = torch.softmax(right_logits, dim=1)
            fused_p = (front_p + left_p + right_p) / 3.0
            return torch.log(torch.clamp(fused_p, min=1e-8))

        front_feat = self._forward_view_features(
            self.front_cnn, videos["front"], kpts, "front"
        )
        left_feat = self._forward_view_features(
            self.left_cnn, videos["left"], kpts, "left"
        )
        right_feat = self._forward_view_features(
            self.right_cnn, videos["right"], kpts, "right"
        )

        if self.fusion_mode == "feature_mean":
            fused_feat = (front_feat + left_feat + right_feat) / 3.0
        elif self.fusion_mode == "feature_concat":
            fused_feat = torch.cat([front_feat, left_feat, right_feat], dim=1)
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        return self.view_fusion_head(fused_feat)

    def _maybe_trim_batch(
        self,
        videos: Dict[str, torch.Tensor],
        kpts: Optional[Dict[str, torch.Tensor]],
        label: torch.Tensor,
    ):
        """
        Simple OOM guard: trim batch if B is too large.
        """
        # TODO: 这里需要修改一个视频内部的多个片段的情况，目前只能按整体 batch size 来裁剪。
        bsz = label.size(0)
        if bsz <= self.video_batch_size:
            return videos, kpts, label

        idx = slice(0, self.video_batch_size)
        videos_trim = {k: v[idx].detach() for k, v in videos.items()}
        kpts_trim = None
        if kpts is not None:
            kpts_trim = {
                k: v[idx].detach() if v is not None else None for k, v in kpts.items()
            }
        label_trim = label[idx]
        return videos_trim, kpts_trim, label_trim

    def _infer_feature_dim(self, model) -> int:
        feature_dim = getattr(model, "feature_dim", None)
        if not feature_dim:
            raise ValueError("Selected model does not expose feature_dim for feature fusion.")
        return int(feature_dim)

    def _forward_view(
        self,
        model,
        video: torch.Tensor,
        kpts: Optional[Dict[str, torch.Tensor]],
        view: str,
    ) -> torch.Tensor:
        if self.input_type == "rgb":
            return model(video)
        if self.input_type == "kpt":
            if kpts is None or kpts.get(view) is None:
                raise ValueError("Keypoint input requested but sam3d_kpt is missing.")
            return model(kpts[view])
        if self.input_type == "rgb_kpt":
            if kpts is None or kpts.get(view) is None:
                raise ValueError("RGB+KPT input requested but sam3d_kpt is missing.")
            return model(video, kpts[view])
        raise ValueError(f"Unknown input_type: {self.input_type}")

    def _forward_view_features(
        self,
        model,
        video: torch.Tensor,
        kpts: Optional[Dict[str, torch.Tensor]],
        view: str,
    ) -> torch.Tensor:
        if not hasattr(model, "forward_features"):
            raise ValueError("Selected model does not support feature fusion.")
        if self.input_type == "rgb":
            return model.forward_features(video)
        if self.input_type == "kpt":
            if kpts is None or kpts.get(view) is None:
                raise ValueError("Keypoint input requested but sam3d_kpt is missing.")
            return model.forward_features(kpts[view])
        if self.input_type == "rgb_kpt":
            if kpts is None or kpts.get(view) is None:
                raise ValueError("RGB+KPT input requested but sam3d_kpt is missing.")
            return model.forward_features(video, kpts[view])
        raise ValueError(f"Unknown input_type: {self.input_type}")

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:

        videos = {k: v.detach() for k, v in batch["video"].items()}
        kpts = batch.get("sam3d_kpt")
        if kpts is not None:
            kpts = {k: (v.detach() if v is not None else None) for k, v in kpts.items()}
        label = batch["label"].view(-1)

        videos, kpts, label = self._maybe_trim_batch(videos, kpts, label)

        logits = self(videos, kpts)  # fused logits
        loss = F.cross_entropy(logits, label.long())

        probs = torch.softmax(logits, dim=1)

        # metrics
        acc = self._accuracy(probs, label)
        precision = self._precision(probs, label)
        recall = self._recall(probs, label)
        f1 = self._f1_score(probs, label)
        _ = self._confusion_matrix(
            probs, label
        )  # if you want to log later, store it yourself

        self.log(
            f"{stage}/loss", loss, on_step=True, on_epoch=True, batch_size=label.size(0)
        )
        self.log_dict(
            {
                f"{stage}/video_acc": acc,
                f"{stage}/video_precision": precision,
                f"{stage}/video_recall": recall,
                f"{stage}/video_f1_score": f1,
            },
            on_step=True,
            on_epoch=True,
            batch_size=label.size(0),
        )
        return loss

    # ---- lightning hooks ----
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        self._shared_step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
            },
        }
