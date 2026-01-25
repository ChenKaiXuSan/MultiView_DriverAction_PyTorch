#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/MultiView_DriverAction_PyTorch/project/dataloader/data_loader.py
Project: /workspace/MultiView_DriverAction_PyTorch/project/dataloader
Created Date: Saturday January 24th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday January 24th 2026 10:51:04 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from typing import Any, Callable, Dict, Optional

import torch
from pytorch_lightning import LightningDataModule
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.labeled_video_dataset import labeled_video_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Resize,
)

from project.dataloader.whole_video_dataset import whole_video_dataset
from project.dataloader.utils import (
    ApplyTransformToKey,
    Div255,
    UniformTemporalSubsample,
)


class DriverDataModule(LightningDataModule):
    def __init__(self, opt, dataset_idx: Dict = None):
        super().__init__()

        self._batch_size = opt.data.batch_size

        self._num_workers = opt.data.num_workers
        self._img_size = opt.data.img_size

        # frame rate
        self._clip_duration = opt.train.clip_duration
        self.uniform_temporal_subsample_num = opt.train.uniform_temporal_subsample_num

        # * this is the dataset idx, which include the train/val dataset idx.
        self._dataset_idx = dataset_idx

        self._class_num = opt.model.model_class_num

        self._experiment = opt.experiment
        self._backbone = opt.model.backbone

        self.mapping_transform = Compose(
            [
                UniformTemporalSubsample(self.uniform_temporal_subsample_num),
                Div255(),
                Resize(size=[self._img_size, self._img_size]),
            ]
        )

    def prepare_data(self) -> None:
        """here prepare the temp val data path,
        because the val dataset not use the gait cycle index,
        so we directly use the pytorchvideo API to load the video.
        AKA, use whole video to validate the model.
        """
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """

        # train dataset
        self.train_gait_dataset = whole_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["train"],
            transform=self.mapping_transform,
        )

        # val dataset
        self.val_gait_dataset = whole_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["val"],
            transform=self.mapping_transform,
        )

        # test dataset
        self.test_gait_dataset = whole_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["val"],
            transform=self.mapping_transform,
        )

    # def collate_fn(self, batch):
    #     """this function process the batch data, and return the batch data.

    #     Args:
    #         batch (list): the batch from the dataset.
    #         The batch include the one patient info from the json file.
    #         Here we only cat the one patient video tensor, and label tensor.

    #     Returns:
    #         dict: {video: torch.tensor, label: torch.tensor, info: list}
    #     """

    #     batch_label = []
    #     batch_video = []
    #     batch_attn_map = []

    #     # * mapping label
    #     for i in batch:
    #         # logging.info(i['video'].shape)
    #         gait_num, *_ = i["video"].shape
    #         disease = i["disease"]

    #         batch_video.append(i["video"])
    #         batch_attn_map.append(i["attn_map"])

    #         for _ in range(gait_num):
    #             if disease in disease_to_num_mapping_Dict[self._class_num].keys():
    #                 assert (
    #                     disease_to_num_mapping_Dict[self._class_num][disease]
    #                     == i["label"]
    #                 ), "The disease label mapping is not correct!"

    #                 batch_label.append(
    #                     disease_to_num_mapping_Dict[self._class_num][disease]
    #                 )
    #             else:
    #                 # * if the disease not in the mapping dict, then set the label to non-ASD.
    #                 batch_label.append(
    #                     disease_to_num_mapping_Dict[self._class_num]["non-ASD"]
    #                 )

    #     # video, b, c, t, h, w, which include the video frame
    #     # attn_map, b, c, t, h, w, which include the attn map
    #     # label, b, which include the label of the video
    #     # sample info, the raw sample info
    #     return {
    #         "video": torch.cat(batch_video, dim=0),
    #         "label": torch.tensor(batch_label),
    #         "attn_map": torch.cat(batch_attn_map, dim=0),
    #         "info": batch,
    #     }

    def train_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        train_data_loader = DataLoader(
            self.train_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )

        return train_data_loader

    def val_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        val_data_loader = DataLoader(
            self.val_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=True,
        )

        return val_data_loader

    def test_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        test_data_loader = DataLoader(
            self.test_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=True,
        )

        return test_data_loader
