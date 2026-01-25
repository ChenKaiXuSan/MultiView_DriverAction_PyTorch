#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Literal

import torch
from torch.utils.data import Dataset
from torchvision.io import read_video

from project.dataloader.prepare_label_dict import prepare_label_dict
from project.cross_validation import Sample, label_mapping_Dict

logger = logging.getLogger(__name__)

ViewName = Literal["front", "left", "right"]


class LabeledVideoDataset(Dataset):
    """
    Multi-view labeled video dataset.

    Output (default):
        sample["video"][view] : Tensor (B, C, T, H, W)  # per-second clips
        sample["label"]       : Tensor (B,) or (B, num_classes) depending on your label logic
        sample["meta"]        : dict
    """

    def __init__(
        self,
        experiment: str,
        index_mapping: list[Sample],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        drop_last: bool = True,
        decode_audio: bool = False,
    ) -> None:
        super().__init__()
        self._experiment = experiment
        self._index_mapping = index_mapping
        self._transform = transform
        self._drop_last = bool(drop_last)
        self._decode_audio = bool(decode_audio)

    def __len__(self) -> int:
        return len(self._index_mapping)

    def _split_into_clips(self, v_cthw: torch.Tensor, fps: int) -> torch.Tensor:
        """
        Split (C,T,H,W) into (B,C,Tc,H,W) clips, where Tc = fps * clip_len_sec.
        """
        c, t, h, w = v_cthw.shape
        clip_t = fps * 1
        if clip_t <= 0:
            raise ValueError(f"Invalid clip_t computed from fps={fps}, clip_len_sec=1")

        if t < clip_t:
            # too short: return empty
            return v_cthw.new_empty((0, c, clip_t, h, w))

        n_full = t // clip_t if self._drop_last else (t + clip_t - 1) // clip_t
        clips = []

        for i in range(n_full):
            s = i * clip_t
            e = s + clip_t
            if e <= t:
                clip = v_cthw[:, s:e, :, :]
            else:
                if self._drop_last:
                    break
                # pad last clip (replicate last frame)
                pad = e - t
                last = v_cthw[:, -1:, :, :].expand(c, pad, h, w)
                clip = torch.cat([v_cthw[:, s:t, :, :], last], dim=1)

            if self._transform is not None:
                # transform expects (C,T,H,W) -> (C,T,H,W)
                clip = self._transform(clip)
            clips.append(clip)

        if len(clips) == 0:
            return v_cthw.new_empty((0, c, clip_t, h, w))

        return torch.stack(clips, dim=0)  # (B,C,Tc,H,W)

    def _load_one_view(self, path: Path) -> tuple[torch.Tensor, int]:
        """
        Load one view video and return (video_cthw, fps).
        """
        # torchvision read_video -> (T,H,W,C), audio, info
        vframes, aframes, info = read_video(
            str(path), pts_unit="sec", output_format="TCHW"
        )
        fps = int(info.get("video_fps", 0))
        if fps <= 0:
            raise ValueError(f"Invalid fps={fps} for video: {path}")

        return vframes, fps

    def _move_transform(self, video: torch.Tensor) -> torch.Tensor:
        """
        Move transform to outside after splitting into clips.
        This is optional and depends on your transform logic.
        """
        if self._transform:
            t, c, h, w = video.shape
            video = self._transform(video)  # T,C,H,W
        return video

    def split_frame_with_label(
        self,
        front_view: torch.Tensor,
        left_view: torch.Tensor,
        right_view: torch.Tensor,
        label_timeline_list: Dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split video frames according to labels.

        Args:
            video: (C,T,H,W)
            labels: {"label_name": [{"start": s, "end": e}, ...]}

        Returns:
            Tensor: (B,C,T,H,W)
        """
        assert (
            front_view.shape[1] == left_view.shape[1] == right_view.shape[1]
        ), "All views must have the same number of frames"

        labels = []
        mapped_labels = []
        batch_front_views = []
        batch_left_views = []
        batch_right_views = []

        # TODO: 目前这里指根据标签时间线切分视频帧，当时后面如果没有标签的帧会丢弃，也要补齐为front吗
        
        # label, num:{"start": s, "end": e}
        for item in label_timeline_list:
            start = int(item["start"])
            end = int(item["end"])
            label = item["label"]
            # T, C, H, W
            batch_front_views.append(self._move_transform(front_view[start:end, ...]))
            batch_left_views.append(self._move_transform(left_view[start:end, ...]))
            batch_right_views.append(self._move_transform(right_view[start:end, ...]))
            for num, mapped_label in label_mapping_Dict.items():
                if mapped_label == label:
                    labels.append(label)
                    mapped_labels.append(num)

        return (
            torch.stack(batch_front_views, dim=0),
            torch.stack(batch_left_views, dim=0),
            torch.stack(batch_right_views, dim=0),
            labels,
            mapped_labels,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._index_mapping[index]

        # load 3 views
        front_view_frames: torch.Tensor = self._load_one_view(item.videos["front"])[0]
        left_view_frames: torch.Tensor = self._load_one_view(item.videos["left"])[0]
        right_view_frames: torch.Tensor = self._load_one_view(item.videos["right"])[0]
        assert (
            front_view_frames.shape[0]
            == left_view_frames.shape[0]
            == right_view_frames.shape[0]
        ), "All views must have the same number of frames"

        # labels
        label_dict = prepare_label_dict(
            item.label_path, total_end=front_view_frames.shape[0]
        )

        (
            batch_front_views,
            batch_left_views,
            batch_right_views,
            labels,
            mapped_labels,
        ) = self.split_frame_with_label(
            front_view_frames,
            left_view_frames,
            right_view_frames,
            label_dict["timeline_list"],
        )

        sample = {
            "video": {
                "front": batch_front_views,  # (B,C,T,H,W)
                "left": batch_left_views,  # (B,C,T,H,W)
                "right": batch_right_views,  # (B,C,T,H,W)
            },
            "label": mapped_labels,  # List[int]
            "label_info": labels,
            "meta": {
                "experiment": self._experiment,
                "index": index,
                "person_id": item.person_id,
                "env_folder": item.env_folder,
                "env_key": item.env_key,
            },
        }
        return sample


def whole_video_dataset(
    experiment: str,
    dataset_idx: list[VideoIndexItem],
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    drop_last: bool = True,
) -> LabeledVideoDataset:
    return LabeledVideoDataset(
        experiment=experiment,
        transform=transform,
        index_mapping=dataset_idx,
        drop_last=drop_last,
    )
