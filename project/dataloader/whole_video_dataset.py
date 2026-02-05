#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Literal, List, Tuple
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.io import read_video

from project.dataloader.prepare_label_dict import prepare_label_dict
from project.map_config import VideoSample, label_mapping_Dict
from project.dataloader.annotation_dict import get_annotation_dict

logger = logging.getLogger(__name__)

ViewName = Literal["front", "left", "right"]


class LabeledVideoDataset(Dataset):
    """
    Multi-view labeled video dataset with SAM 3D body keypoints.

    Output:
        sample["video"][view] : Tensor (B, T, C, H, W)  # segments split by label timeline
        sample["sam3d_kpt"][view] : Tensor (B, T, K, 3)  # 3D keypoints for each view
        sample["label"]       : LongTensor (B,)
        sample["label_info"]  : List[str]
        sample["meta"]        : dict
    """

    def __init__(
        self,
        experiment: str,
        index_mapping: List[VideoSample],
        annotation_file: str,
        sam3d_body_dirs: Optional[Dict[str, Path]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        decode_audio: bool = False,
    ) -> None:
        super().__init__()
        self._experiment = experiment
        self._index_mapping = index_mapping
        self._annotation_dict = get_annotation_dict(annotation_file)
        self._transform = transform
        self._decode_audio = bool(decode_audio)
        self._sam3d_body_dirs = sam3d_body_dirs or {}  # {view: Path}

        # label mapping: {class_id: "label_name"} -> {"label_name": class_id}
        self._label_to_id: Dict[str, int] = {
            v: int(k) for k, v in label_mapping_Dict.items()
        }

    def __len__(self) -> int:
        return len(self._index_mapping)

    # ---------------- IO ----------------
    def _load_one_view(self, path: Path) -> Tuple[torch.Tensor, int]:
        """
        Load one view video and return (video_tchw, fps).

        read_video(output_format="TCHW") returns:
            vframes: (T, C, H, W)
        """
        vframes, aframes, info = read_video(
            str(path),
            pts_unit="sec",
            output_format="TCHW",
        )
        fps = int(info.get("video_fps", 0))
        if fps <= 0:
            raise ValueError(f"Invalid fps={fps} for video: {path}")
        return vframes, fps

    def _load_sam3d_body_kpts(
        self, 
        sam3d_dir: Path, 
        frame_indices: List[int],
    ) -> Optional[torch.Tensor]:
        """
        Load SAM 3D body 3D keypoints for a list of frame indices.
        
        Args:
            sam3d_dir: Directory containing frame npz files (e.g., .../front/)
            frame_indices: List of frame indices to load
        
        Returns:
            Tensor of shape (num_frames, num_keypoints, 3) or None if not available
        """
        if not sam3d_dir.exists():
            logger.warning(f"SAM 3D body directory not found: {sam3d_dir}")
            return None
        
        kpts_list = []
        for frame_idx in frame_indices:
            npz_file = sam3d_dir / f"{frame_idx:06d}_sam3d_body.npz"
            
            if not npz_file.exists():
                logger.debug(f"SAM 3D body file not found: {npz_file}")
                # 如果某一帧数据缺失，使用零向量占位
                kpts_list.append(np.zeros((0, 3), dtype=np.float32))
                continue
            
            try:
                data = np.load(str(npz_file), allow_pickle=True)
                output = data['output'].item()
                
                # 尝试从输出中提取3D keypoints
                # 根据SAM 3D body的输出格式调整这里
                if 'keypoints_3d' in output:
                    kpts_3d = output['keypoints_3d']
                elif 'poses' in output:
                    # 如果是SMPL格式的输出
                    kpts_3d = output['poses']
                else:
                    # 尝试其他可能的键名
                    kpts_3d = np.zeros((0, 3), dtype=np.float32)
                
                kpts_3d = np.asarray(kpts_3d, dtype=np.float32)
                if kpts_3d.ndim == 1:
                    kpts_3d = kpts_3d.reshape(-1, 3)
                
                kpts_list.append(kpts_3d)
            except Exception as e:
                logger.debug(f"Error loading SAM 3D body from {npz_file}: {e}")
                kpts_list.append(np.zeros((0, 3), dtype=np.float32))
        
        # 统一keypoint数量（使用第一个有效帧的keypoint数）
        valid_kpts = [k for k in kpts_list if k.shape[0] > 0]
        if not valid_kpts:
            return None
        
        num_keypoints = valid_kpts[0].shape[0]
        
        # Pad all keypoints to the same dimension
        kpts_array = np.zeros((len(frame_indices), num_keypoints, 3), dtype=np.float32)
        for i, kpts in enumerate(kpts_list):
            if kpts.shape[0] > 0:
                kpts = kpts[:num_keypoints]  # Truncate if too many
                kpts_array[i, :kpts.shape[0]] = kpts
        
        return torch.from_numpy(kpts_array)

    def _apply_transform(self, video_tchw: torch.Tensor) -> torch.Tensor:
        """
        Apply transform on a segment.

        Expect transform: (T,C,H,W) -> (T,C,H,W) or compatible.
        """
        if self._transform is None:
            return video_tchw
        return self._transform(video_tchw)

    # ---------------- Timeline utils ----------------
    @staticmethod
    def _fill_tail_as_front(
        timeline: List[Dict[str, Any]],
        total_frames: int,
        front_label: str = "front",
    ) -> List[Dict[str, Any]]:
        """
        If the timeline doesn't cover [0, total_frames), fill uncovered gaps as front.
        Assumes timeline items have int-like start/end and end is exclusive.
        """
        if total_frames <= 0:
            return []

        # sort by start
        tl = sorted(
            (
                {
                    "start": int(x["start"]),
                    "end": int(x["end"]),
                    "label": str(x["label"]),
                }
                for x in timeline
                if x is not None and "start" in x and "end" in x and "label" in x
            ),
            key=lambda d: (d["start"], d["end"]),
        )

        filled: List[Dict[str, Any]] = []
        cur = 0

        for seg in tl:
            s, e, lb = seg["start"], seg["end"], seg["label"]
            s = max(0, min(s, total_frames))
            e = max(0, min(e, total_frames))
            if e <= s:
                continue

            # gap before this seg
            if s > cur:
                filled.append({"start": cur, "end": s, "label": front_label})

            filled.append({"start": s, "end": e, "label": lb})
            cur = max(cur, e)

        # tail gap
        if cur < total_frames:
            filled.append({"start": cur, "end": total_frames, "label": front_label})

        return filled

    def split_frame_with_label(
        self,
        front_view: torch.Tensor,  # (T,C,H,W)
        left_view: torch.Tensor,  # (T,C,H,W)
        right_view: torch.Tensor,  # (T,C,H,W)
        timeline_list: List[Dict[str, Any]],
        front_kpts: Optional[torch.Tensor] = None,  # (T, K, 3) 
        left_kpts: Optional[torch.Tensor] = None,   # (T, K, 3)
        right_kpts: Optional[torch.Tensor] = None,  # (T, K, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], torch.LongTensor, 
               Dict[str, Optional[torch.Tensor]]]:
        """
        Split video frames and keypoints according to label timeline.

        Returns:
            batch_front: (B, C, T, H, W)  
            batch_left: (B, C, T, H, W)
            batch_right: (B, C, T, H, W)
            labels: List[str]
            mapped_labels: (B,)
            kpts_dict: {"front": (B, T, K, 3) or None, "left": ..., "right": ...}
        """
        assert (
            front_view.shape[0] == left_view.shape[0] == right_view.shape[0]
        ), "All views must have the same number of frames"

        T = int(front_view.shape[0])

        # 1) fill uncovered as front
        timeline = self._fill_tail_as_front(
            timeline_list, total_frames=T, front_label="front"
        )

        batch_front: List[torch.Tensor] = []
        batch_left: List[torch.Tensor] = []
        batch_right: List[torch.Tensor] = []
        batch_front_kpts: List[Optional[torch.Tensor]] = []
        batch_left_kpts: List[Optional[torch.Tensor]] = []
        batch_right_kpts: List[Optional[torch.Tensor]] = []
        labels: List[str] = []
        mapped: List[int] = []

        for seg in timeline:
            s, e, lb = int(seg["start"]), int(seg["end"]), str(seg["label"])
            if e <= s:
                continue

            seg_front = self._apply_transform(front_view[s:e])
            seg_left = self._apply_transform(left_view[s:e])
            seg_right = self._apply_transform(right_view[s:e])

            batch_front.append(seg_front)
            batch_left.append(seg_left)
            batch_right.append(seg_right)
            
            # Extract keypoints for this segment
            if front_kpts is not None:
                batch_front_kpts.append(front_kpts[s:e])
            else:
                batch_front_kpts.append(None)
            
            if left_kpts is not None:
                batch_left_kpts.append(left_kpts[s:e])
            else:
                batch_left_kpts.append(None)
            
            if right_kpts is not None:
                batch_right_kpts.append(right_kpts[s:e])
            else:
                batch_right_kpts.append(None)

            labels.append(lb)
            mapped.append(self._label_to_id.get(lb, -1))  # unknown -> -1

        # Stack video tensors
        batch_front_t = torch.stack(batch_front, dim=0).permute(0, 2, 1, 3, 4)
        batch_left_t = torch.stack(batch_left, dim=0).permute(0, 2, 1, 3, 4)
        batch_right_t = torch.stack(batch_right, dim=0).permute(0, 2, 1, 3, 4)

        mapped_t = torch.tensor(mapped, dtype=torch.long)
        
        # Stack keypoint tensors if available
        kpts_dict = {}
        for view, kpts_list in [("front", batch_front_kpts), 
                                 ("left", batch_left_kpts), 
                                 ("right", batch_right_kpts)]:
            if any(k is not None for k in kpts_list):
                # Get reference shape from first valid keypoint
                valid_kpts = [k for k in kpts_list if k is not None]
                if valid_kpts:
                    kpt_shape = valid_kpts[0].shape[1:]  # (K, 3)
                    max_t = max(k.shape[0] for k in valid_kpts)
                    padded_list = []
                    for k in kpts_list:
                        if k is None:
                            padded_list.append(torch.zeros(max_t, *kpt_shape, dtype=torch.float32))
                        else:
                            if k.shape[0] < max_t:
                                pad = torch.zeros(max_t - k.shape[0], *kpt_shape, dtype=torch.float32)
                                padded_list.append(torch.cat([k, pad], dim=0))
                            else:
                                padded_list.append(k)
                    kpts_dict[view] = torch.stack(padded_list, dim=0)  # (B, T, K, 3)
                else:
                    kpts_dict[view] = None
            else:
                kpts_dict[view] = None

        return batch_front_t, batch_left_t, batch_right_t, labels, mapped_t, kpts_dict

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self._index_mapping[index]

        # load 3 views (T,C,H,W)
        front_frames, fps = self._load_one_view(item.videos["front"])
        left_frames, _ = self._load_one_view(item.videos["left"])
        right_frames, _ = self._load_one_view(item.videos["right"])

        assert (
            front_frames.shape[0] == left_frames.shape[0] == right_frames.shape[0]
        ), "All views must have the same number of frames"

        # Get start and end frame indices from annotation dict if available
        total_frames = int(front_frames.shape[0])
        start_frame = 0
        end_frame = total_frames
        
        # Try to get frame info from annotation dict
        person_key = item.person_id  # e.g., "person_01"
        env_folder = item.env_folder  # e.g., "夜多い"
        
        if person_key in self._annotation_dict and env_folder in self._annotation_dict[person_key]:
            frame_info = self._annotation_dict[person_key][env_folder]
            if frame_info.get("start") is not None:
                start_frame = int(frame_info.get("start", 0))
            if frame_info.get("end") is not None:
                end_frame = int(frame_info.get("end", total_frames))
            
            # Clamp to valid range
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        # Slice video to the specified frame range
        front_frames = front_frames[start_frame:end_frame]
        left_frames = left_frames[start_frame:end_frame]
        right_frames = right_frames[start_frame:end_frame]

        # Load SAM 3D body keypoints for all three views
        frame_indices = list(range(start_frame, end_frame))
        
        front_kpts = None
        left_kpts = None
        right_kpts = None
        
        if "front" in self._sam3d_body_dirs:
            front_kpts = self._load_sam3d_body_kpts(
                self._sam3d_body_dirs["front"], 
                frame_indices
            )
        
        if "left" in self._sam3d_body_dirs:
            left_kpts = self._load_sam3d_body_kpts(
                self._sam3d_body_dirs["left"], 
                frame_indices
            )
        
        if "right" in self._sam3d_body_dirs:
            right_kpts = self._load_sam3d_body_kpts(
                self._sam3d_body_dirs["right"], 
                frame_indices
            )

        # labels (ensure total_end = T)
        label_dict = prepare_label_dict(
            item.label_path, total_end=int(front_frames.shape[0])
        )
        timeline_list = label_dict.get("timeline_list", [])

        (
            batch_front,
            batch_left,
            batch_right,
            labels,
            mapped_labels,
            kpts_dict,
        ) = self.split_frame_with_label(
            front_frames,
            left_frames,
            right_frames,
            timeline_list,
            front_kpts=front_kpts,
            left_kpts=left_kpts,
            right_kpts=right_kpts,
        )

        assert (
            batch_front.shape[0]
            == batch_left.shape[0]
            == batch_right.shape[0]
            == mapped_labels.shape[0]
            == len(labels)
        ), "Batch size mismatch after splitting"

        return {
            "video": {
                "front": batch_front,
                "left": batch_left,
                "right": batch_right,
            },
            "sam3d_kpt": kpts_dict,  # {"front": Tensor or None, "left": ..., "right": ...}
            "label": mapped_labels,  # LongTensor (B,)
            "label_info": labels,  # List[str]
            "meta": {
                "experiment": self._experiment,
                "index": index,
                "person_id": item.person_id,
                "env_folder": item.env_folder,
                "env_key": item.env_key,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": fps,
            },
        }


def whole_video_dataset(
    experiment: str,
    dataset_idx: List[VideoSample],
    annotation_file: str = None,
    sam3d_body_dirs: Optional[Dict[str, Path]] = None,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> LabeledVideoDataset:
    """
    Create a LabeledVideoDataset for whole video processing.
    
    Args:
        experiment: Experiment name
        dataset_idx: List of VideoSample items
        annotation_file: Path to annotation JSON file
        sam3d_body_dirs: Dict mapping view names to SAM 3D body result directories
                        Example: {"front": Path(...), "left": Path(...), "right": Path(...)}
        transform: Optional transform to apply to video frames
    
    Returns:
        LabeledVideoDataset instance
    """
    return LabeledVideoDataset(
        experiment=experiment,
        index_mapping=dataset_idx,
        annotation_file=annotation_file,
        sam3d_body_dirs=sam3d_body_dirs,
        transform=transform,
    )
