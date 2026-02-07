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
from project.map_config import VideoSample, label_mapping_Dict, KEEP_KEYPOINT_INDICES
from project.dataloader.annotation_dict import get_annotation_dict

logger = logging.getLogger(__name__)

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
        Filter keypoints based on KEEP_KEYPOINT_INDICES from map_config.
        
        Args:
            sam3d_dir: Directory containing frame npz files (e.g., .../front/)
            frame_indices: List of frame indices to load
        
        Returns:
            Tensor of shape (num_frames, num_kept_keypoints, 3) or None if not available
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
                kpts_list.append(np.zeros((len(KEEP_KEYPOINT_INDICES), 3), dtype=np.float32))
                continue
            
            try:
                data = np.load(str(npz_file), allow_pickle=True)
                output = data['output'].item()
                
                # 尝试从输出中提取3D keypoints
                # 根据SAM 3D body的输出格式调整这里
                if 'pred_keypoints_3d' in output:
                    kpts_3d = output['pred_keypoints_3d']
                elif 'poses' in output:
                    # 如果是SMPL格式的输出
                    kpts_3d = output['poses']
                else:
                    # 尝试其他可能的键名
                    kpts_3d = np.zeros((0, 3), dtype=np.float32)
                
                kpts_3d = np.asarray(kpts_3d, dtype=np.float32)
                if kpts_3d.ndim == 1:
                    kpts_3d = kpts_3d.reshape(-1, 3)
                
                # 根据 KEEP_KEYPOINT_INDICES 过滤关键点
                try:
                    filtered_kpts = kpts_3d[list(KEEP_KEYPOINT_INDICES)]
                except (IndexError, TypeError):
                    logger.debug(f"Error filtering keypoints with KEEP_KEYPOINT_INDICES for {npz_file}")
                    # 如果索引越界，使用零向量
                    filtered_kpts = np.zeros((len(KEEP_KEYPOINT_INDICES), 3), dtype=np.float32)
                
                kpts_list.append(filtered_kpts)
            except Exception as e:
                logger.debug(f"Error loading SAM 3D body from {npz_file}: {e}")
                kpts_list.append(np.zeros((len(KEEP_KEYPOINT_INDICES), 3), dtype=np.float32))
        
        # 所有帧现在应该有相同的 keypoint 数量
        num_keypoints = len(KEEP_KEYPOINT_INDICES)
        
        # Stack all keypoints
        kpts_array = np.zeros((len(frame_indices), num_keypoints, 3), dtype=np.float32)
        for i, kpts in enumerate(kpts_list):
            if kpts.shape[0] == num_keypoints:
                kpts_array[i] = kpts
            elif kpts.shape[0] > 0:
                # 如果形状不匹配，只复制可用部分
                min_len = min(kpts.shape[0], num_keypoints)
                kpts_array[i, :min_len] = kpts[:min_len]
        
        return torch.from_numpy(kpts_array)

    def _apply_transform(self, video_tchw: torch.Tensor) -> torch.Tensor:
        """
        Apply transform on a segment.

        Expect transform: (T,C,H,W) -> (T,C,H,W) or compatible.
        """
        if self._transform is None:
            return video_tchw
        return self._transform(video_tchw)

    def _validate_output_shapes(
        self,
        batch_front: torch.Tensor,
        batch_left: torch.Tensor,
        batch_right: torch.Tensor,
        mapped_labels: torch.LongTensor,
        labels: List[str],
        front_kpts_batch: Optional[torch.Tensor],
        left_kpts_batch: Optional[torch.Tensor],
        right_kpts_batch: Optional[torch.Tensor],
    ) -> None:
        """
        Validate output tensor shapes and consistency.

        Args:
            batch_front, batch_left, batch_right: Video tensors (B, C, T, H, W)
            mapped_labels: Label tensor (B,)
            labels: List of label strings
            front_kpts_batch, left_kpts_batch, right_kpts_batch: Keypoint tensors (B, T, K, 3) or None
        
        Raises:
            AssertionError: If shapes are inconsistent
        """
        # Video batch consistency check
        assert (
            batch_front.shape[0]
            == batch_left.shape[0]
            == batch_right.shape[0]
            == mapped_labels.shape[0]
            == len(labels)
        ), "Batch size mismatch after splitting"
        
        B = batch_front.shape[0]
        
        # Keypoint tensor dimension and batch size checks
        if front_kpts_batch is not None:
            assert front_kpts_batch.ndim == 4, f"front_kpts must be 4D (B, T, K, 3), got {front_kpts_batch.ndim}D"
            assert front_kpts_batch.shape[0] == B, f"front_kpts batch size {front_kpts_batch.shape[0]} != {B}"
            assert front_kpts_batch.shape[3] == 3, f"front_kpts must have 3 coordinates, got {front_kpts_batch.shape[3]}"
            
        if left_kpts_batch is not None:
            assert left_kpts_batch.ndim == 4, f"left_kpts must be 4D (B, T, K, 3), got {left_kpts_batch.ndim}D"
            assert left_kpts_batch.shape[0] == B, f"left_kpts batch size {left_kpts_batch.shape[0]} != {B}"
            assert left_kpts_batch.shape[3] == 3, f"left_kpts must have 3 coordinates, got {left_kpts_batch.shape[3]}"
            
        if right_kpts_batch is not None:
            assert right_kpts_batch.ndim == 4, f"right_kpts must be 4D (B, T, K, 3), got {right_kpts_batch.ndim}D"
            assert right_kpts_batch.shape[0] == B, f"right_kpts batch size {right_kpts_batch.shape[0]} != {B}"
            assert right_kpts_batch.shape[3] == 3, f"right_kpts must have 3 coordinates, got {right_kpts_batch.shape[3]}"
        
        # Ensure all keypoint tensors have the same T and K if they exist
        valid_kpts_tensors = [t for t in [front_kpts_batch, left_kpts_batch, right_kpts_batch] if t is not None]
        if len(valid_kpts_tensors) > 1:
            T_kpts = valid_kpts_tensors[0].shape[1]
            K_kpts = valid_kpts_tensors[0].shape[2]
            for i, kpts_t in enumerate(valid_kpts_tensors[1:], 1):
                assert kpts_t.shape[1] == T_kpts, f"Keypoint tensor {i} time dim {kpts_t.shape[1]} != {T_kpts}"
                assert kpts_t.shape[2] == K_kpts, f"Keypoint tensor {i} keypoint dim {kpts_t.shape[2]} != {K_kpts}"

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
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], 
               torch.Tensor, torch.Tensor, torch.Tensor, List[str], torch.LongTensor]:
        """
        Split video frames and keypoints according to label timeline.

        Returns:
            front_kpts_batch: (B, T, K, 3) or None
            left_kpts_batch: (B, T, K, 3) or None
            right_kpts_batch: (B, T, K, 3) or None
            batch_front: (B, C, T, H, W)  
            batch_left: (B, C, T, H, W)
            batch_right: (B, C, T, H, W)
            labels: List[str]
            mapped_labels: (B,)
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
        max_t_across_views = 0
        
        # First pass: determine max T across all views to ensure consistency
        for view, kpts_list in [("front", batch_front_kpts), 
                                 ("left", batch_left_kpts), 
                                 ("right", batch_right_kpts)]:
            if any(k is not None for k in kpts_list):
                valid_kpts = [k for k in kpts_list if k is not None]
                if valid_kpts:
                    max_t_view = max(k.shape[0] for k in valid_kpts)
                    max_t_across_views = max(max_t_across_views, max_t_view)
        
        # Second pass: stack and pad keypoint tensors
        for view, kpts_list in [("front", batch_front_kpts), 
                                 ("left", batch_left_kpts), 
                                 ("right", batch_right_kpts)]:
            if any(k is not None for k in kpts_list):
                # Get reference shape from first valid keypoint
                valid_kpts = [k for k in kpts_list if k is not None]
                if valid_kpts:
                    kpt_shape = valid_kpts[0].shape[1:]  # (K, 3)
                    padded_list = []
                    for k in kpts_list:
                        if k is None:
                            padded_list.append(torch.zeros(max_t_across_views, *kpt_shape, dtype=torch.float32))
                        else:
                            if k.shape[0] < max_t_across_views:
                                pad = torch.zeros(max_t_across_views - k.shape[0], *kpt_shape, dtype=torch.float32)
                                padded_list.append(torch.cat([k, pad], dim=0))
                            else:
                                padded_list.append(k)
                    kpts_dict[view] = torch.stack(padded_list, dim=0)  # (B, T, K, 3)
                else:
                    kpts_dict[view] = None
            else:
                kpts_dict[view] = None

        return (
            kpts_dict.get("front"),
            kpts_dict.get("left"),
            kpts_dict.get("right"),
            batch_front_t,
            batch_left_t,
            batch_right_t,
            labels,
            mapped_t,
        )

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
        # Keypoint paths come from item.sam3d_kpts (VideoSample)
        frame_indices = list(range(start_frame, end_frame))
        
        front_kpts = None
        left_kpts = None
        right_kpts = None
        
        if item.sam3d_kpts:
            if "front" in item.sam3d_kpts:
                front_kpts = self._load_sam3d_body_kpts(
                    item.sam3d_kpts["front"], 
                    frame_indices
                )
            
            if "left" in item.sam3d_kpts:
                left_kpts = self._load_sam3d_body_kpts(
                    item.sam3d_kpts["left"], 
                    frame_indices
                )
            
            if "right" in item.sam3d_kpts:
                right_kpts = self._load_sam3d_body_kpts(
                    item.sam3d_kpts["right"], 
                    frame_indices
                )

        # labels (ensure total_end = T)
        label_dict = prepare_label_dict(
            item.label_path, total_end=int(front_frames.shape[0])
        )
        timeline_list = label_dict.get("timeline_list", [])

        (
            front_kpts_batch,
            left_kpts_batch,
            right_kpts_batch,
            batch_front,
            batch_left,
            batch_right,
            labels,
            mapped_labels,
        ) = self.split_frame_with_label(
            front_frames,
            left_frames,
            right_frames,
            timeline_list,
            front_kpts=front_kpts,
            left_kpts=left_kpts,
            right_kpts=right_kpts,
        )

        # Validate output shapes
        self._validate_output_shapes(
            batch_front,
            batch_left,
            batch_right,
            mapped_labels,
            labels,
            front_kpts_batch,
            left_kpts_batch,
            right_kpts_batch,
        )
        return {
            "sam3d_kpt": {
                "front": front_kpts_batch,  
                "left": left_kpts_batch,  
                "right": right_kpts_batch,
            },
            "video": {
                "front": batch_front,
                "left": batch_left,
                "right": batch_right,
            },
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
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> LabeledVideoDataset:
    """
    Create a LabeledVideoDataset for whole video processing.
    
    Args:
        experiment: Experiment name
        dataset_idx: List of VideoSample items (contains sam3d_kpts paths)
        annotation_file: Path to annotation JSON file
        transform: Optional transform to apply to video frames
    
    Returns:
        LabeledVideoDataset instance
    """
    return LabeledVideoDataset(
        experiment=experiment,
        index_mapping=dataset_idx,
        annotation_file=annotation_file,
        sam3d_body_dirs=None,
        transform=transform,
    )
