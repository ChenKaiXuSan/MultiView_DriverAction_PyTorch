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
        annotation_dict: Dict[str, Any],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        decode_audio: bool = False,
        load_rgb: bool = True,
        load_kpt: bool = True,
        max_video_frames: Optional[int] = None,  # 如果设置，将长video分块加载
    ) -> None:
        super().__init__()
        self._experiment = experiment
        self._index_mapping = index_mapping
        self._annotation_dict = annotation_dict
        self._transform = transform
        self._decode_audio = bool(decode_audio)

        # Control what data to load
        self.load_rgb = bool(load_rgb)
        self.load_kpt = bool(load_kpt)
        
        # Video chunking to avoid OOM during loading
        self.max_video_frames = max_video_frames
        self._chunked_index: List[Dict[str, Any]] = []

        # label mapping: {class_id: "label_name"} -> {"label_name": class_id}
        self._label_to_id: Dict[str, int] = {
            v: int(k) for k, v in label_mapping_Dict.items()
        }
        
        # Build chunked index if max_video_frames is set
        if self.max_video_frames is not None:
            self._build_chunked_index()
            logger.info(
                f"Video chunking enabled: {len(self._index_mapping)} videos -> "
                f"{len(self._chunked_index)} chunks (max {self.max_video_frames} frames/chunk)"
            )

    def _build_chunked_index(self) -> None:
        """
        将长video分成多个chunks，每个chunk最多包含max_video_frames帧。
        这样可以避免加载超长video时OOM。
        
        Creates a new index where each item represents a chunk:
        {
            'original_item': VideoSample,
            'chunk_start_frame': int,
            'chunk_end_frame': int,
            'chunk_idx': int,
            'total_chunks': int,
        }
        """
        for item in self._index_mapping:
            # Get video total frames from annotation
            person_key = item.person_id
            env_folder = item.env_folder
            
            total_frames = 0
            start_frame_offset = 0
            end_frame = 0

            if (
                person_key in self._annotation_dict
                and env_folder in self._annotation_dict[person_key]
            ):
                frame_info = self._annotation_dict[person_key][env_folder]
                start_frame_offset = int(frame_info.get("start", 0))
                end_frame = int(frame_info.get("end", 0))

            start_frame_offset = max(0, start_frame_offset)
            end_frame = max(start_frame_offset, end_frame)
            total_frames = end_frame - start_frame_offset
            
            # 如果无法获取帧数或帧数为0，跳过分块，创建单个item
            if total_frames <= 0:
                self._chunked_index.append({
                    'original_item': item,
                    'chunk_start_frame': 0,
                    'chunk_end_frame': None,  # Load all
                    'chunk_idx': 0,
                    'total_chunks': 1,
                    'start_frame_offset': 0,
                })
                continue
            
            # Calculate number of chunks needed
            num_chunks = (total_frames + self.max_video_frames - 1) // self.max_video_frames
            
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * self.max_video_frames
                chunk_end = min(chunk_start + self.max_video_frames, total_frames)
                
                self._chunked_index.append({
                    'original_item': item,
                    'chunk_start_frame': chunk_start,
                    'chunk_end_frame': chunk_end,
                    'chunk_idx': chunk_idx,
                    'total_chunks': num_chunks,
                    'start_frame_offset': start_frame_offset,
                })

    def __len__(self) -> int:
        if self.max_video_frames is not None:
            return len(self._chunked_index)
        return len(self._index_mapping)

    # ---------------- IO ----------------
    def _load_one_view(
        self, 
        path: Path, 
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Load one view video and return (video_tchw, fps).
        
        Args:
            path: Video file path
            start_sec: Start time in seconds (None = from beginning)
            end_sec: End time in seconds (None = to end)

        Returns:
            vframes: (T, C, H, W)
            fps: frames per second
        """
        kwargs = {
            "pts_unit": "sec",
            "output_format": "TCHW",
        }
        
        # Add time range if specified (for chunked loading)
        if start_sec is not None:
            kwargs["start_pts"] = start_sec
        if end_sec is not None:
            kwargs["end_pts"] = end_sec
        
        vframes, aframes, info = read_video(str(path), **kwargs)
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
                kpts_list.append(
                    np.zeros((len(KEEP_KEYPOINT_INDICES), 3), dtype=np.float32)
                )
                continue

            try:
                data = np.load(str(npz_file), allow_pickle=True)
                output = data["output"].item()

                # 尝试从输出中提取3D keypoints
                # 根据SAM 3D body的输出格式调整这里
                if "pred_keypoints_3d" in output:
                    kpts_3d = output["pred_keypoints_3d"]
                elif "poses" in output:
                    # 如果是SMPL格式的输出
                    kpts_3d = output["poses"]
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
                    logger.debug(
                        f"Error filtering keypoints with KEEP_KEYPOINT_INDICES for {npz_file}"
                    )
                    # 如果索引越界，使用零向量
                    filtered_kpts = np.zeros(
                        (len(KEEP_KEYPOINT_INDICES), 3), dtype=np.float32
                    )

                kpts_list.append(filtered_kpts)
            except Exception as e:
                logger.debug(f"Error loading SAM 3D body from {npz_file}: {e}")
                kpts_list.append(
                    np.zeros((len(KEEP_KEYPOINT_INDICES), 3), dtype=np.float32)
                )

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
        batch_front: Optional[torch.Tensor],
        batch_left: Optional[torch.Tensor],
        batch_right: Optional[torch.Tensor],
        mapped_labels: torch.LongTensor,
        labels: List[str],
        front_kpts_batch: Optional[torch.Tensor],
        left_kpts_batch: Optional[torch.Tensor],
        right_kpts_batch: Optional[torch.Tensor],
    ) -> None:
        """
        Validate output tensor shapes and consistency.

        Args:
            batch_front, batch_left, batch_right: Video tensors (B, C, T, H, W) or None (for backward compat)
            mapped_labels: Label tensor (B,)
            labels: List of label strings
            front_kpts_batch, left_kpts_batch, right_kpts_batch: Keypoint tensors (B, T, K, 3) or None (for backward compat)

        Raises:
            AssertionError: If shapes are inconsistent
        """
        # Get batch size from available data
        B = None

        # Try to get batch size from video tensors
        if batch_front is not None:
            B = batch_front.shape[0]
        elif batch_left is not None:
            B = batch_left.shape[0]
        elif batch_right is not None:
            B = batch_right.shape[0]

        # Try to get batch size from keypoint tensors if video is None
        if B is None:
            if front_kpts_batch is not None:
                B = front_kpts_batch.shape[0]
            elif left_kpts_batch is not None:
                B = left_kpts_batch.shape[0]
            elif right_kpts_batch is not None:
                B = right_kpts_batch.shape[0]

        # If we still don't have batch size, use label as reference
        if B is None:
            B = mapped_labels.shape[0]

        # Validate batch size consistency
        assert mapped_labels.shape[0] == B, (
            f"Labels batch size {mapped_labels.shape[0]} != expected {B}"
        )
        assert len(labels) == B, f"Labels list length {len(labels)} != expected {B}"

        # Video batch consistency check (all video tensors should have same B if not None)
        video_tensors = [batch_front, batch_left, batch_right]
        valid_video_tensors = [t for t in video_tensors if t is not None]

        if len(valid_video_tensors) > 0:
            for i, tensor in enumerate(valid_video_tensors):
                assert tensor.ndim == 5, (
                    f"Video tensor {i} should be 5D (B, C, T, H, W), got {tensor.ndim}D"
                )
                assert tensor.shape[0] == B, (
                    f"Video tensor {i} batch size {tensor.shape[0]} != expected {B}"
                )
                assert tensor.shape[1] == 3, (
                    f"Video tensor {i} should have 3 channels, got {tensor.shape[1]}"
                )

        # Keypoint tensor validation (all kpts tensors should have same shape if not None)
        keypoint_tensors = [front_kpts_batch, left_kpts_batch, right_kpts_batch]
        valid_kpts_tensors = [t for t in keypoint_tensors if t is not None]

        if len(valid_kpts_tensors) > 0:
            # All valid keypoint tensors must have same shape
            reference_shape = valid_kpts_tensors[0].shape
            for i, kpts_t in enumerate(valid_kpts_tensors):
                assert kpts_t.ndim == 4, (
                    f"Keypoint tensor {i} should be 4D (B, T, K, 3), got {kpts_t.ndim}D"
                )
                assert kpts_t.shape[0] == B, (
                    f"Keypoint tensor {i} batch size {kpts_t.shape[0]} != {B}"
                )
                assert kpts_t.shape[3] == 3, (
                    f"Keypoint tensor {i} should have 3 coordinates, got {kpts_t.shape[3]}"
                )
                assert kpts_t.shape == reference_shape, (
                    f"Keypoint tensor {i} shape {kpts_t.shape} != reference {reference_shape}"
                )


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
        left_kpts: Optional[torch.Tensor] = None,  # (T, K, 3)
        right_kpts: Optional[torch.Tensor] = None,  # (T, K, 3)
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[str],
        torch.LongTensor,
    ]:
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
        assert front_view.shape[0] == left_view.shape[0] == right_view.shape[0], (
            "All views must have the same number of frames"
        )

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
        max_k_across_views = 0

        # First pass: determine max T and K across all views to ensure consistency
        for view, kpts_list in [
            ("front", batch_front_kpts),
            ("left", batch_left_kpts),
            ("right", batch_right_kpts),
        ]:
            if any(k is not None for k in kpts_list):
                valid_kpts = [k for k in kpts_list if k is not None]
                if valid_kpts:
                    max_t_view = max(k.shape[0] for k in valid_kpts)
                    max_k = valid_kpts[0].shape[1]  # Number of keypoints
                    max_t_across_views = max(max_t_across_views, max_t_view)
                    max_k_across_views = max(max_k_across_views, max_k)

        # Default values if no keypoints exist
        if max_t_across_views == 0:
            max_t_across_views = 1
        if max_k_across_views == 0:
            max_k_across_views = 17

        # Second pass: stack and pad keypoint tensors
        for view, kpts_list in [
            ("front", batch_front_kpts),
            ("left", batch_left_kpts),
            ("right", batch_right_kpts),
        ]:
            # Check if this view has any valid keypoints
            if any(k is not None for k in kpts_list):
                # Get reference shape from first valid keypoint
                valid_kpts = [k for k in kpts_list if k is not None]
                if valid_kpts:
                    kpt_shape = valid_kpts[0].shape[1:]  # (K, 3)
                    padded_list = []
                    for k in kpts_list:
                        if k is None:
                            padded_list.append(
                                torch.zeros(
                                    max_t_across_views, *kpt_shape, dtype=torch.float32
                                )
                            )
                        else:
                            if k.shape[0] < max_t_across_views:
                                pad = torch.zeros(
                                    max_t_across_views - k.shape[0],
                                    *kpt_shape,
                                    dtype=torch.float32,
                                )
                                padded_list.append(torch.cat([k, pad], dim=0))
                            else:
                                padded_list.append(k)
                    kpts_dict[view] = torch.stack(padded_list, dim=0)  # (B, T, K, 3)
                else:
                    # All None in this view, create zero tensor
                    kpts_dict[view] = torch.zeros(len(kpts_list), max_t_across_views, max_k_across_views, 3, dtype=torch.float32)
            else:
                # No valid keypoints in this view at all, create zero tensor
                B = len(kpts_list)
                kpts_dict[view] = torch.zeros(B, max_t_across_views, max_k_across_views, 3, dtype=torch.float32)

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
        # Handle chunked vs non-chunked index
        if self.max_video_frames is not None:
            chunk_info = self._chunked_index[index]
            item = chunk_info['original_item']
            chunk_start_frame = chunk_info['chunk_start_frame']
            chunk_end_frame = chunk_info['chunk_end_frame']
            start_frame_offset = chunk_info['start_frame_offset']
            
            # Calculate time range for video loading
            # Note: We need to get fps first (make a quick probe)
            if self.load_rgb:
                # Quick probe to get fps
                _, fps = self._load_one_view(item.videos["front"])
                
                # Calculate actual start/end in seconds
                # chunk frames are relative to the annotation start
                actual_start_frame = start_frame_offset + chunk_start_frame
                actual_end_frame = start_frame_offset + (chunk_end_frame if chunk_end_frame else chunk_start_frame + self.max_video_frames)
                
                start_sec = actual_start_frame / fps
                end_sec = actual_end_frame / fps
            else:
                start_sec = None
                end_sec = None
                fps = 0
        else:
            item = self._index_mapping[index]
            chunk_start_frame = 0
            chunk_end_frame = None
            start_frame_offset = 0
            start_sec = None
            end_sec = None

        # Load 3 views only if load_rgb is True
        if self.load_rgb:
            # Load with time range if chunking enabled
            front_frames, fps = self._load_one_view(item.videos["front"], start_sec, end_sec)
            left_frames, _ = self._load_one_view(item.videos["left"], start_sec, end_sec)
            right_frames, _ = self._load_one_view(item.videos["right"], start_sec, end_sec)

            assert (
                front_frames.shape[0] == left_frames.shape[0] == right_frames.shape[0]
            ), "All views must have the same number of frames"

            total_frames = int(front_frames.shape[0])
        else:
            # If not loading RGB, set frames to None
            front_frames = None
            left_frames = None
            right_frames = None
            if self.max_video_frames is None:
                fps = 0

            # Still need to determine total_frames from chunk info or annotation
            if self.max_video_frames is not None and chunk_end_frame is not None:
                total_frames = chunk_end_frame - chunk_start_frame
            else:
                total_frames = 0
                person_key = item.person_id
                env_folder = item.env_folder
                if (
                    person_key in self._annotation_dict
                    and env_folder in self._annotation_dict[person_key]
                ):
                    frame_info = self._annotation_dict[person_key][env_folder]
                    start_frame_ann = int(frame_info.get("start", 0))
                    end_frame_ann = int(frame_info.get("end", 1))
                    total_frames = end_frame_ann - start_frame_ann
                else:
                    total_frames = 1

        # Get start and end frame indices
        if self.load_rgb:
            # When chunking, frames are already sliced to the annotation range
            if self.max_video_frames is not None:
                start_frame = 0
                end_frame = total_frames
            else:
                start_frame = 0
                end_frame = total_frames

                # Try to get frame info from annotation dict
                person_key = item.person_id  # e.g., "person_01"
                env_folder = item.env_folder  # e.g., "夜多い"

                if (
                    person_key in self._annotation_dict
                    and env_folder in self._annotation_dict[person_key]
                ):
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

            frame_count = int(front_frames.shape[0])
        else:
            start_frame = 0
            end_frame = total_frames
            frame_count = total_frames

        # Load SAM 3D body keypoints only if load_kpt is True
        # For chunked loading, adjust frame indices
        if self.max_video_frames is not None:
            # Frame indices relative to chunk start
            actual_start = start_frame_offset + chunk_start_frame
            actual_end = actual_start + frame_count
            frame_indices = list(range(actual_start, actual_end))
        else:
            frame_indices = (
                list(range(start_frame, end_frame))
                if self.load_rgb
                else list(range(total_frames))
            )

        if self.load_kpt and item.sam3d_kpts:
            front_kpts = (
                self._load_sam3d_body_kpts(item.sam3d_kpts["front"], frame_indices)
                if "front" in item.sam3d_kpts
                else None
            )

            left_kpts = (
                self._load_sam3d_body_kpts(item.sam3d_kpts["left"], frame_indices)
                if "left" in item.sam3d_kpts
                else None
            )

            right_kpts = (
                self._load_sam3d_body_kpts(item.sam3d_kpts["right"], frame_indices)
                if "right" in item.sam3d_kpts
                else None
            )
        else:
            # If not loading keypoints, set to None
            front_kpts = None
            left_kpts = None
            right_kpts = None

        # labels (ensure total_end = T, or use total_frames if not loading RGB)
        if self.load_rgb:
            label_dict = prepare_label_dict(
                item.label_path, total_end=int(front_frames.shape[0])
            )
        else:
            label_dict = prepare_label_dict(item.label_path, total_end=frame_count)
        timeline_list = label_dict.get("timeline_list", [])

        # Split frames by label only if loading RGB
        if self.load_rgb:
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
        else:
            # If not loading RGB, still process timeline to get labels
            # Generate zero-padded video tensors
            B = len(timeline_list) if timeline_list else len(self._fill_tail_as_front(
                timeline_list, total_frames=frame_count, front_label="front"
            ))
            H, W = 224, 224  # Default video spatial dimensions
            
            # Generate labels from timeline
            labels = []
            mapped_labels_list = []

            timeline = self._fill_tail_as_front(
                timeline_list, total_frames=frame_count, front_label="front"
            )

            # Compute video tensors with zero padding (B, C, T, H, W) per segment
            video_segments_front = []
            video_segments_left = []
            video_segments_right = []

            for seg in timeline:
                s, e = int(seg["start"]), int(seg["end"])
                T_seg = e - s
                # Create zero-padded tensors for missing RGB data
                video_segments_front.append(torch.zeros(3, T_seg, H, W, dtype=torch.float32))
                video_segments_left.append(torch.zeros(3, T_seg, H, W, dtype=torch.float32))
                video_segments_right.append(torch.zeros(3, T_seg, H, W, dtype=torch.float32))

                lb = str(seg["label"])
                labels.append(lb)
                mapped_labels_list.append(self._label_to_id.get(lb, -1))

            mapped_labels = torch.tensor(mapped_labels_list, dtype=torch.long)

            # Stack video tensors (B, C, T, H, W)
            batch_front = torch.stack(video_segments_front, dim=0) if video_segments_front else torch.zeros(B, 3, 1, H, W, dtype=torch.float32)
            batch_left = torch.stack(video_segments_left, dim=0) if video_segments_left else torch.zeros(B, 3, 1, H, W, dtype=torch.float32)
            batch_right = torch.stack(video_segments_right, dim=0) if video_segments_right else torch.zeros(B, 3, 1, H, W, dtype=torch.float32)

            # If loading KPT, process them according to timeline segments
            if self.load_kpt and (
                front_kpts is not None
                or left_kpts is not None
                or right_kpts is not None
            ):
                # Stack keypoint tensors for each segment
                batch_front_kpts_list = []
                batch_left_kpts_list = []
                batch_right_kpts_list = []

                for seg in timeline:
                    s, e = int(seg["start"]), int(seg["end"])

                    if front_kpts is not None:
                        batch_front_kpts_list.append(front_kpts[s:e])
                    else:
                        batch_front_kpts_list.append(None)
                        
                    if left_kpts is not None:
                        batch_left_kpts_list.append(left_kpts[s:e])
                    else:
                        batch_left_kpts_list.append(None)
                        
                    if right_kpts is not None:
                        batch_right_kpts_list.append(right_kpts[s:e])
                    else:
                        batch_right_kpts_list.append(None)

                # Helper function to stack kpts with zero padding
                def stack_kpts_with_padding(kpts_list):
                    valid_kpts = [k for k in kpts_list if k is not None]
                    if not valid_kpts:
                        return None
                    
                    max_t = max(k.shape[0] for k in valid_kpts)
                    kpt_shape = valid_kpts[0].shape[1:]  # (K, 3)
                    padded = []
                    for k in kpts_list:
                        if k is None:
                            # Zero-pad when keypoint is not available
                            padded.append(torch.zeros(max_t, *kpt_shape, dtype=torch.float32))
                        else:
                            if k.shape[0] < max_t:
                                pad = torch.zeros(max_t - k.shape[0], *kpt_shape, dtype=torch.float32)
                                padded.append(torch.cat([k, pad], dim=0))
                            else:
                                padded.append(k)
                    return torch.stack(padded, dim=0)  # (B, T, K, 3)

                # Ensure all views have consistent keypoint shapes
                kpts_shapes = []
                for view, kpts_list in [
                    ("front", batch_front_kpts_list),
                    ("left", batch_left_kpts_list),
                    ("right", batch_right_kpts_list),
                ]:
                    valid = [k for k in kpts_list if k is not None]
                    if valid:
                        max_t = max(k.shape[0] for k in valid)
                        kpts_shapes.append((view, max_t, valid[0].shape[1]))

                # Get maximum T and K across all views
                max_t_all = max([shape[1] for shape in kpts_shapes]) if kpts_shapes else 1
                max_k_all = max([shape[2] for shape in kpts_shapes]) if kpts_shapes else 17

                # Stack kpts for each view
                front_kpts_batch = stack_kpts_with_padding(batch_front_kpts_list)
                left_kpts_batch = stack_kpts_with_padding(batch_left_kpts_list)
                right_kpts_batch = stack_kpts_with_padding(batch_right_kpts_list)

                # If any view is None, create zero-padded tensor to match others
                B_kpt = len(batch_front_kpts_list)
                if front_kpts_batch is None:
                    front_kpts_batch = torch.zeros(B_kpt, max_t_all, max_k_all, 3, dtype=torch.float32)
                if left_kpts_batch is None:
                    left_kpts_batch = torch.zeros(B_kpt, max_t_all, max_k_all, 3, dtype=torch.float32)
                if right_kpts_batch is None:
                    right_kpts_batch = torch.zeros(B_kpt, max_t_all, max_k_all, 3, dtype=torch.float32)
            else:
                # If not loading KPT, create zero-padded tensors
                B_kpt = len(timeline)
                front_kpts_batch = torch.zeros(B_kpt, 1, 17, 3, dtype=torch.float32)
                left_kpts_batch = torch.zeros(B_kpt, 1, 17, 3, dtype=torch.float32)
                right_kpts_batch = torch.zeros(B_kpt, 1, 17, 3, dtype=torch.float32)

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
                "is_chunked": self.max_video_frames is not None,
                "chunk_info": {
                    "chunk_idx": chunk_info['chunk_idx'],
                    "total_chunks": chunk_info['total_chunks'],
                    "chunk_start_frame": chunk_start_frame,  # Relative to annotation start
                    "chunk_end_frame": chunk_end_frame,  # Relative to annotation start
                    "absolute_start_frame": start_frame_offset + chunk_start_frame,  # Absolute frame index in original video
                    "absolute_end_frame": start_frame_offset + (chunk_end_frame if chunk_end_frame else chunk_start_frame),  # Absolute frame index in original video
                    "annotation_start": start_frame_offset,  # Annotation start frame
                } if self.max_video_frames is not None else None,
            },
        }


def whole_video_dataset(
    experiment: str,
    dataset_idx: List[VideoSample],
    annotation_dict: Dict[str, Any],
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    load_rgb: bool = True,
    load_kpt: bool = True,
    max_video_frames: Optional[int] = None,
) -> LabeledVideoDataset:
    """
    Create a LabeledVideoDataset for whole video processing.

    Args:
        experiment: Experiment name
        dataset_idx: List of VideoSample items (contains sam3d_kpts paths)
        annotation_dict: Annotation dictionary
        transform: Optional transform to apply to video frames
        load_rgb: Whether to load video frames (default: True)
        load_kpt: Whether to load keypoint data (default: True)
        max_video_frames: Maximum frames per chunk. If set, long videos will be
            split into multiple chunks to avoid OOM during loading. For example,
            max_video_frames=1000 means videos longer than 1000 frames will be
            split into multiple samples. Recommended: 500-2000 depending on resolution.
            (default: None - load entire video)

    Returns:
        LabeledVideoDataset instance
    """
    return LabeledVideoDataset(
        experiment=experiment,
        index_mapping=dataset_idx,
        annotation_dict=annotation_dict,
        transform=transform,
        load_rgb=load_rgb,
        load_kpt=load_kpt,
        max_video_frames=max_video_frames,
    )
