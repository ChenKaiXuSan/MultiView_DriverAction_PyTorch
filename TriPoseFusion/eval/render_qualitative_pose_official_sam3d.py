#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common import (  # noqa: E402
    ENV_NAMES,
    canonicalize_pose,
    list_sam3d_files,
    load_gt_sequence,
    normalize_frame_id,
    select_common_frame_ids,
)
from eval.render_qualitative_pose_comparison import build_comparison  # noqa: E402
from eval.render_qualitative_pose_samples import choose_sample_indices  # noqa: E402
from map_config import KEEP_KEYPOINT_INDICES  # noqa: E402

DEFAULT_SAM3D_REPO = Path("/home/workspace/kaixu/code/Drive_Face_Mesh_PyTorch/SAM3Dbody")
OFFICIAL_BLUE = (51 / 255.0, 153 / 255.0, 255 / 255.0)
OFFICIAL_LEFT_GREEN = (0 / 255.0, 255 / 255.0, 0 / 255.0)
OFFICIAL_RIGHT_ORANGE = (255 / 255.0, 128 / 255.0, 0 / 255.0)
MODEL52_BODY_ANCHOR_INDICES = (5, 6, 51)


def load_official_mhr70_pose_info(sam3d_repo: Path) -> dict:
    metadata_path = Path(sam3d_repo) / "sam_3d_body" / "metadata" / "mhr70.py"
    if not metadata_path.exists():
        raise FileNotFoundError(f"SAM3D Body MHR70 metadata not found: {metadata_path}")
    spec = importlib.util.spec_from_file_location("_sam3d_mhr70_metadata", metadata_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {metadata_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pose_info


def compact_official_skeleton(pose_info: dict) -> list[tuple[int, int, tuple[float, float, float]]]:
    keep = list(KEEP_KEYPOINT_INDICES)
    original_to_compact = {original: compact for compact, original in enumerate(keep)}
    name_to_original = {
        item["name"]: int(item["id"])
        for item in pose_info["keypoint_info"].values()
    }
    edges = []
    for item in pose_info["skeleton_info"].values():
        name_a, name_b = item["link"]
        idx_a = name_to_original.get(name_a)
        idx_b = name_to_original.get(name_b)
        if idx_a in original_to_compact and idx_b in original_to_compact:
            color = tuple(float(c) / 255.0 for c in item.get("color", [51, 153, 255]))
            edges.append((original_to_compact[idx_a], original_to_compact[idx_b], color))
    bridge_specs = (
        ("left_acromion", "left_wrist", OFFICIAL_LEFT_GREEN),
        ("left_shoulder", "left_wrist", OFFICIAL_LEFT_GREEN),
        ("right_acromion", "right_wrist", OFFICIAL_RIGHT_ORANGE),
        ("right_shoulder", "right_wrist", OFFICIAL_RIGHT_ORANGE),
        ("neck", "left_acromion", OFFICIAL_BLUE),
        ("neck", "right_acromion", OFFICIAL_BLUE),
    )
    existing_pairs = {(a, b) for a, b, _ in edges}
    for name_a, name_b, color in bridge_specs:
        idx_a = name_to_original.get(name_a)
        idx_b = name_to_original.get(name_b)
        if idx_a in original_to_compact and idx_b in original_to_compact:
            compact_pair = (original_to_compact[idx_a], original_to_compact[idx_b])
            if compact_pair not in existing_pairs:
                edges.append((*compact_pair, color))
                existing_pairs.add(compact_pair)
    return edges


def full_official_skeleton(pose_info: dict) -> list[tuple[int, int, tuple[float, float, float]]]:
    name_to_original = {
        item["name"]: int(item["id"])
        for item in pose_info["keypoint_info"].values()
    }
    edges = []
    for item in pose_info["skeleton_info"].values():
        name_a, name_b = item["link"]
        idx_a = name_to_original.get(name_a)
        idx_b = name_to_original.get(name_b)
        if idx_a is not None and idx_b is not None:
            color = tuple(float(c) / 255.0 for c in item.get("color", [51, 153, 255]))
            edges.append((idx_a, idx_b, color))
    return edges


def compact_official_keypoint_colors(pose_info: dict) -> list[tuple[float, float, float]]:
    id_to_color = {
        int(item["id"]): tuple(float(c) / 255.0 for c in item.get("color", [51, 153, 255]))
        for item in pose_info["keypoint_info"].values()
    }
    return [id_to_color.get(original_id, OFFICIAL_BLUE) for original_id in KEEP_KEYPOINT_INDICES]


def full_official_keypoint_colors(pose_info: dict) -> list[tuple[float, float, float]]:
    id_to_color = {
        int(item["id"]): tuple(float(c) / 255.0 for c in item.get("color", [51, 153, 255]))
        for item in pose_info["keypoint_info"].values()
    }
    max_id = max(id_to_color.keys())
    return [id_to_color.get(original_id, OFFICIAL_BLUE) for original_id in range(max_id + 1)]


def load_full_triangulated_pose(
    gt_root: Path,
    subject: str,
    env_folder: str,
    frame_id: str,
    canonicalize: bool = True,
) -> np.ndarray:
    gt_path = Path(gt_root) / subject / env_folder / "keypoints_3d.npz"
    with np.load(gt_path, allow_pickle=False) as data:
        keypoints = np.asarray(data["keypoints_3d"], dtype=np.float32)
        frame_ids = [normalize_frame_id(item) for item in np.asarray(data["frame_ids"]).tolist()]
    lookup = {fid: idx for idx, fid in enumerate(frame_ids)}
    normalized = normalize_frame_id(frame_id)
    if normalized not in lookup:
        raise KeyError(f"Frame id {frame_id} not found in {gt_path}")
    pose = keypoints[lookup[normalized], :, :3]
    if canonicalize:
        pose = canonicalize_pose(
            pose[None, :, :],
            neck_index=69,
            left_shoulder_index=5,
            right_shoulder_index=6,
        )[0]
    return pose


def _hand_valid_ratios(valid_row: np.ndarray) -> tuple[float, float]:
    right = np.asarray(valid_row[21:42], dtype=bool)
    left = np.asarray(valid_row[42:63], dtype=bool)
    return float(right.mean()), float(left.mean())


def select_hand_complete_frame_indices(
    aligned_ids: Sequence[str],
    gt_lookup: dict[str, int],
    gt_valid_mask: np.ndarray,
    samples_per_env: int,
    min_hand_valid_ratio: float = 0.8,
) -> list[int]:
    scored = []
    for aligned_index, frame_id in enumerate(aligned_ids):
        gt_index = gt_lookup[normalize_frame_id(frame_id)]
        right_ratio, left_ratio = _hand_valid_ratios(gt_valid_mask[gt_index])
        both_ratio = min(right_ratio, left_ratio)
        total_ratio = 0.5 * (right_ratio + left_ratio)
        scored.append((aligned_index, both_ratio, total_ratio))
    preferred = [item for item in scored if item[1] >= min_hand_valid_ratio]
    fallback = [item for item in scored if item[1] < min_hand_valid_ratio]
    preferred.sort(key=lambda item: (-item[1], -item[2], item[0]))
    fallback.sort(key=lambda item: (-item[1], -item[2], item[0]))
    selected = [item[0] for item in (preferred + fallback)[:samples_per_env]]
    return sorted(selected)


def align_full_pose_to_model_reference(
    full_pose: np.ndarray,
    model_reference_pose: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    if model_reference_pose is None:
        return np.asarray(full_pose, dtype=np.float32), 1.0
    full_pose = np.asarray(full_pose, dtype=np.float32)
    model_reference_pose = np.asarray(model_reference_pose, dtype=np.float32)
    keep = np.asarray(KEEP_KEYPOINT_INDICES, dtype=np.int64)
    common_full = full_pose[keep]
    common_model = model_reference_pose[: len(keep)]
    valid = np.isfinite(common_full).all(axis=-1) & np.isfinite(common_model).all(axis=-1)
    if int(valid.sum()) < 2:
        return full_pose, 1.0
    source = common_full[valid]
    target = common_model[valid]
    source_center = source.mean(axis=0, keepdims=True)
    target_center = target.mean(axis=0, keepdims=True)
    source_centered = source - source_center
    target_centered = target - target_center
    denom = float(np.sum(source_centered * source_centered))
    if denom < 1e-8:
        return full_pose, 1.0
    scale = float(np.sum(source_centered * target_centered) / denom)
    aligned = (full_pose - source_center.astype(np.float32)) * scale + target_center.astype(np.float32)
    return aligned.astype(np.float32), scale


def align_pose_scale_translation(
    source_pose: np.ndarray,
    target_pose: np.ndarray | None,
    source_indices: Sequence[int] | None = None,
    target_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, float]:
    if target_pose is None:
        return np.asarray(source_pose, dtype=np.float32), 1.0
    source_pose = np.asarray(source_pose, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32)
    if source_indices is None or target_indices is None:
        count = min(source_pose.shape[0], target_pose.shape[0])
        source_common = source_pose[:count]
        target_common = target_pose[:count]
    else:
        source_common = source_pose[np.asarray(source_indices, dtype=np.int64)]
        target_common = target_pose[np.asarray(target_indices, dtype=np.int64)]
    valid = np.isfinite(source_common).all(axis=-1) & np.isfinite(target_common).all(axis=-1)
    if int(valid.sum()) < 2:
        return source_pose, 1.0
    source = source_common[valid]
    target = target_common[valid]
    source_center = source.mean(axis=0, keepdims=True)
    target_center = target.mean(axis=0, keepdims=True)
    source_centered = source - source_center
    target_centered = target - target_center
    source_energy = float(np.sum(source_centered * source_centered))
    target_energy = float(np.sum(target_centered * target_centered))
    if source_energy < 1e-8 or target_energy < 1e-8:
        return source_pose, 1.0
    scale = float(np.sqrt(target_energy / source_energy))
    aligned = (source_pose - source_center.astype(np.float32)) * scale + target_center.astype(np.float32)
    return aligned.astype(np.float32), scale


def align_pose_scale_with_anchor_translation(
    source_pose: np.ndarray,
    target_pose: np.ndarray | None,
    anchor_indices: Sequence[int] = MODEL52_BODY_ANCHOR_INDICES,
    source_indices: Sequence[int] | None = None,
    target_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, float]:
    scaled_pose, scale = align_pose_scale_translation(
        source_pose,
        target_pose,
        source_indices=source_indices,
        target_indices=target_indices,
    )
    if target_pose is None:
        return scaled_pose, scale
    target_pose = np.asarray(target_pose, dtype=np.float32)
    anchors = np.asarray(anchor_indices, dtype=np.int64)
    anchors = anchors[(anchors < scaled_pose.shape[0]) & (anchors < target_pose.shape[0])]
    if anchors.size == 0:
        return scaled_pose, scale
    valid = np.isfinite(scaled_pose[anchors]).all(axis=-1) & np.isfinite(target_pose[anchors]).all(axis=-1)
    if not np.any(valid):
        return scaled_pose, scale
    source_center = scaled_pose[anchors[valid]].mean(axis=0, keepdims=True)
    target_center = target_pose[anchors[valid]].mean(axis=0, keepdims=True)
    aligned = scaled_pose + (target_center - source_center).astype(np.float32)
    return aligned.astype(np.float32), scale


def display_scale_target_pose(poses: dict[str, np.ndarray | None]) -> np.ndarray | None:
    for label in ("TriPoseFusion", "TriPoseFusion (inferred)", "Median fusion", "Single-view"):
        pose = poses.get(label)
        if pose is not None:
            return pose
    return None


def _finite_pose_points(poses: dict[str, np.ndarray | None]) -> np.ndarray:
    points = []
    for pose in poses.values():
        if pose is None:
            continue
        pose = np.asarray(pose, dtype=np.float32)
        valid = np.isfinite(pose).all(axis=-1)
        if np.any(valid):
            points.append(pose[valid])
    if not points:
        return np.zeros((1, 3), dtype=np.float32)
    return np.concatenate(points, axis=0)


def _set_equal_3d_limits(ax, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def comparison_reference_pose(poses: dict[str, np.ndarray | None]) -> np.ndarray | None:
    return poses.get("Pseudo reference")


def reference_center_and_radius(
    poses: dict[str, np.ndarray | None],
    scale_reference_pose: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    reference = scale_reference_pose if scale_reference_pose is not None else comparison_reference_pose(poses)
    points = reference if reference is not None else _finite_pose_points(poses)
    points = np.asarray(points, dtype=np.float32)
    valid = np.isfinite(points).all(axis=-1)
    if not np.any(valid):
        return np.zeros(3, dtype=np.float32), 1.0
    points = points[valid]
    low = np.nanmin(points, axis=0)
    high = np.nanmax(points, axis=0)
    center = ((low + high) * 0.5).astype(np.float32)
    radius = max(float(np.nanmax(high - low)) * 0.65, 1e-3)
    return center, radius


def _draw_pose_3d(
    ax,
    pose: np.ndarray,
    official_edges: Sequence[tuple[int, int, tuple[float, float, float]]],
    keypoint_colors: Sequence[tuple[float, float, float]],
    edge_alpha: float,
    point_alpha: float,
    linewidth: float,
    point_size: float,
    use_official_colors: bool = True,
) -> None:
    pose = np.asarray(pose, dtype=np.float32)
    valid = np.isfinite(pose).all(axis=-1)
    for a, b, color in official_edges:
        if a < pose.shape[0] and b < pose.shape[0] and valid[a] and valid[b]:
            pa = pose[a]
            pb = pose[b]
            ax.plot(
                [pa[0], pb[0]],
                [pa[1], pb[1]],
                [pa[2], pb[2]],
                color=color if use_official_colors else (0.72, 0.72, 0.72),
                linewidth=linewidth,
                alpha=edge_alpha,
            )
    colors = np.asarray(keypoint_colors, dtype=np.float32)
    if not use_official_colors:
        colors = np.repeat(np.asarray([[0.72, 0.72, 0.72]], dtype=np.float32), pose.shape[0], axis=0)
    ax.scatter(
        pose[valid, 0],
        pose[valid, 1],
        pose[valid, 2],
        c=colors[valid],
        s=point_size,
        alpha=point_alpha,
        depthshade=True,
    )


def _edges_and_colors_for_pose(
    pose: np.ndarray,
    compact_edges: Sequence[tuple[int, int, tuple[float, float, float]]],
    compact_colors: Sequence[tuple[float, float, float]],
    full_edges: Sequence[tuple[int, int, tuple[float, float, float]]],
    full_colors: Sequence[tuple[float, float, float]],
) -> tuple[Sequence[tuple[int, int, tuple[float, float, float]]], Sequence[tuple[float, float, float]]]:
    if np.asarray(pose).shape[0] >= 70:
        return full_edges, full_colors
    return compact_edges, compact_colors


def render_official_sam3d_pose_grid(
    poses: dict[str, np.ndarray | None],
    output_path: Path,
    title: str,
    official_edges: Sequence[tuple[int, int, tuple[float, float, float]]],
    keypoint_colors: Sequence[tuple[float, float, float]],
    full_edges: Sequence[tuple[int, int, tuple[float, float, float]]] | None = None,
    full_keypoint_colors: Sequence[tuple[float, float, float]] | None = None,
    overlay_reference_pose: np.ndarray | None = None,
    scale_reference_pose: np.ndarray | None = None,
) -> None:
    full_edges = list(full_edges or official_edges)
    full_keypoint_colors = list(full_keypoint_colors or keypoint_colors)
    reference_pose = overlay_reference_pose if overlay_reference_pose is not None else comparison_reference_pose(poses)
    ref_center, ref_radius = reference_center_and_radius(poses, scale_reference_pose=scale_reference_pose)
    fig = plt.figure(figsize=(16, 4.6), dpi=180)
    fig.suptitle(f"{title} | model 52-keypoint space, reference aligned to model body center", fontsize=12)
    for panel_idx, (label, pose) in enumerate(poses.items(), start=1):
        ax = fig.add_subplot(1, len(poses), panel_idx, projection="3d")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        _set_equal_3d_limits(ax, ref_center, ref_radius)
        ax.view_init(elev=-30, azim=270)
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.set_box_aspect((1, 1, 1))
        if pose is None:
            ax.text(0.0, 0.0, 0.0, "prediction not provided", ha="center")
            continue
        if reference_pose is not None and label != "Pseudo reference":
            _draw_pose_3d(
                ax,
                reference_pose,
                official_edges,
                keypoint_colors,
                edge_alpha=0.22,
                point_alpha=0.22,
                linewidth=1.2,
                point_size=5,
                use_official_colors=False,
            )
        pose_edges, pose_colors = _edges_and_colors_for_pose(
            pose,
            official_edges,
            keypoint_colors,
            full_edges,
            full_keypoint_colors,
        )
        _draw_pose_3d(
            ax,
            pose,
            pose_edges,
            pose_colors,
            edge_alpha=0.92,
            point_alpha=0.94,
            linewidth=2.0,
            point_size=8,
            use_official_colors=True,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path)
    plt.close(fig)


def render_all_samples(
    sam3d_root: Path,
    gt_root: Path,
    output_root: Path,
    sam3d_repo: Path,
    samples_per_env: int,
    single_view: str,
    reference_panel: str = "full70",
) -> tuple[list[dict], list[dict]]:
    pose_info = load_official_mhr70_pose_info(sam3d_repo)
    official_edges = compact_official_skeleton(pose_info)
    keypoint_colors = compact_official_keypoint_colors(pose_info)
    full_edges = full_official_skeleton(pose_info)
    full_keypoint_colors = full_official_keypoint_colors(pose_info)
    manifest = []
    errors = []
    subjects = sorted(path.name for path in Path(gt_root).iterdir() if path.is_dir())
    for subject in subjects:
        for raw_env, env_label in ENV_NAMES.items():
            try:
                view_files = {
                    cam: list_sam3d_files(Path(sam3d_root) / subject / raw_env / cam)
                    for cam in ("front", "left", "right")
                }
                gt_path = Path(gt_root) / subject / raw_env / "keypoints_3d.npz"
                gt_pose, _, gt_frame_ids = load_gt_sequence(gt_path)
                with np.load(gt_path, allow_pickle=False) as full_gt:
                    full_gt_valid = np.asarray(full_gt["valid_mask"], dtype=bool)
                    full_gt_frame_ids = [normalize_frame_id(item) for item in np.asarray(full_gt["frame_ids"]).tolist()]
                full_gt_lookup = {fid: idx for idx, fid in enumerate(full_gt_frame_ids)}
                aligned_ids, _ = select_common_frame_ids(
                    view_files,
                    gt_frame_ids=gt_frame_ids,
                    gt_num_frames=gt_pose.shape[0],
                )
                selected_indices = select_hand_complete_frame_indices(
                    aligned_ids,
                    full_gt_lookup,
                    full_gt_valid,
                    samples_per_env=samples_per_env,
                )
                for sample_no, frame_index in enumerate(selected_indices):
                    poses, _, title = build_comparison(
                        sam3d_root=Path(sam3d_root),
                        gt_root=Path(gt_root),
                        subject=subject,
                        env=raw_env,
                        trifusion_pred=None,
                        single_view=single_view,
                        frame_id=None,
                        frame_index=frame_index,
                        do_canonicalize=True,
                        allow_missing_trifusion=True,
                        infer_trifusion=True,
                    )
                    frame_id = aligned_ids[frame_index]
                    full_gt_index = full_gt_lookup[normalize_frame_id(frame_id)]
                    right_hand_ratio, left_hand_ratio = _hand_valid_ratios(full_gt_valid[full_gt_index])
                    model_reference_pose = poses.get("Pseudo reference")
                    full_reference_pose = load_full_triangulated_pose(
                        gt_root=Path(gt_root),
                        subject=subject,
                        env_folder=raw_env,
                        frame_id=frame_id,
                        canonicalize=True,
                    )
                    full_reference_pose, full_reference_scale = align_full_pose_to_model_reference(
                        full_reference_pose,
                        model_reference_pose,
                    )
                    display_target_pose = display_scale_target_pose(poses)
                    if reference_panel == "full70":
                        poses["Pseudo reference"] = full_reference_pose
                        pseudo_reference_panel = "full_mhr70_triangulated_gt"
                        comparison_policy = "model_52_keypoints_fixed_scale_with_full70_triangulated_reference_panel"
                        display_reference_pose, reference_display_scale = align_pose_scale_with_anchor_translation(
                            full_reference_pose,
                            display_target_pose,
                            source_indices=KEEP_KEYPOINT_INDICES,
                            target_indices=range(len(KEEP_KEYPOINT_INDICES)),
                        )
                    elif reference_panel == "model52":
                        display_reference_pose, reference_display_scale = align_pose_scale_with_anchor_translation(
                            model_reference_pose,
                            display_target_pose,
                        )
                        poses["Pseudo reference"] = display_reference_pose
                        pseudo_reference_panel = "model_52_triangulated_gt_with_wrist_bridge"
                        comparison_policy = "model_52_keypoints_reference_scaled_and_body_center_aligned_to_model"
                    else:
                        raise ValueError(f"Unsupported reference_panel={reference_panel!r}")
                    if reference_panel == "full70":
                        poses["Pseudo reference"] = display_reference_pose
                    out_path = Path(output_root) / subject / env_label / f"sample_{sample_no:02d}_frame_{frame_id}.png"
                    render_official_sam3d_pose_grid(
                        poses,
                        out_path,
                        title,
                        official_edges,
                        keypoint_colors,
                        full_edges=full_edges,
                        full_keypoint_colors=full_keypoint_colors,
                        overlay_reference_pose=display_reference_pose,
                        scale_reference_pose=display_reference_pose,
                    )
                    manifest.append(
                        {
                            "subject": subject,
                            "environment": env_label,
                            "raw_environment": raw_env,
                            "sample": sample_no,
                            "frame_id": frame_id,
                            "frame_index": frame_index,
                            "right_hand_valid_ratio": right_hand_ratio,
                            "left_hand_valid_ratio": left_hand_ratio,
                            "full_reference_to_model52_scale": full_reference_scale,
                            "reference_display_scale_to_model": reference_display_scale,
                            "output": str(out_path),
                            "renderer": "SAM3Dbody official MHR70 metadata + Matplotlib 3D skeleton visualization",
                            "comparison_policy": comparison_policy,
                            "pseudo_reference_panel": pseudo_reference_panel,
                            "sam3d_repo": str(sam3d_repo),
                            "trifusion_prediction": "inferred_confidence_fusion",
                        }
                    )
            except Exception as exc:
                errors.append({"subject": subject, "environment": env_label, "error": str(exc)})
    return manifest, errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render qualitative pose samples using SAM3D Body official MHR70 skeleton metadata.")
    parser.add_argument("--sam3d-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("TriPoseFusion/eval/logs/qualitative_pose_samples_official_sam3d"))
    parser.add_argument("--sam3d-repo", type=Path, default=DEFAULT_SAM3D_REPO)
    parser.add_argument("--samples-per-env", type=int, default=3)
    parser.add_argument("--single-view", choices=("front", "left", "right"), default="front")
    parser.add_argument("--reference-panel", choices=("model52", "full70"), default="full70")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest, errors = render_all_samples(
        sam3d_root=args.sam3d_root,
        gt_root=args.gt_root,
        output_root=args.output_root,
        sam3d_repo=args.sam3d_repo,
        samples_per_env=args.samples_per_env,
        single_view=args.single_view,
        reference_panel=args.reference_panel,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps({"samples": manifest, "errors": errors}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote_samples={len(manifest)}")
    print(f"errors={len(errors)}")
    print(f"manifest={manifest_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
