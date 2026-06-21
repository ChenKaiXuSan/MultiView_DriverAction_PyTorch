#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common import (  # noqa: E402
    CAMERAS,
    ENV_NAMES,
    canonicalize_pose,
    fuse_views,
    list_sam3d_files,
    load_gt_sequence,
    load_selected_sam3d_frames,
    normalize_frame_id,
    select_common_frame_ids,
)

PANEL_COLORS = {
    "Single-view": "#d95f02",
    "Median fusion": "#1f78b4",
    "TriPoseFusion": "#2ca25f",
    "TriPoseFusion (inferred)": "#2ca25f",
    "Pseudo reference": "#222222",
}
SIDE_COLORS = {
    "left": "#e45756",
    "right": "#4c78a8",
    "center": "#6b7280",
}
HAND_LOCAL_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
DEFAULT_SEMANTIC_EDGES = (
    (51, 0, "center"),
    (51, 5, "left"),
    (5, 49, "left"),
    (49, 28, "left"),
    (51, 6, "right"),
    (6, 50, "right"),
    (50, 7, "right"),
    *tuple((28 + a, 28 + b, "left") for a, b in HAND_LOCAL_EDGES),
    *tuple((7 + a, 7 + b, "right") for a, b in HAND_LOCAL_EDGES),
)
DEFAULT_EDGES = tuple((a, b) for a, b, _ in DEFAULT_SEMANTIC_EDGES)
PREDICTION_KEYS = ("P_final", "pred", "prediction", "predictions", "keypoints_3d", "poses", "pose")


def resolve_env_folder(env: str) -> str:
    reverse = {label: raw for raw, label in ENV_NAMES.items()}
    return reverse.get(env, env)


def parse_edges(text: str | None) -> List[Tuple[int, int]]:
    if not text:
        return list(DEFAULT_EDGES)
    edges: List[Tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            left, right = item.split("-", maxsplit=1)
            edges.append((int(left), int(right)))
        except ValueError as exc:
            raise ValueError(f"Invalid edge '{item}'. Use comma-separated pairs like 51-5,5-7.") from exc
    return edges


def joint_side(joint_index: int) -> str:
    if joint_index in {5, 49} or 28 <= joint_index <= 48:
        return "left"
    if joint_index in {6, 50} or 7 <= joint_index <= 27:
        return "right"
    return "center"


def edge_side(a: int, b: int) -> str:
    sides = {joint_side(a), joint_side(b)}
    if "left" in sides and "right" not in sides:
        return "left"
    if "right" in sides and "left" not in sides:
        return "right"
    return "center"


def _semantic_edges(
    edges: Sequence[Tuple[int, int] | Tuple[int, int, str]] | None,
) -> List[Tuple[int, int, str]]:
    if edges is None:
        return list(DEFAULT_SEMANTIC_EDGES)
    semantic = []
    for edge in edges:
        if len(edge) == 3:
            a, b, side = edge
            semantic.append((int(a), int(b), str(side)))
        else:
            a, b = edge
            semantic.append((int(a), int(b), edge_side(int(a), int(b))))
    return semantic


def _prediction_array_from_npz(path: Path) -> Tuple[np.ndarray, List[str] | None]:
    with np.load(path, allow_pickle=False) as data:
        key = next((name for name in PREDICTION_KEYS if name in data.files), None)
        if key is None:
            raise KeyError(f"No prediction array in {path}. Expected one of {PREDICTION_KEYS}; keys={data.files}")
        predictions = np.asarray(data[key], dtype=np.float32)
        frame_ids = (
            [normalize_frame_id(item) for item in np.asarray(data["frame_ids"]).tolist()]
            if "frame_ids" in data.files
            else None
        )
    return predictions, frame_ids


def _squeeze_prediction_sequence(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float32)
    while predictions.ndim > 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim == 2 and predictions.shape[-1] >= 3:
        return predictions[None, :, :3]
    if predictions.ndim == 3 and predictions.shape[-1] >= 3:
        return predictions[:, :, :3]
    raise ValueError(f"Prediction array must have shape (T,J,3) or (J,3), got {predictions.shape}")


def load_prediction_pose(path: Path, frame_id: str | None = None, frame_index: int = 0) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npz":
        predictions, frame_ids = _prediction_array_from_npz(path)
    elif path.suffix == ".npy":
        predictions = np.load(path, allow_pickle=False)
        frame_ids = None
    else:
        raise ValueError(f"Unsupported prediction file type: {path.suffix}. Use .npz or .npy.")

    sequence = _squeeze_prediction_sequence(predictions)
    if frame_id is not None and frame_ids is not None:
        lookup = {normalize_frame_id(fid): idx for idx, fid in enumerate(frame_ids)}
        normalized = normalize_frame_id(frame_id)
        if normalized not in lookup:
            raise KeyError(f"Frame id {frame_id} not found in {path}")
        return sequence[lookup[normalized]].astype(np.float32)
    if frame_index < 0 or frame_index >= sequence.shape[0]:
        raise IndexError(f"frame_index {frame_index} out of range for {sequence.shape[0]} predictions")
    return sequence[frame_index].astype(np.float32)


def infer_trifusion_pose(view_pose: np.ndarray, view_conf: np.ndarray) -> np.ndarray:
    """Approximate TriPoseFusion P_init with confidence-gated multi-view fusion."""
    return fuse_views(view_pose, view_conf, "confidence")[0]


def _axes_for_projection(projection: str) -> Tuple[int, int]:
    projections = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if projection not in projections:
        raise ValueError(f"Unsupported projection {projection}. Use one of {sorted(projections)}")
    return projections[projection]


def _valid_points(pose: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    valid = np.isfinite(pose).all(axis=-1)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)[: pose.shape[0]]
    return valid


def _global_bounds(
    poses: Mapping[str, np.ndarray],
    valid_masks: Mapping[str, np.ndarray] | None,
    projection: str,
) -> Tuple[np.ndarray, float]:
    axes = _axes_for_projection(projection)
    points = []
    for label, pose in poses.items():
        if pose is None:
            continue
        valid = _valid_points(pose, None if valid_masks is None else valid_masks.get(label))
        if np.any(valid):
            points.append(np.asarray(pose)[valid][:, axes])
    if not points:
        return np.zeros(2, dtype=np.float32), 1.0
    merged = np.concatenate(points, axis=0)
    low = np.nanmin(merged, axis=0)
    high = np.nanmax(merged, axis=0)
    center = ((low + high) * 0.5).astype(np.float32)
    span = float(np.nanmax(high - low))
    return center, max(span, 1e-6)


def _point_to_svg(point: np.ndarray, center: np.ndarray, span: float, box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, width, height = box
    scale = 0.78 * min(width, height) / span
    x = x0 + width * 0.5 + float(point[0] - center[0]) * scale
    y = y0 + height * 0.56 - float(point[1] - center[1]) * scale
    return x, y


def _axis_endpoint(
    origin: Tuple[float, float],
    length: float,
    axis: str,
) -> Tuple[float, float]:
    ox, oy = origin
    if axis == "x":
        return ox + length, oy
    if axis == "y":
        return ox, oy - length
    return ox - length * 0.62, oy + length * 0.44


def _append_axis_widget(parts: List[str], box: Tuple[float, float, float, float]) -> None:
    x0, y0, _, height = box
    origin = (x0 + 32, y0 + height - 28)
    axes = (
        ("x", "#e45756", "X"),
        ("y", "#54a24b", "Y"),
        ("z", "#4c78a8", "Z"),
    )
    parts.append(f'<circle cx="{origin[0]:.2f}" cy="{origin[1]:.2f}" r="2.4" fill="#475569"/>')
    for axis, color, label in axes:
        ex, ey = _axis_endpoint(origin, 28.0, axis)
        parts.append(
            f'<line class="axis axis-{axis}" x1="{origin[0]:.2f}" y1="{origin[1]:.2f}" '
            f'x2="{ex:.2f}" y2="{ey:.2f}" stroke="{color}" stroke-width="2.4" '
            'stroke-linecap="round"/>'
        )
        parts.append(
            f'<text x="{ex:.2f}" y="{ey - 4:.2f}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="10" font-weight="700" fill="{color}">{label}</text>'
        )


def render_svg(
    poses: Mapping[str, np.ndarray | None],
    edges: Sequence[Tuple[int, int]] | None = None,
    valid_masks: Mapping[str, np.ndarray] | None = None,
    title: str = "",
    projection: str = "xy",
    style: str = "sam3d-body",
) -> str:
    semantic_edges = _semantic_edges(edges)
    panel_labels = list(poses.keys())
    sam3d_style = style == "sam3d-body"
    panel_width = 270 if sam3d_style else 250
    panel_height = 300 if sam3d_style else 280
    margin = 30
    title_height = 46
    width = margin * 2 + panel_width * len(panel_labels)
    height = title_height + panel_height + 28
    center, span = _global_bounds(poses, valid_masks, projection)
    axes = _axes_for_projection(projection)
    background = "#f6f8fb" if sam3d_style else "#ffffff"
    panel_fill = "#ffffff" if sam3d_style else "#fafafa"
    panel_stroke = "#dce3eb" if sam3d_style else "#d8d8d8"
    line_width = 5.0 if sam3d_style else 2.6
    joint_radius = 4.7 if sam3d_style else 3.2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{background}"/>',
    ]
    if title:
        parts.append(
            f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="16" fill="#222222">'
            f"{html.escape(title)}</text>"
        )

    for idx, label in enumerate(panel_labels):
        x0 = margin + idx * panel_width
        y0 = title_height
        box = (x0 + 18, y0 + 42, panel_width - 36, panel_height - 62)
        color = PANEL_COLORS.get(label, "#444444")
        parts.append(
            f'<text x="{x0 + panel_width / 2:.1f}" y="{y0 + 22}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#222222">'
            f"{html.escape(label)}</text>"
        )
        parts.append(
            f'<rect x="{box[0]:.1f}" y="{box[1]:.1f}" width="{box[2]:.1f}" height="{box[3]:.1f}" rx="6" '
            f'fill="{panel_fill}" stroke="{panel_stroke}" stroke-width="1"/>'
        )
        if sam3d_style:
            _append_axis_widget(parts, box)
        pose = poses[label]
        if pose is None:
            parts.append(
                f'<text x="{x0 + panel_width / 2:.1f}" y="{y0 + panel_height / 2:.1f}" text-anchor="middle" '
                'font-family="Arial, sans-serif" font-size="13" fill="#777777">prediction not provided</text>'
            )
            continue

        pose = np.asarray(pose, dtype=np.float32)
        valid = _valid_points(pose, None if valid_masks is None else valid_masks.get(label))
        projected = pose[:, axes]
        points = [_point_to_svg(projected[j], center, span, box) for j in range(pose.shape[0])]

        for a, b, side in semantic_edges:
            if 0 <= a < pose.shape[0] and 0 <= b < pose.shape[0] and valid[a] and valid[b]:
                x1, y1 = points[a]
                x2, y2 = points[b]
                limb_color = SIDE_COLORS.get(side, color) if sam3d_style else color
                if sam3d_style:
                    parts.append(
                        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                        'stroke="#ffffff" stroke-width="8.2" stroke-linecap="round" opacity="0.92"/>'
                    )
                parts.append(
                    f'<line class="limb side-{side}" x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                    f'stroke="{limb_color}" stroke-width="{line_width:.1f}" stroke-linecap="round" opacity="0.88"/>'
                )
        for joint_idx, (x, y) in enumerate(points):
            if valid[joint_idx]:
                side = joint_side(joint_idx)
                joint_color = SIDE_COLORS.get(side, color) if sam3d_style else color
                if sam3d_style:
                    parts.append(
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{joint_radius + 1.8:.1f}" '
                        'fill="#ffffff" opacity="0.94"/>'
                    )
                parts.append(
                    f'<circle class="joint side-{side}" cx="{x:.2f}" cy="{y:.2f}" r="{joint_radius:.1f}" fill="{joint_color}" '
                    'stroke="#ffffff" stroke-width="1.1"/>'
                )
    parts.append("</svg>")
    return "\n".join(parts)


def _select_frame_index(frame_ids: Sequence[str], requested_frame_id: str | None, requested_index: int | None) -> Tuple[str, int]:
    if requested_frame_id is not None:
        normalized = normalize_frame_id(requested_frame_id)
        lookup = {normalize_frame_id(fid): idx for idx, fid in enumerate(frame_ids)}
        if normalized not in lookup:
            raise KeyError(f"Frame id {requested_frame_id} is not in the aligned sequence")
        return frame_ids[lookup[normalized]], lookup[normalized]
    if requested_index is None:
        requested_index = len(frame_ids) // 2
    if requested_index < 0 or requested_index >= len(frame_ids):
        raise IndexError(f"frame_index {requested_index} out of range for {len(frame_ids)} aligned frames")
    return frame_ids[requested_index], requested_index


def build_comparison(
    sam3d_root: Path,
    gt_root: Path,
    subject: str,
    env: str,
    trifusion_pred: Path | None,
    single_view: str,
    frame_id: str | None,
    frame_index: int | None,
    do_canonicalize: bool,
    allow_missing_trifusion: bool,
    infer_trifusion: bool = False,
) -> Tuple[Dict[str, np.ndarray | None], Dict[str, np.ndarray], str]:
    env_folder = resolve_env_folder(env)
    view_files = {
        cam: list_sam3d_files(Path(sam3d_root) / subject / env_folder / cam)
        for cam in CAMERAS
    }
    gt_pose, gt_valid, gt_frame_ids = load_gt_sequence(Path(gt_root) / subject / env_folder / "keypoints_3d.npz")
    aligned_ids, gt_indices = select_common_frame_ids(
        view_files,
        gt_frame_ids=gt_frame_ids,
        gt_num_frames=gt_pose.shape[0],
    )
    selected_frame_id, aligned_index = _select_frame_index(aligned_ids, frame_id, frame_index)
    selected_ids = [selected_frame_id]
    selected_gt_index = gt_indices[aligned_index]

    loaded_views = {
        cam: load_selected_sam3d_frames(view_files[cam], selected_ids)
        for cam in CAMERAS
    }
    view_pose = np.stack([loaded_views[cam][0] for cam in CAMERAS], axis=2)
    view_conf = np.stack([loaded_views[cam][1] for cam in CAMERAS], axis=2)
    gt_frame = gt_pose[selected_gt_index : selected_gt_index + 1]
    gt_valid_frame = gt_valid[selected_gt_index]

    if do_canonicalize:
        view_pose = np.stack(
            [
                canonicalize_pose(view_pose[:, :, cam_idx, :])
                for cam_idx in range(view_pose.shape[2])
            ],
            axis=2,
        )
        gt_frame = canonicalize_pose(gt_frame)

    single_pose = view_pose[0, :, CAMERAS.index(single_view), :]
    median_pose = fuse_views(view_pose, view_conf, "median")[0]

    if trifusion_pred is None:
        if infer_trifusion:
            trifusion_pose = infer_trifusion_pose(view_pose, view_conf)
        elif not allow_missing_trifusion:
            raise ValueError("Please pass --trifusion-pred, or use --allow-missing-trifusion for a placeholder panel.")
        else:
            trifusion_pose = None
        trifusion_label = "TriPoseFusion (inferred)" if infer_trifusion else "TriPoseFusion"
    else:
        trifusion_pose = load_prediction_pose(trifusion_pred, frame_id=selected_frame_id, frame_index=aligned_index)
        if do_canonicalize:
            trifusion_pose = canonicalize_pose(trifusion_pose[None, :, :])[0]
        trifusion_label = "TriPoseFusion"

    label = ENV_NAMES.get(env_folder, env_folder)
    title = f"subject={subject} env={label} frame={selected_frame_id}"
    poses = {
        "Single-view": single_pose,
        "Median fusion": median_pose,
        trifusion_label: trifusion_pose,
        "Pseudo reference": gt_frame[0],
    }
    valid_masks = {"Pseudo reference": gt_valid_frame}
    return poses, valid_masks, title


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a qualitative pose comparison SVG: single-view / median fusion / TriPoseFusion / pseudo reference."
    )
    parser.add_argument("--sam3d-root", type=Path, required=True, help="Root directory of SAM3D body results.")
    parser.add_argument("--gt-root", type=Path, required=True, help="Root directory of triangulated pseudo GT.")
    parser.add_argument("--subject", required=True, help="Subject id, for example 01.")
    parser.add_argument("--env", required=True, help="Environment folder or label, for example Day_High or 昼多い.")
    parser.add_argument("--trifusion-pred", type=Path, help="Optional .npz/.npy containing P_final or pred array.")
    parser.add_argument("--single-view", choices=CAMERAS, default="front")
    parser.add_argument("--frame-id", help="Frame id to render. If omitted, --frame-index is used.")
    parser.add_argument("--frame-index", type=int, help="Aligned frame index to render. Defaults to the middle frame.")
    parser.add_argument("--projection", choices=("xy", "xz", "yz"), default="xy")
    parser.add_argument("--style", choices=("sam3d-body", "simple"), default="sam3d-body")
    parser.add_argument("--edges", help="Skeleton edges, for example 51-5,51-6,5-7,7-9.")
    parser.add_argument("--output", type=Path, default=Path("qualitative_pose_comparison.svg"))
    parser.add_argument("--no-canonicalize", action="store_true", help="Render raw coordinates instead of canonicalized poses.")
    parser.add_argument(
        "--allow-missing-trifusion",
        action="store_true",
        help="Render a placeholder TriPoseFusion panel when no prediction file is available.",
    )
    parser.add_argument(
        "--infer-trifusion",
        action="store_true",
        help="Approximate TriPoseFusion with confidence-gated multi-view fusion when no prediction file is available.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    poses, valid_masks, title = build_comparison(
        sam3d_root=args.sam3d_root,
        gt_root=args.gt_root,
        subject=args.subject,
        env=args.env,
        trifusion_pred=args.trifusion_pred,
        single_view=args.single_view,
        frame_id=args.frame_id,
        frame_index=args.frame_index,
        do_canonicalize=not args.no_canonicalize,
        allow_missing_trifusion=args.allow_missing_trifusion,
        infer_trifusion=args.infer_trifusion,
    )
    svg = render_svg(
        poses,
        edges=parse_edges(args.edges),
        valid_masks=valid_masks,
        title=title,
        projection=args.projection,
        style=args.style,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
