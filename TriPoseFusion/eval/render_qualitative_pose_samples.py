#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common import (  # noqa: E402
    ENV_NAMES,
    list_sam3d_files,
    load_gt_sequence,
    select_common_frame_ids,
)
from eval.render_qualitative_pose_comparison import (  # noqa: E402
    build_comparison,
    parse_edges,
    render_svg,
)


def choose_sample_indices(num_frames: int, samples_per_env: int) -> list[int]:
    if num_frames <= 0 or samples_per_env <= 0:
        return []
    count = min(num_frames, samples_per_env)
    return [int(idx) for idx in np.linspace(0, num_frames - 1, num=count, dtype=np.int64)]


def available_subjects(gt_root: Path) -> list[str]:
    return sorted(path.name for path in Path(gt_root).iterdir() if path.is_dir())


def render_all_samples(
    sam3d_root: Path,
    gt_root: Path,
    output_root: Path,
    samples_per_env: int,
    single_view: str,
    projection: str,
    style: str,
    infer_trifusion: bool,
    allow_missing_trifusion: bool,
) -> tuple[list[dict], list[dict]]:
    manifest: list[dict] = []
    errors: list[dict] = []
    edges = parse_edges(None)

    for subject in available_subjects(gt_root):
        for raw_env, env_label in ENV_NAMES.items():
            try:
                view_files = {
                    cam: list_sam3d_files(Path(sam3d_root) / subject / raw_env / cam)
                    for cam in ("front", "left", "right")
                }
                gt_pose, _, gt_frame_ids = load_gt_sequence(Path(gt_root) / subject / raw_env / "keypoints_3d.npz")
                aligned_ids, _ = select_common_frame_ids(
                    view_files,
                    gt_frame_ids=gt_frame_ids,
                    gt_num_frames=gt_pose.shape[0],
                )
                for sample_no, frame_index in enumerate(choose_sample_indices(len(aligned_ids), samples_per_env)):
                    poses, valid_masks, title = build_comparison(
                        sam3d_root=Path(sam3d_root),
                        gt_root=Path(gt_root),
                        subject=subject,
                        env=raw_env,
                        trifusion_pred=None,
                        single_view=single_view,
                        frame_id=None,
                        frame_index=frame_index,
                        do_canonicalize=True,
                        allow_missing_trifusion=allow_missing_trifusion,
                        infer_trifusion=infer_trifusion,
                    )
                    frame_id = aligned_ids[frame_index]
                    svg = render_svg(
                        poses,
                        edges=edges,
                        valid_masks=valid_masks,
                        title=title,
                        projection=projection,
                        style=style,
                    )
                    out_path = Path(output_root) / subject / env_label / f"sample_{sample_no:02d}_frame_{frame_id}.svg"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(svg, encoding="utf-8")
                    manifest.append(
                        {
                            "subject": subject,
                            "environment": env_label,
                            "raw_environment": raw_env,
                            "sample": sample_no,
                            "frame_id": frame_id,
                            "frame_index": frame_index,
                            "output": str(out_path),
                            "trifusion_prediction": "inferred_confidence_fusion" if infer_trifusion else None,
                        }
                    )
            except Exception as exc:
                errors.append({"subject": subject, "environment": env_label, "error": str(exc)})
    return manifest, errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render qualitative pose SVG samples for all subjects and environments.")
    parser.add_argument("--sam3d-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("TriPoseFusion/eval/logs/qualitative_pose_samples"))
    parser.add_argument("--samples-per-env", type=int, default=3)
    parser.add_argument("--single-view", choices=("front", "left", "right"), default="front")
    parser.add_argument("--projection", choices=("xy", "xz", "yz"), default="xy")
    parser.add_argument("--style", choices=("sam3d-body", "simple"), default="sam3d-body")
    parser.add_argument("--infer-trifusion", action="store_true")
    parser.add_argument("--allow-missing-trifusion", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest, errors = render_all_samples(
        sam3d_root=args.sam3d_root,
        gt_root=args.gt_root,
        output_root=args.output_root,
        samples_per_env=args.samples_per_env,
        single_view=args.single_view,
        projection=args.projection,
        style=args.style,
        infer_trifusion=args.infer_trifusion,
        allow_missing_trifusion=args.allow_missing_trifusion,
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
