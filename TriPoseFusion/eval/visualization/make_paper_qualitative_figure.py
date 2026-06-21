#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

DEFAULT_SAMPLES_ROOT = Path(
    "/home/workspace/kaixu/code/multiview_trifusion/"
    "TriPoseFusion/eval/logs/"
    "qualitative_pose_samples_hand_complete_model52_reference_body_center_aligned"
)
DEFAULT_PAPER_FIGURES = Path(
    "/home/workspace/kaixu/code/multiview_trifusion/"
    "6a366b68482ae5ce3f81c758/figures"
)
DEFAULT_ROWS = (
    ("(a) Day-high, subject 07, frame 1943", Path("07/Day_High/sample_01_frame_1943.png")),
    ("(b) Day-low, subject 06, frame 1292", Path("06/Day_Low/sample_01_frame_1292.png")),
    ("(c) Night-low, subject 08, frame 5340", Path("08/Night_Low/sample_01_frame_5340.png")),
)


def parse_row(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError("Rows must use LABEL=relative/path.png")
    label, rel_path = text.split("=", maxsplit=1)
    label = label.strip()
    rel_path = rel_path.strip()
    if not label or not rel_path:
        raise ValueError("Rows must include both LABEL and relative path")
    return label, Path(rel_path)


def compose_qualitative_figure(
    samples_root: Path,
    output_png: Path,
    output_pdf: Path,
    rows: Sequence[tuple[str, Path]] = DEFAULT_ROWS,
    row_width: int = 2300,
    label_height: int = 58,
    row_gap: int = 6,
    title_font_size: int = 40,
    source_crop_top: int = 70,
) -> None:
    font = _load_title_font(title_font_size)
    rendered_rows = []
    for label, relative_path in rows:
        source_path = samples_root / relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"Missing qualitative sample: {source_path}")
        image = Image.open(source_path).convert("RGB")
        if source_crop_top > 0:
            crop_top = min(source_crop_top, image.height - 1)
            image = image.crop((0, crop_top, image.width, image.height))
        scale = row_width / image.width
        resized = image.resize((row_width, int(image.height * scale)), Image.LANCZOS)
        row = Image.new("RGB", (row_width, resized.height + label_height), "white")
        row.paste(resized, (0, label_height))
        draw = ImageDraw.Draw(row)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = max(0, (row_width - text_width) // 2 - bbox[0])
        text_y = max(0, (label_height - text_height) // 2 - bbox[1])
        draw.text((text_x, text_y), label, fill=(25, 25, 25), font=font)
        rendered_rows.append(row)

    output_height = sum(row.height for row in rendered_rows) + row_gap * (len(rendered_rows) - 1)
    output = Image.new("RGB", (row_width, output_height), "white")
    y = 0
    for row in rendered_rows:
        output.paste(row, (0, y))
        y += row.height + row_gap

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_png, optimize=True)
    output.save(output_pdf, "PDF", resolution=300.0)


def _load_title_font(font_size: int) -> ImageFont.ImageFont:
    for font_name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose the paper qualitative pose comparison figure.")
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_PAPER_FIGURES / "fig_qualitative_pose.png")
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_PAPER_FIGURES / "fig_qualitative_pose.pdf")
    parser.add_argument("--row-width", type=int, default=2300)
    parser.add_argument("--label-height", type=int, default=58)
    parser.add_argument("--row-gap", type=int, default=6)
    parser.add_argument("--title-font-size", type=int, default=40)
    parser.add_argument(
        "--source-crop-top",
        type=int,
        default=70,
        help="Pixels to crop from the top of each source sample before composing.",
    )
    parser.add_argument(
        "--row",
        action="append",
        type=parse_row,
        help="Override default rows. Format: LABEL=relative/path.png. Pass once per row.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = args.row if args.row else DEFAULT_ROWS
    compose_qualitative_figure(
        samples_root=args.samples_root,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
        rows=rows,
        row_width=args.row_width,
        label_height=args.label_height,
        row_gap=args.row_gap,
        title_font_size=args.title_font_size,
        source_crop_top=args.source_crop_top,
    )
    print(f"wrote_png={args.output_png}")
    print(f"wrote_pdf={args.output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
