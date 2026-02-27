#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Label utilities

- Load label JSON into dict: {label: [{start, end}, ...]}
- Automatically fill unlabeled gaps as `front`
- Convert to a time-ordered timeline list:
    [{"label": "front", "start": 0, "end": 5}, {"label":"left","start":5,"end":10}, ...]
- Print label stats: names + counts
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------
# Loading labels
# ---------------------------------------------------------------------
def load_label_dict(
    json_path: str | Path,
    annotator: Optional[int] = None,
    annotation_id: Optional[int] = None,
) -> Dict[str, Dict[str, List[dict]]]:
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    anns = data.get("annotations", [])
    if not isinstance(anns, list):
        raise ValueError("Invalid json: 'annotations' must be a list")

    filtered_anns = []
    for a in anns:
        if annotation_id is not None and a.get("annotation_id") != annotation_id:
            continue
        if annotator is not None and a.get("annotator") != annotator:
            continue
        filtered_anns.append(a)

    chosen = filtered_anns if (annotator is not None or annotation_id is not None) else anns

    grouped: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))

    for ann in chosen:
        annotator_key = str(ann.get("annotator", "unknown"))
        video_labels = ann.get("videoLabels", [])
        if not isinstance(video_labels, list):
            continue

        for item in video_labels:
            labels = item.get("timelinelabels", [])
            ranges = item.get("ranges", [])
            if not labels or not ranges:
                continue

            for lb in labels:
                for r in ranges:
                    if r is None:
                        continue
                    s = r.get("start")
                    e = r.get("end")
                    if s is None or e is None:
                        continue
                    grouped[annotator_key][str(lb)].append(
                        {"start": float(s), "end": float(e)}
                    )

    grouped_out: Dict[str, Dict[str, List[dict]]] = {}
    for annotator_key, label_dict in grouped.items():
        cleaned: Dict[str, List[dict]] = {}
        for lb, segs in label_dict.items():
            cleaned[lb] = sorted(segs, key=lambda x: (x["start"], x["end"]))
        grouped_out[annotator_key] = cleaned

    return grouped_out


# ---------------------------------------------------------------------
# Timeline conversion (your requested output)
# ---------------------------------------------------------------------
def label_dict_to_timeline(
    label_dict: Dict[str, List[dict]],
    *,
    eps: float = 1e-9,
    sort: bool = True,
) -> List[dict]:
    """
    Convert {label:[{start,end},...]} -> [{"label":lb,"start":s,"end":e}, ...] sorted by time.
    """
    timeline: List[dict] = []
    for lb, segs in label_dict.items():
        for r in segs:
            s, e = r.get("start"), r.get("end")
            if s is None or e is None:
                continue
            s = float(s)
            e = float(e)
            if e <= s + eps:
                continue
            timeline.append({"label": str(lb), "start": s, "end": e})

    if sort:
        timeline.sort(key=lambda x: (x["start"], x["end"], x["label"]))

    return timeline


def print_label_stats(label_dict: Dict[str, List[dict]]) -> None:
    names = sorted(label_dict.keys())
    counts = {k: len(v) for k, v in label_dict.items()}
    total = sum(counts.values())

    print(f"[labels] names({len(names)}): {names}")
    print(f"[labels] total_segments: {total}")
    for k in names:
        print(f"  - {k}: {counts[k]} segments")


# ---------------------------------------------------------------------
# Main prepare function
# ---------------------------------------------------------------------
def prepare_label_dict(
    path: str | Path,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    annotator: Optional[int] = None,
    annotation_id: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
        Returns a dict grouped by annotator:
            {
                "<annotator_id>": {
                    "label_dict": {label:[{start,end},...]},
                    "timeline_list": [{"label":..., "start":..., "end":...}, ...]
                }
            }

    Args:
        path: Path to label JSON file
        start_frame: Optional start frame to filter labels (absolute frame index)
        end_frame: Optional end frame to filter labels (absolute frame index)
        annotator: Optional annotator id filter.
        annotation_id: Optional annotation id filter.

    Returns:
        Dict grouped by annotator
        
    Note:
        If start_frame and end_frame are provided, only labels overlapping 
        with [start_frame, end_frame) will be included in the result.
    """
    grouped = load_label_dict(
        path,
        annotator=annotator,
        annotation_id=annotation_id,
    )

    annotator_dict: Dict[str, Dict[str, Any]] = {}

    for annotator_key, labels in grouped.items():
        timeline_list = label_dict_to_timeline(labels)

        if start_frame is not None or end_frame is not None:
            _start = start_frame if start_frame is not None else 0
            _end = end_frame if end_frame is not None else float("inf")

            filtered_timeline = []
            for seg in timeline_list:
                seg_start = seg["start"]
                seg_end = seg["end"]

                if seg_end <= _start or seg_start >= _end:
                    continue

                clipped_start = max(_start, seg_start)
                clipped_end = min(_end, seg_end)

                if clipped_end <= clipped_start:
                    continue

                filtered_timeline.append(
                    {
                        "label": seg["label"],
                        "start": clipped_start,
                        "end": clipped_end,
                    }
                )

            timeline_list = filtered_timeline

        annotator_dict[annotator_key] = {
            "label_dict": labels,
            "timeline_list": timeline_list,
        }

    return annotator_dict