#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/MultiView_DriverAction_PyTorch/project/dataloader/annotation_dict.py
Project: /workspace/MultiView_DriverAction_PyTorch/project/dataloader
Created Date: Thursday February 5th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday February 5th 2026 10:12:01 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

MAPPING = {
    "night_high": "夜多い",
    "night_low": "夜少ない",
    "day_high": "昼多い",
    "day_low": "昼少ない",
}


def get_annotation_dict(file_path: str) -> dict:
    """
    JSONファイルからアノテーション情報を抽出し、
    { person_id: { env_name: { label: frame_num } } } の形式で返します。
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Annotation file not found: {file_path}")
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    master_dict = {}

    for item in data:
        video_path = item.get("video", "")
        file_name = os.path.basename(video_path)
        parts = file_name.split("_")

        # インデックスエラーを防ぐためのガード
        if len(parts) < 4:
            continue

        person = f"{parts[0]}_{parts[1]}"
        env_key = f"{parts[2]}_{parts[3]}"
        env_name = MAPPING.get(env_key, env_key)

        # 辞書の初期化を setdefault で簡潔に
        person_entry = master_dict.setdefault(person, {})

        # 必要なラベル情報を抽出
        frames = {"start": None, "mid": None, "end": None}
        for label_obj in item.get("videoLabels", []):
            labels = label_obj.get("timelinelabels", [])
            if not labels:
                continue

            label_name = labels[0]
            if label_name in frames:
                # ranges[0]['start'] が存在するか安全に取得
                ranges = label_obj.get("ranges", [{}])
                frames[label_name] = ranges[0].get("start")

        person_entry[env_name] = frames

    return master_dict
