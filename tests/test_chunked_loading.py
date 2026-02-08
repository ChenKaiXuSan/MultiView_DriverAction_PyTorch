#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试video分块加载功能

Usage:
    python test_chunked_loading.py
"""
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from project.dataloader.whole_video_dataset import whole_video_dataset
from project.dataloader.annotation_dict import get_annotation_dict
from project.map_config import get_index_mapping

def test_chunked_loading():
    """测试分块加载功能"""
    
    print("=" * 80)
    print("测试Video分块加载功能")
    print("=" * 80)
    
    # 加载数据
    print("\n1. 加载index mapping和annotation...")
    data_root = Path("./data")
    index_file = data_root / "index_mapping" / "index.json"
    annotation_dir = data_root / "annotation" / "label"
    
    if not index_file.exists():
        print(f"❌ Index file not found: {index_file}")
        return
    
    index_mapping = get_index_mapping(str(index_file))
    annotation_dict = get_annotation_dict(str(annotation_dir))
    
    print(f"✅ 加载了 {len(index_mapping)} 个videos")
    
    # 测试1: 不分块
    print("\n" + "=" * 80)
    print("测试1: 不分块加载 (max_video_frames=None)")
    print("=" * 80)
    
    dataset_no_chunk = whole_video_dataset(
        experiment="test_no_chunk",
        dataset_idx=index_mapping[:5],  # 只测试前5个
        annotation_dict=annotation_dict,
        load_rgb=True,
        load_kpt=False,
        max_video_frames=None,  # 不分块
    )
    
    print(f"Dataset length: {len(dataset_no_chunk)}")
    print(f"原始videos数量: 5")
    print(f"Dataset samples: {len(dataset_no_chunk)}")
    
    # 测试2: 分块加载
    print("\n" + "=" * 80)
    print("测试2: 分块加载 (max_video_frames=1000)")
    print("=" * 80)
    
    dataset_chunked = whole_video_dataset(
        experiment="test_chunked",
        dataset_idx=index_mapping[:5],  # 只测试前5个
        annotation_dict=annotation_dict,
        load_rgb=True,
        load_kpt=False,
        max_video_frames=1000,  # 分块
    )
    
    print(f"Dataset length: {len(dataset_chunked)}")
    print(f"原始videos数量: 5")
    print(f"Dataset chunks: {len(dataset_chunked)}")
    print(f"平均每个video被分成: {len(dataset_chunked) / 5:.1f} 个chunks")
    
    # 测试3: 加载一个chunk
    print("\n" + "=" * 80)
    print("测试3: 加载第一个chunk")
    print("=" * 80)
    
    sample = dataset_chunked[0]
    
    print(f"✅ 成功加载chunk!")
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"\nVideo shapes:")
    for view in ['front', 'left', 'right']:
        shape = sample['video'][view].shape
        print(f"  {view}: {shape}  # (B, C, T, H, W)")
    
    print(f"\nMeta info:")
    meta = sample['meta']
    for key, value in meta.items():
        print(f"  {key}: {value}")
    
    # 测试4: 验证chunk信息
    print("\n" + "=" * 80)
    print("测试4: 验证chunk信息")
    print("=" * 80)
    
    if meta.get('is_chunked'):
        chunk_info = meta['chunk_info']
        print(f"✅ 这是一个chunked sample")
        print(f"  Chunk索引: {chunk_info['chunk_idx']} / {chunk_info['total_chunks']}")
        print(f"  帧范围: {chunk_info['chunk_start_frame']} - {chunk_info['chunk_end_frame']}")
    else:
        print(f"❌ 不是chunked sample（预期应该是）")
    
    # 测试5: 加载多个chunks
    print("\n" + "=" * 80)
    print("测试5: 加载前3个chunks的形状")
    print("=" * 80)
    
    for i in range(min(3, len(dataset_chunked))):
        sample = dataset_chunked[i]
        shape = sample['video']['front'].shape
        meta = sample['meta']
        chunk_info = meta.get('chunk_info', {})
        
        print(f"\nChunk {i}:")
        print(f"  Person: {meta['person_id']}, Env: {meta['env_folder']}")
        print(f"  Shape: {shape}")
        if chunk_info:
            print(f"  Chunk: {chunk_info['chunk_idx']}/{chunk_info['total_chunks']}")
            print(f"  Frames: {chunk_info['chunk_start_frame']}-{chunk_info['chunk_end_frame']}")
    
    # 测试6: 内存估算
    print("\n" + "=" * 80)
    print("测试6: 内存估算")
    print("=" * 80)
    
    sample = dataset_chunked[0]
    video_tensor = sample['video']['front']
    
    # 计算单个view的内存
    bytes_per_element = 4  # float32
    elements = video_tensor.numel()
    mb_per_view = (elements * bytes_per_element) / (1024 * 1024)
    mb_total = mb_per_view * 3  # 3 views
    
    print(f"单个chunk内存占用:")
    print(f"  单个view: {mb_per_view:.2f} MB")
    print(f"  三个views: {mb_total:.2f} MB")
    print(f"  Batch={2}: {mb_total * 2:.2f} MB")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试完成!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        test_chunked_loading()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
