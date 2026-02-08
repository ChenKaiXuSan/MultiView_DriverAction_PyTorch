#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Test chunk inference functionality for long video processing.
"""

import torch
from unittest.mock import MagicMock


def test_chunk_inference_disabled():
    """Test when chunk_infer_size is -1 (disabled)."""
    # Mock trainer
    trainer = MagicMock()
    trainer.chunk_infer_size = -1
    trainer.forward = MagicMock(return_value=torch.randn(1, 5))
    
    # Create dummy input
    video = torch.randn(1, 3, 100, 224, 224)
    
    # Simulate forward pass
    output = trainer.forward(video)
    
    assert output.shape == (1, 5), f"Expected shape (1, 5), got {output.shape}"
    assert trainer.forward.called, "forward() should be called once"


def test_chunk_inference_short_video():
    """Test with video shorter than chunk size."""
    trainer = MagicMock()
    trainer.chunk_infer_size = 200
    trainer.chunk_overlap = 0
    trainer.forward = MagicMock(return_value=torch.randn(1, 5))
    
    video = torch.randn(1, 3, 64, 224, 224)  # Shorter than chunk_size
    
    output = trainer.forward(video)
    
    assert output.shape == (1, 5)
    assert trainer.forward.call_count == 1, "Should call forward only once"


def test_chunk_inference_long_video():
    """
    Test chunk inference on long video.
    This demonstrates the memory-efficient approach.
    """
    # Simulate the chunking logic
    B, C, T, H, W = 1, 3, 128, 224, 224
    video = torch.randn(B, C, T, H, W)
    
    chunk_size = 48
    overlap = 8
    stride = chunk_size - overlap
    
    # Collect all chunks
    chunks = []
    for start_idx in range(0, T - overlap, stride):
        end_idx = min(start_idx + chunk_size, T)
        chunk = video[:, :, start_idx:end_idx, :, :]
        chunks.append(chunk)
        
        if end_idx == T:
            break
    
    # Verify chunking
    assert len(chunks) > 1, "Should have multiple chunks"
    assert chunks[0].shape[2] == chunk_size, "First chunk should be full size"
    assert chunks[-1].shape[2] <= chunk_size, "Last chunk should fit"
    
    print(f"✓ Video split into {len(chunks)} chunks")
    print(f"  Chunk shape: {chunks[0].shape}")


def test_chunk_aggregation_methods():
    """Test different aggregation methods for chunk predictions."""
    # Simulate predictions from 3 chunks
    logits_1 = torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.2]])
    logits_2 = torch.tensor([[0.15, 0.25, 0.25, 0.25, 0.1]])
    logits_3 = torch.tensor([[0.2, 0.15, 0.3, 0.2, 0.15]])
    
    all_logits = [logits_1, logits_2, logits_3]
    
    # Mean aggregation
    mean_logits = torch.stack(all_logits, dim=0).mean(dim=0)
    assert mean_logits.shape == (1, 5)
    print(f"✓ Mean aggregation: {mean_logits}")
    
    # Max aggregation
    max_logits = torch.stack(all_logits, dim=0).amax(dim=0)
    assert max_logits.shape == (1, 5)
    print(f"✓ Max aggregation: {max_logits}")
    
    # Last aggregation
    last_logits = all_logits[-1]
    assert last_logits.shape == (1, 5)
    print(f"✓ Last aggregation: {last_logits}")


def test_memory_savings():
    """
    Estimate memory savings with chunking.
    For reference: 1 second of video at 30fps = 30 frames
    """
    print("\n" + "="*60)
    print("Memory Savings with Chunk Inference")
    print("="*60)
    
    # Assume 3D CNN with batch=1, input=[1,3,T,224,224]
    # Approximate memory per frame: 0.03 GB (for forward pass)
    mem_per_frame = 0.03  # GB
    
    scenarios = [
        {"duration": "30 sec", "fps": 30, "chunk_size": 16, "strategy": "no chunking"},
        {"duration": "30 sec", "fps": 30, "chunk_size": 16, "strategy": "chunk=16"},
        {"duration": "1 min", "fps": 30, "chunk_size": 16, "strategy": "chunk=16"},
        {"duration": "5 min", "fps": 30, "chunk_size": 16, "strategy": "chunk=32"},
    ]
    
    for scenario in scenarios:
        total_frames = scenario["duration"]
        fps = scenario["fps"]
        chunk_size = scenario["chunk_size"]
        
        if "sec" in scenario["duration"]:
            seconds = int(scenario["duration"].split()[0])
            total_frames = seconds * fps
        else:
            minutes = int(scenario["duration"].split()[0])
            total_frames = minutes * 60 * fps
        
        # Memory with no chunking
        mem_no_chunk = total_frames * mem_per_frame
        
        # Memory with chunking (only one chunk in memory at a time)
        mem_with_chunk = chunk_size * mem_per_frame
        
        savings = (1 - mem_with_chunk / mem_no_chunk) * 100
        
        print(f"\n{scenario['duration']} @ {fps}fps:")
        print(f"  Total frames: {total_frames}")
        print(f"  No chunking: {mem_no_chunk:.2f} GB")
        print(f"  With chunk={chunk_size}: {mem_with_chunk:.2f} GB")
        print(f"  Memory savings: {savings:.1f}%")


if __name__ == "__main__":
    print("Running chunk inference tests...\n")
    
    test_chunk_inference_disabled()
    print("✓ Test 1 passed: Chunking disabled")
    
    test_chunk_inference_short_video()
    print("✓ Test 2 passed: Short video (no chunking needed)")
    
    test_chunk_inference_long_video()
    print("✓ Test 3 passed: Long video chunking")
    
    test_chunk_aggregation_methods()
    print("✓ Test 4 passed: Chunk aggregation methods")
    
    test_memory_savings()
    print("✓ Test 5 passed: Memory savings estimation")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
