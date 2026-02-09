#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: test_ts_cva.py
Project: tests
Created Date: 2026-02-09
Author: Kaixu Chen
-----
Comment:
Test script for TS-CVA model to verify forward pass and gradient flow.

Have a good code time :)
-----
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from project.models.ts_cva_model import TSCVAModel, MultiHeadSelfAttention, LearnableGatedAggregation, TemporalConvNet


def create_dummy_hparams():
    """Create dummy hyperparameters for testing."""
    config = {
        'model': {
            'model_class_num': 9,
            'ts_cva_shared_backbone': True,
            'ts_cva_use_view_embedding': True,
            'ts_cva_use_gated_aggregation': True,
            'ts_cva_num_heads': 4,
            'ts_cva_temporal_dim': 512,
            'ts_cva_temporal_layers': 2,
        }
    }
    return OmegaConf.create(config)


def test_multi_head_attention():
    """Test MultiHeadSelfAttention module."""
    print("\n" + "="*60)
    print("Testing MultiHeadSelfAttention...")
    print("="*60)
    
    batch_size = 2
    num_views = 3
    embed_dim = 256
    num_heads = 4
    
    # Create module
    mhsa = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # Create dummy input
    x = torch.randn(batch_size, num_views, embed_dim)
    
    # Forward pass
    output, attn_weights = mhsa(x)
    
    # Check shapes
    assert output.shape == (batch_size, num_views, embed_dim), f"Expected output shape {(batch_size, num_views, embed_dim)}, got {output.shape}"
    assert attn_weights.shape == (batch_size, num_heads, num_views, num_views), f"Expected attention shape {(batch_size, num_heads, num_views, num_views)}, got {attn_weights.shape}"
    
    # Check attention weights sum to 1
    attn_sum = attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), "Attention weights should sum to 1"
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    print("✓ MultiHeadSelfAttention test passed!")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Attention weights shape: {attn_weights.shape}")
    print(f"  - Attention weights sum check: PASSED")
    print(f"  - Gradient flow: PASSED")


def test_gated_aggregation():
    """Test LearnableGatedAggregation module."""
    print("\n" + "="*60)
    print("Testing LearnableGatedAggregation...")
    print("="*60)
    
    batch_size = 2
    num_views = 3
    embed_dim = 256
    
    # Create module
    aggregation = LearnableGatedAggregation(embed_dim=embed_dim, num_views=num_views)
    
    # Create dummy input
    x = torch.randn(batch_size, num_views, embed_dim)
    
    # Forward pass
    fused, weights = aggregation(x)
    
    # Check shapes
    assert fused.shape == (batch_size, embed_dim), f"Expected fused shape {(batch_size, embed_dim)}, got {fused.shape}"
    assert weights.shape == (batch_size, num_views), f"Expected weights shape {(batch_size, num_views)}, got {weights.shape}"
    
    # Check weights sum to 1
    weights_sum = weights.sum(dim=-1)
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5), "Gate weights should sum to 1"
    
    # Test backward pass
    loss = fused.sum()
    loss.backward()
    
    print("✓ LearnableGatedAggregation test passed!")
    print(f"  - Fused shape: {fused.shape}")
    print(f"  - Gate weights shape: {weights.shape}")
    print(f"  - Gate weights sum check: PASSED")
    print(f"  - Gradient flow: PASSED")


def test_temporal_conv_net():
    """Test TemporalConvNet module."""
    print("\n" + "="*60)
    print("Testing TemporalConvNet...")
    print("="*60)
    
    batch_size = 2
    seq_length = 8
    input_dim = 256
    hidden_dim = 512
    
    # Create module
    tcn = TemporalConvNet(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_length, input_dim)
    
    # Forward pass
    output = tcn(x)
    
    # Check shape
    assert output.shape == (batch_size, hidden_dim), f"Expected output shape {(batch_size, hidden_dim)}, got {output.shape}"
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    print("✓ TemporalConvNet test passed!")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Gradient flow: PASSED")


def test_ts_cva_model():
    """Test complete TS-CVA model."""
    print("\n" + "="*60)
    print("Testing TS-CVA Model (Full Pipeline)...")
    print("="*60)
    
    # Create dummy hyperparameters
    hparams = create_dummy_hparams()
    
    # Create model
    model = TSCVAModel(hparams)
    
    # Create dummy input videos
    batch_size = 2
    channels = 3
    time_steps = 16
    height = 224
    width = 224
    
    videos = {
        'front': torch.randn(batch_size, channels, time_steps, height, width),
        'left': torch.randn(batch_size, channels, time_steps, height, width),
        'right': torch.randn(batch_size, channels, time_steps, height, width),
    }
    
    print(f"\nInput shapes:")
    for view, video in videos.items():
        print(f"  - {view}: {video.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    logits = model(videos, return_attention=True)
    
    # Check output shape
    num_classes = hparams.model.model_class_num
    assert logits.shape == (batch_size, num_classes), f"Expected logits shape {(batch_size, num_classes)}, got {logits.shape}"
    
    print(f"✓ Output shape: {logits.shape}")
    
    # Check attention weights
    attn_weights = model.get_attention_weights()
    gate_weights = model.get_gate_weights()
    
    if attn_weights is not None:
        print(f"✓ Attention weights shape: {attn_weights.shape}")
    if gate_weights is not None:
        print(f"✓ Gate weights shape: {gate_weights.shape}")
        
        # Check gate weights sum to 1
        gate_sum = gate_weights.sum(dim=-1)
        assert torch.allclose(gate_sum, torch.ones_like(gate_sum), atol=1e-5), "Gate weights should sum to 1"
        print(f"✓ Gate weights sum check: PASSED")
    
    # Test backward pass
    print("\nTesting backward pass...")
    loss = logits.sum()
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    assert has_gradients, "Model should have gradients after backward pass"
    print("✓ Gradient flow: PASSED")
    
    print("\n✓ TS-CVA Model test passed!")


def test_ablation_modes():
    """Test ablation study configurations."""
    print("\n" + "="*60)
    print("Testing Ablation Modes...")
    print("="*60)
    
    # Use smaller inputs for ablation tests to avoid OOM
    batch_size = 1
    channels = 3
    time_steps = 8
    height = 112
    width = 112
    
    videos = {
        'front': torch.randn(batch_size, channels, time_steps, height, width),
        'left': torch.randn(batch_size, channels, time_steps, height, width),
        'right': torch.randn(batch_size, channels, time_steps, height, width),
    }
    
    # Test 1: Without gated aggregation (mean pooling)
    print("\n1. Testing without gated aggregation...")
    config = create_dummy_hparams()
    config.model.ts_cva_use_gated_aggregation = False
    model1 = TSCVAModel(config)
    logits1 = model1(videos)
    print(f"   ✓ Mean pooling mode: {logits1.shape}")
    
    # Test 2: Without view embeddings
    print("\n2. Testing without view embeddings...")
    config = create_dummy_hparams()
    config.model.ts_cva_use_view_embedding = False
    model2 = TSCVAModel(config)
    logits2 = model2(videos)
    print(f"   ✓ No view embedding mode: {logits2.shape}")
    
    # Test 3: Non-shared backbone
    print("\n3. Testing non-shared backbone...")
    config = create_dummy_hparams()
    config.model.ts_cva_shared_backbone = False
    model3 = TSCVAModel(config)
    logits3 = model3(videos)
    print(f"   ✓ Non-shared backbone mode: {logits3.shape}")
    
    # Test 4: Different number of attention heads
    print("\n4. Testing different number of attention heads...")
    config = create_dummy_hparams()
    config.model.ts_cva_num_heads = 8
    model4 = TSCVAModel(config)
    logits4 = model4(videos)
    print(f"   ✓ 8 attention heads mode: {logits4.shape}")
    
    print("\n✓ All ablation modes test passed!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TS-CVA Model Testing Suite")
    print("="*60)
    
    try:
        # Test individual components
        test_multi_head_attention()
        test_gated_aggregation()
        test_temporal_conv_net()
        
        # Test full model
        test_ts_cva_model()
        
        # Test ablation modes
        test_ablation_modes()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED!")
        print("="*60)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
