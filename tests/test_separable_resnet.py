#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: test_separable_resnet.py
Project: tests
Created Date: 2026-02-09
Author: Kaixu Chen
-----
Comment:
Test script for separable ResNet 3D CNN structure.

Tests that stem, body, and head can be used separately and
produce the same results as the unified model.

Have a good code time :)
-----
"""

import torch
import sys
import os
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from project.models.res_3dcnn import Res3DCNN
from project.models.base_model import BaseModel


def create_dummy_hparams():
    """Create dummy hyperparameters for testing."""
    config = {
        'model': {
            'model_class_num': 9,
        }
    }
    return OmegaConf.create(config)


def test_separable_structure():
    """Test that separable structure works correctly."""
    print("="*60)
    print("Testing Separable ResNet 3D CNN Structure")
    print("="*60)
    
    # Create dummy input with sufficient size for pooling
    batch_size = 2
    channels = 3
    time_steps = 16
    height = 224
    width = 224
    
    video = torch.randn(batch_size, channels, time_steps, height, width)
    
    # Test 1: Standard mode
    print("\n1. Testing standard mode...")
    hparams = create_dummy_hparams()
    model_standard = Res3DCNN(hparams, use_separable=False)
    model_standard.eval()
    
    with torch.no_grad():
        output_standard = model_standard(video)
    
    print(f"   Standard output shape: {output_standard.shape}")
    assert output_standard.shape == (batch_size, 9), "Standard output shape mismatch"
    
    # Test 2: Separable mode
    print("\n2. Testing separable mode...")
    model_separable = Res3DCNN(hparams, use_separable=True)
    model_separable.eval()
    
    with torch.no_grad():
        output_separable = model_separable(video)
    
    print(f"   Separable output shape: {output_separable.shape}")
    assert output_separable.shape == (batch_size, 9), "Separable output shape mismatch"
    
    # Test 3: Manual forward through separable components
    print("\n3. Testing manual forward through stem->body->head...")
    with torch.no_grad():
        stem_out = model_separable.stem(video)
        print(f"   After stem: {stem_out.shape}")
        
        body_out = model_separable.body(stem_out)
        print(f"   After body: {body_out.shape}")
        
        head_out = model_separable.head(body_out)
        print(f"   After head: {head_out.shape}")
    
    assert head_out.shape == (batch_size, 9), "Manual forward output shape mismatch"
    
    # Test 4: Verify outputs are the same
    print("\n4. Verifying manual forward equals model forward...")
    assert torch.allclose(output_separable, head_out, atol=1e-5), "Outputs should be identical"
    print("   ✓ Outputs match!")
    
    # Test 5: Test feature extraction
    print("\n5. Testing feature extraction...")
    with torch.no_grad():
        features_standard = model_standard.forward_features(video)
        features_separable = model_separable.forward_features(video)
    
    print(f"   Standard features shape: {features_standard.shape}")
    print(f"   Separable features shape: {features_separable.shape}")
    
    assert features_standard.shape[0] == batch_size, "Batch size mismatch in features"
    assert features_separable.shape[0] == batch_size, "Batch size mismatch in features"
    
    # Test 6: Test getter methods
    print("\n6. Testing getter methods...")
    stem = model_separable.get_stem()
    body = model_separable.get_body()
    head = model_separable.get_head()
    
    print(f"   Stem type: {type(stem).__name__}")
    print(f"   Body type: {type(body).__name__}")
    print(f"   Head type: {type(head).__name__}")
    
    assert stem is not None, "Stem should not be None"
    assert body is not None, "Body should not be None"
    assert head is not None, "Head should not be None"
    
    # Test 7: Test backward pass
    print("\n7. Testing backward pass...")
    model_separable.train()
    output = model_separable(video)
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist
    has_gradients = False
    for name, param in model_separable.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    assert has_gradients, "Model should have gradients after backward pass"
    print("   ✓ Gradients flow correctly!")
    
    print("\n" + "="*60)
    print("✓ ALL SEPARABLE STRUCTURE TESTS PASSED!")
    print("="*60)


def test_base_model_init_resnet_separable():
    """Test the init_resnet_separable static method."""
    print("\n" + "="*60)
    print("Testing BaseModel.init_resnet_separable()")
    print("="*60)
    
    # Test without feature_dim
    print("\n1. Testing without feature_dim return...")
    stem, body, head = BaseModel.init_resnet_separable(class_num=9)
    
    print(f"   Stem type: {type(stem).__name__}")
    print(f"   Body type: {type(body).__name__}")
    print(f"   Head type: {type(head).__name__}")
    
    assert stem is not None
    assert body is not None
    assert head is not None
    
    # Test with feature_dim
    print("\n2. Testing with feature_dim return...")
    stem, body, head, feature_dim = BaseModel.init_resnet_separable(
        class_num=9, 
        return_feature_dim=True
    )
    
    print(f"   Feature dimension: {feature_dim}")
    assert isinstance(feature_dim, int), "Feature dim should be int"
    assert feature_dim > 0, "Feature dim should be positive"
    
    # Test forward pass
    print("\n3. Testing forward pass through components...")
    batch_size = 1
    video = torch.randn(batch_size, 3, 16, 224, 224)
    
    with torch.no_grad():
        x = stem(video)
        print(f"   After stem: {x.shape}")
        
        x = body(x)
        print(f"   After body: {x.shape}")
        
        x = head(x)
        print(f"   After head: {x.shape}")
    
    assert x.shape == (batch_size, 9), "Final output shape should be (B, num_classes)"
    
    print("\n✓ init_resnet_separable tests passed!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Separable ResNet 3D CNN Testing Suite")
    print("="*60)
    
    try:
        test_base_model_init_resnet_separable()
        test_separable_structure()
        
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
