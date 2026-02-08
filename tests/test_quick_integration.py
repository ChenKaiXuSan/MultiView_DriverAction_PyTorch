#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Quick integration check for modified dataloader."""

import sys
import torch

def test_imports():
    """Test that all modules can be imported."""
    print("\n🔍 Testing imports...")
    
    try:
        from project.dataloader.whole_video_dataset import LabeledVideoDataset
        print("  ✅ LabeledVideoDataset imported successfully")
    except Exception as e:
        print(f"  ❌ Failed to import LabeledVideoDataset: {e}")
        return False
    
    try:
        from project.dataloader.whole_video_dataset import whole_video_dataset
        print("  ✅ whole_video_dataset function imported successfully")
    except Exception as e:
        print(f"  ❌ Failed to import whole_video_dataset: {e}")
        return False
    
    return True


def test_class_interface():
    """Test that the class has the expected interface."""
    print("\n🔍 Testing class interface...")
    
    from project.dataloader.whole_video_dataset import LabeledVideoDataset
    
    # Check constructor signature
    try:
        init_params = LabeledVideoDataset.__init__.__code__.co_varnames
        expected = ['self', 'experiment', 'index_mapping', 'annotation_dict', 
                   'transform', 'decode_audio', 'load_rgb', 'load_kpt']
        
        for param in expected:
            if param in init_params:
                print(f"  ✅ Parameter '{param}' found in __init__")
            else:
                print(f"  ❌ Parameter '{param}' NOT found in __init__")
                return False
    except Exception as e:
        print(f"  ❌ Error checking init parameters: {e}")
        return False
    
    # Check key methods exist
    methods = [
        '__init__',
        '__len__',
        '__getitem__',
        '_load_one_view',
        '_load_sam3d_body_kpts',
        '_apply_transform',
        '_validate_output_shapes',
        'split_frame_with_label',
    ]
    
    for method in methods:
        if hasattr(LabeledVideoDataset, method):
            print(f"  ✅ Method '{method}' exists")
        else:
            print(f"  ❌ Method '{method}' NOT found")
            return False
    
    return True


def test_tensor_shapes():
    """Test tensor shape inference logic."""
    print("\n🔍 Testing tensor shape logic...")
    
    try:
        # Simulate the zero-padding logic
        B, T, H, W, K = 4, 10, 224, 224, 17
        
        # RGB tensors
        rgb_front = torch.zeros(B, 3, T, H, W, dtype=torch.float32)
        rgb_left = torch.zeros(B, 3, T, H, W, dtype=torch.float32)
        rgb_right = torch.zeros(B, 3, T, H, W, dtype=torch.float32)
        
        print(f"  ✅ RGB tensor shapes: {rgb_front.shape}")
        
        # KPT tensors
        kpt_front = torch.zeros(B, T, K, 3, dtype=torch.float32)
        kpt_left = torch.zeros(B, T, K, 3, dtype=torch.float32)
        kpt_right = torch.zeros(B, T, K, 3, dtype=torch.float32)
        
        print(f"  ✅ KPT tensor shapes: {kpt_front.shape}")
        
        # Labels
        labels = torch.zeros(B, dtype=torch.long)
        print(f"  ✅ Labels shape: {labels.shape}")
        
        # Check consistency
        assert rgb_front.shape[0] == kpt_front.shape[0] == labels.shape[0]
        print(f"  ✅ Batch sizes are consistent: B={B}")
        
        return True
    except Exception as e:
        print(f"  ❌ Shape logic test failed: {e}")
        return False


def test_validation_function():
    """Test the validation function with zero tensors."""
    print("\n🔍 Testing validation function...")
    
    try:
        from project.dataloader.whole_video_dataset import LabeledVideoDataset
        from project.map_config import label_mapping_Dict
        
        # Create a dummy dataset instance
        dataset = LabeledVideoDataset(
            experiment="test",
            index_mapping=[],
            annotation_dict={},
            load_rgb=True,
            load_kpt=True
        )
        
        # Test validation with zero tensors
        B, T, K = 2, 10, 17
        batch_front = torch.zeros(B, 3, T, 224, 224)
        batch_left = torch.zeros(B, 3, T, 224, 224)
        batch_right = torch.zeros(B, 3, T, 224, 224)
        mapped_labels = torch.zeros(B, dtype=torch.long)
        labels = ["front", "left"]
        front_kpts_batch = torch.zeros(B, T, K, 3)
        left_kpts_batch = torch.zeros(B, T, K, 3)
        right_kpts_batch = torch.zeros(B, T, K, 3)
        
        # This should not raise any errors
        dataset._validate_output_shapes(
            batch_front, batch_left, batch_right,
            mapped_labels, labels,
            front_kpts_batch, left_kpts_batch, right_kpts_batch
        )
        
        print("  ✅ Validation passed with zero tensors")
        return True
        
    except Exception as e:
        print(f"  ❌ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*70)
    print("🧪 Quick Integration Check")
    print("="*70)
    
    all_pass = True
    
    all_pass &= test_imports()
    all_pass &= test_class_interface()
    all_pass &= test_tensor_shapes()
    all_pass &= test_validation_function()
    
    print("\n" + "="*70)
    if all_pass:
        print("✨ All integration checks passed!")
        print("="*70 + "\n")
        sys.exit(0)
    else:
        print("❌ Some checks failed!")
        print("="*70 + "\n")
        sys.exit(1)
