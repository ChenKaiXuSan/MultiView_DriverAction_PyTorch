#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: example_ts_cva_usage.py
Project: examples
Created Date: 2026-02-09
Author: Kaixu Chen
-----
Comment:
Example script demonstrating how to use TS-CVA model for training and inference.

Usage:
    # Train with default configuration
    python examples/example_ts_cva_usage.py --mode train
    
    # Run inference with trained model
    python examples/example_ts_cva_usage.py --mode inference
    
    # Visualize attention and gate weights
    python examples/example_ts_cva_usage.py --mode visualize

Have a good code time :)
-----
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf

# Example: Create dummy data for demonstration
def create_dummy_video_batch(batch_size=2, num_frames=16, img_size=224):
    """Create dummy multi-view video data for testing."""
    videos = {
        'front': torch.randn(batch_size, 3, num_frames, img_size, img_size),
        'left': torch.randn(batch_size, 3, num_frames, img_size, img_size),
        'right': torch.randn(batch_size, 3, num_frames, img_size, img_size),
    }
    labels = torch.randint(0, 9, (batch_size,))
    return videos, labels


def example_training():
    """
    Example: Training TS-CVA model
    
    In practice, you would use the main training script:
    python project/main.py --config-name config_ts_cva
    """
    print("="*60)
    print("Example: TS-CVA Training")
    print("="*60)
    
    # This would typically be done via Hydra config
    from project.models.ts_cva_model import TSCVAModel
    
    # Create configuration
    config = OmegaConf.create({
        'model': {
            'model_class_num': 9,
            'ts_cva_shared_backbone': True,
            'ts_cva_use_view_embedding': True,
            'ts_cva_use_gated_aggregation': True,
            'ts_cva_num_heads': 4,
            'ts_cva_temporal_dim': 512,
            'ts_cva_temporal_layers': 2,
        }
    })
    
    # Initialize model
    model = TSCVAModel(config)
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop example (simplified)
    print("\nTraining for 3 dummy batches...")
    for epoch in range(3):
        videos, labels = create_dummy_video_batch(batch_size=2)
        
        # Forward pass
        logits = model(videos)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")
    
    print("\n✓ Training example completed!")
    print("\nIn practice, use the full training script:")
    print("  python project/main.py --config-name config_ts_cva")


def example_inference():
    """Example: Running inference with TS-CVA model"""
    print("="*60)
    print("Example: TS-CVA Inference")
    print("="*60)
    
    from project.models.ts_cva_model import TSCVAModel
    
    # Create configuration
    config = OmegaConf.create({
        'model': {
            'model_class_num': 9,
            'ts_cva_shared_backbone': True,
            'ts_cva_use_view_embedding': True,
            'ts_cva_use_gated_aggregation': True,
            'ts_cva_num_heads': 4,
            'ts_cva_temporal_dim': 512,
            'ts_cva_temporal_layers': 2,
        }
    })
    
    # Initialize model
    model = TSCVAModel(config)
    model.eval()
    
    print("\nRunning inference on dummy batch...")
    with torch.no_grad():
        videos, labels = create_dummy_video_batch(batch_size=1)
        
        # Forward pass with attention tracking
        logits = model(videos, return_attention=True)
        
        # Get predictions
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1)
        
        print(f"\nPredicted class: {pred_class.item()}")
        print(f"Confidence: {probs[0, pred_class].item():.4f}")
        
        # Get attention and gate weights
        attention_weights = model.get_attention_weights()
        gate_weights = model.get_gate_weights()
        
        if attention_weights is not None:
            print(f"\nAttention weights shape: {attention_weights.shape}")
            print(f"Gate weights shape: {gate_weights.shape}")
            
            # Average gate weights over time to see overall view importance
            avg_gate_weights = gate_weights.mean(dim=1)[0]
            print(f"\nAverage view importance:")
            print(f"  Front: {avg_gate_weights[0].item():.3f}")
            print(f"  Left:  {avg_gate_weights[1].item():.3f}")
            print(f"  Right: {avg_gate_weights[2].item():.3f}")
    
    print("\n✓ Inference example completed!")


def example_visualization():
    """Example: Visualizing attention and gate weights"""
    print("="*60)
    print("Example: TS-CVA Visualization")
    print("="*60)
    
    from project.models.ts_cva_model import TSCVAModel
    from project.utils.ts_cva_visualization import (
        plot_gate_weights_curve,
        plot_attention_heatmap,
        plot_temporal_attention_evolution
    )
    
    # Create configuration
    config = OmegaConf.create({
        'model': {
            'model_class_num': 9,
            'ts_cva_shared_backbone': True,
            'ts_cva_use_view_embedding': True,
            'ts_cva_use_gated_aggregation': True,
            'ts_cva_num_heads': 4,
            'ts_cva_temporal_dim': 512,
            'ts_cva_temporal_layers': 2,
        }
    })
    
    # Initialize model
    model = TSCVAModel(config)
    model.eval()
    
    # Create output directory
    output_dir = Path("ts_cva_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating visualizations in: {output_dir.absolute()}")
    
    with torch.no_grad():
        videos, labels = create_dummy_video_batch(batch_size=1)
        
        # Forward pass with attention tracking
        logits = model(videos, return_attention=True)
        
        # Get weights
        attention_weights = model.get_attention_weights()
        gate_weights = model.get_gate_weights()
        
        # 1. Plot gate weights over time
        print("\n1. Plotting gate weight curves...")
        plot_gate_weights_curve(
            gate_weights,
            save_path=str(output_dir / "gate_weights.png"),
            title="View Gating Weights Over Time",
            sample_idx=0
        )
        
        # 2. Plot attention heatmap (middle timestep)
        print("2. Plotting attention heatmap...")
        T = attention_weights.shape[1]
        mid_timestep = T // 2
        plot_attention_heatmap(
            attention_weights,
            save_path=str(output_dir / "attention_heatmap.png"),
            title="Cross-View Attention",
            sample_idx=0,
            timestep=mid_timestep,
            head_idx=0
        )
        
        # 3. Plot temporal evolution of attention from front view
        print("3. Plotting temporal attention evolution...")
        plot_temporal_attention_evolution(
            attention_weights,
            save_path=str(output_dir / "attention_evolution_front.png"),
            sample_idx=0,
            head_idx=0,
            query_view=0  # Front view
        )
    
    print(f"\n✓ Visualization completed! Check {output_dir.absolute()}/")


def example_ablation_comparison():
    """Example: Comparing different TS-CVA configurations"""
    print("="*60)
    print("Example: TS-CVA Ablation Study")
    print("="*60)
    
    from project.models.ts_cva_model import TSCVAModel
    
    # Test different configurations
    configs = {
        'Full TS-CVA': {
            'ts_cva_shared_backbone': True,
            'ts_cva_use_view_embedding': True,
            'ts_cva_use_gated_aggregation': True,
        },
        'Without Gating': {
            'ts_cva_shared_backbone': True,
            'ts_cva_use_view_embedding': True,
            'ts_cva_use_gated_aggregation': False,
        },
        'Without View Embedding': {
            'ts_cva_shared_backbone': True,
            'ts_cva_use_view_embedding': False,
            'ts_cva_use_gated_aggregation': True,
        },
    }
    
    videos, labels = create_dummy_video_batch(batch_size=1)
    
    print("\nComparing configurations:")
    print("-" * 60)
    
    for name, config_dict in configs.items():
        config = OmegaConf.create({
            'model': {
                'model_class_num': 9,
                'ts_cva_num_heads': 4,
                'ts_cva_temporal_dim': 512,
                'ts_cva_temporal_layers': 2,
                **config_dict
            }
        })
        
        model = TSCVAModel(config)
        model.eval()
        
        with torch.no_grad():
            logits = model(videos)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
        
        print(f"{name:25s} -> Pred: {pred.item()}, Conf: {probs[0, pred].item():.4f}")
    
    print("\n✓ Ablation comparison completed!")


def main():
    parser = argparse.ArgumentParser(description="TS-CVA Usage Examples")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "inference", "visualize", "ablation", "all"],
        help="Which example to run"
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("TS-CVA Model Usage Examples")
    print("="*60 + "\n")
    
    if args.mode in ["train", "all"]:
        example_training()
        print("\n")
    
    if args.mode in ["inference", "all"]:
        example_inference()
        print("\n")
    
    if args.mode in ["visualize", "all"]:
        example_visualization()
        print("\n")
    
    if args.mode in ["ablation", "all"]:
        example_ablation_comparison()
        print("\n")
    
    print("="*60)
    print("All examples completed successfully!")
    print("="*60)
    print("\nFor real training, use:")
    print("  python project/main.py --config-name config_ts_cva")
    print("\nFor more information, see:")
    print("  doc/TS-CVA_README.md")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
