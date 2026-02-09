#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: ts_cva_visualization.py
Project: project/utils
Created Date: 2026-02-09
Author: Kaixu Chen
-----
Comment:
Visualization utilities for TS-CVA model.

Provides functions to visualize:
1. Gate weight curves over time (w_t^f, w_t^l, w_t^r)
2. Attention heatmaps (3x3 attention matrices)
3. Error analysis with view contributions

Have a good code time :)
-----
Copyright (c) 2026 The University of Tsukuba
-----
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_gate_weights_curve(
    gate_weights: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "View Gate Weights Over Time",
    sample_idx: int = 0
) -> None:
    """
    Plot gate weight curves for three views over time.
    
    Args:
        gate_weights: (B, T, 3) - gate weights tensor
        save_path: path to save the figure
        title: title for the plot
        sample_idx: which sample in the batch to visualize
    """
    if gate_weights is None or len(gate_weights) == 0:
        print("No gate weights available for visualization.")
        return
    
    # Extract weights for the specified sample
    weights = gate_weights[sample_idx].cpu().numpy()  # (T, 3)
    T = weights.shape[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each view's weight curve
    timesteps = np.arange(T)
    view_names = ['Front', 'Left', 'Right']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for view_idx in range(3):
        ax.plot(timesteps, weights[:, view_idx], 
               label=view_names[view_idx], 
               color=colors[view_idx],
               linewidth=2,
               marker='o',
               markersize=4)
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Gate Weight', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gate weights curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Cross-View Attention Heatmap",
    sample_idx: int = 0,
    timestep: int = 0,
    head_idx: int = 0
) -> None:
    """
    Plot attention heatmap for a specific timestep and attention head.
    
    Args:
        attention_weights: (B, T, num_heads, 3, 3) - attention weights tensor
        save_path: path to save the figure
        title: title for the plot
        sample_idx: which sample in the batch to visualize
        timestep: which timestep to visualize
        head_idx: which attention head to visualize
    """
    if attention_weights is None or len(attention_weights) == 0:
        print("No attention weights available for visualization.")
        return
    
    # Extract attention matrix for the specified sample, timestep, and head
    attn_matrix = attention_weights[sample_idx, timestep, head_idx].cpu().numpy()  # (3, 3)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot heatmap
    view_names = ['Front', 'Left', 'Right']
    sns.heatmap(
        attn_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        square=True,
        cbar_kws={'label': 'Attention Weight'},
        xticklabels=view_names,
        yticklabels=view_names,
        ax=ax,
        vmin=0,
        vmax=1
    )
    
    ax.set_xlabel('Key (from view)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query (from view)', fontsize=12, fontweight='bold')
    ax.set_title(f"{title}\n(Timestep {timestep}, Head {head_idx})", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention heatmap saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multiple_attention_heads(
    attention_weights: torch.Tensor,
    save_path: Optional[str] = None,
    sample_idx: int = 0,
    timestep: int = 0
) -> None:
    """
    Plot attention heatmaps for all heads at a specific timestep.
    
    Args:
        attention_weights: (B, T, num_heads, 3, 3) - attention weights tensor
        save_path: path to save the figure
        sample_idx: which sample in the batch to visualize
        timestep: which timestep to visualize
    """
    if attention_weights is None or len(attention_weights) == 0:
        print("No attention weights available for visualization.")
        return
    
    num_heads = attention_weights.shape[2]
    
    # Create subplots
    fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, 5))
    if num_heads == 1:
        axes = [axes]
    
    view_names = ['Front', 'Left', 'Right']
    
    for head_idx in range(num_heads):
        attn_matrix = attention_weights[sample_idx, timestep, head_idx].cpu().numpy()
        
        sns.heatmap(
            attn_matrix,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            square=True,
            cbar_kws={'label': 'Attention'},
            xticklabels=view_names,
            yticklabels=view_names,
            ax=axes[head_idx],
            vmin=0,
            vmax=1
        )
        
        axes[head_idx].set_title(f"Head {head_idx}", fontsize=12, fontweight='bold')
        axes[head_idx].set_xlabel('Key (from view)')
        axes[head_idx].set_ylabel('Query (from view)')
    
    plt.suptitle(f"Multi-Head Attention at Timestep {timestep}", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-head attention heatmap saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_temporal_attention_evolution(
    attention_weights: torch.Tensor,
    save_path: Optional[str] = None,
    sample_idx: int = 0,
    head_idx: int = 0,
    query_view: int = 0
) -> None:
    """
    Plot how attention from one query view to other views evolves over time.
    
    Args:
        attention_weights: (B, T, num_heads, 3, 3) - attention weights tensor
        save_path: path to save the figure
        sample_idx: which sample in the batch to visualize
        head_idx: which attention head to visualize
        query_view: which query view to track (0=front, 1=left, 2=right)
    """
    if attention_weights is None or len(attention_weights) == 0:
        print("No attention weights available for visualization.")
        return
    
    # Extract attention weights for the specified sample, head, and query view
    # Shape: (T, 3) - attention from query_view to all key views over time
    attn_over_time = attention_weights[sample_idx, :, head_idx, query_view, :].cpu().numpy()
    T = attn_over_time.shape[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each key view's attention
    timesteps = np.arange(T)
    view_names = ['Front', 'Left', 'Right']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for key_view in range(3):
        ax.plot(timesteps, attn_over_time[:, key_view],
               label=f'Attend to {view_names[key_view]}',
               color=colors[key_view],
               linewidth=2,
               marker='o',
               markersize=4)
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title(f'Temporal Evolution of Attention from {view_names[query_view]} View',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temporal attention evolution saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_batch_predictions(
    gate_weights: torch.Tensor,
    attention_weights: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    save_dir: str,
    class_names: Optional[List[str]] = None,
    max_samples: int = 5
) -> None:
    """
    Visualize predictions with gate weights and attention for multiple samples.
    
    Args:
        gate_weights: (B, T, 3) - gate weights tensor
        attention_weights: (B, T, num_heads, 3, 3) - attention weights tensor
        predictions: (B,) - predicted labels
        labels: (B,) - ground truth labels
        save_dir: directory to save visualizations
        class_names: optional list of class names
        max_samples: maximum number of samples to visualize
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    B = min(len(predictions), max_samples)
    
    for sample_idx in range(B):
        pred = predictions[sample_idx].item()
        label = labels[sample_idx].item()
        
        # Determine if prediction is correct
        correct = "Correct" if pred == label else "Incorrect"
        
        # Get class names if available
        if class_names:
            pred_name = class_names[pred]
            label_name = class_names[label]
            title_suffix = f"\nPred: {pred_name}, GT: {label_name} ({correct})"
        else:
            title_suffix = f"\nPred: {pred}, GT: {label} ({correct})"
        
        # Plot gate weights
        plot_gate_weights_curve(
            gate_weights,
            save_path=os.path.join(save_dir, f"sample_{sample_idx}_gate_weights.png"),
            title=f"Gate Weights - Sample {sample_idx}{title_suffix}",
            sample_idx=sample_idx
        )
        
        # Plot attention heatmap (middle timestep, first head)
        if attention_weights is not None and len(attention_weights) > 0:
            T = attention_weights.shape[1]
            mid_timestep = T // 2
            plot_attention_heatmap(
                attention_weights,
                save_path=os.path.join(save_dir, f"sample_{sample_idx}_attention.png"),
                title=f"Attention - Sample {sample_idx}{title_suffix}",
                sample_idx=sample_idx,
                timestep=mid_timestep,
                head_idx=0
            )
    
    print(f"Batch visualizations saved to {save_dir}")
