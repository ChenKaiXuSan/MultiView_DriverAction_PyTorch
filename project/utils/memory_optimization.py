#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Memory optimization utilities for training.

This module provides memory management utilities to reduce RAM/VRAM usage during training.
"""

import gc
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def cleanup_memory(aggressive: bool = False):
    """
    Clean up GPU and CPU memory.
    
    Args:
        aggressive: If True, performs more aggressive cleanup (may be slower)
    """
    # Clear Python garbage
    gc.collect()
    
    # Clear PyTorch CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        if aggressive:
            # More aggressive cleanup - synchronize to ensure all operations complete
            torch.cuda.synchronize()


class MemoryOptimizer:
    """
    Memory optimizer for PyTorch Lightning training.
    
    Can be used as a context manager or called directly.
    """
    
    def __init__(self, enable_cleanup: bool = True, log_memory: bool = False):
        """
        Args:
            enable_cleanup: Whether to perform memory cleanup
            log_memory: Whether to log memory usage
        """
        self.enable_cleanup = enable_cleanup
        self.log_memory = log_memory
        self.initial_memory = None
    
    def __enter__(self):
        """Start memory tracking."""
        if self.log_memory and torch.cuda.is_available():
            self.initial_memory = torch.cuda.memory_allocated()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up memory on exit."""
        if self.enable_cleanup:
            cleanup_memory()
        
        if self.log_memory and torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            if self.initial_memory is not None:
                delta = final_memory - self.initial_memory
                logger.debug(
                    f"Memory delta: {delta / 1024**2:.2f} MB "
                    f"(Initial: {self.initial_memory / 1024**2:.2f} MB, "
                    f"Final: {final_memory / 1024**2:.2f} MB)"
                )


def log_memory_stats(tag: str = ""):
    """
    Log current GPU memory statistics.
    
    Args:
        tag: Optional tag to identify the logging point
    """
    if not torch.cuda.is_available():
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    msg = f"Memory Stats"
    if tag:
        msg += f" [{tag}]"
    msg += f": Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB"
    
    logger.info(msg)


def reset_peak_memory_stats():
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class GradientCheckpointing:
    """
    Helper class to enable/disable gradient checkpointing for models.
    
    Gradient checkpointing trades compute for memory by not storing
    intermediate activations during forward pass.
    """
    
    @staticmethod
    def enable_for_transformer(model: torch.nn.Module):
        """
        Enable gradient checkpointing for Transformer models.
        
        Args:
            model: PyTorch model with transformer layers
        """
        if hasattr(model, "encoder"):
            # Standard transformer with encoder
            if hasattr(model.encoder, "enable_gradient_checkpointing"):
                model.encoder.enable_gradient_checkpointing()
            elif hasattr(model.encoder, "gradient_checkpointing_enable"):
                model.encoder.gradient_checkpointing_enable()
            else:
                logger.warning("Could not enable gradient checkpointing - method not found")
        else:
            logger.warning("Model does not have 'encoder' attribute")
    
    @staticmethod
    def enable_for_model(model: torch.nn.Module, use_checkpointing: bool = True):
        """
        Enable gradient checkpointing for a model if supported.
        
        Args:
            model: PyTorch model
            use_checkpointing: Whether to enable checkpointing
        """
        if not use_checkpointing:
            return
        
        # Try various common methods
        if hasattr(model, "use_gradient_checkpointing"):
            model.use_gradient_checkpointing = True
        
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()


# Convenience functions for training loops

def on_train_batch_end_cleanup(enable_cleanup: bool = True):
    """
    Call this at the end of each training batch to clean up memory.
    
    Args:
        enable_cleanup: Whether to perform cleanup
    """
    if enable_cleanup:
        cleanup_memory(aggressive=False)


def on_epoch_end_cleanup(aggressive: bool = True):
    """
    Call this at the end of each epoch for more thorough cleanup.
    
    Args:
        aggressive: Whether to perform aggressive cleanup
    """
    cleanup_memory(aggressive=aggressive)
