# Separable ResNet 3D CNN Structure

## Overview

The ResNet 3D CNN model now supports a **separable structure** where the stem, body, and head can be accessed and used independently. This is useful for:

1. **Feature extraction at different stages**
2. **Custom fusion architectures** (like TS-CVA)
3. **Transfer learning** with frozen backbones
4. **Multi-task learning** with shared backbones

## Architecture Components

### Stem (Initial Layers)
- Input: `(B, 3, T, H, W)` - RGB video
- Conv3d layer with kernel (7, 7, 7)
- BatchNorm + ReLU + Pooling
- Output: `(B, 64, T', H', W')`

### Body (ResNet Stages)
- Input: `(B, 64, T', H', W')`
- ResNet stages (res2, res3, res4, res5)
- Output: `(B, 2048, T'', H'', W'')`

### Head (Classification)
- Input: `(B, 2048, T'', H'', W'')`
- Adaptive global average pooling
- Dropout (optional)
- Linear projection to class logits
- Output: `(B, num_classes)`

## Usage

### Method 1: Using Res3DCNN with separable=True

```python
from omegaconf import OmegaConf
from project.models.res_3dcnn import Res3DCNN
import torch

# Create configuration
config = OmegaConf.create({
    'model': {
        'model_class_num': 9,
    }
})

# Initialize with separable structure
model = Res3DCNN(config, use_separable=True)

# Forward pass (automatic)
video = torch.randn(2, 3, 16, 224, 224)
output = model(video)  # (2, 9)

# Manual forward through components
stem_out = model.stem(video)       # (2, 64, 16, 56, 56)
body_out = model.body(stem_out)    # (2, 2048, 16, 7, 7)
head_out = model.head(body_out)    # (2, 9)

# Extract features before classification
features = model.forward_features(video)  # (2, 2048)
```

### Method 2: Using BaseModel.init_resnet_separable()

```python
from project.models.base_model import BaseModel
import torch

# Initialize separable components
stem, body, head, feature_dim = BaseModel.init_resnet_separable(
    class_num=9,
    return_feature_dim=True
)

print(f"Feature dimension: {feature_dim}")  # 2048

# Forward pass
video = torch.randn(2, 3, 16, 224, 224)
x = stem(video)
x = body(x)
x = head(x)  # (2, 9)
```

### Method 3: Getter Methods (Works with both modes)

```python
# Works with both separable=True and separable=False
model = Res3DCNN(config, use_separable=False)

stem = model.get_stem()
body = model.get_body()
head = model.get_head()

# Use components
video = torch.randn(2, 3, 16, 224, 224)
x = stem(video)
x = body(x)
x = head(x)
```

## Example: Custom Multi-View Fusion

```python
import torch
import torch.nn as nn
from project.models.base_model import BaseModel

class CustomMultiViewModel(nn.Module):
    def __init__(self, num_classes=9, num_views=3):
        super().__init__()
        
        # Create separable components for each view
        self.stems = nn.ModuleList()
        self.bodies = nn.ModuleList()
        
        for _ in range(num_views):
            stem, body, _, _ = BaseModel.init_resnet_separable(
                class_num=num_classes,
                return_feature_dim=True
            )
            self.stems.append(stem)
            self.bodies.append(body)
        
        # Shared classification head
        _, _, head, feature_dim = BaseModel.init_resnet_separable(
            class_num=num_classes,
            return_feature_dim=True
        )
        self.head = head
        
        # Fusion layer
        self.fusion = nn.Conv3d(2048 * num_views, 2048, kernel_size=1)
    
    def forward(self, videos):
        """
        videos: dict with keys 'front', 'left', 'right'
        """
        features = []
        
        for i, view in enumerate(['front', 'left', 'right']):
            x = self.stems[i](videos[view])
            x = self.bodies[i](x)
            features.append(x)
        
        # Concatenate and fuse
        fused = torch.cat(features, dim=1)  # (B, 2048*3, T, H, W)
        fused = self.fusion(fused)  # (B, 2048, T, H, W)
        
        # Classify
        output = self.head(fused)
        return output

# Usage
model = CustomMultiViewModel(num_classes=9)
videos = {
    'front': torch.randn(2, 3, 16, 224, 224),
    'left': torch.randn(2, 3, 16, 224, 224),
    'right': torch.randn(2, 3, 16, 224, 224),
}
output = model(videos)  # (2, 9)
```

## Example: Transfer Learning with Frozen Backbone

```python
from project.models.res_3dcnn import Res3DCNN
from omegaconf import OmegaConf

config = OmegaConf.create({'model': {'model_class_num': 9}})
model = Res3DCNN(config, use_separable=True)

# Freeze stem and body for transfer learning
for param in model.stem.parameters():
    param.requires_grad = False

for param in model.body.parameters():
    param.requires_grad = False

# Only train the head
for param in model.head.parameters():
    param.requires_grad = True

# Now train the model (only head weights will be updated)
```

## Example: Feature Extraction at Different Stages

```python
from project.models.res_3dcnn import Res3DCNN
from omegaconf import OmegaConf
import torch

config = OmegaConf.create({'model': {'model_class_num': 9}})
model = Res3DCNN(config, use_separable=True)
model.eval()

video = torch.randn(1, 3, 16, 224, 224)

with torch.no_grad():
    # Early features (after stem)
    stem_features = model.stem(video)
    print(f"Stem features: {stem_features.shape}")  # (1, 64, 16, 56, 56)
    
    # Mid-level features (after body)
    body_features = model.body(stem_features)
    print(f"Body features: {body_features.shape}")  # (1, 2048, 16, 7, 7)
    
    # High-level features (after pooling, before classification)
    high_features = model.forward_features(video)
    print(f"High features: {high_features.shape}")  # (1, 2048)
    
    # Final classification
    logits = model(video)
    print(f"Logits: {logits.shape}")  # (1, 9)
```

## Differences Between Standard and Separable Modes

| Feature | Standard Mode | Separable Mode |
|---------|--------------|----------------|
| Structure | Unified `model.blocks` | Separate `stem`, `body`, `head` |
| Access | Via block indices | Direct attribute access |
| Forward | Single `model(x)` | Can use `stem->body->head` or `model(x)` |
| Features | `forward_features()` | `forward_features()` or manual |
| Memory | Same | Same |
| Performance | Same | Same |
| Use Case | Simple classification | Custom architectures, transfer learning |

## Benefits of Separable Structure

1. **Modularity**: Each component can be used independently
2. **Flexibility**: Easy to build custom architectures
3. **Clarity**: Explicit separation of concerns
4. **Reusability**: Share components across models
5. **Transfer Learning**: Freeze/unfreeze specific parts easily

## Implementation Details

### Adaptive Pooling in Head

The head always uses adaptive average pooling to ensure compatibility with various input sizes:

```python
# Always pools to (1, 1, 1) regardless of input size
x = nn.functional.adaptive_avg_pool3d(x, (1, 1, 1))
x = x.view(x.size(0), -1)  # Flatten to (B, 2048)
x = self.proj(x)  # Project to (B, num_classes)
```

This ensures the head works with any spatial/temporal input dimensions.

### Backward Compatibility

The standard mode (use_separable=False) maintains full backward compatibility with existing code:

```python
# Old code still works
model = Res3DCNN(config)  # use_separable defaults to False
output = model(video)
```

## Testing

Run the test suite to verify the separable structure:

```bash
python tests/test_separable_resnet.py
```

Tests cover:
- Component initialization
- Forward pass through components
- Standard vs separable equivalence
- Feature extraction
- Gradient flow
- Getter methods

All tests should pass with output:
```
✓ ALL SEPARABLE STRUCTURE TESTS PASSED!
```

## Integration with TS-CVA

The TS-CVA model can now use the separable structure for more efficient feature extraction:

```python
# In TSCVAModel, when extracting view features:
def extract_view_features(self, video, view_idx):
    if self.use_shared_backbone:
        backbone = self.backbone
    else:
        backbone = [self.backbone_front, ...][view_idx]
    
    # Can now use separable components if needed
    stem = backbone.get_stem()
    body = backbone.get_body()
    
    x = stem(video)
    x = body(x)
    return x
```

## Summary

The separable ResNet 3D CNN structure provides:
- ✅ Explicit separation of stem, body, and head
- ✅ Easy access to intermediate features
- ✅ Support for custom fusion architectures
- ✅ Transfer learning capabilities
- ✅ Full backward compatibility
- ✅ Well-tested and documented

Use `use_separable=True` when you need fine-grained control over the model components!
