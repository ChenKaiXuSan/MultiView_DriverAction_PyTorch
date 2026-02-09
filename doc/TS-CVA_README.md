# TS-CVA: Temporal-Synchronous Cross-View Attention

## Overview

**TS-CVA (Temporal-Synchronous Cross-View Attention)** is a novel multi-view fusion method designed for driver action recognition. Unlike traditional early/late fusion approaches that simply concatenate or average features, TS-CVA:

1. **Frame-synchronous view interaction**: Explicitly models complementary relationships between views at each timestep
2. **Dynamic view selection**: Learns time-varying weights to adaptively downweight occluded or unreliable views
3. **Interpretability**: Provides attention weights and gating scores that explain which views are relied upon

This method is particularly effective when dealing with:
- View occlusions (e.g., driver hand blocking front camera)
- Lighting variations across views
- Pose changes that are better captured from certain angles

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    TS-CVA Architecture                        │
└──────────────────────────────────────────────────────────────┘

Input Videos (3 synchronized views)
    ↓
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Front CNN  │  │  Left CNN   │  │  Right CNN  │  ← 3D CNN Encoders
│  (Shared)   │  │  (Shared)   │  │  (Shared)   │     (can be shared)
└─────────────┘  └─────────────┘  └─────────────┘
    ↓                 ↓                 ↓
  F^f ∈ ℝ^(B×C×T×H×W)  F^l             F^r
    ↓                 ↓                 ↓
┌─────────────────────────────────────────────┐
│    Spatial Pooling (GAP over H, W)          │
└─────────────────────────────────────────────┘
    ↓                 ↓                 ↓
  S^f ∈ ℝ^(B×C×T)    S^l               S^r
    ↓                 ↓                 ↓
┌─────────────────────────────────────────────┐
│    Add View Embeddings (optional)           │
└─────────────────────────────────────────────┘
    ↓                 ↓                 ↓
    └─────────────────┴─────────────────┘
                ↓
    For each timestep t:
    ┌──────────────────────────────┐
    │ Multi-Head Self-Attention    │  ← Cross-view attention
    │   X_t = [S^f[t], S^l[t], S^r[t]] │
    │   X'_t = MHSA(X_t)           │
    └──────────────────────────────┘
                ↓
    ┌──────────────────────────────┐
    │ Learnable Gated Aggregation  │  ← Dynamic view weighting
    │   w_t = softmax(MLP(X'_t))   │
    │   z_t = Σ w_t^v · X'^v_t     │
    └──────────────────────────────┘
                ↓
        Z = [z_1, ..., z_T] ∈ ℝ^(B×T×C)
                ↓
    ┌──────────────────────────────┐
    │ Temporal Modeling (TCN)      │  ← Temporal dependencies
    └──────────────────────────────┘
                ↓
    ┌──────────────────────────────┐
    │ Classification Head          │
    └──────────────────────────────┘
                ↓
            Logits ∈ ℝ^(B×K)
```

## Key Components

### 1. Cross-View Attention
- **Purpose**: Model complementary relationships between views
- **Implementation**: Multi-head self-attention on {front, left, right} tokens at each timestep
- **Output**: Attention matrix A ∈ ℝ^(3×3) showing which views attend to each other

### 2. Gated View Aggregation
- **Purpose**: Dynamic view selection based on reliability
- **Implementation**: Learnable MLP produces per-view weights w_t ∈ ℝ^3
- **Benefit**: Automatically downweight occluded/unreliable views

### 3. View Embeddings
- **Purpose**: Help attention distinguish between view roles
- **Implementation**: Learnable embeddings e_v ∈ ℝ^C added to view tokens
- **Benefit**: Front/left/right can learn different functional roles

### 4. Temporal Modeling
- **Purpose**: Model temporal dependencies in fused representation
- **Implementation**: Temporal Convolutional Network (TCN)
- **Alternative**: Can replace with Temporal Transformer if needed

## Usage

### Basic Training

```bash
# Train TS-CVA with default settings
python project/main.py \
    --config-name config \
    model.fuse_method=ts_cva \
    train.view=multi \
    train.view_name=['front','left','right']
```

### Using Pre-configured Settings

```bash
# Use the provided TS-CVA configuration
python project/main.py --config-name config_ts_cva
```

### Configuration Options

All TS-CVA hyperparameters can be set in the config file or via command line:

```yaml
model:
  fuse_method: ts_cva  # REQUIRED: select TS-CVA fusion
  backbone: 3dcnn      # REQUIRED: currently only supports 3dcnn
  
  # TS-CVA specific options
  ts_cva_shared_backbone: true          # Share backbone weights across views
  ts_cva_use_view_embedding: true       # Add view embeddings
  ts_cva_use_gated_aggregation: true    # Use gating (vs mean pooling)
  ts_cva_num_heads: 4                   # Number of attention heads
  ts_cva_temporal_dim: 512              # TCN hidden dimension
  ts_cva_temporal_layers: 2             # Number of TCN layers
```

## Ablation Studies

TS-CVA supports various ablation configurations to analyze component contributions:

### 1. Single View Baselines

```bash
# Front view only
python project/main.py model.fuse_method=ts_cva train.view=single train.view_name=['front']

# Left view only
python project/main.py model.fuse_method=ts_cva train.view=single train.view_name=['left']

# Right view only
python project/main.py model.fuse_method=ts_cva train.view=single train.view_name=['right']
```

### 2. Late Fusion Baseline

```bash
# Traditional late fusion for comparison
python project/main.py \
    model.fuse_method=late \
    model.fusion_mode=logit_mean \
    train.view=multi
```

### 3. TS-CVA without Gating (Mean Pooling)

```bash
python project/main.py \
    model.fuse_method=ts_cva \
    model.ts_cva_use_gated_aggregation=false
```

### 4. TS-CVA without View Embeddings

```bash
python project/main.py \
    model.fuse_method=ts_cva \
    model.ts_cva_use_view_embedding=false
```

### 5. Non-shared Backbone

```bash
python project/main.py \
    model.fuse_method=ts_cva \
    model.ts_cva_shared_backbone=false
```

### 6. Different Attention Heads

```bash
# Test with 2, 4, 8 heads
python project/main.py model.fuse_method=ts_cva model.ts_cva_num_heads=8
```

## Visualization

TS-CVA provides rich visualization capabilities to understand model behavior:

### Gate Weight Curves

Shows how view importance changes over time:

```python
from project.utils.ts_cva_visualization import plot_gate_weights_curve

# During evaluation, gate weights are stored in the model
gate_weights = model.get_gate_weights()  # (B, T, 3)
plot_gate_weights_curve(
    gate_weights,
    save_path='gate_weights.png',
    sample_idx=0
)
```

Output: Line plot showing w_t^front, w_t^left, w_t^right over timesteps

### Attention Heatmaps

Visualize which views attend to which others:

```python
from project.utils.ts_cva_visualization import plot_attention_heatmap

attention_weights = model.get_attention_weights()  # (B, T, H, 3, 3)
plot_attention_heatmap(
    attention_weights,
    save_path='attention.png',
    sample_idx=0,
    timestep=8,
    head_idx=0
)
```

Output: 3×3 heatmap showing attention between front/left/right views

### Temporal Evolution

Track how attention from one view evolves over time:

```python
from project.utils.ts_cva_visualization import plot_temporal_attention_evolution

plot_temporal_attention_evolution(
    attention_weights,
    save_path='temporal_attention.png',
    sample_idx=0,
    head_idx=0,
    query_view=0  # Track front view
)
```

### Batch Visualization

Visualize multiple samples with predictions:

```python
from project.utils.ts_cva_visualization import visualize_batch_predictions

visualize_batch_predictions(
    gate_weights=gate_weights,
    attention_weights=attention_weights,
    predictions=preds,
    labels=labels,
    save_dir='visualizations/',
    class_names=['look_forward', 'check_left', ...]
)
```

## Expected Results

### Recommended Ablation Table

| Method | Acc (%) | F1 (%) | Notes |
|--------|---------|--------|-------|
| Front only | X | X | Single view baseline |
| Left only | X | X | Single view baseline |
| Right only | X | X | Single view baseline |
| Late fusion (mean) | X | X | Traditional multi-view |
| TS-CVA (attn + mean) | X | X | Without gating |
| TS-CVA (attn + gate) | X | X | Full method |
| TS-CVA + view emb | X | X | Add view embeddings |
| TS-CVA (non-shared) | X | X | Separate backbones |

### Interpretation

**Gate weights show**:
- Which views are trusted at different moments
- How model handles occlusions (e.g., hand blocking front camera)
- Temporal patterns in view selection

**Attention weights show**:
- Which views provide complementary information
- Cross-view dependencies (e.g., "front needs left to see blind spot")
- Head specialization (different heads focus on different view pairs)

## Key Contributions

1. **Frame-synchronous fusion**: Unlike late fusion that aggregates after encoding, TS-CVA fuses at each timestep
2. **Interpretability**: Attention and gating provide insights into model decisions
3. **Robustness**: Dynamic weighting handles occlusions and view degradation
4. **Modular design**: Easy to ablate components for analysis

## Citation

If you use TS-CVA in your research, please cite:

```bibtex
@inproceedings{Chen2026TSCVA,
  title     = {Temporal-Synchronous Cross-View Attention for Multi-View Driver Action Recognition},
  author    = {Chen, Kaixu},
  booktitle = {CHI Conference on Human Factors in Computing Systems},
  year      = {2026}
}
```

## Implementation Details

### Computational Efficiency

- **Parallel attention**: All timesteps processed together (batch×time reshape)
- **Shared backbone**: Reduces parameters by 3× compared to separate encoders
- **Memory**: Comparable to late fusion (attention overhead is small)

### Training Tips

1. **Learning rate**: Start with 1e-4, reduce if unstable
2. **Batch size**: 2-4 works well (limited by video memory)
3. **Frame sampling**: 8-16 frames typically sufficient
4. **Resolution**: 224×224 for best quality, 112×112 for faster training

### Common Issues

**Q: OOM during training?**
A: Reduce `data.uniform_temporal_subsample_num` or `data.img_size`

**Q: Gate weights all equal (0.33, 0.33, 0.33)?**
A: Model hasn't learned view differences yet. Check:
- View embeddings enabled? (`ts_cva_use_view_embedding=true`)
- Enough training data?
- Learning rate appropriate?

**Q: Attention weights look random?**
A: Normal in early training. Should become structured after 10+ epochs.

**Q: Performance worse than late fusion?**
A: TS-CVA needs more data to learn view relationships. Try:
- Increase training epochs
- Use view embeddings
- Check if views are truly synchronized
