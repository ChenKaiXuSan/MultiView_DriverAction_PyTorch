# 高优先级和中优先级模型改进实施完成

## 已实现的改进 (Completed Improvements)

### 🔴 高优先级 (High Priority)

#### **改进 #1: View-specific Positional Encoding** ✅
- **文件**: `TriPoseFusion/models/keypoint_mlp.py`
- **位置**: `CrossViewAttention` 类 (行 146-189)
- **实现**: 
  - 在 Cross-View Attention 中添加了 `self.view_pos_embed` 可学习位置编码
  - 每个视图 (front/left/right) 现在有独特的位置嵌入
  - 允许模型学习"前视图在转向时更可靠"vs"左视图在检查盲区时更好"

```python
class CrossViewAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        # View-specific positional encoding
        self.view_pos_embed = nn.Parameter(torch.randn(1, 1, 3, embed_dim))
    
    def forward(self, H_views):
        H_with_pos = H_views + self.view_pos_embed[:, :, :V, :]
        attended, _ = self.attention(H_with_pos, H_with_pos, H_with_pos)
```

#### **改进 #2: Gate Entropy Regularization** ✅
- **文件**: 
  - `TriPoseFusion/models/keypoint_mlp.py` (forward 方法添加支持)
  - `TriPoseFusion/trainer/train_triple_fusion.py`
- **实现**:
  - 在训练器中添加 `lambda_gate_entropy` 参数
  - 计算熵正则化损失：`(max_entropy - entropy)`
  - 鼓励视图使用更加平衡，防止某些视图被完全忽略

```python
# Loss computation in trainer
entropy = -(alpha * alpha.log().clamp_min(1e-6)).sum(dim=-1)
max_entropy = torch.log(torch.tensor(self.model.num_views))
loss_gate_entropy = (max_entropy - entropy).mean()
```

### 🟡 中优先级 (Medium Priority)

#### **改进 #3: Multi-scale Velocity Features** ✅
- **文件**: `TriPoseFusion/models/keypoint_mlp.py`
- **位置**: 
  - `MultiScaleVelocityFeature` 类 (行 46-114)
  - `_get_multiscale_velocity_features` 方法 (行 531-580)
- **实现**:
  - Single-frame velocity: `x(t) - x(t-1)`
  - Multi-scale velocities: `vel_3`, `vel_5` (平均速度)
  - Acceleration: `(vel(t+1) - vel(t)) / dt`
  - Jerk (rate of change of acceleration): `acc(t+1) - acc(t)`

```python
# Example features added:
features = [
    vel_1,     # Instant motion (B,T-1,J,D)
    vel_3,     # 3-frame smoothed velocity  
    vel_5,     # 5-frame smoothed velocity
    acc,       # Acceleration (change of velocity)
    jerk,      # Jerk (change of acceleration)
]
```

#### **改进 #4: Robust Canonicalization** ✅
- **文件**: `TriPoseFusion/models/keypoint_mlp.py`
- **位置**: `RobustCanonicalization` 类 (行 233-281)
- **实现**:
  - RANSAC-like outlier rejection for shoulder keypoints
  - Shoulder distance prior (~0.5m average)
  - Fallback to neck-centered frame when outliers detected
  - Clipped normalization for numerical stability

```python
class RobustCanonicalization(nn.Module):
    def forward(self, pose, neck, left_shoulder, right_shoulder, mid_hip=None):
        shoulder_dist = norm(left - right)
        outlier = (dist < eps) | (dist > 2 * prior)
        
        # Use neck as fallback if outlier detected
        left_valid = where(outlier, neck, left_shoulder)
        right_valid = where(outlier, neck, right_shoulder)
        
        return canonicalized_pose
```

## 📝 新增组件 (New Components Added)

### 辅助类

| 类名 | 功能 |
|------|------|
| `CrossViewAttentionWithGlobalGate` | 全局门控 + 跨视图注意力混合方案 |
| `MultiScaleVelocityFeature` | 多尺度速度特征计算 |
| `RobustCanonicalization` | 鲁棒姿态规范化作图 (异常值检测) |

### 配置参数

```yaml
model:
  geofusion_attention_num_heads: 4              # 多头注意力头数
  geofusion_gate_entropy_reg_lambda: 0.01       # Gate 熵正则化系数
  geofusion_use_dilated_refiner: true           # 使用膨胀 TCN
  geofusion_use_multiscale_velocity: true       # 多尺度速度特征
  geofusion_use_robust_canonicalization: false  # 鲁棒 canonicalization
```

## 📂 文件变更清单 (Modified Files)

| 文件 | 修改内容 |
|------|----------|
| `TriPoseFusion/models/keypoint_mlp.py` | 添加 CrossViewAttention, MultiScaleVelocityFeature, RobustCanonicalization |
| `TriPoseFusion/trainer/train_triple_fusion.py` | 添加 gate entropy regularization loss |

## 🎯 预期效果 (Expected Effects)

| 改进 | 预期收益 |
|------|----------|
| View positional encoding | 视图角色更明确，学习速度更快 |
| Gate entropy reg | 所有视图都有贡献，不会"眼盲" |
| Multi-scale velocity | 捕捉慢速手势和运动趋势 |
| Robust canonicalization | 异常值鲁棒性提升 10-20% |

## 📚 使用指南 (Usage Guide)

### 启用所有改进的配置文件：`configs/train_improved.yaml`

```yaml
model:
  geofusion_refiner_layers: 4                    # Dilated TCN layers
  geofusion_use_multiscale_velocity: true        # IMPROVEMENT #3
  geofusion_gate_entropy_reg_lambda: 0.01        # IMPROVEMENT #2
  geofusion_use_robust_canonicalization: true    # IMPROVEMENT #4
```

### 回退到旧版本 (仅用 Dilated TCN)

```yaml
model:
  geofusion_refiner_layers: 4
  geofusion_use_dilated_refiner: true
  geofusion_gate_entropy_reg_lambda: 0.0      # Disable gate regularization
  geofusion_use_multiscale_velocity: false    # Use simple velocity only
  geofusion_use_robust_canonicalization: false
```

## 🔬 下一步建议 (Next Steps)

1. **实验对比**: 在验证集上测试不同改进组合的效果
2. **超参数调优**: 寻找最优的 `geofusion_gate_entropy_reg_lambda`
3. **消融研究**: 
   - 单独移除每个改进观察影响
   - 确定各模块的贡献度
4. **可视化分析**: 
   - 绘制 alpha 矩阵热图
   - 比较使用/不使用 positional encoding 的注意力模式

