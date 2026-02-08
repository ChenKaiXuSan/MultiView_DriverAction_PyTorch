# Video分块加载 - 快速参考

## 🎯 什么时候使用

**症状**: 加载video时就OOM，还没开始训练
```
RuntimeError: CUDA out of memory when loading video
```

**原因**: Video太长，一次加载整个video占用过多内存

**解决**: 在dataloader层面分块加载

---

## ⚡ 2分钟快速修复

### 步骤1: 修改配置文件

```yaml
# config.yaml
data:
  max_video_frames: 1000  # 添加这一行
```

### 步骤2: 开始训练

```bash
python project/main.py --config configs/your_config.yaml
```

就这么简单！✅

---

## 📏 参数选择指南

| GPU显存 | 分辨率 | 推荐值 |
|--------|-------|--------|
| 8GB | 224×224 | 500-800 |
| 12GB | 224×224 | 800-1200 |
| 16GB+ | 224×224 | 1200-2000 |
| 8GB | 112×112 | 1000-1500 |
| 8GB | 56×56 | 2000-3000 |

**原则**: 尽量大（减少chunks数量），但不要OOM

---

## 🔍 如何验证有效

```bash
# 运行测试
python test_chunked_loading.py

# 期望看到：
# "150 videos → 423 chunks"  ← chunks数量增加了
# "Memory per sample: 2.5GB"  ← 内存降低了
```

---

## 💡 工作原理

```
不分块:
Video (5000帧) → 一次加载5000帧 → OOM ❌

分块:
Video (5000帧) → 分成5个chunks
  ├─ chunk_0 (0-1000帧)    ← 只加载1000帧 ✅
  ├─ chunk_1 (1000-2000帧)
  ├─ chunk_2 (2000-3000帧)  
  ├─ chunk_3 (3000-4000帧)
  └─ chunk_4 (4000-5000帧)
```

每次只加载一个chunk，避免OOM！

---

## 🎛️ 完整优化组合

```yaml
# 适用于8GB GPU
data:
  batch_size: 1
  max_video_frames: 800     # ← 加载时分块
  load_kpt: false

train:
  accumulate_grad_batches: 8  # ← 训练时梯度累积

trainer:
  precision: 16              # ← 混合精度
```

**内存节省**: ~85-90% 🎉

---

## 📚 详细文档

- **完整指南**: [doc/VIDEO_CHUNKING_GUIDE.md](doc/VIDEO_CHUNKING_GUIDE.md)
- **配置示例**: [configs/config_chunked_loading.yaml](configs/config_chunked_loading.yaml)
- **测试脚本**: [test_chunked_loading.py](test_chunked_loading.py)
- **实现总结**: [DATALOADER_CHUNKING_SUMMARY.md](DATALOADER_CHUNKING_SUMMARY.md)

---

## ⚠️ 常见问题

**Q: 会影响训练效果吗？**  
A: 不会。只是改变了数据加载方式，训练数据总量不变。

**Q: Epoch时间会变化吗？**  
A: 会变长（samples增多），但这是正常的。

**Q: 需要修改其他代码吗？**  
A: 不需要。只改配置文件即可。

**Q: 能和其他优化一起用吗？**  
A: 可以！推荐组合使用以达到最大效果。

---

**更新**: 2026-02-08 | **状态**: ✅ 可用
