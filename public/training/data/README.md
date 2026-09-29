# 训练页面数据

⚠️ **这是模拟数据，用于演示训练可视化功能**

## 文件说明

- `training_curve.json` - 训练曲线（3000步，10个检查点）
- `step_500.json` - 第500步的详细训练数据（前向、反向、权重更新）
- `weights.bin` - 权重分布快照（4层×7个矩阵×32×32网格）
- `*.gz` - 压缩版本（网页加载用）

## 数据格式

### training_curve.json
```json
{
  "model_config": {...},
  "training_config": {...},
  "checkpoints": [
    {
      "step": 0,
      "loss": 8.524,
      "perplexity": 5041,
      "lr": 0.001,
      "sample": "生成的文本",
      "quality": "gibberish"
    },
    ...
  ]
}
```

### step_500.json
```json
{
  "step": 500,
  "forward": {
    "input_text": "...",
    "layers": [...],
    "logits_top5": [...],
    "loss": 4.456
  },
  "backward": {
    "layers": [...]
  },
  "weight_updates": {
    "example_weights": [...]
  }
}
```

## 重新生成

```bash
python tools/export_training_mock.py --output public/training/data/
```

要生成真实数据，需要运行：
```bash
python tools/train_small_model.py --output public/training/data/
```
