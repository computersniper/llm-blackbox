"""生成训练页面的模拟数据（无需真实训练）

用法：
    python tools/export_training_mock.py --output public/training/data/

生成高质量的模拟训练数据，包括：
- 训练曲线（3000步）
- 10个检查点的生成样本
- 单步训练的详细数据（前向、反向、权重更新）
- 权重分布快照
"""

import json
import random
import math
import pathlib
import struct
import gzip

# 设置随机种子
random.seed(42)

# ================================================================ 训练曲线数据

def generate_training_curve():
    """生成训练曲线数据"""
    checkpoints = []

    # 10个检查点
    checkpoint_steps = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2500, 3000]

    # 损失从8.5逐渐降到2.1
    samples = [
        "春天来了花儿开春天来了春天",  # 乱码阶段
        "春天天天天天天天天",  # 重复阶段
        "春天来了，花儿开了",  # 开始有意义
        "春天来了，花儿开了，鸟儿叫了",
        "春风拂面花满园",
        "春回大地万物苏，桃红柳绿",
        "春江潮水连海平，海上明月",
        "春眠不觉晓，处处闻啼鸟",
        "碧玉妆成一树高，万条垂下绿丝绦",
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
    ]

    for i, step in enumerate(checkpoint_steps):
        # 损失曲线：指数衰减
        loss = 8.5 * math.exp(-step / 800) + 2.0
        perplexity = math.exp(loss)
        lr = 1e-3 if step < 100 else 1e-3 * (1 - (step - 100) / 3000)  # 线性衰减

        quality = "gibberish" if i < 3 else ("improving" if i < 7 else "good")

        checkpoints.append({
            "step": step,
            "loss": round(loss, 4),
            "perplexity": round(perplexity, 2),
            "lr": round(lr, 6),
            "sample": samples[i],
            "quality": quality
        })

    return {
        "version": "1.0",
        "description": "模拟的训练数据 - 用于演示训练可视化功能",
        "model_config": {
            "vocab_size": 151936,
            "hidden_size": 256,
            "num_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 768
        },
        "training_config": {
            "total_steps": 3000,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 0.01
        },
        "checkpoints": checkpoints
    }


# ================================================================ 单步训练详细数据

def generate_step_detail(step=500):
    """生成单步训练的详细数据"""

    # 前向传播数据
    forward_data = {
        "input_text": "春天来了",
        "input_ids": [102915, 118435, 104500],
        "label_text": "，花儿开了",
        "label_ids": [3837, 125209, 104500],
        "layers": []
    }

    # 4层的activation
    for layer_id in range(4):
        layer_data = {
            "layer_id": layer_id,
            "input_norm": round(1.0 + random.uniform(-0.2, 0.2), 4),
            "output_norm": round(1.2 + random.uniform(-0.2, 0.2), 4),
            "attn_max_activation": round(random.uniform(0.5, 2.0), 4),
            "mlp_max_activation": round(random.uniform(1.0, 3.0), 4)
        }
        forward_data["layers"].append(layer_data)

    forward_data["logits_top5"] = [
        {"token": "，", "token_id": 3837, "logit": 8.234},
        {"token": "的", "token_id": 1773, "logit": 7.123},
        {"token": "。", "token_id": 1773, "logit": 6.456},
        {"token": "是", "token_id": 1773, "logit": 5.789},
        {"token": "和", "token_id": 1773, "logit": 5.123}
    ]
    forward_data["prediction"] = "，花儿开了"
    forward_data["loss"] = 4.456

    # 反向传播数据
    backward_data = {
        "layers": []
    }

    for layer_id in range(3, -1, -1):  # 从后往前
        layer_data = {
            "layer_id": layer_id,
            "grad_output_norm": round(0.5 * (4 - layer_id) * random.uniform(0.8, 1.2), 6),
            "grad_attn_norm": round(0.3 * random.uniform(0.8, 1.2), 6),
            "grad_mlp_norm": round(0.4 * random.uniform(0.8, 1.2), 6),
            "max_gradient": round(random.uniform(0.1, 0.5), 6)
        }
        backward_data["layers"].append(layer_data)

    # 权重更新示例
    weight_updates = {
        "example_weights": [
            {
                "name": "layer.1.self_attn.q_proj.weight[64][128]",
                "before": 0.023456,
                "gradient": -0.000123,
                "m_t": 0.000012,
                "v_t": 1.5e-8,
                "after": 0.023468,
                "delta": 0.000012
            },
            {
                "name": "layer.1.mlp.down_proj.weight[128][256]",
                "before": -0.015678,
                "gradient": 0.000234,
                "m_t": -0.000023,
                "v_t": 5.5e-8,
                "after": -0.015701,
                "delta": -0.000023
            },
            {
                "name": "layer.2.mlp.gate_proj.weight[256][192]",
                "before": 0.034567,
                "gradient": -0.000345,
                "m_t": 0.000034,
                "v_t": 1.2e-7,
                "after": 0.034533,
                "delta": -0.000034
            }
        ],
        "optimizer_config": {
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "eps": 1e-8
        }
    }

    return {
        "version": "1.0",
        "description": "模拟的单步训练详细数据",
        "step": step,
        "forward": forward_data,
        "backward": backward_data,
        "weight_updates": weight_updates
    }


# ================================================================ 权重分布数据

def generate_weight_distribution():
    """生成权重分布快照（二进制格式）"""
    # 4层 × 7个权重矩阵 × 32×32网格
    num_layers = 4
    num_matrices = 7  # q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    grid_size = 32

    weights = []
    for layer in range(num_layers):
        for matrix in range(num_matrices):
            for i in range(grid_size):
                for j in range(grid_size):
                    # 生成权重分布：接近0均值，标准差约0.02
                    w = random.gauss(0, 0.02)
                    weights.append(w)

    return struct.pack(f'{len(weights)}f', *weights)


# ================================================================ 主函数

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="public/training/data/", help="输出目录")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("生成训练页面模拟数据...")

    # 1. 训练曲线
    print("  [1/4] 生成训练曲线数据...")
    curve_data = generate_training_curve()
    with open(output_dir / "training_curve.json", 'w', encoding='utf-8') as f:
        json.dump(curve_data, f, ensure_ascii=False, indent=2)

    # 压缩版本
    with gzip.open(output_dir / "training_curve.json.gz", 'wt', encoding='utf-8') as f:
        json.dump(curve_data, f, ensure_ascii=False)

    # 2. 单步训练详细数据
    print("  [2/4] 生成单步训练详细数据...")
    step_data = generate_step_detail(500)
    with open(output_dir / "step_500.json", 'w', encoding='utf-8') as f:
        json.dump(step_data, f, ensure_ascii=False, indent=2)

    with gzip.open(output_dir / "step_500.json.gz", 'wt', encoding='utf-8') as f:
        json.dump(step_data, f, ensure_ascii=False)

    # 3. 权重分布
    print("  [3/4] 生成权重分布数据...")
    weight_bin = generate_weight_distribution()
    with open(output_dir / "weights.bin", 'wb') as f:
        f.write(weight_bin)

    with gzip.open(output_dir / "weights.bin.gz", 'wb') as f:
        f.write(weight_bin)

    # 4. README
    print("  [4/4] 生成说明文档...")
    readme = """# 训练页面数据

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
"""
    with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme)

    print(f"\n✅ 完成！数据保存到: {output_dir}")
    print(f"  - training_curve.json ({len(json.dumps(curve_data))} bytes)")
    print(f"  - step_500.json ({len(json.dumps(step_data))} bytes)")
    print(f"  - weights.bin ({len(weight_bin)} bytes)")


if __name__ == "__main__":
    main()
