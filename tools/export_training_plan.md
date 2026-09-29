# 训练页面数据导出规划

## 目标
创建类似推理页面的深度展开式训练可视化，从宏观训练曲线到微观权重比特。

## 数据结构设计

### 深度层级（参考推理页面的 D1-D7）

**D1 - 训练曲线总览**
- 损失曲线（3000步）
- 困惑度变化
- 10个检查点，每个有生成样本
- 点击任意检查点可"揭开"进入D2

**D2 - 单步训练展开**
- 选中某一步（如第500步）
- 三个阶段：前向传播 → 反向传播 → 权重更新
- 显示这一步的输入、标签、预测
- 点＋进入各阶段细节

**D3 - 前向传播逐层**
- 28层的activation逐层流动（类似推理页面的D3层塔）
- 每层显示输出范数、最大激活值
- 可选择某一层进入D4

**D4 - 前向一层之内**
- 拆开的一层：RMSNorm → 注意力 → ⊕ → RMSNorm → SwiGLU → ⊕
- 显示每个算子的输入输出
- 点＋进入具体算子

**D5 - 反向传播逐层**
- 梯度从输出层回流到输入层
- 每层显示梯度范数、最大梯度值
- 梯度可视化（颜色编码）
- 可选择某一层查看梯度细节

**D6 - 权重更新微观**
- 选择具体某个权重（如 layer14.mlp.down_proj 的某个元素）
- 显示：
  - 当前权重值：w_t
  - 该位置的梯度：g_t
  - AdamW 的 m_t（一阶动量）
  - AdamW 的 v_t（二阶动量）
  - 更新后的值：w_{t+1}
- 每一步计算公式可视化

**D7 - 权重的比特**
- 显示该权重的16个bf16比特
- 可翻转比特看数值变化
- 显示训练前后的比特差异

## 需要导出的数据

### 1. 训练曲线数据 (training_curve.json)
```json
{
  "model_config": {
    "layers": 4,
    "hidden_size": 256,
    "num_heads": 4,
    "vocab_size": 32000
  },
  "training_steps": 3000,
  "checkpoints": [
    {
      "step": 0,
      "loss": 8.524,
      "perplexity": 5041,
      "lr": 0.0001,
      "sample": "输出的文本样本",
      "quality": "gibberish"
    },
    // ... 10个检查点
  ]
}
```

### 2. 单步训练详细数据 (step_500.json)
选择第500步作为代表，导出：
```json
{
  "step": 500,
  "input_text": "春天来了",
  "input_ids": [1234, 5678],
  "label_text": "，花儿开了",
  "label_ids": [9012, 3456],
  
  "forward": {
    "layers": [
      {
        "layer_id": 0,
        "input_norm": 1.234,
        "output_norm": 2.345,
        "attn_output": [...],
        "mlp_output": [...]
      },
      // ... 4层
    ],
    "logits": [...],
    "prediction": "，花儿开了",
    "loss": 4.456
  },
  
  "backward": {
    "layers": [
      {
        "layer_id": 3,  // 反向传播从后往前
        "grad_output_norm": 0.123,
        "grad_attn": [...],
        "grad_mlp": [...]
      },
      // ... 4层
    ]
  },
  
  "weight_updates": {
    "example_weight": {
      "name": "layer.1.mlp.down_proj.weight[128][64]",
      "before": 0.023456,
      "gradient": -0.000123,
      "m_t": 0.000012,  // AdamW一阶动量
      "v_t": 0.000000015,  // AdamW二阶动量
      "after": 0.023468,
      "lr": 0.0001,
      "beta1": 0.9,
      "beta2": 0.999,
      "weight_decay": 0.01
    }
  }
}
```

### 3. 权重分布快照 (weights_distribution.bin)
在多个检查点保存权重分布缩略图（类似推理页面的 weights.bin）

### 4. 微观数据 (step_500_micro.json)
某个具体权重的详细更新过程

## 实现步骤

### 第一阶段：训练小模型并记录检查点（需要GPU，1-2小时）
```bash
python tools/train_small_model.py --output public/training/data/
```
这会：
1. 从零训练一个4层256维的Qwen3同架构模型
2. 训练3000步，每300步保存检查点
3. 记录损失、困惑度、生成样本
4. 保存10个检查点的模型权重

### 第二阶段：导出单步训练详细数据（需要修改模型代码，1-2小时）
```bash
python tools/export_training_step.py --checkpoint public/training/data/ckpt_500.pt --step 500
```
这会：
1. 加载第500步的模型
2. 用一个真实的训练样本
3. Hook住每一层的前向和反向传播
4. 记录activation、梯度、权重更新
5. 导出详细JSON

### 第三阶段：更新前端代码
修改 public/training/js/training.js 读取真实数据并实现深度展开

## 时间估算
- 训练小模型：2小时（GPU）
- 导出单步数据：1小时
- 更新前端：1小时
- 总计：**4小时**（可以用监控机制，实际人工时间约30分钟）

## 监控方案
训练脚本每10分钟输出一次进度，主会话可以设置定时检查或等待完成通知。
