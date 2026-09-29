# 多模态模型数据文件

本目录包含真实的 Qwen2-VL 视觉模型数据，用于多模态页面的可视化展示。

## 数据来源

数据通过 `tools/export_multimodal.py` 脚本生成，包含：
- 图像的 14×14 patch 切分
- Vision Transformer 的 patch embeddings（每个 patch 768 维向量）
- 逐字符生成时对各个 patch 的注意力权重分布
- 真实的问答对和模型输出

## 文件说明

### manifest.json
数据清单文件，包含所有示例的索引信息：
```json
{
  "version": "1.0",
  "model": "Qwen2-VL-2B-Instruct",
  "description": "真实的Qwen2-VL视觉模型数据",
  "examples": [...]
}
```

### 示例数据文件
每个示例一个 JSON 文件（cat.json, street.json, food.json, document.json），包含：

```json
{
  "id": "cat",
  "name": "猫咪",
  "question": "图片里有什么？",
  "answer": "图片中有一只橘色的猫咪，它看起来很可爱。",
  "image": "images/cat.jpg",
  "patches": [...],      // 196个patch的数据
  "attention": [...]     // 每个字符的注意力权重
}
```

## 数据结构

### Patches
每个 patch 包含：
- `id`: patch 编号（0-195）
- `row`: 行位置（0-13）
- `col`: 列位置（0-13）
- `embedding`: 768 维特征向量

### Attention
每个字符的注意力数据：
- `char`: 生成的字符
- `charIdx`: 字符索引
- `weights`: 对所有 196 个 patch 的注意力权重（归一化后的概率分布）

## 数据大小

每个示例文件约 4.5MB，主要是 patch embeddings 数据。

## 使用方式

前端通过 `multimodal.js` 异步加载：
1. 首先加载 `manifest.json` 获取示例列表
2. 按需加载每个示例的详细数据
3. 渲染图像、patch 切分和注意力可视化

## 重新生成数据

运行以下命令重新生成数据：

```bash
# 使用真实模型（需要安装 transformers 和 qwen_vl_utils）
python tools/export_multimodal.py --model Qwen/Qwen2-VL-2B-Instruct

# 使用模拟数据（不需要模型）
python tools/export_multimodal.py --skip-download
```
