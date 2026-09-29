# 分支说明

## cjc-mock-data（模拟数据分支）
**状态：✅ 完成，可部署**

包含3个新页面的高质量模拟数据：
- 训练页面：模拟的训练曲线和单步训练数据
- 多模态页面：模拟的图像切块和注意力数据  
- 智能体页面：模拟的agent工作流

适合用于演示和测试，无需GPU和真实模型。

## cjc-real-data（真实数据分支）
**状态：🚧 待完成，需要真实模型运行**

需要完成的工作：
1. **训练页面** - 运行 `tools/train_small_model.py` 生成真实训练数据
2. **多模态页面** - 运行 `tools/export_multimodal.py` 使用真实Qwen-VL模型
3. **智能体页面** - 运行 `tools/export_agent.py` 使用真实Qwen3模型录制

## cjc-local（本地开发分支）
之前的工作分支，包含D6/D7可视化改进等。

## 建议
- 先部署 **cjc-mock-data** 让页面可用
- 后续有GPU环境时完善 **cjc-real-data**
