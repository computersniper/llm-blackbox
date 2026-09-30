# Qwen2-VL 真实运行记录

`manifest.json` 的 `source: model` 标识四份真实模型导出：`cat.json`、`street.json`、`food.json`、`sign.json`。对应照片保存在 `../images/`，每份数据包含输入照片的 SHA-256，可核对图片与问答的对应关系。旧版随机向量与热图已移除。

导出环境：本地完整的 `Qwen2-VL-2B-Instruct` 权重，transformers 4.57.6，CUDA，`tools/export_multimodal.py`。导出时限制图像预处理面积为 `256×28×28` 像素；JSON 同时保存原图尺寸、处理后的网格尺寸和模型实际使用的 `image_grid_thw` 推导网格。回答以 `do_sample=False` 贪心解码，最多 64 个输出词元。

`patches` 是视觉编码器经 2×2 空间合并后注入语言序列的最终向量摘要：每个词元保存 1536 维向量的范数和前 12 个分量；未保存 ViT 中间层或完整向量。`tokens[].weights` 来自生成该词元时语言模型最后一层的自注意力，对 12 个头求平均；`tokens[].heads` 保存每个头。热力图在图像词元内归一化，`imageAttentionMass` 单独记录归一化前分配给图像词元的注意力总量。该热图是注意力权重，不构成因果归因。模型回答可能包含识别或表述错误，页面忠实展示原始输出。

复现：`python tools/export_multimodal.py`，默认从 `D:/models/photos` 读取四张照片。`--only cat --probe --max-new-tokens 8` 可先验证一张图而不写入站点。缺模型、缺照片或注意力无法提取时脚本会报错，不生成模拟替代数据。
