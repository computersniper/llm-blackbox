# 揭开黑箱 · Black Box

和真实的开源大模型 **Qwen3-0.6B** 聊天，然后像调试程序一样，一层层揭开它，看它是怎么想出每一个字的。

线上地址：<https://caijiechao.com/blackbox/>

## 玩法

1. **聊天**：用点选式输入法（候选就是问题的真实分词，Qwen3 的词元和编号）拼出问题，发送。模型像平常的 AI 一样流式回答。
2. **＋ 揭开**：点消息旁的 ＋，右侧出现一台 3D 机器。每按一次 ＋ 往里钻一层（单步进入），− 退回（跳出）：

| 深度 | 看到什么 | 每一步是什么 |
| --- | --- | --- |
| D1 黑箱 | 合着的机箱，想完一个词就吐出一块 | 每个词元一步 |
| D2 结构 | 机箱揭开：词元托盘、28 层玻璃层板、残差流光柱、lm_head、采样轨道、自回归回路 | 读入 / 嵌入 / 28 层 / 输出头 / 采样 |
| D3 层塔 | 每层旁边有逻辑透镜读数，层板上有 KV 缓存和注意力光束 | 逐层 |
| D4 一层之内 | 拆开的一层：RMSNorm → 注意力 → ⊕ → RMSNorm → SwiGLU → ⊕ | 逐个算子 |
| D5 算子 | 注意力：Q·K·V / 打分 / softmax / 加权求和，16 个查询头可切换；前馈：3072 个神经元阵列 | 算子内部 |
| D6 一次乘加 | 矩阵乘法的显微镜：在真实权重面板上高亮第 j 列，第 i 行的输入走到格子 (i, j) 相乘，再沿列求和（W_q 另有 q_norm + RoPE 旋转；W_o、W_gate/W_up、W_down、输出头的 logit 同理）；一次 Q·K 打分；SwiGLU 的 SiLU 斜坡与乘法阀门 | 选一列 / 逐项相乘 / 求和 |
| D7 比特 | 被放大的那个权重的 16 个 bf16 比特键帽，可以翻转，看输出元素怎么变 | — |

每一步的调试面板都标出真实的矩阵形状（如 `h [1×1024] @ W_q [1024×2048] → q = 16 头 × 128`）；拆开的层里有按真实形状等比例的权重矩阵，面板上的明暗就是这一层真实的权重分布（每格 32×32 个权重的均方根）。

**自由移动**：左键拖动旋转，右键 / Shift 拖动平移，滚轮推进（到头会穿过去），WASD 移动，Q/E 升降，双击飞到物体跟前，F 回到跟随视角。

3. **调试器**：播放 / 暂停 / 上一步 / 下一步，0.25×–8× 倍速；右侧伪代码高亮当前行，并监视真实变量。键盘：空格、← →、＋ −、1–6 切换倍速。

一路上藏着 19 个“知识碎片”，右上角的图鉴可以查看。

## 数据是怎么来的

`tools/export_qwen.py` 在本地用 GPU 跑真实的 Qwen3-0.6B（非思考模式，T=0.7 / top_k=20 / top_p=0.8，固定种子），为每个预设问题导出：

- 聊天模板后的真实分词、生成的回答、每一步的前 12 名概率、各温度下的概率、top-k/top-p 候选池和采样用的随机数；
- 逻辑透镜（每层输出接最终 RMSNorm + 输出矩阵）的前 3 名；
- 每层每头的注意力（每行前 4 名）和 logsumexp（用来还原真实打分）；
- 每层 SwiGLU 激活的前 16 名、第 14 层完整的 3072 维；
- 焦点层（3 / 14 / 25）的真实乘加：单个 SwiGLU 神经元、一次 Q·K 打分，以及 W_q（含 q_norm、RoPE 前后的值）、W_o、W_down 各一个输出元素；
- 每一步选中词元的 logit 是怎么点积出来的；
- 每层 7 个权重矩阵的真实分布缩略图（`weights.bin`，每格 32×32 个权重的均方根）；
- 每层每个位置的残差范数。

脚本会自己复现一遍注意力（RMSNorm → RoPE → GQA），并和模型输出对比（平均误差约 3×10⁻⁴）。导出结果在 `public/data/`，数据文件额外存一份 `.gz`，网页用 `DecompressionStream` 解压（服务器只对 HTML 做 gzip）。

重新导出：

```bash
python tools/export_qwen.py --model /path/to/Qwen3-0.6B
```

需要 torch、transformers（5.x 已测试）、safetensors；模型可从 ModelScope 或 Hugging Face 下载。

## 本地运行

纯静态站点，没有构建步骤：

```bash
cd public && python3 -m http.server 8765
```

## 目录

```
public/
  js/main.js        聊天模式 / 揭开模式的切换，时间线、舞台、聊天气泡的联动
  js/chat.js        聊天面板和点选输入法（前缀树）
  js/timeline.js    调试器核心：按深度展开的步骤树、单步进入 / 跳出、播放
  js/explain.js     伪代码、每一步的讲解、变量监视（全部用真实数值）
  js/controls.js    调试器界面
  js/data.js        读取导出的数据
  js/stage/         Three.js 舞台：engine（渲染 / 相机 / 拾取）、machine（机器本体）、detail（神经元 / 打分 / 比特）
  js/vendor/three/  自托管的 Three.js r186
  data/             导出的真实模型数据
tools/              导出脚本、截图测试、字体子集、版本号、发布
deploy/             服务器端 post-receive 钩子与初始化脚本
```

改了页面上的中文标题后，重新生成字体子集：`python tools/subset_fonts.py NotoSerifSC-Black.otf NotoSerifSC-SemiBold.otf`。

## 部署

服务器上有 bare 仓库 `/srv/blackbox/repo.git`。推送 `main` 后，钩子导出 `public/`、给 CSS/JS 加 `?v=<commit>`、放进 `/srv/blackbox/releases/<时间>-<commit>/` 并原子切换 `/srv/blackbox/current`；博客根目录下的 `blackbox` 软链接指向它，nginx 不需要改动。保留最近 5 个版本。

```bash
git remote add deploy root@182.61.48.178:/srv/blackbox/repo.git   # 只需一次
git push deploy main                                               # 发布
```

更早的版本保存在 git 标签里：`v1`（九个深度的分页式下潜）、`v2`（聊天 + 调试器初版）。

## 多模态页面（/blackbox/multimodal/）

给真实的视觉语言模型 **Qwen3-VL-2B-Instruct**（ModelScope：`Qwen/Qwen3-VL-2B-Instruct`）看一张图、问一个问题，再一层层揭开它是怎么“看”的。玩法和主页面一样：先选一张图，再用点选输入法拼问题（候选是这张图预设问题的真实分词），模型流式回答；回答完把鼠标放在回复的字上，图上会亮起生成这个字时真实的注意力；点 ＋ 进入调试器。

| 深度 | 看到什么 |
| --- | --- |
| D1 黑箱 | 照片从取景窗塞进机箱，之后每个词元一步 |
| D2 流水线 | 预处理 → 视觉编码器 → 合并器 → 拼进对话 → 语言模型 → 输出；第二个词元起只剩 读入 / 语言模型 / 输出（图片在 KV 缓存里） |
| D3 逐层 | 缩放到 32 的倍数、切成 16×16 图块、归一化（单张图复制成 2 帧）；图块嵌入、位置嵌入；ViT 24 层（每层特征的主成分颜色、平均注意距离）；2×2 合并、MLP、DeepStack；聊天模板、M-RoPE 三维位置；LLM 28 层（生成当前这个字时对图片的注意力热力图、逻辑透镜、图片词元的透镜读数） |
| D4 一层之内 | ViT 块与 LLM 层内部的算子；ViT 第 0/5/11/17/23 层可以点任意图块看它的注意力；LLM 前 3 层的 DeepStack 注入；图块嵌入 = 一组卷积核 |
| D5 一次乘加 | 一个图块的 1536 个输入（3 通道 × 2 帧 × 16 × 16）乘第 c 个卷积核再加偏置，得到嵌入的一个数；第 17 / 20 层 16 个头各自在看图的哪里 |

左上角的监视器是 2D 画布，显示这张图在模型里“此刻的样子”；右侧是伪代码、讲解和变量监视，全部用真实数值。

### 数据

`tools/multimodal/export_qwen3vl.py` 在本地 GPU 上运行模型（bf16，eager 注意力，显存约 5 GB），贪心解码，导出到 `public/multimodal/data/`：

- 模型真正看到的像素（从 `pixel_values` 还原）、网格尺寸、聊天模板后的分词、M-RoPE 的 (t, h, w) 位置；
- ViT 每一级的特征主成分颜色、各头平均注意距离、第 0/5/11/17/23 层合并到 2×2 词元分辨率的完整注意力（脚本自己复现 qkv → 2D RoPE → softmax，并和模块输出核对，平均相对误差约 4×10⁻³）；
- 一个图块 → 嵌入一维的真实乘加（像素、卷积核权重、偏置、位置嵌入）和 16 个卷积核；
- 合并器与三路 DeepStack 输出的主成分颜色、长度；图片词元在 LLM 每层之后的逻辑透镜读数；
- 每个生成步：前 8 名候选概率、每层逻辑透镜前 3 名、每层（16 头平均）对每个视觉词元的注意力、对文字的前 4 名、每头分给图片的比例、第 17 / 20 层每个头在图片上的注意力。

`tools/multimodal/grounding.py` 用几张图上手工标的物体区域，量了“生成某个词时注意力是否落在对应物体上”：第 16–26 层的富集倍数是 3–5 倍，更浅的层基本不看物体，所以网页的总览热力图默认用这几层平均。

数据约 2.5 MB（gzip），另有 7 张图约 0.2 MB。

重新生成：

```bash
# 1. 画图（需要 Noto Sans SC 可变字重 TTF 和 Noto Serif SC，SIL OFL）；两张照片见下
python tools/multimodal/make_images.py --sans NotoSansSC.ttf --serif NotoSerifSC-SemiBold.otf \
    --aldrin Aldrin_Apollo_11_original.jpg --starry Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg
# 2. 导出（需要 torch、transformers 5.x、torchvision、Pillow）
python tools/multimodal/export_qwen3vl.py --model /path/to/Qwen3-VL-2B-Instruct
# 可选：对准层分析
python tools/multimodal/grounding.py
```

### 图片来源

`tools/multimodal/images/` 里是导出脚本的输入，`public/multimodal/data/img/` 是模型实际看到的缩放结果。

- `shapes.png`、`apples.png`、`poem.png`、`chart.png`、`night.png`：`make_images.py` 用 PIL 画的（诗笺用 Noto Serif SC，柱状图用 Noto Sans SC）。
- `moon.jpg`：*Aldrin Apollo 11*（AS11-40-5903），Neil Armstrong / NASA 拍摄，**公共领域**。来源：<https://commons.wikimedia.org/wiki/File:Aldrin_Apollo_11_original.jpg>
- `starry.jpg`：*The Starry Night*（1889），Vincent van Gogh，**公共领域**。来源：<https://commons.wikimedia.org/wiki/File:Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg>

两张照片缩小到长边 640 像素后存入仓库。
