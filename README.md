# 揭开黑箱 · Black Box

和真实的开源大模型 **Qwen3-0.6B** 聊天，然后像调试程序一样，一层层揭开它，看它是怎么想出每一个字的。

线上地址：<https://caijiechao.com/blackbox/>

站点有四个页面，顶栏导航互通，交互方式一致（点选输入、＋ 逐层揭开、调试器式播放），数据都来自真实模型的离线运行：

| 页面 | 地址 | 内容 |
| --- | --- | --- |
| 推理 | `/blackbox/` | Qwen3-0.6B 如何想出每一个字（本文前半部分） |
| 训练 | `/blackbox/train/` | 从零训练一个 Qwen3 同构的唐宋诗小模型；在 Qwen3-0.6B 上真实走三步 SFT |
| 多模态 | `/blackbox/multimodal/` | Qwen3-VL-2B 怎么看图：切块、ViT、合并、插进对话，生成每个字时在看哪里 |
| 智能体 | `/blackbox/agent/` | 编程 agent 怎么在终端里干活：Qwen3-4B 在沙箱里真实录制的 模型 + 工具 + 循环 |

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
| D6 一次乘加 | 矩阵乘法的显微镜：在真实权重面板上高亮第 j 列，输入（蓝）沿第 i 行走到格子 (i, j)（紫），乘积（橙）顺着这一列飞进输出格（青）。舞台下方的**算式板**把它写成 y[j] = Σ x[i] × W[i, j]：前 12 项逐项相乘、飞进累加器，“其余 N−12 项”合成一行，加起来正好等于真实结果；附正 / 负乘积之和、按贡献排序的累计和曲线，以及结果落进输出向量的哪一格、是什么意思（W_q 另有 q_norm + RoPE 旋转；W_o、W_gate/W_up、W_down、输出头的 logit 同理）；一次 Q·K 打分；SwiGLU 的 SiLU 曲线与乘法阀门。28 层每一层、每个生成的词元都有 | 选一列 / 逐项相乘 / 求和 |
| D7 比特 | 被放大的那个数的 16 个 bf16 比特：符号 / 指数 / 尾数三组，算式板写出 值 = (−1)^s × 2^(e−127) × (1 + m/128)。点任意一位翻转（舞台键帽或算式板），精确重算：旧权重 → 新权重、旧输出 → 新输出和差值（神经元输出、注意力权重、候选词的概率也一起重算）。在 D6 的乘法这一步点算式板的某一行，就看那一项 | — |

每一步的调试面板都标出真实的矩阵形状（如 `h [1×1024] @ W_q [1024×2048] → q = 16 头 × 128`）；拆开的层里有按真实形状等比例的权重矩阵，面板上的明暗就是这一层真实的权重分布（每格 32×32 个权重的均方根）。

**自由移动**：左键拖动旋转，右键 / Shift 拖动平移，滚轮推进（到头会穿过去），WASD 移动，Q/E 升降，双击飞到物体跟前，F 回到跟随视角。

3. **调试器**：播放 / 暂停 / 上一步 / 下一步，0.25×–8× 倍速；右侧伪代码高亮当前行，并监视真实变量。键盘：空格、← →、＋ −、1–6 切换倍速。左侧聊天栏、右侧调试器的边缘可以拖动调整宽度（双击恢复默认，宽度会记住；四个页面都一样，手机上不能拖）。

一路上藏着 19 个“知识碎片”，右上角的图鉴可以查看。

## 数据是怎么来的

`tools/export_qwen.py` 在本地用 GPU 跑真实的 Qwen3-0.6B（非思考模式，T=0.7 / top_k=20 / top_p=0.8，固定种子），为每个预设问题导出：

- 聊天模板后的真实分词、生成的回答、每一步的前 12 名概率、各温度下的概率、top-k/top-p 候选池和采样用的随机数；
- 逻辑透镜（每层输出接最终 RMSNorm + 输出矩阵）的前 3 名；
- 每层每头的注意力（每行前 4 名）和 logsumexp（用来还原真实打分）；
- 每层 SwiGLU 激活的前 16 名、第 14 层完整的 3072 维；
- **全部 28 层**、每个生成词元的真实乘加：单个 SwiGLU 神经元（这一步激活最大的那个）、一次 Q·K 打分，以及 W_q（含 q_norm、RoPE 前后的值）、W_o、W_down 各一个输出元素；
- 每一步选中词元的 logit 是怎么点积出来的；
- 上面每一次点积除了贡献最大的 12 项，还附带全部 n 项的汇总（n = 1024 / 2048 / 3072 / 128）：正乘积之和、负乘积之和，以及按 |乘积| 从大到小排好后的累加和（在第 1、2、4、…、2048 和第 n 项处采样，最后一个就是真实结果）；
- 每层 7 个权重矩阵的真实分布缩略图（`weights.bin`，每格 32×32 个权重的均方根）；
- 每层每个位置的残差范数。

脚本会自己复现一遍注意力（RMSNorm → RoPE → GQA），并和模型输出对比（平均误差约 3×10⁻⁴）。导出结果在 `public/data/`，数据文件额外存一份 `.gz`，网页用 `DecompressionStream` 解压（服务器只对 HTML 做 gzip），不支持时退回未压缩的版本。

网页载入的数据分三块（前两块按问题，第三块所有问题共用）：

- **主文件** `qNN.json` + `qNN.bin`：发出问题、回答开始流式输出时就在后台取（gz 后 60–320 KB），点 ＋ 时一般已经到了；
- **“一次乘加”的分块** `qNN/L00.json.gz` … `qNN/L27.json.gz`：每层一个文件，装着这一层所有生成词元的乘加数据（每个 8–41 KB，一个问题 28 层合计 0.2–1.1 MB）。分块**只存 .gz**，不存原始文件；浏览器不支持 `DecompressionStream` 时看不到 D6 的乘加细节（会提示“当前浏览器不支持解压”），其余深度不受影响。网页按需载入：深度 ≥ 5 停在第 L 层时，预取第 L 层和上下相邻两层；真走进某层的 D6 而数据还没到，时间线就在这一步原地等（舞台先画上一级，顶部提示“正在载入第 L 层的乘加数据…”），到了再继续；
- **权重缩略图** `weights.bin`（gz 后约 380 KB）：只有拆开一层（D4）时才用得到，深度 ≥ 3 时才开始取，没到之前权重面板先用装饰纹理。

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
  js/resize.js      左右面板拖动调整宽度（四个页面共用：手柄、键盘、双击复位、存进 localStorage）
  js/prefetch.js    后台预取：聊天页空闲时用 modulepreload 一次性取回舞台的整张模块图（省流量 / 2G 时跳过）
  js/stage/         Three.js 舞台：engine（渲染 / 相机 / 拾取）、machine（机器本体）、detail（神经元 / 打分 / 比特）、
                    mats（权重面板）、micro（矩阵乘法显微镜）、board（D6 / D7 的算式板）
  js/vendor/three/  自托管的 Three.js r186
  data/             导出的真实模型数据（qNN.json / qNN.bin 主文件，qNN/Lxx.json.gz 按层的乘加分块）
tools/              导出脚本、截图测试、字体子集、版本号、发布
deploy/             服务器端 post-receive 钩子与初始化脚本
```

标题用的衬线体（思源宋体，SIL OFL）每个字重拆成两个子集：`-core` 只含首屏一定会用到的字（四个页面 HTML 里的静态文字、开场卡片的大标题、任务卡片等；600 约 60 KB、900 约 20 KB），`-ext` 是其余的字。CSS 用 `unicode-range` 声明两段，页面真的渲染到 ext 里的字时浏览器才去下载它。改了页面上的中文文案后，重新生成字体子集（脚本会同时改写 `css/app.css` 和 `train/css/train.css` 里的 `@font-face`）：`python tools/subset_fonts.py NotoSerifSC-Black.otf NotoSerifSC-SemiBold.otf`。

## 部署

服务器上有 bare 仓库 `/srv/blackbox/repo.git`。推送 `main` 后，钩子导出 `public/`、给 CSS/JS 加 `?v=<commit>`、放进 `/srv/blackbox/releases/<时间>-<commit>/` 并原子切换 `/srv/blackbox/current`；博客根目录下的 `blackbox` 软链接指向它，nginx 不需要改动。保留最近 5 个版本。

```bash
git remote add deploy root@182.61.48.178:/srv/blackbox/repo.git   # 只需一次
git push deploy main                                               # 发布
```

更早的版本保存在 git 标签里：`v1`（九个深度的分页式下潜）、`v2`（聊天 + 调试器初版）。

## 训练页面（`/blackbox/train/`）

看一个大模型是怎么被训练出来的。两段训练都是在本地 RTX 5090 上真实跑出来的，网页只回放导出的记录；标“示意”的部分不是实测。

| 深度 | 看到什么 | 每一步是什么 |
| --- | --- | --- |
| D0 流水线 | 预训练 → 监督微调 → 偏好对齐 / 强化学习（第三段为示意，其中 DPO 损失是用真实对数概率代入公式算的一次） | 每个阶段 |
| D1 训练全程 | 小模型：损失 / 学习率 / 梯度范数曲线、此刻生成的诗、留出的《登鹳雀楼》逐字概率、注意力头、嵌入 PCA、权重局部、损失地形；Qwen3：3 步 SFT 中回答每个词元的概率 | 每个检查点（小模型 41 个；Qwen3 3 步） |
| D2 一步之内 | 训练循环：批次 → 前向 → 损失 → 反向 → 更新 →（演示）重新前向 | 环节 |
| D3 拆开环节 | 批次：取一行 / 编号 / 错开一位（Qwen3：聊天模板 / SFT 遮罩）；前向：逐层的逻辑透镜；损失：逐个位置的 −ln p；反向：逐层的 ‖∂L/∂h‖；更新：一个权重的 AdamW 算式 | 子步骤 |
| D4 细到一个数 | 一个位置的交叉熵（softmax → 取概率 → −ln）；一层 7 个矩阵各自的梯度；写回的那个权重的 fp32 / bf16 比特 | 子步骤 |

操作和主页面一致：播放 / 暂停 / 上一步 / 下一步、0.25–8× 倍速，＋ 单步进入、− 跳出；舞台可以拖动平移、滚轮或双指缩放、双击放大，点卡片飞过去；19 个知识碎片。

### 小模型：从零预训练

- **语料**：[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（MIT 许可，诗歌本身属公有领域）的《全唐诗》《全宋诗》，取自提交 `b8594f8`。用 OpenCC 繁转简，只留“，。”交替的五言 / 七言绝句和律诗，去重，丢掉含低频字（出现不到 3 次）的诗：训练集 215,116 首 / 990 万字，验证集 4,390 首。《登鹳雀楼》被整首留出，训练时从没见过。
- **模型**：transformers 的 `Qwen3ForCausalLM`，改小尺寸：6 层、隐藏维 256、4 个查询头 / 2 个键值头（GQA）、每头 64 维、q_norm / k_norm、RMSNorm、RoPE（θ = 10⁴）、SwiGLU 768、输入输出共享嵌入；字符级词表 7,478（含 `<|endoftext|>`），共 664 万参数。
- **训练**：4,000 步 × 64 行 × 128 字（约 3,277 万字、3.3 轮），AdamW（β = 0.9 / 0.95，权重衰减 0.1，RMSNorm 不衰减），峰值学习率 3e-3、预热 200 步后余弦退火到 3e-4，梯度裁剪 1.0；bf16 autocast 前向，fp32 主权重。训练本身约 3 分钟（含检查点记录共 275 秒），验证损失 8.94 → 4.05。
- **记录**：每一步的损失、学习率、梯度范数；41 个检查点上的生成样例（同一组随机数）、留出诗的逐字概率与所有注意力头、嵌入 PCA（逐帧 Procrustes 对齐）、三块权重和梯度的 48×48 局部、批次第 0 行的逐字损失 / 逻辑透镜 / 残差范数 / 残差梯度 / 更新后重新前向、每个参数张量的梯度范数、4 个权重的完整 AdamW 中间量（脚本会用公式重算一遍核对，误差在 5 个 fp32 ulp 以内）；最后在检查点轨迹的前两个主成分平面上算 41×41 的损失地形（两个主成分解释约 97% 的方差）。

### Qwen3-0.6B：三步 SFT

主页面用的同一个 Qwen3-0.6B，全参数、fp32。一条“自我认知”对话（“你是谁？”→“我是黑箱里的小模型。”）套上非思考模式的聊天模板（24 个词元），提示部分标签设为 −100，只对回答的 8 个位置算损失；在同一条对话上连走 3 步 AdamW（lr 1e-5，β = 0.9 / 0.95，权重衰减 0.1，裁剪 1.0；m、v 放在内存里，公式与 `torch.optim.AdamW` 相同）。回答的损失 5.02 → 1.38 → 0.52 → 0.011。记录逐词元概率与前 5 名、两种损失（只算回答 / 全都算）、逻辑透镜、残差流梯度、每个参数张量的梯度范数、4 个权重的 g / m / v / 偏差校正 / Δw，以及 3 步后有多少权重存回 bf16 会变回原值（85%）。DPO 一例：新回答和原模型自己的贪心回答，在微调前后的模型上算对数概率，代入 DPO 公式（β = 0.1）。

### 重新生成

```bash
# 语料（放 D 盘）：全唐诗 poet.tang.*.json → tang/，全宋诗 poet.song.*.json → song/（都在原仓库的“全唐诗”目录里）
python tools/train/prep_corpus.py --src /mnt/d/cjc/datasets/chinese-poetry --out /mnt/d/cjc/datasets/poetry-train
python tools/train/train_tiny.py            # 约 5 分钟，显存 < 2 GB；--export-only 只用上次的记录重新导出
python tools/train/qwen_step.py --model /mnt/d/cjc/model-weights/qwen3/Qwen3-0.6B   # 约 6 GB 显存
python tools/train/split_data.py --from-git <提交>   # 只重新切分：从某个提交里的旧格式整份文件切，不用 GPU
```

需要 torch、transformers（5.x）、numpy、opencc。页面代码在 `public/train/`：`js/data.js`（数据和分块载入器）、`js/timeline.js`（步骤树）、`js/explain.js`（伪代码 / 讲解 / 变量）、`js/stage.js`（2D 舞台与相机）、`js/scenes/`（各层视图）。

### 数据与载入

导出脚本最后都交给 `tools/train/split_data.py`，把数据拆成“首屏小文件 + 按需分块”，写完再读回来和原始数组逐字节、和元数据逐个数核对（只重新排布，不改任何数值）。全部在 `public/train/data/`，gzip 后合计约 1.16 MB：

| 文件 | 内容 | 什么时候取 | 大小（gzip） |
| --- | --- | --- | --- |
| `tiny.json` + `tiny.bin` | 元数据（配置、语料、词表、41 个检查点的标量和生成的诗）、每一步的损失 / 学习率 / 梯度范数、批次第 0 行的字、损失地形 | 首屏 | 43 + 40 KB |
| `qwen.json` | Qwen3 的对话、逐词元概率与前 5 名、两种损失、DPO、改动统计 | 首屏 | 10 KB |
| `tiny/ck00…40.bin` | D1：留出诗的逐字概率与前 5 名、24 个注意力头（只存下三角）、嵌入 PCA、三块权重 / 梯度局部 | 进入 D1、播放或拖到这个检查点 | 每块约 19 KB |
| `tiny/st00…40.bin` | D2–D4：批次第 0 行的概率与前 5 名、更新后的概率、逻辑透镜、残差范数与梯度、张量梯度范数、AdamW 算式 | 进入一步之内 | 每块约 5 KB |
| `tiny/feat.bin` | 4 个权重每 4 步一个点的 w / g / m / v | D2 起提前取（D3 更新要用） | 48 KB |
| `qwen/st0…2.json` | 每一步的逻辑透镜、残差范数 / 梯度、每个参数张量的梯度范数 | 进入 Qwen3 的 D1 时 | 每块 12–15 KB |

首屏文件另留一份未压缩的（给不支持 `DecompressionStream` 的浏览器）；分块只存 `.gz`，这类浏览器里对应的卡片会写明“不支持解压”。浮点数组按字节分面存放（先放所有数的第 0 个字节……），gzip 能多压 10–30%。

页面（`js/data.js` 的 `Loader`、`js/main.js` 的 `plan()`）每帧算出当前画面要用的分块：这一屏要用的立即取，播放方向上后几个检查点、相邻检查点、往下一层要用的排队预取（同时最多 4 个）；首屏字体下完后在浏览器空闲时一块一块地后台预取（省流量模式 / 2G 不预取），约 10 秒（8 Mbps）取完。数据没到时：D1 的卡片单独显示载入占位，或者先拿最近的已到检查点顶上（调暗并标注是第几步）；D2–D4 整屏先显示最近的已到检查点并在顶部标注；播放时缺块就原地等到了再走；提示都延迟 0.25 秒再淡入，数据很快到时不会闪。失败后自动重试（间隔从 4 秒翻倍到 30 秒）。

在本地模拟 8 Mbps / 60 ms（冷缓存）：首屏从 2.2 秒、1.9 MB（其中数据 1.1 MB）降到约 0.7 秒、0.25 MB（数据 93 KB）；首屏后立刻进入各层，第一次要等的分块约 0.1–0.15 秒，在首屏停留几秒后基本不用等。

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

载入：选图时就在后台取这张图的视觉侧数据，发出问题时取语言侧数据；聊天页空闲时预取 3D 舞台的模块（和推理页共用 `js/prefetch.js`），点 ＋ 时模块和数据并行等待，一般都已经在缓存里。

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

## 智能体页面（/blackbox/agent/）

展示像 Claude Code 这样的编程 agent 是怎么在终端里干活的：**模型 + 工具 + 循环**。页面不冒用任何商业产品的界面或输出，用的是一个真实的开源模型在真实沙箱里的真实运行。

- **玩法**：在任务库里点一张卡片，或者用点选输入法（任务句子的真实分词）拼出任务。先按正常速度看 agent 在屏幕上干活：编辑器里打开、修改文件（diff 高亮），终端里跑命令、输出滚出来、测试变绿，最后给出回答。然后点 ＋ 揭开：

| 深度 | 看到什么 |
| --- | --- |
| D1 屏幕 | 文件树、编辑器、终端，按真实工具调用回放 |
| D2 循环 | 模型 ⇄ harness ⇄ 沙箱，一圈一圈：拼上下文 → 生成 → 解析 &lt;tool_call&gt; → 沙箱执行 → 结果接回对话 |
| D3 上下文 | 每一圈喂给模型的整段上下文：按系统提示 / 工具定义 / 任务 / 模型输出 / 工具结果分段的真实词元数，越滚越长的阶梯，KV 缓存复用了多少、新算了多少，以及聊天模板渲染后的原文 |
| D4 生成 | 一个词元一步，关键决策处停下来看前 5 名候选的真实概率；最后看 harness 怎么把 JSON 解析出来 |
| D5 模型内部 | 每个词元一次完整的 36 层前向计算 + 温度 / top-k / top-p 抽签（候选池按真实概率重新归一化），并链接回推理页面 |

调试器（播放 / 暂停 / 单步 / 0.25–8×、＋ / −）的伪代码就是 harness 的 agent 循环，变量监视全部是录制下来的数值。一路上有 14 个知识碎片。

- **模型**：`Qwen3-4B-Instruct-2507`（bf16，约 8 GB 显存），非思考模式，T=0.7 / top_k=20 / top_p=0.8。先试过 `Qwen3-1.7B`：“算出销售冠军”和“重命名一个函数”两个任务换了 3 个随机种子都没做完（反复执行同一条命令、改不到所有引用），所以换成了能稳定完成的 4B。
- **沙箱**：每个任务把 `tools/agent/sandbox/<任务>/` 复制到临时目录，用 bubblewrap（`bwrap`）挂成 `/work`：系统目录只读、只能写 `/work`、`--unshare-all` 断网、清空环境变量、每条命令 10 秒超时。
- **工具**：`bash`、`read_file`、`write_file`、`edit_file`（字符串替换，原文必须恰好出现一次）。系统提示里写了工作方式和工作目录里的文件列表。
- **任务**：修好失败的测试（`fix-median`）、统计日志里的错误（`log-errors`）、算出销售冠军（`sales-top`）、重命名一个函数（`rename-func`）。每条轨迹录完后，脚本在沙箱里**独立检查**是否真的完成（跑测试、核对数字、确认测试文件只改了名字、脚本真的跑通），没通过就换下一个随机种子重录；尝试记录（包括失败原因）都写进 manifest，页面上如实显示。轨迹里保留了模型自己犯错再纠正的过程：`python` 不存在就换 `python3`，装不了 pandas 就改用 csv 模块。
- **录下来的真实信号**：每一轮聊天模板渲染后的完整上下文和真实词元数（按消息分段）、与上一轮相比能复用的 KV 缓存长度（真的用 `DynamicCache.crop` 复用）、逐词元输出和每个词元前 5 名候选概率、候选池大小、解析出的工具调用、沙箱里的命令输出 / 退出码 / 耗时、每次工具调用前后的文件快照与 diff、预填充与生成的实测耗时。

重新录制（需要 GPU、bubblewrap）：

```bash
python tools/agent/record.py --model /path/to/Qwen3-4B-Instruct-2507               # 录全部任务
python tools/agent/record.py --model ... --only fix-median --seeds 0,1,2            # 只录一个
python tools/agent/record.py --model ... --dry                                       # 只看结果不写文件
python tools/agent/record.py --model ... --chips-only                                # 只重建输入法候选
```

数据在 `public/agent/data/`（`manifest.json` + 每个任务一个 JSON，另存 `.gz`，合计约 70 KB gz）。页面代码在 `public/agent/`：`js/timeline.js`（步骤树）、`js/screen.js`（编辑器 / 终端回放）、`js/views.js`（循环 / 上下文 / 生成 / 模型内部）、`js/explain.js`（伪代码、讲解、变量）、`js/chat.js`（任务库、输入法、对话记录）。

## 推理视频

`tools/video/` 用网站的 3D 舞台和真实数据做了一支 3 分 55 秒的片子《AI 的一个字是怎么思考出来的——走进大模型推理的“黑箱”》。问题是「天空为什么是蓝色的？」。

片子从一个干净的聊天页开始：用拼音一个字一个字打出问题，按发送，问题变成消息气泡，助手显示“正在思考…”。接着镜头推近气泡，文字按真实分词裂成词元块，变成 3D 方块飞进黑箱、落到托盘上（一个连续跟随镜头），片名落在黑箱正面。之后依次是：

- 嵌入；
- 穿过 28 层：逻辑透镜在第 21 层说「因为」，第 24 层改口成「天空」；
- 拆开第 24 层：RMSNorm、Q/K/V、16 个头、一次 Q·K 打分、残差、SwiGLU、一次乘加、bf16 比特；
- 输出头、采样，自回归写完整句回答。

配乐由 `compose.py` 用 numpy / scipy 现场合成。

| 文件 | 作用 |
| --- | --- |
| `film.html` / `film.js` / `film.css` / `board.css` | 电影模式页面：复用 `public/js/stage/` 的机器和算式板，`__film.renderAt(t)` 确定地渲染第 t 秒 |
| `score.js` | 分镜表：每个镜头的机器状态、机位（样条 + 单调三次插值）、字幕、数据条、配乐事件（96 BPM，段落卡在小节线上） |
| `lib/` | 渲染器（多重采样、泛光、景深）、运镜工具、叠加层；`opening.js` 是开场：聊天页（打字、输入法候选、气泡、推近、裂成词元），以及词元交给 3D 方块以后的飞行与落位 |
| `render.mjs` | 无头 Chromium 逐帧截图（WSL 下走 Mesa d3d12 用 GPU），也能抽单帧、导出事件、出封面；WebGL 上下文丢失（GPU 进程崩溃）时自动重开浏览器重渲 |
| `compose.py` | 原创配乐：铺底、琶音、贝斯、鼓、钟和音效，按 `events.json` 对齐画面；母带 −14 LUFS、真峰值 −1 dBTP |
| `encode.sh` | 帧 + 配乐 → H.264（crf 18、slow、yuv420p、+faststart）+ AAC 192k |
| `share.sh` | 分享用小体积版本：1080p30，两遍编码 4.5 Mbps + AAC 192k，约 138 MB |
| `review.py` | 抽帧拼成带时间码的联系表，检查用 |
| `dbg.mjs` | 调试：跳到第 t 秒，在页面里执行一段表达式（可顺带截图） |
| `serve.py` | 本地服务器（仓库根目录 + D 盘上的字体） |

重新生成（帧序列、配乐、成片都放 D 盘，不放仓库里）：

```bash
# 需要 /mnt/d/cjc/videos/llm-inference/fonts/ 下的 NotoSerifSC-{Black,SemiBold}.otf 和 NotoSansSC-VF.ttf（都是 SIL OFL）
python tools/video/serve.py --port 8776 &
O=/mnt/d/cjc/videos/llm-inference
node tools/video/render.mjs frames --out $O/frames60 --fps 60 --workers 4   # 约 3 分钟；已有的帧会跳过，中断了可以接着渲
node tools/video/render.mjs events --out tools/video/events.json
/mnt/d/cjc/venvs/blackbox/bin/python tools/video/compose.py --events tools/video/events.json --out $O/score.wav
bash tools/video/encode.sh $O/frames60 $O/score.wav $O/qwen3-inference-v4.mp4 60
bash tools/video/share.sh $O/frames60 $O/score.wav $O/qwen3-inference-v4-share.mp4
node tools/video/render.mjs poster --out $O/poster-v4.png                   # 默认取片名落版那一刻
python tools/video/review.py --frames $O/frames60 --fps 60 --every 2 --out $O/review   # 可选：联系表
```

浏览器里预览：<http://127.0.0.1:8776/tools/video/film.html?preview&t=0>（拖时间轴、空格暂停）。`render.mjs` 依赖 playwright-core，并把 `LD_LIBRARY_PATH` 指向 chromium 的依赖库（脚本里写好了这台 WSL 的路径）。
