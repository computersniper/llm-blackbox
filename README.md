# 潜入黑箱 · Black Box

一个可交互的大模型内部探索网站：从你写下的一句话出发，一层层下潜到组成模型的最小单位——一个比特，再上浮回来，看它说出下一个词。

线上地址：<https://caijiechao.com/blackbox/>

## 九个深度

| 深度 | 场景 | 可以做什么 |
| --- | --- | --- |
| 00 地表 | 3D 黑箱 + 输入框 | 输入一句话，字符飞进黑箱，镜头冲进去 |
| 01 分词 | 激光切句、词元编号 | 分词实验室：数字、emoji、长单词 |
| 02 嵌入 | 768 维条形码、语义地图 | 向量算术（国王 − 男人 + 女人）、位置编码 |
| 03 层塔 | 12 层等轴测塔、残差流 | 逻辑透镜：逐层看模型的“猜测”怎样成形 |
| 04 注意力 | 弧线、注意力矩阵、12 个头 | 前一词头 / 汇聚头 / 归纳头 / 因果遮罩 |
| 05 前馈网络 | 3072 个神经元阵列 | 稀疏激活、多义神经元 |
| 06 神经元内部 | 逐项乘加、激活函数 | 拖动权重，切换 GELU / ReLU / SiLU |
| 07 比特 | bf16 的 16 个比特 | 翻转比特、FP32/BF16/FP16/INT8/INT4 量化、规模蒙太奇 |
| ↑ 输出 | 概率分布 + 转盘 | 温度、贪心解码、自回归生成 |

一路上藏着 22 个“知识碎片”，靠互动解锁，右上角的图鉴可以查看。

模型结构取自 GPT-2 Small（12 层 · 12 头 · 768 维 · 3072 个 MLP 神经元）。画面中的数值由 `public/js/model.js` 里的确定性规则生成，用来直观展示真实模型中会发生的现象，并非真实模型的推理结果。

## 本地运行

纯静态站点，没有构建步骤：

```bash
cd public && python3 -m http.server 8765
# 打开 http://127.0.0.1:8765/
```

## 目录

```
public/            网站本体（部署的就是这个目录）
  js/main.js       外壳：路由、缩放转场、深度仪表、旁白、图鉴
  js/model.js      示意模型：分词器、嵌入、注意力、神经元、预测、浮点编码
  js/levels/       九个深度，每个导出 { key, name, en, scale, mount(el, app) }
  fonts/           自托管字体（思源宋体子集 + JetBrains Mono）
deploy/            服务器端：post-receive 钩子与一次性初始化脚本
tools/             截图测试、字体子集、版本号、发布脚本
```

改了页面上的中文标题后，重新生成字体子集（否则新字会回退到系统字体）：

```bash
pip install fonttools brotli
python tools/subset_fonts.py NotoSerifSC-Black.otf NotoSerifSC-SemiBold.otf
```

## 部署

服务器上有一个 bare 仓库 `/srv/blackbox/repo.git`。推送 `main` 后，`post-receive` 钩子会导出 `public/`、给 CSS/JS 加上 `?v=<commit>`、放进 `/srv/blackbox/releases/<时间>-<commit>/`，再原子地切换 `/srv/blackbox/current`。博客根目录下的 `blackbox` 是指向它的软链接，nginx 不需要改动。保留最近 5 个版本。

```bash
git remote add deploy root@182.61.48.178:/srv/blackbox/repo.git   # 只需一次
git push deploy main                                               # 发布
```

回滚：在服务器上把 `/srv/blackbox/current` 指回 `releases/` 里的旧目录即可。

首次初始化服务器：

```bash
scp deploy/post-receive root@182.61.48.178:/tmp/blackbox-post-receive
ssh root@182.61.48.178 'sh -s' < deploy/setup-server.sh
```
