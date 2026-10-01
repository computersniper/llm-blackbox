// 调试器里的“代码 / 这一步 / 变量”三块内容，全部用真实导出的数值填充。
import { esc, tokHTML, fmtPct } from '../../js/ui.js';
import { DEPTH_NAMES } from './timeline.js';

const K = (s) => `<span class="kw">${s}</span>`;
const F = (s) => `<span class="fn">${s}</span>`;
const C = (s) => `<span class="cm"># ${s}</span>`;
const N = (s) => `<span class="nu">${s}</span>`;

export function renderCode(el, M, V) {
  const v = M.model.vision, t = M.model.text;
  const g = V ? `${V.gh}×${V.gw}` : 'h×w';
  const CODE = [
    `${K('def')} ${F('chat')}(image, question):`,
    `    px = ${F('resize')}(image, ${N(32)} ${K('的倍数')})       ${C(V ? `${V.gw * 16}×${V.gh * 16}` : '缩放')}`,
    `    x = ${F('patchify')}(px, ${N(16)})            ${C(V ? `${g} = ${V.Np} 块` : '切块')}`,
    `    x = (x - ${N('0.5')}) / ${N('0.5')}               ${C('像素 → [-1, 1]，复制成 2 帧')}`,
    `    v = x @ W_patch + b               ${C(`${v.inDim} → ${v.hidden}`)}`,
    `    v = v + ${F('interp')}(pos_embed, ${g})   ${C(`${v.posGrid}×${v.posGrid} 位置表`)}`,
    `    ${K('for')} i, blk ${K('in')} ${F('enumerate')}(vit):       ${C(`${v.depth} 层`)}`,
    `        h = ${F('layer_norm')}(v)`,
    `        v = v + ${F('attn')}(h, ${F('rope_2d')})     ${C(`${v.heads} 头，图块两两互看`)}`,
    `        v = v + ${F('mlp')}(${F('layer_norm')}(v))     ${C(`${v.hidden}→${v.ffn}→${v.hidden}`)}`,
    `        ${K('if')} i ${K('in')} (${v.deepstack.join(', ')}): deep += [${F('merger_i')}(v)]`,
    `    img = ${F('merger')}(v)                  ${C(`2×2 → ${v.hidden * 4} → ${v.out}`)}`,
    `    ids = ${F('template')}(system, img, question)`,
    `    pos = ${F('mrope_ids')}(ids)               ${C('每个词元 (t, h, w)')}`,
    `    ${K('while')} True:`,
    `        x = ${F('embed')}(ids); x[图片] = img`,
    `        ${K('for')} L, layer ${K('in')} ${F('enumerate')}(llm):   ${C(`${t.layers} 层`)}`,
    `            h = ${F('rms_norm')}(x)`,
    `            x = x + ${F('attn')}(h, ${F('mrope')}(pos))  ${C(`${t.heads} Q / ${t.kvHeads} KV 头`)}`,
    `            x = x + ${F('swiglu')}(${F('rms_norm')}(x))  ${C(`${t.hidden}→${t.ffn}`)}`,
    `            ${K('if')} L < ${N(3)}: x[图片] += deep[L]   ${C('DeepStack')}`,
    `        logits = ${F('rms_norm')}(x[-${N(1)}]) @ embed.T`,
    `        nxt = ${F('argmax')}(logits)            ${C('贪心解码')}`,
    `        ${K('if')} nxt == <span class="nu">"&lt;|im_end|&gt;"</span>: ${K('break')}`,
    `        ids.${F('append')}(nxt)`,
  ];
  el.innerHTML = CODE.map((l, i) => `<span class="ln" data-n="${i + 1}"><span class="bp"></span><span class="no">${i + 1}</span>${l}</span>`).join('');
}

export function linesFor(s, Q) {
  const last = s.g === Q.G - 1 && Q.ended;
  const deep = Q.manifest.model.vision.deepstack;
  switch (s.ph) {
    case 'see': return [2, 3, 4, 5, 6, 7, 12, 13, 14];
    case 'pass': return [16, 17, 22, 23];
    case 'prep': return s.sub ? [{ resize: 2, patch: 3, norm: 4 }[s.sub]] : [2, 3, 4];
    case 'vit':
      if (s.sub === 'embed') return [5];
      if (s.sub === 'pos') return [6];
      if (s.L === undefined) return [5, 6, 7, 8, 9, 10, 11];
      if (!s.op) return deep.includes(s.L) ? [7, 11] : [7];
      return { ln1: [8], attn: [9], add1: [9], ln2: [10], mlp: [10], add2: deep.includes(s.L) ? [10, 11] : [10] }[s.op];
    case 'merge': return !s.sub ? [11, 12] : s.sub === 'deep' ? [11] : [12];
    case 'splice': return !s.sub ? [13, 14] : s.sub === 'template' ? [13] : [14];
    case 'read': return [25];
    case 'llm': return [16, 17, 18, 19, 20, 21];
    case 'layer':
      if (!s.op) return [17];
      return { ln1: [18], attn: [19], add1: [19], ln2: [20], mlp: [20], add2: [20], deep: [21] }[s.op];
    case 'head': return (!s.sub ? [22, 23] : s.sub === 'logits' ? [22] : [23]).concat(last ? [24] : []);
  }
  return [];
}

const STAGE = { see: '看图', pass: '想一个词', prep: '预处理', vit: '视觉编码器', merge: '合并器', splice: '拼进对话', read: '读入', llm: '语言模型', head: '输出' };
const SUBL = { resize: '缩放', patch: '切成图块', norm: '归一化', embed: '图块嵌入', pos: '位置嵌入', group: '2×2 拼接', mlp: 'MLP 投影', deep: 'DeepStack', template: '聊天模板', mrope: 'M-RoPE 位置', logits: '打分', pick: '选词' };
const OPL = { ln1: '归一化', attn: '注意力', add1: '残差 +', ln2: '归一化', mlp: '前馈', add2: '残差 +', deep: '+ DeepStack' };
const MIL = { conv: '一组卷积核', pick: '选一个图块', mul: '逐项相乘', sum: '求和', heads: '16 个头' };

export function stepLabel(s) {
  if (s.ph === 'vit' && s.L !== undefined) return `ViT 第 ${s.L} 层${s.op ? ' · ' + OPL[s.op] : ''}`;
  if (s.ph === 'layer') return `LLM 第 ${s.L} 层${s.op ? ' · ' + OPL[s.op] : ''}${s.mi ? ' · ' + MIL[s.mi] : ''}`;
  if (s.mi) return `${SUBL[s.sub]} · ${MIL[s.mi]}`;
  if (s.sub) return `${STAGE[s.ph]} · ${SUBL[s.sub]}`;
  return STAGE[s.ph] || s.ph;
}

export function crumbs(depth, s) {
  const out = [{ d: 0, label: DEPTH_NAMES[0] }, { d: 1, label: `第 ${s.g + 1} 个词元` }];
  if (depth >= 2) out.push({ d: 2, label: STAGE[s.ph === 'layer' ? 'llm' : s.ph] });
  if (depth >= 3) {
    if (s.ph === 'vit' && s.L !== undefined) out.push({ d: 3, label: `第 ${s.L} 层` });
    else if (s.ph === 'layer') out.push({ d: 3, label: `第 ${s.L} 层` });
    else if (s.sub) out.push({ d: 3, label: SUBL[s.sub] });
  }
  if (depth >= 4 && s.op) out.push({ d: 4, label: OPL[s.op] });
  if (depth >= 4 && s.mi === 'conv') out.push({ d: 4, label: MIL.conv });
  if (depth >= 5 && s.mi && s.mi !== 'conv') out.push({ d: 5, label: MIL[s.mi] });
  return out;
}

const T = (Q, i) => (Q.tokens[i] ? tokHTML(Q.tokens[i].s, `r-${Q.tokens[i].role}`) : '');
const P = (p) => `<em>${fmtPct(p)}</em>`;
const f = (v, d = 3) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
const cell = (V, m) => `(${Math.floor(m / V.mw)}, ${m % V.mw})`;

// 最受关注的视觉词元
export function argmax(a) { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }

// 图片词元在第 L 层“读起来像”的词：取最常见的几个（只算概率 > 5% 的）
export function lensWords(V, L, top = 4) {
  const cnt = new Map();
  for (let m = 0; m < V.Nv; m++) {
    const { s, p } = V.ilens(L + 1, m);
    const w = s.trim();
    if (p < 0.05 || !w || /^[\p{P}\p{S}\d\s]+$/u.test(w) || /<\|/.test(w)) continue;
    cnt.set(w, (cnt.get(w) || 0) + 1);
  }
  return [...cnt.entries()].sort((a, b) => b[1] - a[1]).slice(0, top);
}

export function explain(s, Q, ctx = {}) {
  const V = Q.V, M = Q.manifest.model, v = M.vision, t = M.text;
  const g = s.g, st = Q.steps[g], i = Q.row(g);
  const nxt = tokHTML(st.chosenS, 'r-assistant');
  const im = V.spec;
  const tokPerImg = `${V.mh}×${V.mw} = <b>${V.Nv}</b>`;
  switch (s.ph) {
    case 'see':
      return `图片先进黑箱：被切成 ${V.Np} 个小块、穿过视觉编码器，变成 <b>${V.Nv}</b> 个“视觉词元”，和你的问题一起排成一串。<br>这件事只做一次；之后每生成一个字，图片都已经在缓存里了。`;
    case 'pass':
      return g === 0
        ? `第 <b>1</b> 个词元：语言模型读入 <b>${Q.P}</b> 个词元（其中 ${V.Nv} 个是图片），把 ${t.layers} 层完整算一遍，吐出 ${nxt}。`
        : `第 <b>${g + 1}</b> 个词元：前面 ${Q.P + g} 个词元的 K、V 都在缓存里，只需要为新位置算一遍 ${t.layers} 层，吐出 ${nxt}。`;
    case 'prep': {
      if (!s.sub) return `<b>预处理</b>：原图 ${im.orig[0]}×${im.orig[1]} 缩放到 <b>${V.gw * 16}×${V.gh * 16}</b>（边长是 32 的倍数），切成 ${V.gh}×${V.gw} 个 16×16 的图块，像素值换算到 [-1, 1]。`;
      if (s.sub === 'resize') return `处理器把边长就近取整到 <b>32 的倍数</b>（16 像素的图块 × 2×2 合并），同时让总像素不超过上限 ${Q.manifest.pixels.max.toLocaleString('zh-CN')}。<br>原图 ${im.orig[0]}×${im.orig[1]} → <b>${V.gw * 16}×${V.gh * 16}</b>。图越大，词元越多，算得越慢。`;
      if (s.sub === 'patch') return `切成 <b>${V.gh} 行 × ${V.gw} 列 = ${V.Np}</b> 个 16×16 的图块。每块有 3 个颜色通道 × 256 个像素 = 768 个数。<br>它们会以 2×2 为一组，最后合并成 ${tokPerImg} 个词元。`;
      return `每个像素值 p 换算成 (p/255 − 0.5) / 0.5，落在 [-1, 1]。图块再<b>复制成 2 帧</b>：Qwen3-VL 的视觉编码器同时处理视频，时间方向的块大小是 2，单张图就当成两帧一模一样的视频。每块于是有 3 × 2 × 16 × 16 = <b>${v.inDim}</b> 个输入。`;
    }
    case 'vit': return vitExplain(s, Q, ctx);
    case 'merge': {
      const ds = v.deepstack;
      if (!s.sub) return `<b>合并器</b>：相邻 2×2 个图块的特征拼在一起（4 × ${v.hidden} = ${v.hidden * 4} 维），过一个两层 MLP，投到语言模型的 <b>${v.out}</b> 维。${V.Np} 个图块 → ${tokPerImg} 个视觉词元。`;
      if (s.sub === 'group') return `每 2×2 个相邻图块拼成一个 <b>${v.hidden * 4}</b> 维的向量。词元数变成四分之一：${V.Np} → <b>${V.Nv}</b>。语言模型那边的计算量和词元数的平方成正比，这一步省了很多。`;
      if (s.sub === 'mlp') return `LayerNorm → ${v.hidden * 4}×${v.hidden * 4} 全连接 → GELU → ${v.hidden * 4}×${v.out} 全连接。输出的 ${V.Nv} 个 ${v.out} 维向量，就是语言模型眼里的“图片”。监视器上的颜色是它们的主成分。`;
      let mean = 0;
      for (let m = 0; m < V.Nv; m++) mean += V.dsNorm(0, m) / V.Nv;
      return `<b>DeepStack</b>：视觉编码器第 ${ds.join('、')} 层的中间特征也各走一个合并器，分别<b>加到语言模型前 3 层</b>输出的图片位置上。浅层看纹理、深层看语义，一起交给语言模型。第 ${ds[0]} 层这一路的向量平均长度 ${mean.toFixed(1)}。`;
    }
    case 'splice': {
      const p0 = Q.pos(V.vs), pl = Q.pos(V.vs + V.Nv - 1), pa = Q.pos(V.vs + V.Nv + 1);
      if (!s.sub) return `视觉词元按聊天模板插在 <b>&lt;|vision_start|&gt;</b> 和 <b>&lt;|vision_end|&gt;</b> 之间，后面跟着你的问题。每个词元再配上三维位置 (t, h, w)。`;
      if (s.sub === 'template') return `聊天模板：系统提示、<b>&lt;|vision_start|&gt;</b>、${V.Nv} 个 &lt;|image_pad|&gt; 占位符、<b>&lt;|vision_end|&gt;</b>、你的问题，一共 <b>${Q.P}</b> 个词元。占位符的嵌入被合并器的输出直接替换掉。图片排在问题<b>前面</b>：因果注意力下，图片词元看不到问题，同一张图不管问什么，它们在语言模型里都完全一样。`;
      return `<b>M-RoPE</b>：文字的位置是 (n, n, n)；图片词元是 (${p0[0]}, ${p0[1]}+行, ${p0[2]}+列)，从 ${fmtPos(p0)} 到 ${fmtPos(pl)}。${V.Nv} 个视觉词元只占了 <b>${pa[0] - p0[0] - 1}</b> 个位置，问题从 ${fmtPos(pa)} 接着往下数。64 对旋转频率按 t、h、w 交错分配（${t.mrope.join(' / ')}）。`;
    }
    case 'read': {
      const pp = Q.pos(i);
      return `上一步选出的 ${T(Q, i)} 被接到序列末尾，位置 ${fmtPos(pp)}。<b>自回归</b>：输出立刻变成下一步的输入。图片的 ${V.Nv} 个词元早就在 KV 缓存里，不用再看一遍原图。`;
    }
    case 'llm': {
      const [a, b] = Q.manifest.groundLayers;
      return `语言模型的 <b>${t.layers}</b> 层和纯文字模型一模一样（Qwen3 结构：RMSNorm、${t.heads} Q / ${t.kvHeads} KV 头注意力、SwiGLU）。图片只是一串特殊的词元。生成 ${nxt} 时，第 ${a}–${b} 层的注意力把 ${fmtPct(avgMass(Q, g, a, b))} 分给了图片。`;
    }
    case 'layer': return layerExplain(s, Q, ctx);
    case 'head': {
      const top = st.top.slice(0, 3).map(([, p, tt]) => `${tokHTML(tt, 'r-assistant')} ${P(p)}`).join('，');
      if (!s.sub) return `最后一个位置的向量过 RMSNorm，乘上和嵌入表共用的输出矩阵（${t.hidden} × ${t.vocab.toLocaleString('zh-CN')}），softmax 成概率：${top}……取最高的那个。`;
      if (s.sub === 'logits') return `给词表里每个词元打分（${t.vocab.toLocaleString('zh-CN')} 个），softmax 成概率。前几名：${top}。`;
      const end = st.chosenS === '<|im_end|>' ? ' 它选中了结束标记，这条回复到此为止。' : '';
      return `<b>贪心解码</b>：直接取概率最高的 ${nxt}（${P(st.p)}）。没有随机数，同样的图和问题，永远得到同样的回答。${end}`;
    }
  }
  return '';
}

const fmtPos = (p) => `(${p[0]}, ${p[1]}, ${p[2]})`;

export function avgMass(Q, g, a, b) {
  let m = 0;
  for (let L = a; L <= b; L++) m += Q.mass(g, L) / (b - a + 1);
  return m;
}

function vitExplain(s, Q, ctx) {
  const V = Q.V, v = Q.manifest.model.vision, mic = V.micro;
  if (s.sub === 'embed') {
    if (!s.mi) return `<b>图块嵌入</b>：每个图块的 ${v.inDim} 个数和一个 ${v.inDim} × ${v.hidden} 的矩阵相乘（实现上是一个步长等于核大小的 3D 卷积），变成 ${v.hidden} 维向量。${V.Np} 个图块各自独立地做这件事。`;
    if (s.mi === 'conv') return `${v.hidden} 个输出通道 = ${v.hidden} 个卷积核，每个核也是 3 × 2 × 16 × 16。监视器上是对这个图块反应最强的 16 个核（两帧相加，红绿蓝按真实权重着色）：有的找边缘，有的找某种颜色。`;
    const where = `第 ${mic.row} 行第 ${mic.col} 列的图块`;
    if (s.mi === 'pick') return `放大${where}（它属于这张图在所有问答里累计最受关注的视觉词元 ${cell(V, mic.token)}），看它嵌入向量的<b>第 ${mic.ch} 维</b>是怎么算出来的：${v.inDim} 个像素值，各乘第 ${mic.ch} 个卷积核里对应的权重，再加起来。`;
    if (s.mi === 'mul') return `逐项相乘。贡献最大的 12 项列在下面（两帧像素相同，权重不同，所以同一个像素会出现两次）。橙色为正，蓝色为负；每一项都很小，${v.inDim} 项的绝对值加起来是 ${mic.absSum.toFixed(2)}。`;
    return `${v.inDim} 项全部加起来，再加偏置 b = ${mic.bias.toFixed(4)}：<b>${mic.conv.toFixed(4)}</b>（自己用 float32 重算是 ${mic.mine.toFixed(4)}，差别来自 bf16 卷积的舍入）。接着加上位置嵌入 ${f(mic.pos, 4)}，进入第 0 层时这个数是 <b>${mic.after.toFixed(4)}</b>。`;
  }
  if (s.sub === 'pos') return `<b>位置嵌入</b>：模型里存着一张 ${v.posGrid}×${v.posGrid} 的位置表，按双线性插值缩放到这张图的 ${V.gh}×${V.gw}，加到每个图块上。之后每层注意力里还有<b>二维 RoPE</b>：把 q、k 按图块的行号和列号各旋转一部分维度。`;
  if (s.L === undefined) return `<b>视觉编码器（ViT）</b>：${v.depth} 层 Transformer，${v.hidden} 维，${v.heads} 个头。和语言模型不同，这里<b>没有因果遮罩</b>，${V.Np} 个图块两两互看。监视器上的颜色是每层特征的前三个主成分：越往上，同一个物体的图块颜色越一致。`;
  const L = s.L, dist = V.vitDist[L], dmean = dist.reduce((a, b) => a + b, 0) / dist.length;
  const [xn, an, mn] = V.vitStats[L];
  const has = V.vitAttnLayers.includes(L);
  const q = ctx.vq ?? mic.token;
  const ds = v.deepstack.includes(L) ? ` 这一层的输出还会被 DeepStack 抽走一份，直接送进语言模型。` : '';
  if (!s.op) return `ViT 第 <b>${L}</b> 层：16 个头平均每次看 <b>${dmean.toFixed(1)}</b> 个图块远的地方（平均注意距离）。${has ? `监视器上是从词元 ${cell(V, q)} 出发的注意力（点监视器换一个）。` : '监视器上是这一层输出特征的主成分颜色。'}${ds}`;
  if (s.op === 'ln1' || s.op === 'ln2') return `<b>LayerNorm</b>：减均值、除标准差，再乘缩放加偏置（视觉编码器用的是 LayerNorm，语言模型用的是 RMSNorm）。进来的向量平均长度 ${xn.toFixed(1)}。`;
  if (s.op === 'attn') {
    if (!has) return `<b>自注意力</b>：${v.heads} 个头，每个 ${v.hidden / v.heads} 维。q、k 带着二维旋转位置编码，所以分得清上下左右。各头平均注意距离从 ${Math.min(...dist).toFixed(1)} 到 ${Math.max(...dist).toFixed(1)} 个图块（监视器）。`;
    const a = V.vattn(L, q), self = V.vattnSelf(L, q);
    const top = argmax(a);
    return `从词元 ${cell(V, q)} 出发的注意力（16 头平均，按 2×2 合并显示）：看自己 ${P(self)}，看得最多的是 ${cell(V, top)}（${P(a[top])}）。点监视器上任何一格，换一个查询位置。`;
  }
  if (s.op === 'add1') return `注意力的输出加回残差流（平均长度 ${an.toFixed(1)}）。`;
  if (s.op === 'mlp') return `<b>前馈网络</b>：${v.hidden} → ${v.ffn} → ${v.hidden}，激活函数 GELU（tanh 近似）。每个图块各自加工，不和别的图块交流。`;
  return `前馈的输出加回残差流（平均长度 ${mn.toFixed(1)}），第 ${L} 层结束。${ds}`;
}

function layerExplain(s, Q, ctx) {
  const V = Q.V, M = Q.manifest.model, t = M.text;
  const { L, g } = s, i = Q.row(g);
  const lens = Q.lensAt(g, L);
  const guess = `${tokHTML(lens[0][0], 'r-assistant')} ${P(lens[0][1])}`;
  const mass = Q.mass(g, L);
  const a = Q.attImg(g, L), top = argmax(a);
  const words = lensWords(V, L).map(([w]) => `「${esc(w)}」`).join('');
  const [ga, gb] = Q.manifest.groundLayers;
  const grounded = L >= ga && L <= gb;
  if (!s.op) return `第 <b>${L}</b> 层：${T(Q, i)} 把 <b>${fmtPct(mass)}</b> 的注意力分给图片，最多的落在 ${cell(V, top)}。${grounded ? '这一层属于“对准”的那几层：注意力集中在和当前这个字有关的物体上。' : L < ga ? '这么浅的层，看图基本还是“扫一眼”，不太对得准物体。' : ''}逻辑透镜：如果在这一层停下，它会说 ${guess}。${words ? `<br>图片词元在这一层读起来像：${words}` : ''}`;
  if (s.op === 'ln1') return `<b>RMSNorm</b>：向量除以自己的均方根，再乘一组可学习的系数。`;
  if (s.op === 'ln2') return `再做一次 <b>RMSNorm</b>，准备进入前馈网络。`;
  if (s.op === 'attn') {
    if (s.mi === 'heads') {
      const hs = [...Array(t.heads).keys()].map((h) => ({ h, m: Q.headMass(g, L, h) })).sort((x, y) => y.m - x.m);
      return `第 ${L} 层的 <b>${t.heads}</b> 个头各看各的：有的头几乎只看文字，有的头把大部分注意力都给了图片。看图最多的是第 ${hs[0].h} 头（${P(hs[0].m)}）和第 ${hs[1].h} 头（${P(hs[1].m)}）。监视器上是每个头在图片上的注意力，单独归一化。`;
    }
    const txt = Q.txt(g, L).slice(0, 2).map((x) => `${T(Q, x.j)} ${P(x.w)}`).join('、');
    return `<b>注意力</b>：${T(Q, i)} 的 q 和前面每个词元的 k 做点积（位置用 M-RoPE 旋转，图片词元按 (t, h, w) 转）。图片拿到 <b>${fmtPct(mass)}</b>；文字里看得最多的是 ${txt}。${M.text.heads} 个查询头共用 ${t.kvHeads} 组键值。`;
  }
  const [da, dm] = Q.delta(g, L);
  if (s.op === 'add1') return `注意力的输出（长度 ${da.toFixed(1)}）加回残差流。从图片读到的信息就是这样“搬”进当前这个位置的。`;
  if (s.op === 'mlp') return `<b>SwiGLU 前馈</b>：${t.hidden} → ${t.ffn} → ${t.hidden}，在当前位置自己加工。往残差流里加了长度 ${dm.toFixed(1)} 的向量。`;
  if (s.op === 'add2') return `前馈的输出加回残差流，第 ${L} 层结束。逻辑透镜此刻读到的是 ${guess}。`;
  // deep
  let mean = 0, base = 0;
  for (let m = 0; m < V.Nv; m++) { mean += V.dsNorm(L, m) / V.Nv; base += V.vnorm(L + 1, m) / V.Nv; }
  return `<b>DeepStack</b>：把视觉编码器第 ${M.vision.deepstack[L]} 层的特征（经过自己的合并器，平均长度 ${mean.toFixed(1)}）<b>加到图片位置</b>上，加完后图片位置的残差流平均长度 ${base.toFixed(1)}。只动图片位置，文字位置不变。这一步只在预填充时做一次。`;
}

export function shapeOf(s, Q) {
  const V = Q.V, M = Q.manifest.model, v = M.vision, t = M.text;
  const b = (x) => `<b>${x}</b>`;
  switch (s.ph) {
    case 'prep':
      if (s.sub === 'resize') return `image ${Q.V.spec.orig.join('×')} → ${b(`${V.gw * 16}×${V.gh * 16}`)}`;
      if (s.sub === 'patch') return `${V.gh * 16}×${V.gw * 16}×3 → ${b(`${V.Np} × 16×16×3`)}`;
      if (s.sub === 'norm') return `x ${b(`[${V.Np} × ${v.inDim}]`)} = 3 通道 × 2 帧 × 16 × 16`;
      return null;
    case 'vit':
      if (s.sub === 'embed') return `x [${V.Np}×${v.inDim}] @ W ${b(`[${v.inDim}×${v.hidden}]`)} + b → v [${V.Np}×${v.hidden}]`;
      if (s.sub === 'pos') return `pos [${v.posGrid}×${v.posGrid}×${v.hidden}] → 插值 ${b(`[${V.gh}×${V.gw}×${v.hidden}]`)}`;
      if (s.op === 'attn') return `qkv [${V.Np}×${v.hidden * 3}] → ${v.heads} 头 × ${v.hidden / v.heads} · 分数 ${b(`[${v.heads}×${V.Np}×${V.Np}]`)}`;
      if (s.op === 'mlp') return `[${V.Np}×${v.hidden}] → ${b(`[${V.Np}×${v.ffn}]`)} → [${V.Np}×${v.hidden}]`;
      if (s.L !== undefined) return `v ${b(`[${V.Np}×${v.hidden}]`)}`;
      return null;
    case 'merge':
      if (s.sub === 'group') return `[${V.Np}×${v.hidden}] → ${b(`[${V.Nv}×${v.hidden * 4}]`)}`;
      if (s.sub === 'mlp') return `[${V.Nv}×${v.hidden * 4}] → [${V.Nv}×${v.hidden * 4}] → ${b(`[${V.Nv}×${v.out}]`)}`;
      if (s.sub === 'deep') return `3 × ${b(`[${V.Nv}×${v.out}]`)} → LLM 第 0–2 层`;
      return null;
    case 'splice':
      if (s.sub === 'template') return `ids ${b(`[${Q.P}]`)} = 文字 ${Q.P - V.Nv} + 图片 ${V.Nv}`;
      if (s.sub === 'mrope') return `pos ${b(`[3×${Q.P}]`)} (t, h, w)`;
      return null;
    case 'layer':
      if (s.op === 'attn') return `q [${t.heads}×${t.headDim}] · K 缓存 ${b(`[${t.kvHeads}×${Q.row(s.g) + 1}×${t.headDim}]`)}`;
      if (s.op === 'mlp') return `h [1×${t.hidden}] → ${b(`[1×${t.ffn}]`)} → [1×${t.hidden}]`;
      if (s.op === 'deep') return `x[图片] ${b(`[${V.Nv}×${t.hidden}]`)} += deep[${s.L}]`;
      if (!s.op) return `x ${b(`[${Q.row(s.g) + 1}×${t.hidden}]`)}`;
      return null;
    case 'head':
      if (s.sub === 'logits') return `h [1×${t.hidden}] @ Eᵀ ${b(`[${t.hidden}×${t.vocab}]`)}`;
      return null;
  }
  return null;
}

export function watch(s, Q, ctx) {
  const V = Q.V, M = Q.manifest.model, v = M.vision, t = M.text, g = s.g, st = Q.steps[g], i = Q.row(g);
  const rows = [];
  const add = (k, val) => rows.push([k, val]);
  const head = (k) => rows.push([null, k]);
  head('VARIABLES');
  if (s.ph === 'see' || s.ph === 'prep') {
    add('image', `${V.spec.orig[0]}×${V.spec.orig[1]}`);
    add('px', `${V.gw * 16}×${V.gh * 16}`);
    add('grid (h, w)', `${V.gh} × ${V.gw}`);
    add('patches', V.Np);
    add('tokens', `${V.mh} × ${V.mw} = ${V.Nv}`);
    if (s.sub === 'norm') { const p = V.pix; add('px[0,0] RGB', `${p[0]}, ${p[1]}, ${p[2]}`); add('→ x', `${f(V.px[0], 3)}, ${f(V.px[256], 3)}, ${f(V.px[512], 3)}`); }
  } else if (s.ph === 'vit') {
    const mic = V.micro;
    if (s.mi && s.mi !== 'conv') {
      add('patch (r, c)', `(${mic.row}, ${mic.col})`);
      add('channel', mic.ch);
      if (s.mi !== 'pick') { const e = mic.top[0]; add('x × w', `${f(e.px)} × ${e.w.toFixed(5)}`); add('= ', f(e.prod, 5)); }
      if (s.mi === 'sum') { add('Σ + b', mic.conv.toFixed(4)); add('+ pos', f(mic.pos, 4)); add('v[ch]', mic.after.toFixed(4)); }
    } else if (s.sub) {
      add('W_patch', `${v.hidden} × ${v.inDim}`);
      add('pos_embed', `${v.posGrid}² × ${v.hidden}`);
      add('v', `${V.Np} × ${v.hidden}`);
    } else if (s.L !== undefined) {
      const [xn, an, mn] = V.vitStats[s.L];
      const dist = V.vitDist[s.L];
      add('layer', `${s.L} / ${v.depth - 1}`);
      add('‖v‖ 平均', xn.toFixed(1));
      add('‖attn‖ / ‖mlp‖', `${an.toFixed(1)} / ${mn.toFixed(1)}`);
      add('注意距离 头均', (dist.reduce((a, b) => a + b, 0) / dist.length).toFixed(2));
      const pv = V.pcaVar[s.L + 1];
      add('PCA 方差', pv.map((x) => (x * 100).toFixed(0) + '%').join(' '));
    } else { add('blocks', v.depth); add('dim', v.hidden); add('heads', v.heads); add('patches', V.Np); }
  } else if (s.ph === 'merge') {
    add('in', `${V.Np} × ${v.hidden}`);
    add('group', `${V.Nv} × ${v.hidden * 4}`);
    add('out', `${V.Nv} × ${v.out}`);
    let mn = 0;
    for (let m = 0; m < V.Nv; m++) mn += V.vnorm(0, m) / V.Nv;
    add('‖img‖ 平均', mn.toFixed(1));
    if (s.sub === 'deep') for (let d = 0; d < 3; d++) { let x = 0; for (let m = 0; m < V.Nv; m++) x += V.dsNorm(d, m) / V.Nv; add(`‖deep[${d}]‖`, x.toFixed(1)); }
  } else if (s.ph === 'splice') {
    add('len(ids)', Q.P);
    add('vision_start', `#${V.vs - 1}`);
    add('image_pad', `#${V.vs} … #${V.vs + V.Nv - 1}`);
    add('pos 首个图片', fmtPos(Q.pos(V.vs)));
    add('pos 末个图片', fmtPos(Q.pos(V.vs + V.Nv - 1)));
    add('pos 问题开头', fmtPos(Q.pos(V.vs + V.Nv + 1)));
  } else if (s.ph === 'layer' || s.ph === 'llm' || s.ph === 'pass' || s.ph === 'read') {
    const L = s.L ?? null;
    add('词元', `#${i}`);
    add('pos (t,h,w)', fmtPos(Q.pos(i)));
    if (L != null) {
      const a = Q.attImg(g, L), m = argmax(a);
      add('layer', `${L} / ${t.layers - 1}`);
      add('看图比例', fmtPct(Q.mass(g, L)));
      add('最受关注', `${cell(V, m)} ${fmtPct(a[m])}`);
      const lens = Q.lensAt(g, L);
      add('lens', `${lens[0][0].replace(/\n/g, '↵')} ${fmtPct(lens[0][1])}`);
      const [da, dm] = Q.delta(g, L);
      add('‖Δattn‖ / ‖Δmlp‖', `${da.toFixed(1)} / ${dm.toFixed(1)}`);
      if (s.mi === 'heads') {
        const hs = [...Array(t.heads).keys()].map((h) => ({ h, m: Q.headMass(g, L, h) })).sort((x, y) => y.m - x.m);
        hs.slice(0, 4).forEach((x) => add(`头 ${x.h} 看图`, fmtPct(x.m)));
      }
    } else {
      const [ga, gb] = Q.manifest.groundLayers;
      add(`看图比例 L${ga}–${gb}`, fmtPct(avgMass(Q, g, ga, gb)));
      add('nxt', st.chosenS.replace(/\n/g, '↵'));
    }
  } else if (s.ph === 'head') {
    st.top.slice(0, 5).forEach(([, p, tt]) => add(tt.replace(/\n/g, '↵') || '␣', fmtPct(p)));
  }
  return rows;
}

export function renderWatch(el, rows) {
  el.innerHTML = rows.map(([k, v]) => (k == null ? `<div class="wh">${esc(v)}</div>` : `<div class="k">${esc(k)}</div><div class="v">${esc(String(v))}</div>`)).join('');
}
