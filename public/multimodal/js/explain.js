// 调试器里的“代码 / 这一步 / 变量”三块内容，全部用真实导出的数值填充。
import { esc, tokHTML, fmtPct } from '../../js/ui.js';
import { DEPTH_NAMES } from './timeline.js';
import { isEn, L } from '../../js/i18n.js';

const K = (s) => `<span class="kw">${s}</span>`;
const F = (s) => `<span class="fn">${s}</span>`;
const C = (s) => `<span class="cm"># ${s}</span>`;
const N = (s) => `<span class="nu">${s}</span>`;

export function renderCode(el, M, V) {
  const v = M.model.vision, t = M.model.text;
  const g = V ? `${V.gh}×${V.gw}` : 'h×w';
  const CODE = [
    `${K('def')} ${F('chat')}(image, question):`,
    isEn ? `    px = ${F('resize')}(image, ${K('multiple_of')}=${N(32)})  ${C(V ? `${V.gw * 16}×${V.gh * 16}` : 'resize')}`
      : `    px = ${F('resize')}(image, ${N(32)} ${K('的倍数')})       ${C(V ? `${V.gw * 16}×${V.gh * 16}` : '缩放')}`,
    `    x = ${F('patchify')}(px, ${N(16)})            ${C(V ? L(`${g} = ${V.Np} 块`, `${g} = ${V.Np} patches`) : L('切块', 'patchify'))}`,
    `    x = (x - ${N('0.5')}) / ${N('0.5')}               ${C(L('像素 → [-1, 1]，复制成 2 帧', 'pixels → [-1, 1], copied into 2 frames'))}`,
    `    v = x @ W_patch + b               ${C(`${v.inDim} → ${v.hidden}`)}`,
    `    v = v + ${F('interp')}(pos_embed, ${g})   ${C(L(`${v.posGrid}×${v.posGrid} 位置表`, `${v.posGrid}×${v.posGrid} position table`))}`,
    `    ${K('for')} i, blk ${K('in')} ${F('enumerate')}(vit):       ${C(L(`${v.depth} 层`, `${v.depth} layers`))}`,
    `        h = ${F('layer_norm')}(v)`,
    `        v = v + ${F('attn')}(h, ${F('rope_2d')})     ${C(L(`${v.heads} 头，图块两两互看`, `${v.heads} heads, every patch sees every patch`))}`,
    `        v = v + ${F('mlp')}(${F('layer_norm')}(v))     ${C(`${v.hidden}→${v.ffn}→${v.hidden}`)}`,
    `        ${K('if')} i ${K('in')} (${v.deepstack.join(', ')}): deep += [${F('merger_i')}(v)]`,
    `    img = ${F('merger')}(v)                  ${C(`2×2 → ${v.hidden * 4} → ${v.out}`)}`,
    `    ids = ${F('template')}(system, img, question)`,
    `    pos = ${F('mrope_ids')}(ids)               ${C(L('每个词元 (t, h, w)', '(t, h, w) per token'))}`,
    `    ${K('while')} True:`,
    L(`        x = ${F('embed')}(ids); x[图片] = img`, `        x = ${F('embed')}(ids); x[image] = img`),
    `        ${K('for')} L, layer ${K('in')} ${F('enumerate')}(llm):   ${C(L(`${t.layers} 层`, `${t.layers} layers`))}`,
    `            h = ${F('rms_norm')}(x)`,
    `            x = x + ${F('attn')}(h, ${F('mrope')}(pos))  ${C(L(`${t.heads} Q / ${t.kvHeads} KV 头`, `${t.heads} Q / ${t.kvHeads} KV heads`))}`,
    `            x = x + ${F('swiglu')}(${F('rms_norm')}(x))  ${C(`${t.hidden}→${t.ffn}`)}`,
    L(`            ${K('if')} L < ${N(3)}: x[图片] += deep[L]   ${C('DeepStack')}`, `            ${K('if')} L < ${N(3)}: x[image] += deep[L]  ${C('DeepStack')}`),
    `        logits = ${F('rms_norm')}(x[-${N(1)}]) @ embed.T`,
    `        nxt = ${F('argmax')}(logits)            ${C(L('贪心解码', 'greedy decoding'))}`,
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

const STAGE = L({ see: '看图', pass: '想一个词', prep: '预处理', vit: '视觉编码器', merge: '合并器', splice: '拼进对话', read: '读入', llm: '语言模型', head: '输出' },
  { see: 'Look at the image', pass: 'Think of a token', prep: 'Preprocessing', vit: 'Vision encoder', merge: 'Merger', splice: 'Into the chat', read: 'Read in', llm: 'Language model', head: 'Output' });
const SUBL = L({ resize: '缩放', patch: '切成图块', norm: '归一化', embed: '图块嵌入', pos: '位置嵌入', group: '2×2 拼接', mlp: 'MLP 投影', deep: 'DeepStack', template: '聊天模板', mrope: 'M-RoPE 位置', logits: '打分', pick: '选词' },
  { resize: 'Resize', patch: 'Cut into patches', norm: 'Normalize', embed: 'Patch embedding', pos: 'Position embedding', group: '2×2 concat', mlp: 'MLP projection', deep: 'DeepStack', template: 'Chat template', mrope: 'M-RoPE positions', logits: 'Scores', pick: 'Pick a token' });
const OPL = L({ ln1: '归一化', attn: '注意力', add1: '残差 +', ln2: '归一化', mlp: '前馈', add2: '残差 +', deep: '+ DeepStack' },
  { ln1: 'Norm', attn: 'Attention', add1: 'Residual +', ln2: 'Norm', mlp: 'Feed-forward', add2: 'Residual +', deep: '+ DeepStack' });
const MIL = L({ conv: '一组卷积核', pick: '选一个图块', mul: '逐项相乘', sum: '求和', heads: '16 个头' },
  { conv: 'A bank of kernels', pick: 'Pick one patch', mul: 'Multiply term by term', sum: 'Sum', heads: '16 heads' });

export function stepLabel(s) {
  if (isEn && s.ph === 'vit' && s.L !== undefined) return `ViT layer ${s.L}${s.op ? ' · ' + OPL[s.op] : ''}`;
  if (isEn && s.ph === 'layer') return `LLM layer ${s.L}${s.op ? ' · ' + OPL[s.op] : ''}${s.mi ? ' · ' + MIL[s.mi] : ''}`;
  if (s.ph === 'vit' && s.L !== undefined) return `ViT 第 ${s.L} 层${s.op ? ' · ' + OPL[s.op] : ''}`;
  if (s.ph === 'layer') return `LLM 第 ${s.L} 层${s.op ? ' · ' + OPL[s.op] : ''}${s.mi ? ' · ' + MIL[s.mi] : ''}`;
  if (s.mi) return `${SUBL[s.sub]} · ${MIL[s.mi]}`;
  if (s.sub) return `${STAGE[s.ph]} · ${SUBL[s.sub]}`;
  return STAGE[s.ph] || s.ph;
}

export function crumbs(depth, s) {
  const out = [{ d: 0, label: DEPTH_NAMES[0] }, { d: 1, label: L(`第 ${s.g + 1} 个词元`, `Token ${s.g + 1}`) }];
  if (depth >= 2) out.push({ d: 2, label: STAGE[s.ph === 'layer' ? 'llm' : s.ph] });
  if (depth >= 3) {
    if (s.ph === 'vit' && s.L !== undefined) out.push({ d: 3, label: L(`第 ${s.L} 层`, `Layer ${s.L}`) });
    else if (s.ph === 'layer') out.push({ d: 3, label: L(`第 ${s.L} 层`, `Layer ${s.L}`) });
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
    const { s, p } = V.ilens(L, m);
    const w = s.trim();
    if (p < 0.05 || !w || /^[\p{P}\p{S}\d\s]+$/u.test(w) || /<\|/.test(w)) continue;
    cnt.set(w, (cnt.get(w) || 0) + 1);
  }
  return [...cnt.entries()].sort((a, b) => b[1] - a[1]).slice(0, top);
}

export function explain(s, Q, ctx = {}) {
  if (isEn) return explainEn(s, Q, ctx);
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

// 英文讲解：和中文版一一对应，数字全部取自英文数据（data/en/）这次真实运行
function explainEn(s, Q, ctx) {
  const V = Q.V, M = Q.manifest.model, v = M.vision, t = M.text;
  const g = s.g, st = Q.steps[g], i = Q.row(g);
  const nxt = tokHTML(st.chosenS, 'r-assistant');
  const im = V.spec;
  const tokPerImg = `${V.mh}×${V.mw} = <b>${V.Nv}</b>`;
  switch (s.ph) {
    case 'see':
      return `The image goes into the black box: it is cut into ${V.Np} small patches and passed through the vision encoder, becoming <b>${V.Nv}</b> “vision tokens” that line up with your question in one sequence.<br>This happens only once; for every later token the image is already in the cache.`;
    case 'pass':
      return g === 0
        ? `Token <b>1</b>: the language model reads <b>${Q.P}</b> tokens (${V.Nv} of them are the image), runs all ${t.layers} layers over them, and outputs ${nxt}.`
        : `Token <b>${g + 1}</b>: the K and V of the previous ${Q.P + g} tokens are already in the cache, so the ${t.layers} layers only have to run for the new position. Out comes ${nxt}.`;
    case 'prep': {
      if (!s.sub) return `<b>Preprocessing</b>: the ${im.orig[0]}×${im.orig[1]} original is resized to <b>${V.gw * 16}×${V.gh * 16}</b> (sides are multiples of 32), cut into ${V.gh}×${V.gw} patches of 16×16, and pixel values are mapped to [-1, 1].`;
      if (s.sub === 'resize') return `The processor rounds each side to the nearest <b>multiple of 32</b> (16-pixel patches × 2×2 merging) while keeping the total pixel count under ${Q.manifest.pixels.max.toLocaleString('en-US')}.<br>${im.orig[0]}×${im.orig[1]} → <b>${V.gw * 16}×${V.gh * 16}</b>. A bigger image means more tokens and slower answers.`;
      if (s.sub === 'patch') return `Cut into <b>${V.gh} rows × ${V.gw} columns = ${V.Np}</b> patches of 16×16. Each patch has 3 color channels × 256 pixels = 768 numbers.<br>They will be grouped 2×2 and finally merged into ${tokPerImg} tokens.`;
      return `Each pixel value p becomes (p/255 − 0.5) / 0.5, landing in [-1, 1]. Each patch is then <b>copied into 2 frames</b>: Qwen3-VL’s vision encoder also handles video, with a temporal patch size of 2, so a single image is treated as a video of two identical frames. That gives each patch 3 × 2 × 16 × 16 = <b>${v.inDim}</b> inputs.`;
    }
    case 'vit': return vitExplainEn(s, Q, ctx);
    case 'merge': {
      const ds = v.deepstack;
      if (!s.sub) return `<b>Merger</b>: the features of each 2×2 block of neighboring patches are concatenated (4 × ${v.hidden} = ${v.hidden * 4} dims), passed through a two-layer MLP and projected to the language model’s <b>${v.out}</b> dims. ${V.Np} patches → ${tokPerImg} vision tokens.`;
      if (s.sub === 'group') return `Every 2×2 group of neighboring patches is concatenated into one <b>${v.hidden * 4}</b>-dim vector. The token count drops to a quarter: ${V.Np} → <b>${V.Nv}</b>. The language model’s cost grows with the square of the token count, so this step saves a lot.`;
      if (s.sub === 'mlp') return `LayerNorm → ${v.hidden * 4}×${v.hidden * 4} linear → GELU → ${v.hidden * 4}×${v.out} linear. The resulting ${V.Nv} vectors of ${v.out} dims are “the image” as the language model sees it. The monitor colors them by their principal components.`;
      let mean = 0;
      for (let m = 0; m < V.Nv; m++) mean += V.dsNorm(0, m) / V.Nv;
      return `<b>DeepStack</b>: intermediate features from vision-encoder layers ${ds.join(', ')} each go through their own merger and are <b>added to the image positions in the outputs of the language model’s first 3 layers</b>. Shallow layers see texture, deep layers see meaning; both are handed to the language model. The vectors in the layer-${ds[0]} path have an average length of ${mean.toFixed(1)}.`;
    }
    case 'splice': {
      const p0 = Q.pos(V.vs), pl = Q.pos(V.vs + V.Nv - 1), pa = Q.pos(V.vs + V.Nv + 1);
      if (!s.sub) return `Following the chat template, the vision tokens are inserted between <b>&lt;|vision_start|&gt;</b> and <b>&lt;|vision_end|&gt;</b>, followed by your question. Every token also gets a 3-D position (t, h, w).`;
      if (s.sub === 'template') return `The chat template: system prompt, <b>&lt;|vision_start|&gt;</b>, ${V.Nv} &lt;|image_pad|&gt; placeholders, <b>&lt;|vision_end|&gt;</b>, your question — <b>${Q.P}</b> tokens in all. The placeholders’ embeddings are simply replaced by the merger’s output. The image comes <b>before</b> the question: under causal attention the image tokens cannot see the question, so whatever you ask about the same image, they are exactly the same inside the language model.`;
      return `<b>M-RoPE</b>: a text token’s position is (n, n, n); an image token’s is (${p0[0]}, ${p0[1]}+row, ${p0[2]}+column), from ${fmtPos(p0)} to ${fmtPos(pl)}. The ${V.Nv} vision tokens take up only <b>${pa[0] - p0[0] - 1}</b> position numbers, and the question continues counting from ${fmtPos(pa)}. The 64 rotary frequency pairs are interleaved across t, h and w (${t.mrope.join(' / ')}).`;
    }
    case 'read': {
      const pp = Q.pos(i);
      return `The ${T(Q, i)} chosen in the previous step is appended to the sequence at position ${fmtPos(pp)}. <b>Autoregression</b>: the output immediately becomes the next input. The image’s ${V.Nv} tokens have long been in the KV cache; the original image is never looked at again.`;
    }
    case 'llm': {
      const [a, b] = Q.manifest.groundLayers;
      return `The language model’s <b>${t.layers}</b> layers are exactly the same as in a text-only model (Qwen3 architecture: RMSNorm, attention with ${t.heads} Q / ${t.kvHeads} KV heads, SwiGLU). The image is just a run of special tokens. While generating ${nxt}, layers ${a}–${b} give ${fmtPct(avgMass(Q, g, a, b))} of their attention to the image.`;
    }
    case 'layer': return layerExplainEn(s, Q, ctx);
    case 'head': {
      const top = st.top.slice(0, 3).map(([, p, tt]) => `${tokHTML(tt, 'r-assistant')} ${P(p)}`).join(', ');
      if (!s.sub) return `The vector at the last position goes through RMSNorm and is multiplied by the output matrix shared with the embedding table (${t.hidden} × ${t.vocab.toLocaleString('en-US')}); softmax turns the scores into probabilities: ${top}… and the top one is taken.`;
      if (s.sub === 'logits') return `Every token in the vocabulary gets a score (${t.vocab.toLocaleString('en-US')} of them); softmax turns them into probabilities. The leaders: ${top}.`;
      const end = st.chosenS === '<|im_end|>' ? ' It picked the end-of-turn token, so the reply stops here.' : '';
      return `<b>Greedy decoding</b>: simply take the most likely token, ${nxt} (${P(st.p)}). No random numbers — the same image and question always give the same answer.${end}`;
    }
  }
  return '';
}

function vitExplainEn(s, Q, ctx) {
  const V = Q.V, v = Q.manifest.model.vision, mic = V.micro;
  if (s.sub === 'embed') {
    if (!s.mi) return `<b>Patch embedding</b>: each patch’s ${v.inDim} numbers are multiplied by a ${v.inDim} × ${v.hidden} matrix (implemented as a 3-D convolution whose stride equals its kernel size), giving a ${v.hidden}-dim vector. All ${V.Np} patches do this independently.`;
    if (s.mi === 'conv') return `${v.hidden} output channels = ${v.hidden} kernels, each also 3 × 2 × 16 × 16. The monitor shows the 16 kernels that respond most strongly to this patch (both frames summed, red/green/blue colored by the real weights): some look for edges, some for a particular color.`;
    const where = `the patch at row ${mic.row}, column ${mic.col}`;
    if (s.mi === 'pick') return `Zoom in on ${where} (it belongs to ${cell(V, mic.token)}, the vision token that received the most attention over all the questions about this image) and see how <b>dimension ${mic.ch}</b> of its embedding is computed: ${v.inDim} pixel values, each multiplied by the matching weight in kernel ${mic.ch}, then added up.`;
    if (s.mi === 'mul') return `Multiply term by term. The 12 largest contributions are listed below (the two frames have identical pixels but different weights, so the same pixel can appear twice). Orange is positive, blue negative; every term is tiny — the absolute values of all ${v.inDim} terms add up to ${mic.absSum.toFixed(2)}.`;
    return `Add up all ${v.inDim} terms, plus the bias b = ${mic.bias.toFixed(4)}: <b>${mic.conv.toFixed(4)}</b> (recomputed in float32 it is ${mic.mine.toFixed(4)}; the difference is rounding in the bf16 convolution). Then the position embedding adds ${f(mic.pos, 4)}, so this number enters layer 0 as <b>${mic.after.toFixed(4)}</b>.`;
  }
  if (s.sub === 'pos') return `<b>Position embedding</b>: the model stores a ${v.posGrid}×${v.posGrid} position table, bilinearly interpolated to this image’s ${V.gh}×${V.gw} and added to each patch. On top of that, every attention layer uses <b>2-D RoPE</b>: part of q and k is rotated by the patch’s row and part by its column.`;
  if (s.L === undefined) return `<b>Vision encoder (ViT)</b>: ${v.depth} Transformer layers, ${v.hidden} dims, ${v.heads} heads. Unlike the language model there is <b>no causal mask</b>: all ${V.Np} patches see each other. The monitor colors each layer’s features by their top three principal components: higher up, patches of the same object get more and more similar colors.`;
  const L0 = s.L, dist = V.vitDist[L0], dmean = dist.reduce((a, b) => a + b, 0) / dist.length;
  const [xn, an, mn] = V.vitStats[L0];
  const has = V.vitAttnLayers.includes(L0);
  const q = ctx.vq ?? mic.token;
  const ds = v.deepstack.includes(L0) ? ' DeepStack also takes a copy of this layer’s output and sends it straight to the language model.' : '';
  if (!s.op) return `ViT layer <b>${L0}</b>: on average the 16 heads look <b>${dmean.toFixed(1)}</b> patches away (mean attention distance). ${has ? `The monitor shows attention from token ${cell(V, q)} (click the monitor to pick another).` : 'The monitor shows the principal-component colors of this layer’s output features.'}${ds}`;
  if (s.op === 'ln1' || s.op === 'ln2') return `<b>LayerNorm</b>: subtract the mean, divide by the standard deviation, then scale and shift (the vision encoder uses LayerNorm; the language model uses RMSNorm). Incoming vectors have an average length of ${xn.toFixed(1)}.`;
  if (s.op === 'attn') {
    if (!has) return `<b>Self-attention</b>: ${v.heads} heads of ${v.hidden / v.heads} dims each. q and k carry 2-D rotary position encoding, so up/down/left/right can be told apart. The heads’ mean attention distances range from ${Math.min(...dist).toFixed(1)} to ${Math.max(...dist).toFixed(1)} patches (monitor).`;
    const a = V.vattn(L0, q), self = V.vattnSelf(L0, q);
    const top = argmax(a);
    return `Attention from token ${cell(V, q)} (mean of 16 heads, shown at 2×2-merged resolution): ${P(self)} on itself; the most goes to ${cell(V, top)} (${P(a[top])}). Click any cell on the monitor to change the query position.`;
  }
  if (s.op === 'add1') return `The attention output is added back to the residual stream (average length ${an.toFixed(1)}).`;
  if (s.op === 'mlp') return `<b>Feed-forward network</b>: ${v.hidden} → ${v.ffn} → ${v.hidden}, with a GELU activation (tanh approximation). Each patch is processed on its own, without talking to the others.`;
  return `The feed-forward output is added back to the residual stream (average length ${mn.toFixed(1)}); layer ${L0} is done.${ds}`;
}

function layerExplainEn(s, Q, ctx) {
  const V = Q.V, M = Q.manifest.model, t = M.text;
  const { L: L0, g } = s, i = Q.row(g);
  const lens = Q.lensAt(g, L0);
  const guess = `${tokHTML(lens[0][0], 'r-assistant')} ${P(lens[0][1])}`;
  const mass = Q.mass(g, L0);
  const a = Q.attImg(g, L0), top = argmax(a);
  const words = lensWords(V, L0).map(([w]) => `“${esc(w)}”`).join(' ');
  const [ga, gb] = Q.manifest.groundLayers;
  const grounded = L0 >= ga && L0 <= gb;
  if (!s.op) return `Layer <b>${L0}</b>: ${T(Q, i)} gives <b>${fmtPct(mass)}</b> of its attention to the image, most of it on ${cell(V, top)}. ${grounded ? 'This is one of the “grounded” layers: attention concentrates on the object related to the current token.' : L0 < ga ? 'At such a shallow layer the model mostly just “glances” at the image and doesn’t really lock onto objects.' : ''} Logit lens: if it stopped at this layer, it would say ${guess}.${words ? `<br>At this layer the image tokens read like: ${words}` : ''}`;
  if (s.op === 'ln1') return `<b>RMSNorm</b>: divide the vector by its root mean square, then multiply by a set of learned weights.`;
  if (s.op === 'ln2') return `Another <b>RMSNorm</b>, getting ready for the feed-forward network.`;
  if (s.op === 'attn') {
    if (s.mi === 'heads') {
      const hs = [...Array(t.heads).keys()].map((h) => ({ h, m: Q.headMass(g, L0, h) })).sort((x, y) => y.m - x.m);
      return `The <b>${t.heads}</b> heads of layer ${L0} each look at their own thing: some look almost only at text, others give most of their attention to the image. The heads that look at the image the most are head ${hs[0].h} (${P(hs[0].m)}) and head ${hs[1].h} (${P(hs[1].m)}). The monitor shows each head’s attention on the image, normalized separately.`;
    }
    const txt = Q.txt(g, L0).slice(0, 2).map((x) => `${T(Q, x.j)} ${P(x.w)}`).join(', ');
    return `<b>Attention</b>: the q of ${T(Q, i)} is dotted with the k of every earlier token (positions rotated by M-RoPE; image tokens by their (t, h, w)). The image gets <b>${fmtPct(mass)}</b>; among the text, the most attention goes to ${txt}. The ${M.text.heads} query heads share ${t.kvHeads} key/value groups.`;
  }
  const [da, dm] = Q.delta(g, L0);
  if (s.op === 'add1') return `The attention output (length ${da.toFixed(1)}) is added back to the residual stream. This is how information read from the image gets “moved” into the current position.`;
  if (s.op === 'mlp') return `<b>SwiGLU feed-forward</b>: ${t.hidden} → ${t.ffn} → ${t.hidden}, working on the current position alone. It adds a vector of length ${dm.toFixed(1)} to the residual stream.`;
  if (s.op === 'add2') return `The feed-forward output is added back to the residual stream; layer ${L0} is done. The logit lens now reads ${guess}.`;
  let mean = 0, base = 0;
  for (let m = 0; m < V.Nv; m++) { mean += V.dsNorm(L0, m) / V.Nv; base += V.vnorm(L0 + 1, m) / V.Nv; }
  return `<b>DeepStack</b>: features from vision-encoder layer ${M.vision.deepstack[L0]} (through their own merger, average length ${mean.toFixed(1)}) are <b>added at the image positions</b>; afterwards the residual stream at the image positions has an average length of ${base.toFixed(1)}. Only image positions change; text positions are untouched. This happens only once, during prefill.`;
}

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
      if (s.sub === 'norm') return `x ${b(`[${V.Np} × ${v.inDim}]`)} = ${L('3 通道 × 2 帧 × 16 × 16', '3 channels × 2 frames × 16 × 16')}`;
      return null;
    case 'vit':
      if (s.sub === 'embed') return `x [${V.Np}×${v.inDim}] @ W ${b(`[${v.inDim}×${v.hidden}]`)} + b → v [${V.Np}×${v.hidden}]`;
      if (s.sub === 'pos') return `pos [${v.posGrid}×${v.posGrid}×${v.hidden}] → ${L('插值', 'interp')} ${b(`[${V.gh}×${V.gw}×${v.hidden}]`)}`;
      if (s.op === 'attn') return L(`qkv [${V.Np}×${v.hidden * 3}] → ${v.heads} 头 × ${v.hidden / v.heads} · 分数 ${b(`[${v.heads}×${V.Np}×${V.Np}]`)}`, `qkv [${V.Np}×${v.hidden * 3}] → ${v.heads} heads × ${v.hidden / v.heads} · scores ${b(`[${v.heads}×${V.Np}×${V.Np}]`)}`);
      if (s.op === 'mlp') return `[${V.Np}×${v.hidden}] → ${b(`[${V.Np}×${v.ffn}]`)} → [${V.Np}×${v.hidden}]`;
      if (s.L !== undefined) return `v ${b(`[${V.Np}×${v.hidden}]`)}`;
      return null;
    case 'merge':
      if (s.sub === 'group') return `[${V.Np}×${v.hidden}] → ${b(`[${V.Nv}×${v.hidden * 4}]`)}`;
      if (s.sub === 'mlp') return `[${V.Nv}×${v.hidden * 4}] → [${V.Nv}×${v.hidden * 4}] → ${b(`[${V.Nv}×${v.out}]`)}`;
      if (s.sub === 'deep') return `3 × ${b(`[${V.Nv}×${v.out}]`)} → ${L('LLM 第 0–2 层', 'LLM layers 0–2')}`;
      return null;
    case 'splice':
      if (s.sub === 'template') return `ids ${b(`[${Q.P}]`)} = ${L(`文字 ${Q.P - V.Nv} + 图片 ${V.Nv}`, `text ${Q.P - V.Nv} + image ${V.Nv}`)}`;
      if (s.sub === 'mrope') return `pos ${b(`[3×${Q.P}]`)} (t, h, w)`;
      return null;
    case 'layer':
      if (s.op === 'attn') return `q [${t.heads}×${t.headDim}] · ${L('K 缓存', 'K cache')} ${b(`[${t.kvHeads}×${Q.row(s.g) + 1}×${t.headDim}]`)}`;
      if (s.op === 'mlp') return `h [1×${t.hidden}] → ${b(`[1×${t.ffn}]`)} → [1×${t.hidden}]`;
      if (s.op === 'deep') return `${L('x[图片]', 'x[image]')} ${b(`[${V.Nv}×${t.hidden}]`)} += deep[${s.L}]`;
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
      add(L('‖v‖ 平均', '‖v‖ mean'), xn.toFixed(1));
      add('‖attn‖ / ‖mlp‖', `${an.toFixed(1)} / ${mn.toFixed(1)}`);
      add(L('注意距离 头均', 'attn dist (head avg)'), (dist.reduce((a, b) => a + b, 0) / dist.length).toFixed(2));
      const pv = V.pcaVar[s.L + 1];
      add(L('PCA 方差', 'PCA variance'), pv.map((x) => (x * 100).toFixed(0) + '%').join(' '));
    } else { add('blocks', v.depth); add('dim', v.hidden); add('heads', v.heads); add('patches', V.Np); }
  } else if (s.ph === 'merge') {
    add('in', `${V.Np} × ${v.hidden}`);
    add('group', `${V.Nv} × ${v.hidden * 4}`);
    add('out', `${V.Nv} × ${v.out}`);
    let mn = 0;
    for (let m = 0; m < V.Nv; m++) mn += V.vnorm(0, m) / V.Nv;
    add(L('‖img‖ 平均', '‖img‖ mean'), mn.toFixed(1));
    if (s.sub === 'deep') for (let d = 0; d < 3; d++) { let x = 0; for (let m = 0; m < V.Nv; m++) x += V.dsNorm(d, m) / V.Nv; add(`‖deep[${d}]‖`, x.toFixed(1)); }
  } else if (s.ph === 'splice') {
    add('len(ids)', Q.P);
    add('vision_start', `#${V.vs - 1}`);
    add('image_pad', `#${V.vs} … #${V.vs + V.Nv - 1}`);
    add(L('pos 首个图片', 'pos first image'), fmtPos(Q.pos(V.vs)));
    add(L('pos 末个图片', 'pos last image'), fmtPos(Q.pos(V.vs + V.Nv - 1)));
    add(L('pos 问题开头', 'pos question start'), fmtPos(Q.pos(V.vs + V.Nv + 1)));
  } else if (s.ph === 'layer' || s.ph === 'llm' || s.ph === 'pass' || s.ph === 'read') {
    const LY = s.L ?? null;
    add(L('词元', 'token'), `#${i}`);
    add('pos (t,h,w)', fmtPos(Q.pos(i)));
    if (LY != null) {
      const a = Q.attImg(g, LY), m = argmax(a);
      add('layer', `${LY} / ${t.layers - 1}`);
      add(L('看图比例', 'on image'), fmtPct(Q.mass(g, LY)));
      add(L('最受关注', 'most attended'), `${cell(V, m)} ${fmtPct(a[m])}`);
      const lens = Q.lensAt(g, LY);
      add('lens', `${lens[0][0].replace(/\n/g, '↵')} ${fmtPct(lens[0][1])}`);
      const [da, dm] = Q.delta(g, LY);
      add('‖Δattn‖ / ‖Δmlp‖', `${da.toFixed(1)} / ${dm.toFixed(1)}`);
      if (s.mi === 'heads') {
        const hs = [...Array(t.heads).keys()].map((h) => ({ h, m: Q.headMass(g, LY, h) })).sort((x, y) => y.m - x.m);
        hs.slice(0, 4).forEach((x) => add(L(`头 ${x.h} 看图`, `head ${x.h} on image`), fmtPct(x.m)));
      }
    } else {
      const [ga, gb] = Q.manifest.groundLayers;
      add(L(`看图比例 L${ga}–${gb}`, `on image L${ga}–${gb}`), fmtPct(avgMass(Q, g, ga, gb)));
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
