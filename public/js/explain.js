// 调试器里的“代码 / 这一步 / 变量”三块内容，全部用真实导出的数值填充。
import { esc, tokHTML, tokPlain, fmtPct } from './ui.js';
import { DEPTH_NAMES } from './timeline.js';

const K = (s) => `<span class="kw">${s}</span>`;
const F = (s) => `<span class="fn">${s}</span>`;
const C = (s) => `<span class="cm"># ${s}</span>`;
const N = (s) => `<span class="nu">${s}</span>`;

export const CODE = [
  `${K('def')} ${F('chat')}(question):`,
  `    ids = ${F('apply_chat_template')}(system, question)  ${C('读入')}`,
  `    ${K('while')} True:`,
  `        x = embed_tokens[ids]            ${C('151936×1024 查表')}`,
  `        ${K('for')} layer ${K('in')} layers:            ${C('28 层')}`,
  `            h = ${F('rms_norm')}(x)`,
  `            q, k, v = h@Wq, h@Wk, h@Wv    ${C('16 Q 头 / 8 KV 头')}`,
  `            q, k = ${F('rope')}(${F('norm')}(q)), ${F('rope')}(${F('norm')}(k))`,
  `            cache.${F('append')}(k, v)            ${C('KV 缓存')}`,
  `            s = q @ cache.k.T / ${F('sqrt')}(${N('128')})`,
  `            a = ${F('softmax')}(s + causal_mask)`,
  `            x = x + (a @ cache.v) @ Wo    ${C('残差')}`,
  `            h = ${F('rms_norm')}(x)`,
  `            g, u = h@W_gate, h@W_up      ${C('1024→3072')}`,
  `            x = x + (${F('silu')}(g) * u) @ W_down  ${C('SwiGLU')}`,
  `        logits = ${F('rms_norm')}(x[-1]) @ embed_tokens.T`,
  `        probs = ${F('softmax')}(logits / ${N('0.7')})`,
  `        nxt = ${F('sample')}(${F('top_p')}(${F('top_k')}(probs, ${N('20')}), ${N('0.8')}))`,
  `        ${K('if')} nxt == <span class="nu">"&lt;|im_end|&gt;"</span>: ${K('break')}`,
  `        ids.${F('append')}(nxt)`,
];

export function linesFor(s, Q) {
  const last = s.g === Q.G - 1 && Q.ended;
  switch (s.ph) {
    case 'pass': return [4, 5, 16, 17, 18];
    case 'read': return s.g === 0 ? [2] : [20];
    case 'embed': return [4];
    case 'layers': return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    case 'head': return !s.sub ? [16, 17] : s.sub === 'softmax' ? [17] : [16];
    case 'sample': return (!s.sub ? [17, 18] : s.sub === 'temp' ? [17] : [18]).concat(last ? [19] : []);
    case 'layer':
      if (s.mi) {
        if (s.op === 'attn') return [10];
        return s.mi === 'mul' || s.mi === 'sum' ? [14] : [15];
      }
      if (s.sub) return { qkv: [7, 8, 9], score: [10], softmax: [11], mix: [12], up: [14], act: [15], down: [15] }[s.sub];
      if (s.op) return { ln1: [6], attn: [7, 8, 9, 10, 11, 12], add1: [12], ln2: [13], mlp: [14, 15], add2: [15] }[s.op];
      return [5];
  }
  return [];
}

export function renderCode(el) {
  el.innerHTML = CODE.map((l, i) => `<span class="ln" data-n="${i + 1}"><span class="bp"></span><span class="no">${i + 1}</span>${l}</span>`).join('');
}

// 这一行里，除了开头之外注意力最集中的头（通常最能说明问题）
export function interestingHead(Q, L, i) {
  let best = 0, bw = -1;
  for (let h = 0; h < Q.H; h++) {
    for (const { j, w } of Q.att(L, h, i)) if (j !== 0 && j !== i && w > bw) { bw = w; best = h; }
  }
  return best;
}

const T = (Q, i) => (Q.tokens[i] ? tokHTML(Q.tokens[i].s, `r-${Q.tokens[i].role}${Q.tokens[i].sp ? ' sp' : ''}`) : '');
const P = (p) => `<em>${fmtPct(p)}</em>`;

export function explain(s, Q, ctx = {}) {
  const g = s.g, i = Q.row(g), st = Q.steps[g];
  const cur = T(Q, i);
  const nxt = tokHTML(st.chosenS, 'r-assistant');
  const ctxLen = Q.P + g;
  switch (s.ph) {
    case 'pass':
      return `第 <b>${g + 1}</b> 个词元：模型读入前面的 <b>${ctxLen}</b> 个词元，把 28 层完整算一遍，最后吐出 ${nxt}。<br>回答里的每一个字，都是这样一轮完整计算的结果。`;
    case 'read':
      return g === 0
        ? `聊天模板把系统提示、你的问题和几个特殊标记拼在一起，一共 <b>${Q.P}</b> 个词元。<br>模型看到的不是“对话”，只是一串编号；<b>&lt;|im_start|&gt;assistant</b> 之后该轮到它接着写了。`
        : `上一步选出的 ${T(Q, i)} 被接到序列末尾，成为第 <b>${ctxLen}</b> 个词元。<b>自回归</b>：输出立刻变成下一步的输入。`;
    case 'embed':
      return g === 0
        ? `${Q.P} 个词元同时去 <b>151936 × 1024</b> 的嵌入表里取出各自的一行，变成 1024 维向量。这一步叫<b>预填充</b>，所有位置可以并行计算。`
        : `只有新词元 ${cur} 需要查表。前面词元在每一层算出的 K、V 都存在 <b>KV 缓存</b>里，不用重算，所以越往后生成越省事。`;
    case 'layers':
      return `向量依次穿过 <b>28 个结构完全相同</b>、但权重各不相同的 Transformer 层。每层都先让词元之间交流（注意力），再让每个词元各自加工（前馈网络）。`;
    case 'layer': return layerExplain(s, Q, ctx, i, cur);
    case 'head': {
      const top = st.top.slice(0, 3).map(([, p, t]) => `${tokHTML(t, 'r-assistant')} ${P(p)}`).join('，');
      if (!s.sub) return `最后一个位置的向量做一次 RMSNorm，再乘上<b>和嵌入表共用的那块权重</b>（1024 × 151936），得到词表里每个词元的分数，softmax 成概率：${top}……`;
      if (s.sub === 'norm') return `最终 RMSNorm：把第 28 层输出的向量拉回统一的尺度。`;
      if (s.sub === 'unembed') return `乘上输出矩阵，得到 <b>151936</b> 个分数（logits）。Qwen3-0.6B 的输出矩阵和嵌入表是同一块权重（tied embeddings），省下了 1.5 亿个参数。`;
      return `softmax 把分数变成概率：${top}……剩下十五万个词元分掉其余的概率。`;
    }
    case 'sample': {
      const pool = st.pool.length;
      const rank = st.chosenRank === 0 ? '概率最高的那个' : `第 ${st.chosenRank + 1} 名`;
      const end = st.chosenS === '<|im_end|>' ? ' 它选中了结束标记，这条回复到此为止。' : '';
      if (!s.sub) return `采样：温度 0.7 → 只留前 20 名 → 累计概率凑够 80% 就截断，剩 <b>${pool}</b> 个候选。随机数 u = <b>${st.u.toFixed(3)}</b> 落在 ${nxt} 的区间里（${rank}）。${end}`;
      if (s.sub === 'temp') return `温度 <b>0.7</b>：所有分数除以 0.7，高分和低分拉得更开，分布变得更“尖”。`;
      if (s.sub === 'topk') return `top-k：只保留分数最高的 <b>20</b> 个词元，其余全部出局。`;
      if (s.sub === 'topp') return `top-p：从高到低累加概率，凑够 <b>80%</b> 就停，最后剩下 <b>${pool}</b> 个候选。`;
      return `掷出随机数 <b>u = ${st.u.toFixed(4)}</b>，它落在 ${nxt} 的区间（${rank}，原始概率 ${P(st.chosenP1)}）。${end}`;
    }
  }
  return '';
}

function layerExplain(s, Q, ctx, i, cur) {
  const { L, g } = s;
  const lens = Q.lensAt(g, L);
  const guess = `${tokHTML(lens.top[0][2], 'r-assistant')} ${P(lens.top[0][1])}`;
  const mean = Q.attMean(L, i, 2);
  const tgt = mean[0] ? `${T(Q, mean[0].j)}（${fmtPct(mean[0].w)}）` : '';
  if (!s.op) return `第 <b>${L}</b> 层：${cur} 平均把最多的注意力给了 ${tgt}。如果模型在这一层就停下直接开口，它会说 ${guess}（逻辑透镜）。`;
  if (s.op === 'ln1') return `<b>RMSNorm</b>：把向量除以它的均方根，再乘一组可学习的缩放系数。不减均值、没有偏置，比 LayerNorm 更省。`;
  if (s.op === 'ln2') return `再做一次 <b>RMSNorm</b>，准备进入前馈网络。`;
  if (s.op === 'add1') return `<b>残差相加</b>：注意力的输出（经过 o_proj）直接加回原来的向量。每层只是“往上加一点”，信息不会被覆盖。当前向量长度 ‖x‖ = <b>${Q.norm(L, i).toFixed(1)}</b>。`;
  if (s.op === 'add2') return `前馈网络的输出加回残差流，第 ${L} 层结束。逻辑透镜此刻读到的是 ${guess}。`;
  const h = ctx.head ?? interestingHead(Q, L, i);
  if (s.op === 'attn') {
    if (!s.sub) return `<b>分组查询注意力（GQA）</b>：16 个查询头，每 2 个共用一组键值头，共 8 组。${cur} 平均最关注 ${tgt}。`;
    const row = Q.att(L, h, i);
    const best = row.filter((r) => r.j !== i)[0] || row[0];
    if (s.sub === 'qkv') return `新向量分别乘 W<sub>q</sub>、W<sub>k</sub>、W<sub>v</sub>：得到 <b>16 × 128</b> 维的 Q，和各 <b>8 × 128</b> 维的 K、V。Q、K 先各自做 RMSNorm（Qwen3 新加的），再按位置旋转（<b>RoPE</b>，θ = 10⁶）。K、V 存进缓存。`;
    if (s.sub === 'score') return `第 <b>${h}</b> 头用自己的 Q 和缓存里每个词元的 K 做点积，再除以 √128。它打分最高的是 ${best ? T(Q, best.j) : ''}：<b>${best ? best.s.toFixed(2) : ''}</b>。`;
    if (s.sub === 'softmax') return `softmax 把分数变成加起来等于 1 的权重。第 ${h} 头给 ${best ? T(Q, best.j) : ''} ${best ? P(best.w) : ''}。后面的词元被因果遮罩挡住，永远看不见。`;
    if (s.sub === 'mix') return `按权重把各个词元的 V 加起来。16 个头的结果拼成 2048 维，再乘 W<sub>o</sub> 回到 1024 维。`;
    const d = Q.dotAt(L, g);
    if (!d) return '';
    if (s.mi === 'mul') return `第 ${d.head} 个 Q 头（用第 ${d.kv} 组 KV）和 ${T(Q, d.key)} 的 K 在 128 个维度上逐一相乘。这里显示乘积最大的 12 项。`;
    if (s.mi === 'sum') return `128 项全部加起来：<b>${d.sum.toFixed(3)}</b>。`;
    return `除以 √128 ≈ 11.31，得到打分 <b>${d.score.toFixed(3)}</b>。softmax 之后，这个头给 ${T(Q, d.key)} 的权重是 ${P(d.w)}。`;
  }
  // mlp
  const count = Q.mlpCount(g, L);
  if (!s.sub) return `<b>SwiGLU 前馈网络</b>：先扩到 3072 维，经过门控激活，再压回 1024 维。这一步有 <b>${count}</b> 个神经元明显激活。`;
  if (s.sub === 'up') return `gate_proj 和 up_proj 两个矩阵，同时把 1024 维扩到 <b>3072</b> 维，得到两组数：<b>g</b> 和 <b>u</b>。`;
  if (s.sub === 'act') return `g 经过 <b>SiLU</b> 激活，再和 u 逐个相乘。g 像一道<b>阀门</b>，决定每个神经元放多少 u 通过。明显激活的有 <b>${count}</b> 个。`;
  if (s.sub === 'down') return `down_proj 把 3072 维压回 1024 维，准备加回残差流。`;
  const n = Q.neuronAt(L, g);
  if (!n) return '';
  if (s.mi === 'mul') return `神经元 <b>#${n.n}</b>：1024 个输入分别乘 gate 和 up 两组权重。这里显示贡献最大的 12 项（全是真实的 bf16 权重）。`;
  if (s.mi === 'sum') return `全部加起来：g = <b>${n.gz.toFixed(3)}</b>，u = <b>${n.uz.toFixed(3)}</b>。`;
  if (s.mi === 'silu') return `SiLU(g) = g · σ(g) = <b>${n.silu.toFixed(3)}</b>。负数会被压到接近 0。`;
  return `SiLU(g) × u = <b>${(n.silu * n.uz).toFixed(3)}</b>：这就是神经元 #${n.n} 的输出。`;
}

// 这一步的矩阵形状（预填充时 n = 提示长度，之后每步 n = 1）
const D = (s) => `<b>${s}</b>`;
export function shapeOf(s, Q) {
  const M = Q.M, n = s.g === 0 ? Q.P : 1, L = Q.P + s.g;
  const H = M.hidden, qd = M.heads * M.headDim, kd = M.kvHeads * M.headDim, F = M.ffn, V = M.vocab;
  const x = `[${n}×${H}]`;
  switch (s.ph) {
    case 'read': return s.g === 0 ? `ids ${D(`[${Q.P}]`)}：${Q.P} 个词元编号` : `ids ${D(`[${L}]`)}：末尾接上 1 个`;
    case 'embed': return `one-hot ${D(`[${n}×${V}]`)} @ E ${D(`[${V}×${H}]`)} → x ${D(x)}<br><span class="dimmed">实际实现是按编号直接取出 E 的第 id 行</span>`;
    case 'layers': case 'pass': return `x ${D(x)} → 28 × Transformer 块 → ${D(x)}`;
    case 'head':
      if (s.sub === 'norm') return `x[-1] ${D(`[1×${H}]`)} ÷ RMS × γ ${D(`[${H}]`)}`;
      if (s.sub === 'softmax') return `softmax(logits ${D(`[1×${V}]`)}) → p ${D(`[1×${V}]`)}，和为 1`;
      return `x[-1] ${D(`[1×${H}]`)} @ Eᵀ ${D(`[${H}×${V}]`)} → logits ${D(`[1×${V}]`)}`;
    case 'sample': return `p ${D(`[${V}]`)} → ÷0.7 → 前 20 → 累计 80% → 候选 ${D(`[${Q.steps[s.g].pool.length}]`)} → 1 个词元`;
    case 'layer': {
      if (!s.op) return `x ${D(x)} → 注意力 → 前馈 → ${D(x)}`;
      if (s.op === 'ln1' || s.op === 'ln2') return `h = x ${D(x)} ÷ RMS(x) × γ ${D(`[${H}]`)} → ${D(x)}`;
      if (s.op === 'add1' || s.op === 'add2') return `x ${D(x)} + Δx ${D(x)} → ${D(x)}`;
      if (s.op === 'attn') {
        if (!s.sub || s.sub === 'qkv') return `h ${D(x)} @ W<sub>q</sub> ${D(`[${H}×${qd}]`)} → q ${D(`[${n}×${qd}]`)} = ${M.heads} 头 × ${M.headDim}<br>h ${D(x)} @ W<sub>k</sub> ${D(`[${H}×${kd}]`)} → k = ${M.kvHeads} 头 × ${M.headDim}<br>h ${D(x)} @ W<sub>v</sub> ${D(`[${H}×${kd}]`)} → v = ${M.kvHeads} 头 × ${M.headDim}`;
        if (s.sub === 'score') {
          if (s.mi) return `q<sub>头</sub> ${D(`[${M.headDim}]`)} · k ${D(`[${M.headDim}]`)} → 标量，÷ √${M.headDim}`;
          return `q ${D(`[${M.heads}×${n}×${M.headDim}]`)} @ Kᵀ ${D(`[${M.heads}×${M.headDim}×${L}]`)} → s ${D(`[${M.heads}×${n}×${L}]`)}<br><span class="dimmed">K 只有 ${M.kvHeads} 头，每个复用给 2 个 Q 头</span>`;
        }
        if (s.sub === 'softmax') return `softmax(s ${D(`[${M.heads}×${n}×${L}]`)} + 遮罩) → a，每行和为 1`;
        return `a ${D(`[${M.heads}×${n}×${L}]`)} @ V ${D(`[${M.heads}×${L}×${M.headDim}]`)} → ${D(`[${M.heads}×${n}×${M.headDim}]`)}<br>拼接 ${D(`[${n}×${qd}]`)} @ W<sub>o</sub> ${D(`[${qd}×${H}]`)} → Δx ${D(x)}`;
      }
      if (s.op === 'mlp') {
        if (s.mi) return `x ${D(`[${H}]`)} · w<sub>gate</sub> ${D(`[${H}]`)} → g（标量）；x · w<sub>up</sub> → u；输出 silu(g)·u`;
        if (!s.sub || s.sub === 'up') return `h ${D(x)} @ W<sub>gate</sub> ${D(`[${H}×${F}]`)} → g ${D(`[${n}×${F}]`)}<br>h ${D(x)} @ W<sub>up</sub> ${D(`[${H}×${F}]`)} → u ${D(`[${n}×${F}]`)}`;
        if (s.sub === 'act') return `silu(g) ⊙ u：${D(`[${n}×${F}]`)} 逐个相乘`;
        return `${D(`[${n}×${F}]`)} @ W<sub>down</sub> ${D(`[${F}×${H}]`)} → Δx ${D(x)}`;
      }
    }
  }
  return '';
}

export function watch(s, Q, ctx = {}) {
  const g = s.g, i = Q.row(g), st = Q.steps[g];
  const rows = [
    ['wh', '变量'],
    ['step', `${g + 1} / ${Q.G}`],
    ['len(ids)', Q.P + g],
    ['ids[-1]', `${tokPlain(Q.tokens[i].s)}  #${Q.tokens[i].id}`],
  ];
  if (s.ph === 'layer' || s.ph === 'layers') {
    if (s.L !== undefined) {
      rows.push(['layer', `${s.L} / ${Q.NL - 1}`]);
      rows.push(['‖x‖', Q.norm(s.L, i).toFixed(2)]);
      const lens = Q.lensAt(g, s.L).top[0];
      rows.push(['lens(x)', `${tokPlain(lens[2])} ${fmtPct(lens[1])}`]);
    }
    if (s.op === 'attn') rows.push(['head', ctx.head == null ? `${interestingHead(Q, s.L, i)}` : `${ctx.head}`], ['kv_group', `${Math.floor((ctx.head ?? interestingHead(Q, s.L, i)) / 2)}`]);
    if (s.op === 'mlp') rows.push(['active', `${Q.mlpCount(g, s.L)} / 3072`]);
    if (s.mi && s.op === 'mlp') { const n = Q.neuronAt(s.L, g); if (n) rows.push(['neuron', `#${n.n}`], ['g', n.gz.toFixed(4)], ['u', n.uz.toFixed(4)], ['silu(g)*u', (n.silu * n.uz).toFixed(4)]); }
    if (s.mi && s.op === 'attn') { const d = Q.dotAt(s.L, g); if (d) rows.push(['q·k', d.sum.toFixed(4)], ['score', d.score.toFixed(4)], ['weight', fmtPct(d.w)]); }
  }
  if (s.ph === 'head' || s.ph === 'sample' || s.ph === 'pass') {
    rows.push(['top1', `${tokPlain(st.top[0][2])} ${fmtPct(st.top[0][1])}`]);
    if (s.ph !== 'head') rows.push(['u', st.u.toFixed(4)], ['candidates', st.pool.length], ['nxt', `${tokPlain(st.chosenS)}  #${st.chosen}`]);
  }
  return rows;
}

export function renderWatch(el, rows) {
  el.innerHTML = rows.map(([k, v]) => (k === 'wh' ? `<div class="wh">${esc(v)}</div>` : `<span class="k">${esc(k)}</span><span class="v" title="${esc(String(v))}">${esc(String(v))}</span>`)).join('');
}

const OP_NAME = { ln1: 'RMSNorm', attn: '注意力', add1: '残差 ⊕', ln2: 'RMSNorm', mlp: '前馈 SwiGLU', add2: '残差 ⊕' };
const SUB_NAME = { qkv: 'Q·K·V', score: '打分', softmax: 'softmax', mix: '加权求和', up: '升维', act: '门控激活', down: '降维', norm: 'RMSNorm', unembed: '输出矩阵', temp: '温度', topk: 'top-k', topp: 'top-p', draw: '掷骰子' };
const MI_NAME = { mul: '逐项相乘', sum: '求和', scale: '÷√128', silu: 'SiLU', gate: '× u' };

export function stepLabel(s) {
  switch (s.ph) {
    case 'pass': return `第 ${s.g + 1} 个词元`;
    case 'read': return s.g === 0 ? '套聊天模板' : '接上新词元';
    case 'embed': return '嵌入';
    case 'layers': return '28 层';
    case 'head': return s.sub ? `输出 · ${SUB_NAME[s.sub]}` : '输出头';
    case 'sample': return s.sub ? `采样 · ${SUB_NAME[s.sub]}` : '采样';
    case 'layer': return [`第 ${s.L} 层`, s.op && OP_NAME[s.op], s.sub && SUB_NAME[s.sub], s.mi && MI_NAME[s.mi]].filter(Boolean).join(' · ');
  }
  return '';
}

// 面包屑：每一级是 {d, label}
export function crumbs(depth, s) {
  const out = [{ d: 0, label: '对话' }, { d: 1, label: '黑箱' }];
  if (depth >= 2) out.push({ d: 2, label: '结构' });
  if (depth >= 3) {
    if (s.ph === 'layer') out.push({ d: 3, label: '层塔' });
    else out.push({ d: 3, label: { read: '读入', embed: '嵌入', head: '输出头', sample: '采样' }[s.ph] || DEPTH_NAMES[3] });
  }
  if (depth >= 4 && s.ph === 'layer') out.push({ d: 4, label: `第 ${s.L} 层` });
  if (depth >= 5 && s.op) out.push({ d: 5, label: OP_NAME[s.op] });
  if (depth >= 6 && s.sub) out.push({ d: 6, label: SUB_NAME[s.sub] });
  if (depth >= 7 && s.mi === 'mul') out.push({ d: 7, label: '比特' });
  return out;
}
