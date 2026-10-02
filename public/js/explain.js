// 调试器里的“代码 / 这一步 / 变量”三块内容，全部用真实导出的数值填充。
// 中英两版：tr(中文, English)；数字都从当前问题的真实数据里取，两种语言用的是各自那一批真实运行。
import { esc, tokHTML, tokPlain, fmtPct } from './ui.js';
import { DEPTH_NAMES } from './timeline.js';
import { sums, neuronId, neuronMM, ropePos } from './stage/fields.js';
import { isEn, L as tr } from './i18n.js';

const K = (s) => `<span class="kw">${s}</span>`;
const F = (s) => `<span class="fn">${s}</span>`;
const C = (s) => `<span class="cm"># ${s}</span>`;
const N = (s) => `<span class="nu">${s}</span>`;

export const CODE = [
  `${K('def')} ${F('chat')}(question):`,
  `    ids = ${F('apply_chat_template')}(system, question)  ${C(tr('读入', 'read input'))}`,
  `    ${K('while')} True:`,
  `        x = embed_tokens[ids]            ${C(tr('151936×1024 查表', '151936×1024 lookup'))}`,
  `        ${K('for')} layer ${K('in')} layers:            ${C(tr('28 层', '28 layers'))}`,
  `            h = ${F('rms_norm')}(x)`,
  `            q, k, v = h@Wq, h@Wk, h@Wv    ${C(tr('16 Q 头 / 8 KV 头', '16 Q heads / 8 KV heads'))}`,
  `            q, k = ${F('rope')}(${F('norm')}(q)), ${F('rope')}(${F('norm')}(k))`,
  `            cache.${F('append')}(k, v)            ${C(tr('KV 缓存', 'KV cache'))}`,
  `            s = q @ cache.k.T / ${F('sqrt')}(${N('128')})`,
  `            a = ${F('softmax')}(s + causal_mask)`,
  `            x = x + (a @ cache.v) @ Wo    ${C(tr('残差', 'residual'))}`,
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
      if (s.mi) return { qkv: s.mi === 'rope' ? [8] : [7], score: [10], mix: [12], up: [14], act: [15], down: [15] }[s.sub] || [5];
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
      return tr(`第 <b>${g + 1}</b> 个词元：模型读入前面的 <b>${ctxLen}</b> 个词元，把 28 层完整算一遍，最后吐出 ${nxt}。<br>回答里的每一个字，都是这样一轮完整计算的结果。`,
        `Token <b>${g + 1}</b>: the model reads all <b>${ctxLen}</b> tokens so far, runs them through all 28 layers once, and finally outputs ${nxt}.<br>Every single token of the reply comes out of one full pass like this.`);
    case 'read':
      return g === 0
        ? tr(`聊天模板把系统提示、你的问题和几个特殊标记拼在一起，一共 <b>${Q.P}</b> 个词元。<br>模型看到的不是“对话”，只是一串编号；<b>&lt;|im_start|&gt;assistant</b> 之后该轮到它接着写了。`,
          `The chat template glues the system prompt, your question and a few special markers together: <b>${Q.P}</b> tokens in all.<br>The model doesn't see a "conversation", just a list of ID numbers. After <b>&lt;|im_start|&gt;assistant</b>, it's its turn to keep writing.`)
        : tr(`上一步选出的 ${T(Q, i)} 被接到序列末尾，成为第 <b>${ctxLen}</b> 个词元。<b>自回归</b>：输出立刻变成下一步的输入。`,
          `The ${T(Q, i)} picked in the previous step is appended to the sequence as token number <b>${ctxLen}</b>. That's <b>autoregression</b>: each output immediately becomes the next input.`);
    case 'embed':
      return g === 0
        ? tr(`${Q.P} 个词元同时去 <b>151936 × 1024</b> 的嵌入表里取出各自的一行，变成 1024 维向量。这一步叫<b>预填充</b>，所有位置可以并行计算。`,
          `All ${Q.P} tokens look up their own row in the <b>151936 × 1024</b> embedding table at once, each becoming a vector of 1024 numbers. This is called <b>prefill</b>: every position can be computed in parallel.`)
        : tr(`只有新词元 ${cur} 需要查表。前面词元在每一层算出的 K、V 都存在 <b>KV 缓存</b>里，不用重算，所以越往后生成越省事。`,
          `Only the new token ${cur} needs a lookup. The K and V that earlier tokens produced in every layer are kept in the <b>KV cache</b>, so nothing is recomputed, and each new token is cheap.`);
    case 'layers':
      return tr(`向量依次穿过 <b>28 个结构完全相同</b>、但权重各不相同的 Transformer 层。每层都先让词元之间交流（注意力），再让每个词元各自加工（前馈网络）。`,
        `The vectors pass through <b>28 Transformer layers that share exactly the same structure</b> but each have their own weights. In every layer the tokens first exchange information (attention), then each token is processed on its own (feed-forward network).`);
    case 'layer': return layerExplain(s, Q, ctx, i, cur);
    case 'head': {
      const top = st.top.slice(0, 3).map(([, p, t]) => `${tokHTML(t, 'r-assistant')} ${P(p)}`).join(tr('，', ', '));
      if (s.mi) {
        const e = Q.headMMAt(g);
        if (ctx.view === 'bits') return bitsExplain(tr('嵌入表里被选中的那个权重', 'the selected weight in the embedding table'));
        return mmExplain(s, e, { y: 'logit', x: 'h', w: 'E', wi: e ? `E[${e.j}, i]` : '', n: 1024, tail: e ? tr(`最终向量 ${X('h')} 的 1024 个数，和嵌入表里「${esc(tokPlain(e.token))}」那一行逐项相乘再相加，就是它的分数（logit）。模型给词表里全部 151936 个词元都这样算一遍，再用 softmax 变成概率。`,
          ` The 1024 numbers of the final vector ${X('h')} are multiplied one by one with the row of “${esc(tokPlain(e.token))}” in the embedding table and added up: that's its score (logit). The model does this for all 151,936 tokens in the vocabulary, then softmax turns the scores into probabilities.`) : '' });
      }
      if (!s.sub) return tr(`最后一个位置的向量做一次 RMSNorm，再乘上<b>和嵌入表共用的那块权重</b>（1024 × 151936），得到词表里每个词元的分数，softmax 成概率：${top}……`,
        `The vector at the last position gets one more RMSNorm, then is multiplied by <b>the same weights as the embedding table</b> (1024 × 151936). That gives a score for every token in the vocabulary, and softmax turns them into probabilities: ${top}…`);
      if (s.sub === 'norm') return tr(`最终 RMSNorm：把第 28 层输出的向量拉回统一的尺度。`, `Final RMSNorm: brings the vector coming out of the last (28th) layer back to a standard scale.`);
      if (s.sub === 'unembed') return tr(`乘上输出矩阵，得到 <b>151936</b> 个分数（logits）。Qwen3-0.6B 的输出矩阵和嵌入表是同一块权重（tied embeddings），省下了 1.5 亿个参数。`,
        `Multiply by the output matrix to get <b>151,936</b> scores (logits). In Qwen3-0.6B the output matrix and the embedding table are the same weights (tied embeddings), which saves about 156 million parameters.`);
      return tr(`softmax 把分数变成概率：${top}……剩下十五万个词元分掉其余的概率。`, `Softmax turns the scores into probabilities: ${top}… The other ~150,000 tokens share what's left.`);
    }
    case 'sample': {
      const pool = st.pool.length;
      const rank = st.chosenRank === 0 ? tr('概率最高的那个', 'the most likely one') : tr(`第 ${st.chosenRank + 1} 名`, `ranked #${st.chosenRank + 1}`);
      const end = st.chosenS === '<|im_end|>' ? tr(' 它选中了结束标记，这条回复到此为止。', ' It picked the end marker, so the reply stops here.') : '';
      if (!s.sub) return tr(`采样：温度 0.7 → 只留前 20 名 → 累计概率凑够 80% 就截断，剩 <b>${pool}</b> 个候选。随机数 u = <b>${st.u.toFixed(3)}</b> 落在 ${nxt} 的区间里（${rank}）。${end}`,
        `Sampling: temperature 0.7 → keep only the top 20 → cut off once the running total reaches 80%, leaving <b>${pool}</b> candidate${pool === 1 ? '' : 's'}. The random number u = <b>${st.u.toFixed(3)}</b> lands in the slot of ${nxt} (${rank}).${end}`);
      if (s.sub === 'temp') return tr(`温度 <b>0.7</b>：所有分数除以 0.7，高分和低分拉得更开，分布变得更“尖”。`, `Temperature <b>0.7</b>: every score is divided by 0.7, which pulls high and low scores further apart and makes the distribution "sharper".`);
      if (s.sub === 'topk') return tr(`top-k：只保留分数最高的 <b>20</b> 个词元，其余全部出局。`, `top-k: keep only the <b>20</b> highest-scoring tokens; everything else is out.`);
      if (s.sub === 'topp') return tr(`top-p：从高到低累加概率，凑够 <b>80%</b> 就停，最后剩下 <b>${pool}</b> 个候选。`, `top-p: add up probabilities from the top down and stop as soon as they reach <b>80%</b>. <b>${pool}</b> candidate${pool === 1 ? ' is' : 's are'} left.`);
      return tr(`掷出随机数 <b>u = ${st.u.toFixed(4)}</b>，它落在 ${nxt} 的区间（${rank}，原始概率 ${P(st.chosenP1)}）。${end}`,
        `Roll a random number: <b>u = ${st.u.toFixed(4)}</b>. It lands in the slot of ${nxt} (${rank}, original probability ${P(st.chosenP1)}).${end}`);
    }
  }
  return '';
}

// 一次乘加 / 比特：和算式板、3D 舞台同一套颜色（输入蓝、权重紫、乘积橙、结果青）
const X = (t) => `<span class="cx">${t}</span>`;
const W = (t) => `<span class="cw">${t}</span>`;
const Y = (t) => `<span class="cy">${t}</span>`;
const PR = (t) => `<span class="cp">${t}</span>`;
const sg = (v, d = 3) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`;
const LOADING = `<span class="dimmed">${tr('正在载入这一层的乘加数据…', 'Loading the multiply-add data for this layer…')}</span>`;

function bitsExplain(what) {
  return tr(`<b>比特</b>：${what}在显存里就是 16 个 0/1（bfloat16）：1 位符号、8 位指数、7 位尾数，值 = (−1)<sup>s</sup> × 2<sup>e−127</sup> × (1 + m/128)。点算式板或舞台上的任意一位把它翻过来：别的数都没变，所以结果可以精确地重算出来。<span class="dimmed">翻指数位，数会成倍地变；翻尾数位，只改一点点。</span>`,
    `<b>Bits</b>: in memory, ${what} is just 16 zeros and ones (bfloat16): 1 sign bit, 8 exponent bits and 7 mantissa bits, with value = (−1)<sup>s</sup> × 2<sup>e−127</sup> × (1 + m/128). Click any bit on the board or on the stage to flip it. Nothing else changes, so the result can be recomputed exactly.<span class="dimmed"> Flip an exponent bit and the number jumps by powers of two; flip a mantissa bit and it changes only a little.</span>`);
}

// 矩阵乘法里的一个输出元素：y[j] = Σ x[i] · W[i, j]
function mmExplain(s, e, { y, x, w, wi, n, tail = '' }) {
  if (!e) return LOADING;
  const sm = sums(e, n), N = sm.n;
  const yj = Y(`${y}[${e.j}]`);
  if (s.mi === 'pick') return tr(`放大 ${W(w)} 的<b>第 ${e.j} 列</b>：${yj} = Σ ${X(`${x}[i]`)} × ${W(wi || `${w}[i, ${e.j}]`)}，输入的 ${N} 个数和这一列的 ${N} 个权重一一配对、相乘、再全部加起来。${tail}`,
    `Zoom in on <b>column ${e.j}</b> of ${W(w)}: ${yj} = Σ ${X(`${x}[i]`)} × ${W(wi || `${w}[i, ${e.j}]`)}. The ${N} input numbers are paired one-to-one with the ${N} weights in this column, multiplied, and all added up.${tail}`);
  if (s.mi === 'mul') return tr(`逐项相乘：舞台上${X('蓝点')}沿第 i 行走到这一列（${W('紫色')}格子），乘出来的积（${PR('橙色')}）顺着这一列飞进输出格（${Y('青色')}）。算式板按贡献从大到小列出前 12 项。<span class="dimmed">点算式板的一行（或舞台上的格子），再按 ＋ 看那个权重的 16 个比特。</span>`,
    `Multiply term by term: on the stage, a ${X('blue dot')} walks along row i to this column (the ${W('purple')} cell), and the product (${PR('orange')}) flies up the column into the output cell (${Y('cyan')}). The board lists the top 12 terms, largest contribution first.<span class="dimmed"> Click a row on the board (or a cell on the stage), then press ＋ to see that weight's 16 bits.</span>`);
  const pn = sm.pos != null && sm.neg != null ? tr(`所有正的乘积加起来 ${sg(sm.pos, 2)}，负的 ${sg(sm.neg, 2)}，大部分相互抵消了。`, ` All the positive products add up to ${sg(sm.pos, 2)} and the negative ones to ${sg(sm.neg, 2)}: most of it cancels out.`) : '';
  return tr(`${N} 项全部加起来：${yj} = <b>${sg(e.total)}</b>。前 12 项合计 ${sg(e.shown)}，其余 ${N - 12} 项合计 ${sg(e.total - e.shown)}：单项都很小，但合起来不能忽略。${pn}`,
    `All ${N} terms added up: ${yj} = <b>${sg(e.total)}</b>. The top 12 terms give ${sg(e.shown)} and the other ${N - 12} give ${sg(e.total - e.shown)}: each one is tiny, but together they can't be ignored.${pn}`);
}

// D6 / D7 的每一个微观步骤
function microExplain(s, Q, ctx, L, g) {
  const bits = ctx.view === 'bits';
  if (s.sub === 'score') {
    const d = Q.dotAt(L, g);
    if (!d) return LOADING;
    const k = T(Q, d.key);
    if (bits) return bitsExplain(tr(`第 ${d.head} 号头的 q 里被选中的那一维（激活值也是 bf16）`, `the selected dimension of head ${d.head}'s q (activations are bf16 too)`));
    if (s.mi === 'mul') return tr(`第 <b>${d.head}</b> 号头（用第 ${d.kv} 组键值头）拿当前词元的 ${X('q')} 和 ${k} 的 ${W('k')}，在 128 个维度上逐一相乘。算式板列出乘积最大的 12 维。`,
      `Head <b>${d.head}</b> (which uses KV group ${d.kv}) takes the current token's ${X('q')} and the ${W('k')} of ${k}, and multiplies them across all 128 dimensions. The board lists the 12 largest products.`);
    if (s.mi === 'sum') { const sm = sums(d, 128); return tr(`128 项全部加起来：${Y('q·k')} = <b>${sg(d.sum, 2)}</b>。${sm.pos != null ? `正的乘积合计 ${sg(sm.pos, 2)}，负的 ${sg(sm.neg, 2)}。` : ''}两个向量越“同向”，点积越大。`,
      `All 128 terms added up: ${Y('q·k')} = <b>${sg(d.sum, 2)}</b>. ${sm.pos != null ? `The positive products total ${sg(sm.pos, 2)}, the negative ones ${sg(sm.neg, 2)}. ` : ''}The more two vectors point the same way, the bigger their dot product.`); }
    return tr(`除以 √128 ≈ 11.31 得到打分 <b>${sg(d.score)}</b>；和前面所有词元的打分一起做 softmax 之后，这个头给 ${k} 的权重是 ${P(d.w)}。`,
      `Divide by √128 ≈ 11.31 to get the score <b>${sg(d.score)}</b>. After a softmax together with the scores of all earlier tokens, this head gives ${k} a weight of ${P(d.w)}.`);
  }
  if (s.sub === 'act') {
    const n = Q.neuronAt(L, g);
    if (!n) return LOADING;
    const id = neuronId(n);
    if (s.mi === 'silu') return tr(`${X('g')} = ${sg(n.gz)} 先过 <b>SiLU</b>：SiLU(g) = g × σ(g) = <b>${sg(n.silu)}</b>。σ 是 0 到 1 之间的 S 形曲线，所以负的 g 被压到接近 0，正的 g 几乎原样通过。`,
      `${X('g')} = ${sg(n.gz)} first goes through <b>SiLU</b>: SiLU(g) = g × σ(g) = <b>${sg(n.silu)}</b>. σ is an S-shaped curve between 0 and 1, so a negative g gets squashed to nearly 0 while a positive g passes through almost unchanged.`);
    return tr(`再乘上 ${W('u')} = ${sg(n.uz)}：${Y(`a[${id}]`)} = SiLU(g) × u = <b>${sg(n.silu * n.uz)}</b>，这就是神经元 #${id} 的输出。SiLU(g) 像一道<b>阀门</b>，决定放多少 u 通过。`,
      `Then multiply by ${W('u')} = ${sg(n.uz)}: ${Y(`a[${id}]`)} = SiLU(g) × u = <b>${sg(n.silu * n.uz)}</b>. That's the output of neuron #${id}. SiLU(g) works like a <b>valve</b> that decides how much of u gets through.`);
  }
  if (s.sub === 'qkv') {
    const e = Q.mmAt(L, g)?.q;
    if (!e) return LOADING;
    if (bits) return bitsExplain(tr(`W<sub>q</sub> 里被选中的那个权重`, `the selected weight in W<sub>q</sub>`));
    if (s.mi === 'rope') return tr(`${Y(`q[${e.j}]`)} 还要再加工两次：先做 <b>q_norm</b>（第 ${e.head} 号头的 128 个数一起除以均方根 ${e.rms.toFixed(3)}，再乘缩放 γ），得到 ${sg(e.qn, 4)}；再做 <b>RoPE</b>：和第 ${e.partner} 维配成一对，按位置 ${ropePos(e)} 转一个角度，得到 <b>${sg(e.qr, 4)}</b>。位置越靠后转得越多，两个词元的点积因此只和它们的相对距离有关。`,
      `${Y(`q[${e.j}]`)} gets two more touches. First <b>q_norm</b>: all 128 numbers of head ${e.head} are divided by their root mean square ${e.rms.toFixed(3)} and multiplied by a scale γ, giving ${sg(e.qn, 4)}. Then <b>RoPE</b>: paired with dimension ${e.partner}, it is rotated by an angle set by its position ${ropePos(e)}, giving <b>${sg(e.qr, 4)}</b>. Later positions rotate further, so the dot product of two tokens depends only on how far apart they are.`);
    return mmExplain(s, e, { y: 'q', x: 'h', w: 'W<sub>q</sub>', n: 1024, tail: tr(`q 一共 2048 个数（16 个头 × 128），这里是第 ${e.head} 号头的第 ${e.dim} 维。k、v 也完全一样，只是换成 W<sub>k</sub>、W<sub>v</sub>。`,
      ` q has 2048 numbers in all (16 heads × 128); this is dimension ${e.dim} of head ${e.head}. k and v work exactly the same way, just with W<sub>k</sub> and W<sub>v</sub>.`) });
  }
  if (s.sub === 'mix') {
    const e = Q.mmAt(L, g)?.o;
    if (bits && e) return bitsExplain(tr(`W<sub>o</sub> 里被选中的那个权重`, `the selected weight in W<sub>o</sub>`));
    return mmExplain(s, e, { y: 'Δx', x: 'z', w: 'W<sub>o</sub>', n: 2048, tail: tr(`16 个头各自加权求和的结果拼成 2048 个数（${X('z')}），乘 W<sub>o</sub> 得到加回残差流的 1024 个数之一。`,
      ` The weighted sums of the 16 heads are joined into 2048 numbers (${X('z')}); multiplying by W<sub>o</sub> gives one of the 1024 numbers that get added back into the residual stream.`) });
  }
  if (s.sub === 'down') {
    const e = Q.mmAt(L, g)?.down;
    if (bits && e) return bitsExplain(tr(`W<sub>down</sub> 里被选中的那个权重`, `the selected weight in W<sub>down</sub>`));
    return mmExplain(s, e, { y: 'Δx', x: 'a', w: 'W<sub>down</sub>', n: 3072, tail: tr(`3072 个神经元的输出（${X('a')}）乘 W<sub>down</sub>，得到加回残差流的 1024 个数之一。`,
      ` The outputs of the 3072 neurons (${X('a')}) are multiplied by W<sub>down</sub>, giving one of the 1024 numbers that get added back into the residual stream.`) });
  }
  if (s.sub === 'up') {
    const n = Q.neuronAt(L, g);
    if (!n) return LOADING;
    if (bits) return bitsExplain(tr(`W<sub>gate</sub> 里被选中的那个权重`, `the selected weight in W<sub>gate</sub>`));
    const id = neuronId(n);
    return mmExplain(s, neuronMM(n).g, { y: 'g', x: 'h', w: 'W<sub>gate</sub>', n: 1024, tail: tr(`同一个神经元 #${id} 在 W<sub>up</sub> 里也有一列，同样乘加得到 ${W(`u[${id}]`)} = <b>${sg(n.uz)}</b>。`,
      ` The same neuron #${id} also has a column in W<sub>up</sub>; the same multiply-add gives ${W(`u[${id}]`)} = <b>${sg(n.uz)}</b>.`) });
  }
  return '';
}

function layerExplain(s, Q, ctx, i, cur) {
  const { L, g } = s;
  const lens = Q.lensAt(g, L);
  const guess = `${tokHTML(lens.top[0][2], 'r-assistant')} ${P(lens.top[0][1])}`;
  const mean = Q.attMean(L, i, 2);
  const tgt = mean[0] ? `${T(Q, mean[0].j)}${tr(`（${fmtPct(mean[0].w)}）`, ` (${fmtPct(mean[0].w)})`)}` : '';
  if (!s.op) return tr(`第 <b>${L}</b> 层：${cur} 平均把最多的注意力给了 ${tgt}。如果模型在这一层就停下直接开口，它会说 ${guess}（逻辑透镜）。`,
    `Layer <b>${L}</b>: averaged over heads, ${cur} pays the most attention to ${tgt}. If the model stopped at this layer and spoke right away, it would say ${guess} (logit lens).`);
  if (s.op === 'ln1') return tr(`<b>RMSNorm</b>：把向量除以它的均方根，再乘一组可学习的缩放系数。不减均值、没有偏置，比 LayerNorm 更省。`, `<b>RMSNorm</b>: divide the vector by its root mean square, then multiply by a set of learned scale factors. No mean subtraction and no bias, so it's cheaper than LayerNorm.`);
  if (s.op === 'ln2') return tr(`再做一次 <b>RMSNorm</b>，准备进入前馈网络。`, `Another <b>RMSNorm</b>, getting ready for the feed-forward network.`);
  if (s.op === 'add1') return tr(`<b>残差相加</b>：注意力的输出（经过 o_proj）直接加回原来的向量。每层只是“往上加一点”，信息不会被覆盖。当前向量长度 ‖x‖ = <b>${Q.norm(L, i).toFixed(1)}</b>。`,
    `<b>Residual add</b>: the attention output (after o_proj) is added straight back onto the original vector. Each layer only "adds a little on top", so nothing gets overwritten. Current vector length ‖x‖ = <b>${Q.norm(L, i).toFixed(1)}</b>.`);
  if (s.op === 'add2') return tr(`前馈网络的输出加回残差流，第 ${L} 层结束。逻辑透镜此刻读到的是 ${guess}。`, `The feed-forward output is added back into the residual stream, and layer ${L} is done. Right now the logit lens reads ${guess}.`);
  if (s.mi) return microExplain(s, Q, ctx, L, g);
  const h = ctx.head ?? interestingHead(Q, L, i);
  if (s.op === 'attn') {
    if (!s.sub) return tr(`<b>分组查询注意力（GQA）</b>：16 个查询头，每 2 个共用一组键值头，共 8 组。${cur} 平均最关注 ${tgt}。`, `<b>Grouped-query attention (GQA)</b>: 16 query heads, with every 2 sharing one key-value head, 8 groups in all. Averaged over heads, ${cur} attends most to ${tgt}.`);
    const row = Q.att(L, h, i);
    const best = row.filter((r) => r.j !== i)[0] || row[0];
    if (s.sub === 'qkv') return tr(`新向量分别乘 W<sub>q</sub>、W<sub>k</sub>、W<sub>v</sub>：得到 <b>16 × 128</b> 维的 Q，和各 <b>8 × 128</b> 维的 K、V。Q、K 先各自做 RMSNorm（Qwen3 新加的），再按位置旋转（<b>RoPE</b>，θ = 10⁶）。K、V 存进缓存。`,
      `The new vector is multiplied by W<sub>q</sub>, W<sub>k</sub> and W<sub>v</sub>, giving a <b>16 × 128</b> Q and a K and V of <b>8 × 128</b> each. Q and K each get their own RMSNorm first (new in Qwen3), then are rotated according to position (<b>RoPE</b>, θ = 10⁶). K and V go into the cache.`);
    if (s.sub === 'score') return tr(`第 <b>${h}</b> 头用自己的 Q 和缓存里每个词元的 K 做点积，再除以 √128。它打分最高的是 ${best ? T(Q, best.j) : ''}：<b>${best ? best.s.toFixed(2) : ''}</b>。`,
      `Head <b>${h}</b> takes the dot product of its Q with the K of every token in the cache, then divides by √128. Its highest score goes to ${best ? T(Q, best.j) : ''}: <b>${best ? best.s.toFixed(2) : ''}</b>.`);
    if (s.sub === 'softmax') return tr(`softmax 把分数变成加起来等于 1 的权重。第 ${h} 头给 ${best ? T(Q, best.j) : ''} ${best ? P(best.w) : ''}。后面的词元被因果遮罩挡住，永远看不见。`,
      `Softmax turns the scores into weights that add up to 1. Head ${h} gives ${best ? T(Q, best.j) : ''} ${best ? P(best.w) : ''}. Later tokens are hidden behind the causal mask and can never be seen.`);
    return tr(`按权重把各个词元的 V 加起来。16 个头的结果拼成 2048 维，再乘 W<sub>o</sub> 回到 1024 维。`, `Add up the tokens' V vectors, weighted by attention. The results of the 16 heads are joined into 2048 numbers, then multiplied by W<sub>o</sub> to get back to 1024.`);
  }
  // mlp
  const count = Q.mlpCount(g, L);
  if (!s.sub) return tr(`<b>SwiGLU 前馈网络</b>：先扩到 3072 维，经过门控激活，再压回 1024 维。这一步有 <b>${count}</b> 个神经元明显激活。`, `<b>SwiGLU feed-forward network</b>: expand to 3072 dimensions, apply a gated activation, then squeeze back to 1024. At this step <b>${count}</b> neurons are clearly active.`);
  if (s.sub === 'up') return tr(`gate_proj 和 up_proj 两个矩阵，同时把 1024 维扩到 <b>3072</b> 维，得到两组数：<b>g</b> 和 <b>u</b>。`, `Two matrices, gate_proj and up_proj, both expand the 1024 dimensions to <b>3072</b>, giving two sets of numbers: <b>g</b> and <b>u</b>.`);
  if (s.sub === 'act') return tr(`g 经过 <b>SiLU</b> 激活，再和 u 逐个相乘。g 像一道<b>阀门</b>，决定每个神经元放多少 u 通过。明显激活的有 <b>${count}</b> 个。`, `g goes through the <b>SiLU</b> activation, then is multiplied element by element with u. g works like a <b>valve</b> that decides how much of u each neuron lets through. <b>${count}</b> are clearly active.`);
  return tr(`down_proj 把 3072 维压回 1024 维，准备加回残差流。`, `down_proj squeezes the 3072 dimensions back down to 1024, ready to be added back into the residual stream.`);
}

// 这一步的矩阵形状（预填充时 n = 提示长度，之后每步 n = 1）
const D = (s) => `<b>${s}</b>`;
const ONE = tr('1 个数', '1 number');
export function shapeOf(s, Q, ctx = {}) {
  if (s.mi && ctx.view === 'bits') return tr(`值 = (−1)<sup>s</sup> × 2<sup>e−127</sup> × (1 + m/128)<br><span class="dimmed">bfloat16：1 位符号 + 8 位指数 + 7 位尾数 = 16 位</span>`, `value = (−1)<sup>s</sup> × 2<sup>e−127</sup> × (1 + m/128)<br><span class="dimmed">bfloat16: 1 sign bit + 8 exponent bits + 7 mantissa bits = 16 bits</span>`);
  const M = Q.M, n = s.g === 0 ? Q.P : 1, L = Q.P + s.g;
  const H = M.hidden, qd = M.heads * M.headDim, kd = M.kvHeads * M.headDim, F = M.ffn, V = M.vocab;
  const x = `[${n}×${H}]`;
  const heads = (k) => tr(`${k} 头`, `${k} heads`);
  switch (s.ph) {
    case 'read': return s.g === 0 ? tr(`ids ${D(`[${Q.P}]`)}：${Q.P} 个词元编号`, `ids ${D(`[${Q.P}]`)}: ${Q.P} token IDs`) : tr(`ids ${D(`[${L}]`)}：末尾接上 1 个`, `ids ${D(`[${L}]`)}: one more appended`);
    case 'embed': return tr(`one-hot ${D(`[${n}×${V}]`)} @ E ${D(`[${V}×${H}]`)} → x ${D(x)}<br><span class="dimmed">实际实现是按编号直接取出 E 的第 id 行</span>`, `one-hot ${D(`[${n}×${V}]`)} @ E ${D(`[${V}×${H}]`)} → x ${D(x)}<br><span class="dimmed">in practice, row id of E is simply looked up</span>`);
    case 'layers': case 'pass': return tr(`x ${D(x)} → 28 × Transformer 块 → ${D(x)}`, `x ${D(x)} → 28 × Transformer block → ${D(x)}`);
    case 'head':
      if (s.sub === 'norm') return `x[-1] ${D(`[1×${H}]`)} ÷ RMS × γ ${D(`[${H}]`)}`;
      if (s.sub === 'softmax') return tr(`softmax(logits ${D(`[1×${V}]`)}) → p ${D(`[1×${V}]`)}，和为 1`, `softmax(logits ${D(`[1×${V}]`)}) → p ${D(`[1×${V}]`)}, sums to 1`);
      if (s.mi) return `${Y('logit[id]')} = ${X('h')} ${D(`[${H}]`)} · ${W('E[id]')} ${D(`[${H}]`)} → ${ONE}`;
      return `x[-1] ${D(`[1×${H}]`)} @ Eᵀ ${D(`[${H}×${V}]`)} → logits ${D(`[1×${V}]`)}`;
    case 'sample': return tr(`p ${D(`[${V}]`)} → ÷0.7 → 前 20 → 累计 80% → 候选 ${D(`[${Q.steps[s.g].pool.length}]`)} → 1 个词元`, `p ${D(`[${V}]`)} → ÷0.7 → top 20 → running total 80% → candidates ${D(`[${Q.steps[s.g].pool.length}]`)} → 1 token`);
    case 'layer': {
      if (!s.op) return tr(`x ${D(x)} → 注意力 → 前馈 → ${D(x)}`, `x ${D(x)} → attention → feed-forward → ${D(x)}`);
      if (s.op === 'ln1' || s.op === 'ln2') return `h = x ${D(x)} ÷ RMS(x) × γ ${D(`[${H}]`)} → ${D(x)}`;
      if (s.op === 'add1' || s.op === 'add2') return `x ${D(x)} + Δx ${D(x)} → ${D(x)}`;
      if (s.mi && s.sub === 'qkv') return s.mi === 'rope' ? tr(`q<sub>头</sub> ${D(`[${M.headDim}]`)} ÷ RMS × γ → 每两维一对旋转 θ = 位置 / 10⁶<sup>2k/128</sup>`, `q<sub>head</sub> ${D(`[${M.headDim}]`)} ÷ RMS × γ → rotate each pair of dims by θ = position / 10⁶<sup>2k/128</sup>`) : `${Y('q[j]')} = ${X('h')} ${D(`[${H}]`)} · ${W('W<sub>q</sub>[:, j]')} ${D(`[${H}]`)} → ${ONE}`;
      if (s.mi && s.sub === 'mix') return `${Y('Δx[j]')} = ${X('z')} ${D(`[${qd}]`)} · ${W('W<sub>o</sub>[:, j]')} ${D(`[${qd}]`)} → ${ONE}<br><span class="dimmed">${tr('z = 16 个头的输出拼在一起', 'z = the outputs of the 16 heads, joined together')}</span>`;
      if (s.mi && s.sub === 'down') return `${Y('Δx[j]')} = ${X('a')} ${D(`[${F}]`)} · ${W('W<sub>down</sub>[:, j]')} ${D(`[${F}]`)} → ${ONE}`;
      if (s.mi && s.sub === 'up') return `${Y('g[n]')} = ${X('h')} ${D(`[${H}]`)} · ${W('W<sub>gate</sub>[:, n]')} ${D(`[${H}]`)}${tr('；u[n] 同理用 W<sub>up</sub>', '; u[n] likewise with W<sub>up</sub>')}`;
      if (s.op === 'attn') {
        if (!s.sub || s.sub === 'qkv') return `h ${D(x)} @ W<sub>q</sub> ${D(`[${H}×${qd}]`)} → q ${D(`[${n}×${qd}]`)} = ${heads(M.heads)} × ${M.headDim}<br>h ${D(x)} @ W<sub>k</sub> ${D(`[${H}×${kd}]`)} → k = ${heads(M.kvHeads)} × ${M.headDim}<br>h ${D(x)} @ W<sub>v</sub> ${D(`[${H}×${kd}]`)} → v = ${heads(M.kvHeads)} × ${M.headDim}`;
        if (s.sub === 'score') {
          if (s.mi) return `${Y('q·k')} = ${X('q')} ${D(`[${M.headDim}]`)} · ${W('k')} ${D(`[${M.headDim}]`)} → ${tr(`1 个数，再 ÷ √${M.headDim}`, `1 number, then ÷ √${M.headDim}`)}`;
          return `q ${D(`[${M.heads}×${n}×${M.headDim}]`)} @ Kᵀ ${D(`[${M.heads}×${M.headDim}×${L}]`)} → s ${D(`[${M.heads}×${n}×${L}]`)}<br><span class="dimmed">${tr(`K 只有 ${M.kvHeads} 头，每个复用给 2 个 Q 头`, `K has only ${M.kvHeads} heads, each one reused by 2 Q heads`)}</span>`;
        }
        if (s.sub === 'softmax') return tr(`softmax(s ${D(`[${M.heads}×${n}×${L}]`)} + 遮罩) → a，每行和为 1`, `softmax(s ${D(`[${M.heads}×${n}×${L}]`)} + mask) → a, each row sums to 1`);
        return `a ${D(`[${M.heads}×${n}×${L}]`)} @ V ${D(`[${M.heads}×${L}×${M.headDim}]`)} → ${D(`[${M.heads}×${n}×${M.headDim}]`)}<br>${tr('拼接', 'concat')} ${D(`[${n}×${qd}]`)} @ W<sub>o</sub> ${D(`[${qd}×${H}]`)} → Δx ${D(x)}`;
      }
      if (s.op === 'mlp') {
        if (s.mi) return `${Y('a[n]')} = silu(${X('g[n]')}) × ${W('u[n]')} → ${ONE}`;
        if (!s.sub || s.sub === 'up') return `h ${D(x)} @ W<sub>gate</sub> ${D(`[${H}×${F}]`)} → g ${D(`[${n}×${F}]`)}<br>h ${D(x)} @ W<sub>up</sub> ${D(`[${H}×${F}]`)} → u ${D(`[${n}×${F}]`)}`;
        if (s.sub === 'act') return tr(`silu(g) ⊙ u：${D(`[${n}×${F}]`)} 逐个相乘`, `silu(g) ⊙ u: ${D(`[${n}×${F}]`)} element-wise`);
        return `${D(`[${n}×${F}]`)} @ W<sub>down</sub> ${D(`[${F}×${H}]`)} → Δx ${D(x)}`;
      }
    }
  }
  return '';
}

export function watch(s, Q, ctx = {}) {
  const g = s.g, i = Q.row(g), st = Q.steps[g];
  const rows = [
    ['wh', tr('变量', 'Variables')],
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
    if (s.mi && s.op === 'mlp') { const n = Q.neuronAt(s.L, g); if (n) rows.push(['neuron', `#${neuronId(n)}`], ['g', n.gz.toFixed(4)], ['u', n.uz.toFixed(4)], ['silu(g)*u', (n.silu * n.uz).toFixed(4)]); }
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

const OP_NAME = isEn
  ? { ln1: 'RMSNorm', attn: 'Attention', add1: 'Residual ⊕', ln2: 'RMSNorm', mlp: 'SwiGLU feed-forward', add2: 'Residual ⊕' }
  : { ln1: 'RMSNorm', attn: '注意力', add1: '残差 ⊕', ln2: 'RMSNorm', mlp: '前馈 SwiGLU', add2: '残差 ⊕' };
const SUB_NAME = isEn
  ? { qkv: 'Q·K·V', score: 'Scores', softmax: 'softmax', mix: 'Weighted sum', up: 'Expand', act: 'Gated activation', down: 'Project down', norm: 'RMSNorm', unembed: 'Output matrix', temp: 'Temperature', topk: 'top-k', topp: 'top-p', draw: 'Roll the dice' }
  : { qkv: 'Q·K·V', score: '打分', softmax: 'softmax', mix: '加权求和', up: '升维', act: '门控激活', down: '降维', norm: 'RMSNorm', unembed: '输出矩阵', temp: '温度', topk: 'top-k', topp: 'top-p', draw: '掷骰子' };
const MI_NAME = isEn
  ? { mul: 'Multiply', sum: 'Sum', scale: '÷√128', silu: 'SiLU', gate: '× u', pick: 'Pick a column', rope: 'q_norm + RoPE' }
  : { mul: '逐项相乘', sum: '求和', scale: '÷√128', silu: 'SiLU', gate: '× u', pick: '选一列', rope: 'q_norm + RoPE' };

export function stepLabel(s) {
  switch (s.ph) {
    case 'pass': return tr(`第 ${s.g + 1} 个词元`, `Token ${s.g + 1}`);
    case 'read': return s.g === 0 ? tr('套聊天模板', 'Apply chat template') : tr('接上新词元', 'Append new token');
    case 'embed': return tr('嵌入', 'Embedding');
    case 'layers': return tr('28 层', '28 layers');
    case 'head': return s.mi ? `${tr('输出', 'Output')} · ${SUB_NAME[s.sub]} · ${MI_NAME[s.mi]}` : s.sub ? `${tr('输出', 'Output')} · ${SUB_NAME[s.sub]}` : tr('输出头', 'Output head');
    case 'sample': return s.sub ? `${tr('采样', 'Sampling')} · ${SUB_NAME[s.sub]}` : tr('采样', 'Sampling');
    case 'layer': return [tr(`第 ${s.L} 层`, `Layer ${s.L}`), s.op && OP_NAME[s.op], s.sub && SUB_NAME[s.sub], s.mi && MI_NAME[s.mi]].filter(Boolean).join(' · ');
  }
  return '';
}

// 面包屑：每一级是 {d, label}
export function crumbs(depth, s) {
  const out = [{ d: 0, label: DEPTH_NAMES[0] }, { d: 1, label: DEPTH_NAMES[1] }];
  if (depth >= 2) out.push({ d: 2, label: DEPTH_NAMES[2] });
  if (depth >= 3) {
    if (s.ph === 'layer') out.push({ d: 3, label: DEPTH_NAMES[3] });
    else out.push({ d: 3, label: (isEn ? { read: 'Read input', embed: 'Embedding', head: 'Output head', sample: 'Sampling' } : { read: '读入', embed: '嵌入', head: '输出头', sample: '采样' })[s.ph] || DEPTH_NAMES[3] });
  }
  if ((s.ph === 'head' || s.ph === 'sample') && depth >= 4 && s.sub) out.push({ d: 4, label: SUB_NAME[s.sub] });
  if (s.ph === 'head' && depth >= 5 && s.mi) out.push({ d: 5, label: DEPTH_NAMES[6] });
  if (s.ph === 'head' && depth >= 6 && s.mi === 'mul') out.push({ d: 6, label: DEPTH_NAMES[7] });
  if (depth >= 4 && s.ph === 'layer') out.push({ d: 4, label: tr(`第 ${s.L} 层`, `Layer ${s.L}`) });
  if (depth >= 5 && s.op) out.push({ d: 5, label: OP_NAME[s.op] });
  if (depth >= 6 && s.sub && s.ph === 'layer') out.push({ d: 6, label: SUB_NAME[s.sub] });
  if (depth >= 7 && s.mi === 'mul' && s.ph === 'layer') out.push({ d: 7, label: DEPTH_NAMES[7] });
  return out;
}
