// 玻璃小模型这一章的调试器内容：代码（伪代码，高亮当前行）、这一步的白话讲解 + 真实数字、变量监视。
// 接口和 ../explain.js 一样：codeFor / linesFor / explain / watch / stepLabel / crumbs / shapeOf。
import { esc } from '../../../js/ui.js';
import { fmtP, sciSup, fmtInt } from '../draw.js';
import { isEn, L } from '../lang.js';
import { TENSOR_LABEL } from './data.js';
import { OP_NAME } from './scenes/step.js';
import { paramName } from './heat.js';
import { adamAt, chainTerms, defaultParam } from './scenes/param.js';

const K_ = (s) => `<span class="kw">${s}</span>`;
const F_ = (s) => `<span class="fn">${s}</span>`;
const C_ = (s) => `<span class="cm"># ${s}</span>`;
const N_ = (s) => `<span class="nu">${s}</span>`;
const P = (p) => `<em>${fmtP(p)}</em>`;
const f4 = (v) => (v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e4 ? Number(v.toPrecision(4)).toString() : sciSup(v, 3));
const ch = (D, id) => esc(D.ch(id));

export function codeFor(R) {
  return [
    `W = ${F_('init')}(std=${N_('0.02')})   ${C_(L('2,928 个参数，γ = 1', '2,928 params, γ = 1'))}`,
    `opt = ${F_('AdamW')}(lr=${N_('1e-2')}, wd=${N_('0.1')})`,
    `${K_('for')} step ${K_('in')} ${F_('range')}(${N_(200)}):`,
    `    x = ${F_('batch')}()         ${C_(L('8 段 × 9 个字', '8 windows × 9 chars'))}`,
    `    inp, tgt = x[:, :${N_('-1')}], x[:, ${N_('1')}:]`,
    `    h = E[inp]             ${C_(L('查嵌入表', 'embedding lookup'))}`,
    `    h = h + ${F_('attn')}(${F_('rms')}(h))   ${C_(L('2 个头 + RoPE', '2 heads + RoPE'))}`,
    `    h = h + ${F_('ffn')}(${F_('rms')}(h))    ${C_('SwiGLU 16→32→16')}`,
    `    logits = ${F_('rms')}(h) @ E.T  ${C_(L('共用 E', 'shared E'))}`,
    `    loss = ${F_('cross_entropy')}(logits, tgt)`,
    `    loss.${F_('backward')}()       ${C_(L('链式法则，从后往前', 'chain rule, back to front'))}`,
    `    ${F_('clip_grad_norm_')}(W, ${N_('1.0')})`,
    `    ${K_('for')} w, g ${K_('in')} W:          ${C_('AdamW')}`,
    `        m = ${N_('.9')}m + ${N_('.1')}g;  v = ${N_('.95')}v + ${N_('.05')}g²`,
    `        w −= lr·(m̂/(√v̂+ε) + λ·w)`,
  ];
}

export function linesFor(s) {
  switch (s.ph) {
    case 'init': return [1];
    case 'run': return [3];
    case 'batch': return s.sub === 'pick' ? [4] : s.sub === 'shift' ? [5] : [4, 5];
    case 'fwd': return !s.sub ? [6, 7, 8, 9] : { emb: [6], norm1: [7], qkv: [7], attn: [7], wo: [7], norm2: [8], ffn: [8], wd: [8], normf: [9], logits: [9] }[s.sub];
    case 'loss': return [10];
    case 'bwd': return [11];
    case 'upd': return !s.sub ? [12, 13, 14, 15] : s.sub === 'clip' ? [12] : !s.mi ? [13, 14, 15] : { g: [13], m: [14], v: [14], bc: [15], dw: [15], write: [15] }[s.mi];
  }
  return [];
}

/* ---------------------------------------------------------------- 名字 */

const PH = isEn ? { batch: 'Batch', fwd: 'Forward', loss: 'Loss', bwd: 'Backward', upd: 'Update' } : { batch: '取批次', fwd: '前向', loss: '损失', bwd: '反向', upd: '更新' };
const INIT = isEn ? { model: 'The whole model', hist: 'Initial values', zero: 'Control: all zeros', big: 'Control: 50× larger' } : { model: '整个模型', hist: '初始值的分布', zero: '对照：全零', big: '对照：放大 50 倍' };
const ADAM = isEn ? { g: 'gradient g', m: 'first moment m', v: 'second moment v', bc: 'bias correction', dw: 'compute Δw', write: 'write back' } : { g: '梯度 g', m: '一阶动量 m', v: '二阶动量 v', bc: '偏差校正', dw: '算出 Δw', write: '写回' };

function selOf(D, ctx, s) {
  if (!s.t) return null;
  const gs = ctx.gsel;
  return gs != null && D.locate(gs).p.name === s.t ? gs : defaultParam(D, s.t);
}

export function stepLabel(s, R, depth, ctx) {
  const D = R.D;
  if (s.ph === 'init') return INIT[s.sub];
  if (s.ph === 'run') return L(`第 ${D.FR[s.k] + 1} 步`, `Step ${D.FR[s.k] + 1}`);
  if (!s.sub) return PH[s.ph];
  switch (s.ph) {
    case 'batch': return `${PH.batch} · ${s.sub === 'pick' ? L('取 8 段', 'pick 8 windows') : L('错开一位', 'shift by one')}`;
    case 'fwd': return `${PH.fwd} · ${OP_NAME[s.sub]}`;
    case 'loss': return s.sub === 'mean' ? `${PH.loss} · ${L('平均', 'mean')}` : `${PH.loss} · ${L(`位置 ${s.i}`, `position ${s.i}`)}「${ch(D, D.fixed[s.i])}→${ch(D, D.fixed[s.i + 1])}」`;
    case 'bwd': return s.mi ? `${PH.bwd} · ${ctx ? esc(paramName(D, selOf(D, ctx, s))) : TENSOR_LABEL[s.t]} ${L('的梯度', 'gradient')}` : `${PH.bwd} · ${OP_NAME[s.sub]}`;
    case 'upd': return s.sub === 'clip' ? `${PH.upd} · ${L('梯度裁剪', 'gradient clipping')}` : `${PH.upd} · ${TENSOR_LABEL[s.t]}${s.mi ? ` · ${ADAM[s.mi]}` : ''}`;
  }
  return '';
}

export function crumbs(depth, s, R, ctx) {
  const D = R.D;
  const out = [{ d: 0, label: L('初始化', 'Initialization') }];
  if (depth === 0) return out;
  out.push({ d: 1, label: L('训练全程', 'Whole run') });
  if (depth >= 2) out.push({ d: 2, label: L(`第 ${D.FR[s.k] + 1} 步`, `Step ${D.FR[s.k] + 1}`) });
  if (depth >= 3 && s.sub) out.push({ d: 3, label: PH[s.ph] });
  if (depth >= 4 && s.mi) out.push({ d: 4, label: paramName(D, selOf(D, ctx, s)) });
  return out;
}

export function shapeOf(s, R) {
  const D = R.D;
  if (s.ph === 'init' || s.ph === 'run') return '';
  const B = D.B, T = D.T;
  if (s.ph === 'batch') return L(`x <b>[${B} × ${T + 1}]</b> → inp <b>[${B} × ${T}]</b>，tgt <b>[${B} × ${T}]</b>`, `x <b>[${B} × ${T + 1}]</b> → inp <b>[${B} × ${T}]</b>, tgt <b>[${B} × ${T}]</b>`);
  if (s.ph === 'fwd') {
    const sh = { emb: `h₀ <b>[${B} × ${T} × 16]</b>`, norm1: `n₁ <b>[${B} × ${T} × 16]</b>`, qkv: `q, k, v <b>[${B} × ${T} × 2 × 8]</b>`, attn: L(`注意力 <b>[${B} × 2 × ${T} × ${T}]</b>`, `attention <b>[${B} × 2 × ${T} × ${T}]</b>`), wo: `h₁ <b>[${B} × ${T} × 16]</b>`, norm2: `n₂ <b>[${B} × ${T} × 16]</b>`, ffn: `gate, up, act <b>[${B} × ${T} × 32]</b>`, wd: `h₂ <b>[${B} × ${T} × 16]</b>`, normf: `n_f <b>[${B} × ${T} × 16]</b>`, logits: `logits <b>[${B} × ${T} × 20]</b>` };
    return s.sub ? `${sh[s.sub]}${L('；舞台上画第 0 段', '; the stage shows row 0')}` : L(`inp <b>[${B} × ${T}]</b> → … → logits <b>[${B} × ${T} × 20]</b>；舞台上画第 0 段 <b>[${T} × ·]</b>`, `inp <b>[${B} × ${T}]</b> → … → logits <b>[${B} × ${T} × 20]</b>; the stage shows row 0 <b>[${T} × ·]</b>`);
  }
  if (s.ph === 'loss') return L(`logits <b>[${B} × ${T} × 20]</b> → 每个位置 −ln p → 64 个数平均成 1 个`, `logits <b>[${B} × ${T} × 20]</b> → −ln p per position → 64 numbers averaged into 1`);
  if (s.ph === 'bwd') return L(`每个激活、每个参数都得到同形状的梯度：参数的梯度共 <b>${fmtInt(D.P)}</b> 个数`, `every activation and parameter gets a gradient of the same shape: <b>${fmtInt(D.P)}</b> numbers for the parameters`);
  if (s.ph === 'upd') return L(`m、v：和参数同形状，各 <b>${fmtInt(D.P)}</b> 个`, `m, v: same shape as the parameters, <b>${fmtInt(D.P)}</b> each`);
  return '';
}

/* ---------------------------------------------------------------- 讲解 */

function topOf(D, k, i) { let b = 0; for (let v = 1; v < D.V; v++) if (D.probs(k, i, v) > D.probs(k, i, b)) b = v; return b; }

function runNarrative(D, k) {
  const t = D.FR[k], ev = D.evalLoss[t];
  if (!D.has('w', k)) return '';
  const x = D.fixed, pc = (i) => D.probs(k, i, x[i + 1]);
  const ok = Array.from({ length: D.T }, (_, i) => topOf(D, k, i) === x[i + 1]).filter(Boolean).length;
  if (t < 3) return L(`刚开始：每个位置对 20 个字几乎一视同仁（正确答案的概率都在 5% 左右），注意力也平均分给前面每个字。`, `Right at the start every position treats all 20 characters almost alike (about 5% each for the right answer), and attention is spread evenly over the preceding characters.`);
  if (ev > 1.6) return L(`它先学的是“哪些字常出现、某个字后面常跟哪个字”。固定样例 8 个位置里已经有 ${ok} 个第一名猜对。`, `It first learns which characters are common and which character usually follows which. In the fixed sample, ${ok} of 8 positions already have the right answer on top.`);
  if (ev > 0.3) return L(`大部分位置已经猜对（${ok}/8）。难的是只看前一个字分不清的地方：「头」后面是「望」还是「思」（现在 ${P(pc(7))}）、「月」后面是「光」还是「，」（现在 ${P(pc(4))}）——要往回多看一两个字，这正是注意力在学的。`, `Most positions are right (${ok}/8). The hard ones can’t be told apart from the previous character alone: is 头 followed by 望 or 思 (now ${P(pc(7))}), is 月 followed by 光 or ， (now ${P(pc(4))})? It has to look back one or two more characters — exactly what attention is learning.`);
  return L(`几乎全对（${ok}/8）。最后才稳下来的是「月→，」（现在 ${P(pc(4))}）：只有往回看到「望」才知道这是第二个“明月”。全零初始化的那次训练，这时损失还停在 2.996。`, `Almost all right (${ok}/8). The last to settle is 月→， (now ${P(pc(4))}): only by looking back to 望 can it tell this is the second 明月. The all-zero run is still stuck at 2.996.`);
}

export function explain(s, R, ctx, depth) {
  const D = R.D, m = D.meta;
  if (s.ph === 'init') return initExplain(s.sub, D);
  const k = s.k, t = D.FR[k];
  if (s.ph === 'run') {
    const ph = t < m.train.warmup ? L(`预热中（前 ${m.train.warmup} 步从 0 线性升到 10⁻²）`, `warming up (rising linearly to 10⁻² over the first ${m.train.warmup} steps)`) : L('余弦退火中', 'cosine annealing');
    const gap = k < D.NF - 1 ? D.FR[k + 1] - t : 0;
    return L(`第 <b>${t + 1}</b> 步（共 ${D.S} 步）。这一步 8 段的损失 <b>${D.loss[t].toFixed(3)}</b>，全部 25 段平均 <b>${D.evalLoss[t].toFixed(3)}</b>；学习率 ${sciSup(D.lr[t], 3)}，${ph}。<br>${runNarrative(D, k)}<br><span class="dimmed">${gap > 1 ? `下一帧是第 ${D.FR[k + 1] + 1} 步（中间 ${gap - 1} 步只记了损失）。` : ''}点任意一个格子看这个参数的一生；按 ＋ 拆开这一步。</span>`,
      `Step <b>${t + 1}</b> of ${D.S}. Loss on this step’s 8 windows <b>${D.loss[t].toFixed(3)}</b>, mean over all 25 windows <b>${D.evalLoss[t].toFixed(3)}</b>; learning rate ${sciSup(D.lr[t], 3)}, ${ph}.<br>${runNarrative(D, k)}<br><span class="dimmed">${gap > 1 ? `The next frame is step ${D.FR[k + 1] + 1} (only the loss was kept for the ${gap - 1} steps in between). ` : ''}Click any cell to see that parameter’s life; press + to open this step.</span>`);
  }
  if (!s.sub) return phaseExplain(s.ph, D, k);
  switch (s.ph) {
    case 'batch': return batchExplain(s, D, k);
    case 'fwd': return fwdExplain(s.sub, D, k);
    case 'loss': return lossExplain(s, D, k);
    case 'bwd': return s.mi ? chainExplain(D, k, selOf(D, ctx, s)) : bwdExplain(s.sub, D, k);
    case 'upd': return s.sub === 'clip' ? clipExplain(D, k) : s.mi ? adamExplain(s.mi, D, k, selOf(D, ctx, s)) : updTensorExplain(s.t, D, k);
  }
  return '';
}

function initExplain(sub, D) {
  const z = D.meta.runs.zero, b = D.meta.runs.big;
  if (sub === 'model') return L(`这个小模型只有 <b>${fmtInt(D.P)}</b> 个数：一张 20 × 16 的嵌入表 E、注意力的 4 个 16 × 16 矩阵、前馈的 3 个矩阵，外加 3 条 RMSNorm 的缩放 γ。舞台上每个格子就是其中一个数（蓝 = 负，琥珀 = 正，越亮绝对值越大）。<br>它要学的是李白《静夜思》：看前面几个字，猜下一个字。`,
    `This tiny model has just <b>${fmtInt(D.P)}</b> numbers: a 20 × 16 embedding table E, four 16 × 16 attention matrices, three FFN matrices, plus the scales γ of three RMSNorms. Every cell on the stage is one of them (blue = negative, amber = positive, brighter = larger).<br>Its job: Li Bai’s “Quiet Night Thought” (静夜思) — read the previous characters, guess the next one.`);
  if (sub === 'hist') {
    let s = 0, n = 0;
    for (const p of D.params) if (!p.norm) for (let i = p.off; i < p.off + p.n; i++) { s += D.w0[i] ** 2; n++; }
    return L(`训练开始前，每个矩阵元素都从均值 0、标准差 0.02 的正态分布里随机抽一个（种子 0）——所以图上是一片细碎的噪点。实际抽到的 ${fmtInt(n)} 个数标准差 <b>${Math.sqrt(s / n).toPrecision(3)}</b>；48 个 γ 都从 1 开始。`,
      `Before training, every matrix entry is drawn at random from a normal distribution with mean 0 and standard deviation 0.02 (seed 0) — hence the fine-grained noise. The ${fmtInt(n)} numbers actually drawn have std <b>${Math.sqrt(s / n).toPrecision(3)}</b>; all 48 γ start at 1.`);
  }
  if (sub === 'zero') return L(`如果全设成 0：每个位置的隐藏向量都是 0，20 个字的 logits 全是 0，概率都是 1/20，损失 ln 20 = <b>${D.zLoss[0].toFixed(3)}</b>。反向时，误差信号要乘上全零的权重才能往回传，乘出来还是 0；参数的梯度又要乘上全零的激活，也是 0。<br>真实训练 200 步：梯度范数始终 <b>${z.maxGnorm}</b>，参数一个都没动（最大 |w| = ${z.maxAbsW}），损失一直 ${z.finalEval.toFixed(3)}。每个数一模一样，得到的梯度也一模一样——对称性没被打破，就永远分不开。`,
    `Set everything to 0 and every position’s hidden vector is 0, all 20 logits are 0, every probability is 1/20, and the loss is ln 20 = <b>${D.zLoss[0].toFixed(3)}</b>. On the way back the error signal must be multiplied by all-zero weights, which gives 0; each parameter’s gradient is multiplied by all-zero activations, also 0.<br>Real run, 200 steps: the gradient norm stays at <b>${z.maxGnorm}</b>, not one parameter moves (max |w| = ${z.maxAbsW}), and the loss stays ${z.finalEval.toFixed(3)}. Identical numbers get identical gradients — the symmetry is never broken, so they can never become different.`);
  return L(`放大 50 倍（标准差 1）：一开始的 logits 很大，模型对随机的答案非常自信，第 1 步的损失 <b>${b.firstLoss.toFixed(2)}</b>，比瞎猜（3.00）还高得多。同样训练 200 步，最后 <b>${b.finalEval.toFixed(3)}</b>，是正常初始化（${D.meta.train.finalEval.toFixed(3)}）的 ${(b.finalEval / D.meta.train.finalEval).toFixed(0)} 倍。所以初始值要小：让模型从“谁都不偏向”出发。<br><span class="dimmed">按 ＋ 开始训练。</span>`,
    `50× larger (std 1): the initial logits are huge, so the model is very sure of random answers — the first loss is <b>${b.firstLoss.toFixed(2)}</b>, far worse than a blind guess (3.00). After the same 200 steps it ends at <b>${b.finalEval.toFixed(3)}</b>, ${(b.finalEval / D.meta.train.finalEval).toFixed(0)}× the normal init (${D.meta.train.finalEval.toFixed(3)}). That’s why initial values are small: start from “no preference for anything”.<br><span class="dimmed">Press + to start training.</span>`);
}

function phaseExplain(ph, D, k) {
  const t = D.FR[k];
  if (ph === 'batch') return L(`从首尾相接的《静夜思》里取 8 段、每段 9 个字：前 8 个是输入，错开一位的 8 个是答案。第 0 段固定是“举头望明月，低头思”，其余 7 段这一步从第 ${[1, 2, 3, 4, 5, 6, 7].map((b) => D.offs(t, b) + 1).join('、')} 个字开始。一共 64 道“猜下一个字”的题。`,
    `Take 8 windows of 9 characters from the looped poem: the first 8 are the input, the 8 shifted by one are the answers. Row 0 is always 举头望明月，低头思; this step the other 7 start at characters ${[1, 2, 3, 4, 5, 6, 7].map((b) => D.offs(t, b) + 1).join(', ')}. 64 “guess the next character” questions in all.`);
  if (ph === 'fwd') return L(`8 段一起算，舞台上画第 0 段：8 个字 → 查嵌入表 → 注意力 → 前馈 → 20 个字的概率。每个格子都是这一步真实的激活值；左边那条竖线是残差，每一层的输出都是“加”到它上面的。`, `All 8 windows are computed together; the stage shows row 0: 8 characters → embedding lookup → attention → FFN → probabilities over 20 characters. Every cell is a real activation from this step; the vertical line on the left is the residual stream, and each layer’s output is added onto it.`);
  if (ph === 'loss') { let s = 0; for (let i = 0; i < D.T; i++) s += -Math.log(Math.max(D.probs(k, i, D.fixed[i + 1]), 1e-9)); return L(`每个位置取正确答案的概率 p，算 −ln p：猜得越准越接近 0，瞎猜是 ln 20 = 3.0。第 0 段 8 个位置平均 <b>${(s / D.T).toFixed(3)}</b>，整批 64 个位置平均 = 这一步的损失 <b>${D.loss[t].toFixed(4)}</b>。`, `At each position take the probability p of the right answer and compute −ln p: near 0 when confident and right, ln 20 = 3.0 for a blind guess. Row 0 averages <b>${(s / D.T).toFixed(3)}</b>; the mean over all 64 positions is this step’s loss <b>${D.loss[t].toFixed(4)}</b>.`); }
  if (ph === 'bwd') return L(`从损失往回算：先是 logits 的梯度 (p − 1{正确}) / 64，再一层层往回乘（链式法则：上游梯度 × 本层的局部导数），每个激活、每个参数都拿到自己的梯度 ∂L/∂·。右边一栏是流到每一层的梯度，最右边的参数换成了它们的梯度。全部参数梯度的长度 ‖g‖ = <b>${D.gnorm[t].toFixed(3)}</b>。`, `Work back from the loss: first the logits’ gradient (p − 1{answer}) / 64, then multiply back layer by layer (chain rule: upstream gradient × this layer’s local derivative). Every activation and every parameter gets its own gradient ∂L/∂·. The right column shows the gradient reaching each layer; the parameters on the far right switch to their gradients. Length of all parameter gradients: ‖g‖ = <b>${D.gnorm[t].toFixed(3)}</b>.`);
  return L(`AdamW 让每个参数按自己的 m、v 挪一小步，步长大约是学习率 ${sciSup(D.lr[t], 3)}。最右边的参数先显示这一步的 Δw（色标 ±lr），再变成更新后的值（实际是全部参数同时更新，这里按行依次展示）。走完这一步，回到下一步的批次。`, `AdamW moves every parameter a small step according to its own m and v — roughly one learning rate (${sciSup(D.lr[t], 3)}). The parameters on the far right first show this step’s Δw (color scale ±lr), then their updated values (in reality all parameters update at once; here they are shown row by row). Then back to the next step’s batch.`);
}

function batchExplain(s, D, k) {
  const t = D.FR[k];
  if (s.sub === 'pick') return L(`《静夜思》20 个字加一个分隔符 ⏎，首尾相接成 25 个字的圈。第 0 段固定从「举」开始；其余 7 段的起点由随机数决定（种子 1），这一步是 ${[1, 2, 3, 4, 5, 6, 7].map((b) => `「${ch(D, D.stream[D.offs(t, b)])}」`).join('')}。`, `The poem’s 20 characters plus a separator ⏎ form a 25-character loop. Row 0 always starts at 举; the other 7 start points are random (seed 1) — this step: ${[1, 2, 3, 4, 5, 6, 7].map((b) => `“${ch(D, D.stream[D.offs(t, b)])}”`).join(' ')}.`);
  return L(`每段 9 个字：输入取前 8 个，目标取后 8 个（错开一位）。于是第 i 个位置的任务就是：看着输入的前 i+1 个字，猜第 i+2 个字。训练时 8 个位置一起算——这叫教师强制。`, `Each window has 9 characters: the input is the first 8, the target the last 8 (shifted by one). So position i’s task is: having seen the first i+1 input characters, guess character i+2. All 8 positions are trained at once — this is called teacher forcing.`);
}

function fwdExplain(op, D, k) {
  const x = D.fixed, T = D.T;
  switch (op) {
    case 'emb': { const e = D.pIndex.get('E'); return L(`每个字查 E 的一行：「举」是第 ${x[0]} 号，取 E 第 ${x[0]} 行的 16 个数……8 个位置得到 h₀（8 × 16）。查表没有乘法；反向时梯度也只会流回被查到的那几行。`, `Each character looks up one row of E: 举 is id ${x[0]}, so it takes the 16 numbers in row ${x[0]}… 8 positions give h₀ (8 × 16). A lookup has no multiplication; on the way back the gradient flows only into the rows that were used.`) + (e ? '' : ''); }
    case 'norm1': { const inv = D.act(k, 'inv1'); return L(`RMSNorm：每个位置的向量除以自己的均方根，再逐维乘 γ₁。位置 0 的均方根是 ${f4(1 / inv[0])}，除完以后大小拉回 1 左右——后面的乘法就不会越乘越大或越乘越小。`, `RMSNorm: divide each position’s vector by its root mean square, then scale each dimension by γ₁. Position 0’s RMS is ${f4(1 / inv[0])}; after dividing, sizes are back around 1, so later multiplications neither blow up nor fade away.`); }
    case 'qkv': return L(`n₁ 分别乘 W_q、W_k、W_v，得到 q、k、v（各 16 维 = 2 个头 × 8 维）。q 像“我要找什么”，k 像“我这里有什么”，v 像“我能提供什么”。`, `n₁ is multiplied by W_q, W_k, W_v to give q, k, v (16 dims each = 2 heads × 8). Think of q as “what I’m looking for”, k as “what I have”, v as “what I can pass on”.`);
    case 'attn': {
      const a = (h, j) => D.att(k, h, T - 1, j);   // 最后一个位置的注意力（在权重块里）
      const best = (h) => { let b = 0; for (let j = 0; j < T; j++) if (a(h, j) > a(h, b)) b = j; return b; };
      const b0 = best(0), b1 = best(1);
      return L(`每个头里，每个位置拿自己的 q 和前面每个位置的 k 做点积（先按位置用 RoPE 旋转），除以 √8，softmax 成注意力，再对 v 加权求和。最后一个位置「头」：头 0 把 ${P(a(0, b0))} 给了「${ch(D, x[b0])}」，头 1 把 ${P(a(1, b1))} 给了「${ch(D, x[b1])}」。`, `In each head, every position dots its q with the k of every earlier position (after rotating them by position with RoPE), divides by √8, applies softmax, and takes the weighted sum of v. The last position 头: head 0 gives ${P(a(0, b0))} to “${ch(D, x[b0])}”, head 1 gives ${P(a(1, b1))} to “${ch(D, x[b1])}”.`);
    }
    case 'wo': return L(`两个头的输出拼回 16 维，乘 W_o 得到 o，再加回残差：h₁ = h₀ + o。残差让原来的信息（以及反向时的梯度）可以不经过注意力直接通过。`, `The two heads’ outputs are joined back into 16 dims and multiplied by W_o to give o, which is added onto the residual: h₁ = h₀ + o. The residual lets the original information (and, on the way back, the gradient) pass straight through without going through attention.`);
    case 'norm2': return L(`再做一次 RMSNorm（缩放是 γ₂），给前馈层准备输入 n₂。`, `Another RMSNorm (scale γ₂) prepares the FFN’s input n₂.`);
    case 'ffn': return L(`SwiGLU：n₂ 乘 W_gate、W_up 各得到 32 维；gate 过 silu（负的大多被压成 0），再和 up 逐元素相乘，得到 act。gate 像一排开关，决定 up 的哪些维度能通过。`, `SwiGLU: n₂ times W_gate and W_up gives 32 dims each; gate goes through silu (most negatives are squashed to 0) and is multiplied elementwise with up to give act. gate acts like a row of switches deciding which dims of up get through.`);
    case 'wd': return L(`act 乘 W_down 回到 16 维，加回残差：h₂ = h₁ + f。`, `act times W_down returns to 16 dims, added onto the residual: h₂ = h₁ + f.`);
    case 'normf': return L(`最后一次 RMSNorm（缩放 γ_f），得到 n_f。`, `A final RMSNorm (scale γ_f) gives n_f.`);
    case 'logits': { const b = topOf(D, k, T - 1), pc = D.probs(k, T - 1, x[T]); return L(`n_f 乘 Eᵀ：和 20 个字的嵌入各做一次点积，得到 20 个分数（logits），softmax 成概率。最后一个位置最想说「${ch(D, b)}」（${P(D.probs(k, T - 1, b))}），正确答案「${ch(D, x[T])}」是 ${P(pc)}。绿框 = 每个位置的正确答案。`, `n_f times Eᵀ: a dot product with each of the 20 characters’ embeddings gives 20 scores (logits), turned into probabilities by softmax. The last position’s favourite is “${ch(D, b)}” (${P(D.probs(k, T - 1, b))}); the right answer “${ch(D, x[T])}” gets ${P(pc)}. Green boxes = each position’s right answer.`); }
  }
  return '';
}

function lossExplain(s, D, k) {
  const x = D.fixed, t = D.FR[k];
  if (s.sub === 'mean') { let sm = 0; for (let i = 0; i < D.T; i++) sm += -Math.log(Math.max(D.probs(k, i, x[i + 1]), 1e-9)); return L(`第 0 段 8 个位置平均 <b>${(sm / D.T).toFixed(4)}</b>；整批 64 个位置一起平均，就是这一步的损失 <b>${D.loss[t].toFixed(4)}</b>——训练要让它变小。`, `Row 0 averages <b>${(sm / D.T).toFixed(4)}</b> over its 8 positions; averaging all 64 positions gives this step’s loss <b>${D.loss[t].toFixed(4)}</b> — the number training pushes down.`); }
  const i = s.i, p = D.probs(k, i, x[i + 1]);
  return L(`位置 ${i}：看到“${esc(x.slice(0, i + 1).map(D.ch).join(''))}”，正确答案「${ch(D, x[i + 1])}」的概率 ${P(p)}，损失 −ln p = <b>${(-Math.log(Math.max(p, 1e-9))).toFixed(3)}</b>。`, `Position ${i}: having seen “${esc(x.slice(0, i + 1).map(D.ch).join(''))}”, the right answer “${ch(D, x[i + 1])}” has probability ${P(p)}, loss −ln p = <b>${(-Math.log(Math.max(p, 1e-9))).toFixed(3)}</b>.`);
}

function bwdExplain(op, D, k) {
  const T = D.T, G = D.G(k);
  const gn = (n) => { const p = D.pIndex.get(n); let s = 0; for (let i = p.off; i < p.off + p.n; i++) s += G[i] * G[i]; return Math.sqrt(s); };
  switch (op) {
    case 'logits': { const dl = D.grad(k, 'logits'), y = D.fixed[T]; return L(`起点：∂L/∂logits = (p − 1{正确}) / 64。正确答案那一格是负的（要把它推高），其余是正的（往下压）。最后一个位置「${ch(D, y)}」那一格是 ${f4(dl[(T - 1) * D.V + y])}。再往回：∂L/∂n_f = ∂L/∂logits · E；E 在输出这一路的梯度 = ∂L/∂logitsᵀ · n_f。`, `Start: ∂L/∂logits = (p − 1{answer}) / 64. The answer’s cell is negative (push it up), the rest positive (push down). At the last position the cell for “${ch(D, y)}” is ${f4(dl[(T - 1) * D.V + y])}. Further back: ∂L/∂n_f = ∂L/∂logits · E; E’s gradient on this output path = ∂L/∂logitsᵀ · n_f.`); }
    case 'normf': return L(`穿过最后的 RMSNorm：∂L/∂h₂ = 上游的 ∂L/∂n_f × RMSNorm 的局部导数；γ_f 的梯度 = Σ（归一化后的值 × ∂L/∂n_f），‖∂L/∂γ_f‖ = ${f4(gn('gf'))}。`, `Through the final RMSNorm: ∂L/∂h₂ = upstream ∂L/∂n_f × RMSNorm’s local derivative; γ_f’s gradient = Σ (normalized value × ∂L/∂n_f), ‖∂L/∂γ_f‖ = ${f4(gn('gf'))}.`);
    case 'wd': return L(`h₂ = h₁ + act·W_down，所以：∂L/∂act = ∂L/∂h₂ · W_downᵀ（上游梯度 × 局部导数 W_down）；∂L/∂W_down = actᵀ · ∂L/∂h₂（输入 × 上游梯度，64 个位置加起来），‖·‖ = ${f4(gn('Wd'))}。残差那一路，∂L/∂h₁ 直接拿到一份 ∂L/∂h₂。`, `h₂ = h₁ + act·W_down, so: ∂L/∂act = ∂L/∂h₂ · W_downᵀ (upstream × local derivative W_down); ∂L/∂W_down = actᵀ · ∂L/∂h₂ (input × upstream, summed over 64 positions), norm ${f4(gn('Wd'))}. Along the residual, ∂L/∂h₁ simply receives a copy of ∂L/∂h₂.`);
    case 'ffn': return L(`act = silu(gate) ⊙ up：∂L/∂up = ∂L/∂act ⊙ silu(gate)，∂L/∂gate = ∂L/∂act ⊙ up ⊙ silu′(gate)——被 silu 压成 0 的维度，梯度也几乎过不去。再乘回去：∂L/∂W_gate = n₂ᵀ · ∂L/∂gate（${f4(gn('Wg'))}），∂L/∂W_up = n₂ᵀ · ∂L/∂up（${f4(gn('Wu'))}）。`, `act = silu(gate) ⊙ up: ∂L/∂up = ∂L/∂act ⊙ silu(gate), ∂L/∂gate = ∂L/∂act ⊙ up ⊙ silu′(gate) — dims squashed to 0 by silu barely let the gradient through. Then: ∂L/∂W_gate = n₂ᵀ · ∂L/∂gate (${f4(gn('Wg'))}), ∂L/∂W_up = n₂ᵀ · ∂L/∂up (${f4(gn('Wu'))}).`);
    case 'norm2': return L(`穿过 RMSNorm ②，梯度加回残差：∂L/∂h₁ = 残差直接来的那份 + 从前馈绕回来的那份。‖∂L/∂γ₂‖ = ${f4(gn('g2'))}。`, `Through RMSNorm ②, the gradient joins the residual: ∂L/∂h₁ = the copy that came straight down + the part that came back through the FFN. ‖∂L/∂γ₂‖ = ${f4(gn('g2'))}.`);
    case 'wo': return L(`o = ao · W_o：∂L/∂ao = ∂L/∂h₁ · W_oᵀ；∂L/∂W_o = aoᵀ · ∂L/∂h₁，‖·‖ = ${f4(gn('Wo'))}。`, `o = ao · W_o: ∂L/∂ao = ∂L/∂h₁ · W_oᵀ; ∂L/∂W_o = aoᵀ · ∂L/∂h₁, norm ${f4(gn('Wo'))}.`);
    case 'attn': return L(`穿过注意力：∂L/∂v = 注意力ᵀ · ∂L/∂ao；∂L/∂a（每条注意力连线要变强还是变弱）= ∂L/∂ao · vᵀ，再穿过 softmax 得到 q、k 的梯度。中间那两格 ∂L/∂a 就是“这条连线该加强（琥珀）还是减弱（蓝）”。`, `Through attention: ∂L/∂v = attentionᵀ · ∂L/∂ao; ∂L/∂a (should each attention link get stronger or weaker) = ∂L/∂ao · vᵀ, then through softmax to the gradients of q and k. The two ∂L/∂a squares show which links should strengthen (amber) or weaken (blue).`);
    case 'qkv': return L(`q、k、v 都是 n₁ 乘矩阵：∂L/∂W_q = n₁ᵀ · ∂L/∂q（${f4(gn('Wq'))}），∂L/∂W_k（${f4(gn('Wk'))}），∂L/∂W_v（${f4(gn('Wv'))}）；三路梯度乘回去加在一起成为 ∂L/∂n₁。${D.FR[k] < 5 ? '一开始 W_q、W_k 的梯度几乎是 0：注意力还是平均分，q、k 稍微改一点几乎不影响结果。' : ''}`, `q, k, v are all n₁ times a matrix: ∂L/∂W_q = n₁ᵀ · ∂L/∂q (${f4(gn('Wq'))}), ∂L/∂W_k (${f4(gn('Wk'))}), ∂L/∂W_v (${f4(gn('Wv'))}); the three paths multiplied back add up to ∂L/∂n₁.${D.FR[k] < 5 ? ' Early on W_q and W_k get almost no gradient: attention is still spread evenly, so nudging q or k barely changes anything.' : ''}`);
    case 'norm1': return L(`穿过 RMSNorm ①，再加上残差那一路，得到 ∂L/∂h₀。‖∂L/∂γ₁‖ = ${f4(gn('g1'))}。`, `Through RMSNorm ①, plus the residual path, giving ∂L/∂h₀. ‖∂L/∂γ₁‖ = ${f4(gn('g1'))}.`);
    case 'emb': return L(`查表的反向：∂L/∂h₀ 的每一行加回到 E 里被查到的那一行；没出现在这批里的字，输入这一路的梯度是 0。加上输出那一路，E 的梯度合计 ‖·‖ = ${f4(gn('E'))}——是所有参数里最大的。`, `The lookup in reverse: each row of ∂L/∂h₀ is added back into the row of E it came from; characters not in this batch get nothing on this path. Together with the output path, E’s total gradient norm is ${f4(gn('E'))} — the largest of all parameters.`);
  }
  return '';
}

function chainExplain(D, k, gi) {
  const ct = chainTerms(D, k, gi), G = D.G(k)[gi];
  let r0 = 0, n = 0;
  for (const gp of ct.groups) for (const tm of gp.terms) { r0 += tm.x * tm.d; n++; }
  return L(`${esc(paramName(D, gi))} 的梯度不是凭空来的：每个位置都有一项“输入 × 上游梯度”。第 0 段这里列出 ${n} 项，加起来 ${f4(r0)}；另外 7 段（56 个位置）合计 ${f4(G - r0)}；总共 <b>${f4(G)}</b>，就是记录的 ∂L/∂w。`, `${esc(paramName(D, gi))}’s gradient doesn’t come from nowhere: every position contributes one term, input × upstream gradient. Row 0’s ${n} terms are listed, adding up to ${f4(r0)}; the other 7 rows (56 positions) add ${f4(G - r0)}; the total <b>${f4(G)}</b> is the recorded ∂L/∂w.`);
}

function clipExplain(D, k) {
  const t = D.FR[k], gn = D.gnorm[t], c = D.clip[t];
  return L(`先看全部 2,928 个梯度合起来有多长：‖g‖ = <b>${gn.toFixed(4)}</b>。${c < 1 ? `超过 1.0，所有梯度一起乘 ${c.toFixed(4)}，方向不变，只是步子收短。` : '没超过 1.0，不裁剪。'}`, `First, how long are all 2,928 gradients together: ‖g‖ = <b>${gn.toFixed(4)}</b>. ${c < 1 ? `Over 1.0, so every gradient is multiplied by ${c.toFixed(4)} — same direction, shorter step.` : 'Not over 1.0, so no clipping.'}`);
}

function updTensorExplain(name, D, k) {
  const p = D.pIndex.get(name), DW = D.DW(k), lr = D.lr[D.FR[k]];
  let s = 0, mx = 0;
  for (let i = p.off; i < p.off + p.n; i++) { s += Math.abs(DW[i]); mx = Math.max(mx, Math.abs(DW[i])); }
  const mean = s / p.n;
  return L(`${TENSOR_LABEL[name]} 的 ${p.n} 个参数各自按 AdamW 挪一步：平均 |Δw| = ${sciSup(mean, 3)}（≈ ${(mean / lr).toFixed(2)} × lr），最大 ${sciSup(mx, 3)}。${p.decay ? '' : 'RMSNorm 的 γ 不做权重衰减。'}按 ＋ 看其中一个参数的完整算式。`, `Each of ${TENSOR_LABEL[name]}’s ${p.n} parameters takes its own AdamW step: mean |Δw| = ${sciSup(mean, 3)} (≈ ${(mean / lr).toFixed(2)} × lr), max ${sciSup(mx, 3)}.${p.decay ? '' : ' RMSNorm’s γ gets no weight decay.'} Press + to see the full formula for one of them.`);
}

function adamExplain(mi, D, k, gi) {
  const a = adamAt(D, k, gi), nm = esc(paramName(D, gi));
  switch (mi) {
    case 'g': return L(`${nm} 这一步的梯度 ∂L/∂w = ${f4(a.graw)}${a.clip < 1 ? `，裁剪后乘 ${a.clip.toFixed(4)}` : ''}，g = <b>${f4(a.g)}</b>。负的梯度意味着“把 w 调大，损失会变小”。`, `${nm}’s gradient this step is ∂L/∂w = ${f4(a.graw)}${a.clip < 1 ? `, times ${a.clip.toFixed(4)} after clipping` : ''}, g = <b>${f4(a.g)}</b>. A negative gradient means “increasing w lowers the loss”.`);
    case 'm': return L(`一阶动量是梯度的滑动平均：m = 0.9 × 上一步的 m + 0.1 × g = <b>${f4(a.m)}</b>。它把每一步忽左忽右的梯度平滑成一个稳定的方向。`, `The first moment is a moving average of gradients: m = 0.9 × previous m + 0.1 × g = <b>${f4(a.m)}</b>. It smooths step-to-step jitter into a steady direction.`);
    case 'v': return L(`二阶动量是梯度平方的滑动平均：v = 0.95 × 上一步的 v + 0.05 × g² = <b>${sciSup(a.v, 3)}</b>，√v ≈ 梯度的典型大小。`, `The second moment averages squared gradients: v = 0.95 × previous v + 0.05 × g² = <b>${sciSup(a.v, 3)}</b>; √v ≈ the typical gradient size.`);
    case 'bc': return L(`m、v 都从 0 开始，前几步偏小，除以 (1 − βᵗ) 校正回来：m̂ = ${f4(a.mh)}，v̂ = ${sciSup(a.vh, 3)}。${a.t === 1 ? '第 1 步正好 m̂ = g、v̂ = g²。' : `第 ${a.t} 步的校正系数 ${(1 / (1 - a.b1 ** a.t)).toFixed(3)} / ${(1 / (1 - a.b2 ** a.t)).toFixed(3)}。`}`, `m and v start at 0 and are too small early on; dividing by (1 − βᵗ) corrects them: m̂ = ${f4(a.mh)}, v̂ = ${sciSup(a.vh, 3)}.${a.t === 1 ? ' At step 1, m̂ = g and v̂ = g² exactly.' : ` Correction factors at step ${a.t}: ${(1 / (1 - a.b1 ** a.t)).toFixed(3)} / ${(1 / (1 - a.b2 ** a.t)).toFixed(3)}.`}`);
    case 'dw': return L(`Δw = −lr × m̂/(√v̂ + ε) − lr × λ × w = ${sciSup(a.adam, 3)} ${a.decay >= 0 ? '+' : '−'} ${sciSup(Math.abs(a.decay), 3)} = <b>${sciSup(a.dw, 4)}</b>。m̂/√v̂ 的大小总在 1 左右，所以每个参数每步都挪“差不多一个学习率”，不管它的梯度本身多大。`, `Δw = −lr × m̂/(√v̂ + ε) − lr × λ × w = ${sciSup(a.adam, 3)} ${a.decay >= 0 ? '+' : '−'} ${sciSup(Math.abs(a.decay), 3)} = <b>${sciSup(a.dw, 4)}</b>. m̂/√v̂ is always around 1 in size, so every parameter moves “about one learning rate” per step, however large its gradient is.`);
    case 'write': return L(`写回：${f4(a.w)} → <b>${Number(a.w1.toPrecision(7))}</b>。${a.exact ? '和记录的下一步逐位一致（float32）。' : `和记录的 Δw 相差 ${a.rec.dw ? ((Math.abs(a.dw - a.rec.dw) / Math.abs(a.rec.dw)) * 100).toFixed(2) : 0}%（float16 记录的精度）。`}`, `Write back: ${f4(a.w)} → <b>${Number(a.w1.toPrecision(7))}</b>.${a.exact ? ' Matches the recorded next value bit for bit (float32).' : ` Differs from the recorded Δw by ${a.rec.dw ? ((Math.abs(a.dw - a.rec.dw) / Math.abs(a.rec.dw)) * 100).toFixed(2) : 0}% (float16 record precision).`}`);
  }
  return '';
}

/* ---------------------------------------------------------------- 变量 */

export function watch(s, R, ctx, depth) {
  const D = R.D;
  const rows = [['wh', L('GLASS · 《静夜思》', 'GLASS · 静夜思')]];
  if (s.ph === 'init') {
    rows.push([L('参数', 'params'), fmtInt(D.P)], ['std', '0.02'], [L('全零的损失', 'zero-init loss'), D.zLoss[0].toFixed(4)], [L('放大 50 倍', '50× init'), D.bLoss[0].toFixed(3)]);
    return rows;
  }
  const t = D.FR[s.k];
  rows.push(['step', `${t + 1} / ${D.S}`], ['lr', sciSup(D.lr[t], 3)], ['loss', D.loss[t].toFixed(4)], ['‖g‖', D.gnorm[t].toFixed(4)]);
  if (s.ph === 'run') { rows.push([L('25 段平均', 'all 25'), D.evalLoss[t].toFixed(4)]); return rows; }
  if (!D.ready(s.k)) return rows;
  if (s.ph === 'upd' || s.ph === 'bwd') {
    if (s.mi) {
      const gi = selOf(D, ctx, s), a = adamAt(D, s.k, gi);
      rows.push(['wh', paramName(D, gi)], ['w', Number(a.w.toPrecision(6))], ['∂L/∂w', f4(a.graw)], ['m', f4(a.m)], ['v', sciSup(a.v, 3)], ['Δw', sciSup(a.dw, 3)]);
    } else rows.push([L('裁剪系数', 'clip'), D.clip[t] < 1 ? D.clip[t].toFixed(4) : '1']);
  }
  if (s.ph === 'loss' && s.sub === 'pos') { const p = D.probs(s.k, s.i, D.fixed[s.i + 1]); rows.push(['p', fmtP(p)], ['−ln p', (-Math.log(Math.max(p, 1e-9))).toFixed(4)]); }
  return rows;
}
