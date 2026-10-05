// 唐宋诗小模型这一章（3D 机器）的调试器内容：代码（伪代码，高亮当前行）、这一步的白话讲解 + 真实数字、变量监视。
// 接口和 ../explain.js、../glass/explain.js 一样：codeFor / linesFor / stepLabel / crumbs / shapeOf / explain / watch。
import { esc } from '../../../js/ui.js';
import { fmtP, sciSup, fmtInt } from '../draw.js';
import { isEn, L, compact, featLabel } from '../lang.js';
import { FWD, BWD, opIndex, ADAM_SUBS } from './timeline.js';
import { tensorName, KIND_LABEL } from './data.js';

const K_ = (s) => `<span class="kw">${s}</span>`;
const F_ = (s) => `<span class="fn">${s}</span>`;
const C_ = (s) => `<span class="cm"># ${s}</span>`;
const N_ = (s) => `<span class="nu">${s}</span>`;
const P = (p) => `<em>${fmtP(p)}</em>`;
const f4 = (v) => (v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e4 ? Number(v.toPrecision(4)).toString() : sciSup(v, 3));
const ch = (X, id) => esc(X.ch(id));

export function codeFor(X) {
  const m = X.meta;
  return [
    `model = ${F_('Qwen3ForCausalLM')}(cfg) ${C_(L(`${(m.model.params / 1e4).toFixed(0)} 万参数`, `${compact(m.model.params)} params`))}`,
    `opt = ${F_('AdamW')}(lr=${N_('3e-3')}, wd=${N_('0.1')})`,
    `${K_('for')} step ${K_('in')} ${F_('range')}(${N_(m.train.steps)}):`,
    `    x = ${F_('next_batch')}()   ${C_(L(`${m.train.batch} × ${m.train.seq + 1} 字`, `${m.train.batch} × ${m.train.seq + 1} chars`))}`,
    `    inp, tgt = x[:, :${N_('-1')}], x[:, ${N_('1')}:]`,
    `    h = E[inp]              ${C_(L('查嵌入表', 'embedding lookup'))}`,
    `    ${K_('for')} layer ${K_('in')} layers:   ${C_(L('6 层', '6 layers'))}`,
    `        h = h + ${F_('attn')}(${F_('rms')}(h)) ${C_(L('4 头 / 2 KV 头', '4 heads / 2 KV'))}`,
    `        h = h + ${F_('ffn')}(${F_('rms')}(h))  ${C_('SwiGLU 256→768')}`,
    `    logits = ${F_('rms')}(h) @ E.T   ${C_(L('共用 E', 'shared E'))}`,
    `    loss = ${F_('cross_entropy')}(logits, tgt)`,
    `    loss.${F_('backward')}()        ${C_(L('反向', 'backward'))}`,
    `    ${F_('clip_grad_norm_')}(params, ${N_('1.0')})`,
    `    ${K_('for')} w, g ${K_('in')} params:   ${C_('AdamW')}`,
    `        m = ${N_('.9')}m + ${N_('.1')}g;  v = ${N_('.95')}v + ${N_('.05')}g²`,
    `        w −= lr·(m̂/(√v̂+ε) + λ·w)`,
  ];
}

export function linesFor(s) {
  switch (s.ph) {
    case 'run': return [3];
    case 'batch': return !s.sub ? [4, 5] : s.sub === 'shift' ? [5] : [4];
    case 'fwd': return !s.sub ? [6, 7, 8, 9, 10] : { emb: [6], attn: [7, 8], ffn: [7, 9], head: [10] }[s.sub];
    case 'loss': return [11];
    case 'bwd': return [12];
    case 'upd': return !s.sub ? [13, 14, 15, 16] : s.sub === 'clip' ? [13] : !s.mi ? [14, 15, 16] : { g: [14], m: [15], v: [15], bc: [16], dw: [16], write: [16] }[s.mi];
  }
  return [];
}

/* ---------------------------------------------------------------- 名字 */

const PH = isEn ? { batch: 'Batch', fwd: 'Forward', loss: 'Loss', bwd: 'Backward', upd: 'Update' } : { batch: '取批次', fwd: '前向', loss: '损失', bwd: '反向', upd: '更新' };
const BATCH = isEn ? { row: 'take row 0', ids: 'chars → IDs', shift: 'shift by one' } : { row: '取第 0 行', ids: '字 → 编号', shift: '错开一位' };
const ADAM = isEn ? { g: 'gradient g', m: 'first moment m', v: 'second moment v', bc: 'bias correction', dw: 'compute Δw', write: 'write back' } : { g: '梯度 g', m: '一阶动量 m', v: '二阶动量 v', bc: '偏差校正', dw: '算出 Δw', write: '写回' };
export const CROP_NAME = isEn ? ['Embedding E (48 × 48)', 'Layer 2 W_q (48 × 48)', 'Layer 4 W_down (48 × 48)'] : ['嵌入 E 的局部', '第 2 层 W_q 的局部', '第 4 层 W_down 的局部'];

export function opName(s) {
  if (s.sub === 'emb') return L('嵌入', 'embedding');
  if (s.sub === 'head') return L('输出头', 'output head');
  return s.sub === 'attn' ? L(`第 ${s.L} 层 · 注意力`, `layer ${s.L} · attention`) : L(`第 ${s.L} 层 · 前馈`, `layer ${s.L} · FFN`);
}

export function stepLabel(s, X) {
  if (s.ph === 'run') return L(`第 ${fmtInt(X.step(s.k) + 1)} 步`, `Step ${fmtInt(X.step(s.k) + 1)}`);
  if (!s.sub) return PH[s.ph];
  switch (s.ph) {
    case 'batch': return `${PH.batch} · ${BATCH[s.sub]}`;
    case 'fwd': case 'bwd': {
      let lab = `${PH[s.ph]} · ${opName(s)}`;
      if (s.f != null) lab += ` · ${featLabel(X.feats[s.f], 'tiny')}`;
      else if (s.c != null) lab += ` · ${CROP_NAME[s.c]}`;
      return lab;
    }
    case 'loss':
      if (s.sub === 'mean') return `${PH.loss} · ${L('整批平均', 'batch mean')}`;
      if (s.i == null) return `${PH.loss} · ${L('64 个位置', '64 positions')}`;
      return `${PH.loss} · ${L(`位置 ${s.i}`, `position ${s.i}`)}「${ch(X, X.row(s.k, s.i))}→${ch(X, X.tgt(s.k, s.i))}」`;
    case 'upd':
      if (s.sub === 'clip') return `${PH.upd} · ${L('梯度裁剪', 'gradient clipping')}`;
      if (s.f != null) return `${PH.upd} · ${featLabel(X.feats[s.f], 'tiny')} · ${ADAM[s.mi]}`;
      if (s.c != null) return `${PH.upd} · ${CROP_NAME[s.c]}`;
      return `${PH.upd} · AdamW`;
  }
  return '';
}

export function crumbs(depth, s, X) {
  const out = [{ d: 1, label: L('训练全程', 'Whole run') }];
  if (depth >= 2) out.push({ d: 2, label: L(`第 ${fmtInt(X.step(s.k) + 1)} 步`, `Step ${fmtInt(X.step(s.k) + 1)}`) });
  if (depth >= 3 && s.sub) out.push({ d: 3, label: PH[s.ph] });
  if (depth >= 4 && s.c != null && s.f == null) out.push({ d: 4, label: CROP_NAME[s.c] });
  else if (depth >= 4 && s.i != null) out.push({ d: 4, label: L(`位置 ${s.i}`, `position ${s.i}`) });
  if (depth >= 5 && s.f != null) out.push({ d: 5, label: featLabel(X.feats[s.f], 'tiny') });
  return out;
}

export function shapeOf(s, X) {
  const m = X.meta, B = m.train.batch, T = m.train.seq, H = m.model.hidden, V = m.model.vocab;
  if (s.ph === 'run') return '';
  if (s.ph === 'batch') return L(`x <b>[${B} × ${T + 1}]</b> → inp <b>[${B} × ${T}]</b>，tgt <b>[${B} × ${T}]</b>；机器上画第 0 行的前 64 个位置`, `x <b>[${B} × ${T + 1}]</b> → inp <b>[${B} × ${T}]</b>, tgt <b>[${B} × ${T}]</b>; the machine shows the first 64 positions of row 0`);
  if (s.ph === 'fwd' || s.ph === 'bwd') {
    const g = s.ph === 'bwd' ? '∂L/∂' : '';
    if (s.f != null) return L(`一个数：${esc(featLabel(X.feats[s.f], 'tiny'))}`, `one number: ${esc(featLabel(X.feats[s.f], 'tiny'))}`);
    if (s.c != null) return L(`${g}${s.c === 0 ? 'E' : s.c === 1 ? 'W_q' : 'W_down'} 的 <b>48 × 48</b> 局部（整块 <b>${s.c === 0 ? '7478 × 256' : s.c === 1 ? '256 × 256' : '768 × 256'}</b>）`, `the <b>48 × 48</b> corner of ${g}${s.c === 0 ? 'E' : s.c === 1 ? 'W_q' : 'W_down'} (whole: <b>${s.c === 0 ? '7478 × 256' : s.c === 1 ? '256 × 256' : '768 × 256'}</b>)`);
    if (!s.sub) return L(`inp <b>[${B} × ${T}]</b> → h <b>[${B} × ${T} × ${H}]</b> → 6 层 → logits <b>[${B} × ${T} × ${fmtInt(V)}]</b>`, `inp <b>[${B} × ${T}]</b> → h <b>[${B} × ${T} × ${H}]</b> → 6 layers → logits <b>[${B} × ${T} × ${fmtInt(V)}]</b>`);
    if (s.sub === 'emb') return `${g}E <b>[${fmtInt(V)} × ${H}]</b>：inp → h₀ <b>[${B} × ${T} × ${H}]</b>`;
    if (s.sub === 'head') return `${g}h <b>[${B} × ${T} × ${H}]</b> @ Eᵀ <b>[${H} × ${fmtInt(V)}]</b> → logits <b>[${B} × ${T} × ${fmtInt(V)}]</b>`;
    if (s.sub === 'attn') return `${g}W_q <b>[256 × 256]</b>，W_k / W_v <b>[256 × 128]</b>，W_o <b>[256 × 256]</b>；q <b>[${B} × ${T} × 4 × 64]</b>，k / v <b>[${B} × ${T} × 2 × 64]</b>`;
    return `${g}W_gate / W_up <b>[256 × 768]</b>，W_down <b>[768 × 256]</b>；gate、up <b>[${B} × ${T} × 768]</b>`;
  }
  if (s.ph === 'loss') return L(`logits <b>[${B} × ${T} × ${fmtInt(V)}]</b> → 每个位置 −ln p → ${fmtInt(B * T)} 个数平均成 1 个`, `logits <b>[${B} × ${T} × ${fmtInt(V)}]</b> → −ln p per position → ${fmtInt(B * T)} numbers averaged into 1`);
  if (s.ph === 'upd') return L(`m、v：和参数同形状，各 <b>${fmtInt(m.model.params)}</b> 个 fp32`, `m, v: same shape as the parameters, <b>${fmtInt(m.model.params)}</b> fp32 numbers each`);
  return '';
}

/* ---------------------------------------------------------------- 统计 */

function lensStats(X, k, b) {
  let ok = 0, p = 0;
  for (let i = 0; i < 64; i++) { if (X.lensTop(k, b, i) === X.tgt(k, i)) ok++; p += X.lensP(k, b, i); }
  return { acc: ok / 64, p: p / 64 };
}
function rowStats(X, k) {
  let nl = 0, best = 0, worst = 0;
  for (let i = 0; i < 64; i++) { const v = -Math.log(Math.max(X.rowP(k, i), 1e-12)); nl += v; if (v < -Math.log(Math.max(X.rowP(k, best), 1e-12))) best = i; if (v > -Math.log(Math.max(X.rowP(k, worst), 1e-12))) worst = i; }
  return { mean: nl / 64, best, worst };
}
function residMean(X, k, b, grad) { let s = 0; for (let i = 0; i < 64; i++) s += grad ? X.residGrad(k, b, i) : X.residNorm(k, b, i); return s / 64; }
function headStats(X, k, l, h) {
  let prev = 0, self = 0, back6 = 0, first = 0, n = 0;
  for (let i = 7; i < 25; i++) { prev += X.attn(k, l, h, i, i - 1); self += X.attn(k, l, h, i, i); back6 += X.attn(k, l, h, i, i - 6); first += X.attn(k, l, h, i, 0); n++; }
  return { prev: prev / n, self: self / n, back6: back6 / n, first: first / n };
}
function headLine(X, k, l) {
  if (!X.has('ck', k)) return '';
  const nm = isEn ? { prev: 'the previous char', back6: 'the same spot in the previous line', self: 'itself', first: 'the start' } : { prev: '前一个字', back6: '上一句同一位置', self: '自己', first: '开头' };
  const parts = [0, 1, 2, 3].map((h) => { const s = headStats(X, k, l, h); const [key, v] = Object.entries(s).sort((a, b) => b[1] - a[1])[0]; return L(`头 ${h} 最常看${nm[key]}（${fmtP(v)}）`, `head ${h} mostly looks at ${nm[key]} (${fmtP(v)})`); });
  return L(`在留出的《登鹳雀楼》上：${parts.join('，')}。`, `On the held-out poem: ${parts.join('; ')}.`);
}
function gradList(X, k, l, kinds) {
  return kinds.map((kd) => `${KIND_LABEL[kd]} ${sciSup(X.tgrad(k, l, kd), 2)}`).join(L('，', ', '));
}

function runNarrative(X, k) {
  const m = X.meta, t = X.step(k), D = X.D;
  if (!D.has('ck', k)) return L('<span class="dimmed">（这个检查点的逐字概率正在载入…）</span>', '<span class="dimmed">(loading this checkpoint’s per-character probabilities…)</span>');
  let punct = 0, np = 0, cont = 0, nc = 0;
  for (let i = 0; i < D.Lv; i++) {
    const c = m.held.text[i] ?? '⏎', p = D.valP(k, i);
    if (c === '，' || c === '。') { punct += p; np++; } else if (c !== '⏎') { cont += p; nc++; }
  }
  punct /= np; cont /= nc;
  if (t < 20) return L(`刚刚随机初始化：模型对 ${fmtInt(m.model.vocab)} 个字几乎一视同仁，损失接近 ln ${m.model.vocab} = ${Math.log(m.model.vocab).toFixed(2)}；打印机打出来的是随机的字，透镜屏上每一层都在乱写。`, `Freshly initialized at random: the model treats all ${fmtInt(m.model.vocab)} characters almost alike, the loss is close to ln ${m.model.vocab} = ${Math.log(m.model.vocab).toFixed(2)}; the printer prints random characters and every lens screen is gibberish.`);
  if (punct < 0.6) return L('它先学会的是最常见的字和标点的大致频率——透镜屏上开始出现一排排“，”“。”“一”“不”，打印机写出的字开始“像诗里的字”，但还没有句子。', 'It first picks up the rough frequency of the commonest characters and punctuation — the lens screens fill with rows of “，” “。” “一” “不”, and the printer’s characters start to “look like poem characters”, but no sentences yet.');
  if (cont < 0.08) return L(`<b>格式先于内容</b>：《登鹳雀楼》里标点的平均概率已经到 ${P(punct)}，正文的字只有 ${P(cont)}。它先学会了“五个字一个逗号”。`, `<b>Format before content</b>: punctuation in “On the Stork Tower” already averages ${P(punct)}, the characters only ${P(cont)}. It has learned “a comma every five characters” first.`);
  return L(`正文的字也在变得可猜：《登鹳雀楼》正文平均 ${P(cont)}，标点 ${P(punct)}。越往上的透镜屏绿字越多：答案是一层层拼出来的。`, `The characters are becoming guessable too: ${P(cont)} on average for the text of “On the Stork Tower”, ${P(punct)} for punctuation. The higher the lens screen, the more green characters: the answer is built up layer by layer.`);
}

/* ---------------------------------------------------------------- 讲解 */

export function explain(s, X, ctx, depth) {
  const m = X.meta, k = s.k, t = X.step(k) + 1, c = m.ckpts[k];
  if (s.ph === 'run') {
    const lrPhase = c.t < m.train.warmup ? L(`还在预热（前 ${m.train.warmup} 步线性升到 ${m.train.peakLr}）`, `still warming up (rising linearly to ${m.train.peakLr} over the first ${m.train.warmup} steps)`) : L('在余弦退火中慢慢变小', 'slowly shrinking under cosine annealing');
    return L(`第 <b>${fmtInt(t)}</b> 步（共 ${fmtInt(m.train.steps)} 步）。这一批的损失 <b>${c.loss.toFixed(3)}</b>，没训练过的验证集 <b>${c.valLoss.toFixed(3)}</b>；学习率 ${sciSup(c.lr, 3)}，${lrPhase}。<br>${runNarrative(X, k)}<br><span class="dimmed">一拍里机器走完一整步：第 0 行的字进托盘 → 光脉冲一层层往上（透镜屏亮起）→ 顶上的柱子升起 → −ln p → 玫红脉冲倒流、面板按梯度发光 → 局部按 ΔW 变色。按 ＋ 拆开这一步；拖动左上角的曲线可以来回看。</span>`,
      `Step <b>${fmtInt(t)}</b> of ${fmtInt(m.train.steps)}. Loss on this batch <b>${c.loss.toFixed(3)}</b>, on the held-out validation set <b>${c.valLoss.toFixed(3)}</b>; learning rate ${sciSup(c.lr, 3)}, ${lrPhase}.<br>${runNarrative(X, k)}<br><span class="dimmed">Each beat runs one whole step on the machine: row 0’s characters enter the tray → a pulse climbs layer by layer (lens screens light up) → the pillars on top rise → −ln p → a rose pulse flows back and panels glow with their gradients → crops change color by ΔW. Press + to open this step; drag the curve at top left to scrub. The model learns only from classical Chinese poems, so everything it reads and writes is Chinese.</span>`);
  }
  if (!X.has('st', k)) return L('<span class="dimmed">正在载入这一步的真实记录…</span>', '<span class="dimmed">Loading the real record of this step…</span>');
  switch (s.ph) {
    case 'batch': return batchExplain(s, X, k);
    case 'fwd': return fwdExplain(s, X, k);
    case 'loss': return lossExplain(s, X, k);
    case 'bwd': return bwdExplain(s, X, k);
    case 'upd': return updExplain(s, X, k);
  }
  return '';
}

function batchExplain(s, X, k) {
  const m = X.meta, row = Array.from({ length: 65 }, (_, i) => X.ch(X.row(k, i))).join('');
  if (!s.sub) return L(`从 ${(m.corpus.chars.train / 1e4).toFixed(0)} 万字的训练语料里随机挑 ${m.train.batch} 处，每处从一首诗的开头起连续取 ${m.train.seq + 1} 个字（会跨过好几首诗，诗与诗之间是 ⏎）。<br>托盘里是第 0 行的前 65 个字（真实的字）：「${esc(row.slice(0, 26))}…」。<b>其余 63 行没有记录</b>，画成托盘下面一叠空白的薄片。一个批次 = ${fmtInt(m.train.tokensPerStep)} 道“猜下一个字”的题。`,
    `Pick ${m.train.batch} random spots in the ${compact(m.corpus.chars.train, 1)}-character corpus, each at the start of a poem, and take ${m.train.seq + 1} consecutive characters from each (spanning several poems, separated by ⏎).<br>The tray holds the first 65 characters of row 0 (the real ones): 「${esc(row.slice(0, 26))}…」. <b>The other 63 rows weren’t recorded</b> — they’re the blank sheets under the tray. One batch = ${fmtInt(m.train.tokensPerStep)} “guess the next character” questions.`);
  if (s.sub === 'row') return L(`第 0 行：「${esc(row)}」<br>⏎ 是 &lt;|endoftext|&gt;，和 Qwen3 一样用它分隔文档。模型并不知道“诗”是什么，只看到一长串字；字块按 16 个一排摆成 4 排（第 0–15 个字在最后一排），64 根光纤就从它们底下开始。`, `Row 0: 「${esc(row)}」<br>⏎ is &lt;|endoftext|&gt;, which separates documents just as in Qwen3. The model has no idea what a “poem” is — it just sees a long string of characters. The tiles sit 16 to a row in 4 rows (characters 0–15 in the back row), and the 64 fibers start right under them.`);
  if (s.sub === 'ids') {
    const ex = Array.from({ length: 8 }, (_, i) => `${X.ch(X.row(k, i))}→${X.row(k, i)}`).join(L('，', ', '));
    return L(`<b>字级分词</b>：每个字查词表换成一个编号（${esc(ex)}……）。词表按字频排序：「，」是 1 号、「。」是 2 号……一共 ${fmtInt(m.model.vocab)} 个。编号就是嵌入表 E 的行号——E 平躺在地上往后伸，越常见的字越靠前。<br><span class="dimmed">真实的 Qwen3 用 15 万词表的 BPE，一个词元可能是半个字、一个字或一个词。</span>`, `<b>Character-level tokenization</b>: each character is looked up and replaced by an ID (${esc(ex)}…). The vocabulary is sorted by frequency: “，” is 1, “。” is 2… ${fmtInt(m.model.vocab)} in all. The ID is the row number in the embedding table E, which lies on the floor stretching backward — the commoner the character, the closer to the front.<br><span class="dimmed">Real Qwen3 uses a 150K-entry BPE vocabulary, where a token can be part of a character, a character or a word.</span>`);
  }
  return L(`<b>错开一位</b>：输入是第 0–${m.train.seq - 1} 个字，答案是第 1–${m.train.seq} 个字。托盘前面那排绿色的字就是每个位置的答案（从“下一个字”滑过来）。<br>位置 i 只能看到前 i+1 个字，要猜第 i+1 个；一行 ${m.train.seq} 道题，在一次前向里同时作答（教师强制）。`, `<b>Shift by one</b>: the input is characters 0–${m.train.seq - 1}, the answers are characters 1–${m.train.seq}. The green row in front of the tray holds each position’s answer (slid over from “the next character”).<br>Position i sees only the first i+1 characters and must guess character i+1; ${m.train.seq} questions per row, answered all at once in one forward pass (teacher forcing).`);
}

function fwdExplain(s, X, k) {
  const m = X.meta;
  if (!s.sub) return L(`${m.train.batch} × ${m.train.seq} 个位置一起送进 6 层 Transformer，<b>每个位置同时预测它的下一个字</b>，因果遮罩保证它看不到后面。<br>机器上：64 根光纤（第 0 行）上的青色光环一层层往上，环的大小是这里残差的长度 ‖h‖（真实）；每过一个层边界，左边的透镜屏写出这一层此刻最想写的字（绿 = 正好是答案）。`, `All ${m.train.batch} × ${m.train.seq} positions go through the 6-layer Transformer together, and <b>every position predicts its next character at once</b>; the causal mask keeps it from peeking ahead.<br>On the machine: cyan rings climb the 64 fibers (row 0), sized by the residual length ‖h‖ there (real); past each layer boundary the lens screen on the left writes what this layer would guess right now (green = exactly the answer).`);
  if (s.f != null) return featFwdBwd(s, X, k);
  if (s.c != null && s.sub !== 'emb' && s.sub !== 'head') return cropExplain(s, X, k);
  if (s.sub === 'emb') {
    const st = lensStats(X, k, 0);
    return L(`<b>查嵌入表</b>：64 个字各取 E 的一行（7478 行 × 256 维；光束飞向它在长桌上的行号，生僻字飞得远）。这就是残差流的起点 h₀。<br>透镜屏（嵌入之后）：只靠 E 自己（输出层共用 E），最想写的几乎就是<b>输入的字本身</b>——第 0 层之前它还不会“往下猜”。猜中 ${P(st.acc)}。${s.c === 0 ? `<br>${cropBlurb(X, k, 0)}` : ''}`, `<b>Embedding lookup</b>: each of the 64 characters fetches its row of E (7,478 rows × 256 dims; the beam flies to its row on the long table — rare characters fly far). This is the start of the residual stream, h₀.<br>Lens screen (after embedding): with only E itself (the output layer shares E), its top guess is almost always <b>the input character itself</b> — before layer 0 it can’t yet “guess ahead”. Right: ${P(st.acc)}.${s.c === 0 ? `<br>${cropBlurb(X, k, 0)}` : ''}`);
  }
  if (s.sub === 'head') {
    const st = lensStats(X, k, 6), rs = rowStats(X, k);
    return L(`<b>输出头</b>：最后的 RMSNorm，再乘 Eᵀ（和最底下的 E 是同一张表），给 7478 个字各打一个分，softmax 变成概率。<br>顶上的柱子 = 正确答案的概率，柱顶的字 = 它最想写的字（绿 = 猜对）。第 0 行前 64 个位置：猜中第一名 ${P(st.acc)}，正确答案平均 ${P(st.p)}，平均 −ln p = ${rs.mean.toFixed(3)}。`, `<b>Output head</b>: the final RMSNorm, then multiply by Eᵀ (the same table as E at the bottom) to score all 7,478 characters; softmax turns the scores into probabilities.<br>Pillars on top = probability of the right answer; the character on each = its top guess (green = right). First 64 positions of row 0: top-1 right ${P(st.acc)}, right answer averages ${P(st.p)}, mean −ln p = ${rs.mean.toFixed(3)}.`);
  }
  const l = s.L;
  if (s.sub === 'attn') {
    const a = lensStats(X, k, l);
    return L(`<b>第 ${l} 层 · 注意力</b>：RMSNorm（γ₁）→ W_q / W_k / W_v（扫描线扫过）→ 4 个查询头各自决定看前面哪些字（GQA：每 2 个查询头共用 1 组 K/V，所以 W_k、W_v 只有 W_q 的一半宽）→ W_o → ⊕ 加回残差。<br>${headLine(X, k, l)}<br>进这一层之前的透镜读数：猜中 ${P(a.acc)}，平均 ${P(a.p)}。<span class="dimmed">这块面板只显示统计量；第 2 层的 W_q 角上有真实的 48 × 48。</span>`, `<b>Layer ${l} · attention</b>: RMSNorm (γ₁) → W_q / W_k / W_v (scan lines sweep) → each of the 4 query heads decides which earlier characters to look at (GQA: every 2 query heads share 1 K/V pair, so W_k and W_v are half as wide as W_q) → W_o → ⊕ back into the residual.<br>${headLine(X, k, l)}<br>Lens reading before this layer: right ${P(a.acc)}, average ${P(a.p)}. <span class="dimmed">This panel shows statistics only; layer 2’s W_q has a real 48 × 48 corner.</span>`);
  }
  const b = lensStats(X, k, l + 1), a = lensStats(X, k, l);
  return L(`<b>第 ${l} 层 · 前馈（SwiGLU）</b>：RMSNorm（γ₂）→ W_gate、W_up 把 256 维升到 768 维 → silu(gate) ⊙ up → W_down 降回 256 维 → ⊕ 加回残差。<br>这一层的透镜读数：猜中 ${P(a.acc)} → <b>${P(b.acc)}</b>，正确答案平均 ${P(a.p)} → <b>${P(b.p)}</b>；残差长度 ‖h‖ 平均 ${residMean(X, k, l + 1).toFixed(1)}。`, `<b>Layer ${l} · FFN (SwiGLU)</b>: RMSNorm (γ₂) → W_gate and W_up lift 256 dims to 768 → silu(gate) ⊙ up → W_down back to 256 → ⊕ into the residual.<br>Lens reading across this layer: right ${P(a.acc)} → <b>${P(b.acc)}</b>, right answer ${P(a.p)} → <b>${P(b.p)}</b>; mean residual length ‖h‖ ${residMean(X, k, l + 1).toFixed(1)}.`);
}

function cropBlurb(X, k, c) {
  const sc = X.meta.ckpts[k].wScale[c] * 127;
  return L(`角上那块 48 × 48 是真实数值（${c === 0 ? '最常见的 48 个字 × 前 48 维' : '左上角'}），此刻最大 |w| = ${f4(sc)}。`, `The 48 × 48 corner holds real values (${c === 0 ? 'the 48 most common characters × first 48 dims' : 'top-left corner'}); max |w| right now = ${f4(sc)}.`);
}

function cropExplain(s, X, k) {
  const c = s.c, m = X.meta, sw = m.ckpts[k].wScale[c] * 127, sg = m.ckpts[k].gScale[c] * 127;
  const what = c === 0 ? L('嵌入 E：每一行是一个字（左边写着），每一列是一维', 'Embedding E: each row is a character (written on the left), each column a dimension') : c === 1 ? L('第 2 层 W_q 左上角：行 = 输入第 0–47 维，列 = 输出第 0–47 维（第 0 个查询头的前 48 维）', 'Layer 2 W_q, top-left: rows = input dims 0–47, columns = output dims 0–47 (the first 48 dims of query head 0)') : L('第 4 层 W_down 左上角：行 = 768 个神经元里的第 0–47 个，列 = 输出第 0–47 维', 'Layer 4 W_down, top-left: rows = neurons 0–47 of 768, columns = output dims 0–47');
  if (s.ph === 'fwd') return L(`<b>放大一块真实的权重</b>：${what}。48 × 48 = 2,304 个方块，颜色是正负（琥珀正、蓝负），高度是 |w|；此刻最大 |w| = ${f4(sw)}。<br>前向时它和输入相乘（扫描线一列列扫过）。悬停任意方块看 W、这一步的梯度 ∇ 和到下一个检查点的 ΔW。<br><span class="dimmed">整块矩阵太大，只导出了这 48 × 48；按 int8 存储，误差约 1%。</span>`, `<b>Magnify one real block of weights</b>: ${what}. 48 × 48 = 2,304 cubes; color = sign (amber +, blue −), height = |w|; max |w| right now = ${f4(sw)}.<br>In the forward pass it multiplies the input (the scan line sweeps the columns). Hover any cube for W, this step’s gradient ∇ and the ΔW to the next checkpoint.<br><span class="dimmed">The whole matrix is too big; only this 48 × 48 was exported, stored as int8 with about 1% error.</span>`);
  if (s.ph === 'bwd') return L(`<b>这块的梯度</b>：每个方块的玫红（正）/ 紫（负）光是它这一步真实的 ∂L/∂w（裁剪前），这块里最大 |∇| = ${sciSup(sg, 3)}。<br>矩阵的梯度 = 流进来的“上游梯度”和它的“输入”做外积，再把批次里 ${fmtInt(m.train.tokensPerStep)} 个位置加起来——所以整行、整列一起亮。`, `<b>This block’s gradient</b>: each cube’s rose (+) / violet (−) glow is its real ∂L/∂w this step (before clipping); largest |∇| here = ${sciSup(sg, 3)}.<br>A matrix’s gradient = the outer product of the incoming “upstream gradient” and its “input”, summed over all ${fmtInt(m.train.tokensPerStep)} positions in the batch — which is why whole rows and columns light up together.`);
  const sp = X.dwSpan(k);
  if (!sp) return L('最后一个检查点：后面没有记录，看不到它之后怎么变。', 'Last checkpoint: nothing was recorded after it, so we can’t see how it changes next.');
  return L(`<b>这块的 ΔW</b>：方块按真实的 ΔW 顶出（ΔW &gt; 0）或沉下，再落到新值。${sp[1] - sp[0] === 1 ? `第 ${sp[0] + 1} → ${sp[1] + 1} 步正好一步，这就是 AdamW 一步的更新${sp[0] < 1 ? '：第 1 步 m̂/√v̂ = ±1，每个方块挪的距离都一样（一个 lr）' : ''}。` : `<b>到下一个检查点</b>（第 ${sp[0] + 1} → ${sp[1] + 1} 步）累计 ${sp[1] - sp[0]} 步的变化；单步的 ΔW 没有记录。`}<br><span class="dimmed">ΔW 是导出时用两个检查点的 float32 权重相减算出来的（int8 的局部直接相减，前几个检查点全是量化误差）。</span>`, `<b>This block’s ΔW</b>: cubes push out (ΔW &gt; 0) or sink by the real ΔW, then settle at their new values. ${sp[1] - sp[0] === 1 ? `Step ${sp[0] + 1} → ${sp[1] + 1} is exactly one step: this is one AdamW update${sp[0] < 1 ? ' — at step 1 m̂/√v̂ = ±1, so every cube moves the same distance (one lr)' : ''}.` : `The change <b>up to the next checkpoint</b> (step ${sp[0] + 1} → ${sp[1] + 1}), ${sp[1] - sp[0]} steps combined; single-step ΔW wasn’t recorded.`}<br><span class="dimmed">ΔW was computed at export time by subtracting the float32 weights of two checkpoints (subtracting the int8 crops would be pure quantization error for the first few checkpoints).</span>`);
}

function featFwdBwd(s, X, k) {
  const a = X.adam(k, s.f), nm = esc(featLabel(X.feats[s.f], 'tiny'));
  if (s.ph === 'fwd') return L(`跟踪的 4 个权重之一：<b>${nm}</b>。镜头推到它在面板上的真实位置（十字线 = 它所在的那一行、那一列）。这一步开始时 w = <b>${f4(a.w0)}</b>。<br>底部算式板是它每 4 步一个点的一生：w、梯度、m、√v。`, `One of the 4 tracked weights: <b>${nm}</b>. The camera moves to its real spot on the panel (crosshair = its row and column). At the start of this step w = <b>${f4(a.w0)}</b>.<br>The board at the bottom shows its life, one point every 4 steps: w, gradient, m, √v.`);
  return L(`<b>${nm}</b> 这一步的梯度：∂L/∂w = <b>${f4(a.gRaw)}</b>（裁剪前），裁剪后 ${f4(a.g)}。<br>它是批次里 ${fmtInt(X.meta.train.tokensPerStep)} 个位置的贡献加起来的；${a.g > 0 ? '正' : '负'}号表示 w ${a.g > 0 ? '减小' : '增大'}一点能让损失下降。`, `This step’s gradient for <b>${nm}</b>: ∂L/∂w = <b>${f4(a.gRaw)}</b> (before clipping), ${f4(a.g)} after.<br>It sums the contributions of the batch’s ${fmtInt(X.meta.train.tokensPerStep)} positions; the ${a.g > 0 ? 'positive' : 'negative'} sign means ${a.g > 0 ? 'lowering' : 'raising'} w a little would reduce the loss.`);
}

function lossExplain(s, X, k) {
  const m = X.meta, rs = rowStats(X, k);
  if (s.sub === 'mean') return L(`整批 ${fmtInt(m.train.tokensPerStep)} 个位置的 −ln p 取平均：<b>${X.loss(k).toFixed(4)}</b>（损失管）。第 0 行 128 个位置的平均是 ${X.rowLoss(k).toFixed(4)}，前 64 个是 ${rs.mean.toFixed(4)}。<br>完全瞎猜是 ln ${m.model.vocab} = ${Math.log(m.model.vocab).toFixed(2)}（管上的玫红环）。`, `Average −ln p over all ${fmtInt(m.train.tokensPerStep)} positions in the batch: <b>${X.loss(k).toFixed(4)}</b> (the gauge). Row 0’s 128 positions average ${X.rowLoss(k).toFixed(4)}, the first 64 ${rs.mean.toFixed(4)}.<br>A blind guess costs ln ${m.model.vocab} = ${Math.log(m.model.vocab).toFixed(2)} (the rose ring on the gauge).`);
  if (s.i == null) return L(`每个位置的损失 = <b>−ln p(正确的下一个字)</b>。柱子右边 64 根是第 0 行前 64 个位置各自的 −ln p：最容易的是位置 ${rs.best}「${ch(X, X.row(k, rs.best))}→${ch(X, X.tgt(k, rs.best))}」（${(-Math.log(Math.max(X.rowP(k, rs.best), 1e-12))).toFixed(3)}），最难的是位置 ${rs.worst}「${ch(X, X.row(k, rs.worst))}→${ch(X, X.tgt(k, rs.worst))}」（${(-Math.log(Math.max(X.rowP(k, rs.worst), 1e-12))).toFixed(2)}）。<br>标点、常见搭配几乎不扣分，诗的第一个字最难猜。按 ＋ 一个位置一个位置看。`, `The loss at each position = <b>−ln p(the correct next character)</b>. The 64 bars to the right of the pillars are row 0’s first 64 positions: easiest is position ${rs.best} 「${ch(X, X.row(k, rs.best))}→${ch(X, X.tgt(k, rs.best))}」 (${(-Math.log(Math.max(X.rowP(k, rs.best), 1e-12))).toFixed(3)}), hardest is position ${rs.worst} 「${ch(X, X.row(k, rs.worst))}→${ch(X, X.tgt(k, rs.worst))}」 (${(-Math.log(Math.max(X.rowP(k, rs.worst), 1e-12))).toFixed(2)}).<br>Punctuation and common pairings cost almost nothing; the first character of a poem is hardest. Press + to go position by position.`);
  const i = s.i, p = X.rowP(k, i), tg = X.tgt(k, i);
  let ctx = '';
  for (let j = Math.max(0, i - 9); j <= i; j++) ctx += X.ch(X.row(k, j));
  const top = X.rowTop(k, i).map((tp) => `${tp.id === tg ? '<b>' : ''}${ch(X, tp.id)} ${fmtP(tp.p)}${tp.id === tg ? '</b>' : ''}`).join(L('、', ', '));
  return L(`位置 <b>${i}</b>：上文「…${esc(ctx)}」，正确答案「<b>${ch(X, tg)}</b>」（${tg} 号）。<br>模型给它的概率 ${P(p)}，损失 −ln p = <b>${(-Math.log(Math.max(p, 1e-12))).toFixed(3)}</b>。<br>前 5 名（立在顶上 Eᵀ 的影子上，按字的编号排在 7478 行里的真实位置）：${top}。`, `Position <b>${i}</b>: context “…${esc(ctx)}”, correct answer “<b>${ch(X, tg)}</b>” (#${tg}).<br>The model gives it probability ${P(p)}, so −ln p = <b>${(-Math.log(Math.max(p, 1e-12))).toFixed(3)}</b>.<br>Top 5 (standing on the shadow of Eᵀ at the top, each at its real row among the 7,478): ${top}.`);
}

function bwdExplain(s, X, k) {
  const m = X.meta;
  if (!s.sub) return L(`<b>loss.backward()</b> 用链式法则，从输出头开始一层层往回，把损失对每个参数的梯度算出来。所有 ${fmtInt(m.model.params)} 个梯度合起来的长度 ‖g‖ = <b>${X.gnorm(k).toFixed(3)}</b>。<br>机器上：玫红光环沿光纤往下，环的大小是这里的 ‖∂L/∂h‖（真实）；每块面板按自己真实的 ‖∇W‖ 发光（这一步最大的那块 = 满格），有局部的角上换成梯度热力。`, `<b>loss.backward()</b> uses the chain rule, starting at the output head and working back layer by layer, to get the gradient of the loss for every parameter. Combined length of all ${fmtInt(m.model.params)} gradients ‖g‖ = <b>${X.gnorm(k).toFixed(3)}</b>.<br>On the machine: rose rings flow down the fibers, sized by ‖∂L/∂h‖ there (real); each panel glows by its own real ‖∇W‖ (this step’s largest = full), and the crops switch to gradient heat.`);
  if (s.f != null) return featFwdBwd(s, X, k);
  if (s.c != null && s.sub !== 'emb' && s.sub !== 'head') return cropExplain(s, X, k);
  if (s.sub === 'head') return L(`梯度从损失流进输出头：最后的 RMSNorm γ_f（‖∇‖ = ${sciSup(X.tgrad(k, -1, 'nf'), 2)}）和 Eᵀ。Eᵀ 就是 E，所以 E 的梯度在这里先收到一份（给每个字打分那一路），到最底下查表那一路再收一份。<br>最后一层之后每个位置的 ‖∂L/∂h‖ 平均 ${sciSup(residMean(X, k, 6, true), 3)}。`, `The gradient flows from the loss into the output head: the final RMSNorm γ_f (‖∇‖ = ${sciSup(X.tgrad(k, -1, 'nf'), 2)}) and Eᵀ. Eᵀ is E, so E gets one share here (the scoring path) and another at the bottom (the lookup path).<br>Mean ‖∂L/∂h‖ per position after the last layer: ${sciSup(residMean(X, k, 6, true), 3)}.`);
  if (s.sub === 'emb') return L(`梯度流到最底下：每个位置的 ∂L/∂h₀ 加到它那个字在 E 里的那一行上（只有这一批里出现过的字的行才有梯度）。加上输出头那一路，E 的梯度范数是 ${sciSup(X.tgrad(k, -1, 'E'), 3)}——通常是全模型最大的。${s.c === 0 ? `<br>角上的局部换成了梯度热力：最常见的 48 个字几乎每一批都出现，所以它们的行一直有梯度。` : ''}`, `The gradient reaches the bottom: each position’s ∂L/∂h₀ is added to its character’s row of E (only rows of characters that appear in this batch get a gradient). With the output-head path added, E’s gradient norm is ${sciSup(X.tgrad(k, -1, 'E'), 3)} — usually the largest in the model.${s.c === 0 ? '<br>The corner crop now shows gradient heat: the 48 most common characters appear in almost every batch, so their rows always get a gradient.' : ''}`);
  const l = s.L;
  if (s.sub === 'ffn') return L(`梯度流过<b>第 ${l} 层 · 前馈</b>：残差连接让它直接抄近路往下，前馈这一路再贡献一份。<br>这几块的 ‖∇W‖：${gradList(X, k, l, ['down', 'gate', 'up', 'ln2'])}。这一层之后每个位置的 ‖∂L/∂h‖ 平均 ${sciSup(residMean(X, k, l + 1, true), 3)}。`, `The gradient flows through <b>layer ${l} · FFN</b>: the residual connection lets it shortcut straight down, and the FFN path adds its own share.<br>‖∇W‖ of these blocks: ${gradList(X, k, l, ['down', 'gate', 'up', 'ln2'])}. Mean ‖∂L/∂h‖ per position after this layer: ${sciSup(residMean(X, k, l + 1, true), 3)}.`);
  return L(`梯度流过<b>第 ${l} 层 · 注意力</b>：W_o → 4 个头 → W_q / W_k / W_v → γ₁。<br>这几块的 ‖∇W‖：${gradList(X, k, l, ['o', 'q', 'k', 'v', 'ln1'])}；q_norm / k_norm 的 γ：${gradList(X, k, l, ['qn', 'kn'])}。这一层之前每个位置的 ‖∂L/∂h‖ 平均 ${sciSup(residMean(X, k, l, true), 3)}。`, `The gradient flows through <b>layer ${l} · attention</b>: W_o → the 4 heads → W_q / W_k / W_v → γ₁.<br>‖∇W‖ of these blocks: ${gradList(X, k, l, ['o', 'q', 'k', 'v', 'ln1'])}; q_norm / k_norm γ: ${gradList(X, k, l, ['qn', 'kn'])}. Mean ‖∂L/∂h‖ per position before this layer: ${sciSup(residMean(X, k, l, true), 3)}.`);
}

function updExplain(s, X, k) {
  const m = X.meta, lr = X.lr(k);
  const sp = X.dwSpan(k);
  const span = !sp ? L('最后一个检查点之后没有记录，面板不再变化。', 'Nothing was recorded after the last checkpoint, so the panels stay put.') : sp[1] - sp[0] === 1 ? L(`第 ${sp[0] + 1} → ${sp[1] + 1} 步正好一步。`, `Step ${sp[0] + 1} → ${sp[1] + 1} is exactly one step.`) : L(`面板的紫光和局部的变化是<b>到下一个检查点</b>（第 ${sp[0] + 1} → ${sp[1] + 1} 步，累计 ${sp[1] - sp[0]} 步）的真实 RMS(ΔW) / ΔW；单步的没有记录。`, `The panels’ violet glow and the crops’ change are the real RMS(ΔW) / ΔW <b>up to the next checkpoint</b> (step ${sp[0] + 1} → ${sp[1] + 1}, ${sp[1] - sp[0]} steps combined); single steps weren’t recorded.`);
  if (!s.sub) return L(`AdamW 用梯度更新<b>每一个</b>权重，${fmtInt(m.model.params)} 个各算各的。学习率 ${sciSup(lr, 3)}（${X.step(k) < m.train.warmup ? '预热中' : '余弦退火中'}）。<br>${span}<br><span class="dimmed">按 ＋ 先看裁剪，再一块块看局部，最后细到一个权重的算式。</span>`, `AdamW updates <b>every single</b> weight with its gradient — all ${fmtInt(m.model.params)}, each on its own. Learning rate ${sciSup(lr, 3)} (${X.step(k) < m.train.warmup ? 'warming up' : 'cosine annealing'}).<br>${span}<br><span class="dimmed">Press + for clipping, then the crops one by one, then one weight’s formula.</span>`);
  if (s.sub === 'clip') return L(`所有梯度合起来的长度 ‖g‖ = <b>${X.gnorm(k).toFixed(4)}</b>。${X.clip(k) < 1 ? `超过 1.0，全部梯度乘以 ${X.clip(k).toFixed(5)} 缩到 1.0。` : '没超过 1.0，不用裁剪。'}<br>裁剪只改长度、不改方向，防止某一批数据把模型“踹”得太远。刚开始的几百步梯度大，几乎每步都被裁剪。`, `Combined length of all gradients ‖g‖ = <b>${X.gnorm(k).toFixed(4)}</b>. ${X.clip(k) < 1 ? `Above 1.0, so every gradient is multiplied by ${X.clip(k).toFixed(5)} to bring it to 1.0.` : 'Not above 1.0, so no clipping.'}<br>Clipping changes only the length, not the direction, so one batch can’t kick the model too far. In the first few hundred steps the gradients are big and get clipped almost every step.`);
  if (s.f != null) return adamExplain(s, X, k);
  if (s.c != null) return cropExplain({ ...s, ph: 'upd' }, X, k);
  return L(`AdamW 一次更新全部 ${fmtInt(m.model.params)} 个参数，学习率 ${sciSup(lr, 3)}。m̂/√v̂ 通常在 1 左右，所以每个权重每步大约挪一个 lr，不管它的梯度是大是小；矩阵另外做权重衰减（λ = 0.1），RMSNorm 的 γ 不做。<br>${span}`, `AdamW updates all ${fmtInt(m.model.params)} parameters at once, learning rate ${sciSup(lr, 3)}. m̂/√v̂ is usually around 1, so each weight moves about one lr per step, whether its gradient is big or small; matrices also get weight decay (λ = 0.1), RMSNorm γ doesn’t.<br>${span}`);
}

function adamExplain(s, X, k) {
  const a = X.adam(k, s.f), lr = X.lr(k), tt = a.t, b1 = 0.9, b2 = 0.95, eps = 1e-8;
  const name = `<b>${esc(featLabel(X.feats[s.f], 'tiny'))}</b>`;
  switch (s.mi) {
    case 'g': return L(`${name}的梯度（裁剪后）g = <b>${f4(a.g)}</b>：w 增大一点点，损失就会${a.g > 0 ? '上升' : '下降'}约 |g| 倍。所以它要往${a.g > 0 ? '小' : '大'}的方向挪。`, `The (clipped) gradient of ${name} is g = <b>${f4(a.g)}</b>: nudge w up a tiny bit and the loss ${a.g > 0 ? 'rises' : 'falls'} by about |g| times as much. So it should move ${a.g > 0 ? 'down' : 'up'}.`);
    case 'm': return L(`<b>一阶动量</b>（梯度的滑动平均）：m = 0.9 × ${f4(a.m0)} + 0.1 × ${f4(a.g)} = <b>${f4(a.m)}</b>。它记住过去的方向，抹平批次之间的噪声。`, `<b>First moment</b> (moving average of the gradient): m = 0.9 × ${f4(a.m0)} + 0.1 × ${f4(a.g)} = <b>${f4(a.m)}</b>. It remembers past directions and smooths out batch-to-batch noise.`);
    case 'v': return L(`<b>二阶动量</b>（梯度平方的滑动平均）：v = 0.95 × ${f4(a.v0)} + 0.05 × (${f4(a.g)})² = <b>${f4(a.v)}</b>。√v 是这个权重梯度的“典型大小”。`, `<b>Second moment</b> (moving average of the squared gradient): v = 0.95 × ${f4(a.v0)} + 0.05 × (${f4(a.g)})² = <b>${f4(a.v)}</b>. √v is the “typical size” of this weight’s gradient.`);
    case 'bc': return L(`<b>偏差校正</b>：m、v 从 0 开始累积，前几步偏小，所以除以 (1 − βᵗ)，t = ${tt}：m̂ = <b>${f4(a.mh)}</b>，v̂ = <b>${f4(a.vh)}</b>。${tt === 1 ? '第 1 步 m̂ 正好等于 g、v̂ 正好等于 g²：更新量只剩下符号。' : `t 越大，除数越接近 1（现在 ${(1 - b1 ** tt).toPrecision(4)} / ${(1 - b2 ** tt).toPrecision(4)}）。`}`, `<b>Bias correction</b>: m and v start from 0 and come out too small at first, so divide by (1 − βᵗ), t = ${tt}: m̂ = <b>${f4(a.mh)}</b>, v̂ = <b>${f4(a.vh)}</b>. ${tt === 1 ? 'At step 1, m̂ equals g and v̂ equals g² exactly: only the sign is left.' : `The larger t, the closer the divisors get to 1 (now ${(1 - b1 ** tt).toPrecision(4)} / ${(1 - b2 ** tt).toPrecision(4)}).`}`);
    case 'dw': {
      const ad = -lr * a.mh / (Math.sqrt(a.vh) + eps), dc = -lr * a.wd * a.w0;
      return L(`Δw = −lr × m̂/(√v̂+ε) − lr × λ × w = ${sciSup(ad, 3)} ${dc >= 0 ? '+' : '−'} ${sciSup(Math.abs(dc), 3)} = <b>${sciSup(ad + dc, 3)}</b>（lr = ${sciSup(lr, 3)}）。${a.wd ? `权重衰减 λ = ${a.wd} 每步把 w 往 0 拉一点。` : 'RMSNorm 的 γ 按惯例不做权重衰减。'}`, `Δw = −lr × m̂/(√v̂+ε) − lr × λ × w = ${sciSup(ad, 3)} ${dc >= 0 ? '+' : '−'} ${sciSup(Math.abs(dc), 3)} = <b>${sciSup(ad + dc, 3)}</b> (lr = ${sciSup(lr, 3)}). ${a.wd ? `Weight decay λ = ${a.wd} pulls w a little toward 0 every step.` : 'By convention RMSNorm’s γ gets no weight decay.'}`);
    }
    case 'write': return L(`写回：w = ${f4(a.w0)} → <b>${f4(a.w1)}</b>，改动只有原值的 ${sciSup(Math.abs((a.w1 - a.w0) / (a.w0 || 1)), 2)}。导出时用 PyTorch 同样的 float32 运算把这条算式在全部 4000 步上核对过，最大误差 ${X.meta.train.adamCheckUlp} ulp。`, `Write back: w = ${f4(a.w0)} → <b>${f4(a.w1)}</b>, a change of only ${sciSup(Math.abs((a.w1 - a.w0) / (a.w0 || 1)), 2)} of its value. At export this formula was checked on all 4,000 steps with PyTorch’s own float32 ops: max error ${X.meta.train.adamCheckUlp} ulp.`);
  }
  return '';
}

/* ---------------------------------------------------------------- 变量监视 */

export function watch(s, X, ctx, depth) {
  const m = X.meta, k = s.k;
  const rows = [['wh', L('TINY QWEN3 · 唐宋诗', 'TINY QWEN3 · POETRY')], ['step', `${fmtInt(X.step(k) + 1)} / ${fmtInt(m.train.steps)}`], ['lr', sciSup(X.lr(k), 3)], ['loss', X.loss(k).toFixed(4)], ['val loss', X.valLoss(k).toFixed(4)], ['‖g‖', X.gnorm(k).toFixed(4)], ['clip', X.clip(k) < 1 ? `×${X.clip(k).toFixed(5)}` : '—'], ['tokens', fmtInt(X.step(k) * m.train.tokensPerStep)]];
  if (s.ph === 'run' || !X.has('st', k)) return rows;
  if (s.ph === 'loss' && s.i != null) {
    const p = X.rowP(k, s.i);
    rows.push(['wh', 'POSITION'], ['i', s.i], ['target', `${X.ch(X.tgt(k, s.i))} #${X.tgt(k, s.i)}`], ['p', p.toPrecision(4)], ['−ln p', (-Math.log(Math.max(p, 1e-12))).toFixed(4)]);
  }
  if ((s.ph === 'fwd' || s.ph === 'bwd') && s.sub) {
    const b = s.sub === 'emb' ? 0 : s.sub === 'head' ? 6 : s.ph === 'fwd' ? s.L + 1 : s.L;
    const ls = lensStats(X, k, b);
    rows.push(['wh', L(`边界 ${b}`, `BOUNDARY ${b}`)], ['mean ‖h‖', residMean(X, k, b).toFixed(2)], ['lens top-1', fmtP(ls.acc)], ['lens p', fmtP(ls.p)]);
    if (s.ph === 'bwd') rows.push(['mean ‖∂L/∂h‖', sciSup(residMean(X, k, b, true), 3)]);
    if (s.sub === 'attn' || s.sub === 'ffn') for (const kd of s.sub === 'attn' ? ['q', 'k', 'v', 'o'] : ['gate', 'up', 'down']) rows.push([`‖∇${KIND_LABEL[kd]}‖`, sciSup(X.tgrad(k, s.L, kd), 3)]);
    if (s.sub === 'emb' || s.sub === 'head') rows.push(['‖∇E‖', sciSup(X.tgrad(k, -1, 'E'), 3)]);
  }
  if (s.ph === 'upd' && s.f != null) {
    const a = X.adam(k, s.f);
    rows.push(['wh', L('ADAMW · 一个权重', 'ADAMW · ONE WEIGHT')], ['t', a.t], ['w', f4(a.w0)], ['g', f4(a.g)], ['m', f4(a.m)], ['v', f4(a.v)], ['m̂', f4(a.mh)], ['v̂', f4(a.vh)], ['λ', a.wd], ['Δw', sciSup(a.w1 - a.w0, 4)], ['w′', f4(a.w1)]);
  } else if (s.ph === 'upd') {
    const sp = X.dwSpan(k);
    rows.push(['wh', 'ΔW'], ['span', sp ? `${sp[0] + 1} → ${sp[1] + 1}` : '—'], ['RMS(ΔW) max', X.has('dw', k) && sp ? sciSup(X.dwRmsMax(k), 3) : '—']);
  }
  return rows;
}
