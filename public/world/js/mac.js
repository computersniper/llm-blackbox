// 一次乘加：把某一层的某一个输出拆成 y = b + Σ x·w，每一项的 x、w 都是这一帧真实的激活值和真实的权重。
// 加完和前向算出来的值比一比（float32 的前向 vs 这里用双精度重新加一遍，差在 1e-6 量级）。
import { ENC, DEC, Z, KMIX, HID } from './nn.js';
import { CAR_Y } from './game.js';
import { dimOrder, dimMeaning, OP_LABEL } from './explain.js';
import { esc } from '../../js/ui.js';
import { L as Lx } from '../../js/i18n.js';

const GATE = { gi: [0, 'i', 'σ'], gf: [1, 'f', 'σ'], gg: [2, 'g', 'tanh'], go: [3, 'o', 'σ'] };
const sig = (v) => 1 / (1 + Math.exp(-v));
const RGB = ['R', 'G', 'B'];

function argmaxAbs(a) { let b = 0; for (let i = 1; i < a.length; i++) if (Math.abs(a[i]) > Math.abs(a[b])) b = i; return b; }
function argmax(a) { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }

// 这一步要拆的那个输出（用户点选过就用点选的，否则挑一个有代表性的）
export function defaultSel(op, sim, model, meta) {
  const F = sim.cur, rec = F.rec, det = sim.detail(F);
  const L = [...ENC, ...DEC].find((l) => l.id === op);
  if (op === 'd4') {
    const G = sim.next;
    const x = G ? G.snap.carX : 32;
    return { c: 1, y: CAR_Y + 4, x };
  }
  if (L) {
    const a = op[0] === 'e' ? det.enc[op] : det.dec[op];
    const i = argmax(a), H = L.outH;
    return { c: Math.floor(i / (H * H)), y: Math.floor((i % (H * H)) / H), x: i % H };
  }
  if (op === 'mu') return { j: dimOrder(meta)[0] };
  if (op === 'dfc') return { j: argmaxAbs(det.dec.dfc) };
  if (GATE[op]) {
    const off = GATE[op][0] * HID;
    return { j: argmaxAbs(rec.L.pre.subarray(off, off + HID)) };
  }
  if (op === 'mdn') { const d = sim._mdnDim ?? dimOrder(meta)[0]; return { d, k: rec.S.k[d] }; }
  return {};
}

export function buildMac(op, sel, sim, model, meta) {
  const F = sim.cur, rec = F.rec, det = sim.detail(F), T = model.T;
  const terms = [];
  let bias = 0, stored = 0, act = 'none', where = '', outLabel = 'y', ctxNote = '';
  const L = [...ENC, ...DEC].find((l) => l.id === op);
  if (L && op[0] === 'e') {
    const inp = op === 'e1' ? det.enc.x : det.enc[`e${Number(op[1]) - 1}`];
    const W = T[`${op}.weight`], { cin, cout, k, s, inH, outH } = L, { c: co, y: oy, x: ox } = sel;
    for (let ci = 0; ci < cin; ci++) for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
      const x = inp[(ci * inH + oy * s + ky) * inH + ox * s + kx], w = W[((co * cin + ci) * k + ky) * k + kx];
      terms.push({ lab: op === 'e1' ? `${RGB[ci]}${ky},${kx}` : `c${ci}·${ky},${kx}`, x, w, g: ci });
    }
    bias = T[`${op}.bias`][co];
    stored = det.enc[op][(co * outH + oy) * outH + ox];
    act = 'relu';
    where = Lx(`${OP_LABEL[op]} · 通道 ${co} · 位置 (${oy}, ${ox})`, `${OP_LABEL[op]} · channel ${co} · position (${oy}, ${ox})`);
    ctxNote = op === 'e1' ? Lx(`输入画面里 (${oy * s}…${oy * s + k - 1}, ${ox * s}…${ox * s + k - 1}) 这块 4×4 的像素，3 个颜色通道`, `the 4×4 block of input pixels at (${oy * s}…${oy * s + k - 1}, ${ox * s}…${ox * s + k - 1}), 3 color channels`) : Lx(`上一层 ${cin} 张特征图在 (${oy * s}…${oy * s + k - 1}, ${ox * s}…${ox * s + k - 1}) 的 4×4 小块`, `the 4×4 patch at (${oy * s}…${oy * s + k - 1}, ${ox * s}…${ox * s + k - 1}) in each of the previous layer’s ${cin} feature maps`);
  } else if (L) {
    const inp = op === 'd1' ? det.dec.dfc : det.dec[`d${Number(op[1]) - 1}`];
    const W = T[`${op}.weight`], { cin, cout, k, s, inH, outH } = L, { c: co, y: oy, x: ox } = sel;
    for (let ci = 0; ci < cin; ci++) {
      for (let ky = 0; ky < k; ky++) {
        if ((oy - ky) % s !== 0) continue;
        const iy = (oy - ky) / s;
        if (iy < 0 || iy >= inH) continue;
        for (let kx = 0; kx < k; kx++) {
          if ((ox - kx) % s !== 0) continue;
          const ix = (ox - kx) / s;
          if (ix < 0 || ix >= inH) continue;
          const x = inp[(ci * inH + iy) * inH + ix], w = W[((ci * cout + co) * k + ky) * k + kx];
          terms.push({ lab: inH === 1 ? `c${ci}` : `c${ci}·${iy},${ix}`, x, w, g: ci });
        }
      }
    }
    bias = T[`${op}.bias`][co];
    stored = op === 'd4' ? det.dec.d4pre[(co * outH + oy) * outH + ox] : det.dec[op][(co * outH + oy) * outH + ox];
    act = op === 'd4' ? 'sigmoid' : 'relu';
    where = op === 'd4' ? Lx(`梦里的像素 (${oy}, ${ox}) · ${['红', '绿', '蓝'][co]}色通道`, `dream pixel (${oy}, ${ox}) · ${['red', 'green', 'blue'][co]} channel`) : Lx(`${OP_LABEL[op]} · 通道 ${co} · 位置 (${oy}, ${ox})`, `${OP_LABEL[op]} · channel ${co} · position (${oy}, ${ox})`);
    ctxNote = inH === 1 ? Lx('1×1×512 的输入，每个通道和这个核在 (oy, ox) 处的那一格相乘', 'a 1×1×512 input: each channel is multiplied by this kernel’s cell at (oy, ox)') : Lx(`反卷积：只有“盖章”盖到 (${oy}, ${ox}) 的那些输入位置和核上的那一格才算进来`, `deconvolution: only the input positions whose “stamp” lands on (${oy}, ${ox}), times the matching kernel cell, count`);
  } else if (op === 'mu') {
    const d = sel.j, W = T['mu.weight'];
    for (let i = 0; i < 512; i++) terms.push({ lab: `x${i}`, x: det.enc.e4[i], w: W[d * 512 + i], g: 0 });
    bias = T['mu.bias'][d];
    stored = det.enc.mu[d];
    where = Lx(`μ 的第 ${d} 维（${dimMeaning(meta, d)}）`, `dimension ${d} of μ (${dimMeaning(meta, d)})`);
    ctxNote = Lx('卷积 4 的 512 个输出（2×2×128 摊平）', 'the 512 outputs of conv 4 (2×2×128, flattened)');
  } else if (op === 'dfc') {
    const j = sel.j, W = T['dfc.weight'];
    for (let i = 0; i < Z; i++) terms.push({ lab: `z${i}`, x: det.dec.z[i], w: W[j * Z + i], g: 0 });
    bias = T['dfc.bias'][j];
    stored = det.dec.dfc[j];
    where = Lx(`全连接的第 ${j} 个输出`, `output ${j} of the fully connected layer`);
    ctxNote = Lx('输入就是 M 预测的 ẑ 的 32 个数', 'the input is the 32 numbers of ẑ predicted by M');
  } else if (GATE[op]) {
    const [gk, nm, fn] = GATE[op], j = sel.j, r = gk * HID + j;
    const Wi = T['lstm.weight_ih_l0'], Wh = T['lstm.weight_hh_l0'];
    for (let i = 0; i < Z + 3; i++) terms.push({ lab: i < Z ? `z${i}` : `a${i - Z}`, x: rec.L.x[i], w: Wi[r * (Z + 3) + i], g: 0 });
    for (let i = 0; i < HID; i++) terms.push({ lab: `h${i}`, x: F.h[i], w: Wh[r * HID + i], g: 1 });
    bias = T['lstm.bias_ih_l0'][r] + T['lstm.bias_hh_l0'][r];
    stored = rec.L.pre[r];
    act = fn === 'σ' ? 'sigmoid' : 'tanh';
    where = Lx(`${OP_LABEL[op]} ${nm} 的第 ${j} 维`, `${OP_LABEL[op]} ${nm}, dimension ${j}`);
    ctxNote = Lx('输入是 [z（32）, 动作（3）] 和上一步的记忆 h（256），一共 291 项；偏置是 PyTorch 的两份偏置之和', 'inputs are [z (32), action (3)] and last step’s memory h (256), 291 terms in all; the bias is the sum of PyTorch’s two biases');
  } else if (op === 'mdn' || op === 'done') {
    const W = T['head.weight'];
    let idx, label;
    if (op === 'done') { idx = 3 * Z * KMIX; label = Lx('撞车 logit', 'crash logit'); act = 'sigmoid'; }
    else { idx = Z * KMIX + sel.d * KMIX + sel.k; label = Lx(`z${sel.d} 第 ${sel.k} 个高斯的中心 μ`, `center μ of Gaussian ${sel.k} for z${sel.d}`); }
    for (let i = 0; i < HID; i++) terms.push({ lab: `h${i}`, x: rec.L.h[i], w: W[idx * HID + i], g: 0 });
    bias = T['head.bias'][idx];
    stored = rec.M.raw[idx];
    where = Lx(`MDN 头 · ${label}`, `MDN head · ${label}`);
    ctxNote = Lx('输入是这一步新的记忆 h\'（256 个数）', 'the input is this step’s new memory h\' (256 numbers)');
  }
  for (const t of terms) t.p = t.x * t.w;
  return new Mac(op, sel, terms, bias, stored, act, where, ctxNote);
}

const fmt = (v, d = 4) => {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-4 || a >= 1e4) return v.toExponential(2).replace('-', '−');
  return (v < 0 ? '−' : '') + a.toFixed(d);
};

export class Mac {
  constructor(op, sel, terms, bias, stored, act, where, note) {
    Object.assign(this, { op, sel, terms, bias, stored, act, where, note });
    this.n = terms.length;
    this.order = terms.map((t, i) => i).sort((a, b) => Math.abs(terms[b].p) - Math.abs(terms[a].p));
    let s = 0, pos = 0, neg = 0;
    for (const t of terms) { s += t.p; if (t.p > 0) pos += t.p; else neg += t.p; }
    this.sum = s;
    this.pos = pos;
    this.neg = neg;
    this.pre = s + bias;
    this.out = act === 'relu' ? Math.max(0, this.pre) : act === 'sigmoid' ? sig(this.pre) : act === 'tanh' ? Math.tanh(this.pre) : this.pre;
    // 卷积层存的是 ReLU 之后的激活值，其余存的是激活之前的值
    this.post = act === 'relu';
    this.err = Math.abs((this.post ? this.out : this.pre) - stored);
    // 按 |乘积| 从大到小累加
    this.cum = [];
    let c = 0;
    for (const i of this.order) { c += terms[i].p; this.cum.push(c); }
  }

  get top() { return this.order.slice(0, 12); }

  actFormula() {
    const p = fmt(this.pre);
    if (this.act === 'relu') return `ReLU(${p}) = max(0, ${p}) = ${fmt(this.out)}`;
    if (this.act === 'sigmoid') return `σ(${p}) = 1 / (1 + e<sup>−(${p})</sup>) = ${fmt(this.out)}`;
    if (this.act === 'tanh') return `tanh(${p}) = ${fmt(this.out)}`;
    return Lx(`没有激活函数：${fmt(this.out)}`, `no activation function: ${fmt(this.out)}`);
  }

  explain(mi) {
    const nTop = Math.min(12, this.n);
    switch (mi) {
      case 'pick': return Lx(`<p>拆开这一个数：<b>${esc(this.where)}</b>。</p><p>它由 <b>${this.n}</b> 项乘积加起来，再加上偏置。${esc(this.note)}。</p><p class="dimmed">在上一层（D3 / D4）点任意一格可以换一个数来拆。</p>`, `<p>Take this one number apart: <b>${esc(this.where)}</b>.</p><p>It is the sum of <b>${this.n}</b> products plus a bias — ${esc(this.note)}.</p><p class="dimmed">Click any cell one level up (D3 / D4) to take apart a different number.</p>`);
      case 'mul': return Lx(`<p>每一项 = 输入 x × 权重 w。${this.n} 项里，绝对值最大的 ${nTop} 项列在算式板上，其余合成一行。</p><p>正的乘积往上推，负的往下拉：正项之和 <b>${fmt(this.pos)}</b>，负项之和 <b>${fmt(this.neg)}</b>。</p>`, `<p>Each term = input x × weight w. Of the ${this.n} terms, the ${nTop} largest in absolute value are listed on the board; the rest are merged into one row.</p><p>Positive products push up, negative ones pull down: the positive terms sum to <b>${fmt(this.pos)}</b>, the negative terms to <b>${fmt(this.neg)}</b>.</p>`);
      case 'sum': return Lx(`<p>全部 ${this.n} 项加起来是 <b>${fmt(this.sum)}</b>，再加偏置 ${fmt(this.bias)}，得到 <b>${fmt(this.pre)}</b>。</p><p class="dimmed">前向（float32${this.post ? '，已经过 ReLU' : ''}）算出来的是 ${fmt(this.stored, 6)}，这里用双精度重新加一遍，差 ${this.err.toExponential(1)}：加法的顺序不同，误差在 float32 的舍入范围内。</p>`, `<p>All ${this.n} terms add up to <b>${fmt(this.sum)}</b>; adding the bias ${fmt(this.bias)} gives <b>${fmt(this.pre)}</b>.</p><p class="dimmed">The forward pass (float32${this.post ? ', after ReLU' : ''}) computed ${fmt(this.stored, 6)}; re-adding here in double precision differs by ${this.err.toExponential(1)}: the additions happen in a different order, and the gap is within float32 rounding.</p>`);
      case 'act': return `<p>${this.actFormula()}</p><p>${this.act === 'relu' ? (this.pre < 0 ? Lx('加起来是负数，ReLU 把它截成 0：这个位置这一帧“没被点亮”。', 'The sum is negative, so ReLU clips it to 0: this position is “not lit” in this frame.') : Lx('ReLU 只截掉负数，正数原样通过。', 'ReLU only clips negatives; positives pass through unchanged.')) : this.act === 'sigmoid' ? (this.op === 'd4' ? Lx('sigmoid 把它压到 0–1，就是梦里这个像素这个颜色通道的亮度（×255 就是 0–255 的颜色值）。', 'The sigmoid squashes it into 0–1: the brightness of this color channel of this dream pixel (×255 gives the 0–255 color value).') : Lx('压到 0–1：门开多大。', 'Squashed into 0–1: how far the gate opens.')) : this.act === 'tanh' ? Lx('压到 −1–1。', 'Squashed into −1 to 1.') : ''}</p>`;
    }
    return '';
  }

  watch() {
    return [['wh', Lx('一次乘加', 'One multiply-add')], [Lx('项数 n', 'Terms n'), this.n], ['Σ x·w', fmt(this.sum)], [Lx('偏置 b', 'Bias b'), fmt(this.bias)], ['b + Σ', fmt(this.pre)], [Lx('激活后', 'After activation'), fmt(this.out)], [Lx('前向里的值', 'Forward-pass value'), fmt(this.stored, 6)], [Lx('差', 'Difference'), this.err.toExponential(1)]];
  }
}

// 算式板（DOM，样式沿用 ../css/app.css 的 .board）
export function renderBoard(el, mac, mi, p) {
  const steps = ['pick', 'mul', 'sum', 'act'];
  const k = steps.indexOf(mi);
  const names = Lx(['选一个输出', '逐项相乘', '加起来', '激活'], ['Pick an output', 'Multiply', 'Add up', 'Activate']);
  const top = mac.top;
  const shownRows = mi === 'pick' ? 0 : mi === 'mul' ? Math.ceil(top.length * Math.min(1, p * 1.15 + 0.05)) : top.length;
  const restDone = k >= 2;
  const maxP = Math.max(...top.map((i) => Math.abs(mac.terms[i].p)), 1e-9);
  const rows = top.map((i, r) => {
    const t = mac.terms[i];
    const w = Math.min(50, (Math.abs(t.p) / maxP) * 50);
    const cls = r >= shownRows ? 'pend' : r === shownRows - 1 && mi === 'mul' ? 'cur' : '';
    return `<div class="bd-tr ${cls}"><span class="i">${esc(t.lab)}</span><span class="cx">${fmt(t.x)}</span><span class="o">×</span><span class="cw">${fmt(t.w)}</span><span class="o">=</span><span class="cp">${fmt(t.p)}</span><span class="bar"><i class="${t.p >= 0 ? 'pos' : 'neg'}" style="width:${w}%"></i></span></div>`;
  }).join('');
  const restSum = mac.sum - top.reduce((s, i) => s + mac.terms[i].p, 0);
  const rest = mac.n > top.length ? `<div class="bd-tr bd-rest ${restDone ? 'done' : ''}"><span class="i"></span><span class="rest-t">${Lx(`其余 ${mac.n - top.length} 项<small>|乘积| 都更小</small>`, `Other ${mac.n - top.length} terms<small>all with smaller |product|</small>`)}</span><span class="cp">${fmt(restSum)}</span><span class="bar"><i class="${restSum >= 0 ? 'pos' : 'neg'}" style="width:${Math.min(50, Math.abs(restSum) / maxP * 50)}%"></i></span></div>` : '';
  const accVal = k <= 0 ? 0 : k === 1 ? top.slice(0, shownRows).reduce((s, i) => s + mac.terms[i].p, 0) : mac.pre;
  const final = k >= 2;
  const meterW = Math.min(50, Math.abs(accVal) / (Math.abs(mac.pos) + Math.abs(mac.neg) + Math.abs(mac.bias) || 1) * 100);
  el.innerHTML = `
    <div class="bd-h"><span class="bd-where">${esc(mac.where)}</span><ol class="bd-steps">${names.map((n, i) => `<li class="${i < k ? 'done' : i === k ? 'on' : ''}"><b>${i + 1}</b>${n}</li>`).join('')}</ol><button class="bd-fold" type="button" title="${Lx('收起', 'Collapse')}">–</button></div>
    <div class="bd-scroll">
      <div class="bd-body">
        <div class="bd-main">
          <div class="bd-fx">y = b + Σ<span class="lim">i=1…${mac.n}</span> x<sub>i</sub> × w<sub>i</sub><small>${esc(mac.note)}</small></div>
          <div class="bd-tbl">
            <div class="bd-tr bd-th"><span class="i">${Lx('项', 'term')}</span><span class="cx">${Lx('x（输入）', 'x (input)')}</span><span class="o"></span><span class="cw">${Lx('w（权重）', 'w (weight)')}</span><span class="o"></span><span class="cp">x × w</span><span></span></div>
            ${rows}${rest}
          </div>
        </div>
        <div class="bd-side">
          <div class="bd-acc ${final ? 'final' : ''}"><div class="bd-acc-k">${k <= 1 ? Lx('累加（按 |乘积| 从大到小）', 'Running sum (largest |product| first)') : 'b + Σ x·w'}</div><div class="bd-acc-v">${k >= 2 ? `<span class="eq">${fmt(mac.sum)} + ${fmt(mac.bias)} = </span>` : ''}<b>${fmt(accVal)}</b></div><div class="bd-meter"><span class="zero"></span><b style="${accVal >= 0 ? `left:50%;width:${meterW}%` : `left:${50 - meterW}%;width:${meterW}%`}"></b></div></div>
          <div class="bd-pn ${k >= 1 ? 'on' : ''}"><span>${Lx('正项之和', 'Positive sum')}</span><span>${Lx('负项之和', 'Negative sum')}</span><b>${fmt(mac.pos)}</b><b>${fmt(mac.neg)}</b></div>
          <div class="bd-dest ${k >= 3 ? 'on' : ''}"><div class="bd-dest-h">${Lx('激活', 'Activation')}</div><p class="bd-big ${k >= 3 ? 'on' : ''}">${mac.actFormula()}</p><p>${Lx('前向里的值', 'Forward-pass value')} ${fmt(mac.stored, 6)} · ${Lx('差', 'diff')} ${mac.err.toExponential(1)} ${mac.err < 1e-4 ? '✓' : ''}</p></div>
        </div>
      </div>
    </div>`;
}
