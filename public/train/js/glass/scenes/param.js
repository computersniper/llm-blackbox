// 玻璃小模型 · D4 一个参数：随便点一个格子，看它的一生。
//   左上：它在哪个张量里（点别的格子换一个）；中间：从初始化到训练结束，每一帧的 w、梯度 g、m、√v、Δw；
//   右边：① 梯度从哪来——链式法则：g = Σ（输入 × 上游梯度），第 0 段 8 个位置逐项列出，其余 7 段合成一项；
//         ② 这一步的 AdamW 算式，全是真实数字，最后和记录的下一步对一下。
// 配色和站里的算式板一致：输入蓝、权重紫、乘积橙、结果青；梯度用玫红（全页的“反向”色）。
import { COL, text, rr, card, pill, line, dot, clamp, ease, seg, fmtP, sciSup, fmtInt, hexA, wrap, waitBox, measure, badge } from '../../draw.js';
import { heat, fnum, cellAt, mark } from '../heat.js';
import { ADAM_SUBS, UPD_TENSORS } from '../timeline.js';
import { TENSOR_LABEL } from '../data.js';
import { paramName } from './overview.js';
import { animP } from './step.js';
import { isEn, L } from '../../lang.js';
import { esc } from '../../../../js/ui.js';

const num = (v, d = 4) => (v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e4 ? Number(v.toPrecision(d + 1)).toString().replace('-', '−') : sciSup(v, d));
const par = (v, d = 4) => (v < 0 ? `(${num(v, d)})` : num(v, d));

// 默认看哪个参数：这个张量里从初始化到训练结束变化最大的那个（每次进来都一样，不随帧跳）
export function defaultParam(D, name) {
  const p = D.pIndex.get(name);
  const last = D.has('w', D.NF - 1) ? D.W(D.NF - 1) : null;
  let best = p.off, bv = -1;
  const lo = name === 'E' ? p.off + D.chars.indexOf('月') * p.cols : p.off, hi = name === 'E' ? lo + p.cols : p.off + p.n;
  for (let i = lo; i < hi; i++) { const v = last ? Math.abs(last[i] - (p.norm ? 1 : D.w0[i])) : Math.abs(D.w0[i]); if (v > bv) { bv = v; best = i; } }
  return best;
}

// 链式法则：一个参数的梯度由哪两个量相乘、加起来
export function chainTerms(D, k, gi) {
  const lc = D.locate(gi), p = lc.p, T = D.T, i = lc.i, j = lc.j;
  const A = (n) => D.act(k, n), Gd = (n) => D.grad(k, n);
  const rowsDot = (xa, xs, da, ds, xi, dj) => { const out = []; for (let t = 0; t < T; t++) out.push({ t, x: xa[t * xs + xi], d: da[t * ds + dj] }); return out; };
  const MAP = { Wq: ['n1', 'q'], Wk: ['n1', 'k'], Wv: ['n1', 'v'], Wo: ['ao', 'h1'], Wg: ['n2', 'gate'], Wu: ['n2', 'up'], Wd: ['act', 'h2'] };
  if (MAP[p.name]) {
    const [xn, dn] = MAP[p.name];
    const xa = A(xn), da = Gd(dn);
    return { groups: [{ terms: rowsDot(xa, xa.length / T, da, da.length / T, i, j), xl: `${xn === 'act' ? 'act' : xn.replace('n1', 'n₁').replace('n2', 'n₂')}[t, ${i}]`, dl: `∂L/∂${dn.replace('h1', 'h₁').replace('h2', 'h₂')}[t, ${j}]` }] };
  }
  if (p.norm) {
    const src = { g1: ['h0', 'inv1', 'n1'], g2: ['h1', 'inv2', 'n2'], gf: ['h2', 'invf', 'nf'] }[p.name];
    const h = A(src[0]), inv = A(src[1]), dn = Gd(src[2]), d = D.d;
    const terms = [];
    for (let t = 0; t < T; t++) terms.push({ t, x: h[t * d + i] * inv[t], d: dn[t * d + i] });
    const hn = { g1: 'h₀', g2: 'h₁', gf: 'h₂' }[p.name], nn = { g1: 'n₁', g2: 'n₂', gf: 'n_f' }[p.name];
    return { groups: [{ terms, xl: `(${hn}/rms)[t, ${i}]`, dl: `∂L/∂${nn}[t, ${i}]` }] };
  }
  // E：输入一路（这个字出现的位置，∂L/∂h₀ 直接加过来）+ 输出一路（∂L/∂logits[t, 这个字] × n_f[t, j]）
  const x = D.fixed, d = D.d, dh0 = Gd('h0'), dl = Gd('logits'), nf = A('nf');
  const inT = [], outT = [];
  for (let t = 0; t < T; t++) { if (x[t] === i) inT.push({ t, x: 1, d: dh0[t * d + j] }); outT.push({ t, x: nf[t * d + j], d: dl[t * D.V + i] }); }
  return { groups: [
    { terms: inT, xl: L(`[x_t = 「${D.ch(i)}」]`, `[x_t = “${D.ch(i)}”]`), dl: `∂L/∂h₀[t, ${j}]`, note: L('输入一路：查表时取过这一行的位置', 'input path: positions that looked up this row') },
    { terms: outT, xl: `n_f[t, ${j}]`, dl: L(`∂L/∂logits[t, 「${D.ch(i)}」]`, `∂L/∂logits[t, “${D.ch(i)}”]`), note: L('输出一路：每个位置给「' + D.ch(i) + '」打分时用到这一行', `output path: every position scores “${D.ch(i)}” with this row`) },
  ] };
}

// 这一步的 AdamW：exact 参数用 float32 原值，其余用帧里的 float16 记录
export function adamAt(D, k, gi) {
  const t = D.FR[k], lc = D.locate(gi), wd = lc.p.decay ? D.meta.train.wd : 0;
  const b1 = D.meta.train.beta1, b2 = D.meta.train.beta2, eps = D.meta.train.eps;
  const EX = D.exact();
  let w, graw, clip, m0, v0, lr, rec, exact = false;
  if (EX && EX.pos.has(gi)) {
    const j = EX.pos.get(gi), n = EX.n;
    w = EX.w[t * n + j]; graw = EX.g[t * n + j]; m0 = EX.m[t * n + j]; v0 = EX.v[t * n + j]; clip = EX.clip[t]; lr = EX.lr[t];
    rec = { w1: EX.w[(t + 1) * n + j], m1: EX.m[(t + 1) * n + j], v1: EX.v[(t + 1) * n + j] };
    exact = true;
  } else {
    w = D.W(k)[gi]; graw = D.G(k)[gi]; m0 = D.Mo(k)[gi]; v0 = D.Vo(k)[gi]; clip = D.clip[t]; lr = D.lr[t];
    rec = { dw: D.DW(k)[gi] };
  }
  const g = graw * clip;
  const m = b1 * m0 + (1 - b1) * g, v = b2 * v0 + (1 - b2) * g * g;
  const bc1 = 1 - b1 ** (t + 1), bc2 = 1 - b2 ** (t + 1);
  const mh = m / bc1, vh = v / bc2;
  const adam = -lr * mh / (Math.sqrt(vh) + eps), decay = -lr * wd * w;
  const dw = adam + decay;
  return { t: t + 1, w, graw, clip, g, m0, v0, m, v, mh, vh, lr, wd, adam, decay, dw, w1: w + dw, rec, exact, b1, b2, eps };
}

export class GParam {
  constructor(app, R) { this.app = app; this.R = R; this.D = R.D; }

  layout(env) {
    const P = (this.portrait = env.portrait);
    if (!P) {
      this.W = 1250;
      this.mapC = { x: 0, y: 74, w: 320, h: 496 };
      this.chainC = { x: 336, y: 74, w: 450, h: 496 };
      this.adamC = { x: 802, y: 74, w: 450, h: 496 };
      this.lifeC = { x: 0, y: 584, w: 1252, h: 214, wide: true };
      this.bounds = { x: -10, y: -4, w: 1272, h: 812 };
    } else {
      this.W = 380;
      this.mapC = { x: 0, y: 84, w: 380, h: 290 };
      this.chainC = { x: 0, y: 388, w: 380, h: 600 };
      this.adamC = { x: 0, y: 1002, w: 380, h: 560 };
      this.lifeC = { x: 0, y: 1576, w: 380, h: 420 };
      this.bounds = { x: -8, y: -4, w: 396, h: 2010 };
    }
  }

  focus(st) {
    if (!this.portrait) return this.bounds;
    const c = st.step.mi === 'chain' || st.step.mi === 'g' ? this.chainC : this.adamC;
    return { x: -8, y: c.y - 330, w: 396, h: c.h + 340 };
  }

  sel(st) {
    const D = this.D, s = st.step, ctx = this.app.ctx;
    const gsel = ctx.gsel;
    if (gsel != null && D.locate(gsel).p.name === s.t) return gsel;
    return defaultParam(D, s.t);
  }

  draw(g, st, env) {
    st = { ...st, p: animP(this, st, env, 1.6) };
    const D = this.D, k = st.k, s = st.step, P = this.portrait;
    if (!D.ready(k)) { waitBox(g, 0, 60, this.W, 400, env, st.wait, { withCard: true, label: L('正在载入这一步的真实记录…', 'Loading the real record of this step…') }); return; }
    const gi = this.sel(st);
    const t = D.FR[k];
    const name = paramName(D, gi);
    text(g, L(`ONE PARAMETER · 第 ${t + 1} 步 · ${s.ph === 'bwd' ? '反向：它的梯度' : '更新：AdamW'}`, `ONE PARAMETER · STEP ${t + 1} · ${s.ph === 'bwd' ? 'BACKWARD: ITS GRADIENT' : 'UPDATE: ADAMW'}`), 0, 16, { size: 10, kind: 'mono', color: COL.dim });
    text(g, L(`${name} 的一生`, `The life of ${name}`), 0, 50, { size: P ? 21 : 27, kind: 'serif', weight: 900, color: COL.ink });
    const exact = D.exact() && D.exact().pos.has(gi);
    const bx = measure(g, L(`${name} 的一生`, `The life of ${name}`), P ? 21 : 27, 'serif', 900) + 14;
    if (!P || bx < 250) badge(g, bx, 32, exact ? L('float32 原值 · 每一步都记', 'float32 · every step') : L('float16 记录 · 约 3 位有效数字', 'float16 record · ~3 significant digits'), exact ? COL.cyan : COL.dim, 'left', 10);
    this.drawMap(g, st, env, gi);
    this.drawLife(g, st, env, gi);
    this.drawChain(g, st, env, gi);
    this.drawAdam(g, st, env, gi);
  }

  /* ---------------------------------------------------------------- 在哪个张量里 */

  drawMap(g, st, env, gi) {
    const D = this.D, k = st.k, C = this.mapC, P = this.portrait;
    const lc = D.locate(gi), p = lc.p;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: L('WHERE · 点别的格子换一个', 'WHERE · CLICK ANOTHER CELL'), title: `${TENSOR_LABEL[p.name]}  [${p.norm ? p.rows : `${p.rows} × ${p.cols}`}]`, accent: COL.violet });
    // 张量切换
    let x = C.x + 14, y = C.y + 50;
    for (const n of UPD_TENSORS) {
      const lab = TENSOR_LABEL[n], w = measure(g, lab, 10, 'mono') + 14;
      if (x + w > C.x + C.w - 10) { x = C.x + 14; y += 24; }
      pill(g, x, y, w, 20, lab, { on: n === p.name, color: COL.violet, size: 10, kind: 'mono' });
      env.hit(x, y, w, 20, { click: true, act: () => { this.app.pickParam(n === p.name ? gi : defaultParam(D, n), true); }, tip: `${TENSOR_LABEL[n]}` });
      x += w + 5;
    }
    const top = y + 32;
    const rows = p.norm ? 1 : p.rows, cols = p.norm ? p.rows : p.cols;
    const avail = C.w - 40, availH = C.y + C.h - top - 24;
    const cell = Math.max(3, Math.min(14, Math.floor(Math.min(avail / cols, availH / rows))));
    const x0 = C.x + (C.w - cols * cell) / 2 + (p.name === 'E' ? 8 : 0), y0 = top;
    const W = D.W(k);
    const val = (r, c) => { const q = p.off + (p.norm ? c : r * p.cols + c); return p.norm ? W[q] - 1 : W[q]; };
    let sc = 0;
    for (let q = p.off; q < p.off + p.n; q++) sc = Math.max(sc, Math.abs(p.norm ? W[q] - 1 : W[q]));
    heat(g, x0, y0, rows, cols, cell, cell, val, { scale: sc * 0.9 || 1, s: env.s });
    mark(g, x0, y0, p.norm ? 0 : lc.i, p.norm ? lc.i : lc.j, cell, cell, '#ffffff', 2);
    if (p.name === 'E') for (let r = 0; r < rows; r++) text(g, D.ch(r), x0 - 4, y0 + r * cell + cell * 0.82, { size: Math.min(10, cell), color: r === lc.i ? COL.ink : COL.dim, align: 'right' });
    env.hit(x0, y0, cols * cell, rows * cell, {
      click: true,
      tipAt: (wx, wy) => { const c = cellAt(wx, wy, x0, y0, rows, cols, cell, cell); if (!c) return null; const q = p.off + (p.norm ? c.c : c.r * p.cols + c.c); return `<span class="k">${esc(paramName(D, q))}</span>w <span class="v">${fnum(W[q], 4)}</span><br><span style="color:var(--dim)">${L('点一下换成看它', 'Click to look at this one')}</span>`; },
      act: (h) => { const c = h && cellAt(h.wx, h.wy, x0, y0, rows, cols, cell, cell); if (c) this.app.pickParam(p.off + (p.norm ? c.c : c.r * p.cols + c.c), true); },
    });
  }

  /* ---------------------------------------------------------------- 一生的曲线 */

  series(gi) {
    const D = this.D, EX = D.exact();
    if (EX && EX.pos.has(gi)) {
      const j = EX.pos.get(gi), n = EX.n, S = D.S;
      const xs = [], w = [], gg = [], m = [], sv = [], dw = [];
      for (let t = 0; t < S; t++) { xs.push(t); w.push(EX.w[t * n + j]); gg.push(EX.g[t * n + j]); m.push(EX.m[t * n + j]); sv.push(Math.sqrt(EX.v[t * n + j])); dw.push(EX.w[(t + 1) * n + j] - EX.w[t * n + j]); }
      return { xs, w, g: gg, m, sv, dw, full: true };
    }
    const xs = [], w = [], gg = [], m = [], sv = [], dw = [];
    for (let k = 0; k < D.NF; k++) {
      if (!D.has('w', k) || !D.has('f', k)) continue;
      xs.push(D.FR[k]); w.push(D.W(k)[gi]); gg.push(D.G(k)[gi]); m.push(D.Mo(k)[gi]); sv.push(Math.sqrt(D.Vo(k)[gi])); dw.push(D.DW(k)[gi]);
    }
    return { xs, w, g: gg, m, sv, dw, full: false };
  }

  drawLife(g, st, env, gi) {
    const D = this.D, C = this.lifeC, S = D.S, t = D.FR[st.k];
    const sr = this.series(gi);
    card(g, C.x, C.y, C.w, C.h, { eyebrow: sr.full ? L(`HISTORY · 全部 ${S} 步`, `HISTORY · ALL ${S} STEPS`) : L(`HISTORY · ${sr.xs.length} / ${D.NF} 帧已载入`, `HISTORY · ${sr.xs.length} / ${D.NF} FRAMES LOADED`), title: L('从初始化到训练结束', 'From initialization to the end'), accent: COL.cyan });
    const rowsDef = [['w', L('权重 w', 'weight w'), COL.violet], ['g', L('梯度 ∂L/∂w', 'gradient ∂L/∂w'), COL.rose], ['m', L('一阶动量 m', '1st moment m'), COL.amber], ['sv', L('√v（梯度的典型大小）', '√v (typical gradient size)'), COL.blue], ['dw', L('每步的更新 Δw', 'update Δw'), COL.cyan]];
    const wide = !!C.wide;
    const n = rowsDef.length, gap = 18;
    const cw = wide ? (C.w - 28 - gap * (n - 1)) / n : C.w - 28;
    const top = C.y + 54, rh = wide ? C.h - 66 : (C.h - 66) / n;
    rowsDef.forEach(([key, lab, col], ri) => {
      const x0 = C.x + 14 + (wide ? ri * (cw + gap) : 0), w = cw;
      const X = (tt) => x0 + (tt / (S - 1)) * w;
      const arr = sr[key], y0 = wide ? top : top + ri * rh, h = rh - 24;
      text(g, lab, x0, y0 + 9, { size: 10, color: col });
      if (!arr.length) return;
      let lo = Math.min(...arr), hi = Math.max(...arr);
      if (key === 'w') { const pad = (hi - lo) * 0.1 || 0.01; lo -= pad; hi += pad; } else if (key === 'sv') { lo = 0; } else { const mm = Math.max(Math.abs(lo), Math.abs(hi)) || 1e-9; lo = -mm; hi = mm; }
      const Y = (v) => y0 + 14 + h - ((v - lo) / (hi - lo || 1)) * h;
      if (key !== 'w' && key !== 'sv') { g.strokeStyle = COL.line; g.beginPath(); g.moveTo(x0, Y(0)); g.lineTo(x0 + w, Y(0)); g.stroke(); }
      const pts = sr.xs.map((xx, i) => [X(xx), Y(arr[i])]);
      line(g, pts, hexA(col, key === 'g' ? 0.55 : 0.95), key === 'g' ? 0.9 : 1.4);
      // 当前这一步
      let ci = 0;
      sr.xs.forEach((xx, i) => { if (Math.abs(xx - t) < Math.abs(sr.xs[ci] - t)) ci = i; });
      const cx = X(t);
      g.strokeStyle = hexA(COL.amber, 0.45); g.beginPath(); g.moveTo(cx, y0 + 12); g.lineTo(cx, y0 + 14 + h); g.stroke();
      if (sr.xs[ci] === t) dot(g, cx, Y(arr[ci]), 3, COL.amber);
      text(g, num(arr[ci], 3), x0 + w, y0 + 9, { size: 9.5, kind: 'mono', color: COL.ink2, align: 'right' });
      if (wide) { g.strokeStyle = COL.line; g.strokeRect(x0, y0 + 14, w, h); text(g, L('第 1 步', 'step 1'), x0, y0 + h + 26, { size: 8.5, kind: 'mono', color: COL.faint }); text(g, `${S}`, x0 + w, y0 + h + 26, { size: 8.5, kind: 'mono', color: COL.faint, align: 'right' }); }
      env.hit(x0, y0, w, rh, { tipAt: (wx) => { const tt = ((wx - x0) / w) * (S - 1); let bi = 0; sr.xs.forEach((xx, i) => { if (Math.abs(xx - tt) < Math.abs(sr.xs[bi] - tt)) bi = i; }); return `<span class="k">${L(`第 ${sr.xs[bi] + 1} 步`, `step ${sr.xs[bi] + 1}`)}</span>${lab} = <span class="v">${num(arr[bi], 5)}</span>`; } });
    });
  }

  /* ---------------------------------------------------------------- 梯度从哪来 */

  drawChain(g, st, env, gi) {
    const D = this.D, k = st.k, s = st.step, C = this.chainC, P = this.portrait, T = D.T;
    const on = s.mi === 'chain' || s.mi === 'g';
    const lc = D.locate(gi), p = lc.p;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: L('CHAIN RULE · 梯度从哪来', 'CHAIN RULE · WHERE THE GRADIENT COMES FROM'), title: L('输入 × 上游梯度，64 个位置加起来', 'input × upstream gradient, summed over 64 positions'), accent: COL.rose, active: on });
    const ch = chainTerms(D, k, gi);
    const G = D.G(k)[gi];
    const x0 = C.x + 16, w = C.w - 32;
    let y = C.y + 64;
    // 一句话
    const rule = p.name === 'E'
      ? L('E 既在输入端被查表、又在输出端乘 Eᵀ，所以梯度有两路。', 'E is looked up at the input and multiplied as Eᵀ at the output, so its gradient has two paths.')
      : p.norm ? L('γ 逐维乘在归一化后的向量上：∂L/∂γ = Σ（归一化后的值 × 流到输出的梯度）。', 'γ scales each dimension of the normalized vector: ∂L/∂γ = Σ (normalized value × gradient at the output).')
        : L(`y = x·W，所以 ∂L/∂W[${lc.i}, ${lc.j}] = Σ 输入 x[${lc.i}] × 上游梯度 ∂L/∂y[${lc.j}]。`, `y = x·W, so ∂L/∂W[${lc.i}, ${lc.j}] = Σ input x[${lc.i}] × upstream gradient ∂L/∂y[${lc.j}].`);
    y += wrap(g, rule, x0, y, w, 16, { size: 11.5, color: COL.ink2 }) * 16 + 8;
    const reveal = on ? ease(seg(st.p, 0, 0.7)) : 1;
    let row0 = 0, nShown = 0;
    const totalTerms = ch.groups.reduce((a, gp) => a + gp.terms.length, 0);
    const maxAbs = Math.max(1e-30, ...ch.groups.flatMap((gp) => gp.terms.map((tm) => Math.abs(tm.x * tm.d))), Math.abs(G));
    const lh = P ? 19 : 21;
    for (const gp of ch.groups) {
      if (gp.note) { text(g, gp.note, x0, y + 4, { size: 10, color: COL.dim, max: w }); y += 16; }
      // 表头
      text(g, L('位置', 'pos'), x0, y + 10, { size: 9.5, color: COL.dim });
      text(g, gp.xl, x0 + (P ? 96 : 120), y + 10, { size: 9.5, kind: 'mono', color: COL.blue, align: 'right' });
      text(g, gp.dl, x0 + (P ? 196 : 250), y + 10, { size: 9.5, kind: 'mono', color: COL.rose, align: 'right', max: P ? 110 : 140 });
      text(g, L('乘积', 'product'), x0 + w, y + 10, { size: 9.5, color: COL.amber, align: 'right' });
      y += 16;
      if (!gp.terms.length) { text(g, L('（第 0 段里没有这个字，这一路是 0）', '(this character isn’t in row 0, so this path is 0)'), x0, y + 12, { size: 10.5, color: COL.faint }); y += lh; }
      for (const tm of gp.terms) {
        const vis = clamp(reveal * totalTerms - nShown, 0, 1);
        nShown++;
        const prod = tm.x * tm.d;
        if (vis <= 0) { y += lh; continue; }
        g.globalAlpha = vis;
        row0 += prod;
        text(g, `${tm.t}「${D.ch(D.fixed[tm.t])}」`, x0, y + 13, { size: 10.5, color: COL.ink2 });
        text(g, num(tm.x, 3), x0 + (P ? 96 : 120), y + 13, { size: 10.5, kind: 'mono', color: COL.blue, align: 'right' });
        text(g, '×', x0 + (P ? 106 : 134), y + 13, { size: 10, color: COL.dim, align: 'center' });
        text(g, num(tm.d, 3), x0 + (P ? 196 : 250), y + 13, { size: 10.5, kind: 'mono', color: COL.rose, align: 'right' });
        // 乘积的条
        const bx = x0 + (P ? 206 : 266), bw = w - (P ? 206 : 266) - (P ? 64 : 76);
        const len = (Math.abs(prod) / maxAbs) * bw / 2;
        g.fillStyle = hexA(COL.amber, 0.7);
        if (prod >= 0) g.fillRect(bx + bw / 2, y + 5, len, 9); else g.fillRect(bx + bw / 2 - len, y + 5, len, 9);
        g.fillStyle = COL.line2; g.fillRect(bx + bw / 2, y + 3, 1, 13);
        text(g, num(prod, 2), x0 + w, y + 13, { size: 10, kind: 'mono', color: COL.amber, align: 'right' });
        g.globalAlpha = 1;
        y += lh;
      }
      y += 4;
    }
    // 合计
    const done = reveal >= 1;
    g.strokeStyle = COL.line2; g.beginPath(); g.moveTo(x0, y); g.lineTo(x0 + w, y); g.stroke();
    y += 6;
    const rest = G - row0;
    if (done) {
      text(g, L('第 0 段这 8 个位置合计', 'row 0 (these 8 positions)'), x0, y + 14, { size: 11, color: COL.ink2 });
      text(g, num(row0, 4), x0 + w, y + 14, { size: 11.5, kind: 'mono', color: COL.amber, align: 'right' });
      text(g, L('其余 7 段（56 个位置）合计 = 总梯度 − 第 0 段', 'other 7 rows (56 positions) = total − row 0'), x0, y + 34, { size: 11, color: COL.ink2, max: w - 90 });
      text(g, num(rest, 4), x0 + w, y + 34, { size: 11.5, kind: 'mono', color: COL.amber, align: 'right' });
      rr(g, x0 - 6, y + 44, w + 12, 30, 7); g.fillStyle = hexA(COL.cyan, 0.08); g.fill();
      text(g, L(`总梯度 ∂L/∂${TENSOR_LABEL[p.name]}（记录的）`, `total ∂L/∂${TENSOR_LABEL[p.name]} (recorded)`), x0, y + 64, { size: 11.5, color: COL.ink, weight: 600 });
      text(g, num(G, 4), x0 + w, y + 64, { size: 13, kind: 'mono', weight: 700, color: COL.cyan, align: 'right' });
      y += 86;
      const share = Math.abs(G) > 0 ? row0 / G : 0;
      const head = share >= 0 && share <= 1 ? L(`第 0 段贡献了 ${(share * 100).toFixed(0)}%。`, `Row 0 contributes ${(share * 100).toFixed(0)}%. `) : L('第 0 段和其余 7 段的方向相反，互相抵消了一部分。', 'Row 0 and the other 7 rows point opposite ways and partly cancel. ');
      wrap(g, head + L('每一项乘积就是“这个位置想让它往哪边改、改多少”；64 个位置的意见加起来，才是这一步的梯度。', 'Each product is “which way, and how much, this position wants it to move”; the gradient is all 64 positions’ votes added up.'), x0, y + 6, w, 15, { size: 10.5, color: COL.dim });
    }
  }

  /* ---------------------------------------------------------------- AdamW 算式 */

  drawAdam(g, st, env, gi) {
    const D = this.D, k = st.k, s = st.step, C = this.adamC, P = this.portrait;
    const a = adamAt(D, k, gi);
    const on = s.ph === 'upd';
    card(g, C.x, C.y, C.w, C.h, { eyebrow: L(`ADAMW · 第 ${a.t} 步`, `ADAMW · STEP ${a.t}`), title: L('这一步它挪了多少', 'How far it moves this step'), accent: COL.violet, active: on });
    const cur = on ? ADAM_SUBS.indexOf(s.mi) : -1;
    const x0 = C.x + 16, fx = x0 + (P ? 64 : 84), mw = C.x + C.w - fx - 12;
    text(g, `lr = ${sciSup(a.lr, 3)}　β₁ = ${a.b1}　β₂ = ${a.b2}　ε = 10⁻⁸　λ = ${a.wd}　t = ${a.t}`, x0, C.y + 62, { size: 10, kind: 'mono', color: COL.dim, max: C.w - 32 });
    const lines = [
      { k: L('梯度', 'gradient'), l: `g = ∂L/∂w × ${L('裁剪系数', 'clip')} = ${par(a.graw)} × ${a.clip < 1 ? a.clip.toPrecision(4) : '1'}`, r: `g = ${num(a.g)}`, c: COL.rose },
      { k: L('一阶动量', '1st moment'), l: `m = ${a.b1} × ${par(a.m0)} + ${(1 - a.b1).toFixed(1)} × ${par(a.g)}`, r: `m = ${num(a.m)}`, c: COL.amber },
      { k: L('二阶动量', '2nd moment'), l: `v = ${a.b2} × ${par(a.v0, 3)} + ${(1 - a.b2).toFixed(2)} × ${par(a.g, 3)}²`, r: `v = ${num(a.v)}`, c: COL.amber },
      { k: L('偏差校正', 'bias corr.'), l: `m̂ = m / (1 − ${a.b1}^${a.t}),  v̂ = v / (1 − ${a.b2}^${a.t})`, r: `m̂ = ${num(a.mh)},  v̂ = ${num(a.vh, 3)}`, c: COL.amber },
      { k: 'Δw', l: `−lr·m̂/(√v̂+ε) − lr·λ·w = ${sciSup(a.adam, 3)} ${a.decay >= 0 ? '+' : '−'} ${sciSup(Math.abs(a.decay), 3)}`, r: `Δw = ${sciSup(a.dw, 4)}`, c: COL.cyan },
      { k: L('写回', 'write'), l: `w′ = ${num(a.w, 7)} + (${sciSup(a.dw, 3)})`, r: `w′ = ${num(a.w1, 7)}`, c: COL.cyan },
    ];
    const lh = P ? 58 : 56;
    let y = C.y + 78;
    lines.forEach((ln, i) => {
      const vis = cur < 0 ? 1 : i < cur ? 1 : i === cur ? ease(seg(st.p, 0, 0.35)) : 0.14;
      g.globalAlpha = vis;
      if (i === cur) { rr(g, C.x + 8, y - 4, C.w - 16, lh - 8, 8); g.fillStyle = hexA(COL.amber, 0.08); g.fill(); g.fillStyle = COL.amber; g.fillRect(C.x + 8, y - 4, 2.5, lh - 8); }
      text(g, ln.k, x0 + 4, y + 14, { size: 10.5, color: i === cur ? COL.amber : COL.dim });
      text(g, ln.l, fx, y + 14, { size: P ? 10 : 11, kind: 'mono', color: COL.ink2, max: mw });
      text(g, ln.r, fx, y + 36, { size: P ? 12 : 13, kind: 'mono', weight: 700, color: i === cur ? ln.c : COL.ink, max: mw });
      g.globalAlpha = 1;
      y += lh;
    });
    // 和记录对一下
    y += 2;
    const showCheck = cur < 0 || cur >= ADAM_SUBS.length - 1;
    g.globalAlpha = showCheck ? 1 : 0.18;
    rr(g, C.x + 10, y, C.w - 20, P ? 92 : 84, 8); g.fillStyle = 'rgba(255,255,255,0.03)'; g.fill(); g.strokeStyle = COL.line2; g.stroke();
    if (a.exact) {
      const d = a.w1 - a.rec.w1;
      text(g, L('记录的下一步 w（float32）', 'recorded next w (float32)'), x0, y + 20, { size: 10.5, color: COL.ink2 });
      text(g, num(a.rec.w1, 7), C.x + C.w - 18, y + 20, { size: 12, kind: 'mono', weight: 700, color: COL.cyan, align: 'right' });
      wrap(g, L(`按公式算出的和记录的相差 ${Math.abs(d) < 1e-12 ? '0' : sciSup(Math.abs(d), 2)}（这里用双精度算；导出脚本用 PyTorch 同样的 float32 运算重算全部 585,600 次更新，逐位一致）`, `Formula vs record differ by ${Math.abs(d) < 1e-12 ? '0' : sciSup(Math.abs(d), 2)} (computed here in double precision; the export script redoes all 585,600 updates with PyTorch’s own float32 ops and matches bit for bit)`), x0, y + 40, C.w - 32, 14, { size: 10, color: COL.dim });
    } else {
      const rel = Math.abs(a.rec.dw) > 0 ? Math.abs(a.dw - a.rec.dw) / Math.abs(a.rec.dw) : 0;
      text(g, L('记录的 Δw', 'recorded Δw'), x0, y + 20, { size: 10.5, color: COL.ink2 });
      text(g, sciSup(a.rec.dw, 4), C.x + C.w - 18, y + 20, { size: 12, kind: 'mono', weight: 700, color: COL.cyan, align: 'right' });
      wrap(g, L(`按公式算出的和记录的相差 ${(rel * 100).toFixed(rel < 0.001 ? 3 : 2)}%：页面上的 w、g、m、v 是 float16 记录（约 3 位有效数字）。三个 γ 和「月」那一行存了 float32 原值，点它们能逐位对上。`, `Formula vs record differ by ${(rel * 100).toFixed(rel < 0.001 ? 3 : 2)}%: w, g, m, v here are float16 records (~3 significant digits). The three γ and the row of 月 are stored as float32 — click them for an exact match.`), x0, y + 40, C.w - 32, 14, { size: 10, color: COL.dim });
    }
    g.globalAlpha = 1;
    if (a.t === 1 && (cur < 0 || cur >= 3)) text(g, L('第 1 步：m̂ = g、v̂ = g²，Δw ≈ −lr × g 的符号——每个参数都挪了差不多一个 lr', 'Step 1: m̂ = g and v̂ = g², so Δw ≈ −lr × sign(g) — every parameter moves about one lr'), x0, C.y + C.h - 14, { size: 10, color: COL.violet, max: C.w - 32 });
  }
}
