// 玻璃小模型 · 一个参数的算式（纯函数，数字全部来自记录）：
//   fwdTerms   —— 前向：它乘了谁（第 0 段 8 个位置，每个位置一项“输入 × 它”）
//   chainTerms —— 反向：它的梯度从哪来（第 0 段 8 个位置，每个位置一项“输入 × 上游梯度”）
//   adamAt     —— 更新：这一步的 AdamW 算式
//   defaultParam —— 某个张量里默认看哪个参数
import { L } from '../lang.js';

// 矩阵：y = x · W，输入是哪个激活、输出是哪个激活
export const MM_IO = { Wq: ['n1', 'q'], Wk: ['n1', 'k'], Wv: ['n1', 'v'], Wo: ['ao', 'o'], Wg: ['n2', 'gate'], Wu: ['n2', 'up'], Wd: ['act', 'f'] };
// RMSNorm 的 γ：归一化前的激活、1/rms、输出
export const NORM_IO = { g1: ['h0', 'inv1', 'n1'], g2: ['h1', 'inv2', 'n2'], gf: ['h2', 'invf', 'nf'] };
const SUBN = (s) => s.replace('n1', 'n₁').replace('n2', 'n₂').replace('nf', 'n_f').replace('h0', 'h₀').replace('h1', 'h₁').replace('h2', 'h₂');

// 默认看哪个参数：这个张量里从初始化到训练结束变化最大的那个（每次进来都一样，不随帧跳）
export function defaultParam(D, name) {
  const p = D.pIndex.get(name);
  const last = D.has('w', D.NF - 1) ? D.W(D.NF - 1) : null;
  let best = p.off, bv = -1;
  const lo = name === 'E' ? p.off + D.chars.indexOf('月') * p.cols : p.off, hi = name === 'E' ? lo + p.cols : p.off + p.n;
  for (let i = lo; i < hi; i++) { const v = last ? Math.abs(last[i] - (p.norm ? 1 : D.w0[i])) : Math.abs(D.w0[i]); if (v > bv) { bv = v; best = i; } }
  return best;
}

// 前向：这个参数在第 0 段的 8 个位置上各乘了谁、加进了哪个输出
//   矩阵 W[i, j]：y[t, j] 里有一项 x[t, i] × W[i, j]（y[t, j] 是 16 / 32 项之和，记录里有）
//   γ[i]：n[t, i] = (h[t, i] / rms[t]) × γ[i]（就这一项）
//   E[i, j]：输入一路，x_t =「i」的位置直接取 h₀[t, j] = E[i, j]；输出一路，logits[t, i] 里有一项 n_f[t, j] × E[i, j]
export function fwdTerms(D, k, gi) {
  const lc = D.locate(gi), p = lc.p, T = D.T, i = lc.i, j = lc.j;
  const W = D.W(k)[gi];
  const A = (n) => D.act(k, n);
  if (MM_IO[p.name]) {
    const [xn, yn] = MM_IO[p.name];
    const xa = A(xn), ya = A(yn), xs = xa.length / T, ys = ya.length / T;
    const terms = [];
    for (let t = 0; t < T; t++) terms.push({ t, x: xa[t * xs + i], y: ya[t * ys + j] });
    return { w: W, groups: [{ terms, xl: `${SUBN(xn)}[t, ${i}]`, yl: `${SUBN(yn)}[t, ${j}]`, n: xs }] };
  }
  if (p.norm) {
    const [hn, invn, nn] = NORM_IO[p.name];
    const h = A(hn), inv = A(invn), n = A(nn), d = D.d;
    const terms = [];
    for (let t = 0; t < T; t++) terms.push({ t, x: h[t * d + i] * inv[t], y: n[t * d + i] });
    return { w: W, groups: [{ terms, xl: `(${SUBN(hn)}/rms)[t, ${i}]`, yl: `${SUBN(nn)}[t, ${i}]`, n: 1 }] };
  }
  const x = D.fixed, d = D.d, h0 = A('h0'), nf = A('nf'), lg = A('logits');
  const inT = [], outT = [];
  for (let t = 0; t < T; t++) {
    if (x[t] === i) inT.push({ t, x: 1, y: h0[t * d + j] });
    outT.push({ t, x: nf[t * d + j], y: lg[t * D.V + i] });
  }
  return { w: W, groups: [
    { terms: inT, xl: L(`[x_t = 「${D.ch(i)}」]`, `[x_t = “${D.ch(i)}”]`), yl: `h₀[t, ${j}]`, n: 1, note: L('输入一路：查表时取到这一行的位置', 'input path: positions that look up this row') },
    { terms: outT, xl: `n_f[t, ${j}]`, yl: L(`logits[t, 「${D.ch(i)}」]`, `logits[t, “${D.ch(i)}”]`), n: d, note: L(`输出一路：每个位置给「${D.ch(i)}」打分`, `output path: every position scores “${D.ch(i)}”`) },
  ] };
}

// 链式法则：一个参数的梯度由哪两个量相乘、加起来
export function chainTerms(D, k, gi) {
  const lc = D.locate(gi), p = lc.p, T = D.T, i = lc.i, j = lc.j;
  const A = (n) => D.act(k, n), Gd = (n) => D.grad(k, n);
  const rowsDot = (xa, xs, da, ds, xi, dj) => { const out = []; for (let t = 0; t < T; t++) out.push({ t, x: xa[t * xs + xi], d: da[t * ds + dj] }); return out; };
  // o、f 的梯度就是残差 h₁、h₂ 的梯度（h₁ = h₀ + o）
  const GRAD_OF = { q: 'q', k: 'k', v: 'v', o: 'h1', gate: 'gate', up: 'up', f: 'h2' };
  if (MM_IO[p.name]) {
    const [xn, yn] = MM_IO[p.name], dn = GRAD_OF[yn];
    const xa = A(xn), da = Gd(dn);
    return { groups: [{ terms: rowsDot(xa, xa.length / T, da, da.length / T, i, j), xl: `${SUBN(xn)}[t, ${i}]`, dl: `∂L/∂${SUBN(dn)}[t, ${j}]` }] };
  }
  if (p.norm) {
    const [hn, invn, nn] = NORM_IO[p.name];
    const h = A(hn), inv = A(invn), dn = Gd(nn), d = D.d;
    const terms = [];
    for (let t = 0; t < T; t++) terms.push({ t, x: h[t * d + i] * inv[t], d: dn[t * d + i] });
    return { groups: [{ terms, xl: `(${SUBN(hn)}/rms)[t, ${i}]`, dl: `∂L/∂${SUBN(nn)}[t, ${i}]` }] };
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
