// 3D 舞台旁边的 2D 卡片（画在 board.js 的小画布上）：训练曲线、初始值的直方图、三种初始化的对照、
// 一个参数的一生（w / g / m / √v / Δw）、前向它乘了谁、梯度从哪来、AdamW 算式。数字全部来自记录。
import { COL, text, rr, card, line, dot, clamp, ease, seg, sciSup, hexA, wrap, waitBox } from '../draw.js';
import { heat, fnum, rgb, divRGB, paramName } from '../glass/heat.js';
import { ADAM_SUBS } from '../glass/timeline.js';
import { TENSOR_LABEL } from '../glass/data.js';
import { fwdTerms, chainTerms, adamAt } from '../glass/math.js';
import { L, isEn } from '../lang.js';

const num = (v, d = 4) => (v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e4 ? Number(v.toPrecision(d + 1)).toString().replace('-', '−') : sciSup(v, d));
const par = (v, d = 4) => (v < 0 ? `(${num(v, d)})` : num(v, d));
const INIT_SCALE = 0.06;

/* ---------------------------------------------------------------- 训练曲线（D1–D4 的监视器） */

// t：当前的“连续”步数；k：当前帧。drag(k) 拖动时换帧
export function drawLoss(g, C, D, t, k, env, { drag, compact = false } = {}) {
  const S = D.S;
  const gap = k > 0 ? D.FR[k] - D.FR[k - 1] : 1;
  card(g, C.x, C.y, C.w, C.h, { eyebrow: L(`LOSS · 第 ${Math.round(t) + 1} / ${S} 步`, `LOSS · STEP ${Math.round(t) + 1} / ${S}`), accent: COL.cyan });
  const tn = Math.min(S - 1, Math.max(0, Math.floor(t)));
  text(g, D.evalLoss[tn].toFixed(3), C.x + C.w - 14, C.y + 19, { size: 12, kind: 'mono', weight: 700, color: COL.amber, align: 'right' });
  const x0 = C.x + 34, w = C.w - 48, y0 = C.y + 32, h = C.h - (compact ? 50 : 62);
  const lo = Math.log(0.03), hi = Math.log(4);
  const Y = (v) => y0 + h - ((Math.log(clamp(v, 0.03, 4)) - lo) / (hi - lo)) * h, X = (tt) => x0 + (tt / (S - 1)) * w;
  for (const v of [0.1, 1, 3]) { const y = Y(v); g.strokeStyle = COL.line; g.beginPath(); g.moveTo(x0, y); g.lineTo(x0 + w, y); g.stroke(); text(g, String(v), x0 - 5, y + 3, { size: 9, kind: 'mono', color: COL.dim, align: 'right' }); }
  g.setLineDash([3, 3]);
  line(g, [[x0, Y(D.zLoss[0])], [x0 + w, Y(D.zLoss[0])]], hexA(COL.rose, 0.5), 1);
  g.setLineDash([]);
  if (!compact) text(g, L('全零初始化（一直不动）', 'all-zero init (never moves)'), x0 + w, Y(D.zLoss[0]) - 4, { size: 9, color: hexA(COL.rose, 0.8), align: 'right' });
  const pb = [], pe = [];
  for (let i = 0; i <= tn; i++) { pb.push([X(i), Y(D.loss[i])]); pe.push([X(i), Y(D.evalLoss[i])]); }
  line(g, pb, hexA(COL.cyan, 0.3), 0.9);
  line(g, pe, COL.cyan, 1.6);
  for (const fr of D.FR) { g.fillStyle = fr <= t ? hexA(COL.violet, 0.75) : hexA(COL.violet, 0.22); g.fillRect(X(fr) - 0.5, y0 + h + 3, 1, 4); }
  const xc = X(t);
  g.strokeStyle = hexA(COL.amber, 0.6); g.beginPath(); g.moveTo(xc, y0); g.lineTo(xc, y0 + h); g.stroke();
  dot(g, xc, Y(D.evalLoss[tn]), 3.2, COL.amber);
  if (!compact) {
    const note = gap > 1 ? L(`这一帧和上一帧隔 ${gap} 步（第 50 步以后每 5 / 10 步记一帧，中间的步只记了损失）`, `${gap} steps since the last frame (after step 50 a frame every 5 / 10 steps; only the loss was kept in between)`)
      : L('亮线：全部 25 段平均　暗线：这一步的 8 段　紫色刻度：逐数记录的帧', 'bright: all 25 windows · faint: this step’s 8 · ticks: fully recorded frames');
    text(g, note, C.x + 14, C.y + C.h - 11, { size: 9.5, color: gap > 1 ? COL.violet : COL.dim, max: C.w - 28 });
  }
  const toK = (wx) => {
    const tt = clamp(((wx - x0) / w) * (S - 1), 0, S - 1);
    let best = 0;
    D.FR.forEach((fr, i) => { if (Math.abs(fr - tt) < Math.abs(D.FR[best] - tt)) best = i; });
    return best;
  };
  env.hit(x0 - 4, y0 - 6, w + 8, h + 14, {
    drag: drag ? (wx, wy, phase) => { if (phase !== 'end') drag(toK(wx)); } : null,
    tipAt: (wx) => { const tt = clamp(Math.round(((wx - x0) / w) * (S - 1)), 0, S - 1); return `<span class="k">${L(`第 ${tt + 1} 步`, `step ${tt + 1}`)}</span>${L('8 段的损失', 'batch loss')} <span class="v">${D.loss[tt].toFixed(3)}</span>　${L('全部 25 段', 'all 25')} <span class="v">${D.evalLoss[tt].toFixed(3)}</span><br>${L('学习率', 'learning rate')} <span class="a">${sciSup(D.lr[tt], 3)}</span>　‖g‖ ${D.gnorm[tt].toFixed(3)}${drag ? `<br><span style="color:var(--dim)">${L('按住拖动来回看', 'Drag to scrub')}</span>` : ''}`; },
  });
}

/* ---------------------------------------------------------------- D0 初始化 */

// 初始值的直方图（真实的 2,880 个矩阵参数）+ 理论的正态曲线
export function drawHist(g, C, D, which, on) {
  if (which === 'zero') return drawAfter(g, C, D);
  const big = which === 'big';
  card(g, C.x, C.y, C.w, C.h, { eyebrow: L('INIT · 初始值的分布（真实的 2,880 个）', 'INIT · DISTRIBUTION OF THE REAL 2,880'), title: big ? L('放大 50 倍：N(0, 1²)', '50× larger: N(0, 1²)') : L('正态分布 N(0, 0.02²)', 'Normal distribution N(0, 0.02²)'), accent: COL.cyan, active: on });
  const src = big ? D.bW0 : D.w0, std = big ? 1 : 0.02, lim = 4 * std;
  const nb = 32, cnt = new Float32Array(nb);
  let n = 0;
  for (const p of D.params) { if (p.norm) continue; for (let i = p.off; i < p.off + p.n; i++) { const b = Math.floor(((src[i] + lim) / (2 * lim)) * nb); if (b >= 0 && b < nb) cnt[b]++; n++; } }
  const x0 = C.x + 18, w = C.w - 36, y0 = C.y + 62, h = C.h - 98;
  const mx = Math.max(...cnt), bw = w / nb;
  for (let b = 0; b < nb; b++) {
    const v = (b + 0.5) / nb * 2 * lim - lim, bh = (cnt[b] / mx) * h;
    g.fillStyle = rgb(divRGB(big ? v / (INIT_SCALE * 50) : v / INIT_SCALE), 0.9);
    g.fillRect(x0 + b * bw + 0.5, y0 + h - bh, bw - 1, bh);
  }
  const pts = [];
  for (let i = 0; i <= 80; i++) {
    const v = -lim + (i / 80) * 2 * lim;
    const dens = (n * (2 * lim / nb)) * Math.exp(-(v * v) / (2 * std * std)) / (std * Math.sqrt(2 * Math.PI));
    pts.push([x0 + ((v + lim) / (2 * lim)) * w, y0 + h - (dens / mx) * h]);
  }
  line(g, pts, hexA(COL.cyan, 0.8), 1.3);
  g.strokeStyle = COL.line2; g.beginPath(); g.moveTo(x0, y0 + h); g.lineTo(x0 + w, y0 + h); g.stroke();
  for (const v of [-lim, -lim / 2, 0, lim / 2, lim]) text(g, fnum(v, 2), x0 + ((v + lim) / (2 * lim)) * w, y0 + h + 13, { size: 9, kind: 'mono', color: COL.dim, align: 'center' });
  let s = 0;
  for (const p of D.params) if (!p.norm) for (let i = p.off; i < p.off + p.n; i++) s += src[i] * src[i];
  text(g, L(`实际标准差 ${Math.sqrt(s / n).toPrecision(3)} · 48 个 γ 都是 1`, `actual std ${Math.sqrt(s / n).toPrecision(3)} · all 48 γ are 1`), C.x + 18, C.y + C.h - 12, { size: 10, color: COL.dim });
}

// 全零对照：训练 200 步以后，两次真实训练的 E 和 W_q 并排
function drawAfter(g, C, D) {
  const K = D.NF - 1;
  card(g, C.x, C.y, C.w, C.h, { eyebrow: L('AFTER 200 STEPS · 两次真实训练并排', 'AFTER 200 STEPS · TWO REAL RUNS SIDE BY SIDE'), title: L('训练 200 步以后', 'After 200 training steps'), accent: COL.rose, active: true });
  const have = D.has('w', K), W = have ? D.W(K) : null;
  [[L('正态初始化', 'normal init'), COL.cyan, false], [L('全零初始化', 'all-zero init'), COL.rose, true]].forEach(([lab, col, zero], ci) => {
    const x0 = C.x + 18 + ci * ((C.w - 36) / 2);
    text(g, lab, x0, C.y + 58, { size: 11, color: col, weight: 600 });
    [['E', 20, 16, 3.6], ['Wq', 16, 16, 3.6]].forEach(([name, rows, ncols, cell], j) => {
      const p = D.pIndex.get(name), x = x0 + j * (ncols * cell + 14), y = C.y + 66;
      text(g, TENSOR_LABEL[name], x, y + rows * cell + 12, { size: 9, kind: 'mono', color: COL.dim });
      if (!zero && !have) { rr(g, x, y, ncols * cell, rows * cell, 3); g.strokeStyle = COL.line2; g.stroke(); return; }
      let sc = 0;
      if (!zero) for (let q = p.off; q < p.off + p.n; q++) sc = Math.max(sc, Math.abs(W[q]));
      heat(g, x, y, rows, ncols, cell, cell, (r, c) => (zero ? 0 : W[p.off + r * ncols + c]), { scale: sc * 0.8 || 1 });
    });
  });
  const z = D.meta.runs.zero;
  wrap(g, L(`右边：200 步里梯度范数始终 ${z.maxGnorm}、max |w| 始终 ${z.maxAbsW}，损失一直 ${z.finalEval.toFixed(3)}。左边同样 200 步，损失降到 ${D.meta.train.finalEval.toFixed(3)}。`, `Right: over 200 steps the gradient norm stayed ${z.maxGnorm}, max |w| stayed ${z.maxAbsW}, loss stuck at ${z.finalEval.toFixed(3)}. Left, same 200 steps: loss down to ${D.meta.train.finalEval.toFixed(3)}.`), C.x + 18, C.y + C.h - 30, C.w - 36, 13, { size: 9.5, color: COL.dim, maxLines: 2 });
}

// 三次真实训练：正态 / 全零 / 放大 50 倍，同样的批次、同样 200 步
export function drawRuns(g, C, D, which, on, env) {
  const S = D.S;
  card(g, C.x, C.y, C.w, C.h, { eyebrow: L('THREE REAL RUNS · 同样的批次，各训练 200 步', 'THREE REAL RUNS · SAME BATCHES, 200 STEPS EACH'), title: L('三种初始化的损失', 'Loss under three inits'), accent: COL.amber, active: on });
  const x0 = C.x + 40, w = C.w - 56, y0 = C.y + 58, h = C.h - 92;
  const lo = Math.log(0.03), hi = Math.log(9);
  const Y = (v) => y0 + h - ((Math.log(Math.max(v, 0.03)) - lo) / (hi - lo)) * h, X = (t) => x0 + (t / (S - 1)) * w;
  for (const v of [0.1, 1, 3]) { const y = Y(v); g.strokeStyle = COL.line; g.beginPath(); g.moveTo(x0, y); g.lineTo(x0 + w, y); g.stroke(); text(g, String(v), x0 - 5, y + 3, { size: 9, kind: 'mono', color: COL.dim, align: 'right' }); }
  [[D.evalLoss, COL.cyan, L('正态 0.02', 'normal 0.02'), 'normal'], [D.zLoss, COL.rose, L('全零', 'all zero'), 'zero'], [D.bEval, COL.amber, L('放大 50 倍', '50× larger'), 'big']].forEach(([arr, col, lab, key], si) => {
    const pts = [];
    for (let t = 0; t < S; t++) pts.push([X(t), Y(arr[t])]);
    const hot = !on || which === key || key === 'normal';
    line(g, pts, hexA(col, hot ? 0.95 : 0.3), hot ? 1.6 : 1);
    text(g, lab, C.x + 14 + si * 112, C.y + C.h - 12, { size: 10, color: hexA(col, hot ? 1 : 0.5) });
  });
  text(g, `ln 20 = ${Math.log(20).toFixed(2)}`, x0 + w, Y(Math.log(20)) - 5, { size: 9, kind: 'mono', color: COL.rose, align: 'right' });
  text(g, L('全部 25 段上的平均损失（对数刻度）', 'mean loss over all 25 windows (log scale)'), x0, y0 - 6, { size: 9.5, color: COL.dim });
  env.hit(x0, y0, w, h, { tipAt: (wx) => { const t = clamp(Math.round(((wx - x0) / w) * (S - 1)), 0, S - 1); return `<span class="k">${L(`第 ${t + 1} 步`, `step ${t + 1}`)}</span>${L('正态', 'normal')} <span class="v">${D.evalLoss[t].toFixed(3)}</span> · ${L('全零', 'zero')} <span class="v">${D.zLoss[t].toFixed(3)}</span> · ${L('放大', '50×')} <span class="a">${D.bEval[t].toFixed(3)}</span>`; } });
}

/* ---------------------------------------------------------------- D5 一个参数 */

// 从初始化到训练结束，每一帧的 w、g、m、√v、Δw
export function series(D, gi) {
  const EX = D.exact();
  const xs = [], w = [], gg = [], m = [], sv = [], dw = [];
  if (EX && EX.pos.has(gi)) {
    const j = EX.pos.get(gi), n = EX.n;
    for (let t = 0; t < D.S; t++) { xs.push(t); w.push(EX.w[t * n + j]); gg.push(EX.g[t * n + j]); m.push(EX.m[t * n + j]); sv.push(Math.sqrt(EX.v[t * n + j])); dw.push(EX.w[(t + 1) * n + j] - EX.w[t * n + j]); }
    return { xs, w, g: gg, m, sv, dw, full: true };
  }
  for (let k = 0; k < D.NF; k++) {
    if (!D.has('w', k) || !D.has('f', k)) continue;
    xs.push(D.FR[k]); w.push(D.W(k)[gi]); gg.push(D.G(k)[gi]); m.push(D.Mo(k)[gi]); sv.push(Math.sqrt(D.Vo(k)[gi])); dw.push(D.DW(k)[gi]);
  }
  return { xs, w, g: gg, m, sv, dw, full: false };
}

// 一生的曲线：wide = 五张小图排成一行，否则竖着排
export function drawLife(g, C, D, gi, t, env, { wide = true } = {}) {
  const S = D.S, sr = series(D, gi);
  card(g, C.x, C.y, C.w, C.h, { eyebrow: sr.full ? L(`HISTORY · 全部 ${S} 步 · float32`, `HISTORY · ALL ${S} STEPS · FLOAT32`) : L(`HISTORY · ${sr.xs.length} / ${D.NF} 帧已载入`, `HISTORY · ${sr.xs.length} / ${D.NF} FRAMES LOADED`), title: L(`${paramName(D, gi)} 的一生`, `The life of ${paramName(D, gi)}`), accent: COL.cyan });
  const rowsDef = [['w', L('权重 w', 'weight w'), COL.violet], ['g', L('梯度 ∂L/∂w', 'gradient ∂L/∂w'), COL.rose], ['m', L('一阶动量 m', '1st moment m'), COL.amber], ['sv', L('√v（梯度的典型大小）', '√v (typical gradient size)'), COL.blue], ['dw', L('每步的更新 Δw', 'update Δw'), COL.cyan]];
  const n = rowsDef.length, gap = 16;
  const cw = wide ? (C.w - 28 - gap * (n - 1)) / n : C.w - 28;
  const top = C.y + 52, rh = wide ? C.h - 62 : (C.h - 62) / n;
  rowsDef.forEach(([key, lab, col], ri) => {
    const x0 = C.x + 14 + (wide ? ri * (cw + gap) : 0), w = cw;
    const X = (tt) => x0 + (tt / (S - 1)) * w;
    const arr = sr[key], y0 = wide ? top : top + ri * rh, h = rh - 24;
    text(g, lab, x0, y0 + 9, { size: 10, color: col, max: w - 50 });
    if (!arr.length) return;
    let lo = Math.min(...arr), hi = Math.max(...arr);
    if (key === 'w') { const pad = (hi - lo) * 0.1 || 0.01; lo -= pad; hi += pad; } else if (key === 'sv') lo = 0; else { const mm = Math.max(Math.abs(lo), Math.abs(hi)) || 1e-9; lo = -mm; hi = mm; }
    const Y = (v) => y0 + 14 + h - ((v - lo) / (hi - lo || 1)) * h;
    if (key !== 'w' && key !== 'sv') { g.strokeStyle = COL.line; g.beginPath(); g.moveTo(x0, Y(0)); g.lineTo(x0 + w, Y(0)); g.stroke(); }
    line(g, sr.xs.map((xx, i) => [X(xx), Y(arr[i])]), hexA(col, key === 'g' ? 0.55 : 0.95), key === 'g' ? 0.9 : 1.4);
    let ci = 0;
    sr.xs.forEach((xx, i) => { if (Math.abs(xx - t) < Math.abs(sr.xs[ci] - t)) ci = i; });
    const cx = X(t);
    g.strokeStyle = hexA(COL.amber, 0.45); g.beginPath(); g.moveTo(cx, y0 + 12); g.lineTo(cx, y0 + 14 + h); g.stroke();
    if (sr.xs[ci] === t) dot(g, cx, Y(arr[ci]), 3, COL.amber);
    text(g, num(arr[ci], 3), x0 + w, y0 + 9, { size: 9.5, kind: 'mono', color: COL.ink2, align: 'right' });
    g.strokeStyle = COL.line; g.strokeRect(x0, y0 + 14, w, h);
    env.hit(x0, y0, w, rh, { tipAt: (wx) => { const tt = ((wx - x0) / w) * (S - 1); let bi = 0; sr.xs.forEach((xx, i) => { if (Math.abs(xx - tt) < Math.abs(sr.xs[bi] - tt)) bi = i; }); return `<span class="k">${L(`第 ${sr.xs[bi] + 1} 步`, `step ${sr.xs[bi] + 1}`)}</span>${lab} = <span class="v">${num(arr[bi], 5)}</span>`; } });
  });
}

// 一张“逐项相乘、加起来”的表（前向和反向共用）：rows = [{t, a, b, prod}]
function termTable(g, C, D, groups, { x0, w, y, reveal, aCol, bCol, aKey, bKey, lh }) {
  const total = groups.reduce((a, gp) => a + gp.terms.length, 0);
  const maxAbs = Math.max(1e-30, ...groups.flatMap((gp) => gp.terms.map((tm) => Math.abs(tm[aKey] * tm[bKey]))));
  let shown = 0, sum = 0;
  const narrow = w < 360;
  const cA = x0 + (narrow ? 96 : 120), cB = x0 + (narrow ? 196 : 250);
  for (const gp of groups) {
    if (gp.note) { text(g, gp.note, x0, y + 4, { size: 10, color: COL.dim, max: w }); y += 16; }
    text(g, L('位置', 'pos'), x0, y + 10, { size: 9.5, color: COL.dim });
    text(g, gp.xl, cA, y + 10, { size: 9.5, kind: 'mono', color: aCol, align: 'right', max: narrow ? 90 : 116 });
    text(g, gp.bl, cB, y + 10, { size: 9.5, kind: 'mono', color: bCol, align: 'right', max: narrow ? 96 : 126 });
    text(g, L('乘积', 'product'), x0 + w, y + 10, { size: 9.5, color: COL.amber, align: 'right' });
    y += 16;
    if (!gp.terms.length) { text(g, L('（第 0 段里没有这个字，这一路是 0）', '(this character isn’t in row 0, so this path is 0)'), x0, y + 12, { size: 10.5, color: COL.faint }); y += lh; }
    for (const tm of gp.terms) {
      const vis = clamp(reveal * total - shown, 0, 1);
      shown++;
      const prod = tm[aKey] * tm[bKey];
      if (vis <= 0) { y += lh; continue; }
      g.globalAlpha = vis;
      sum += prod;
      text(g, `${tm.t}「${D.ch(D.fixed[tm.t])}」`, x0, y + 13, { size: 10.5, color: COL.ink2 });
      text(g, num(tm[aKey], 3), cA, y + 13, { size: 10.5, kind: 'mono', color: aCol, align: 'right' });
      text(g, '×', cA + (narrow ? 10 : 14), y + 13, { size: 10, color: COL.dim, align: 'center' });
      text(g, num(tm[bKey], 3), cB, y + 13, { size: 10.5, kind: 'mono', color: bCol, align: 'right' });
      const bx = cB + 12, bw = w - (cB - x0) - 12 - (narrow ? 62 : 74);
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
  return { y, sum };
}

// 前向：这个参数乘了谁
export function drawMul(g, C, D, k, gi, st, env) {
  const on = st.step.mi === 'mul', lc = D.locate(gi), p = lc.p;
  card(g, C.x, C.y, C.w, C.h, { eyebrow: L('FORWARD · 它乘了谁', 'FORWARD · WHAT IT MULTIPLIES'), title: L('输入 × 它，写进输出的一格', 'input × it, into one output cell'), accent: COL.blue, active: on });
  const ft = fwdTerms(D, k, gi);
  const x0 = C.x + 16, w = C.w - 32;
  let y = C.y + 62;
  const rule = p.name === 'E'
    ? L(`E 用了两次：开头每个「${D.ch(lc.i)}」查表时直接取出这一行；最后每个位置给「${D.ch(lc.i)}」打分时乘上这一行。`, `E is used twice: each “${D.ch(lc.i)}” at the input copies this row; at the end every position scores “${D.ch(lc.i)}” with it.`)
    : p.norm ? L(`RMSNorm 把每个位置的向量除以均方根，再把第 ${lc.i} 维乘上 γ[${lc.i}] = ${num(ft.w, 4)}。`, `RMSNorm divides each position’s vector by its RMS, then multiplies dim ${lc.i} by γ[${lc.i}] = ${num(ft.w, 4)}.`)
      : L(`y = x · W：每个位置的输出第 ${lc.j} 维 = Σ_i x[i] × W[i, ${lc.j}]，其中一项就是 x[${lc.i}] × 它（${num(ft.w, 4)}）。`, `y = x · W: each position’s output dim ${lc.j} = Σ_i x[i] × W[i, ${lc.j}]; one term is x[${lc.i}] × it (${num(ft.w, 4)}).`);
  y += wrap(g, rule, x0, y, w, 15, { size: 11, color: COL.ink2 }) * 15 + 6;
  const reveal = on ? ease(seg(st.p, 0, 0.7)) : 1;
  const groups = ft.groups.map((gp) => ({ ...gp, bl: 'w', terms: gp.terms.map((tm) => ({ ...tm, w: ft.w })) }));
  const r = termTable(g, C, D, groups, { x0, w, y, reveal, aCol: COL.blue, bCol: COL.violet, aKey: 'x', bKey: 'w', lh: C.w < 400 ? 18 : 19 });
  y = r.y;
  if (reveal >= 1 && y < C.y + C.h - 30) {
    g.strokeStyle = COL.line2; g.beginPath(); g.moveTo(x0, y); g.lineTo(x0 + w, y); g.stroke();
    const gp = ft.groups[ft.groups.length - 1], tm = gp.terms[gp.terms.length - 1];
    if (tm) {
      const share = gp.n > 1 && Math.abs(tm.y) > 1e-9 ? (tm.x * ft.w) / tm.y : null;
      const s = gp.n > 1
        ? L(`位置 ${tm.t}：这一项 ${num(tm.x * ft.w, 3)}，整格 ${gp.yl.replace('t', String(tm.t))} = ${num(tm.y, 4)}（${gp.n} 项加起来）${share != null ? `，它占 ${(share * 100).toFixed(0)}%` : ''}`, `position ${tm.t}: this term ${num(tm.x * ft.w, 3)}, the whole cell ${gp.yl.replace('t', String(tm.t))} = ${num(tm.y, 4)} (sum of ${gp.n} terms)${share != null ? `, it is ${(share * 100).toFixed(0)}%` : ''}`)
        : L(`记录的 ${gp.yl.replace('t', String(tm.t))} = ${num(tm.y, 4)}`, `recorded ${gp.yl.replace('t', String(tm.t))} = ${num(tm.y, 4)}`);
      wrap(g, s, x0, y + 16, w, 14, { size: 10.5, color: COL.cyan, maxLines: 2 });
    }
  }
}

// 反向：梯度从哪来（第 0 段 8 个位置逐项，其余 7 段合成一项）
export function drawChain(g, C, D, k, gi, st, env) {
  const on = st.step.mi === 'chain', lc = D.locate(gi), p = lc.p;
  card(g, C.x, C.y, C.w, C.h, { eyebrow: L('CHAIN RULE · 梯度从哪来', 'CHAIN RULE · WHERE THE GRADIENT COMES FROM'), title: L('输入 × 上游梯度，64 个位置加起来', 'input × upstream gradient, summed over 64 positions'), accent: COL.rose, active: on });
  const ch = chainTerms(D, k, gi), G = D.G(k)[gi];
  const x0 = C.x + 16, w = C.w - 32;
  let y = C.y + 62;
  const rule = p.name === 'E'
    ? L('E 既在输入端被查表、又在输出端乘 Eᵀ，所以梯度有两路。', 'E is looked up at the input and multiplied as Eᵀ at the output, so its gradient has two paths.')
    : p.norm ? L('γ 逐维乘在归一化后的向量上：∂L/∂γ = Σ（归一化后的值 × 流到输出的梯度）。', 'γ scales each dimension of the normalized vector: ∂L/∂γ = Σ (normalized value × gradient at the output).')
      : L(`y = x·W，所以 ∂L/∂W[${lc.i}, ${lc.j}] = Σ 输入 x[${lc.i}] × 上游梯度 ∂L/∂y[${lc.j}]。`, `y = x·W, so ∂L/∂W[${lc.i}, ${lc.j}] = Σ input x[${lc.i}] × upstream gradient ∂L/∂y[${lc.j}].`);
  y += wrap(g, rule, x0, y, w, 15, { size: 11, color: COL.ink2 }) * 15 + 6;
  const reveal = on ? ease(seg(st.p, 0, 0.7)) : 1;
  const groups = ch.groups.map((gp) => ({ ...gp, bl: gp.dl }));
  const r = termTable(g, C, D, groups, { x0, w, y, reveal, aCol: COL.blue, bCol: COL.rose, aKey: 'x', bKey: 'd', lh: C.w < 400 ? 18 : 19 });
  y = r.y;
  if (reveal < 1) return;
  g.strokeStyle = COL.line2; g.beginPath(); g.moveTo(x0, y); g.lineTo(x0 + w, y); g.stroke();
  y += 6;
  text(g, L('第 0 段这 8 个位置合计', 'row 0 (these 8 positions)'), x0, y + 13, { size: 10.5, color: COL.ink2 });
  text(g, num(r.sum, 4), x0 + w, y + 13, { size: 11, kind: 'mono', color: COL.amber, align: 'right' });
  text(g, L('其余 7 段（56 个位置）= 总梯度 − 第 0 段', 'other 7 rows (56 positions) = total − row 0'), x0, y + 31, { size: 10.5, color: COL.ink2, max: w - 90 });
  text(g, num(G - r.sum, 4), x0 + w, y + 31, { size: 11, kind: 'mono', color: COL.amber, align: 'right' });
  rr(g, x0 - 6, y + 40, w + 12, 26, 7); g.fillStyle = hexA(COL.cyan, 0.08); g.fill();
  text(g, L(`总梯度 ∂L/∂${TENSOR_LABEL[p.name]}（记录的）`, `total ∂L/∂${TENSOR_LABEL[p.name]} (recorded)`), x0, y + 58, { size: 11, color: COL.ink, weight: 600 });
  text(g, num(G, 4), x0 + w, y + 58, { size: 12.5, kind: 'mono', weight: 700, color: COL.cyan, align: 'right' });
}

// 更新：这一步的 AdamW，全是真实数字，最后和记录对一下
export function drawAdam(g, C, D, k, gi, st, env) {
  const s = st.step, a = adamAt(D, k, gi), on = s.ph === 'upd';
  card(g, C.x, C.y, C.w, C.h, { eyebrow: L(`ADAMW · 第 ${a.t} 步`, `ADAMW · STEP ${a.t}`), title: L('这一步它挪了多少', 'How far it moves this step'), accent: COL.violet, active: on });
  const narrow = C.w < 400;
  const cur = on ? ADAM_SUBS.indexOf(s.mi) : -1;
  const x0 = C.x + 16, fx = x0 + (isEn ? (narrow ? 90 : 96) : (narrow ? 64 : 84)), mw = C.x + C.w - fx - 12;
  text(g, `lr = ${sciSup(a.lr, 3)}　β₁ = ${a.b1}　β₂ = ${a.b2}　ε = 10⁻⁸　λ = ${a.wd}　t = ${a.t}`, x0, C.y + 60, { size: 10, kind: 'mono', color: COL.dim, max: C.w - 32 });
  const lines = [
    { k: L('梯度', 'gradient'), l: `g = ∂L/∂w × ${L('裁剪系数', 'clip')} = ${par(a.graw)} × ${a.clip < 1 ? a.clip.toPrecision(4) : '1'}`, r: `g = ${num(a.g)}`, c: COL.rose },
    { k: L('一阶动量', '1st moment'), l: `m = ${a.b1} × ${par(a.m0)} + ${(1 - a.b1).toFixed(1)} × ${par(a.g)}`, r: `m = ${num(a.m)}`, c: COL.amber },
    { k: L('二阶动量', '2nd moment'), l: `v = ${a.b2} × ${par(a.v0, 3)} + ${(1 - a.b2).toFixed(2)} × ${par(a.g, 3)}²`, r: `v = ${num(a.v)}`, c: COL.amber },
    { k: L('偏差校正', 'bias corr.'), l: `m̂ = m / (1 − ${a.b1}^${a.t}),  v̂ = v / (1 − ${a.b2}^${a.t})`, r: `m̂ = ${num(a.mh)},  v̂ = ${num(a.vh, 3)}`, c: COL.amber },
    { k: 'Δw', l: `−lr·m̂/(√v̂+ε) − lr·λ·w = ${sciSup(a.adam, 3)} ${a.decay >= 0 ? '+' : '−'} ${sciSup(Math.abs(a.decay), 3)}`, r: `Δw = ${sciSup(a.dw, 4)}`, c: COL.cyan },
    { k: L('写回', 'write'), l: `w′ = ${num(a.w, 7)} + (${sciSup(a.dw, 3)})`, r: `w′ = ${num(a.w1, 7)}`, c: COL.cyan },
  ];
  const avail = C.h - 78 - (narrow ? 96 : 76);
  const lh = Math.max(40, Math.min(narrow ? 56 : 52, avail / 6));
  let y = C.y + 72;
  lines.forEach((ln, i) => {
    const vis = cur < 0 ? 1 : i < cur ? 1 : i === cur ? ease(seg(st.p, 0, 0.35)) : 0.14;
    g.globalAlpha = vis;
    if (i === cur) { rr(g, C.x + 8, y - 4, C.w - 16, lh - 6, 8); g.fillStyle = hexA(COL.amber, 0.08); g.fill(); g.fillStyle = COL.amber; g.fillRect(C.x + 8, y - 4, 2.5, lh - 6); }
    text(g, ln.k, x0 + 4, y + 13, { size: 10.5, color: i === cur ? COL.amber : COL.dim });
    text(g, ln.l, fx, y + 13, { size: narrow ? 10 : 10.5, kind: 'mono', color: COL.ink2, max: mw });
    text(g, ln.r, fx, y + 32, { size: narrow ? 11.5 : 12.5, kind: 'mono', weight: 700, color: i === cur ? ln.c : COL.ink, max: mw });
    g.globalAlpha = 1;
    y += lh;
  });
  y += 2;
  const showCheck = cur < 0 || cur >= ADAM_SUBS.length - 1;
  g.globalAlpha = showCheck ? 1 : 0.18;
  const bh = narrow ? 88 : 70;
  rr(g, C.x + 10, y, C.w - 20, bh, 8); g.fillStyle = 'rgba(255,255,255,0.03)'; g.fill(); g.strokeStyle = COL.line2; g.stroke();
  if (a.exact) {
    const d = a.w1 - a.rec.w1;
    text(g, L('记录的下一步 w（float32）', 'recorded next w (float32)'), x0, y + 19, { size: 10.5, color: COL.ink2 });
    text(g, num(a.rec.w1, 7), C.x + C.w - 18, y + 19, { size: 12, kind: 'mono', weight: 700, color: COL.cyan, align: 'right' });
    wrap(g, L(`按公式算出的和记录的相差 ${Math.abs(d) < 1e-12 ? '0' : sciSup(Math.abs(d), 2)}（这里用双精度算；导出脚本用 PyTorch 同样的 float32 运算重算全部 585,600 次更新，逐位一致）`, `Formula vs record differ by ${Math.abs(d) < 1e-12 ? '0' : sciSup(Math.abs(d), 2)} (computed here in double precision; the export script redoes all 585,600 updates with PyTorch’s own float32 ops and matches bit for bit)`), x0, y + 37, C.w - 32, 13, { size: 9.5, color: COL.dim, maxLines: narrow ? 4 : 3 });
  } else {
    const rel = Math.abs(a.rec.dw) > 0 ? Math.abs(a.dw - a.rec.dw) / Math.abs(a.rec.dw) : 0;
    text(g, L('记录的 Δw', 'recorded Δw'), x0, y + 19, { size: 10.5, color: COL.ink2 });
    text(g, sciSup(a.rec.dw, 4), C.x + C.w - 18, y + 19, { size: 12, kind: 'mono', weight: 700, color: COL.cyan, align: 'right' });
    wrap(g, L(`相差 ${(rel * 100).toFixed(rel < 0.001 ? 3 : 2)}%：这里的 w、g、m、v 是 float16 记录（约 3 位有效数字）。三个 γ 和「月」那一行存了 float32 原值，点它们能逐位对上。`, `Differ by ${(rel * 100).toFixed(rel < 0.001 ? 3 : 2)}%: w, g, m, v here are float16 records (~3 significant digits). The three γ and the row of 月 are float32 — click them for an exact match.`), x0, y + 37, C.w - 32, 13, { size: 9.5, color: COL.dim, maxLines: narrow ? 4 : 3 });
  }
  g.globalAlpha = 1;
  void env;
}

export function drawWait(g, C, env, wait, label) { waitBox(g, C.x, C.y, C.w, C.h, env, wait, { withCard: true, label }); }
