// D3 · 更新：一个具体权重的 AdamW 算式（全是真实数字）+ 它在整段训练里的轨迹；
// D4 · 写回的比特：更新前后的 fp32，以及如果只用 bf16 存会怎样。
import { COL, text, rr, card, hexA, clamp, ease, seg, sciSup, fmtInt, line, dot, pill, wrap, measure, waitBox } from '../draw.js';
import { f32Bits, bf16Round } from '../explain.js';
import { UPD_SUBS } from '../timeline.js';
import { esc } from '../../../js/ui.js';
import { isEn, L as tr, featLabel } from '../lang.js';

const B1 = 0.9, B2 = 0.95, EPS = 1e-8;
const num = (v, d = 4) => (v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e4 ? Number(v.toPrecision(d + 1)).toString() : sciSup(v, d));
const par = (v, d = 4) => (v < 0 ? `(${num(v, d)})` : num(v, d));

export class AdamView {
  constructor(app, R) { this.app = app; this.R = R; }

  layout(env) {
    this.portrait = env.portrait;
    const P = this.portrait;
    if (!P) {
      this.chips = { x: 0, y: 64, w: 1160 };
      this.form = { x: 0, y: 110, w: 600, h: 520 };
      this.hist = { x: 620, y: 110, w: 540, h: 520 };
      this.bits = { x: 0, y: 650, w: 1160, h: 380 };
      this.W = 1160;
    } else {
      this.chips = { x: 0, y: 64, w: 380 };
      this.form = { x: 0, y: 150, w: 380, h: 500 };
      this.hist = { x: 0, y: 664, w: 380, h: 470 };
      this.bits = { x: 0, y: 1148, w: 380, h: 600 };
      this.W = 380;
    }
  }

  focus(st) {
    if (st.step.mi) { const b = this.bits; return { x: b.x - 12, y: b.y - 12, w: b.w + 24, h: b.h + 24 }; }
    return this.portrait ? { x: -6, y: 50, w: 392, h: 606 } : { x: -12, y: 0, w: 1184, h: 640 };
  }

  draw(g, st, env) {
    const R = this.R, k = st.k, s = st.step, P = this.portrait;
    const f = this.app.ctx.feat ?? 0;
    const ft = R.feats[f];
    const a = R.adam(k, f);
    text(g, tr(`ADAMW · 第 ${a.t} 步 · 一个权重的完整算式`, `ADAMW · STEP ${a.t} · ONE WEIGHT, THE FULL FORMULA`), 0, 18, { size: 10, kind: 'mono', color: COL.dim });
    text(g, tr('更新：每个权重各算各的', 'Update: every weight on its own'), 0, 52, { size: P ? 19 : 24, kind: 'serif', weight: 900, color: COL.ink });
    // 选权重
    let x = this.chips.x, y = this.chips.y;
    R.feats.forEach((ff, i) => {
      const lab = featLabel(ff, R.kind);
      const w = measure(g, lab, 11) + 22;
      if (x + w > this.chips.x + this.chips.w) { x = this.chips.x; y += 32; }
      pill(g, x, y, w, 26, lab, { on: i === f, color: COL.violet, size: 11 });
      env.hit(x, y, w, 26, { click: true, act: () => { this.app.setFeat(i); this.app.sfx('click'); }, tip: `<span class="k">${tr('换一个权重看', 'look at another weight')}</span>${esc(ff.name)}[${ff.index.join(', ')}]${ff.decay ? '' : tr('<br>RMSNorm 的缩放：不做权重衰减', '<br>RMSNorm scale: no weight decay')}` });
      x += w + 8;
    });
    this.drawForm(g, st, env, a, ft);
    this.drawHist(g, st, env, a, ft, f);
    this.drawBits(g, st, env, a);
  }

  drawForm(g, st, env, a, ft) {
    const R = this.R, k = st.k, s = st.step, C = this.form, P = this.portrait;
    const lr = R.lr(k);
    card(g, C.x, C.y, C.w, C.h, { eyebrow: `${ft.name}[${ft.index.join(', ')}]`, title: featLabel(ft, R.kind), accent: COL.violet, active: !!s.sub });
    const cur = s.sub ? UPD_SUBS.indexOf(s.sub) : UPD_SUBS.length;
    const t = a.t;
    const adamPart = -lr * a.mh / (Math.sqrt(a.vh) + EPS), decay = -lr * a.wd * a.w0;
    const L = [
      { sub: 'clip', k: tr('裁剪', 'clip'), l: `‖g‖ = ${R.gradNorm(k).toFixed(4)} ${R.clip(k) < 1 ? tr(`> 1，乘 ${R.clip(k).toPrecision(5)}`, `> 1, × ${R.clip(k).toPrecision(5)}`) : tr('≤ 1，不变', '≤ 1, unchanged')}`, r: `g = ${par(a.gRaw)} × ${R.clip(k).toPrecision(4)}` },
      { sub: 'g', k: tr('梯度', 'gradient'), l: `g = ∂L/∂w`, r: `= ${num(a.g)}` },
      { sub: 'm', k: tr('一阶动量', '1st moment'), l: `m = 0.9 × ${par(a.m0)} + 0.1 × ${par(a.g)}`, r: `= ${num(a.m)}` },
      { sub: 'v', k: tr('二阶动量', '2nd moment'), l: `v = 0.95 × ${par(a.v0)} + 0.05 × ${par(a.g)}²`, r: `= ${num(a.v)}` },
      { sub: 'bc', k: tr('偏差校正', 'bias corr.'), l: `m̂ = m / (1 − 0.9^${t}),  v̂ = v / (1 − 0.95^${t})`, r: `m̂ = ${num(a.mh)},  v̂ = ${num(a.vh)}` },
      { sub: 'dw', k: 'Δw', l: `−lr·m̂/(√v̂+ε) − lr·λ·w = ${sciSup(adamPart, 3)} ${decay >= 0 ? '+' : '−'} ${sciSup(Math.abs(decay), 3)}`, r: `Δw = ${sciSup(adamPart + decay, 4)}` },
      { sub: 'write', k: tr('写回', 'write'), l: `w′ = ${num(a.w0, 8)} + (${sciSup(a.w1 - a.w0, 3)})`, r: `= ${num(a.w1, 8)}` },
    ];
    const x0 = C.x + 16, lh = P ? 56 : 58;
    let y = C.y + 76;
    text(g, `lr = ${sciSup(lr, 3)}　β₁ = 0.9　β₂ = 0.95　ε = 10⁻⁸　λ = ${a.wd}　t = ${t}`, x0, y - 8, { size: 10.5, kind: 'mono', color: COL.dim, max: C.w - 32 });
    y += 10;
    L.forEach((ln, i) => {
      const vis = i < cur ? 1 : i === cur ? ease(seg(st.p, 0, 0.35)) : s.sub ? 0.12 : 1;
      g.globalAlpha = vis;
      if (i === cur) { rr(g, C.x + 8, y - 4, C.w - 16, lh - 8, 8); g.fillStyle = hexA(COL.amber, 0.08); g.fill(); g.fillStyle = COL.amber; g.fillRect(C.x + 8, y - 4, 2.5, lh - 8); }
      text(g, ln.k, x0 + 4, y + 13, { size: 10.5, color: i === cur ? COL.amber : COL.dim });
      const fx = x0 + (P ? 70 : 82);
      text(g, ln.l, fx, y + 13, { size: P ? 10.5 : 11.5, kind: 'mono', color: COL.ink2, max: C.x + C.w - fx - 12 });
      text(g, ln.r, fx, y + 34, { size: P ? 12 : 13, kind: 'mono', weight: 700, color: i === cur ? COL.amber : COL.ink, max: C.x + C.w - fx - 12 });
      g.globalAlpha = 1;
      y += lh;
    });
    if (t === 1 && cur >= 4) text(g, tr('第 1 步：m̂ = g、v̂ = g²，更新只剩下 g 的符号', 'Step 1: m̂ = g and v̂ = g², so the update is just the sign of g'), x0, C.y + C.h - 16, { size: 10.5, color: COL.violet, max: C.w - 32 });
  }

  drawHist(g, st, env, a, ft, f) {
    const R = this.R, k = st.k, C = this.hist, T = R.kind === 'tiny', P = this.portrait;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: T ? tr(`HISTORY · 全部 ${fmtInt(R.D.meta.train.steps)} 步（每 4 步一个点）`, `HISTORY · ALL ${fmtInt(R.D.meta.train.steps)} STEPS (A POINT EVERY 4)`) : tr('HISTORY · 3 步', 'HISTORY · 3 STEPS'), title: tr('这个权重的一生', 'The life of this weight'), accent: COL.cyan });
    const series = isEn ? [['w', 'weight w', COL.ink], ['g', 'gradient g (clipped)', COL.rose], ['m', 'first moment m', COL.amber], ['v', '√v (typical gradient size)', COL.violet]] : [['w', '权重 w', COL.ink], ['g', '梯度 g（裁剪后）', COL.rose], ['m', '一阶动量 m', COL.amber], ['v', '√v（梯度的典型大小）', COL.violet]];
    const x0 = C.x + 16, w = C.w - 32;
    const ch = (C.h - 80) / 4;
    const tNow = a.t;
    if (T && !R.D.has('feat')) {
      // 4 个权重的完整轨迹单独一块，还没到时先占位
      waitBox(g, C.x + 14, C.y + 50, C.w - 28, C.h - 64, env, st.wait, { label: tr('正在载入这个权重的轨迹…', 'Loading this weight’s history…') });
    } else if (T) {
      const H = R.featHist(f);
      const n = H.n, stride = H.stride;
      series.forEach(([key, lab, col], j) => {
        const y0 = C.y + 62 + j * ch, h = ch - 26;
        const arr = key === 'v' ? Array.from(H.v, Math.sqrt) : Array.from(H[key]);
        let lo = Math.min(...arr), hi = Math.max(...arr);
        if (key !== 'w') { const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = key === 'v' ? 0 : -m; hi = m; }
        const X = (i) => x0 + (i / (n - 1)) * w, Y = (v) => y0 + 18 + h - ((v - lo) / (hi - lo || 1)) * h;
        text(g, lab, x0, y0 + 10, { size: 10.5, color: col });
        if (key !== 'w' && key !== 'v') { g.strokeStyle = COL.line; g.beginPath(); g.moveTo(x0, Y(0)); g.lineTo(x0 + w, Y(0)); g.stroke(); }
        g.beginPath();
        for (let i = 0; i < n; i++) { const xx = X(i), yy = Y(arr[i]); i ? g.lineTo(xx, yy) : g.moveTo(xx, yy); }
        g.strokeStyle = hexA(col.startsWith('#') ? col : '#e9eff9', key === 'g' ? 0.45 : 0.9);
        g.lineWidth = key === 'g' ? 0.8 : 1.4;
        g.stroke();
        const ci = clamp(Math.round((tNow - 1) / stride), 0, n - 1);
        const cx = X(ci);
        g.strokeStyle = hexA(COL.amber, 0.5);
        g.beginPath(); g.moveTo(cx, y0 + 14); g.lineTo(cx, y0 + 18 + h); g.stroke();
        dot(g, cx, Y(arr[ci]), 3, COL.amber);
        text(g, key === 'w' ? `${num(hi, 3)}` : sciSup(hi, 2), x0 + w, y0 + 10, { size: 9, kind: 'mono', color: COL.faint, align: 'right' });
        env.hit(x0, y0, w, ch, { tipAt: (wx) => { const i = clamp(Math.round(((wx - x0) / w) * (n - 1)), 0, n - 1); return `<span class="k">${tr(`第 ${fmtInt(i * stride + 1)} 步之后`, `after step ${fmtInt(i * stride + 1)}`)}</span>${lab} = <span class="v">${num(arr[i], 5)}</span>`; } });
      });
      text(g, tr('梯度 g 噪声很大；m 把它平滑成一个稳定的方向；w 沿着 −m̂/√v̂ 慢慢挪', 'g is very noisy; m smooths it into a steady direction; w creeps along −m̂/√v̂'), x0, C.y + C.h - 12, { size: 10, color: COL.dim, max: w });
    } else {
      // 只有 3 步：直接列成表
      const m = R.D.meta;
      const rows = [[tr('w（更新前）', 'w (before)'), 'w0'], [tr('g（裁剪后）', 'g (clipped)'), 'g'], ['m', 'm'], ['v', 'v'], ['m̂', 'mh'], ['v̂', 'vh'], ['Δw', 'dw'], [tr('w（更新后）', 'w (after)'), 'w1']];
      const cx0 = x0 + (P ? 70 : 110), cwid = (w - (cx0 - x0)) / 3;
      for (let t = 0; t < 3; t++) {
        const on = t === tNow - 1;
        if (on) { rr(g, cx0 + t * cwid + 2, C.y + 60, cwid - 4, rows.length * 44 + 30, 8); g.fillStyle = hexA(COL.amber, 0.07); g.fill(); }
        text(g, tr(`第 ${t + 1} 步`, `step ${t + 1}`), cx0 + t * cwid + cwid - 10, C.y + 80, { size: 11, color: on ? COL.amber : COL.dim, align: 'right' });
      }
      rows.forEach(([lab, key], j) => {
        const yy = C.y + 116 + j * 44;
        text(g, lab, x0, yy, { size: 11, color: COL.ink2 });
        for (let t = 0; t < 3; t++) {
          const d = m.steps[t].feats[f];
          const v = key === 'dw' ? d.w1 - d.w0 : d[key];
          const on = t === tNow - 1;
          text(g, key === 'w0' || key === 'w1' ? num(v, 7) : sciSup(v, 3), cx0 + t * cwid + cwid - 10, yy, { size: P ? 10 : 11.5, kind: 'mono', color: on ? COL.amber : COL.ink, align: 'right', max: cwid - 8 });
        }
      });
      wrap(g, tr('第 1 步 m̂/√v̂ ≈ ±1：几乎每个权重都挪了整整一个 lr；之后 m、v 有了历史，步子开始各不相同。', 'At step 1, m̂/√v̂ ≈ ±1: almost every weight moves exactly one lr; after that m and v have a history and the steps start to differ.'), x0, C.y + C.h - 30, w, 16, { size: 10.5, color: COL.dim });
    }
  }

  drawBits(g, st, env, a) {
    const s = st.step, C = this.bits, P = this.portrait;
    const sub = s.mi ? ['fp32a', 'fp32b', 'bf16'].indexOf(s.mi) : -1;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: tr('BITS · 写回内存的那个数', 'BITS · THE NUMBER WRITTEN BACK TO MEMORY'), title: tr('比特：为什么要 fp32 主权重', 'Bits: why fp32 master weights'), accent: COL.amber, active: sub >= 0 });
    const b0 = f32Bits(a.w0), b1 = f32Bits(a.w1);
    const r0 = bf16Round(a.w0), r1 = bf16Round(a.w1);
    const rb0 = f32Bits(r0) >>> 16, rb1 = f32Bits(r1) >>> 16;
    const kw = P ? 10.4 : 24, kh = P ? 22 : 32, gap = P ? 0.8 : 3;
    const x0 = C.x + (P ? 10 : 150);
    const rows = [
      { lab: tr('fp32 · 更新前', 'fp32 · before'), bits: b0, n: 32, val: a.w0, vis: sub < 0 || sub >= 0 },
      { lab: tr('fp32 · 更新后', 'fp32 · after'), bits: b1, n: 32, val: a.w1, vis: sub < 0 || sub >= 1, cmp: b0 },
      { lab: tr('bf16 · 更新前', 'bf16 · before'), bits: rb0, n: 16, val: r0, vis: sub < 0 || sub >= 2 },
      { lab: tr('bf16 · 更新后', 'bf16 · after'), bits: rb1, n: 16, val: r1, vis: sub < 0 || sub >= 2, cmp: rb0 },
    ];
    rows.forEach((row, ri) => {
      const y = C.y + (P ? 80 : 72) + ri * (P ? 112 : 66) + (ri >= 2 ? (P ? 18 : 18) : 0);
      if (!row.vis) { g.globalAlpha = 0.12; }
      text(g, row.lab, P ? C.x + 10 : C.x + 16, P ? y - 8 : y + kh / 2 + 4, { size: P ? 10.5 : 11.5, color: COL.ink2 });
      for (let i = 0; i < row.n; i++) {
        const bit = (row.bits >>> (row.n - 1 - i)) & 1;
        const field = i === 0 ? 'sign' : i <= 8 ? 'exp' : 'man';
        const col = field === 'sign' ? COL.rose : field === 'exp' ? COL.amber : COL.cyan;
        const changed = row.cmp != null && (((row.bits ^ row.cmp) >>> (row.n - 1 - i)) & 1);
        const bx = x0 + i * (kw + gap), by = P ? y : y;
        rr(g, bx, by, kw, kh, P ? 2 : 5);
        g.fillStyle = bit ? hexA(col, 0.35) : 'rgba(255,255,255,0.03)';
        g.fill();
        g.strokeStyle = changed ? '#fff' : hexA(col, 0.5);
        g.lineWidth = changed ? 1.8 : 1;
        g.stroke();
        text(g, bit, bx + kw / 2, by + kh / 2 + (P ? 3.5 : 5), { size: P ? 9 : 14, kind: 'mono', color: bit ? COL.ink : COL.faint, align: 'center' });
      }
      const vx = P ? C.x + 10 : x0 + 32 * (kw + gap) + 14;
      text(g, num(row.val, 9), vx, P ? y + kh + 16 : y + kh / 2 + 5, { size: P ? 11 : 13, kind: 'mono', color: COL.ink, weight: 700 });
      if (row.cmp != null) {
        let d = 0;
        for (let i = 0; i < row.n; i++) if (((row.bits ^ row.cmp) >>> i) & 1) d++;
        text(g, d ? tr(`变了 ${d} 个比特`, `${d} bits changed`) : tr('一个比特都没变', 'no bit changed'), vx + (P ? 130 : 0), P ? y + kh + 16 : y + kh / 2 + 22, { size: 10.5, color: d ? COL.amber : COL.rose });
      }
      g.globalAlpha = 1;
    });
    // 字段说明
    const ly = C.y + C.h - (P ? 40 : 34);
    const leg = isEn ? [[COL.rose, 'sign: 1 bit'], [COL.amber, 'exponent: 8 bits'], [COL.cyan, 'mantissa (fp32: 23 bits / bf16: 7)'], ['#ffffff', 'white frame = differs from before']] : [[COL.rose, '符号 1 位'], [COL.amber, '指数 8 位'], [COL.cyan, '尾数（fp32 23 位 / bf16 7 位）'], ['#ffffff', '白框 = 和更新前不同']];
    let lx = C.x + 16;
    for (const [c, sx] of leg) {
      if (P && lx + measure(g, sx, 10.5) + 24 > C.x + C.w) { lx = C.x + 16; }
      text(g, '■', lx, ly, { size: 11, color: c });
      text(g, sx, lx + 13, ly, { size: 10.5, color: COL.dim });
      lx += 30 + measure(g, sx, 10.5);
    }
    if (sub === 2 || sub < 0) {
      const lost = r0 === r1;
      text(g, lost ? tr('只用 bf16 存：更新前后是同一个数，这一步白走了', 'Stored only in bf16: the same number before and after — this step was wasted') : tr('只用 bf16 存：这次恰好跨过了一个 bf16 的间隔', 'Stored only in bf16: this time it happened to cross one bf16 spacing'), P ? C.x + 16 : x0, C.y + C.h - (P ? 18 : 12), { size: 11.5, color: lost ? COL.rose : COL.cyan, weight: 600, max: C.w - 32 });
    }
  }
}
