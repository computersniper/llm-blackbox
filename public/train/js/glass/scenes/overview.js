// 玻璃小模型 · D0 初始化 / D1 训练全程：全部 2,928 个参数按结构摆开的全景图 + 右侧几张卡片。
//   D0：初始值（正态 N(0, 0.02²)，γ = 1）→ 直方图 → 全零对照 → 放大 50 倍对照（三次真实训练的损失曲线）
//   D1：拖动 / 播放时全部权重跟着变；损失曲线、固定样例“举头望明月，低头”每个位置的预测、两个注意力头、嵌入的相似度
import { COL, text, rr, card, pill, line, dot, clamp, lerp, ease, fmtP, sciSup, fmtInt, hexA, wrap, waitBox, measure } from '../../draw.js';
import { heat, legend, fnum, cellAt, mark, rgb, divRGB, animP, paramName } from '../heat.js';
import { ModelMap } from '../model.js';
import { TENSOR_LABEL } from '../data.js';
import { isEn, L } from '../../lang.js';
import { esc } from '../../../../js/ui.js';

const INIT_SCALE = 0.06;          // D0 的色标：±3 个标准差
const disp = (s) => (s === '⏎' ? '⏎' : s);

// 一个确定的伪随机数（D0 里“一个个抽出来”的顺序）
const hash01 = (i) => { let x = (i + 1) * 2654435761 >>> 0; x ^= x >>> 15; x = Math.imul(x, 2246822519) >>> 0; x ^= x >>> 13; return (x >>> 0) / 4294967296; };

class Base {
  constructor(app, R) {
    this.app = app;
    this.R = R;
    this.D = R.D;
    this.map = new ModelMap(this.D);
  }
  // γ 从 1 开始：上色画的是 γ − 1
  isNorm(gi) { const lc = this.D.locate(gi); return lc && lc.p.norm; }
}

/* ================================================================ D0 初始化 */

export class GInit extends Base {
  layout(env) {
    const P = (this.portrait = env.portrait);
    if (!P) {
      this.map.layout(false, 0, 92);
      const x = 920, w = 350;
      this.cards = { list: { x, y: 60, w, h: 222 }, hist: { x, y: 296, w, h: 200 }, runs: { x, y: 510, w, h: 206 } };
      this.bounds = { x: -10, y: -6, w: 1290, h: 730 };
    } else {
      // 竖屏：对照组的损失曲线、直方图放在全景图前面（全零 / 放大 50 倍时先看到证据），参数清单放最后
      this.cards = { runs: { x: 0, y: 70, w: 380, h: 206 }, hist: { x: 0, y: 290, w: 380, h: 200 } };
      this.map.layout(true, 0, 540);
      const y = 540 + this.map.H + 10;
      this.cards.list = { x: 0, y, w: 380, h: 222 };
      this.bounds = { x: -8, y: -6, w: 396, h: y + 232 };
    }
  }

  focus(st) {
    if (!this.portrait) return this.bounds;
    const sub = st.step.sub;
    if (sub === 'model') return { x: -8, y: 480, w: 396, h: 620 };
    if (sub === 'hist') return { x: -8, y: 250, w: 396, h: 620 };
    return { x: -8, y: -6, w: 396, h: 520 };
  }

  draw(g, st, env) {
    st = { ...st, p: animP(this, st, env, 2.2) };
    const D = this.D, sub = st.step.sub, P = this.portrait;
    const titles = isEn
      ? { model: 'The whole model is these 2,928 numbers', hist: 'Where they start: small random numbers', zero: 'Why random? Start everything at zero and nothing moves', big: 'Why small? Start 50× bigger and it starts out confidently wrong' }
      : { model: '整个模型，就是这 2,928 个数', hist: '一开始：每个数都是一个很小的随机数', zero: '为什么要随机：全设成 0，就一步也学不动', big: '为什么要小：放大 50 倍，一开始就“很有把握地猜错”' };
    text(g, L('GLASS MODEL · 初始化 · 还没训练', 'GLASS MODEL · INITIALIZATION · BEFORE TRAINING'), 0, 16, { size: 10, kind: 'mono', color: COL.dim });
    text(g, titles[sub], 0, 46, { size: P ? 18 : 24, kind: 'serif', weight: 900, color: COL.ink, max: P ? 380 : 1120 });
    // 全景图：哪一种初始化
    const which = sub === 'zero' ? 'zero' : sub === 'big' ? 'big' : 'normal';
    const w0 = D.w0, bw = D.bW0;
    const reveal = sub === 'hist' ? ease(clamp(st.p * 1.6, 0, 1)) : 1;
    const val = (gi) => {
      const norm = this.isNorm(gi);
      if (norm) return 0;                             // γ 一律从 1 开始（画 γ − 1 = 0）
      if (which === 'zero') return 0;
      if (hash01(gi) > reveal) return 0;
      return which === 'big' ? bw[gi] : w0[gi];
    };
    const my = this.map.oy;
    const tag = { normal: L('正态初始化 N(0, 0.02²)', 'Normal init N(0, 0.02²)'), zero: L('全零初始化（γ 仍为 1）', 'All-zero init (γ still 1)'), big: L('放大 50 倍：N(0, 1²)', '50× larger: N(0, 1²)') }[which];
    if (!P) text(g, `${tag} · ${L(`色标 ±${INIT_SCALE}（γ 画的是 γ − 1）`, `color scale ±${INIT_SCALE} (γ drawn as γ − 1)`)}`, 0, 74, { size: 11.5, color: which === 'normal' ? COL.cyan : which === 'zero' ? COL.rose : COL.amber });
    this.map.draw(g, env, {
      value: val,
      scale: () => INIT_SCALE,
      tip: (name, gi) => {
        const v = this.isNorm(gi) ? 1 : which === 'zero' ? 0 : which === 'big' ? bw[gi] : w0[gi];
        return `<span class="k">${esc(paramName(D, gi))}</span>${L('初始值', 'initial value')} <span class="v">${fnum(v, 4)}</span>${this.isNorm(gi) ? L('<br>RMSNorm 的缩放 γ 一律从 1 开始', '<br>RMSNorm scales γ always start at 1') : ''}${which === 'big' && Math.abs(v) > INIT_SCALE ? L('<br>超出色标，按最亮画', '<br>beyond the color scale, drawn at full brightness') : ''}<br><span style="color:var(--dim)">${L('点一下看这个参数的一生', 'Click to see this parameter’s life')}</span>`;
      },
      pick: (name, gi) => this.app.pickParam(gi),
      sel: this.app.ctx.gsel,
    });
    if (!P) legend(g, this.map.ox + 30, my + this.map.H + 4, 150, INIT_SCALE, (v) => fnum(v, 2));
    this.drawList(g, st, env, sub === 'model');
    this.drawHist(g, st, env, sub === 'hist' || sub === 'big', which);
    this.drawRuns(g, st, env, sub === 'zero' || sub === 'big', which);
  }

  drawList(g, st, env, on) {
    const C = this.cards.list, D = this.D;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: L('PARAMETERS · 全部参数', 'PARAMETERS · ALL OF THEM'), title: L('11 个张量，2,928 个数', '11 tensors, 2,928 numbers'), accent: COL.violet, active: on });
    const rows = [
      ['E', L('词嵌入（输出层共用）', 'embeddings (shared with output)')],
      ['g1', L('注意力前的 RMSNorm', 'RMSNorm before attention')],
      ['Wq', L('W_q / W_k / W_v / W_o：注意力', 'W_q / W_k / W_v / W_o: attention')],
      ['g2', L('前馈前的 RMSNorm', 'RMSNorm before the FFN')],
      ['Wg', L('W_gate / W_up / W_down：前馈', 'W_gate / W_up / W_down: FFN')],
      ['gf', L('最后的 RMSNorm', 'final RMSNorm')],
    ];
    const cnt = { E: 320, g1: 16, Wq: 1024, g2: 16, Wg: 1536, gf: 16 };
    const shp = { E: '20 × 16', g1: '16', Wq: '4 × 16 × 16', g2: '16', Wg: '2 × 16 × 32 + 32 × 16', gf: '16' };
    let y = C.y + 66;
    for (const [n, lab] of rows) {
      text(g, lab, C.x + 14, y, { size: 11, color: COL.ink2, max: 200 });
      text(g, shp[n], C.x + C.w - 74, y, { size: 9.5, kind: 'mono', color: COL.dim, align: 'right' });
      text(g, fmtInt(cnt[n]), C.x + C.w - 14, y, { size: 11, kind: 'mono', color: COL.ink, align: 'right' });
      y += 22;
    }
    g.strokeStyle = COL.line2; g.beginPath(); g.moveTo(C.x + 14, y - 12); g.lineTo(C.x + C.w - 14, y - 12); g.stroke();
    text(g, L('合计', 'total'), C.x + 14, y + 6, { size: 11.5, color: COL.ink, weight: 600 });
    text(g, fmtInt(D.P), C.x + C.w - 14, y + 6, { size: 13, kind: 'mono', weight: 700, color: COL.violet, align: 'right' });
    text(g, L('对照：Qwen3-0.6B 约 6 亿个数，是同一类零件（28 层）', 'For scale: Qwen3-0.6B has ~600 million, the same kinds of parts (28 layers)'), C.x + 14, C.y + C.h - 12, { size: 10, color: COL.dim, max: C.w - 28 });
  }

  // 全零对照：训练 200 步以后，两次真实训练的 E 和 W_q 并排（全零那次 200 步里 max |w| 始终是 0，所以画出来就是全零）
  drawAfter(g, st, env) {
    const C = this.cards.hist, D = this.D, K = D.NF - 1;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: L('AFTER 200 STEPS · 两次真实训练并排', 'AFTER 200 STEPS · TWO REAL RUNS SIDE BY SIDE'), title: L('训练 200 步以后', 'After 200 training steps'), accent: COL.rose, active: true });
    const cols = [[L('正态初始化', 'normal init'), COL.cyan, false], [L('全零初始化', 'all-zero init'), COL.rose, true]];
    const have = D.has('w', K);
    const W = have ? D.W(K) : null;
    cols.forEach(([lab, col, zero], ci) => {
      const x0 = C.x + 18 + ci * ((C.w - 36) / 2);
      text(g, lab, x0, C.y + 58, { size: 11, color: col, weight: 600 });
      [['E', 20, 16, 3.6], ['Wq', 16, 16, 3.6]].forEach(([name, rows, ncols, cell], j) => {
        const p = D.pIndex.get(name), x = x0 + j * (ncols * cell + 14), y = C.y + 66;
        text(g, TENSOR_LABEL[name], x, y + rows * cell + 12, { size: 9, kind: 'mono', color: COL.dim });
        if (!zero && !have) { waitBox(g, x, y, ncols * cell, rows * cell, env, st.wait, { label: '', size: 9 }); return; }
        let sc = 0;
        if (!zero) for (let q = p.off; q < p.off + p.n; q++) sc = Math.max(sc, Math.abs(W[q]));
        heat(g, x, y, rows, ncols, cell, cell, (r, c) => (zero ? 0 : W[p.off + r * ncols + c]), { scale: sc * 0.8 || 1, s: env.s });
      });
    });
    const z = D.meta.runs.zero;
    wrap(g, L(`右边：200 步里梯度范数始终 ${z.maxGnorm}、max |w| 始终 ${z.maxAbsW}，损失一直 ${z.finalEval.toFixed(3)}。左边同样 200 步，损失降到 ${D.meta.train.finalEval.toFixed(3)}。`, `Right: over 200 steps the gradient norm stayed ${z.maxGnorm}, max |w| stayed ${z.maxAbsW}, loss stuck at ${z.finalEval.toFixed(3)}. Left, same 200 steps: loss down to ${D.meta.train.finalEval.toFixed(3)}.`), C.x + 18, C.y + C.h - 30, C.w - 36, 13, { size: 9.5, color: COL.dim, maxLines: 2 });
  }

  // 初始值的直方图（真实的 2,880 个矩阵参数）+ 理论的正态曲线
  drawHist(g, st, env, on, which) {
    if (which === 'zero') return this.drawAfter(g, st, env);
    const C = this.cards.hist, D = this.D;
    const big = which === 'big';
    card(g, C.x, C.y, C.w, C.h, { eyebrow: L('INIT · 初始值的分布（真实的 2,880 个）', 'INIT · DISTRIBUTION OF THE REAL 2,880'), title: big ? L('放大 50 倍：N(0, 1²)', '50× larger: N(0, 1²)') : L('正态分布 N(0, 0.02²)', 'Normal distribution N(0, 0.02²)'), accent: COL.cyan, active: on });
    const src = big ? D.bW0 : D.w0, std = big ? 1 : 0.02, lim = 4 * std;
    const nb = 32, cnt = new Float32Array(nb);
    let n = 0;
    for (const p of D.params) { if (p.norm) continue; for (let i = p.off; i < p.off + p.n; i++) { const b = Math.floor(((src[i] + lim) / (2 * lim)) * nb); if (b >= 0 && b < nb) cnt[b]++; n++; } }
    const x0 = C.x + 18, w = C.w - 36, y0 = C.y + 62, h = C.h - 98;
    const mx = Math.max(...cnt);
    const bw = w / nb;
    for (let b = 0; b < nb; b++) {
      const v = (b + 0.5) / nb * 2 * lim - lim;
      const bh = (cnt[b] / mx) * h;
      g.fillStyle = rgb(divRGB(big ? v / (INIT_SCALE * 50) : v / INIT_SCALE), 0.9);
      g.fillRect(x0 + b * bw + 0.5, y0 + h - bh, bw - 1, bh);
    }
    // 理论曲线
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
    const sd = Math.sqrt(s / n);
    text(g, L(`实际标准差 ${sd.toPrecision(3)} · 48 个 γ 都是 1`, `actual std ${sd.toPrecision(3)} · all 48 γ are 1`), C.x + 18, C.y + C.h - 12, { size: 10, color: COL.dim });
  }

  // 三次真实训练：正态 / 全零 / 放大 50 倍，同样的批次、同样 200 步
  drawRuns(g, st, env, on, which) {
    const C = this.cards.runs, D = this.D, S = D.S;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: L('THREE REAL RUNS · 同样的批次，各训练 200 步', 'THREE REAL RUNS · SAME BATCHES, 200 STEPS EACH'), title: L('三种初始化的损失', 'Loss under three inits'), accent: COL.amber, active: on });
    const x0 = C.x + 40, w = C.w - 56, y0 = C.y + 58, h = C.h - 92;
    const lo = Math.log(0.03), hi = Math.log(9);
    const Y = (v) => y0 + h - ((Math.log(Math.max(v, 0.03)) - lo) / (hi - lo)) * h, X = (t) => x0 + (t / (S - 1)) * w;
    for (const v of [0.1, 1, 3]) { const y = Y(v); g.strokeStyle = COL.line; g.beginPath(); g.moveTo(x0, y); g.lineTo(x0 + w, y); g.stroke(); text(g, String(v), x0 - 5, y + 3, { size: 9, kind: 'mono', color: COL.dim, align: 'right' }); }
    const series = [
      [D.evalLoss, COL.cyan, L('正态 0.02', 'normal 0.02'), 'normal'],
      [D.zLoss, COL.rose, L('全零', 'all zero'), 'zero'],
      [D.bEval, COL.amber, L('放大 50 倍', '50× larger'), 'big'],
    ];
    series.forEach(([arr, col, lab, key], si) => {
      const pts = [];
      for (let t = 0; t < S; t++) pts.push([X(t), Y(arr[t])]);
      const hot = !on || which === key || key === 'normal';
      line(g, pts, hexA(col, hot ? 0.95 : 0.3), hot ? 1.6 : 1);
      text(g, lab, C.x + 14 + si * 112, C.y + C.h - 12, { size: 10, color: hexA(col, hot ? 1 : 0.5) });
    });
    // ln 20：完全瞎猜
    text(g, `ln 20 = ${Math.log(20).toFixed(2)}`, x0 + w, Y(Math.log(20)) - 5, { size: 9, kind: 'mono', color: COL.rose, align: 'right' });
    text(g, L('全部 25 段上的平均损失（对数刻度）', 'mean loss over all 25 windows (log scale)'), x0, y0 - 6, { size: 9.5, color: COL.dim });
    env.hit(x0, y0, w, h, { tipAt: (wx) => { const t = clamp(Math.round(((wx - x0) / w) * (S - 1)), 0, S - 1); return `<span class="k">${L(`第 ${t + 1} 步`, `step ${t + 1}`)}</span>${L('正态', 'normal')} <span class="v">${D.evalLoss[t].toFixed(3)}</span> · ${L('全零', 'zero')} <span class="v">${D.zLoss[t].toFixed(3)}</span> · ${L('放大', '50×')} <span class="a">${D.bEval[t].toFixed(3)}</span>`; } });
  }
}

/* ================================================================ D1 训练全程 */

export class GRun extends Base {
  constructor(app, R) {
    super(app, R);
    this.mode = 'w';           // 'w' = 数值；'d' = 和初始化的差 w − w₀
    this.fixed = false;        // 色标固定为训练结束时
  }

  layout(env) {
    const P = (this.portrait = env.portrait);
    if (!P) {
      this.map.layout(false, 0, 92);
      const x = 920, w = 350;
      this.cards = { loss: { x, y: 60, w, h: 190 }, pred: { x, y: 262, w, h: 252 }, attn: { x, y: 526, w, h: 200 } };
      this.sim = { x: 0, y: 92 + 296, w: 170, h: 170 };
      this.bounds = { x: -10, y: -6, w: 1290, h: 732 };
    } else {
      this.cards = { loss: { x: 0, y: 92, w: 380, h: 196 }, pred: { x: 0, y: 302, w: 380, h: 262 } };
      this.map.layout(true, 0, 620);
      const y = 620 + this.map.H + 10;
      this.cards.attn = { x: 0, y, w: 380, h: 186 };
      this.sim = { x: 0, y: y + 200, w: 380, h: 330 };
      this.bounds = { x: -8, y: -6, w: 396, h: y + 540 };
    }
  }

  focus() { return this.portrait ? { x: -8, y: -6, w: 396, h: 900 } : this.bounds; }

  // 播放头所在的“连续”步数和插值系数
  pos(st) {
    const D = this.D, k = st.k;
    const t0 = D.FR[k];
    if (k >= D.NF - 1 || st.depth !== 1) return { t: t0, f: 0, k1: k };
    return { t: lerp(t0, D.FR[k + 1], st.p), f: st.p, k1: k + 1 };
  }

  draw(g, st, env) {
    const D = this.D, k = st.k, P = this.portrait;
    const { t, f, k1 } = this.pos(st);
    const haveW = D.has('w', k);
    const tInt = Math.round(t);
    text(g, L(`GLASS MODEL · 训练全程 · 第 ${fmtInt(tInt + 1)} / ${D.S} 步`, `GLASS MODEL · WHOLE RUN · STEP ${fmtInt(tInt + 1)} / ${D.S}`), 0, 16, { size: 10, kind: 'mono', color: COL.dim });
    text(g, L('看规律一点点被写进这 2,928 个数里', 'Watch the pattern get written into these 2,928 numbers'), 0, 46, { size: P ? 18 : 24, kind: 'serif', weight: 900, color: COL.ink, max: P ? 380 : 1120 });
    // 开关：数值 / 变化；色标跟着变 / 固定
    const by = P ? 58 : 60;
    const pills = [
      [L('数值 w', 'value w'), this.mode === 'w', () => { this.mode = 'w'; }],
      [L('变化 w − w₀', 'change w − w₀'), this.mode === 'd', () => { this.mode = 'd'; }],
      [L('色标跟着变', 'auto scale'), !this.fixed, () => { this.fixed = false; }],
      [L('色标固定', 'fixed scale'), this.fixed, () => { this.fixed = true; }],
    ];
    let px = 0;
    pills.forEach(([lab, on, act], i) => {
      const w = measure(g, lab, 11) + 22;
      if (i === 2) px += 10;
      pill(g, px, by, w, 24, lab, { on, color: i < 2 ? COL.cyan : COL.violet, size: 11 });
      env.hit(px, by, w, 24, { click: true, act: () => { act(); this.app.sfx('click'); }, tip: i < 2 ? L('看参数本身，还是看它离初始化走了多远', 'Show the parameter itself, or how far it has moved from its initial value') : L('“跟着变”：按此刻的数值范围上色，看得清图案；“固定”：按训练结束时的范围，看得出整体变大', '“Auto”: colors follow the current range so patterns stay visible; “fixed”: use the end-of-training range so you can see values grow') });
      px += w + 6;
    });
    if (!haveW) {
      const b = this.map.bounds();
      waitBox(g, b.x, b.y, b.w, b.h, env, st.wait, { withCard: true, label: L('正在载入这一段的权重…', 'Loading the weights for this stretch…') });
    } else {
      const Wa = D.W(k), Wb = D.has('w', k1) ? D.W(k1) : Wa;
      const w0 = D.w0;
      const cur = (gi) => (f > 0 ? Wa[gi] + (Wb[gi] - Wa[gi]) * f : Wa[gi]);
      const val = (gi) => {
        const v = cur(gi);
        if (this.mode === 'd') return v - (this.isNorm(gi) ? 1 : w0[gi]);
        return this.isNorm(gi) ? v - 1 : v;
      };
      const sc = this.scales(Wa, f > 0 ? Wb : null, f);
      this.map.draw(g, env, {
        value: val,
        scale: (name) => (name.startsWith('g') ? sc.norm : sc.mat),
        tip: (name, gi) => {
          const v = cur(gi), v0 = this.isNorm(gi) ? 1 : w0[gi];
          return `<span class="k">${esc(paramName(D, gi))} · ${L(`第 ${tInt + 1} 步`, `step ${tInt + 1}`)}</span>${L('现在', 'now')} <span class="v">${fnum(v, 4)}</span>　${L('初始', 'initial')} ${fnum(v0, 4)}　${L('变化', 'change')} <span class="a">${v - v0 >= 0 ? '+' : ''}${fnum(v - v0, 3)}</span><br><span style="color:var(--dim)">${L('点一下看这个参数的一生', 'Click to see this parameter’s life')}</span>`;
        },
        pick: (name, gi) => this.app.pickParam(gi),
        sel: this.app.ctx.gsel,
        // 标题右边：这一步流到这个张量的梯度有多长（反向传播在训练中怎么变）
        sub: (name) => `‖∂L/∂${name === 'E' ? 'E' : 'W'}‖ = ${fnum(D.tgn(Math.min(D.S - 1, tInt), D.pIndex.get(name).j), 2)}`,
        subColor: hexA(COL.rose, 0.85),
      });
      if (!P) {
        legend(g, 300, this.map.oy + this.map.H + 6, 150, sc.mat, (v) => fnum(v, 2), { label: L('矩阵', 'matrices') });
        legend(g, 560, this.map.oy + this.map.H + 6, 150, sc.norm, (v) => fnum(v, 2), { label: this.mode === 'd' ? 'γ' : 'γ − 1' });
      }
      this.drawSim(g, st, env, cur);
    }
    this.drawLoss(g, st, env, t);
    this.drawPred(g, st, env, k, k1, f);
    this.drawAttn(g, st, env, k, k1, f);
  }

  // 色标：矩阵和 γ 各一个（γ 画 γ − 1）。跟着变 = 此刻的 98% 分位；固定 = 最后一帧的
  scales(Wa, Wb, f) {
    const D = this.D;
    const W = this.fixed ? (D.has('w', D.NF - 1) ? D.W(D.NF - 1) : Wa) : Wa;
    const mats = [], norms = [];
    for (const p of D.params) for (let i = p.off; i < p.off + p.n; i++) {
      const v = Wb && !this.fixed ? W[i] + (Wb[i] - W[i]) * f : W[i];
      if (p.norm) norms.push(Math.abs(v - 1)); else mats.push(Math.abs(this.mode === 'd' ? v - D.w0[i] : v));
    }
    const q = (a) => { a.sort((x, y) => x - y); return a[Math.floor(0.985 * (a.length - 1))] || 1e-3; };
    return { mat: Math.max(q(mats), 1e-3), norm: Math.max(q(norms), 0.02) };
  }

  drawLoss(g, st, env, t) {
    const C = this.cards.loss, D = this.D, S = D.S;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: L('LOSS · 每一步都记', 'LOSS · EVERY STEP'), title: L('损失：从瞎猜到猜对', 'Loss: from guessing to knowing'), accent: COL.cyan });
    const x0 = C.x + 40, w = C.w - 56, y0 = C.y + 60, h = C.h - 96;
    const lo = Math.log(0.03), hi = Math.log(4);
    const Y = (v) => y0 + h - ((Math.log(clamp(v, 0.03, 4)) - lo) / (hi - lo)) * h, X = (tt) => x0 + (tt / (S - 1)) * w;
    for (const v of [0.1, 0.3, 1, 3]) { const y = Y(v); g.strokeStyle = COL.line; g.beginPath(); g.moveTo(x0, y); g.lineTo(x0 + w, y); g.stroke(); text(g, String(v), x0 - 5, y + 3, { size: 9, kind: 'mono', color: COL.dim, align: 'right' }); }
    // 全零对照：一条平线
    g.setLineDash([3, 3]);
    line(g, [[x0, Y(D.zLoss[0])], [x0 + w, Y(D.zLoss[0])]], hexA(COL.rose, 0.55), 1);
    g.setLineDash([]);
    text(g, L('全零初始化（一直不动）', 'all-zero init (never moves)'), x0 + w, Y(D.zLoss[0]) - 4, { size: 9, color: hexA(COL.rose, 0.8), align: 'right' });
    const tn = Math.floor(t);
    const pb = [], pe = [];
    for (let i = 0; i <= tn && i < S; i++) { pb.push([X(i), Y(D.loss[i])]); pe.push([X(i), Y(D.evalLoss[i])]); }
    line(g, pb, hexA(COL.cyan, 0.3), 0.9);
    line(g, pe, COL.cyan, 1.7);
    // 帧的位置（逐数记录的步）
    for (const fr of D.FR) { g.fillStyle = fr <= t ? hexA(COL.violet, 0.7) : hexA(COL.violet, 0.22); g.fillRect(X(fr) - 0.5, y0 + h + 3, 1, 4); }
    const xc = X(t);
    g.strokeStyle = hexA(COL.amber, 0.6); g.beginPath(); g.moveTo(xc, y0); g.lineTo(xc, y0 + h); g.stroke();
    dot(g, xc, Y(D.evalLoss[Math.min(S - 1, tn)]), 3.5, COL.amber);
    text(g, D.evalLoss[Math.min(S - 1, tn)].toFixed(3), xc + 6, Y(D.evalLoss[Math.min(S - 1, tn)]) - 6, { size: 10.5, kind: 'mono', color: COL.amber });
    text(g, L('亮线：全部 25 段的平均　暗线：这一步的 8 段　紫色刻度：逐数记录的步', 'bright: mean over all 25 windows · faint: this step’s 8 windows · ticks: fully recorded steps'), C.x + 14, C.y + C.h - 12, { size: 9.5, color: COL.dim, max: C.w - 28 });
    const app = this.app;
    const toK = (wx) => {
      const tt = clamp(((wx - x0) / w) * (S - 1), 0, S - 1);
      let best = 0;
      D.FR.forEach((fr, i) => { if (Math.abs(fr - tt) < Math.abs(D.FR[best] - tt)) best = i; });
      return best;
    };
    env.hit(x0 - 4, y0 - 6, w + 8, h + 14, {
      drag: (wx, wy, phase) => { if (phase !== 'end') app.scrub(toK(wx)); },
      tipAt: (wx) => { const tt = clamp(Math.round(((wx - x0) / w) * (S - 1)), 0, S - 1); return `<span class="k">${L(`第 ${tt + 1} 步`, `step ${tt + 1}`)}</span>${L('8 段的损失', 'batch loss')} <span class="v">${D.loss[tt].toFixed(3)}</span>　${L('全部 25 段', 'all 25')} <span class="v">${D.evalLoss[tt].toFixed(3)}</span><br>${L('学习率', 'learning rate')} <span class="a">${sciSup(D.lr[tt], 3)}</span>　‖g‖ ${D.gnorm[tt].toFixed(3)}<br><span style="color:var(--dim)">${L('按住拖动来回看', 'Drag to scrub')}</span>`; },
    });
  }

  // 固定样例“举头望明月，低头”：每个位置猜下一个字
  drawPred(g, st, env, k, k1, f) {
    const C = this.cards.pred, D = this.D, T = D.T;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: L('FIXED SAMPLE · 每一步批次里的第 0 段', 'FIXED SAMPLE · ROW 0 OF EVERY BATCH'), title: L('“举头望明月，低头”：下一个字', '“举头望明月，低头”: the next character'), accent: COL.amber });
    if (!D.has('w', k)) { waitBox(g, C.x + 10, C.y + 50, C.w - 20, C.h - 60, env, st.wait); return; }
    const x = D.fixed;
    const y0 = C.y + 60, rh = (C.h - 92) / T;
    const pc = (kk, i) => D.probs(kk, i, x[i + 1]);
    const has1 = D.has('w', k1);
    let tot = 0;
    for (let i = 0; i < T; i++) {
      const p = f > 0 && has1 ? lerp(pc(k, i), pc(k1, i), f) : pc(k, i);
      tot += -Math.log(Math.max(p, 1e-9));
      // 第一名
      let best = 0;
      for (let v = 1; v < D.V; v++) if (D.probs(k, i, v) > D.probs(k, i, best)) best = v;
      const y = y0 + i * rh;
      text(g, disp(D.ch(x[i])), C.x + 22, y + rh * 0.7, { size: 13, color: COL.ink, align: 'center' });
      text(g, '→', C.x + 40, y + rh * 0.68, { size: 10, color: COL.dim, align: 'center' });
      text(g, disp(D.ch(x[i + 1])), C.x + 58, y + rh * 0.7, { size: 13, color: COL.amber, align: 'center', weight: 600 });
      const bx = C.x + 76, bw = C.w - 76 - 92;
      rr(g, bx, y + rh * 0.28, bw, rh * 0.44, 3); g.fillStyle = 'rgba(255,255,255,0.04)'; g.fill();
      rr(g, bx, y + rh * 0.28, Math.max(2, bw * p), rh * 0.44, 3); g.fillStyle = p > 0.5 ? hexA(COL.cyan, 0.75) : p > 0.15 ? hexA(COL.amber, 0.7) : hexA(COL.rose, 0.6); g.fill();
      text(g, fmtP(p), bx + bw + 8, y + rh * 0.68, { size: 10.5, kind: 'mono', color: COL.ink });
      const ok = best === x[i + 1];
      text(g, ok ? '✓' : disp(D.ch(best)), C.x + C.w - 14, y + rh * 0.7, { size: 12, color: ok ? COL.green : COL.rose, align: 'right' });
      env.hit(C.x + 10, y, C.w - 20, rh, { tip: () => {
        const top = Array.from({ length: D.V }, (_, v) => [v, D.probs(k, v === -1 ? 0 : i, v)]).sort((a, b) => b[1] - a[1]).slice(0, 4);
        return `<span class="k">${L(`位置 ${i}：看到“${esc(D.fixed.slice(0, i + 1).map(D.ch).join(''))}”`, `position ${i}: has seen “${esc(D.fixed.slice(0, i + 1).map(D.ch).join(''))}”`)}</span>${L('正确答案', 'answer')} <b>${esc(D.ch(x[i + 1]))}</b> ${L('的概率', 'probability')} <span class="v">${fmtP(pc(k, i))}</span>，−ln p = ${(-Math.log(Math.max(pc(k, i), 1e-9))).toFixed(3)}<br>${L('前 4 名', 'top 4')}：${top.map(([v, pp]) => `${esc(D.ch(v))} ${fmtP(pp)}`).join('　')}`;
      } });
    }
    text(g, L(`平均 −ln p = ${(tot / T).toFixed(3)}　右边：第一名（✓ = 猜对）`, `mean −ln p = ${(tot / T).toFixed(3)} · right: top guess (✓ = right)`), C.x + 14, C.y + C.h - 12, { size: 9.5, color: COL.dim, max: C.w - 28 });
  }

  drawAttn(g, st, env, k, k1, f) {
    const C = this.cards.attn, D = this.D, T = D.T;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: L('ATTENTION · 固定样例', 'ATTENTION · FIXED SAMPLE'), title: L('两个注意力头：每个字往回看谁', 'Two heads: who each position looks back at'), accent: COL.violet });
    if (!D.has('w', k)) { waitBox(g, C.x + 10, C.y + 50, C.w - 20, C.h - 60, env, st.wait); return; }
    const cell = Math.min(13, (C.w - 90) / (2 * T + 2));
    const has1 = D.has('w', k1);
    for (let h = 0; h < D.H; h++) {
      const x0 = C.x + 30 + h * (T * cell + 46), y0 = C.y + 58;
      heat(g, x0, y0, T, T, cell, cell, (i, j) => (f > 0 && has1 ? lerp(D.att(k, h, i, j), D.att(k1, h, i, j), f) : D.att(k, h, i, j)), { seq: true, scale: 1, s: env.s });
      text(g, L(`头 ${h}`, `head ${h}`), x0, y0 - 6, { size: 10, kind: 'mono', color: COL.ink2 });
      for (let i = 0; i < T; i++) {
        text(g, D.ch(D.fixed[i]), x0 - 4, y0 + i * cell + cell * 0.8, { size: cell * 0.8, color: COL.dim, align: 'right' });
        text(g, D.ch(D.fixed[i]), x0 + i * cell + cell / 2, y0 + T * cell + cell * 0.95, { size: cell * 0.8, color: COL.dim, align: 'center' });
      }
      env.hit(x0, y0, T * cell, T * cell, { tipAt: (wx, wy) => { const c = cellAt(wx, wy, x0, y0, T, T, cell, cell); if (!c) return null; const a = D.att(k, h, c.r, c.c); return `<span class="k">${L(`头 ${h}`, `head ${h}`)}</span>${L(`位置 ${c.r}「${esc(D.ch(D.fixed[c.r]))}」看位置 ${c.c}「${esc(D.ch(D.fixed[c.c]))}」`, `position ${c.r} “${esc(D.ch(D.fixed[c.r]))}” → position ${c.c} “${esc(D.ch(D.fixed[c.c]))}”`)}：<span class="v">${fmtP(a)}</span>${c.c > c.r ? L('<br>因果遮罩：看不到后面的字', '<br>causal mask: cannot see later characters') : ''}`; } });
    }
    text(g, L('每一行加起来是 1；右上角是因果遮罩（不能偷看后面）', 'each row sums to 1; upper right is the causal mask (no peeking ahead)'), C.x + 14, C.y + C.h - 12, { size: 9.5, color: COL.dim, max: C.w - 28 });
  }

  // 嵌入表 E 的行两两之间的余弦相似度（按诗里第一次出现的顺序排）
  drawSim(g, st, env, cur) {
    const S = this.sim, D = this.D, V = D.V, d = D.d, E = D.pIndex.get('E').off;
    const P = this.portrait;
    if (!P) {
      text(g, L('E 的行两两相似度', 'Similarity between rows of E'), S.x, S.y - 18, { size: 11.5, kind: 'serif', weight: 600, color: COL.ink });
      text(g, L('余弦，按诗里出现的顺序', 'cosine, in poem order'), S.x, S.y - 4, { size: 9.5, color: COL.dim });
    } else {
      card(g, S.x, S.y, S.w, S.h, { eyebrow: L('EMBEDDINGS · 余弦相似度', 'EMBEDDINGS · COSINE SIMILARITY'), title: L('嵌入表里哪些字越来越像', 'Which characters’ embeddings grow alike'), accent: COL.cyan });
    }
    const rows = [];
    for (let a = 0; a < V; a++) { const r = new Float32Array(d); let n = 0; for (let j = 0; j < d; j++) { r[j] = cur(E + a * d + j); n += r[j] * r[j]; } n = Math.sqrt(n) || 1; for (let j = 0; j < d; j++) r[j] /= n; rows.push(r); }
    const sim = (a, b) => { let s = 0; for (let j = 0; j < d; j++) s += rows[a][j] * rows[b][j]; return s; };
    const cell = P ? 15 : 7.4, x0 = S.x + (P ? 28 : 16), y0 = S.y + (P ? 62 : 12);
    heat(g, x0, y0, V, V, cell, cell, (a, b) => (a === b ? 0 : sim(a, b)), { scale: 0.7, s: env.s });
    for (let a = 0; a < V; a++) {
      text(g, D.ch(a), x0 - 3, y0 + a * cell + cell * 0.82, { size: Math.min(10, cell * 0.85), color: COL.dim, align: 'right' });
      text(g, D.ch(a), x0 + a * cell + cell / 2, y0 + V * cell + cell * 0.95, { size: Math.min(10, cell * 0.85), color: COL.dim, align: 'center' });
    }
    env.hit(x0, y0, V * cell, V * cell, { tipAt: (wx, wy) => { const c = cellAt(wx, wy, x0, y0, V, V, cell, cell); if (!c || c.r === c.c) return null; return `<span class="k">${L('嵌入的余弦相似度', 'embedding cosine similarity')}</span>「${esc(D.ch(c.r))}」·「${esc(D.ch(c.c))}」 = <span class="v">${sim(c.r, c.c).toFixed(2)}</span>`; } });
    if (P) wrap(g, L('对角线两旁亮起来：诗里前后相邻的字越来越像——输入、输出共用 E，“月”的向量被拉向“光”', 'Next to the diagonal lights up: neighbours in the poem grow alike — input and output share E, so 月 (moon) is pulled toward 光 (light)'), S.x + 14, S.y + S.h - 30, S.w - 28, 13, { size: 9.5, color: COL.dim });
  }
}
