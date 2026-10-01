// D3 · 损失：逐个位置看 −ln p；D4 · 一个位置的交叉熵：softmax → 取正确答案的概率 → 取负对数。
import { COL, text, rr, card, hexA, fmtP, clamp, ease, seg, seqColor, line, dot, wrap, sciSup, fmtInt, lerp } from '../draw.js';
import { rowLayout, tileBase, disp, short } from './row.js';
import { esc } from '../../../js/ui.js';

export class LossView {
  constructor(app, R) { this.app = app; this.R = R; }

  layout(env) {
    this.portrait = env.portrait;
    this.W = this.portrait ? 380 : 1100;
  }

  focus(st) {
    const P = this.portrait;
    if (st.step.mi) return P ? { x: -6, y: this.detailY - 10, w: 392, h: 1040 } : { x: -12, y: this.detailY - 16, w: 1124, h: 400 };
    return P ? { x: -6, y: -6, w: 392, h: (this.detailY || 500) + 420 } : { x: -12, y: -12, w: 1124, h: (this.detailY || 400) + 400 };
  }

  draw(g, st, env) {
    const R = this.R, k = st.k, s = st.step, T = R.kind === 'tiny', P = this.portrait, W = this.W;
    const ci = s.i;
    text(g, 'CROSS-ENTROPY · 每个位置 −ln p(正确答案)', 0, 18, { size: 10, kind: 'mono', color: COL.dim });
    text(g, '损失：把每个位置猜错的程度加起来', 0, 50, { size: P ? 18 : 22, kind: 'serif', weight: 900, color: COL.ink });
    // 字块：第 j 块是位置 j−1 的目标
    const rows = rowLayout(g, R, k, 0, 78, W, T ? { tile: P ? 30 : 36, th: 40, gap: 4, lineGap: 26 } : { tile: 34, th: 40, gap: 5, lineGap: 28 });
    let sum = 0, cnt = 0;
    for (const r of rows) {
      const pi = r.j - 1;
      const has = pi >= 0 && pi < R.rowLen;
      const counted = has && R.counted(pi);
      const done = has && counted && pi <= ci;
      const p = has ? R.p(k, pi) : 0;
      const nll = -Math.log(Math.max(p, 1e-12));
      if (done) { sum += nll; cnt++; }
      const cur = pi === ci;
      tileBase(g, r, { fill: done ? seqColor(Math.sqrt(p), 0.4) : 'rgba(255,255,255,0.03)', stroke: cur ? COL.amber : COL.line2, dim: has && !counted });
      if (cur) { g.lineWidth = 2; g.strokeStyle = COL.amber; rr(g, r.x - 2, r.y - 2, r.w + 4, r.h + 4, 9); g.stroke(); }
      if (done) {
        const bw = r.w - 6;
        const grow = cur ? ease(seg(st.p, 0, 0.5)) : 1;
        rr(g, r.x + 3, r.y + r.h + 4, Math.max(1.5, bw * Math.min(1, nll / 10) * grow), 5, 2);
        g.fillStyle = nll > 6 ? COL.rose : nll > 2 ? COL.amber : COL.cyan;
        g.fill();
        if (!T || cur) text(g, nll.toFixed(2), r.x + r.w / 2, r.y + r.h + 21, { size: 9, kind: 'mono', color: cur ? COL.amber : COL.dim, align: 'center' });
      }
      if (has && counted) env.hit(r.x, r.y, r.w, r.h + 20, {
        click: true,
        act: () => { this.app.seek((b) => b.ph === 'loss' && b.i === pi); },
        tip: `<span class="k">位置 ${pi} → 「${esc(disp(R.tok(k, r.j)))}」</span>p = <span class="v">${fmtP(p)}</span>　−ln p = ${nll.toFixed(3)}<br><span style="color:var(--dim)">点一下跳到这个位置</span>`,
      });
    }
    const ry = 78 + rows.height + 36;
    // 累计
    text(g, `已经累计 ${cnt} 个位置，平均 −ln p = ${cnt ? (sum / cnt).toFixed(3) : '—'}`, 0, ry, { size: 12, kind: 'mono', color: COL.ink2 });
    text(g, T ? `整批 ${fmtInt(R.D.meta.train.tokensPerStep)} 个位置的平均（真正的 loss）= ${R.loss(k).toFixed(4)}` : `${R.D.sftPos.length} 个回答位置的平均（真正的 loss）= ${R.loss(k).toFixed(4)}`, P ? 0 : W, P ? ry + 20 : ry, { size: 12, kind: 'mono', color: COL.amber, align: P ? 'left' : 'right' });
    this.detailY = ry + (P ? 44 : 28);
    this.drawDetail(g, st, env, this.detailY);
  }

  drawDetail(g, st, env, y0) {
    const R = this.R, k = st.k, s = st.step, T = R.kind === 'tiny', P = this.portrait, W = this.W;
    const i = s.i, p = R.p(k, i), tgt = R.tok(k, i + 1);
    const nll = -Math.log(Math.max(p, 1e-12));
    const top = R.top(k, i);
    const tid = R.tokId(k, i + 1);
    const inTop = top.findIndex((t) => t.id === tid);
    const mi = s.mi;
    const phase = mi ? ['softmax', 'pick', 'log'].indexOf(mi) : 3;
    const cw = P ? 380 : (W - 28) / 3, chh = P ? 320 : 340;
    const boxes = [0, 1, 2].map((j) => (P ? { x: 0, y: y0 + j * (chh + 14), w: 380, h: chh } : { x: j * (cw + 14), y: y0, w: cw, h: chh }));
    // ① softmax
    {
      const b = boxes[0];
      card(g, b.x, b.y, b.w, b.h, { eyebrow: '① SOFTMAX · 分数 → 概率', title: `位置 ${i} 的预测`, accent: COL.cyan, active: phase === 0 });
      const ctxN = T ? 8 : 4;
      const ctx = Array.from({ length: Math.min(i + 1, ctxN) }, (_, j) => disp(R.tok(k, i - Math.min(i, ctxN - 1) + j))).join(T ? '' : '·');
      text(g, `上文「…${ctx}」→ ？`, b.x + 14, b.y + 66, { size: 12, color: COL.ink2, max: b.w - 28 });
      const items = top.map((t) => ({ s: disp(t.s), p: t.p, tg: t.id === tid }));
      const rest = 1 - items.reduce((a, t) => a + t.p, 0);
      if (inTop < 0) items.push({ s: disp(tgt), p, tg: true, out: true });
      items.push({ s: `其余 ${fmtInt(R.model.vocab - items.length)} 个`, p: Math.max(0, rest - (inTop < 0 ? p : 0)), rest: true });
      const grow = phase === 0 ? ease(seg(st.p, 0, 0.7)) : 1;
      const bx = b.x + 96, bw = b.w - 160;
      items.forEach((t, j) => {
        const yy = b.y + 92 + j * 30;
        const hl = phase >= 1 && t.tg;
        const dimmed = phase >= 1 && !t.tg;
        g.globalAlpha = dimmed ? 0.35 : 1;
        text(g, t.s, b.x + 86, yy + 12, { size: 12.5, color: t.tg ? COL.amber : t.rest ? COL.dim : COL.ink, align: 'right', max: 80 });
        rr(g, bx, yy, Math.max(1.5, bw * t.p * grow), 16, 3);
        g.fillStyle = t.tg ? COL.amber : t.rest ? COL.faint : COL.cyan;
        g.fill();
        text(g, fmtP(t.p), bx + Math.max(2, bw * t.p * grow) + 6, yy + 12, { size: 10.5, kind: 'mono', color: hl ? COL.amber : COL.dim });
        g.globalAlpha = 1;
      });
      text(g, `${fmtInt(R.model.vocab)} 个分数 → e^z / Σe^z → 加起来等于 1`, b.x + 14, b.y + b.h - 14, { size: 10, color: COL.faint, max: b.w - 28 });
    }
    // ② 取正确答案的概率
    {
      const b = boxes[1];
      card(g, b.x, b.y, b.w, b.h, { eyebrow: '② PICK · 只看正确答案', title: `正确答案「${disp(tgt)}」`, accent: COL.amber, active: phase === 1 });
      const a = phase >= 1 ? 1 : 0.35;
      g.globalAlpha = a;
      text(g, short(tgt), b.x + b.w / 2, b.y + 150, { size: 54, color: COL.amber, align: 'center', weight: 600, max: b.w - 30 });
      text(g, `p = ${p < 1e-3 ? sciSup(p, 3) : p.toFixed(4)}`, b.x + b.w / 2, b.y + 200, { size: 20, kind: 'mono', color: COL.ink, align: 'center' });
      text(g, inTop >= 0 ? `它是第 ${inTop + 1} 名` : '它不在前 5 名里', b.x + b.w / 2, b.y + 226, { size: 12, color: COL.dim, align: 'center' });
      g.globalAlpha = 1;
      wrap(g, '其余词元不直接出现在损失里，但都在 softmax 的分母里：梯度会把它们统统压低一点，把正确答案抬高。', b.x + 14, b.y + b.h - 58, b.w - 28, 17, { size: 10.5, color: COL.dim });
    }
    // ③ −ln p
    {
      const b = boxes[2];
      card(g, b.x, b.y, b.w, b.h, { eyebrow: '③ −LN P · 越没把握，罚得越重', title: `损失 = ${nll.toFixed(3)}`, accent: COL.rose, active: phase === 2 });
      const px = b.x + 40, pw = b.w - 64, py = b.y + 70, ph = b.h - 120;
      const maxL = 10;
      const X = (pp) => px + pp * pw, Y = (v) => py + ph - (Math.min(v, maxL) / maxL) * ph;
      g.strokeStyle = COL.line;
      g.beginPath(); g.moveTo(px, py); g.lineTo(px, py + ph); g.lineTo(px + pw, py + ph); g.stroke();
      const pts = [];
      for (let t = 0; t <= 200; t++) { const pp = Math.max(1e-5, t / 200); pts.push([X(pp), Y(-Math.log(pp))]); }
      line(g, pts, hexA(COL.rose, 0.8), 1.6);
      text(g, '0', px - 6, py + ph + 4, { size: 9, kind: 'mono', color: COL.faint, align: 'right' });
      text(g, '1', px + pw, py + ph + 14, { size: 9, kind: 'mono', color: COL.faint, align: 'center' });
      text(g, 'p', px + pw / 2, py + ph + 16, { size: 10, kind: 'mono', color: COL.dim, align: 'center' });
      text(g, `${maxL}`, px - 6, py + 4, { size: 9, kind: 'mono', color: COL.faint, align: 'right' });
      // 均匀瞎猜
      const lv = Math.log(R.model.vocab);
      g.setLineDash([2, 3]);
      g.strokeStyle = 'rgba(255,107,147,0.3)';
      g.beginPath(); g.moveTo(px, Y(lv)); g.lineTo(px + pw, Y(lv)); g.stroke();
      g.setLineDash([]);
      text(g, `瞎猜 ${lv.toFixed(2)}`, px + pw, Y(lv) - 4, { size: 9, color: 'rgba(255,107,147,0.7)', align: 'right' });
      const f = phase === 2 ? ease(seg(st.p, 0, 0.6)) : phase > 2 ? 1 : 0;
      if (f > 0) {
        const pp = lerp(1, Math.max(p, 1e-5), f);
        const mx = X(pp), my = Y(-Math.log(pp));
        g.setLineDash([2, 3]);
        g.strokeStyle = hexA(COL.amber, 0.6);
        g.beginPath(); g.moveTo(mx, py + ph); g.lineTo(mx, my); g.lineTo(px, my); g.stroke();
        g.setLineDash([]);
        dot(g, mx, my, 5, COL.amber, '#fff');
        text(g, (-Math.log(pp)).toFixed(3), mx + 8, my - 6, { size: 12, kind: 'mono', color: COL.amber, weight: 700 });
      }
      text(g, `−ln ${p < 1e-3 ? sciSup(p, 2) : p.toFixed(3)} = ${nll.toFixed(3)}`, b.x + 14, b.y + b.h - 14, { size: 11, kind: 'mono', color: COL.ink2 });
    }
  }
}
