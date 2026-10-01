// D2 · 一步之内：批次 → 前向 → 损失 → 反向 → 更新 的训练循环，右边是这一步真实的批次第 0 行。
import { COL, text, rr, card, dot, hexA, wrap, fmtInt, fmtP, clamp, lerp, ease, sciSup, seqColor, heatColor, arrow, badge } from '../draw.js';
import { rowLayout, tileBase, disp } from './row.js';
import { esc } from '../../../js/ui.js';

const NODES = [
  { ph: 'batch', name: '批次', en: 'BATCH', a: -90, c: COL.blue },
  { ph: 'fwd', name: '前向', en: 'FORWARD', a: -18, c: COL.cyan },
  { ph: 'loss', name: '损失', en: 'LOSS', a: 54, c: COL.amber },
  { ph: 'bwd', name: '反向', en: 'BACKWARD', a: 126, c: COL.rose },
  { ph: 'upd', name: '更新', en: 'UPDATE', a: 198, c: COL.violet },
];

export class Loop {
  constructor(app, R) { this.app = app; this.R = R; }

  layout(env) {
    this.portrait = env.portrait;
    if (!this.portrait) {
      this.ring = { cx: 290, cy: 320, rx: 230, ry: 220, r: 52 };
      this.panel = { x: 600, y: 40, w: 600, h: 560 };
      this.bounds = { x: 0, y: 20, w: 1210, h: 600 };
    } else {
      this.ring = { cx: 190, cy: 150, rx: 128, ry: 112, r: 34 };
      this.panel = { x: 0, y: 290, w: 380, h: 560 };
      this.bounds = { x: -6, y: 14, w: 392, h: 640 };
    }
  }

  focus() { return this.bounds; }

  pos(i) {
    const n = NODES[i], R = this.ring;
    const a = (n.a * Math.PI) / 180;
    return [R.cx + Math.cos(a) * R.rx, R.cy + Math.sin(a) * R.ry];
  }

  draw(g, st, env) {
    const R = this.R, k = st.k, s = st.step;
    const cur = NODES.findIndex((n) => n.ph === s.ph);
    const RG = this.ring;
    // 环
    g.strokeStyle = COL.line2;
    g.lineWidth = 1.5;
    g.beginPath();
    g.ellipse(RG.cx, RG.cy, RG.rx, RG.ry, 0, 0, Math.PI * 2);
    g.stroke();
    // 方向箭头（顺时针）
    for (let i = 0; i < NODES.length; i++) {
      const a0 = NODES[i].a, a1 = NODES[(i + 1) % NODES.length].a + (i === NODES.length - 1 ? 360 : 0);
      const am = (((a0 + a1) / 2) * Math.PI) / 180, ad = am + 0.04;
      const [x1, y1] = [RG.cx + Math.cos(am) * RG.rx, RG.cy + Math.sin(am) * RG.ry];
      const [x2, y2] = [RG.cx + Math.cos(ad) * RG.rx, RG.cy + Math.sin(ad) * RG.ry];
      arrow(g, x1, y1, x2, y2, i === cur ? COL.amber : COL.line3, 1.5, 8);
    }
    // 彗星：当前环节做完后沿环跑向下一个
    if (cur >= 0) {
      const a0 = NODES[cur].a, a1 = NODES[(cur + 1) % NODES.length].a + (cur === NODES.length - 1 ? 360 : 0);
      const f = ease(clamp((st.p - 0.55) / 0.45, 0, 1));
      if (f > 0 && f < 1) {
        for (let j = 0; j < 14; j++) {
          const ff = Math.max(0, f - j * 0.018);
          const a = ((a0 + (a1 - a0) * ff) * Math.PI) / 180;
          dot(g, RG.cx + Math.cos(a) * RG.rx, RG.cy + Math.sin(a) * RG.ry, 4 - j * 0.25, hexA(COL.amber, 0.9 - j * 0.06));
        }
      }
    }
    // 中心
    const T = R.kind === 'tiny';
    text(g, T ? 'STEP' : 'SFT STEP', RG.cx, RG.cy - 34, { size: 10, kind: 'mono', color: COL.dim, align: 'center' });
    text(g, `第 ${fmtInt(R.stepNo(k))} 步`, RG.cx, RG.cy - 4, { size: this.portrait ? 22 : 28, kind: 'serif', weight: 900, color: COL.ink, align: 'center' });
    text(g, `loss ${R.loss(k).toFixed(4)}`, RG.cx, RG.cy + 22, { size: 12, kind: 'mono', color: COL.cyan, align: 'center' });
    text(g, `lr ${sciSup(R.lr(k), 3)}`, RG.cx, RG.cy + 40, { size: 11, kind: 'mono', color: COL.amber, align: 'center' });
    // 节点
    const info = {
      batch: R.batchShape,
      fwd: `${R.model.layers} 层 · 并行`,
      loss: R.loss(k).toFixed(3),
      bwd: `‖g‖ ${R.gradNorm(k).toFixed(T ? 3 : 1)}`,
      upd: `${fmtInt(R.model.params / 1e4)} 万个权重`,
    };
    NODES.forEach((n, i) => {
      const [x, y] = this.pos(i);
      const on = i === cur;
      const r = RG.r * (on ? 1.08 + 0.03 * Math.sin(env.t * 5) * (st.playing ? 1 : 0) : 1);
      g.beginPath();
      g.arc(x, y, r, 0, Math.PI * 2);
      g.fillStyle = on ? hexA(n.c, 0.16) : 'rgba(10,17,31,0.92)';
      g.fill();
      g.lineWidth = on ? 2 : 1.2;
      g.strokeStyle = on ? n.c : hexA(n.c, 0.45);
      g.stroke();
      text(g, n.en, x, y - 12, { size: 8.5, kind: 'mono', color: on ? n.c : COL.dim, align: 'center' });
      text(g, n.name, x, y + 8, { size: this.portrait ? 15 : 17, kind: 'serif', weight: 600, color: on ? COL.ink : COL.ink2, align: 'center' });
      text(g, info[n.ph], x, y + 26, { size: 9.5, kind: 'mono', color: COL.dim, align: 'center', max: r * 2 - 8 });
      env.hit(x - r, y - r, r * 2, r * 2, { click: true, act: () => { this.app.seek((b) => b.ph === n.ph); this.app.sfx('click'); }, tip: `<span class="k">${n.en}</span>${n.name}：点一下跳到这里，再按 ＋ 拆开` });
    });
    // 演示用的“重新前向”
    {
      const [ux, uy] = this.pos(4), [bx, by] = this.pos(0);
      const cx = (ux + bx) / 2 - (this.portrait ? 28 : 54), cy = (uy + by) / 2 - (this.portrait ? 30 : 40);
      const on = s.ph === 'check';
      g.setLineDash([3, 4]);
      g.strokeStyle = on ? COL.amber : COL.line3;
      g.beginPath(); g.moveTo(ux - 10, uy - 30); g.lineTo(cx, cy + 22); g.stroke();
      g.setLineDash([]);
      rr(g, cx - 46, cy - 18, 92, 40, 12);
      g.fillStyle = on ? hexA(COL.amber, 0.14) : 'rgba(10,17,31,0.9)';
      g.fill();
      g.setLineDash([3, 2]);
      g.strokeStyle = on ? COL.amber : hexA(COL.amber, 0.4);
      g.stroke();
      g.setLineDash([]);
      text(g, '重新前向', cx, cy, { size: 12, color: on ? COL.amber : COL.ink2, align: 'center', weight: 600 });
      text(g, '演示 · 不是训练的一部分', cx, cy + 14, { size: 8.5, color: COL.dim, align: 'center' });
      env.hit(cx - 46, cy - 18, 92, 40, { click: true, act: () => { this.app.seek((b) => b.ph === 'check'); this.app.sfx('click'); }, tip: '更新完以后，对同一批数据再算一次损失，看这一步改变了什么（演示用）' });
    }
    this.drawPanel(g, st, env, s.ph);
  }

  drawPanel(g, st, env, ph) {
    const R = this.R, k = st.k, C = this.panel, T = R.kind === 'tiny';
    const titles = { batch: '这一步的数据', fwd: '前向：每个位置同时预测下一个', loss: '损失：猜中正确答案的概率', bwd: '反向：梯度流回到每个输入位置', upd: '更新：每个权重挪一小步', check: '重新前向：同一批数据，更新前后' };
    card(g, C.x, C.y, C.w, C.h, { eyebrow: T ? `BATCH ROW 0 · 第 0 行的前 ${R.rowLen} 个位置` : `CONVERSATION · ${R.rowLen + 1} 个词元`, title: titles[ph] || '', accent: COL.cyan, demo: ph === 'check' });
    const rows = rowLayout(g, R, k, C.x + 16, C.y + 64, C.w - 32, T ? { tile: this.portrait ? 30 : 38, th: this.portrait ? 36 : 42, gap: this.portrait ? 3 : 4, lineGap: this.portrait ? 14 : 18 } : { tile: 34, th: this.portrait ? 36 : 42, gap: 5, lineGap: this.portrait ? 18 : 22 });
    // 反向用：嵌入处每个位置的梯度（对数刻度）
    let gmax = 1e-30;
    if (ph === 'bwd') for (let i = 0; i < R.rowLen; i++) gmax = Math.max(gmax, R.residGrad(k, 0, i));
    for (const r of rows) {
      const j = r.j;
      const pi = j - 1; // 以这个字为目标的位置
      const has = pi >= 0 && pi < R.rowLen;
      const counted = has && R.counted(pi);
      const p = has ? R.p(k, pi) : 0;
      let fill = 'rgba(255,255,255,0.03)', stroke = COL.line2, dim = false;
      if (ph === 'fwd' && has) { const lt = R.lensTop(k, R.NL, pi); stroke = lt.ok ? hexA(COL.green, 0.8) : COL.line2; }
      if (ph === 'loss' || ph === 'check') { dim = has && !counted; if (counted) fill = seqColor(Math.sqrt(p), 0.35); }
      if (ph === 'bwd' && j < R.rowLen) { const v = Math.log10(R.residGrad(k, 0, j) / gmax + 1e-12); fill = heatColor(clamp(1 + v / 4, 0, 1), 0.6); }
      if (ph === 'upd') dim = true;
      if (!T && ph === 'batch') dim = false;
      tileBase(g, r, { fill, stroke, dim });
      if (ph === 'fwd' && has) {
        const lt = R.lensTop(k, R.NL, pi);
        text(g, disp(lt.s).slice(0, 4), r.x + r.w - 3, r.y + 10, { size: 8.5, color: lt.ok ? COL.green : COL.dim, align: 'right' });
      }
      if ((ph === 'loss' || ph === 'check') && counted) {
        const nll = -Math.log(Math.max(p, 1e-9));
        const bw = r.w - 6;
        if (ph === 'loss') {
          rr(g, r.x + 3, r.y + r.h + 3, Math.max(1.5, bw * Math.min(1, nll / 10)), 4, 2);
          g.fillStyle = nll > 6 ? COL.rose : nll > 2 ? COL.amber : COL.cyan;
          g.fill();
        } else {
          const pa = R.pAfter(k, pi);
          rr(g, r.x + 3, r.y + r.h + 3, Math.max(1, bw * p), 3, 1.5); g.fillStyle = COL.faint; g.fill();
          rr(g, r.x + 3, r.y + r.h + 8, Math.max(1, bw * pa), 3, 1.5); g.fillStyle = pa >= p ? COL.cyan : COL.rose; g.fill();
        }
      }
      env.hit(r.x, r.y, r.w, r.h, { tip: () => {
        let h = `<span class="k">第 ${j} 个${T ? '字' : '词元'}${T ? '' : ` · ${{ user: '用户', answer: '回答', tpl: '模板' }[r.role]}`}</span><b>${esc(disp(R.tok(k, j)))}</b>　编号 <span class="v">${R.tokId(k, j)}</span>`;
        if (has) h += `<br>从前面猜中它的概率 <span class="v">${fmtP(p)}</span>${counted ? `，损失 ${(-Math.log(Math.max(p, 1e-12))).toFixed(3)}` : '（不计入损失）'}`;
        if (has && ph === 'check') h += `<br>更新后 <span class="a">${fmtP(R.pAfter(k, pi))}</span>`;
        if (ph === 'bwd' && j < R.rowLen) h += `<br>嵌入处的梯度 ‖∂L/∂h‖ = ${sciSup(R.residGrad(k, 0, j), 3)}`;
        return h;
      } });
    }
    const y = C.y + 64 + rows.height + 34;
    this.drawFacts(g, st, ph, C.x + 16, y, C.w - 32, C.y + C.h - y - 14);
  }

  drawFacts(g, st, ph, x, y, w, h) {
    const R = this.R, k = st.k, T = R.kind === 'tiny', m = R.D.meta;
    const big = (v, lab, col, xx, yy) => { text(g, v, xx, yy, { size: this.portrait ? 20 : 24, kind: 'mono', weight: 700, color: col }); text(g, lab, xx, yy + 18, { size: 10.5, color: COL.dim }); };
    const cw = w / 3;
    if (ph === 'batch') {
      big(T ? `${m.train.batch}` : '1', T ? '行（每行一段连续的诗）' : '条对话', COL.blue, x, y + 20);
      big(T ? `${m.train.seq}` : `${R.rowLen}`, '个位置 / 行', COL.blue, x + cw, y + 20);
      big(T ? fmtInt(m.train.tokensPerStep) : `${R.D.sftPos.length}`, T ? '道“猜下一个字”的题' : '个位置计入损失', COL.cyan, x + 2 * cw, y + 20);
    } else if (ph === 'fwd') {
      let ok = 0, n = 0;
      for (let i = 0; i < R.rowLen; i++) if (R.counted(i)) { n++; if (R.lensTop(k, R.NL, i).ok) ok++; }
      big(`${R.model.layers}`, '层，所有位置一起算', COL.cyan, x, y + 20);
      big(fmtP(ok / n), '第一名就猜对的位置', COL.green, x + cw, y + 20);
      text(g, '格子右上角的小字是模型最想说的下一个字（绿色 = 猜对）', x, y + 64, { size: 10.5, color: COL.dim, max: w });
    } else if (ph === 'loss') {
      big(R.loss(k).toFixed(4), T ? `整批 ${fmtInt(m.train.tokensPerStep)} 个位置的平均` : '回答 8 个位置的平均', COL.amber, x, y + 20);
      big(Math.log(R.model.vocab).toFixed(2), '完全瞎猜时的损失', COL.faint, x + cw * 1.4, y + 20);
      text(g, '格子底色 = 猜中它的概率；下面的短条 = 损失 −ln p', x, y + 64, { size: 10.5, color: COL.dim, max: w });
    } else if (ph === 'bwd') {
      big(R.gradNorm(k).toFixed(T ? 3 : 1), '全部梯度合起来的长度', COL.rose, x, y + 20);
      big(R.clip(k) < 1 ? `×${R.clip(k).toFixed(4)}` : '不裁剪', '裁剪到 1.0', COL.violet, x + cw * 1.2, y + 20);
      text(g, '格子底色 = 梯度流回到这个位置的嵌入时有多大（对数刻度）', x, y + 64, { size: 10.5, color: COL.dim, max: w });
    } else if (ph === 'upd') {
      text(g, '跟踪的几个权重（真实数值）', x, y, { size: 11, color: COL.ink2 });
      R.feats.forEach((f, i) => {
        const a = R.adam(k, i);
        const yy = y + 22 + i * 20;
        if (yy > y + h) return;
        text(g, f.label, x, yy, { size: 10.5, color: COL.dim, max: w * 0.5 });
        text(g, `${a.w0.toPrecision(6)} → ${a.w1.toPrecision(6)}`, x + w, yy, { size: 10.5, kind: 'mono', color: COL.ink, align: 'right' });
      });
    } else if (ph === 'check') {
      const a = R.loss(k), b = R.lossAfter(k);
      big(`${a.toFixed(4)} → ${b.toFixed(4)}`, T ? '同一批数据，更新前 → 更新后' : '回答的损失，更新前 → 更新后', b <= a ? COL.cyan : COL.rose, x, y + 20);
      text(g, '格子下方：灰 = 更新前的概率，亮 = 更新后', x, y + 64, { size: 10.5, color: COL.dim, max: w });
    }
  }
}
