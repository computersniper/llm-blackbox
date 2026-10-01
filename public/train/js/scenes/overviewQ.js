// D1 · SFT 三步（Qwen3-0.6B）：同一条对话连走 3 步，回答里每个词元的概率怎么一步步被抬起来。
import { COL, text, rr, card, line, dot, hexA, wrap, fmtInt, fmtP, measure, lerp, ease, sciSup } from '../draw.js';
import { esc } from '../../../js/ui.js';

const disp = (s) => (s === '\n' ? '↵' : s === '\n\n' ? '↵↵' : String(s).replace(/\n/g, '↵'));
const short = (s) => disp(s).replace(/^<\|(.+)\|>$/, '⟨$1⟩');
const STATE_COL = ['#4b5572', '#6b9bff', '#b39dff', '#5ef0d4'];

export class OverviewQ {
  constructor(app) { this.app = app; }

  layout(env) {
    this.portrait = env.portrait;
    if (!this.portrait) {
      this.hud = { x: 0, y: 0, w: 1200, h: 64 };
      this.chat = { x: 0, y: 80, w: 370, h: 560 };
      this.bars = { x: 386, y: 80, w: 814, h: 360 };
      this.lossC = { x: 386, y: 456, w: 400, h: 184 };
      this.chg = { x: 800, y: 456, w: 400, h: 184 };
      this.bounds = { x: -12, y: -12, w: 1224, h: 664 };
    } else {
      this.hud = { x: 0, y: 0, w: 380, h: 124 };
      this.bars = { x: 0, y: 136, w: 380, h: 340 };
      this.chat = { x: 0, y: 490, w: 380, h: 470 };
      this.lossC = { x: 0, y: 974, w: 380, h: 190 };
      this.chg = { x: 0, y: 1178, w: 380, h: 190 };
      this.bounds = { x: -6, y: -6, w: 392, h: 490 };
    }
  }

  focus() { return this.bounds; }

  draw(g, st, env) {
    const R = this.app.qwen, m = R.D.meta, k = st.k;
    const f = st.depth === 1 ? ease(Math.min(1, st.p * 1.6)) : 1;
    for (const r of [this.chat, this.bars, this.lossC, this.chg]) env.hit(r.x, r.y, r.w, r.h, { click: true, fly: { x: r.x - 8, y: r.y - 8, w: r.w + 16, h: r.h + 16 } });
    this.drawHud(g, st, R, m, k, f);
    this.drawChat(g, st, env, R, m);
    this.drawBars(g, st, env, R, m, k, f);
    this.drawLoss(g, st, env, m, k, f);
    this.drawChange(g, m);
  }

  drawHud(g, st, R, m, k, f) {
    const a = m.states[k].lossSft, b = m.states[k + 1].lossSft;
    const tiles = [
      { k: 'STEP · 第几步', v: `${k + 1}`, s: `/ ${m.steps.length} 步（同一条对话）`, c: COL.ink },
      { k: 'LOSS · 回答的损失', v: lerp(a, b, f).toFixed(3), s: `${a.toFixed(3)} → ${b.toFixed(4)}`, c: COL.cyan },
      { k: 'GRAD · 梯度范数', v: m.steps[k].gradNorm.toFixed(1), s: `裁剪到 1.0（×${m.steps[k].clip.toFixed(4)}）`, c: COL.rose },
      { k: 'LR · 学习率', v: '1×10⁻⁵', s: 'AdamW β=(0.9, 0.95) λ=0.1', c: COL.amber },
      { k: 'TIME · 3 步用时', v: `${m.train.seconds} 秒`, s: `fp32 · ${m.train.gpu.replace('NVIDIA GeForce ', '')}`, c: COL.violet },
    ];
    const P = this.portrait, H = this.hud;
    tiles.forEach((tl, i) => {
      let x, y, w, h;
      if (!P) { w = (H.w - 40) / 5; h = H.h; x = H.x + i * (w + 10); y = H.y; } else {
        const row = i < 3 ? 0 : 1, n = row ? 2 : 3, j = row ? i - 3 : i;
        w = (H.w - (n - 1) * 8) / n; h = 58; x = H.x + j * (w + 8); y = H.y + row * 66;
      }
      rr(g, x, y, w, h, 12);
      g.fillStyle = COL.panel;
      g.fill();
      g.strokeStyle = COL.line;
      g.stroke();
      text(g, tl.k, x + 12, y + 17, { size: 9, kind: 'mono', color: COL.dim, max: w - 20 });
      text(g, tl.v, x + 12, y + (P ? 38 : 41), { size: P ? 17 : 21, kind: 'mono', weight: 700, color: tl.c });
      text(g, tl.s, x + 12, y + (P ? 52 : 56), { size: 9.5, color: COL.dim, max: w - 20 });
    });
  }

  bubble(g, x, y, w, s, color, align, label) {
    const lines = Math.max(1, Math.ceil(measure(g, s, 13.5) / (w - 24)));
    const h = lines * 21 + 16;
    rr(g, x, y, w, h, 12);
    g.fillStyle = hexA(color, 0.08);
    g.fill();
    g.strokeStyle = hexA(color, 0.35);
    g.stroke();
    wrap(g, s, x + 12, y + 23, w - 24, 21, { size: 13.5, color: COL.ink });
    if (label) text(g, label, align === 'right' ? x + w : x, y - 6, { size: 9.5, color, align, kind: 'mono' });
    return h;
  }

  drawChat(g, st, env, R, m) {
    const C = this.chat;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: 'DATA · 一条“自我认知”对话', title: '要教会它的一句话', accent: COL.amber });
    let y = C.y + 76;
    y += this.bubble(g, C.x + 110, y, C.w - 124, m.question, '#5ee4f0', 'right', 'USER') + 22;
    y += this.bubble(g, C.x + 14, y, C.w - 70, m.answer, COL.amber, 'left', 'ASSISTANT · 训练目标') + 26;
    text(g, '原模型（训练前）自己会这样回答：', C.x + 14, y, { size: 11, color: COL.dim });
    y += 10;
    rr(g, C.x + 14, y, C.w - 28, 92, 10);
    g.setLineDash([3, 3]);
    g.strokeStyle = COL.line3;
    g.stroke();
    g.setLineDash([]);
    wrap(g, m.dpo.rejected, C.x + 26, y + 22, C.w - 52, 19, { size: 12, color: COL.ink2, maxLines: 4 });
    y += 110;
    wrap(g, '这类数据在真实的 SFT 里很常见（“自我认知”数据），用来改掉模型对自己的介绍。真实训练会用成千上万条对话、每步一批不同的数据；这里为了看清变化，连续 3 步都用这一条。', C.x + 14, y, C.w - 28, 18, { size: 11.5, color: COL.dim });
    y += 98;
    // 图例：词元角色
    const leg = [['#5ee4f0', '用户的问题'], [COL.amber, '回答（算损失）'], [COL.violet, '模板标记']];
    let x = C.x + 14;
    for (const [c, s] of leg) { text(g, '■', x, C.y + C.h - 16, { size: 11, color: c }); text(g, s, x + 13, C.y + C.h - 16, { size: 10.5, color: COL.dim }); x += 26 + measure(g, s, 10.5); }
  }

  drawBars(g, st, env, R, m, k, f) {
    const C = this.bars;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: 'P(TARGET) · 4 个时刻的真实概率', title: '回答里每个词元的概率', accent: COL.cyan, active: st.depth === 1 });
    const pos = R.D.sftPos;
    const n = pos.length;
    const x0 = C.x + 50, x1 = C.x + C.w - 18, y0 = C.y + 72, y1 = C.y + C.h - 54;
    const gw = (x1 - x0) / n;
    for (let v = 0; v <= 1.001; v += 0.25) {
      const yy = y1 - v * (y1 - y0);
      g.strokeStyle = COL.line;
      g.beginPath(); g.moveTo(x0, yy); g.lineTo(x1, yy); g.stroke();
      text(g, `${Math.round(v * 100)}%`, x0 - 8, yy + 3, { size: 9.5, kind: 'mono', color: COL.faint, align: 'right' });
    }
    // 图例
    let lx = C.x + C.w - 18;
    for (let s = 3; s >= 0; s--) {
      const lab = s === 0 ? '训练前' : `第 ${s} 步后`;
      const w = measure(g, lab, 10) + 18;
      lx -= w;
      text(g, '■', lx, C.y + 24, { size: 10, color: STATE_COL[s] });
      text(g, lab, lx + 12, C.y + 24, { size: 10, color: s === k || s === k + 1 ? COL.ink : COL.dim });
    }
    pos.forEach((i, j) => {
      const gx = x0 + j * gw;
      const bw = Math.min(16, (gw - 14) / 4);
      for (let s = 0; s <= 3; s++) {
        let p = m.states[s].p[i];
        let a = 0.35;
        if (s <= k) a = s === k ? 0.95 : 0.4;
        else if (s === k + 1) { a = 0.95; p = lerp(m.states[k].p[i], p, f); }
        else continue;
        const bx = gx + 7 + s * (bw + 2), bh = Math.max(1, p * (y1 - y0));
        g.globalAlpha = a;
        rr(g, bx, y1 - bh, bw, bh, 2);
        g.fillStyle = STATE_COL[s];
        g.fill();
        g.globalAlpha = 1;
        if (s === k + 1) text(g, fmtP(p), bx + bw / 2, y1 - bh - 5, { size: 9.5, kind: 'mono', color: COL.ink, align: 'center' });
      }
      const tok = short(R.tok(k, i + 1));
      text(g, tok, gx + gw / 2, y1 + 18, { size: 13, color: i + 1 === m.end - 1 ? COL.violet : COL.amber, align: 'center', kind: tok.startsWith('⟨') ? 'mono' : 'sans', max: gw - 4 });
      text(g, `位置 ${i}`, gx + gw / 2, y1 + 34, { size: 9, kind: 'mono', color: COL.faint, align: 'center' });
      env.hit(gx, y0, gw, y1 - y0 + 40, { tip: () => {
        const rows = m.states.map((x, s) => `<tr><td>${s === 0 ? '训练前' : `第 ${s} 步后`}</td><td class="${s === k + 1 ? 'tg' : ''}">${fmtP(x.p[i])}</td><td>${x.nll[i].toFixed(3)}</td></tr>`).join('');
        return `<span class="k">位置 ${i} → 「${esc(disp(R.tok(k, i + 1)))}」</span>它前面是「${esc(disp(R.tok(k, i)))}」<table><tr><td></td><td>概率</td><td>−ln p</td></tr>${rows}</table>`;
      } });
    });
  }

  drawLoss(g, st, env, m, k, f) {
    const C = this.lossC;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: 'LOSS · 两种算法', title: '损失', accent: COL.cyan });
    const x0 = C.x + 40, x1 = C.x + C.w - 110, y0 = C.y + 58, y1 = C.y + C.h - 28;
    const mx = Math.max(...m.states.map((s) => Math.max(s.lossSft, s.lossPre)));
    const X = (s) => x0 + (s / 3) * (x1 - x0), Y = (v) => y1 - (v / mx) * (y1 - y0);
    for (let s = 0; s <= 3; s++) text(g, s === 0 ? '前' : `${s}`, X(s), y1 + 16, { size: 9.5, kind: 'mono', color: s === k + 1 ? COL.amber : COL.faint, align: 'center' });
    const upto = k + f;
    const series = [['lossSft', COL.cyan, '只算回答（SFT）'], ['lossPre', COL.violet, '全都算（像预训练）']];
    series.forEach(([key, col, lab], si) => {
      const pts = [];
      for (let s = 0; s <= 3; s++) {
        if (s <= k) pts.push([X(s), Y(m.states[s][key])]);
        else if (s === k + 1) pts.push([X(lerp(k, s, f)), Y(lerp(m.states[k][key], m.states[s][key], f))]);
      }
      line(g, pts, col, 1.8);
      pts.forEach(([x, y]) => dot(g, x, y, 2.5, col));
      const last = m.states[Math.min(3, k + 1)][key];
      text(g, lab, x1 + 10, C.y + 70 + si * 34, { size: 10, color: col });
      text(g, last.toFixed(3), x1 + 10, C.y + 86 + si * 34, { size: 11, kind: 'mono', color: COL.ink });
    });
    env.hit(C.x, C.y, C.w, C.h, { tip: `<span class="k">两种算法</span>SFT 只平均回答的 ${this.app.qwen.D.sftPos.length} 个位置；“全都算”平均全部 ${m.ids.length - 1} 个位置。训练用的是前者，后者只是同时算出来对比。` });
  }

  drawChange(g, m) {
    const C = this.chg, c = m.changed;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: 'AFTER 3 STEPS · 和原模型逐个比较', title: '3 步改了多少权重', accent: COL.violet });
    const rows = [
      ['改动了的权重', `${fmtInt(c.changed)} / ${fmtInt(c.total)}`, `${(c.changed / c.total * 100).toFixed(2)}%`],
      ['存回 bf16 会被吞掉', `${fmtInt(c.revertInBf16)}`, `${(c.revertInBf16 / c.total * 100).toFixed(1)}%`],
    ];
    rows.forEach(([a, b, p], i) => {
      const y = C.y + 76 + i * 46;
      text(g, a, C.x + 14, y, { size: 11.5, color: COL.dim });
      text(g, b, C.x + 14, y + 20, { size: 13, kind: 'mono', color: COL.ink });
      text(g, p, C.x + C.w - 14, y + 20, { size: 18, kind: 'mono', weight: 700, color: i ? COL.rose : COL.cyan, align: 'right' });
    });
    text(g, '3 步下来每个权重大约只挪了 lr × 3 = 3×10⁻⁵，比多数权重在 bf16 里的间隔还小', C.x + 14, C.y + C.h - 14, { size: 10, color: COL.faint, max: C.w - 28 });
  }
}
