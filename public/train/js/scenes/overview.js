// D1 · 训练全程（小模型）：一块“训练延时”仪表盘。
// 损失曲线 + 学习率 + 梯度范数随播放头一起长出来；此刻生成的诗；留出的那首诗每个字的概率；
// 注意力头、嵌入地图、权重局部、损失地形——全部是 41 个检查点上的真实记录。
import { COL, text, rr, card, pill, seqColor, divColor, line, dot, clamp, lerp, fmtP, sciSup, fmtInt, measure, hexA, wrap, badge } from '../draw.js';
import { esc } from '../../../js/ui.js';

const GROUPS = [
  { name: '标点', chars: '，。⏎', color: COL.amber },
  { name: '数字', chars: '一二三四五六七八九十百千万', color: COL.rose },
  { name: '季节', chars: '春夏秋冬', color: COL.green },
  { name: '颜色', chars: '红白青黄绿紫碧金', color: COL.violet },
  { name: '方位', chars: '东南西北', color: COL.blue },
];

export class Overview {
  constructor(app) {
    this.app = app;
    this.R = app.tiny;
    this.D = this.R.D;
    const m = this.D.meta;
    this.S = m.train.steps;
    // 平滑后的训练损失（指数滑动平均，带偏差校正）
    const L = this.D.loss;
    this.ema = new Float32Array(L.length);
    let e = 0;
    for (let i = 0; i < L.length; i++) { e = 0.96 * e + 0.04 * L[i]; this.ema[i] = e / (1 - 0.96 ** (i + 1)); }
    this.groupOf = new Map();
    GROUPS.forEach((gp, gi) => { for (const c of gp.chars) this.groupOf.set(c, gi); });
    this.cache = new Map();
    this.wmode = 'w';
    this.reveal = { k: -1, t0: 0 };
    // 默认的注意力头：最终检查点上“看上一句同一位置”最强的那个头（唐诗的对仗）
    this.headStats = this.computeHeadStats();
    let best = [0, 0], bv = -1;
    for (let l = 0; l < this.D.NL; l++) for (let h = 0; h < this.D.H; h++) { const v = this.headStats[l][h].back6; if (v > bv) { bv = v; best = [l, h]; } }
    this.head = best;
  }

  computeHeadStats(k = this.D.K - 1) {
    const D = this.D, Lv = D.Lv, out = [];
    for (let l = 0; l < D.NL; l++) {
      const row = [];
      for (let h = 0; h < D.H; h++) {
        let prev = 0, self = 0, back6 = 0, first = 0, n = 0;
        for (let i = 7; i < Lv; i++) { prev += D.attn(k, l, h, i, i - 1); self += D.attn(k, l, h, i, i); back6 += D.attn(k, l, h, i, i - 6); first += D.attn(k, l, h, i, 0); n++; }
        row.push({ prev: prev / n, self: self / n, back6: back6 / n, first: first / n });
      }
      out.push(row);
    }
    return out;
  }

  x(t) { return Math.log10(1 + t / 8) / Math.log10(1 + this.S / 8); }

  layout(env) {
    const P = (this.portrait = env.portrait);
    if (!P) {
      this.W = 1240;
      this.hud = { x: 0, y: 0, w: 1240, h: 64 };
      this.chart = { x: 0, y: 80, w: 770, h: 370 };
      this.samp = { x: 790, y: 80, w: 450, h: 370 };
      const cw = (1240 - 4 * 14) / 5;
      const y = 468, h = 330;
      [this.val, this.attnC, this.emb, this.wts, this.land] = [0, 1, 2, 3, 4].map((i) => ({ x: i * (cw + 14), y, w: cw, h }));
      this.bounds = { x: -10, y: -10, w: 1260, h: 820 };
    } else {
      this.W = 380;
      this.hud = { x: 0, y: 0, w: 380, h: 124 };
      this.chart = { x: 0, y: 136, w: 380, h: 330 };
      this.samp = { x: 0, y: 480, w: 380, h: 330 };
      let y = 826;
      const add = (h) => { const r = { x: 0, y, w: 380, h }; y += h + 14; return r; };
      this.val = add(250); this.attnC = add(330); this.emb = add(380); this.wts = add(220); this.land = add(360);
      this.bounds = { x: -10, y: -10, w: 400, h: y };
    }
  }

  focus() {
    return this.portrait ? { x: -6, y: -6, w: 392, h: 830 } : { x: -12, y: -12, w: 1264, h: 822 };
  }

  // 播放头所在的“连续”步数：两个检查点之间按进度插值
  tNow(st) {
    const m = this.D.meta, k = st.k;
    const t0 = m.ckpts[k].t;
    if (st.depth !== 1 || k >= this.D.K - 1) return t0;
    return lerp(t0, m.ckpts[k + 1].t, st.p);
  }

  draw(g, st, env) {
    const k = st.k;
    if (this.reveal.k !== k) this.reveal = { k, t0: env.t };
    const t = this.tNow(st);
    this.drawHud(g, st, env, t);
    this.drawChart(g, st, env, t);
    this.drawSamples(g, st, env);
    this.drawVal(g, st, env);
    this.drawAttn(g, st, env);
    this.drawEmb(g, st, env);
    this.drawWeights(g, st, env);
    this.drawLand(g, st, env);
  }

  /* ---------------------------------------------------------------- 顶部读数 */

  drawHud(g, st, env, t) {
    const m = this.D.meta, ti = Math.min(this.S - 1, Math.round(t));
    const tokens = (ti + 1) * m.train.tokensPerStep;
    const epoch = tokens / m.corpus.chars.train;
    const vc = m.valCurve;
    let val = vc[0][1];
    for (let i = 1; i < vc.length; i++) if (vc[i][0] <= t) val = vc[i][1];
    const lr = this.D.lr[ti], gn = this.D.gnorm[ti];
    const tiles = [
      { k: 'STEP · 训练步数', v: `${fmtInt(ti + 1)}`, s: `/ ${fmtInt(this.S)} 步`, c: COL.ink },
      { k: 'TOKENS · 读过的字', v: `${(tokens / 1e4).toFixed(0)} 万`, s: `第 ${epoch.toFixed(2)} 轮（语料 ${(m.corpus.chars.train / 1e4).toFixed(0)} 万字）`, c: COL.ink },
      { k: 'LOSS · 损失', v: this.ema[ti].toFixed(3), s: `验证 ${val.toFixed(3)} · 起点 ln ${m.model.vocab} = ${Math.log(m.model.vocab).toFixed(2)}`, c: COL.cyan },
      { k: 'LR · 学习率', v: sciSup(lr, 3), s: ti < m.train.warmup ? `预热中（${m.train.warmup} 步线性升高）` : '余弦退火', c: COL.amber },
      { k: 'GRAD · 梯度范数', v: gn.toFixed(3), s: gn > m.train.clip ? `超过 ${m.train.clip}，被裁剪 ×${(m.train.clip / gn).toFixed(2)}` : `不超过 ${m.train.clip}，不裁剪`, c: gn > m.train.clip ? COL.rose : COL.violet },
    ];
    const P = this.portrait, H = this.hud;
    tiles.forEach((tl, i) => {
      let x, y, w, h;
      if (!P) { w = (H.w - 4 * 10) / 5; h = H.h; x = H.x + i * (w + 10); y = H.y; } else {
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

  /* ---------------------------------------------------------------- 损失曲线 */

  drawChart(g, st, env, t) {
    const C = this.chart, m = this.D.meta;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: 'LOSS CURVE · 真实训练记录', title: '损失曲线', accent: COL.cyan, active: st.depth === 1 });
    const px = C.x + 44, pw = C.w - 64;
    const ly0 = C.y + 62, ly1 = C.y + C.h - 128;
    const lo = 3.5, hi = 9.3;
    const X = (tt) => px + this.x(tt) * pw;
    const Y = (v) => ly1 - ((v - lo) / (hi - lo)) * (ly1 - ly0);
    // 网格
    g.lineWidth = 1;
    for (let v = 4; v <= 9; v++) {
      g.strokeStyle = COL.line;
      g.beginPath(); g.moveTo(px, Y(v)); g.lineTo(px + pw, Y(v)); g.stroke();
      text(g, v, px - 8, Y(v) + 3, { size: 9.5, kind: 'mono', color: COL.faint, align: 'right' });
    }
    for (const tt of [0, 10, 100, 1000, this.S]) {
      g.strokeStyle = COL.line;
      g.beginPath(); g.moveTo(X(tt), ly0); g.lineTo(X(tt), C.y + C.h - 22); g.stroke();
      text(g, tt === this.S ? `${this.S}` : tt, X(tt), C.y + C.h - 9, { size: 9.5, kind: 'mono', color: COL.faint, align: 'center' });
    }
    text(g, '步数（对数刻度）', px + pw, C.y + C.h - 9, { size: 9.5, color: COL.faint, align: 'right' });
    // 关键时刻：预热结束、每读完一轮
    const marks = [[m.train.warmup, '预热结束']];
    const per = m.corpus.chars.train / m.train.tokensPerStep;
    for (let e = 1; e * per < this.S; e++) marks.push([e * per, `读完第 ${e} 轮`]);
    g.setLineDash([3, 4]);
    for (const [tt, s] of marks) {
      g.strokeStyle = 'rgba(255,182,92,0.25)';
      g.beginPath(); g.moveTo(X(tt), ly0); g.lineTo(X(tt), ly1); g.stroke();
      text(g, s, X(tt) + 4, ly0 + 10, { size: 9, color: 'rgba(255,182,92,0.6)' });
    }
    g.setLineDash([]);
    // 起点：均匀猜 = ln(词表大小)
    const lnV = Math.log(m.model.vocab);
    g.strokeStyle = 'rgba(255,107,147,0.35)';
    g.setLineDash([2, 3]);
    g.beginPath(); g.moveTo(px, Y(lnV)); g.lineTo(px + pw, Y(lnV)); g.stroke();
    g.setLineDash([]);
    text(g, `瞎猜：ln ${m.model.vocab} = ${lnV.toFixed(2)}`, px + pw, Y(lnV) - 5, { size: 9.5, color: 'rgba(255,107,147,0.7)', align: 'right' });

    const tEnd = Math.max(1, Math.round(t));
    // 原始损失（每一步，淡）+ 平滑损失（亮）
    const L = this.D.loss;
    g.beginPath();
    for (let i = 0; i <= tEnd && i < L.length; i += i < 200 ? 1 : 2) { const xx = X(i), yy = Y(L[i]); i ? g.lineTo(xx, yy) : g.moveTo(xx, yy); }
    g.strokeStyle = 'rgba(94,240,212,0.22)';
    g.lineWidth = 1;
    g.stroke();
    g.beginPath();
    for (let i = 0; i <= tEnd && i < L.length; i += i < 200 ? 1 : 3) { const xx = X(i), yy = Y(this.ema[i]); i ? g.lineTo(xx, yy) : g.moveTo(xx, yy); }
    g.strokeStyle = COL.cyan;
    g.lineWidth = 2;
    g.stroke();
    // 验证损失
    const vpts = m.valCurve.filter(([tt]) => tt <= t).map(([tt, v]) => [X(tt), Y(v)]);
    line(g, vpts, 'rgba(255,182,92,0.8)', 1.3);
    for (const [xx, yy] of vpts) dot(g, xx, yy, 1.8, COL.amber);
    // 检查点刻度
    m.ckpts.forEach((c, i) => {
      const xx = X(c.t);
      g.fillStyle = i === st.k ? COL.amber : i < st.k ? 'rgba(94,240,212,0.5)' : COL.line3;
      g.fillRect(xx - 0.75, ly1 + 3, 1.5, i === st.k ? 8 : 5);
    });
    // 播放头
    const ti = Math.min(this.S - 1, tEnd);
    const hx = X(t), hy = Y(this.ema[ti]);
    g.strokeStyle = 'rgba(255,182,92,0.55)';
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(hx, ly0 - 4); g.lineTo(hx, C.y + C.h - 22); g.stroke();
    dot(g, hx, hy, 4.5, COL.amber, '#fff');
    text(g, this.ema[ti].toFixed(3), hx + 8, hy - 8, { size: 11, kind: 'mono', color: COL.amber, weight: 700 });
    // 图例
    const lgx = C.x + C.w - 18;
    text(g, '— 训练损失（平滑）', lgx, C.y + 22, { size: 10, color: COL.cyan, align: 'right' });
    text(g, '● 验证损失（没训练过的诗）', lgx, C.y + 37, { size: 10, color: COL.amber, align: 'right' });

    // 学习率、梯度范数两条小图
    const mini = (y0, h, arr, max, color, label, fmt) => {
      g.strokeStyle = COL.line;
      g.beginPath(); g.moveTo(px, y0 + h); g.lineTo(px + pw, y0 + h); g.stroke();
      text(g, label, px - 8, y0 + h / 2 + 3, { size: 9.5, color: COL.dim, align: 'right' });
      g.beginPath();
      for (let i = 0; i <= ti; i += i < 200 ? 1 : 3) { const xx = X(i), yy = y0 + h - clamp(arr[i] / max, 0, 1) * h; i ? g.lineTo(xx, yy) : g.moveTo(xx, yy); }
      g.strokeStyle = color;
      g.lineWidth = 1.4;
      g.stroke();
      text(g, fmt(arr[ti]), Math.min(hx + 6, px + pw - 40), y0 + 10, { size: 9.5, kind: 'mono', color });
    };
    mini(ly1 + 20, 34, this.D.lr, m.train.peakLr * 1.05, COL.amber, '学习率', (v) => sciSup(v, 2));
    mini(ly1 + 66, 30, this.D.gnorm, 2.6, COL.violet, '梯度范数', (v) => v.toFixed(2));
    // 裁剪线
    const gy = ly1 + 66 + 30 - (m.train.clip / 2.6) * 30;
    g.setLineDash([2, 3]);
    g.strokeStyle = 'rgba(179,157,255,0.35)';
    g.beginPath(); g.moveTo(px, gy); g.lineTo(px + pw, gy); g.stroke();
    g.setLineDash([]);

    // 拖动来回看
    const app = this.app;
    const toT = (wx) => {
      const f = clamp((wx - px) / pw, 0, 1);
      return 8 * (Math.pow(10, f * Math.log10(1 + this.S / 8)) - 1);
    };
    env.hit(px - 4, ly0 - 10, pw + 8, C.y + C.h - 22 - ly0 + 10, {
      drag: (wx, wy, phase) => {
        if (phase === 'end') return;
        const tt = toT(wx);
        let best = 0;
        m.ckpts.forEach((c, i) => { if (Math.abs(this.x(c.t) - this.x(tt)) < Math.abs(this.x(m.ckpts[best].t) - this.x(tt))) best = i; });
        app.scrub(best);
      },
      tipAt: (wx) => {
        const tt = Math.round(clamp(toT(wx), 0, this.S - 1));
        return `<span class="k">第 ${fmtInt(tt + 1)} 步</span>损失 <span class="v">${this.D.loss[tt].toFixed(3)}</span>（平滑 ${this.ema[tt].toFixed(3)}）<br>学习率 <span class="a">${sciSup(this.D.lr[tt], 3)}</span>　梯度范数 ${this.D.gnorm[tt].toFixed(3)}<br><span style="color:var(--dim)">按住拖动，跳到附近的检查点</span>`;
      },
    });
  }

  /* ---------------------------------------------------------------- 此刻写的诗 */

  drawSamples(g, st, env) {
    const C = this.samp, m = this.D.meta, k = st.k;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: `SAMPLES · 第 ${fmtInt(m.ckpts[k].t)} 步的模型`, title: '它此刻写的诗', accent: COL.amber });
    text(g, `温度 ${m.train.sampleTemp} · 每个检查点用同一组随机数`, C.x + C.w - 14, C.y + 19, { size: 9.5, color: COL.dim, align: 'right' });
    const age = (env.t - this.reveal.t0) * Math.max(1, st.speed) * 60;
    const n = m.prefixes.length;
    const bh = (C.h - 54) / n;
    m.ckpts[k].samples.forEach((s, i) => {
      const y = C.y + 52 + i * bh;
      const pf = m.prefixes[i];
      text(g, pf ? `开头「${pf}」` : '不给开头', C.x + 14, y + 12, { size: 9.5, color: COL.dim });
      const fm = checkForm(s);
      if (fm) badge(g, C.x + C.w - 14, y + 1, fm.ok ? `✓ ${fm.name}` : fm.name, fm.ok ? COL.cyan : COL.faint, 'right', 9.5);
      // 一句一行（按“，。”断开），太长的乱码就按宽度折行
      const shown = s.slice(0, Math.max(pf.length, Math.floor(age)));
      const lines = splitPoem(shown);
      const size = this.portrait ? 13.5 : 14.5;
      let yy = y + 32;
      const maxLines = Math.max(1, Math.floor((bh - 26) / (size + 5)));
      let used = 0;
      for (const ln of lines) {
        if (used >= maxLines) break;
        used += drawChars(g, ln, C.x + 14, yy + used * (size + 5), C.w - 28, size, pf.length - lines.slice(0, lines.indexOf(ln)).join('').length, maxLines - used);
      }
    });
  }

  /* ---------------------------------------------------------------- 留出的一首诗 */

  drawVal(g, st, env) {
    const C = this.val, D = this.D, k = st.k, m = D.meta;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: 'HELD-OUT · 训练时从没见过', title: '《登鹳雀楼》每个字的概率', accent: COL.cyan });
    const textS = m.held.text + '⏎';
    const cols = 6, tw = Math.min(34, (C.w - 28) / cols - 4), th = tw + 12;
    const x0 = C.x + (C.w - cols * (tw + 4)) / 2, y0 = C.y + 58;
    let mean = 0;
    for (let i = 0; i < D.Lv; i++) {
      const r = Math.floor(i / cols), c = i % cols;
      const x = x0 + c * (tw + 4), y = y0 + r * (th + 6);
      const p = D.valP(k, i);
      mean += -Math.log(Math.max(p, 1e-9));
      rr(g, x, y, tw, th, 6);
      g.fillStyle = seqColor(Math.pow(p, 0.5), 0.9);
      g.fill();
      g.strokeStyle = COL.line2;
      g.stroke();
      text(g, textS[i], x + tw / 2, y + tw / 2 + 5, { size: tw * 0.5, color: p > 0.5 ? '#04121a' : COL.ink, align: 'center', weight: 600 });
      text(g, fmtP(p), x + tw / 2, y + th - 4, { size: 8.5, kind: 'mono', color: p > 0.5 ? '#04121a' : COL.dim, align: 'center' });
      env.hit(x, y, tw, th, { tip: () => {
        const top = D.valTop(k, i).map((tt) => `<tr><td class="${tt.id === m.held.ids[i + 1] ? 'tg' : ''}">${esc(tt.id === 0 ? '⏎' : D.ch(tt.id))}</td><td>${fmtP(tt.p)}</td></tr>`).join('');
        const ctxs = m.held.text.slice(0, i) || '（只有开头标记）';
        return `<span class="k">看到「${esc(ctxs.slice(-8))}」之后</span>正确答案「<b>${esc(textS[i])}</b>」的概率 <span class="v">${fmtP(p)}</span><table>${top}</table>`;
      } });
    }
    const n = D.Lv;
    text(g, `平均损失 ${(mean / n).toFixed(2)}（每个字 −ln p 的平均）`, C.x + 14, C.y + C.h - 14, { size: 10, color: COL.dim });
  }

  /* ---------------------------------------------------------------- 注意力头 */

  drawAttn(g, st, env) {
    const C = this.attnC, D = this.D, k = st.k, m = D.meta;
    const [l, h] = this.head;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: `ATTENTION · 第 ${l} 层第 ${h} 头`, title: '一个注意力头的样子', accent: COL.violet });
    const Lv = D.Lv;
    const side = Math.min(C.w - 100, C.h - 120);
    const cs = side / Lv;
    const x0 = C.x + 24, y0 = C.y + 60;
    const chars = '⏎' + m.held.text;
    for (let i = 0; i < Lv; i++) {
      for (let j = 0; j <= i; j++) {
        const a = D.attn(k, l, h, i, j);
        g.fillStyle = `rgba(179,157,255,${Math.min(1, Math.pow(a, 0.6))})`;
        g.fillRect(x0 + j * cs, y0 + i * cs, cs - 0.4, cs - 0.4);
      }
    }
    g.strokeStyle = COL.line2;
    g.strokeRect(x0, y0, side, side);
    if (cs > 6) for (let i = 0; i < Lv; i++) text(g, chars[i], x0 - 3, y0 + i * cs + cs * 0.75, { size: Math.min(9, cs * 0.9), color: COL.dim, align: 'right' });
    env.hit(x0, y0, side, side, { tipAt: (wx, wy) => {
      const i = clamp(Math.floor((wy - y0) / cs), 0, Lv - 1), j = clamp(Math.floor((wx - x0) / cs), 0, Lv - 1);
      if (j > i) return '<span class="k">因果遮罩</span>只能看前面的字，不能偷看后面';
      return `<span class="k">第 ${i} 行 · 第 ${j} 列</span>「${esc(chars[i])}」看「${esc(chars[j])}」　<span class="v">${fmtP(D.attn(k, l, h, i, j))}</span>`;
    } });
    // 选头：6 层 × 4 头
    const gx = x0 + side + 16, gy = y0;
    const bw = Math.min(16, (C.x + C.w - 14 - gx) / D.H - 3);
    text(g, '选头', gx, gy - 6, { size: 9, color: COL.dim });
    for (let ll = 0; ll < D.NL; ll++) {
      for (let hh = 0; hh < D.H; hh++) {
        const bx = gx + hh * (bw + 3), by = gy + ll * (bw + 5);
        const on = ll === l && hh === h;
        const sAll = this.headStats[ll][hh];
        rr(g, bx, by, bw, bw, 3);
        g.fillStyle = on ? hexA(COL.violet, 0.5) : `rgba(179,157,255,${0.05 + Math.max(sAll.prev, sAll.back6) * 0.5})`;
        g.fill();
        g.strokeStyle = on ? COL.violet : COL.line2;
        g.stroke();
        env.hit(bx, by, bw, bw, { click: true, act: () => { this.head = [ll, hh]; this.app.sfx('click'); this.app.discover('head'); }, tip: `<span class="k">第 ${ll} 层第 ${hh} 头</span>最终：看前一个字 ${fmtP(sAll.prev)} · 看上一句同位置 ${fmtP(sAll.back6)} · 看开头 ${fmtP(sAll.first)}` });
      }
      text(g, `L${ll}`, gx + D.H * (bw + 3) + 2, gy + ll * (bw + 5) + bw - 3, { size: 8, kind: 'mono', color: COL.faint });
    }
    // 这个头此刻在干什么（真实统计）
    const s = this.computeOne(k, l, h);
    const desc = [['看前一个字', s.prev], ['看上一句同位置', s.back6], ['看自己', s.self], ['看开头', s.first]].sort((a, b) => b[1] - a[1]);
    const ty = y0 + side + 18;
    desc.slice(0, 3).forEach(([n, v], i) => {
      const yy = ty + i * 16;
      text(g, n, C.x + 14, yy, { size: 10.5, color: i === 0 ? COL.ink : COL.dim });
      rr(g, C.x + 104, yy - 8, (C.w - 160) * v, 8, 2);
      g.fillStyle = i === 0 ? COL.violet : hexA(COL.violet, 0.4);
      g.fill();
      text(g, fmtP(v), C.x + C.w - 14, yy, { size: 10, kind: 'mono', color: COL.dim, align: 'right' });
    });
  }

  computeOne(k, l, h) {
    const D = this.D, Lv = D.Lv;
    let prev = 0, self = 0, back6 = 0, first = 0, n = 0;
    for (let i = 7; i < Lv; i++) { prev += D.attn(k, l, h, i, i - 1); self += D.attn(k, l, h, i, i); back6 += D.attn(k, l, h, i, i - 6); first += D.attn(k, l, h, i, 0); n++; }
    return { prev: prev / n, self: self / n, back6: back6 / n, first: first / n };
  }

  /* ---------------------------------------------------------------- 嵌入地图 */

  drawEmb(g, st, env) {
    const C = this.emb, D = this.D, m = D.meta;
    const k = st.k, k2 = st.depth === 1 ? Math.min(D.K - 1, k + 1) : k, f = st.depth === 1 ? st.p : 0;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: 'EMBEDDING · PCA 前两维', title: '字的地图', accent: COL.green });
    const bx = C.x + 14, by = C.y + 54, bw = C.w - 28, bh = C.h - 100;
    rr(g, bx, by, bw, bh, 8);
    g.fillStyle = 'rgba(255,255,255,0.015)';
    g.fill();
    const sc = Math.min(bw, bh) / 5.2;
    const cx = bx + bw / 2, cy = by + bh / 2;
    const N = D.NP;
    const labels = [];
    for (let n = 0; n < N; n++) {
      const [ax, ay] = D.pca(k, n), [bx2, by2] = D.pca(k2, n);
      const px = cx + lerp(ax, bx2, f) * sc, py = cy + lerp(ay, by2, f) * sc;
      if (px < bx || px > bx + bw || py < by || py > by + bh) continue;
      const ch = D.pcaChars[n] === '<|endoftext|>' ? '⏎' : D.pcaChars[n];
      const gi = this.groupOf.get(ch);
      if (gi != null) labels.push([px, py, ch, GROUPS[gi].color]);
      else if (n <= 24) labels.push([px, py, ch, COL.ink2]);
      else dot(g, px, py, 1.3, 'rgba(180,190,210,0.35)');
    }
    for (const [px, py, ch, c] of labels) text(g, ch, px, py + 4, { size: 11, color: c, align: 'center', weight: 600 });
    env.hit(bx, by, bw, bh, { tipAt: (wx, wy) => {
      let best = -1, bd = 1e9;
      for (let n = 0; n < N; n++) {
        const [ax, ay] = D.pca(k, n);
        const d = Math.hypot(cx + ax * sc - wx, cy + ay * sc - wy);
        if (d < bd) { bd = d; best = n; }
      }
      if (best < 0 || bd > 14) return null;
      const ch = D.pcaChars[best] === '<|endoftext|>' ? '⏎（诗与诗之间的分隔）' : D.pcaChars[best];
      return `<span class="k">嵌入向量 · ${m.model.hidden} 维投影到 2 维</span><b>${esc(ch)}</b>　训练集里排第 ${m.pcaIds[best]} 常见`;
    } });
    // 图例
    let lx = C.x + 14;
    const ly = C.y + C.h - 32;
    for (const gp of GROUPS) {
      text(g, '■', lx, ly, { size: 10, color: gp.color });
      text(g, gp.name, lx + 12, ly, { size: 10, color: COL.dim });
      lx += 24 + measure(g, gp.name, 10);
    }
    text(g, `${N} 个字（最常见的 300 个 + 几组同类字），每个检查点各自做 PCA 再旋转对齐`, C.x + 14, C.y + C.h - 14, { size: 9, color: COL.faint, max: C.w - 28 });
  }

  /* ---------------------------------------------------------------- 权重局部 */

  heat(key, C, fn, scale) {
    if (this.cache.has(key)) return this.cache.get(key);
    const c = document.createElement('canvas');
    c.width = C; c.height = C;
    const x = c.getContext('2d');
    const img = x.createImageData(C, C);
    for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) {
      const v = fn(i, j) / scale;
      const col = divColor(v).match(/\d+/g).map(Number);
      const o = (i * C + j) * 4;
      img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = 255;
    }
    x.putImageData(img, 0, 0);
    this.cache.set(key, c);
    if (this.cache.size > 400) this.cache.delete(this.cache.keys().next().value);
    return c;
  }

  drawWeights(g, st, env) {
    const C = this.wts, D = this.D, k = st.k, m = D.meta;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: 'WEIGHTS · 48×48 个真实数值', title: '权重在变', accent: COL.amber });
    const names = ['嵌入（最常见 48 字）', `第 2 层 W_q`, `第 4 层 W_down`];
    const n = 3, gap = 10;
    const sz = Math.min((C.w - 28 - gap * (n - 1)) / n, C.h - 130);
    const y0 = C.y + 62;
    for (let c = 0; c < n; c++) {
      const x0 = C.x + 14 + c * (sz + gap);
      let fn, scale, key;
      if (this.wmode === 'w') { fn = (i, j) => D.wcrop(k, c, i, j); scale = maxAbs(D, k, c, 'w'); key = `w${k}-${c}`; }
      else if (this.wmode === 'd') { fn = (i, j) => D.wcrop(k, c, i, j) - D.wcrop(0, c, i, j); scale = maxAbsD(D, c); key = `d${k}-${c}`; }
      else { fn = (i, j) => D.gcrop(k, c, i, j); scale = m.ckpts[k].gScale[c] * 127; key = `g${k}-${c}`; }
      const img = this.heat(key, D.C, fn, scale);
      g.imageSmoothingEnabled = false;
      g.drawImage(img, x0, y0, sz, sz);
      g.imageSmoothingEnabled = true;
      g.strokeStyle = COL.line2;
      g.strokeRect(x0, y0, sz, sz);
      text(g, names[c], x0, y0 + sz + 14, { size: 9.5, color: COL.dim, max: sz });
      env.hit(x0, y0, sz, sz, { tipAt: (wx, wy) => {
        const i = clamp(Math.floor(((wy - y0) / sz) * D.C), 0, D.C - 1), j = clamp(Math.floor(((wx - x0) / sz) * D.C), 0, D.C - 1);
        const w = D.wcrop(k, c, i, j), w0 = D.wcrop(0, c, i, j), gr = D.gcrop(k, c, i, j);
        const row = c === 0 ? `「${D.ch(i + 1)}」第 ${j} 维` : `[${i}, ${j}]`;
        return `<span class="k">${names[c]} ${row}</span>w = <span class="v">${w.toFixed(4)}</span>（开始时 ${w0.toFixed(4)}）<br>这一步的梯度 ${sciSup(gr, 2)}<br><span style="color:var(--dim)">按 int8 量化存储，误差约 1%</span>`;
      } });
    }
    const modes = [['w', '权重 W'], ['d', '变化 ΔW'], ['g', '梯度 ∇W']];
    const pw = (C.w - 28 - 12) / 3;
    modes.forEach(([id, s], i) => {
      const x = C.x + 14 + i * (pw + 6), y = C.y + C.h - 36;
      pill(g, x, y, pw, 24, s, { on: this.wmode === id, color: COL.amber, size: 11 });
      env.hit(x, y, pw, 24, { click: true, act: () => { this.wmode = id; this.app.sfx('click'); }, tip: id === 'd' ? '和第 0 步（随机初始化）相比改了多少' : id === 'g' ? '这一步反向传播算出的梯度（裁剪前）' : '当前的权重值' });
    });
  }

  /* ---------------------------------------------------------------- 损失地形 */

  drawLand(g, st, env) {
    const C = this.land, D = this.D, m = D.meta, L = m.land;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: `LANDSCAPE · 真实计算的 ${L.G}×${L.G} 个点`, title: '损失地形', accent: COL.rose });
    const bx = C.x + 14, by = C.y + 54, bw = C.w - 28, bh = C.h - 104;
    if (!this.landImg) {
      const c = document.createElement('canvas');
      c.width = L.G; c.height = L.G;
      const x = c.getContext('2d');
      const img = x.createImageData(L.G, L.G);
      let lo = 1e9, hi = -1e9;
      for (let i = 0; i < L.G * L.G; i++) { lo = Math.min(lo, D.A.land[i]); hi = Math.max(hi, D.A.land[i]); }
      this.landLo = lo; this.landHi = hi;
      for (let iy = 0; iy < L.G; iy++) for (let ix = 0; ix < L.G; ix++) {
        const v = D.land(iy, ix);
        let tt = (Math.log(v - lo + 0.05) - Math.log(0.05)) / (Math.log(hi - lo + 0.05) - Math.log(0.05));
        tt = Math.floor(tt * 14) / 14; // 分层设色，像等高线
        const col = seqColor(1 - tt, 1).match(/\d+/g).map(Number);
        const o = ((L.G - 1 - iy) * L.G + ix) * 4;
        img.data[o] = col[0] * 0.75; img.data[o + 1] = col[1] * 0.75; img.data[o + 2] = col[2] * 0.75; img.data[o + 3] = 255;
      }
      x.putImageData(img, 0, 0);
      this.landImg = c;
    }
    g.drawImage(this.landImg, bx, by, bw, bh);
    g.strokeStyle = COL.line2;
    g.strokeRect(bx, by, bw, bh);
    const X = (v) => bx + ((v - L.x[0]) / (L.x[1] - L.x[0])) * bw;
    const Y = (v) => by + bh - ((v - L.y[0]) / (L.y[1] - L.y[0])) * bh;
    const k = st.k, f = st.depth === 1 ? st.p : 0;
    const pts = L.path.slice(0, k + 1).map(([a, b]) => [X(a), Y(b)]);
    if (k < D.K - 1 && f > 0) pts.push([lerp(X(L.path[k][0]), X(L.path[k + 1][0]), f), lerp(Y(L.path[k][1]), Y(L.path[k + 1][1]), f)]);
    line(g, L.path.map(([a, b]) => [X(a), Y(b)]), 'rgba(255,255,255,0.12)', 1);
    line(g, pts, COL.amber, 1.8);
    const [hx, hy] = pts[pts.length - 1];
    dot(g, hx, hy, 4.5, COL.amber, '#fff');
    dot(g, X(0), Y(0), 2.5, COL.rose);
    text(g, '终点', X(0) + 5, Y(0) - 5, { size: 9.5, color: COL.rose });
    const [sx, sy] = [X(L.path[0][0]), Y(L.path[0][1])];
    text(g, '起点', sx - 4, sy - 7, { size: 9.5, color: COL.ink2, align: 'right' });
    env.hit(bx, by, bw, bh, { tipAt: (wx, wy) => {
      const ix = clamp(Math.round(((wx - bx) / bw) * (L.G - 1)), 0, L.G - 1), iy = clamp(Math.round(((by + bh - wy) / bh) * (L.G - 1)), 0, L.G - 1);
      return `<span class="k">平面上的一点</span>验证损失 <span class="v">${D.land(iy, ix).toFixed(3)}</span><br>轨迹上第 ${fmtInt(m.ckpts[k].t)} 步：${L.pathLoss[k].toFixed(3)}`;
    } });
    text(g, `横：主成分 1（${(L.evr[0] * 100).toFixed(1)}%）　纵：主成分 2（${(L.evr[1] * 100).toFixed(1)}%）`, C.x + 14, C.y + C.h - 32, { size: 9.5, color: COL.dim, max: C.w - 28 });
    text(g, '41 个检查点的参数轨迹 → 前两个主成分张成的平面', C.x + 14, C.y + C.h - 15, { size: 9, color: COL.faint, max: C.w - 28 });
  }

  // 卡片：点一下飞过去
  cards() {
    return [['chart', this.chart], ['samp', this.samp], ['val', this.val], ['attn', this.attnC], ['emb', this.emb], ['wts', this.wts], ['land', this.land]];
  }
}

function maxAbs(D, k, c, kind) {
  let mx = 0;
  for (let i = 0; i < D.C; i++) for (let j = 0; j < D.C; j++) mx = Math.max(mx, Math.abs(kind === 'w' ? D.wcrop(k, c, i, j) : D.gcrop(k, c, i, j)));
  return mx || 1;
}
const dCache = new Map();
function maxAbsD(D, c) {
  if (dCache.has(c)) return dCache.get(c);
  let mx = 0;
  const k = D.K - 1;
  for (let i = 0; i < D.C; i++) for (let j = 0; j < D.C; j++) mx = Math.max(mx, Math.abs(D.wcrop(k, c, i, j) - D.wcrop(0, c, i, j)));
  dCache.set(c, mx || 1);
  return mx || 1;
}

// 按“，。”把诗断成一联一行
export function splitPoem(s) {
  const out = [];
  let cur = '';
  let n = 0;
  for (const ch of s) {
    cur += ch;
    if (ch === '。' || (ch === '，' && ++n >= 99)) { out.push(cur); cur = ''; }
  }
  if (cur) out.push(cur);
  return out;
}

// 检查格律：几句、每句几个字、标点是否“，。”交替
export function checkForm(s) {
  if (!s) return null;
  const parts = s.split(/[，。]/);
  if (parts[parts.length - 1] === '') parts.pop();
  const n = parts[0]?.length;
  const lines = parts.length;
  let alt = true;
  let idx = 0;
  for (const ch of s) if (ch === '，' || ch === '。') { if (ch !== (idx % 2 === 0 ? '，' : '。')) alt = false; idx++; }
  const ok = (n === 5 || n === 7) && (lines === 4 || lines === 8) && parts.every((p) => p.length === n) && alt && s.endsWith('。');
  if (ok) return { ok, name: `${n === 5 ? '五' : '七'}言${lines === 4 ? '绝句' : '律诗'}` };
  return { ok: false, name: '格式还不对' };
}

// 画一行字（自动折行），返回用掉的行数
function drawChars(g, s, x, y, w, size, prefixLeft, maxLines) {
  let cx = x, used = 1, i = 0;
  for (const ch of s) {
    const cw = measure(g, ch, size);
    if (cx + cw > x + w) { if (used >= maxLines) break; used++; cx = x; }
    const isPf = i < prefixLeft;
    text(g, ch, cx, y + (used - 1) * (size + 5), { size, color: isPf ? COL.amber : ch === '，' || ch === '。' ? COL.dim : COL.ink });
    cx += cw;
    i++;
  }
  return used;
}
