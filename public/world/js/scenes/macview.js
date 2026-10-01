// D5 一次乘加的显微镜：左边是这个数在哪（输入里参与计算的那些位置、输出里的那一格），
// 中间三块方格是全部 n 项的输入 x、权重 w、乘积 x·w（颜色：琥珀 = 正，蓝 = 负），绝对值最大的 12 项描了边。
// 具体数字在下方的算式板上（DOM，和推理页的算式板同一套样式）。
import { COL, rr, text, hexA, ease, clamp } from '../../../train/js/draw.js';
import { Pix, featGrid, divRGB } from '../pix.js';
import { ENC, DEC } from '../nn.js';
import { OP_LABEL } from '../explain.js';

const SHAPE = {
  e1: [3, 64, 3], e2: [16, 31, 4], e3: [32, 14, 8], e4: [64, 6, 8],
  d1: [512, 1, 32], d2: [64, 5, 8], d3: [32, 13, 8], d4: [16, 30, 4],
};
const OUT = { e1: [16, 31, 4], e2: [32, 14, 8], e3: [64, 6, 8], e4: [128, 2, 16], d1: [64, 5, 8], d2: [32, 13, 8], d3: [16, 30, 4] };

export class MacView {
  constructor(app) {
    this.app = app;
    this.key = null;
    this.img = new Pix();
  }

  layout(env) {
    this.portrait = env.portrait;
    const B = (x, y, w, h) => ({ x, y, w, h });
    if (this.portrait) {
      this.B = { inp: B(0, 26, 170, 170), out: B(196, 26, 170, 170), x: B(0, 236, 112, 112), w: B(127, 236, 112, 112), p: B(254, 236, 112, 112), res: B(0, 372, 366, 60) };
      this.box = { x: -8, y: 0, w: 384, h: 440 };
    } else {
      this.B = { inp: B(0, 30, 220, 220), out: B(250, 30, 220, 220), x: B(520, 30, 170, 170), w: B(710, 30, 170, 170), p: B(900, 30, 170, 170), res: B(520, 222, 550, 50) };
      this.box = { x: -14, y: -6, w: 1100, h: 300 };
    }
  }

  focus() { return this.box; }

  // 输入 / 输出张量的网格图（按这一帧缓存）
  grids(mac, det) {
    const op = mac.op;
    const k = `${op}`;
    if (this.key === det && this.gk === k) return this.G;
    this.key = det; this.gk = k;
    const G = {};
    if (SHAPE[op]) {
      const [C, H, cols] = SHAPE[op];
      const inp = op === 'e1' ? null : op[0] === 'e' ? det.enc[`e${Number(op[1]) - 1}`] : op === 'd1' ? det.dec.dfc : det.dec[`d${Number(op[1]) - 1}`];
      if (op === 'e1') { this.img.chw(det.enc.x); G.inp = { pix: this.img, cols: 1, w: 64, h: 64, gap: 0, cw: 64, ch: 64 }; }
      else G.inp = featGrid(inp, C, H, H, cols, { gap: H === 1 ? 0 : 1, signed: H === 1 });
    }
    if (OUT[op]) { const [C, H, cols] = OUT[op]; G.out = featGrid(op[0] === 'e' ? det.enc[op] : det.dec[op], C, H, H, cols); }
    if (op === 'd4') { const p = new Pix(); p.chw(det.dec.y); G.out = { pix: p, cols: 1, w: 64, h: 64, gap: 0, cw: 64, ch: 64 }; }
    this.G = G;
    return G;
  }

  draw(g, st, env) {
    const mac = this.app.mac;
    if (!mac) return;
    const { sim } = this.app;
    const det = sim.detail();
    const s = st.step, p = st.p, B = this.B, pt = this.portrait;
    const k = ['pick', 'mul', 'sum', 'act'].indexOf(s.mi);
    const G = this.grids(mac, det);
    // ---- 输入、输出在哪
    this.tensor(g, B.inp, G.inp, '输入', mac, 'in', env);
    this.tensor(g, B.out, G.out, '输出', mac, 'out', env);
    // ---- n 项：x、w、x·w
    const n = mac.n, cols = Math.ceil(Math.sqrt(n)), rows = Math.ceil(n / cols);
    let mx = 0, mw = 0, mp = 0;
    for (const t of mac.terms) { mx = Math.max(mx, Math.abs(t.x)); mw = Math.max(mw, Math.abs(t.w)); mp = Math.max(mp, Math.abs(t.p)); }
    const top = new Set(mac.top);
    const reveal = k <= 0 ? 0 : k === 1 ? ease(clamp(p * 1.1, 0, 1)) : 1;
    const draw3 = (r, key, scale, title, color, rv) => {
      text(g, title, r.x, r.y - 9, { size: pt ? 10 : 12, kind: 'serif', weight: 600, color });
      rr(g, r.x - 2, r.y - 2, r.w + 4, r.h + 4, 5); g.fillStyle = 'rgba(4,8,16,0.92)'; g.fill(); g.strokeStyle = COL.line2; g.lineWidth = 1; g.stroke();
      const cs = Math.min(r.w / cols, r.h / rows);
      const lim = Math.ceil(n * rv);
      for (let i = 0; i < n; i++) {
        const t = mac.terms[i];
        const x = r.x + (i % cols) * cs, y = r.y + Math.floor(i / cols) * cs;
        if (i >= lim) { g.fillStyle = 'rgba(255,255,255,0.03)'; g.fillRect(x + 0.5, y + 0.5, cs - 1, cs - 1); continue; }
        const c = divRGB(t[key] / (scale || 1));
        g.fillStyle = `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
        g.fillRect(x + 0.5, y + 0.5, Math.max(0.6, cs - 1), Math.max(0.6, cs - 1));
        if (top.has(i) && k >= 1 && cs > 2.5) { g.strokeStyle = hexA(COL.amber, 0.9); g.lineWidth = 1; g.strokeRect(x, y, cs, cs); }
      }
      env.hit(r.x, r.y, r.w, r.h, { tipAt: (wx, wy) => { const i = Math.floor((wy - r.y) / cs) * cols + Math.floor((wx - r.x) / cs); const t = mac.terms[i]; if (!t) return ''; return `<span class="k">第 ${i} 项 · ${t.lab}</span>x = <span class="v">${t.x.toFixed(4)}</span><br>w = <span class="v">${t.w.toFixed(4)}</span><br>x·w = <span class="v">${t.p.toFixed(5)}</span>`; } });
    };
    draw3(B.x, 'x', mx, `输入 x（${n} 个）`, COL.ink, 1);
    draw3(B.w, 'w', mw, `权重 w（${n} 个）`, COL.ink, 1);
    draw3(B.p, 'p', mp, '乘积 x·w', k >= 1 ? COL.ink : COL.faint, reveal);
    if (!pt) {
      text(g, '×', (B.x.x + B.x.w + B.w.x) / 2, B.x.y + B.x.h / 2 + 6, { size: 18, color: COL.dim, align: 'center' });
      text(g, '=', (B.w.x + B.w.w + B.p.x) / 2, B.x.y + B.x.h / 2 + 6, { size: 18, color: COL.dim, align: 'center' });
    }
    // ---- 结果
    const r = B.res;
    rr(g, r.x, r.y, r.w, r.h, 10);
    g.fillStyle = k >= 2 ? 'rgba(94,240,212,0.06)' : 'rgba(10,17,31,0.82)'; g.fill();
    g.strokeStyle = k >= 2 ? hexA(COL.cyan, 0.5) : COL.line2; g.lineWidth = 1; g.stroke();
    const sumTxt = k >= 2 ? `b + Σ x·w = ${mac.bias.toFixed(4)} + ${mac.sum.toFixed(4)} = ${mac.pre.toFixed(4)}` : k === 1 ? `正 ${mac.pos.toFixed(3)} · 负 ${mac.neg.toFixed(3)}` : `y = b + Σ x·w，${n} 项`;
    text(g, sumTxt, r.x + 14, r.y + (pt ? 24 : 22), { size: pt ? 10.5 : 13, kind: 'mono', color: COL.ink, max: r.w - 28 });
    if (k >= 3) text(g, mac.actFormula().replace(/<[^>]+>/g, ''), r.x + 14, r.y + (pt ? 44 : 40), { size: pt ? 10 : 12, kind: 'mono', color: COL.cyan, max: r.w - 28 });
    else text(g, mac.where, r.x + 14, r.y + (pt ? 44 : 40), { size: pt ? 9.5 : 11, color: COL.dim, max: r.w - 28 });
  }

  // 张量网格：输入时标出参与这一次乘加的位置，输出时标出这一格
  tensor(g, r, Gr, title, mac, which, env) {
    const pt = this.portrait;
    const label = which === 'in' ? `${title}${mac.op === 'e1' ? '：真实画面' : ''}` : `${title}：${mac.op === 'd4' ? '梦见的画面' : OP_LABEL[mac.op]}`;
    text(g, label, r.x, r.y - 9, { size: pt ? 10 : 12, kind: 'serif', weight: 600, color: COL.ink2 });
    rr(g, r.x - 2, r.y - 2, r.w + 4, r.h + 4, 5); g.fillStyle = 'rgba(4,8,16,0.92)'; g.fill(); g.strokeStyle = COL.line2; g.lineWidth = 1; g.stroke();
    if (!Gr) {
      // 向量（全连接、LSTM 的门、MDN）：画成方格
      const n = which === 'in' ? mac.n : 1;
      text(g, which === 'in' ? `${mac.n} 个输入（右边 x 方格里就是全部）` : mac.where, r.x + r.w / 2, r.y + r.h / 2, { size: 10.5, color: COL.dim, align: 'center', max: r.w - 16 });
      return;
    }
    const sc = Math.min(r.w / Gr.cw, r.h / Gr.ch);
    const w = Gr.cw * sc, h = Gr.ch * sc, ox = r.x + (r.w - w) / 2, oy = r.y + (r.h - h) / 2;
    Gr.pix.draw(g, ox, oy, w, h, 0.9);
    const L = [...ENC, ...DEC].find((l) => l.id === mac.op);
    const sel = mac.sel;
    const cell = (c, y, x) => [ox + ((c % Gr.cols) * (Gr.w + Gr.gap) + x) * sc, oy + (Math.floor(c / Gr.cols) * (Gr.h + Gr.gap) + y) * sc];
    g.strokeStyle = COL.amber; g.lineWidth = 1.2;
    if (which === 'out') {
      const [x, y] = cell(mac.op === 'd4' ? 0 : sel.c, sel.y, sel.x);
      g.strokeRect(x - 1, y - 1, sc + 2, sc + 2);
      return;
    }
    if (!L) return;
    // 输入里参与计算的位置（每个输入通道都画；通道太多时只画前 64 个）
    const seen = new Set();
    for (const t of mac.terms) {
      const m = /^c?(\d+)·(\d+),(\d+)$/.exec(t.lab);
      let c, y, x;
      if (mac.op === 'e1') { c = 0; const q = /^[RGB](\d+),(\d+)$/.exec(t.lab); y = sel.y * 2 + Number(q[1]); x = sel.x * 2 + Number(q[2]); }
      else if (L.inH === 1) { c = Number(t.lab.slice(1)); y = 0; x = 0; }
      else if (m) {
        c = Number(m[1]);
        if (mac.op[0] === 'e') { y = sel.y * L.s + Number(m[2]); x = sel.x * L.s + Number(m[3]); } else { y = Number(m[2]); x = Number(m[3]); }
      } else continue;
      const key = `${c},${y},${x}`;
      if (seen.has(key) || c >= 64 && L.inH > 1) continue;
      seen.add(key);
      const [px, py] = cell(c, y, x);
      g.strokeRect(px - 0.5, py - 0.5, sc + 1, sc + 1);
    }
  }
}
