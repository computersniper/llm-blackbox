// D3 V 的内部：上面一排是编码器（真实画面 → 4 层卷积 → μ / σ），右边经过 M，下面一排是解码器
// （M 预测的 ẑ → 全连接 → 4 层反卷积 → 梦见的画面），排成一个 U。每一层的特征图都是这一帧真实算出来的；
// 点任意一格选中它，按 ＋ 看它的一次乘加。走到 z 时，下面把用得最多的几维各扫一遍。
import { COL, rr, text, card, hexA, line, ease, seg } from '../../../train/js/draw.js';
import { Pix, featGrid } from '../pix.js';
import { dimOrder, OP_LABEL, fmtMSE } from '../explain.js';
import { ENC, DEC } from '../nn.js';

const LAYERS = {
  e1: { C: 16, H: 31, cols: 4 }, e2: { C: 32, H: 14, cols: 8 }, e3: { C: 64, H: 6, cols: 8 }, e4: { C: 128, H: 2, cols: 16 },
  d1: { C: 64, H: 5, cols: 8 }, d2: { C: 32, H: 13, cols: 8 }, d3: { C: 16, H: 30, cols: 4 },
};
const SWEEP = [-1, -0.66, -0.33, 0, 0.33, 0.66, 1];

export class VaeView {
  constructor(app) {
    this.app = app;
    this.inp = new Pix();
    this.out = new Pix();
    this.rec = new Pix();
    this.grids = {};
    this.key = null;
    this.sweep = { key: null, done: 0, pix: [] };
  }

  layout(env) {
    this.portrait = env.portrait;
    const B = (x, y, w, h) => ({ x, y, w, h });
    if (this.portrait) {
      // 竖屏：编码器一列往下，解码器一列往下（右边），z 和 M 在中间
      this.B = {
        in: B(0, 24, 96, 96), e1: B(0, 150, 96, 96), e2: B(0, 268, 96, 48), e3: B(0, 340, 96, 96), e4: B(0, 460, 96, 48), mu: B(0, 534, 150, 90),
        zd: B(214, 24, 150, 90), dfc: B(214, 140, 120, 60), d1: B(214, 224, 96, 96), d2: B(214, 344, 120, 60), d3: B(214, 428, 120, 120), out: B(214, 572, 120, 120),
        rec: B(0, 650, 96, 96), M: B(170, 640, 0, 0), sweep: B(0, 790, 370, 300),
      };
      this.box = { x: -8, y: 0, w: 390, h: 1100 };
    } else {
      this.B = {
        in: B(0, 34, 128, 128), e1: B(172, 18, 160, 160), e2: B(368, 56, 168, 84), e3: B(572, 26, 144, 144), e4: B(752, 59, 160, 80), mu: B(950, 20, 170, 150),
        zd: B(950, 310, 170, 140), dfc: B(760, 344, 144, 72), d1: B(588, 310, 141, 141), d2: B(386, 340, 162, 81), d3: B(196, 302, 156, 156), out: B(0, 316, 128, 128),
        rec: B(0, 196, 84, 84), M: B(990, 196, 92, 74), sweep: B(0, 500, 1120, 236),
      };
      this.box = { x: -14, y: -10, w: 1150, h: 760 };
    }
  }

  focus(st) {
    const op = st.step.op, B = this.B;
    if (this.portrait) {
      if (op === 'z') return { x: -8, y: 520, w: 390, h: 580 };
      const b = B[op === 'mu' ? 'mu' : op === 'd4' ? 'out' : op] || B.in;
      return { x: -8, y: Math.max(-10, b.y - 150), w: 390, h: 420 };
    }
    if (op === 'z') return { x: -14, y: 0, w: 1150, h: 750 };
    return this.box.h > 600 ? { x: -14, y: -10, w: 1150, h: 480 } : this.box;
  }

  ensure(F, det) {
    if (this.key === det) return;
    this.key = det;
    this.grids = {};
    for (const [id, L] of Object.entries(LAYERS)) {
      const a = id[0] === 'e' ? det.enc[id] : det.dec?.[id];
      if (a) this.grids[id] = featGrid(a, L.C, L.H, L.H, L.cols);
    }
    this.grids.dfc = featGrid(det.dec ? det.dec.dfc : new Float32Array(512), 512, 1, 1, 32, { gap: 0, signed: true });
    this.grids.e4 = featGrid(det.enc.e4, 128, 2, 2, 16, { gap: 1 });
    this.inp.palette(F.o, F.o);
    this.rec.chw(det.recon, det.recon);
    if (det.dec) this.out.chw(det.dec.y, det.dec.y);
  }

  draw(g, st, env) {
    const { sim, meta } = this.app;
    const F = sim.cur, rec = F.rec;
    if (!rec || rec.kind !== 'step') return;
    const det = sim.detail(F);
    this.ensure(F, det);
    const s = st.step, p = st.p, B = this.B, pt = this.portrait;
    const usedEnc = rec.src !== 'dream';
    const order = ['e1', 'e2', 'e3', 'e4', 'mu', 'z', 'M', 'dfc', 'd1', 'd2', 'd3', 'd4'];
    const curOp = s.ph === 'rnn' || s.ph === 'sample' ? 'M' : s.op || (s.ph === 'enc' ? 'e1' : s.ph === 'dec' ? 'dfc' : '');
    const ci = order.indexOf(curOp);
    const st8 = (op) => (op === curOp ? 'on' : order.indexOf(op) < ci ? 'done' : 'todo');
    const sel = this.app.sel;

    // ---- 编码器
    this.pixBox(g, B.in, this.inp, '真实画面 obs', '64×64×3', COL.cyan, 'done', env);
    const encOps = ['e1', 'e2', 'e3', 'e4'];
    let prev = B.in;
    for (const op of encOps) {
      const L = ENC.find((l) => l.id === op);
      this.arrow(g, prev, B[op], st8(op) === 'on' ? p : st8(op) === 'done' ? 1 : 0, COL.cyan, pt);
      this.gridBox(g, B[op], op, `${OP_LABEL[op]}`, `${L.outH}×${L.outH}×${L.cout}`, st8(op), p, env, sel, usedEnc);
      prev = B[op];
    }
    this.arrow(g, B.e4, B.mu, st8('mu') === 'on' ? p : st8('mu') === 'done' ? 1 : 0, COL.cyan, pt);
    this.muBox(g, B.mu, det.enc, st8('mu') === 'on' || st8('z') === 'on', env);
    // ---- 重建：V 自己看一眼再画回来
    const r = B.rec;
    this.recon(g, r, det, F);
    // ---- 中间：M
    if (!pt) {
      const m = B.M;
      card(g, m.x, m.y, m.w, m.h, { r: 10, active: curOp === 'M', accent: COL.violet });
      text(g, 'M', m.x + m.w / 2, m.y + 32, { size: 20, kind: 'serif', weight: 600, align: 'center', color: curOp === 'M' ? COL.ink : COL.ink2 });
      text(g, 'D4 拆开', m.x + m.w / 2, m.y + 52, { size: 10, color: COL.dim, align: 'center' });
      line(g, [[B.mu.x + B.mu.w / 2, B.mu.y + B.mu.h + 4], [m.x + m.w / 2, m.y - 4]], hexA(COL.violet, 0.5), 1.2);
      line(g, [[m.x + m.w / 2, m.y + m.h + 4], [B.zd.x + B.zd.w / 2, B.zd.y - 18]], hexA(COL.amber, 0.5), 1.2);
      text(g, rec.src === 'dream' ? '闭眼：M 的输入是上一步的 ẑ' : 'z = μ', m.x - 10, m.y + 20, { size: 10, color: COL.dim, align: 'right' });
    }
    // ---- 解码器（M 预测的 ẑ）
    this.zBox(g, B.zd, rec.S.z, 'ẑ（M 预测的下一帧）', curOp === 'dfc' || curOp === 'M', env);
    this.arrow(g, B.zd, B.dfc, st8('dfc') === 'on' ? p : st8('dfc') === 'done' ? 1 : 0, COL.amber, pt, true);
    this.gridBox(g, B.dfc, 'dfc', '全连接', '1×1×512', st8('dfc'), p, env, sel, true, true);
    prev = B.dfc;
    for (const op of ['d1', 'd2', 'd3']) {
      const L = DEC.find((l) => l.id === op);
      this.arrow(g, prev, B[op], st8(op) === 'on' ? p : st8(op) === 'done' ? 1 : 0, COL.amber, pt, true);
      this.gridBox(g, B[op], op, OP_LABEL[op], `${L.outH}×${L.outH}×${L.cout}`, st8(op), p, env, sel, true);
      prev = B[op];
    }
    this.arrow(g, B.d3, B.out, st8('d4') === 'on' ? p : st8('d4') === 'done' ? 1 : 0, COL.amber, pt, true);
    const rev = st8('d4') === 'on' ? ease(seg(p, 0.2, 1)) : st8('d4') === 'done' ? 1 : 0;
    this.pixBox(g, B.out, this.out, '梦见的下一帧 ô', '反卷积 4 + sigmoid', COL.amber, st8('d4'), env, rev, 'd4');
    // ---- z 的每一维扫一遍
    if (s.op === 'z' || s.op === 'mu') this.sweepPanel(g, B.sweep, det, env, s.op === 'z');
    else {
      const sw = B.sweep;
      text(g, '走到 z 这一步时，这里会把用得最多的几维各扫一遍', sw.x, sw.y + 18, { size: 11.5, color: COL.faint });
    }
  }

  pixBox(g, r, pix, title, sub, accent, state, env, reveal = 1, selOp = null) {
    const on = state === 'on';
    text(g, title, r.x, r.y - 10, { size: this.portrait ? 10.5 : 12, kind: 'serif', weight: 600, color: on ? COL.ink : COL.ink2 });
    if (!this.portrait) text(g, sub, r.x + r.w, r.y - 10, { size: 9, kind: 'mono', color: COL.dim, align: 'right' });
    rr(g, r.x - 3, r.y - 3, r.w + 6, r.h + 6, 6);
    g.fillStyle = 'rgba(4,8,16,0.92)'; g.fill();
    g.strokeStyle = hexA(on ? COL.amber : accent, on ? 0.85 : 0.35); g.lineWidth = on ? 1.6 : 1; g.stroke();
    if (reveal > 0) {
      g.save(); g.beginPath(); g.rect(r.x, r.y, r.w, r.h * reveal); g.clip();
      pix.draw(g, r.x, r.y, r.w, r.h);
      g.restore();
    }
    if (selOp) {
      const sl = this.app.sel[selOp];
      if (sl && reveal > 0) {
        const cs = r.w / 64;
        g.strokeStyle = COL.amber; g.lineWidth = 1.2;
        g.strokeRect(r.x + sl.x * cs - 1, r.y + sl.y * cs - 1, cs + 2, cs + 2);
      }
      env.hit(r.x, r.y, r.w, r.h, {
        tipAt: (wx, wy) => { const x = Math.floor((wx - r.x) / (r.w / 64)), y = Math.floor((wy - r.y) / (r.h / 64)); const d = this.app.sim.detail().dec; if (!d) return ''; const i = y * 64 + x; return `<span class="k">梦里的像素 (${y}, ${x})</span>R ${d.y[i].toFixed(3)} · G ${d.y[4096 + i].toFixed(3)} · B ${d.y[8192 + i].toFixed(3)}<br><span class="v">点一下选中它（绿色通道），按 ＋ 看它怎么算出来</span>`; },
        clickAt: (wx, wy) => { const x = Math.floor((wx - r.x) / (r.w / 64)), y = Math.floor((wy - r.y) / (r.h / 64)); this.app.select('d4', { c: 1, y, x }); },
      });
    }
  }

  gridBox(g, r, op, title, sub, state, p, env, sel, live = true, flat = false) {
    const on = state === 'on', G = this.grids[op];
    text(g, title, r.x, r.y - 10, { size: this.portrait ? 10.5 : 12, kind: 'serif', weight: 600, color: on ? COL.ink : live ? COL.ink2 : COL.faint });
    if (!this.portrait) text(g, sub, r.x + r.w, r.y - 10, { size: 9, kind: 'mono', color: COL.dim, align: 'right' });
    rr(g, r.x - 3, r.y - 3, r.w + 6, r.h + 6, 6);
    g.fillStyle = 'rgba(4,8,16,0.92)'; g.fill();
    g.strokeStyle = on ? hexA(COL.amber, 0.85) : COL.line2; g.lineWidth = on ? 1.6 : 1; g.stroke();
    if (!G || !live) { text(g, '闭眼：这一帧没用到', r.x + r.w / 2, r.y + r.h / 2 + 4, { size: 10, color: COL.faint, align: 'center' }); return; }
    const reveal = state === 'todo' ? 0 : on ? ease(seg(p, 0, 0.75)) : 1;
    if (reveal > 0) {
      g.save(); g.beginPath(); g.rect(r.x, r.y, r.w, r.h * reveal); g.clip();
      G.pix.draw(g, r.x, r.y, r.w, r.h, state === 'todo' ? 0.25 : 1);
      g.restore();
      if (on && reveal < 1) { g.fillStyle = hexA(COL.amber, 0.7); g.fillRect(r.x, r.y + r.h * reveal - 1, r.w, 1.5); }
    } else G.pix.draw(g, r.x, r.y, r.w, r.h, 0.12);
    // 选中的那一格
    const sx = r.w / G.cw, sy = r.h / G.ch;
    const cell = (c, y, x) => ({ x: r.x + ((c % G.cols) * (G.w + G.gap) + x) * sx, y: r.y + (Math.floor(c / G.cols) * (G.h + G.gap) + y) * sy, w: sx, h: sy });
    const sl = sel[op];
    if (sl && state !== 'todo') {
      const q = flat ? cell(sl.j, 0, 0) : cell(sl.c, sl.y, sl.x);
      g.strokeStyle = COL.amber; g.lineWidth = 1.4;
      g.strokeRect(q.x - 1, q.y - 1, q.w + 2, q.h + 2);
    }
    const pick = (wx, wy) => {
      const gx = Math.floor((wx - r.x) / sx), gy = Math.floor((wy - r.y) / sy);
      const tc = Math.floor(gx / (G.w + G.gap)), tr = Math.floor(gy / (G.h + G.gap));
      const x = gx - tc * (G.w + G.gap), y = gy - tr * (G.h + G.gap);
      if (x >= G.w || y >= G.h || tc >= G.cols) return null;
      const c = tr * G.cols + tc;
      if (c >= (flat ? 512 : LAYERS[op].C)) return null;
      return flat ? { j: c } : { c, y, x };
    };
    const val = (q) => {
      const det = this.app.sim.detail();
      const a = op[0] === 'e' ? det.enc[op] : det.dec?.[op];
      if (!a) return 0;
      if (flat) return a[q.j];
      const L = LAYERS[op];
      return a[(q.c * L.H + q.y) * L.H + q.x];
    };
    env.hit(r.x, r.y, r.w, r.h, {
      tipAt: (wx, wy) => { const q = pick(wx, wy); if (!q) return ''; return `<span class="k">${OP_LABEL[op]} · ${flat ? `第 ${q.j} 个` : `通道 ${q.c} · (${q.y}, ${q.x})`}</span>值 <span class="v">${val(q).toFixed(4)}</span><br><span class="v">点一下选中，按 ＋ 看它怎么乘加出来</span>`; },
      clickAt: (wx, wy) => { const q = pick(wx, wy); if (q) this.app.select(op, q); },
    });
  }

  arrow(g, a, b, pr, color, pt, back = false) {
    let p0, p1;
    if (pt) { p0 = [a.x + a.w / 2, a.y + a.h + 6]; p1 = [b.x + b.w / 2, b.y - 18]; }
    else if (back) { p0 = [a.x - 6, a.y + a.h / 2]; p1 = [b.x + b.w + 6, b.y + b.h / 2]; }
    else { p0 = [a.x + a.w + 6, a.y + a.h / 2]; p1 = [b.x - 6, b.y + b.h / 2]; }
    line(g, [p0, p1], hexA(color, 0.25), 1.2);
    if (pr > 0) line(g, [p0, [p0[0] + (p1[0] - p0[0]) * pr, p0[1] + (p1[1] - p0[1]) * pr]], hexA(color, 0.85), 1.6);
    const ang = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]);
    g.beginPath();
    g.moveTo(p1[0], p1[1]);
    g.lineTo(p1[0] - 6 * Math.cos(ang - 0.45), p1[1] - 6 * Math.sin(ang - 0.45));
    g.lineTo(p1[0] - 6 * Math.cos(ang + 0.45), p1[1] - 6 * Math.sin(ang + 0.45));
    g.closePath();
    g.fillStyle = hexA(color, pr >= 1 ? 0.9 : 0.35);
    g.fill();
  }

  // μ 和 σ：每一维一根柱子（μ），两边的细线是 ±σ
  muBox(g, r, enc, on, env) {
    const { meta } = this.app;
    text(g, 'μ 和 σ（32 维）', r.x, r.y - 10, { size: this.portrait ? 10.5 : 12, kind: 'serif', weight: 600, color: on ? COL.ink : COL.ink2 });
    card(g, r.x, r.y, r.w, r.h, { r: 8, active: on });
    const order = dimOrder(meta), pad = 6, bw = (r.w - pad * 2) / 32, mid = r.y + r.h / 2, R = 3.2;
    line(g, [[r.x + pad, mid], [r.x + r.w - pad, mid]], COL.line2, 1);
    order.forEach((d, i) => {
      const used = meta.dims[d].kl > 0.02;
      const x = r.x + pad + i * bw;
      const m = enc.mu[d], s = Math.exp(enc.lv[d] / 2);
      const y = (v) => mid - Math.max(-1, Math.min(1, v / R)) * (r.h / 2 - pad);
      g.fillStyle = 'rgba(179,157,255,0.12)';
      g.fillRect(x + 0.5, y(m + s), bw - 1, y(m - s) - y(m + s));
      g.fillStyle = used ? hexA(COL.violet, 0.9) : 'rgba(122,133,158,0.4)';
      g.fillRect(x + 1, Math.min(mid, y(m)), Math.max(1, bw - 2), Math.max(0.8, Math.abs(y(m) - mid)));
    });
    env.hit(r.x, r.y, r.w, r.h, { tip: '<span class="k">μ 和 σ</span>柱子是 μ，淡紫色的带子是 ±σ。没用上的维度（灰色）μ≈0、σ≈1：编码器对它们“什么都没说”。' });
  }

  zBox(g, r, z, title, on, env) {
    const { meta } = this.app;
    text(g, title, r.x, r.y - 10, { size: this.portrait ? 10.5 : 12, kind: 'serif', weight: 600, color: on ? COL.ink : COL.ink2 });
    card(g, r.x, r.y, r.w, r.h, { r: 8, active: on });
    const order = dimOrder(meta), pad = 6, bw = (r.w - pad * 2) / 32, mid = r.y + r.h / 2, R = 3.2;
    line(g, [[r.x + pad, mid], [r.x + r.w - pad, mid]], COL.line2, 1);
    order.forEach((d, i) => {
      const used = meta.dims[d].kl > 0.02;
      const v = Math.max(-1, Math.min(1, z[d] / R)) * (r.h / 2 - pad);
      g.fillStyle = used ? hexA(COL.amber, 0.85) : 'rgba(122,133,158,0.4)';
      g.fillRect(r.x + pad + i * bw + 1, Math.min(mid, mid - v), Math.max(1, bw - 2), Math.max(0.8, Math.abs(v)));
    });
    env.hit(r.x, r.y, r.w, r.h, { tip: '<span class="k">ẑ</span>M 预测、按温度采样出来的下一帧 z，解码器把它画成梦。' });
  }

  recon(g, r, det, F) {
    const pt = this.portrait;
    text(g, pt ? 'V 自己的重建' : '只看 V：μ 直接解码回去', r.x, r.y - 8, { size: 10.5, color: COL.ink2 });
    rr(g, r.x - 2, r.y - 2, r.w + 4, r.h + 4, 5);
    g.fillStyle = 'rgba(4,8,16,0.92)'; g.fill(); g.strokeStyle = COL.line2; g.lineWidth = 1; g.stroke();
    this.rec.draw(g, r.x, r.y, r.w, r.h);
    const x = new Float32Array(det.enc.x);
    let s = 0;
    for (let i = 0; i < x.length; i++) { const d = x[i] - det.recon[i]; s += d * d; }
    text(g, `重建误差 ${fmtMSE(s / x.length)}`, r.x + r.w + 10, r.y + 16, { size: 10.5, kind: 'mono', color: COL.ink2 });
    text(g, '草丛、虚线的位置', r.x + r.w + 10, r.y + 34, { size: 10, color: COL.dim });
    text(g, '这类小细节被抹平了', r.x + r.w + 10, r.y + 48, { size: 10, color: COL.dim });
  }

  // 潜变量每一维扫一遍：以这一帧的 μ 为底，只改一维（从数据里的 1% 分位到 99% 分位），其余不动，解码
  sweepPanel(g, r, det, env, active) {
    const { meta, model } = this.app;
    const dims = dimOrder(meta).filter((d) => meta.dims[d].kl > 0.02).slice(0, this.portrait ? 6 : 8);
    const key = det;
    const sw = this.sweep;
    if (sw.key !== key) { sw.key = key; sw.pix = dims.map(() => SWEEP.map(() => null)); sw.done = 0; }
    // 每帧最多解码 4 张，免得卡
    let budget = 4;
    for (let i = 0; i < dims.length && budget > 0; i++) {
      for (let j = 0; j < SWEEP.length && budget > 0; j++) {
        if (sw.pix[i][j]) continue;
        const d = dims[i], info = meta.dims[d];
        const z = det.enc.mu.slice();
        const lo = info.lo, hi = info.hi;
        z[d] = lo + (hi - lo) * (SWEEP[j] + 1) / 2;
        const y = model.decode(z).y;
        sw.pix[i][j] = new Pix().chw(y);
        budget--;
      }
    }
    const pt = this.portrait;
    text(g, '潜空间每一维的含义：只改这一维，其余不动，解码出来的画面', r.x, r.y + 2, { size: pt ? 11 : 13, kind: 'serif', weight: 600, color: active ? COL.ink : COL.ink2 });
    const cols = pt ? 1 : 2, rows = Math.ceil(dims.length / cols);
    const cw = r.w / cols, rh = (r.h - 18) / rows;
    const th = Math.min(rh - 8, pt ? 34 : 44);
    dims.forEach((d, i) => {
      const cx = r.x + (i % cols) * cw, cy = r.y + 16 + Math.floor(i / cols) * rh;
      const info = meta.dims[d];
      const lw = pt ? 108 : 150;
      text(g, `z${d}`, cx, cy + th / 2 - 3, { size: 11, kind: 'mono', color: COL.amber });
      text(g, meta.feats[info.feat] || '', cx + 28, cy + th / 2 - 3, { size: pt ? 9.5 : 10.5, color: COL.ink2, max: lw - 28 });
      text(g, `相关 ${info.corr.toFixed(2)} · KL ${info.kl.toFixed(2)}`, cx + 28, cy + th / 2 + 11, { size: 9, kind: 'mono', color: COL.dim });
      SWEEP.forEach((v, j) => {
        const x = cx + lw + j * (th + 4);
        rr(g, x - 1, cy - 1, th + 2, th + 2, 3); g.fillStyle = 'rgba(4,8,16,0.9)'; g.fill();
        const px = sw.pix[i]?.[j];
        if (px) px.draw(g, x, cy, th, th);
      });
      env.hit(cx, cy, cw - 10, th, { tip: `<span class="k">z<sub>${d}</sub> 扫一遍</span>从左到右：${info.lo.toFixed(2)} → ${info.hi.toFixed(2)}（数据里 1% 到 99% 的范围），其余 31 维固定为这一帧的 μ。<br>和「${meta.feats[info.feat]}」的相关系数 <span class="v">${info.corr.toFixed(2)}</span>` });
    });
  }
}
