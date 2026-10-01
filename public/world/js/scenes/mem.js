// D4 M 的内部：LSTM 的一步（拼输入 → 四个门 → 细胞状态 → 隐状态）和混合密度输出（32 维 × 5 个高斯）、撞车概率、采样。
// 256 维的向量都排成 16×16 的方格；右边 32 个小图是每一维下一帧 z 的概率密度，下面放大的是选中的那一维。
import { COL, rr, text, card, hexA, line, ease, clamp } from '../../../train/js/draw.js';
import { vecGrid } from '../pix.js';
import { ACTIONS } from '../game.js';
import { dimOrder, dimMeaning, used as isUsed } from '../explain.js';
import { Z, KMIX } from '../nn.js';

const GATES = [['gi', 'i', '输入门', 'σ'], ['gf', 'f', '遗忘门', 'σ'], ['gg', 'g', '候选记忆', 'tanh'], ['go', 'o', '输出门', 'σ']];
const KCOL = ['#5ef0d4', '#ffb65c', '#b39dff', '#ff6b93', '#6b9bff'];

export class MemView {
  constructor(app) {
    this.app = app;
    this.key = null;
  }

  layout(env) {
    this.portrait = env.portrait;
    const B = (x, y, w, h) => ({ x, y, w, h });
    if (this.portrait) {
      this.B = {
        z: B(0, 24, 250, 50), a: B(266, 24, 104, 50), h: B(0, 112, 84, 84), c: B(96, 112, 84, 84),
        gi: B(0, 236, 84, 84), gf: B(96, 236, 84, 84), gg: B(192, 236, 84, 84), go: B(288, 236, 84, 84),
        fc: B(0, 360, 84, 84), ig: B(96, 360, 84, 84), c2: B(192, 360, 84, 84), h2: B(288, 360, 84, 84),
        mdn: B(0, 488, 372, 230), big: B(0, 752, 372, 170), done: B(196, 112, 176, 60),
      };
      this.box = { x: -8, y: 0, w: 390, h: 940 };
    } else {
      this.B = {
        z: B(0, 30, 190, 64), a: B(0, 126, 190, 30), h: B(0, 196, 124, 124), c: B(0, 360, 124, 124),
        gi: B(236, 30, 124, 124), gf: B(390, 30, 124, 124), gg: B(544, 30, 124, 124), go: B(698, 30, 124, 124),
        fc: B(236, 220, 124, 124), ig: B(390, 220, 124, 124), c2: B(544, 220, 124, 124), h2: B(698, 220, 124, 124),
        mdn: B(870, 30, 300, 330), big: B(236, 400, 586, 190), done: B(870, 400, 300, 80),
      };
      this.box = { x: -14, y: -10, w: 1200, h: 620 };
    }
  }

  focus(st) {
    if (!this.portrait) return this.box;
    const op = st.step.op;
    const y = op === 'cat' ? 0 : ['gi', 'gf', 'gg', 'go'].includes(op) ? 90 : op === 'cell' || op === 'hid' ? 200 : op === 'mdn' ? 440 : op === 'done' ? 60 : 480;
    return { x: -8, y, w: 390, h: 470 };
  }

  ensure(F, rec) {
    if (this.key === rec) return;
    this.key = rec;
    const L = rec.L;
    const fc = new Float32Array(256), ig = new Float32Array(256), tc = new Float32Array(256);
    for (let j = 0; j < 256; j++) { fc[j] = L.f[j] * F.c[j]; ig[j] = L.i[j] * L.g[j]; tc[j] = Math.tanh(L.c[j]); }
    let cmax = 0;
    for (let j = 0; j < 256; j++) cmax = Math.max(cmax, Math.abs(F.c[j]), Math.abs(L.c[j]));
    cmax = Math.max(1, cmax);
    this.v = {
      h: vecGrid(F.h, 16), c: vecGrid(F.c, 16, { scale: cmax }),
      gi: vecGrid(L.i, 16, { signed: false }), gf: vecGrid(L.f, 16, { signed: false }), gg: vecGrid(L.g, 16), go: vecGrid(L.o, 16, { signed: false }),
      fc: vecGrid(fc, 16, { scale: cmax }), ig: vecGrid(ig, 16), c2: vecGrid(L.c, 16, { scale: cmax }), h2: vecGrid(L.h, 16),
    };
    this.raw = { h: F.h, c: F.c, gi: L.i, gf: L.f, gg: L.g, go: L.o, fc, ig, c2: L.c, h2: L.h };
  }

  draw(g, st, env) {
    const { sim, meta } = this.app;
    const F = sim.cur, rec = F.rec;
    if (!rec || rec.kind !== 'step') return;
    this.ensure(F, rec);
    const s = st.step, p = st.p, B = this.B, pt = this.portrait;
    const op = s.ph === 'sample' ? (s.op || 'pick') : s.op || 'cat';
    const seq = ['cat', 'gi', 'gf', 'gg', 'go', 'cell', 'hid', 'mdn', 'done', 'pick', 'draw'];
    const ci = seq.indexOf(op);
    const stOf = (o) => (o === op ? 'on' : seq.indexOf(o) < ci ? 'done' : 'todo');
    const sel = this.app.sel;

    // ---- 输入：z、动作、上一步的记忆
    this.zRow(g, B.z, rec.z, rec.src === 'dream' ? 'z（闭眼：上一步的 ẑ）' : 'z（这一帧的编码）', stOf('cat') === 'on', env);
    this.actRow(g, B.a, rec.a, stOf('cat') === 'on');
    this.grid(g, B.h, 'h', '记忆 h', 'tanh 范围 −1…1', 'done', env);
    this.grid(g, B.c, 'c', '细胞 c', '长期记忆', 'done', env);
    // ---- 四个门
    for (const [id, , name, act] of GATES) {
      const state = stOf(id);
      this.grid(g, B[id], id, `${name} ${id[1]}`, `${act}(W·[x, h] + b)`, state, env, state === 'on' ? ease(p) : 1, sel[id]?.j);
      // 输入连线
      if (state === 'on') {
        for (const src of [B.z, B.h]) line(g, [[src.x + src.w + 4, src.y + src.h / 2], [B[id].x - 4, B[id].y + B[id].h / 2]], hexA(COL.violet, 0.35 + 0.4 * Math.sin(env.t * 6) ** 2), 1);
      }
    }
    // ---- 细胞状态、隐状态
    const cs = stOf('cell'), hs = stOf('hid');
    this.grid(g, B.fc, 'fc', 'f ⊙ c', '留下的旧记忆', cs, env, cs === 'on' ? ease(p) : 1);
    this.grid(g, B.ig, 'ig', 'i ⊙ g', '写进去的新内容', cs, env, cs === 'on' ? ease(p) : 1);
    this.grid(g, B.c2, 'c2', "c' = f⊙c + i⊙g", '新的细胞状态', cs, env, cs === 'on' ? ease(clamp(p * 1.5 - 0.5, 0, 1)) : 1);
    this.grid(g, B.h2, 'h2', "h' = o ⊙ tanh(c')", '新的记忆', hs, env, hs === 'on' ? ease(p) : 1);
    if (!pt) {
      const y = B.fc.y + B.fc.h / 2;
      text(g, '+', (B.fc.x + B.fc.w + B.ig.x) / 2, y + 6, { size: 18, color: COL.dim, align: 'center' });
      text(g, '=', (B.ig.x + B.ig.w + B.c2.x) / 2, y + 6, { size: 18, color: COL.dim, align: 'center' });
      text(g, '→', (B.c2.x + B.c2.w + B.h2.x) / 2, y + 6, { size: 16, color: COL.dim, align: 'center' });
    }
    // ---- 混合密度
    this.mdn(g, B.mdn, rec, stOf('mdn') !== 'todo', op === 'mdn' || op === 'pick' || op === 'draw', env);
    this.big(g, B.big, rec, op, p, env);
    this.doneGauge(g, B.done, rec, stOf('done'), env);
  }

  zRow(g, r, z, title, on, env) {
    const { meta } = this.app;
    text(g, title, r.x, r.y - 9, { size: this.portrait ? 10 : 11.5, kind: 'serif', weight: 600, color: on ? COL.ink : COL.ink2 });
    card(g, r.x, r.y, r.w, r.h, { r: 6, active: on });
    const order = dimOrder(meta), pad = 4, bw = (r.w - pad * 2) / 32, mid = r.y + r.h / 2;
    order.forEach((d, i) => {
      const v = clamp(z[d] / 3.2, -1, 1) * (r.h / 2 - pad);
      g.fillStyle = isUsed(meta, d) ? hexA(COL.violet, 0.85) : 'rgba(122,133,158,0.4)';
      g.fillRect(r.x + pad + i * bw + 0.5, Math.min(mid, mid - v), Math.max(1, bw - 1), Math.max(0.8, Math.abs(v)));
    });
    env.hit(r.x, r.y, r.w, r.h, { tip: '<span class="k">输入 x 的前 32 个数</span>这一帧的 z（按 KL 排序显示）' });
  }

  actRow(g, r, a, on) {
    text(g, '动作 one-hot', r.x, r.y - 9, { size: this.portrait ? 10 : 11.5, kind: 'serif', weight: 600, color: on ? COL.ink : COL.ink2 });
    const w = r.w / 3;
    for (let k = 0; k < 3; k++) {
      rr(g, r.x + k * w + 2, r.y, w - 4, r.h, 5);
      g.fillStyle = k === a ? hexA(COL.amber, 0.25) : 'rgba(255,255,255,0.03)'; g.fill();
      g.strokeStyle = k === a ? hexA(COL.amber, 0.8) : COL.line2; g.lineWidth = 1; g.stroke();
      text(g, `${ACTIONS[k]} ${k === a ? 1 : 0}`, r.x + k * w + w / 2, r.y + r.h / 2 + 4, { size: this.portrait ? 9.5 : 11, color: k === a ? COL.amber : COL.dim, align: 'center' });
    }
  }

  grid(g, r, id, title, sub, state, env, alpha = 1, selJ = null) {
    const on = state === 'on';
    text(g, title, r.x, r.y - 9, { size: this.portrait ? 9.5 : 11.5, kind: 'serif', weight: 600, color: on ? COL.ink : state === 'todo' ? COL.faint : COL.ink2, max: r.w + 30 });
    rr(g, r.x - 2, r.y - 2, r.w + 4, r.h + 4, 5);
    g.fillStyle = 'rgba(4,8,16,0.92)'; g.fill();
    g.strokeStyle = on ? hexA(COL.amber, 0.85) : COL.line2; g.lineWidth = on ? 1.5 : 1; g.stroke();
    this.v[id].draw(g, r.x, r.y, r.w, r.h, state === 'todo' ? 0.15 : alpha);
    if (!this.portrait) text(g, sub, r.x, r.y + r.h + 14, { size: 9.5, color: COL.dim, max: r.w + 20 });
    const cs = r.w / 16;
    if (selJ != null && state !== 'todo') {
      g.strokeStyle = COL.amber; g.lineWidth = 1.3;
      g.strokeRect(r.x + (selJ % 16) * cs - 1, r.y + Math.floor(selJ / 16) * cs - 1, cs + 2, cs + 2);
    }
    const isGate = id[0] === 'g' && id.length === 2;
    env.hit(r.x, r.y, r.w, r.h, {
      tipAt: (wx, wy) => { const j = clamp(Math.floor((wy - r.y) / cs), 0, 15) * 16 + clamp(Math.floor((wx - r.x) / cs), 0, 15); return `<span class="k">${title} · 第 ${j} 维</span><span class="v">${this.raw[id][j].toFixed(4)}</span>${isGate ? '<br><span class="v">点一下选中，按 ＋ 看它的 291 项乘加</span>' : ''}`; },
      clickAt: isGate ? (wx, wy) => { const j = clamp(Math.floor((wy - r.y) / cs), 0, 15) * 16 + clamp(Math.floor((wx - r.x) / cs), 0, 15); this.app.select(id, { j }); } : null,
    });
  }

  // 32 个小图：每一维下一帧 z 的混合密度（实线 τ=1，琥珀色是按当前温度调过的），竖线是采样出来的值
  mdn(g, r, rec, shown, on, env) {
    const { meta } = this.app;
    const pt = this.portrait;
    text(g, '下一帧 z 的分布（每一维 5 个高斯）', r.x, r.y - 9, { size: pt ? 10 : 11.5, kind: 'serif', weight: 600, color: on ? COL.ink : COL.ink2 });
    card(g, r.x, r.y, r.w, r.h, { r: 8, active: on });
    if (!shown) { text(g, '算到“混合密度”这一步才有', r.x + r.w / 2, r.y + r.h / 2, { size: 11, color: COL.faint, align: 'center' }); return; }
    const order = dimOrder(meta);
    const cols = 4, rows = 8, pad = 6;
    const cw = (r.w - pad * 2) / cols, ch = (r.h - pad * 2) / rows;
    const selD = this.selDim(rec);
    order.forEach((d, i) => {
      const x = r.x + pad + (i % cols) * cw, y = r.y + pad + Math.floor(i / cols) * ch;
      const used = isUsed(meta, d);
      if (d === selD) { rr(g, x + 1, y + 1, cw - 2, ch - 2, 3); g.strokeStyle = hexA(COL.amber, 0.8); g.lineWidth = 1; g.stroke(); }
      this.density(g, { x: x + 3, y: y + 3, w: cw - 6, h: ch - 8 }, rec, d, used ? 1 : 0.35, false);
      text(g, `${d}`, x + 4, y + 10, { size: 7.5, kind: 'mono', color: COL.faint });
      env.hit(x, y, cw, ch, { tip: `<span class="k">z<sub>${d}</sub> 的下一帧分布</span>${dimMeaning(meta, d)}<br>采样到 <span class="v">${rec.S.z[d].toFixed(3)}</span>（第 ${rec.S.k[d]} 个高斯）<br><span class="v">点一下放大</span>`, click: true, act: () => { this.app.sel.mdnDim = d; } });
    });
  }

  selDim(rec) {
    const { meta } = this.app;
    if (this.app.sel.mdnDim != null) return this.app.sel.mdnDim;
    // 默认挑用得上的维度里分布最“散”的（模型最拿不准的）
    let best = dimOrder(meta)[0], bv = -1;
    for (const d of dimOrder(meta)) {
      if (!isUsed(meta, d)) continue;
      let m = 0, v = 0;
      for (let k = 0; k < KMIX; k++) m += Math.exp(rec.M.logpi[d * KMIX + k]) * rec.M.mu[d * KMIX + k];
      for (let k = 0; k < KMIX; k++) { const pk = Math.exp(rec.M.logpi[d * KMIX + k]); v += pk * ((rec.M.mu[d * KMIX + k] - m) ** 2 + Math.exp(2 * rec.M.logsig[d * KMIX + k])); }
      if (v > bv) { bv = v; best = d; }
    }
    return best;
  }

  range(rec, d) {
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < KMIX; k++) {
      if (Math.exp(rec.M.logpi[d * KMIX + k]) < 0.01) continue;
      const m = rec.M.mu[d * KMIX + k], s = Math.exp(rec.M.logsig[d * KMIX + k]);
      lo = Math.min(lo, m - 3.2 * s); hi = Math.max(hi, m + 3.2 * s);
    }
    lo = Math.min(lo, rec.S.z[d] - 0.3); hi = Math.max(hi, rec.S.z[d] + 0.3);
    return [lo, hi];
  }

  // 一维的混合密度曲线；big = true 时画出每个分量
  density(g, r, rec, d, alpha, big, opts = {}) {
    const M = rec.M, tau = rec.tau;
    const [lo, hi] = opts.range || this.range(rec, d);
    const n = big ? 160 : 48;
    const comp = (k, x, t) => {
      const s = Math.exp(M.logsig[d * KMIX + k]) * Math.sqrt(t);
      return Math.exp(-0.5 * ((x - M.mu[d * KMIX + k]) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
    };
    const pis = [], pist = [];
    let sum = 0;
    for (let k = 0; k < KMIX; k++) { pis.push(Math.exp(M.logpi[d * KMIX + k])); pist.push(rec.S.pi[d * KMIX + k]); sum += pist[k]; }
    const xs = Array.from({ length: n }, (_, i) => lo + (hi - lo) * i / (n - 1));
    const p1 = xs.map((x) => pis.reduce((s, p, k) => s + p * comp(k, x, 1), 0));
    const pt = xs.map((x) => pist.reduce((s, p, k) => s + p * comp(k, x, tau), 0));
    const mx = Math.max(...p1, ...pt) || 1;
    const X = (x) => r.x + (x - lo) / (hi - lo) * r.w, Y = (v) => r.y + r.h - (v / mx) * r.h;
    g.save();
    g.globalAlpha *= alpha;
    if (big) {
      for (let k = 0; k < KMIX; k++) {
        if (pis[k] < 0.005) continue;
        line(g, xs.map((x) => [X(x), Y(pis[k] * comp(k, x, 1))]), hexA(KCOL[k], opts.pickK === k ? 0.95 : 0.4), opts.pickK === k ? 1.6 : 1);
      }
    }
    // 实线：模型给的分布（τ=1）；琥珀色：按温度调过、真正拿来采样的
    g.beginPath();
    g.moveTo(X(lo), r.y + r.h);
    xs.forEach((x, i) => g.lineTo(X(x), Y(p1[i])));
    g.lineTo(X(hi), r.y + r.h);
    g.fillStyle = 'rgba(179,157,255,0.12)';
    g.fill();
    line(g, xs.map((x, i) => [X(x), Y(p1[i])]), hexA(COL.violet, 0.9), big ? 1.8 : 1);
    if (Math.abs(tau - 1) > 0.01) line(g, xs.map((x, i) => [X(x), Y(pt[i])]), hexA(COL.amber, 0.85), big ? 1.5 : 0.9);
    if (opts.showSample !== false) {
      const zx = X(rec.S.z[d]);
      g.strokeStyle = COL.amber; g.lineWidth = big ? 2 : 1;
      g.beginPath(); g.moveTo(zx, r.y); g.lineTo(zx, r.y + r.h); g.stroke();
    }
    if (opts.truth != null) {
      const tx = X(opts.truth);
      g.strokeStyle = 'rgba(233,239,249,0.9)'; g.lineWidth = 1.5; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(tx, r.y); g.lineTo(tx, r.y + r.h); g.stroke(); g.setLineDash([]);
    }
    g.restore();
    return { X, lo, hi };
  }

  big(g, r, rec, op, p, env) {
    const { meta, sim } = this.app;
    const pt = this.portrait;
    const d = this.selDim(rec);
    const on = op === 'mdn' || op === 'pick' || op === 'draw';
    const shown = ['mdn', 'done', 'pick', 'draw'].includes(op);
    text(g, `放大：z${d}（${dimMeaning(meta, d)}）`, r.x, r.y - 9, { size: pt ? 10 : 11.5, kind: 'serif', weight: 600, color: on ? COL.ink : COL.ink2 });
    card(g, r.x, r.y, r.w, r.h, { r: 8, active: on });
    if (!shown) { text(g, '算到“混合密度”这一步才有', r.x + r.w / 2, r.y + r.h / 2, { size: 11, color: COL.faint, align: 'center' }); return; }
    // 睁眼时，下一帧真实画面的编码作参照（白色虚线）
    const G = sim.next;
    const truth = rec.mode === 'open' && G ? sim.encOf(G).mu[d] : null;
    const P = { x: r.x + 14, y: r.y + 14, w: r.w - (pt ? 28 : 200), h: r.h - 40 };
    const showSample = op === 'draw' ? p > 0.4 : op !== 'mdn' && op !== 'done' && op !== 'pick' ? true : false;
    const k = rec.S.k[d];
    const geo = this.density(g, P, rec, d, 1, true, { pickK: op === 'pick' || op === 'draw' ? k : null, truth, showSample, range: truth != null ? (() => { const [a, b] = this.range(rec, d); return [Math.min(a, truth - 0.3), Math.max(b, truth + 0.3)]; })() : null });
    line(g, [[P.x, P.y + P.h], [P.x + P.w, P.y + P.h]], COL.line2, 1);
    text(g, geo.lo.toFixed(2), P.x, P.y + P.h + 13, { size: 9, kind: 'mono', color: COL.dim });
    text(g, geo.hi.toFixed(2), P.x + P.w, P.y + P.h + 13, { size: 9, kind: 'mono', color: COL.dim, align: 'right' });
    if (op === 'draw' && p > 0.4) text(g, `ẑ = ${rec.S.z[d].toFixed(3)}`, geo.X(rec.S.z[d]) + 6, P.y + 12, { size: 11, kind: 'mono', color: COL.amber });
    if (truth != null) text(g, '真实下一帧', geo.X(truth) + 5, P.y + 26, { size: 9.5, color: COL.ink2 });
    // 右侧：5 个分量的 π、μ、σ
    if (!pt) {
      const tx = r.x + r.w - 176;
      text(g, '分量   π      π(τ)   μ       σ', tx, r.y + 22, { size: 9.5, kind: 'mono', color: COL.dim });
      for (let kk = 0; kk < KMIX; kk++) {
        const pi = Math.exp(rec.M.logpi[d * KMIX + kk]), pit = rec.S.pi[d * KMIX + kk];
        const yy = r.y + 40 + kk * 17;
        const hl = (op === 'pick' || op === 'draw') && kk === k;
        if (hl) { rr(g, tx - 4, yy - 11, 172, 15, 3); g.fillStyle = hexA(COL.amber, 0.15); g.fill(); }
        g.fillStyle = KCOL[kk]; g.fillRect(tx, yy - 7, 8, 8);
        text(g, `${(pi * 100).toFixed(1).padStart(5)}% ${(pit * 100).toFixed(1).padStart(5)}% ${rec.M.mu[d * KMIX + kk].toFixed(2).padStart(6)} ${Math.exp(rec.M.logsig[d * KMIX + kk]).toFixed(3)}`, tx + 14, yy, { size: 9.5, kind: 'mono', color: hl ? COL.amber : COL.ink2 });
      }
      text(g, `τ = ${rec.tau.toFixed(2)} · 挑中第 ${k} 个`, tx, r.y + 140, { size: 10, color: COL.dim });
      text(g, `ε = ${rec.S.eps[d].toFixed(3)}`, tx, r.y + 156, { size: 10, kind: 'mono', color: COL.dim });
    }
    env.hit(r.x, r.y, r.w, r.h, { tip: `<span class="k">z<sub>${d}</sub> 的分布</span>紫色：M 给出的 5 个高斯的混合；琥珀色：按温度 τ 调过之后真正拿来采样的；琥珀竖线：采样结果${truth != null ? '；白色虚线：真实下一帧编码出来的值' : ''}。` });
  }

  doneGauge(g, r, rec, state, env) {
    const on = state === 'on';
    const pd = rec.M.done;
    card(g, r.x, r.y, r.w, r.h, { r: 8, active: on, accent: COL.rose });
    text(g, '这一步撞车了吗', r.x + 12, r.y + 20, { size: this.portrait ? 10.5 : 12, kind: 'serif', weight: 600, color: on ? COL.ink : COL.ink2 });
    const shown = state !== 'todo';
    text(g, shown ? `${(pd * 100).toFixed(pd < 0.1 ? 2 : 1)}%` : '—', r.x + r.w - 12, r.y + 21, { size: 14, kind: 'mono', color: pd > 0.5 ? COL.rose : COL.ink, align: 'right' });
    const w = r.w - 24;
    rr(g, r.x + 12, r.y + 32, w, 6, 3); g.fillStyle = 'rgba(255,255,255,0.06)'; g.fill();
    if (shown) { rr(g, r.x + 12, r.y + 32, w * Math.min(1, pd), 6, 3); g.fillStyle = COL.rose; g.fill(); }
    g.strokeStyle = COL.line3; g.beginPath(); g.moveTo(r.x + 12 + w / 2, r.y + 28); g.lineTo(r.x + 12 + w / 2, r.y + 42); g.stroke();
    if (r.h > 60) text(g, `logit ${rec.M.doneLogit.toFixed(2)} · 超过 50%（竖线）梦就“撞车”`, r.x + 12, r.y + 60, { size: 9.5, color: COL.dim });
    env.hit(r.x, r.y, r.w, r.h, { tip: '<span class="k">撞车概率</span>σ(w·h + b)。训练时只有每局最后一步是 1。' });
  }
}
