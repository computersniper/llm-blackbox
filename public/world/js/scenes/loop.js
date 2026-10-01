// D2 循环：一帧之内，真实画面 → V 编码 → z → M（LSTM + MDN）→ 采样 ẑ → V 解码 → 梦见的下一帧，
// 上面一条是真实世界自己走一步。闭眼时从真实画面进来的那条路熄灭，ẑ 从下面绕回来当下一步的 z。
// 底下三张卡片是 V、M、C 真实的训练记录。
import { COL, rr, text, card, hexA, line, clamp, ease, seg } from '../../../train/js/draw.js';
import { Pix, vecGrid } from '../pix.js';
import { ACTIONS } from '../game.js';
import { dimOrder, fmtMSE } from '../explain.js';

const ORDER_STAGE = ['obs', 'enc', 'rnn', 'sample', 'dec', 'cmp'];

export class LoopView {
  constructor(app) {
    this.app = app;
    this.obs = new Pix();
    this.real1 = new Pix();
    this.dream = new Pix();
    this.diffp = new Pix();
  }

  layout(env) {
    this.portrait = env.portrait;
    if (this.portrait) {
      this.L = {
        O: { x: 0, y: 30, w: 120, h: 120 }, VE: { x: 132, y: 40, w: 56, h: 100 }, ZT: { x: 200, y: 45, w: 92, h: 90 },
        M: { x: 0, y: 186, w: 400, h: 136 }, ZN: { x: 0, y: 366, w: 92, h: 90 }, VD: { x: 104, y: 361, w: 56, h: 100 },
        D: { x: 172, y: 351, w: 120, h: 120 }, R1: { x: 304, y: 351, w: 96, h: 96 }, ENV: null,
        CARDS: { x: 0, y: 500, w: 400, h: 230 },
      };
      this.box = { x: -8, y: 0, w: 416, h: 740 };
    } else {
      this.L = {
        O: { x: 0, y: 190, w: 150, h: 150 }, VE: { x: 182, y: 195, w: 104, h: 140 }, ZT: { x: 314, y: 205, w: 130, h: 120 },
        M: { x: 478, y: 168, w: 226, h: 196 }, ZN: { x: 738, y: 205, w: 130, h: 120 }, VD: { x: 896, y: 195, w: 104, h: 140 },
        D: { x: 1030, y: 190, w: 150, h: 150 }, R1: { x: 1030, y: 8, w: 150, h: 150 }, ENV: { x: 478, y: 40, w: 226, h: 70 },
        CARDS: { x: 0, y: 432, w: 1180, h: 168 },
      };
      this.box = { x: -14, y: -14, w: 1208, h: 628 };
    }
  }

  focus() { return this.box; }

  draw(g, st, env) {
    const { sim, meta } = this.app;
    const F = sim.cur, rec = F.rec, G = sim.next;
    const L = this.L, pt = this.portrait;
    const s = st.step, p = st.p;
    const k = ORDER_STAGE.indexOf(s.ph);
    const on = (ph) => s.ph === ph;
    const done = (ph) => k > ORDER_STAGE.indexOf(ph);
    const closed = rec?.src === 'dream';
    if (!rec || rec.kind !== 'step') { this.drawReset(g, F, env); return; }

    // ---- 真实画面
    this.obs.palette(F.o, F.o);
    this.screen(g, L.O, this.obs, pt ? '真实画面 obs' : '真实画面 obs<sub>t</sub>', COL.cyan, on('obs'), env);
    // ---- 编码
    const encLive = !closed;
    this.trap(g, L.VE, 'V', '编码', pt ? 'down' : 'right', on('enc') && encLive, encLive ? 1 : 0.3, env);
    this.flow(g, [[L.O.x + L.O.w + 4, L.O.y + L.O.h / 2], [L.VE.x - 4, L.VE.y + L.VE.h / 2]], COL.cyan, on('enc') && encLive ? p : done('enc') && encLive ? 1 : 0, encLive ? 1 : 0.25);
    this.flow(g, [[L.VE.x + L.VE.w + 4, L.VE.y + L.VE.h / 2], [L.ZT.x - 4, L.ZT.y + L.ZT.h / 2]], COL.cyan, on('enc') ? p : done('enc') ? 1 : 0, encLive ? 1 : 0.25);
    if (!encLive) text(g, '闭眼：不看', L.VE.x + L.VE.w / 2, L.VE.y - 8, { size: 10, color: COL.amber, align: 'center' });
    // ---- z_t
    const zFill = on('enc') ? ease(p) : done('enc') || on('obs') === false ? 1 : 0;
    this.bars(g, L.ZT, rec.z, pt ? 'z（32）' : 'z<sub>t</sub> · 32 个数', closed ? COL.amber : COL.violet, k >= 1 ? zFill : 0, on('enc'), env, closed ? '来自上一步的 ẑ' : 'V 编码出来的 μ');
    // ---- 动作
    const ax = L.M.x + L.M.w / 2, ay = pt ? L.M.y - 10 : L.M.y - 22;
    const aTxt = `a = ${ACTIONS[rec.a]}${rec.by === 'key' ? '（你按的）' : rec.by === 'heur' ? '（自动驾驶）' : rec.by === 'ctrl' ? '（C）' : ''}`;
    // ---- M
    this.mBox(g, L.M, F, rec, on('rnn') || on('sample'), on('rnn') ? p : done('rnn') ? 1 : 0, env);
    this.flow(g, pt ? [[L.ZT.x + L.ZT.w / 2, L.ZT.y + L.ZT.h + 4], [L.ZT.x + L.ZT.w / 2, L.M.y - 4]] : [[L.ZT.x + L.ZT.w + 4, L.ZT.y + L.ZT.h / 2], [L.M.x - 4, L.ZT.y + L.ZT.h / 2]], COL.violet, on('rnn') ? Math.min(1, p * 2) : done('rnn') ? 1 : 0, 1);
    this.chip(g, pt ? L.M.x + 80 : ax, pt ? L.M.y - 14 : ay, aTxt, COL.amber, on('rnn'));
    // ---- 采样 ẑ_{t+1}
    const zn = on('sample') ? ease(p) : done('sample') ? 1 : 0;
    this.flow(g, pt ? [[L.M.x + 46, L.M.y + L.M.h + 4], [L.ZN.x + L.ZN.w / 2, L.ZN.y - 4]] : [[L.M.x + L.M.w + 4, L.ZN.y + L.ZN.h / 2], [L.ZN.x - 4, L.ZN.y + L.ZN.h / 2]], COL.amber, on('sample') ? p : done('sample') ? 1 : 0, 1);
    this.bars(g, L.ZN, rec.S.z, pt ? 'ẑ（采样）' : `ẑ<sub>t+1</sub> · 采样 τ=${rec.tau.toFixed(2)}`, COL.amber, zn, on('sample'), env, '从 M 给的分布里抽出来的');
    // ---- 解码
    this.trap(g, L.VD, 'V', '解码', pt ? 'right-open' : 'left', on('dec'), 1, env);
    this.flow(g, [[L.ZN.x + L.ZN.w + 4, L.ZN.y + L.ZN.h / 2], [L.VD.x - 4, L.VD.y + L.VD.h / 2]], COL.amber, on('dec') ? Math.min(1, p * 2) : done('dec') ? 1 : 0, 1);
    this.flow(g, [[L.VD.x + L.VD.w + 4, L.VD.y + L.VD.h / 2], [L.D.x - 4, L.D.y + L.D.h / 2]], COL.amber, on('dec') ? Math.max(0, p * 2 - 1) : done('dec') ? 1 : 0, 1);
    // 梦见的下一帧：解码时从上往下一行行显现
    if (G) {
      const y = sim.dreamOf(G);
      this.dream.chw(y, y);
      const rev = on('dec') ? ease(seg(p, 0.35, 1)) : done('dec') ? 1 : 0;
      this.screen(g, L.D, this.dream, pt ? '梦见的下一帧 ô' : '梦见的下一帧 ô<sub>t+1</sub>', COL.amber, on('dec') || on('cmp'), env, rev);
      // 真实世界的下一帧
      this.real1.palette(G.o, G.o);
      const realRev = on('cmp') ? 1 : done('cmp') ? 1 : 0.0;
      this.screen(g, L.R1, this.real1, pt ? '真实下一帧' : '真实世界的下一帧 obs<sub>t+1</sub>', COL.cyan, on('cmp'), env, realRev, !realRev ? '对照时揭晓' : '');
      if (on('cmp')) {
        const a = ease(p);
        const mx = pt ? L.R1.x + L.R1.w / 2 : L.D.x + L.D.w / 2, my = pt ? L.R1.y + L.R1.h + 16 : L.R1.y + L.R1.h + 16;
        text(g, `差 ${fmtMSE(G.mse)}`, mx, my, { size: pt ? 11 : 12.5, kind: 'mono', color: hexA(COL.rose, 0.4 + 0.6 * a), align: 'center' });
      }
    }
    // ---- 真实世界自己走一步（上面那条）
    if (L.ENV) {
      const E = L.ENV;
      rr(g, E.x, E.y, E.w, E.h, 10);
      g.fillStyle = on('cmp') ? 'rgba(94,240,212,0.08)' : COL.panel; g.fill();
      g.strokeStyle = on('cmp') ? hexA(COL.cyan, 0.6) : COL.line2; g.lineWidth = 1; g.stroke();
      text(g, '真实世界：env.step(a)', E.x + E.w / 2, E.y + 28, { size: 13, kind: 'serif', weight: 600, align: 'center' });
      text(g, '游戏引擎，不经过模型', E.x + E.w / 2, E.y + 48, { size: 10.5, color: COL.dim, align: 'center' });
      const pr = on('cmp') ? p : done('cmp') ? 1 : 0;
      this.flow(g, [[L.O.x + L.O.w / 2, L.O.y - 6], [L.O.x + L.O.w / 2, E.y + E.h / 2], [E.x - 4, E.y + E.h / 2]], COL.cyan, pr > 0 ? Math.min(1, pr * 2) : 0, 0.8);
      this.flow(g, [[E.x + E.w + 4, E.y + E.h / 2], [L.R1.x - 4, E.y + E.h / 2]], COL.cyan, pr > 0 ? Math.max(0, pr * 2 - 1) : 0, 0.8);
      // 同一个动作也发给真实世界
      g.setLineDash([2, 3]);
      line(g, [[ax, ay - 10], [ax, E.y + E.h + 4]], hexA(COL.amber, 0.45), 1);
      g.setLineDash([]);
    }
    // ---- 闭眼的回路：ẑ 绕回来当下一步的 z
    if (!pt) {
      const y0 = L.ZN.y + L.ZN.h + 34;
      const pts = [[L.ZN.x + L.ZN.w / 2, L.ZN.y + L.ZN.h + 20], [L.ZN.x + L.ZN.w / 2, y0], [L.ZT.x + L.ZT.w / 2, y0], [L.ZT.x + L.ZT.w / 2, L.ZT.y + L.ZT.h + 20]];
      const nowClosed = sim.mode === 'closed';
      g.setLineDash([4, 4]);
      line(g, pts, hexA(COL.amber, nowClosed ? 0.55 : 0.18), 1.2);
      g.setLineDash([]);
      text(g, nowClosed ? '闭眼：这个 ẑ 就是下一帧的 z（梦自己往下想）' : '闭眼时走这条：ẑ 当作下一帧的 z', (L.ZT.x + L.ZN.x + L.ZN.w) / 2, y0 + 16, { size: 11, color: nowClosed ? COL.amber : COL.faint, align: 'center' });
    }
    this.cards(g, L.CARDS, env);
  }

  drawReset(g, F, env) {
    const L = this.L;
    this.obs.palette(F.o, F.o);
    this.screen(g, L.O, this.obs, '真实画面', COL.rose, true, env);
    text(g, F.why === 'offroad' ? '冲出路面，这一局结束' : '撞车了，这一局结束', L.ZT.x, L.ZT.y + 30, { size: 16, kind: 'serif', weight: 600, color: COL.rose });
    text(g, '下一步换一条新路；梦重新从真实画面开始，h、c 清零', L.ZT.x, L.ZT.y + 56, { size: 12, color: COL.ink2 });
    this.cards(g, L.CARDS, env);
  }

  screen(g, r, pix, title, accent, active, env, reveal = 1, note = '') {
    text(g, title.replace(/<\/?sub>/g, ''), r.x, r.y - 9, { size: this.portrait ? 10.5 : 12, kind: 'serif', weight: 600, color: active ? COL.ink : COL.ink2 });
    rr(g, r.x - 3, r.y - 3, r.w + 6, r.h + 6, 7);
    g.fillStyle = 'rgba(4,8,16,0.92)'; g.fill();
    g.strokeStyle = hexA(accent, active ? 0.8 : 0.35); g.lineWidth = active ? 1.6 : 1; g.stroke();
    if (reveal > 0) {
      g.save();
      g.beginPath(); g.rect(r.x, r.y, r.w, r.h * reveal); g.clip();
      pix.draw(g, r.x, r.y, r.w, r.h);
      g.restore();
      if (reveal < 1) { g.fillStyle = hexA(accent, 0.7); g.fillRect(r.x, r.y + r.h * reveal - 1, r.w, 2); }
    }
    if (note) text(g, note, r.x + r.w / 2, r.y + r.h / 2 + 4, { size: 10.5, color: COL.faint, align: 'center' });
  }

  // 梯形：编码器收窄、解码器展开
  trap(g, r, a, b, dir, active, alpha, env) {
    g.save();
    g.globalAlpha *= alpha;
    const n = 0.36;
    g.beginPath();
    if (dir === 'right') { g.moveTo(r.x, r.y); g.lineTo(r.x + r.w, r.y + r.h * (0.5 - n / 2)); g.lineTo(r.x + r.w, r.y + r.h * (0.5 + n / 2)); g.lineTo(r.x, r.y + r.h); }
    else if (dir === 'left' || dir === 'right-open') { g.moveTo(r.x, r.y + r.h * (0.5 - n / 2)); g.lineTo(r.x + r.w, r.y); g.lineTo(r.x + r.w, r.y + r.h); g.lineTo(r.x, r.y + r.h * (0.5 + n / 2)); }
    else { g.moveTo(r.x, r.y); g.lineTo(r.x + r.w, r.y + r.h * (0.5 - n / 2)); g.lineTo(r.x + r.w, r.y + r.h * (0.5 + n / 2)); g.lineTo(r.x, r.y + r.h); }
    g.closePath();
    g.fillStyle = active ? 'rgba(179,157,255,0.14)' : 'rgba(179,157,255,0.06)';
    g.fill();
    g.strokeStyle = hexA(COL.violet, active ? 0.85 : 0.4);
    g.lineWidth = active ? 1.5 : 1;
    g.stroke();
    text(g, a, r.x + r.w / 2, r.y + r.h / 2 - 2, { size: this.portrait ? 16 : 22, kind: 'serif', weight: 600, color: active ? COL.ink : COL.ink2, align: 'center' });
    text(g, b, r.x + r.w / 2, r.y + r.h / 2 + 15, { size: 10, color: COL.dim, align: 'center' });
    g.restore();
    env.hit(r.x, r.y, r.w, r.h, { tip: `<span class="k">V · 卷积变分自编码器</span>${b === '编码' ? '64×64×3 → 4 层卷积 → 32 个数' : '32 个数 → 全连接 + 4 层反卷积 → 64×64×3'}<br><span class="v">点 ＋ 看里面的每一层</span>`, click: true });
  }

  bars(g, r, z, title, color, fill, active, env, sub) {
    const { meta } = this.app;
    text(g, title.replace(/<\/?sub>/g, ''), r.x, r.y - 9, { size: this.portrait ? 10.5 : 12, kind: 'serif', weight: 600, color: active ? COL.ink : COL.ink2 });
    card(g, r.x, r.y, r.w, r.h, { r: 8, active });
    const order = dimOrder(meta);
    const pad = 6, bw = (r.w - pad * 2) / 32, mid = r.y + r.h / 2;
    g.strokeStyle = COL.line2; g.lineWidth = 1;
    g.beginPath(); g.moveTo(r.x + pad, mid); g.lineTo(r.x + r.w - pad, mid); g.stroke();
    order.forEach((d, i) => {
      const used = meta.dims[d].kl > 0.02;
      const R = Math.max(1.2, Math.abs(meta.dims[d].lo), Math.abs(meta.dims[d].hi)) * 1.15;
      const v = clamp(z[d] / R, -1, 1) * fill;
      const h = v * (r.h / 2 - pad);
      g.fillStyle = used ? hexA(color, 0.85) : 'rgba(122,133,158,0.35)';
      g.fillRect(r.x + pad + i * bw + 0.5, Math.min(mid, mid - h), Math.max(1, bw - 1), Math.max(0.8, Math.abs(h)));
    });
    if (sub && !this.portrait) text(g, sub, r.x + r.w / 2, r.y + r.h + 14, { size: 9.5, color: COL.dim, align: 'center' });
    env.hit(r.x, r.y, r.w, r.h, { tip: `<span class="k">${title}</span>${sub}。按“用得多不多”排序，灰色的维度模型没用上。<br>|z| = <span class="v">${Math.sqrt(z.reduce((s, v) => s + v * v, 0)).toFixed(3)}</span>` });
  }

  mBox(g, r, F, rec, active, p, env) {
    const pt = this.portrait;
    card(g, r.x, r.y, r.w, r.h, { r: 12, active, accent: COL.violet });
    text(g, 'M', r.x + 14, r.y + 26, { size: 22, kind: 'serif', weight: 600, color: active ? COL.ink : COL.ink2 });
    text(g, 'LSTM + 混合密度网络', r.x + 40, r.y + 24, { size: 11, color: COL.dim });
    const gs = pt ? 62 : 70;
    const gy = r.y + (pt ? 42 : 46);
    if (this.hKey !== rec) { this.hKey = rec; this.h0 = vecGrid(F.h, 16, { scale: 1 }); this.h1 = vecGrid(rec.L.h, 16, { scale: 1 }); }
    const h0 = this.h0, h1 = this.h1;
    const x0 = r.x + 16, x1 = r.x + (pt ? 120 : 132);
    h0.draw(g, x0, gy, gs, gs);
    h1.draw(g, x1, gy, gs, gs, p > 0 ? 0.3 + 0.7 * ease(p) : 0.25);
    g.strokeStyle = COL.line2; g.strokeRect(x0 - 0.5, gy - 0.5, gs + 1, gs + 1); g.strokeRect(x1 - 0.5, gy - 0.5, gs + 1, gs + 1);
    text(g, '记忆 h', x0, gy + gs + 13, { size: 10, color: COL.dim });
    text(g, "新的 h'", x1, gy + gs + 13, { size: 10, color: COL.dim });
    line(g, [[x0 + gs + 6, gy + gs / 2], [x1 - 6, gy + gs / 2]], hexA(COL.violet, 0.6), 1.2);
    // 撞车概率
    const dx = pt ? r.x + 220 : r.x + 16, dy = pt ? r.y + 50 : r.y + r.h - 34;
    const pd = rec.M.done;
    text(g, `撞车概率 ${(pd * 100).toFixed(pd < 0.1 ? 1 : 0)}%`, dx, dy, { size: 11, color: pd > 0.5 ? COL.rose : COL.ink2 });
    rr(g, dx, dy + 6, pt ? 150 : 194, 5, 2.5); g.fillStyle = 'rgba(255,255,255,0.06)'; g.fill();
    rr(g, dx, dy + 6, (pt ? 150 : 194) * Math.min(1, pd) * (p > 0 ? ease(p) : 0), 5, 2.5); g.fillStyle = COL.rose; g.fill();
    if (pt) text(g, '输出：下一帧 z 的分布', dx, dy + 34, { size: 10, color: COL.dim });
    env.hit(r.x, r.y, r.w, r.h, { tip: '<span class="k">M · MDN-RNN</span>LSTM（256 维记忆）+ 混合密度输出：下一帧 z 的每一维是 5 个高斯的混合，再加一个“撞车了吗”。<br><span class="v">点 ＋ 看门和记忆</span>', click: true });
  }

  chip(g, x, y, s, color, active) {
    g.font = '11px -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif';
    const w = g.measureText(s).width + 16;
    rr(g, x - w / 2, y - 10, w, 20, 10);
    g.fillStyle = active ? hexA(color, 0.2) : 'rgba(10,17,31,0.9)'; g.fill();
    g.strokeStyle = hexA(color, active ? 0.8 : 0.45); g.lineWidth = 1; g.stroke();
    text(g, s, x, y + 4, { size: 11, color, align: 'center' });
  }

  // 一条带流动光点的箭头：pr = 0..1 是数据走到了哪
  flow(g, pts, color, pr, alpha = 1) {
    g.save();
    g.globalAlpha *= alpha;
    line(g, pts, hexA(color, 0.3), 1.2);
    const lens = [];
    let tot = 0;
    for (let i = 1; i < pts.length; i++) { const l = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); lens.push(l); tot += l; }
    const at = (d) => {
      for (let i = 0; i < lens.length; i++) {
        if (d <= lens[i]) { const f = d / lens[i]; return [pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f]; }
        d -= lens[i];
      }
      return pts[pts.length - 1];
    };
    if (pr > 0) {
      const d = tot * clamp(pr, 0, 1);
      const pts2 = [pts[0]];
      let acc = 0;
      for (let i = 0; i < lens.length && acc + lens[i] < d; i++) { acc += lens[i]; pts2.push(pts[i + 1]); }
      pts2.push(at(d));
      line(g, pts2, hexA(color, 0.85), 1.6);
      if (pr < 1) { const q = at(d); g.beginPath(); g.arc(q[0], q[1], 3, 0, Math.PI * 2); g.fillStyle = color; g.fill(); }
    }
    // 箭头
    const a = pts[pts.length - 2], b = pts[pts.length - 1];
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    g.beginPath();
    g.moveTo(b[0], b[1]);
    g.lineTo(b[0] - 6 * Math.cos(ang - 0.45), b[1] - 6 * Math.sin(ang - 0.45));
    g.lineTo(b[0] - 6 * Math.cos(ang + 0.45), b[1] - 6 * Math.sin(ang + 0.45));
    g.closePath();
    g.fillStyle = hexA(color, pr >= 1 ? 0.9 : 0.4);
    g.fill();
    g.restore();
  }

  // V、M、C 真实的训练记录
  cards(g, r, env) {
    const { meta } = this.app;
    const T = meta.train;
    const pt = this.portrait;
    const items = [
      { t: 'V 怎么训出来的', a: `${(T.collect.frames / 1e4).toFixed(0)} 万帧 · ${T.V.epochs} 轮 · ${Math.round(T.V.seconds)} 秒`, b: `重建误差（加权平方和）${T.V.finalRecon} · KL ${T.V.finalKL}`, curve: T.Vcurve.map((c) => c[1]), color: COL.violet },
      { t: 'M 怎么训出来的', a: `${T.M.iters.toLocaleString()} 步 · 每批 ${T.M.batch}×${T.M.seqLen} · ${Math.round(T.M.seconds)} 秒`, b: `负对数似然 ${T.M.finalNLL} · 测试集 ${T.M.testNLL}`, curve: T.Mcurve.map((c) => c[1]), color: COL.amber },
    ];
    if (T.C) items.push({ t: 'C 在梦里学开车', a: `CMA-ES ${T.C.gens} 代 × ${T.C.pop} 个 · ${Math.round(T.C.seconds)} 秒`, b: `梦里活 ${T.C.dreamLife} 步 · 真实游戏 ${T.C.realLife} 步（乱开 ${T.C.randomLife}）`, curve: T.Ccurve.map((c) => c[1]), color: COL.cyan });
    const n = items.length;
    const gap = 14;
    const w = pt ? r.w : (r.w - gap * (n - 1)) / n, h = pt ? (r.h - gap * (n - 1)) / n : r.h;
    items.forEach((it, i) => {
      const x = pt ? r.x : r.x + i * (w + gap), y = pt ? r.y + i * (h + gap) : r.y;
      card(g, x, y, w, h, { r: 10, accent: it.color });
      text(g, it.t, x + 14, y + 24, { size: pt ? 12 : 13.5, kind: 'serif', weight: 600 });
      text(g, it.a, x + 14, y + 42, { size: pt ? 9.5 : 10.5, kind: 'mono', color: COL.ink2, max: w - 28 });
      text(g, it.b, x + 14, y + 57, { size: pt ? 9.5 : 10.5, color: COL.dim, max: w - 28 });
      const P = { x: x + 14, y: y + (pt ? 64 : 70), w: w - 28, h: h - (pt ? 72 : 82) };
      if (P.h > 14) {
        const c = it.curve;
        let mn = Infinity, mx = -Infinity;
        for (const v of c) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
        const pts = c.map((v, j) => [P.x + (j / (c.length - 1)) * P.w, P.y + P.h - ((v - mn) / (mx - mn || 1)) * P.h]);
        g.strokeStyle = COL.line; g.beginPath(); g.moveTo(P.x, P.y + P.h); g.lineTo(P.x + P.w, P.y + P.h); g.stroke();
        line(g, pts, hexA(it.color, 0.85), 1.3);
      }
    });
    text(g, `${T.gpu || 'GPU'} · PyTorch ${T.torch}`, r.x + r.w, r.y + r.h + 16, { size: 9.5, kind: 'mono', color: COL.faint, align: 'right' });
  }
}
