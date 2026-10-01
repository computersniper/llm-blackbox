// D1 玩：左边是真实的游戏，右边是模型的梦，同一串动作驱动两边。
// 右上是两幅画面逐像素的差，差异曲线记着梦怎么一点点走样；下面 32 根柱子是梦此刻的 z，可以拖。
import { COL, rr, text, card, hexA, line, clamp } from '../../../train/js/draw.js';
import { Pix } from '../pix.js';
import { toCHW } from '../game.js';
import { dimOrder, dimShort, dimMeaning, used as isUsed, fmtMSE } from '../explain.js';

export class PlayView {
  constructor(app) {
    this.app = app;
    this.fixed = true;
    this.real = new Pix();
    this.dream = new Pix();
    this.dpix = new Pix();
    this.drag = null;
  }

  layout(env) {
    this.portrait = env.portrait;
    if (this.portrait) {
      const s = 172;
      this.R = { x: 0, y: 30, w: s, h: s };
      this.D = { x: 188, y: 30, w: s, h: s };
      this.HM = null;
      this.CH = { x: 0, y: 244, w: 360, h: 92 };
      this.EQ = { x: 0, y: 358, w: 360, h: 128 };
      this.box = { x: -6, y: 0, w: 372, h: 492 };
    } else {
      this.R = { x: 0, y: 34, w: 300, h: 300 };
      this.D = { x: 330, y: 34, w: 300, h: 300 };
      this.HM = { x: 664, y: 34, w: 128, h: 128 };
      this.CH = { x: 664, y: 194, w: 266, h: 140 };
      this.EQ = { x: 0, y: 372, w: 930, h: 150 };
      this.box = { x: -10, y: 0, w: 950, h: 530 };
    }
  }

  focus() { return this.box; }

  draw(g, st, env) {
    const { sim, meta } = this.app;
    const F = sim.cur;
    const y = sim.dreamOf(F);
    this.real.palette(F.o, F.o);
    this.dream.chw(y, y);
    const pt = this.portrait;
    // ---- 两块屏幕
    this.screen(g, this.R, this.real, '真实世界', 'game.js · 和训练数据同一个游戏', COL.cyan, env);
    const closed = sim.mode === 'closed';
    const eyeTxt = closed ? (F.dreamAge > 0 ? `闭眼 · 自己想了 ${F.dreamAge} 步` : '闭眼') : '睁眼 · 每步看着真实画面预测';
    this.screen(g, this.D, this.dream, '模型的梦', pt ? 'V 解码 M 的预测' : 'V.decode(M 预测的 z) · 浏览器现场算', closed ? COL.amber : COL.violet, env, eyeTxt);
    // 里程、撞车
    text(g, `里程 ${F.t}${sim.best > F.t ? `  最远 ${sim.best}` : ''}`, this.R.x + 8, this.R.y + this.R.h - 8, { size: pt ? 9.5 : 11, kind: 'mono', color: 'rgba(233,239,249,0.75)' });
    if (F.realDone) this.flash(g, this.R, F.why === 'offroad' ? '冲出路面' : '撞车了', '下一步换一条新路', env);
    if (F.dreamDone > 0.5) this.flash(g, this.D, '梦里撞车了', `M 预测撞车的概率 ${(F.dreamDone * 100).toFixed(0)}%`, env);
    else if (F.dreamDone > 0.15) text(g, `撞车概率 ${(F.dreamDone * 100).toFixed(0)}%`, this.D.x + this.D.w - 8, this.D.y + this.D.h - 8, { size: pt ? 9.5 : 11, kind: 'mono', color: COL.rose, align: 'right' });
    if (F.edited) text(g, '改过的 z', this.D.x + 8, this.D.y + this.D.h - 8, { size: pt ? 9.5 : 11, color: COL.amber });
    // 动作
    const a = F.rec?.a ?? sim.keys();
    const ax = (this.R.x + this.R.w + this.D.x) / 2, ay = this.R.y + this.R.h / 2;
    this.actionChip(g, ax, ay, a, pt);
    // ---- 逐像素的差
    const x = toCHW(F.o);
    if (this.HM) {
      this.dpix.diff(x, y, y);
      const r = this.HM;
      text(g, '|真实 − 梦|', r.x, r.y - 12, { size: 12.5, kind: 'serif', weight: 600, color: COL.ink });
      rr(g, r.x - 3, r.y - 3, r.w + 6, r.h + 6, 6);
      g.fillStyle = COL.panel; g.fill(); g.strokeStyle = COL.line2; g.lineWidth = 1; g.stroke();
      this.dpix.draw(g, r.x, r.y, r.w, r.h);
      const tx = r.x + r.w + 14;
      text(g, '每像素均方误差', tx, r.y + 14, { size: 10.5, color: COL.dim });
      text(g, fmtMSE(F.mse), tx, r.y + 38, { size: 20, kind: 'mono', color: closed ? COL.amber : COL.cyan });
      const ref = meta.drift;
      text(g, `睁眼平均 ${fmtMSE(ref.open[0])}`, tx, r.y + 62, { size: 10, kind: 'mono', color: COL.dim });
      text(g, `只看 V 的重建 ${fmtMSE(ref.recon)}`, tx, r.y + 78, { size: 10, kind: 'mono', color: COL.dim });
      text(g, closed ? '闭眼越久，差得越多' : '每一步都重新看一眼', tx, r.y + 100, { size: 10.5, color: COL.ink2 });
      env.hit(r.x, r.y, r.w, r.h, { tip: `<span class="k">逐像素的差</span>真实画面和梦里那一帧每个像素差了多少，越亮差得越多。<br>均方误差 <span class="v">${fmtMSE(F.mse)}</span>` });
    }
    this.chart(g, this.CH, env);
    this.equalizer(g, this.EQ, env, st);
  }

  screen(g, r, pix, title, sub, accent, env, chip = '') {
    const pt = this.portrait;
    text(g, title, r.x, r.y - (pt ? 10 : 14), { size: pt ? 13 : 16, kind: 'serif', weight: 600, color: COL.ink });
    if (!pt) text(g, sub, r.x + (title.length * 16 + 10), r.y - 14, { size: 10.5, kind: 'mono', color: COL.dim, max: r.w - title.length * 16 - 10 });
    rr(g, r.x - 4, r.y - 4, r.w + 8, r.h + 8, 8);
    g.fillStyle = 'rgba(4,8,16,0.9)'; g.fill();
    g.strokeStyle = hexA(accent, 0.55); g.lineWidth = 1.2; g.stroke();
    pix.draw(g, r.x, r.y, r.w, r.h);
    if (chip) {
      const fs = pt ? 9 : 10.5;
      g.font = `${fs}px ${'-apple-system, "PingFang SC", "Microsoft YaHei", sans-serif'}`;
      const w = g.measureText(chip).width + 14;
      rr(g, r.x + 6, r.y + 6, w, fs + 9, (fs + 9) / 2);
      g.fillStyle = 'rgba(6,12,24,0.82)'; g.fill();
      g.strokeStyle = hexA(accent, 0.6); g.stroke();
      text(g, chip, r.x + 13, r.y + 6 + fs + 2.5, { size: fs, color: accent });
    }
  }

  flash(g, r, big, small, env) {
    const k = 0.55 + 0.25 * Math.sin(env.t * 9);
    g.fillStyle = `rgba(120,20,40,${0.32 * k})`;
    g.fillRect(r.x, r.y, r.w, r.h);
    const pt = this.portrait;
    text(g, big, r.x + r.w / 2, r.y + r.h / 2 - 2, { size: pt ? 16 : 24, kind: 'serif', weight: 600, color: '#ffd0da', align: 'center' });
    text(g, small, r.x + r.w / 2, r.y + r.h / 2 + (pt ? 16 : 22), { size: pt ? 9 : 11, color: '#ffb3c2', align: 'center' });
  }

  // 竖屏时排成一行放在两块屏幕下面，横屏时竖着放在两块屏幕中间
  actionChip(g, x, y, a, pt) {
    const s = pt ? 10 : 13;
    const items = [['◀', 1], ['▲', 0], ['▶', 2]];
    if (pt) y = this.R.y + this.R.h + 16;
    items.forEach(([ch, v], k) => {
      const cx = pt ? x + (k - 1) * (s * 2 + 6) : x, cy = pt ? y : y + (k - 1) * (s + 10);
      const on = a === v;
      rr(g, cx - s, cy - s * 0.8, s * 2, s * 1.6, 5);
      g.fillStyle = on ? hexA(COL.amber, 0.22) : 'rgba(255,255,255,0.03)';
      g.fill();
      g.strokeStyle = on ? hexA(COL.amber, 0.7) : COL.line2; g.lineWidth = 1; g.stroke();
      text(g, ch, cx, cy + s * 0.32, { size: s * 0.8, color: on ? COL.amber : COL.faint, align: 'center' });
    });
    if (pt) text(g, '两边同一个动作', x + s * 3 + 18, y + 3.5, { size: 9, color: COL.dim });
    else text(g, '同一个动作', x, y + (s + 10) * 1.5 + s * 0.6, { size: 9.5, color: COL.dim, align: 'center' });
  }

  // 差异曲线：最近 120 帧；闭眼时叠上“平均来说会涨成这样”（测试集上的离线统计）
  chart(g, r, env) {
    const { sim, meta } = this.app;
    const pt = this.portrait;
    card(g, r.x, r.y, r.w, r.h, { r: 10 });
    text(g, '梦和真实差多少', r.x + 10, r.y + 17, { size: pt ? 11 : 12.5, kind: 'serif', weight: 600 });
    text(g, '每像素均方误差 · 最近 120 帧', r.x + r.w - 10, r.y + 16, { size: 9.5, color: COL.dim, align: 'right' });
    const P = { x: r.x + 10, y: r.y + 26, w: r.w - 20, h: r.h - 36 };
    const frames = sim.recent(120);
    const ref = meta.drift;
    let mx = 0.006;
    for (const f of frames) if (f.mse != null) mx = Math.max(mx, f.mse * 1.2);
    mx = Math.min(0.08, mx);
    const X = (i) => P.x + (i / 119) * P.w, Y = (v) => P.y + P.h - Math.min(1, v / mx) * P.h;
    g.strokeStyle = COL.line; g.lineWidth = 1;
    for (const q of [0.25, 0.5, 0.75]) { g.beginPath(); g.moveTo(P.x, P.y + P.h * q); g.lineTo(P.x + P.w, P.y + P.h * q); g.stroke(); }
    g.beginPath(); g.moveTo(P.x, P.y + P.h); g.lineTo(P.x + P.w, P.y + P.h); g.strokeStyle = COL.line2; g.stroke();
    text(g, fmtMSE(mx), P.x + 2, P.y + 9, { size: 8.5, kind: 'mono', color: COL.faint });
    // 睁眼的平均水平
    g.setLineDash([3, 4]);
    line(g, [[P.x, Y(ref.open[0])], [P.x + P.w, Y(ref.open[0])]], hexA(COL.cyan, 0.35), 1);
    g.setLineDash([]);
    const off = 120 - frames.length;
    // 闭眼：从最近一次“闭眼开始”的那一帧起，画出平均会涨成的样子
    const F = sim.cur;
    if (sim.mode === 'closed' && F.dreamAge > 0) {
      const start = frames.length - 1 - F.dreamAge;
      const key = sim.tau <= 0.75 ? 'closed_0.5' : 'closed_1.0';
      const pts = [];
      for (let k = 0; k < ref[key].length; k++) {
        const i = off + start + 1 + k;
        if (i > 119 + 20) break;
        if (i >= 0) pts.push([X(Math.min(i, 119)), Y(ref[key][k])]);
      }
      g.setLineDash([2, 3]);
      line(g, pts, hexA(COL.amber, 0.45), 1.2);
      g.setLineDash([]);
    }
    // 快进时有的帧没解码过（没有差异值）：跳过它们，把同一局里前后有值的帧连起来
    let pk = -1;
    for (let k = 0; k < frames.length; k++) {
      const b = frames[k];
      if (b.mse == null) continue;
      if (pk >= 0 && frames[pk].gameNo === b.gameNo) line(g, [[X(off + pk), Y(frames[pk].mse)], [X(off + k), Y(b.mse)]], b.dreamAge > 0 ? COL.amber : COL.cyan, 1.6);
      pk = k;
    }
    const last = frames[frames.length - 1];
    if (last?.mse != null) { g.beginPath(); g.arc(X(119), Y(last.mse), 3, 0, Math.PI * 2); g.fillStyle = last.dreamAge > 0 ? COL.amber : COL.cyan; g.fill(); }
    if (!pt) {
      text(g, '— 睁眼', P.x + 60, P.y + 9, { size: 9, color: COL.cyan });
      text(g, '— 闭眼', P.x + 104, P.y + 9, { size: 9, color: COL.amber });
    }
    env.hit(r.x, r.y, r.w, r.h, { tip: `<span class="k">梦和真实的差异</span>青色：睁眼（每一步都看一眼真实画面再预测下一帧）；琥珀色：闭眼（梦用自己上一步想出来的 z 往下想）。<br>虚线是测试集上 ${ref.episodes} 局的平均：闭眼 10 步后约 <span class="v">${fmtMSE(ref['closed_1.0'][9])}</span>，30 步后约 <span class="v">${fmtMSE(ref['closed_1.0'][29])}</span>。` });
  }

  // 32 根柱子：梦此刻的 z（按 KL 从大到小排，KL≈0 的维度模型没用上）。白色小刻度是真实画面编码出来的 z
  equalizer(g, r, env, st) {
    const { sim, meta } = this.app;
    const pt = this.portrait;
    const F = sim.cur;
    const order = dimOrder(meta);
    // 白色刻度要把真实画面再编码一次：闭眼播放时省掉（睁眼时本来就要编码）
    const enc = sim.mode === 'open' || !st.playing || F.enc ? sim.encOf(F).mu : null;
    text(g, '梦此刻的 z：32 个数', r.x, r.y + 2, { size: pt ? 12 : 14, kind: 'serif', weight: 600 });
    text(g, pt ? '拖动柱子改梦（会自动闭眼）' : '拖动柱子改梦里的画面（会自动闭眼）· 白色刻度 = 真实画面编码出来的 z · 按“用得多不多”（KL）排序', r.x + (pt ? 128 : 160), r.y + 2, { size: pt ? 9 : 10.5, color: COL.dim, max: r.w - (pt ? 128 : 160) });
    const top = r.y + 14, bh = r.h - (pt ? 34 : 40);
    const gap = pt ? 2 : 4, bw = (r.w - gap * 31) / 32;
    const mid = top + bh / 2;
    order.forEach((d, k) => {
      const info = meta.dims[d];
      const x = r.x + k * (bw + gap);
      const used = isUsed(meta, d);
      const R = Math.max(1.2, Math.abs(info.lo), Math.abs(info.hi)) * 1.15;
      const Yv = (v) => mid - clamp(v / R, -1, 1) * (bh / 2);
      rr(g, x, top, bw, bh, 3);
      g.fillStyle = used ? 'rgba(255,255,255,0.035)' : 'rgba(255,255,255,0.015)'; g.fill();
      const hov = env.hover?.dim === d || this.drag?.d === d;
      if (hov) { g.strokeStyle = hexA(COL.amber, 0.6); g.lineWidth = 1; g.stroke(); }
      // 数据里常见的范围
      g.fillStyle = used ? 'rgba(94,240,212,0.06)' : 'rgba(255,255,255,0.02)';
      const y1 = Yv(info.hi), y2 = Yv(info.lo);
      g.fillRect(x + 1, y1, bw - 2, y2 - y1);
      const v = F.zhat[d], yv = Yv(v);
      g.fillStyle = !used ? 'rgba(122,133,158,0.35)' : sim.mode === 'closed' ? hexA(COL.amber, 0.75) : hexA(COL.violet, 0.75);
      g.fillRect(x + 2, Math.min(mid, yv), bw - 4, Math.max(1, Math.abs(yv - mid)));
      // 真实画面的编码
      if (enc) {
        const ye = Yv(enc[d]);
        g.fillStyle = 'rgba(233,239,249,0.85)';
        g.fillRect(x, ye - 0.75, bw, 1.5);
      }
      text(g, `${d}`, x + bw / 2, top + bh + (pt ? 9 : 11), { size: pt ? 7 : 9, kind: 'mono', color: used ? COL.dim : COL.faint, align: 'center' });
      if (!pt && used && k < 12) text(g, dimShort(meta, d), x + bw / 2, top + bh + 24, { size: 9.5, color: COL.ink2, align: 'center', max: bw + gap });
      env.hit(x, top, bw, bh, {
        dim: d,
        tip: () => `<span class="k">z<sub>${d}</sub> · KL ${info.kl.toFixed(2)}</span>${used ? `扫一遍：${dimMeaning(meta, d)}` : '这一维几乎没被用上（KL≈0）：编码器总给它同一个数，解码器也不看它'}<br>梦：<span class="v">${F.zhat[d].toFixed(3)}</span>　真实画面：<span class="v">${sim.encOf(F).mu[d].toFixed(3)}</span><br><span class="v">上下拖动改梦</span>`,
        drag: (wx, wy, phase) => {
          if (phase === 'start') { this.drag = { d }; this.app.dragStart?.(); }
          if (phase === 'end') { this.drag = null; this.app.dragEnd?.(); return; }
          const nv = clamp((mid - wy) / (bh / 2), -1, 1) * R;
          sim.editZ(d, nv);
        },
      });
    });
  }
}
