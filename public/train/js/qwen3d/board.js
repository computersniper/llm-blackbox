// 第三章 3D 舞台上的 2D 浮层（画法和训练页其余的卡片一样，../draw.js）：
//   监视器（左上角，D1–D3）：回答 8 个词元的真实概率，原模型 → 第 1 / 2 / 3 步之后，最下面一行是损失；
//   算式板（底部）：D4 一个张量的梯度 / 更新量；D5 一个权重的 AdamW 算式和 bf16 比特；D3 “存回 bf16”；D1 结尾的 DPO 浮层（示意）。
import { COL, text, rr, card, hexA, clamp, seg, sciSup, fmtInt, line, dot, wrap, measure, fmtP, badge } from '../draw.js';
import { f32Bits, bf16Round } from '../explain.js';
import { tokPlain } from '../../../js/ui.js';
import { tname, TSHAPE, LAYER_T, numel } from './data.js';
import { ADAM_SUBS } from './timeline.js';
import { t as T_, L, isEn } from './lang.js';
import { SFT_EN } from '../lang.js';

const small = () => matchMedia('(max-width: 900px)').matches;
const num = (v, d = 4) => (v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e4 ? Number(v.toPrecision(d + 1)).toString().replace('-', '−') : sciSup(v, d));
const par = (v, d = 4) => (v < 0 ? `(${num(v, d)})` : num(v, d));
const tk = (s) => tokPlain(s).replace(/^<\|(.+)\|>$/, '⟨$1⟩');

class Pane {
  constructor(el, onTip) {
    this.el = el;
    this.cv = document.createElement('canvas');
    el.append(this.cv);
    this.g = this.cv.getContext('2d');
    this.hits = [];
    this.onTip = onTip;
    this.w = this.h = 0;
    const at = (e) => { const r = this.cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    const find = (p) => { for (let i = this.hits.length - 1; i >= 0; i--) { const h = this.hits[i]; if (p.x >= h.x && p.x <= h.x + h.w && p.y >= h.y && p.y <= h.y + h.h) return h; } return null; };
    this.cv.addEventListener('pointerdown', (e) => { const h = find(at(e)); if (h?.info?.act) h.info.act(); });
    this.cv.addEventListener('pointermove', (e) => {
      const h = find(at(e));
      this.onTip(h?.info?.tip ?? null, e);
      this.cv.style.cursor = h?.info?.act ? 'pointer' : '';
    });
    this.cv.addEventListener('pointerleave', () => this.onTip(null));
  }
  size(w, h) {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    if (w === this.w && h === this.h && dpr === this.dpr) return;
    this.w = w; this.h = h; this.dpr = dpr;
    this.cv.width = Math.round(w * dpr);
    this.cv.height = Math.round(h * dpr);
    this.cv.style.width = `${w}px`;
    this.cv.style.height = `${h}px`;
  }
  draw(fn) {
    const g = this.g;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, this.cv.width, this.cv.height);
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.hits = [];
    fn(g, { hit: (x, y, w, h, info) => this.hits.push({ x, y, w, h, info }) });
  }
}

export class QwenBoard {
  constructor(host, { onTip, E, R, app }) {
    this.E = E;
    this.R = R;
    this.app = app;
    const mk = (cls, lab) => { const el = document.createElement('section'); el.className = cls; el.hidden = true; el.setAttribute('aria-label', lab); host.append(el); return el; };
    this.hudEl = mk('qhud', T_('q3.hudAria'));
    this.boardEl = mk('qboard', T_('q3.boardAria'));
    this.hud = new Pane(this.hudEl, onTip);
    this.board = new Pane(this.boardEl, onTip);
    this.ovKey = '';
    new ResizeObserver(() => { this.ovKey = ''; }).observe(this.boardEl);
    addEventListener('resize', () => { this.ovKey = ''; });
  }

  hide() { this.hudEl.hidden = this.boardEl.hidden = true; document.body.classList.remove('q-board'); this.E.setOverlay(0, 0); this.ovKey = ''; }

  // 底部算式板要画什么
  boardKind(st) {
    const s = st.step, v = st.view;
    if (v === 'q-end') return 'end';
    if (v === 'q-param') return s.ph === 'upd' ? 'adam' : 'wgrad';
    if (v === 'q-mat') return 'tensor';
    if (s.ph === 'upd' && s.sub === 'bf16' && st.depth >= 3) return 'bf16';
    return null;
  }

  update(st, t, F) {
    const kind = this.boardKind(st);
    const showHud = !kind && st.view !== 'q-mat' && st.view !== 'q-param';
    this.hudEl.hidden = !showHud;
    this.boardEl.hidden = !kind;
    document.body.classList.toggle('q-board', !!kind);
    if (showHud) this.drawHud(st, F);
    if (kind) this.drawBoard(kind, st, F);
    const roomy = (this.E.w || 0) - (this.E.insetR || 0) >= 1000;
    this.measure(kind ? this.boardEl : showHud && (!roomy || st.view === 'q-op') ? this.hudEl : null);
  }

  /* ---------------------------------------------------------------- 监视器 */

  drawHud(st, F) {
    const R = this.R, sm = small();
    const w = this.hudEl.clientWidth || (sm ? 300 : 320);
    const rowH = sm ? 11 : 16, A = R.ans.length;
    const h = sm ? 40 + A * rowH + 26 : 50 + A * rowH + 30;
    this.hud.size(w, h);
    const cur = st.view === 'q-end' ? R.K : F && (F.after > 0.5 || F.next > 0.5) ? st.k + 1 : st.k;
    this.hud.draw((g, env) => {
      card(g, 0.5, 0.5, w - 1, h - 1, { eyebrow: T_('q3.hudEye'), accent: COL.green });
      const x0 = sm ? 52 : 64, cw = (w - x0 - 12) / (R.K + 1), y0 = sm ? 34 : 44;
      const heads = [T_('q3.hudS0'), T_('q3.hudS', { n: 1 }), T_('q3.hudS', { n: 2 }), T_('q3.hudS', { n: 3 })];
      for (let s = 0; s <= R.K; s++) {
        const x = x0 + s * cw;
        if (s === cur) { rr(g, x + 1, y0 - (sm ? 11 : 13), cw - 2, A * rowH + (sm ? 30 : 36), 6); g.fillStyle = hexA(COL.amber, 0.07); g.fill(); g.strokeStyle = hexA(COL.amber, 0.45); g.lineWidth = 1; g.stroke(); }
        text(g, heads[s], x + cw / 2, y0 - 2, { size: sm ? 8.5 : 9.5, color: s === cur ? COL.amber : COL.dim, align: 'center' });
      }
      R.ans.forEach((i, a) => {
        const y = y0 + 4 + a * rowH;
        text(g, tk(R.tgtStr(i)), x0 - 8, y + rowH * 0.72, { size: sm ? 9 : 11, color: COL.ink2, align: 'right', max: x0 - 12 });
        for (let s = 0; s <= R.K; s++) {
          const pv = R.pState(s, i), x = x0 + s * cw + 4, bw = cw - 8;
          rr(g, x, y + 2, bw, rowH - 4, 2);
          g.fillStyle = 'rgba(255,255,255,0.03)';
          g.fill();
          const c = pv > 0.5 ? COL.green : pv > 0.1 ? COL.amber : COL.rose;
          g.fillStyle = hexA(c, s === cur ? 0.75 : 0.4);
          g.fillRect(x, y + 2, Math.max(1, bw * pv), rowH - 4);
          if (!sm) text(g, fmtP(pv), x + bw - 2, y + rowH * 0.72, { size: 9, kind: 'mono', color: s === cur ? COL.ink : COL.dim, align: 'right' });
          env.hit(x, y, bw, rowH, { tip: `<span class="k">${heads[s]}</span>「${tk(R.tgtStr(i))}」 p = <span class="v">${fmtP(pv)}</span>` });
        }
      });
      const ly = y0 + 4 + A * rowH + (sm ? 13 : 17);
      text(g, T_('q3.hudLoss'), x0 - 8, ly, { size: sm ? 9 : 10.5, color: COL.rose, align: 'right' });
      for (let s = 0; s <= R.K; s++) text(g, R.lossState(s).toFixed(s === R.K ? 3 : 2), x0 + s * cw + cw / 2, ly, { size: sm ? 9.5 : 11, kind: 'mono', weight: 700, color: s === cur ? COL.amber : COL.ink2, align: 'center' });
    });
  }

  /* ---------------------------------------------------------------- 算式板 */

  drawBoard(kind, st, F) {
    const sm = small(), W = this.boardEl.clientWidth || 800, R = this.R, k = st.k;
    const need3 = kind !== 'end' && !R.has3(k), needSt = (kind === 'tensor' || kind === 'wgrad') && !R.hasSt(k);
    if (need3 || needSt) {
      this.board.size(W, 90);
      this.board.draw((g) => { card(g, 0.5, 0.5, W - 1, 89, {}); text(g, T_('q3.loadingStep'), 20, 50, { size: 12.5, color: COL.dim }); });
      return;
    }
    const s = st.step;
    if (kind === 'end') {
      if (sm) { const h1 = 360, h2 = 190; this.board.size(W, h1 + 10 + h2); this.board.draw((g, env) => { this.drawDpo(g, { x: 0.5, y: 0.5, w: W - 1, h: h1 }, env); this.drawChanged(g, { x: 0.5, y: h1 + 10, w: W - 1, h: h2 - 1 }, env); }); }
      else { const h = 300, mw = Math.min(640, W * 0.6); this.board.size(W, h); this.board.draw((g, env) => { this.drawDpo(g, { x: 0.5, y: 0.5, w: mw, h: h - 1 }, env); this.drawChanged(g, { x: mw + 10, y: 0.5, w: W - mw - 10.5, h: h - 1 }, env); }); }
      return;
    }
    if (kind === 'bf16') {
      const h = sm ? 400 : 270;
      this.board.size(W, h);
      this.board.draw((g, env) => this.drawBf16(g, { x: 0.5, y: 0.5, w: W - 1, h: h - 1 }, st, env));
      return;
    }
    if (kind === 'tensor') {
      if (sm) { const h1 = 250, h2 = 210; this.board.size(W, h1 + 10 + h2); this.board.draw((g, env) => { this.drawTensor(g, { x: 0.5, y: 0.5, w: W - 1, h: h1 }, st, env); this.drawTensorChart(g, { x: 0.5, y: h1 + 10, w: W - 1, h: h2 - 1 }, st, env); }); }
      else { const h = 260, mw = Math.min(520, Math.max(400, W * 0.48)); this.board.size(W, h); this.board.draw((g, env) => { this.drawTensor(g, { x: 0.5, y: 0.5, w: mw, h: h - 1 }, st, env); this.drawTensorChart(g, { x: mw + 10, y: 0.5, w: W - mw - 10.5, h: h - 1 }, st, env); }); }
      return;
    }
    if (kind === 'wgrad') {
      const h = sm ? 300 : 230;
      this.board.size(W, h);
      this.board.draw((g, env) => this.drawWGrad(g, { x: 0.5, y: 0.5, w: W - 1, h: h - 1 }, st, env));
      return;
    }
    // AdamW
    if (sm) { const h1 = 330, h2 = 300; this.board.size(W, h1 + 10 + h2); this.board.draw((g, env) => { this.drawAdam(g, { x: 0.5, y: 0.5, w: W - 1, h: h1 }, st, env, true); this.drawBits(g, { x: 0.5, y: h1 + 10, w: W - 1, h: h2 - 1 }, st, env, true); }); }
    else { const h = 300, mw = Math.min(600, Math.max(470, W * 0.5)); this.board.size(W, h); this.board.draw((g, env) => { this.drawAdam(g, { x: 0.5, y: 0.5, w: mw, h: h - 1 }, st, env, false); this.drawBits(g, { x: mw + 10, y: 0.5, w: W - mw - 10.5, h: h - 1 }, st, env, false); }); }
    void s;
  }

  tLabel(st, tn) {
    const l = this.app.tl?.L ?? 27;
    return tn === 'embed' || tn === 'norm' ? T_(`q3.t.${tn}`) : `${T_('q3.layerN', { l })} · ${T_(`q3.t.${tn}`)}`;
  }

  // D4：一个张量的梯度、更新量、bf16
  drawTensor(g, C, st, env) {
    const R = this.R, k = st.k, s = st.step, tn = s.t, l = this.app.tl?.L ?? 27;
    const L_ = tn === 'embed' || tn === 'norm' ? -1 : l;
    const ts = R.tensor(k, L_, tn);
    const ph = s.ph;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: T_('q3.tenEye', { n: k + 1 }), title: this.tLabel(st, tn), accent: ph === 'bwd' ? COL.rose : ph === 'upd' ? COL.violet : COL.cyan, active: true });
    const sh = TSHAPE[tn];
    const rows = [];
    rows.push([T_('q3.tenShape'), `${sh.length === 2 ? `[${sh[0]} × ${sh[1]}]` : `[${sh[0]}]`} · ${T_('q3.tenN', { n: fmtInt(numel(tn)) })}`, COL.ink2, ph === 'fwd']);
    rows.push([T_('q3.tenW'), ts.w != null ? num(ts.w, 3) : '—', COL.ink2, ph === 'fwd']);
    rows.push([T_('q3.tenG'), `${num(ts.g, 3)}  × ${ts.clip.toPrecision(3)} → ${num(ts.g * ts.clip, 3)}`, COL.rose, ph === 'bwd']);
    if (k === 0 && ts.gPre != null) rows.push([T_('q3.tenGPre'), num(ts.gPre, 3), COL.dim, ph === 'bwd']);
    rows.push([T_('q3.tenDw'), ts.dw != null ? `${num(ts.dw, 3)} · ${T_('q3.tenDwMax', { v: num(ts.dwMax, 2) })}` : '—', COL.violet, ph === 'upd']);
    if (ts.dw != null && ts.w) rows.push([T_('q3.tenRel'), num(ts.dw / ts.w, 2), COL.violet, ph === 'upd']);
    if (ts.revert != null) rows.push([T_('q3.tenRev'), T_('q3.tenRevV', { c: fmtInt(ts.changed), r: fmtInt(ts.revert), p: ((ts.revert / ts.n) * 100).toFixed(1) }), COL.amber, ph === 'upd']);
    const lx = C.x + 16, vx = C.x + (isEn ? 168 : 150);
    rows.forEach(([kk, vv, c, hot], j) => {
      const y = C.y + 66 + j * 25;
      if (hot) { rr(g, C.x + 8, y - 15, C.w - 16, 22, 5); g.fillStyle = hexA(c === COL.ink2 ? COL.cyan : c, 0.07); g.fill(); }
      text(g, kk, lx, y, { size: 11, color: COL.dim, max: vx - lx - 8 });
      text(g, vv, vx, y, { size: 11.5, kind: 'mono', color: c, max: C.x + C.w - vx - 12 });
    });
    const note = ph === 'fwd' ? T_('q3.tenNoteF') : ph === 'bwd' ? T_('q3.tenNoteB') : T_('q3.tenNoteU');
    wrap(g, note, C.x + 16, C.y + C.h - 30, C.w - 32, 14, { size: 10.5, color: COL.dim, maxLines: 2 });
  }

  // D4 右边：这一层 11 个张量的梯度（或嵌入表梯度最大的行）+ 同一个张量在 28 层里的梯度
  drawTensorChart(g, C, st, env) {
    const R = this.R, k = st.k, s = st.step, tn = s.t, l = this.app.tl?.L ?? 27, NL = R.NL;
    if (tn === 'embed') {
      const em = R.emb(k);
      card(g, C.x, C.y, C.w, C.h, { eyebrow: T_('q3.embEye', { n: k + 1 }), title: T_('q3.embTitle'), accent: COL.violet });
      const rows = em.top.slice(0, 10), mx = rows[0][1];
      const x0 = C.x + 90, bw = C.w - 160;
      rows.forEach(([id, v], j) => {
        const y = C.y + 64 + j * 17;
        const conv = em.conv.some(([cid]) => cid === id);
        text(g, tk(R.tok(id)), x0 - 8, y + 10, { size: 11, color: conv ? COL.amber : COL.ink2, align: 'right', max: 80 });
        g.fillStyle = hexA(conv ? COL.amber : COL.violet, 0.55);
        g.fillRect(x0, y + 2, Math.max(1, bw * v / mx), 11);
        text(g, num(v, 3), x0 + bw * v / mx + 6, y + 11, { size: 9.5, kind: 'mono', color: COL.dim });
      });
      wrap(g, T_('q3.embNote'), C.x + 16, C.y + C.h - 26, C.w - 32, 13, { size: 10, color: COL.dim, maxLines: 2 });
      return;
    }
    if (tn === 'norm') {
      card(g, C.x, C.y, C.w, C.h, { eyebrow: T_('q3.normEye'), title: T_('q3.normTitle'), accent: COL.blue });
      const vals = [0, 1, 2].map((kk) => R.gradOf(kk, tname(-1, 'norm')));
      const mx = Math.max(...vals);
      vals.forEach((v, kk) => {
        const y = C.y + 70 + kk * 30;
        text(g, T_('q3.stepN', { n: kk + 1 }), C.x + 16, y + 12, { size: 11, color: kk === k ? COL.amber : COL.dim });
        g.fillStyle = hexA(COL.rose, kk === k ? 0.7 : 0.35);
        g.fillRect(C.x + 80, y + 2, (C.w - 170) * v / mx, 14);
        text(g, num(v, 3), C.x + 86 + (C.w - 170) * v / mx, y + 14, { size: 10, kind: 'mono', color: COL.ink2 });
      });
      wrap(g, T_('q3.normNote'), C.x + 16, C.y + C.h - 26, C.w - 32, 13, { size: 10, color: COL.dim, maxLines: 2 });
      return;
    }
    const ph = s.ph;
    const useDw = ph === 'upd' && R.has3(k);
    const val = (L_, t) => (useDw ? R.tensor(k, L_, t).dw ?? 0 : R.gradOf(k, tname(L_, t)));
    card(g, C.x, C.y, C.w, C.h, { eyebrow: useDw ? T_('q3.chartEyeD', { n: k + 1 }) : T_('q3.chartEyeG', { n: k + 1 }), accent: useDw ? COL.violet : COL.rose });
    const half = C.w > 520;
    // 这一层 11 个张量（对数刻度）
    const A = { x: C.x + 16, y: C.y + 40, w: half ? C.w * 0.5 - 24 : C.w - 32, h: half ? C.h - 70 : (C.h - 80) / 2 };
    const vals = LAYER_T.map((t) => Math.max(1e-12, val(l, t)));
    const lo = Math.log10(Math.min(...vals)) - 0.2, hi = Math.log10(Math.max(...vals)) + 0.1;
    const bh = Math.min(14, (A.h - 18) / vals.length - 2);
    text(g, T_('q3.chartLayer', { l }), A.x, A.y + 8, { size: 10, color: COL.dim });
    LAYER_T.forEach((t, j) => {
      const y = A.y + 16 + j * (bh + 2), on = t === tn;
      const f = (Math.log10(vals[j]) - lo) / (hi - lo);
      text(g, T_(`q3.t.${t}`), A.x + 62, y + bh - 2, { size: 9.5, color: on ? COL.amber : COL.ink2, align: 'right' });
      g.fillStyle = hexA(useDw ? COL.violet : COL.rose, on ? 0.8 : 0.35);
      g.fillRect(A.x + 68, y, Math.max(1, (A.w - 120) * f), bh);
      text(g, num(vals[j], 2), A.x + 72 + (A.w - 120) * f, y + bh - 2, { size: 9, kind: 'mono', color: on ? COL.ink : COL.dim });
      env.hit(A.x, y, A.w, bh + 2, { tip: `<span class="k">${T_('q3.layerN', { l })} · ${T_(`q3.t.${t}`)}</span>${useDw ? '‖Δw‖' : '‖∇‖'} = <span class="v">${num(vals[j], 4)}</span>` });
    });
    // 同一个张量在 28 层里
    const B = half ? { x: C.x + C.w * 0.5 + 8, y: C.y + 40, w: C.w * 0.5 - 24, h: C.h - 70 } : { x: C.x + 16, y: A.y + A.h + 14, w: C.w - 32, h: (C.h - 80) / 2 };
    const across = Array.from({ length: NL }, (_, L_) => Math.max(1e-12, val(L_, tn)));
    const mx = Math.max(...across);
    text(g, T_('q3.chartAcross', { t: T_(`q3.t.${tn}`) }), B.x, B.y + 8, { size: 10, color: COL.dim });
    const bw = (B.w - 8) / NL, by0 = B.y + B.h - 4;
    across.forEach((v, L_) => {
      const hgt = (B.h - 30) * v / mx, x = B.x + 4 + L_ * bw;
      g.fillStyle = hexA(useDw ? COL.violet : COL.rose, L_ === l ? 0.9 : 0.35);
      g.fillRect(x + 0.5, by0 - hgt, Math.max(1, bw - 1.5), hgt);
      env.hit(x, B.y + 14, bw, B.h - 14, { tip: `<span class="k">${T_('q3.layerN', { l: L_ })} · ${T_(`q3.t.${tn}`)}</span>${useDw ? '‖Δw‖' : '‖∇‖'} = <span class="v">${num(v, 4)}</span><br><span style="color:var(--dim)">${T_('q3.tipLayerClick')}</span>`, act: () => this.app.tl?.setLayer(L_) });
    });
    text(g, '0', B.x + 4, by0 + 11, { size: 8.5, kind: 'mono', color: COL.faint });
    text(g, String(NL - 1), B.x + B.w - 4, by0 + 11, { size: 8.5, kind: 'mono', color: COL.faint, align: 'right' });
    if (useDw && k === 0) wrap(g, T_('q3.chartNoteD0'), C.x + 16, C.y + C.h - 16, C.w - 32, 12, { size: 9.5, color: COL.dim, maxLines: 1 });
  }

  // D5 反向：这个权重的梯度
  drawWGrad(g, C, st, env) {
    const R = this.R, k = st.k, tn = st.step.t, l = this.app.tl?.L ?? 27;
    const L_ = tn === 'embed' || tn === 'norm' ? -1 : l;
    const a = R.adamT(k, L_, tn), idx = R.trackedIndex(L_, tn), ts = R.tensor(k, L_, tn);
    card(g, C.x, C.y, C.w, C.h, { eyebrow: T_('q3.wgEye', { n: k + 1 }), title: `${this.tLabel(st, tn)} [${idx.join(', ')}]`, accent: COL.rose, active: true });
    const rms = ts.g / Math.sqrt(ts.n);
    const lines = [
      [T_('q3.wgWhich'), tn === 'embed' ? T_('q3.wgWhichE', { s: tk(R.tok(idx[0])), d: idx[1] }) : T_('q3.wgWhichT')],
      [T_('q3.wgG'), `∂L/∂w = ${num(a.gRaw, 4)}`],
      [T_('q3.wgRms'), T_('q3.wgRmsV', { r: num(rms, 3), x: (Math.abs(a.gRaw) / rms).toFixed(0) })],
      [T_('q3.wgClip'), `${num(a.gRaw, 4)} × ${a.g / a.gRaw > 0 ? (a.g / a.gRaw).toPrecision(4) : '—'} = ${num(a.g, 4)}`],
    ];
    lines.forEach(([kk, vv], j) => {
      const y = C.y + 70 + j * 26;
      text(g, kk, C.x + 16, y, { size: 11, color: COL.dim });
      text(g, vv, C.x + (isEn ? 190 : 150), y, { size: 12, kind: 'mono', color: j === 1 ? COL.rose : COL.ink2, max: C.w - 170 });
    });
    // 三步的原始梯度
    const gs = [0, 1, 2].map((kk) => R.adamT(kk, L_, tn)?.gRaw ?? 0);
    const mx = Math.max(...gs.map(Math.abs), 1e-12);
    const bx = C.x + C.w - 200, by = C.y + 60;
    if (C.w > 560) {
      text(g, T_('q3.wg3'), bx, by, { size: 10, color: COL.dim });
      gs.forEach((v, kk) => {
        const y = by + 14 + kk * 26, w = 120 * Math.abs(v) / mx;
        g.fillStyle = hexA(v < 0 ? COL.violet : COL.rose, kk === k ? 0.8 : 0.35);
        g.fillRect(bx + 50, y, w, 14);
        text(g, T_('q3.stepN', { n: kk + 1 }), bx, y + 11, { size: 10, color: kk === k ? COL.amber : COL.dim });
        text(g, num(v, 3), bx + 54 + w, y + 11, { size: 9.5, kind: 'mono', color: COL.ink2 });
      });
    }
    wrap(g, T_('q3.wgNote'), C.x + 16, C.y + C.h - 30, C.w - 32, 14, { size: 10.5, color: COL.dim, maxLines: 2 });
  }

  // D5 更新：AdamW 算式（全部是这个权重这一步的真实数字）
  drawAdam(g, C, st, env, P) {
    const R = this.R, k = st.k, s = st.step, tn = s.t, l = this.app.tl?.L ?? 27;
    const L_ = tn === 'embed' || tn === 'norm' ? -1 : l;
    const a = R.adamT(k, L_, tn), idx = R.trackedIndex(L_, tn);
    const mi = ADAM_SUBS.indexOf(s.mi);
    card(g, C.x, C.y, C.w, C.h, { eyebrow: T_('q3.adamEye', { n: a.t }), title: `${this.tLabel(st, tn)} [${idx.join(', ')}]`, accent: COL.violet, active: true });
    const b1 = 0.9, b2 = 0.95, lr = a.lr, wd = a.wd, eps = a.eps;
    const ratio = a.mh / (Math.sqrt(a.vh) + eps);
    const dw = a.w1 - a.w0;
    const L1 = [
      [`g = ${num(a.gRaw)} × ${(a.g / a.gRaw).toPrecision(4)}`, `= ${num(a.g)}`, T_('q3.aG')],
      [`m = ${b1}·${par(a.m0)} + ${(1 - b1).toFixed(1)}·${par(a.g)}`, `= ${num(a.m)}`, T_('q3.aM')],
      [`v = ${b2}·${par(a.v0)} + ${(1 - b2).toFixed(2)}·${par(a.g)}²`, `= ${num(a.v)}`, T_('q3.aV')],
      [`m̂ = m/${a.bc1.toPrecision(3)},  v̂ = v/${a.bc2.toPrecision(3)}`, `= ${num(a.mh)}, ${num(a.vh)}`, T_('q3.aBc', { t: a.t })],
      [`Δw = −${lr}·(${num(ratio, 3)} + ${wd}·${par(a.w0)})`, `= ${num(dw, 3)}`, wd ? T_('q3.aDw') : T_('q3.aDwNo')],
      [`w ← ${num(a.w0, 6)} ${dw >= 0 ? '+' : '−'} ${num(Math.abs(dw), 3)}`, `= ${num(a.w1, 7)}`, T_('q3.aW')],
    ];
    const lh = P ? 46 : 36;
    L1.forEach(([lhs, rhs, note], j) => {
      const y = C.y + 64 + j * lh;
      const on = j === Math.min(mi, 5) || (mi === 6 && j === 5);
      const done = j < mi;
      if (on) { rr(g, C.x + 8, y - 15, C.w - 16, lh - 6, 6); g.fillStyle = hexA(COL.violet, 0.1); g.fill(); g.strokeStyle = hexA(COL.violet, 0.5); g.lineWidth = 1; g.stroke(); }
      g.globalAlpha = on || done ? 1 : 0.35;
      text(g, lhs, C.x + 16, y, { size: P ? 11 : 12, kind: 'mono', color: COL.ink, max: P ? C.w - 32 : C.w * 0.62 });
      text(g, rhs, P ? C.x + 16 : C.x + C.w - 14, P ? y + 16 : y, { size: P ? 11 : 12, kind: 'mono', weight: 700, color: on ? COL.amber : COL.ink2, align: P ? 'left' : 'right' });
      if (!P && on) text(g, note, C.x + 16, y + 16, { size: 10, color: COL.dim, max: C.w - 32 });
      else if (P && on) text(g, note, C.x + 140, y + 16, { size: 9.5, color: COL.dim, max: C.w - 156 });
      g.globalAlpha = 1;
    });
    if (a.t === 1 && mi >= 3) text(g, T_('q3.aSign'), C.x + 16, C.y + C.h - 12, { size: 10.5, color: COL.cyan, max: C.w - 32 });
  }

  // D5 比特：原模型的这个权重（bf16 存的）、这一步之后的 fp32、存回 bf16 会不会变回原值
  drawBits(g, C, st, env, P) {
    const R = this.R, k = st.k, s = st.step, tn = s.t, l = this.app.tl?.L ?? 27;
    const L_ = tn === 'embed' || tn === 'norm' ? -1 : l;
    const a = R.adamT(k, L_, tn), a0 = R.adamT(0, L_, tn);
    const on = s.mi === 'bits' || s.mi === 'write';
    card(g, C.x, C.y, C.w, C.h, { eyebrow: T_('q3.bitsEye'), title: T_('q3.bitsTitle'), accent: COL.amber, active: on });
    const orig = a0 ? a0.w0 : a.w0, w1 = a.w1, back = bf16Round(w1);
    const rows = [
      { lab: T_('q3.bitsOrig'), bits: f32Bits(orig) >>> 16, n: 16, val: orig },
      { lab: T_('q3.bitsNow', { n: k + 1 }), bits: f32Bits(w1), n: 32, val: w1, cmp: f32Bits(orig) },
      { lab: T_('q3.bitsBack'), bits: f32Bits(back) >>> 16, n: 16, val: back, cmp: f32Bits(orig) >>> 16 },
    ];
    const kw = Math.max(4, Math.min(P ? 9 : 11, (C.w - 40) / 33 - 1.2)), kh = P ? 16 : 18, gap = 1.2;
    rows.forEach((row, ri) => {
      const y = C.y + 62 + ri * (P ? 60 : 62);
      g.globalAlpha = on || ri === 0 ? 1 : 0.45;
      text(g, row.lab, C.x + 16, y, { size: 10.5, color: COL.ink2, max: C.w - 32 });
      for (let i = 0; i < row.n; i++) {
        const bit = (row.bits >>> (row.n - 1 - i)) & 1;
        const field = i === 0 ? 'sign' : i <= 8 ? 'exp' : 'man';
        const c = field === 'sign' ? COL.rose : field === 'exp' ? COL.amber : COL.cyan;
        const changed = row.cmp != null && ((((row.n === 32 ? row.bits : row.bits) ^ row.cmp) >>> (row.n - 1 - i)) & 1);
        const bx = C.x + 16 + i * (kw + gap), by = y + 8;
        rr(g, bx, by, kw, kh, 2);
        g.fillStyle = bit ? hexA(c, 0.35) : 'rgba(255,255,255,0.03)';
        g.fill();
        g.strokeStyle = changed ? '#fff' : hexA(c, 0.45);
        g.lineWidth = changed ? 1.5 : 1;
        g.stroke();
      }
      text(g, num(row.val, 9), C.x + 16, y + kh + 22, { size: 11, kind: 'mono', weight: 700, color: COL.ink });
      g.globalAlpha = 1;
    });
    const same = back === orig;
    text(g, same ? T_('q3.bitsSame') : T_('q3.bitsDiff'), C.x + 16, C.y + C.h - 12, { size: 11, weight: 600, color: same ? COL.rose : COL.cyan, max: C.w - 32 });
  }

  // D3：存回 bf16（全模型的真实统计）
  drawBf16(g, C, st, env) {
    const R = this.R, k = st.k, sm = small();
    card(g, C.x, C.y, C.w, C.h, { eyebrow: T_('q3.bfEye'), title: T_('q3.bfTitle'), accent: COL.amber, active: true });
    const tot = R.total;
    const x0 = C.x + 16, w = C.w - 32;
    for (let kk = 0; kk < R.K; kk++) {
      const y = C.y + 70 + kk * (sm ? 56 : 42);
      const ch = R.changedAfter(kk), rv = R.revertAfter(kk);
      text(g, T_('q3.bfStep', { n: kk + 1 }), x0, y, { size: 11, color: kk === k ? COL.amber : COL.dim });
      const bx = x0 + (sm ? 0 : 90), bw = w - (sm ? 0 : 90), by = y + (sm ? 8 : -11);
      rr(g, bx, by, bw, 14, 3);
      g.fillStyle = 'rgba(255,255,255,0.04)';
      g.fill();
      g.fillStyle = hexA(COL.rose, kk === k ? 0.6 : 0.3);
      g.fillRect(bx, by, bw * (rv / tot), 14);
      g.fillStyle = hexA(COL.cyan, kk === k ? 0.6 : 0.3);
      g.fillRect(bx + bw * (rv / tot), by, bw * ((ch - rv) / tot), 14);
      text(g, T_('q3.bfRow', { c: (ch / 1e8).toFixed(2), r: (rv / 1e8).toFixed(2), p: ((rv / tot) * 100).toFixed(1) }), bx, by + 28, { size: 10.5, kind: 'mono', color: COL.ink2, max: bw });
      env.hit(bx, by, bw, 14, { tip: T_('q3.bfTip', { n: kk + 1, c: fmtInt(ch), r: fmtInt(rv), t: fmtInt(tot) }) });
    }
    wrap(g, T_('q3.bfNote'), x0, C.y + C.h - (sm ? 60 : 44), w, 15, { size: 11, color: COL.ink2, maxLines: sm ? 4 : 3 });
  }

  // D1 结尾：DPO（示意，用真实对数概率代入公式）
  drawDpo(g, C, env) {
    const d = this.R.D.meta.dpo;
    card(g, C.x, C.y, C.w, C.h, { eyebrow: T_('q3.dpoEye'), title: T_('q3.dpoTitle'), accent: COL.rose, demo: true });
    let y = C.y + 66;
    wrap(g, T_('q3.dpoIntro'), C.x + 16, y, C.w - 32, 16, { size: 11.5, color: COL.ink2, maxLines: 2 });
    y += 40;
    const rows = [[T_('q3.dpoGood'), d.chosen, d.policy.chosen, d.ref.chosen, COL.cyan, SFT_EN.answer], [T_('q3.dpoBad'), d.rejected, d.policy.rejected, d.ref.rejected, COL.rose, SFT_EN.rejected]];
    const tx = C.x + (isEn ? 64 : 52);
    for (const [tag, s, pol, ref, c, en] of rows) {
      rr(g, C.x + 14, y, C.w - 28, 50, 8);
      g.fillStyle = hexA(c, 0.05); g.fill();
      g.strokeStyle = hexA(c, 0.3); g.lineWidth = 1; g.stroke();
      text(g, tag, C.x + 24, y + 19, { size: isEn ? 11 : 13, color: c, weight: 700 });
      text(g, s, tx, y + 19, { size: 11.5, color: COL.ink, max: C.x + C.w - 30 - tx });
      text(g, T_('q3.dpoLp', { p: pol.toFixed(2), r: ref.toFixed(2), d: `${pol - ref >= 0 ? '+' : ''}${(pol - ref).toFixed(2)}` }), tx, y + 39, { size: 10.5, kind: 'mono', color: COL.dim, max: C.x + C.w - 30 - tx });
      if (isEn) env.hit(C.x + 14, y, C.w - 28, 50, { tip: `<span class="k">${tag} · translation</span>“${en}”` });
      y += 58;
    }
    text(g, T_('q3.dpoF'), C.x + 16, y + 10, { size: 11.5, kind: 'mono', color: COL.ink2, max: C.w - 32 });
    text(g, `= −log σ(${d.beta} × ${d.margin.toFixed(1)}) = ${d.loss.toFixed(4)}`, C.x + 16, y + 30, { size: 11.5, kind: 'mono', color: COL.amber, max: C.w - 32 });
    wrap(g, T_('q3.dpoNote'), C.x + 16, y + 50, C.w - 32, 14, { size: 10.5, color: COL.dim, maxLines: 2 });
  }

  // D1 结尾：3 步改了多少
  drawChanged(g, C, env) {
    const R = this.R, tot = R.total, ch = R.changedAfter(R.K - 1), rv = R.revertAfter(R.K - 1);
    card(g, C.x, C.y, C.w, C.h, { eyebrow: T_('q3.chgEye'), title: T_('q3.chgTitle'), accent: COL.green });
    const rows = [[T_('q3.chgLoss'), `${R.lossState(0).toFixed(3)} → ${R.lossState(R.K).toFixed(3)}`, COL.amber], [T_('q3.chgPre'), `${R.lossPreState(0).toFixed(3)} → ${R.lossPreState(R.K).toFixed(3)}`, COL.ink2], [T_('q3.chgN'), `${fmtInt(ch)} / ${fmtInt(tot)}`, COL.cyan], [T_('q3.chgBf'), `${((rv / tot) * 100).toFixed(1)}%`, COL.rose]];
    rows.forEach(([kk, vv, c], j) => {
      const y = C.y + 66 + j * 28;
      text(g, kk, C.x + 16, y, { size: 11, color: COL.dim, max: C.w * 0.5 });
      text(g, vv, C.x + C.w - 14, y, { size: 11.5, kind: 'mono', weight: 700, color: c, align: 'right' });
    });
    wrap(g, T_('q3.chgNote'), C.x + 16, C.y + C.h - 34, C.w - 32, 13, { size: 10, color: COL.dim, maxLines: 2 });
  }

  // 浮层挡住的那一块告诉引擎，3D 在剩下的区域里取景
  measure(el) {
    if (!el) { if (this.ovKey !== 'off') { this.E.setOverlay(0, 0); this.ovKey = 'off'; } return; }
    const r = el.getBoundingClientRect(), h = this.E.host.getBoundingClientRect();
    const key = `${Math.round(r.top)}|${Math.round(r.bottom)}|${Math.round(h.top)}|${Math.round(h.bottom)}`;
    if (key === this.ovKey || !r.height) return;
    this.ovKey = key;
    const atTop = r.top + r.height / 2 < h.top + h.height / 2;
    if (atTop) this.E.setOverlay(Math.max(0, r.bottom - h.top + 8), 0);
    else this.E.setOverlay(0, Math.max(0, h.bottom - r.top + 8));
  }
}
