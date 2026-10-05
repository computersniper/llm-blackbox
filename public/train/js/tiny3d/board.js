// 3D 舞台上的 2D 浮层（每块一张小画布，悬停有提示，曲线能拖动）：
//   监视器（左上角，D1–D4）：训练曲线；D1 里按住拖动换检查点；
//   左边一列（D1，宽屏）：打印机打出的诗、留出的《登鹳雀楼》逐字概率；
//   算式板（底部，D5）：这个权重的值 / 梯度或 AdamW 算式，加上它的一生。
import { drawCurves, drawReceipt, drawHeld, drawWhere, drawAdam5, drawLife5 } from './cards.js';
import { waitBox } from '../draw.js';
import { L } from '../lang.js';

const small = () => matchMedia('(max-width: 900px)').matches;

class Pane {
  constructor(el, onTip) {
    this.el = el;
    this.cv = document.createElement('canvas');
    el.append(this.cv);
    this.g = this.cv.getContext('2d');
    this.hits = [];
    this.w = this.h = 0;
    let drag = null;
    const at = (e) => { const r = this.cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    const find = (p) => { for (let i = this.hits.length - 1; i >= 0; i--) { const h = this.hits[i]; if (p.x >= h.x && p.x <= h.x + h.w && p.y >= h.y && p.y <= h.y + h.h) return h; } return null; };
    this.cv.addEventListener('pointerdown', (e) => {
      const p = at(e), h = find(p);
      if (h?.info?.drag) { drag = h; this.cv.setPointerCapture(e.pointerId); h.info.drag(p.x, p.y, 'start'); e.preventDefault(); }
    });
    this.cv.addEventListener('pointermove', (e) => {
      const p = at(e);
      if (drag) { drag.info.drag(p.x, p.y, 'move'); return; }
      const h = find(p);
      const html = h?.info ? (h.info.tipAt ? h.info.tipAt(p.x, p.y) : typeof h.info.tip === 'function' ? h.info.tip() : h.info.tip) : null;
      onTip(html, e);
      this.cv.style.cursor = h?.info?.drag ? 'ew-resize' : '';
    });
    const end = () => { if (drag) { drag.info.drag(0, 0, 'end'); drag = null; } };
    this.cv.addEventListener('pointerup', end);
    this.cv.addEventListener('pointercancel', end);
    this.cv.addEventListener('pointerleave', () => { if (!drag) onTip(null); });
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

  draw(fn, t) {
    const g = this.g;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, this.cv.width, this.cv.height);
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.hits = [];
    fn(g, { t, s: 1, hit: (x, y, w, h, info) => this.hits.push({ x, y, w, h, info }) });
  }
}

export class TinyBoard {
  constructor(host, { onTip, scrub, E }) {
    this.E = E;
    this.scrub = scrub;
    const mk = (cls, label) => { const el = document.createElement('section'); el.className = cls; el.hidden = true; el.setAttribute('aria-label', label); host.append(el); return el; };
    this.hudEl = mk('t3hud', L('训练曲线', 'Training curve'));
    this.sideEl = mk('t3side', L('打印机和留出的诗', 'Printer and held-out poem'));
    this.boardEl = mk('t3board', L('算式板', 'Formula board'));
    this.hud = new Pane(this.hudEl, onTip);
    this.side = new Pane(this.sideEl, onTip);
    this.board = new Pane(this.boardEl, onTip);
    this.ovKey = '';
    this.insetL = 0;
    this.printK = -1;
    this.printT0 = 0;
    new ResizeObserver(() => { this.ovKey = ''; }).observe(this.boardEl);
    addEventListener('resize', () => { this.ovKey = ''; });
  }

  hide() { this.hudEl.hidden = this.sideEl.hidden = this.boardEl.hidden = true; document.body.classList.remove('t3-board'); this.E.setOverlay(0, 0); this.ovKey = ''; this.insetL = 0; }

  update(st, X, t) {
    const v = st.view, sm = small();
    const free = (this.E.w || 0) - (this.E.insetR || 0);
    const showHud = v !== 't-param';
    const showSide = v === 't-run' && !sm && free >= 900;
    const showBoard = v === 't-param';
    this.hudEl.hidden = !showHud;
    this.sideEl.hidden = !showSide;
    this.boardEl.hidden = !showBoard;
    document.body.classList.toggle('t3-board', showBoard);
    if (showHud) this.drawHud(st, X, t);
    if (showSide) this.drawSide(st, X, t);
    if (showBoard) this.drawBoard(st, X, t);
    // 宽屏的训练全程：左边一列卡片挡住的那一条，3D 在剩下的区域里取景
    this.insetL = showSide ? this.sideEl.getBoundingClientRect().right - this.E.host.getBoundingClientRect().left + 10 : 0;
    this.measure(showBoard ? this.boardEl : showHud && (sm || v !== 't-run') ? this.hudEl : null);
  }

  drawHud(st, X, t) {
    const sm = small(), compact = sm || st.view !== 't-run';
    this.hudEl.classList.toggle('compact', compact && !sm);
    const w = this.hudEl.clientWidth || (sm ? 300 : 340), h = sm ? 84 : compact ? 104 : 170;
    this.hud.size(w, h);
    const k = st.k, m = X.meta;
    const tt = st.view === 't-run' && k < X.K - 1 ? m.ckpts[k].t + (m.ckpts[k + 1].t - m.ckpts[k].t) * Math.min(1, st.p) : m.ckpts[k].t;
    this.hud.draw((g, env) => drawCurves(g, { x: 0.5, y: 0.5, w: w - 1, h: h - 1 }, X, tt, k, env, { drag: st.view === 't-run' ? this.scrub : null, compact }), t);
  }

  drawSide(st, X, t) {
    const w = this.sideEl.clientWidth || 340;
    const rh = 262, hh = 132, gap = 10, h = rh + gap + hh;
    this.side.size(w, h);
    const k = st.k;
    if (this.printK !== k) { this.printK = k; this.printT0 = t; }
    const age = st.playing ? (t - this.printT0) * Math.max(1, st.speed) * 40 : 999;
    this.side.draw((g, env) => {
      drawReceipt(g, { x: 0.5, y: 0.5, w: w - 1, h: rh }, X, k, age, env);
      drawHeld(g, { x: 0.5, y: rh + gap + 0.5, w: w - 1, h: hh - 1 }, X, k, env);
    }, t);
  }

  drawBoard(st, X, t) {
    const sm = small(), s = st.step, k = st.k, f = s.f;
    const W = this.boardEl.clientWidth || 800;
    const ready = X.has('st', k);
    const main = s.mi === 'w' || s.mi === 'g' ? drawWhere : drawAdam5;
    const p = this.ap ?? 1;
    if (sm) {
      const h1 = 300, h2 = 360;
      this.board.size(W, h1 + 10 + h2);
      this.board.draw((g, env) => {
        if (!ready) { waitBox(g, 0.5, 0.5, W - 1, h1, env, st.wait, { withCard: true, label: L('正在载入这一步的真实记录…', 'Loading the real record of this step…') }); return; }
        main(g, { x: 0.5, y: 0.5, w: W - 1, h: h1 }, X, k, f, s.mi, p, env);
        drawLife5(g, { x: 0.5, y: h1 + 10, w: W - 1, h: h2 - 1 }, X, f, X.step(k), env, { wide: false });
      }, t);
    } else {
      const h = 290, mw = Math.min(520, Math.max(400, W * 0.46));
      this.board.size(W, h);
      this.board.draw((g, env) => {
        if (!ready) { waitBox(g, 0.5, 0.5, W - 1, h - 1, env, st.wait, { withCard: true, label: L('正在载入这一步的真实记录…', 'Loading the real record of this step…') }); return; }
        main(g, { x: 0.5, y: 0.5, w: mw, h: h - 1 }, X, k, f, s.mi, p, env);
        drawLife5(g, { x: mw + 10, y: 0.5, w: W - mw - 10.5, h: h - 1 }, X, f, X.step(k), env, { wide: W - mw > 620 });
      }, t);
    }
  }

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
