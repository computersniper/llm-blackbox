// 3D 舞台上的两块 2D 浮层：
//   监视器（左上角，D1–D4）：训练曲线，标着现在是第几步、和上一帧隔了几步；D1 里按住拖动换帧；
//   算式板（底部，D0 和 D5）：初始化时是初始值的直方图和三种初始化的对照；一个参数时是“它乘了谁 / 梯度从哪来 / AdamW 算式”和它一生的曲线。
// 画法和训练页其余的卡片一样（../draw.js），每块浮层一张小画布，悬停有提示、能拖动。
import { drawLoss, drawHist, drawRuns, drawLife, drawMul, drawChain, drawAdam, drawWait } from './cards.js';
import { defaultParam } from '../glass/math.js';
import { L } from '../lang.js';

const small = () => matchMedia('(max-width: 900px)').matches;

// 一张小画布：画布坐标 = CSS 像素；hit(x, y, w, h, info) 注册悬停 / 点击 / 拖动区域
class Pane {
  constructor(el, onTip) {
    this.el = el;
    this.cv = document.createElement('canvas');
    el.append(this.cv);
    this.g = this.cv.getContext('2d');
    this.hits = [];
    this.onTip = onTip;
    this.w = this.h = 0;
    let drag = null;
    const at = (e) => { const r = this.cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    const find = (p) => { for (let i = this.hits.length - 1; i >= 0; i--) { const h = this.hits[i]; if (p.x >= h.x && p.x <= h.x + h.w && p.y >= h.y && p.y <= h.y + h.h) return h; } return null; };
    this.cv.addEventListener('pointerdown', (e) => {
      const p = at(e), h = find(p);
      if (h?.info?.drag) { drag = h; this.cv.setPointerCapture(e.pointerId); h.info.drag(p.x, p.y, 'start'); e.preventDefault(); }
      else if (h?.info?.act) h.info.act(p);
    });
    this.cv.addEventListener('pointermove', (e) => {
      const p = at(e);
      if (drag) { drag.info.drag(p.x, p.y, 'move'); return; }
      const h = find(p);
      const html = h?.info ? (h.info.tipAt ? h.info.tipAt(p.x, p.y) : h.info.tip) : null;
      this.onTip(html, e);
      this.cv.style.cursor = h?.info?.drag ? 'ew-resize' : h?.info?.act ? 'pointer' : '';
    });
    const end = () => { if (drag) { drag.info.drag(0, 0, 'end'); drag = null; } };
    this.cv.addEventListener('pointerup', end);
    this.cv.addEventListener('pointercancel', end);
    this.cv.addEventListener('pointerleave', () => { if (!drag) this.onTip(null); });
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

export class GlassBoard {
  constructor(host, { onTip, scrub, E }) {
    this.E = E;
    this.scrub = scrub;
    const mk = (cls, label) => { const el = document.createElement('section'); el.className = cls; el.hidden = true; el.setAttribute('aria-label', label); host.append(el); return el; };
    this.hudEl = mk('ghud', L('训练曲线', 'Training curve'));
    this.boardEl = mk('gboard', L('算式板', 'Formula board'));
    this.hud = new Pane(this.hudEl, onTip);
    this.board = new Pane(this.boardEl, onTip);
    this.ovKey = '';
    new ResizeObserver(() => { this.ovKey = ''; }).observe(this.boardEl);
    addEventListener('resize', () => { this.ovKey = ''; });
  }

  hide() { this.hudEl.hidden = this.boardEl.hidden = true; document.body.classList.remove('g-board'); this.E.setOverlay(0, 0); this.ovKey = ''; }

  update(st, D, ctx, t, ap) {
    const v = st.view;
    const showHud = v === 'g-run' || v === 'g-step' || v === 'g-op' || v === 'g-mat';
    const showBoard = v === 'g-init' || v === 'g-param';
    this.hudEl.hidden = !showHud;
    this.boardEl.hidden = !showBoard;
    document.body.classList.toggle('g-board', showBoard);
    if (showHud) this.drawHud(st, D, t);
    if (showBoard) this.drawBoard(st, D, ctx, t, ap);
    // 近景时监视器挡住的那一条也让出来（训练全程的远景不让：机器在正中，左上角本来就空）
    this.measure(showBoard ? this.boardEl : showHud && (v !== 'g-run' || small()) ? this.hudEl : null);
  }

  drawHud(st, D, t) {
    const sm = small(), compact = sm || st.view !== 'g-run';
    this.hudEl.classList.toggle('compact', compact && !sm);
    const w = this.hudEl.clientWidth || (sm ? 300 : 330), h = sm ? 84 : compact ? 104 : 150;
    this.hud.size(w, h);
    const k = st.k;
    const tt = st.view === 'g-run' && k < D.NF - 1 ? D.FR[k] + (D.FR[k + 1] - D.FR[k]) * Math.min(1, st.p) : D.FR[k];
    this.hud.draw((g, env) => drawLoss(g, { x: 0.5, y: 0.5, w: w - 1, h: h - 1 }, D, tt, k, env, { drag: st.view === 'g-run' ? this.scrub : null, compact }), t);
  }

  drawBoard(st, D, ctx, t, ap) {
    const sm = small(), s = st.step;
    const W = this.boardEl.clientWidth || 800;
    const stA = { ...st, p: ap };
    if (st.view === 'g-init') {
      const which = s.sub === 'zero' ? 'zero' : s.sub === 'big' ? 'big' : 'normal';
      if (sm) {
        const h = 214;
        this.board.size(W, h);
        this.board.draw((g, env) => {
          if (s.sub === 'zero' || s.sub === 'big') drawRuns(g, { x: 0.5, y: 0.5, w: W - 1, h: h - 1 }, D, which, true, env);
          else drawHist(g, { x: 0.5, y: 0.5, w: W - 1, h: h - 1 }, D, which, s.sub === 'hist');
        }, t);
      } else {
        const h = 214, cw = (W - 10) / 2;
        this.board.size(W, h);
        this.board.draw((g, env) => {
          drawHist(g, { x: 0.5, y: 0.5, w: cw, h: h - 1 }, D, which, s.sub === 'hist' || s.sub === 'big');
          drawRuns(g, { x: cw + 10, y: 0.5, w: cw - 0.5, h: h - 1 }, D, which, s.sub === 'zero' || s.sub === 'big', env);
        }, t);
      }
      return;
    }
    // 一个参数
    const k = st.k;
    const gi = ctx.gsel != null && D.locate(ctx.gsel)?.p.name === s.t ? ctx.gsel : defaultParam(D, s.t);
    const main = s.mi === 'mul' ? drawMul : s.mi === 'chain' ? drawChain : drawAdam;
    const ready = D.has('w', k) && D.has('f', k);
    if (sm) {
      const h1 = s.mi === 'write' || s.ph === 'upd' ? 470 : 430, h2 = 520;
      this.board.size(W, h1 + 10 + h2);
      this.board.draw((g, env) => {
        if (!ready) { drawWait(g, { x: 0.5, y: 0.5, w: W - 1, h: h1 }, env, st.wait, L('正在载入这一步的真实记录…', 'Loading the real record of this step…')); return; }
        main(g, { x: 0.5, y: 0.5, w: W - 1, h: h1 }, D, k, gi, stA, env);
        drawLife(g, { x: 0.5, y: h1 + 10, w: W - 1, h: h2 - 1 }, D, gi, D.FR[k], env, { wide: false });
      }, t);
    } else {
      const h = 330, mw = Math.min(470, Math.max(380, W * 0.42));
      this.board.size(W, h);
      this.board.draw((g, env) => {
        if (!ready) { drawWait(g, { x: 0.5, y: 0.5, w: W - 1, h: h - 1 }, env, st.wait, L('正在载入这一步的真实记录…', 'Loading the real record of this step…')); return; }
        main(g, { x: 0.5, y: 0.5, w: mw, h: h - 1 }, D, k, gi, stA, env);
        drawLife(g, { x: mw + 10, y: 0.5, w: W - mw - 10.5, h: h - 1 }, D, gi, D.FR[k], env, { wide: W - mw > 640 });
      }, t);
    }
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
