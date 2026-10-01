// 2D 舞台：一块可以平移缩放的画布（从训练页的 ../../train/js/stage.js 改来）。跟随镜头自动对准当前步骤；
// 拖动 / 滚轮 / 双指进入自由视角。键盘留给游戏（A / D 转向），所以这里没有 WASD。
// 视图的 fixed = true 时（D1 玩）镜头不让拖，免得开车时手一滑把画面拖走。
// 每个视图（scene）在自己的“世界坐标”里画，舞台负责相机、换场过渡（快照淡出 + 缩放）和拾取（hit 区域可以带 drag）。

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

export class Stage {
  constructor(host, canvas, { onFrame, onHover, onPick, onFreeChange } = {}) {
    this.host = host;
    this.cv = canvas;
    this.g = canvas.getContext('2d');
    this.onFrame = onFrame;
    this.onHover = onHover;
    this.onPick = onPick;
    this.onFreeChange = onFreeChange;
    this.scene = null;
    this.cam = { x: 0, y: 0, s: 1 };
    this.tgt = { x: 0, y: 0, s: 1 };
    this.free = false;
    this.goal = null;
    this.insets = { r: 0, b: 0, t: 0 };
    this.hits = [];
    this.hover = null;
    this.snap = document.createElement('canvas');
    this.trans = null;
    this.clock = 0;
    this.bind();
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    this.last = performance.now();
    requestAnimationFrame((t) => this.loop(t));
  }

  resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.dpr = dpr;
    this.W = Math.max(1, this.host.clientWidth);
    this.H = Math.max(1, this.host.clientHeight);
    this.cv.width = Math.round(this.W * dpr);
    this.cv.height = Math.round(this.H * dpr);
    this.portrait = this.W < 700 || this.W / this.H < 0.9;
    if (this.scene) { this.scene.layout(this.env()); this.refit(true); }
  }

  setInsets(r, b, t) { this.insets = { r, b, t }; this.refit(false); }

  // 可视区（没被面板挡住的那一块）
  get view() {
    const { r, b, t } = this.insets;
    const w = Math.max(160, this.W - r), h = Math.max(160, this.H - b - t);
    return { x: 0, y: t, w, h, cx: w / 2, cy: t + h / 2 };
  }

  fit(rect, margin = 0.92) {
    const v = this.view;
    const s = clamp(Math.min(v.w / rect.w, v.h / rect.h) * margin, 0.12, 6);
    return { x: rect.x + rect.w / 2, y: rect.y + rect.h / 2, s };
  }

  env() { return { W: this.W, H: this.H, portrait: this.portrait, view: this.view }; }

  setScene(scene, dir = 0) {
    if (scene === this.scene) return;
    // 把上一帧拍成快照，在新场景上面淡出
    if (this.scene && dir !== 0 && !reduced) {
      this.snap.width = this.cv.width;
      this.snap.height = this.cv.height;
      this.snap.getContext('2d').drawImage(this.cv, 0, 0);
      this.trans = { t0: this.clock, dir };
    } else this.trans = null;
    this.scene = scene;
    scene.layout(this.env());
    this.exitFree(true);
    this.refit(true);
    if (this.trans) this.cam.s *= dir > 0 ? 0.82 : 1.25;
  }

  refit(snap) {
    if (!this.scene || !this.focusRect) return;
    const f = this.fit(this.focusRect);
    this.tgt = f;
    if (snap) this.cam = { ...f };
  }

  /* ---------------- 自由视角 ---------------- */

  enterFree() {
    if (this.free) return;
    this.free = true;
    this.onFreeChange?.(true);
  }

  exitFree(silent = false) {
    this.goal = null;
    if (!this.free) return;
    this.free = false;
    if (!silent) this.onFreeChange?.(false);
    else this.onFreeChange?.(false);
  }

  // 把相机飞到某个世界矩形（点卡片放大）
  flyTo(rect) {
    this.enterFree();
    this.goal = this.fit(rect, 0.94);
  }

  toWorld(px, py) {
    const v = this.view, c = this.cam;
    return { x: (px - v.cx) / c.s + c.x, y: (py - v.cy) / c.s + c.y };
  }

  zoomAt(px, py, f) {
    this.enterFree();
    this.goal = null;
    const before = this.toWorld(px, py);
    this.cam.s = clamp(this.cam.s * f, 0.12, 12);
    const after = this.toWorld(px, py);
    this.cam.x += before.x - after.x;
    this.cam.y += before.y - after.y;
  }

  local(e) {
    const r = this.cv.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  hitAt(px, py) {
    const w = this.toWorld(px, py);
    for (let i = this.hits.length - 1; i >= 0; i--) {
      const h = this.hits[i];
      if (w.x >= h.x && w.x <= h.x + h.w && w.y >= h.y && w.y <= h.y + h.h) return { ...h, wx: w.x, wy: w.y };
    }
    return null;
  }

  bind() {
    const el = this.cv;
    const pts = new Map();
    let mode = null, moved = 0, last = null, pinch = null, dragHit = null;
    const two = () => { const [a, b] = [...pts.values()]; return { d: Math.hypot(a.x - b.x, a.y - b.y), m: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } }; };
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('pointerdown', (e) => {
      el.setPointerCapture(e.pointerId);
      const p = this.local(e);
      pts.set(e.pointerId, p);
      if (pts.size === 1) {
        const h = e.button === 0 ? this.hitAt(p.x, p.y) : null;
        dragHit = h?.info?.drag ? h : null;
        mode = dragHit ? 'drag' : 'pan';
        if (dragHit) dragHit.info.drag(dragHit.wx, dragHit.wy, 'start');
        last = p;
        moved = 0;
        this.downButton = e.button;
      } else if (pts.size === 2) {
        mode = 'pinch';
        pinch = two();
      }
      this.host.classList.add('grabbing');
      this.onHover?.(null);
    });
    el.addEventListener('pointermove', (e) => {
      const p = this.local(e);
      if (!pts.has(e.pointerId)) {
        const h = this.hitAt(p.x, p.y);
        this.hover = h?.info || null;
        this.onHover?.(h, e);
        el.style.cursor = h?.info?.drag ? 'ew-resize' : h?.info?.click ? 'pointer' : '';
        return;
      }
      pts.set(e.pointerId, p);
      if (mode === 'pinch' && pts.size >= 2 && !this.scene?.fixed) {
        const { d, m } = two();
        if (pinch.d > 0 && d > 0) this.zoomAt(m.x, m.y, d / pinch.d);
        this.cam.x -= (m.x - pinch.m.x) / this.cam.s;
        this.cam.y -= (m.y - pinch.m.y) / this.cam.s;
        pinch = { d, m };
        moved += 10;
        return;
      }
      const dx = p.x - last.x, dy = p.y - last.y;
      last = p;
      moved += Math.abs(dx) + Math.abs(dy);
      if (mode === 'drag') { const w = this.toWorld(p.x, p.y); dragHit.info.drag(w.x, w.y, 'move'); return; }
      if (moved < 4 || this.scene?.fixed) return;
      this.enterFree();
      this.goal = null;
      this.cam.x -= dx / this.cam.s;
      this.cam.y -= dy / this.cam.s;
    });
    const end = (e) => {
      if (!pts.has(e.pointerId)) return;
      pts.delete(e.pointerId);
      if (pts.size === 0) {
        if (mode === 'drag') dragHit.info.drag(0, 0, 'end');
        else if (moved < 5 && e.type === 'pointerup' && this.downButton === 0) {
          const p = this.local(e);
          const h = this.hitAt(p.x, p.y);
          this.onPick?.(h?.info || null, e, h);
        }
        mode = null;
        dragHit = null;
        this.host.classList.remove('grabbing');
      } else if (pts.size === 1) { mode = 'pan'; last = [...pts.values()][0]; }
    };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('pointerleave', () => { if (!pts.size) { this.hover = null; this.onHover?.(null); } });
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (this.scene?.fixed) return;
      const p = this.local(e);
      this.zoomAt(p.x, p.y, Math.exp(-clamp(e.deltaY, -60, 60) * 0.0025));
    }, { passive: false });
    el.addEventListener('dblclick', (e) => {
      const p = this.local(e);
      const h = this.hitAt(p.x, p.y);
      if (h?.info?.drag || this.scene?.fixed) return;
      const w = this.toWorld(p.x, p.y);
      this.enterFree();
      const s = clamp(this.cam.s * 2, 0.12, 12);
      this.goal = { x: w.x, y: w.y, s };
    });
  }

  loop(now) {
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    this.clock += dt;
    const st = this.onFrame?.(dt, this.clock);
    if (this.scene && st) {
      // 视图有时要画过一帧才知道自己的取景范围（被占位挡住时还没画过）：拿到的不是有限数就沿用上一个
      const f = this.scene.focus(st, this.env());
      if (f && Number.isFinite(f.x + f.y + f.w + f.h)) this.focusRect = f;
      const target = this.free ? this.goal : this.fit(this.focusRect);
      if (target) {
        const k = 1 - Math.exp(-dt * (this.free ? 5 : 3.4));
        this.cam.x += (target.x - this.cam.x) * k;
        this.cam.y += (target.y - this.cam.y) * k;
        this.cam.s = Math.exp(Math.log(this.cam.s) + (Math.log(target.s) - Math.log(this.cam.s)) * k);
        if (this.free && Math.abs(target.x - this.cam.x) * this.cam.s < 0.5 && Math.abs(Math.log(target.s / this.cam.s)) < 0.003) this.goal = null;
      }
      this.draw(st, dt);
    }
    requestAnimationFrame((t) => this.loop(t));
  }

  draw(st, dt) {
    const g = this.g, dpr = this.dpr, v = this.view, c = this.cam;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, this.cv.width, this.cv.height);
    g.setTransform(dpr * c.s, 0, 0, dpr * c.s, dpr * (v.cx - c.x * c.s), dpr * (v.cy - c.y * c.s));
    this.hits = [];
    const env = {
      t: this.clock, dt, s: c.s, portrait: this.portrait, hover: this.hover,
      hit: (x, y, w, h, info) => this.hits.push({ x, y, w, h, info }),
      // 当前可视区在世界坐标里的范围（用来做跟随屏幕的元素）
      vis: { x: c.x - v.w / 2 / c.s, y: c.y - v.h / 2 / c.s, w: v.w / c.s, h: v.h / c.s },
    };
    g.globalAlpha = 1;
    this.scene.draw(g, st, env);
    // 换场：旧画面的快照在上面放大（下潜）或缩小（上浮）并淡出
    if (this.trans) {
      const e = (this.clock - this.trans.t0) / 0.75;
      if (e >= 1) this.trans = null;
      else {
        const k = 1 - Math.pow(1 - e, 3);
        const sc = this.trans.dir > 0 ? 1 + 0.5 * k : 1 - 0.3 * k;
        g.setTransform(1, 0, 0, 1, 0, 0);
        g.globalAlpha = 1 - k;
        const cx = v.cx * dpr, cy = v.cy * dpr;
        g.translate(cx, cy);
        g.scale(sc, sc);
        g.translate(-cx, -cy);
        g.drawImage(this.snap, 0, 0);
        g.globalAlpha = 1;
      }
    }
  }
}
