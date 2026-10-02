// 拖动调整左右面板的宽度（推理 / 训练 / 多模态 / 智能体四个页面共用）。
// 面板宽度来自 CSS 变量（--chat-w / --side-w / --dbg-w）：这里只改 <body> 上的这个变量，
// 控制条、提示、算式板、舞台边距这些依赖它的东西自己跟着走；改完调一次 onChange（重新算舞台的遮挡边距）。
// 手柄什么时候出现交给样式表（app.css / train.css 末尾）：聊天模式、调试器收起、窄屏（≤900px，面板是覆盖式的）都藏起来。

import { L as tr } from './i18n.js';

const NARROW = matchMedia('(max-width: 900px)');
const STEP = 16;
const DEFAULTS = { left: { min: 280, max: 720 }, right: { min: 260, max: 680 } };

const load = (key) => { try { const v = JSON.parse(localStorage.getItem(key)); return v && typeof v === 'object' && !Array.isArray(v) ? v : {}; } catch { return {}; } };
const store = (key, v) => { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* 忽略 */ } };
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

/**
 * page：存进 localStorage 用的页面名；
 * left / right：{ el, v, name, min?, max? }，el 是面板（选择器或元素），v 是它的宽度变量，name 用在无障碍标签里；
 * reserve：两侧面板之外，舞台至少要留出的宽度（含调试器外侧的边距）；
 * onChange：宽度变了之后调用（拖动时每帧最多一次）。
 */
export function initPanes(page, { left, right, reserve = 500, onChange } = {}) {
  const KEY = `blackbox:panes:${page}`;
  const body = document.body;
  const saved = load(KEY);
  const panes = [];
  for (const [side, o] of [['left', left], ['right', right]]) {
    const el = typeof o?.el === 'string' ? document.querySelector(o.el) : o?.el;
    if (el) panes.push({ ...DEFAULTS[side], ...o, side, el });
  }
  if (!panes.length) return;

  const width = (p) => parseFloat(getComputedStyle(body).getPropertyValue(p.v)) || p.el.offsetWidth;
  // 可调范围：另一侧面板按它当前的宽度算（不管它此刻收没收起，免得展开时把舞台挤没）；
  // keep 是现在的宽度：窗口变窄后已经超出上限的面板，拖动时只能往回收，不会一下子跳过去
  const range = (p, keep = 0) => {
    const others = panes.reduce((s, q) => (q === p ? s : s + width(q)), 0);
    const hi = Math.min(p.max, Math.max(innerWidth - others - reserve, keep));
    return [p.min, Math.max(p.min, hi)];
  };
  const set = (p, w) => {
    const val = `${Math.round(w)}px`;
    if (body.style.getPropertyValue(p.v) !== val) body.style.setProperty(p.v, val);
  };
  const aria = (p) => {
    const w = Math.round(width(p)), [lo, hi] = range(p, w);
    p.grip.setAttribute('aria-valuemin', lo);
    p.grip.setAttribute('aria-valuemax', Math.round(hi));
    p.grip.setAttribute('aria-valuenow', w);
    p.grip.setAttribute('aria-valuetext', tr(`${w} 像素`, `${w} pixels`));
  };
  // 按存下来的宽度重新摆一遍（窗口大小变了也走这里）：窄屏一律用样式表里的默认值
  const fit = () => {
    for (const p of panes) {
      if (NARROW.matches || !Number.isFinite(saved[p.v])) body.style.removeProperty(p.v);
      else set(p, saved[p.v]);
    }
    if (!NARROW.matches) for (const p of panes) if (Number.isFinite(saved[p.v])) set(p, clamp(saved[p.v], ...range(p)));
    panes.forEach(aria);
  };

  for (const p of panes) {
    const g = (p.grip = document.createElement('div'));
    const sign = p.side === 'left' ? 1 : -1;   // 手柄往右挪：左栏变宽、调试器变窄
    g.className = `pane-grip ${p.side === 'left' ? 'l' : 'r'}`;
    g.tabIndex = 0;
    g.setAttribute('role', 'separator');
    g.setAttribute('aria-orientation', 'vertical');
    g.setAttribute('aria-label', tr(`调整${p.name || '面板'}宽度`, `Resize the ${p.name || 'panel'}`));
    if (p.el.id) g.setAttribute('aria-controls', p.el.id);
    g.title = tr('拖动调整宽度，双击恢复默认', 'Drag to resize, double-click to reset');
    p.el.append(g);

    let drag = null, want = 0, raf = 0;
    const flush = () => {
      cancelAnimationFrame(raf);
      raf = 0;
      if (!drag || !want) return;
      set(p, want);
      onChange?.();
      aria(p);
    };
    g.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || NARROW.matches) return;
      e.preventDefault();   // 不选中文字、不抢焦点（拖完按 ← → 仍然是单步）
      const w0 = width(p);
      drag = { id: e.pointerId, x0: e.clientX, w0, r: range(p, w0) };
      want = 0;
      g.setPointerCapture(e.pointerId);   // 指针移出窗口也能收到抬起
      g.classList.add('on');
      body.classList.add('pane-drag');
    });
    g.addEventListener('pointermove', (e) => {
      if (!drag || e.pointerId !== drag.id) return;
      want = clamp(drag.w0 + sign * (e.clientX - drag.x0), ...drag.r);
      if (!raf) raf = requestAnimationFrame(flush);
    });
    const end = (e) => {
      if (!drag || e.pointerId !== drag.id) return;
      flush();
      if (want && Math.round(want) !== Math.round(drag.w0)) { saved[p.v] = Math.round(want); store(KEY, saved); }
      drag = null;
      g.classList.remove('on');
      body.classList.remove('pane-drag');
      if (g.hasPointerCapture(e.pointerId)) g.releasePointerCapture(e.pointerId);
    };
    g.addEventListener('pointerup', end);
    g.addEventListener('pointercancel', end);
    g.addEventListener('lostpointercapture', end);
    // 双击：恢复样式表里的默认宽度
    g.addEventListener('dblclick', () => {
      delete saved[p.v];
      store(KEY, saved);
      fit();
      onChange?.();
    });
    // 键盘：← → 挪手柄（Shift 一次挪 4 倍），Home / End 到最窄 / 最宽
    g.addEventListener('keydown', (e) => {
      const w = width(p), [lo, hi] = range(p, w), d = e.shiftKey ? STEP * 4 : STEP;
      let n;
      if (e.key === 'ArrowRight') n = w + sign * d;
      else if (e.key === 'ArrowLeft') n = w - sign * d;
      else if (e.key === 'Home') n = lo;
      else if (e.key === 'End') n = hi;
      else return;
      e.preventDefault();
      e.stopPropagation();   // 别让页面把方向键当成单步
      n = Math.round(clamp(n, lo, hi));
      set(p, n);
      saved[p.v] = n;
      store(KEY, saved);
      onChange?.();
      aria(p);
    });
    g.addEventListener('focus', () => aria(p));
  }

  fit();
  addEventListener('resize', () => { fit(); onChange?.(); });
}
