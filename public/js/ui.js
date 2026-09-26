import { hashStr, clamp } from './model.js';

export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ESC[c]);

export const hueOf = (text) => hashStr(text, 3) % 360;

// 词元的可见写法：换行显示为 ↵，空格显示为 ␣
export function tokInner(text) {
  if (text === '\n') return '<i class="ws">↵</i>';
  if (/^\s+$/.test(text)) return '<i class="ws">␣</i>';
  if (text.startsWith(' ')) return `<i class="ws">␣</i>${esc(text.slice(1))}`;
  return esc(text);
}
export const tokPlain = (text) => (text === '\n' ? '↵' : /^\s+$/.test(text) ? '␣' : text.replace(/^ /, '␣'));

export function tokHTML(t, cls = '', attrs = '') {
  const text = typeof t === 'string' ? t : t.text;
  return `<span class="tok ${cls}" style="--h:${hueOf(text)}" ${attrs}>${tokInner(text)}</span>`;
}

// 发散色：负值偏蓝，正值偏琥珀
const BASE = [18, 22, 36], POS = [255, 184, 96], NEG = [92, 142, 255];
export function divergeRGB(v, max = 1) {
  const t = clamp(v / max, -1, 1);
  const c = t >= 0 ? POS : NEG;
  const a = Math.pow(Math.abs(t), 0.75);
  return [0, 1, 2].map((i) => Math.round(BASE[i] + (c[i] - BASE[i]) * a));
}
export const diverge = (v, max = 1) => `rgb(${divergeRGB(v, max).join(',')})`;

export const fmtPct = (p) => (p >= 0.995 ? '100%' : p >= 0.1 ? `${(p * 100).toFixed(0)}%` : p >= 0.001 ? `${(p * 100).toFixed(1)}%` : p > 0 ? '<0.1%' : '0%');
export const fmtNum = (n) => Math.round(n).toLocaleString('zh-CN');
export const signed = (v, d = 2) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);

export function countUp(el, to, { dur = 900, from = 0, fmt = fmtNum } = {}) {
  if (reducedMotion) { el.textContent = fmt(to); return; }
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - t0) / dur);
    const e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(from + (to - from) * e);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// 逐字打出一段 HTML（标签整体跳过）。返回取消函数。
export function typeHTML(el, html, speed = 16) {
  if (reducedMotion) { el.innerHTML = html; return () => {}; }
  const parts = html.match(/<[^>]+>|&[a-z#0-9]+;|[\s\S]/gi) || [];
  let i = 0, cancelled = false, out = '';
  el.innerHTML = '';
  const caret = '<span class="caret"></span>';
  const tick = () => {
    if (cancelled) return;
    let n = 0;
    while (i < parts.length && n < 2) {
      out += parts[i];
      if (!parts[i].startsWith('<')) n++;
      i++;
    }
    el.innerHTML = out + (i < parts.length ? caret : '');
    if (i < parts.length) setTimeout(tick, speed);
  };
  tick();
  return () => { cancelled = true; el.innerHTML = html; };
}

// 给场景里的 .rv 元素依次加上入场动画
export function reveal(root, base = 380, step = 70) {
  $$('.rv', root).forEach((el, i) => {
    el.style.animationDelay = `${base + (el.dataset.d ? Number(el.dataset.d) : i * step)}ms`;
  });
}

export function svgEl(tag, attrs = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

// 让 canvas 跟随容器尺寸，返回 {g, w, h}
export function fitCanvas(canvas) {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  // 用布局尺寸而不是 getBoundingClientRect：转场缩放期间后者是被缩放过的
  const w = Math.max(1, Math.round(canvas.clientWidth)), h = Math.max(1, Math.round(canvas.clientHeight));
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, w, h };
}

export function onResize(el, fn) {
  const ro = new ResizeObserver(() => fn());
  ro.observe(el);
  return () => ro.disconnect();
}
