export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smooth = (t) => t * t * (3 - 2 * t);
export const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ESC[c]);

export function hashStr(str, seed = 0) {
  let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return 4294967296 * (2097151 & h2) + (h1 >>> 0);
}
export const hueOf = (text) => hashStr(text, 3) % 360;

// 词元的可见写法：换行显示为 ↵，空格显示为 ␣
export function tokInner(text) {
  if (text === '\n') return '<i class="ws">↵</i>';
  if (/^\n+$/.test(text)) return `<i class="ws">${'↵'.repeat(Math.min(text.length, 3))}</i>`;
  if (/^\s+$/.test(text)) return '<i class="ws">␣</i>';
  let out = esc(text.replace(/\n/g, '↵'));
  if (text.startsWith(' ')) out = `<i class="ws">␣</i>${esc(text.slice(1).replace(/\n/g, '↵'))}`;
  return out;
}
export const tokPlain = (text) => (text === '\n' ? '↵' : /^\s+$/.test(text) ? '␣' : text.replace(/^ /, '␣').replace(/\n/g, '↵'));
export const shortSpecial = (s) => s.replace(/^<\|(.+)\|>$/, '⟨$1⟩');

export function tokHTML(text, cls = '', hue = null) {
  const style = hue == null ? '' : ` style="--h:${hue}"`;
  const sp = /^<\|.*\|>$|^<\/?think>$/.test(text);
  return `<span class="tok ${cls}${sp ? ' sp' : ''}"${style}>${sp ? esc(shortSpecial(text)) : tokInner(text)}</span>`;
}

export const fmtPct = (p) => (p >= 0.995 ? '100%' : p >= 0.1 ? `${(p * 100).toFixed(0)}%` : p >= 0.001 ? `${(p * 100).toFixed(1)}%` : p > 0 ? '<0.1%' : '0%');
export const fmtNum = (n) => Math.round(n).toLocaleString('zh-CN');
export const signed = (v, d = 2) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
