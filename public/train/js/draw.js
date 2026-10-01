// 画布上的小工具：颜色、字体、圆角卡片、文字、色带、坐标轴。所有尺寸都是世界坐标。

export const COL = {
  bg: '#060d1a', ink: '#e9eff9', ink2: '#b4bed2', dim: '#7a859e', faint: '#4b5572',
  line: 'rgba(150,180,230,0.11)', line2: 'rgba(150,180,230,0.2)', line3: 'rgba(150,180,230,0.34)',
  panel: 'rgba(10,17,31,0.82)', panel2: 'rgba(15,24,42,0.92)',
  cyan: '#5ef0d4', amber: '#ffb65c', rose: '#ff6b93', blue: '#6b9bff', violet: '#b39dff', green: '#8ee07a',
};
export const FONT = {
  sans: '-apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans SC", system-ui, sans-serif',
  mono: '"JetBrains Mono", ui-monospace, Menlo, Consolas, monospace',
  serif: '"BB Serif", "Noto Serif SC", "Source Han Serif SC", "Songti SC", "STSong", serif',
};

export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const seg = (p, a, b) => clamp((p - a) / (b - a), 0, 1);
export const ease = (t) => 1 - Math.pow(1 - clamp(t, 0, 1), 3);
export const easeIO = (t) => { t = clamp(t, 0, 1); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };

export function font(g, size, kind = 'sans', weight = 400) {
  g.font = `${weight} ${size}px ${FONT[kind]}`;
}

export function text(g, s, x, y, { size = 13, color = COL.ink, kind = 'sans', weight = 400, align = 'left', base = 'alphabetic', max = 0 } = {}) {
  font(g, size, kind, weight);
  g.fillStyle = color;
  g.textAlign = align;
  g.textBaseline = base;
  if (max > 0) {
    let t = String(s);
    if (g.measureText(t).width > max) {
      while (t.length > 1 && g.measureText(t + '…').width > max) t = t.slice(0, -1);
      t += '…';
    }
    g.fillText(t, x, y);
    return;
  }
  g.fillText(String(s), x, y);
}

export function measure(g, s, size = 13, kind = 'sans', weight = 400) {
  font(g, size, kind, weight);
  return g.measureText(String(s)).width;
}

// 自动换行（中文按字、英文按词），返回行数
export function wrap(g, s, x, y, w, lh, opt = {}) {
  font(g, opt.size || 13, opt.kind || 'sans', opt.weight || 400);
  const lines = [];
  let cur = '';
  for (const ch of String(s)) {
    if (ch === '\n') { lines.push(cur); cur = ''; continue; }
    if (g.measureText(cur + ch).width > w && cur) { lines.push(cur); cur = ch.trim() ? ch : ''; } else cur += ch;
  }
  if (cur) lines.push(cur);
  const n = opt.maxLines ? Math.min(lines.length, opt.maxLines) : lines.length;
  for (let i = 0; i < n; i++) text(g, i === n - 1 && n < lines.length ? lines[i].slice(0, -1) + '…' : lines[i], x, y + i * lh, opt);
  return n;
}

export function rr(g, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}

// 面板卡片：标题（衬线）+ 小号英文眉标
export function card(g, x, y, w, h, { title = '', eyebrow = '', accent = null, active = false, fill = COL.panel, demo = false, r = 14 } = {}) {
  rr(g, x, y, w, h, r);
  g.fillStyle = fill;
  g.fill();
  g.lineWidth = active ? 1.4 : 1;
  g.strokeStyle = active ? (accent || COL.cyan) : COL.line2;
  g.stroke();
  if (accent && !active) {
    g.fillStyle = accent;
    g.globalAlpha *= 0.85;
    rr(g, x + 14, y, 26, 2, 1);
    g.fill();
    g.globalAlpha /= 0.85;
  }
  let ty = y + 22;
  if (eyebrow) { text(g, eyebrow, x + 14, y + 19, { size: 9.5, kind: 'mono', color: COL.dim }); ty = y + 39; }
  if (title) text(g, title, x + 14, ty, { size: 15, kind: 'serif', weight: 600, color: COL.ink, max: w - 28 });
  if (demo) badge(g, x + w - 14, y + 13, '示意', COL.amber, 'right');
}

export function badge(g, x, y, s, color = COL.cyan, align = 'left', size = 10) {
  const w = measure(g, s, size, 'sans') + 12;
  const bx = align === 'right' ? x - w : x;
  rr(g, bx, y, w, size + 8, (size + 8) / 2);
  g.setLineDash([3, 2]);
  g.strokeStyle = color;
  g.lineWidth = 1;
  g.stroke();
  g.setLineDash([]);
  text(g, s, bx + 6, y + size + 3, { size, color });
  return w;
}

export function pill(g, x, y, w, h, s, { on = false, color = COL.cyan, size = 11, kind = 'sans' } = {}) {
  rr(g, x, y, w, h, h / 2);
  g.fillStyle = on ? hexA(color, 0.14) : 'rgba(255,255,255,0.03)';
  g.fill();
  g.strokeStyle = on ? hexA(color, 0.6) : COL.line2;
  g.lineWidth = 1;
  g.stroke();
  text(g, s, x + w / 2, y + h / 2 + 0.5, { size, color: on ? color : COL.ink2, align: 'center', base: 'middle', kind });
}

export function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

export function mix(a, b, t) {
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
  const c = (s) => Math.round(lerp((pa >> s) & 255, (pb >> s) & 255, t));
  return `rgb(${c(16)},${c(8)},${c(0)})`;
}

// 概率色带：深蓝 → 青 → 浅
const SEQ = [[10, 22, 40], [16, 60, 90], [30, 130, 140], [70, 210, 190], [190, 250, 235]];
export function seqColor(t, a = 1) {
  t = clamp(t, 0, 1) * (SEQ.length - 1);
  const i = Math.min(SEQ.length - 2, Math.floor(t)), f = t - i;
  const c = SEQ[i].map((v, j) => Math.round(lerp(v, SEQ[i + 1][j], f)));
  return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}
// 正负：蓝 ← 0 → 琥珀
export function divColor(t, a = 1) {
  t = clamp(t, -1, 1);
  const z = [12, 20, 36];
  const pos = [255, 182, 92], neg = [107, 155, 255];
  const e = Math.pow(Math.abs(t), 0.75);
  const c = (t >= 0 ? pos : neg).map((v, j) => Math.round(lerp(z[j], v, e)));
  return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}
// 热度（梯度）：深 → 紫 → 玫红 → 琥珀
const HEAT = [[12, 18, 34], [70, 40, 120], [180, 70, 140], [255, 107, 147], [255, 190, 110]];
export function heatColor(t, a = 1) {
  t = clamp(t, 0, 1) * (HEAT.length - 1);
  const i = Math.min(HEAT.length - 2, Math.floor(t)), f = t - i;
  const c = HEAT[i].map((v, j) => Math.round(lerp(v, HEAT[i + 1][j], f)));
  return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}

export function line(g, pts, color, w = 1.5) {
  if (pts.length < 2) return;
  g.beginPath();
  g.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]);
  g.strokeStyle = color;
  g.lineWidth = w;
  g.lineJoin = 'round';
  g.stroke();
}

export function dot(g, x, y, r, color, ring = null) {
  g.beginPath();
  g.arc(x, y, r, 0, Math.PI * 2);
  g.fillStyle = color;
  g.fill();
  if (ring) { g.strokeStyle = ring; g.lineWidth = 1.2; g.stroke(); }
}

export function arrow(g, x1, y1, x2, y2, color, w = 1.4, head = 6) {
  g.beginPath();
  g.moveTo(x1, y1);
  g.lineTo(x2, y2);
  g.strokeStyle = color;
  g.lineWidth = w;
  g.stroke();
  const a = Math.atan2(y2 - y1, x2 - x1);
  g.beginPath();
  g.moveTo(x2, y2);
  g.lineTo(x2 - head * Math.cos(a - 0.45), y2 - head * Math.sin(a - 0.45));
  g.lineTo(x2 - head * Math.cos(a + 0.45), y2 - head * Math.sin(a + 0.45));
  g.closePath();
  g.fillStyle = color;
  g.fill();
}

// 数字格式
export const fmtP = (p) => (p >= 0.995 ? '100%' : p >= 0.1 ? `${(p * 100).toFixed(0)}%` : p >= 0.001 ? `${(p * 100).toFixed(1)}%` : p > 0 ? '<0.1%' : '0%');
export function sci(v, d = 3) {
  if (v === 0) return '0';
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e4) return (v < 0 ? '−' : '') + a.toPrecision(d + 1).replace(/\.?0+$/, (m) => (m.includes('.') ? '' : m));
  const e = Math.floor(Math.log10(a));
  const m = a / 10 ** e;
  return `${v < 0 ? '−' : ''}${m.toFixed(d - 1)}e${e}`;
}
// 上标形式：1.23 × 10⁻⁵
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
export function sciSup(v, d = 3) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e4) return (v < 0 ? '−' : '') + Number(a.toPrecision(d)).toString();
  const e = Math.floor(Math.log10(a));
  const m = a / 10 ** e;
  return `${v < 0 ? '−' : ''}${m.toFixed(d - 1)}×10${String(e).split('').map((c) => SUP[c]).join('')}`;
}
export const fmtInt = (n) => Math.round(n).toLocaleString('zh-CN');
