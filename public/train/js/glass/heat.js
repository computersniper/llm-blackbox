// 玻璃小模型用的热力图：每个格子一个真实的数。发散色（负 = 蓝，正 = 琥珀，零 = 暗），和站里的 divColor 同一套。
// 画法：先把 rows × cols 个颜色写进一张小位图（一格一个像素），再按格子大小放大画出来（最近邻），格子之间描一道暗线。
import { COL, text, rr, hexA, clamp } from '../draw.js';

const Z = [12, 20, 36], POS = [255, 182, 92], NEG = [107, 155, 255];
export function divRGB(t) {
  t = clamp(t, -1, 1);
  const e = Math.pow(Math.abs(t), 0.75), c = t >= 0 ? POS : NEG;
  return [Z[0] + (c[0] - Z[0]) * e, Z[1] + (c[1] - Z[1]) * e, Z[2] + (c[2] - Z[2]) * e];
}
const SEQ = [[10, 22, 40], [16, 60, 90], [30, 130, 140], [70, 210, 190], [190, 250, 235]];
export function seqRGB(t) {
  t = clamp(t, 0, 1) * (SEQ.length - 1);
  const i = Math.min(SEQ.length - 2, Math.floor(t)), f = t - i;
  return SEQ[i].map((v, j) => v + (SEQ[i + 1][j] - v) * f);
}
export const rgb = (c, a = 1) => `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a})`;

const pool = new Map();
function bitmap(rows, cols) {
  const key = `${rows}x${cols}`;
  let b = pool.get(key);
  if (!b) {
    const cv = document.createElement('canvas');
    cv.width = cols;
    cv.height = rows;
    const ctx = cv.getContext('2d');
    b = { cv, ctx, img: ctx.createImageData(cols, rows) };
    pool.set(key, b);
  }
  return b;
}

// get(r, c) → 数值；scale：数值除以它再上色（发散色），seq = true 时用顺序色（概率）。alpha：整体透明度
export function heat(g, x, y, rows, cols, cw, ch, get, { scale = 1, seq = false, grid = true, alpha = 1, s = 1 } = {}) {
  const b = bitmap(rows, cols), d = b.img.data;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = get(r, c);
      const col = seq ? seqRGB(v / scale) : divRGB(v / scale);
      const o = (r * cols + c) * 4;
      d[o] = col[0]; d[o + 1] = col[1]; d[o + 2] = col[2]; d[o + 3] = 255;
    }
  }
  b.ctx.putImageData(b.img, 0, 0);
  const pa = g.globalAlpha;
  g.globalAlpha = pa * alpha;
  const sm = g.imageSmoothingEnabled;
  g.imageSmoothingEnabled = false;
  g.drawImage(b.cv, x, y, cols * cw, rows * ch);
  g.imageSmoothingEnabled = sm;
  // 格子线：屏幕上每格够大时才描（否则糊成一片）
  if (grid && Math.min(cw, ch) * s >= 3.2) {
    g.strokeStyle = 'rgba(6,13,26,0.7)';
    g.lineWidth = Math.min(1.2, 0.9 / s);
    g.beginPath();
    for (let c = 1; c < cols; c++) { g.moveTo(x + c * cw, y); g.lineTo(x + c * cw, y + rows * ch); }
    for (let r = 1; r < rows; r++) { g.moveTo(x, y + r * ch); g.lineTo(x + cols * cw, y + r * ch); }
    g.stroke();
  }
  g.strokeStyle = COL.line2;
  g.lineWidth = 1 / Math.max(1, s);
  g.strokeRect(x, y, cols * cw, rows * ch);
  g.globalAlpha = pa;
}

// 世界坐标 → 第几行第几列
export function cellAt(wx, wy, x, y, rows, cols, cw, ch) {
  const c = Math.floor((wx - x) / cw), r = Math.floor((wy - y) / ch);
  return r >= 0 && r < rows && c >= 0 && c < cols ? { r, c } : null;
}

// 选中 / 悬停的那一格：白框
export function mark(g, x, y, r, c, cw, ch, color = '#ffffff', w = 1.6) {
  g.strokeStyle = color;
  g.lineWidth = w;
  g.strokeRect(x + c * cw - 0.5, y + r * ch - 0.5, cw + 1, ch + 1);
}

// 小色标：−scale … 0 … +scale
export function legend(g, x, y, w, scale, fmt, { seq = false, label = '' } = {}) {
  const n = 40, h = 6;
  for (let i = 0; i < n; i++) {
    const t = seq ? i / (n - 1) : (i / (n - 1)) * 2 - 1;
    g.fillStyle = rgb(seq ? seqRGB(t) : divRGB(t));
    g.fillRect(x + (i / n) * w, y, w / n + 0.5, h);
  }
  g.strokeStyle = COL.line2;
  g.lineWidth = 1;
  g.strokeRect(x, y, w, h);
  const o = { size: 9, kind: 'mono', color: COL.dim };
  if (seq) {
    text(g, '0', x, y + h + 10, o);
    text(g, fmt(scale), x + w, y + h + 10, { ...o, align: 'right' });
  } else {
    text(g, `−${fmt(scale)}`, x, y + h + 10, o);
    text(g, '0', x + w / 2, y + h + 10, { ...o, align: 'center' });
    text(g, `+${fmt(scale)}`, x + w, y + h + 10, { ...o, align: 'right' });
  }
  if (label) text(g, label, x - 6, y + h, { size: 9.5, color: COL.dim, align: 'right' });
}

// 一个张量的“标题牌”：名字 + 形状 + 个数
export function tag(g, x, y, name, sub, { color = COL.ink, size = 12, subColor = COL.dim } = {}) {
  text(g, name, x, y, { size, kind: 'mono', weight: 600, color });
  if (sub) text(g, sub, x, y + size + 3, { size: 9.5, kind: 'mono', color: subColor });
}

// 数值的简短写法（热力图提示、格子旁的读数）
export function fnum(v, d = 3) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e4) return (v < 0 ? '−' : '') + Number(a.toPrecision(d)).toString();
  let e = Math.floor(Math.log10(a)), m = a / 10 ** e;
  if (Number(m.toFixed(d - 1)) >= 10) { e += 1; m /= 10; }
  return `${v < 0 ? '−' : ''}${m.toFixed(d - 1)}e${e}`;
}

// 一组数的“典型大小”：绝对值的某个分位（色标用；比最大值稳，不会被一个特别大的数压暗全图）
export function absQuantile(arr, q = 0.98, a = 0, b = arr.length) {
  const n = b - a;
  if (n <= 0) return 1;
  const tmp = new Float32Array(n);
  for (let i = 0; i < n; i++) tmp[i] = Math.abs(arr[a + i]);
  tmp.sort();
  return tmp[Math.min(n - 1, Math.floor(q * (n - 1)))] || tmp[n - 1] || 1;
}

export function frame(g, x, y, w, h, color, a = 0.9, lw = 1.4) {
  rr(g, x, y, w, h, 4);
  g.strokeStyle = hexA(color, a);
  g.lineWidth = lw;
  g.stroke();
}
