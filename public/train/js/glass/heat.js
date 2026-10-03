// 玻璃小模型的小工具：2D 热力图（算式板的卡片里用）、数值的简短写法、参数的名字。
// 发散色（负 = 蓝，正 = 琥珀，零 = 暗）和站里的 divColor、3D 机器的方块（../glass3d/palette.js）同一套。
import { COL, clamp } from '../draw.js';
import { TENSOR_LABEL } from './data.js';

const Z = [12, 20, 36], POS = [255, 182, 92], NEG = [107, 155, 255];
export function divRGB(t) {
  t = clamp(t, -1, 1);
  const e = Math.pow(Math.abs(t), 0.75), c = t >= 0 ? POS : NEG;
  return [Z[0] + (c[0] - Z[0]) * e, Z[1] + (c[1] - Z[1]) * e, Z[2] + (c[2] - Z[2]) * e];
}
const SEQ = [[10, 22, 40], [16, 60, 90], [30, 130, 140], [70, 210, 190], [190, 250, 235]];
function seqRGB(t) {
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

// get(r, c) → 数值；scale：数值除以它再上色（发散色），seq = true 时用顺序色（概率）
// 画法：先把 rows × cols 个颜色写进一张小位图（一格一个像素），再按格子大小放大画出来（最近邻），格子之间描一道暗线
export function heat(g, x, y, rows, cols, cw, ch, get, { scale = 1, seq = false, grid = true, s = 1 } = {}) {
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
  const sm = g.imageSmoothingEnabled;
  g.imageSmoothingEnabled = false;
  g.drawImage(b.cv, x, y, cols * cw, rows * ch);
  g.imageSmoothingEnabled = sm;
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
}

// 数值的简短写法（提示、读数）
export function fnum(v, d = 3) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e4) return (v < 0 ? '−' : '') + Number(a.toPrecision(d)).toString();
  let e = Math.floor(Math.log10(a)), m = a / 10 ** e;
  if (Number(m.toFixed(d - 1)) >= 10) { e += 1; m /= 10; }
  return `${v < 0 ? '−' : ''}${m.toFixed(d - 1)}e${e}`;
}

// 一组数的“典型大小”：绝对值的某个分位（色标 / 发光的满格用；比最大值稳，不会被一个特别大的数压暗全图）
export function absQuantile(arr, q = 0.98, a = 0, b = arr.length) {
  const n = b - a;
  if (n <= 0) return 1;
  const tmp = new Float32Array(n);
  for (let i = 0; i < n; i++) tmp[i] = Math.abs(arr[a + i]);
  tmp.sort();
  return tmp[Math.min(n - 1, Math.floor(q * (n - 1)))] || tmp[n - 1] || 1;
}

// 参数的名字：E[「字」, 列]、W_q[行, 列]、γ₁[维]（提示、标题、调试器里用）
export function paramName(D, gi) {
  const lc = D.locate(gi);
  if (!lc) return '';
  const p = lc.p;
  if (p.name === 'E') return `E[「${D.ch(lc.i)}」, ${lc.j}]`;
  if (p.norm) return `${TENSOR_LABEL[p.name]}[${lc.i}]`;
  return `${TENSOR_LABEL[p.name]}[${lc.i}, ${lc.j}]`;
}
