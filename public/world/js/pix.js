// 像素画面和特征图：都先画进离屏画布，再按整数倍放大贴到舞台上（关掉平滑，保持像素感）。
import { PALETTE, W, H } from './game.js';

const N = W * H;

// 一块 w×h 的离屏画布 + ImageData
export class Pix {
  constructor(w = W, h = H) {
    this.c = document.createElement('canvas');
    this.c.width = w;
    this.c.height = h;
    this.g = this.c.getContext('2d');
    this.img = this.g.createImageData(w, h);
    this.key = null;
  }
  // 调色板编号（真实游戏）
  palette(idx, key) {
    if (key !== undefined && key === this.key) return this;
    const d = this.img.data;
    for (let i = 0; i < N; i++) { const c = PALETTE[idx[i]]; d[i * 4] = c[0]; d[i * 4 + 1] = c[1]; d[i * 4 + 2] = c[2]; d[i * 4 + 3] = 255; }
    this.g.putImageData(this.img, 0, 0);
    this.key = key;
    return this;
  }
  // CHW 浮点（模型输出，0..1）
  chw(y, key) {
    if (key !== undefined && key === this.key) return this;
    const d = this.img.data, n = this.c.width * this.c.height;
    for (let i = 0; i < n; i++) {
      d[i * 4] = y[i] * 255; d[i * 4 + 1] = y[n + i] * 255; d[i * 4 + 2] = y[2 * n + i] * 255; d[i * 4 + 3] = 255;
    }
    this.g.putImageData(this.img, 0, 0);
    this.key = key;
    return this;
  }
  // 两幅画面逐像素的差（|真实 − 梦| 三个通道取平均），越亮差得越多
  diff(a, b, key) {
    if (key !== undefined && key === this.key) return this;
    const d = this.img.data;
    for (let i = 0; i < N; i++) {
      const v = (Math.abs(a[i] - b[i]) + Math.abs(a[N + i] - b[N + i]) + Math.abs(a[2 * N + i] - b[2 * N + i])) / 3;
      const t = Math.min(1, v * 2.2);
      d[i * 4] = 14 + 241 * t; d[i * 4 + 1] = 18 + 150 * t * t; d[i * 4 + 2] = 32 + 60 * t; d[i * 4 + 3] = 255;
    }
    this.g.putImageData(this.img, 0, 0);
    this.key = key;
    return this;
  }
  draw(g, x, y, w, h, alpha = 1) {
    const s = g.imageSmoothingEnabled;
    g.imageSmoothingEnabled = false;
    const a = g.globalAlpha;
    g.globalAlpha = a * alpha;
    g.drawImage(this.c, x, y, w, h);
    g.globalAlpha = a;
    g.imageSmoothingEnabled = s;
  }
}

// 激活值的色带：深蓝 → 青 → 浅（不要太亮）
const SEQ = [[9, 16, 30], [14, 48, 72], [24, 104, 116], [62, 176, 164], [168, 232, 214]];
export function seqRGB(t) {
  t = Math.max(0, Math.min(1, t)) * (SEQ.length - 1);
  const i = Math.min(SEQ.length - 2, Math.floor(t)), f = t - i;
  return [0, 1, 2].map((j) => SEQ[i][j] + (SEQ[i + 1][j] - SEQ[i][j]) * f);
}
// 有正有负：蓝 ← 0 → 琥珀
export function divRGB(t) {
  t = Math.max(-1, Math.min(1, t));
  const z = [12, 20, 36], pos = [255, 182, 92], neg = [107, 155, 255];
  const e = Math.pow(Math.abs(t), 0.75);
  return (t >= 0 ? pos : neg).map((v, j) => z[j] + (v - z[j]) * e);
}

// 一层的特征图（C 个 h×w）排成 cols 列的网格，每格之间留 gap 个像素。返回画布和几何信息，供点选、高亮
export function featGrid(act, C, h, w, cols, { gap = 1, signed = false, scale = 0 } = {}) {
  const rows = Math.ceil(C / cols);
  const cw = cols * (w + gap) - gap, ch = rows * (h + gap) - gap;
  const p = new Pix(cw, ch);
  const d = p.img.data;
  for (let i = 0; i < d.length; i += 4) { d[i] = 6; d[i + 1] = 10; d[i + 2] = 20; d[i + 3] = 255; }
  let mx = scale;
  if (!mx) { for (let i = 0; i < C * h * w; i++) mx = Math.max(mx, Math.abs(act[i])); }
  mx = mx || 1;
  for (let c = 0; c < C; c++) {
    const ox = (c % cols) * (w + gap), oy = Math.floor(c / cols) * (h + gap);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const v = act[(c * h + y) * w + x] / mx;
      const rgb = signed ? divRGB(v) : seqRGB(v);
      const k = ((oy + y) * cw + ox + x) * 4;
      d[k] = rgb[0]; d[k + 1] = rgb[1]; d[k + 2] = rgb[2];
    }
  }
  p.g.putImageData(p.img, 0, 0);
  return { pix: p, cols, rows, w, h, gap, cw, ch, max: mx };
}

// 一个向量排成 rows×cols 的方格（LSTM 的门、隐状态：256 = 16×16）
export function vecGrid(v, cols, { signed = true, scale = 1 } = {}) {
  const rows = Math.ceil(v.length / cols);
  const p = new Pix(cols, rows);
  const d = p.img.data;
  for (let i = 0; i < cols * rows; i++) {
    const t = i < v.length ? v[i] / scale : 0;
    const rgb = signed ? divRGB(t) : seqRGB(t);
    d[i * 4] = rgb[0]; d[i * 4 + 1] = rgb[1]; d[i * 4 + 2] = rgb[2]; d[i * 4 + 3] = i < v.length ? 255 : 0;
  }
  p.g.putImageData(p.img, 0, 0);
  return p;
}
