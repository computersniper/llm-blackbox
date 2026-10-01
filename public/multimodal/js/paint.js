// 2D 画布上的小工具：把视觉词元网格上的数值画成叠在图片上的热力图、PCA 颜色、网格线。

export const AMBER = [255, 182, 92];

// 数值 → 颜色：低处透明，高处琥珀色，最高处偏白（克制，不刺眼）
export function heatRGBA(v) {
  v = Math.max(0, Math.min(1, v));
  const a = Math.pow(v, 0.9) * 0.88;
  const w = Math.max(0, v - 0.75) / 0.25;
  return [255, Math.round(150 + 60 * v + 30 * w), Math.round(70 + 40 * v + 110 * w), a];
}

// 把 rows×cols 的数值做成一张小画布（每格一个像素），放大时由浏览器做平滑插值
export function gridCanvas(values, rows, cols, color = heatRGBA) {
  const c = document.createElement('canvas');
  c.width = cols; c.height = rows;
  const g = c.getContext('2d');
  const im = g.createImageData(cols, rows);
  for (let i = 0; i < rows * cols; i++) {
    const [r, gg, b, a] = color(values[i], i);
    im.data[i * 4] = r; im.data[i * 4 + 1] = gg; im.data[i * 4 + 2] = b; im.data[i * 4 + 3] = Math.round(a * 255);
  }
  g.putImageData(im, 0, 0);
  return c;
}

// 归一化到 [0, 1]（按最大值），可选 gamma
export function normMax(vals, gamma = 1) {
  let mx = 0;
  for (const v of vals) if (v > mx) mx = v;
  const out = new Float32Array(vals.length);
  if (mx <= 0) return out;
  for (let i = 0; i < vals.length; i++) out[i] = Math.pow(vals[i] / mx, gamma);
  return out;
}

// 图片调暗后，叠上热力图
export function drawHeat(g, img, W, H, values, rows, cols, { dim = 0.42, smooth = true, gamma = 1 } = {}) {
  g.clearRect(0, 0, W, H);
  g.globalAlpha = 1;
  if (img) {
    g.drawImage(img, 0, 0, W, H);
    g.fillStyle = `rgba(5, 10, 20, ${1 - dim})`;
    g.fillRect(0, 0, W, H);
  }
  if (!values) return;
  const v = normMax(values, gamma);
  const hc = gridCanvas(v, rows, cols);
  g.imageSmoothingEnabled = smooth;
  g.imageSmoothingQuality = 'high';
  g.drawImage(hc, 0, 0, W, H);
  g.imageSmoothingEnabled = true;
}

// 画网格线：每 step 个单元一条
export function drawGrid(g, W, H, rows, cols, color = 'rgba(200, 220, 255, .28)', width = 1) {
  g.strokeStyle = color;
  g.lineWidth = width;
  g.beginPath();
  for (let c = 0; c <= cols; c++) { const x = Math.round((c / cols) * W) + 0.5; g.moveTo(x, 0); g.lineTo(x, H); }
  for (let r = 0; r <= rows; r++) { const y = Math.round((r / rows) * H) + 0.5; g.moveTo(0, y); g.lineTo(W, y); }
  g.stroke();
}

export function rgbOf(arr, i, k = 0.85) {
  return [arr[i * 3] * k, arr[i * 3 + 1] * k, arr[i * 3 + 2] * k];
}
