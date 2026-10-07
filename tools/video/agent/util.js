// 小工具：缓动、插值、二维“镜头”的关键帧（不依赖 three.js，智能体视频全是 DOM / 2D canvas）。
export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const seg = (t, a, b) => clamp((t - a) / (b - a));
export const smooth = (t) => { t = clamp(t); return t * t * (3 - 2 * t); };
export const smoother = (t) => { t = clamp(t); return t * t * t * (t * (t * 6 - 15) + 10); };
export const easeOut = (t) => 1 - Math.pow(1 - clamp(t), 3);
export const easeIn = (t) => Math.pow(clamp(t), 3);
export const easeInOut = (t) => { t = clamp(t); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
// 一段时间 [t0, t1] 内的可见度：渐入 fin 秒、渐出 fout 秒
export const vis = (t, t0, t1, fin = 0.35, fout = 0.35) => (t < t0 || t > t1 ? 0 : Math.min(fin > 0 ? smooth((t - t0) / fin) : 1, fout > 0 ? smooth((t1 - t) / fout) : 1));

export function el(tag, cls, html, parent) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  if (parent) parent.append(e);
  return e;
}

// 单调三次插值（Fritsch–Carlson），端点斜率为 0：关键帧之间起停自然、不会冲过头
export function pchip(xs, ys) {
  const n = xs.length;
  if (n === 1) return () => ys[0];
  const h = [], d = [];
  for (let i = 0; i < n - 1; i++) { h.push(xs[i + 1] - xs[i]); d.push((ys[i + 1] - ys[i]) / h[i]); }
  const m = new Array(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    if (d[i - 1] * d[i] <= 0) m[i] = 0;
    else { const w1 = 2 * h[i] + h[i - 1], w2 = h[i] + 2 * h[i - 1]; m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]); }
  }
  return (x) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let i = 0;
    while (i < n - 2 && x > xs[i + 1]) i++;
    const t = (x - xs[i]) / h[i], t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h[i] * m[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h[i] * m[i + 1];
  };
}

// 二维镜头：keys = [{ t, x, y, s }]，(x, y) 是被摄物体上的一点（物体自己的坐标），放到画面中心；s 是缩放。
// 缩放按对数插值（推拉的速度感均匀），位置按“屏幕上的移动”插值，避免缩放时画面乱晃
export function cam2(keys) {
  const ts = keys.map((k) => k.t);
  const fx = pchip(ts, keys.map((k) => k.x));
  const fy = pchip(ts, keys.map((k) => k.y));
  const fs = pchip(ts, keys.map((k) => Math.log(k.s)));
  const fcx = pchip(ts, keys.map((k) => k.cx ?? 960));
  const fcy = pchip(ts, keys.map((k) => k.cy ?? 540));
  return (t) => ({ x: fx(t), y: fy(t), s: Math.exp(fs(t)), cx: fcx(t), cy: fcy(t) });
}

// 确定的伪随机（给粒子、抖动用）
export function rng(seed) {
  let s = (seed % 2147483646) + 1;
  return () => ((s = (s * 16807) % 2147483647) / 2147483647);
}

// 二次贝塞尔上的一点
export const bez = (a, c, b, u) => ({ x: (1 - u) * (1 - u) * a.x + 2 * (1 - u) * u * c.x + u * u * b.x, y: (1 - u) * (1 - u) * a.y + 2 * (1 - u) * u * c.y + u * u * b.y });
