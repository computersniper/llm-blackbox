// 运镜工具：关键帧 → 样条曲线。位置和注视点各走一条向心 Catmull-Rom 曲线；
// 时间到曲线参数的映射用单调三次插值（PCHIP），两端速度为 0，中间速度连续，所以起停自然、不会顿。
import { THREE } from './engine.js';

export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const seg = (t, a, b) => clamp((t - a) / (b - a));
export const smooth = (t) => { t = clamp(t); return t * t * (3 - 2 * t); };
export const smoother = (t) => { t = clamp(t); return t * t * t * (t * (t * 6 - 15) + 10); };
export const easeOut = (t) => 1 - Math.pow(1 - clamp(t), 3);
export const easeIn = (t) => Math.pow(clamp(t), 3);
export const easeInOut = (t) => { t = clamp(t); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
export const expoOut = (t) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * clamp(t)));
export const v3 = (x, y, z) => new THREE.Vector3(x, y, z);
const V = (a) => (a.isVector3 ? a.clone() : v3(a[0], a[1], a[2]));

// 单调三次插值（Fritsch–Carlson），端点斜率为 0
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

// keys: [{ t, p:[x,y,z], l:[x,y,z], fov?, roll? }]，t 为镜头内的局部时间
export function path(keys) {
  const ts = keys.map((k) => k.t);
  const idx = pchip(ts, keys.map((_, i) => i));
  const n = keys.length;
  const P = n > 1 ? new THREE.CatmullRomCurve3(keys.map((k) => V(k.p)), false, 'centripetal') : null;
  const Lk = n > 1 ? new THREE.CatmullRomCurve3(keys.map((k) => V(k.l)), false, 'centripetal') : null;
  const fov = pchip(ts, keys.map((k) => k.fov ?? 34));
  const roll = pchip(ts, keys.map((k) => k.roll ?? 0));
  return (t) => {
    if (n === 1) return { pos: V(keys[0].p), look: V(keys[0].l), fov: keys[0].fov ?? 34, roll: keys[0].roll ?? 0 };
    const u = clamp(idx(t) / (n - 1));
    return { pos: P.getPoint(u), look: Lk.getPoint(u), fov: fov(t), roll: roll(t) };
  };
}

// 两个机位之间平滑过渡（k: 0 → 1）
export function blendCam(a, b, k) {
  return { pos: a.pos.clone().lerp(b.pos, k), look: a.look.clone().lerp(b.look, k), fov: lerp(a.fov ?? 34, b.fov ?? 34, k), roll: lerp(a.roll ?? 0, b.roll ?? 0, k) };
}

// 绕注视点转：yaw（水平，度）、pitch（俯仰，度）、dist 倍数
export function orbit(cam, yawDeg = 0, pitchDeg = 0, distK = 1) {
  const off = cam.pos.clone().sub(cam.look);
  const sph = new THREE.Spherical().setFromVector3(off);
  sph.theta += (yawDeg * Math.PI) / 180;
  sph.phi = clamp(sph.phi - (pitchDeg * Math.PI) / 180, 0.05, Math.PI - 0.05);
  sph.radius *= distK;
  return { ...cam, pos: cam.look.clone().add(new THREE.Vector3().setFromSpherical(sph)) };
}

// 手持感：几个不成比例的正弦叠加，极轻微（确定性，不用随机数）
export function handheld(cam, t, amp = 0.02) {
  if (!amp) return cam;
  const n = (a, b, c) => Math.sin(t * a + c) * 0.6 + Math.sin(t * b + c * 1.7) * 0.4;
  const d = cam.pos.distanceTo(cam.look);
  const k = amp * d;
  return {
    ...cam,
    pos: cam.pos.clone().add(v3(n(0.71, 1.37, 0.3) * k, n(0.53, 1.13, 1.9) * k * 0.7, n(0.37, 0.91, 4.1) * k * 0.5)),
    look: cam.look.clone().add(v3(n(0.43, 1.07, 2.2) * k * 0.4, n(0.61, 0.83, 3.3) * k * 0.3, 0)),
  };
}
