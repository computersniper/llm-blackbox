// 3D 舞台的配色（线性空间，直接写进 InstancedMesh 的 instanceColor）。和 2D 热力图同一套：
//   数值：负 = 蓝，正 = 琥珀，零 = 暗；梯度：正 = 玫红，负 = 紫；概率：暗青 → 亮青。
const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
const L3 = (r, g, b) => [lin(r), lin(g), lin(b)];

// 在 sRGB 里插值再转线性，和 2D 热力图的观感一致
function lut(zero, pos, neg, n = 257) {
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * 2 - 1, e = Math.pow(Math.abs(t), 0.75), c = t >= 0 ? pos : neg;
    for (let j = 0; j < 3; j++) out[i * 3 + j] = lin(zero[j] + (c[j] - zero[j]) * e);
  }
  return out;
}

const DIV = lut([12, 20, 36], [255, 182, 92], [107, 155, 255]);
const GRAD = lut([14, 18, 34], [255, 107, 147], [150, 128, 255]);
const SEQ = (() => {
  const st = [[10, 22, 40], [16, 60, 90], [30, 130, 140], [70, 210, 190], [190, 250, 235]], n = 129, out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * (st.length - 1), a = Math.min(st.length - 2, Math.floor(t)), f = t - a;
    for (let j = 0; j < 3; j++) out[i * 3 + j] = lin(st[a][j] + (st[a + 1][j] - st[a][j]) * f);
  }
  return out;
})();

// v / scale ∈ [−1, 1] → 写进 arr[o..o+2]，再乘 k
export function divInto(arr, o, t, k = 1, tab = DIV) {
  t = t > 1 ? 1 : t < -1 ? -1 : t;
  const i = Math.round((t + 1) * 128) * 3;
  arr[o] = tab[i] * k; arr[o + 1] = tab[i + 1] * k; arr[o + 2] = tab[i + 2] * k;
}
export const gradInto = (arr, o, t, k = 1) => divInto(arr, o, t, k, GRAD);
export function seqInto(arr, o, t, k = 1) {
  t = t > 1 ? 1 : t < 0 ? 0 : t;
  const i = Math.round(t * 128) * 3;
  arr[o] = SEQ[i] * k; arr[o + 1] = SEQ[i + 1] * k; arr[o + 2] = SEQ[i + 2] * k;
}
export const LIN = { rose: L3(255, 107, 147), violet: L3(179, 157, 255), cyan: L3(94, 240, 212), amber: L3(255, 182, 92), blue: L3(107, 155, 255), green: L3(142, 224, 122), white: [1, 1, 1], idle: L3(17, 27, 46) };
