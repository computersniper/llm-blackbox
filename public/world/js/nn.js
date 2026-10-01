// 在浏览器里跑训练好的世界模型（真实权重，纯 JS + typed array）。
// V：卷积 VAE（编码 64×64×3 → μ, logσ²；解码 z → 64×64×3），M：LSTM + 混合密度输出（MDN），C：线性控制器。
// 权重由 tools/world/export_world.py 导出成 float16（model.bin）+ 描述（model.json），这里展开成 float32 计算，
// 张量的排布和 PyTorch 一致（卷积 [Cout, Cin, k, k]，反卷积 [Cin, Cout, k, k]，激活值都是 CHW）。
// tools/world/check_nn.mjs 在 Node 里用同一份代码和 PyTorch 的输出逐元素核对。

export const Z = 32, NA = 3, HID = 256, KMIX = 5;

// float16 → float32（查表，不依赖 Float16Array）
const F16 = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

export function unpack(meta, buf) {
  const T = {};
  for (const [k, s] of Object.entries(meta.tensors)) {
    const n = s.shape.reduce((a, b) => a * b, 1);
    const raw = new Uint16Array(buf, s.offset, n);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = F16[raw[i]];
    T[k] = out;
  }
  return T;
}

/* ---------------------------------------------------------------- 算子（输出都是新的 Float32Array） */

// 卷积，无填充：in [C, H, W] → out [CO, OH, OW]
export function conv2d(inp, C, H, W, w, b, CO, k, s, relu = true) {
  const OH = ((H - k) / s | 0) + 1, OW = ((W - k) / s | 0) + 1, HW = OH * OW, out = new Float32Array(CO * HW);
  for (let co = 0; co < CO; co++) {
    const ob = co * HW;
    out.fill(b[co], ob, ob + HW);
    for (let ci = 0; ci < C; ci++) {
      const ib = ci * H * W, wb = (co * C + ci) * k * k;
      for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
        const wv = w[wb + ky * k + kx];
        for (let oy = 0; oy < OH; oy++) {
          let ii = ib + (oy * s + ky) * W + kx, oi = ob + oy * OW;
          for (let ox = 0; ox < OW; ox++, ii += s, oi++) out[oi] += wv * inp[ii];
        }
      }
    }
  }
  if (relu) for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0;
  return { a: out, C: CO, H: OH, W: OW };
}

// 反卷积（转置卷积）：每个输入像素把 k×k 的核“盖章”到输出上，步长 s
export function convT2d(inp, C, H, W, w, b, CO, k, s) {
  const OH = (H - 1) * s + k, OW = (W - 1) * s + k, HW = OH * OW, out = new Float32Array(CO * HW);
  for (let co = 0; co < CO; co++) out.fill(b[co], co * HW, (co + 1) * HW);
  if (H === 1 && W === 1) {
    // 1×1 的输入：就是一次矩阵乘向量，权重 [Cin, Cout·k·k] 正好连续
    const n = CO * k * k;
    for (let ci = 0; ci < C; ci++) {
      const xv = inp[ci], wb = ci * n;
      if (xv === 0) continue;
      for (let j = 0; j < n; j++) out[j] += xv * w[wb + j];
    }
  } else {
    for (let ci = 0; ci < C; ci++) {
      const ib = ci * H * W;
      for (let co = 0; co < CO; co++) {
        const ob = co * HW, wb = (ci * CO + co) * k * k;
        for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
          const wv = w[wb + ky * k + kx];
          for (let iy = 0; iy < H; iy++) {
            let ii = ib + iy * W, oi = ob + (iy * s + ky) * OW + kx;
            for (let ix = 0; ix < W; ix++, ii++, oi += s) out[oi] += wv * inp[ii];
          }
        }
      }
    }
  }
  return { a: out, C: CO, H: OH, W: OW };
}

// 全连接：w [out, in]
export function linear(x, w, b, nOut) {
  const nIn = x.length, out = new Float32Array(nOut);
  for (let j = 0; j < nOut; j++) {
    let acc = b ? b[j] : 0;
    const wb = j * nIn;
    for (let i = 0; i < nIn; i++) acc += w[wb + i] * x[i];
    out[j] = acc;
  }
  return out;
}

export const sigmoid = (v) => 1 / (1 + Math.exp(-v));
const relu = (a) => { for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0; return a; };

// 编码器、解码器的每一层（讲解和“一次乘加”用）
export const ENC = [
  { id: 'e1', name: '卷积 1', cin: 3, cout: 16, k: 4, s: 2, inH: 64, outH: 31 },
  { id: 'e2', name: '卷积 2', cin: 16, cout: 32, k: 4, s: 2, inH: 31, outH: 14 },
  { id: 'e3', name: '卷积 3', cin: 32, cout: 64, k: 4, s: 2, inH: 14, outH: 6 },
  { id: 'e4', name: '卷积 4', cin: 64, cout: 128, k: 4, s: 2, inH: 6, outH: 2 },
];
export const DEC = [
  { id: 'd1', name: '反卷积 1', cin: 512, cout: 64, k: 5, s: 2, inH: 1, outH: 5 },
  { id: 'd2', name: '反卷积 2', cin: 64, cout: 32, k: 5, s: 2, inH: 5, outH: 13 },
  { id: 'd3', name: '反卷积 3', cin: 32, cout: 16, k: 6, s: 2, inH: 13, outH: 30 },
  { id: 'd4', name: '反卷积 4', cin: 16, cout: 3, k: 6, s: 2, inH: 30, outH: 64 },
];

/* ---------------------------------------------------------------- 世界模型 */

export class WorldModel {
  constructor(meta, T) {
    this.meta = meta;
    this.T = T;
    // LSTM 的两份偏置合成一份
    const bi = T['lstm.bias_ih_l0'], bh = T['lstm.bias_hh_l0'];
    this.bLstm = new Float32Array(4 * HID);
    for (let j = 0; j < 4 * HID; j++) this.bLstm[j] = bi[j] + bh[j];
    this.params = Object.values(T).reduce((s, a) => s + a.length, 0);
  }

  // x：3×64×64（0..1）。keep = true 时把每一层的激活值也带回来
  encode(x, keep = false) {
    const T = this.T;
    const a1 = conv2d(x, 3, 64, 64, T['e1.weight'], T['e1.bias'], 16, 4, 2);
    const a2 = conv2d(a1.a, 16, 31, 31, T['e2.weight'], T['e2.bias'], 32, 4, 2);
    const a3 = conv2d(a2.a, 32, 14, 14, T['e3.weight'], T['e3.bias'], 64, 4, 2);
    const a4 = conv2d(a3.a, 64, 6, 6, T['e4.weight'], T['e4.bias'], 128, 4, 2);
    const mu = linear(a4.a, T['mu.weight'], T['mu.bias'], Z);
    const lv = linear(a4.a, T['lv.weight'], T['lv.bias'], Z);
    const r = { mu, lv };
    if (keep) Object.assign(r, { x, e1: a1.a, e2: a2.a, e3: a3.a, e4: a4.a });
    return r;
  }

  decode(z, keep = false) {
    const T = this.T;
    const f = linear(z, T['dfc.weight'], T['dfc.bias'], 512);
    const b1 = convT2d(f, 512, 1, 1, T['d1.weight'], T['d1.bias'], 64, 5, 2); relu(b1.a);
    const b2 = convT2d(b1.a, 64, 5, 5, T['d2.weight'], T['d2.bias'], 32, 5, 2); relu(b2.a);
    const b3 = convT2d(b2.a, 32, 13, 13, T['d3.weight'], T['d3.bias'], 16, 6, 2); relu(b3.a);
    const b4 = convT2d(b3.a, 16, 30, 30, T['d4.weight'], T['d4.bias'], 3, 6, 2);
    const pre = keep ? b4.a.slice() : null;
    const y = b4.a;
    for (let i = 0; i < y.length; i++) y[i] = 1 / (1 + Math.exp(-y[i]));
    const r = { y };
    if (keep) Object.assign(r, { z, dfc: f, d1: b1.a, d2: b2.a, d3: b3.a, d4pre: pre });
    return r;
  }

  // 一步 LSTM。PyTorch 的门顺序：i, f, g, o
  lstm(z, a, h, c) {
    const T = this.T, x = new Float32Array(Z + NA);
    x.set(z);
    x[Z + a] = 1;
    const Wi = T['lstm.weight_ih_l0'], Wh = T['lstm.weight_hh_l0'], b = this.bLstm;
    const pre = new Float32Array(4 * HID);
    for (let j = 0; j < 4 * HID; j++) {
      let acc = b[j];
      const wi = j * (Z + NA), wh = j * HID;
      for (let k = 0; k < Z + NA; k++) acc += Wi[wi + k] * x[k];
      for (let k = 0; k < HID; k++) acc += Wh[wh + k] * h[k];
      pre[j] = acc;
    }
    const gi = new Float32Array(HID), gf = new Float32Array(HID), gg = new Float32Array(HID), go = new Float32Array(HID);
    const c2 = new Float32Array(HID), h2 = new Float32Array(HID);
    for (let j = 0; j < HID; j++) {
      gi[j] = sigmoid(pre[j]);
      gf[j] = sigmoid(pre[HID + j]);
      gg[j] = Math.tanh(pre[2 * HID + j]);
      go[j] = sigmoid(pre[3 * HID + j]);
      c2[j] = gf[j] * c[j] + gi[j] * gg[j];
      h2[j] = go[j] * Math.tanh(c2[j]);
    }
    return { x, pre, i: gi, f: gf, g: gg, o: go, c: c2, h: h2 };
  }

  // 混合密度输出：每一维 5 个高斯分量的 log π（已归一化）、μ、log σ，外加撞车的 logit
  mdn(h) {
    const T = this.T, n = Z * KMIX;
    const raw = linear(h, T['head.weight'], T['head.bias'], 3 * n + 1);
    const logpi = new Float32Array(n), mu = raw.slice(n, 2 * n), logsig = raw.slice(2 * n, 3 * n);
    for (let d = 0; d < Z; d++) {
      let m = -Infinity;
      for (let k = 0; k < KMIX; k++) m = Math.max(m, raw[d * KMIX + k]);
      let s = 0;
      for (let k = 0; k < KMIX; k++) s += Math.exp(raw[d * KMIX + k] - m);
      const lse = m + Math.log(s);
      for (let k = 0; k < KMIX; k++) logpi[d * KMIX + k] = raw[d * KMIX + k] - lse;
    }
    const doneLogit = raw[3 * n];
    return { raw, logpi, mu, logsig, doneLogit, done: sigmoid(doneLogit) };
  }

  // 按温度 τ 采样下一帧的 z（论文的做法：log π / τ 再 softmax 选分量，σ 乘 √τ）
  sample(m, tau, rand) {
    const z = new Float32Array(Z), ks = new Uint8Array(Z), eps = new Float32Array(Z), pis = new Float32Array(Z * KMIX);
    for (let d = 0; d < Z; d++) {
      let mx = -Infinity;
      for (let k = 0; k < KMIX; k++) mx = Math.max(mx, m.logpi[d * KMIX + k] / tau);
      let s = 0;
      for (let k = 0; k < KMIX; k++) { const e = Math.exp(m.logpi[d * KMIX + k] / tau - mx); pis[d * KMIX + k] = e; s += e; }
      for (let k = 0; k < KMIX; k++) pis[d * KMIX + k] /= s;
      let u = rand(), k = 0;
      while (k < KMIX - 1 && u >= pis[d * KMIX + k]) { u -= pis[d * KMIX + k]; k++; }
      ks[d] = k;
      eps[d] = gauss(rand);
      z[d] = m.mu[d * KMIX + k] + Math.exp(m.logsig[d * KMIX + k]) * Math.sqrt(tau) * eps[d];
    }
    return { z, k: ks, eps, pi: pis };
  }

  // 线性控制器：a = argmax(W [z, h, 1])
  ctrl(z, h) {
    const W = this.T['ctrl.weight'];
    if (!W) return null;
    const logits = new Float32Array(NA);
    for (let a = 0; a < NA; a++) {
      let acc = W[(Z + HID) * NA + a];
      for (let i = 0; i < Z; i++) acc += W[i * NA + a] * z[i];
      for (let i = 0; i < HID; i++) acc += W[(Z + i) * NA + a] * h[i];
      logits[a] = acc;
    }
    let best = 0;
    for (let a = 1; a < NA; a++) if (logits[a] > logits[best]) best = a;
    return { logits, a: best };
  }
}

// 标准正态（Box–Muller）
export function gauss(rand) {
  let u = 0;
  while (u <= 1e-12) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

// 可复现的随机数（mulberry32）：同一个种子，梦里的每一次采样都一样，调试器退回再前进时梦不会变
export function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
