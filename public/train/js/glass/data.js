// 玻璃小模型的真实训练记录（tools/train/glassbox.py 导出）：2,928 个参数、200 步，前 50 步每步都记，之后每 5 / 10 步一帧。
//
//   首屏 glass.json + glass.bin：配置、语料、每一步的损失 / 学习率 / 梯度范数（总的和每个张量的）/ 裁剪系数 / 批次、初始化时的全部参数、两个对照组（全零 / 放大 50 倍）
//   glass/wN    ：第 N 组帧（每组 9 帧）的全部参数（float16，按帧做差存）+ 固定样例的预测概率、注意力、−ln p
//   glass/fNN   ：第 NN 帧的全部梯度、更新前的 m 和 √v、这一步的 Δw（按张量缩放的 float16）+ 固定样例每一层的激活和激活的梯度
//   glass/exact ：三个 RMSNorm γ 和嵌入表里“月”那一行，每一步的 float32 原值（w、g、m、v），能逐位重算 AdamW
//
// 帧 k（0..NF−1）对应训练的第 FR[k] 步（从 0 数）：记录的是这一步更新前的参数、这一步的批次 / 前向 / 反向，以及这一步的更新量。
import { f16tab, view, fetchData, decodeJSON, readChunk, NO_GUNZIP } from '../data.js';

const pad2 = (k) => String(k).padStart(2, '0');

async function fetchGz(url) {
  if (typeof DecompressionStream === 'undefined') throw Object.assign(new Error(NO_GUNZIP), { unsupported: true });
  const r = await fetch(`${url}.gz`);
  if (!r.ok || !r.body) throw new Error(`${url}.gz ${r.status}`);
  return new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}
export { fetchGz };

export const TENSOR_LABEL = { E: 'E', g1: 'γ₁', Wq: 'W_q', Wk: 'W_k', Wv: 'W_v', Wo: 'W_o', g2: 'γ₂', Wg: 'W_gate', Wu: 'W_up', Wd: 'W_down', gf: 'γ_f' };

export async function loadGlass(loader, base = 'data/') {
  const [metaBuf, buf] = await Promise.all([fetchData(`${base}glass.json`), fetchData(`${base}glass.bin`)]);
  const meta = decodeJSON(metaBuf);
  const A = {};
  for (const [k, spec] of Object.entries(meta.bin)) A[k] = view(buf, spec);
  const M = meta.model, Tn = meta.train;
  const S = Tn.steps, T = M.ctx, B = Tn.batch, V = M.vocab, H = M.heads, d = M.d, Fh = M.ffn, P = M.params;
  const FR = meta.frames, NF = FR.length;
  const chars = meta.corpus.chars, stream = meta.corpus.stream, NS = stream.length;
  const params = meta.params.map((p, j) => ({ ...p, j, label: TENSOR_LABEL[p.name] || p.name, rows: p.shape[0], cols: p.shape.length > 1 ? p.shape[1] : 1, norm: p.name.startsWith('g') }));
  const pIndex = new Map(params.map((p) => [p.name, p]));
  const frameOfStep = new Map(FR.map((t, k) => [t, k]));
  const groups = meta.chunks.w;                        // [[起始帧, 结束帧), …]
  const groupOf = new Int16Array(NF);
  groups.forEach(([a, b], gi) => { for (let k = a; k < b; k++) groupOf[k] = gi; });
  const TRI = (T * (T + 1)) / 2;
  const triIdx = (i, j) => (i * (i + 1)) / 2 + j;

  // 分块到了以后放这里
  const WG = new Array(groups.length).fill(null);      // { w: [每帧 Float32Array(P)], p, att, nll }
  const FC = new Array(NF).fill(null);                  // { A: 数组, J }
  let EX = null;
  const dir = `${base}${meta.chunks.dir}`;
  groups.forEach(([a, b], gi) => loader.define(`glass:w:${gi}`, `${dir}/w${gi}.bin`, (bf) => (WG[gi] = decodeW(bf, b - a))));
  for (let k = 0; k < NF; k++) loader.define(`glass:f:${k}`, `${dir}/f${pad2(k)}.bin`, (bf) => (FC[k] = readChunk(bf)));
  loader.define('glass:exact', `${dir}/exact.bin`, (bf) => (EX = decodeExact(readChunk(bf).A)));

  // 权重块：w 是 float16 比特模式沿帧的差（mod 2¹⁶），累加回去再查表
  function decodeW(bf, n) {
    const h = new DataView(bf).getUint32(0, true);
    const head = decodeJSON(new Uint8Array(bf, 4, h));
    const base0 = 4 + h + ((8 - ((4 + h) % 8)) % 8);
    const out = {};
    for (const [k, spec] of Object.entries(head.bin)) {
      if (k === 'w') continue;
      out[k] = view(bf, spec, base0);
    }
    const dw = view(bf, head.bin.w, base0);             // Uint16Array [n × P]
    const acc = new Uint16Array(P);
    out.w = [];
    for (let f = 0; f < n; f++) {
      const wf = new Float32Array(P);
      for (let i = 0; i < P; i++) { acc[i] = (acc[i] + dw[f * P + i]) & 0xffff; wf[i] = f16tab[acc[i]]; }
      out.w.push(wf);
    }
    return out;
  }

  // exact：idx [n]、w / m / v [S+1, n]、g [S, n]、lr [S]（float64）、clip [S]
  function decodeExact(X) {
    const pos = new Map(Array.from(X.idx, (gi, j) => [gi, j]));
    return { ...X, n: X.idx.length, pos };
  }

  // 按张量缩放的 float16 → 真实数值（每帧缓存）
  const cache = new Map();
  function unscaled(k, key) {
    const ck = `${k}:${key}`;
    if (cache.has(ck)) return cache.get(ck);
    const a = FC[k].A, x = a[key], sc = a[`${key}S`];
    const out = new Float32Array(P);
    for (const p of params) { const s = sc[p.j]; for (let i = p.off; i < p.off + p.n; i++) out[i] = x[i] * s; }
    if (key === 'v') for (let i = 0; i < P; i++) out[i] *= out[i];   // 存的是 √v
    cache.set(ck, out);
    if (cache.size > 64) cache.delete(cache.keys().next().value);
    return out;
  }

  const win = (o) => { const r = []; for (let i = 0; i <= T; i++) r.push(stream[(o + i) % NS]); return r; };

  const D = {
    kind: 'glass', meta, A, M, S, T, B, V, H, d, Fh, P, FR, NF, chars, stream, params, pIndex, TRI, groups,
    hd: M.headDim,
    ch: (id) => chars[id] ?? '?',
    loss: A.loss, evalLoss: A.evalLoss, lr: A.lr, gnorm: A.gnorm, clip: A.clip, w0: A.w0,
    tgn: (t, j) => A.tgn[t * params.length + j],   // 第 t 步第 j 个张量的梯度长度（裁剪前）
    zLoss: A.zLoss, zGnorm: A.zGnorm, zMaxW: A.zMaxW, bLoss: A.bLoss, bEval: A.bEval, bGnorm: A.bGnorm, bW0: A.bW0, bP0: A.bP0, zP0: A.zP0, zAtt0: A.zAtt0,
    win,
    // 第 t 步的批次：B 段，每段 T+1 个字的编号（第 0 段固定是“举头望明月，低头思”）
    batch: (t) => { const out = []; for (let b = 0; b < B; b++) out.push(win(A.offs[t * B + b])); return out; },
    offs: (t, b) => A.offs[t * B + b],
    fixed: win(meta.corpus.fixed),
    frameOf: (t) => frameOfStep.get(t) ?? -1,
    stepOf: (k) => FR[k],
    groupOf: (k) => groupOf[k],
    key: (part, k) => (part === 'exact' ? 'glass:exact' : part === 'w' ? `glass:w:${groupOf[k]}` : `glass:f:${k}`),
    has: (part, k) => (part === 'exact' ? !!EX : part === 'w' ? !!WG[groupOf[k]] : !!FC[k]),
    ready: (k) => !!WG[groupOf[k]] && !!FC[k],
    // —— 权重块
    W: (k) => WG[groupOf[k]].w[k - groups[groupOf[k]][0]],
    probs: (k, i, v) => WG[groupOf[k]].p[((k - groups[groupOf[k]][0]) * T + i) * V + v],
    nll: (k, i) => WG[groupOf[k]].nll[(k - groups[groupOf[k]][0]) * T + i],
    att: (k, h, i, j) => (j > i ? 0 : WG[groupOf[k]].att[((k - groups[groupOf[k]][0]) * H + h) * TRI + triIdx(i, j)]),
    // —— 每帧块
    G: (k) => unscaled(k, 'g'),          // 裁剪前的梯度 ∂L/∂w
    Mo: (k) => unscaled(k, 'm'),         // 这一步更新前的 m
    Vo: (k) => unscaled(k, 'v'),         // 这一步更新前的 v
    DW: (k) => unscaled(k, 'dw'),        // 这一步的更新量 Δw
    // 固定样例（第 0 段）的激活 / 激活的梯度；att 展开成 [H, T, T]
    act(k, name) { return expand(FC[k].A[`f.${name}`], name); },
    grad(k, name) { return expand(FC[k].A[`b.${name}`], name); },
    // —— exact
    exact: () => EX,
    isExact: (gi) => !!meta.exact.idx.includes(gi),
  };
  function expand(a, name) {
    if (!a || !meta.chunks.tri.includes(name)) return a;
    const out = new Float32Array(H * T * T);
    for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) out[(h * T + i) * T + j] = a[h * TRI + triIdx(i, j)];
    return out;
  }
  // 参数下标 ↔（张量, 行, 列）
  D.locate = (gi) => { for (const p of params) if (gi >= p.off && gi < p.off + p.n) { const r = gi - p.off; return { p, i: Math.floor(r / p.cols), j: r % p.cols }; } return null; };
  D.index = (name, i, j) => { const p = pIndex.get(name); return p.off + i * p.cols + j; };
  return D;
}

// 给页面 / 控制条用的统一接口（和 ../run.js 的 wrapTiny / wrapQwen 同一套字段）
export function wrapGlass(D) {
  return {
    kind: 'glass', D, K: D.NF,
    stepNo: (k) => D.FR[k] + 1,
    stepReady: (k) => D.ready(k),
    loss: (k) => D.loss[D.FR[k]],
    lr: (k) => D.lr[D.FR[k]],
    gradNorm: (k) => D.gnorm[D.FR[k]],
    clip: (k) => D.clip[D.FR[k]],
  };
}
