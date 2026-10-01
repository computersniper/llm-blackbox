// 读取 tools/train/ 导出的两段真实训练记录：tiny（唐宋诗小模型从零训练）、qwen（Qwen3-0.6B 三步 SFT）。

// float16 → float32（不依赖 Float16Array）
const f16tab = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

function view(buf, spec) {
  const n = spec.shape.reduce((a, b) => a * b, 1);
  switch (spec.dtype) {
    case 'uint8': return new Uint8Array(buf, spec.offset, n);
    case 'int8': return new Int8Array(buf, spec.offset, n);
    case 'uint16': return new Uint16Array(buf, spec.offset, n);
    case 'float32': return new Float32Array(buf, spec.offset, n);
    case 'float16': {
      const raw = new Uint16Array(buf, spec.offset, n);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = f16tab[raw[i]];
      return out;
    }
    default: throw new Error(`dtype ${spec.dtype}`);
  }
}

// 数据文件预先 gzip 过（服务器只压缩 HTML），浏览器里用 DecompressionStream 解开；不支持时退回未压缩版本
async function fetchData(url) {
  if (typeof DecompressionStream !== 'undefined') {
    try {
      const r = await fetch(`${url}.gz`);
      if (r.ok && r.body) return await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    } catch { /* 退回未压缩 */ }
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} ${r.status}`);
  return r.arrayBuffer();
}

const json = async (url) => JSON.parse(new TextDecoder().decode(await fetchData(url)));

/* ---------------------------------------------------------------- 小模型 */

export async function loadTiny() {
  const [meta, buf] = await Promise.all([json('data/tiny.json'), fetchData('data/tiny.bin')]);
  const A = {};
  for (const [k, spec] of Object.entries(meta.bin)) A[k] = view(buf, spec);
  const M = meta.model, NL = M.layers, H = M.heads, K = meta.ckpts.length;
  const Lv = meta.held.ids.length - 1, R = meta.rowN, NB = NL + 1, TOP = 5;
  const NP = meta.pcaIds.length, C = meta.crops[0].rows;
  const S = meta.train.steps;
  const D = {
    kind: 'tiny', meta, A, M, NL, H, K, Lv, R, NB, S, NP, C,
    vocab: meta.vocab,
    ck: (k) => meta.ckpts[k],
    ch: (id) => meta.vocab[id] ?? '?',
    // 留出的《登鹳雀楼》：第 i 个位置预测第 i+1 个字
    valP: (k, i) => A.val_p[k * Lv + i],
    valTop(k, i) { const out = []; for (let j = 0; j < TOP; j++) out.push({ id: A.val_top_id[(k * Lv + i) * TOP + j], p: A.val_top_p[(k * Lv + i) * TOP + j] }); return out; },
    attn: (k, l, h, i, j) => A.attn[(((k * NL + l) * H + h) * Lv + i) * Lv + j] / 255,
    pca: (k, n) => [A.pca[(k * NP + n) * 2], A.pca[(k * NP + n) * 2 + 1]],
    wcrop: (k, c, i, j) => A.wcrop[((k * 3 + c) * C + i) * C + j] * meta.ckpts[k].wScale[c],
    gcrop: (k, c, i, j) => A.gcrop[((k * 3 + c) * C + i) * C + j] * meta.ckpts[k].gScale[c],
    // 批次第 0 行（前 R 个位置）
    rowId: (k, i) => A.row_ids[k * (R + 1) + i],
    rowP: (k, i) => A.row_p[k * R + i],
    rowPAfter: (k, i) => A.row_p_after[k * R + i],
    rowTop(k, i) { const out = []; for (let j = 0; j < TOP; j++) out.push({ id: A.row_top_id[(k * R + i) * TOP + j], p: A.row_top_p[(k * R + i) * TOP + j] }); return out; },
    lensTop: (k, b, i) => A.lens_top[(k * NB + b) * R + i],
    lensP: (k, b, i) => A.lens_p[(k * NB + b) * R + i],
    residNorm: (k, b, i) => A.resid_norm[(k * NB + b) * R + i],
    residGrad: (k, b, i) => A.resid_grad[(k * NB + b) * R + i],
    tgrad: (k, n) => A.tgrad[k * meta.tensors.length + n],
    loss: A.loss, lr: A.lr, gnorm: A.gnorm,
    feat: (f) => {
      const n = A.feat_w.length / meta.feats.length;
      return { w: A.feat_w.subarray(f * n, (f + 1) * n), g: A.feat_g.subarray(f * n, (f + 1) * n), m: A.feat_m.subarray(f * n, (f + 1) * n), v: A.feat_v.subarray(f * n, (f + 1) * n), n, stride: meta.featStride };
    },
    land: (iy, ix) => A.land[iy * meta.land.G + ix],
  };
  D.tIndex = new Map(meta.tensors.map((n, i) => [n, i]));
  // 训练集里的字频顺序就是编号顺序（1 号最常见）；PCA 里每个点是哪个字
  D.pcaChars = meta.pcaIds.map((id) => D.ch(id));
  return D;
}

/* ---------------------------------------------------------------- Qwen3-0.6B */

export async function loadQwen() {
  const meta = await json('data/qwen.json');
  const N = meta.ids.length;
  const D = {
    kind: 'qwen', meta, N,
    NL: meta.model.layers,
    NB: meta.model.layers + 1,
    K: meta.steps.length,
    tok: (id) => meta.vocab[String(id)] ?? `#${id}`,
    roleOf: (i) => meta.roles[i],
    // 位置 i 预测第 i+1 个词元
    sftPos: meta.sftMask.map((m, i) => (m ? i : -1)).filter((i) => i >= 0),
  };
  return D;
}
