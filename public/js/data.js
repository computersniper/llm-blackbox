// 读取 tools/export_qwen.py 导出的真实模型数据。

const cache = new Map();

export async function loadManifest() {
  const r = await fetch('data/manifest.json');
  if (!r.ok) throw new Error(`manifest ${r.status}`);
  return r.json();
}

// float16 → float32（不依赖 Float16Array，旧浏览器也能用）
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

// 每层每个矩阵的真实权重分布缩略图（每格 = 32×32 个权重的均方根）
let thumbs = null;
export async function loadThumbs() {
  if (!thumbs) thumbs = fetchData('data/weights.bin').then((b) => new Uint8Array(b));
  return thumbs;
}

const decodeJSON = (buf) => JSON.parse(new TextDecoder().decode(buf));

export async function loadQuestion(id, manifest) {
  if (cache.has(id)) return cache.get(id);
  const [jbuf, buf] = await Promise.all([fetchData(`data/${id}.json`), fetchData(`data/${id}.bin`)]);
  const meta = decodeJSON(jbuf);
  const arr = {};
  for (const [k, spec] of Object.entries(meta.bin)) arr[k] = view(buf, spec);
  const M = manifest.model;
  const NL = M.layers, H = M.heads, G = meta.G, K = manifest.attTopk, MK = manifest.mlpTopk;
  // “一次乘加”的数据（单神经元 / 单次打分 / 矩阵乘法的一个输出元素）28 层都有，但按层拆成小文件
  // data/qNN/Lxx.json，进入某一层的 D6 时才去取。micro 存已经到手的层，pending 合并同一层的并发请求
  const micro = new Map();
  const pending = new Map();
  const Q = {
    ...meta,
    M,
    manifest,
    NL, H, K,
    row: (g) => meta.P - 1 + g,
    // 第 L 层第 h 个头，第 i 行（查询位置）的前 K 个键。只导出了生成步的那些行（i = P-1+g）
    att(L, h, i) {
      const g = i - (meta.P - 1);
      if (g < 0 || g >= G) return [];
      const base = ((L * H + h) * G + g) * K;
      const out = [];
      for (let k = 0; k < K; k++) {
        const w = arr.attW[base + k] / 255;
        if (w <= 0) continue;
        const j = arr.attIdx[base + k];
        if (j > i) continue;
        out.push({ j, w, s: Math.log(w) + arr.attLse[(L * H + h) * G + g] });
      }
      return out;
    },
    // 16 个头平均后的前几名（每个头只有前 K 个，其余视为 0）
    attMean(L, i, top = 4) {
      const acc = new Map();
      for (let h = 0; h < H; h++) for (const { j, w } of Q.att(L, h, i)) acc.set(j, (acc.get(j) || 0) + w / H);
      return [...acc.entries()].map(([j, w]) => ({ j, w })).sort((a, b) => b.w - a.w).slice(0, top);
    },
    mlpTop(g, L) {
      const base = (g * NL + L) * MK;
      const out = [];
      for (let k = 0; k < MK; k++) out.push({ n: arr.mlpIdx[base + k], v: arr.mlpVal[base + k] });
      return out;
    },
    mlpCount: (g, L) => arr.mlpCount[g * NL + L],
    mlpFull(g) {
      const F = M.ffn, s = arr.mlpFullScale[g], out = new Float32Array(F);
      for (let n = 0; n < F; n++) out[n] = arr.mlpFull[g * F + n] * s;
      return out;
    },
    norm: (L, i) => meta.norms[L][i],
    lensAt: (g, L) => meta.lens[g][L],
    // 第 L 层的乘加数据到了没有；没到时下面三个访问器返回 null
    hasMicro: (L) => micro.has(L),
    // 取第 L 层的乘加数据：已缓存就立即完成，同一层同时只发一个请求；失败后下次调用会重试
    ensureMicro(L) {
      if (micro.has(L)) return Promise.resolve(micro.get(L));
      if (pending.has(L)) return pending.get(L);
      if (!Number.isInteger(L) || L < 0 || L >= NL) return Promise.reject(new Error(`没有第 ${L} 层`));
      const p = fetchData(`data/${id}/L${String(L).padStart(2, '0')}.json`).then((b) => {
        const d = decodeJSON(b);
        micro.set(L, d);
        pending.delete(L);
        return d;
      }, (e) => { pending.delete(L); throw e; });
      pending.set(L, p);
      return p;
    },
    neuronAt: (L, g) => micro.get(L)?.neuron?.[g] ?? null,
    mmAt: (L, g) => micro.get(L)?.mm?.[g] ?? null,
    headMMAt: (g) => meta.headMM?.[g] ?? null,
    dotAt: (L, g) => micro.get(L)?.dot?.[g] ?? null,
  };
  cache.set(id, Q);
  return Q;
}
