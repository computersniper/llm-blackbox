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

export async function loadQuestion(id, manifest) {
  if (cache.has(id)) return cache.get(id);
  const [jbuf, buf] = await Promise.all([fetchData(`data/${id}.json`), fetchData(`data/${id}.bin`)]);
  const meta = JSON.parse(new TextDecoder().decode(jbuf));
  const arr = {};
  for (const [k, spec] of Object.entries(meta.bin)) arr[k] = view(buf, spec);
  const M = manifest.model;
  const NL = M.layers, H = M.heads, G = meta.G, K = manifest.attTopk, MK = manifest.mlpTopk;
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
    neuronAt: (L, g) => meta.neuron[String(L)]?.[g] ?? null,
    dotAt: (L, g) => meta.dot[String(L)]?.[g] ?? null,
  };
  cache.set(id, Q);
  return Q;
}
