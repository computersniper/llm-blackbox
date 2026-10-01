// 读取 tools/multimodal/export_qwen3vl.py 导出的真实模型数据。
// 每张图一份视觉侧数据（ViT、合并器、DeepStack，和问题无关），每个问题一份语言侧数据。

const cache = new Map();

export async function loadManifest() {
  const r = await fetch('data/manifest.json');
  if (!r.ok) throw new Error(`manifest ${r.status}`);
  return r.json();
}

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
    case 'int16': return new Int16Array(buf, spec.offset, n);
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

async function loadPair(stem) {
  const [jbuf, buf] = await Promise.all([fetchData(`data/${stem}.json`), fetchData(`data/${stem}.bin`)]);
  const meta = JSON.parse(new TextDecoder().decode(jbuf));
  const arr = {};
  for (const [k, spec] of Object.entries(meta.bin)) arr[k] = view(buf, spec);
  return { meta, arr };
}

// 图片本身（模型真正看到的像素）
const imgCache = new Map();
export function loadImage(url) {
  if (!imgCache.has(url)) {
    imgCache.set(url, new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = () => rej(new Error(`图片 ${url} 加载失败`));
      im.src = url;
    }));
  }
  return imgCache.get(url);
}

// 一张图的视觉侧数据
export async function loadVision(spec) {
  const key = `v:${spec.id}`;
  if (cache.has(key)) return cache.get(key);
  const p = (async () => {
    const [{ meta, arr }, img] = await Promise.all([loadPair(spec.id), loadImage(`data/${spec.file}`)]);
    const { Np, Nv } = meta;
    const [gh, gw] = meta.grid, [mh, mw] = meta.merged;
    const V = {
      ...meta, img, spec, gh, gw, mh, mw,
      // ViT 第 k 级（0 = 图块嵌入 + 位置嵌入之后，1..24 = 第 k-1 个块之后）每个图块的 PCA 颜色，光栅顺序
      pca: (k) => arr.pca.subarray(k * Np * 3, (k + 1) * Np * 3),
      mergePca: arr.mergePca,
      dsPca: (d) => arr.dsPca.subarray(d * Nv * 3, (d + 1) * Nv * 3),
      // ViT 注意力（合并到 2×2 词元的分辨率，16 头平均）：第 li 个导出层，查询词元 q 看每个词元的权重
      vattn(li, q) {
        const k = meta.vitAttnLayers.indexOf(li);
        if (k < 0) return null;
        const base = (k * Nv + q) * Nv, mx = arr.vattnMax[k * Nv + q], out = new Float32Array(Nv);
        for (let j = 0; j < Nv; j++) out[j] = (arr.vattn[base + j] / 255) * mx;
        return out;
      },
      vattnSelf: (li, q) => arr.vattnSelf[meta.vitAttnLayers.indexOf(li) * Nv + q],
      vnorm: (L, m) => arr.vnorm[L * Nv + m],             // L = 0 是合并器输出，L = 1..28 是第 L-1 层之后
      dsNorm: (d, m) => arr.dsNorm[d * Nv + m],
      ilens: (L, m) => ({ s: meta.lensTab[arr.ilensId[L * Nv + m]], p: arr.ilensP[L * Nv + m] / 255 }),
      pix: arr.pix, px: arr.px, w: arr.w, kern: arr.kern, bank: arr.bank,
      // 光栅位置 → 所属的合并词元
      tokOfPatch: (r, c) => Math.floor(r / 2) * mw + Math.floor(c / 2),
    };
    return V;
  })();
  cache.set(key, p);
  p.catch(() => cache.delete(key));
  return p;
}

// 一个问题的语言侧数据
export async function loadQuestion(q, manifest) {
  const key = `q:${q.id}`;
  if (cache.has(key)) return cache.get(key);
  const p = (async () => {
    const spec = manifest.images.find((im) => im.questions.some((x) => x.id === q.id));
    const [{ meta, arr }, V] = await Promise.all([loadPair(q.id), loadVision(spec)]);
    const M = manifest.model.text;
    const NL = M.layers, H = M.heads, G = meta.G, Nv = meta.Nv, T = meta.T, F = manifest.focusLayers.length;
    const Q = {
      ...meta, V, M, manifest, NL, H, F, q,
      row: (g) => meta.P - 1 + g,
      isImg: (i) => i >= meta.vs && i < meta.vs + Nv,
      pos: (i) => [arr.pos[i], arr.pos[T + i], arr.pos[2 * T + i]],
      // 第 g 步、第 L 层，16 头平均后分给每个视觉词元的注意力（真实权重）
      attImg(g, L) {
        const base = (g * NL + L) * Nv, mx = arr.attMax[g * NL + L], out = new Float32Array(Nv);
        for (let m = 0; m < Nv; m++) out[m] = (arr.attImg[base + m] / 255) * mx;
        return out;
      },
      mass: (g, L) => arr.attMass[g * NL + L],
      // 若干层平均（默认“对准层”）
      attAvg(g, [a, b] = manifest.groundLayers) {
        const out = new Float32Array(Nv);
        for (let L = a; L <= b; L++) { const r = Q.attImg(g, L); for (let m = 0; m < Nv; m++) out[m] += r[m] / (b - a + 1); }
        return out;
      },
      txt(g, L) {
        const out = [];
        for (let k = 0; k < 4; k++) {
          const w = arr.txtW[(g * NL + L) * 4 + k] / 255;
          if (w > 0.004) out.push({ j: arr.txtIdx[(g * NL + L) * 4 + k], w });
        }
        return out;
      },
      headMass: (g, L, h) => arr.headMass[(g * NL + L) * H + h] / 255,
      attHead(g, f, h) {
        const base = ((g * F + f) * H + h) * Nv, mx = arr.attHeadMax[(g * F + f) * H + h], out = new Float32Array(Nv);
        for (let m = 0; m < Nv; m++) out[m] = (arr.attHead[base + m] / 255) * mx;
        return out;
      },
      delta: (g, L) => [arr.deltas[(g * NL + L) * 2], arr.deltas[(g * NL + L) * 2 + 1]],
      lensAt: (g, L) => meta.lens[g][L],
    };
    return Q;
  })();
  cache.set(key, p);
  p.catch(() => cache.delete(key));
  return p;
}
