// 读取 tools/train/ 导出的两段真实训练记录：tiny（唐宋诗小模型从零训练）、qwen（Qwen3-0.6B 三步 SFT）。
//
// 数据分两层（切分方式见 tools/train/split_data.py）：
//   首屏：tiny.json + tiny.bin、qwen.json——元数据、每一步的损失 / 学习率 / 梯度范数、批次第 0 行的字、损失地形，约 90 KB（gzip）；
//   分块：tiny/ckNN（D1 第 NN 个检查点的诗概率、注意力、嵌入、权重局部）、tiny/stNN（D2–D4 一步之内）、tiny/feat（权重轨迹）、
//         qwen/stN（D2–D4）。进入对应视图、播放或拖到对应检查点时才去取，空闲时在后台慢慢预取（见 Loader）。

import { L } from './lang.js';

// float16 → float32（不依赖 Float16Array）
export const f16tab = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

const SIZE = { uint8: 1, int8: 1, uint16: 2, float16: 2, float32: 4, float64: 8 };

// buf 里从 base + spec.offset 开始的一个数组。shuffle：按字节分面存放（先放所有数的第 0 个字节……），先拼回原来的字节顺序
export function view(buf, spec, base = 0) {
  const n = spec.shape.reduce((a, b) => a * b, 1);
  const size = SIZE[spec.dtype];
  if (!size) throw new Error(`dtype ${spec.dtype}`);
  let off = base + spec.offset;
  if (spec.shuffle && size > 1) {
    const src = new Uint8Array(buf, off, n * size), dst = new Uint8Array(n * size);
    for (let b = 0; b < size; b++) for (let i = 0, o = b * n; i < n; i++) dst[i * size + b] = src[o + i];
    buf = dst.buffer;
    off = 0;
  }
  switch (spec.dtype) {
    case 'uint8': return new Uint8Array(buf, off, n);
    case 'int8': return new Int8Array(buf, off, n);
    case 'uint16': return new Uint16Array(buf, off, n);
    case 'float32': return new Float32Array(buf, off, n);
    case 'float64': return new Float64Array(buf, off, n);
    case 'float16': {
      const raw = new Uint16Array(buf, off, n);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = f16tab[raw[i]];
      return out;
    }
  }
  throw new Error(`dtype ${spec.dtype}`);
}

// 首屏文件预先 gzip 过，浏览器里用 DecompressionStream 解开；不支持时退回未压缩版本
export async function fetchData(url) {
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

// 分块只存了 .gz：浏览器不支持解压时直接失败（err.unsupported），不去请求不存在的原始文件
export const NO_GUNZIP = L('当前浏览器不支持解压，这部分数据无法显示', 'This browser can’t decompress gzip, so this part of the data can’t be shown');
async function fetchGz(url) {
  if (typeof DecompressionStream === 'undefined') throw Object.assign(new Error(NO_GUNZIP), { unsupported: true });
  const r = await fetch(`${url}.gz`);
  if (!r.ok || !r.body) throw new Error(`${url}.gz ${r.status}`);
  return new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}

export const decodeJSON = (buf) => JSON.parse(new TextDecoder().decode(buf));
const json = async (url) => decodeJSON(await fetchData(url));

// .bin 分块：u32 头长度 + JSON 头 {bin, json} + 补齐到 8 字节 + 数组
export function readChunk(buf) {
  const h = new DataView(buf).getUint32(0, true);
  const head = decodeJSON(new Uint8Array(buf, 4, h));
  const base = 4 + h + ((8 - ((4 + h) % 8)) % 8);
  const A = {};
  for (const [k, spec] of Object.entries(head.bin || {})) A[k] = view(buf, spec, base);
  return { A, J: head.json || null };
}

/* ---------------------------------------------------------------- 分块载入器 */

const RETRY_MS = 4000;          // 失败后隔这么久才会再试，之后每失败一次间隔翻倍，最长 30 秒
const MAX_INFLIGHT = 4;         // 同时在路上的分块
const RECENT_MS = 400;          // 这么久没人再要的请求，降回后台的优先级

// 每一块只发一个请求；want(key, 优先级) 每帧由页面调用（3 = 这一屏马上要用，2 = 播放 / 相邻，1 = 很可能下一步要用），
// pump() 按“优先级高、最近要过”的顺序发请求；没人要的时候，后台按 bgOrder 在浏览器空闲时一块一块地取。
export class Loader {
  constructor() {
    this.items = new Map();
    this.order = [];
    this.inflight = 0;
    this.listeners = new Set();
    this.bg = false;
    this.bgTimer = 0;
    this.bytes = 0;
  }

  define(key, url, decode) { const it = { key, url, decode, state: 'idle', prio: 0, stamp: 0 }; this.items.set(key, it); this.order.push(key); }
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  has(key) { return this.items.get(key)?.state === 'ok'; }
  // 'ok' | 'loading' | 'idle' | 'err'
  state(key) { return this.items.get(key)?.state ?? 'none'; }
  error(key) { const it = this.items.get(key); return it?.state === 'err' ? it.err : null; }
  // 还在等（没到、也没失败）
  pending(key) { const s = this.state(key); return s === 'idle' || s === 'loading'; }

  want(key, prio) {
    const it = this.items.get(key);
    if (!it || it.state === 'ok' || it.state === 'loading') return;
    if (it.state === 'err' && (it.err?.unsupported || performance.now() - it.t < Math.min(30000, RETRY_MS * 2 ** (it.fails - 1)))) return;
    if (it.state === 'err') it.state = 'idle';
    it.prio = Math.max(prio, performance.now() - it.stamp < RECENT_MS ? it.prio : 0);
    it.stamp = performance.now();
  }

  pump() {
    if (this.inflight >= MAX_INFLIGHT) return;
    const now = performance.now();
    const cand = [];
    for (const it of this.items.values()) if (it.state === 'idle' && it.prio > 0 && now - it.stamp < RECENT_MS) cand.push(it);
    cand.sort((a, b) => b.prio - a.prio || b.stamp - a.stamp);
    for (const it of cand) {
      if (this.inflight >= MAX_INFLIGHT) break;
      this.start(it);
    }
  }

  start(it) {
    it.state = 'loading';
    this.inflight++;
    const p = fetchGz(it.url).then((buf) => {
      this.bytes += buf.byteLength;
      it.val = it.decode(buf);
      it.state = 'ok';
    }, (e) => {
      it.state = 'err';
      it.err = e;
      it.t = performance.now();
      it.fails = (it.fails || 0) + 1;
      if (!e.unsupported) console.warn('分块载入失败，稍后重试：', it.url, e.message);
    }).finally(() => {
      this.inflight--;
      for (const fn of this.listeners) fn(it.key, it.state);
      this.pump();
      this.idle();
    });
    return p;
  }

  // 后台预取：浏览器空闲、没有别的请求在路上时，按 order 取下一块。省流量模式 / 2G 网络不预取
  startBackground(order) {
    const c = navigator.connection;
    if (c && (c.saveData || /(^|-)2g$/.test(c.effectiveType || ''))) return false;
    if (order) this.order = order.filter((k) => this.items.has(k));
    this.bg = true;
    this.idle();
    return true;
  }

  idle() {
    if (!this.bg || this.bgTimer || this.inflight > 0) return;
    const next = this.order.find((k) => this.items.get(k).state === 'idle');
    if (!next) return;
    const go = () => {
      this.bgTimer = 0;
      if (this.inflight > 0) return;
      const it = this.items.get(next);
      if (it.state === 'idle') this.start(it);
      else this.idle();
    };
    this.bgTimer = typeof requestIdleCallback === 'function' ? requestIdleCallback(go, { timeout: 1500 }) : setTimeout(go, 200);
  }
}

/* ---------------------------------------------------------------- 小模型 */

const pad2 = (k) => String(k).padStart(2, '0');

export async function loadTiny(loader, base = 'data/') {
  const [meta, buf] = await Promise.all([json(`${base}tiny.json`), fetchData(`${base}tiny.bin`)]);
  const A = {};
  for (const [k, spec] of Object.entries(meta.bin)) A[k] = view(buf, spec);
  const M = meta.model, NL = M.layers, H = M.heads, K = meta.ckpts.length;
  const Lv = meta.held.ids.length - 1, R = meta.rowN, NB = NL + 1, TOP = 5;
  const NP = meta.pcaIds.length, C = meta.crops[0].rows;
  const S = meta.train.steps;
  const TRI = (Lv * (Lv + 1)) / 2;
  // 分块到了以后放在这里：CK[k] / ST[k] = { A: 数组, J: json }
  const CK = new Array(K).fill(null), ST = new Array(K).fill(null);
  let FEAT = null;
  const dir = `${base}${meta.chunks.dir}`;
  for (let k = 0; k < K; k++) {
    loader.define(`tiny:ck:${k}`, `${dir}/ck${pad2(k)}.bin`, (b) => (CK[k] = readChunk(b)));
    loader.define(`tiny:st:${k}`, `${dir}/st${pad2(k)}.bin`, (b) => (ST[k] = readChunk(b)));
  }
  loader.define('tiny:feat', `${dir}/feat.bin`, (b) => (FEAT = readChunk(b)));
  const D = {
    kind: 'tiny', meta, A, M, NL, H, K, Lv, R, NB, S, NP, C,
    vocab: meta.vocab,
    ck: (k) => meta.ckpts[k],
    ch: (id) => meta.vocab[id] ?? '?',
    // 分块：part = 'ck' | 'st' | 'feat'
    key: (part, k) => (part === 'feat' ? 'tiny:feat' : `tiny:${part}:${k}`),
    has: (part, k) => (part === 'ck' ? !!CK[k] : part === 'st' ? !!ST[k] : !!FEAT),
    // 离 k 最近的、已经到了的 D1 分块（拖动时先拿它顶上）；一个都没有返回 -1
    nearestCk(k) {
      for (let d = 0; d < K; d++) { if (k - d >= 0 && CK[k - d]) return k - d; if (k + d < K && CK[k + d]) return k + d; }
      return -1;
    },
    // 留出的《登鹳雀楼》：第 i 个位置预测第 i+1 个字
    valP: (k, i) => CK[k].A.val_p[i],
    valTop(k, i) { const a = CK[k].A, out = []; for (let j = 0; j < TOP; j++) out.push({ id: a.val_top_id[i * TOP + j], p: a.val_top_p[i * TOP + j] }); return out; },
    // 注意力只存了下三角（j ≤ i）；上三角是因果遮罩，本来就是 0
    attn: (k, l, h, i, j) => (j > i ? 0 : CK[k].A.attn[(l * H + h) * TRI + (i * (i + 1)) / 2 + j] / 255),
    pca: (k, n) => { const a = CK[k].A.pca; return [a[n * 2], a[n * 2 + 1]]; },
    wcrop: (k, c, i, j) => CK[k].A.wcrop[(c * C + i) * C + j] * meta.ckpts[k].wScale[c],
    gcrop: (k, c, i, j) => CK[k].A.gcrop[(c * C + i) * C + j] * meta.ckpts[k].gScale[c],
    // 批次第 0 行（前 R 个位置）。字本身在首屏数据里，其余在 st 分块里
    rowId: (k, i) => A.row_ids[k * (R + 1) + i],
    rowP: (k, i) => ST[k].A.row_p[i],
    rowPAfter: (k, i) => ST[k].A.row_p_after[i],
    rowTop(k, i) { const a = ST[k].A, out = []; for (let j = 0; j < TOP; j++) out.push({ id: a.row_top_id[i * TOP + j], p: a.row_top_p[i * TOP + j] }); return out; },
    lensTop: (k, b, i) => ST[k].A.lens_top[b * R + i],
    lensP: (k, b, i) => ST[k].A.lens_p[b * R + i],
    residNorm: (k, b, i) => ST[k].A.resid_norm[b * R + i],
    residGrad: (k, b, i) => ST[k].A.resid_grad[b * R + i],
    tgrad: (k, n) => ST[k].A.tgrad[n],
    adam: (k, f) => ST[k].J.adam[f],
    loss: A.loss, lr: A.lr, gnorm: A.gnorm,
    feat: (f) => {
      const a = FEAT.A, n = a.feat_w.length / meta.feats.length;
      return { w: a.feat_w.subarray(f * n, (f + 1) * n), g: a.feat_g.subarray(f * n, (f + 1) * n), m: a.feat_m.subarray(f * n, (f + 1) * n), v: a.feat_v.subarray(f * n, (f + 1) * n), n, stride: meta.featStride };
    },
    land: (iy, ix) => A.land[iy * meta.land.G + ix],
  };
  D.tIndex = new Map(meta.tensors.map((n, i) => [n, i]));
  // 训练集里的字频顺序就是编号顺序（1 号最常见）；PCA 里每个点是哪个字
  D.pcaChars = meta.pcaIds.map((id) => D.ch(id));
  return D;
}

/* ---------------------------------------------------------------- Qwen3-0.6B */

export async function loadQwen(loader, base = 'data/') {
  const meta = await json(`${base}qwen.json`);
  const N = meta.ids.length;
  const K = meta.steps.length;
  // 第 k 步的逻辑透镜、残差范数 / 梯度、张量梯度范数在分块里；到了就并回 meta.steps[k]，其余代码照旧读 meta
  const got = new Array(K).fill(false);
  for (let k = 0; k < K; k++) {
    loader.define(`qwen:st:${k}`, `${base}${meta.chunks.dir}/st${k}.json`, (b) => {
      const js = decodeJSON(b);
      if (js.pretrainCompare) { Object.assign(meta.pretrainCompare, js.pretrainCompare); delete js.pretrainCompare; }
      Object.assign(meta.steps[k], js);
      got[k] = true;
    });
  }
  const D = {
    kind: 'qwen', meta, N,
    NL: meta.model.layers,
    NB: meta.model.layers + 1,
    K,
    key: (part, k) => `qwen:${part}:${k}`,
    has: (part, k) => part === 'st' && !!got[k],
    tok: (id) => meta.vocab[String(id)] ?? `#${id}`,
    roleOf: (i) => meta.roles[i],
    // 位置 i 预测第 i+1 个词元
    sftPos: meta.sftMask.map((m, i) => (m ? i : -1)).filter((i) => i >= 0),
  };
  return D;
}
