// 第三章的数据：原来那份三步 SFT 的记录（../data.js 的 loadQwen → qwen.json + qwen/stN）
// + 给 3D 机器补记的一份（tools/train/qwen_step_3d.py：同样设置重跑、逐项核对过）→ qwen3d.json + qwen3d/stN.bin：
//   每层注意力（16 头平均）、注意力 / 前馈输出长度、中间残差的梯度、每个张量的 ‖Δw‖、每个张量一个跟踪权重的 AdamW 全过程、
//   28 层 × 7 个矩阵的梯度 / Δw 缩略图、嵌入表梯度最大的行、存回 bf16 会变回原值的个数。
// 另外借推理页的权重缩略图（../../data/weights.bin：每格 32×32 个权重的均方根，原模型 = 微调的起点），给权重面板贴上真实的分布。
import { fetchData, readChunk, decodeJSON } from '../data.js';
import { wrapQwen } from '../run.js';

export const TKEY = {
  ln1: 'input_layernorm.weight', q: 'self_attn.q_proj.weight', qn: 'self_attn.q_norm.weight', k: 'self_attn.k_proj.weight', kn: 'self_attn.k_norm.weight',
  v: 'self_attn.v_proj.weight', o: 'self_attn.o_proj.weight', ln2: 'post_attention_layernorm.weight', gate: 'mlp.gate_proj.weight', up: 'mlp.up_proj.weight', down: 'mlp.down_proj.weight',
};
export const LAYER_T = ['ln1', 'q', 'qn', 'k', 'kn', 'v', 'o', 'ln2', 'gate', 'up', 'down'];
export const MAT_T = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'];
export const tname = (L, t) => (t === 'embed' ? 'model.embed_tokens.weight' : t === 'norm' ? 'model.norm.weight' : `model.layers.${L}.${TKEY[t]}`);
// torch 里的形状 [输出, 输入]（向量只有一维）
export const TSHAPE = { embed: [151936, 1024], norm: [1024], ln1: [1024], ln2: [1024], qn: [128], kn: [128], q: [2048, 1024], k: [1024, 1024], v: [1024, 1024], o: [1024, 2048], gate: [3072, 1024], up: [3072, 1024], down: [1024, 3072] };
export const numel = (t) => TSHAPE[t].reduce((a, b) => a * b, 1);
export const isVec = (t) => TSHAPE[t].length === 1;

const B1 = 0.9, B2 = 0.95;

export async function loadQwen3d(loader, Dq, base = 'data/') {
  const meta = decodeJSON(await fetchData(`${base}qwen3d.json`));
  const K = Dq.K, NL = Dq.NL, N1 = Dq.N - 1;
  const X = new Array(K).fill(null);
  for (let k = 0; k < K; k++) loader.define(`qwen3d:st:${k}`, `${base}qwen3d/st${k}.bin`, (b) => (X[k] = readChunk(b)));
  const R = wrapQwen(Dq);
  const m = Dq.meta;
  const tok = (id) => m.vocab[String(id)] ?? meta.vocab[String(id)] ?? `#${id}`;
  const ans = Dq.sftPos;   // 回答的 8 个位置（位置 i 预测第 i+1 个词元）

  // 权重缩略图（推理页导出的，原模型）：用到时才取
  let thumbs = null, thumbP = null;
  const loadThumbs = () => {
    if (thumbs || thumbP) return thumbP;
    thumbP = Promise.all([fetch(`${base}../../data/manifest.json`).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }), fetchData(`${base}../../data/weights.bin`)])
      .then(([man, buf]) => { thumbs = { index: man.thumbs.index, bytes: new Uint8Array(buf) }; return thumbs; })
      .catch((e) => { console.warn('权重缩略图载入失败', e); thumbP = null; throw e; });
    return thumbP;
  };

  const st = (k) => m.steps[k];
  const layerSum = (obj, L, f) => { let s = 0; for (const t of LAYER_T) { const v = f(obj[tname(L, t)]); s += v * v; } return Math.sqrt(s); };

  Object.assign(R, {
    kind: 'qwen3d', meta3: meta, ans, tok, N1,
    key3: (k) => `qwen3d:st:${k}`,
    keySt: (k) => `qwen:st:${k}`,
    has3: (k) => !!X[k],
    hasSt: (k) => Dq.has('st', k),
    x3: (k) => X[k],
    loadThumbs,
    get thumbs() { return thumbs; },
    // 目标词元（位置 i 要猜的）
    tgt: (i) => m.ids[i + 1],
    tgtStr: (i) => tok(m.ids[i + 1]),
    inStr: (i) => tok(m.ids[i]),
    pState: (s, i) => m.states[s].p[i],
    nllState: (s, i) => m.states[s].nll[i],
    lossState: (s) => m.states[s].lossSft,
    lossPreState: (s) => m.states[s].lossPre,
    topState: (s, i) => m.states[s].top[i].map(([id, p]) => ({ id, s: tok(id), p })),
    // 第 k 步每一层的梯度长度（这一层 11 个张量合起来，裁剪前）
    layerGrad: (k, L) => (Dq.has('st', k) ? layerSum(st(k).gradNorms, L, (v) => v ?? 0) : 0),
    layerDw: (k, L) => (X[k] ? layerSum(X[k].J.tensors, L, (v) => v?.[0] ?? 0) : 0),
    gradOf: (k, name) => (Dq.has('st', k) ? st(k).gradNorms[name] ?? 0 : 0),
    gradPre: (name) => m.pretrainCompare.gradNorms?.[name] ?? null,
    // 一个张量在第 k 步的数：梯度（裁剪前）、更新量、更新前的长度、变了几个 / 存回 bf16 会变回原值的有几个（和原权重比）
    tensor(k, L, t) {
      const name = tname(L, t);
      const ts = X[k]?.J.tensors[name], rv = X[k]?.J.revTensor[name];
      return { name, g: this.gradOf(k, name), gPre: k === 0 ? this.gradPre(name) : null, dw: ts?.[0] ?? null, dwMax: ts?.[1] ?? null, w: ts?.[2] ?? null, changed: rv?.[0] ?? null, revert: rv?.[1] ?? null, n: numel(t), clip: m.steps[k].clip };
    },
    // 跟踪的那个权重（每个张量一个）：位置 + 第 k 步的 AdamW 全过程
    trackedIndex: (L, t) => meta.tracked[tname(L, t)],
    adamT(k, L, t) {
      const a = X[k]?.J.tracked[tname(L, t)];
      if (!a) return null;
      const tt = k + 1;
      return { ...a, t: tt, bc1: 1 - B1 ** tt, bc2: 1 - B2 ** tt, lr: m.train.lr, eps: m.train.eps };
    },
    // 第 L 层注意力：位置 i 看位置 j 的权重（16 个头平均）
    att(k, L, i, j) { const a = X[k]?.A.att; return a ? a[(L * N1 + i) * N1 + j] / 255 : 0; },
    attnOut: (k, L, i) => X[k]?.J.attnOut[L][i] ?? 0,
    mlpOut: (k, L, i) => X[k]?.J.mlpOut[L][i] ?? 0,
    midGrad: (k, L, i) => X[k]?.J.midGrad[L][i] ?? 0,
    // 缩略图：每格 64×64 个数的均方根，取对数（3 个数量级）量化到 0..255；行 = 输入（第 0 行在上）、列 = 输出
    thumb(k, L, t, kind = 'g') {
      const x = X[k];
      if (!x || !MAT_T.includes(t)) return null;
      const sp = meta.thumbIndex[`${L}:${t}`], arr = kind === 'g' ? x.A.gthumb : x.A.dthumb;
      const mi = L * 7 + MAT_T.indexOf(t);
      return { data: arr.subarray(sp.offset, sp.offset + sp.w * sp.h), w: sp.w, h: sp.h, mx: (kind === 'g' ? x.J.gmax : x.J.dmax)[mi], decades: meta.decades, block: meta.block };
    },
    emb: (k) => X[k]?.J.emb ?? null,
    changedAfter: (k) => meta.steps[k].changed,
    revertAfter: (k) => meta.steps[k].revertBf16,
    total: meta.total,
  });
  return R;
}
