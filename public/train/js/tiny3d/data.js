// 唐宋诗小模型这一章（3D 机器）用到的数据：沿用 ../data.js 的 loadTiny（首屏 + 按检查点的 ck / st 分块 + feat），
// 另加一份补充分块 tiny3d/dwNN（tools/train/tiny3d_dw.py 从检查点在 CPU 上算的）：
//   三块权重局部到下一个检查点的真实 ΔW（int8 + 比例），每个参数张量的 RMS(W)、到下一个检查点的 RMS(ΔW)。
import { readChunk } from '../data.js';
import { L } from '../lang.js';

const pad2 = (k) => String(k).padStart(2, '0');
const SUFFIX = {
  q: 'self_attn.q_proj.weight', k: 'self_attn.k_proj.weight', v: 'self_attn.v_proj.weight', o: 'self_attn.o_proj.weight',
  qn: 'self_attn.q_norm.weight', kn: 'self_attn.k_norm.weight', gate: 'mlp.gate_proj.weight', up: 'mlp.up_proj.weight', down: 'mlp.down_proj.weight',
  ln1: 'input_layernorm.weight', ln2: 'post_attention_layernorm.weight',
};
export const tensorName = (l, kind) => (kind === 'E' ? 'model.embed_tokens.weight' : kind === 'nf' ? 'model.norm.weight' : `model.layers.${l}.${SUFFIX[kind]}`);
// 显示用的名字和形状（矩阵写成 [输入 × 输出]，和机器上 y = x · W 的摆法一致）
export const KIND_LABEL = { E: 'E', nf: 'γ_f', ln1: 'γ₁', ln2: 'γ₂', q: 'W_q', k: 'W_k', v: 'W_v', o: 'W_o', qn: 'q_norm γ', kn: 'k_norm γ', gate: 'W_gate', up: 'W_up', down: 'W_down' };
export function kindShape(kind) {
  return { E: [7478, 256], nf: [256], ln1: [256], ln2: [256], q: [256, 256], k: [256, 128], v: [256, 128], o: [256, 256], qn: [64], kn: [64], gate: [256, 768], up: [256, 768], down: [768, 256] }[kind];
}
export const kindCount = (kind) => kindShape(kind).reduce((a, b) => a * b, 1);
// 三块权重局部：c = 0 嵌入、1 第 2 层 W_q、2 第 4 层 W_down
export const CROPS = [{ l: -1, kind: 'E' }, { l: 2, kind: 'q' }, { l: 4, kind: 'down' }];
export const cropOf = (l, kind) => CROPS.findIndex((c) => c.kind === kind && (c.l === l || kind === 'E'));

export function wrapTiny3d(R, loader) {
  const D = R.D, m = D.meta, K = D.K, C = 48;
  const DW = new Array(K).fill(null);
  for (let k = 0; k < K; k++) loader.define(`tiny3d:dw:${k}`, `data/tiny3d/dw${pad2(k)}.bin`, (b) => (DW[k] = readChunk(b)));
  const tIdx = D.tIndex;
  const X = {
    R, D, meta: m, K, C,
    S: m.train.steps, NPOS: m.rowN, NL: D.NL, H: D.H,
    key: (part, k) => (part === 'dw' ? `tiny3d:dw:${k}` : D.key(part, k)),
    has: (part, k) => (part === 'dw' ? !!DW[k] : D.has(part, k)),
    // 检查点 → 训练步数（0 起）
    step: (k) => m.ckpts[k].t,
    ch: (id) => (id === 0 ? '⏎' : D.ch(id)),
    // 批次第 0 行：第 i 个字（i = 0..64），位置 i 的答案是第 i+1 个字
    row: (k, i) => D.rowId(k, i),
    tgt: (k, i) => D.rowId(k, i + 1),
    rowP: (k, i) => D.rowP(k, i),
    rowTop: (k, i) => D.rowTop(k, i),
    rowPAfter: (k, i) => D.rowPAfter(k, i),
    lensTop: (k, b, i) => D.lensTop(k, b, i),
    lensP: (k, b, i) => D.lensP(k, b, i),
    residNorm: (k, b, i) => D.residNorm(k, b, i),
    residGrad: (k, b, i) => D.residGrad(k, b, i),
    tgrad: (k, l, kind) => D.tgrad(k, tIdx.get(tensorName(l, kind))),
    tgradAll: (k) => { let mx = 0; for (let i = 0; i < m.tensors.length; i++) mx = Math.max(mx, D.tgrad(k, i)); return mx; },
    // 局部：PyTorch 的下标 [i, j]（嵌入：第 i+1 号字的第 j 维；矩阵：[输出 i, 输入 j]）
    wcrop: (k, c, i, j) => D.wcrop(k, c, i, j),
    gcrop: (k, c, i, j) => D.gcrop(k, c, i, j),
    dwcrop: (k, c, i, j) => { const d = DW[k]; return d && d.A.dwcrop ? d.A.dwcrop[(c * C + i) * C + j] * d.J.dwScale[c] : 0; },
    hasDw: (k) => !!DW[k]?.A.dwcrop,
    dwSpan: (k) => (DW[k]?.J.t1 != null ? [DW[k].J.t, DW[k].J.t1] : null),
    wRms: (k, l, kind) => DW[k]?.J.wRms[tIdx.get(tensorName(l, kind))] ?? null,
    dwRms: (k, l, kind) => DW[k]?.J.dwRms?.[tIdx.get(tensorName(l, kind))] ?? null,
    dwRmsMax: (k) => { const a = DW[k]?.J.dwRms; return a ? Math.max(...a) : 0; },
    attn: (k, l, h, i, j) => D.attn(k, l, h, i, j),
    lr: (k) => m.ckpts[k].lr,
    loss: (k) => m.ckpts[k].loss,
    lossAfter: (k) => m.ckpts[k].lossAfter,
    rowLoss: (k) => m.ckpts[k].rowLoss,
    gnorm: (k) => m.ckpts[k].gradNorm,
    clip: (k) => m.ckpts[k].clip,
    valLoss: (k) => m.ckpts[k].valLoss,
    adam: (k, f) => D.adam(k, f),
    feats: m.feats,
    held: m.held,
    vocab: m.model.vocab,
  };
  // 训练损失的指数滑动平均（带偏差校正），曲线用
  const LS = D.loss;
  X.ema = new Float32Array(LS.length);
  let e = 0;
  for (let i = 0; i < LS.length; i++) { e = 0.96 * e + 0.04 * LS[i]; X.ema[i] = e / (1 - 0.96 ** (i + 1)); }
  X.name = L('唐宋诗小模型', 'Tiny poetry model');
  return X;
}
