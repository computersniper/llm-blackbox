// 把两段训练记录包成同一套接口，让“一步之内”往下的各个视图不用关心是哪个模型。
import { L } from './lang.js';

const TENSOR = {
  q: 'self_attn.q_proj.weight', k: 'self_attn.k_proj.weight', v: 'self_attn.v_proj.weight', o: 'self_attn.o_proj.weight',
  gate: 'mlp.gate_proj.weight', up: 'mlp.up_proj.weight', down: 'mlp.down_proj.weight',
  ln1: 'input_layernorm.weight', ln2: 'post_attention_layernorm.weight', qn: 'self_attn.q_norm.weight', kn: 'self_attn.k_norm.weight',
};
export const TENSOR_NAME = { q: 'W_q', k: 'W_k', v: 'W_v', o: 'W_o', gate: 'W_gate', up: 'W_up', down: 'W_down', ln1: L('RMSNorm γ（注意力前）', 'RMSNorm γ (pre-attention)'), ln2: L('RMSNorm γ（前馈前）', 'RMSNorm γ (pre-FFN)'), qn: 'q_norm γ', kn: 'k_norm γ' };

// 矩阵形状 [输入维, 输出维]
export function shapes(M) {
  const qd = M.heads * (M.headDim || M.hidden / M.heads), kd = M.kvHeads * (M.headDim || M.hidden / M.heads);
  return { q: [M.hidden, qd], k: [M.hidden, kd], v: [M.hidden, kd], o: [qd, M.hidden], gate: [M.hidden, M.ffn], up: [M.hidden, M.ffn], down: [M.ffn, M.hidden] };
}

export function wrapTiny(D) {
  const m = D.meta;
  const tIdx = D.tIndex;
  return {
    kind: 'tiny', D, K: D.K, NL: D.NL, NB: D.NB, R: D.R,
    model: { ...m.model, headDim: m.model.headDim },
    name: L('唐宋诗小模型', 'Tiny poetry model'),
    rowLen: D.R,
    stepNo: (k) => m.ckpts[k].t + 1,
    lr: (k) => m.ckpts[k].lr,
    loss: (k) => m.ckpts[k].loss,
    lossAfter: (k) => m.ckpts[k].lossAfter,
    gradNorm: (k) => m.ckpts[k].gradNorm,
    clip: (k) => m.ckpts[k].clip,
    // 输入的第 i 个词元（i = 0..R），目标是第 i+1 个
    tok: (k, i) => { const id = D.rowId(k, i); return id === 0 ? '⏎' : D.ch(id); },
    tokId: (k, i) => D.rowId(k, i),
    isSep: (k, i) => D.rowId(k, i) === 0,
    role: () => 'char',
    counted: () => true,
    p: (k, i) => D.rowP(k, i),
    pAfter: (k, i) => D.rowPAfter(k, i),
    top: (k, i) => D.rowTop(k, i).map((t) => ({ s: t.id === 0 ? '⏎' : D.ch(t.id), p: t.p, id: t.id })),
    lensTop: (k, b, i) => { const id = D.lensTop(k, b, i); return { s: id === 0 ? '⏎' : D.ch(id), ok: id === D.rowId(k, i + 1) }; },
    lensP: (k, b, i) => D.lensP(k, b, i),
    residNorm: (k, b, i) => D.residNorm(k, b, i),
    residGrad: (k, b, i) => D.residGrad(k, b, i),
    tgrad(k, L, t) {
      const name = L < 0 ? (t === 'embed' ? 'model.embed_tokens.weight' : 'model.norm.weight') : `model.layers.${L}.${TENSOR[t]}`;
      return D.tgrad(k, tIdx.get(name));
    },
    feats: m.feats,
    adam: (k, f) => D.adam(k, f),
    featHist: (f) => D.feat(f),
    // 一步之内（D2–D4）的数据分块到了没有
    stepReady: (k) => D.has('st', k),
    batchShape: L(`${m.train.batch} 行 × ${m.train.seq + 1} 字`, `${m.train.batch} rows × ${m.train.seq + 1}`),
    tokensSeen: (k) => m.ckpts[k].t * m.train.tokensPerStep,
  };
}

export function wrapQwen(D) {
  const m = D.meta;
  const N = D.N;
  return {
    kind: 'qwen', D, K: D.K, NL: D.NL, NB: D.NB, R: N - 1, sftPos: D.sftPos,
    model: { ...m.model, headDim: 128 },
    name: 'Qwen3-0.6B',
    rowLen: N - 1,
    stepNo: (k) => k + 1,
    lr: () => m.train.lr,
    loss: (k) => m.steps[k].loss,
    lossAfter: (k) => m.states[k + 1].lossSft,
    gradNorm: (k) => m.steps[k].gradNorm,
    clip: (k) => m.steps[k].clip,
    tok: (k, i) => D.tok(m.ids[i]),
    tokId: (k, i) => m.ids[i],
    isSep: () => false,
    role: (i) => m.roles[i],
    counted: (i) => m.sftMask[i],
    p: (k, i) => m.states[k].p[i],
    pAfter: (k, i) => m.states[k + 1].p[i],
    top: (k, i) => m.states[k].top[i].map(([id, p]) => ({ s: D.tok(id), p, id })),
    lensTop: (k, b, i) => { const id = m.steps[k].lens.top[b][i]; return { s: D.tok(id), ok: id === m.ids[i + 1] }; },
    lensP: (k, b, i) => m.steps[k].lens.p[b][i],
    residNorm: (k, b, i) => m.steps[k].residNorm[b][i],
    residGrad: (k, b, i) => m.steps[k].residGrad[b][i],
    tgrad(k, L, t) {
      const name = L < 0 ? (t === 'embed' ? 'model.embed_tokens.weight' : 'model.norm.weight') : `model.layers.${L}.${TENSOR[t]}`;
      return m.steps[k].gradNorms[name] ?? 0;
    },
    feats: m.feats,
    adam: (k, f) => ({ t: k + 1, ...m.steps[k].feats[f] }),
    featHist: null,
    stepReady: (k) => D.has('st', k),
    batchShape: L(`1 条对话 × ${N} 个词元`, `1 × ${N} tokens`),
    tokensSeen: (k) => k * N,
  };
}
