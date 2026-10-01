// 训练页：真实训练录像（tools/train_poet.py 导出）。
//
//   D1 训练全程   31 个录制点（第 1、100、200 … 3000 步）：损失曲线、生成的诗、“静夜思”逐字预测、字向量
//   D2 一步训练   取数据 / 前向 / 损失 / 反向 / 裁剪 / AdamW
//   D3 逐层       前向第 0→3 层；反向第 3→0 层；损失逐位置；AdamW 的每一行公式
//   D4 一层之内   RMSNorm → 注意力 → ⊕ → RMSNorm → SwiGLU → ⊕（反向：前馈的梯度、注意力的梯度）
//   D5 头 / 神经元 4 个注意力头各自的注意力矩阵；SwiGLU 的升维 / 门控 / 降维及活跃神经元摘要

import { Lab, fetchJSON, fmt, fmtP, heat, linePath } from '../js/lab/core.js';
import { TraceStage } from '../js/lab/stage3d.js';
import { $, esc } from '../js/ui.js';

const ctx = { pos: 11, head: null };
let run = null;
const steps = new Map();
let traceStage;
let scenePicks = new Map();

// ---------------------------------------------------------------- 数据

async function loadStep(s) {
  if (!steps.has(s)) steps.set(s, fetchJSON(`data/step_${String(s).padStart(4, '0')}.json`).then((d) => { steps.set(s, d); return d; }));
  return steps.get(s);
}
const stepData = (s) => { const d = steps.get(s); return d && !(d instanceof Promise) ? d : null; };

// ---------------------------------------------------------------- 步骤树

const OPS = [
  ['ln1', 'RMSNorm', 1.0], ['attn', '注意力', 1.8], ['add1', '⊕ 残差', 0.9],
  ['ln2', 'RMSNorm', 1.0], ['mlp', 'SwiGLU 前馈', 1.8], ['add2', '⊕ 残差', 0.9],
];

function phases(s) {
  const N = (t, label, extra = {}) => ({ t, s, label, ...extra });
  return [
    N('batch', '取一批数据', { dur: 2.2, kids: () => [N('tok', '按字切分', { dur: 2 }), N('shift', '输入与目标错开一位', { dur: 2 })] }),
    N('fwd', '前向传播', {
      dur: 2.4,
      kids: () => [
        N('emb', '查嵌入表', { dur: 1.6 }),
        ...[0, 1, 2, 3].map((L) => N('layer', `第 ${L} 层`, {
          L, dur: 1.4, crumb: `第 ${L} 层`,
          kids: () => OPS.map(([op, lab, dur]) => N('op', lab, {
            L, op, dur, crumb: lab,
            kids: op === 'attn' ? () => [0, 1, 2, 3].map((h) => N('head', `头 ${h}`, { L, h, dur: 1.6 }))
              : op === 'mlp' ? () => [N('up', '升维 W_gate / W_up', { L, dur: 1.6 }), N('act', '门控 SiLU(g)·u', { L, dur: 1.8 }), N('down', '降维 W_down', { L, dur: 1.6 })]
                : null,
          })),
        })),
        N('head', '输出头', { dur: 1.8, isHead: true }),
      ],
    }),
    N('loss', '算损失', { dur: 2.2, kids: () => [N('probs', 'softmax 概率', { dur: 1.8 }), N('nll', '每个位置 −log p', { dur: 1.8 }), N('mean', '对整批取平均', { dur: 1.6 })] }),
    N('bwd', '反向传播', {
      dur: 2.4,
      kids: () => [
        N('dhead', '输出头的梯度', { dur: 1.6 }),
        ...[3, 2, 1, 0].map((L) => N('blayer', `第 ${L} 层（反向）`, {
          L, dur: 1.4, crumb: `第 ${L} 层 ←`,
          kids: () => [N('bmlp', '前馈的梯度', { L, dur: 1.6 }), N('battn', '注意力的梯度', { L, dur: 1.6 })],
        })),
        N('demb', '嵌入表的梯度', { dur: 1.6 }),
      ],
    }),
    N('clip', '梯度裁剪', { dur: 1.8 }),
    N('upd', 'AdamW 更新', {
      dur: 2.4,
      kids: () => [
        N('apick', '盯住一个权重', { dur: 2 }), N('am', '一阶矩 m', { dur: 1.8 }), N('av', '二阶矩 v', { dur: 1.8 }),
        N('ahat', '偏差修正', { dur: 1.8 }), N('astep', '更新权重', { dur: 2.2 }), N('abits', '看它的比特', { dur: 2.2 }),
      ],
    }),
  ];
}

// ---------------------------------------------------------------- 伪代码

const CODE = [
  'for step in range(1, 3001):',
  '    batch = sample(poems, 64)          # 64 首诗',
  '    x, y = batch[:, :-1], batch[:, 1:] # 目标 = 下一个字',
  '    h = embed[x]                       # [T, 256]',
  '    for L in range(4):',
  '        n = rmsnorm(h)',
  '        h = h + attention(n)           # 4 头 / 2 组 KV',
  '        n = rmsnorm(h)',
  '        h = h + W_down(silu(W_gate n) * W_up n)',
  '    logits = rmsnorm(h) @ embed.T      # 输出头共享嵌入',
  '    p = softmax(logits)',
  '    loss = mean(-log(p[y]))',
  '    loss.backward()                    # 链式法则，从后往前',
  '    clip_grad_norm_(params, 1.0)',
  '    lr = warmup_cosine(step)',
  '    for w in params:                   # AdamW',
  '        m = 0.9 * m + 0.1 * g',
  '        v = 0.95 * v + 0.05 * g * g',
  '        mh, vh = m / (1 - 0.9**t), v / (1 - 0.95**t)',
  '        w = w * (1 - lr * 0.1) - lr * mh / (sqrt(vh) + 1e-8)',
];

const LINES = {
  ck: [1], batch: [2, 3], tok: [2], shift: [3], fwd: [4, 5, 6, 7, 8, 9, 10], emb: [4], layer: [5, 6, 7, 8, 9],
  ln1: [6], attn: [7], add1: [7], ln2: [8], mlp: [9], add2: [9], head: [7], up: [9], act: [9], down: [9],
  isHead: [10, 11], loss: [11, 12], probs: [11], nll: [12], mean: [12], bwd: [13], dhead: [13], blayer: [13], bmlp: [13], battn: [13], demb: [13],
  clip: [14], upd: [15, 16, 17, 18, 19, 20], apick: [16], am: [17], av: [18], ahat: [19], astep: [20], abits: [20],
};

// ---------------------------------------------------------------- 小部件

const chars = (d) => d.tokens.slice(0, d.T); // 输入（第 i 个位置预测第 i+1 个字）
const tgt = (d, i) => d.tokens[i + 1];
const showTok = (c) => (c === '<s>' ? '⟨s⟩' : c === '</s>' ? '⟨/s⟩' : c);

function chipRow(d, { cls = () => '', sub = () => '', click = true } = {}) {
  return `<div class="chips pos-chips">${chars(d).map((c, i) => `<span class="chip ${click ? 'click' : ''} ${i === ctx.pos ? 'cur' : ''} ${c.length > 1 ? 'sp' : ''} ${cls(i)}" data-pos="${i}" title="位置 ${i}：输入“${esc(c)}”，目标“${esc(tgt(d, i))}”">${esc(showTok(c))}<small>${sub(i)}</small></span>`).join('')}</div>`;
}

function lossChart(s, { w = 640, h = 190 } = {}) {
  const L = run.log.loss, n = L.length;
  const ema = [];
  let e = L[0];
  for (let i = 0; i < n; i++) { e = i ? 0.95 * e + 0.05 * L[i] : L[i]; ema.push(e); }
  const xs = L.map((_, i) => i + 1);
  const y0 = Math.floor(Math.min(...L, ...run.val.map((v) => v[1])) * 10) / 10 - 0.2, y1 = Math.ceil(Math.max(...L.slice(0, 5)) * 2) / 2;
  const px = (x) => (x / n) * w, py = (y) => h - ((y - y0) / (y1 - y0)) * h;
  const raw = linePath(xs, L, 0, n, y0, y1, w, h);
  const sm = linePath(xs, ema, 0, n, y0, y1, w, h);
  const vp = linePath(run.val.map((v) => Math.max(1, v[0])), run.val.map((v) => v[1]), 0, n, y0, y1, w, h);
  const ticks = [];
  for (let y = Math.ceil(y0); y <= y1; y++) ticks.push(`<line x1="0" x2="${w}" y1="${py(y)}" y2="${py(y)}" class="grid"/><text x="4" y="${py(y) - 3}" class="tk">${y}</text>`);
  for (const x of [500, 1000, 1500, 2000, 2500, 3000]) ticks.push(`<text x="${px(x) - 4}" y="${h + 14}" class="tk" text-anchor="end">${x}</text>`);
  const marks = run.detail.map((d) => `<circle cx="${px(d)}" cy="${py(ema[d - 1])}" r="${d === s ? 5 : 2.2}" class="${d === s ? 'mk on' : 'mk'}" data-step="${d}"/>`).join('');
  return `<svg class="fit loss-chart" viewBox="0 -6 ${w} ${h + 22}" preserveAspectRatio="none">
    ${ticks.join('')}
    <path d="${raw}" class="raw"/><path d="${sm}" class="sm"/><path d="${vp}" class="val"/>
    <line x1="${px(s)}" x2="${px(s)}" y1="0" y2="${h}" class="cur"/>
    ${marks}
  </svg>
  <div class="legend"><span><i style="background:rgba(94,240,212,.3)"></i>每一步的训练损失</span><span><i style="background:var(--acc)"></i>平滑后</span><span><i style="background:var(--amber)"></i>验证损失（没见过的 512 首诗）</span><span><i style="background:#fff"></i>录制点（可点）</span></div>`;
}

function samplesHTML(s) {
  const list = run.samples[String(s)] || [];
  return list.map((smp, k) => {
    const txt = smp.text;
    const lines = txt.replace(/([，。])/g, '$1\n').split('\n').filter(Boolean);
    return `<div class="poem ${k === list.length - 1 ? 'greedy' : ''}"><div class="pm-h">${k === list.length - 1 ? '贪心（每次取最可能的字）' : `以“${esc(smp.prompt)}”开头 · T=0.8`}</div>${lines.map((l) => `<div>${esc(l)}</div>`).join('') || '<div class="muted">（什么也没写出来）</div>'}</div>`;
  }).join('');
}

function probeHTML(s) {
  const by = run.probe.by[String(s)];
  if (!by) return '';
  const toks = run.probe.tokens;
  const avg = by.reduce((a, b) => a + b.p, 0) / by.length;
  return `<div class="chips">${toks.slice(1).map((c, i) => {
    const p = by[i].p;
    return `<span class="chip" style="background:rgba(94,240,212,${(0.05 + 0.8 * p).toFixed(2)})" title="读到“${esc(toks.slice(0, i + 1).join('').replace('<s>', ''))}”之后：猜“${esc(c)}”的概率 ${fmtP(p)}；它最想写：${by[i].top.map((t) => `${t[0]} ${fmtP(t[1])}`).join('、')}">${esc(showTok(c))}<small>${fmtP(p)}</small></span>`;
  }).join('')}</div><p class="note">每个格子：模型读完前面的字之后，猜中这个字的概率（颜色越亮越有把握）。这首诗<b>不在训练集里</b>；平均 ${fmtP(avg)}。</p>`;
}

function pcaHTML(s, { w = 420, h = 300 } = {}) {
  const by = run.pca.by[String(s)] || run.pca.by['0'];
  // 每个时刻按自己的范围缩放（嵌入向量在训练中整体会变长），取 95% 分位数避免被个别点拉开
  const mags = by.flatMap(([x, y]) => [Math.abs(x), Math.abs(y)]).sort((a, b) => a - b);
  const mx = mags[Math.floor(mags.length * 0.95)] * 1.25 || 1;
  const X = (x) => w / 2 + (x / mx) * (w / 2), Y = (y) => h / 2 - (y / mx) * (h / 2);
  const pts = by.map(([x, y], i) => {
    const c = run.pca.chars[i];
    const cls = /[，。]/.test(c) ? 'pu' : c.length > 1 ? 'sp' : i < 60 ? 'hi' : '';
    return `<text x="${X(x).toFixed(1)}" y="${Y(y).toFixed(1)}" class="pc ${cls}">${esc(c)}</text>`;
  }).join('');
  return `<svg class="fit pca" viewBox="0 0 ${w} ${h}">${pts}</svg><p class="note">最常见的 240 个字，每个字的 256 维嵌入向量投影到平面上（投影方向固定用训练结束时的两个主成分，每个时刻自动缩放）。一开始是随机的一团；训练中意思相近、用法相近的字慢慢聚到一起。</p>`;
}

function barsHTML(items, win) {
  const mx = Math.max(...items.map((t) => t[1]), 1e-9);
  return `<div class="bars">${items.map((t) => `<div class="bar ${t[0] === win ? 'win' : ''}"><span class="t">${esc(showTok(t[0]))}</span><span class="b"><i style="width:${((t[1] / mx) * 100).toFixed(1)}%"></i></span><span class="v">${fmtP(t[1])}</span></div>`).join('')}</div>`;
}

// 层 × 位置 的热力表（残差范数 / 梯度范数 / 逻辑透镜）
function gridHTML(d, rows, { cell, label, cur }) {
  const T = d.T;
  let h = `<div class="lgrid" style="--T:${T}"><span></span>${chars(d).map((c, i) => `<span class="gh ${i === ctx.pos ? 'on' : ''}" data-pos="${i}">${esc(showTok(c))}</span>`).join('')}`;
  rows.forEach((r, k) => {
    h += `<span class="rl ${cur === k ? 'on' : ''}">${label(k)}</span>`;
    for (let i = 0; i < T; i++) h += cell(k, i);
  });
  return h + '</div>';
}

function drawAttn(canvas, rows, T) {
  const px = Math.max(4, Math.min(12, Math.floor(360 / T)));
  canvas.width = T * px; canvas.height = T * px;
  const g = canvas.getContext('2d');
  g.fillStyle = '#060c18'; g.fillRect(0, 0, T * px, T * px);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j <= i; j++) {
      const v = rows[i][j] / 255;
      g.fillStyle = `rgba(94,240,212,${Math.min(1, 0.05 + v * 1.4)})`;
      g.fillRect(j * px, i * px, px - (px > 5 ? 1 : 0), px - (px > 5 ? 1 : 0));
    }
  }
  g.strokeStyle = '#ffb65c'; g.lineWidth = 2;
  g.strokeRect(0, ctx.pos * px, T * px, px);
}

// ---------------------------------------------------------------- 渲染

const view = () => $('#view');

// The spatial stage uses measured values only. The numerical panels below retain the full recorded detail.
const traceNumber = (x, digits = 3) => Number.isFinite(x) ? Number(x).toFixed(digits) : '—';
const sampleLoss = (s) => run.log.loss.slice(Math.max(0, s - 64), s);

function traceValues(n, d) {
  const i = Math.min(ctx.pos, (d?.T || 1) - 1), L = n.L;
  if (n.t === 'ck') return sampleLoss(n.s);
  if (!d) return [];
  if (n.t === 'batch' || n.t === 'tok' || n.t === 'shift') return d.ids.slice(0, d.T);
  if (n.t === 'fwd') return d.fwd.res.at(-1);
  if (n.t === 'emb') return d.fwd.res[0];
  if (n.t === 'layer') return d.fwd.res[L + 1];
  if (n.t === 'op') {
    if (n.op === 'attn') return d.fwd.attn[L][0][i].map((x) => x / 255);
    if (n.op === 'mlp') return d.fwd.mlp[L][i].map((x) => x[1]);
    return d.fwd.res[L + (n.op === 'add2' ? 1 : 0)];
  }
  if (n.t === 'head') return n.isHead ? d.out.pTarget : d.fwd.attn[L][n.h][i].map((x) => x / 255);
  if (n.t === 'act') return d.fwd.mlp[L][i].map((x) => x[1]);
  if (n.t === 'down') return d.fwd.mlpOut[L];
  if (n.t === 'loss' || n.t === 'nll' || n.t === 'mean') return d.out.lossPos;
  if (n.t === 'probs') return d.out.pTarget;
  if (n.t === 'bwd' || n.t === 'demb') return d.bwd.res[0];
  if (n.t === 'blayer') return d.bwd.res[L + 1];
  if (n.t === 'bmlp') return d.bwd.mlpOut[L];
  if (n.t === 'battn') return d.bwd.attnOut[L];
  if (n.t === 'clip') return run.log.gnorm.slice(Math.max(0, n.s - 64), n.s);
  if (n.t === 'upd' || n.t === 'apick') return d.adam.row8;
  if (n.t === 'am') return [d.adam.m0, d.adam.g, d.adam.m1];
  if (n.t === 'av') return [d.adam.v0, d.adam.g ** 2, d.adam.v1];
  if (n.t === 'ahat') return [d.adam.m1, d.adam.v1];
  if (n.t === 'astep' || n.t === 'abits') return [d.adam.w, d.adam.w1];
  return [];
}

function traceDetail(n, d) {
  if (n.t === 'ck') return `训练损失 ${traceNumber(run.log.loss[n.s - 1])} · 近 64 步实测`;
  if (!d) return '载入这一步的实测数值…';
  const i = Math.min(ctx.pos, d.T - 1), L = n.L;
  if (n.t === 'batch') return `${d.batch} 首 · ${d.tokensInBatch} 个目标字`;
  if (n.t === 'fwd') return `末层残差范数 ${traceNumber(d.fwd.res[4][i])}`;
  if (n.t === 'emb') return `${d.T} 个位置 · 256 维字向量`;
  if (n.t === 'layer') return `位置 ${i} · ‖h‖ ${traceNumber(d.fwd.res[L + 1][i])}`;
  if (n.t === 'op') return `第 ${L} 层 · 位置 ${i} · 实测摘要`;
  if (n.t === 'head') return n.isHead ? `目标字概率 ${fmtP(d.out.pTarget[i])}` : `位置 ${i} · K/V 组 ${n.h >> 1} · 实测注意力`;
  if (n.t === 'loss' || n.t === 'nll') return `本批损失 ${traceNumber(d.loss)}`;
  if (n.t === 'bwd') return `裁剪前梯度范数 ${traceNumber(d.gradNorm)}`;
  if (n.t === 'blayer') return `位置 ${i} · ‖∂h‖ ${d.bwd.res[L + 1][i].toExponential(2)}`;
  if (n.t === 'bmlp') return `位置 ${i} · ‖∂m‖ ${d.bwd.mlpOut[L][i].toExponential(2)}`;
  if (n.t === 'battn') return `位置 ${i} · ‖∂a‖ ${d.bwd.attnOut[L][i].toExponential(2)}`;
  if (n.t === 'clip') return `‖g‖ ${traceNumber(d.gradNorm)} → ×${traceNumber(d.clip)}`;
  if (n.t === 'upd' || n.t === 'apick') return `embed[${d.adam.row},${d.adam.col}] · ${traceNumber(d.adam.w, 6)} → ${traceNumber(d.adam.w1, 6)}`;
  if (n.t === 'am') return `m: ${traceNumber(d.adam.m0, 6)} → ${traceNumber(d.adam.m1, 6)}`;
  if (n.t === 'av') return `v: ${traceNumber(d.adam.v0, 6)} → ${traceNumber(d.adam.v1, 6)}`;
  if (n.t === 'astep' || n.t === 'abits') return `w: ${traceNumber(d.adam.w, 6)} → ${traceNumber(d.adam.w1, 6)}`;
  if (n.t === 'act') return '录下最活跃的 8 个神经元';
  if (n.t === 'down') return `位置 ${i} · ‖m‖ ${traceNumber(d.fwd.mlpOut[L][i])}`;
  return n.t === 'up' ? 'W_gate / W_up · 768 维' : '实测训练步骤';
}

function traceKind(n) {
  if (n.t === 'layer' || n.t === 'blayer') return 'layer';
  if (n.t === 'head') return n.isHead ? 'matrix' : 'head';
  if (['op', 'emb', 'up', 'act', 'down', 'bmlp', 'battn', 'dhead'].includes(n.t)) return 'matrix';
  if (['upd', 'apick', 'am', 'av', 'ahat', 'astep', 'abits'].includes(n.t)) return 'weight';
  if (['tok', 'shift', 'probs', 'nll'].includes(n.t)) return 'token';
  return 'box';
}

function traceScene(n, path, tree, d) {
  scenePicks = new Map();
  let nodes, layout = 'flow';
  if (tree.depth === 1) {
    nodes = [{ id: 'current', label: `第 ${n.s} 步`, detail: `训练损失 ${traceNumber(run.log.loss[n.s - 1])}`, kind: 'box', values: sampleLoss(n.s) }];
    scenePicks.set('current', () => tree.into());
    nodes.push({ id: 'loop', label: '一步训练', detail: '取数据 → 前向 → 损失 → 反向 → AdamW', kind: 'box', values: run.log.gnorm.slice(Math.max(0, n.s - 64), n.s) });
    nodes.push({ id: 'model', label: '4 层 Qwen3 架构', detail: '256 维 · 4 头 / 2 组 KV', kind: 'tower', layers: run.model.layers });
    scenePicks.set('loop', () => tree.into());
    scenePicks.set('model', () => tree.seekWhere((x) => x.t === 'fwd'));
  } else if (n.t === 'abits' && d) {
    const a = d.adam;
    nodes = [
      { id: 'before', label: '更新前 · float32', detail: `${a.w.toPrecision(9)} · 32 个实测比特`, kind: 'weight', values: [...a.bitsBefore].map(Number) },
      { id: 'changed', label: '更新改变的位', detail: `${[...a.bitsBefore].filter((b, k) => b !== a.bitsAfter[k]).length} / 32 位改变`, kind: 'weight', values: [...a.bitsBefore].map((b, k) => Number(b !== a.bitsAfter[k])) },
      { id: 'after', label: '更新后 · float32', detail: `${a.w1.toPrecision(9)} · 32 个实测比特`, kind: 'weight', values: [...a.bitsAfter].map(Number) },
    ];
    for (const bit of nodes) scenePicks.set(bit.id, () => { if (!$('#stage').classList.contains('trace-data-open')) $('#stage .trace-data-button')?.click(); });
  } else {
    const anchorDepth = Math.min(tree.depth - 2, path.length - 1);
    const anchor = path[anchorDepth];
    let candidates = tree.list.map((p, i) => ({ p, i })).filter(({ p }) => p[anchorDepth] === anchor);
    if (tree.depth >= 4 && candidates.length > 12) {
      const selected = candidates.findIndex(({ i }) => i === tree.i);
      const start = Math.max(0, Math.min(candidates.length - 12, selected - 5));
      candidates = candidates.slice(start, start + 12);
    }
    nodes = candidates.map(({ p, i }) => {
      const x = p.at(-1), id = `step-${i}`;
      scenePicks.set(id, () => { if (i === tree.i && tree.canInto()) tree.into(); else tree.seekIndex(i); });
      return { id, label: x.label, detail: i === tree.i ? traceDetail(x, d) : '', kind: traceKind(x), values: traceValues(x, d) };
    });
    if (tree.depth === 3 && ['fwd', 'bwd'].includes(path[1]?.t)) layout = 'tower';
    else if (tree.depth === 5 && path[3]?.op === 'attn') layout = 'grid';
  }
  const active = tree.depth === 1 ? 'current' : n.t === 'abits' && d ? 'changed' : `step-${tree.i}`;
  return { title: tree.depth === 1 ? `训练全程 · 第 ${path[0].s} 步` : `第 ${path[0].s} 步 · ${n.label}`, depth: tree.depth, layout, nodes, active };
}

function render(n, path, tree, kind) {
  const s = path[0].s;
  const d = stepData(s);
  sideRender(s);
  traceStage?.update(traceScene(n, path, tree, d));
  if (n.t !== 'ck' && !d) { view().innerHTML = '<div class="panel">正在载入这一步的录像…</div>'; loadStep(s).then(() => tree.emit('step')); return; }
  const f = R[n.t] || R[n.op] || R.ck;
  view().innerHTML = f(n, d, s);
  view().querySelectorAll('canvas[data-attn]').forEach((c) => { const [L, h] = c.dataset.attn.split(',').map(Number); drawAttn(c, d.fwd.attn[L][h], d.T); });
  view().querySelectorAll('.loss-chart .mk').forEach((m) => m.addEventListener('click', () => { const k = run.detail.indexOf(Number(m.dataset.step)); tree.pause(); tree.seekRoot(k); }));
  view().querySelectorAll('[data-pos]').forEach((el) => el.addEventListener('click', () => { ctx.pos = Number(el.dataset.pos); lab.update('step'); }));
}

const stepHead = (d) => `<div class="panel"><h3>第 ${d.step} 步 · 这一批的第 0 首诗 <small>lr ${fmt(d.lr, 5)} · 这一批的损失 ${d.loss.toFixed(3)}</small></h3>${chipRow(d, { sub: (i) => (i === ctx.pos ? '▲' : '') })}<p class="note">点任意一个字，换一个位置细看。第 i 个位置读入这个字，要猜出<b>下一个字</b>。当前位置 ${ctx.pos}：读入“${esc(showTok(chars(d)[ctx.pos]))}”，目标“${esc(showTok(tgt(d, ctx.pos)))}”。</p></div>`;

function flowHTML(cur, bwd = false) {
  const names = [['batch', '取数据'], ['fwd', '前向'], ['loss', '损失'], ['bwd', '反向'], ['clip', '裁剪'], ['upd', 'AdamW']];
  const k = names.findIndex((x) => x[0] === cur);
  return `<div class="flow">${names.map(([t, l], i) => `${i ? '<span class="arr">→</span>' : ''}<div class="node ${t === 'bwd' ? 'bwd' : ''} ${i === k ? 'on' : i < k ? 'done' : ''}">${l}</div>`).join('')}</div>`;
}

const R = {
  ck(n, d, s) {
    const i = run.detail.indexOf(s);
    const loss = run.log.loss[s - 1];
    const vl = run.val.filter((v) => v[0] <= s).pop();
    const best = run.val.reduce((a, b) => (b[1] < a[1] ? b : a));
    const over = s > best[0] + 200 ? `<div class="panel note">⚠ <b>过拟合</b>：验证损失在第 ${best[0]} 步降到最低（${best[1].toFixed(3)}）之后开始回升，现在是 ${vl[1].toFixed(3)}；训练损失却还在下降。1.7 万首诗对 435 万个参数来说不算多，模型开始<b>背</b>训练集里的诗，而不是学会一般的写法。真实的大模型训练会用多得多的数据，或者在验证损失最低的地方停下。</div>` : '';
    return `<div class="panel"><h3>训练全程 <small>第 ${s} / ${run.train.steps} 步 · 损失 ${loss.toFixed(3)} · 验证 ${vl[1].toFixed(3)} · 困惑度 ${Math.exp(vl[1]).toFixed(0)}</small></h3>${lossChart(s)}</div>
    <div class="row">
      <div class="panel"><h3>此刻它写的诗 <small>第 ${s} 步更新之后</small></h3><div class="poems">${samplesHTML(s)}</div></div>
      <div class="panel"><h3>字向量 <small>第 ${s} 步</small></h3>${pcaHTML(s)}</div>
    </div>
    <div class="panel"><h3>它能背出《静夜思》吗</h3>${probeHTML(s)}</div>${over}
    ${i === 0 ? '<div class="panel note">第 1 步：权重刚刚随机初始化，词表有 4699 个字，每个字的概率都差不多是 1/4699，所以损失 ≈ ln 4699 ≈ 8.45，写出来的是随机的字。按 <b>▶</b> 看它一路学下去，或按 <b>＋</b> 钻进这一步训练。</div>' : ''}`;
  },

  batch(n, d) {
    return `<div class="panel">${flowHTML('batch')}</div>${stepHead(d)}
    <div class="panel"><h3>同一批里的另外几首 <small>一批共 ${d.batch} 首 · ${d.tokensInBatch} 个要预测的字</small></h3><div class="poems small">${d.others.map((t) => `<div class="poem">${esc(t)}</div>`).join('')}</div>
    <p class="note">训练集 ${run.data.poems.toLocaleString()} 首五言诗，每一步随机抽 64 首。所有诗的损失加起来取平均，一起算梯度。</p></div>`;
  },
  tok(n, d) {
    return `${stepHead(d)}<div class="panel"><h3>按字切分 <small>每个字查词表得到编号</small></h3>
    <div class="chips">${d.tokens.map((c, i) => `<span class="chip ${c.length > 1 ? 'sp' : ''}">${esc(showTok(c))}<small>${d.ids[i]}</small></span>`).join('')}</div>
    <p class="note">这个小模型的词表就是语料里出现过至少两次的 ${run.model.vocab - 4} 个汉字和标点，外加 4 个特殊记号（⟨s⟩ 开头、⟨/s⟩ 结尾）。推理页的 Qwen3-0.6B 用的是 15 万个 BPE 词元，道理一样。</p></div>`;
  },
  shift(n, d) {
    const inp = d.tokens.slice(0, d.T), out = d.tokens.slice(1, d.T + 1);
    return `${stepHead(d)}<div class="panel"><h3>输入 x 与目标 y <small>错开一位</small></h3>
    <div class="shift"><div><span class="lab">x</span>${inp.map((c, i) => `<span class="chip ${i === ctx.pos ? 'cur' : ''}">${esc(showTok(c))}</span>`).join('')}</div>
    <div><span class="lab">y</span>${out.map((c, i) => `<span class="chip ${i === ctx.pos ? 'cur' : ''}">${esc(showTok(c))}</span>`).join('')}</div></div>
    <p class="note">一首 ${d.T + 1} 个记号的诗，一次就能出 ${d.T} 道题：看到前 i 个字，猜第 i+1 个。因果掩码保证每个位置只能看到自己和前面的字。</p></div>`;
  },

  fwd(n, d) {
    return `<div class="panel">${flowHTML('fwd')}</div>${stepHead(d)}${towerHTML(d, -1)}`;
  },
  emb(n, d) {
    return `${stepHead(d)}${towerHTML(d, 0)}<div class="panel"><h3>查嵌入表 <small>embed [${run.model.vocab} × 256]</small></h3><p class="note">每个字取出嵌入表里自己那一行（256 个数），排成 [${d.T} × 256] 的矩阵，这就是残差流的起点。嵌入表同时也是最后的输出头（权重共享），所以它会从两头都收到梯度。</p></div>`;
  },
  layer(n, d) { return `${stepHead(d)}${towerHTML(d, n.L + 1)}${layerOpsHTML(n, d)}`; },
  op(n, d) { return `${stepHead(d)}${layerOpsHTML(n, d)}${opDetail(n, d)}`; },
  head(n, d) {
    if (n.isHead) {
      const i = ctx.pos, top = d.out.top[i];
      return `${stepHead(d)}${towerHTML(d, 5)}<div class="panel"><h3>输出头 <small>rmsnorm(h) @ embed.T → ${run.model.vocab} 个分数 → softmax</small></h3>
      <p class="note">位置 ${i} 读到“${esc(showTok(chars(d)[i]))}”，最想写的 5 个字（目标是“${esc(showTok(tgt(d, i)))}”）：</p>${barsHTML(top, tgt(d, i))}</div>`;
    }
    return `${stepHead(d)}${layerOpsHTML({ ...n, op: 'attn' }, d)}${headDetail(n, d)}`;
  },
  up(n, d) { return `${stepHead(d)}${layerOpsHTML({ ...n, op: 'mlp' }, d)}${mlpDetail(n, d, 'up')}`; },
  act(n, d) { return `${stepHead(d)}${layerOpsHTML({ ...n, op: 'mlp' }, d)}${mlpDetail(n, d, 'act')}`; },
  down(n, d) { return `${stepHead(d)}${layerOpsHTML({ ...n, op: 'mlp' }, d)}${mlpDetail(n, d, 'down')}`; },

  loss(n, d) { return `<div class="panel">${flowHTML('loss')}</div>${stepHead(d)}${lossPosHTML(d, 'nll')}`; },
  probs(n, d) {
    const i = ctx.pos;
    return `${stepHead(d)}<div class="panel"><h3>softmax 概率 <small>位置 ${i}</small></h3><p class="note">目标“${esc(showTok(tgt(d, i)))}”的概率是 <b class="amber">${fmtP(d.out.pTarget[i])}</b>；前 5 名：</p>${barsHTML(d.out.top[i], tgt(d, i))}</div>${lossPosHTML(d, 'p')}`;
  },
  nll(n, d) { return `${stepHead(d)}${lossPosHTML(d, 'nll')}`; },
  mean(n, d) {
    const m = d.out.lossPos.reduce((a, b) => a + b, 0) / d.T;
    return `${stepHead(d)}${lossPosHTML(d, 'nll')}<div class="panel"><h3>取平均</h3><div class="formula">这首诗：mean(−log p) = <span class="val">${m.toFixed(3)}</span>　·　整批 64 首、${d.tokensInBatch} 个字：loss = <span class="on">${d.loss.toFixed(4)}</span></div><p class="note">这一个数就是要最小化的目标。接下来反向传播会算出：每个权重往哪个方向动一点，能让它变小。</p></div>`;
  },

  bwd(n, d) { return `<div class="panel">${flowHTML('bwd')}</div>${stepHead(d)}${gradGridHTML(d, -1)}`; },
  dhead(n, d) {
    const i = ctx.pos, p = d.out.pTarget[i];
    return `${stepHead(d)}${gradGridHTML(d, 5)}<div class="panel"><h3>输出头的梯度 <small>∂loss/∂logits = p − onehot(y)</small></h3>
    <p class="note">softmax 加交叉熵的梯度特别简单：每个字的概率减去“正确答案是 1、其他是 0”。位置 ${i}：目标“${esc(showTok(tgt(d, i)))}”的分数被往上推（p − 1 = ${(p - 1).toFixed(3)}），其余 ${run.model.vocab - 1} 个字按各自的概率往下压。再除以这一批的 ${d.tokensInBatch} 个字。</p></div>`;
  },
  blayer(n, d) { return `${stepHead(d)}${gradGridHTML(d, n.L + 1)}${paramsHTML(d, n.L)}`; },
  bmlp(n, d) { return `${stepHead(d)}${gradGridHTML(d, n.L + 1, 'mlp')}${paramsHTML(d, n.L, 'mlp')}`; },
  battn(n, d) { return `${stepHead(d)}${gradGridHTML(d, n.L + 1, 'attn')}${paramsHTML(d, n.L, 'self_attn')}`; },
  demb(n, d) { return `${stepHead(d)}${gradGridHTML(d, 0)}${paramsHTML(d, -1)}`; },

  clip(n, d) {
    const pct = Math.min(100, (d.gradNorm / Math.max(d.gradNorm, 1.5)) * 100), thr = Math.min(100, (1 / Math.max(d.gradNorm, 1.5)) * 100);
    const hist = run.log.gnorm;
    const w = 640, h = 80, mx = Math.max(...hist);
    const pth = linePath(hist.map((_, i) => i + 1), hist, 0, hist.length, 0, mx, w, h);
    return `<div class="panel">${flowHTML('clip')}</div><div class="panel"><h3>梯度裁剪 <small>所有 ${d.params.length} 个参数矩阵的梯度拼成一个向量，看它的长度</small></h3>
    <div class="gauge"><i style="width:${pct}%" class="${d.gradNorm > 1 ? 'over' : ''}"></i><b style="left:${thr}%"></b></div>
    <div class="formula">‖g‖ = <span class="val">${d.gradNorm.toFixed(3)}</span>　${d.gradNorm > 1 ? `&gt; 1.0 → 所有梯度乘以 <span class="on">${d.clip.toFixed(3)}</span>` : '≤ 1.0 → 不用裁剪'}</div>
    <p class="note">梯度太长时整体缩短到 1.0，方向不变。它防止某一批的异常数据把权重一下推得太远。</p>
    <svg class="fit" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><path d="${pth}" class="gn"/><line x1="0" x2="${w}" y1="${h - h / mx}" y2="${h - h / mx}" class="thr"/><line x1="${(d.step / hist.length) * w}" x2="${(d.step / hist.length) * w}" y1="0" y2="${h}" class="cur"/></svg>
    <div class="legend"><span><i style="background:var(--violet)"></i>每一步裁剪前的梯度范数</span><span><i style="background:var(--rose)"></i>阈值 1.0</span></div></div>`;
  },

  upd(n, d) { return `<div class="panel">${flowHTML('upd')}</div>${adamHTML(d, 'all')}${updTable(d)}`; },
  apick(n, d) { return adamHTML(d, 'pick'); },
  am(n, d) { return adamHTML(d, 'm'); },
  av(n, d) { return adamHTML(d, 'v'); },
  ahat(n, d) { return adamHTML(d, 'hat'); },
  astep(n, d) { return adamHTML(d, 'step'); },
  abits(n, d) { return adamHTML(d, 'bits'); },
};

function towerHTML(d, cur) {
  // 行：嵌入、第 0..3 层、输出；格子：残差范数（颜色）+ 逻辑透镜读出的字
  const rows = [0, 1, 2, 3, 4, 5];
  let mx = 0;
  for (const r of d.fwd.res) for (const v of r) mx = Math.max(mx, v);
  return `<div class="panel"><h3>前向：残差流逐层 <small>颜色 = 这个位置向量的长度；字 = 在这一层就接上输出头，模型会猜什么（逻辑透镜）</small></h3>${gridHTML(d, rows, {
    cur,
    label: (k) => (k === 0 ? '嵌入' : k === 5 ? '输出' : `L${k - 1}`),
    cell: (k, i) => {
      if (k === 0) { const v = d.fwd.res[0][i]; return `<span class="gc" style="background:${heat(v, mx)}"></span>`; }
      if (k === 5) { const t = d.out.top[i][0]; const ok = t[0] === tgt(d, i); return `<span class="gc lens ${ok ? 'ok' : ''} ${i === ctx.pos ? 'on' : ''}" title="${esc(t[0])} ${fmtP(t[1])}">${esc(showTok(t[0]))}</span>`; }
      const v = d.fwd.res[k][i], l = d.fwd.lens[k - 1][i], ok = l[0] === tgt(d, i);
      return `<span class="gc lens ${ok ? 'ok' : ''} ${i === ctx.pos ? 'on' : ''}" style="background:${heat(v, mx)}" title="第 ${k - 1} 层后 · 长度 ${v.toFixed(2)} · 透镜：${esc(l[0])} ${fmtP(l[1])}">${esc(showTok(l[0]))}</span>`;
    },
  })}<p class="note">绿字 = 已经猜对了下一个字。看看是哪一层开始猜对的。</p></div>`;
}

function layerOpsHTML(n, d) {
  const L = n.L, i = ctx.pos;
  const vals = { ln1: '', attn: `‖a‖ ${d.fwd.attnOut[L][i].toFixed(2)}`, add1: `‖h‖ ${d.fwd.mid[L][i].toFixed(2)}`, ln2: '', mlp: `‖m‖ ${d.fwd.mlpOut[L][i].toFixed(2)}`, add2: `‖h‖ ${d.fwd.res[L + 1][i].toFixed(2)}` };
  const k = OPS.findIndex((o) => o[0] === n.op);
  return `<div class="panel"><h3>第 ${L} 层之内 <small>位置 ${i} 输入 ‖h‖ = ${d.fwd.res[L][i].toFixed(2)}</small></h3><div class="flow">${OPS.map(([op, lab], j) => `${j ? '<span class="arr">→</span>' : ''}<div class="node ${j === k ? 'on' : j < k ? 'done' : ''}">${lab}<small>${vals[op]}</small></div>`).join('')}</div></div>`;
}

function opDetail(n, d) {
  const L = n.L, i = ctx.pos;
  switch (n.op) {
    case 'ln1': case 'ln2':
      return `<div class="panel"><h3>RMSNorm</h3><div class="formula">n = h / √(mean(h²) + 1e-6) · γ</div><p class="note">把每个位置的向量缩放到差不多一样长，再乘一组学出来的系数 γ。它让深层网络好训练得多。</p></div>`;
    case 'attn':
      return `<div class="panel"><h3>注意力 <small>4 个查询头 · 每 2 个共用一组 K/V · 每头 64 维</small></h3><div class="heads4">${[0, 1, 2, 3].map((h) => `<div><canvas data-attn="${L},${h}"></canvas><span>头 ${h}</span></div>`).join('')}</div><p class="note">每张图是一个头的注意力矩阵：第 i 行 = 位置 i 把注意力分给前面每个位置的比例（越亮越多），右上角被因果掩码挡住。琥珀框是当前位置。按 ＋ 逐个头看。</p></div>`;
    case 'add1': case 'add2': {
      const a = n.op === 'add1' ? d.fwd.attnOut[L][i] : d.fwd.mlpOut[L][i];
      const before = n.op === 'add1' ? d.fwd.res[L][i] : d.fwd.mid[L][i];
      const after = n.op === 'add1' ? d.fwd.mid[L][i] : d.fwd.res[L + 1][i];
      return `<div class="panel"><h3>残差相加</h3><div class="formula">h ← h + ${n.op === 'add1' ? 'a' : 'm'}　·　‖h‖ ${before.toFixed(2)} + ‖${n.op === 'add1' ? 'a' : 'm'}‖ ${a.toFixed(2)} → <span class="val">${after.toFixed(2)}</span></div><p class="note">子层不替换残差流，只往上面“加一笔”。反向传播时这条直通的加法让梯度可以不衰减地流回前面的层。</p></div>`;
    }
    case 'mlp':
      return mlpDetail(n, d, 'all');
    default: return '';
  }
}

function headDetail(n, d) {
  const L = n.L, h = n.h, i = ctx.pos;
  const row = d.fwd.attn[L][h][i];
  const pairs = row.map((v, j) => [j, v / 255]).sort((a, b) => b[1] - a[1]).slice(0, 6);
  return `<div class="panel"><h3>第 ${L} 层 · 头 ${h} <small>K/V 组 ${h >> 1}</small></h3><div class="row"><div><canvas data-attn="${L},${h}" class="big"></canvas></div>
  <div style="flex:1;min-width:200px"><p class="note">位置 ${i}（“${esc(showTok(chars(d)[i]))}”）这一行看得最多的是：</p>${barsHTML(pairs.map(([j, w]) => [`${showTok(chars(d)[j])}·${j}`, w]), null)}</div></div></div>`;
}

function mlpDetail(n, d, part) {
  const L = n.L, i = ctx.pos;
  const top = d.fwd.mlp[L][i];
  const mx = Math.max(...top.map((t) => Math.abs(t[1])));
  const cells = top.map(([j, v]) => `<div class="neu" style="background:${heat(v, mx)}"><b>#${j}</b><span>${v.toFixed(3)}</span></div>`).join('');
  const txt = {
    up: `n [1×256] 分别乘 W_gate、W_up [256×768]，得到 768 个门控值 g 和 768 个候选值 u。`,
    act: `每个神经元：SiLU(g)·u。SiLU(g) = g·σ(g) 像一个平滑的阀门：g 很负时几乎关死，g 为正时放行。下面是位置 ${i} 这一层里最活跃的 8 个神经元（共 768 个）。`,
    down: `768 个激活值乘 W_down [768×256] 压回 256 维，得到 m，加回残差流。本层 ‖m‖ = ${d.fwd.mlpOut[L][i].toFixed(2)}。`,
    all: `SwiGLU：升维到 768 → 门控 → 降维回 256。下面是位置 ${i} 最活跃的 8 个神经元（按 ＋ 分三步看）。`,
  }[part];
  return `<div class="panel"><h3>SwiGLU 前馈 <small>第 ${L} 层 · 位置 ${i}</small></h3><div class="formula">m = W_down( SiLU(W_gate·n) ⊙ (W_up·n) )</div><p class="note">${txt}</p><div class="neus">${cells}</div></div>`;
}

function lossPosHTML(d, mode) {
  const vals = mode === 'p' ? d.out.pTarget : d.out.lossPos;
  const mx = Math.max(...vals);
  return `<div class="panel"><h3>${mode === 'p' ? '每个位置猜中目标的概率' : '每个位置的损失 −log p(目标)'} <small>这首诗平均 ${(d.out.lossPos.reduce((a, b) => a + b, 0) / d.T).toFixed(3)}</small></h3>
  <div class="lossbars">${vals.map((v, i) => `<div class="lb ${i === ctx.pos ? 'on' : ''}" data-pos="${i}" title="位置 ${i} → “${esc(tgt(d, i))}”：p=${fmtP(d.out.pTarget[i])}，损失 ${d.out.lossPos[i].toFixed(2)}"><i style="height:${((v / mx) * 100).toFixed(1)}%"></i><span>${esc(showTok(tgt(d, i)))}</span></div>`).join('')}</div>
  <p class="note">横轴是要猜的目标字。${mode === 'p' ? '' : '猜得越没把握，损失越大：p=1 时损失为 0；p=1/4699（瞎猜）时损失 8.45。'}标点和固定格式通常最先学会。</p></div>`;
}

function gradGridHTML(d, cur, part) {
  const rows = [0, 1, 2, 3, 4, 5];
  const all = d.bwd.res.flat();
  const lmx = Math.log10(Math.max(...all)), lmn = Math.log10(Math.max(1e-12, Math.min(...all.filter((v) => v > 0))));
  const col = (v) => { const t = (Math.log10(Math.max(v, 1e-12)) - lmn) / (lmx - lmn || 1); return `rgba(255,107,147,${(0.06 + 0.9 * t).toFixed(2)})`; };
  return `<div class="panel"><h3>反向：梯度逐层往回流 <small>颜色 = ‖∂loss/∂h‖（对数刻度）· 当前位置 ${ctx.pos}</small></h3>${gridHTML(d, rows, {
    cur,
    label: (k) => (k === 0 ? '嵌入' : k === 5 ? '输出' : `L${k - 1}`),
    cell: (k, i) => {
      if (k === 5) { const p = d.out.pTarget[i]; return `<span class="gc" style="background:rgba(255,107,147,${(0.1 + 0.8 * (1 - p)).toFixed(2)})" title="1 − p = ${(1 - p).toFixed(3)}"></span>`; }
      const v = d.bwd.res[k][i];
      return `<span class="gc ${i === ctx.pos ? 'on' : ''}" style="background:${col(v)}" title="${v.toExponential(2)}"></span>`;
    },
  })}${part ? `<div class="formula">第 ${cur - 1} 层 · 位置 ${ctx.pos}：‖∂loss/∂${part === 'mlp' ? 'm' : 'a'}‖ = <span class="val">${(part === 'mlp' ? d.bwd.mlpOut : d.bwd.attnOut)[cur - 1][ctx.pos].toExponential(3)}</span></div>` : ''}
  <p class="note">梯度从输出（最下面）往上流回嵌入。因为每层都是 h + f(h)，梯度可以沿着“+”直接穿过去，所以前面的层也能收到差不多大小的信号。越往前的位置影响越多的后续预测，梯度通常也越大。</p></div>`;
}

function paramsHTML(d, L, filter) {
  const pre = L < 0 ? 'embed_tokens' : `layers.${L}.`;
  const list = d.params.filter((p) => p.name.startsWith(pre) && (!filter || p.name.includes(filter)));
  const mx = Math.max(...list.map((p) => p.g));
  return `<div class="panel"><h3>${L < 0 ? '嵌入表' : `第 ${L} 层`}每个参数矩阵的梯度 <small>‖g‖ · 更新量 ‖Δw‖ / ‖w‖</small></h3><div class="ptab">${list.map((p) => `<div class="pr"><span class="pn">${esc(p.name.replace(pre, '').replace('.weight', ''))}</span><span class="ps">[${p.shape.join('×')}]</span><span class="pb"><i style="width:${((p.g / mx) * 100).toFixed(1)}%"></i></span><span class="pv">${p.g.toExponential(2)}</span><span class="pv">${((p.upd / p.w) * 100).toFixed(3)}%</span></div>`).join('')}</div>
  <p class="note">表中的百分比是这一步更新向量的长度与原权重长度之比，各矩阵数值不同。AdamW 按每个权重自己的历史梯度大小来缩放步长，所以梯度小的矩阵也能正常学习。</p></div>`;
}

function updTable(d) {
  const list = d.params.filter((p) => p.shape.length === 2);
  return `<div class="panel"><h3>所有矩阵这一步动了多少 <small>‖Δw‖ / ‖w‖</small></h3><div class="ptab">${list.map((p) => `<div class="pr"><span class="pn">${esc(p.name.replace('.weight', ''))}</span><span class="ps">[${p.shape.join('×')}]</span><span class="pb"><i style="width:${Math.min(100, (p.upd / p.w) * 100 * 30).toFixed(1)}%"></i></span><span class="pv">${((p.upd / p.w) * 100).toFixed(3)}%</span></div>`).join('')}</div></div>`;
}

function bitsHTML(b, other) {
  const out = [];
  for (let k = 0; k < 32; k++) {
    const cls = k === 0 ? 's' : k <= 8 ? 'e' : 'm';
    out.push(`<span class="${cls} ${b[k] === '1' ? 'one' : ''} ${other && other[k] !== b[k] ? 'diff' : ''}">${b[k]}</span>`);
    if (k === 0 || k === 8) out.push('<span class="gap"></span>');
  }
  return `<div class="bits">${out.join('')}</div>`;
}

// 这个权重在 31 个录制点上的值（每个录制点都记下了更新前后的值）
function trajHTML(d) {
  const pts = run.detail.map((s) => stepData(s)).filter(Boolean).map((x) => [x.step, x.adam.w1]);
  if (pts.length < 3) return '';
  const w = 640, h = 90;
  const ys = pts.map((p) => p[1]), lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo) * 0.15 || 1e-3;
  const path = linePath(pts.map((p) => p[0]), ys, 0, run.train.steps, lo - pad, hi + pad, w, h);
  const cx = (d.step / run.train.steps) * w, cy = h - ((d.adam.w1 - lo + pad) / (hi - lo + 2 * pad)) * h;
  return `<div class="panel"><h3>embed[${d.adam.row}, ${d.adam.col}] 在整个训练里的轨迹 <small>${lo.toFixed(4)} ~ ${hi.toFixed(4)}</small></h3>
  <svg class="fit traj" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><path d="${path}"/><circle cx="${cx}" cy="${cy}" r="4"/></svg>
  <p class="note">一开始是随机初始化的小数；学习率大的前期挪得快，后期学习率衰减、方向也稳定下来，就慢慢停在一个位置。</p></div>`;
}

function adamHTML(d, part) {
  const a = d.adam;
  const b1t = 1 - a.beta1 ** a.t, b2t = 1 - a.beta2 ** a.t;
  const mh = a.m1 / b1t, vh = a.v1 / b2t;
  const stepv = (a.lr * mh) / (Math.sqrt(vh) + a.eps), decay = a.w * a.lr * a.wd;
  const pred = a.w * (1 - a.lr * a.wd) - stepv;
  const on = (k) => (part === k || part === 'all' ? 'on' : '');
  const e = (v) => fmt(v, 6);
  let extra = '';
  if (part === 'pick' || part === 'all') {
    extra = `<div class="panel"><h3>盯住一个权重 <small>“${esc(a.char)}”字嵌入向量的第 ${a.col} 维（embed[${a.row}, ${a.col}]）</small></h3>
    <div class="emrow">${a.row8.map((v, k) => `<span style="background:${heat(v, 0.3)}">${v.toFixed(3)}</span>`).join('')}<span class="more">… 共 256 维</span></div>
    <p class="note">我们从第 1 步就一直盯着同一个数。因为输出头和嵌入表共享权重，每一步不管这批诗里有没有“${esc(a.char)}”，它都会收到梯度：模型需要把“${esc(a.char)}”的分数往上或往下调。这一步的原始梯度 g = ${e(a.gRaw)}，裁剪后 ${e(a.g)}。</p></div>`;
  }
  if (part === 'bits') {
    extra = `<div class="panel"><h3>它的 32 个比特 <small>float32 · 1 位符号 · 8 位指数 · 23 位尾数</small></h3>
    <div class="note">更新前 ${a.w.toPrecision(9)}</div>${bitsHTML(a.bitsBefore, a.bitsAfter)}
    <div class="note">更新后 ${a.w1.toPrecision(9)}</div>${bitsHTML(a.bitsAfter, a.bitsBefore)}
    <p class="note">琥珀框是这一步翻转了的比特；比较更新前后，能看到 float32 如何存储这次变化。学习就是数百万个权重每一步各自挪动一点。（推理页的 Qwen3 存成 bfloat16，只有 7 位尾数；训练时通常用 float32 或混合精度，避免小更新被舍入掉。）</p></div>`;
  }
  return `${extra}${trajHTML(d)}<div class="panel"><h3>AdamW 更新这一个权重 <small>第 t = ${a.t} 步 · lr = ${fmt(a.lr, 6)}</small></h3>
  <div class="formula"><div class="${on('m')}">m = 0.9 × <span class="val">${e(a.m0)}</span> + 0.1 × <span class="val">${e(a.g)}</span> = <span class="val">${e(a.m1)}</span></div>
  <div class="${on('v')}">v = 0.95 × <span class="val">${fmt(a.v0, 4)}</span> + 0.05 × <span class="val">${e(a.g)}</span>² = <span class="val">${fmt(a.v1, 4)}</span></div>
  <div class="${on('hat')}">m̂ = m / (1 − 0.9<sup>${a.t}</sup>) = <span class="val">${e(mh)}</span>　v̂ = v / (1 − 0.95<sup>${a.t}</sup>) = <span class="val">${fmt(vh, 4)}</span></div>
  <div class="${on('step')}">w = <span class="val">${e(a.w)}</span> − ${fmt(a.lr, 6)}×0.1×w − ${fmt(a.lr, 6)} × m̂/(√v̂+1e-8) = <span class="val">${e(a.w)}</span> − <span class="val">${e(decay)}</span> − <span class="val">${e(stepv)}</span> = <span class="on">${e(pred)}</span></div></div>
  <p class="note">${{
    m: 'm 是梯度的滑动平均（动量）：最近的梯度占 10%，历史占 90%。它让更新方向不随某一批数据乱跳。',
    v: 'v 是梯度平方的滑动平均，衡量这个权重的梯度通常有多大。',
    hat: '刚开始 m、v 都从 0 起步，会偏小；除以 (1 − β^t) 把这个偏差修正回来。t 越大修正越接近 1。',
    step: `m̂/√v̂ 大约是“梯度方向 ÷ 梯度的典型大小”，所以每个权重每步移动的量级都接近学习率。再加上权重衰减（每步把权重往 0 拉一点点）。PyTorch 实际算出的新值：<b class="cyanc">${e(a.w1)}</b>，和上面手算的一致。`,
    pick: '', bits: '', all: '按 ＋ 逐行看这四行公式。',
  }[part]}</p></div>`;
}

// ---------------------------------------------------------------- 左侧

let sideStep = -1;
function sideRender(s) {
  if (s === sideStep) { $('#sideBody').querySelectorAll('.ckl').forEach((el) => el.classList.toggle('on', Number(el.dataset.step) === s)); return; }
  sideStep = s;
  const list = run.detail.map((st) => {
    const smp = run.samples[String(st)]?.[4]?.text || '';
    return `<button type="button" class="ckl ${st === s ? 'on' : ''}" data-step="${st}"><span class="st">第 ${st} 步</span><span class="ls">${run.log.loss[st - 1].toFixed(2)}</span><span class="sm">${esc(smp.slice(0, 12))}</span></button>`;
  }).join('');
  $('#sideBody').innerHTML = `<div><h4>录制点 · 点一下跳过去</h4><div class="cklist">${list}</div></div>
  <div class="blk"><b>模型</b>：Qwen3ForCausalLM（transformers）· ${run.model.layers} 层 · 隐藏 ${run.model.hidden} · ${run.model.heads} 头 / ${run.model.kvHeads} 组 KV · SwiGLU ${run.model.ffn} · 词表 ${run.model.vocab} · 共 ${(run.model.params / 1e6).toFixed(2)}M 参数。<br><b>训练</b>：AdamW（β=${run.train.betas.join(', ')}，权重衰减 ${run.train.wd}）· 峰值学习率 ${run.train.lr} · 预热 ${run.train.warmup} 步后余弦衰减 · 每批 ${run.train.batch} 首 · 梯度裁剪 ${run.train.clip} · 单卡 RTX 5060 用时 ${Math.round(run.elapsed)} 秒。</div>`;
  $('#sideBody').querySelectorAll('.ckl').forEach((b) => b.addEventListener('click', () => { lab.tree.pause(); lab.tree.seekRoot(run.detail.indexOf(Number(b.dataset.step)), lab.tree.depth); document.body.classList.remove('side-open'); $('#btnSideOpen').setAttribute('aria-expanded', 'false'); }));
  $('#sideBody').querySelector('.ckl.on')?.scrollIntoView({ block: 'nearest' });
}

// ---------------------------------------------------------------- 讲解与变量

function explain(n, path) {
  const s = path[0].s, d = stepData(s), i = ctx.pos;
  const lines = LINES[n.op] || LINES[n.isHead ? 'isHead' : n.t] || [1];
  const w = [];
  if (d) {
    w.push('— 这一步');
    w.push(['step', `${d.step}`], ['lr', fmt(d.lr, 6), true], ['loss', d.loss.toFixed(4)], ['‖g‖', d.gradNorm.toFixed(3)]);
    w.push(`— 位置 ${i}`);
    w.push(['x[i]', showTok(chars(d)[i])], ['y[i]', showTok(tgt(d, i))], ['p(y)', fmtP(d.out.pTarget[i])], ['−log p', d.out.lossPos[i].toFixed(3)]);
    if (n.L != null && n.t !== 'blayer' && n.t !== 'bmlp' && n.t !== 'battn') w.push(`— 第 ${n.L} 层`, ['‖h 入‖', d.fwd.res[n.L][i].toFixed(3)], ['‖attn‖', d.fwd.attnOut[n.L][i].toFixed(3)], ['‖mlp‖', d.fwd.mlpOut[n.L][i].toFixed(3)], ['透镜', `${d.fwd.lens[n.L][i][0]} ${fmtP(d.fwd.lens[n.L][i][1])}`]);
    if (n.L != null && (n.t === 'blayer' || n.t === 'bmlp' || n.t === 'battn')) w.push(`— 第 ${n.L} 层（反向）`, ['‖∂h‖', d.bwd.res[n.L + 1][i].toExponential(2)], ['‖∂attn‖', d.bwd.attnOut[n.L][i].toExponential(2)], ['‖∂mlp‖', d.bwd.mlpOut[n.L][i].toExponential(2)]);
    if (path.some((x) => x.t === 'upd')) {
      const a = d.adam;
      w.push(`— embed[${a.row},${a.col}]（“${a.char}”）`, ['w', a.w.toPrecision(7)], ['g', a.g.toExponential(3)], ['m', a.m1.toExponential(3)], ['v', a.v1.toExponential(3)], ['w′', a.w1.toPrecision(7)]);
    }
  } else {
    const vl = run.val.filter((v) => v[0] <= s).pop();
    w.push('— 训练进度', ['step', `${s} / ${run.train.steps}`], ['loss', run.log.loss[s - 1].toFixed(4)], ['val', vl[1].toFixed(4)], ['困惑度', Math.exp(vl[1]).toFixed(1)], ['lr', fmt(run.log.lr[s - 1], 6), true], ['‖g‖', run.log.gnorm[s - 1].toFixed(3)]);
  }
  const shape = {
    emb: `x [${d?.T}] → embed [${run.model.vocab}×256] → h [${d?.T}×256]`,
    layer: `h [T×256] → 注意力 → + → SwiGLU → + → h [T×256]`,
    attn: 'n [T×256] @ W_q [256×256] → 4 头×64 · W_k/W_v [256×128] → 2 组×64',
    head: `h [T×256] @ embed.T [256×${run.model.vocab}] → logits [T×${run.model.vocab}]`,
    mlp: 'n [T×256] → W_gate, W_up [256×768] → ⊙ → W_down [768×256]',
    up: 'n [1×256] @ W_gate, W_up [256×768] → g, u [1×768]', act: 'SiLU(g) ⊙ u → [1×768]', down: '[1×768] @ W_down [768×256] → m [1×256]',
    ahead: 'q [1×64] · K [T×64]ᵀ / √64 → softmax → 权重 [1×T] @ V [T×64]',
  }[n.isHead ? 'head' : n.t === 'head' ? 'ahead' : n.op || n.t];
  return { lines, shape: shape ? esc(shape) : '', html: EXPLAIN[n.op || n.t]?.(n, d) || '', watch: w };
}

const EXPLAIN = {
  ck: () => '这是训练中的一个录制点。每一步：抽 64 首诗 → 前向算出每个字的预测 → 和真实的下一个字比，得到损失 → 反向传播求出每个权重的梯度 → AdamW 更新 435 万个权重。按 <b>＋</b> 钻进这一步。',
  batch: () => '从训练集随机抽 64 首诗组成一批。下面只追踪其中第 0 首的内部，数值都是这一步真实计算出来的。',
  tok: () => '模型只认数字：每个字换成它在词表里的编号。',
  shift: () => '语言模型的训练目标：<b>预测下一个字</b>。把同一首诗错开一位，就得到输入和答案。',
  fwd: () => '前向传播：和推理完全一样的计算，只是一次处理整首诗的所有位置。',
  emb: () => '嵌入：编号 → 向量。这些向量一开始是随机数，训练会把它们慢慢调成有意义的样子。',
  layer: (n) => `第 ${n.L} 层：先注意力（位置之间交换信息），再前馈（每个位置自己加工）。每个子层的结果都<b>加</b>回残差流。`,
  ln1: () => '进注意力之前先做 RMSNorm。', ln2: () => '进前馈之前再做一次 RMSNorm。',
  attn: () => '注意力：每个位置决定从前面哪些位置取信息。4 个头各看各的。',
  add1: () => '残差：把注意力的输出加回 h。', add2: () => '残差：把前馈的输出加回 h。',
  mlp: () => 'SwiGLU 前馈：768 个神经元，每个都是“门控 × 候选值”。',
  head: (n) => (n.isHead ? '最后一次 RMSNorm，然后乘嵌入表的转置，得到词表里每个字的分数。' : `第 ${n.L} 层头 ${n.h} 的注意力矩阵。`),
  up: () => '升维：两个矩阵乘法，一个算门控，一个算候选值。', act: () => '激活：SiLU(门控) × 候选值。', down: () => '降维：768 → 256，加回残差流。',
  loss: () => '交叉熵损失：对每个位置，取正确答案概率的负对数，再取平均。',
  probs: () => 'softmax 把分数变成概率（全部加起来 = 1）。', nll: () => '每个位置的损失 = −log p(正确的字)。', mean: () => '整批平均，得到这一步要最小化的那个数。',
  bwd: () => '反向传播：从损失出发，按链式法则把“损失对每个中间量的导数”一层层往回传，最后得到对每个权重的梯度。',
  dhead: () => 'softmax + 交叉熵的梯度：p − onehot(y)。', blayer: (n) => `梯度穿过第 ${n.L} 层：先经过前馈，再经过注意力（和前向的顺序相反）。`,
  bmlp: () => '前馈子层的梯度：W_down、W_up、W_gate 各自得到一个和自己同形状的梯度矩阵。', battn: () => '注意力子层的梯度：W_q、W_k、W_v、W_o 各自的梯度。',
  demb: () => '嵌入表的梯度：来自两头——输入端被查到的那几行，和作为输出头时的每一行。',
  clip: () => '把所有梯度看成一个超长向量；太长就整体缩短到 1.0。',
  upd: () => 'AdamW：对 435 万个权重中的每一个，独立地执行这四行。下面盯住其中一个。',
  apick: () => '选一个具体的权重，看它这一步是怎么被更新的。', am: () => '一阶矩：梯度的滑动平均。', av: () => '二阶矩：梯度平方的滑动平均。',
  ahat: () => '偏差修正。', astep: () => '真正改动权重。', abits: () => '权重在内存里就是 32 个 0 和 1。',
};

// ---------------------------------------------------------------- 启动

let lab;

async function boot() {
  try {
    run = await fetchJSON('data/run.json');
  } catch (e) {
    $('#lead').textContent = `训练记录载入失败：${e.message}`;
    return;
  }
  const last = run.detail[run.detail.length - 1];
  const v0 = run.val[0][1], vN = run.val[run.val.length - 1][1];
  $('#lead').innerHTML = `下面是一次<b>真实的训练</b>：用 transformers 的 <b>Qwen3ForCausalLM</b>（和推理页 Qwen3-0.6B 同一套结构，缩小到 4 层 · 256 维 · ${(run.model.params / 1e6).toFixed(1)}M 参数），从随机初始化开始，在 ${run.data.poems.toLocaleString()} 首唐代五言诗上训练 ${run.train.steps} 步。验证损失从 ${v0.toFixed(2)} 降到 ${vN.toFixed(2)}。我们在 ${run.detail.length} 个时刻录下逐层、逐位置以及选定权重的数值。`;
  $('#spec').innerHTML = `<span>架构 <b>Qwen3 · ${run.model.layers} 层</b></span><span>隐藏 <b>${run.model.hidden}</b></span><span>注意力 <b>${run.model.heads} 头 / ${run.model.kvHeads} 组 KV</b></span><span>SwiGLU <b>${run.model.ffn}</b></span><span>词表 <b>${run.model.vocab} 字</b></span><span>优化器 <b>AdamW</b></span><span>批大小 <b>${run.train.batch}</b></span><span>用时 <b>${Math.round(run.elapsed)} 秒</b></span>`;
  const pickSteps = [1, 300, 1000, last];
  $('#cards').innerHTML = pickSteps.map((st) => {
    const g = run.samples[String(st)]?.[4]?.text || '';
    return `<button type="button" class="card" data-step="${st}"><span class="tag">第 ${st} 步 · 损失 ${run.log.loss[st - 1].toFixed(2)}</span><h3>${st === 1 ? '刚出生：随机乱码' : st === last ? '训练结束' : st === 300 ? '学会了格式' : '开始像诗'}</h3><p class="poem-line">${esc(g.slice(0, 24)) || '—'}</p><span class="go">从这里开始看 →</span></button>`;
  }).join('');
  $('#credit').innerHTML = `语料：<a href="https://github.com/chinese-poetry/chinese-poetry" target="_blank" rel="noopener">chinese-poetry</a> 全唐诗（MIT），繁转简后只保留五言绝句 / 律诗。训练脚本 tools/train_poet.py，数据全部来自这一次运行。`;

  traceStage = new TraceStage($('#stage'), {
    onPick: (id) => {
      lab?.tree.pause();
      scenePicks.get(id)?.();
    },
  });
  lab = new Lab({
    roots: run.detail.map((s) => ({ t: 'ck', s, label: `第 ${s} 步`, crumb: `第 ${s} 步`, dur: 2.4, kids: () => phases(s) })),
    maxDepth: 5,
    depthNames: ['', '训练全程', '一步训练', '逐层', '一层之内', '头 / 活跃神经元'],
    code: CODE,
    explain,
    render,
    frame: (_node, progress) => traceStage?.frame(progress),
    posText: (t) => `第 <b>${t.root.s}</b> 步 · ${t.r + 1}/${t.roots.length}<br>步骤 <b>${t.i + 1}</b>/${t.list.length}`,
    onExit: () => document.body.classList.replace('mode-inspect', 'mode-pick'),
  });
  window.__lab = lab;

  const enter = (st) => {
    document.body.classList.remove('side-open');
    $('#btnSideOpen').setAttribute('aria-expanded', 'false');
    document.body.classList.replace('mode-pick', 'mode-inspect');
    lab.tree.seekRoot(Math.max(0, run.detail.indexOf(st)), 1);
    Promise.all(run.detail.map(loadStep)).catch(() => {});
  };
  $('#cards').addEventListener('click', (e) => { const c = e.target.closest('.card'); if (c) enter(Number(c.dataset.step)); });
  $('#btnSideOpen').addEventListener('click', () => { document.body.classList.add('side-open'); $('#btnSideOpen').setAttribute('aria-expanded', 'true'); });
  $('#btnSideClose').addEventListener('click', () => { document.body.classList.remove('side-open'); $('#btnSideOpen').setAttribute('aria-expanded', 'false'); });
  $('#btnBack').addEventListener('click', () => { lab.tree.pause(); document.body.classList.remove('side-open'); document.body.classList.replace('mode-inspect', 'mode-pick'); });
  const want = new URLSearchParams(location.search).get('step');
  if (want) enter(Number(want));
}

boot();
