// 玻璃小模型 · D2 一步之内 / D3 拆开环节：把一步训练摊开成一张从上到下的“玻璃计算图”。
//   最上面是这一步的批次（8 段，第 0 段固定是“举头望明月，低头思”）；往下每一行是一个算子：
//   左边写算式，中间是第 0 段在这里的真实激活（8 个位置 × 维度），右边是反向时流到这里的梯度 ∂L/∂·，最右是这一行用到的参数。
//   前向：一行行从上往下亮起来；损失：每个位置 −ln p；反向：梯度从下往上一行行出现，参数换成它们的梯度；更新：参数换成 Δw，再变成新值。
import { COL, text, rr, card, clamp, ease, seg, fmtP, sciSup, fmtInt, hexA, wrap, waitBox, arrow, dot, line, measure } from '../../draw.js';
import { heat, legend, fnum, cellAt, mark, absQuantile, frame, rgb, divRGB } from '../heat.js';
import { FWD_OPS, BWD_OPS, BWD_TENSORS, UPD_TENSORS } from '../timeline.js';
import { TENSOR_LABEL } from '../data.js';
import { paramName } from './overview.js';
import { isEn, L } from '../../lang.js';
import { esc } from '../../../../js/ui.js';

// 每一行：值（第 0 段的激活）、梯度、参数
const ROWS = [
  { id: 'emb', vals: [['h0', 16]], grads: [['h0', 16]], ws: ['E'] },
  { id: 'norm1', vals: [['n1', 16]], grads: [['n1', 16]], ws: ['g1'] },
  { id: 'qkv', vals: [['q', 16], ['k', 16], ['v', 16]], grads: [['q', 16], ['k', 16], ['v', 16]], ws: ['Wq', 'Wk', 'Wv'] },
  { id: 'attn', vals: [['att0', 8], ['att1', 8], ['ao', 16]], grads: [['att0', 8], ['att1', 8], ['ao', 16]], ws: [] },
  { id: 'wo', vals: [['o', 16], ['h1', 16]], grads: [['h1', 16]], ws: ['Wo'] },
  { id: 'norm2', vals: [['n2', 16]], grads: [['n2', 16]], ws: ['g2'] },
  { id: 'ffn', vals: [['gate', 32], ['up', 32], ['act', 32]], grads: [['gate', 32], ['up', 32], ['act', 32]], ws: ['Wg', 'Wu'] },
  { id: 'wd', vals: [['f', 16], ['h2', 16]], grads: [['h2', 16]], ws: ['Wd'] },
  { id: 'normf', vals: [['nf', 16]], grads: [['nf', 16]], ws: ['gf'] },
  { id: 'logits', vals: [['logits', 20], ['probs', 20]], grads: [['logits', 20]], ws: ['E'] },
];
const ROW_OF = new Map(ROWS.map((r, i) => [r.id, i]));

export const OP_NAME = isEn
  ? { emb: 'Embedding lookup', norm1: 'RMSNorm ①', qkv: 'q, k, v', attn: 'Attention (2 heads)', wo: 'W_o + residual', norm2: 'RMSNorm ②', ffn: 'SwiGLU', wd: 'W_down + residual', normf: 'Final RMSNorm', logits: 'logits → probabilities' }
  : { emb: '查嵌入表', norm1: 'RMSNorm ①', qkv: '算 q、k、v', attn: '注意力（两个头）', wo: 'W_o + 残差', norm2: 'RMSNorm ②', ffn: 'SwiGLU 前馈', wd: 'W_down + 残差', normf: '最后的 RMSNorm', logits: 'logits → 概率' };
const FORMULA = {
  emb: 'h₀ = E[x]', norm1: 'n₁ = h₀ / rms(h₀) ⊙ γ₁', qkv: 'q,k,v = n₁·W_q, n₁·W_k, n₁·W_v', attn: 'a = softmax(q·kᵀ/√8)　ao = a·v',
  wo: 'o = ao·W_o　h₁ = h₀ + o', norm2: 'n₂ = h₁ / rms(h₁) ⊙ γ₂', ffn: 'act = silu(n₂·W_gate) ⊙ (n₂·W_up)', wd: 'f = act·W_down　h₂ = h₁ + f',
  normf: 'n_f = h₂ / rms(h₂) ⊙ γ_f', logits: 'logits = n_f·Eᵀ　p = softmax',
};
const NODE_LABEL = { h0: 'h₀', n1: 'n₁', q: 'q', k: 'k', v: 'v', att0: L('注意力 · 头 0', 'attn · head 0'), att1: L('注意力 · 头 1', 'attn · head 1'), ao: 'ao', o: 'o', h1: 'h₁', n2: 'n₂', gate: 'gate', up: 'up', act: 'act', f: 'f', h2: 'h₂', nf: 'n_f', logits: 'logits', probs: L('概率 p', 'probabilities p') };
const BWD_NOTE = { h1: L('∂L/∂h₁（= ∂L/∂o）', '∂L/∂h₁ (= ∂L/∂o)'), h2: L('∂L/∂h₂（= ∂L/∂f）', '∂L/∂h₂ (= ∂L/∂f)') };

// 暂停时单步过来：这一步的动画也自己播一遍（约 1.4 秒）然后停在结束的样子；播放时跟着时间轴走
export function animP(sc, st, env, dur = 1.4) {
  const key = `${st.depth}:${st.k}:${st.i}`;
  if (sc._pk !== key) { sc._pk = key; sc._pt0 = env.t; }
  return st.playing ? st.p : Math.max(st.p, Math.min(1, (env.t - sc._pt0) / dur));
}

export class GStep {
  constructor(app, R) { this.app = app; this.R = R; this.D = R.D; }

  layout(env) {
    const P = (this.portrait = env.portrait);
    const D = this.D;
    this.rows = [];
    if (!P) {
      this.W = 1290;
      this.col = { label: 0, val: 182, grad: 640, w: 1088 };
      this.slot = 420;
      this.batch = { x: 0, y: 70, w: 1290, h: 240 };
      let y = 340;
      for (const r of ROWS) { const h = r.id === 'emb' || r.id === 'logits' ? 168 : r.id === 'qkv' || r.id === 'ffn' ? 150 : 142; this.rows.push({ ...r, y, h }); y += h + 12; }
      this.lossRow = { y, h: 170 };
      this.H = y + 190;
    } else {
      this.W = 380;
      this.col = { label: 0, val: 16, grad: 16, w: 16 };
      this.slot = 360;
      this.batch = { x: 0, y: 70, w: 380, h: 360 };
      let y = 460;
      for (const r of ROWS) { const h = 470; this.rows.push({ ...r, y, h }); y += h + 14; }
      this.lossRow = { y, h: 260 };
      this.H = y + 280;
    }
  }

  focus(st) {
    const s = st.step, P = this.portrait;
    const all = { x: -12, y: 0, w: this.W + 24, h: this.H };
    const rowRect = (i, pad = 18) => { const r = this.rows[i]; return { x: -12, y: r.y - pad - 10, w: this.W + 24, h: r.h + pad * 2 + 10 }; };
    if (s.ph === 'batch') return { x: -12, y: 0, w: this.W + 24, h: this.batch.y + this.batch.h + 20 };
    if (s.ph === 'loss') {
      const r = this.rows[ROW_OF.get('logits')];
      return { x: -12, y: r.y - 30, w: this.W + 24, h: this.lossRow.y + this.lossRow.h - r.y + 40 };
    }
    if (!s.sub) {
      // 一步之内（D2）：前向 / 反向时镜头跟着正在算的那一行走（看得清真实数值）；更新时看全部参数
      if (s.ph === 'upd') return { x: -12, y: this.batch.y + this.batch.h - 10, w: this.W + 24, h: this.lossRow.y + this.lossRow.h - this.batch.y - this.batch.h + 20 };
      const yc = this.followY({ ...st, p: this._pk === `${st.depth}:${st.k}:${st.i}` && this._ap != null ? this._ap : st.p });
      const h = P ? 560 : 640;
      return { x: -12, y: yc - h / 2, w: this.W + 24, h };
    }
    if (s.ph === 'fwd' || s.ph === 'bwd') return rowRect(ROW_OF.get(s.sub), P ? 10 : 40);
    if (s.ph === 'upd') {
      if (s.sub === 'clip') return { x: -12, y: this.lossRow.y - 30, w: this.W + 24, h: this.lossRow.h + 50 };
      const i = this.rowOfTensor(s.t);
      return rowRect(i, P ? 10 : 40);
    }
    return all;
  }

  // 前向 / 反向进行到的位置（世界坐标的 y），在相邻两行之间连续移动
  followY(st) {
    const s = st.step, n = ROWS.length;
    const x = clamp(st.p * n * 1.05, 0, n - 0.001);
    const i = Math.floor(x), f = x - i;
    const rowIdx = (j) => (s.ph === 'bwd' ? ROW_OF.get(BWD_OPS[j]) : j);
    const c = (j) => { const r = this.rows[rowIdx(Math.min(n - 1, j))]; return r.y + r.h / 2; };
    return c(i) + (c(Math.min(n - 1, i + 1)) - c(i)) * f;
  }

  rowOfTensor(t) { return t === 'E' ? 0 : this.rows.findIndex((r) => r.ws.includes(t)); }

  // 前向 / 反向进行到哪一行（D2 按进度，D3 按算子）
  fwdState(st) {
    const s = st.step;
    if (s.ph === 'batch') return { cur: -1, f: 0 };
    if (s.ph !== 'fwd') return { cur: ROWS.length, f: 1 };
    if (!s.sub) { const x = st.p * ROWS.length * 1.05; return { cur: Math.min(ROWS.length, Math.floor(x)), f: x - Math.floor(x) }; }
    return { cur: ROW_OF.get(s.sub), f: ease(seg(st.p, 0, 0.6)) };
  }
  bwdState(st) {
    const s = st.step;
    if (s.ph === 'upd') return { cur: ROWS.length, f: 1 };
    if (s.ph !== 'bwd') return { cur: -1, f: 0 };
    if (!s.sub) { const x = st.p * ROWS.length * 1.05; return { cur: Math.min(ROWS.length, Math.floor(x)), f: x - Math.floor(x) }; }
    return { cur: BWD_OPS.indexOf(s.sub), f: ease(seg(st.p, 0, 0.6)) };
  }
  activeRow(st) {
    const s = st.step;
    if (s.ph === 'fwd') { const c = this.fwdState(st).cur; return Math.min(ROWS.length - 1, c); }
    if (s.ph === 'bwd') { const c = this.bwdState(st).cur; return ROW_OF.get(BWD_OPS[Math.min(BWD_OPS.length - 1, c)]); }
    return -1;
  }

  draw(g, st, env) {
    st = { ...st, p: animP(this, st, env, st.step.sub ? 1.4 : 3) };
    this._ap = st.p;   // 镜头跟随用（focus 在 draw 之前调用，晚一帧没关系）
    const D = this.D, k = st.k, s = st.step, P = this.portrait;
    const t = D.FR[k];
    if (!D.ready(k)) { waitBox(g, 0, 60, this.W, 400, env, st.wait, { withCard: true, label: L('正在载入这一步的真实记录…', 'Loading the real record of this step…') }); return; }
    const phName = isEn ? { batch: '① Batch', fwd: '② Forward', loss: '③ Loss', bwd: '④ Backward', upd: '⑤ Update' } : { batch: '① 取批次', fwd: '② 前向', loss: '③ 损失', bwd: '④ 反向', upd: '⑤ 更新' };
    text(g, L(`GLASS MODEL · 第 ${t + 1} 步的里面 · 全部是真实数值`, `GLASS MODEL · INSIDE STEP ${t + 1} · ALL REAL NUMBERS`), 0, 16, { size: 10, kind: 'mono', color: COL.dim });
    const sub = s.sub ? (s.ph === 'fwd' || s.ph === 'bwd' ? ` · ${OP_NAME[s.sub]}` : s.ph === 'upd' ? (s.sub === 'clip' ? L(' · 梯度裁剪', ' · gradient clipping') : ` · ${TENSOR_LABEL[s.t]}`) : s.ph === 'loss' ? (s.sub === 'mean' ? L(' · 平均', ' · mean') : L(` · 位置 ${s.i}`, ` · position ${s.i}`)) : s.sub === 'pick' ? L(' · 取 8 段', ' · pick 8 windows') : L(' · 错开一位', ' · shift by one')) : '';
    text(g, `${phName[s.ph]}${sub}`, 0, 48, { size: P ? 19 : 26, kind: 'serif', weight: 900, color: COL.ink, max: this.W });
    this.drawBatch(g, st, env);
    const F = this.fwdState(st), B = this.bwdState(st);
    // 左边的残差主干：h₀ → h₁ → h₂
    this.drawSpine(g, st, env, F, B);
    this.rows.forEach((r, i) => this.drawRow(g, st, env, r, i, F, B));
    this.drawLoss(g, st, env, F, B);
  }

  /* ---------------------------------------------------------------- 批次 */

  drawBatch(g, st, env) {
    const D = this.D, k = st.k, t = D.FR[k], s = st.step, C = this.batch, P = this.portrait;
    const on = s.ph === 'batch';
    card(g, C.x, C.y, C.w, C.h, { eyebrow: L(`BATCH · 第 ${t + 1} 步 · 8 段 × 9 个字`, `BATCH · STEP ${t + 1} · 8 WINDOWS × 9 CHARACTERS`), title: L('《静夜思》首尾相接成一个 25 字的圈，从里面取 8 段', 'Li Bai’s “Quiet Night Thought” as a 25-character loop; take 8 windows from it'), accent: COL.blue, active: on });
    const NS = D.stream.length, B = D.B, T = D.T;
    const pick = on && s.sub === 'pick' ? st.p : on && !s.sub ? clamp(st.p * 1.6, 0, 1) : 1;
    const shift = on && s.sub === 'shift' ? ease(seg(st.p, 0.1, 0.7)) : on && !s.sub ? ease(seg(st.p, 0.55, 0.95)) : 1;
    // 诗的圈（展开成一行）
    const tw = P ? 13.6 : 26, x0 = C.x + 16, y0 = C.y + 62;
    const offs = []; for (let b = 0; b < B; b++) offs.push(D.offs(t, b));
    for (let i = 0; i < NS; i++) {
      const used = offs.some((o, b) => b / B < pick && ((i - o + NS) % NS) <= T);
      rr(g, x0 + i * tw, y0, tw - 2, tw + 4, 4);
      g.fillStyle = used ? hexA(COL.blue, 0.16) : 'rgba(255,255,255,0.03)'; g.fill();
      g.strokeStyle = used ? hexA(COL.blue, 0.5) : COL.line2; g.lineWidth = 1; g.stroke();
      text(g, D.ch(D.stream[i]), x0 + i * tw + (tw - 2) / 2, y0 + tw * 0.72, { size: P ? 10 : 14, color: COL.ink, align: 'center' });
    }
    // 8 段的起点
    for (let b = 0; b < B; b++) {
      if (b / B >= pick) break;
      const o = offs[b], y = y0 + tw + 10 + b * (P ? 4 : 5);
      const col = b === 0 ? COL.amber : hexA(COL.blue, 0.75);
      for (let i = 0; i <= T; i++) { const xi = (o + i) % NS; g.fillStyle = col; g.fillRect(x0 + xi * tw + 1, y, tw - 4, P ? 2 : 3); }
    }
    // 8 段：输入 → 目标
    const ty0 = P ? y0 + tw + 60 : C.y + 64, tx0 = P ? C.x + 16 : C.x + 16 + NS * tw + 30;
    const cs = P ? 18 : 19, gap = P ? 18 : 26;
    text(g, L('输入（前 8 个字）', 'input (first 8)'), tx0 + 22, ty0 - 6, { size: 10, color: COL.dim });
    text(g, L('目标（错开一位）', 'target (shifted by one)'), tx0 + 22 + T * cs + gap, ty0 - 6, { size: 10, color: COL.dim });
    for (let b = 0; b < B; b++) {
      if (b / B >= pick) break;
      const y = ty0 + 4 + b * (cs + 3), w = D.batch(t)[b];
      text(g, b === 0 ? L('固定', 'fixed') : `#${b}`, tx0 + 18, y + cs * 0.7, { size: 9.5, kind: 'mono', color: b === 0 ? COL.amber : COL.dim, align: 'right' });
      for (let i = 0; i < T; i++) {
        const x = tx0 + 22 + i * cs;
        rr(g, x, y, cs - 2, cs, 3); g.fillStyle = b === 0 ? hexA(COL.amber, 0.1) : 'rgba(255,255,255,0.04)'; g.fill();
        text(g, D.ch(w[i]), x + (cs - 2) / 2, y + cs * 0.72, { size: 11.5, color: COL.ink, align: 'center' });
        const xt = tx0 + 22 + T * cs + gap + i * cs - (1 - shift) * (T * cs + gap - cs);
        g.globalAlpha = shift;
        rr(g, xt, y, cs - 2, cs, 3); g.fillStyle = hexA(COL.cyan, 0.1); g.fill();
        text(g, D.ch(w[i + 1]), xt + (cs - 2) / 2, y + cs * 0.72, { size: 11.5, color: COL.cyan, align: 'center' });
        g.globalAlpha = 1;
      }
    }
    if (!P) text(g, L(`第 0 段每一步都一样（方便对比）；另外 7 段的起点每一步随机抽。8 × 8 = 64 道“猜下一个字”的题`, `Row 0 is the same every step (for comparison); the other 7 start at random places. 8 × 8 = 64 “guess the next character” questions`), C.x + 16, C.y + C.h - 14, { size: 10.5, color: COL.dim });
  }

  /* ---------------------------------------------------------------- 残差主干 */

  drawSpine(g, st, env, F, B) {
    if (this.portrait) return;
    const r0 = this.rows[0], r4 = this.rows[ROW_OF.get('wo')], r7 = this.rows[ROW_OF.get('wd')], r8 = this.rows[ROW_OF.get('normf')];
    const x = this.col.val - 22;
    const y0 = r0.y + 70, y1 = r8.y + 60;
    g.strokeStyle = 'rgba(150,180,230,0.22)'; g.lineWidth = 3;
    g.beginPath(); g.moveTo(x, y0); g.lineTo(x, y1); g.stroke();
    for (const r of [r4, r7]) { const y = r.y + 72; dot(g, x, y, 7, COL.bg, COL.line3); text(g, '+', x, y + 4, { size: 12, color: COL.ink2, align: 'center' }); }
    text(g, L('残差', 'residual'), x - 4, y0 - 8, { size: 9, color: COL.dim, align: 'center' });
    // 前向的光点往下走，反向的往上走
    const pulse = (y, col) => { const gr = g.createRadialGradient(x, y, 0, x, y, 12); gr.addColorStop(0, hexA(col, 0.9)); gr.addColorStop(1, hexA(col, 0)); g.fillStyle = gr; g.beginPath(); g.arc(x, y, 12, 0, Math.PI * 2); g.fill(); };
    const s = st.step;
    if (s.ph === 'fwd' && F.cur < ROWS.length) { const r = this.rows[Math.min(ROWS.length - 1, F.cur)]; pulse(clamp(r.y + 20 + F.f * r.h, y0, y1), COL.cyan); }
    if (s.ph === 'bwd' && B.cur < ROWS.length) { const r = this.rows[ROW_OF.get(BWD_OPS[Math.min(BWD_OPS.length - 1, B.cur)])]; pulse(clamp(r.y + r.h - B.f * r.h, y0, y1), COL.rose); }
  }

  /* ---------------------------------------------------------------- 一行 */

  nodeData(k, name, grad) {
    const D = this.D;
    if (name === 'att0' || name === 'att1') {
      const h = name === 'att0' ? 0 : 1, T = D.T;
      if (!grad) return { get: (r, c) => D.att(k, h, r, c), rows: T, cols: T, arr: null };   // 注意力本身在权重块里
      const a = D.grad(k, 'att');
      return { get: (r, c) => a[(h * T + r) * T + c], rows: T, cols: T, arr: a.subarray(h * T * T, (h + 1) * T * T) };
    }
    if (name === 'probs') { const T = D.T, V = D.V; return { get: (r, c) => D.probs(k, r, c), rows: T, cols: V, arr: null }; }
    const a = grad ? D.grad(k, name) : D.act(k, name);
    const cols = a.length / D.T;
    return { get: (r, c) => a[r * cols + c], rows: D.T, cols, arr: a };
  }

  // 一个 slot 里摆几个节点：返回每个节点的 x 和格子宽
  slotLayout(list, x0) {
    const n = list.length, gap = 12, wAll = this.slot - gap * (n - 1);
    const tot = list.reduce((a, [, c]) => a + c, 0);
    const P = this.portrait;
    // 16 维的单个节点不拉满（格子太扁）；其余按维数分宽度
    if (n === 1 && list[0][1] === 16) return [{ x: x0, cw: P ? 14 : 12 }];
    let x = x0;
    const out = [];
    for (const [, c] of list) {
      const w = n === 3 && list[0][1] === 16 ? wAll / 3 : (wAll * c) / tot;
      out.push({ x, cw: Math.min(P ? 14 : 12, w / c) });
      x += Math.min(w, (P ? 14 : 12) * c) + gap;
    }
    return out;
  }

  drawRow(g, st, env, r, i, F, B) {
    const D = this.D, k = st.k, s = st.step, P = this.portrait, T = D.T;
    const ch = P ? 11 : 12;
    const fwdDone = i < F.cur, fwdNow = i === F.cur && s.ph === 'fwd';
    const bi = BWD_OPS.indexOf(r.id);
    const bwdDone = bi < B.cur, bwdNow = bi === B.cur && s.ph === 'bwd';
    const isActive = (s.ph === 'fwd' && (fwdNow || (s.sub === r.id))) || (s.ph === 'bwd' && bwdNow) || (s.ph === 'upd' && s.sub === 't' && r.ws.includes(s.t) && !(s.t === 'E' && r.id === 'logits'));
    // 行底
    const y = r.y;
    rr(g, -8, y - 8, this.W + 16, r.h + 4, 12);
    g.fillStyle = isActive ? 'rgba(15,24,42,0.94)' : 'rgba(10,17,31,0.55)'; g.fill();
    g.strokeStyle = isActive ? hexA(s.ph === 'bwd' ? COL.rose : s.ph === 'upd' ? COL.violet : COL.cyan, 0.7) : COL.line; g.lineWidth = isActive ? 1.4 : 1; g.stroke();
    // 左：名字和算式
    const lx = 6, ly = y + 16;
    text(g, OP_NAME[r.id], lx, ly, { size: P ? 14 : 15, kind: 'serif', weight: 600, color: isActive ? COL.ink : COL.ink2 });
    if (P) text(g, FORMULA[r.id], lx + measure(g, OP_NAME[r.id], 14, 'serif', 600) + 10, ly, { size: 10, kind: 'mono', color: COL.dim, max: 380 - lx - measure(g, OP_NAME[r.id], 14, 'serif', 600) - 14 });
    else wrap(g, FORMULA[r.id], lx, ly + 20, 160, 14, { size: 10.5, kind: 'mono', color: COL.dim });
    // 中：值
    const vy = P ? y + 42 : y + 30;
    const showVal = fwdDone || fwdNow;
    const va = fwdNow ? F.f : 1;
    const lay = this.slotLayout(r.vals, this.col.val);
    r.vals.forEach(([name], j) => {
      const L0 = lay[j], nd = this.nodeData(k, name, false);
      const w = nd.cols * L0.cw;
      text(g, NODE_LABEL[name], L0.x, vy - 6, { size: 10, kind: 'mono', color: COL.ink2 });
      if (!showVal) { rr(g, L0.x, vy, w, nd.rows * ch, 3); g.strokeStyle = COL.line2; g.setLineDash([3, 3]); g.stroke(); g.setLineDash([]); return; }
      const seq = name.startsWith('att') || name === 'probs';
      const sc = seq ? 1 : absQuantile(nd.arr, 0.99);
      // 正在算：从左往右一列列出现
      const cut = fwdNow ? Math.ceil(nd.cols * va) : nd.cols;
      heat(g, L0.x, vy, nd.rows, nd.cols, L0.cw, ch, (rr_, c) => (c < cut ? nd.get(rr_, c) : 0), { scale: sc, seq, s: env.s });
      if (j === 0 || name.startsWith('att')) for (let p = 0; p < T; p++) text(g, D.ch(D.fixed[p]), L0.x - 3, vy + p * ch + ch * 0.8, { size: Math.min(10, ch * 0.85), color: COL.dim, align: 'right' });
      if (name.startsWith('att')) for (let p = 0; p < T; p++) text(g, D.ch(D.fixed[p]), L0.x + p * L0.cw + L0.cw / 2, vy + T * ch + 9, { size: 8.5, color: COL.dim, align: 'center' });
      if (name === 'probs') this.markTargets(g, L0.x, vy, L0.cw, ch);
      env.hit(L0.x, vy, w, nd.rows * ch, { tipAt: (wx, wy) => { const c = cellAt(wx, wy, L0.x, vy, nd.rows, nd.cols, L0.cw, ch); return c ? this.nodeTip(name, c.r, c.c, nd.get(c.r, c.c), false) : null; } });
    });
    // 右：梯度（反向走到这一行之后才有）
    const gy = P ? vy + T * ch + 34 : vy;
    const showG = bwdDone || bwdNow;
    const gx0 = this.col.grad;
    if (!P) { g.strokeStyle = COL.line; g.beginPath(); g.moveTo(gx0 - 22, y + 4); g.lineTo(gx0 - 22, y + r.h - 12); g.stroke(); }
    const glay = this.slotLayout(r.grads, gx0);
    r.grads.forEach(([name], j) => {
      const L0 = glay[j], nd = this.nodeData(k, name, true);
      const w = nd.cols * L0.cw;
      text(g, BWD_NOTE[name] || `∂L/∂${NODE_LABEL[name].replace(/ · .*/, '') === NODE_LABEL[name] ? NODE_LABEL[name] : (name === 'att0' ? 'a₀' : 'a₁')}`, L0.x, gy - 6, { size: 10, kind: 'mono', color: showG ? COL.rose : COL.faint });
      if (!showG) { rr(g, L0.x, gy, w, nd.rows * ch, 3); g.strokeStyle = 'rgba(255,107,147,0.16)'; g.setLineDash([3, 3]); g.stroke(); g.setLineDash([]); return; }
      const sc = absQuantile(nd.arr, 0.99);
      const cut = bwdNow ? Math.ceil(nd.cols * B.f) : nd.cols;
      heat(g, L0.x, gy, nd.rows, nd.cols, L0.cw, ch, (rr_, c) => (c >= nd.cols - cut ? nd.get(rr_, c) : 0), { scale: sc, s: env.s });
      frame(g, L0.x - 2, gy - 2, w + 4, nd.rows * ch + 4, COL.rose, 0.5, 1);
      if (P && j === 0) for (let p = 0; p < T; p++) text(g, D.ch(D.fixed[p]), L0.x - 3, gy + p * ch + ch * 0.8, { size: Math.min(10, ch * 0.85), color: COL.dim, align: 'right' });
      env.hit(L0.x, gy, w, nd.rows * ch, { tipAt: (wx, wy) => { const c = cellAt(wx, wy, L0.x, gy, nd.rows, nd.cols, L0.cw, ch); return c ? this.nodeTip(name, c.r, c.c, nd.get(c.r, c.c), true) : null; } });
    });
    if (!showG && !P && r.grads.length) text(g, s.ph === 'bwd' ? L('等反向传到这里…', 'waiting for the backward pass…') : L('反向时这里出现梯度 ∂L/∂·', 'gradients ∂L/∂· appear here on the backward pass'), gx0, gy + T * ch + 18, { size: 10, color: COL.faint });
    if (r.id === 'attn') wrap(g, L('这一行没有参数：RoPE 按位置旋转 q、k，是固定的公式，不用学', 'No parameters in this row: RoPE rotates q and k by position with a fixed formula — nothing to learn'), this.col.w, P ? gy + T * ch + 40 : vy + 14, P ? 350 : 190, 14, { size: 10, color: COL.dim });
    // 最右：参数（前向时是数值，反向时是梯度，更新时是 Δw → 新值）
    const wy = P ? gy + T * ch + 40 : vy;
    this.drawWeights(g, st, env, r, i, wy, { bwdDone: bwdDone || (bwdNow && B.f > 0.5) });
  }

  // 概率格子里把正确答案框出来
  markTargets(g, x, y, cw, ch) {
    const D = this.D;
    for (let p = 0; p < D.T; p++) { g.strokeStyle = COL.green; g.lineWidth = 1.4; g.strokeRect(x + D.fixed[p + 1] * cw - 0.5, y + p * ch - 0.5, cw + 1, ch + 1); }
  }

  nodeTip(name, r, c, v, grad) {
    const D = this.D;
    const pos = L(`位置 ${r}「${esc(D.ch(D.fixed[r]))}」`, `position ${r} “${esc(D.ch(D.fixed[r]))}”`);
    if (name.startsWith('att')) return `<span class="k">${grad ? '∂L/∂' : ''}${NODE_LABEL[name]}</span>${pos} → ${L(`位置 ${c}「${esc(D.ch(D.fixed[c]))}」`, `position ${c} “${esc(D.ch(D.fixed[c]))}”`)}：<span class="v">${grad ? fnum(v, 3) : fmtP(v)}</span>`;
    if (name === 'probs' || name === 'logits') return `<span class="k">${grad ? '∂L/∂' : ''}${NODE_LABEL[name]} · ${pos}</span>${L('字', 'char')}「${esc(D.ch(c))}」：<span class="v">${name === 'probs' ? fmtP(v) : fnum(v, 4)}</span>${c === D.fixed[r + 1] ? L('　← 正确答案', '　← the answer') : ''}${grad && name === 'logits' ? L('<br>= (p − 1{正确}) / 64', '<br>= (p − 1{answer}) / 64') : ''}`;
    return `<span class="k">${grad ? (BWD_NOTE[name] || `∂L/∂${NODE_LABEL[name]}`) : NODE_LABEL[name]} · ${pos}</span>${L(`第 ${c} 维`, `dim ${c}`)}：<span class="v">${fnum(v, 4)}</span>`;
  }

  /* ---------------------------------------------------------------- 参数 */

  drawWeights(g, st, env, r, ri, wy, { bwdDone }) {
    const D = this.D, k = st.k, s = st.step, P = this.portrait;
    if (!r.ws.length) return;
    const Wk = D.W(k), G = D.G(k), DW = D.DW(k), lr = D.lr[D.FR[k]];
    const upd = s.ph === 'upd';
    let x = this.col.w;
    const ws = r.ws;
    const cellOf = (p) => (P ? (p.norm ? 14 : p.name === 'E' ? 7 : p.cols === 32 ? 5 : p.rows === 32 ? 5 : 8) : (p.norm ? 11 : p.name === 'E' ? 5 : p.cols === 32 ? 3 : p.rows === 32 ? 3 : ws.length === 3 ? 4 : 6));
    for (const name of ws) {
      const p = D.pIndex.get(name);
      const cw = cellOf(p);
      const rows = p.norm ? 1 : p.rows, cols = p.norm ? p.rows : p.cols;
      const w = cols * cw, h = rows * cw;
      // 这一格显示什么
      // 前向：数值 w；反向走到这一行以后：梯度 g；更新：这一步的 Δw → 更新后的值（D3 按张量依次来，还没轮到的仍显示梯度）
      let mode = 'w';
      if (upd && !s.sub) mode = st.p < 0.55 ? 'dw' : 'w1';
      else if (upd && s.sub === 't') mode = s.t === name && !(name === 'E' && r.id === 'logits') ? (st.p < 0.6 ? 'dw' : 'w1') : UPD_TENSORS.indexOf(s.t) > UPD_TENSORS.indexOf(name) ? 'w1' : 'g';
      else if (upd && s.sub === 'clip') mode = 'g';
      else if (bwdDone) mode = 'g';
      const val = (gi) => (mode === 'g' ? G[gi] : mode === 'dw' ? DW[gi] : mode === 'w1' ? Wk[gi] + DW[gi] - (p.norm ? 1 : 0) : Wk[gi] - (p.norm ? 1 : 0));
      let sc;
      if (mode === 'dw') sc = lr * 1.05;
      else if (mode === 'g') sc = absQuantile(G, 0.99, p.off, p.off + p.n) || 1e-9;
      else if (p.norm) { sc = 0.05; for (let q = p.off; q < p.off + p.n; q++) sc = Math.max(sc, Math.abs(Wk[q] - 1)); }
      else sc = absQuantile(Wk, 0.99, p.off, p.off + p.n);
      const gi = (rr_, c) => p.off + (p.norm ? c : rr_ * p.cols + c);
      const lab = name === 'E' && r.id === 'logits' ? L('Eᵀ（同一张表）', 'Eᵀ (same table)') : TENSOR_LABEL[name];
      const modeLab = { w: '', g: ' · ∂L/∂W', dw: ' · Δw', w1: L(' · 更新后', ' · updated') }[mode];
      const col = { w: COL.ink2, g: COL.rose, dw: COL.violet, w1: COL.cyan }[mode];
      text(g, lab + modeLab, x, wy - 6, { size: 10, kind: 'mono', color: col });
      heat(g, x, wy, rows, cols, cw, cw, (a, b) => val(gi(a, b)), { scale: sc, s: env.s });
      if (mode !== 'w') frame(g, x - 2, wy - 2, w + 4, h + 4, col, 0.6, 1);
      const sel = this.app.ctx.gsel;
      if (sel != null && sel >= p.off && sel < p.off + p.n) { const lc = D.locate(sel); mark(g, x, wy, p.norm ? 0 : lc.i, p.norm ? lc.i : lc.j, cw, cw); }
      env.hit(x, wy, w, h, {
        click: true,
        tipAt: (wx, wy2) => { const c = cellAt(wx, wy2, x, wy, rows, cols, cw, cw); if (!c) return null; const q = gi(c.r, c.c); return `<span class="k">${esc(paramName(D, q))}</span>w <span class="v">${fnum(Wk[q], 4)}</span>　∂L/∂w <span class="a">${fnum(G[q], 3)}</span>　Δw ${fnum(DW[q], 3)}<br><span style="color:var(--dim)">${L('点一下看这个参数的一生', 'Click to see this parameter’s life')}</span>`; },
        act: (h) => { const c = h && cellAt(h.wx, h.wy, x, wy, rows, cols, cw, cw); if (c) this.app.pickParam(gi(c.r, c.c)); },
      });
      x += w + (P ? 14 : 10);
      if (P && x > 300) { x = this.col.w; wy += h + 26; }
    }
  }

  /* ---------------------------------------------------------------- 损失 */

  drawLoss(g, st, env, F, B) {
    const D = this.D, k = st.k, t = D.FR[k], s = st.step, P = this.portrait, T = D.T;
    const y = this.lossRow.y, h = this.lossRow.h;
    const on = s.ph === 'loss' || (s.ph === 'upd' && s.sub === 'clip');
    rr(g, -8, y - 8, this.W + 16, h, 12);
    g.fillStyle = on ? 'rgba(15,24,42,0.94)' : 'rgba(10,17,31,0.55)'; g.fill();
    g.strokeStyle = on ? hexA(COL.amber, 0.7) : COL.line; g.lineWidth = on ? 1.4 : 1; g.stroke();
    text(g, L('损失', 'Loss'), 6, y + 16, { size: 15, kind: 'serif', weight: 600, color: on ? COL.ink : COL.ink2 });
    text(g, 'L = mean(−ln p[正确])'.replace('正确', L('正确', 'answer')), 6, y + 36, { size: 10.5, kind: 'mono', color: COL.dim });
    if (F.cur < ROWS.length) { text(g, L('前向算完才有', 'available after the forward pass'), this.col.val, y + 60, { size: 11, color: COL.faint }); return; }
    // 每个位置的 −ln p
    const prog = s.ph === 'loss' ? (s.sub === 'pos' ? (s.i + ease(seg(st.p, 0, 0.5))) / T : s.sub === 'mean' ? 1 : clamp(st.p * 1.3, 0, 1)) : 1;
    const x0 = this.col.val, bw = P ? 40 : 52, maxL = 3.2;
    let sum = 0;
    for (let i = 0; i < T; i++) {
      const p = D.probs(k, i, D.fixed[i + 1]), nl = -Math.log(Math.max(p, 1e-9));
      const x = x0 + i * (bw + 6);
      const vis = clamp(prog * T - i, 0, 1);
      const cur = s.ph === 'loss' && s.sub === 'pos' && s.i === i;
      text(g, `${D.ch(D.fixed[i])}→${D.ch(D.fixed[i + 1])}`, x + bw / 2, y + 60, { size: 11, color: cur ? COL.amber : COL.ink2, align: 'center' });
      const bh = (P ? 70 : 80) * Math.min(1, nl / maxL) * vis;
      const by = y + (P ? 150 : 150);
      rr(g, x + 6, by - bh, bw - 12, Math.max(1, bh), 3); g.fillStyle = nl > 1 ? hexA(COL.rose, 0.75) : nl > 0.3 ? hexA(COL.amber, 0.75) : hexA(COL.cyan, 0.75); g.fill();
      if (vis > 0.5) { text(g, nl.toFixed(2), x + bw / 2, by - bh - 5, { size: 10, kind: 'mono', color: COL.ink, align: 'center' }); text(g, `p ${fmtP(p)}`, x + bw / 2, y + 74, { size: 9, kind: 'mono', color: COL.dim, align: 'center' }); }
      if (vis > 0) sum += nl;
      env.hit(x, y + 50, bw, 110, { tip: `<span class="k">${L(`位置 ${i}`, `position ${i}`)}</span>${L('正确答案', 'answer')}「${esc(D.ch(D.fixed[i + 1]))}」p = <span class="v">${fmtP(p)}</span><br>−ln p = <span class="a">${nl.toFixed(4)}</span>` });
    }
    const mx = x0 + T * (bw + 6) + 20;
    if (!P && prog >= 1) {
      text(g, L(`第 0 段平均 ${(sum / T).toFixed(4)}`, `row 0 mean ${(sum / T).toFixed(4)}`), mx, y + 92, { size: 12, kind: 'mono', color: COL.amber });
      text(g, L(`整批 64 个位置平均 = 这一步的损失 ${D.loss[t].toFixed(4)}`, `mean over all 64 positions = this step’s loss ${D.loss[t].toFixed(4)}`), mx, y + 114, { size: 12, kind: 'mono', color: COL.ink });
      text(g, L(`瞎猜时 ln 20 = ${Math.log(20).toFixed(3)}`, `blind guess: ln 20 = ${Math.log(20).toFixed(3)}`), mx, y + 136, { size: 10.5, color: COL.dim });
    } else if (P && prog >= 1) {
      text(g, L(`第 0 段 ${(sum / T).toFixed(3)} · 整批 ${D.loss[t].toFixed(3)}`, `row 0 ${(sum / T).toFixed(3)} · batch ${D.loss[t].toFixed(3)}`), x0, y + 180, { size: 11.5, kind: 'mono', color: COL.amber });
    }
    // 裁剪（更新的第一小步）：全部梯度的长度
    if (s.ph === 'upd' && s.sub === 'clip') {
      const gn = D.gnorm[t], c = D.clip[t];
      const tx = P ? x0 : mx, ty = P ? y + 210 : y + 30;
      text(g, L(`‖g‖ = ${gn.toFixed(4)}`, `‖g‖ = ${gn.toFixed(4)}`), tx, ty, { size: 13, kind: 'mono', weight: 700, color: COL.violet });
      text(g, c < 1 ? L(`> 1.0：全部梯度乘 ${c.toFixed(4)}`, `> 1.0: every gradient × ${c.toFixed(4)}`) : L('≤ 1.0：不裁剪', '≤ 1.0: no clipping'), tx + (P ? 150 : 150), ty, { size: 11.5, color: COL.ink2 });
    }
  }
}
