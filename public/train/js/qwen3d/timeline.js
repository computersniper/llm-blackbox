// 第三章（Qwen3-0.6B 三步 SFT）的“步骤树”，接口和 ../glass/timeline.js 的 GlassTimeline 一样（页面和控制条不用关心是哪一章）。
//
//   D1 三步全程  每一步一拍：进料 → 前向 → 损失 → 反向 → 更新；三步走完再加一拍“结果”（3 步之后的概率 + DPO 浮层）
//   D2 一步之内  ① 进料 → ② 前向 → ③ 损失 → ④ 反向 → ⑤ 更新（走完自动进入下一步）
//   D3 一层之内  进料：套模板 / 遮住提示 / 错开一位；前向、反向：选中的那一层逐个算子（其余层一拍带过）；
//               损失：回答的 8 个位置逐个；更新：裁剪 / AdamW / 存回 bf16
//   D4 一个张量  有参数的算子拆成每个张量（W_q、q_norm、W_k……），看它的梯度、更新量
//   D5 一个权重  更新：这个张量里跟踪的那个权重的 AdamW 算式（g → m → v → 偏差校正 → Δw → 写回 → bf16 比特）；
//               反向：这个权重在梯度热力图里的位置和它的真实梯度
// 选中的层 L 由页面设置（点任意一块层板换层），默认第 27 层（离损失最近，反向第一个到）。
import { isEn } from '../lang.js';

export const Q_MAX_DEPTH = 5;
export const Q_MIN_DEPTH = 1;
export const Q_DEPTH_NAMES = isEn ? ['', 'Three steps', 'One step', 'One layer', 'One tensor', 'One weight'] : ['', '三步全程', '一步之内', '一层之内', '一个张量', '一个权重'];
export const Q_PHASES = ['batch', 'fwd', 'loss', 'bwd', 'upd'];
export const BATCH_SUBS = ['tpl', 'mask', 'shift'];
// 一层之内的前向算子（顺序就是计算顺序）；lo / hi = 选中层下面 / 上面的那些层（一拍带过）
export const LAYER_OPS = ['ln1', 'qkv', 'attn', 'o', 'ln2', 'ffn', 'down'];
export const OP_TENSORS = {
  emb: ['embed'], ln1: ['ln1'], qkv: ['q', 'qn', 'k', 'kn', 'v'], attn: [], o: ['o'], ln2: ['ln2'], ffn: ['gate', 'up'], down: ['down'], head: ['norm', 'embed'],
};
// 反向里每个算子产生哪些张量的梯度（顺序和前向相反）
export const OP_TENSORS_BWD = {
  head: ['embed', 'norm'], down: ['down'], ffn: ['up', 'gate'], ln2: ['ln2'], o: ['o'], attn: [], qkv: ['v', 'kn', 'k', 'qn', 'q'], ln1: ['ln1'], emb: ['embed'],
};
// 更新：选中层的 11 个张量 + 嵌入表 + 最后的 RMSNorm
export const UPD_TENSORS = ['embed', 'ln1', 'q', 'qn', 'k', 'kn', 'v', 'o', 'ln2', 'gate', 'up', 'down', 'norm'];
export const ADAM_SUBS = ['g', 'm', 'v', 'bc', 'dw', 'write', 'bits'];

const DUR = { run: 7.5, end: 6, batch: 4.2, fwd: 6.5, loss: 3.6, bwd: 6.5, upd: 4.6, tpl: 3.2, mask: 3.4, shift: 2.6, emb: 2.2, pass: 3, op: 2.3, head: 2.6, pos: 1.7, mean: 2.4, clip: 2.8, adam: 3.2, bf16: 4.2, ten: 2.6, wsub: 2.5, wg: 2.8 };

export function fwdOps(L, NL) {
  const out = ['emb'];
  if (L > 0) out.push('lo');
  out.push(...LAYER_OPS);
  if (L < NL - 1) out.push('hi');
  out.push('head');
  return out;
}
export function bwdOps(L, NL) { return fwdOps(L, NL).slice().reverse(); }

export function qBuild(depth, k, R, L) {
  const NL = R.NL, K = R.K;
  if (depth <= 1) return k === K - 1 ? [{ k, ph: 'run' }, { k, ph: 'end' }] : [{ k, ph: 'run' }];
  if (depth === 2) return Q_PHASES.map((ph) => ({ k, ph }));
  const out = [];
  for (const sub of BATCH_SUBS) out.push({ k, ph: 'batch', sub });
  const ops = (ph, list, TS) => {
    for (const op of list) {
      const ts = TS[op] || [];
      if (depth >= 4 && ts.length) for (const t of ts) out.push(depth >= 5 && ph === 'bwd' ? { k, ph, sub: op, t, mi: 'g' } : { k, ph, sub: op, t });
      else out.push({ k, ph, sub: op });
    }
  };
  ops('fwd', fwdOps(L, NL), OP_TENSORS);
  for (const i of R.sftPos) out.push({ k, ph: 'loss', sub: 'pos', i });
  out.push({ k, ph: 'loss', sub: 'mean' });
  ops('bwd', bwdOps(L, NL), OP_TENSORS_BWD);
  out.push({ k, ph: 'upd', sub: 'clip' });
  if (depth >= 4) {
    for (const t of UPD_TENSORS) {
      if (depth >= 5) for (const mi of ADAM_SUBS) out.push({ k, ph: 'upd', sub: 'adam', t, mi });
      else out.push({ k, ph: 'upd', sub: 'adam', t });
    }
  } else out.push({ k, ph: 'upd', sub: 'adam' });
  out.push({ k, ph: 'upd', sub: 'bf16' });
  return out;
}

export function qIsPrefix(a, b) {
  if (a.k !== b.k) return false;
  if (a.ph === 'run') return b.ph !== 'end';
  if (a.ph === 'end' || b.ph === 'end') return a.ph === b.ph;
  if (b.ph === 'run') return false;
  if (a.ph !== b.ph) return false;
  for (const key of ['sub', 'i', 't', 'mi']) if (a[key] !== undefined && a[key] !== b[key]) return false;
  return true;
}
const same = (a, b) => a && b && qIsPrefix(a, b) && qIsPrefix(b, a);

export function qView(depth, s) {
  if (depth <= 1) return s.ph === 'end' ? 'q-end' : 'q-run';
  if (depth === 2) return 'q-step';
  if (s.mi) return 'q-param';
  if (depth >= 4 && s.t) return 'q-mat';
  return 'q-op';
}

function durOf(s) {
  if (s.ph === 'run') return DUR.run;
  if (s.ph === 'end') return DUR.end;
  if (!s.sub) return DUR[s.ph];
  if (s.mi) return s.mi === 'g' && s.ph === 'bwd' ? DUR.wg : DUR.wsub;
  if (s.t) return DUR.ten;
  if (s.ph === 'batch') return DUR[s.sub];
  if (s.ph === 'loss') return s.sub === 'mean' ? DUR.mean : DUR.pos;
  if (s.ph === 'upd') return DUR[s.sub];
  if (s.sub === 'lo' || s.sub === 'hi') return DUR.pass;
  if (s.sub === 'emb') return DUR.emb;
  if (s.sub === 'head') return DUR.head;
  return DUR.op;
}

export class QwenTimeline {
  constructor(R, L = R.NL - 1) {
    this.R = R;
    this.K = R.K;
    this.L = L;
    this.maxDepth = Q_MAX_DEPTH;
    this.minDepth = Q_MIN_DEPTH;
    this.depthNames = Q_DEPTH_NAMES;
    this.depth = 1;
    this.k = 0;
    this.i = 0;
    this.p = 0;
    this.playing = false;
    this.speed = 1;
    this.done = false;
    this.list = this.build(1, 0);
    this.listeners = new Set();
  }

  build(depth, k) { return qBuild(depth, k, this.R, this.L); }
  get step() { return this.list[this.i]; }
  get view() { return qView(this.depth, this.step); }
  get dur() { return durOf(this.step); }
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(type) { for (const fn of this.listeners) fn(type, this); }

  canInto() {
    if (this.depth >= Q_MAX_DEPTH) return false;
    const s = this.step;
    const deeper = this.build(this.depth + 1, this.k).find((b) => qIsPrefix(s, b));
    return !!deeper && (!same(deeper, s) || qView(this.depth + 1, s) !== qView(this.depth, s));
  }

  setDepth(d) {
    d = Math.max(Q_MIN_DEPTH, Math.min(Q_MAX_DEPTH, d));
    if (d === this.depth) return;
    const s = this.step;
    const list = this.build(d, this.k);
    let idx;
    if (d > this.depth) idx = list.findIndex((b) => qIsPrefix(s, b));
    else idx = list.findIndex((a) => qIsPrefix(a, s));
    this.depth = d;
    this.list = list;
    this.i = Math.max(0, idx);
    this.p = 0;
    this.done = false;
    this.emit('step');
  }

  // 换选中的层：同一深度、同一个环节（换层前后都有的算子停在同一个算子上）
  setLayer(L) {
    L = Math.max(0, Math.min(this.R.NL - 1, L));
    if (L === this.L) return;
    const s = this.step;
    this.L = L;
    this.list = this.build(this.depth, this.k);
    const i = this.list.findIndex((b) => b.ph === s.ph && b.sub === s.sub && b.i === s.i && b.t === s.t && b.mi === s.mi);
    this.i = i >= 0 ? i : Math.max(0, this.list.findIndex((b) => b.ph === s.ph));
    this.p = 0;
    this.done = false;
    this.emit('step');
  }

  // 直接跳到某一深度的某一步
  jump(d, pred) {
    this.depth = d;
    this.list = this.build(d, this.k);
    this.i = Math.max(0, this.list.findIndex(pred));
    this.p = 0;
    this.done = false;
    this.emit('step');
  }

  into() { if (this.canInto()) this.setDepth(this.depth + 1); }

  advance() {
    if (this.i < this.list.length - 1) { this.i++; this.p = 0; this.emit('step'); return true; }
    if (this.k < this.K - 1) {
      this.k++;
      this.list = this.build(this.depth, this.k);
      this.i = 0;
      this.p = 0;
      this.emit('ck');
      this.emit('step');
      return true;
    }
    this.p = 1;
    if (!this.done) { this.done = true; this.emit('end'); }
    return false;
  }

  back() {
    this.done = false;
    if (this.p > 0.15) { this.p = 0; this.emit('step'); return; }
    if (this.i > 0) { this.i--; this.p = 0; this.emit('step'); return; }
    if (this.k > 0) {
      this.k--;
      this.list = this.build(this.depth, this.k);
      this.i = this.list.length - 1;
      this.p = 0;
      this.emit('ck');
      this.emit('step');
    }
  }

  next() { this.done = false; if (this.p < 1 && this.playing) this.p = 0; this.advance(); }

  seekIndex(i) { this.i = Math.max(0, Math.min(this.list.length - 1, i)); this.p = 0; this.done = false; this.emit('step'); }
  seek(pred) { const i = this.list.findIndex(pred); if (i >= 0) this.seekIndex(i); }

  // D1 的轨道：0..K−1 = 三步，K = 结果那一拍
  seekCk(k, keepPos = true) {
    if (this.depth === 1 && k === this.K) { this.k = this.K - 1; this.list = this.build(1, this.k); this.i = this.list.length - 1; this.p = 0; this.done = false; this.emit('ck'); this.emit('step'); return; }
    k = Math.max(0, Math.min(this.K - 1, k));
    if (k === this.k && !(this.depth === 1 && this.step.ph === 'end')) return;
    const s = this.step;
    this.k = k;
    this.list = this.build(this.depth, k);
    this.i = keepPos && s && this.depth > 1 ? Math.max(0, this.list.findIndex((b) => b.ph === s.ph && b.sub === s.sub && b.i === s.i && b.t === s.t && b.mi === s.mi)) : 0;
    this.p = 0;
    this.done = false;
    this.emit('ck');
    this.emit('step');
  }

  play() { if (this.done) return; this.playing = true; this.emit('play'); }
  pause() { this.playing = false; this.emit('play'); }
  toggle() { this.playing ? this.pause() : this.play(); }

  tick(dt) {
    if (!this.playing) return;
    this.p += (dt * this.speed) / this.dur;
    let guard = 0;
    while (this.p >= 1 && guard++ < 50) {
      const carry = this.p - 1;
      if (!this.advance()) { this.playing = false; this.emit('play'); break; }
      this.p = Math.min(carry, 0.99);
    }
  }
}
