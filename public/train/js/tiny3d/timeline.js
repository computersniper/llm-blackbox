// 唐宋诗小模型这一章的“步骤树”（接口和 ../timeline.js 的 Timeline 一样，页面和控制条不用关心是哪一章）。
//
//   D1 训练全程    41 个检查点，每个一拍：进料 → 前向 → 损失 → 反向 → 更新（延时摄影）
//   D2 一步之内    ① 取批次 ② 前向 ③ 损失 ④ 反向 ⑤ 更新（走完自动进入下一个检查点）
//   D3 一层之内    批次：取一行 / 编号 / 错开一位；前向、反向：嵌入、每层的注意力一半和前馈一半、输出头；
//                  损失：64 个位置、平均；更新：裁剪、AdamW
//   D4 一块权重    三块有真实局部的矩阵（嵌入、第 2 层 W_q、第 4 层 W_down）放大成 48 × 48 个方块；损失拆到每个位置
//   D5 一个权重    数据里跟踪的 4 个权重：前向时它的值、反向时它的梯度、更新时完整的 AdamW 算式
import { isEn } from '../lang.js';

export const T3_MAX = 5, T3_MIN = 1;
export const T3_NAMES = isEn ? ['', 'Whole run', 'One step', 'One layer', 'One weight block', 'One weight'] : ['', '训练全程', '一步之内', '一层之内', '一块权重', '一个权重'];
export const PHASES = ['batch', 'fwd', 'loss', 'bwd', 'upd'];
export const NL = 6;
// 前向的 14 个算子：嵌入、每层的注意力一半 / 前馈一半、输出头；反向倒过来
export const FWD = [{ sub: 'emb' }];
for (let l = 0; l < NL; l++) FWD.push({ sub: 'attn', L: l }, { sub: 'ffn', L: l });
FWD.push({ sub: 'head' });
export const BWD = FWD.slice().reverse();
export const opKey = (o) => `${o.sub}${o.L ?? ''}`;
export const opIndex = (list, s) => list.findIndex((o) => o.sub === s.sub && o.L === s.L);
// 有真实局部的算子（c = 局部编号）；有跟踪权重的算子（f = 权重编号，按在算子里出现的先后）
export const CROP_AT = { emb: 0, attn2: 1, ffn4: 2, head: 0 };
export const FEAT_AT = { emb: [0], attn1: [3, 1], ffn5: [2] };
export const FEAT_CROP = [0, 1, 2, undefined];
export const ADAM_SUBS = ['g', 'm', 'v', 'bc', 'dw', 'write'];
const NPOS = 64;

const DUR = { run: 3.8, batch: 3.2, fwd: 7, loss: 3, bwd: 7, upd: 4.2, sub: 2.4, op: 2.2, nll: 3, mean: 2, clip: 2.4, adam: 3.6, crop: 3.4, pos: 1.0, w: 3.6, sub5: 2.6 };

export function t3Build(depth, k) {
  if (depth <= 1) return [{ k, ph: 'run' }];
  if (depth === 2) return PHASES.map((ph) => ({ k, ph }));
  const out = [];
  for (const sub of ['row', 'ids', 'shift']) out.push({ k, ph: 'batch', sub });
  const ops = (ph, list, mi) => {
    for (const op of list) {
      const key = opKey(op), c = CROP_AT[key], fs = FEAT_AT[key];
      if (depth >= 5 && fs) for (const f of fs) out.push({ k, ph, ...op, c, f, mi });
      else if (depth >= 4 && c != null) out.push({ k, ph, ...op, c });
      else out.push({ k, ph, ...op });
    }
  };
  ops('fwd', FWD, 'w');
  if (depth >= 4) for (let i = 0; i < NPOS; i++) out.push({ k, ph: 'loss', sub: 'pos', i });
  else out.push({ k, ph: 'loss', sub: 'pos' });
  out.push({ k, ph: 'loss', sub: 'mean' });
  ops('bwd', BWD, 'g');
  out.push({ k, ph: 'upd', sub: 'clip' });
  if (depth >= 5) for (let f = 0; f < 4; f++) for (const mi of ADAM_SUBS) out.push({ k, ph: 'upd', sub: 'adam', c: FEAT_CROP[f], f, mi });
  else if (depth === 4) for (let c = 0; c < 3; c++) out.push({ k, ph: 'upd', sub: 'adam', c });
  else out.push({ k, ph: 'upd', sub: 'adam' });
  return out;
}

// a 是不是 b 的“祖先或自身”：a 写了的每一项 b 都要一样
export function t3IsPrefix(a, b) {
  if (a.k !== b.k) return false;
  if (a.ph === 'run') return true;
  if (a.ph !== b.ph) return false;
  for (const key of ['sub', 'L', 'c', 'i', 'f', 'mi']) if (a[key] !== undefined && a[key] !== b[key]) return false;
  return true;
}
const same = (a, b) => a && b && t3IsPrefix(a, b) && t3IsPrefix(b, a);

// 视图：决定镜头怎么取景
export function t3View(depth, s) {
  if (depth <= 1) return 't-run';
  if (depth === 2) return 't-step';
  if (depth >= 5 && s.f != null) return 't-param';
  if (depth >= 4 && s.c != null) return 't-crop';
  if (depth >= 4 && s.ph === 'loss' && s.i != null) return 't-pos';
  return 't-op';
}

function durOf(s) {
  if (s.ph === 'run') return DUR.run;
  if (!s.sub) return DUR[s.ph];
  if (s.mi === 'w' || s.mi === 'g') return DUR.w;
  if (s.mi) return DUR.sub5;
  if (s.ph === 'batch') return DUR.sub;
  if (s.ph === 'loss') return s.sub === 'mean' ? DUR.mean : s.i != null ? DUR.pos : DUR.nll;
  if (s.ph === 'upd') return s.sub === 'clip' ? DUR.clip : s.c != null ? DUR.crop : DUR.adam;
  return s.c != null ? DUR.crop : DUR.op;
}

export class TinyTimeline {
  constructor(X) {
    this.X = X;
    this.R = X;
    this.K = X.K;
    this.maxDepth = T3_MAX;
    this.minDepth = T3_MIN;
    this.depthNames = T3_NAMES;
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

  build(depth, k) { return t3Build(depth, k); }
  get step() { return this.list[this.i]; }
  get view() { return t3View(this.depth, this.step); }
  get dur() { return durOf(this.step); }
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(type) { for (const fn of this.listeners) fn(type, this); }

  // 往下最近的、真的有新东西的一层（第 1 层的注意力在 D4 没有局部，但 D5 有跟踪的权重：直接跳过 D4）
  intoDepth() {
    const s = this.step;
    for (let d = this.depth + 1; d <= T3_MAX; d++) {
      const deeper = this.build(d, this.k).find((b) => t3IsPrefix(s, b));
      if (!deeper) return -1;
      if (!same(deeper, s) || t3View(d, deeper) !== t3View(this.depth, s)) return d;
    }
    return -1;
  }
  canInto() { return this.depth < T3_MAX && this.intoDepth() > 0; }

  setDepth(d) {
    d = Math.max(T3_MIN, Math.min(T3_MAX, d));
    if (d === this.depth) return;
    const s = this.step;
    const list = this.build(d, this.k);
    let idx = d > this.depth ? list.findIndex((b) => t3IsPrefix(s, b)) : list.findIndex((a) => t3IsPrefix(a, s));
    // 往外退时找不到祖先（D5 的 γ 在 D4 没有对应的局部）：停在同一环节的第一步
    if (idx < 0) idx = list.findIndex((b) => b.ph === s.ph && b.sub === s.sub);
    if (idx < 0) idx = list.findIndex((b) => b.ph === s.ph);
    this.depth = d;
    this.list = list;
    this.i = Math.max(0, idx);
    this.p = 0;
    this.done = false;
    this.emit('step');
  }

  jump(d, pred) {
    this.depth = d;
    this.list = this.build(d, this.k);
    this.i = Math.max(0, this.list.findIndex(pred));
    this.p = 0;
    this.done = false;
    this.emit('step');
  }

  into() { const d = this.intoDepth(); if (d > 0) this.setDepth(d); }

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

  seekCk(k, keepPos = true) {
    k = Math.max(0, Math.min(this.K - 1, k));
    if (k === this.k) return;
    const s = this.step;
    this.k = k;
    this.list = this.build(this.depth, k);
    this.i = keepPos && s ? Math.max(0, this.list.findIndex((b) => same({ ...b, k: 0 }, { ...s, k: 0 }))) : 0;
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
