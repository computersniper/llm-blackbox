// 玻璃小模型这一章的“步骤树”（接口和 ../timeline.js 的 Timeline 一样，页面和控制条不用关心是哪一章）。
//
//   D0 初始化    整个模型 → 初始值从哪来（直方图）→ 为什么要随机（全零对照）→ 为什么要小（放大 50 倍对照）
//   D1 训练全程  每一帧一步（前 50 步每步一帧，之后每 5 / 10 步一帧）：全部权重、损失、固定样例的预测、注意力、嵌入相似度
//   D2 一步之内  ① 取批次 → ② 前向 → ③ 损失 → ④ 反向 → ⑤ 更新（走完自动进入下一帧）
//   D3 拆开环节  批次：取 8 段 / 错开一位；前向、反向：逐个算子；损失：逐个位置；更新：先裁剪，再一个张量一个张量地更新
//   D4 一个参数  反向：一个权重的梯度怎么由“输入 × 上游梯度”加出来；更新：一个权重的 AdamW 算式（真实数字）
import { isEn } from '../lang.js';

export const G_MAX_DEPTH = 4;
export const G_DEPTH_NAMES = isEn ? ['Initialization', 'Whole run', 'One step', 'Each operation', 'One parameter'] : ['初始化', '训练全程', '一步之内', '拆开环节', '一个参数'];
export const INIT_SUBS = ['model', 'hist', 'zero', 'big'];
export const G_PHASES = ['batch', 'fwd', 'loss', 'bwd', 'upd'];
export const FWD_OPS = ['emb', 'norm1', 'qkv', 'attn', 'wo', 'norm2', 'ffn', 'wd', 'normf', 'logits'];
// 反向：顺序和前向相反；每个算子产生哪些参数的梯度
export const BWD_OPS = ['logits', 'normf', 'wd', 'ffn', 'norm2', 'wo', 'attn', 'qkv', 'norm1', 'emb'];
export const BWD_TENSORS = { logits: ['E'], normf: ['gf'], wd: ['Wd'], ffn: ['Wg', 'Wu'], norm2: ['g2'], wo: ['Wo'], attn: [], qkv: ['Wq', 'Wk', 'Wv'], norm1: ['g1'], emb: ['E'] };
export const UPD_TENSORS = ['E', 'g1', 'Wq', 'Wk', 'Wv', 'Wo', 'g2', 'Wg', 'Wu', 'Wd', 'gf'];
export const ADAM_SUBS = ['g', 'm', 'v', 'bc', 'dw', 'write'];

const DUR = { init: 7, run: 0.9, phase: 4.2, batch: 2.6, fwd: 1.5, loss: 1.0, bwd: 1.6, upd: 1.5, clip: 2.2, chain: 4.2, adam: 2.4 };

export function gBuild(depth, k, D) {
  if (depth <= 0) return INIT_SUBS.map((sub) => ({ ph: 'init', sub }));
  if (depth === 1) return [{ k, ph: 'run' }];
  if (depth === 2) return G_PHASES.map((ph) => ({ k, ph }));
  const out = [];
  out.push({ k, ph: 'batch', sub: 'pick' }, { k, ph: 'batch', sub: 'shift' });
  for (const op of FWD_OPS) out.push({ k, ph: 'fwd', sub: op });
  for (let i = 0; i < D.T; i++) out.push({ k, ph: 'loss', sub: 'pos', i });
  out.push({ k, ph: 'loss', sub: 'mean' });
  for (const op of BWD_OPS) {
    const ts = BWD_TENSORS[op];
    if (depth >= 4 && ts.length) for (const t of ts) out.push({ k, ph: 'bwd', sub: op, t, mi: 'chain' });
    else out.push({ k, ph: 'bwd', sub: op });
  }
  out.push({ k, ph: 'upd', sub: 'clip' });
  for (const t of UPD_TENSORS) {
    if (depth >= 4) for (const mi of ADAM_SUBS) out.push({ k, ph: 'upd', sub: 't', t, mi });
    else out.push({ k, ph: 'upd', sub: 't', t });
  }
  return out;
}

export function gIsPrefix(a, b) {
  if (a.ph === 'init') return b.ph === 'init' && a.sub === b.sub;
  if (b.ph === 'init') return false;
  if (a.k !== b.k) return false;
  if (a.ph === 'run') return true;
  if (a.ph !== b.ph) return false;
  // a 写了的每一项 b 都要一样（i 只有损失用、t / mi 只有反向和更新用，不能遇到第一个空项就停）
  for (const key of ['sub', 'i', 't', 'mi']) if (a[key] !== undefined && a[key] !== b[key]) return false;
  return true;
}
const same = (a, b) => a && b && gIsPrefix(a, b) && gIsPrefix(b, a);

export function gView(depth, s) {
  if (depth <= 0) return 'g-init';
  if (depth === 1) return 'g-run';
  if (s.mi) return 'g-param';
  return 'g-step';
}

function durOf(s, depth) {
  if (s.ph === 'init') return DUR.init;
  if (s.ph === 'run') return DUR.run;
  if (!s.sub) return s.ph === 'batch' ? DUR.batch + 0.6 : DUR.phase;
  if (s.mi === 'chain') return DUR.chain;
  if (s.mi) return DUR.adam;
  if (s.ph === 'upd') return s.sub === 'clip' ? DUR.clip : DUR.upd;
  return DUR[s.ph] || DUR.phase;
}

export class GlassTimeline {
  constructor(R) {
    this.R = R;
    this.K = R.K;
    this.maxDepth = G_MAX_DEPTH;
    this.depthNames = G_DEPTH_NAMES;
    this.depth = 0;
    this.k = 0;
    this.i = 0;
    this.p = 0;
    this.playing = false;
    this.speed = 1;
    this.done = false;
    this.list = this.build(0, 0);
    this.listeners = new Set();
  }

  build(depth, k) { return gBuild(depth, k, this.R.D); }
  get step() { return this.list[this.i]; }
  get view() { return gView(this.depth, this.step); }
  get dur() { return durOf(this.step, this.depth); }
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(type) { for (const fn of this.listeners) fn(type, this); }

  canInto() {
    if (this.depth >= G_MAX_DEPTH) return false;
    if (this.depth === 0) return true;
    const s = this.step;
    const deeper = this.build(this.depth + 1, this.k).find((b) => gIsPrefix(s, b));
    return !!deeper && (!same(deeper, s) || gView(this.depth + 1, s) !== gView(this.depth, s));
  }

  setDepth(d) {
    d = Math.max(0, Math.min(G_MAX_DEPTH, d));
    if (d === this.depth) return;
    const s = this.step;
    const list = this.build(d, this.k);
    let idx;
    if (this.depth === 0 || d === 0) idx = 0;
    else if (d > this.depth) idx = list.findIndex((b) => gIsPrefix(s, b));
    else idx = list.findIndex((a) => gIsPrefix(a, s));
    this.depth = d;
    this.list = list;
    this.i = Math.max(0, idx);
    this.p = 0;
    this.done = false;
    this.emit('step');
  }

  // 直接跳到某一深度的某一步（点格子看“一个参数”时用）
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
    if (this.depth > 0 && this.k < this.K - 1) {
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
    if (this.depth > 0 && this.k > 0) {
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
    if (k === this.k || this.depth === 0) return;
    const s = this.step;
    this.k = k;
    this.list = this.build(this.depth, k);
    this.i = keepPos && s ? Math.max(0, this.list.findIndex((b) => b.ph === s.ph && b.sub === s.sub && b.i === s.i && b.t === s.t && b.mi === s.mi)) : 0;
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
