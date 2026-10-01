// 调试器的核心：一段训练被拆成一棵“步骤树”，深度越大，步子越细。
//
//   d0 流水线    预训练 → 监督微调 → 偏好对齐，每个阶段一步
//   d1 训练全程  每个检查点一步（小模型 41 个检查点；Qwen3 是 3 步 SFT）
//   d2 一步之内  批次 → 前向 → 损失 → 反向 → 更新 →（演示）重新前向
//   d3 拆开环节  批次：取一行 / 编号 / 错开一位；前向、反向：逐层；损失：逐个位置；更新：AdamW 的每一项
//   d4 一个数    一个位置的交叉熵；一层里 7 个矩阵的梯度；一个权重的 fp32 / bf16 比特
//
// “＋”= 单步进入（step into），“−”= 跳出（step out），上一步 / 下一步 = 在当前深度逐步执行。

export const MAX_DEPTH = 4;
export const DEPTH_NAMES = ['流水线', '训练全程', '一步之内', '拆开环节', '细到一个数'];
export const PHASES = ['batch', 'fwd', 'loss', 'bwd', 'upd', 'check'];
export const UPD_SUBS = ['clip', 'g', 'm', 'v', 'bc', 'dw', 'write'];
export const LAYER_TENSORS = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'];
export const CE_SUBS = ['softmax', 'pick', 'log'];
export const BITS_SUBS = ['fp32a', 'fp32b', 'bf16'];
const BATCH_SUBS = { tiny: ['row', 'ids', 'shift'], qwen: ['tpl', 'mask', 'shift'] };

const DUR = {
  pipe: 6, ck: { tiny: 2.2, qwen: 5.5 },
  phase: 2.6, batch: 2.8,
  fwd: { tiny: 1.1, qwen: 0.55 }, loss: { tiny: 0.42, qwen: 1.7 },
  upd: 2.6, ce: 2.0, layer: 1.0, bits: 2.8,
};

export function buildSteps(depth, k, R) {
  if (depth <= 0) return [0, 1, 2].map((st) => ({ ph: 'pipe', st }));
  if (depth === 1) return [{ k, ph: 'ck' }];
  const out = [];
  const NL = R.NL;
  for (const ph of PHASES) {
    if (depth === 2 || ph === 'check') { out.push({ k, ph }); continue; }
    if (ph === 'batch') for (const sub of BATCH_SUBS[R.kind]) out.push({ k, ph, sub });
    else if (ph === 'fwd') for (let i = 0; i <= NL; i++) out.push({ k, ph, sub: 'b', i });
    else if (ph === 'bwd') {
      for (let i = NL; i >= 0; i--) {
        if (depth >= 4 && i >= 1) for (const mi of LAYER_TENSORS) out.push({ k, ph, sub: 'b', i, mi });
        else out.push({ k, ph, sub: 'b', i });
      }
    } else if (ph === 'loss') {
      const pos = R.kind === 'tiny' ? Array.from({ length: R.R }, (_, i) => i) : R.sftPos;
      for (const i of pos) {
        if (depth >= 4) for (const mi of CE_SUBS) out.push({ k, ph, sub: 'pos', i, mi });
        else out.push({ k, ph, sub: 'pos', i });
      }
    } else if (ph === 'upd') {
      for (const sub of UPD_SUBS) {
        if (depth >= 4 && sub === 'write') for (const mi of BITS_SUBS) out.push({ k, ph, sub, mi });
        else out.push({ k, ph, sub });
      }
    }
  }
  return out;
}

// a 是否是 b 的“祖先或自身”
export function isPrefix(a, b) {
  if (a.ph === 'pipe') return b.ph === 'pipe' && a.st === b.st;
  if (b.ph === 'pipe') return false;
  if (a.k !== b.k) return false;
  if (a.ph === 'ck') return true;
  if (a.ph !== b.ph) return false;
  for (const key of ['sub', 'i', 'mi']) {
    if (a[key] === undefined) return true;
    if (a[key] !== b[key]) return false;
  }
  return true;
}

export const sameStep = (a, b) => a && b && isPrefix(a, b) && isPrefix(b, a);

// 舞台上的“视图”
export function viewOf(depth, s, kind) {
  if (depth <= 0) return 'pipeline';
  if (depth === 1) return kind === 'tiny' ? 'overview' : 'overviewQ';
  if (depth === 2 || !s.sub) return 'loop';
  switch (s.ph) {
    case 'batch': return 'batch';
    case 'fwd': return 'fwd';
    case 'loss': return s.mi ? 'ce' : 'loss';
    case 'bwd': return s.mi ? 'layer' : 'bwd';
    case 'upd': return s.mi ? 'bits' : 'adam';
  }
  return 'loop';
}

function durOf(s, kind) {
  switch (s.ph) {
    case 'pipe': return DUR.pipe;
    case 'ck': return DUR.ck[kind];
    case 'batch': return s.sub ? DUR.batch : DUR.phase;
    case 'fwd': return s.sub ? DUR.fwd[kind] : DUR.phase;
    case 'bwd': return s.mi ? DUR.layer : s.sub ? DUR.fwd[kind] : DUR.phase;
    case 'loss': return s.mi ? DUR.ce : s.sub ? DUR.loss[kind] : DUR.phase;
    case 'upd': return s.mi ? DUR.bits : s.sub ? DUR.upd : DUR.phase;
  }
  return DUR.phase;
}

export class Timeline {
  constructor(R) {
    this.R = R;
    this.K = R.K;
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

  build(depth, k) { return buildSteps(depth, k, this.R); }
  get step() { return this.list[this.i]; }
  get view() { return viewOf(this.depth, this.step, this.R.kind); }
  get dur() { return durOf(this.step, this.R.kind); }
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(type) { for (const fn of this.listeners) fn(type, this); }

  canInto() {
    if (this.depth >= MAX_DEPTH) return false;
    if (this.depth === 0) return this.step.st < 2;
    const s = this.step;
    const deeper = this.build(this.depth + 1, this.k).find((b) => isPrefix(s, b));
    return !!deeper && (!sameStep(deeper, s) || viewOf(this.depth + 1, s, this.R.kind) !== viewOf(this.depth, s, this.R.kind));
  }

  setDepth(d) {
    d = Math.max(1, Math.min(MAX_DEPTH, d));
    if (d === this.depth) return;
    const s = this.step;
    const list = this.build(d, this.k);
    let idx;
    if (this.depth === 0) idx = 0;
    else if (d > this.depth) idx = list.findIndex((b) => isPrefix(s, b));
    else idx = list.findIndex((a) => isPrefix(a, s));
    this.depth = d;
    this.list = list;
    this.i = Math.max(0, idx);
    this.p = 0;
    this.done = false;
    this.emit('step');
  }

  toPipeline(st) {
    this.depth = 0;
    this.list = this.build(0, 0);
    this.i = st;
    this.p = 0;
    this.done = false;
    this.emit('step');
  }

  into() { if (this.canInto()) this.setDepth(this.depth + 1); }

  // 前进一步；跨过最后一步 = 进入下一个检查点
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

  seek(pred) {
    const i = this.list.findIndex(pred);
    if (i >= 0) this.seekIndex(i);
  }

  seekCk(k, keepPos = true) {
    k = Math.max(0, Math.min(this.K - 1, k));
    if (k === this.k) return;
    const s = this.step;
    this.k = k;
    this.list = this.build(this.depth, k);
    // 换检查点时，尽量停在同一个环节
    this.i = keepPos && s ? Math.max(0, this.list.findIndex((b) => b.ph === s.ph && b.sub === s.sub && b.i === s.i && b.mi === s.mi)) : 0;
    this.p = 0;
    this.done = false;
    this.emit('ck');
    this.emit('step');
  }

  play() {
    if (this.done) return;
    this.playing = true;
    this.emit('play');
  }
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
