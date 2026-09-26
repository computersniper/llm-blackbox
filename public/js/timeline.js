// 调试器的核心：一次生成被拆成一棵“步骤树”，深度越大，步子越细。
//
//   d1 黑箱    每个词元一步：读入 → 算 → 吐出
//   d2 结构    读入 / 嵌入 / 28 层 / 输出头 / 采样
//   d3 层塔    28 层逐层
//   d4 一层内  RMSNorm / 注意力 / 残差 / RMSNorm / 前馈 / 残差
//   d5 算子    注意力：QKV / 打分 / softmax / 加权求和；前馈：升维 / 门控激活 / 降维
//   d6 乘加    单次打分 Q·K、单个 SwiGLU 神经元（只在导出了细节的焦点层）
//   d7 比特    权重本身的 bf16 比特
//
// “＋”= 单步进入（step into），“−”= 跳出（step out），上一步 / 下一步 = 在当前深度逐步执行。

export const OPS = ['ln1', 'attn', 'add1', 'ln2', 'mlp', 'add2'];
const SUBS = { attn: ['qkv', 'score', 'softmax', 'mix'], mlp: ['up', 'act', 'down'] };
const MICROS = { score: ['mul', 'sum', 'scale'], act: ['mul', 'sum', 'silu', 'gate'] };
export const MAX_DEPTH = 7;

export const DEPTH_NAMES = ['对话', '黑箱', '结构', '层塔', '一层之内', '算子', '一次乘加', '比特'];

const DUR = {
  pass: 1.6, read0: 2.4, read: 1.1, embed: 1.3, layers: 2.6, layer: 0.62, head: 1.5, sample: 2.4,
  ln1: 0.8, attn: 1.8, add1: 0.8, ln2: 0.8, mlp: 1.8, add2: 0.8,
  qkv: 1.6, score: 1.6, softmax: 1.5, mix: 1.7, up: 1.5, act: 1.9, down: 1.5,
  mul: 2.2, sum: 1.7, scale: 1.3, silu: 1.9, gate: 1.7,
  norm: 0.9, unembed: 1.5, softmaxH: 1.5, temp: 1.3, topk: 1.3, topp: 1.5, draw: 2.4,
};

export function buildSteps(depth, g, NL, focus) {
  if (depth <= 1) return [{ g, ph: 'pass' }];
  const s = [{ g, ph: 'read' }, { g, ph: 'embed' }];
  if (depth === 2) s.push({ g, ph: 'layers' });
  else {
    for (let L = 0; L < NL; L++) {
      if (depth === 3) { s.push({ g, ph: 'layer', L }); continue; }
      for (const op of OPS) {
        if (depth === 4 || !SUBS[op]) { s.push({ g, ph: 'layer', L, op }); continue; }
        for (const sub of SUBS[op]) {
          const micros = depth >= 6 && focus.includes(L) ? MICROS[sub] : null;
          if (micros) for (const mi of micros) s.push({ g, ph: 'layer', L, op, sub, mi });
          else s.push({ g, ph: 'layer', L, op, sub });
        }
      }
    }
  }
  if (depth <= 3) s.push({ g, ph: 'head' }, { g, ph: 'sample' });
  else {
    for (const sub of ['norm', 'unembed', 'softmax']) s.push({ g, ph: 'head', sub });
    for (const sub of ['temp', 'topk', 'topp', 'draw']) s.push({ g, ph: 'sample', sub });
  }
  return s;
}

// a 是否是 b 的“祖先或自身”
export function isPrefix(a, b) {
  if (a.g !== b.g) return false;
  if (a.ph === 'pass') return true;
  if (a.ph === 'layers') return b.ph === 'layer' || b.ph === 'layers';
  if (a.ph !== b.ph) return false;
  for (const k of ['L', 'op', 'sub', 'mi']) {
    if (a[k] === undefined) return true;
    if (a[k] !== b[k]) return false;
  }
  return true;
}

export const sameStep = (a, b) => a && b && isPrefix(a, b) && isPrefix(b, a);

// 相机与场景使用的“视图”
export function viewOf(depth, s) {
  if (depth <= 1) return 'box';
  if (depth === 2) return 'machine';
  if (s.ph === 'read' || s.ph === 'embed') return 'tray';
  if (s.ph === 'head' || s.ph === 'sample') return 'head';
  if (depth === 3) return 'tower';
  if (depth === 4 || !s.sub) return 'layer';
  if (depth === 5 || !s.mi) return s.op === 'attn' ? 'attn' : 'mlp';
  if (depth === 6 || s.mi !== 'mul') return s.op === 'attn' ? 'dot' : 'neuron';
  return 'bits';
}

function durOf(s) {
  if (s.ph === 'read') return s.g === 0 ? DUR.read0 : DUR.read;
  if (s.ph === 'layer') return DUR[s.mi || s.sub || s.op || 'layer'];
  if (s.ph === 'head') return s.sub ? DUR[s.sub === 'softmax' ? 'softmaxH' : s.sub] : DUR.head;
  if (s.ph === 'sample') return s.sub ? DUR[s.sub] : DUR.sample;
  return DUR[s.ph];
}

export class Timeline {
  constructor(Q, focus) {
    this.Q = Q;
    this.NL = Q.NL;
    this.G = Q.G;
    this.focus = focus;
    this.depth = 1;
    this.g = 0;
    this.i = 0;
    this.p = 0;
    this.playing = false;
    this.speed = 1;
    this.list = this.build(1, 0);
    this.listeners = new Set();
    this.done = false;
  }

  build(depth, g) { return buildSteps(depth, g, this.NL, this.focus); }
  get step() { return this.list[this.i]; }
  get view() { return viewOf(this.depth, this.step); }
  get dur() { return durOf(this.step); }
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(type) { for (const fn of this.listeners) fn(type, this); }

  canInto() {
    if (this.depth >= MAX_DEPTH) return false;
    const s = this.step;
    const deeper = this.build(this.depth + 1, this.g).find((b) => isPrefix(s, b));
    return !!deeper && (!sameStep(deeper, s) || viewOf(this.depth + 1, s) !== viewOf(this.depth, s));
  }

  setDepth(d) {
    d = Math.max(1, Math.min(MAX_DEPTH, d));
    if (d === this.depth) return;
    const s = this.step;
    const list = this.build(d, this.g);
    let idx;
    if (d > this.depth) idx = list.findIndex((b) => isPrefix(s, b));
    else idx = list.findIndex((a) => isPrefix(a, s));
    this.depth = d;
    this.list = list;
    this.i = Math.max(0, idx);
    this.p = 0;
    this.emit('step');
  }

  into() { if (this.canInto()) this.setDepth(this.depth + 1); }
  out() { if (this.depth > 1) this.setDepth(this.depth - 1); }

  // 前进一步；跨过最后一步 = 吐出第 g 个词元，进入下一个
  advance() {
    if (this.i < this.list.length - 1) { this.i++; this.p = 0; this.emit('step'); return true; }
    if (this.g < this.G - 1) {
      this.g++;
      this.list = this.build(this.depth, this.g);
      this.i = 0;
      this.p = 0;
      this.emit('token');
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
    if (this.g > 0) {
      this.g--;
      this.list = this.build(this.depth, this.g);
      this.i = this.list.length - 1;
      this.p = 0;
      this.emit('token');
      this.emit('step');
    }
  }

  next() { this.done = false; if (this.p < 1 && this.playing) this.p = 0; this.advance(); }

  seekIndex(i) { this.i = Math.max(0, Math.min(this.list.length - 1, i)); this.p = 0; this.done = false; this.emit('step'); }

  seek(pred) {
    const i = this.list.findIndex(pred);
    if (i >= 0) this.seekIndex(i);
  }

  seekToken(g) {
    g = Math.max(0, Math.min(this.G - 1, g));
    if (g === this.g) return;
    this.g = g;
    this.list = this.build(this.depth, g);
    this.i = 0;
    this.p = 0;
    this.done = false;
    this.emit('token');
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
