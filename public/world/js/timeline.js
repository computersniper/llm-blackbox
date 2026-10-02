// 调试器的核心：世界模型的“一帧”被拆成一棵步骤树，深度越大，步子越细。
//
//   D1 玩         每一帧一步：真实世界和梦一起往前走
//   D2 循环       看（真实画面）→ 编码 → 记忆与预测（M）→ 采样 → 解码（梦）→ 对照
//   D3 V 的内部   编码、解码拆成一层一层：卷积 1–4、μ / σ、z；全连接、反卷积 1–4
//   D4 M 的内部   M 拆开：拼输入、输入门 / 遗忘门 / 候选 / 输出门、细胞、隐状态、混合密度、撞车概率；采样拆成选分量、加噪声
//   D5 一次乘加   每一层再拆成：选一个输出 → 逐项相乘 → 加起来 + 偏置 → 激活
//
// V 的层在 D3 就展开，M 的在 D4 才展开：某一步在某个深度没有新东西时，＋ / − 会直接跳过那一层，
// 面包屑也只列出这一步真正经过的深度。“＋”= 单步进入，“−”= 跳出，上一步 / 下一步 = 在当前深度逐步执行。

import { L } from '../../js/i18n.js';

export const MAX_DEPTH = 5;
export const DEPTH_NAMES = L(['', '玩', '循环', 'V 的内部', 'M 的内部', '一次乘加'], ['', 'Play', 'Loop', 'Inside V', 'Inside M', 'One multiply-add']);
export const STAGES = ['obs', 'enc', 'rnn', 'sample', 'dec', 'cmp'];
export const ENC_OPS = ['e1', 'e2', 'e3', 'e4', 'mu', 'z'];
export const DEC_OPS = ['dfc', 'd1', 'd2', 'd3', 'd4'];
export const RNN_OPS = ['cat', 'gi', 'gf', 'gg', 'go', 'cell', 'hid', 'mdn', 'done'];
export const SAMPLE_OPS = ['pick', 'draw'];
// 有矩阵乘法 / 卷积、可以拆成一次乘加的算子
export const MAC_OPS = new Set(['e1', 'e2', 'e3', 'e4', 'mu', 'dfc', 'd1', 'd2', 'd3', 'd4', 'gi', 'gf', 'gg', 'go', 'mdn', 'done']);
export const MICROS = ['pick', 'mul', 'sum', 'act'];

const DUR = {
  frame: 1 / 20,          // D1：1× 是每秒 20 帧
  obs: 1.0, enc: 1.5, rnn: 1.8, sample: 1.3, dec: 1.5, cmp: 1.4, reset: 1.6,
  op: 1.4, z: 1.8, cell: 1.6, hid: 1.4, mdn: 2.0, done: 1.4, cat: 1.4, pick: 1.5, draw: 1.5,
  mi: { pick: 1.2, mul: 2.6, sum: 1.7, act: 1.3 },
};

// rec：这一帧的转移记录（sim.prepare() 算好的）。闭眼时编码器没有用到，这一帧的编码只占一个“跳过”的步骤
export function buildSteps(depth, rec) {
  if (depth <= 1) return [{ ph: 'frame' }];
  if (!rec || rec.kind === 'reset') return [{ ph: 'reset' }];
  const out = [];
  const usedEnc = rec.src !== 'dream';
  for (const ph of STAGES) {
    let ops = null;
    if (ph === 'enc' && depth >= 3 && usedEnc) ops = ENC_OPS;
    if (ph === 'dec' && depth >= 3) ops = DEC_OPS;
    if (ph === 'rnn' && depth >= 4) ops = RNN_OPS;
    if (ph === 'sample' && depth >= 4) ops = SAMPLE_OPS;
    if (!ops) { out.push({ ph }); continue; }
    for (const op of ops) {
      if (depth >= 5 && MAC_OPS.has(op)) for (const mi of MICROS) out.push({ ph, op, mi });
      else out.push({ ph, op });
    }
  }
  return out;
}

// a 是否是 b 的“祖先或自身”
export function isPrefix(a, b) {
  if (a.ph === 'frame') return true;
  if (a.ph !== b.ph) return false;
  if (a.op === undefined) return true;
  if (a.op !== b.op) return false;
  return a.mi === undefined || a.mi === b.mi;
}
export const sameStep = (a, b) => !!a && !!b && isPrefix(a, b) && isPrefix(b, a);

// 舞台上的视图
export function viewOf(depth, s) {
  if (depth <= 1 || s.ph === 'frame') return 'play';
  if (s.ph === 'reset' || !s.op) return 'loop';
  if (s.mi) return 'mac';
  return s.ph === 'enc' || s.ph === 'dec' ? 'vae' : 'mem';
}

function durOf(s) {
  if (s.mi) return DUR.mi[s.mi];
  if (s.op) return DUR[s.op] || DUR.op;
  return DUR[s.ph];
}

export class Timeline {
  constructor(sim) {
    this.sim = sim;
    this.depth = 1;
    this.i = 0;
    this.p = 0;
    this.playing = false;
    this.speed = 1;
    this.list = buildSteps(1, null);
    this.listeners = new Set();
  }

  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(type) { for (const fn of this.listeners) fn(type, this); }

  get step() { return this.list[this.i]; }
  get view() { return viewOf(this.depth, this.step); }
  get dur() { return durOf(this.step); }
  get rec() { return this.sim.cur.rec; }

  // 当前帧在这个深度的步骤表；深度 ≥ 2 时要先把这一帧的计算做完
  build(depth) {
    if (depth >= 2) this.sim.prepare();
    return buildSteps(depth, this.sim.cur.rec);
  }
  rebuild(keep = true) {
    const s = this.step;
    this.list = this.build(this.depth);
    let i = keep && s ? this.list.findIndex((b) => sameStep(s, b)) : 0;
    if (i < 0) i = keep && s ? Math.max(0, this.list.findIndex((b) => isPrefix(b, s) || isPrefix(s, b))) : 0;
    this.i = Math.max(0, i);
  }

  // 第 d 层里，和当前这一步对应的那一步（祖先或第一个后代）
  stepAt(d) {
    if (d === 1) return { ph: 'frame' };
    if (d === this.depth) return this.step;
    const s = this.step, list = this.build(d);
    return d > this.depth ? list.find((b) => isPrefix(s, b)) : list.find((a) => isPrefix(a, s));
  }

  // 这一步真正“经过”的深度（面包屑用）：比上一层多出了新东西的那些
  path() {
    const out = [];
    let prev = null;
    for (let d = 1; d <= this.depth; d++) {
      const a = this.stepAt(d);
      if (!a) continue;
      const v = viewOf(d, a);
      if (!prev || !sameStep(prev.a, a) || prev.v !== v) out.push({ d, step: a, v });
      prev = { a, v };
    }
    // 当前所在的深度总是列在最后（比如 D3 里“看真实画面”这一步和 D2 一样，但你确实在 D3）
    if (out[out.length - 1].d !== this.depth) out.push({ d: this.depth, step: this.step, v: viewOf(this.depth, this.step) });
    return out;
  }

  intoTarget() {
    if (this.depth === 1) return 2;
    const s = this.step, v = viewOf(this.depth, s);
    for (let d = this.depth + 1; d <= MAX_DEPTH; d++) {
      const b = this.build(d).find((x) => isPrefix(s, x));
      if (!b) return 0;
      if (!sameStep(b, s) || viewOf(d, b) !== v) return d;
    }
    return 0;
  }
  outTarget() {
    const p = this.path();
    return p.length >= 2 ? p[p.length - 2].d : 0;
  }
  canInto() { return this.intoTarget() > 0; }

  setDepth(d) {
    d = Math.max(1, Math.min(MAX_DEPTH, d));
    if (d === this.depth) return;
    const s = this.step;
    const list = this.build(d);
    let idx = d > this.depth ? list.findIndex((b) => isPrefix(s, b)) : list.findIndex((a) => isPrefix(a, s));
    this.depth = d;
    this.list = list;
    this.i = Math.max(0, idx);
    this.p = 0;
    this.emit('step');
  }
  into() { const d = this.intoTarget(); if (d) this.setDepth(d); }
  out() { const d = this.outTarget(); if (d) this.setDepth(d); }

  // 前进一步；跨过最后一步 = 真实世界和梦都走到下一帧
  advance() {
    if (this.i < this.list.length - 1) { this.i++; this.p = 0; this.emit('step'); return true; }
    this.sim.forward(this.depth >= 2);
    this.list = this.build(this.depth);
    this.i = 0;
    this.p = 0;
    this.emit('frame');
    this.emit('step');
    return true;
  }

  back() {
    if (this.p > 0.15 && this.depth >= 2) { this.p = 0; this.emit('step'); return; }
    if (this.i > 0) { this.i--; this.p = 0; this.emit('step'); return; }
    if (this.sim.back()) {
      this.list = this.build(this.depth);
      this.i = this.list.length - 1;
      this.p = 0;
      this.emit('frame');
      this.emit('step');
    }
  }

  next() { if (this.p < 1 && this.playing) this.p = 0; this.advance(); }
  seekIndex(i) { this.i = Math.max(0, Math.min(this.list.length - 1, i)); this.p = 0; this.emit('step'); }
  seek(pred) { const i = this.list.findIndex(pred); if (i >= 0) this.seekIndex(i); }

  play() { this.playing = true; this.emit('play'); }
  pause() { this.playing = false; this.emit('play'); }
  toggle() { this.playing ? this.pause() : this.play(); }

  // budget：这一帧最多花多少毫秒算新帧（8× 时算不过来就慢一点，不卡住页面）
  tick(dt, budget = 12) {
    if (!this.playing) return;
    this.p += (dt * this.speed) / this.dur;
    const t0 = performance.now();
    let guard = 0;
    while (this.p >= 1 && guard++ < 40) {
      const carry = this.p - 1;
      this.advance();
      this.p = this.depth === 1 ? carry : Math.min(carry, 0.99);   // D1 一帧很短：8× 时一次要走好几帧
      if (performance.now() - t0 > budget) { this.p = Math.min(this.p, 0.5); break; }
    }
  }
}
