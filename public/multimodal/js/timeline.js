// 调试器的核心：一次“看图回答”被拆成一棵步骤树，深度越大，步子越细。
//
//   d1 黑箱      看图（只在第一个词元）+ 每个词元一步
//   d2 流水线    预处理 / 视觉编码器 / 合并器 / 拼进对话 / 语言模型 / 输出；之后的词元只有 读入 / 语言模型 / 输出
//   d3 逐层      缩放 · 切块 · 归一化；图块嵌入 · 位置嵌入 · ViT 24 层；2×2 合并 · MLP · DeepStack；聊天模板 · M-RoPE；LLM 28 层
//   d4 一层之内  ViT 块与 LLM 层内部的算子；图块嵌入 = 一组卷积核
//   d5 一次乘加  一个图块的 1536 个像素值 × 卷积核 → 嵌入的一个数；焦点层的 16 个注意力头各自在看哪
//
// 图片只在生成第一个词元之前编码一次，之后视觉词元都躺在 KV 缓存里。

export const VIT_OPS = ['ln1', 'attn', 'add1', 'ln2', 'mlp', 'add2'];
export const LLM_OPS = ['ln1', 'attn', 'add1', 'ln2', 'mlp', 'add2'];
export const MAX_DEPTH = 5;
export const DEPTH_NAMES = ['对话', '黑箱', '流水线', '逐层', '一层之内', '一次乘加'];

const DUR = {
  see: 2.6, pass: 1.7, read: 1.1,
  prep: 2.4, vit: 2.8, merge: 2.4, splice: 2.4, llm: 2.4, head: 1.7,
  resize: 2.4, patch: 2.6, norm: 2.0, embed: 2.2, pos: 2.0, vitL: 0.55, vitOp: 0.7, vitAttn: 1.5,
  group: 2.2, mlp: 2.0, deep: 2.4, template: 2.6, mrope: 3.0,
  layer: 0.6, ln1: 0.8, attn: 1.7, add1: 0.8, ln2: 0.8, mlpL: 1.5, add2: 0.8, deepL: 1.6, heads: 2.6,
  logits: 1.6, pick: 1.7, conv: 2.6, mpick: 1.6, mul: 2.8, sum: 2.2,
};

export function buildSteps(depth, g, C) {
  if (depth <= 1) return g === 0 ? [{ g, ph: 'see' }, { g, ph: 'pass' }] : [{ g, ph: 'pass' }];
  const s = [];
  if (g === 0) {
    if (depth === 2) s.push({ g, ph: 'prep' }, { g, ph: 'vit' }, { g, ph: 'merge' }, { g, ph: 'splice' });
    else {
      for (const sub of ['resize', 'patch', 'norm']) s.push({ g, ph: 'prep', sub });
      if (depth === 3) s.push({ g, ph: 'vit', sub: 'embed' });
      else if (depth === 4) s.push({ g, ph: 'vit', sub: 'embed', mi: 'conv' });
      else for (const mi of ['pick', 'mul', 'sum']) s.push({ g, ph: 'vit', sub: 'embed', mi });
      s.push({ g, ph: 'vit', sub: 'pos' });
      for (let L = 0; L < C.NV; L++) {
        if (depth === 3) s.push({ g, ph: 'vit', L });
        else for (const op of VIT_OPS) s.push({ g, ph: 'vit', L, op });
      }
      for (const sub of ['group', 'mlp', 'deep']) s.push({ g, ph: 'merge', sub });
      for (const sub of ['template', 'mrope']) s.push({ g, ph: 'splice', sub });
    }
  } else s.push({ g, ph: 'read' });
  if (depth === 2) s.push({ g, ph: 'llm' });
  else {
    for (let L = 0; L < C.NL; L++) {
      if (depth === 3) { s.push({ g, ph: 'layer', L }); continue; }
      const ops = L < C.deep && g === 0 ? [...LLM_OPS, 'deep'] : LLM_OPS;  // DeepStack 只在预填充时加到图片位置上
      for (const op of ops) {
        if (depth >= 5 && op === 'attn' && C.focus.includes(L)) s.push({ g, ph: 'layer', L, op, mi: 'heads' });
        else s.push({ g, ph: 'layer', L, op });
      }
    }
  }
  if (depth === 2) s.push({ g, ph: 'head' });
  else s.push({ g, ph: 'head', sub: 'logits' }, { g, ph: 'head', sub: 'pick' });
  return s;
}

const FAMILY = { see: ['prep', 'vit', 'merge', 'splice'], pass: ['read', 'llm', 'layer', 'head'], llm: ['layer'] };

// a 是否是 b 的“祖先或自身”
export function isPrefix(a, b) {
  if (a.g !== b.g) return false;
  if (a.ph !== b.ph) return !!FAMILY[a.ph]?.includes(b.ph);
  for (const k of ['sub', 'L', 'op', 'mi']) {
    if (a[k] === undefined) continue;
    if (a[k] !== b[k]) return false;
  }
  return true;
}

export const sameStep = (a, b) => a && b && isPrefix(a, b) && isPrefix(b, a);

// 相机、监视器使用的“视图”
export function viewOf(depth, s) {
  if (depth <= 1) return 'box';
  switch (s.ph) {
    case 'prep': return 'image';
    case 'vit':
      if (s.mi === 'conv') return 'conv';
      if (s.mi) return 'micro';
      if (s.sub) return 'embed';
      if (depth === 2) return 'vit';
      return s.op ? 'vitop' : 'vitlayer';
    case 'merge': return 'merge';
    case 'splice': return 'splice';
    case 'read': return 'tray';
    case 'llm': return 'tower';
    case 'layer': return s.mi ? 'heads' : s.op ? 'layerop' : 'layer';
    case 'head': return 'head';
  }
  return 'box';
}

function durOf(s) {
  switch (s.ph) {
    case 'vit':
      if (s.mi) return DUR[s.mi === 'pick' ? 'mpick' : s.mi];
      if (s.sub) return DUR[s.sub];
      if (s.op) return s.op === 'attn' ? DUR.vitAttn : DUR.vitOp;
      return s.L === undefined ? DUR.vit : DUR.vitL;
    case 'layer':
      if (s.mi) return DUR.heads;
      if (s.op) return DUR[s.op === 'mlp' ? 'mlpL' : s.op === 'deep' ? 'deepL' : s.op];
      return DUR.layer;
    case 'prep': case 'merge': case 'splice': case 'head':
      return DUR[s.sub || s.ph];
  }
  return DUR[s.ph] || 1.5;
}

export class Timeline {
  constructor(Q, C) {
    this.Q = Q;
    this.C = C;
    this.G = Q.G;
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

  build(depth, g) { return buildSteps(depth, g, this.C); }
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
