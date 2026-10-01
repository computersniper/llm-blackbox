// 调试器的核心：一条 agent 轨迹被拆成一棵“步骤树”，深度越大，步子越细。
//
//   d1 屏幕      每一轮：说一句话 / 每次动手（命令在终端里跑、文件在编辑器里改）/ 最终回答
//   d2 循环      组装上下文 → 模型生成 → 解析工具调用 → 沙箱执行 → 结果接回对话（一圈一圈）
//   d3 上下文    组装上下文拆成：拼出整段历史 / 复用 KV 缓存、只算新增部分
//   d4 生成      模型生成拆成：一个词元一步，关键处停下来看候选概率
//   d5 模型内部  每个词元：一次完整的 36 层前向计算 → 抽签选出下一个词元
//
// “＋”= 单步进入（step into），“−”= 跳出（step out），上一步 / 下一步 = 在当前深度逐步执行。
// 外层循环是“轮”（turn）：模型每说完一次（以 <|im_end|> 结束）算一轮。
import { isDecision } from './data.js';

export const MAX_DEPTH = 5;
export const DEPTH_NAMES = ['任务', '屏幕', '循环', '上下文', '生成', '模型内部'];

// 每一级的键：k1 屏幕事件、k2 循环阶段、k3 上下文子步骤、k4 词元、k5 前向子步骤
const KEYS = ['k1', 'k2', 'k3', 'k4', 'k5'];

function d1Steps(t, T) {
  if (T.final) return [{ t, k1: 'answer' }];
  const s = T.say ? [{ t, k1: 'think' }] : [];
  T.calls.forEach((c, i) => s.push({ t, k1: `act${i}`, c: i }));
  return s;
}

// 第二级：一圈循环的各个阶段，挂到第一级的事件下面
function d2Steps(t, T) {
  const head = ['prompt', 'gen', 'parse'];
  const out = [];
  const g1 = T.final ? 'answer' : T.say ? 'think' : 'act0';
  for (const p of head) out.push({ t, k1: g1, k2: p, p });
  if (T.final) out.push({ t, k1: 'answer', k2: 'done', p: 'done' });
  T.calls.forEach((c, i) => {
    out.push({ t, k1: `act${i}`, c: i, k2: `exec${i}`, p: 'exec' });
    out.push({ t, k1: `act${i}`, c: i, k2: `append${i}`, p: 'append' });
  });
  return out;
}

export function buildSteps(depth, R, t) {
  const T = R.turns[t];
  if (depth <= 1) return d1Steps(t, T);
  const out = [];
  for (const s of d2Steps(t, T)) {
    if (depth >= 3 && s.p === 'prompt') {
      out.push({ ...s, k3: 'ctx', sub: 'ctx' }, { ...s, k3: 'kv', sub: 'kv' });
    } else if (depth >= 4 && s.p === 'gen') {
      T.gen.toks.forEach((tk, j) => {
        if (depth >= 5) for (const f of ['fwd', 'pick']) out.push({ ...s, k3: 'tok', k4: j, j, k5: f, f });
        else out.push({ ...s, k3: 'tok', k4: j, j });
      });
    } else out.push(s);
  }
  return out;
}

// a 是否是 b 的“祖先或自身”
export function isPrefix(a, b) {
  if (a.t !== b.t) return false;
  for (const k of KEYS) {
    if (a[k] === undefined) return true;
    if (a[k] !== b[k]) return false;
  }
  return true;
}
export const sameStep = (a, b) => a && b && isPrefix(a, b) && isPrefix(b, a);

// 舞台上显示哪一块
export function viewOf(depth, s) {
  if (depth <= 1) return 'screen';
  if (depth === 2) return 'loop';
  if (s.p === 'gen' && s.j !== undefined) return s.f ? 'fwd' : 'tok';
  if (s.p === 'parse' && depth >= 4) return 'tok';
  return 'ctx';
}

// 每一步多长（秒，1× 速度下）
function durOf(depth, s, R) {
  const T = R.turns[s.t];
  if (s.k1 === 'think' && depth === 1) return clamp(0.9 + T.say.length / 40, 1.6, 7);
  if (s.k1 === 'answer' && depth === 1) return clamp(1.2 + T.say.length / 36, 2.4, 9);
  if (depth === 1) return actDur(T, s.c, T.say ? 0 : 1);
  switch (s.p) {
    case 'prompt': return s.sub === 'ctx' ? 2.4 : s.sub === 'kv' ? 2.6 : 2.0;
    case 'gen': {
      if (s.j === undefined) return depth === 2 ? clamp(1.2 + T.gen.text.length / 70, 1.8, 6) : 2.4;
      const tk = T.gen.toks[s.j];
      if (s.f) return s.f === 'fwd' ? 1.7 : 1.5;
      return isDecision(tk, s.j) ? 1.7 : 0.14;
    }
    case 'parse': return depth >= 4 ? 2.2 : 1.6;
    case 'exec': return depth === 2 ? actDur(T, s.c, 0) : 1.8;
    case 'append': return 1.8;
    case 'done': return 2.4;
  }
  return 1.5;
}

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

function actDur(T, ci, extra) {
  const c = T.calls[ci];
  const lines = (c.result.match(/\n/g) || []).length + 1;
  let d = 1.4 + extra;
  if (c.name === 'bash') d += String(c.args.command || '').length / 38 + Math.min(lines, 40) * 0.05 + 0.8;
  else if (c.name === 'read_file') d += 1.0 + Math.min(lines, 60) * 0.025;
  else if (c.name === 'edit_file') d += 2.2;
  else if (c.name === 'write_file') d += Math.min(4, String(c.args.content || '').length / 140) + 0.8;
  return clamp(d, 2, 8);
}

export class Timeline {
  constructor(R) {
    this.R = R;
    this.T = R.T;
    this.depth = 1;
    this.t = 0;
    this.i = 0;
    this.p = 0;
    this.playing = false;
    this.speed = 1;
    this.done = false;
    this.list = this.build(1, 0);
    this.listeners = new Set();
  }

  build(depth, t) { return buildSteps(depth, this.R, t); }
  get step() { return this.list[this.i]; }
  get view() { return viewOf(this.depth, this.step); }
  get dur() { return durOf(this.depth, this.step, this.R); }
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(type) { for (const fn of this.listeners) fn(type, this); }

  canInto() {
    if (this.depth >= MAX_DEPTH) return false;
    const s = this.step;
    const deeper = this.build(this.depth + 1, this.t).find((b) => isPrefix(s, b));
    return !!deeper && (!sameStep(deeper, s) || viewOf(this.depth + 1, deeper) !== viewOf(this.depth, s));
  }

  setDepth(d) {
    d = Math.max(1, Math.min(MAX_DEPTH, d));
    if (d === this.depth) return;
    const s = this.step;
    const list = this.build(d, this.t);
    let idx;
    if (d > this.depth) idx = list.findIndex((b) => isPrefix(s, b));
    else idx = list.findIndex((a) => isPrefix(a, s));
    this.depth = d;
    this.list = list;
    this.i = Math.max(0, idx);
    this.p = 0;
    this.emit('depth');
    this.emit('step');
  }

  into() { if (this.canInto()) this.setDepth(this.depth + 1); }
  out() { if (this.depth > 1) this.setDepth(this.depth - 1); }

  // 前进一步；跨过最后一步 = 进入下一轮
  advance() {
    if (this.i < this.list.length - 1) { this.i++; this.p = 0; this.emit('step'); return true; }
    if (this.t < this.T - 1) {
      this.t++;
      this.list = this.build(this.depth, this.t);
      this.i = 0;
      this.p = 0;
      this.emit('turn');
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
    if (this.t > 0) {
      this.t--;
      this.list = this.build(this.depth, this.t);
      this.i = this.list.length - 1;
      this.p = 0;
      this.emit('turn');
      this.emit('step');
    }
  }

  next() { this.done = false; this.advance(); }

  seekIndex(i) { this.i = Math.max(0, Math.min(this.list.length - 1, i)); this.p = 0; this.done = false; this.emit('step'); }

  seekTurn(t, pred = null) {
    t = Math.max(0, Math.min(this.T - 1, t));
    if (t !== this.t) {
      this.t = t;
      this.list = this.build(this.depth, t);
      this.emit('turn');
    }
    const i = pred ? this.list.findIndex(pred) : 0;
    this.i = Math.max(0, i);
    this.p = 0;
    this.done = false;
    this.emit('step');
  }

  restart() { this.done = false; this.seekTurn(0); }

  play() { if (this.done) return; this.playing = true; this.emit('play'); }
  pause() { this.playing = false; this.emit('play'); }
  toggle() { this.playing ? this.pause() : this.play(); }

  tick(dt) {
    if (!this.playing) return;
    this.p += (dt * this.speed) / this.dur;
    let guard = 0;
    while (this.p >= 1 && guard++ < 60) {
      const carry = this.p - 1;
      if (!this.advance()) { this.playing = false; this.emit('play'); break; }
      this.p = Math.min(carry, 0.99);
    }
  }
}
