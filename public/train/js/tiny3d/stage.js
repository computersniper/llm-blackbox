// 唐宋诗小模型这一章的 3D 舞台：引擎（和推理页、玻璃小模型共用 ../../../js/stage/engine.js）+ 机器（machine.js）+ 2D 浮层（board.js）。
// 页面（../main.js）只管：第一次进这一章时建它、每帧调用 frame（推进时间轴、按需取数据）、把调试器交给 ui.js。
import { Engine } from '../../../js/stage/engine.js';
import { TinyMachine } from './machine.js';
import { TinyBoard } from './board.js';
import { L } from '../lang.js';
import { esc } from '../../../js/ui.js';

export class TinyStage {
  constructor(host, stageEl, X, app, { state, onTip, onFreeChange, scrub }) {
    this.X = X;
    this.app = app;
    this.state = state;
    this.onTip = onTip;
    this.host = host;
    this.E = new Engine(host, { onFrame: (dt, t) => this.frame(dt, t), onHover: (info, e) => this.hover(info, e), onPick: (info) => this.pick(info), onFreeChange });
    this.M = new TinyMachine(this.E, X, app);
    this.B = new TinyBoard(stageEl, { onTip, scrub, E: this.E });
    this.pill = document.createElement('div');
    this.pill.className = 'wait-pill t3pill';
    this.pill.setAttribute('role', 'status');
    stageEl.append(this.pill);
    this.lastView = '';
    this.lastDepth = -1;
    this.started = false;
  }

  get free() { return this.E.free; }
  exitFree() { this.E.exitFree(); }
  setInsets(r, b) { this.E.setInsets(r, b); }

  set active(on) {
    this.E.active = on;
    if (!on) { this.B.hide(); this.onTip(null); this.pill.classList.remove('on'); }
  }
  get active() { return this.E.active; }

  frame(dt, t) {
    const st = this.state(dt, t);
    if (!st) return;
    const M = this.M;
    M.update(st, dt, t);
    this.B.ap = M.p;
    this.B.update(st, this.X, t);
    M.insetL = this.B.insetL;
    const w = st.wait;
    this.waitPill(w.waiting && t - w.since > 0.25, w.err);
    const cam = M.camera(st);
    if (!this.started) {
      this.started = true;
      this.E.setView(cam.pos.clone().sub(cam.look).multiplyScalar(1.7).add(cam.look), cam.look, { snap: true });
    }
    const same = st.view === this.lastView && st.depth === this.lastDepth;
    this.E.setView(cam.pos, cam.look, { speed: same ? 2.4 : 2.0 });
    if (!same) { this.lastView = st.view; this.lastDepth = st.depth; }
  }

  waitPill(on, err) {
    const html = err ? esc(err.unsupported ? err.message : L('这一步的数据没有载入成功，稍后会自动重试…', 'This step’s data failed to load; retrying automatically…')) : `<span class="spin"></span>${L('正在载入这一步的真实记录…', 'Loading the real record of this step…')}`;
    if (this.pill._h !== html) { this.pill.innerHTML = html; this.pill._h = html; }
    this.pill.classList.toggle('on', !!on);
  }

  hover(info, e) {
    this.M.hover = info;
    if (!info) return this.onTip(null);
    this.onTip(this.M.tip(info), e);
  }

  // 点预测柱 / −ln p：跳到“一块权重”那一层的这个位置的损失
  pick(info) {
    this.onTip(null);
    if (info?.type === 'pillar' || info?.type === 'nll') this.app.pickPos?.(info.i);
  }
}

// 这一屏要用的数据分块（见 ../main.js 的 plan()）：ck = D1 的检查点数据，st = 一步之内，dw = 补充的 ΔW，feat = 权重轨迹
export function planTiny(tl, X, p) {
  const K = X.K, k = tl.k;
  const at = (part) => (j) => (j >= 0 && j < K ? X.key(part, j) : null);
  const ck = at('ck'), stp = at('st'), dw = at('dw');
  // 3D 机器的每一拍都走一遍前向 / 反向 / 更新：D1 起三种都要
  p.want.push(ck(k), stp(k), dw(k));
  if (tl.depth === 1) {
    p.hold.push(ck(k), stp(k), dw(k), ck(k + 1));
    const ahead = tl.playing ? Math.min(6, 1 + Math.ceil(tl.speed)) : 1;
    for (let j = 1; j <= ahead; j++) p.soon.push(ck(k + j), stp(k + j), dw(k + j));
  } else {
    p.need.push(stp(k));
    p.hold.push(stp(k), ck(k), dw(k));
    p.soon.push(stp(k + 1), ck(k + 1), dw(k + 1), stp(k - 1));
    if (tl.depth >= 4) p.soon.push(X.key('feat'));
    if (tl.depth >= 5) p.want.push(X.key('feat'));
  }
  for (const key of ['need', 'want', 'hold', 'soon']) p[key] = [...new Set(p[key].filter(Boolean))];
  return p;
}

// 知识碎片：和原来的第二章同一批（格式先于内容、预热、余弦退火、教师强制、逻辑透镜……）
export function discoverTiny(tl, X, discover) {
  const s = tl.step, d = tl.depth, t = X.step(s.k);
  if (d === 1 && t >= 46 && t < 400) discover('format');
  if (d === 1 && t >= 200 && t < 260) discover('warmup');
  if (d === 1 && s.k === X.K - 1) discover('cosine');
  if (d === 1 && s.k >= 20) discover('head');
  if (s.ph === 'batch' && s.sub === 'shift') discover('teacher');
  if (s.ph === 'fwd' && s.sub && s.sub !== 'emb' && d >= 3) discover('lens');
  if (s.ph === 'loss' && s.i != null) discover('ce');
  if (s.ph === 'bwd' && d >= 2) discover('backprop');
  if (s.ph === 'bwd' && s.c != null) discover('outer');
  if (s.ph === 'upd' && s.sub === 'clip') discover('clip');
  if (s.mi === 'bc' && X.has('st', s.k)) discover(X.adam(s.k, s.f).t === 1 ? 'sign' : 'bias');
  if (s.mi === 'dw') discover('decay');
}
