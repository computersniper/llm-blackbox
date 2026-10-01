// 真实世界和模型的梦，按同一串动作一起往前走。
//
// 每一帧 F 记着：真实画面 o（调色板编号）、梦的当前 z（ẑ）、梦画出来的画面 ô = V.decode(ẑ)、
// LSTM 进这一帧之前的状态 (h, c)。从 F 走到下一帧（rec）：
//   睁眼：z = V.encode(o).μ       闭眼：z = ẑ（用梦自己上一步想出来的）
//   (h, c) ← LSTM([z, onehot(a)], h, c)；π, μ, σ, done ← MDN(h)；ẑ' ~ 按温度 τ 采样；ô' = V.decode(ẑ')
//   真实游戏 game.step(a) → o'
// 所有帧都存在 hist 里：调试器的“上一步”就是退回之前的帧，再前进时按原样重放（除非你又按了方向键）。

import { Game, heuristic, toCHW, W, H } from './game.js';
import { mulberry, HID } from './nn.js';

const CAP = 600;   // 最多留多少帧
const N = W * H;

export class Sim {
  constructor(model, { seed = 20261002, dreamSeed = 7 } = {}) {
    this.m = model;
    this.mode = 'open';
    this.tau = 1;
    this.auto = 'heur';
    this.rand = mulberry(dreamSeed);
    this.game = new Game(seed);
    this.hist = [];
    this.idx = -1;
    this.frameNo = 0;
    this.gameNo = 0;
    this.keys = () => 0;          // 现在按着的方向（main.js 提供）：0 没按，1 左，2 右
    this.listeners = new Set();
    this.push(this.firstFrame(seed));
  }

  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(type, data) { for (const fn of this.listeners) fn(type, data); }

  get cur() { return this.hist[this.idx]; }
  get next() { return this.hist[this.idx + 1] || null; }

  // 一局的第一帧：梦从真实画面的编码开始（ẑ = μ），LSTM 状态清零
  firstFrame(seed) {
    this.game.reset(seed);
    this.gameNo++;
    const o = this.game.render();
    const enc = this.m.encode(toCHW(o));
    return this.frame({ o, zhat: enc.mu, h: new Float32Array(HID), c: new Float32Array(HID), resync: 'start', enc });
  }

  frame(f) {
    return { n: this.frameNo++, dreamAge: 0, t: this.game.t, seed: this.game.seed, gameNo: this.gameNo, snap: this.game.save(), realDone: this.game.done, why: this.game.why, dreamDone: 0, rec: null, dreamY: null, mse: null, enc: null, ...f };
  }

  push(f) {
    this.hist.push(f);
    this.idx = this.hist.length - 1;
    // 梦画出来的整幅画面（每帧 4.8 万个数）只给最近 40 帧留着，更早的退回去时再解码（差异值留着画曲线）
    const old = this.hist[this.idx - 40];
    if (old && old !== this._detailOf) old.dreamY = null;
    if (this.hist.length > CAP) { const k = this.hist.length - CAP; this.hist.splice(0, k); this.idx -= k; }
  }

  truncate() {
    if (this.idx < this.hist.length - 1) this.hist.length = this.idx + 1;
    if (this.cur) { this.cur.rec = null; this.cur._detail = null; }
  }

  // 这一帧的真实画面编码（缓存）
  encOf(F) {
    if (!F.enc) F.enc = this.m.encode(toCHW(F.o));
    return F.enc;
  }

  // 梦画出来的这一帧（缓存）；顺手算出和真实画面的均方误差
  dreamOf(F) {
    if (!F.dreamY) {
      F.dreamY = this.m.decode(F.zhat).y;
      const x = toCHW(F.o);
      let s = 0;
      for (let i = 0; i < 3 * N; i++) { const d = F.dreamY[i] - x[i]; s += d * d; }
      F.mse = s / (3 * N);
    }
    return F.dreamY;
  }

  // 这一步该做什么动作：方向键优先，然后是自动驾驶
  chooseAction(F) {
    const k = this.keys();
    if (k) return { a: k, by: 'key' };
    if (this.auto === 'heur') { this.game.load(F.snap); return { a: heuristic(this.game), by: 'heur' }; }
    if (this.auto === 'ctrl') {
      const z = this.zIn(F);
      const r = this.m.ctrl(z, F.h);
      if (r) return { a: r.a, by: 'ctrl', logits: r.logits };
    }
    return { a: 0, by: 'none' };
  }

  // 这一帧喂给 M 的 z：睁眼用真实画面的编码，闭眼用梦自己的；梦刚“撞车”过就重新睁眼对齐一次
  zIn(F) {
    if (this.mode === 'open' || F.resyncNext) return this.encOf(F).mu;
    return F.zhat;
  }

  // 从当前帧算出下一帧（不移动 idx）。latched = 动作已经定下（调试器里一帧拆成很多小步时，动作在这一帧开始时读一次）
  computeNext(F, act) {
    this.game.load(F.snap);
    if (F.realDone) {
      // 真实世界撞车了：换一条新路，梦也重新对齐
      F.rec = { kind: 'reset', a: act.a, by: act.by, mode: this.mode, tau: this.tau };
      const seed = (Math.random() * 4294967296) >>> 0;
      return this.firstFrame(seed);
    }
    const src = this.mode === 'open' ? 'enc' : F.resyncNext ? 'resync' : 'dream';
    const z = this.zIn(F);
    const L = this.m.lstm(z, act.a, F.h, F.c);
    const M = this.m.mdn(L.h);
    const S = this.m.sample(M, this.tau, this.rand);
    F.rec = { kind: 'step', a: act.a, by: act.by, logits: act.logits || null, mode: this.mode, tau: this.tau, src, z, L, M, S };
    this.game.step(act.a);
    const o = this.game.render();
    // dreamAge：梦已经闭着眼自己往下想了几步（睁眼时每一步都是“看着真实画面预测下一帧”，记 0）
    const G = this.frame({ o, zhat: S.z, h: L.h, c: L.c, dreamDone: M.done, dreamAge: src === 'dream' ? (F.dreamAge || 0) + 1 : 0 });
    // 梦里预测“撞车了”（闭眼时）：这一段梦到此为止，下一步重新睁眼对齐
    if (M.done > 0.5 && this.mode === 'closed') G.resyncNext = true;
    return G;
  }

  // 准备好下一帧（调试器拆开一帧时要先知道这一帧的全部计算）
  prepare() {
    const F = this.cur;
    if (!this.next || !F.rec) {
      this.truncate();
      const G = this.computeNext(F, this.chooseAction(F));
      this.hist.push(G);
      if (this.hist.length > CAP) { this.hist.shift(); this.idx--; }
    }
    return F.rec;
  }

  // 前进一帧。已经算过的下一帧：动作一致（或这一帧的动作已经锁定）就照原样重放，否则从这里重新算
  forward(latched = false) {
    const F = this.cur;
    const nx = this.next;
    if (nx && F.rec) {
      if (latched || (this.keys() === 0 && this.auto === 'off') || this.chooseAction(F).a === F.rec.a) {
        this.idx++;
        this.game.load(this.cur.snap);
        this.emit('frame', this.cur);
        return this.cur;
      }
    }
    this.truncate();
    const G = this.computeNext(F, this.chooseAction(F));
    this.push(G);
    this.game.load(G.snap);
    this.emit('frame', G);
    if (G.realDone) this.emit('crash', G);
    if (G.dreamDone > 0.5) this.emit('dreamCrash', G);
    if (G.resync === 'start') this.emit('newgame', G);
    return G;
  }

  back() {
    if (this.idx <= 0) return false;
    this.idx--;
    this.game.load(this.cur.snap);
    this.emit('frame', this.cur);
    return true;
  }

  // 下面这些都会改写“从这一帧往后”的历史
  setMode(m) {
    if (m === this.mode) return;
    this.mode = m;
    this.truncate();
    this.emit('mode', m);
  }
  setTau(v) { this.tau = v; this.truncate(); }
  setAuto(v) { this.auto = v; this.truncate(); }

  // 把梦拉回真实：ẑ = 真实画面的编码（LSTM 的记忆保留）
  sync() {
    const F = this.cur;
    this.truncate();
    F.zhat = this.encOf(F).mu.slice();
    F.dreamY = null;
    F.resync = 'sync';
    F.resyncNext = false;
    F.dreamAge = 0;
    this.emit('frame', F);
  }

  // 拖动潜变量：改的是梦的当前 z，之后的梦从改过的 z 往下想（所以会自动闭眼）
  editZ(d, v) {
    const F = this.cur;
    this.truncate();
    if (this.mode !== 'closed') this.setMode('closed');
    F.zhat = F.zhat.slice();
    F.zhat[d] = v;
    F.dreamY = null;
    F.edited = true;
    this.emit('frame', F);
  }

  newGame(seed = (Math.random() * 4294967296) >>> 0) {
    this.truncate();
    this.push(this.firstFrame(seed));
    this.emit('frame', this.cur);
    this.emit('newgame', this.cur);
  }

  // D3 以下要看的中间结果（每一层的激活值）：只给当前这一帧算，换帧时丢掉
  detail(F = this.cur) {
    if (F._detail) return F._detail;
    const r = F.rec;
    const d = { enc: this.m.encode(toCHW(F.o), true) };
    d.recon = this.m.decode(d.enc.mu).y;
    if (r && r.kind === 'step') d.dec = this.m.decode(r.S.z, true);
    if (this._detailOf && this._detailOf !== F) this._detailOf._detail = null;
    this._detailOf = F;
    F._detail = d;
    return d;
  }

  // 最近 n 帧的差异（画差异曲线用；还没解码过的帧没有值）
  recent(n) {
    const a = Math.max(0, this.idx - n + 1);
    return this.hist.slice(a, this.idx + 1);
  }
}

export function mseOf(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return s / a.length;
}
