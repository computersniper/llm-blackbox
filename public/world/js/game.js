// 「夜路」：世界模型要学的那个小游戏。俯视角，小车在一条蜿蜒的夜路上往前开，躲开抛锚的车。
// 画面 64×64，每个像素是调色板里的一个编号；三个动作：0 直行、1 向左、2 向右。
//
// 这份代码和 tools/world/game.py 是同一个游戏的两份实现，只用整数运算（随机数是 xorshift32），
// 同一个种子、同一串动作 → 两边逐像素相同（tools/world/check_game.mjs 核对）。所以网页上可以让
// “真实世界”和“模型的梦”并排跑：真实的这一边就是 Python 采集训练数据时用的那个游戏本身。
// 本文件不碰 DOM，Node 里也能直接 import。

export const W = 64, H = 64;
export const HALF = 12;          // 路面半宽（中心线两侧各 11 格沥青，第 12 格是路沿）
export const CAR_Y = 48;         // 小车车头所在的屏幕行（车身占 48..55 行）
export const CAR_H = 8, CAR_W = 5;
export const OB_H = 7;           // 抛锚车高 7 行
export const SPEED = 2;          // 每一步路面往下滚 2 行
export const STEER = 2;          // 每一步左右挪 2 格
export const FP = 16;            // 路中心用 1/16 像素的定点数
const LO = (HALF + 3) * FP, HI = (W - 1 - HALF - 3) * FP;
const TARGETS = [-10, -6, -3, 0, 0, 3, 6, 10];   // 弯道的“目标斜率”（1/16 像素每行）
const SEG = 24;                  // 每 24 行换一次目标斜率
const SPAWN = 30;                // 每行刷出抛锚车的概率：30/1000（两辆之间至少隔 14 行）
const GAP = 14;
export const ACTIONS = ['直行', '向左', '向右'];

// 调色板：深色夜景，不要太亮
export const PALETTE = [
  [15, 32, 29],    // 0 草地
  [24, 49, 41],    // 1 草丛
  [35, 40, 51],    // 2 沥青
  [93, 101, 119],  // 3 路沿
  [184, 146, 63],  // 4 中线
  [72, 214, 189],  // 5 小车
  [11, 47, 44],    // 6 车窗
  [208, 86, 111],  // 7 抛锚车
  [77, 26, 39],    // 8 抛锚车车窗
  [47, 53, 66],    // 9 车灯照亮的路面
  [22, 48, 43],    // 10 车灯照亮的草地
  [240, 211, 138], // 11 车头灯
];

// 小车（车头朝上）和抛锚车的像素图；-1 是透明
const _ = -1;
const CAR = [
  [11, 5, 5, 5, 11],
  [5, 5, 5, 5, 5],
  [5, 6, 6, 6, 5],
  [5, 6, 6, 6, 5],
  [5, 5, 5, 5, 5],
  [5, 5, 5, 5, 5],
  [5, 6, 6, 6, 5],
  [5, 5, 5, 5, 5],
];
const OB = [
  [_, 7, 7, 7, _],
  [7, 8, 8, 8, 7],
  [7, 7, 7, 7, 7],
  [7, 7, 7, 7, 7],
  [7, 8, 8, 8, 7],
  [7, 7, 7, 7, 7],
  [7, _, _, _, 7],
];

// xorshift32：x 始终是 0..2^32-1 的整数
export function rngInit(seed) {
  let x = ((seed >>> 0) ^ 0x9e3779b9) >>> 0;
  if (x === 0) x = 1;
  const r = { x };
  for (let i = 0; i < 8; i++) rngNext(r);
  return r;
}
export function rngNext(r) {
  let x = r.x;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  r.x = x >>> 0;
  return r.x;
}

// 草丛纹理：只看 (x, 世界行号) 的整数哈希，路面滚动时草丛跟着走
function tuft(x, wr) {
  let h = (Math.imul(x, 374761393) + Math.imul(wr, 668265263)) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
  return (h >>> 24) < 16;
}

export class Game {
  constructor(seed) { this.reset(seed); }

  reset(seed) {
    this.seed = seed >>> 0;
    this.rng = rngInit(this.seed);
    this.cx = new Int32Array(64);      // 路中心（定点数），按 世界行号 & 63 存
    this.cxv = 32 * FP;                // 当前生成到的路中心
    this.vx = 0;                       // 当前斜率
    this.tv = 0;                       // 目标斜率
    this.nextOk = 40;                  // 开头 40 行不刷抛锚车
    this.obs = [];                     // 抛锚车 { n: 最下面那一行的世界行号, x: 中心列 }
    this.base = 0;                     // 屏幕最下面一行（第 63 行）是第几个世界行
    this.top = -1;                     // 已经生成到的世界行号
    while (this.top < 63) this.genRow();
    this.carX = this.centerAt(this.base + 12);
    this.t = 0;
    this.done = false;
    this.why = '';
    return this;
  }

  // 生成下一行路：每行固定取 3 个随机数（弯道目标、要不要刷车、刷在哪），两边实现的随机数序列才对得上
  genRow() {
    const n = ++this.top;
    const r1 = rngNext(this.rng), r2 = rngNext(this.rng), r3 = rngNext(this.rng);
    if (n % SEG === 0) this.tv = TARGETS[r1 % 8];
    this.vx += Math.sign(this.tv - this.vx);
    this.cxv += this.vx;
    if (this.cxv < LO) { this.cxv = LO; this.vx = 0; this.tv = Math.abs(this.tv); }
    if (this.cxv > HI) { this.cxv = HI; this.vx = 0; this.tv = -Math.abs(this.tv); }
    this.cx[n & 63] = this.cxv;
    if (n >= this.nextOk && r2 % 1000 < SPAWN) {
      this.obs.push({ n, x: ((this.cxv + 8) >> 4) + (r3 % 19) - 9 });
      this.nextOk = n + GAP;
    }
  }

  // 某个世界行的路中心（整像素）
  centerAt(wr) { return (this.cx[wr & 63] + 8) >> 4; }

  step(a) {
    if (this.done) return;
    if (a === 1) this.carX -= STEER;
    else if (a === 2) this.carX += STEER;
    this.carX = Math.max(2, Math.min(W - 3, this.carX));
    for (let k = 0; k < SPEED; k++) { this.base++; this.genRow(); }
    this.obs = this.obs.filter((o) => o.n + OB_H - 1 >= this.base);
    this.t++;
    // 撞车：小车占世界行 base+8 .. base+15
    const lo = this.base + (H - 1 - (CAR_Y + CAR_H - 1)), hi = this.base + (H - 1 - CAR_Y);
    for (const o of this.obs) {
      if (o.n <= hi && o.n + OB_H - 1 >= lo && Math.abs(o.x - this.carX) <= CAR_W - 1) { this.done = true; this.why = 'crash'; }
    }
    // 冲出路面：车身中间那一行，离路中心超过 11 格
    if (!this.done && Math.abs(this.carX - this.centerAt(this.base + 12)) > HALF - 1) { this.done = true; this.why = 'offroad'; }
  }

  // 画成调色板编号（Uint8Array(64*64)，行优先）
  render(out = new Uint8Array(W * H)) {
    const base = this.base;
    for (let r = 0; r < H; r++) {
      const wr = base + H - 1 - r;
      const c = this.centerAt(wr);
      const dash = ((wr >> 2) & 1) === 0;
      const row = r * W;
      for (let x = 0; x < W; x++) {
        const d = Math.abs(x - c);
        let p;
        if (d < HALF) p = x === c && dash ? 4 : 2;
        else if (d === HALF) p = 3;
        else p = tuft(x, wr) ? 1 : 0;
        out[row + x] = p;
      }
    }
    // 车灯照亮前方：车头上面 14 行，越远越宽
    for (let d = 1; d <= 14; d++) {
      const r = CAR_Y - d, hw = 1 + ((d / 3) | 0);
      for (let x = Math.max(0, this.carX - hw); x <= Math.min(W - 1, this.carX + hw); x++) {
        const i = r * W + x, p = out[i];
        if (p === 2) out[i] = 9;
        else if (p === 0 || p === 1) out[i] = 10;
      }
    }
    for (const o of this.obs) {
      for (let k = 0; k < OB_H; k++) {
        const r = base + (H - 1) - (o.n + OB_H - 1) + k;
        if (r < 0 || r >= H) continue;
        for (let j = 0; j < CAR_W; j++) {
          const x = o.x - 2 + j, p = OB[k][j];
          if (p >= 0 && x >= 0 && x < W) out[r * W + x] = p;
        }
      }
    }
    for (let k = 0; k < CAR_H; k++) {
      for (let j = 0; j < CAR_W; j++) {
        const x = this.carX - 2 + j, p = CAR[k][j];
        if (p >= 0 && x >= 0 && x < W) out[(CAR_Y + k) * W + x] = p;
      }
    }
    return out;
  }

  // 存档 / 读档（调试器的“上一步”要退回之前的帧）
  save() {
    return { seed: this.seed, rx: this.rng.x, cx: this.cx.slice(), cxv: this.cxv, vx: this.vx, tv: this.tv, nextOk: this.nextOk, obs: this.obs.map((o) => ({ ...o })), base: this.base, top: this.top, carX: this.carX, t: this.t, done: this.done, why: this.why };
  }
  load(s) {
    this.seed = s.seed; this.rng = { x: s.rx }; this.cx = s.cx.slice(); this.cxv = s.cxv; this.vx = s.vx; this.tv = s.tv;
    this.nextOk = s.nextOk; this.obs = s.obs.map((o) => ({ ...o })); this.base = s.base; this.top = s.top; this.carX = s.carX; this.t = s.t; this.done = s.done; this.why = s.why;
    return this;
  }
}

// 采集训练数据用的启发式司机：往前看 20 步（屏幕上已经看得见的路和抛锚车），用动态规划挑一条不撞车、
// 尽量贴着路中间的走法，返回第一步的动作。只用整数，和 Python 版逐步一致。网页上的“自动驾驶”也用它
const PLAN = 20, BAD = 1000000;
export function heuristic(g) {
  let next = new Int32Array(W);           // 第 k+1 步之后的最好得分（第 PLAN+1 步为 0）
  let cur = new Int32Array(W);
  for (let k = PLAN; k >= 1; k--) {
    const b = g.base + SPEED * k, lo = b + 8, hi = b + 15, c = g.centerAt(b + 12);
    for (let x = 2; x <= W - 3; x++) {
      let ok = Math.abs(x - c) <= HALF - 1;
      for (const o of g.obs) if (o.n <= hi && o.n + OB_H - 1 >= lo && Math.abs(o.x - x) <= CAR_W - 1) ok = false;
      if (!ok) { cur[x] = -BAD * (PLAN - k + 1); continue; }
      let best = -BAD * PLAN * 2;
      if (k < PLAN) for (const d of [0, -STEER, STEER]) { const v = next[Math.max(2, Math.min(W - 3, x + d))]; if (v > best) best = v; }
      else best = 0;
      cur[x] = best - Math.abs(x - c);
    }
    [next, cur] = [cur, next];
  }
  let a = 0, best = -BAD * PLAN * 4;
  for (const [act, d] of [[0, 0], [1, -STEER], [2, STEER]]) {
    const v = next[Math.max(2, Math.min(W - 3, g.carX + d))];
    if (v > best) { best = v; a = act; }
  }
  return a;
}

// 调色板编号 → RGBA（给 ImageData 用）
export function toRGBA(idx, out = new Uint8ClampedArray(W * H * 4)) {
  for (let i = 0; i < W * H; i++) {
    const c = PALETTE[idx[i]];
    out[i * 4] = c[0]; out[i * 4 + 1] = c[1]; out[i * 4 + 2] = c[2]; out[i * 4 + 3] = 255;
  }
  return out;
}

// 调色板编号 → 模型的输入（3×64×64，CHW，0..1 的浮点数，和 PyTorch 那边一致）
export function toCHW(idx, out = new Float32Array(3 * W * H)) {
  const N = W * H;
  for (let i = 0; i < N; i++) {
    const c = PALETTE[idx[i]];
    out[i] = c[0] / 255; out[N + i] = c[1] / 255; out[2 * N + i] = c[2] / 255;
  }
  return out;
}
