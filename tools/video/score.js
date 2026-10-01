// 分镜表：每个镜头的时间、机器状态、机位、字幕、数据条、事件。
// 节拍：96 BPM（一拍 0.625 秒，一小节 2.5 秒）。段落表 PLAN 里每段占几个小节，所有段落都卡在小节线上；
// 配乐（compose.py）读 render.mjs 导出的同一张段落表和事件表。
// 字幕按每秒不超过 7–8 个字来控制长短（数字读得快，可以略多）。
//
// 问题：「天空为什么是蓝色的？」（q01）。字幕和数据条里的数字全部从真实数据里现取。
import { THREE } from './lib/engine.js';
import { path, blendCam, orbit, handheld, clamp, lerp, seg, smooth, smoother, easeIn, easeInOut, v3, pchip } from './lib/cam.js';
import { viewOf } from '/public/js/timeline.js';
import { tokPlain, fmtPct, esc } from '/public/js/ui.js';
import { bf16Bits, bf16Value } from '/public/js/num.js';

export const BPM = 96, BEAT = 60 / BPM, BAR = 4 * BEAT;
const B = (bar, beat = 0) => bar * BAR + beat * BEAT;

// 段落：名字、小节数、能量（配乐用）
const PLAN = [
  ['chat', 5, 0.1], ['fly', 3, 0.55], ['land', 3, 0.4], ['embed', 3, 0.5], ['layers1', 10, 0.78],
  ['attn', 35, 0.4], ['ffn', 23, 0.45], ['sample1', 7, 0.6], ['loop1', 6, 0.72], ['layersK', 6, 0.85], ['sampleK', 4, 0.7],
  ['loop2', 7, 0.9], ['end', 10, 0.22],
];
// 顶部章节进度条的六章
const CHAPTERS = ['切成词元', '查表', '穿过 28 层', '注意力', '前馈', '选字'];

const tk = (s) => esc(tokPlain(s));
const q = (s) => `<q>${s === '<|endoftext|>' ? '结束符' : tk(s)}</q>`;
const m = (s) => `<span class="m">${s}</span>`;
const pct = (p) => fmtPct(p);
const f2 = (v, d = 2) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
const V = (a) => v3(a[0], a[1], a[2]);
const cam = (p, l, fov = 32) => ({ pos: V(p), look: V(l), fov });
// 机位沿视线方向推远 / 拉近 k 倍
const farther = (c, k) => ({ ...c, pos: c.look.clone().add(c.pos.clone().sub(c.look).multiplyScalar(k)) });
// 角色配色（和网站算式板一致）：输入蓝、权重紫、乘积橙、结果青
const cx = (s) => `<span class="cx">${s}</span>`, cw = (s) => `<span class="cw">${s}</span>`, cy = (s) => `<span class="cy">${s}</span>`;

function mst(depth, g, step, p, o = {}) {
  const s = { g, ...step };
  return { step: s, p: clamp(p), g, depth, dAnim: o.dAnim ?? depth, view: o.view ?? viewOf(depth, s), head: o.head ?? null };
}

// 顺序排列的若干段：[{ ...data, d: 时长 }] 从 t0 开始 → [{ ...data, t0, t1 }]
function lay(t0, items) {
  let t = t0;
  return items.map((it) => { const a = { ...it, t0: t, t1: t + it.d }; t += it.d; return a; });
}
function pick(list, t) {
  for (let i = 0; i < list.length; i++) if (t < list[i].t1 || i === list.length - 1) return { ...list[i], i, p: clamp((t - list[i].t0) / (list[i].t1 - list[i].t0)) };
  return null;
}

export function buildScore(Q) {
  const NL = Q.NL;
  const subs = [], strips = [], chapters = [], cards = [], events = [], sections = [];
  const sub = (t0, t1, html) => subs.push({ t0, t1, html });
  const strip = (t0, t1, html, o = {}) => strips.push({ t0, t1, html, ...o });
  const chapter = (t0, t1, n, zh, en) => chapters.push({ t0, t1, html: `<span class="n">${n}</span><span class="bar"></span><span class="zh">${zh}</span><span class="en">${en}</span>` });
  const card = (t0, t1, html, o = {}) => cards.push({ t0, t1, html, transform: '', ...o });
  const ev = (t, type, o = {}) => events.push({ t, type, ...o });
  // v5：字幕上方的小号术语标签（一次一个）；章节进度（从某一时刻起第 i 章亮）
  const terms = [], progs = [];
  const term = (t0, t1, html) => terms.push({ t0, t1, html });
  const prog = (t, i) => progs.push({ t, i });

  // 段落表 → 每段的起止时间
  const SEC = {};
  let bb = 0;
  for (const [name, bars, energy] of PLAN) {
    SEC[name] = { b0: bb, bars, t0: B(bb), t1: B(bb + bars) };
    sections.push({ name, t0: B(bb), t1: B(bb + bars), bars, energy });
    bb += bars;
  }
  const end = B(bb);

  const shots = [];
  const shot = (name, fn) => shots.push({ name, t0: SEC[name].t0, t1: SEC[name].t1, fn });
  const shotSpan = (name, t0, t1, fn) => shots.push({ name, t0, t1, fn });
  const endCam = new Map(); // 每个镜头（及镜头内每个小段）最后一帧的机位：下一段从这里平滑接过去
  const prevCam = (name, fallback) => endCam.get(name) || fallback;

  const S = (g) => Q.steps[g];
  const user = Q.tokens.map((t, i) => ({ ...t, i })).filter((t) => t.role === 'user' && !t.sp && t.s !== 'user' && t.s !== '\n');
  const lens = (g, L) => Q.lensAt(g, L).top;
  const lensTop = (g, L) => lens(g, L)[0];
  const G0 = 0, LX = 24, GK = 14; // 拆开的是第 1 个词的第 24 层；第 15 个词「散」是第二个重点
  const row0 = Q.row(G0);
  const dot = Q.dotAt(LX, G0), neu = Q.neuronAt(LX, G0);
  const attD = Q.att(LX, dot.head, row0);
  const keyTok = Q.tokens[dot.key].s;
  const sinkHeads = Array.from({ length: Q.H }, (_, h) => Q.att(LX, h, row0)[0]).filter((r) => r && r.j === 0).length;
  const hm0 = Q.headMMAt(G0);
  const X = (M, i) => M.x(i);

  /* ------------------------------------------------------------ 开场：聊天页 → 打字 → 发送 → 气泡裂成词元 → 飞进黑箱 → 落进托盘（一个连续镜头） */
  // 打字：中文用拼音输入，每个词敲完拼音、候选一闪、上屏。上屏的时刻卡在八分音符上（配乐的拨弦跟着它走）
  const PY = {
    '天空': ["tian'kong", ['天空', '填空', '天', '田']],
    '为什么': ["wei'shen'me", ['为什么', '为', '喂', '未']],
    '是': ['shi', ['是', '时', '事', '使', '十']],
    '蓝色': ["lan'se", ['蓝色', '蓝', '篮', '拦']],
    '的': ['de', ['的', '得', '地', '德']],
  };
  const upAt = [B(0, 3), B(1, 0.5), B(1, 1.5), B(1, 2.5), B(1, 3.5), B(2, 0.5)];
  const jit = (i) => (((i * 7919) % 13) / 13 - 0.5) * 0.03; // 敲键间隔的一点不均匀（确定的）
  const typing = user.map((u, k) => {
    const [segd, cands] = PY[u.s] || [null, null];
    const up = upAt[k] ?? upAt[upAt.length - 1] + (k - upAt.length + 1) * BEAT;
    if (!segd) return { s: u.s, py: null, keys: [up], commit: up };
    const n = segd.replace(/'/g, '').length;
    const keys = Array.from({ length: n }, (_, j) => up - 0.17 - (n - 1 - j) * 0.074 + jit(k * 10 + j));
    return { s: u.s, py: segd.replace(/'/g, ''), seg: segd, cands, keys, commit: up };
  });
  const OPEN = {
    typing,
    sendT: B(2, 2),                               // 按下发送（Enter）
    boxY0: 560, boxY1: 912,                        // 输入框：屏幕中间 → 贴底
    push0: B(2, 2) + 1.15, push1: B(4) - 0.1,      // 镜头推近你的消息气泡
    zoom: 1.9, zoom2: 0.06, focusY: 470,
    split0: B(3, 2) - 0.25, splitGap: 0.1, splitD: 0.5, // 文字按真实分词裂成词元块
    depth: 7,                                      // 交给 3D 方块时离镜头的距离
    boxOpen0: SEC.fly.t0 + 4.3, boxOpen1: SEC.fly.t0 + 6.3,
    landStart0: SEC.fly.t0 + 6.2, landGap: 0.08,   // 6 个词元开始离开编队、落向托盘
    dropD: 0.45,
    landAt: {},
    end: SEC.land.t1,
  };
  OPEN.bubT = OPEN.sendT + 0.06;                   // 你的消息出现在对话区
  OPEN.botT = OPEN.sendT + 0.4;                    // 助手：正在思考…
  OPEN.diss0 = B(4) - 0.05; OPEN.diss1 = B(4) + 0.6; // 气泡以外的界面散去
  OPEN.swapT = B(4) + 0.7;                         // 词元块换成 3D 方块
  OPEN.warpT = OPEN.swapT - 1.0;                   // 背景粒子“跃迁”
  OPEN.fog0 = OPEN.swapT + 0.5; OPEN.fog1 = SEC.fly.t0 + 0.4; // 远处的黑箱从黑暗里显出来
  const titleIn = SEC.fly.t0 + 0.1, titleOut0 = SEC.fly.t0 + 2.8, titleOut1 = SEC.fly.t0 + 3.6; // 片名落版
  OPEN.titleIn = titleIn; OPEN.titleOut0 = titleOut0; OPEN.titleOut1 = titleOut1;
  user.forEach((u, k) => { OPEN.landAt[u.i] = SEC.land.t0 + k * OPEN.landGap; });
  // 聊天模板的其余词元：以问题为中心，一圈圈往两边落下来
  const uFirst = user[0].i, uLast = user[user.length - 1].i;
  for (let i = 0; i < Q.P; i++) {
    if (OPEN.landAt[i] != null) continue;
    const r = i < uFirst ? uFirst - i : i - uLast;
    OPEN.landAt[i] = SEC.land.t0 + 0.5 + 0.065 * r + OPEN.dropD;
  }
  // 镜头：词元交接以后从远处一路飞到黑箱跟前，越过倒下的前面板，降到托盘上；再稍微拉开看模板把问题包起来
  const camOpen = path([
    { t: 0, p: [0, 5.8, 124], l: [0, 5.8, 0], fov: 32 },
    { t: OPEN.swapT - 0.3, p: [0, 5.8, 124], l: [0, 5.8, 0], fov: 32 },
    { t: OPEN.swapT + 0.5, p: [0, 5.85, 121.6], l: [0, 5.8, 0], fov: 32 },
    { t: SEC.fly.t0, p: [-1.0, 6.3, 97], l: [-1.5, 5.6, 0], fov: 32 },
    { t: SEC.fly.t0 + 2.5, p: [-3.0, 6.6, 62], l: [-4.0, 4.8, 0], fov: 32 },
    { t: SEC.fly.t0 + 4.5, p: [-4.6, 5.9, 33], l: [-5.0, 3.0, 0], fov: 32 },
    { t: SEC.fly.t0 + 5.8, p: [-5.0, 4.3, 16], l: [-5.0, 1.3, 0], fov: 32 },
    { t: SEC.fly.t0 + 6.9, p: [-5.3, 2.6, 8.0], l: [-5.0, 0.35, 0], fov: 32 },
    { t: SEC.land.t0 + 0.1, p: [-5.6, 2.0, 5.6], l: [-5.0, 0.25, 0], fov: 32 },
    { t: SEC.land.t0 + 2.1, p: [-8.2, 3.6, 11.5], l: [-7.5, 0.4, 0], fov: 32 },
    { t: SEC.land.t0 + 4.5, p: [-6.6, 2.3, 7.0], l: [-5.6, 0.3, 0], fov: 32 },
    { t: SEC.land.t1, p: [-5.9, 1.8, 5.2], l: [-5.1, 0.22, 0], fov: 32 },
  ]);
  OPEN.camAt = camOpen;
  let kn = 0;
  for (const w of typing) if (w.py) for (const k of w.keys) ev(k, 'key', { i: kn++ });
  typing.forEach((w, i) => ev(w.commit, 'type', { i }));
  ev(OPEN.sendT, 'send');
  ev(OPEN.swapT, 'whoosh', { k: 0.8 });
  ev(OPEN.swapT, 'rise', { d: SEC.fly.t0 - OPEN.swapT });   // 词元飞起来，一路推到片名
  ev(SEC.fly.t0, 'hit', { k: 1 });
  ev(OPEN.boxOpen0 - 2.0, 'rise', { d: 2.0 });
  ev(OPEN.boxOpen0, 'open');
  for (let i = 0; i < Q.P; i++) ev(OPEN.landAt[i], 'tick', { i, k: i >= uFirst && i <= uLast ? 1 : 0.4 });
  // 片名在画面上的那几秒不放字幕
  sub(OPEN.split0 + 0.45, titleIn - 0.35, '发出去的问题，先被切成词元。');
  sub(OPEN.boxOpen0 + 0.1, SEC.land.t0 - 0.2, '黑箱打开，词元落进托盘。');
  sub(SEC.land.t0 + 0.3, SEC.land.t0 + 3.3, '聊天模板：给问题包上提示和标记');
  term(SEC.land.t0 + 0.4, SEC.land.t0 + 3.3, '聊天模板 chat template');
  sub(SEC.land.t0 + 3.5, SEC.land.t0 + 5.2, `一共 ${m(Q.P)} 个词元`);
  sub(SEC.land.t0 + 5.4, SEC.land.t1 - 0.2, `模型只认编号：${q(user[3].s)} = ${m(user[3].id)}`);
  term(SEC.land.t0 + 5.5, SEC.land.t1 - 0.2, '词元编号 token id');
  prog(OPEN.boxOpen0 + 0.1, 0);
  shotSpan('opening', 0, SEC.land.t1, (lt, t) => {
    const opened = t >= OPEN.boxOpen0;
    const ft = t - SEC.fly.t0; // 片名在飞行途中出现
    return {
      st: opened ? mst(2, 0, { ph: 'read' }, 1, { dAnim: lerp(1, 2.6, smoother(seg(t, OPEN.boxOpen0, OPEN.boxOpen1))) }) : mst(1, 0, { ph: 'pass' }, 0.04),
      cam: () => camOpen(t),
      fade: 1 - smooth(seg(t, OPEN.diss0, OPEN.diss0 + 0.3)),
      // 黑箱正面的型号字：片名淡出以后才亮，再开箱
      boxLabel: smooth(seg(t, titleOut1 + 0.05, titleOut1 + 0.55)),
      extras: { ids: smooth(seg(t, SEC.land.t0 + 3.3, SEC.land.t0 + 3.9)), idsFocus: true },
      ov: { band: smooth(seg(t, OPEN.diss0, OPEN.diss1)), title: smooth(seg(t, titleIn, titleIn + 0.8)) * (1 - smooth(seg(t, titleOut0, titleOut1))), titleK: seg(ft, 0.05, 2.6), titleBlur: 7 * smooth(seg(t, titleOut0, titleOut1)), titleOnBox: true },
    };
  });

  /* ------------------------------------------------------------ 02 嵌入 */
  {
    const T0 = SEC.embed.t0, T1 = SEC.embed.t1;
    prog(T0, 1);
    const k0 = T0 + 0.6, k1 = T0 + 4.0; // 从表里取向量（光点落下）
    for (let i = 0; i < Q.P; i += 2) ev(k0 + (k1 - k0) * ((i / Q.P) * 0.5 + 0.5), 'blip', { i });
    sub(T0 + 0.4, T0 + 3.8, `拿编号去表里查，取出一行：${m('1024')} 个数`);
    term(T0 + 0.5, T0 + 3.8, `嵌入表 ${m('151,936')} 行 × ${m('1,024')} 列`);
    sub(T0 + 4.1, T1 - 0.2, `从此，每个词元都是 ${m('1024')} 个数`);
    let embPath = null;
    shot('embed', (lt, t, { M }) => {
      const last = camOpen(SEC.land.t1);
      const uc = (X(M, user[0].i) + X(M, user[user.length - 1].i)) / 2;
      embPath ||= path([
        { t: 0, p: last.pos.toArray(), l: last.look.toArray(), fov: 32 },
        { t: 2.2, p: [uc + 0.6, 3.9, 6.6], l: [uc - 0.2, 2.6, -1.0], fov: 32 },   // 先贴近墙上问题那几行
        { t: 3.6, p: [uc + 0.4, 3.7, 7.6], l: [uc - 0.3, 2.3, -0.8], fov: 32 },
        { t: 7.5, p: [-7.6, 4.9, 19.4], l: [-8.1, 4.5, -0.6], fov: 32 },          // 再拉开：光柱从托盘长起来
      ]);
      return {
        st: mst(2, 0, { ph: 'embed' }, seg(t, T0 + 3.4, T1 - 0.5), { dAnim: 2.6 }),
        cam: () => embPath(lt),
        slabDim: 0.88 * smooth(seg(lt, 0, 1.0)) * (1 - smooth(seg(t, T1 - 1.3, T1))),
        extras: { ids: 1 - smooth(seg(lt, 0, 0.8)), idsFocus: true, emb: smooth(seg(lt, 0.2, 1.0)) * (1 - smooth(seg(t, T1 - 1.1, T1 - 0.1))), embK: seg(t, k0, k1) },
      };
    });
  }

  /* ------------------------------------------------------------ ③ 第 1 个词：穿过 28 层 */
  const towerCam = (M, Lc, g, o = {}) => {
    const y = 0.95 + Lc * 0.26;
    const xf = M.x(Q.row(g));
    const look = v3(xf + (o.lx ?? -1.2), y - 0.25, -0.2);
    return orbit({ pos: look.clone().add(v3(o.dx ?? 6.6, o.dy ?? 3.4, o.dz ?? 8.6)), look, fov: 32 }, o.yaw ?? 0, 0, o.dist ?? 1);
  };
  const bumpAt = (t, t0, d) => smooth(seg(t, t0 - 0.15, t0 + 0.35)) * (1 - smooth(seg(t, t0 + d - 0.3, t0 + d + 0.6)));
  {
    const T0 = SEC.layers1.t0, T1 = SEC.layers1.t1;
    prog(T0, 2);
    const sched1 = lay(T0 + 3 * BEAT, [
      { L: 0, d: 2 * BEAT },
      ...Array.from({ length: 16 }, (_, k) => ({ L: k + 1, d: BEAT / 2 })),
      ...Array.from({ length: 4 }, (_, k) => ({ L: 17 + k, d: BEAT })),
      { L: 21, d: 3 * BEAT }, { L: 22, d: BEAT }, { L: 23, d: BEAT },
      { L: 24, d: 5 * BEAT },
      { L: 25, d: BEAT }, { L: 26, d: BEAT }, { L: 27, d: BEAT },
    ]);
    const l1 = (L) => sched1.find((s) => s.L === L);
    sched1.forEach((s) => ev(s.t0, 'layer', { L: s.L, k: s.d > BEAT * 1.5 ? 1 : 0.6 }));
    ev(l1(21).t0, 'reveal', { k: 0.7 });
    ev(l1(24).t0, 'reveal', { k: 1.1 });
    const hold1 = l1(27).t1; // 28 层走完
    const dive0 = T1 - 4.4;  // 往第 24 层扎下去
    const l21 = lensTop(0, 21), l24 = lensTop(0, 24), l27 = lensTop(0, 27);
    sub(T0 + 0.3, T0 + 3.4, '接着，穿过 28 层。');
    term(T0 + 0.5, T0 + 3.4, 'Transformer 层 × 28');
    sub(T0 + 3.6, l1(17).t0 - 0.15, '每过一层，偷看一眼：它此刻最想说什么？');
    term(T0 + 3.8, l1(17).t0 - 0.15, '逻辑透镜 logit lens');
    sub(l1(17).t0 + 0.05, l1(21).t0 - 0.1, '前十几层，还说不出像样的词。');
    sub(l1(21).t0 + 0.05, l1(24).t0 - 0.1, `第 21 层：想说${q(l21[2])} ${m(pct(l21[1]))}`);
    sub(l1(24).t0 + 0.05, l1(25).t0 - 0.05, `第 24 层，它改口了：${q(l24[2])} ${m(pct(l24[1]))}`);
    sub(l1(25).t0 + 0.05, dive0 - 0.2, `最后一层：${q(l27[2])} ${m(pct(l27[1]))}`);
    sub(dive0, T1 - 0.15, `第 ${LX} 层发生了什么？拆开看看。`);
    const camLayers1 = (M, t) => {
      const Ls = [[T0, 0.5], [l1(0).t0, 0.5], [l1(17).t0, 15.5], [l1(21).t0, 20.0], [l1(24).t0, 23.2], [l1(25).t0, 24.3], [hold1, 26.5]];
      const Lc = pchip(Ls.map((a) => a[0]), Ls.map((a) => a[1]))(t);
      // 透镜改口的两个时刻（第 21 层「因为」、第 24 层「天空」）镜头推近一下
      const push = 0.2 * bumpAt(t, l1(21).t0, l1(21).d) + 0.3 * bumpAt(t, l1(24).t0, l1(24).d);
      return towerCam(M, Lc, 0, { yaw: lerp(-6, 12, smooth(seg(t, T0, hold1))), dist: 1 - push });
    };
    let e1In = null;
    shot('layers1', (lt, t, { M }) => {
      const s = pick(sched1, t);
      const before = t < sched1[0].t0;
      return {
        st: mst(3, 0, { ph: 'layer', L: s.L }, before ? 0 : s.p, { dAnim: 3 }),
        cam: () => {
          const live = camLayers1(M, t);
          if (lt < 2.4) { e1In ||= prevCam('embed', live); return blendCam(e1In, live, smoother(lt / 2.4)); } // 从嵌入的远景接进来
          if (t > hold1) { // 拉开看整座塔，再往第 24 层扎下去
            const wide = towerCam(M, 14, 0, { lx: -5, dx: 13, dy: 4.6, dz: 22 });
            const dive = cam([3.4, 8.9, 7.6], [0.5, 7.7, 0.2]);
            return blendCam(blendCam(live, wide, smoother(seg(t, hold1, dive0 + 0.3))), dive, smoother(seg(t, dive0 + 0.3, T1)));
          }
          return handheld(live, t, 0.003);
        },
        lensWin: t > hold1 ? 28 : 9,
        ov: { lens: { a: smooth(seg(t, T0 + 3.6, T0 + 4.3)) * (1 - smooth(seg(t, dive0, dive0 + 0.8))), g: 0, upto: before ? -0.01 : s.L + (s.L === 27 ? 1 : s.p) - 0.0001, other: lensTop(0, 21)[2], side: 'left' } },
      };
    });
  }

  /* ------------------------------------------------------------ ④⑤ 拆开第 24 层：注意力、前馈（一次只讲一件事） */
  const headCam = (M, st, o = {}) => {
    const hx = M.headX(st);
    const look = v3(hx + (o.lx ?? -0.4), M.yTop + (o.ly ?? 2.75), 0.1);
    return { pos: look.clone().add(v3(o.dx ?? 0.9, o.dy ?? 1.0, o.dz ?? 10.0)), look, fov: 32 };
  };
  {
    const T0 = SEC.attn.t0, T1 = SEC.ffn.t1;
    const D = lay(T0, [
      // 一层里的几步 + RMSNorm
      { key: 'over', depth: 4, s: { op: 'ln1' }, d: 8, p0: true },
      { key: 'ln1', depth: 4, s: { op: 'ln1' }, d: 8 },
      // ④ 注意力
      { key: 'intro', depth: 5, s: { op: 'attn', sub: 'qkv' }, d: 8, p0: true },
      { key: 'lib', depth: 5, s: { op: 'attn', sub: 'qkv' }, d: 4, p0: true },
      { key: 'q', depth: 5, s: { op: 'attn', sub: 'qkv' }, d: 8 },
      { key: 'k', depth: 5, s: { op: 'attn', sub: 'qkv' }, d: 8 },
      { key: 'v', depth: 5, s: { op: 'attn', sub: 'qkv' }, d: 8 },
      { key: 'score', depth: 5, s: { op: 'attn', sub: 'score' }, d: 12 },
      { key: 'dot', depth: 6, s: { op: 'attn', sub: 'score', mi: 'mul' }, d: 6 },
      { key: 'dsum', depth: 6, s: { op: 'attn', sub: 'score', mi: 'sum' }, d: 6 },
      { key: 'scale', depth: 6, s: { op: 'attn', sub: 'score', mi: 'scale' }, d: 8 },
      { key: 'softmax', depth: 5, s: { op: 'attn', sub: 'softmax' }, d: 12 },
      { key: 'mix', depth: 5, s: { op: 'attn', sub: 'softmax' }, d: 12 },
      { key: 'recap', depth: 5, s: { op: 'attn', sub: 'softmax' }, d: 4 },
      { key: 'heads', depth: 5, s: { op: 'attn', sub: 'softmax' }, d: 16 },
      { key: 'add1', depth: 5, s: { op: 'add1' }, d: 12 },
      // ⑤ 前馈
      { key: 'up', depth: 5, s: { op: 'mlp', sub: 'up' }, d: 8 },
      { key: 'act', depth: 5, s: { op: 'mlp', sub: 'act' }, d: 10 },
      { key: 'gate', depth: 5, s: { op: 'mlp', sub: 'act' }, d: 8 },
      { key: 'down', depth: 5, s: { op: 'mlp', sub: 'down' }, d: 8 },
      { key: 'pick', depth: 6, s: { op: 'mlp', sub: 'up', mi: 'pick' }, d: 6 },
      { key: 'mul', depth: 6, s: { op: 'mlp', sub: 'up', mi: 'mul' }, d: 8 },
      { key: 'sum', depth: 6, s: { op: 'mlp', sub: 'up', mi: 'sum' }, d: 8 },
      { key: 'silu', depth: 6, s: { op: 'mlp', sub: 'act', mi: 'silu' }, d: 8 },
      { key: 'gate2', depth: 6, s: { op: 'mlp', sub: 'act', mi: 'gate' }, d: 8 },
      { key: 'bits', depth: 7, s: { op: 'mlp', sub: 'up', mi: 'mul' }, d: 16 },
      { key: 'out', depth: 4, s: null, d: 4 },
    ].map((x) => ({ ...x, d: x.d * BEAT })));
    const Dk = (k) => D.find((x) => x.key === k);
    // 章节进度：RMSNorm 和注意力算 ④，前馈、乘加、比特算 ⑤
    prog(T0, 2); prog(Dk('ln1').t0, 3); prog(Dk('up').t0, 4);
    D.forEach((x) => { if (!['over', 'lib', 'recap', 'dsum', 'out', 'bits'].includes(x.key)) ev(x.t0, 'step', { k: 0.5 }); });
    ev(Dk('over').t0, 'hit', { k: 0.7 });
    ev(Dk('score').t0, 'reveal', { k: 0.5 });
    ev(Dk('softmax').t0, 'reveal', { k: 0.7 });
    ev(Dk('recap').t0, 'reveal', { k: 1.0 });
    ev(Dk('up').t0, 'hit', { k: 0.5 });
    for (let k = 0; k < 12; k++) ev(Dk('dot').t0 + (k / 12) * Dk('dot').d * 0.94, 'tick', { k: 0.3 });
    for (let k = 0; k < 12; k++) ev(Dk('mul').t0 + (k / 12) * Dk('mul').d * 0.94, 'tick', { k: 0.3 });
    for (let k = 0; k < 16; k++) ev(Dk('bits').t0 + 0.8 + k * 0.08, 'bit', { k });
    ev(Dk('out').t0, 'whoosh', { k: 1 });

    // 真实数字
    const inNorm = Q.norm(LX - 1, row0);
    const wb = neu.wg[0];
    const FLIP = 1; // 最高的指数位
    const fb = bf16Bits(wb); fb[FLIP] ^= 1;
    const wFlip = bf16Value(fb).value, gFlip = neu.gz + neu.x[0] * (wFlip - wb);
    const sci = (v) => { const e = Math.floor(Math.log10(Math.abs(v))); return `${(v / 10 ** e).toFixed(1)}×10<sup>${e}</sup>`; };
    const flipT0 = Dk('bits').t0 + 6.0, flipT1 = Dk('bits').t1 - 0.2;
    ev(flipT0, 'flip');
    const att6 = Q.att(LX, dot.head, row0);
    const nm = (s) => (s === '<|im_start|>' ? '开头' : tk(s));
    const wOf = (j) => att6.find((r) => r.j === j)?.w ?? 0;
    const a1 = att6[0], a2 = att6[1], a3 = att6[2];
    const HEADS = [dot.head, 10, 14, 1, 0, 9].filter((h, i, a) => a.indexOf(h) === i).slice(0, 6);
    const lensX = lensTop(0, LX);
    const n1 = (v) => v.toFixed(1);

    // 字幕（白话，一行）+ 术语标签（小号）
    const say = (k, html, tg, o = {}) => {
      const x = Dk(k);
      const a = x.t0 + (o.a ?? 0.2), b = (o.b != null ? x.t0 + o.b : x.t1 - 0.15);
      sub(a, b, html);
      if (tg) term(a + 0.1, b, tg);
    };
    say('over', `拆开第 ${LX} 层：每层都是同样几步`);
    say('ln1', '先把这串数，拉回统一的音量', `RMSNorm 归一化 · 均方根 ${m((inNorm / 32).toFixed(2))} → ${m('1')}`);
    say('intro', '注意力：回头看看前面说了什么', '自注意力 Self-Attention');
    say('lib', '就像在图书馆里查资料');
    say('q', 'Q 是提问：我想找什么？', `Query 查询 · q = h × W<sub>q</sub>`);
    say('k', 'K 是每个词的标签：我是什么', `Key 键 · k = h × W<sub>k</sub>`);
    say('v', 'V 是这个词真正能提供的内容', `Value 值 · v = h × W<sub>v</sub>`);
    say('score', '拿提问，去对每个词的标签', `点积 Q·K<sup>T</sup>`, { b: 3.7 });
    say('score', '越对得上，匹配分越高', `点积 Q·K<sup>T</sup>`, { a: 3.9 });
    say('dot', `放大一次：${m('128')} 对数，逐个相乘`, `第 ${m(dot.head)} 个头 · 每个头 ${m('128')} 维`);
    say('dsum', `全部加起来：${cy(n1(dot.sum))}`);
    say('scale', '再除以 √128，免得分数太大', `d = ${m('128')} · √d ≈ ${m('11.3')}`);
    say('softmax', 'softmax：分数变成权重，总和为 1', 'softmax', { b: 3.7 });
    say('softmax', `${q('天空')}一个就拿走 ${m(pct(dot.w))}`, 'softmax', { a: 3.9 });
    say('mix', '按权重，把各词的 V 混在一起', '加权求和 Σ w × V', { b: 3.7 });
    say('mix', '得到它读完上下文的新理解', '加权求和 Σ w × V', { a: 3.9 });
    say('recap', '这就是注意力的全部公式');
    say('heads', `同样的查找，同时有 ${m('16')} 个头`, `多头注意力 · ${m('16')} 个头`, { b: 4.8 });
    say('heads', `各找各的：有的盯${q('天空')}，有的盯开头`, `多头注意力 · ${m('16')} 个头`, { a: 5.0 });
    say('add1', '把结果加回去：只改一点，不推倒重来', '残差连接', { b: 3.7 });
    say('add1', `这一层之后，它最想说${q(lensX[2])}`, `逻辑透镜 · ${q(lensX[2])} ${m(pct(lensX[1]))}`, { a: 3.9 });
    say('up', `接下来是前馈：${m('3072')} 个小开关`, '前馈网络 FFN');
    say('act', '每个开关判断：某个特征出现了没？', null, { b: 3.4 });
    say('act', `这一次，明显亮起 ${m(Q.mlpCount(G0, LX))} 个`, '激活的神经元', { a: 3.6 });
    say('gate', '门控再决定：每个放行多少', 'SwiGLU 门控');
    say('down', `再压回 ${m('1024')} 个数，加回去`, '残差连接');
    say('pick', '屏幕上每个数，都是这样算出来的', `第 ${m(neu.j)} 号开关的输入 g`);
    say('mul', `${cx(m('1024'))} 个输入，各乘一个${cw('权重')}……`, '乘加 multiply–add');
    say('sum', `……其余 ${m('1012')} 项也加进来：${cy('g = ' + neu.gz.toFixed(3))}`, null, { a: 0.6 });
    say('silu', `开关开多大：SiLU(${m(neu.gz.toFixed(3))}) = ${m(neu.silu.toFixed(3))}`, 'SiLU 激活函数');
    say('gate2', `再乘另一路 ${m(neu.uz.toFixed(3))}，输出 ${cy((neu.silu * neu.uz).toFixed(2))}`, '门控：silu(g) × u');
    say('bits', `每个权重，在内存里只是 ${m('16')} 个 0/1`, 'bf16 格式', { b: 3.1 });
    say('bits', '1 位正负，8 位指数，7 位尾数', `${cw(m(wb.toFixed(4)))} 在内存里的样子`, { a: 3.3, b: 6.0 - 0.15 });
    say('bits', `翻错一位：权重变成 ${m(sci(wFlip))}`, '指数位翻转', { a: 6.1, b: 8.1 });
    say('bits', `g 跟着变成 ${m(sci(gFlip))}：全乱了`, '一位都错不得', { a: 8.2 });
    say('out', `${m('5.96')} 亿个这样的数，算出下一个字`);

    const L = (M) => ({ xf: M.x(row0), y: M.yL(LX), e: M.e });
    const dCam = {
      over: (M, p) => { const { xf, y } = L(M); return blendCam(cam([xf + 2.8, y + 2.3, 6.6], [xf + 0.7, y + 1.25, 0.2]), cam([xf + 1.1, y + 1.55, 3.9], [xf + 0.65, y + 1.25, 0.2]), smooth(p)); },
      ln1: (M, p) => { const { xf, y } = L(M); return blendCam(cam([xf - 0.9, y + 0.85, 3.5], [xf + 0.35, y + 0.3, 0.1]), cam([xf - 0.55, y + 0.65, 2.75], [xf + 0.35, y + 0.28, 0.1]), smooth(p)); },
      // Q / K / V：机位跟着正在讲的那块矩阵慢慢平移
      qkv: (M, p, st, t) => {
        const { xf, y } = L(M);
        const A = M.mats.attn;
        const ax = (pn) => (A && A.visible ? A.position.x + pn.position.x + pn.w / 2 : xf + 1.8);
        const tq = Dk('q').t0, tk_ = Dk('k').t0, tv = Dk('v').t0;
        const cx_ = pchip([Dk('intro').t0, tq - 0.2, tq + 0.9, tk_ - 0.2, tk_ + 0.9, tv - 0.2, tv + 0.9, Dk('v').t1], [xf + 2.2, xf + 2.2, ax(A.q), ax(A.q), ax(A.k), ax(A.k), ax(A.v), ax(A.v) + 0.15])(t);
        const zz = pchip([Dk('intro').t0, tq, tq + 1.2, Dk('v').t1], [5.3, 5.3, 4.2, 4.0])(t);
        return cam([cx_ - 0.35, y + 1.95, zz], [cx_, y + 1.45, 0.62]);
      },
      score: (M, p) => { const { xf, y } = L(M); const kx = M.x(dot.key); return blendCam(cam([(kx + xf) / 2 - 0.6, y + 2.3, 8.6], [(kx + xf) / 2, y + 1.3, 0]), cam([(kx + xf) / 2 + 0.2, y + 2.0, 7.6], [(kx + xf) / 2 + 0.1, y + 1.3, 0]), smooth(p)); },
      soft: (M, p) => { const { xf, y } = L(M); const kx = M.x(dot.key); return blendCam(cam([(kx + xf) / 2 + 0.2, y + 2.0, 7.6], [(kx + xf) / 2 + 0.1, y + 1.3, 0]), cam([(kx + xf) / 2 + 0.9, y + 2.1, 7.2], [(kx + xf) / 2 + 0.5, y + 1.3, 0]), smooth(p)); },
      heads: (M, p) => { const { xf, y } = L(M); return blendCam(cam([xf + 0.3, y + 1.6, 13.2], [xf + 0.5, y + 1.75, 0]), cam([xf + 0.6, y + 1.5, 12.6], [xf + 0.55, y + 1.75, 0]), smooth(p)); },
      dot: (M, p) => { const c = M.detail.dotCenter || v3(L(M).xf - 1, L(M).y + 1.6, 0.7); return farther(blendCam(cam([c.x - 0.05, c.y + 0.55, c.z + 3.3], [c.x - 0.3, c.y + 0.3, c.z]), cam([c.x - 0.25, c.y + 0.42, c.z + 2.75], [c.x - 0.3, c.y + 0.3, c.z]), smooth(p)), 1.5); },
      add1: (M, p) => { const { xf, y } = L(M); return blendCam(cam([xf - 0.4, y + 1.6, 2.9], [xf + 0.45, y + 1.1, 0.1]), cam([xf - 0.2, y + 1.7, 4.2], [xf + 0.6, y + 1.15, 0.1]), smooth(p)); },
      // 前馈：先贴近神经元阵列（灯一盏盏亮起），再拉开把三块矩阵也框进来
      mlp: (M, p) => { const { xf, y } = L(M); const a = cam([xf + 1.6, y + 3.6, 7.6], [xf + 1.2, y + 2.85, -0.2]), b = cam([xf + 1.25, y + 3.4, 6.3], [xf + 1.2, y + 2.85, -0.2]); const panel = cam([xf + 0.5, y + 3.5, 4.4], [xf + 0.2, y + 3.15, -0.9]); return p < 0.6 ? blendCam(a, panel, smooth(seg(p, 0.12, 0.3)) * (1 - smooth(seg(p, 0.42, 0.6)))) : blendCam(a, b, smooth(seg(p, 0.6, 1))); },
      mm: (M, p, st) => { const c = M.micro.camera(st); if (c) endCam.set('mm', c); const b = endCam.get('mm') || dCam.mlp(M, 1); return orbit({ ...b, fov: 32 }, lerp(-6, 4, smooth(p)), 0, lerp(1.6, 1.4, smooth(p))); },
      neuron: (M, p, st) => { const c = M.detail.camera(st); return orbit({ ...c, fov: 32 }, lerp(-6, 5, smooth(p)), 0, lerp(1.55, 1.4, smooth(p))); },
      bits: (M, p, st) => { const c = M.detail.camera(st); return orbit({ ...c, fov: 32 }, lerp(-12, 4, smooth(p)), lerp(8, 2, smooth(p)), lerp(2.3, 1.5, easeInOut(seg(p, 0, 0.6)))); },
    };
    const camKey = { over: 'over', ln1: 'ln1', intro: 'qkv', lib: 'qkv', q: 'qkv', k: 'qkv', v: 'qkv', score: 'score', dot: 'dot', dsum: 'dot', scale: 'dot', softmax: 'soft', mix: 'soft', recap: 'soft', heads: 'heads', add1: 'add1', up: 'mlp', act: 'mlp', gate: 'mlp', down: 'mlp', pick: 'mm', mul: 'mm', sum: 'mm', silu: 'neuron', gate2: 'neuron', bits: 'bits' };
    const camSpan = {}; // 同一个机位 key 跨越的时间段
    D.forEach((x) => { const k = camKey[x.key]; if (!k) return; camSpan[k] ||= { t0: x.t0, t1: x.t1 }; camSpan[k].t1 = x.t1; });
    // 算式板（网站 D6/D7）只在前馈的乘加、SiLU、比特这几段出现：画面往左让出位置
    const BOARD = ['mm', 'neuron', 'bits'];
    const boardShift = (t) => smoother(seg(t, camSpan.mm.t0 - 0.2, camSpan.mm.t0 + 0.9)) * (1 - smoother(seg(t, camSpan.bits.t1 - 0.7, camSpan.bits.t1 + 0.4)));
    const BLEND = { over: 2.8, ln1: 1.3, qkv: 1.5, score: 1.4, soft: 1.2, heads: 1.6, dot: 1.4, add1: 1.2, mlp: 1.5, mm: 1.5, neuron: 0.9, bits: 1.0 };

    // 公式板：随着讲解一项一项搭出来（show = 已出现的项，hi = 正在讲的项）
    const FORM = {
      score: { show: ['qk'], hi: 'qk', cap: `${q('天空')}的匹配分：Q·K = ${m(n1(dot.sum))}` },
      dot: { show: ['qk'], hi: 'qk', cap: `${q('天空')}：q·k = Σ q<sub>i</sub> × k<sub>i</sub>（${m('128')} 项）` },
      dsum: { show: ['qk'], hi: 'qk', cap: `${q('天空')}：q·k = ${m(n1(dot.sum))}` },
      scale: { show: ['qk', 'sd'], hi: 'sd', cap: `${m(n1(dot.sum))} ÷ √${m('128')} = ${m(n1(dot.score))}` },
      softmax: { show: ['qk', 'sd', 'sm'], hi: 'sm', cap: `${q(nm(Q.tokens[a1.j].s))} ${m(pct(a1.w))} · ${q(nm(Q.tokens[a2.j].s))} ${m(pct(a2.w))} · ${q(nm(Q.tokens[a3.j].s))} ${m(pct(a3.w))} …… 合计 ${m('100%')}` },
      mix: { show: ['qk', 'sd', 'sm', 'v'], hi: 'v', cap: `${m(a1.w.toFixed(2))} × V${q(nm(Q.tokens[a1.j].s))} + ${m(a2.w.toFixed(2))} × V${q(nm(Q.tokens[a2.j].s))} + ……` },
      recap: { show: ['lhs', 'qk', 'sd', 'sm', 'v'], hi: 'all', cap: `第 ${m(LX)} 层 · 第 ${m(dot.head)} 个头 · d = ${m('128')}` },
    };
    const formKeys = Object.keys(FORM);
    const fFirst = Dk('score').t0, fLast = Dk('recap').t1;

    shotSpan('dissect', T0, T1, (lt, t, { M }) => {
      const s = pick(D, t);
      if (s.key === 'out') { // 拉出来：回到塔顶的输出头
        const st = mst(4, G0, { ph: 'head', sub: 'norm' }, 0, { dAnim: 4 });
        return {
          st,
          cam: () => {
            const a = endCam.get('d-bits') || headCam(M, st);
            const far = cam([M.x(row0) + 9, 15.5, 23], [M.x(row0) - 1.5, 8.6, 0], 34);
            const target = headCam(M, st, { dz: 7.2, ly: 0.9, lx: 1.8, dx: 1.6 });
            return s.p < 0.5 ? blendCam(a, far, easeIn(s.p / 0.5)) : blendCam(far, target, smoother((s.p - 0.5) / 0.5));
          },
        };
      }
      // Q / K / V 三步：矩阵扫描连续推进（不在每一步重来）
      let p = s.p0 ? 0 : s.p;
      if (['q', 'k', 'v'].includes(s.key)) p = seg(t, Dk('q').t0, Dk('v').t1 - 0.6);
      let head = null;
      const hk = Dk('heads');
      const hOn = s.key === 'heads' ? HEADS[Math.min(HEADS.length - 1, Math.floor(clamp((t - hk.t0 - 4.0) / (hk.d - 4.6)) * HEADS.length))] : dot.head;
      if (['score', 'softmax', 'mix', 'recap', 'dot', 'dsum', 'scale'].includes(s.key)) head = dot.head;
      if (s.key === 'heads') head = t < hk.t0 + 4.0 ? dot.head : hOn;
      const st = mst(s.depth, G0, { ph: 'layer', L: LX, ...s.s }, p, { dAnim: s.depth, head });
      const ck = camKey[s.key], span = camSpan[ck];
      // 指示环
      const A = M.mats.attn;
      const vecAt = (o) => () => (A && A.visible && o.visible ? o.localToWorld(v3(o.len / 2, 0, 0)) : null);
      const beamEnd = (j, end) => () => { const b = (M.beamList || []).find((x) => x.j === j); return b ? b.m.localToWorld(b.geo.parameters.path[end].clone()) : null; };
      const ringA = (k, a = 0.25, b = 0.15) => { const x = Dk(k); return smooth(seg(t, x.t0 + a, x.t0 + a + 0.45)) * (1 - smooth(seg(t, x.t1 - b - 0.35, x.t1 - b))); };
      const rings = [];
      if (s.key === 'q') rings.push({ at: vecAt(A.qo), w: A.qo.len + 0.08, h: 0.06, pad: 40, label: 'Q', top: true, a: ringA('q', 0.5) });
      if (s.key === 'k') rings.push({ at: vecAt(A.ko), w: A.ko.len + 0.08, h: 0.06, pad: 40, label: 'K', top: true, a: ringA('k', 0.5), cls: 'amber' });
      if (s.key === 'v') rings.push({ at: vecAt(A.vo), w: A.vo.len + 0.08, h: 0.06, pad: 40, label: 'V', top: true, a: ringA('v', 0.5) });
      if (s.key === 'score') rings.push({ at: beamEnd(dot.key, 'v2'), w: 0.2, h: 0.2, label: q('天空'), cls: 'amber', a: smooth(seg(t, s.t0 + 3.9, s.t0 + 4.4)) * (1 - smooth(seg(t, s.t1 - 0.4, s.t1 - 0.1))) });
      if (s.key === 'softmax') rings.push({ at: beamEnd(dot.key, 'v2'), w: 0.2, h: 0.2, label: `${q('天空')} ${pct(dot.w)}`, cls: 'amber', a: smooth(seg(t, s.t0 + 3.9, s.t0 + 4.4)) * (1 - smooth(seg(t, s.t1 - 0.4, s.t1 - 0.1))) });
      if (s.key === 'mix') rings.push({ at: beamEnd(dot.key, 'v0'), w: 0.2, h: 0.2, label: '新的理解', a: smooth(seg(t, s.t0 + 3.9, s.t0 + 4.4)) * (1 - smooth(seg(t, s.t1 - 0.4, s.t1 - 0.1))) });
      const P = M.mats.mlp;
      if (s.key === 'gate') rings.push({ at: () => (P && P.visible ? P.din.localToWorld(v3(0, P.din.len / 2, 0)) : null), w: 0.06, h: P.din.len, pad: 34, label: '门控后的输出', a: ringA('gate', 0.6) });
      if (s.key === 'down') rings.push({ at: () => (P && P.visible ? P.dout.localToWorld(v3(P.dout.len / 2, 0, 0)) : null), w: P.dout.len + 0.08, h: 0.06, pad: 40, label: '加回去', top: true, a: ringA('down', 0.8) });
      if (s.key === 'add1') rings.push({ at: () => M.exAdd1.getWorldPosition(v3(0, 0, 0)), w: 0.26, h: 0.26, label: '⊕', a: ringA('add1', 0.4, 4.4) });
      // 公式板
      let formula = null;
      if (t >= fFirst - 0.4 && t < fLast + 0.6 && s.key !== 'heads') {
        const fk = formKeys.includes(s.key) ? s.key : 'score';
        formula = { ...FORM[fk], a: smooth(seg(t, fFirst, fFirst + 0.6)) * (1 - smooth(seg(t, fLast - 0.1, fLast + 0.5))) };
      }
      return {
        st,
        iso: smooth(seg(lt, 0, 2)) * (s.key === 'over' ? 0.55 : 0.85),
        cam: () => {
          const live = handheld(dCam[ck](M, clamp((t - span.t0) / (span.t1 - span.t0)), st, t), t, 0.0025);
          let out = live;
          const bl = BLEND[ck];
          if (bl && t - span.t0 < bl) {
            const prevKey = s.key === 'over' ? null : camKey[D[D.findIndex((x) => x.t0 === span.t0) - 1].key];
            const from = prevKey ? endCam.get(`d-${prevKey}`) : prevCam('layers1', live);
            if (from) out = blendCam(from, live, smoother((t - span.t0) / bl));
          }
          endCam.set(`d-${ck}`, out);
          return out;
        },
        hide: [...(['score', 'soft', 'heads', 'dot'].includes(ck) ? ['attnMats'] : []), ...(ck === 'bits' ? ['exDeco', 'mlpVecs'] : [])],
        attnFocus: s.key === 'q' ? 'q' : s.key === 'k' ? 'k' : s.key === 'v' ? 'v' : null,
        attnFocusK: ['q', 'k', 'v'].includes(s.key) ? smooth(seg(t, Dk('q').t0, Dk('q').t0 + 0.6)) * (1 - smooth(seg(t, Dk('v').t1 - 0.5, Dk('v').t1))) : 0,
        beamLabels: !['score'].includes(s.key),
        exOnly: s.key === 'over' ? undefined : (s.s?.op ?? null),
        hideMlpLabels: ['up', 'act', 'gate', 'down'].includes(s.key),
        flipBit: t >= flipT0 && t < flipT1 ? FLIP : null,
        boardA: BOARD.includes(ck) ? smooth(seg(t, span.t0 + 0.05, span.t0 + 0.45)) * (1 - smooth(seg(t, span.t1 - 0.35, span.t1 - 0.02))) : 0,
        shiftX: 430 * boardShift(t),
        dof: ck === 'bits' ? { range: 0.5, blur: 6 } : ['dot', 'mm', 'neuron'].includes(ck) ? { range: 1.6, blur: 3.5 } : null,
        ov: {
          formula,
          rings,
          heads6: s.key === 'heads' ? { a: smooth(seg(t, hk.t0 + 0.2, hk.t0 + 0.8)) * (1 - smooth(seg(t, hk.t1 - 0.4, hk.t1 - 0.05))), k: seg(t, hk.t0 + 0.3, hk.t0 + 3.6), list: HEADS, on: t < hk.t0 + 4.0 ? null : hOn, L: LX, g: G0, sky: '天空' } : null,
        },
      };
    });
  }

  /* ------------------------------------------------------------ ⑥ 第 1 个词：输出与采样 */
  const samp = (t0, beats) => lay(t0, [
    { depth: 4, s: { ph: 'head', sub: 'norm' }, d: beats[0] * BEAT },
    ...(beats[1] ? [
      { depth: 5, s: { ph: 'head', sub: 'unembed', mi: 'pick' }, d: beats[1] * BEAT * 0.25 },
      { depth: 5, s: { ph: 'head', sub: 'unembed', mi: 'mul' }, d: beats[1] * BEAT * 0.5 },
      { depth: 5, s: { ph: 'head', sub: 'unembed', mi: 'sum' }, d: beats[1] * BEAT * 0.25 },
    ] : []),
    { depth: 4, s: { ph: 'head', sub: 'softmax' }, d: beats[2] * BEAT },
    { depth: 4, s: { ph: 'sample', sub: 'temp' }, d: beats[3] * BEAT },
    { depth: 4, s: { ph: 'sample', sub: 'topk' }, d: beats[4] * BEAT },
    { depth: 4, s: { ph: 'sample', sub: 'topp' }, d: beats[5] * BEAT },
    { depth: 4, s: { ph: 'sample', sub: 'draw' }, d: beats[6] * BEAT },
  ]);
  const spk = (list, k) => list.find((x) => (x.s.sub === k && !x.s.mi) || x.s.mi === k);
  const emitOf = (list) => { const d = list[list.length - 1]; return d.t0 + d.d * 0.8; };
  const sampleShot = (name, g, list, prevName, blendIn) => shot(name, (lt, t, { M }) => {
    const s = pick(list, t);
    const st = mst(s.depth, g, s.s, s.p, { dAnim: s.depth });
    const mmA = list.find((x) => x.s.mi === 'pick'), mmZ = list.find((x) => x.s.sub === 'softmax');
    const dur = SEC[name].t1 - SEC[name].t0;
    return {
      st,
      cam: () => {
        // 归一化 / 乘输出矩阵时概率柱还没升起来：镜头先低一点、近一点；之后按这一步最高的概率柱取景
        const up = smooth(seg(t, mmZ.t0 - 0.9, mmZ.t0 + 0.5));
        const hmax = 3.2 * Math.max(Q.steps[g].temps['0.7'][0], Q.steps[g].top[0][1]);
        const head = headCam(M, st, { dz: lerp(7.2, 6.4 + hmax * 1.2 - 0.8 * smooth(lt / dur), up), dx: lerp(1.6, 0.4, smooth(lt / dur)), ly: lerp(0.9, 1.25 + hmax * 0.52, up), lx: lerp(1.8, -0.6, up) });
        let live = head;
        if (mmA) {
          const mmK = smooth(seg(t, mmA.t0, mmA.t0 + 1.1)) * (1 - smooth(seg(t, mmZ.t0, mmZ.t0 + 1.1)));
          if (mmK > 0) {
            const mc = M.micro.camera({ ...st, step: s.s.mi ? st.step : { g, ph: 'head', sub: 'unembed', mi: 'sum' } });
            if (mc) endCam.set(`${name}-mm`, { ...mc, fov: 32 });
            const mcc = endCam.get(`${name}-mm`);
            if (mcc) live = blendCam(head, orbit(mcc, lerp(-6, 6, seg(t, mmA.t0, mmZ.t0)), 0, 1.5), mmK);
          }
        }
        return blendCam(prevCam(prevName, live), live, smoother(seg(lt, 0, blendIn)));
      },
      shiftX: mmA ? 430 * smoother(seg(t, mmA.t0 - 0.2, mmA.t0 + 0.9)) * (1 - smoother(seg(t, mmZ.t0 - 0.7, mmZ.t0 + 0.4))) : 0,
      boardA: mmA ? smooth(seg(t, mmA.t0 + 0.05, mmA.t0 + 0.45)) * (1 - smooth(seg(t, mmZ.t0 - 0.35, mmZ.t0 - 0.02))) : 1,
    };
  });
  const st0 = S(0);
  {
    const T0 = SEC.sample1.t0, T1 = SEC.sample1.t1;
    prog(T0, 5);
    var sp1 = samp(T0, [2, 8, 4, 3, 3, 3, 5]);
    ev(spk(sp1, 'draw').t0 + 0.05, 'dice');
    ev(emitOf(sp1), 'emit', { g: 0, k: 1.2 });
    ev(spk(sp1, 'pick').t0, 'step', { k: 0.6 });
    ev(spk(sp1, 'softmax').t0, 'step', { k: 0.6 });
    const sx = (k) => spk(sp1, k);
    sub(T0 + 0.15, sx('mul').t0 + 0.4, `最后：给 ${m('15')} 万个候选词打分`);
    term(T0 + 0.25, sx('mul').t0 + 0.4, `输出头 · 词表 ${m('151,936')} 个词`);
    sub(sx('mul').t0 + 0.6, sx('sum').t1 - 0.1, `每个分数，也是 ${m('1024')} 项乘加`);
    term(sx('mul').t0 + 0.7, sx('sum').t1 - 0.1, `${q(hm0.token)}的分数 = ${m(hm0.total.toFixed(2))}`);
    sub(sx('softmax').t0 + 0.1, sx('softmax').t1 - 0.1, `分数变成概率：${q(st0.top[0][2])} ${m(pct(st0.top[0][1]))}`);
    term(sx('softmax').t0 + 0.2, sx('softmax').t1 - 0.1, 'softmax');
    sub(sx('temp').t0 + 0.05, sx('temp').t1 - 0.1, '温度：把概率拉尖，或者抹平');
    term(sx('temp').t0 + 0.1, sx('temp').t1 - 0.1, `温度 T = ${m('0.7')}`);
    sub(sx('topk').t0 + 0.05, sx('topk').t1 - 0.1, `只留前 ${m('20')} 名`);
    term(sx('topk').t0 + 0.1, sx('topk').t1 - 0.1, `top-k = ${m('20')}`);
    sub(sx('topp').t0 + 0.05, sx('topp').t1 - 0.1, `从高到低，凑满 ${m('80%')} 就停`);
    term(sx('topp').t0 + 0.1, sx('topp').t1 - 0.1, `top-p = ${m('0.8')}`);
    sub(sx('draw').t0 + 0.05, T1 - 0.15, `按概率抽签：抽中${q(st0.chosenS)}`);
    term(sx('draw').t0 + 1.6, T1 - 0.15, `随机数 u = ${m(st0.u.toFixed(3))}`); // 等舞台上的随机数停下来再给数字
    sampleShot('sample1', 0, sp1, 'dissect', 1.0);
  }

  /* ------------------------------------------------------------ 自回归：第 2 – 14 个词 */
  // 每个词元：读入 → 嵌入 → 28 层 → 输出头 → 采样
  const PH = [{ ph: 'read', d: 0.2 }, { ph: 'embed', d: 0.1 }, { ph: 'layers', d: 0.32 }, { ph: 'head', d: 0.12 }, { ph: 'sample', d: 0.26 }];
  const emitFrac = 1 - 0.26 * 0.2;
  const toks = (g0, t0, durs) => { let t = t0; return durs.map((d, k) => { const a = { g: g0 + k, t0: t, t1: t + d, d, emit: t + d * emitFrac }; t += d; return a; }); };
  const loopState = (list, t) => {
    const s = list.find((x) => t < x.t1) || list[list.length - 1];
    const u = clamp((t - s.t0) / s.d);
    let acc = 0;
    for (const ph of PH) { if (u < acc + ph.d || ph === PH[PH.length - 1]) return { g: s.g, ph: ph.ph, p: clamp((u - acc) / ph.d) }; acc += ph.d; }
    return null;
  };
  // 自回归：从机器右前方斜着看，新词元沿右侧的回路落回托盘、在新的一列里往上算，KV 缓存一格格变长
  const loopCam = (M, fx, t, o = {}) => {
    const look = v3(fx - (o.back ?? 3), o.ly ?? 5.6, 0);
    return orbit({ pos: look.clone().add(v3(0, o.dy ?? 3.6, o.dz ?? 23)), look, fov: 32 }, o.yaw ?? 28, 0, 1);
  };
  {
    const T0 = SEC.loop1.t0, T1 = SEC.loop1.t1;
    // 时长都取拍子的整数倍，词元吐出来的时刻正好落在节拍上（配乐里的铃声跟着它）
    const D1 = 12 * BEAT;
    const loop1 = toks(1, T0, [D1, ...Array.from({ length: 12 }, () => BEAT)]);
    loop1.forEach((s) => ev(s.emit, 'emit', { g: s.g, k: s.g === 1 ? 1.1 : 0.7, rank: Q.steps[s.g].chosenRank }));
    const sampleAt = T0 + D1 * 0.74;
    ev(sampleAt + 0.05, 'dice');
    const st1 = S(1);
    sub(T0 + 0.3, T0 + 2.9, `${q(st0.chosenS)}接回末尾，整套再算一遍`);
    term(T0 + 0.4, T0 + 2.9, '自回归：一次只写一个字');
    sub(T0 + 3.1, sampleAt + 0.1, `第 2 个字：${q(st1.top[0][2])} ${m(pct(st1.top[0][1]))}，${q(st1.top[1][2])} ${m(pct(st1.top[1][1]))}`);
    sub(sampleAt + 0.3, loop1[1].t0 + 1.0, `抽中了${q(st1.chosenS)}：不是第一名`);
    term(sampleAt + 0.4, loop1[1].t0 + 1.0, `随机数 u = ${m(st1.u.toFixed(3))}`);
    sub(loop1[1].t0 + 1.2, T1 - 3.0, '算过的 K、V 都存着，不用重算');
    term(loop1[1].t0 + 1.3, T1 - 3.0, 'KV 缓存');
    sub(T1 - 2.8, T1 - 0.15, '一个字，一个字，往下写……');
    shot('loop1', (lt, t, { M }) => {
      const s = loopState(loop1, t);
      const detail = s.g === 1 && (s.ph === 'head' || s.ph === 'sample');
      const st = detail ? mst(4, 1, { ph: s.ph, sub: s.ph === 'head' ? 'softmax' : 'draw' }, s.p, { dAnim: 2.7 }) : mst(2, s.g, { ph: s.ph }, s.p, { dAnim: 2.7 });
      const done = loop1.filter((x) => t >= x.emit);
      const newest = done[done.length - 1];
      return {
        st,
        cam: () => {
          const wide = loopCam(M, lerp(M.x(Q.row(1)), M.x(Q.row(13)), smooth(seg(t, T0, T1))), t, { yaw: lerp(34, 20, smooth(lt / 15)), dz: lerp(21, 25, smooth(lt / 15)) });
          const h1 = 3.2 * Q.steps[1].temps['0.7'][0];
          const close = headCam(M, mst(4, 1, { ph: 'sample', sub: 'draw' }, 0), { dz: 6.4 + h1 * 1.2, ly: 1.25 + h1 * 0.52, lx: -0.6 });
          const kc = smooth(seg(t, T0 + D1 * 0.5, T0 + D1 * 0.64)) * (1 - smooth(seg(t, loop1[1].t0 + 0.2, loop1[1].t0 + 1.6)));
          const live = blendCam(wide, close, kc);
          return blendCam(prevCam('sample1', live), live, smoother(seg(lt, 0, 2.2)));
        },
        hide: detail ? [] : ['bars'],
        ov: { reply: { a: smooth(seg(lt, 0.2, 0.8)), n: 1 + done.length, k: newest ? smooth(seg(t, newest.emit, newest.emit + 0.35)) : 1 } },
      };
    });
  }

  /* ------------------------------------------------------------ 第 15 个词「散」：再穿过 28 层 */
  const stK = S(GK);
  {
    const T0 = SEC.layersK.t0, T1 = SEC.layersK.t1;
    prog(T0, 2);
    const schedK = lay(T0, [
      { s: { ph: 'read' }, d: BEAT * 1.2 }, { s: { ph: 'embed' }, d: BEAT * 0.8 },
      ...Array.from({ length: 20 }, (_, k) => ({ L: k, d: BEAT / 2 })),
      ...Array.from({ length: 5 }, (_, k) => ({ L: 20 + k, d: BEAT })),
      { L: 25, d: 1.5 * BEAT }, { L: 26, d: 1.5 * BEAT }, { L: 27, d: 4 * BEAT },
    ]);
    const lk = (Lx) => schedK.find((s) => s.L === Lx);
    schedK.forEach((s) => { if (s.L != null) ev(s.t0, 'layer', { L: s.L, k: s.d > BEAT * 1.5 ? 1 : 0.6 }); });
    ev(lk(23).t0, 'reveal', { k: 0.6 });
    ev(lk(27).t0, 'reveal', { k: 1.2 });
    const reflect = lens(GK, 23)[0];
    sub(T0 + 0.3, lk(20).t0 - 0.1, `第 ${m(GK + 1)} 个字：同样的 28 层，再走一遍`);
    sub(lk(20).t0 + 0.05, lk(25).t0 - 0.1, `第 23 层，它想说${q(reflect[2])} ${m(pct(reflect[1]))}`);
    term(lk(20).t0 + 0.15, lk(27).t0 - 0.1, '逻辑透镜');
    sub(lk(25).t0 + 0.05, lk(27).t0 - 0.1, `后来又变成${q(lensTop(GK, 25)[2])}、${q(lensTop(GK, 26)[2])}……`);
    sub(lk(27).t0 + 0.05, T1 - 0.1, `最后一层，${q(lensTop(GK, 27)[2])}才排到第一：${m(pct(lensTop(GK, 27)[1]))}`);
    shot('layersK', (lt, t, { M }) => {
      const s = pick(schedK, t);
      const st = s.L == null ? mst(3, GK, s.s, s.p, { dAnim: 3 }) : mst(3, GK, { ph: 'layer', L: s.L }, s.p, { dAnim: 3 });
      const Ls = [[T0, 0], [lk(0).t0, 0], [lk(20).t0, 18.5], [lk(25).t0, 23.6], [T1, 26.8]];
      const Lc = pchip(Ls.map((a) => a[0]), Ls.map((a) => a[1]))(t);
      return {
        st,
        cam: () => {
          const push = 0.18 * bumpAt(t, lk(23).t0, lk(23).d) + 0.3 * bumpAt(t, lk(27).t0, lk(27).d + 0.5);
          const live = handheld(towerCam(M, Lc, GK, { lx: -1.6, dx: -5.2, dy: 2.0, dz: 9.0, yaw: lerp(6, -8, smooth(lt / 15)), dist: 1 - push }), t, 0.003);
          return blendCam(prevCam('loop1', live), live, smoother(seg(lt, 0, 2.0)));
        },
        lensWin: 9,
        ov: {
          lens: { a: smooth(seg(t, T0 + 3.7, T0 + 4.4)) * (1 - smooth(seg(t, T1 - 0.5, T1))), g: GK, upto: s.L == null ? -0.01 : s.L + s.p - 0.0001, other: reflect[2], side: 'right' },
          reply: { a: 1 - smooth(seg(lt, 0, 0.8)), n: GK, k: 1 },
        },
      };
    });
  }

  /* ------------------------------------------------------------ 「散」：输出与采样 */
  {
    const T0 = SEC.sampleK.t0, T1 = SEC.sampleK.t1;
    prog(T0, 5);
    const spK = samp(T0, [1, 0, 4, 2, 1, 3, 5]);
    ev(spk(spK, 'draw').t0 + 0.05, 'dice');
    ev(emitOf(spK), 'emit', { g: GK, k: 1.4 });
    sub(T0 + 0.2, spk(spK, 'temp').t0 - 0.05, stK.top.slice(0, 3).map((x) => `${q(x[2])} ${m(pct(x[1]))}`).join('，'));
    term(T0 + 0.3, spk(spK, 'temp').t0 - 0.05, '概率前三名');
    sub(spk(spK, 'temp').t0 + 0.1, spk(spK, 'draw').t0 - 0.05, `截断之后，只剩 ${m(stK.pool.length)} 个候选`);
    term(spk(spK, 'temp').t0 + 0.2, spk(spK, 'draw').t0 - 0.05, `候选池 ${stK.pool.map((x) => m(pct(x[1]))).join(' / ')}`);
    sub(spk(spK, 'draw').t0 + 0.05, T1 - 0.1, `抽中${q(stK.chosenS)}：差一点就是别的字`);
    term(spk(spK, 'draw').t0 + 0.1, T1 - 0.1, `随机数 u = ${m(stK.u.toFixed(3))}`);
    sampleShot('sampleK', GK, spK, 'layersK', 1.6);
  }

  /* ------------------------------------------------------------ 自回归写完 */
  {
    const T0 = SEC.loop2.t0, T1 = SEC.loop2.t1;
    const rest = Q.G - (GK + 1);
    const durs2 = Array.from({ length: rest }, (_, k) => (k === rest - 1 ? 0 : k < 4 ? 1.5 : 0.75) * BEAT);
    durs2[rest - 1] = T1 - T0 - durs2.reduce((a, b) => a + b, 0);
    const loop2 = toks(GK + 1, T0, durs2);
    loop2.forEach((s) => ev(s.emit, s.g === Q.G - 1 ? 'end' : 'emit', { g: s.g, k: 0.55, rank: Q.steps[s.g].chosenRank }));
    const lowRank = Q.steps.filter((x) => x.chosenRank > 0).length;
    const stEnd = S(Q.G - 1);
    const lastT = loop2[loop2.length - 1].t0;
    sub(T0 + 0.3, T0 + 4.0, '剩下的字，一个接一个写出来');
    sub(T0 + 4.3, lastT - 3.1, `红线标出的 ${m(lowRank)} 个字，都不是当时的第一名`);
    term(T0 + 4.4, lastT - 3.1, '下划线长度 = 选中时的概率');
    sub(lastT - 2.9, lastT - 0.15, '所以，每次回答都可能不一样');
    sub(lastT + 0.1, T1 - 0.1, '选中结束标记：回答写完了');
    term(lastT + 0.2, T1 - 0.1, `${m('&lt;|im_end|&gt;')} ${m(pct(stEnd.chosenP1))}`);
    shot('loop2', (lt, t, { M }) => {
      const s = loopState(loop2, t);
      const done = loop2.filter((x) => t >= x.emit);
      const newest = done[done.length - 1];
      return {
        st: mst(2, s.g, { ph: s.ph }, s.p, { dAnim: 2.7 }),
        cam: () => {
          const live = loopCam(M, lerp(M.x(Q.row(GK + 1)), M.x(Q.row(Q.G - 1)), smooth(seg(t, T0, T1 - 2))), t, { yaw: lerp(18, 30, smooth(lt / 17.5)), dz: lerp(24, 28, smooth(lt / 17.5)), back: lerp(3, 6, smooth(lt / 17.5)) });
          return blendCam(prevCam('sampleK', live), live, smoother(seg(lt, 0, 2.2)));
        },
        hide: ['bars'],
        ov: { reply: { a: 1, n: Math.min(Q.G, GK + 1 + done.length), k: newest ? smooth(seg(t, newest.emit, newest.emit + 0.3)) : 1 } },
      };
    });
  }

  /* ------------------------------------------------------------ 片尾 */
  // 一共做了多少次乘加：每层的矩阵乘法 + 注意力里的 Q·K 和加权求和 + 输出头
  const Mo = Q.M, perTokLayer = Mo.hidden * (Mo.heads * Mo.headDim) * 2 + Mo.hidden * (Mo.kvHeads * Mo.headDim) * 2 + Mo.hidden * Mo.ffn * 3;
  let macs = 0;
  for (let pos = 0; pos < Q.P + Q.G - 1; pos++) macs += NL * (perTokLayer + 2 * Mo.heads * Mo.headDim * (pos + 1));
  macs += Q.G * Mo.hidden * Mo.vocab;
  const statLine = `${Q.G} 个词元 · 每个都走完 28 层 · 约 ${Math.round(macs / 1e8)} 亿次乘加`;
  {
    const T0 = SEC.end.t0, T1 = SEC.end.t1;
    // 0–9 s 完整回答；9.6 s 起「继续探索」：一句引子 → 站内另外几页各一行 → 延伸学习 + 二维码；配乐在这里收尾
    const E1 = 9.6, E2 = 13.6, E3 = 19.6;
    ev(T0 + 1.6, 'hit', { k: 0.5 });
    ev(T0 + E1, 'reveal', { k: 0.5 });
    ev(T0 + E2, 'step', { k: 0.4 });
    ev(T0 + E3, 'end', { k: 0.4 });
    shot('end', (lt, t, { M }) => ({
      st: mst(2, Q.G - 1, { ph: 'sample' }, 1, { dAnim: lerp(2.7, 1, smoother(seg(lt, 0.4, 3.9))), view: 'machine' }),
      cam: () => {
        const far = cam([-11, 5.5, 70], [0, 11.5, 0], 30);
        const live = loopCam(M, M.x(Q.row(Q.G - 1)), t, { yaw: 30, dz: 28, back: 6 });
        return blendCam(prevCam('loop2', live), far, smoother(seg(lt, 0, 10)));
      },
      hide: ['bars'],
      dof: { focus: 14, range: 24, blur: 6 * smooth(seg(lt, 2.5, 5)) },
      fade: lerp(0, 0.86, smooth(seg(lt, 0.6, 2.0))) + 0.14 * smooth(seg(t, T1 - 1.2, T1 - 0.05)),
      ov: {
        reply: { a: 1 - smooth(seg(lt, 0.4, 1.4)), n: Q.G, k: 1 },
        band: 1 - smooth(seg(lt, 0, 1)),
        end: {
          a1: smooth(seg(lt, 1.4, 2.4)) * (1 - smooth(seg(lt, E1 - 0.8, E1 - 0.1))), k1: smooth(seg(lt, 4.0, 5.0)),
          a3: smooth(seg(lt, E1, E1 + 0.8)) * (1 - smooth(seg(lt, E2 - 0.6, E2 - 0.05))),
          a4: smooth(seg(lt, E2, E2 + 0.5)) * (1 - smooth(seg(lt, E3 - 0.6, E3 - 0.05))), k4: seg(lt, E2 + 0.2, E2 + 3.2),
          a5: smooth(seg(lt, E3, E3 + 0.8)) * (1 - smooth(seg(t, T1 - 0.8, T1 - 0.05))), k5: seg(lt, E3, E3 + 1.6),
        },
      },
    }));
  }

  function frame(t, ctx) {
    t = clamp(t, 0, end - 1e-6);
    const i = shots.findIndex((s) => t >= s.t0 && t < s.t1);
    const s = shots[i];
    const r = s.fn(t - s.t0, t, ctx);
    const camFn = typeof r.cam === 'function' ? r.cam : () => r.cam;
    r.cam = () => { const c = camFn(); endCam.set(s.name, c); return c; };
    r.shot = s.name;
    r.ov = r.ov || {};
    return r;
  }

  progs.sort((a, b) => a.t - b.t);
  const progAt = (t) => {
    let i = -1;
    for (const p of progs) if (t >= p.t) i = p.i;
    const a = progs.length ? smooth(seg(t, progs[0].t, progs[0].t + 0.6)) * (1 - smooth(seg(t, SEC.end.t0, SEC.end.t0 + 1.0))) : 0;
    return { i, a: i < 0 ? 0 : a };
  };
  return { end, frame, subs, strips, chapters, cards, terms, chapterNames: CHAPTERS, progAt, events, sections, open: OPEN, statLine, bpm: BPM, shots: shots.map((s) => ({ name: s.name, t0: s.t0, t1: s.t1 })) };
}
