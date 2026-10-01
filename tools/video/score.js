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
  ['cold', 5, 0.12], ['title', 3, 0.35], ['tokenize', 5, 0.45], ['embed', 3, 0.5], ['layers1', 10, 0.78],
  ['dissect', 35, 0.55], ['sample1', 6, 0.6], ['loop1', 6, 0.72], ['layersK', 6, 0.85], ['sampleK', 4, 0.7],
  ['loop2', 7, 0.9], ['end', 6, 0.22],
];

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

  /* ------------------------------------------------------------ 冷开场 */
  const qTimes = user.map((_, k) => B(0, 3) + k * BEAT);
  qTimes.forEach((t, i) => ev(t, 'type', { i }));
  sub(B(2, 3), B(3, 3) + 0.2, '你问它一个问题。');
  sub(B(3, 3) + 0.4, B(5) - 0.15, '在它开口之前，黑箱里发生了什么？');
  const coldCam = path([{ t: 0, p: [-15, 3.0, 74], l: [0, 5.8, 0], fov: 30 }, { t: B(5), p: [-9, 4.0, 58], l: [0, 5.9, 0], fov: 30 }]);
  shot('cold', (lt, t) => ({
    st: mst(1, 0, { ph: 'pass' }, 0.04 + 0.42 * smooth(seg(t, 9.5, 12.4))),
    cam: () => coldCam(t),
    dof: { focus: 14, range: 18, blur: 9 * (1 - smooth(seg(t, 10.5, 12.5))) },
    fade: t < 4.5 ? lerp(1, 0.62, smooth(seg(t, 0.4, 4.5))) : lerp(0.62, 0.35, seg(t, 4.5, 12.4)),
    ov: { band: 0.5, qline: smooth(seg(t, qTimes[0] - 0.6, qTimes[0])) * (1 - smooth(seg(t, 11.5, 12.35))), qScale: 1 + 0.05 * smooth(seg(t, 11.5, 12.35)), qBlur: 7 * smooth(seg(t, 11.5, 12.35)) },
  }));

  /* ------------------------------------------------------------ 片名 */
  ev(SEC.title.t0, 'hit', { k: 1 });
  const titleCam = (lt) => {
    const k = smooth(lt / 7.5);
    const yaw = THREE.MathUtils.degToRad(lerp(-34, -13, k)), R = lerp(50, 45, k), h = lerp(2.4, 3.8, k);
    const look = v3(0, 5.6, 0);
    return { pos: look.clone().add(v3(R * Math.sin(yaw), h - 5.6, R * Math.cos(yaw))), look, fov: 32 };
  };
  shot('title', (lt) => ({
    st: mst(1, 0, { ph: 'pass' }, 0.04),
    cam: () => titleCam(lt),
    fade: 0.38 * (1 - smooth(seg(lt, 0, 0.6))) + 0.25,
    ov: { band: 0, title: smooth(seg(lt, 0.1, 1.0)) * (1 - smooth(seg(lt, 6.3, 7.3))), titleK: seg(lt, 0.05, 3.4), titleBlur: 7 * smooth(seg(lt, 6.3, 7.3)) },
    dof: { focus: 20, range: 30, blur: 3 },
  }));

  /* ------------------------------------------------------------ 01 揭开 + 分词 */
  {
    const T0 = SEC.tokenize.t0, T1 = SEC.tokenize.t1;
    chapter(T0 + 0.3, T1 - 0.2, '01', '分词', 'TOKENIZE');
    ev(T0, 'open');
    ev(T0 + BAR, 'whoosh', { k: 0.6 });
    var dropT0 = T0 + 3.0, dropD = 4.0;
    var landT = (i) => dropT0 + dropD * ((i / Q.P) * 0.7 + 0.3);
    for (let i = 0; i < Q.P; i++) ev(landT(i), 'tick', { i, k: Q.tokens[i].role === 'user' ? 1 : 0.45 });
    sub(T0 + 0.4, T0 + 2.4, '第一步：分词。');
    sub(dropT0, landT(Q.P - 1) - 0.3, `系统提示、你的问题和几个特殊标记，拼成 ${m(Q.P)} 个词元。`);
    sub(landT(Q.P - 1) + 0.1, T1 - 0.2, `模型看到的不是文字，是编号：${q(user[3].s)} = ${m(user[3].id)}。`);
    strip(landT(Q.P - 1) + 0.1, T1 - 0.2, user.map((t) => `<span class="tk">${tk(t.s)}</span> ${m(t.id)}`).join('<span class="sep"></span>'));
    var tokCamKeys = (M) => {
      const tr = (i) => ({ p: [X(M, i) - 3.0, 1.75, 4.4], l: [X(M, i) + 0.9, 0.22, 0] });
      const uc = (X(M, user[0].i) + X(M, user[user.length - 1].i)) / 2;
      return [
        { t: 2.5, ...tr(0), fov: 30 },
        { t: landT(0) - T0 - 0.1, ...tr(0), fov: 30 },
        { t: landT(19) - T0, ...tr(19), fov: 30 },
        { t: landT(38) - T0, ...tr(38), fov: 30 },
        { t: 7.6, p: [uc - 1.5, 1.85, 4.9], l: [uc + 0.25, 0.2, 0], fov: 30 },
        { t: 12.5, p: [uc - 0.9, 1.45, 3.9], l: [uc + 0.25, 0.18, 0], fov: 30 },
      ];
    };
    let tokPath = null, openPath = null;
    shot('tokenize', (lt, t, { M }) => ({
      st: mst(2, 0, { ph: 'read' }, seg(t, dropT0, dropT0 + dropD), { dAnim: lerp(1, 2.6, smoother(seg(lt, 0, 2.4))) }),
      cam: () => {
        // 前 2.5 秒看机箱打开；卡在小节线上硬切到托盘的低机位跟拍
        if (lt < 2.5) { const T = titleCam(7.5); openPath ||= path([{ t: 0, p: T.pos.toArray(), l: T.look.toArray(), fov: 32 }, { t: 2.5, p: [-6.5, 7.8, 39], l: [-1.0, 4.6, 0], fov: 32 }]); return openPath(lt); }
        tokPath ||= path(tokCamKeys(M)); return tokPath(lt);
      },
      extras: { ids: smooth(seg(t, landT(0), landT(0) + 0.5)), idsFocus: t > landT(Q.P - 1) + 0.1 },
    }));
  }

  /* ------------------------------------------------------------ 02 嵌入 */
  {
    const T0 = SEC.embed.t0, T1 = SEC.embed.t1;
    chapter(T0 + 0.3, T1 - 0.2, '02', '嵌入', 'EMBEDDING');
    const k0 = T0 + 0.6, k1 = T0 + 4.0; // 从表里取向量（光点落下）
    for (let i = 0; i < Q.P; i += 2) ev(k0 + (k1 - k0) * ((i / Q.P) * 0.5 + 0.5), 'blip', { i });
    sub(T0 + 0.4, T0 + 3.8, `每个编号去嵌入表里，取出自己那一行：${m('1024')} 个数。`);
    sub(T0 + 4.1, T1 - 0.2, `从此，每个词元都是一个 ${m('1024')} 维的向量。`);
    strip(T0 + 0.6, T1 - 0.2, `嵌入表 ${m('151,936 × 1,024')}<span class="sep"></span>${m('155,582,464')} 个参数<span class="sep"></span>${q(user[0].s)}在第 ${m(user[0].id)} 行`);
    let embPath = null;
    shot('embed', (lt, t, { M }) => {
      const keys = tokCamKeys(M);
      const last = keys[keys.length - 1];
      const uc = (X(M, user[0].i) + X(M, user[user.length - 1].i)) / 2;
      embPath ||= path([
        { t: 0, p: last.p, l: last.l, fov: 30 },
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

  /* ------------------------------------------------------------ 03 第 1 个词：穿过 28 层 */
  const towerCam = (M, Lc, g, o = {}) => {
    const y = 0.95 + Lc * 0.26;
    const xf = M.x(Q.row(g));
    const look = v3(xf + (o.lx ?? -1.2), y - 0.25, -0.2);
    return orbit({ pos: look.clone().add(v3(o.dx ?? 6.6, o.dy ?? 3.4, o.dz ?? 8.6)), look, fov: 32 }, o.yaw ?? 0, 0, o.dist ?? 1);
  };
  const bumpAt = (t, t0, d) => smooth(seg(t, t0 - 0.15, t0 + 0.35)) * (1 - smooth(seg(t, t0 + d - 0.3, t0 + d + 0.6)));
  {
    const T0 = SEC.layers1.t0, T1 = SEC.layers1.t1;
    chapter(T0 + 0.3, T1 - 0.2, '03', '穿过 28 层', 'TRANSFORMER LAYERS');
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
    sub(T0 + 0.3, T0 + 3.4, '然后，穿过 28 层 Transformer。');
    sub(T0 + 3.6, l1(17).t0 - 0.15, '每过一层，用「逻辑透镜」偷看：此刻开口，会说什么？');
    sub(l1(17).t0 + 0.05, l1(21).t0 - 0.1, '前十几层，还说不出像样的词。');
    sub(l1(21).t0 + 0.05, l1(24).t0 - 0.1, `第 21 层：${q(lensTop(0, 21)[2])} ${m(pct(lensTop(0, 21)[1]))}，想直接回答「为什么」。`);
    sub(l1(24).t0 + 0.05, l1(25).t0 - 0.05, `第 24 层，它改口了：${q(lensTop(0, 24)[2])} ${m(pct(lensTop(0, 24)[1]))}。`);
    sub(l1(25).t0 + 0.05, dive0 - 0.2, `最后一层：${q(lensTop(0, 27)[2])} ${m(pct(lensTop(0, 27)[1]))}。先复述问题，再解释原因。`);
    sub(dive0, T1 - 0.15, `第 ${LX} 层发生了什么？停下来，拆开看看。`);
    for (const s of sched1) {
      const top = lensTop(0, s.L);
      strip(s.t0, s.t1 + (s.L === 27 ? dive0 - hold1 - 0.1 : 0), `<span class="k">L${String(s.L).padStart(2, '0')}</span><span class="v">此刻开口：${q(top[2])} ${m(pct(top[1]))}</span>`, { fin: 0.06, fout: 0.06 });
    }
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
        ov: { lens: { a: smooth(seg(t, T0 + 3.6, T0 + 4.3)) * (1 - smooth(seg(t, dive0, dive0 + 0.8))), g: 0, upto: before ? -0.01 : s.L + (s.L === 27 ? 1 : s.p) - 0.0001, other: lensTop(0, 21)[2], side: 'left' }, corner: { g: 0, L: before ? null : s.L }, cornerA: smooth(seg(lt, 0, 0.6)) },
      };
    });
  }

  /* ------------------------------------------------------------ 04 拆开第 24 层 */
  const headCam = (M, st, o = {}) => {
    const hx = M.headX(st);
    const look = v3(hx + (o.lx ?? -0.4), M.yTop + (o.ly ?? 2.75), 0.1);
    return { pos: look.clone().add(v3(o.dx ?? 0.9, o.dy ?? 1.0, o.dz ?? 10.0)), look, fov: 32 };
  };
  {
    const T0 = SEC.dissect.t0, T1 = SEC.dissect.t1;
    chapter(T0 + 0.3, T1 - 0.2, '04', `拆开第 ${LX} 层`, `INSIDE LAYER ${LX}`);
    const D = lay(T0, [
      { key: 'over', depth: 4, s: { op: 'ln1' }, d: 2 * BAR, p0: true },
      { key: 'ln1', depth: 4, s: { op: 'ln1' }, d: 2 * BAR },
      { key: 'qkv', depth: 5, s: { op: 'attn', sub: 'qkv' }, d: 3 * BAR },
      { key: 'score', depth: 5, s: { op: 'attn', sub: 'score' }, d: BAR },
      { key: 'softmax', depth: 5, s: { op: 'attn', sub: 'softmax' }, d: BAR },
      { key: 'heads', depth: 5, s: { op: 'attn', sub: 'softmax' }, d: 3 * BAR },
      { key: 'dmul', depth: 6, s: { op: 'attn', sub: 'score', mi: 'mul' }, d: 5 * BEAT },
      { key: 'dsum', depth: 6, s: { op: 'attn', sub: 'score', mi: 'sum' }, d: 4 * BEAT },
      { key: 'dscale', depth: 6, s: { op: 'attn', sub: 'score', mi: 'scale' }, d: 3 * BEAT },
      { key: 'mix', depth: 5, s: { op: 'attn', sub: 'mix' }, d: 6 * BEAT },
      { key: 'add1', depth: 5, s: { op: 'add1' }, d: 6 * BEAT },
      { key: 'up', depth: 5, s: { op: 'mlp', sub: 'up' }, d: 6 * BEAT },
      { key: 'act', depth: 5, s: { op: 'mlp', sub: 'act' }, d: 2 * BAR },
      { key: 'down', depth: 5, s: { op: 'mlp', sub: 'down' }, d: 6 * BEAT },
      { key: 'pick', depth: 6, s: { op: 'mlp', sub: 'up', mi: 'pick' }, d: 4 * BEAT },
      { key: 'mul', depth: 6, s: { op: 'mlp', sub: 'up', mi: 'mul' }, d: 7 * BEAT },
      { key: 'sum', depth: 6, s: { op: 'mlp', sub: 'up', mi: 'sum' }, d: 5 * BEAT },
      { key: 'silu', depth: 6, s: { op: 'mlp', sub: 'act', mi: 'silu' }, d: 6 * BEAT },
      { key: 'gate', depth: 6, s: { op: 'mlp', sub: 'act', mi: 'gate' }, d: 6 * BEAT },
      { key: 'bits', depth: 7, s: { op: 'mlp', sub: 'up', mi: 'mul' }, d: 4 * BAR },
      { key: 'out', depth: 4, s: null, d: BAR },
    ]);
    const Dk = (k) => D.find((x) => x.key === k);
    const hk = Dk('heads'), CYCLE = 3.2; // 16 个头先轮一遍，再停在第 6 头
    D.forEach((x) => { if (['ln1', 'qkv', 'score', 'mix', 'add1', 'up', 'act', 'down', 'pick', 'silu', 'gate'].includes(x.key)) ev(x.t0, 'step', { k: 0.6 }); });
    ev(Dk('over').t0, 'hit', { k: 0.8 });
    ev(Dk('dmul').t0, 'hit', { k: 0.5 });
    for (let h = 0; h < 16; h++) ev(hk.t0 + 0.1 + h * (CYCLE / 16), 'tick', { k: 0.4 });
    ev(hk.t0 + 0.2 + CYCLE, 'reveal', { k: 0.6 });
    for (let k = 0; k < 12; k++) ev(Dk('mul').t0 + (k / 12) * Dk('mul').d * 0.94, 'tick', { k: 0.35 });
    for (let k = 0; k < 12; k++) ev(Dk('dmul').t0 + (k / 12) * Dk('dmul').d * 0.94, 'tick', { k: 0.3 });
    for (let k = 0; k < 16; k++) ev(Dk('bits').t0 + 0.8 + k * 0.08, 'bit', { k });
    ev(Dk('out').t0, 'whoosh', { k: 1 });
    const inNorm = Q.norm(LX - 1, row0);
    const wb = neu.wg[0];
    const e2 = Math.floor(Math.log2(Math.abs(wb))), mant = Math.abs(wb) / 2 ** e2;
    const FLIP = 1; // 最高的指数位
    const fb = bf16Bits(wb); fb[FLIP] ^= 1;
    const wFlip = bf16Value(fb).value, gFlip = neu.gz + neu.x[0] * (wFlip - wb);
    const sci = (v) => { const e = Math.floor(Math.log10(Math.abs(v))); return `${(v / 10 ** e).toFixed(1)} × 10<sup>${e}</sup>`; };
    const flipT0 = Dk('bits').t0 + 6.0, flipT1 = Dk('bits').t1 - 0.2;
    ev(flipT0, 'flip');
    sub(Dk('over').t0 + 0.3, Dk('over').t1 - 0.2, `这是第 ${LX} 层：六个步骤，每一层都一样。`);
    sub(Dk('ln1').t0 + 0.2, Dk('ln1').t1 - 0.2, 'RMSNorm：除以均方根，把向量的尺度拉回来。');
    strip(Dk('ln1').t0 + 0.4, Dk('ln1').t1 - 0.2, `进入第 ${LX} 层的向量 ‖x‖ = ${m(inNorm.toFixed(1))}<span class="sep"></span>均方根 ${m((inNorm / 32).toFixed(2))} → ${m('1')}`);
    sub(Dk('qkv').t0 + 0.3, Dk('qkv').t0 + 3.8, '注意力：乘三块权重矩阵，得到 Q、K、V。');
    sub(Dk('qkv').t0 + 4.0, Dk('qkv').t1 - 0.2, 'Q 有 16 个头，K、V 只有 8 个（GQA）。');
    card(Dk('qkv').t0 + 0.6, Dk('qkv').t1 - 0.2, `<span class="k">预填充：${Q.P} 个位置一起算</span><span class="m">h [${Q.P}×1024] @ W<sub>q</sub> [1024×2048] → Q：<b>16</b> 头 × 128</span><br><span class="m">h [${Q.P}×1024] @ W<sub>k</sub> [1024×1024] → K：<b>8</b> 头 × 128</span><br><span class="m">h [${Q.P}×1024] @ W<sub>v</sub> [1024×1024] → V：<b>8</b> 头 × 128</span><br>K、V 存进缓存；下面只看最后一个位置`, { style: 'left:76px;top:150px' });
    sub(Dk('score').t0 + 0.2, Dk('score').t1 - 0.1, '最后一个位置的 Q，和每个 K 做点积。');
    sub(Dk('softmax').t0 + 0.1, Dk('softmax').t1 - 0.1, 'softmax：变成加起来为 1 的权重。');
    sub(hk.t0 + 0.2, hk.t0 + CYCLE + 0.1, `16 个头各看各的，${m(sinkHeads)} 个盯着开头的标记。`);
    sub(hk.t0 + CYCLE + 0.3, hk.t1 - 0.2, `第 ${m(dot.head)} 头：${m(pct(attD[0].w))} 的注意力给了${q(keyTok)}。`);
    strip(hk.t0 + CYCLE + 0.3, hk.t1 - 0.2, `注意力汇：多数头把最多的注意力放在开头的 ${m('&lt;|im_start|&gt;')} 上`);
    sub(Dk('dmul').t0 + 0.2, Dk('dmul').t1 - 0.1, `放大这一次打分：${cx('q')} 和 ${cw('k')} 逐维相乘。`);
    sub(Dk('dsum').t0 + 0.1, Dk('dscale').t1 - 0.15, `加起来 ${cy(f2(dot.sum))}，÷ √128 = ${cy(f2(dot.score))}，softmax 后 ${cy(pct(dot.w))}。`);
    sub(Dk('mix').t0 + 0.2, Dk('mix').t1 - 0.15, `按权重把 V 加起来：${q(keyTok)}的信息搬到了这里。`);
    sub(Dk('add1').t0 + 0.2, Dk('add1').t1 - 0.15, `加回残差流。这一层之后，透镜读到的就是${q(lensTop(0, LX)[2])}。`);
    sub(Dk('up').t0 + 0.2, Dk('up').t1 - 0.15, `前馈 SwiGLU：${m('1024')} 维先扩成 ${m('3072')} 维。`);
    sub(Dk('act').t0 + 0.2, Dk('act').t0 + 2.5, 'SiLU 像一道阀门，决定放行多少。');
    sub(Dk('act').t0 + 2.7, Dk('act').t1 - 0.15, `明显激活 ${m(Q.mlpCount(G0, LX))} 个，最亮的是 #${m(neu.j)}。`);
    sub(Dk('down').t0 + 0.2, Dk('down').t1 - 0.15, `再压回 ${m('1024')} 维，加回残差流。`);
    sub(Dk('pick').t0 + 0.2, Dk('pick').t1 + 0.2, `放大神经元 #${m(neu.j)}：它的 g 是怎么来的？`);
    sub(Dk('pick').t1 + 0.4, Dk('mul').t1 - 0.15, `${cx('1024 个输入 h')}，逐个乘上 ${cw('W_gate 第 ' + neu.j + ' 列')}……`);
    sub(Dk('sum').t0 + 0.1, Dk('sum').t1 - 0.15, `……加起来：${cy('g = ' + neu.gz.toFixed(3))}。`);
    sub(Dk('silu').t0 + 0.2, Dk('silu').t1 - 0.15, `SiLU(${m(neu.gz.toFixed(3))}) = ${m(neu.silu.toFixed(3))}。`);
    sub(Dk('gate').t0 + 0.2, Dk('gate').t1 - 0.15, `再乘 u = ${m(neu.uz.toFixed(3))}：输出 ${cy((neu.silu * neu.uz).toFixed(2))}。`);
    sub(Dk('bits').t0 + 0.2, Dk('bits').t0 + 3.2, '再放大：这个权重在显存里只是 16 个比特。');
    sub(Dk('bits').t0 + 3.4, flipT0 - 0.15, '1 位符号、8 位指数、7 位尾数。');
    strip(Dk('bits').t0 + 3.4, flipT0 - 0.15, `${cw(`W<sub>gate</sub>[${neu.dims[0]}, ${neu.j}]`)} = ${m(`2<sup>${e2}</sup> × ${mant.toFixed(4)} = ${wb.toFixed(4)}`)}`);
    sub(flipT0 + 0.1, flipT1, `假如翻错最高的指数位，g 会变成 ${m(sci(gFlip))}。`);
    strip(flipT0 + 0.1, flipT1, `权重 ${m(wb.toFixed(4))} → ${m(sci(wFlip))}<span class="sep"></span>一位都错不得`);
    sub(Dk('out').t0 + 0.1, Dk('out').t1 - 0.1, `${m('5.96')} 亿个这样的数，算出了下一个词。`);
    const L = (M) => ({ xf: M.x(row0), y: M.yL(LX), e: M.e });
    const dCam = {
      over: (M, p) => { const { xf, y } = L(M); return blendCam(cam([xf + 2.8, y + 2.3, 6.6], [xf + 0.7, y + 1.25, 0.2]), cam([xf + 1.1, y + 1.55, 3.9], [xf + 0.65, y + 1.25, 0.2]), smooth(p)); },
      ln1: (M, p) => { const { xf, y } = L(M); return blendCam(cam([xf - 0.9, y + 0.85, 3.5], [xf + 0.35, y + 0.3, 0.1]), cam([xf - 0.55, y + 0.65, 2.75], [xf + 0.35, y + 0.28, 0.1]), smooth(p)); },
      qkv: (M, p) => { const { xf, y } = L(M); return blendCam(cam([xf + 1.3, y + 1.85, 4.95], [xf + 1.8, y + 1.35, 0.62]), cam([xf + 3.2, y + 1.9, 4.6], [xf + 2.6, y + 1.4, 0.62]), smooth(p)); },
      score: (M, p) => { const { xf, y } = L(M); const kx = M.x(dot.key); return blendCam(cam([(kx + xf) / 2 - 0.6, y + 1.9, 8.4], [(kx + xf) / 2, y + 0.95, 0]), cam([(kx + xf) / 2 + 0.2, y + 1.6, 7.2], [(kx + xf) / 2 + 0.1, y + 0.95, 0]), smooth(p)); },
      heads: (M, p) => { const { xf, y } = L(M); return blendCam(cam([xf + 0.3, y + 2.6, 14.2], [xf + 0.5, y + 0.9, 0]), cam([xf + 0.6, y + 2.3, 13.2], [xf + 0.55, y + 0.9, 0]), smooth(p)); },
      dot: (M, p) => { const c = M.detail.dotCenter || v3(L(M).xf - 1, L(M).y + 1.6, 0.7); return farther(blendCam(cam([c.x + 0.35, c.y + 0.3, c.z + 3.3], [c.x + 0.05, c.y + 0.02, c.z]), cam([c.x + 0.1, c.y + 0.18, c.z + 2.75], [c.x + 0.05, c.y + 0.02, c.z]), smooth(p)), 1.45); },
      mix: (M, p) => { const { xf, y } = L(M); return blendCam(cam([xf + 0.8, y + 1.85, 4.3], [xf + 1.3, y + 1.55, 0.62]), cam([xf + 1.8, y + 1.85, 4.0], [xf + 1.4, y + 1.55, 0.62]), smooth(p)); },
      add1: (M, p) => { const { xf, y } = L(M); return blendCam(cam([xf - 0.4, y + 1.6, 2.9], [xf + 0.45, y + 1.1, 0.1]), cam([xf - 0.2, y + 1.7, 4.2], [xf + 0.6, y + 1.15, 0.1]), smooth(p)); },
      // 前馈：先贴近神经元阵列（灯一盏盏亮起），再拉开把三块矩阵也框进来
      mlp: (M, p) => { const { xf, y } = L(M); const a = cam([xf + 1.6, y + 3.6, 7.6], [xf + 1.2, y + 2.85, -0.2]), b = cam([xf + 1.25, y + 3.4, 6.3], [xf + 1.2, y + 2.85, -0.2]); const panel = cam([xf + 0.5, y + 3.5, 4.4], [xf + 0.2, y + 3.15, -0.9]); return p < 0.6 ? blendCam(a, panel, smooth(seg(p, 0.15, 0.35)) * (1 - smooth(seg(p, 0.45, 0.6)))) : blendCam(a, b, smooth(seg(p, 0.6, 1))); },
      mm: (M, p, st) => { const c = M.micro.camera(st); if (c) endCam.set('mm', c); const b = endCam.get('mm') || dCam.mlp(M, 1); return orbit({ ...b, fov: 32 }, lerp(-6, 4, smooth(p)), 0, lerp(1.6, 1.4, smooth(p))); },
      neuron: (M, p, st) => { const c = M.detail.camera(st); return orbit({ ...c, fov: 32 }, lerp(-6, 5, smooth(p)), 0, lerp(1.55, 1.4, smooth(p))); },
      bits: (M, p, st) => { const c = M.detail.camera(st); return orbit({ ...c, fov: 32 }, lerp(-12, 4, smooth(p)), lerp(8, 2, smooth(p)), lerp(2.3, 1.5, easeInOut(seg(p, 0, 0.6)))); },
    };
    const camKey = { over: 'over', ln1: 'ln1', qkv: 'qkv', score: 'score', softmax: 'score', heads: 'heads', dmul: 'dot', dsum: 'dot', dscale: 'dot', mix: 'mix', add1: 'add1', up: 'mlp', act: 'mlp', down: 'mlp', pick: 'mm', mul: 'mm', sum: 'mm', silu: 'neuron', gate: 'neuron', bits: 'bits' };
    const camSpan = {}; // 同一个机位 key 跨越的时间段
    D.forEach((x) => { const k = camKey[x.key]; if (!k) return; camSpan[k] ||= { t0: x.t0, t1: x.t1 }; camSpan[k].t1 = x.t1; });
    // 算式板出现的几段：画面往左让出位置
    const BOARD = ['dot', 'mm', 'neuron', 'bits'];
    const boardShift = (t) => Math.max(...[[camSpan.dot.t0, camSpan.dot.t1], [camSpan.mm.t0, camSpan.bits.t1]].map(([a, b]) => smoother(seg(t, a - 0.2, a + 0.9)) * (1 - smoother(seg(t, b - 0.7, b + 0.4)))));
    const BLEND = { over: 2.8, ln1: 1.3, qkv: 1.5, score: 1.4, heads: 1.6, dot: 1.4, mix: 1.3, add1: 1.1, mlp: 1.5, mm: 1.5, neuron: 0.9, bits: 1.0 };
    const WHAT = { over: '', ln1: 'RMSNorm', qkv: '注意力 · Q K V', score: '注意力 · 打分', softmax: '注意力 · softmax', heads: '注意力 · 16 个头', dmul: '一次 Q·K 打分', dsum: '一次 Q·K 打分', dscale: '一次 Q·K 打分', mix: '注意力 · 加权求和', add1: '残差 ⊕', up: '前馈 · 升维', act: '前馈 · 门控激活', down: '前馈 · 降维', pick: '一次乘加', mul: '一次乘加', sum: '一次乘加', silu: 'SiLU', gate: 'SiLU × u', bits: 'bf16 比特' };
    shot('dissect', (lt, t, { M }) => {
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
          ov: { corner: { g: G0 }, cornerA: 1 },
        };
      }
      let head = null;
      if (['score', 'softmax', 'dmul', 'dsum', 'dscale'].includes(s.key)) head = dot.head;
      if (s.key === 'heads') head = t < hk.t0 + 0.1 + CYCLE ? Math.min(15, Math.floor(clamp((t - hk.t0 - 0.1) / CYCLE) * 16)) : dot.head;
      const st = mst(s.depth, G0, { ph: 'layer', L: LX, ...s.s }, s.p0 ? 0 : s.p, { dAnim: s.depth, head });
      const ck = camKey[s.key], span = camSpan[ck];
      return {
        st,
        iso: smooth(seg(lt, 0, 2)) * (s.key === 'over' ? 0.55 : 0.85),
        cam: () => {
          const live = handheld(dCam[ck](M, clamp((t - span.t0) / (span.t1 - span.t0)), st), t, 0.0025);
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
        hide: [...(['score', 'heads', 'dot'].includes(ck) ? ['attnMats'] : []), ...(ck === 'bits' ? ['exDeco', 'mlpVecs'] : [])],
        flipBit: t >= flipT0 && t < flipT1 ? FLIP : null,
        boardA: BOARD.includes(ck) ? smooth(seg(t, span.t0 + 0.05, span.t0 + 0.45)) * (1 - smooth(seg(t, span.t1 - 0.35, span.t1 - 0.02))) : 1,
        shiftX: 430 * boardShift(t),
        dof: ck === 'bits' ? { range: 0.5, blur: 6 } : ['dot', 'mm', 'neuron'].includes(ck) ? { range: 1.6, blur: 3.5 } : null,
        ov: {
          att: { a: smooth(seg(t, hk.t0 + 0.1, hk.t0 + 0.8)) * (1 - smooth(seg(t, hk.t1 - 0.3, hk.t1 + 0.3))), L: LX, g: G0, head: s.key === 'heads' ? head : dot.head, reveal: seg(t, hk.t0 + 0.1, hk.t0 + 1.2), side: 'right' },
          corner: { g: G0, L: LX, what: WHAT[s.key] }, cornerA: 1,
        },
      };
    });
  }

  /* ------------------------------------------------------------ 05 第 1 个词：输出与采样 */
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
      ov: { corner: { g, what: s.s.ph === 'sample' ? '采样' : '输出头' }, cornerA: 1 },
    };
  });
  const st0 = S(0);
  {
    const T0 = SEC.sample1.t0, T1 = SEC.sample1.t1;
    chapter(T0 + 0.3, T1 - 0.2, '05', '输出与采样', 'LOGITS & SAMPLING');
    var sp1 = samp(T0, [3, 9, 4, 2, 1, 1, 4]);
    ev(spk(sp1, 'draw').t0 + 0.05, 'dice');
    ev(emitOf(sp1), 'emit', { g: 0, k: 1.2 });
    ev(spk(sp1, 'pick').t0, 'step', { k: 0.6 });
    ev(spk(sp1, 'softmax').t0, 'step', { k: 0.6 });
    sub(T0 + 0.15, spk(sp1, 'pick').t0 - 0.1, '最后一次 RMSNorm。');
    sub(spk(sp1, 'pick').t0 + 0.1, spk(sp1, 'mul').t0 + 1.4, '再乘输出矩阵：它就是那张嵌入表。');
    sub(spk(sp1, 'mul').t0 + 1.6, spk(sp1, 'sum').t1 - 0.1, `${q(hm0.token)}的分数：${m('1024')} 项乘加 = ${cy(hm0.total.toFixed(2))}。`);
    sub(spk(sp1, 'softmax').t0 + 0.1, spk(sp1, 'softmax').t1 - 0.1, `softmax：${q(st0.top[0][2])} ${m(pct(st0.top[0][1]))}。`);
    sub(spk(sp1, 'temp').t0 + 0.05, spk(sp1, 'topp').t1 - 0.1, `温度和截断之后，只剩 ${m(st0.pool.length)} 个候选。`);
    sub(spk(sp1, 'draw').t0 + 0.05, T1 - 0.15, `u = ${m(st0.u.toFixed(3))}。第一个词：${q(st0.chosenS)}。`);
    strip(spk(sp1, 'pick').t0 + 0.3, spk(sp1, 'softmax').t0, `logit[${m(hm0.j)}] = h ${m('[1024]')} · E[${m(hm0.j)}] ${m('[1024]')}<span class="sep"></span>词表里 ${m('151,936')} 个词元，每个都算一遍`);
    strip(spk(sp1, 'softmax').t0 + 0.2, spk(sp1, 'draw').t0, `softmax → ÷ 温度 ${m('0.7')} → 前 ${m('20')} 名（top-k）→ 累计 ${m('80%')}（top-p）`);
    strip(spk(sp1, 'draw').t0 + 0.2, T1 - 0.15, `候选池：${q(st0.pool[0][2])} ${m(pct(st0.pool[0][1]))}<span class="sep"></span>随机数 u = ${m(st0.u.toFixed(4))}`);
    sampleShot('sample1', 0, sp1, 'dissect', 1.0);
  }

  /* ------------------------------------------------------------ 06 自回归：第 2 – 14 个词 */
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
    chapter(T0 + 0.3, T1 - 0.2, '06', '自回归', 'AUTOREGRESSION');
    // 时长都取拍子的整数倍，词元吐出来的时刻正好落在节拍上（配乐里的铃声跟着它）
    const D1 = 12 * BEAT;
    const loop1 = toks(1, T0, [D1, ...Array.from({ length: 12 }, () => BEAT)]);
    loop1.forEach((s) => ev(s.emit, 'emit', { g: s.g, k: s.g === 1 ? 1.1 : 0.7, rank: Q.steps[s.g].chosenRank }));
    const sampleAt = T0 + D1 * 0.74;
    ev(sampleAt + 0.05, 'dice');
    const st1 = S(1);
    sub(T0 + 0.3, T0 + 2.9, `${q(st0.chosenS)}接回末尾，再算一遍：自回归。`);
    sub(T0 + 3.1, sampleAt + 0.1, `第 2 个词：${q(st1.top[0][2])} ${m(pct(st1.top[0][1]))}，${q(st1.top[1][2])} ${m(pct(st1.top[1][1]))}。`);
    sub(sampleAt + 0.3, loop1[1].t0 + 1.0, `u = ${m(st1.u.toFixed(3))}：选中了${q(st1.chosenS)}，不是第一名。`);
    sub(loop1[1].t0 + 1.2, T1 - 3.0, '前面的 K、V 都存在缓存里，每步只算新来的词。');
    sub(T1 - 2.8, T1 - 0.15, `${Q.steps.slice(2, 14).map((x) => tk(x.chosenS)).join('')}……`);
    strip(sampleAt, loop1[1].t0 + 1.0, `候选池：${st1.pool.map((x) => `${q(x[2])} ${m(pct(x[1]))}`).join(' · ')}<span class="sep"></span>u = ${m(st1.u.toFixed(4))}`);
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
        ov: { reply: { a: smooth(seg(lt, 0.2, 0.8)), n: 1 + done.length, k: newest ? smooth(seg(t, newest.emit, newest.emit + 0.35)) : 1 }, corner: { g: s.g }, cornerA: 1 },
      };
    });
  }

  /* ------------------------------------------------------------ 07 第 15 个词「散」：再穿过 28 层 */
  const stK = S(GK);
  {
    const T0 = SEC.layersK.t0, T1 = SEC.layersK.t1;
    chapter(T0 + 0.3, SEC.sampleK.t1 - 0.2, '07', `第 ${GK + 1} 个词`, 'THE KEY WORD');
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
    sub(T0 + 0.3, T0 + 3.5, `第 ${GK + 1} 个词。「${Q.steps.slice(9, 14).map((x) => tk(x.chosenS)).join('')}」之后呢？`);
    sub(T0 + 3.7, lk(20).t0 - 0.1, '同样的 28 层，再走一遍。');
    sub(lk(20).t0 + 0.05, lk(27).t0 - 0.1, `第 20 层起它想说${q(reflect[2])}（第 23 层 ${m(pct(reflect[1]))}），后来又变成${q(lensTop(GK, 25)[2])}、${q(lensTop(GK, 26)[2])}……`);
    sub(lk(27).t0 + 0.05, T1 - 0.1, `直到最后一层，${q(lensTop(GK, 27)[2])}才排到第一：${m(pct(lensTop(GK, 27)[1]))}。`);
    for (const s of schedK) {
      if (s.L == null) continue;
      const top = lensTop(GK, s.L);
      strip(s.t0, s.t1, `<span class="k">L${String(s.L).padStart(2, '0')}</span><span class="v">此刻开口：${q(top[2])} ${m(pct(top[1]))}</span>`, { fin: 0.06, fout: 0.06 });
    }
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
          corner: { g: GK, L: s.L }, cornerA: 1,
        },
      };
    });
  }

  /* ------------------------------------------------------------ 「散」：输出与采样 */
  {
    const T0 = SEC.sampleK.t0, T1 = SEC.sampleK.t1;
    const spK = samp(T0, [1, 0, 4, 2, 1, 3, 5]);
    ev(spk(spK, 'draw').t0 + 0.05, 'dice');
    ev(emitOf(spK), 'emit', { g: GK, k: 1.4 });
    sub(T0 + 0.2, spk(spK, 'temp').t0 - 0.05, `softmax：${stK.top.slice(0, 3).map((x) => `${q(x[2])} ${m(pct(x[1]))}`).join('，')}。`);
    sub(spk(spK, 'temp').t0 + 0.1, spk(spK, 'draw').t0 - 0.05, `温度、截断之后剩 ${m(stK.pool.length)} 个候选：${stK.pool.map((x) => m(pct(x[1]))).join('、')}。`);
    sub(spk(spK, 'draw').t0 + 0.05, T1 - 0.1, `u = ${m(stK.u.toFixed(3))} → ${q(stK.chosenS)}。差一点就说成了别的。`);
    strip(spk(spK, 'topp').t0 + 0.1, T1 - 0.1, `候选池：${stK.pool.map((x) => `${q(x[2])} ${m(pct(x[1]))}`).join(' · ')}<span class="sep"></span>u = ${m(stK.u.toFixed(4))}`);
    sampleShot('sampleK', GK, spK, 'layersK', 1.6);
  }

  /* ------------------------------------------------------------ 08 自回归写完 */
  {
    const T0 = SEC.loop2.t0, T1 = SEC.loop2.t1;
    chapter(T0 + 0.3, T1 - 0.2, '08', '写完整句', 'TO THE END');
    const rest = Q.G - (GK + 1);
    const durs2 = Array.from({ length: rest }, (_, k) => (k === rest - 1 ? 0 : k < 4 ? 1.5 : 0.75) * BEAT);
    durs2[rest - 1] = T1 - T0 - durs2.reduce((a, b) => a + b, 0);
    const loop2 = toks(GK + 1, T0, durs2);
    loop2.forEach((s) => ev(s.emit, s.g === Q.G - 1 ? 'end' : 'emit', { g: s.g, k: 0.55, rank: Q.steps[s.g].chosenRank }));
    const lowRank = Q.steps.filter((x) => x.chosenRank > 0).length;
    const stEnd = S(Q.G - 1);
    const lastT = loop2[loop2.length - 1].t0;
    sub(T0 + 0.3, T0 + 4.0, '剩下的词，一个接一个写出来。');
    sub(T0 + 4.3, lastT - 3.1, `下划线是选中时的概率；红色的 ${m(lowRank)} 个，都不是当时的第一名。`);
    sub(lastT - 2.9, lastT - 0.15, '采样让每次回答都可能不一样。');
    sub(lastT + 0.1, T1 - 0.1, `最后选中结束标记 ${m('&lt;|im_end|&gt;')}（${m(pct(stEnd.chosenP1))}）：回答结束。`);
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
        ov: { reply: { a: 1, n: Math.min(Q.G, GK + 1 + done.length), k: newest ? smooth(seg(t, newest.emit, newest.emit + 0.3)) : 1 }, corner: { g: s.g }, cornerA: 1 },
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
    ev(T0 + 1.6, 'hit', { k: 0.5 });
    ev(T0 + 10.0, 'hit', { k: 0.35 });
    shot('end', (lt, t, { M }) => ({
      st: mst(2, Q.G - 1, { ph: 'sample' }, 1, { dAnim: lerp(2.7, 1, smoother(seg(lt, 0.4, 3.9))), view: 'machine' }),
      cam: () => {
        const far = cam([-11, 5.5, 70], [0, 11.5, 0], 30);
        const live = loopCam(M, M.x(Q.row(Q.G - 1)), t, { yaw: 30, dz: 28, back: 6 });
        return blendCam(prevCam('loop2', live), far, smoother(seg(lt, 0, 10)));
      },
      dof: { focus: 14, range: 24, blur: 6 * smooth(seg(lt, 2.5, 5)) },
      fade: lerp(0, 0.86, smooth(seg(lt, 0.6, 2.0))) + 0.14 * smooth(seg(t, T1 - 1.2, T1 - 0.05)),
      ov: {
        reply: { a: 1 - smooth(seg(lt, 0.4, 1.4)), n: Q.G, k: 1 },
        band: 1 - smooth(seg(lt, 0, 1)),
        end: { a1: smooth(seg(lt, 1.4, 2.4)) * (1 - smooth(seg(lt, 9.0, 9.8))), k1: smooth(seg(lt, 4.0, 5.0)), a2: smooth(seg(lt, 10.0, 10.9)) * (1 - smooth(seg(t, T1 - 1.0, T1 - 0.1))), k2: seg(lt, 10.0, 11.6) },
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

  return { end, frame, subs, strips, chapters, cards, events, sections, qTimes, statLine, bpm: BPM, shots: shots.map((s) => ({ name: s.name, t0: s.t0, t1: s.t1 })) };
}
