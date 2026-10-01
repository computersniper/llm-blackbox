// 分镜表：每个镜头的时间、机器状态、机位、字幕、数据条、事件。
// 节拍：96 BPM（一拍 0.625 秒，一小节 2.5 秒），所有段落都卡在小节线上，配乐（compose.py）用同一张表。
import { THREE } from './lib/engine.js';
import { path, blendCam, orbit, handheld, clamp, lerp, seg, smooth, smoother, easeOut, easeIn, easeInOut, v3, pchip } from './lib/cam.js';
import { viewOf } from '/public/js/timeline.js';
import { tokPlain, shortSpecial, fmtPct, esc } from '/public/js/ui.js';

export const BPM = 96, BEAT = 60 / BPM, BAR = 4 * BEAT;
const B = (bar, beat = 0) => bar * BAR + beat * BEAT;

const tk = (s) => esc(tokPlain(s));
const q = (s) => `<q>${tk(s)}</q>`;
const m = (s) => `<span class="m">${s}</span>`;
const pct = (p) => fmtPct(p);
const f = (v, d = 2) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);

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
  const section = (name, t0, t1, o = {}) => sections.push({ name, t0, t1, ...o });

  const shots = [];
  const shot = (name, t0, t1, fn) => shots.push({ name, t0, t1, fn });
  const endCam = new Map(); // 每个镜头最后一帧的机位（下一个镜头从这里平滑接过去）
  const prevCam = (name, fallback) => endCam.get(name) || fallback;

  const st0 = Q.steps[0], st4 = Q.steps[4];
  const user = Q.tokens.map((t, i) => ({ ...t, i })).filter((t) => t.role === 'user' && !t.sp && t.s !== 'user' && t.s !== '\n');
  const G4 = 4, LX = 20; // 拆开的那一步、那一层
  const row4 = Q.row(G4);
  const dot = Q.dotAt(LX, G4), neu = Q.neuronAt(LX, G4), hm4 = Q.headMMAt(G4);
  const att13 = Q.att(LX, dot.head, row4);
  const sinkHeads = Array.from({ length: Q.H }, (_, h) => Q.att(LX, h, row4)[0]).filter((r) => r && r.j === 0).length;
  const lensTop = (g, L) => Q.lensAt(g, L).top[0];

  /* ------------------------------------------------------------ A 冷开场 0 – 12.5 */
  const qTimes = user.map((_, k) => B(0, 3) + k * BEAT);
  qTimes.forEach((t, i) => ev(t, 'type', { i }));
  section('cold', 0, B(5), { energy: 0.15 });
  sub(B(2, 3), B(3, 3) + 0.2, '你问它一个问题。');
  sub(B(3, 3) + 0.4, B(5) - 0.15, '在它开口之前，黑箱里发生了什么？');
  const coldCam = path([{ t: 0, p: [-9.5, 3.0, 50], l: [0, 5.8, 0], fov: 30 }, { t: B(5), p: [-5.2, 3.9, 37.5], l: [0, 5.9, 0], fov: 30 }]);
  shot('cold', 0, B(5), (lt, t) => ({
    st: mst(1, 0, { ph: 'pass' }, 0.04 + 0.42 * smooth(seg(t, 9.5, 12.4))),
    cam: () => coldCam(t),
    fade: t < 4.5 ? lerp(1, 0.56, smooth(seg(t, 0.4, 4.5))) : lerp(0.56, 0.3, seg(t, 4.5, 12.4)),
    ov: { band: 0.5, qline: smooth(seg(t, qTimes[0] - 0.6, qTimes[0])) * (1 - smooth(seg(t, 11.5, 12.35))), qScale: 1 + 0.05 * smooth(seg(t, 11.5, 12.35)), qBlur: 7 * smooth(seg(t, 11.5, 12.35)) },
  }));

  /* ------------------------------------------------------------ B 片名 12.5 – 20 */
  ev(B(5), 'hit', { k: 1 });
  section('title', B(5), B(8), { energy: 0.35 });
  const titleCam = (lt) => {
    const k = smooth(lt / 7.5);
    const yaw = THREE.MathUtils.degToRad(lerp(-33, -12, k)), R = lerp(31, 27.5, k), h = lerp(2.3, 3.6, k);
    const look = v3(0, 5.6, 0);
    return { pos: look.clone().add(v3(R * Math.sin(yaw), h - 5.6, R * Math.cos(yaw))), look, fov: 32 };
  };
  shot('title', B(5), B(8), (lt) => ({
    st: mst(1, 0, { ph: 'pass' }, 0.04),
    cam: () => titleCam(lt),
    fade: 0.32 * (1 - smooth(seg(lt, 0, 0.6))),
    ov: { band: 0, title: smooth(seg(lt, 0.1, 1.0)) * (1 - smooth(seg(lt, 6.3, 7.3))), titleK: seg(lt, 0.05, 3.4), titleBlur: 7 * smooth(seg(lt, 6.3, 7.3)) },
  }));

  /* ------------------------------------------------------------ C 揭开 + 分词 20 – 30 */
  section('tokenize', B(8), B(12), { energy: 0.45 });
  chapter(B(8) + 0.3, B(12) - 0.2, '01', '分词', 'TOKENIZE');
  ev(B(8), 'open');
  ev(B(9) - 0.1, 'whoosh');
  const dropT0 = 23.0, dropD = 4.0;
  const landT = (i) => dropT0 + dropD * ((i / Q.P) * 0.7 + 0.3);
  for (let i = 0; i < Q.P; i++) ev(landT(i), 'tick', { i, k: Q.tokens[i].role === 'user' ? 1 : 0.5 });
  sub(B(8) + 0.4, B(9) + 0.6, '第一步：分词。');
  sub(B(9) + 0.9, B(10) + 1.8, `聊天模板把系统提示、你的问题和特殊标记拼在一起：一共 ${m(Q.P)} 个词元。`);
  sub(B(10) + 2.1, B(12) - 0.2, '注意，「首都是」被切成了「首」和「都是」——分词并不管词义。');
  strip(B(10) + 2.1, B(12) - 0.2, user.map((t) => `<span class="tk">${tk(t.s)}</span> ${m(t.id)}`).join('<span class="sep"></span>'));
  const xEnd = (i) => (i < 0 ? -1 : 0); void xEnd;
  const tokCamKeys = (M) => {
    const X = (i) => M.x(i);
    const tr = (i, dx = -3.4, dy = 1.25, dz = 4.6) => ({ p: [X(i) + dx, 0.3 + dy, dz], l: [X(i) + 0.6, 0.28, 0] });
    const u0 = user[0].i, u1 = user[user.length - 1].i, uc = (X(u0) + X(u1)) / 2;
    const T = titleCam(7.5);
    return [
      { t: 0, p: T.pos.toArray(), l: T.look.toArray(), fov: 32 },
      { t: 2.4, p: [-2.6, 8.2, 19.0], l: [-1.2, 2.4, 0], fov: 32 },
      { t: landT(0) - B(8) - 0.15, ...tr(0), fov: 30 },
      { t: landT(19) - B(8), ...tr(19), fov: 30 },
      { t: landT(38) - B(8), ...tr(38, -3.0, 1.3, 4.5), fov: 30 },
      { t: 7.6, p: [uc - 1.6, 1.75, 4.6], l: [uc + 0.25, 0.22, 0], fov: 30 },
      { t: 10.0, p: [uc - 1.0, 1.35, 3.85], l: [uc + 0.25, 0.2, 0], fov: 30 },
    ];
  };
  let tokPath = null;
  shot('tokenize', B(8), B(12), (lt, t, { M }) => {
    const dA = lerp(1, 2.6, smoother(seg(t, 20.0, 22.4)));
    return {
      st: mst(2, 0, { ph: 'read' }, seg(t, dropT0, dropT0 + dropD), { dAnim: dA }),
      cam: () => { tokPath ||= path(tokCamKeys(M)); return tokPath(lt); },
      extras: { ids: smooth(seg(t, landT(0), landT(0) + 0.5)), idsFocus: t > 27.4 },
      ov: { cornerA: 0 },
    };
  });

  /* ------------------------------------------------------------ D 嵌入 30 – 37.5 */
  section('embed', B(12), B(15), { energy: 0.5 });
  chapter(B(12) + 0.3, B(15) - 0.2, '02', '嵌入', 'EMBEDDING');
  for (let i = 0; i < Q.P; i += 3) ev(30.5 + (i / Q.P) * 0.55 * 2.7, 'blip', { i });
  sub(B(12) + 0.4, B(13) + 1.8, `每个编号去嵌入表里，取出属于自己的那一行：${m('1024')} 个数。`);
  sub(B(13) + 2.1, B(15) - 0.2, `从这里开始，每个词元都是一个 ${m('1024')} 维的向量。`);
  strip(B(12) + 0.6, B(15) - 0.2, `嵌入表 ${m('151,936 × 1,024')}<span class="sep"></span>${m('155,582,464')} 个参数<span class="sep"></span>「法国」= 第 ${m(user[0].id)} 行`);
  let embPath = null;
  shot('embed', B(12), B(15), (lt, t, { M }) => {
    const keys = tokCamKeys(M);
    const last = keys[keys.length - 1];
    embPath ||= path([
      { t: 0, p: last.p, l: last.l, fov: 30 },
      { t: 2.6, p: [-6.2, 4.5, 17.0], l: [-3.4, 3.0, 0], fov: 32 },
      { t: 7.5, p: [-3.6, 5.9, 18.6], l: [-1.4, 3.5, 0], fov: 32 },
    ]);
    return {
      st: mst(2, 0, { ph: 'embed' }, seg(t, 32.0, 36.2), { dAnim: 2.6 }),
      cam: () => embPath(lt),
      extras: { ids: 1 - smooth(seg(t, 30.0, 30.8)), idsFocus: true, emb: smooth(seg(t, 30.0, 30.7)) * (1 - smooth(seg(t, 36.6, 37.5))), embK: seg(t, 30.4, 33.6) },
      ov: { cornerA: 0 },
    };
  });

  /* ------------------------------------------------------------ E 第 1 个词：28 层 37.5 – 60 */
  section('layers1', B(15), B(24), { energy: 0.75 });
  chapter(B(15) + 0.3, B(24) - 0.2, '03', '穿过 28 层', 'TRANSFORMER LAYERS');
  const sched1 = lay(B(15, 3), [
    { L: 0, d: 2 * BEAT },
    ...Array.from({ length: 16 }, (_, k) => ({ L: k + 1, d: BEAT / 2 })),
    ...Array.from({ length: 4 }, (_, k) => ({ L: 17 + k, d: BEAT })),
    { L: 21, d: 2 * BEAT }, { L: 22, d: BEAT }, { L: 23, d: BEAT },
    { L: 24, d: 4 * BEAT },
    { L: 25, d: BEAT }, { L: 26, d: BEAT }, { L: 27, d: BEAT },
  ]);
  sched1.forEach((s) => ev(s.t0, 'layer', { L: s.L, k: s.d > BEAT * 1.5 ? 1 : 0.6 }));
  ev(sched1[21].t0, 'reveal', { k: 0.8 });
  ev(sched1[24].t0, 'reveal', { k: 1 });
  const l1 = (L) => sched1.find((s) => s.L === L);
  sub(B(15) + 0.3, B(16) + 1.2, '然后，穿过 28 层 Transformer。');
  sub(B(16) + 1.5, l1(17).t0 - 0.15, '每过一层，用「逻辑透镜」偷看一眼：如果此刻就开口，它会说什么？');
  sub(l1(17).t0 + 0.1, l1(21).t0 - 0.1, `前二十层，还说不出像样的词：${q(lensTop(0, 17)[2])}、${q(lensTop(0, 18)[2] === '<|endoftext|>' ? '结束符' : lensTop(0, 18)[2])}……`);
  sub(l1(21).t0 + 0.05, l1(24).t0 - 0.1, `第 21 层：${q(lensTop(0, 21)[2])} ${m(pct(lensTop(0, 21)[1]))}。`);
  sub(l1(24).t0 + 0.05, l1(25).t0 + 0.4, `第 24 层，它想到的是${q(lensTop(0, 24)[2])}（${m(pct(lensTop(0, 24)[1]))}）——答案已经在了。`);
  sub(l1(25).t0 + 0.6, B(22) + 1.0, `可现在要说的是第一个词：最后一层，${q(lensTop(0, 27)[2])} ${m(pct(lensTop(0, 27)[1]))}。`);
  sub(B(22) + 1.3, B(24) - 0.2, `向量一层层往上累加：长度从 ${m(Q.norm(0, Q.row(0)).toFixed(1))} 涨到 ${m(Math.max(...Array.from({ length: NL }, (_, L) => Q.norm(L, Q.row(0)))).toFixed(0))}。`);
  for (const s of sched1) {
    const top = lensTop(0, s.L);
    strip(s.t0, s.t1 + (s.L === 27 ? 4.6 : 0), `<span class="k">L${String(s.L).padStart(2, '0')}</span><span class="v">此刻开口：${q(top[2])} ${m(pct(top[1]))}</span>`, { fin: 0.06, fout: 0.06 });
  }
  const camLayers1 = (M, t) => {
    const Ls = [[B(15), 0], [l1(0).t0, 0], [l1(17).t0, 15.5], [l1(21).t0, 20.0], [l1(24).t0, 23.2], [l1(25).t0, 24.4], [B(22), 27.0]];
    const Lc = pchip(Ls.map((a) => a[0]), Ls.map((a) => a[1]))(t);
    const y = 0.95 + Lc * 0.26;
    const yaw = lerp(-6, 10, smooth(seg(t, B(15), B(22))));
    const look = v3(7.6, y + 0.1, 0.3);
    const base = { pos: look.clone().add(v3(5.0, 1.45, 8.6)), look, fov: 32 };
    return orbit(base, yaw, 0, 1);
  };
  let l1Path = null;
  shot('layers1', B(15), B(24), (lt, t, { M }) => {
    const s = pick(sched1, t);
    const before = t < sched1[0].t0;
    const st = mst(3, 0, { ph: 'layer', L: s.L }, before ? 0 : s.p, { dAnim: 3 });
    const cam = () => {
      const live = camLayers1(M, t);
      if (lt < 2.2) { // 从嵌入的远景接进来
        l1Path ||= path([{ t: 0, p: [-3.6, 5.9, 18.6], l: [-1.4, 3.5, 0], fov: 32 }, { t: 2.2, p: camLayers1(M, B(15) + 2.2).pos.toArray(), l: camLayers1(M, B(15) + 2.2).look.toArray(), fov: 32 }]);
        return blendCam(l1Path(lt), live, smoother(lt / 2.2));
      }
      if (t > B(22)) { // 拉开：整座塔
        const far = { pos: v3(19.5, 9.4, 21.5), look: v3(3.8, 5.0, 0), fov: 32 };
        return blendCam(live, far, smoother(seg(t, B(22), B(24) - 0.3)));
      }
      return handheld(live, t, 0.004);
    };
    return {
      st, cam,
      ov: { lens: { a: smooth(seg(t, B(16) + 1.5, B(16) + 2.2)) * (1 - smooth(seg(t, B(24) - 0.8, B(24)))), g: 0, upto: before ? -0.01 : s.L + (s.L === 27 ? 1 : s.p) - 0.0001, other: '巴黎' }, corner: { g: 0, L: before ? null : s.L }, cornerA: smooth(seg(t, B(15), B(15) + 0.6)) },
    };
  });

  /* ------------------------------------------------------------ F0 第 1 个词：输出与采样 60 – 70 */
  section('sample1', B(24), B(28), { energy: 0.6 });
  chapter(B(24) + 0.3, B(28) - 0.2, '04', '输出与采样', 'SAMPLING');
  const samp1 = lay(B(24), [
    { s: { ph: 'head', sub: 'norm' }, d: 1.5 * BEAT },
    { s: { ph: 'head', sub: 'unembed' }, d: 1.5 * BEAT },
    { s: { ph: 'head', sub: 'softmax' }, d: 3 * BEAT },
    { s: { ph: 'sample', sub: 'temp' }, d: 2 * BEAT },
    { s: { ph: 'sample', sub: 'topk' }, d: BEAT },
    { s: { ph: 'sample', sub: 'topp' }, d: 2 * BEAT },
    { s: { ph: 'sample', sub: 'draw' }, d: 6 * BEAT },
  ]);
  const draw1 = samp1[6];
  ev(draw1.t0 + 0.05, 'dice');
  ev(draw1.t0 + draw1.d * 0.8, 'emit', { g: 0 });
  sub(B(24) + 0.3, samp1[2].t0 + 0.9, `输出头给 ${m('151,936')} 个词元各打一个分，softmax 成概率。`);
  sub(samp1[2].t0 + 1.2, samp1[4].t0 - 0.1, `${q(st0.top[0][2])} ${m(pct(st0.top[0][1]))}，${q(st0.top[1][2])} ${m(pct(st0.top[1][1]))}。温度 0.7 让高的更高：${m(pct(st0.temps['0.7'][0]))}。`);
  sub(samp1[4].t0 + 0.1, draw1.t0 + 1.5, `top-k、top-p 截断之后，只剩 ${m(st0.pool.length)} 个候选。`);
  sub(draw1.t0 + 1.8, B(28) - 0.15, `随机数 u = ${m(st0.u.toFixed(3))}。第一个词：${q(st0.chosenS)}。`);
  strip(samp1[2].t0 + 0.2, samp1[3].t0, `logits ${m('[1 × 151936]')} → softmax`);
  strip(samp1[3].t0 + 0.1, samp1[5].t0 + 0.4, `÷ 温度 ${m('0.7')} → 前 ${m('20')} 名 → 累计 ${m('80%')}`);
  strip(draw1.t0 + 0.2, B(28) - 0.15, `候选池：${q(st0.pool[0][2])} ${m(pct(st0.pool[0][1]))}<span class="sep"></span>u = ${m(st0.u.toFixed(4))}`);
  const headCam = (M, st, dx = 1.3, dy = 0.85, dz = 7.7) => {
    const hx = M.headX(st);
    const look = v3(hx, M.yTop + 1.65, 0.1);
    return { pos: look.clone().add(v3(dx, dy, dz)), look, fov: 32 };
  };
  shot('sample1', B(24), B(28), (lt, t, { M }) => {
    const s = pick(samp1, t);
    const st = mst(4, 0, s.s, s.p, { dAnim: 4 });
    return {
      st,
      cam: () => {
        const live = headCam(M, st, lerp(1.6, 0.6, smooth(lt / 10)), lerp(1.1, 0.7, smooth(lt / 10)), lerp(8.6, 7.2, smooth(lt / 10)));
        const from = prevCam('layers1', live);
        return blendCam(from, live, smoother(seg(lt, 0, 2.0)));
      },
      ov: { corner: { g: 0, what: s.s.sub === 'draw' ? '采样' : '输出头' }, cornerA: 1 },
    };
  });

  /* ------------------------------------------------------------ G1 自回归：第 2 – 4 个词 70 – 80 */
  section('loop1', B(28), B(32), { energy: 0.7 });
  chapter(B(28) + 0.3, B(32) - 0.2, '05', '自回归', 'AUTOREGRESSION');
  const PH = [{ ph: 'read', d: 0.2 }, { ph: 'embed', d: 0.1 }, { ph: 'layers', d: 0.32 }, { ph: 'head', d: 0.12 }, { ph: 'sample', d: 0.26 }];
  const tokSched = (g, t0, d) => ({ g, t0, t1: t0 + d, d, emit: t0 + d * (1 - 0.26 * 0.2) });
  const loop1 = [tokSched(1, B(28), 1.5 * BAR), tokSched(2, B(29, 2), 1.25 * BAR), tokSched(3, B(30, 3), 1.25 * BAR)];
  loop1.forEach((s) => ev(s.emit, 'emit', { g: s.g }));
  const loopState = (list, t) => {
    const s = list.find((x) => t < x.t1) || list[list.length - 1];
    let u = clamp((t - s.t0) / s.d), acc = 0;
    for (const ph of PH) { if (u < acc + ph.d || ph === PH[PH.length - 1]) return { g: s.g, ph: ph.ph, p: clamp((u - acc) / ph.d) }; acc += ph.d; }
    return null;
  };
  sub(B(28) + 0.3, B(29) + 1.6, `${q(st0.chosenS)} 接回序列末尾，整套计算再来一遍：这叫自回归。`);
  sub(B(29) + 1.9, B(31) - 0.1, '前面词元的 K、V 都存在缓存里，每一步只需要算新来的这一个。');
  sub(B(31) + 0.1, B(32) - 0.15, `${q(Q.steps[1].chosenS)}、${q(Q.steps[2].chosenS)}、${q(Q.steps[3].chosenS)}……下一个，是关键的那个词。`);
  const machineCam = (M, t, yaw, distK = 1, dy = 0) => {
    const look = v3(0.8, 4.6 + dy, 0);
    const base = { pos: look.clone().add(v3(1.5, 2.6, 25.5 * distK)), look, fov: 32 };
    return orbit(base, yaw, 0, 1);
  };
  shot('loop1', B(28), B(32), (lt, t, { M }) => {
    const s = loopState(loop1, t);
    const st = mst(2, s.g, { ph: s.ph }, s.p, { dAnim: 2.7 });
    const n = 1 + loop1.filter((x) => t >= x.emit).length;
    const newest = loop1.filter((x) => t >= x.emit).pop();
    return {
      st,
      cam: () => {
        const live = machineCam(M, t, lerp(24, 4, smooth(lt / 10)), lerp(0.92, 1.0, smooth(lt / 10)));
        return blendCam(prevCam('sample1', live), live, smoother(seg(lt, 0, 2.4)));
      },
      ov: { reply: { a: smooth(seg(lt, 0.2, 0.8)), n, k: newest ? smooth(seg(t, newest.emit, newest.emit + 0.4)) : 1 }, corner: { g: s.g }, cornerA: 1 },
    };
  });

  /* ------------------------------------------------------------ G2 第 5 个词：再穿过 28 层 80 – 95 */
  section('layers5', B(32), B(38), { energy: 0.8 });
  chapter(B(32) + 0.3, B(38) - 0.2, '06', `第 5 个词`, 'THE KEY TOKEN');
  const sched5 = lay(B(32), [
    { s: { ph: 'read' }, d: 0.6 }, { s: { ph: 'embed' }, d: 2 * BEAT - 0.6 },
    ...Array.from({ length: 16 }, (_, k) => ({ L: k, d: BEAT / 2 })),
    ...Array.from({ length: 4 }, (_, k) => ({ L: 16 + k, d: BEAT })),
    { L: 20, d: 2 * BEAT }, { L: 21, d: 2 * BEAT }, { L: 22, d: 3 * BEAT },
    { L: 22, d: B(38) - B(32) - (2 + 8 + 4 + 2 + 2 + 3) * BEAT, hold: true },
  ]);
  sched5.forEach((s) => { if (s.L != null && !s.hold) ev(s.t0, 'layer', { L: s.L, k: s.d > BEAT * 1.5 ? 1 : 0.6 }); });
  const l5 = (L) => sched5.find((s) => s.L === L && !s.hold);
  ev(l5(20).t0, 'reveal', { k: 0.7 });
  ev(l5(22).t0, 'reveal', { k: 1.2 });
  sub(B(32) + 0.3, B(33) + 1.0, `第 5 个词。同样的 28 层，再走一遍。`);
  sub(B(33) + 1.3, l5(16).t0 - 0.1, `透镜里先是一些标点、${q(lensTop(4, 17)[2])}、${q(lensTop(4, 18)[2])}……`);
  sub(l5(16).t0 + 0.1, l5(20).t0 - 0.1, '它在等：后面该填一个地名。');
  sub(l5(20).t0 + 0.05, l5(21).t0 - 0.1, `第 20 层：好几个注意力头回头盯住了「法国」。`);
  sub(l5(21).t0 + 0.05, l5(22).t0 - 0.1, `第 21 层：${q(lensTop(4, 21)[2])} ${m(pct(lensTop(4, 21)[1]))}。`);
  sub(l5(22).t0 + 0.05, B(38) - 0.15, `第 22 层：${q(lensTop(4, 22)[2])} ${m(pct(lensTop(4, 22)[1]))}。它是怎么想到的？回到第 20 层。`);
  for (const s of sched5) {
    if (s.L == null || s.hold) continue;
    const top = lensTop(4, s.L);
    strip(s.t0, s.t1 + (s.L === 22 ? sched5[sched5.length - 1].d : 0), `<span class="k">L${String(s.L).padStart(2, '0')}</span><span class="v">此刻开口：${q(top[2])} ${m(pct(top[1]))}</span>`, { fin: 0.06, fout: 0.06 });
  }
  const camLayers5 = (M, t) => {
    const Ls = [[B(32), -1.5], [l5(0).t0, 0], [l5(16).t0, 14.5], [l5(20).t0, 19.0], [l5(22).t0, 21.4], [B(37), 22.0], [B(38), 20.4]];
    const Lc = pchip(Ls.map((a) => a[0]), Ls.map((a) => a[1]))(t);
    const y = 0.95 + Lc * 0.26;
    const look = v3(6.4, y + 0.15, 0.2);
    const base = { pos: look.clone().add(v3(-2.6, 2.3, 8.9)), look, fov: 32 };
    return orbit(base, lerp(4, -10, smooth(seg(t, B(32), B(38)))), 0, lerp(1, 0.88, smooth(seg(t, l5(20).t0, B(38)))));
  };
  shot('layers5', B(32), B(38), (lt, t, { M }) => {
    const s = pick(sched5, t);
    const st = s.L == null ? mst(3, G4, s.s, s.p, { dAnim: 3 }) : mst(3, G4, { ph: 'layer', L: s.L }, s.hold ? 1 : s.p, { dAnim: 3 });
    return {
      st,
      cam: () => { const live = handheld(camLayers5(M, t), t, 0.004); return blendCam(prevCam('loop1', live), live, smoother(seg(lt, 0, 2.2))); },
      ov: {
        lens: { a: smooth(seg(t, B(33) + 1.3, B(33) + 2.0)) * (1 - smooth(seg(t, B(38) - 0.6, B(38)))), g: G4, upto: s.L == null ? -0.01 : s.L + (s.hold ? 1 : s.p) - 0.0001, other: '法国' },
        reply: { a: 1 - smooth(seg(lt, 0, 0.8)), n: 4, k: 1 },
        corner: { g: G4, L: s.L }, cornerA: 1,
      },
    };
  });

  /* ------------------------------------------------------------ H 拆开第 20 层 95 – 165 */
  section('dissect', B(38), B(66), { energy: 0.55 });
  chapter(B(38) + 0.3, B(66) - 0.2, '07', '拆开第 20 层', 'INSIDE LAYER 20');
  const D = lay(B(38), [
    { key: 'over', depth: 4, s: { op: 'ln1' }, d: 2 * BAR, p0: true },
    { key: 'ln1', depth: 4, s: { op: 'ln1' }, d: 2 * BAR },
    { key: 'qkv', depth: 5, s: { op: 'attn', sub: 'qkv' }, d: 3 * BAR },
    { key: 'score', depth: 5, s: { op: 'attn', sub: 'score' }, d: BAR },
    { key: 'softmax', depth: 5, s: { op: 'attn', sub: 'softmax' }, d: BAR },
    { key: 'heads', depth: 5, s: { op: 'attn', sub: 'softmax' }, d: 2 * BAR },
    { key: 'dmul', depth: 6, s: { op: 'attn', sub: 'score', mi: 'mul' }, d: 3 * BEAT },
    { key: 'dsum', depth: 6, s: { op: 'attn', sub: 'score', mi: 'sum' }, d: 3 * BEAT },
    { key: 'dscale', depth: 6, s: { op: 'attn', sub: 'score', mi: 'scale' }, d: 2 * BEAT },
    { key: 'mix', depth: 5, s: { op: 'attn', sub: 'mix' }, d: BAR },
    { key: 'add1', depth: 5, s: { op: 'add1' }, d: BAR },
    { key: 'up', depth: 5, s: { op: 'mlp', sub: 'up' }, d: BAR },
    { key: 'act', depth: 5, s: { op: 'mlp', sub: 'act' }, d: 2 * BAR },
    { key: 'down', depth: 5, s: { op: 'mlp', sub: 'down' }, d: BAR },
    { key: 'pick', depth: 6, s: { op: 'mlp', sub: 'up', mi: 'pick' }, d: 2 * BEAT },
    { key: 'mul', depth: 6, s: { op: 'mlp', sub: 'up', mi: 'mul' }, d: 6 * BEAT },
    { key: 'sum', depth: 6, s: { op: 'mlp', sub: 'up', mi: 'sum' }, d: 4 * BEAT },
    { key: 'silu', depth: 6, s: { op: 'mlp', sub: 'act', mi: 'silu' }, d: 4 * BEAT },
    { key: 'gate', depth: 6, s: { op: 'mlp', sub: 'act', mi: 'gate' }, d: 4 * BEAT },
    { key: 'bits', depth: 7, s: { op: 'mlp', sub: 'up', mi: 'mul' }, d: 3 * BAR },
    { key: 'out', depth: 4, s: null, d: BAR },
  ]);
  const Dk = (k) => D.find((x) => x.key === k);
  // 事件
  D.forEach((x) => { if (['ln1', 'qkv', 'score', 'mix', 'add1', 'up', 'act', 'down', 'pick', 'silu', 'gate'].includes(x.key)) ev(x.t0, 'step', { k: 0.6 }); });
  ev(Dk('over').t0, 'hit', { k: 0.7 });
  ev(Dk('dmul').t0, 'hit', { k: 0.5 });
  for (let h = 0; h < 16; h++) ev(Dk('heads').t0 + h * (Dk('heads').d / 16), 'tick', { k: h === dot.head ? 1 : 0.45 });
  for (let k = 0; k < 12; k++) ev(Dk('mul').t0 + (k / 12) * Dk('mul').d / 1.1, 'tick', { k: 0.4 });
  for (let k = 0; k < 16; k++) ev(Dk('bits').t0 + 0.6 + k * 0.09, 'bit', { k });
  ev(Dk('out').t0, 'whoosh', { k: 1 });
  ev(Dk('out').t1, 'hit', { k: 0.6 });
  // 字幕
  const inNorm = Q.norm(LX - 1, row4);
  sub(Dk('over').t0 + 0.3, Dk('over').t0 + 2.4, '把第 20 层拆开。');
  sub(Dk('over').t0 + 2.6, Dk('ln1').t0 - 0.1, '每一层都是同样的六步：归一化、注意力、残差，归一化、前馈、残差。');
  sub(Dk('ln1').t0 + 0.2, Dk('qkv').t0 - 0.15, 'RMSNorm：把向量除以它的均方根，再乘一组学到的缩放系数。');
  strip(Dk('ln1').t0 + 0.4, Dk('qkv').t0 - 0.15, `进入第 20 层的向量 ‖x‖ = ${m(inNorm.toFixed(1))}<span class="sep"></span>均方根 ${m((inNorm / 32).toFixed(2))} → ${m('1')}`);
  sub(Dk('qkv').t0 + 0.3, Dk('qkv').t0 + 3.5, '注意力：先乘三块权重矩阵，得到 Q、K、V。');
  sub(Dk('qkv').t0 + 3.8, Dk('score').t0 - 0.15, 'Q 有 16 个头，K、V 只有 8 个：每两个查询头共用一组（GQA）。');
  card(Dk('qkv').t0 + 0.6, Dk('score').t0 - 0.2, `<span class="k">SHAPES · 解码阶段每步只算 1 个位置</span><span class="m">h [1×1024] @ W<sub>q</sub> [1024×2048] → q：<b>16</b> 头 × 128</span><br><span class="m">h [1×1024] @ W<sub>k</sub> [1024×1024] → k：<b>8</b> 头 × 128</span><br><span class="m">h [1×1024] @ W<sub>v</sub> [1024×1024] → v：<b>8</b> 头 × 128</span><br>k、v 存进 KV 缓存：现在一共 <b>${row4 + 1}</b> 个位置`, { style: 'left:76px;top:150px' });
  sub(Dk('score').t0 + 0.2, Dk('softmax').t0 - 0.1, `新词元的 Q，和缓存里 ${m(row4 + 1)} 个 K 逐一做点积，再除以 √128。`);
  sub(Dk('softmax').t0 + 0.1, Dk('heads').t0 - 0.1, 'softmax：打分变成权重，每一行加起来等于 1。');
  sub(Dk('heads').t0 + 0.1, Dk('heads').t0 + 2.4, `16 个头各看各的。${m(sinkHeads)} 个头把最多的注意力放在开头的标记上——这叫「注意力汇」。`);
  sub(Dk('heads').t0 + 2.6, Dk('dmul').t0 - 0.1, `第 ${dot.head} 头：${m(pct(att13[0].w))} 的注意力给了${q(Q.tokens[att13[0].j].s)}。`);
  sub(Dk('dmul').t0 + 0.1, Dk('dsum').t0 - 0.1, `放大这一次打分：两个 128 维向量，逐项相乘。`);
  sub(Dk('dsum').t0 + 0.05, Dk('mix').t0 - 0.1, `加起来 q·k = ${m(f(dot.sum, 2))}，÷ √128 = ${m(f(dot.score, 2))}，softmax 后 ${m(pct(dot.w))}。`);
  sub(Dk('mix').t0 + 0.1, Dk('add1').t0 - 0.1, '按权重把 V 加起来：「法国」的信息被搬到了这里。');
  sub(Dk('add1').t0 + 0.1, Dk('up').t0 - 0.1, '再乘 W_o 回到 1024 维，加回残差流。');
  sub(Dk('up').t0 + 0.1, Dk('act').t0 - 0.1, '前馈网络 SwiGLU：先把 1024 维扩成 3072 维。');
  sub(Dk('act').t0 + 0.1, Dk('act').t0 + 2.4, 'g 经过 SiLU，像一道阀门，决定每个神经元放行多少 u。');
  sub(Dk('act').t0 + 2.6, Dk('down').t0 - 0.1, `这一步明显激活的有 ${m(Q.mlpCount(G4, LX))} 个神经元，最亮的是 #${m(neu.j)}。`);
  sub(Dk('down').t0 + 0.1, Dk('pick').t0 - 0.1, '再用 W_down 压回 1024 维，加回残差流。');
  sub(Dk('pick').t0 + 0.1, Dk('mul').t0 + 0.6, `放大到一个神经元：#${m(neu.j)}。`);
  sub(Dk('mul').t0 + 0.8, Dk('sum').t0 - 0.1, `1024 个输入，和 W_gate 第 ${m(neu.j)} 列逐项相乘……`);
  sub(Dk('sum').t0 + 0.1, Dk('silu').t0 - 0.1, `……再全部加起来：g = ${m(neu.gz.toFixed(3))}。同一列在 W_up 里得到 u = ${m(neu.uz.toFixed(3))}。`);
  sub(Dk('silu').t0 + 0.1, Dk('gate').t0 - 0.1, `SiLU(g) = g · σ(g) = ${m(neu.silu.toFixed(3))}。`);
  sub(Dk('gate').t0 + 0.1, Dk('bits').t0 - 0.1, `${m(neu.silu.toFixed(3))} × ${m(neu.uz.toFixed(3))} = ${m((neu.silu * neu.uz).toFixed(2))}：这就是神经元 #${m(neu.j)} 的输出。`);
  const wb = neu.wg[0], bits = Array.from(new Uint8Array(new Float32Array([wb]).buffer)); void bits;
  const e2 = Math.floor(Math.log2(Math.abs(wb))), mant = Math.abs(wb) / 2 ** e2;
  sub(Dk('bits').t0 + 0.2, Dk('bits').t0 + 3.6, `再放大：权重 W_gate[${m(neu.dims[0])}, ${m(neu.j)}] = ${m(wb.toFixed(4))}，在内存里只是 16 个比特。`);
  sub(Dk('bits').t0 + 3.8, Dk('out').t0 - 0.1, `1 位符号、8 位指数、7 位尾数：${m(`2^${e2} × ${mant.toFixed(4)}`)}。`);
  sub(Dk('out').t0 + 0.1, Dk('out').t1 - 0.1, `${m('596,049,920')} 个这样的数，一起算出了下一个词。`);
  const fxD = (M, st) => M.camera(st);
  shot('dissect', B(38), B(66), (lt, t, { M }) => {
    const s = pick(D, t);
    if (s.key === 'out') { // 拉出来：回到输出头
      const st = mst(4, G4, { ph: 'head', sub: 'norm' }, 0, { dAnim: 4 });
      return {
        st,
        cam: () => {
          const target = headCam(M, st, 1.6, 1.1, 8.8);
          const far = { pos: v3(17, 13.5, 21), look: v3(5.5, 8.0, 0), fov: 34 };
          const k = s.p;
          if (k < 0.55) return blendCam(endCam.get('dissect-bits') || far, far, easeIn(k / 0.55) * 0.85 + 0.15 * smoother(k / 0.55));
          return blendCam(far, target, smoother((k - 0.55) / 0.45));
        },
        ov: { corner: { g: G4 }, cornerA: 1 },
      };
    }
    let head = null;
    if (s.key === 'score' || s.key === 'softmax' || s.key.startsWith('d')) head = dot.head;
    if (s.key === 'heads') head = Math.min(15, Math.floor(s.p * 16 * 1.0));
    const st = mst(s.depth, G4, { ph: 'layer', L: LX, ...s.s }, s.p0 ? 0 : s.p, { dAnim: s.depth, head });
    const camFor = (M2, k) => {
      const stc = { ...st, head: head == null ? null : dot.head };
      let c = fxD(M2, stc);
      c = { ...c, fov: 32 };
      switch (k) {
        case 'over': return orbit(c, lerp(22, 8, smooth(s.p)), lerp(4, 0, smooth(s.p)), lerp(1.25, 1.02, smooth(s.p)));
        case 'ln1': {
          const xf = M2.x(row4), y = M2.yL(LX) + 0.25 * M2.e;
          const look = v3(xf + 0.35, y + 0.12, 0);
          return { pos: look.clone().add(v3(lerp(-1.7, -1.2, s.p), lerp(1.0, 0.75, s.p), lerp(3.9, 3.2, s.p))), look, fov: 32 };
        }
        case 'qkv': return orbit(c, lerp(-16, 6, smooth(s.p)), 2, lerp(1.1, 0.97, smooth(s.p)));
        case 'score': case 'softmax': case 'heads': {
          const u = smooth(seg(t, Dk('score').t0, Dk('heads').t1));
          return orbit(c, lerp(-18, 14, u), lerp(-2, 4, u), lerp(1.05, 0.95, u));
        }
        case 'dmul': case 'dsum': case 'dscale': return orbit(c, lerp(-8, 6, smooth(seg(t, Dk('dmul').t0, Dk('dscale').t1))), 0, 1.0);
        case 'mix': return orbit(c, lerp(-10, 4, smooth(s.p)), 0, 1.02);
        case 'add1': return orbit(c, lerp(12, 4, smooth(s.p)), 3, 1.0);
        case 'up': case 'act': case 'down': return orbit(c, lerp(-12, 10, smooth(seg(t, Dk('up').t0, Dk('down').t1))), 0, lerp(1.08, 0.94, smooth(seg(t, Dk('up').t0, Dk('down').t1))));
        case 'pick': case 'mul': case 'sum': return orbit(c, lerp(-6, 5, smooth(seg(t, Dk('pick').t0, Dk('sum').t1))), 0, lerp(1.06, 0.98, smooth(seg(t, Dk('pick').t0, Dk('sum').t1))));
        case 'silu': case 'gate': return orbit(c, lerp(-5, 5, smooth(seg(t, Dk('silu').t0, Dk('gate').t1))), 0, 1.0);
        case 'bits': return orbit(c, lerp(-10, 8, smooth(s.p)), lerp(6, 0, smooth(s.p)), lerp(1.15, 0.92, smooth(s.p)));
      }
      return c;
    };
    // 每换一种视图，镜头用 1 秒左右从上一种视图的机位平滑过渡过来
    const VIEW_BLEND = { over: 2.6, ln1: 1.2, qkv: 1.4, score: 1.3, mix: 1.2, add1: 1.1, up: 1.4, pick: 1.4, silu: 1.2, bits: 1.6, dmul: 1.3 };
    return {
      st,
      cam: () => {
        const live = handheld(camFor(M, s.key), t, 0.003);
        let out = live;
        const bl = VIEW_BLEND[s.key];
        if (bl) {
          const from = s.key === 'over' ? prevCam('layers5', live) : endCam.get(`dissect-${D[s.i - 1].key}`) || live;
          out = blendCam(from, live, smoother(clamp((t - s.t0) / bl)));
        }
        endCam.set(`dissect-${s.key}`, out);
        return out;
      },
      dof: s.key === 'bits' ? { range: 0.6, blur: 7 } : (s.key.startsWith('d') || ['pick', 'mul', 'sum', 'silu', 'gate'].includes(s.key)) ? { range: 2.2, blur: 4 } : null,
      ov: {
        att: { a: smooth(seg(t, Dk('softmax').t0 + 0.2, Dk('softmax').t0 + 0.8)) * (1 - smooth(seg(t, Dk('dmul').t0 - 0.2, Dk('dmul').t0 + 0.4))), L: LX, g: G4, head: s.key === 'heads' ? head : dot.head, reveal: seg(t, Dk('softmax').t0 + 0.2, Dk('softmax').t0 + 1.4) },
        corner: { g: G4, L: LX, what: { over: '', ln1: 'RMSNorm', qkv: '注意力 · Q K V', score: '注意力 · 打分', softmax: '注意力 · softmax', heads: '注意力 · 16 个头', dmul: '一次 Q·K 打分', dsum: '一次 Q·K 打分', dscale: '一次 Q·K 打分', mix: '注意力 · 加权求和', add1: '残差 ⊕', up: '前馈 · 升维', act: '前馈 · 门控激活', down: '前馈 · 降维', pick: '一次乘加', mul: '一次乘加', sum: '一次乘加', silu: 'SiLU', gate: 'SiLU × u', bits: 'bf16 比特' }[s.key] },
        cornerA: 1,
      },
    };
  });

  /* ------------------------------------------------------------ I 第 5 个词：输出与采样 165 – 180 */
  section('sample5', B(66), B(72), { energy: 0.7 });
  chapter(B(66) + 0.3, B(72) - 0.2, '08', '输出与采样', 'SAMPLING');
  const samp5 = lay(B(66), [
    { depth: 4, s: { ph: 'head', sub: 'norm' }, d: 2 * BEAT },
    { depth: 5, s: { ph: 'head', sub: 'unembed', mi: 'pick' }, d: 2 * BEAT },
    { depth: 5, s: { ph: 'head', sub: 'unembed', mi: 'mul' }, d: 4 * BEAT },
    { depth: 5, s: { ph: 'head', sub: 'unembed', mi: 'sum' }, d: 2 * BEAT },
    { depth: 4, s: { ph: 'head', sub: 'softmax' }, d: 3 * BEAT },
    { depth: 4, s: { ph: 'sample', sub: 'temp' }, d: 2 * BEAT },
    { depth: 4, s: { ph: 'sample', sub: 'topk' }, d: BEAT },
    { depth: 4, s: { ph: 'sample', sub: 'topp' }, d: 2 * BEAT },
    { depth: 4, s: { ph: 'sample', sub: 'draw' }, d: 6 * BEAT },
  ]);
  const draw5 = samp5[8];
  ev(draw5.t0 + 0.05, 'dice');
  ev(draw5.t0 + draw5.d * 0.8, 'emit', { g: 4, k: 1.4 });
  ev(samp5[1].t0, 'step', { k: 0.6 });
  ev(samp5[4].t0, 'step', { k: 0.6 });
  sub(B(66) + 0.15, samp5[1].t0 - 0.1, '最后一次 RMSNorm。');
  sub(samp5[1].t0 + 0.1, samp5[3].t0 - 0.1, `再乘输出矩阵：它和嵌入表是同一块权重。${q(hm4.token)}的分数 = h · E[${tk(hm4.token)}]……`);
  sub(samp5[3].t0 + 0.05, samp5[4].t0 + 0.3, `……${m('1024')} 项乘加，= ${m(hm4.total.toFixed(2))}。151936 个词元，每个都这样算一遍。`);
  sub(samp5[4].t0 + 0.5, samp5[6].t0 - 0.05, `softmax：${q(st4.top[0][2])} ${m(pct(st4.top[0][1]))}；温度 0.7 之后 ${m(pct(st4.temps['0.7'][0]))}。`);
  sub(samp5[6].t0 + 0.05, draw5.t0 + 1.2, `top-p 截断：只剩 ${m(st4.pool.length)} 个候选。`);
  sub(draw5.t0 + 1.5, B(72) - 0.15, `u = ${m(st4.u.toFixed(3))}。第 5 个词：${q(st4.chosenS)}。`);
  strip(samp5[1].t0 + 0.3, samp5[4].t0, `logit[${m(hm4.j)}] = h ${m('[1024]')} · E[${m(hm4.j)}] ${m('[1024]')}`);
  strip(draw5.t0 + 0.2, B(72) - 0.15, `候选池：${q(st4.pool[0][2])} ${m(pct(st4.pool[0][1]))}<span class="sep"></span>u = ${m(st4.u.toFixed(4))}`);
  shot('sample5', B(66), B(72), (lt, t, { M }) => {
    const s = pick(samp5, t);
    const st = mst(s.depth, G4, s.s, s.p, { dAnim: s.depth });
    const isMM = !!s.s.mi;
    return {
      st,
      cam: () => {
        const head = headCam(M, st, lerp(1.4, 0.7, smooth(lt / 15)), lerp(1.0, 0.75, smooth(lt / 15)), lerp(8.2, 7.4, smooth(lt / 15)));
        let live = head;
        const mmK = smooth(seg(t, samp5[1].t0, samp5[1].t0 + 1.2)) * (1 - smooth(seg(t, samp5[4].t0, samp5[4].t0 + 1.2)));
        if (mmK > 0) {
          const mc = M.micro.camera({ ...st, step: isMM ? st.step : { g: G4, ph: 'head', sub: 'unembed', mi: 'sum' } });
          if (mc) endCam.set('mm5', { ...mc, fov: 32 });
          const mcc = endCam.get('mm5');
          if (mcc) live = blendCam(head, orbit(mcc, lerp(-6, 6, seg(t, samp5[1].t0, samp5[4].t0)), 0, 1.0), mmK);
        }
        return blendCam(prevCam('dissect', live), live, smoother(seg(lt, 0, 0.3)));
      },
      ov: { corner: { g: G4, what: s.s.ph === 'sample' ? '采样' : '输出头' }, cornerA: 1 },
    };
  });

  /* ------------------------------------------------------------ J 句号与结束符 180 – 190 */
  section('loop2', B(72), B(76), { energy: 0.5 });
  chapter(B(72) + 0.3, B(76) - 0.2, '09', '说完', 'END OF TURN');
  const loop2 = [tokSched(5, B(72), 1.75 * BAR), tokSched(6, B(73, 3), 2.25 * BAR)];
  ev(loop2[0].emit, 'emit', { g: 5 });
  ev(loop2[1].emit, 'end');
  sub(B(72) + 0.3, B(73) + 2.0, `${q(st4.chosenS)} 接回去，再算一轮：句号。`);
  sub(B(73) + 2.3, B(76) - 0.2, `最后选中了结束标记 ${m('&lt;|im_end|&gt;')}（${m(pct(Q.steps[6].chosenP1))}）：回答到此为止。`);
  shot('loop2', B(72), B(76), (lt, t, { M }) => {
    const s = loopState(loop2, t);
    const st = mst(2, s.g, { ph: s.ph }, s.p, { dAnim: 2.7 });
    const n = 5 + loop2.filter((x) => t >= x.emit).length;
    const newest = loop2.filter((x) => t >= x.emit).pop();
    return {
      st,
      cam: () => { const live = machineCam(M, t, lerp(-14, 6, smooth(lt / 10)), lerp(0.95, 1.06, smooth(lt / 10))); return blendCam(prevCam('sample5', live), live, smoother(seg(lt, 0, 2.4))); },
      ov: { reply: { a: smooth(seg(lt, 0.2, 0.8)), n: Math.max(5, n), k: newest ? smooth(seg(t, newest.emit, newest.emit + 0.4)) : 1 }, corner: { g: s.g }, cornerA: 1 },
    };
  });

  /* ------------------------------------------------------------ K 片尾 190 – 210 */
  section('end', B(76), B(84), { energy: 0.25 });
  // 一共做了多少次乘加：每层的矩阵乘法 + 注意力里的 Q·K 和加权求和 + 输出头
  const Mo = Q.M, perTokLayer = Mo.hidden * (Mo.heads * Mo.headDim) * 2 + Mo.hidden * (Mo.kvHeads * Mo.headDim) * 2 + Mo.hidden * Mo.ffn * 3;
  let macs = 0;
  for (let pos = 0; pos < Q.P + Q.G - 1; pos++) macs += NL * (perTokLayer + 2 * Mo.heads * Mo.headDim * (pos + 1));
  macs += Q.G * Mo.hidden * Mo.vocab;
  const statLine = `${Q.G} 个词元 · 每个都走完 28 层 · 约 ${Math.round(macs / 1e8)} 亿次乘加`;
  shot('end', B(76), B(84), (lt, t, { M }) => {
    const dA = lerp(2.7, 1, smoother(seg(t, B(76) + 0.4, B(77) + 1.4)));
    return {
      st: mst(2, 6, { ph: 'sample' }, 1, { dAnim: dA, view: 'machine' }),
      cam: () => {
        const far = { pos: v3(-7.5, 4.0, 46), look: v3(0, 5.8, 0), fov: 30 };
        const live = machineCam(M, B(76), 6, 1.06);
        return blendCam(prevCam('loop2', live), far, smoother(seg(lt, 0, 9)));
      },
      fade: lerp(0, 0.6, smooth(seg(t, B(77), B(78)))) + 0.4 * smooth(seg(t, B(83) + 1.0, B(84) - 0.05)),
      ov: {
        reply: { a: 1 - smooth(seg(lt, 0.2, 1.2)), n: Q.G, k: 1 },
        band: 1 - smooth(seg(lt, 0, 1)),
        end: { a1: smooth(seg(t, B(77) + 0.4, B(77) + 1.4)) * (1 - smooth(seg(t, B(80) + 0.4, B(80) + 1.2))), k1: smooth(seg(t, B(78) + 0.6, B(78) + 1.4)), a2: smooth(seg(t, B(80) + 1.4, B(81) + 0.2)) * (1 - smooth(seg(t, B(83) + 1.2, B(84) - 0.1))), k2: seg(t, B(80) + 1.4, B(81) + 1.0) },
      },
    };
  });
  ev(B(77) + 0.4, 'hit', { k: 0.5 });

  const end = B(84);
  shots.forEach((s, i) => { s.i = i; });

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
