// 多模态电影的分镜表：每个镜头的时间、舞台状态、机位、字幕、术语标签、章节、配乐事件。
// 节拍：96 BPM（一拍 0.625 秒，一小节 2.5 秒），段落都卡在小节线上；配乐（../compose.py）读 render.mjs 导出的段落表和事件表。
//
// 图和问题：「桌上的苹果」+「绿色的苹果是第几个？」（apples-2），回答「绿色的苹果是第3个。」。
// 字幕、术语标签里的数字全部从真实数据里现取（manifest、apples.json/.bin、apples-2.json/.bin）。
import { THREE } from '../lib/engine.js';
import { path, blendCam, orbit, handheld, clamp, lerp, seg, smooth, smoother, easeInOut, easeOut, v3 } from '../lib/cam.js';
import { viewOf } from '/public/multimodal/js/timeline.js';
import { tokPlain, esc } from '/public/js/ui.js';

export const BPM = 96, BEAT = 60 / BPM, BAR = 4 * BEAT;
const CMP = 6;   // ViT 对比用的那一级：第 6 层之后五个苹果（连绿的那个）主成分颜色最一致（看过 25 级以后选的）
const B = (bar, beat = 0) => bar * BAR + beat * BEAT;

// 舞台的几何（和 public/multimodal/js/scene.js 里的常数一致）
export const G = { PS: 0.1, LVL: 0.16, VX: -3.7, Y0: 0.32, PX: -7.6, PY: 1.7, S: 0.27, LY0: 0.6, LG: 0.16, X0: -1.3 };
G.MY = G.Y0 + 24 * G.LVL + 0.55;

// 段落：名字、小节数、能量（配乐用）
const PLAN = [
  ['chat', 6, 0.12], ['fly', 4, 0.6], ['patch', 4, 0.35], ['calc', 8, 0.3], ['vit', 7, 0.62], ['merge', 5, 0.5],
  ['splice', 5, 0.45], ['answer', 13, 0.55], ['lens', 5, 0.7], ['reply', 4, 0.8], ['end', 10, 0.22],
];
const CHAPTERS = ['切成图块', '视觉编码器', '四合一', '插进对话', '看图回答'];

const m = (s) => `<span class="m">${s}</span>`;
const pct = (p) => `${(p * 100).toFixed(p >= 0.995 ? 0 : p < 0.1 ? 1 : 0)}%`;
const sgn = (v, d = 3) => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(d);
const num = (v, d = 3) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
const V3 = (a) => v3(a[0], a[1], a[2]);
const cam = (p, l, fov = 32) => ({ pos: V3(p), look: V3(l), fov });

export function mst(depth, g, step, p, o = {}) {
  const s = { g, ...step };
  return { step: s, p: clamp(p), g, depth, dAnim: o.dAnim ?? depth, view: o.view ?? viewOf(depth, s), vq: o.vq ?? null };
}

// 顺序排列的若干段：[{ ...data, d }] 从 t0 开始
function lay(t0, items) {
  let t = t0;
  return items.map((it) => { const a = { ...it, t0: t, t1: t + it.d }; t += it.d; return a; });
}
function pick(list, t) {
  for (let i = 0; i < list.length; i++) if (t < list[i].t1 || i === list.length - 1) return { ...list[i], i, p: clamp((t - list[i].t0) / (list[i].t1 - list[i].t0)) };
  return null;
}

// 从真实数据里算出片子要讲的数
export function facts(Q) {
  const V = Q.V, man = Q.manifest;
  const gAns = Q.steps.findIndex((s) => /^\s*3\s*$/.test(s.chosenS));   // 说出「3」的那一步
  const gDi = gAns - 1;                                                    // 「第」
  // 绿苹果：生成「3」时第 16–26 层平均注意力最强的那个视觉词元；区域取 tools/multimodal/grounding.py 里标注的绿苹果框（合并网格的行 4–7、列 6–8）
  const [ga, gb] = man.groundLayers;
  const avg = Q.attAvg(gAns, [ga, gb]);
  let hot = 0;
  for (let k = 1; k < V.Nv; k++) if (avg[k] > avg[hot]) hot = k;
  const box = { r0: 4, r1: 7, c0: 6, c1: 8 };
  const inBox = (k) => { const r = Math.floor(k / V.mw), c = k % V.mw; return r >= box.r0 && r <= box.r1 && c >= box.c0 && c <= box.c1; };
  let nIn = 0;
  for (let k = 0; k < V.Nv; k++) if (inBox(k)) nIn++;
  const base = nIn / V.Nv;
  const enr = Array.from({ length: Q.NL }, (_, L) => {
    const a = Q.attImg(gAns, L);
    let s = 0, si = 0;
    for (let k = 0; k < V.Nv; k++) { s += a[k]; if (inBox(k)) si += a[k]; }
    return s > 0 ? si / s / base : 0;
  });
  const mean = (a, b) => enr.slice(a, b + 1).reduce((x, y) => x + y, 0) / (b - a + 1);
  // 逻辑透镜：「3」这一步每层的第一名
  const lens = Array.from({ length: Q.NL }, (_, L) => Q.lensAt(gAns, L)[0]);
  // 心里先想的：第一次以过半的把握读出带「三」的词（第 21 层「三个」）
  let thinkL = lens.findIndex((x) => /三/.test(x[0]) && x[1] >= 0.5);
  if (thinkL < 0) thinkL = Q.NL - 1;
  const stA = Q.steps[gAns];
  // 绿苹果正中的那个视觉词元：模型看到的像素里，每个 2×2 词元格子平均“绿得最突出”的那一格（M-RoPE、合并举例用）
  let apple = hot;
  try {
    const cv = new OffscreenCanvas(V.img.naturalWidth, V.img.naturalHeight);
    const g2 = cv.getContext('2d');
    g2.drawImage(V.img, 0, 0);
    const px = g2.getImageData(0, 0, cv.width, cv.height).data;
    const cw = cv.width / V.mw, chh = cv.height / V.mh;
    let best = -1e9;
    for (let k = 0; k < V.Nv; k++) {
      const r = Math.floor(k / V.mw), c = k % V.mw;
      let s = 0, n = 0;
      for (let y = Math.floor(r * chh); y < Math.floor((r + 1) * chh); y += 2) for (let x = Math.floor(c * cw); x < Math.floor((c + 1) * cw); x += 2) { const o = (y * cv.width + x) * 4; s += px[o + 1] - (px[o] + px[o + 2]) / 2; n++; }
      if (s / n > best) { best = s / n; apple = k; }
    }
  } catch { /* 取不到像素就用热点 */ }
  // M-RoPE
  const p0 = Q.pos(V.vs), pHot = Q.pos(V.vs + hot), pApple = Q.pos(V.vs + apple);
  const pLast = Q.pos(V.vs + V.Nv - 1);
  const firstQ = Q.tokens.findIndex((t, i) => i > V.vs + V.Nv && t.role === 'user' && !t.sp);
  const pQ = Q.pos(firstQ);
  // ViT 每层（16 头平均）的注意距离
  const dist = V.vitDist.map((h) => h.reduce((a, b) => a + b, 0) / h.length);
  return {
    gAns, gDi, hot, apple, pApple, thinkL, box, base, enr, enrLo: mean(0, ga - 1), enrHi: mean(ga, gb), ga, gb, lens, stA,
    p0, pHot, pLast, pQ, firstQ, nPos: pLast[2] - p0[2] + 1, dist, distMin: Math.min(...dist), distMax: Math.max(...dist),
  };
}

export function buildScore(Q, cap = null) {
  const V = Q.V, man = Q.manifest, mv = man.model.vision, mt = man.model.text, mic = V.micro;
  const F = facts(Q);
  const subs = [], terms = [], events = [], sections = [], progs = [];
  const sub = (t0, t1, html) => subs.push({ t0, t1, html });
  const term = (t0, t1, html) => terms.push({ t0, t1, html });
  const ev = (t, type, o = {}) => events.push({ t, type, ...o });
  const prog = (t, i) => progs.push({ t, i });

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
  const endCam = new Map();
  const prevCam = (name, fallback) => endCam.get(name) || fallback;
  const user = Q.tokens.map((t, i) => ({ ...t, i })).filter((t) => t.role === 'user' && !t.sp && t.s !== 'user' && t.s !== '\n');
  const pw = V.gw * G.PS, ph = V.gh * G.PS;
  const W0 = V.spec.orig[0], H0 = V.spec.orig[1], W1 = V.gw * mv.patch, H1 = V.gh * mv.patch;

  /* ------------------------------------------------------------ 开场：聊天页 → 选图 → 打字 → 发送 → 推近 → 飞进黑箱 */
  const PY = {
    '绿色': ["lv'se", ['绿色', '绿', '率', '律']],
    '的': ['de', ['的', '得', '地', '德']],
    '苹果': ["ping'guo", ['苹果', '平果', '苹', '平']],
    '是': ['shi', ['是', '时', '事', '使']],
    '第': ['di', ['第', '地', '低', '帝']],
    '几个': ["ji'ge", ['几个', '及格', '几', '机']],
  };
  const upAt = [B(2, 0), B(2, 1), B(2, 2), B(2, 3), B(3, 0), B(3, 1), B(3, 1.75)];
  const jit = (i) => (((i * 7919) % 13) / 13 - 0.5) * 0.03;
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
    attachT: B(1, 0), pickT: B(1, 2),                  // 点「图片」按钮；点中苹果那张
    sendT: B(3, 3),
    boxY0: 590, boxY1: 912,
    push0: B(3, 3) + 1.2, push1: B(6) - 0.1,
    zoom: 1.75, zoom2: 0.05, focusY: 470,
    split0: B(5, 1.5), splitGap: 0.09, splitD: 0.45,
    diss0: B(6) - 0.05, diss1: B(6) + 0.6,
    swapT: B(6) + 0.7,
    depth: 7,
  };
  OPEN.bubT = OPEN.sendT + 0.06;
  OPEN.botT = OPEN.sendT + 0.4;
  OPEN.warpT = OPEN.swapT - 1.0;
  OPEN.fog0 = OPEN.swapT + 0.4; OPEN.fog1 = SEC.fly.t0 + 1.6;
  // 飞行：图和词元先钻进黑箱的取景窗；镜头再拉开，片名落在黑箱正面；然后黑箱打开
  const F0 = SEC.fly.t0;
  OPEN.dock0 = F0 + 2.2;                                             // 离开跟随编队，飞向取景窗
  OPEN.enter0 = F0 + 3.6; OPEN.enter1 = F0 + 4.6;                    // 图片从取景窗推进去
  const titleIn = F0 + 5.0, titleOut0 = F0 + 7.4, titleOut1 = F0 + 8.0;
  Object.assign(OPEN, { titleIn, titleOut0, titleOut1 });
  OPEN.open0 = F0 + 7.9; OPEN.open1 = SEC.patch.t0 + 1.0;            // 黑箱打开
  OPEN.end = SEC.patch.t0;
  // 飞行：图片和词元交给 3D 以后，镜头跟着它们从远处飞到黑箱的取景窗前
  const box = { x0: G.PX - pw / 2 - 0.6, z0: Math.max(V.mh * G.S, ph) / 2 + 0.6 };
  const winZ = box.z0 + 2.6;   // 图片进窗前停的位置（scene.js 里 enter = 0 时照片的 z）
  OPEN.photoEnd = v3(G.PX, G.PY, winZ);
  const camOpen = path([
    { t: 0, p: [G.PX, 4.0, 130], l: [G.PX, 4.0, 0], fov: 32 },
    { t: OPEN.swapT - 0.3, p: [G.PX, 4.0, 130], l: [G.PX, 4.0, 0], fov: 32 },
    { t: OPEN.swapT + 0.5, p: [G.PX + 0.4, 4.1, 112], l: [G.PX + 0.4, 3.9, 0], fov: 32 },
    { t: OPEN.dock0, p: [G.PX + 1.6, 3.4, 40], l: [G.PX + 1.0, 2.4, 0], fov: 32 },
    { t: OPEN.enter0, p: [G.PX + 2.0, 2.7, 13.5], l: [G.PX + 0.3, 1.8, 2.5], fov: 32 },
    { t: OPEN.enter1 + 0.2, p: [G.PX + 1.5, 2.5, 10.5], l: [G.PX + 0.2, 1.7, 1.5], fov: 32 },
    { t: titleIn + 0.9, p: [2.6, 3.7, 31], l: [3.4, 2.5, 0], fov: 32 },
    { t: OPEN.open0, p: [2.9, 3.5, 28], l: [3.4, 2.4, 0], fov: 32 },
    { t: OPEN.open1 - 0.4, p: [G.PX + 1.2, 3.2, 9.2], l: [G.PX + 0.4, 1.6, 0], fov: 32 },
    { t: SEC.patch.t0 + 2.5, p: [G.PX, 2.25, 6.9], l: [G.PX, 1.7, 0], fov: 32 },
  ]);
  OPEN.camAt = camOpen;
  let kn = 0;
  for (const w of typing) if (w.py) for (const k of w.keys) ev(k, 'key', { i: kn++ });
  typing.forEach((w, i) => ev(w.commit, 'type', { i }));
  ev(OPEN.attachT, 'tick', { k: 0.5 });
  ev(OPEN.pickT, 'tick', { k: 0.8 });
  ev(OPEN.sendT, 'send');
  ev(OPEN.swapT, 'whoosh', { k: 0.8 });
  ev(OPEN.swapT, 'rise', { d: OPEN.enter0 - OPEN.swapT });
  ev(OPEN.enter0, 'whoosh', { k: 0.6 });
  ev(titleIn - 1.0, 'rise', { d: 1.0 });
  ev(titleIn, 'hit', { k: 1 });
  ev(OPEN.open0 - 1.6, 'rise', { d: 1.6 });
  ev(OPEN.open0, 'open');
  sub(OPEN.split0 + 0.3, OPEN.diss0 + 0.3, '发出去的，是一张图和一句话');
  sub(OPEN.dock0 + 0.2, titleIn - 0.3, '图和字，一起进了黑箱');
  sub(OPEN.open0 + 0.2, SEC.patch.t0 - 0.15, '打开看看：图怎么变成“词”？');
  shotSpan('opening', 0, SEC.patch.t0, (lt, t) => {
    const opened = t >= OPEN.open0;
    return {
      st: opened ? mst(3, 0, { ph: 'prep', sub: 'resize' }, 0, { dAnim: lerp(1, 3, smoother(seg(t, OPEN.open0, OPEN.open1))) }) : mst(1, 0, { ph: 'see' }, seg(t, OPEN.enter0, OPEN.enter1)),
      cam: () => camOpen(t),
      fade: 1 - smooth(seg(t, OPEN.diss0, OPEN.diss0 + 0.3)),
      boxLabel: 1 - smooth(seg(t, titleIn - 0.6, titleIn)) * (1 - smooth(seg(t, titleOut1 + 0.05, titleOut1 + 0.55))),
      photoA: t >= OPEN.enter0 - 0.02 ? 1 : 0,
      ov: { band: smooth(seg(t, OPEN.diss0, OPEN.diss1)), title: smooth(seg(t, titleIn, titleIn + 0.8)) * (1 - smooth(seg(t, titleOut0, titleOut1))), titleK: seg(t, titleIn + 0.05, titleIn + 2.6), titleBlur: 7 * smooth(seg(t, titleOut0, titleOut1)), titleOnBox: true },
    };
  });

  /* ------------------------------------------------------------ ① 切成图块 */
  const photoCam = (k = 1) => ({ pos: v3(G.PX, G.PY + 0.55 * k, 6.9 * k), look: v3(G.PX, G.PY, 0), fov: 32 });
  {
    const T0 = SEC.patch.t0, T1 = SEC.patch.t1;
    prog(T0, 0);
    const rz0 = T0 + 0.6, rz1 = T0 + 3.4, cut0 = T0 + 5.0, cut1 = T0 + 8.2;
    ev(rz0, 'reveal', { k: 0.5 });
    for (let r = 0; r < V.gh; r += 2) ev(cut0 + ((cut1 - cut0) * r) / V.gh, 'blip', { i: r });
    sub(T0 + 0.3, T0 + 4.6, '先缩放：边长凑成 32 的倍数');
    term(T0 + 0.4, T0 + 4.6, `${m(`${W0}×${H0}`)} → ${m(`${W1}×${H1}`)}`);
    sub(T0 + 4.8, T1 - 0.15, `再切成 ${mv.patch}×${mv.patch} 像素的小方块`);
    term(T0 + 4.9, T1 - 0.15, `图块 patch · ${m(V.gh)} × ${m(V.gw)} = ${m(V.Np)} 块`);
    shot('patch', (lt, t) => {
      const cutting = t >= cut0;
      return {
        st: cutting ? mst(3, 0, { ph: 'prep', sub: 'patch' }, seg(t, cut0, cut1) / 1.3) : mst(3, 0, { ph: 'prep', sub: 'resize' }, seg(t, rz0, rz1)),
        cam: () => { const c = photoCam(lerp(1, 1.08, smooth(seg(t, cut0, T1)))); return lt < 2.5 ? blendCam(camOpen(t), c, smoother(lt / 2.5)) : c; },
        dims: { a: smooth(seg(t, rz0, rz0 + 0.5)) * (1 - smooth(seg(t, T1 - 0.6, T1))), resize: seg(t, rz0, rz1), cut: seg(t, cut0, cut1) },
      };
    });
  }

  /* ------------------------------------------------------------ ①′ 一次乘加：一个图块 × 卷积核 → 嵌入的一个数 */
  const hotPatch = { r: mic.row, c: mic.col, i: mic.patch, tok: mic.token };
  {
    const T0 = SEC.calc.t0, T1 = SEC.calc.t1;
    const k = (x) => T0 + x;
    const S1 = k(0.2), S2 = k(4.0), S3 = k(7.6), S4 = k(11.4), S5 = k(14.8), S6 = k(17.4);
    // 板上一步步出现：挑出一块 → 1536 个数 → 卷积核 → 乘积 → 求和 → 1024 个核
    const CALC = { lift: [k(0.3), k(2.0)], board: [k(1.6), k(2.4)], pix: k(2.0), frames: k(4.6), kern: k(8.0), prods: [k(9.6), k(11.2)], sum: k(12.6), bias: k(14.0), res: k(15.2), many: k(17.6), out: [T1 - 0.9, T1 - 0.1] };
    ev(CALC.lift[0], 'whoosh', { k: 0.45 });
    ev(CALC.frames, 'tick', { k: 0.6 });
    ev(CALC.kern, 'reveal', { k: 0.6 });
    for (let j = 0; j < 3; j++) ev(lerp(CALC.prods[0], CALC.prods[1], j / 3), 'blip', { i: j });
    ev(CALC.sum, 'tick', { k: 0.7 });
    ev(CALC.res, 'hit', { k: 0.55 });
    sub(S1, S2 - 0.1, '拿出一块，看它怎么变成数字');
    term(S1 + 0.1, S2 - 0.1, `绿苹果上的一块 · 第 ${m(hotPatch.r)} 行第 ${m(hotPatch.c)} 列`);
    sub(S2, S3 - 0.1, `这一小块，有 ${m(mv.inDim)} 个数`);
    term(S2 + 0.1, S3 - 0.1, `${mv.patch}×${mv.patch} 像素 × 红绿蓝 3 × ${mv.temporal} 帧`);
    sub(S3, S4 - 0.1, '每个数，乘上一个权重');
    term(S3 + 0.1, S4 - 0.1, `卷积核 #${mic.ch} · 也是 ${m(mv.inDim)} 个数`);
    sub(S4, S5 - 0.1, `${m(mv.inDim)} 个乘积，全部加起来`);
    term(S4 + 0.1, S5 - 0.1, `再加上偏置 ${m(num(mic.bias))}`);
    sub(S5, S6 - 0.1, `得到一个数：${m(num(mic.mine))}`);
    term(S5 + 0.1, S6 - 0.1, `图块嵌入的第 ${m(mic.ch)} 维`);
    sub(S6, T1 - 0.15, `${m(mv.hidden)} 个卷积核，就得到 ${m(mv.hidden)} 个数`);
    term(S6 + 0.1, T1 - 0.15, `图块嵌入 · 每块变成 ${m(mv.hidden)} 维向量`);
    shot('calc', (lt, t) => {
      const lift = smoother(seg(t, CALC.lift[0], CALC.lift[1])) * (1 - smoother(seg(t, CALC.out[0], CALC.out[1])));
      return {
        st: mst(3, 0, { ph: 'prep', sub: 'patch' }, 1),
        cam: () => blendCam(photoCam(1.08), photoCam(1.0), smooth(seg(t, T0, T0 + 3))),
        lift,
        fade: 0.62 * smooth(seg(t, CALC.board[0], CALC.board[1])) * (1 - smooth(seg(t, CALC.out[0], CALC.out[1]))),
        calc: { a: smooth(seg(t, CALC.board[0], CALC.board[1])) * (1 - smooth(seg(t, CALC.out[0] - 0.2, CALC.out[1]))), t, C: CALC },
        hideLabels: true,
      };
    });
  }

  /* ------------------------------------------------------------ ② 视觉编码器：24 层 */
  {
    const T0 = SEC.vit.t0, T1 = SEC.vit.t1;
    prog(T0, 1);
    const fly0 = T0 + 0.2, fly1 = T0 + 3.2, up0 = T0 + 3.4, up1 = T0 + 12.4;
    ev(fly0, 'whoosh', { k: 0.7 });
    ev(up0, 'rise', { d: up1 - up0 });
    for (let L = 0; L < 24; L += 2) ev(lerp(up0, up1, (L + 1) / 25), 'layer', { L, n: 24, k: 0.5 });
    const cmp0 = up1 + 0.2;
    ev(cmp0, 'reveal', { k: 0.8 });
    sub(T0 + 0.3, up0 + 1.4, `${m(V.Np)} 块一起，送进视觉编码器`);
    term(T0 + 0.4, up0 + 1.4, `视觉编码器 ViT · ${m(mv.depth)} 层 · 每块 ${m(mv.hidden)} 维`);
    sub(up0 + 1.6, up0 + 5.4, '每一层，每块都在看其他块');
    term(up0 + 1.7, up0 + 5.4, `注意力 · 平均看 ${m(F.distMin.toFixed(0))}–${m(F.distMax.toFixed(0))} 块远`);
    sub(up0 + 5.6, cmp0 - 0.1, '颜色 = 模型对这一块的“理解”');
    term(up0 + 5.7, cmp0 - 0.1, '特征的前 3 个主成分 → 红绿蓝');
    sub(cmp0 + 0.1, cmp0 + 2.6, '一开始：颜色跟着位置和像素走');
    term(cmp0 + 0.2, cmp0 + 2.6, '红的、黄的、绿的，各是各的');
    sub(cmp0 + 2.7, T1 - 0.15, `第 ${CMP - 1} 层之后：五个苹果一个颜色`);
    term(cmp0 + 2.8, T1 - 0.15, '连绿苹果也一样：它们都是“苹果”');
    const vitCam = (lv, o = {}) => {
      const look = v3(G.VX, G.Y0 + lv * G.LVL, 0);
      return orbit({ pos: look.clone().add(v3(o.dx ?? 3.2, o.dy ?? 3.1, o.dz ?? 5.6)), look, fov: 32 }, o.yaw ?? 0, 0, o.dist ?? 1);
    };
    shot('vit', (lt, t) => {
      const flying = t < up0;
      const st = flying ? mst(3, 0, { ph: 'vit', sub: 'embed' }, seg(t, fly0, fly1)) : mst(2, 0, { ph: 'vit' }, seg(t, up0, up1) * 0.999);
      const lv = flying ? 0 : Math.min(24, seg(t, up0, up1) * 25);
      return {
        st,
        cam: () => {
          const embedCam = cam([G.VX - 0.6, 3.9, 9.6], [G.VX - 1.0, 1.1, 0.2]);
          if (flying) return blendCam(prevCam('calc', photoCam()), embedCam, smoother(seg(t, T0, fly1)));
          const live = vitCam(lv, { yaw: lerp(-8, 14, seg(t, up0, T1)), dist: lerp(1.0, 1.25, smooth(seg(t, up1, T1))) });
          return blendCam(embedCam, live, smoother(seg(t, up0, up0 + 2.2)));
        },
        panel: { kind: 'pca', a: smooth(seg(t, up0 + 0.4, up0 + 1.2)) * (1 - smooth(seg(t, T1 - 0.6, T1))), lv: Math.floor(lv), cmp: smooth(seg(t, cmp0, cmp0 + 1.0)), cmpL: CMP, cmpK: smooth(seg(t, cmp0 + 2.6, cmp0 + 3.4)) },
        vitHl: !flying,
      };
    });
  }

  /* ------------------------------------------------------------ ③ 四合一 + DeepStack */
  {
    const T0 = SEC.merge.t0, T1 = SEC.merge.t1;
    prog(T0, 2);
    const g0 = T0 + 0.4, g1 = T0 + 3.6, ds0 = T0 + 7.6;
    ev(g0, 'whoosh', { k: 0.5 });
    ev(g1, 'hit', { k: 0.5 });
    ev(ds0, 'reveal', { k: 0.7 });
    sub(T0 + 0.3, T0 + 4.4, `相邻 2×2 块，合成一个“词”`);
    term(T0 + 0.4, T0 + 4.4, `${m(V.Np)} 块 → ${m(V.Nv)} 个视觉词元`);
    sub(T0 + 4.6, ds0 - 0.1, `再翻译成语言模型的 ${m(mv.out)} 维`);
    term(T0 + 4.7, ds0 - 0.1, `合并器 · ${m(mv.hidden * 4)} → ${m(mv.out)}`);
    sub(ds0 + 0.1, T1 - 0.15, '中途三层，也抄近路送过去');
    term(ds0 + 0.2, T1 - 0.15, `DeepStack · 第 ${mv.deepstack.join('、')} 层 → 语言模型前 3 层`);
    const mergeCam = cam([G.VX + 1.9, G.MY + 2.8, 4.4], [G.VX, G.MY - 0.25, 0]);
    const dsCam = cam([G.VX + 5.2, 4.4, 10.5], [G.VX + 3.6, 2.6, 0]);
    shot('merge', (lt, t) => {
      const st = t < ds0 ? mst(3, 0, { ph: 'merge', sub: t < g1 + 0.6 ? 'group' : 'mlp' }, t < g1 + 0.6 ? seg(t, g0, g1) : 1) : mst(3, 0, { ph: 'merge', sub: 'deep' }, seg(t, ds0, T1));
      return {
        st,
        cam: () => {
          const c = blendCam(prevCam('vit', mergeCam), mergeCam, smoother(seg(lt, 0, 2.0)));
          return blendCam(c, dsCam, smoother(seg(t, ds0 - 0.6, ds0 + 1.6)));
        },
        panel: { kind: 'merge', a: smooth(seg(t, g0 - 0.2, g0 + 0.5)) * (1 - smooth(seg(t, ds0 - 0.6, ds0))), k: seg(t, g0, g1) },
      };
    });
  }

  /* ------------------------------------------------------------ ④ 插进对话 + M-RoPE */
  {
    const T0 = SEC.splice.t0, T1 = SEC.splice.t1;
    prog(T0, 3);
    const l0 = T0 + 0.3, l1 = T0 + 3.6, mr0 = T0 + 5.0;
    ev(l0, 'whoosh', { k: 0.6 });
    for (let j = 0; j < 6; j++) ev(lerp(l0 + 0.8, l1, j / 6), 'tick', { i: j, k: 0.35 });
    ev(mr0, 'reveal', { k: 0.6 });
    const ph = F.pApple, pq = F.pQ;
    sub(T0 + 0.3, mr0 - 0.1, `${m(V.Nv)} 个视觉词，插进对话里`);
    term(T0 + 0.4, mr0 - 0.1, `${m('&lt;|vision_start|&gt;')} ×${V.Nv} ${m('&lt;|vision_end|&gt;')}`);
    sub(mr0, mr0 + 2.4, '每个词带三个坐标');
    term(mr0 + 0.1, mr0 + 2.4, 'M-RoPE ·（时间, 行, 列）');
    sub(mr0 + 2.5, mr0 + 4.9, `绿苹果这一格：${m(`(${ph.join(', ')})`)}`);
    term(mr0 + 2.6, mr0 + 4.9, `问题里的「${esc(tokPlain(Q.tokens[F.firstQ].s))}」：${m(`(${pq.join(', ')})`)}`);
    sub(mr0 + 5.0, T1 - 0.15, `${m(V.Nv)} 个词，只占 ${m(F.nPos)} 个位置号`);
    term(mr0 + 5.1, T1 - 0.15, `图片 ${m(F.p0[0])}–${m(F.pLast[2])} · 文字从 ${m(pq[0])} 接着数`);
    const cx = (i) => G.X0 + i * G.S;   // 近似：序列前段的 x
    shot('splice', (lt, t) => {
      const st = t < mr0 ? mst(3, 0, { ph: 'splice', sub: 'template' }, seg(t, l0, l1)) : mst(3, 0, { ph: 'splice', sub: 'mrope' }, 1);
      return {
        st,
        cam: (ctx) => {
          const Sc = ctx.S;
          const wide = cam([(G.VX + Sc.bcx) / 2 + 1.2, 5.8, 15.5], [(G.VX + Sc.bcx) / 2 + 0.9, 2.0, 0]);
          const near = cam([Sc.bcx + 0.4, 4.6, 4.6], [Sc.bcx + 0.4, 0.1, 0.2]);
          const c = blendCam(prevCam('merge', wide), wide, smoother(seg(lt, 0, 2.2)));
          return blendCam(c, near, smoother(seg(t, mr0 - 0.8, mr0 + 1.4)));
        },
        panel: { kind: 'mrope', a: smooth(seg(t, mr0 + 0.4, mr0 + 1.1)) * (1 - smooth(seg(t, T1 - 0.6, T1))), hl: t > mr0 + 2.5 ? 'hot' : null, rows: t > mr0 + 5.0 },
      };
    });
  }

  /* ------------------------------------------------------------ ⑤ 看图回答：语言模型 28 层，「3」的热力图 */
  const gA = F.gAns;
  const sched = (() => {
    const T0 = SEC.answer.t0;
    // 前几个字快速带过（每个 1.3 秒），「3」逐层慢慢看
    const pre = lay(T0 + 4.2, Array.from({ length: gA }, (_, g) => ({ g, d: 1.25 })));
    return pre;
  })();
  {
    const T0 = SEC.answer.t0, T1 = SEC.answer.t1;
    prog(T0, 4);
    const preEnd = sched[sched.length - 1].t1;
    // 「3」：28 层，前 16 层快、16–26 层慢
    const A0 = preEnd + 2.6;
    const raw = Array.from({ length: mt.layers }, (_, L) => (L < F.ga ? 0.22 : L <= F.gb ? 0.6 : 0.4));
    const layers = lay(A0, raw.map((d, L) => ({ L, d })));
    const A1 = layers[layers.length - 1].t1;
    const lens0 = A1 + 0.4;
    sched.forEach((s) => ev(s.t1 - 0.05, 'emit', { g: s.g, k: 0.45 }));
    sched.forEach((s) => { for (let L = 0; L < 28; L += 7) ev(s.t0 + (L / 28) * s.d, 'layer', { L, n: 28, k: 0.25 }); });
    layers.forEach((s) => { if (s.L % 2 === 0 || (s.L >= F.ga && s.L <= F.gb)) ev(s.t0, 'layer', { L: s.L, n: 28, k: s.L >= F.ga && s.L <= F.gb ? 0.7 : 0.4 }); });
    ev(A0, 'climb', { d: A1 - A0 });
    ev(layers[F.ga].t0, 'reveal', { k: 1.0 });
    ev(A1, 'emit', { g: gA, k: 0.9 });
    const lt3 = F.lens;
    const firstSan = F.thinkL;
    const lastL = mt.layers - 1;
    sub(T0 + 0.3, T0 + 4.0, `语言模型 ${m(mt.layers)} 层，开始回答`);
    term(T0 + 0.4, T0 + 4.0, `Qwen3 语言模型 · ${m(mt.layers)} 层 · ${m(mt.hidden)} 维`);
    sub(T0 + 4.2, preEnd - 0.1, '一个字一个字地往外说');
    term(T0 + 4.3, preEnd - 0.1, `每个字都要穿过 ${m(mt.layers)} 层 · 层板上亮的是在看图的哪里`);
    sub(preEnd + 0.1, A0 - 0.1, `说到“第”，下一个字是几？`);
    term(preEnd + 0.2, A0 - 0.1, `候选：「${esc(F.stA.top[0][2])}」${m(pct(F.stA.top[0][1]))} ·「${esc(F.stA.top[1][2])}」${m(pct(F.stA.top[1][1]))}`);
    sub(A0 + 0.05, layers[F.ga].t0 - 0.1, `前 ${F.ga} 层：东张西望`);
    term(A0 + 0.15, layers[F.ga].t0 - 0.1, `生成「3」时的注意力 · 16 个头平均`);
    sub(layers[F.ga].t0 + 0.05, layers[F.gb].t0, `第 ${F.ga} 层起：盯住了绿苹果`);
    term(layers[F.ga].t0 + 0.15, layers[F.gb].t0, `对准倍数 ${m(F.enrLo.toFixed(1) + '×')} → ${m(F.enrHi.toFixed(1) + '×')}`);
    sub(layers[F.gb].t0 + 0.1, A1 + 0.3, `看绿苹果的比例，是瞎看的 ${m(F.enrHi.toFixed(1))} 倍`);
    term(layers[F.gb].t0 + 0.2, A1 + 0.3, `第 ${F.ga}–${F.gb} 层平均`);
    sub(lens0, lens0 + 3.4, `它心里先想的是“${esc(lt3[firstSan >= 0 ? firstSan : 21][0])}”`);
    term(lens0 + 0.1, lens0 + 3.4, `逻辑透镜 · 第 ${firstSan} 层「${esc(lt3[firstSan][0])}」${m(pct(lt3[firstSan][1]))}`);
    sub(lens0 + 3.5, T1 - 0.15, `最后写下：“${esc(F.stA.chosenS)}”`);
    term(lens0 + 3.6, T1 - 0.15, `输出 ·「${esc(F.stA.chosenS)}」${m(pct(F.stA.p))}`);
    const towerCam = (Sc, L, o = {}) => {
      const y = G.LY0 + L * G.LG;
      const x1 = Sc.bx - 0.3, x2 = Sc.tx[Q.row(gA)] + 0.6;
      const look = v3((x1 + x2) / 2 + (o.lx ?? 0), y, 0);
      return orbit({ pos: look.clone().add(v3(o.dx ?? 0.6, o.dy ?? 5.4, o.dz ?? 7.6)), look, fov: 32 }, o.yaw ?? 0, 0, o.dist ?? 1);
    };
    shot('answer', (lt, t) => {
      let st, L = -1;
      if (t < sched[0].t0) st = mst(2, 0, { ph: 'llm' }, seg(t, T0 + 0.6, sched[0].t0) * 0.5, { dAnim: 3 });
      else if (t < preEnd) { const s = pick(sched, t); st = mst(2, s.g, { ph: 'llm' }, Math.min(0.999, s.p * 1.08), { dAnim: 3 }); }
      else if (t < A0) st = mst(3, gA, { ph: 'read' }, 1);
      else if (t < A1) { const s = pick(layers, t); L = s.L; st = mst(3, gA, { ph: 'layer', L }, s.p); }
      else { L = lastL; st = mst(3, gA, { ph: 'layer', L: lastL }, 1); }
      const nRep = t < sched[0].t0 ? 0 : t < preEnd ? pick(sched, t).g + (t >= pick(sched, t).t1 - 0.05 ? 1 : 0) : t < A1 ? gA : gA + 1;
      return {
        st,
        cam: (ctx) => {
          const Sc = ctx.S;
          const wide = towerCam(Sc, 13, { dx: 1.6, dy: 3.4, dz: 15.5, lx: 1.2 });
          const c0 = blendCam(prevCam('splice', wide), wide, smoother(seg(lt, 0, 3)));
          if (t < A0 - 1.2) return handheld(c0, t, 0.002);
          const Lc = L < 0 ? 0 : L + (t < A1 ? pick(layers, t).p : 0);
          const live = towerCam(Sc, Math.min(lastL, Lc), { yaw: lerp(-6, 6, seg(t, A0, A1)) });
          return blendCam(c0, live, smoother(seg(t, A0 - 1.2, A0 + 0.6)));
        },
        reply: { a: smooth(seg(t, sched[0].t0 - 0.4, sched[0].t0 + 0.2)), n: nRep, hl: t >= A1 ? gA : -1 },
        panel: L >= 0 ? { kind: 'heat', a: smooth(seg(t, A0, A0 + 0.5)), g: gA, L, enr: F.enr, upto: L + (t < A1 ? pick(layers, t).p : 1), avg: t >= A1 + 0.2 } : null,
        lens3: { a: smooth(seg(t, lens0, lens0 + 0.5)) * (1 - smooth(seg(t, T1 - 0.5, T1))), k: seg(t, lens0 + 0.3, lens0 + 3.4), fin: seg(t, lens0 + 3.5, lens0 + 4.0) },
      };
    });
  }

  /* ------------------------------------------------------------ ⑤′ 图片词元的逻辑透镜：读起来像词 */
  {
    const T0 = SEC.lens.t0, T1 = SEC.lens.t1;
    const LA = 22, LB = 24;
    ev(T0 + 2.6, 'reveal', { k: 0.8 });
    ev(T0 + 7.4, 'reveal', { k: 0.6 });
    sub(T0 + 0.3, T0 + 2.5, '那图片位置上，又“读”出什么？');
    term(T0 + 0.4, T0 + 2.5, '把逻辑透镜用在图片的词元上');
    sub(T0 + 2.6, T0 + 7.3, `第 ${LA} 层：五个、红色……`);
    term(T0 + 2.7, T0 + 7.3, '每格写的是这个位置“想说”的词');
    sub(T0 + 7.4, T1 - 0.15, `第 ${LB} 层：苹果`);
    term(T0 + 7.5, T1 - 0.15, '图块，变成了能说出口的“词”');
    shot('lens', (lt, t) => ({
      st: mst(3, gA, { ph: 'layer', L: t < T0 + 7.4 ? LA : LB }, 0.5),
      cam: (ctx) => {
        const Sc = ctx.S;
        const y = G.LY0 + LA * G.LG;
        const c = cam([Sc.bcx + 0.2, y + 5.0, 4.6], [Sc.bcx + 0.2, y, 0.1]);
        return blendCam(prevCam('answer', c), c, smoother(seg(lt, 0, 2.2)));
      },
      reply: { a: 1 - smooth(seg(lt, 0, 0.6)), n: gA + 1, hl: gA },
      panel: { kind: 'lens', center: true, a: smooth(seg(t, T0 + 2.4, T0 + 3.0)) * (1 - smooth(seg(t, T1 - 0.6, T1))), L: t < T0 + 7.4 ? LA : LB, k: seg(t, t < T0 + 7.4 ? T0 + 2.6 : T0 + 7.4, (t < T0 + 7.4 ? T0 + 2.6 : T0 + 7.4) + 1.6) },
      shiftX: 0,
      fade: 0.55 * smooth(seg(t, T0 + 2.0, T0 + 3.0)) * (1 - smooth(seg(t, T1 - 0.6, T1))),
      hideHeat: true,
    }));
  }

  /* ------------------------------------------------------------ ⑤″ 写完：每个字都知道它在看哪 */
  {
    const T0 = SEC.reply.t0, T1 = SEC.reply.t1;
    // 挑回答里“说的是图上东西”的几个字：绿色、苹果、第、3（按真实分词找）
    const keyG = Q.steps.map((st, g) => ({ g, s: st.chosenS })).filter((x) => /绿色|苹果|第|^\s*3\s*$/.test(x.s)).map((x) => x.g);
    const each = (T1 - T0 - 1.0) / keyG.length;
    const hov = lay(T0 + 0.6, keyG.map((g) => ({ g, d: each })));
    hov.forEach((h) => ev(h.t0, 'tick', { k: 0.3 }));
    ev(T1 - 0.4, 'end', { k: 0.6 });
    sub(T0 + 0.3, T1 - 0.15, '回答的每个字，都能看到它在看哪');
    term(T0 + 0.4, T1 - 0.15, `第 ${F.ga}–${F.gb} 层平均 · 16 个头平均`);
    shot('reply', (lt, t) => {
      const h = pick(hov, t);
      const g = Math.min(Q.G - 1, h.g);
      return {
        st: mst(3, g, { ph: 'layer', L: 20 }, 0.5),
        cam: (ctx) => {
          const Sc = ctx.S;
          const c = cam([Sc.bcx + 3.0, 6.2, 12.5], [Sc.bcx + 2.4, 2.4, 0]);
          return blendCam(prevCam('lens', c), c, smoother(seg(lt, 0, 2.4)));
        },
        fade: 0.55 * smooth(seg(lt, 0, 1.5)),
        reply: { a: 1, n: Q.G, hl: g, big: true },
        panel: { kind: 'hover', center: true, a: smooth(seg(lt, 0.2, 0.8)) * (1 - smooth(seg(t, T1 - 0.4, T1 + 0.2))), g },
        shiftX: 0,
        hideHeat: true,
      };
    });
  }

  /* ------------------------------------------------------------ 片尾：多模态页的真实录屏 + 网址 + 二维码 */
  let endSite = null;
  {
    const T0 = SEC.end.t0, T1 = SEC.end.t1;
    const D0 = 0.6;
    let FIN = 16.5;
    const A = cap?.actions || [];
    const at = (name, k = 0) => A.filter((a) => a.name === name)[k]?.t ?? 0;
    const segs = [];
    let f = D0;
    const add = (d, c0, c1, note) => { segs.push({ f0: f, f1: f + d, c0, c1, note }); f += d; };
    if (A.length) {
      add(2.8, at('pick') - 0.6, at('send') + 0.2, 'ask');            // 选图、点选词元、发送
      add(1.2, at('send') + 0.2, at('replyDone') + 0.2, 'reply');      // 回答写出来
      add(3.0, at('hover') - 0.3, at('hoverEnd') + 0.1, 'hover');      // 鼠标放在「3」上：图上亮起热力图（镜头推近那张图）
      add(1.6, at('plus') - 0.4, at('plus') + 2.6, 'reveal');          // 点 ＋ 揭开
      const dives = A.filter((a) => a.name === 'in');
      dives.forEach((a) => add(1.3, a.t - 0.25, a.t + 1.6, 'dive'));
      add(Math.max(2.8, Math.min(3.6, at('end') - at('mon') + 0.4)), at('mon') - 0.4, at('end'), 'mon');   // 最后：监视器上的热力图（推近）
      FIN = f;
    }
    const capAt = (lt) => { if (!segs.length) return 0; const s = segs.find((x) => lt < x.f1) || segs[segs.length - 1]; return lerp(s.c0, s.c1, clamp((lt - s.f0) / (s.f1 - s.f0))); };
    const filmOf = (ct) => { const s = segs.find((x) => ct >= x.c0 && ct <= x.c1); return s ? s.f0 + ((ct - s.c0) / (s.c1 - s.c0)) * (s.f1 - s.f0) : null; };
    for (const a of A) { if (a.x == null || a.click === false) continue; const lt = filmOf(a.t); if (lt != null) ev(T0 + lt, a.name === 'in' ? 'step' : 'tick', { k: a.name === 'in' ? 0.55 : 0.3 }); }
    ev(T0 + FIN, 'end', { k: 0.4 });
    const sg = (note) => segs.find((s) => s.note === note);
    const boxOf = (name) => A.find((a) => a.name === name)?.box || null;
    // 推近：在某一段里把窗口放大到 z 倍，让录屏里的那块区域移到画面中间
    const zoomAt = (lt) => {
      const zs = [['hover', 'heatbox', 2.0], ['mon', 'monbox', 1.75]];
      for (const [note, bx, z] of zs) {
        const s2 = sg(note), b = boxOf(bx);
        if (!s2 || !b) continue;
        const k = smoother(seg(lt, s2.f0 + 0.2, s2.f0 + 1.0)) * (1 - smoother(seg(lt, s2.f1 - 0.5, s2.f1 + (note === 'mon' ? 0.6 : 0.1))));
        if (k > 0) return { x: b.x + b.w / 2, y: b.y + b.h / 2, z, k };
      }
      return null;
    };
    if (segs.length) {
      sub(T0 + D0 + 0.2, T0 + sg('hover').f0 - 0.1, '想自己一步步打开看看？');
      term(T0 + D0 + 0.3, T0 + sg('hover').f0 - 0.1, '选一张图，点选词元拼出问题');
      sub(T0 + sg('hover').f0 + 0.1, T0 + sg('hover').f1 - 0.1, '鼠标放在字上，看它在看哪');
      term(T0 + sg('hover').f0 + 0.2, T0 + sg('hover').f1 - 0.1, '生成「3」时真实的注意力');
      const dv = segs.filter((s) => s.note === 'dive');
      const r = sg('reveal');
      sub(T0 + r.f0 + 0.1, T0 + (dv.length ? dv[dv.length - 1].f1 : r.f1) - 0.1, '点 ＋，一层层钻进去');
      term(T0 + r.f0 + 0.2, T0 + (dv.length ? dv[dv.length - 1].f1 : r.f1) - 0.1, '黑箱 → 流水线 → 逐层 → 一层之内');
      sub(T0 + sg('mon').f0 + 0.1, T0 + FIN - 0.15, '每一层在看哪，都能停下来看');
      term(T0 + sg('mon').f0 + 0.2, T0 + FIN - 0.15, '左上角的监视器：第 20 层的热力图');
    }
    shot('end', (lt, t) => ({
      st: mst(3, Q.G - 1, { ph: 'layer', L: 20 }, 0.5, { dAnim: lerp(3, 1, smoother(seg(lt, 0, 3))) }),
      cam: (ctx) => {
        const Sc = ctx.S;
        const c = cam([Sc.bcx + 3.0, 6.2, 12.5], [Sc.bcx + 2.4, 2.4, 0]);
        const far = cam([2, 6, 60], [2, 3, 0], 30);
        return blendCam(c, far, smoother(seg(lt, 0, 8)));
      },
      fade: lerp(0.55, 0.92, smooth(seg(lt, 0, 1.2))),
      reply: { a: 1 - smooth(seg(lt, 0, 0.6)), n: Q.G, hl: -1, big: true },
      ov: {
        band: 1 - smooth(seg(lt, 0, 1)),
        site: segs.length ? {
          a: smooth(seg(lt, D0 - 0.3, D0 + 0.4)) * (1 - smooth(seg(t, T1 - 0.9, T1 - 0.05))),
          ct: capAt(Math.min(lt, FIN)),
          zoom: zoomAt(lt),
          fin: easeInOut(seg(lt, FIN, FIN + 1.0)),
          finA: smooth(seg(lt, FIN + 0.5, FIN + 1.3)) * (1 - smooth(seg(t, T1 - 0.9, T1 - 0.05))),
          cursor: lt < FIN ? smooth(seg(lt, D0 + 0.2, D0 + 0.6)) * (1 - smooth(seg(lt, FIN - 0.4, FIN))) : 0,
        } : null,
      },
    }));
    endSite = { segs, D0, FIN };
  }

  function frame(t, ctx) {
    t = clamp(t, 0, end - 1e-6);
    const i = shots.findIndex((s) => t >= s.t0 && t < s.t1);
    const s = shots[i];
    const r = s.fn(t - s.t0, t, ctx);
    const camFn = typeof r.cam === 'function' ? r.cam : () => r.cam;
    r.cam = () => { const c = camFn(ctx); endCam.set(s.name, c); return c; };
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
  return { end, frame, subs, terms, chapterNames: CHAPTERS, progAt, endSite, events, sections, open: OPEN, F, SEC, bpm: BPM, hotPatch, shots: shots.map((s) => ({ name: s.name, t0: s.t0, t1: s.t1 })) };
}
