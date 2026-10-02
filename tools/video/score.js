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
  ['chat', 5, 0.1], ['fly', 3, 0.55], ['land', 2, 0.4], ['embed', 2, 0.5], ['layers1', 7, 0.78],
  ['attn', 29, 0.4], ['ffn', 13, 0.45], ['pick', 8, 0.6], ['loop1', 5, 0.72], ['loop2', 4, 0.9], ['end', 10, 0.22],
];
// 顶部章节进度条的六章
const CHAPTERS_ZH = ['切成词元', '查表', '穿过 28 层', '注意力', '前馈', '选字'];

const tkZh = (s) => esc(tokPlain(s));
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

// 英文版的戏剧点：从英文数据里找（看过数据以后可以用 opts.drama 手动覆盖）
//   flip：逻辑透镜从这一层起一直是最终说出口的第一个词；alt：改口之前最自信的另一个想法
//   LX：拆开哪一层（导出的点积里，最集中地看问题里某个词、又离 flip 不远的那层）
//   heads：并排展示的 6 个头；idWord：托盘里拿来举例的编号；rand：第一次没选第一名的那一步
export function findDrama(Q, ov = {}) {
  const NL = Q.NL, row = Q.row(0);
  const chosen = Q.steps[0].chosenS;
  const top = (L) => Q.lensAt(0, L).top[0];
  let flip = NL - 1;
  for (let L = NL - 1; L >= 0; L--) { if (top(L)[2] === chosen) flip = L; else break; }
  let alt = Math.max(1, flip - 3), best = -1;
  for (let L = Math.max(1, flip - 8); L < flip; L++) { const t = top(L); if (t[2] !== chosen && t[1] > best && (t[2] === '<|endoftext|>' || !/^<\|/.test(t[2]))) { best = t[1]; alt = L; } }
  const users = Q.tokens.map((t, i) => ({ ...t, i })).filter((t) => t.role === 'user' && !t.sp && t.s !== 'user' && t.s !== '\n');
  const userIdx = new Set(users.map((u) => u.i));
  let LX = flip, bw = -1;
  for (let L = 0; L < NL; L++) {
    const d = Q.dotAt(L, 0);
    if (!d) continue;
    const sc = d.w * (userIdx.has(d.key) ? 1 : 0.2) * (Math.abs(L - flip) <= 4 ? 1 : 0.6);
    if (sc > bw) { bw = sc; LX = L; }
  }
  const dot = Q.dotAt(LX, 0);
  const tops = Array.from({ length: Q.H }, (_, h) => ({ h, r: Q.att(LX, h, row)[0] })).filter((x) => x.r);
  const byW = (a, b) => b.r.w - a.r.w;
  const heads = [dot.head];
  for (const x of tops.filter((x) => userIdx.has(x.r.j)).sort(byW)) if (heads.length < 3 && !heads.includes(x.h)) heads.push(x.h);
  for (const x of tops.filter((x) => x.r.j === 0).sort(byW)) if (heads.length < 5 && !heads.includes(x.h)) heads.push(x.h);
  for (const x of tops.filter((x) => x.r.j === row || x.r.j >= Q.P - 4).sort(byW)) if (heads.length < 6 && !heads.includes(x.h)) heads.push(x.h);
  for (const x of tops.sort(byW)) if (heads.length < 6 && !heads.includes(x.h)) heads.push(x.h);
  let idWord = users.findIndex((u) => u.i === dot.key);
  if (idWord < 0) idWord = users.reduce((bi, u, k) => (u.s.trim().length > users[bi].s.trim().length ? k : bi), 0);
  let rand = 1;
  for (let g = 1; g < Math.min(Q.G - 2, 12); g++) if (Q.steps[g].chosenRank > 0) { rand = g; break; }
  return { LX, alt, flip, idWord, heads, rand, ...ov };
}

export function buildScore(Q, cap = null, opts = {}) {
  let endSite = null;
  // 语言：中文版用 q01，英文版用 e01（两份都是真实运行）；英文字幕不是直译，按英文数据自己的戏剧点写
  const EN = opts.lang === 'en';
  const T_ = (zh, en) => (EN ? en : zh);
  const CHAPTERS = EN ? ['Tokens', 'Lookup', '28 Layers', 'Attention', 'Feed-forward', 'Next word'] : CHAPTERS_ZH;
  const tk = (s) => (EN ? esc(String(s).replace(/^\s+/, '').replace(/\n/g, '↵')) : tkZh(s));
  const q = (s) => `<q>${s === '<|endoftext|>' ? T_('结束符', 'end-of-text') : s === '<|im_start|>' ? T_('开头', 'start') : tk(s)}</q>`;
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
  // 戏剧点：中文版是看过数据以后手选的（第 21 层「因为」→ 第 24 层「天空」、拆开第 24 层）；英文版由 findDrama 从英文数据里找
  const DR = EN ? findDrama(Q, opts.drama) : { LX: 24, alt: 21, flip: 24, idWord: 3, heads: null, rand: 1 };
  const G0 = 0, LX = DR.LX;
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
  // 英文：逐字母敲（没有输入法候选），空格稍微停一下；每个词敲完那一下当作“上屏”（配乐的拨弦）
  let tEn = B(0, 2);
  const typingEn = user.map((u, k) => {
    const keys = [...u.s].map((ch, j) => { const t = tEn; tEn += 0.098 + jit(k * 10 + j) * 1.6 + (ch === ' ' ? 0.07 : 0); return t; });
    return { s: u.s, py: null, keys, commit: keys[keys.length - 1], chars: true };
  });
  const typing = EN ? typingEn : user.map((u, k) => {
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
    { t: SEC.land.t0 + 1.8, p: [-8.2, 3.6, 11.5], l: [-7.5, 0.4, 0], fov: 32 },
    { t: SEC.land.t0 + 3.6, p: [-6.6, 2.3, 7.0], l: [-5.6, 0.3, 0], fov: 32 },
    { t: SEC.land.t1, p: [-5.9, 1.8, 5.2], l: [-5.1, 0.22, 0], fov: 32 },
  ]);
  OPEN.camAt = camOpen;
  let kn = 0;
  for (const w of typing) if (w.py || w.chars) for (const k of w.keys) ev(k, 'key', { i: kn++ });
  typing.forEach((w, i) => ev(w.commit, 'type', { i }));
  ev(OPEN.sendT, 'send');
  ev(OPEN.swapT, 'whoosh', { k: 0.8 });
  ev(OPEN.swapT, 'rise', { d: SEC.fly.t0 - OPEN.swapT });   // 词元飞起来，一路推到片名
  ev(SEC.fly.t0, 'hit', { k: 1 });
  ev(OPEN.boxOpen0 - 2.0, 'rise', { d: 2.0 });
  ev(OPEN.boxOpen0, 'open');
  for (let i = 0; i < Q.P; i++) ev(OPEN.landAt[i], 'tick', { i, k: i >= uFirst && i <= uLast ? 1 : 0.4 });
  // 片名在画面上的那几秒不放字幕
  sub(OPEN.split0 + 0.45, titleIn - 0.35, T_('发出去的问题，先被切成词元。', 'First, your question is chopped into tokens.'));
  sub(OPEN.boxOpen0 + 0.1, SEC.land.t0 - 0.2, T_('黑箱打开，词元落进托盘。', 'The box opens, and the tokens drop into the tray.'));
  sub(SEC.land.t0 + 0.3, SEC.land.t0 + 2.4, T_('聊天模板：给问题包上提示和标记', 'A chat template wraps it in instructions and markers'));
  term(SEC.land.t0 + 0.4, SEC.land.t0 + 2.4, T_('聊天模板 chat template', 'chat template'));
  const idw = user[DR.idWord] || user[0];
  sub(SEC.land.t0 + 2.6, SEC.land.t1 - 0.2, T_(`模型只认编号：${q(idw.s)} = ${m(idw.id)}`, `The model only sees numbers: ${q(idw.s)} = ${m(idw.id)}`));
  term(SEC.land.t0 + 2.7, SEC.land.t1 - 0.2, T_('词元编号 token id', 'token ID'));
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
      extras: { ids: smooth(seg(t, SEC.land.t0 + 2.4, SEC.land.t0 + 3.0)), idsFocus: true },
      ov: { band: smooth(seg(t, OPEN.diss0, OPEN.diss1)), title: smooth(seg(t, titleIn, titleIn + 0.8)) * (1 - smooth(seg(t, titleOut0, titleOut1))), titleK: seg(ft, 0.05, 2.6), titleBlur: 7 * smooth(seg(t, titleOut0, titleOut1)), titleOnBox: true },
    };
  });

  /* ------------------------------------------------------------ 02 嵌入 */
  {
    const T0 = SEC.embed.t0, T1 = SEC.embed.t1;
    prog(T0, 1);
    const k0 = T0 + 0.4, k1 = T0 + 2.8; // 从表里取向量（光点落下）
    for (let i = 0; i < Q.P; i += 2) ev(k0 + (k1 - k0) * ((i / Q.P) * 0.5 + 0.5), 'blip', { i });
    sub(T0 + 0.3, T1 - 0.2, T_(`拿编号去表里查：取出 ${m('1024')} 个数`, `Each ID looks up its row: 1,024 numbers`));
    term(T0 + 0.4, T1 - 0.2, T_(`嵌入表 ${m('151,936')} 行 × ${m('1,024')} 列`, `embedding table · 151,936 rows × 1,024`));
    let embPath = null;
    shot('embed', (lt, t, { M }) => {
      const last = camOpen(SEC.land.t1);
      const uc = (X(M, user[0].i) + X(M, user[user.length - 1].i)) / 2;
      embPath ||= path([
        { t: 0, p: last.pos.toArray(), l: last.look.toArray(), fov: 32 },
        { t: 1.5, p: [uc + 0.6, 3.9, 6.6], l: [uc - 0.2, 2.6, -1.0], fov: 32 },   // 先贴近墙上问题那几行
        { t: 2.4, p: [uc + 0.4, 3.7, 7.6], l: [uc - 0.3, 2.3, -0.8], fov: 32 },
        { t: 5.0, p: [-7.6, 4.9, 19.4], l: [-8.1, 4.5, -0.6], fov: 32 },          // 再拉开：光柱从托盘长起来
      ]);
      return {
        st: mst(2, 0, { ph: 'embed' }, seg(t, T0 + 2.2, T1 - 0.4), { dAnim: 2.6 }),
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
    // 每层停多久：开头那层 1 拍；离「改口」越近越慢；想法出现的那层 2.5 拍、改口那层 4 拍；总长固定（20.25 拍）
    const AL = DR.alt, FL = DR.flip;
    const raw = Array.from({ length: NL }, (_, L) => (L === 0 ? 1 : L === AL ? 2.5 : L === FL ? 4 : L < AL - 4 ? 0.375 : 0.75));
    const fixed = raw.map((d, L) => L === 0 || L === AL || L === FL);
    const flex = raw.reduce((a, d, L) => a + (fixed[L] ? 0 : d), 0), fix = raw.reduce((a, d, L) => a + (fixed[L] ? d : 0), 0);
    const kf = (20.25 - fix) / flex;
    const sched1 = lay(T0 + 2 * BEAT, raw.map((d, L) => ({ L, d: (fixed[L] ? d : d * kf) * BEAT })));
    const l1 = (L) => sched1.find((s) => s.L === L);
    sched1.forEach((s) => { if (s.L % 2 === 0 || s.L > 16) ev(s.t0, 'layer', { L: s.L, k: s.d > BEAT * 1.5 ? 1 : 0.6 }); });
    ev(l1(AL).t0, 'reveal', { k: 0.7 });
    ev(l1(FL).t0, 'reveal', { k: 1.1 });
    const hold1 = l1(NL - 1).t1; // 28 层走完
    const dive0 = T1 - 3.0;  // 往要拆开的那层扎下去
    const lA = lensTop(0, AL), lF = lensTop(0, FL), lZ = lensTop(0, NL - 1);
    const afterF = l1(Math.min(NL - 1, FL + 1));
    sub(T0 + 0.2, T0 + 2.0, T_('接着，穿过 28 层。', 'Next, it passes through 28 layers.'));
    term(T0 + 0.3, T0 + 2.0, T_('Transformer 层 × 28', 'Transformer layers × 28'));
    sub(T0 + 2.2, l1(AL).t0 - 0.1, T_('每过一层，偷看一眼：它此刻最想说什么？', 'After each layer, we peek: what would it say right now?'));
    term(T0 + 2.3, l1(AL).t0 - 0.1, T_('逻辑透镜 logit lens', 'logit lens'));
    sub(l1(AL).t0 + 0.05, l1(FL).t0 - 0.1, T_(`第 ${AL} 层：想说${q(lA[2])} ${m(pct(lA[1]))}`, DR.altText ?? (lA[2] === '<|endoftext|>' ? `Layer ${AL}: its top pick is to stop and say nothing (${m(pct(lA[1]))})` : `Layer ${AL}: it leans toward ${q(lA[2])} (${m(pct(lA[1]))})`)));
    sub(l1(FL).t0 + 0.05, (FL < NL - 1 ? afterF.t0 : dive0) - 0.05, T_(`第 ${FL} 层，它改口了：${q(lF[2])} ${m(pct(lF[1]))}`, DR.flipText ?? `Layer ${FL}: ${q(lF[2])} takes the lead (${m(pct(lF[1]))})`));
    if (FL < NL - 1) sub(afterF.t0 + 0.05, dive0 - 0.15, T_(`最后一层：${q(lZ[2])} ${m(pct(lZ[1]))}`, `Last layer: ${q(lZ[2])} at ${m(pct(lZ[1]))}`));
    sub(dive0, T1 - 0.15, T_(`第 ${LX} 层发生了什么？拆开看看。`, `What happens inside layer ${LX}? Let's open it up.`));
    const camLayers1 = (M, t) => {
      const Ls = [[T0, 0.5], [l1(0).t0, 0.5], [l1(Math.max(1, AL - 4)).t0, Math.max(1, AL - 5.5)], [l1(AL).t0, AL - 1], [l1(FL).t0, FL - 0.8]];
      if (FL < NL - 1) Ls.push([afterF.t0, FL + 0.3]);
      Ls.push([hold1, 26.5]);
      const Lc = pchip(Ls.map((a) => a[0]), Ls.map((a) => a[1]))(t);
      const push = 0.2 * bumpAt(t, l1(AL).t0, l1(AL).d) + 0.3 * bumpAt(t, l1(FL).t0, l1(FL).d);
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
          if (lt < 2.0) { e1In ||= prevCam('embed', live); return blendCam(e1In, live, smoother(lt / 2.0)); } // 从嵌入的远景接进来
          if (t > hold1) { // 拉开看整座塔，再往第 24 层扎下去
            const wide = towerCam(M, 14, 0, { lx: -5, dx: 13, dy: 4.6, dz: 22 });
            const dy = (LX - 24) * 0.26;
            const dive = cam([3.4, 8.9 + dy, 7.6], [0.5, 7.7 + dy, 0.2]);
            return blendCam(blendCam(live, wide, smoother(seg(t, hold1, dive0 + 0.2))), dive, smoother(seg(t, dive0 + 0.2, T1)));
          }
          return handheld(live, t, 0.003);
        },
        lensWin: t > hold1 ? 28 : 9,
        ov: { lens: { a: smooth(seg(t, T0 + 2.2, T0 + 2.9)) * (1 - smooth(seg(t, dive0, dive0 + 0.8))), g: 0, upto: before ? -0.01 : s.L + (s.L === NL - 1 ? 1 : s.p) - 0.0001, other: lensTop(0, AL)[2], side: 'left' } },
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
      { key: 'over', depth: 4, s: { op: 'ln1' }, d: 6, p0: true },
      { key: 'ln1', depth: 4, s: { op: 'ln1' }, d: 6 },
      // ④ 注意力（全片唯一的一次逐数计算在这里：一次点积）
      { key: 'intro', depth: 5, s: { op: 'attn', sub: 'qkv' }, d: 6, p0: true },
      { key: 'lib', depth: 5, s: { op: 'attn', sub: 'qkv' }, d: 4, p0: true },
      { key: 'q', depth: 5, s: { op: 'attn', sub: 'qkv' }, d: 7 },
      { key: 'k', depth: 5, s: { op: 'attn', sub: 'qkv' }, d: 7 },
      { key: 'v', depth: 5, s: { op: 'attn', sub: 'qkv' }, d: 7 },
      { key: 'score', depth: 5, s: { op: 'attn', sub: 'score' }, d: 10 },
      { key: 'dot', depth: 6, s: { op: 'attn', sub: 'score', mi: 'mul' }, d: 6 },
      { key: 'dsum', depth: 6, s: { op: 'attn', sub: 'score', mi: 'sum' }, d: 5 },
      { key: 'scale', depth: 6, s: { op: 'attn', sub: 'score', mi: 'scale' }, d: 6 },
      { key: 'softmax', depth: 5, s: { op: 'attn', sub: 'softmax' }, d: 10 },
      { key: 'mix', depth: 5, s: { op: 'attn', sub: 'softmax' }, d: 10 },
      { key: 'recap', depth: 5, s: { op: 'attn', sub: 'softmax' }, d: 4 },
      { key: 'heads', depth: 5, s: { op: 'attn', sub: 'softmax' }, d: 12 },
      { key: 'add1', depth: 5, s: { op: 'add1' }, d: 10 },
      // ⑤ 前馈：结构看得见（1024 → 3072 → 1024），门 / 内容 / 相乘 / 收回，不做逐数推导
      { key: 'fintro', depth: 5, s: { op: 'mlp', sub: 'up' }, d: 8, p0: true },
      { key: 'shape', depth: 5, s: { op: 'mlp', sub: 'up' }, d: 6 },
      { key: 'gate', depth: 5, s: { op: 'mlp', sub: 'up' }, d: 6 },
      { key: 'up', depth: 5, s: { op: 'mlp', sub: 'up' }, d: 6 },
      { key: 'mul', depth: 5, s: { op: 'mlp', sub: 'act' }, d: 6 },
      { key: 'silu', depth: 5, s: { op: 'mlp', sub: 'act' }, d: 6 },
      { key: 'act', depth: 5, s: { op: 'mlp', sub: 'act' }, d: 6 },
      { key: 'down', depth: 5, s: { op: 'mlp', sub: 'down' }, d: 8 },
    ].map((x) => ({ ...x, d: x.d * BEAT })));
    const Dk = (k) => D.find((x) => x.key === k);
    // 章节进度：RMSNorm 和注意力算 ④，前馈算 ⑤
    prog(T0, 2); prog(Dk('ln1').t0, 3); prog(Dk('fintro').t0, 4);
    D.forEach((x) => { if (!['over', 'lib', 'recap', 'dsum'].includes(x.key)) ev(x.t0, 'step', { k: 0.5 }); });
    ev(Dk('over').t0, 'hit', { k: 0.7 });
    ev(Dk('score').t0, 'reveal', { k: 0.5 });
    ev(Dk('softmax').t0, 'reveal', { k: 0.7 });
    ev(Dk('recap').t0, 'reveal', { k: 1.0 });
    ev(Dk('fintro').t0, 'hit', { k: 0.5 });
    ev(Dk('act').t0, 'reveal', { k: 0.6 });
    for (let k = 0; k < 12; k++) ev(Dk('dot').t0 + (k / 12) * Dk('dot').d * 0.94, 'tick', { k: 0.3 });

    // 真实数字
    const inNorm = Q.norm(LX - 1, row0);
    const att6 = Q.att(LX, dot.head, row0);
    const nm = (s) => s;
    const a1 = att6[0], a2 = att6[1], a3 = att6[2];
    const HEADS = (DR.heads || [dot.head, 10, 14, 1, 0, 9]).filter((h, i, a) => a.indexOf(h) === i).slice(0, 6);
    const lensX = lensTop(0, LX);
    const n1 = (v) => v.toFixed(1);
    const nOn = Q.mlpCount(G0, LX);

    // 字幕（白话，一行）+ 术语标签（小号）
    const say = (k, html, tg, o = {}) => {
      const x = Dk(k);
      const a = x.t0 + (o.a ?? 0.2), b = (o.b != null ? x.t0 + o.b : x.t1 - 0.15);
      sub(a, b, html);
      if (tg) term(a + 0.1, b, tg);
    };
    say('over', T_(`拆开第 ${LX} 层：每层都是同样几步`, `Inside layer ${LX}: every layer runs the same steps`));
    say('ln1', T_('先把这串数，拉回统一的音量', 'First, the numbers are evened out to a standard volume'), T_(`RMSNorm 归一化 · 均方根 ${m((inNorm / 32).toFixed(2))} → ${m('1')}`, `RMSNorm · RMS ${m((inNorm / 32).toFixed(2))} → ${m('1')}`));
    say('intro', T_('注意力：回头看看前面说了什么', 'Attention: look back at what came before'), T_('自注意力 Self-Attention', 'self-attention'));
    say('lib', T_('就像在图书馆里查资料', "Think of it as looking something up in a library"));
    say('q', T_('Q 是提问：我想找什么？', 'Q is the question: what am I looking for?'), T_(`Query 查询 · q = h × W<sub>q</sub>`, `Query · q = h × W<sub>q</sub>`));
    say('k', T_('K 是每个词的标签：我是什么', "K is each word's label: what I'm about"), T_(`Key 键 · k = h × W<sub>k</sub>`, `Key · k = h × W<sub>k</sub>`));
    say('v', T_('V 是这个词真正能提供的内容', 'V is what the word actually has to offer'), T_(`Value 值 · v = h × W<sub>v</sub>`, `Value · v = h × W<sub>v</sub>`));
    say('score', T_('拿提问，去对每个词的标签', 'Check the question against every label'), T_(`点积 Q·K<sup>T</sup>`, `dot product Q·K<sup>T</sup>`), { b: 3.0 });
    say('score', T_('越对得上，匹配分越高', 'The better the match, the higher the score'), T_(`点积 Q·K<sup>T</sup>`, `dot product Q·K<sup>T</sup>`), { a: 3.2 });
    say('dot', T_(`放大一次：${m('128')} 对数，逐个相乘`, `Zoom in on one: ${m('128')} pairs of numbers, multiplied`), T_(`第 ${m(dot.head)} 个头 · 每个头 ${m('128')} 维`, `head ${m(dot.head)} · ${m('128')} dimensions per head`));
    say('dsum', T_(`全部加起来：${cy(n1(dot.sum))}`, `Add them all up: ${cy(n1(dot.sum))}`));
    say('scale', T_('再除以 √128，免得分数太大', "Divide by √128 so the scores don't blow up"), `d = ${m('128')} · √d ≈ ${m('11.3')}`);
    say('softmax', T_('softmax：分数变成权重，总和为 1', 'Softmax turns the scores into weights that add up to 1'), 'softmax', { b: 3.0 });
    say('softmax', T_(`${q(keyTok)}一个就拿走 ${m(pct(dot.w))}`, `${q(keyTok)} alone takes ${m(pct(dot.w))}`), 'softmax', { a: 3.2 });
    say('mix', T_('按权重，把各词的 V 混在一起', "Blend every word's V, by weight"), T_('加权求和 Σ w × V', 'weighted sum Σ w × V'), { b: 3.0 });
    say('mix', T_('得到它读完上下文的新理解', 'The result: a new understanding, informed by context'), T_('加权求和 Σ w × V', 'weighted sum Σ w × V'), { a: 3.2 });
    say('recap', T_('这就是注意力的全部公式', "That's the whole attention formula"));
    say('heads', T_(`同样的查找，同时有 ${m('16')} 个头`, `The same search runs in ${m('16')} heads at once`), T_(`多头注意力 · ${m('16')} 个头`, `multi-head attention · ${m('16')} heads`), { b: 3.4 });
    say('heads', T_(`各找各的：有的盯${q(keyTok)}，有的盯开头`, DR.headsText ?? `Different heads, different targets: ${[...new Set(HEADS.map((h) => Q.att(LX, h, row0)[0]?.j).filter((j) => j != null).map((j) => (j === 0 ? 'the start' : /^\s*\n/.test(Q.tokens[j].s) || Q.tokens[j].sp ? null : q(Q.tokens[j].s))).filter(Boolean))].slice(0, 3).join(', ')}`), T_(`多头注意力 · ${m('16')} 个头`, `multi-head attention · ${m('16')} heads`), { a: 3.6 });
    say('add1', T_('把结果加回去：只改一点，不推倒重来', 'Add the result back: a small edit, not a rewrite'), T_('残差连接', 'residual connection'), { b: 3.0 });
    say('add1', T_(`这一层之后，它最想说${q(lensX[2])}`, `After this layer, its top pick is ${q(lensX[2])}`), T_(`逻辑透镜 · ${q(lensX[2])} ${m(pct(lensX[1]))}`, `logit lens · ${q(lensX[2])} ${m(pct(lensX[1]))}`), { a: 3.2 });
    say('fintro', T_('注意力：把上下文的信息收集过来了', 'Attention gathered what the context had to say'), null, { b: 2.4 });
    say('fintro', T_('前馈：每个词自己消化、联想', 'Feed-forward: each word digests it on its own'), T_('前馈网络 FFN', 'feed-forward network'), { a: 2.6 });
    say('shape', T_(`先展开成 ${m('3072')} 个特征，筛一遍，再收回`, `Expand into 3,072 features, filter them, fold back`), `${m('1024')} → ${m('3072')} → ${m('1024')}`);
    say('gate', T_('门：这个特征该不该放行、放多少', 'The gate: should this feature pass, and how much?'), T_('gate（门）', 'gate'));
    say('up', T_('内容：这个特征说的是什么', 'The content: what this feature has to say'), T_('up（内容）', 'up (content)'));
    say('mul', T_('两者相乘：门开多大，就放多少内容', 'Multiply: the wider the gate, the more content gets through'), 'SwiGLU = SiLU(gate) × up');
    say('silu', T_('SiLU：小于 0 基本关上，大于 0 照常通过', 'SiLU: below zero it mostly shuts, above zero it lets through'), T_('SiLU 激活函数', 'SiLU activation'));
    say('act', T_(`这一次，${m('3072')} 个里只亮了 ${m(nOn)} 个`, `This time, only ${m(nOn)} of 3,072 light up`), T_('大多数特征这次没用上', 'most features stay quiet'));
    say('down', T_(`down：把筛过的特征写回 ${m('1024')} 个数`, `Down: fold the filtered features back into 1,024 numbers`), T_('down（收回）', 'down'), { b: 2.6 });
    say('down', T_('再加回原来的理解', '…and add them to what it already knew'), T_('残差连接', 'residual connection'), { a: 2.8 });

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
      // 前馈：三块矩阵放在画面中下部（结构图在上面），镜头慢慢推近
      ffn: (M, p) => { const { xf, y } = L(M); return blendCam(cam([xf + 1.9, y + 3.65, 8.6], [xf + 1.55, y + 3.05, -0.2]), cam([xf + 1.75, y + 3.55, 7.9], [xf + 1.5, y + 3.0, -0.2]), smooth(p)); },
      // 神经元阵列：3072 个开关，这次亮了几个
      neurons: (M, p) => { const { xf, y } = L(M); return blendCam(cam([xf + 0.6, y + 3.55, 4.7], [xf + 0.2, y + 3.15, -0.9]), cam([xf + 0.45, y + 3.5, 4.3], [xf + 0.2, y + 3.15, -0.9]), smooth(p)); },
      down: (M, p) => { const { xf, y } = L(M); return blendCam(cam([xf + 2.3, y + 3.9, 8.0], [xf + 2.0, y + 3.3, -0.2]), cam([xf + 2.5, y + 3.85, 7.5], [xf + 2.2, y + 3.3, -0.2]), smooth(p)); },
    };
    const camKey = { over: 'over', ln1: 'ln1', intro: 'qkv', lib: 'qkv', q: 'qkv', k: 'qkv', v: 'qkv', score: 'score', dot: 'dot', dsum: 'dot', scale: 'dot', softmax: 'soft', mix: 'soft', recap: 'soft', heads: 'heads', add1: 'add1', fintro: 'ffn', shape: 'ffn', gate: 'ffn', up: 'ffn', mul: 'ffn', silu: 'ffn', act: 'neurons', down: 'down' };
    const camSpan = {}; // 同一个机位 key 跨越的时间段
    D.forEach((x) => { const k = camKey[x.key]; if (!k) return; camSpan[k] ||= { t0: x.t0, t1: x.t1 }; camSpan[k].t1 = x.t1; });
    const BLEND = { over: 2.4, ln1: 1.2, qkv: 1.4, score: 1.3, soft: 1.1, heads: 1.4, dot: 1.3, add1: 1.1, ffn: 1.5, neurons: 1.1, down: 1.2 };

    // 注意力公式板：随着讲解一项一项搭出来（show = 已出现的项，hi = 正在讲的项）
    const FORM = {
      score: { show: ['qk'], hi: 'qk', cap: T_(`${q(keyTok)}的匹配分：Q·K = ${m(n1(dot.sum))}`, `match score for ${q(keyTok)}: Q·K = ${m(n1(dot.sum))}`) },
      dot: { show: ['qk'], hi: 'qk', cap: T_(`${q(keyTok)}：q·k = Σ q<sub>i</sub> × k<sub>i</sub>（${m('128')} 项）`, `${q(keyTok)}: q·k = Σ q<sub>i</sub> × k<sub>i</sub> (${m('128')} terms)`) },
      dsum: { show: ['qk'], hi: 'qk', cap: T_(`${q(keyTok)}：q·k = ${m(n1(dot.sum))}`, `${q(keyTok)}: q·k = ${m(n1(dot.sum))}`) },
      scale: { show: ['qk', 'sd'], hi: 'sd', cap: `${m(n1(dot.sum))} ÷ √${m('128')} = ${m(n1(dot.score))}` },
      softmax: { show: ['qk', 'sd', 'sm'], hi: 'sm', cap: `${q(nm(Q.tokens[a1.j].s))} ${m(pct(a1.w))} · ${q(nm(Q.tokens[a2.j].s))} ${m(pct(a2.w))} · ${q(nm(Q.tokens[a3.j].s))} ${m(pct(a3.w))} ${T_('…… 合计', '… total')} ${m('100%')}` },
      mix: { show: ['qk', 'sd', 'sm', 'v'], hi: 'v', cap: `${m(a1.w.toFixed(2))} × V${q(nm(Q.tokens[a1.j].s))} + ${m(a2.w.toFixed(2))} × V${q(nm(Q.tokens[a2.j].s))} + ${T_('……', '…')}` },
      recap: { show: ['lhs', 'qk', 'sd', 'sm', 'v'], hi: 'all', cap: T_(`第 ${m(LX)} 层 · 第 ${m(dot.head)} 个头 · d = ${m('128')}`, `layer ${m(LX)} · head ${m(dot.head)} · d = ${m('128')}`) },
    };
    const formKeys = Object.keys(FORM);
    const fFirst = Dk('score').t0, fLast = Dk('recap').t1;
    // 前馈结构图
    const FFN = {
      shape: { hi: 'shape', cap: T_(`${m('1024')} → ${m('3072')} → ${m('1024')}：展开，筛一遍，再收回`, `${m('1024')} → ${m('3072')} → ${m('1024')}: expand, filter, fold back`) },
      gate: { hi: 'gate', cap: T_('门开得越大，放行得越多', 'the wider it opens, the more gets through') },
      up: { hi: 'up', cap: T_('每个特征带来的内容', 'what each feature brings') },
      mul: { hi: 'mul', cap: T_('门开多大，就放多少内容', 'as much content as the gate allows') },
      silu: { hi: 'silu', cap: 'silu' },
      down: { hi: 'down', cap: T_(`写回 ${m('1024')} 个数，加回原来的理解`, `back to 1,024 numbers, added to the original`) },
    };
    const mFirst = Dk('shape').t0, mLast = Dk('down').t1;

    shotSpan('dissect', T0, T1, (lt, t, { M }) => {
      const s = pick(D, t);
      // Q / K / V 三步、前馈的展开三步：矩阵扫描连续推进（不在每一步重来）
      let p = s.p0 ? 0 : s.p;
      if (['q', 'k', 'v'].includes(s.key)) p = seg(t, Dk('q').t0, Dk('v').t1 - 0.6);
      if (['shape', 'gate', 'up'].includes(s.key)) p = seg(t, Dk('shape').t0, Dk('up').t1 - 0.4);
      if (['mul', 'silu', 'act'].includes(s.key)) p = seg(t, Dk('mul').t0, Dk('act').t1);
      let head = null;
      const hk = Dk('heads');
      const hOn = s.key === 'heads' ? HEADS[Math.min(HEADS.length - 1, Math.floor(clamp((t - hk.t0 - 3.0) / (hk.d - 3.4)) * HEADS.length))] : dot.head;
      if (['score', 'softmax', 'mix', 'recap', 'dot', 'dsum', 'scale'].includes(s.key)) head = dot.head;
      if (s.key === 'heads') head = t < hk.t0 + 3.0 ? dot.head : hOn;
      const st = mst(s.depth, G0, { ph: 'layer', L: LX, ...s.s }, p, { dAnim: s.depth, head });
      const ck = camKey[s.key], span = camSpan[ck];
      // 指示环
      const A = M.mats.attn, P = M.mats.mlp;
      const vecAt = (o, vertical = false) => () => (o && o.parent && o.parent.visible && o.visible ? o.localToWorld(vertical ? v3(0, o.len / 2, 0) : v3(o.len / 2, 0, 0)) : null);
      const beamEnd = (j, end) => () => { const b = (M.beamList || []).find((x) => x.j === j); return b ? b.m.localToWorld(b.geo.parameters.path[end].clone()) : null; };
      const ringA = (k, a = 0.25, b = 0.15) => { const x = Dk(k); return smooth(seg(t, x.t0 + a, x.t0 + a + 0.45)) * (1 - smooth(seg(t, x.t1 - b - 0.35, x.t1 - b))); };
      const late = (a = 3.2) => smooth(seg(t, s.t0 + a, s.t0 + a + 0.5)) * (1 - smooth(seg(t, s.t1 - 0.4, s.t1 - 0.1)));
      const rings = [];
      if (s.key === 'q') rings.push({ at: vecAt(A.qo), w: A.qo.len + 0.08, h: 0.06, pad: 40, label: 'Q', top: true, a: ringA('q', 0.5) });
      if (s.key === 'k') rings.push({ at: vecAt(A.ko), w: A.ko.len + 0.08, h: 0.06, pad: 40, label: 'K', top: true, a: ringA('k', 0.5), cls: 'amber' });
      if (s.key === 'v') rings.push({ at: vecAt(A.vo), w: A.vo.len + 0.08, h: 0.06, pad: 40, label: 'V', top: true, a: ringA('v', 0.5) });
      if (s.key === 'score') rings.push({ at: beamEnd(dot.key, 'v2'), w: 0.2, h: 0.2, label: q(keyTok), cls: 'amber', a: late() });
      if (s.key === 'softmax') rings.push({ at: beamEnd(dot.key, 'v2'), w: 0.2, h: 0.2, label: `${q(keyTok)} ${pct(dot.w)}`, cls: 'amber', a: late() });
      if (s.key === 'mix') rings.push({ at: beamEnd(dot.key, 'v0'), w: 0.2, h: 0.2, label: T_('新的理解', 'new understanding'), a: late() });
      if (s.key === 'add1') rings.push({ at: () => M.exAdd1.getWorldPosition(v3(0, 0, 0)), w: 0.26, h: 0.26, label: '⊕', a: ringA('add1', 0.4, 3.6) });
      if (s.key === 'gate') rings.push({ at: vecAt(P.go), w: P.go.len + 0.08, h: 0.06, pad: 40, label: T_('门', 'gate'), top: true, a: ringA('gate', 0.4) });
      if (s.key === 'up') rings.push({ at: () => (P && P.visible ? P.u.localToWorld(v3(P.u.w / 2, P.u.h / 2, 0)) : null), w: P.u.w + 0.06, h: P.u.h + 0.04, pad: 20, label: T_('内容', 'content'), top: true, a: ringA('up', 0.4), cls: 'violet' });
      if (s.key === 'mul') rings.push({ at: vecAt(P.din, true), w: 0.06, h: P.din.len, pad: 34, label: T_('门 × 内容', 'gate × content'), top: true, a: ringA('mul', 0.5) });
      if (s.key === 'down') rings.push({ at: vecAt(P.dout), w: P.dout.len + 0.08, h: 0.06, pad: 40, label: T_(`${m('1024')} 个数`, `1,024 numbers`), top: true, cls: 'amber', a: ringA('down', 0.5, 2.4) });
      // 公式板 / 前馈结构图
      let formula = null;
      if (t >= fFirst - 0.4 && t < fLast + 0.6 && s.key !== 'heads') {
        const fk = formKeys.includes(s.key) ? s.key : 'score';
        formula = { ...FORM[fk], a: smooth(seg(t, fFirst, fFirst + 0.6)) * (1 - smooth(seg(t, fLast - 0.1, fLast + 0.5))) };
      }
      let ffnmap = null;
      if (t >= mFirst - 0.4 && t < mLast + 0.4 && s.key !== 'act') {
        const fk = FFN[s.key] ? s.key : 'shape';
        const actS = Dk('act');
        ffnmap = { ...FFN[fk], a: smooth(seg(t, mFirst, mFirst + 0.6)) * (1 - smooth(seg(t, mLast - 0.5, mLast - 0.05))) * (1 - smooth(seg(t, actS.t0 - 0.3, actS.t0 + 0.2)) * (1 - smooth(seg(t, actS.t1 - 0.2, actS.t1 + 0.3)))) };
      }
      const MF = { gate: 'g', up: 'u', mul: 'din', silu: 'din', down: 'd' };
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
          endCam.set('d-last', out);
          return out;
        },
        hide: [...(['score', 'soft', 'heads', 'dot'].includes(ck) ? ['attnMats'] : [])],
        attnFocus: s.key === 'q' ? 'q' : s.key === 'k' ? 'k' : s.key === 'v' ? 'v' : null,
        attnFocusK: ['q', 'k', 'v'].includes(s.key) ? smooth(seg(t, Dk('q').t0, Dk('q').t0 + 0.6)) * (1 - smooth(seg(t, Dk('v').t1 - 0.5, Dk('v').t1))) : 0,
        mlpFocus: MF[s.key] ?? null,
        mlpFocusK: MF[s.key] ? smooth(seg(t, s.t0, s.t0 + 0.5)) : 0,
        beamLabels: !['score'].includes(s.key),
        exOnly: s.key === 'over' ? undefined : (s.s?.op ?? null),
        hideMlpLabels: s.s?.op === 'mlp',
        boardA: 0,
        dof: ck === 'dot' ? { range: 1.6, blur: 3.5 } : null,
        ov: {
          formula,
          ffnmap,
          rings,
          heads6: s.key === 'heads' ? { a: smooth(seg(t, hk.t0 + 0.2, hk.t0 + 0.8)) * (1 - smooth(seg(t, hk.t1 - 0.4, hk.t1 - 0.05))), k: seg(t, hk.t0 + 0.3, hk.t0 + 2.8), list: HEADS, on: t < hk.t0 + 3.0 ? null : hOn, L: LX, g: G0, sky: keyTok, en: EN } : null,
        },
      };
    });
  }

  /* ------------------------------------------------------------ ⑥ 选字：理解完之后，怎么把下一个词说出来 */
  const st0 = S(0);
  {
    const T0 = SEC.pick.t0, T1 = SEC.pick.t1;
    prog(T0, 5);
    const sp = lay(T0, [
      { key: 'norm', depth: 4, s: { ph: 'head', sub: 'norm' }, d: 8 },
      { key: 'cmp', depth: 4, s: { ph: 'head', sub: 'unembed' }, d: 7 },
      { key: 'soft', depth: 4, s: { ph: 'head', sub: 'softmax' }, d: 6 },
      { key: 'temp', depth: 4, s: { ph: 'sample', sub: 'temp' }, d: 3 },
      { key: 'topk', depth: 4, s: { ph: 'sample', sub: 'topk' }, d: 2 },
      { key: 'topp', depth: 4, s: { ph: 'sample', sub: 'topp' }, d: 2 },
      { key: 'draw', depth: 4, s: { ph: 'sample', sub: 'draw' }, d: 4 },
    ].map((x) => ({ ...x, d: x.d * BEAT })));
    const P_ = (k) => sp.find((x) => x.key === k);
    ev(P_('norm').t0, 'whoosh', { k: 0.8 });
    ev(P_('cmp').t0, 'step', { k: 0.6 });
    ev(P_('soft').t0, 'reveal', { k: 0.8 });
    ev(P_('temp').t0, 'step', { k: 0.5 });
    ev(P_('draw').t0 + 0.05, 'dice');
    ev(P_('draw').t0 + P_('draw').d * 0.8, 'emit', { g: 0, k: 1.2 });
    const say = (k, html, tg, o = {}) => { const x = P_(k); const a = x.t0 + (o.a ?? 0.2), b = o.b != null ? x.t0 + o.b : (o.until ? P_(o.until).t1 : x.t1) - 0.15; sub(a, b, html); if (tg) term(a + 0.1, b, tg); };
    say('norm', T_('28 层走完：这个位置浓缩了它想说的话', 'After 28 layers, this position holds what it wants to say'), T_(`最后一个位置的向量 · ${m('1024')} 个数`, `last position · 1,024 numbers`), { a: 1.2 });
    say('cmp', T_('拿它和词表里每个词比一比', 'Compare it with every word in the vocabulary'), T_(`点积 · 词表 ${m('151,936')} 个词`, `dot product · 151,936 tokens`), { b: 2.1 });
    say('cmp', T_(`越像，分越高：${m('15')} 万个词各有一分`, `Closer means higher: all 151,936 get a score`), T_(`${q(hm0.token)}得分最高：${m(hm0.total.toFixed(2))}`, `top score: ${q(hm0.token)} ${m(hm0.total.toFixed(2))}`), { a: 2.3 });
    say('soft', T_(`softmax：分数变成概率，${q(st0.top[0][2])} ${m(pct(st0.top[0][1]))}`, `Softmax turns scores into odds: ${q(st0.top[0][2])} ${m(pct(st0.top[0][1]))}`), 'softmax');
    say('temp', T_('按概率抽签：只在最靠谱的几个里抽', 'Then a weighted draw, among only the likeliest few'), T_(`温度 ${m('0.7')} · 前 ${m('20')} 名 · 累计 ${m('80%')}`, `temperature ${m('0.7')} · top-k ${m('20')} · top-p ${m('0.8')}`), { until: 'topp' });
    say('draw', T_(`抽中${q(st0.chosenS)}：说出口`, `It draws ${q(st0.chosenS)}, and says it`));
    const nrm = P_('norm'), smx = P_('soft'), cmpT = P_('cmp');
    shot('pick', (lt, t, { M }) => {
      const s = pick(sp, t);
      const st = mst(s.depth, 0, s.s, s.p, { dAnim: s.depth });
      return {
        st,
        cam: () => {
          const up = smooth(seg(t, smx.t0 - 0.9, smx.t0 + 0.5));
          const hmax = 3.2 * Math.max(Q.steps[0].temps['0.7'][0], Q.steps[0].top[0][1]);
          const head = headCam(M, st, { dz: lerp(7.2, 6.4 + hmax * 1.2 - 0.8 * smooth(lt / 20), up), dx: lerp(1.6, 0.4, smooth(lt / 20)), ly: lerp(0.9, 1.25 + hmax * 0.52, up), lx: lerp(1.8, -0.6, up) });
          if (t >= nrm.t1) return head;
          // 从第 24 层拉出来：看一眼整座塔，再落到塔顶最后一个位置
          const a = endCam.get('d-last') || head;
          const far = cam([M.x(row0) + 9, 15.5, 23], [M.x(row0) - 1.5, 8.6, 0], 34);
          const k = seg(t, nrm.t0, nrm.t0 + 3.4);
          return k < 0.5 ? blendCam(a, far, easeIn(k / 0.5)) : blendCam(far, head, smoother((k - 0.5) / 0.5));
        },
        ov: {
          rings: [
            { at: () => M.normRing.getWorldPosition(v3(0, 0, 0)), w: 0.3, h: 0.3, label: T_('最后一个位置', 'last position'), a: smooth(seg(t, nrm.t0 + 3.0, nrm.t0 + 3.5)) * (1 - smooth(seg(t, nrm.t1 - 0.3, nrm.t1))) },
            // 输出矩阵（和嵌入表共用）：词表里每个词一个向量
            { at: () => { const c = M.lm.getWorldPosition(v3(0, 0, 0)); return v3(M.x(row0) - 1.4, c.y, c.z); }, w: 3.2, h: 0.3, label: T_(`词表：${m('151,936')} 个词`, `vocabulary: 151,936 tokens`), top: true, cls: 'violet', a: smooth(seg(t, cmpT.t0 + 0.4, cmpT.t0 + 0.9)) * (1 - smooth(seg(t, cmpT.t1 - 0.4, cmpT.t1 - 0.1))) },
          ],
        },
      };
    });
  }

  /* ------------------------------------------------------------ 说出口、接回去，再说下一个 */
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
  // 自回归：从机器右前方斜着看，新词元沿右侧的回路落回托盘、在新的一列里往上算
  const loopCam = (M, fx, t, o = {}) => {
    const look = v3(fx - (o.back ?? 3), o.ly ?? 5.6, 0);
    return orbit({ pos: look.clone().add(v3(0, o.dy ?? 3.6, o.dz ?? 23)), look, fov: 32 }, o.yaw ?? 28, 0, 1);
  };
  {
    // 抽签有随机性：中文版是第 2 个字（「是」57% /「之所以」31%，抽中了「之所以」）；英文版是数据里第一次没选第一名的那一步。
    // 那一步之前的词先快速过一遍，然后只给它一个镜头
    const T0 = SEC.loop1.t0, T1 = SEC.loop1.t1;
    const R = DR.rand;
    const stR = S(R);
    const pre = R > 1 ? Math.min(4.2, 0.6 * (R - 1)) : 0;
    const fast = R > 1 ? toks(1, T0, Array.from({ length: R - 1 }, () => pre / (R - 1))) : [];
    fast.forEach((s) => ev(s.emit, 'emit', { g: s.g, k: 0.5, rank: Q.steps[s.g].chosenRank }));
    const tr = T0 + pre; // 这一步开始
    const PH1 = R > 1
      ? [{ ph: 'read', t1: 0.4 }, { ph: 'embed', t1: 0.7 }, { ph: 'layers', t1: 1.9 }, { ph: 'head', t1: 5.4 }, { ph: 'sample', t1: T1 - tr }]
      : [{ ph: 'read', t1: 0.9 }, { ph: 'embed', t1: 1.5 }, { ph: 'layers', t1: 3.8 }, { ph: 'head', t1: 7.8 }, { ph: 'sample', t1: T1 - T0 }];
    const phAt = (lt) => { let a = 0; for (const x of PH1) { if (lt < x.t1 || x === PH1[PH1.length - 1]) return { ph: x.ph, p: clamp((lt - a) / (x.t1 - a)) }; a = x.t1; } return null; };
    const headT = tr + PH1[2].t1, sampT = tr + PH1[3].t1;
    const emitR = sampT + (T1 - sampT) * 0.8;
    ev(sampT + 0.05, 'dice');
    ev(emitR, 'emit', { g: R, k: 1.1, rank: stR.chosenRank });
    sub(T0 + 0.3, headT - 0.2, T_('接到句子后面，再整个过一遍', 'Append it, then run the whole thing again'));
    term(T0 + 0.4, headT - 0.2, T_('自回归：一次只说一个字', 'autoregression: one token at a time'));
    const top2 = `${q(stR.top[0][2])} ${m(pct(stR.top[0][1]))}，${q(stR.top[1][2])} ${m(pct(stR.top[1][1]))}`;
    const top2En = `${q(stR.top[0][2])} ${m(pct(stR.top[0][1]))}, ${q(stR.top[1][2])} ${m(pct(stR.top[1][1]))}`;
    sub(headT + 0.1, sampT - 0.2, T_(`第 ${R + 1} 个字：${top2}`, `${R === 1 ? 'The next word' : `Word ${R + 1}`}: ${top2En}`));
    sub(sampT + 0.1, T1 - 0.2, T_(`这次抽中了${q(stR.chosenS)}：抽签有随机性`, stR.chosenRank > 0 ? `It drew ${q(stR.chosenS)}, not the favourite: it really is a draw` : `It drew ${q(stR.chosenS)}, but it could have gone the other way`));
    shot('loop1', (lt, t, { M }) => {
      let st, detail = false, gNow = R;
      if (t < tr) { const s = loopState(fast, t); gNow = s.g; st = mst(2, s.g, { ph: s.ph }, s.p, { dAnim: 2.7 }); }
      else {
        const { ph, p } = phAt(t - tr);
        detail = ph === 'head' || ph === 'sample';
        st = detail ? mst(4, R, { ph, sub: ph === 'head' ? 'softmax' : 'draw' }, p, { dAnim: 2.7 }) : mst(2, R, { ph }, p, { dAnim: 2.7 });
      }
      const doneFast = fast.filter((x) => t >= x.emit).length;
      const n = 1 + doneFast + (t >= emitR ? 1 : 0);
      return {
        st,
        cam: () => {
          const wide = loopCam(M, M.x(Q.row(gNow)), t, { yaw: lerp(34, 26, smooth(lt / 12)), dz: 22 });
          const hR = 3.2 * Q.steps[R].temps['0.7'][0];
          const close = headCam(M, mst(4, R, { ph: 'sample', sub: 'draw' }, 0), { dz: 6.4 + hR * 1.2, ly: 1.25 + hR * 0.52, lx: -0.6 });
          const kc = smooth(seg(t, headT - 0.8, headT + 0.5));
          const live = blendCam(wide, close, kc);
          return blendCam(prevCam('pick', live), live, smoother(seg(lt, 0, 1.8)));
        },
        hide: detail ? [] : ['bars'],
        ov: { reply: { a: smooth(seg(lt, 0.2, 0.8)), n, k: t >= emitR ? smooth(seg(t, emitR, emitR + 0.35)) : 1 } },
      };
    });
  }
  {
    // 剩下的字：蒙太奇，一口气写完
    const T0 = SEC.loop2.t0, T1 = SEC.loop2.t1;
    const g0 = DR.rand + 1;
    const rest = Q.G - g0;
    const last = 1.2, each = (T1 - T0 - last) / (rest - 1);
    const durs2 = Array.from({ length: rest }, (_, k) => (k === rest - 1 ? last : each));
    const loop2 = toks(g0, T0, durs2);
    loop2.forEach((s, k) => { if (s.g === Q.G - 1) ev(s.emit, 'end', { g: s.g, k: 0.55 }); else if (k % 2 === 0) ev(s.emit, 'emit', { g: s.g, k: 0.45, rank: Q.steps[s.g].chosenRank }); });
    const stEnd = S(Q.G - 1);
    sub(T0 + 0.3, T0 + 5.2, T_('就这样，一个字一个字往下说', 'And so on, one word after another'));
    sub(T0 + 5.4, T1 - 0.15, T_('直到选中结束标记：回答写完', 'until it picks the end marker, and the answer is done'));
    term(T0 + 5.5, T1 - 0.15, `${m('&lt;|im_end|&gt;')} ${m(pct(stEnd.chosenP1))}`);
    shot('loop2', (lt, t, { M }) => {
      const s = loopState(loop2, t);
      const done = loop2.filter((x) => t >= x.emit);
      const newest = done[done.length - 1];
      return {
        st: mst(2, s.g, { ph: s.ph }, s.p, { dAnim: 2.7 }),
        cam: () => {
          const live = loopCam(M, lerp(M.x(Q.row(g0)), M.x(Q.row(Q.G - 1)), smooth(seg(t, T0, T1 - 1))), t, { yaw: lerp(22, 30, smooth(lt / 10)), dz: lerp(24, 28, smooth(lt / 10)), back: lerp(3, 6, smooth(lt / 10)) });
          return blendCam(prevCam('loop1', live), live, smoother(seg(lt, 0, 1.8)));
        },
        hide: ['bars'],
        ov: { reply: { a: 1, n: Math.min(Q.G, g0 + done.length), k: newest ? smooth(seg(t, newest.emit, newest.emit + 0.2)) : 1 } },
      };
    });
  }

  /* ------------------------------------------------------------ 片尾 */
  // 一共做了多少次乘加：每层的矩阵乘法 + 注意力里的 Q·K 和加权求和 + 输出头
  const Mo = Q.M, perTokLayer = Mo.hidden * (Mo.heads * Mo.headDim) * 2 + Mo.hidden * (Mo.kvHeads * Mo.headDim) * 2 + Mo.hidden * Mo.ffn * 3;
  let macs = 0;
  for (let pos = 0; pos < Q.P + Q.G - 1; pos++) macs += NL * (perTokLayer + 2 * Mo.heads * Mo.headDim * (pos + 1));
  macs += Q.G * Mo.hidden * Mo.vocab;
  const statLine = T_(`${Q.G} 个词元 · 每个都走完 28 层 · 约 ${Math.round(macs / 1e8)} 亿次乘加`, `${Q.G} tokens · each through all 28 layers · about ${Math.round(macs / 1e9)} billion multiply-adds`);
  {
    const T0 = SEC.end.t0, T1 = SEC.end.t1;
    // 0–6.2 s 完整回答；6.4 s 起用推理页的真实录屏演示一遍（点选发送 → 点 ＋ 揭开 → 一层层往里钻 → 调试器），最后落版网址 + 二维码
    const D0 = 6.4, FIN = 18.4;
    ev(T0 + 1.6, 'hit', { k: 0.5 });
    // 录屏的时间表：电影里的一段 [f0, f1] 对应录屏里的 [c0, c1]（可以变速；段与段之间直接剪接）
    const A = (cap?.actions || []);
    const at = (name, k = 0) => A.filter((a) => a.name === name)[k]?.t ?? 0;
    const segs = [];
    let f = D0;
    const add = (d, c0, c1, note) => { segs.push({ f0: f, f1: f + d, c0, c1, note }); f += d; };
    if (A.length) {
      add(1.6, at('chip', 0) - 0.5, at('send') + 0.15, 'ask');          // 点选词元、发送
      add(1.2, at('send') + 0.15, at('replyDone') + 0.2, 'reply');      // 回答流式写出来
      add(1.4, at('plus') - 0.4, at('plus') + 3.7, 'reveal');           // 点 ＋ 揭开黑箱
      const dives = A.filter((a) => a.name === 'in');
      dives.forEach((a) => add(0.9, a.t - 0.25, a.t + 1.0, 'dive'));     // 每点一次 ＋ 往里钻一层
      add(FIN - f, at('play') - 0.3, at('step', 1) + 0.8, 'debug');     // 播放、倍速、暂停、单步
    }
    const capAt = (lt) => { if (!segs.length) return 0; const s = segs.find((x) => lt < x.f1) || segs[segs.length - 1]; return lerp(s.c0, s.c1, clamp((lt - s.f0) / (s.f1 - s.f0))); };
    const filmOf = (ct) => { const s = segs.find((x) => ct >= x.c0 && ct <= x.c1); return s ? s.f0 + ((ct - s.c0) / (s.c1 - s.c0)) * (s.f1 - s.f0) : null; };
    for (const a of A) { if (a.x == null) continue; const lt = filmOf(a.t); if (lt != null) ev(T0 + lt, a.name === 'in' ? 'step' : 'tick', { k: a.name === 'in' ? 0.55 : 0.3 }); }
    ev(T0 + FIN, 'end', { k: 0.4 });
    const depthNames = ['结构', '层塔', '一层之内', '注意力', '一次乘加'];
    const dv = segs.filter((s) => s.note === 'dive');
    sub(T0 + D0 + 0.2, T0 + (dv[0]?.f0 ?? D0 + 4) - 1.4 - 0.1, T_('想自己一步步打开看看？', 'Want to open it up yourself, step by step?'));
    if (dv.length) {
      const r = segs.find((s) => s.note === 'reveal');
      sub(T0 + r.f0 + 0.1, T0 + dv[dv.length - 1].f1 - 0.15, T_('在网页上，每点一次 ＋，就往里钻一层', 'On the website, every ＋ takes you one level deeper'));
      // 一路钻下去的层次：一个标签从头到尾挂着（和网页顶栏的面包屑一致）
      term(T0 + r.f0 + 0.2, T0 + dv[dv.length - 1].f1 - 0.15, T_('黑箱 → 结构 → 层塔 → 一层之内 → 注意力 → 一次乘加', DR.depthPath ?? 'black box → structure → layer tower → inside a layer → attention → one multiply-add'));
      const dbg = segs.find((s) => s.note === 'debug');
      sub(T0 + dbg.f0 + 0.1, T0 + FIN - 0.15, T_('暂停、单步、倍速，像调试程序一样看它思考', 'Pause, step, speed up: debug its thinking like a program'));
      term(T0 + dbg.f0 + 0.2, T0 + FIN - 0.15, T_('播放 · 倍速 · 暂停 · 单步', 'play · speed · pause · step'));
    }
    shot('end', (lt, t, { M }) => ({
      st: mst(2, Q.G - 1, { ph: 'sample' }, 1, { dAnim: lerp(2.7, 1, smoother(seg(lt, 0.4, 3.9))), view: 'machine' }),
      cam: () => {
        const far = cam([-11, 5.5, 70], [0, 11.5, 0], 30);
        const live = loopCam(M, M.x(Q.row(Q.G - 1)), t, { yaw: 30, dz: 28, back: 6 });
        return blendCam(prevCam('loop2', live), far, smoother(seg(lt, 0, 10)));
      },
      hide: ['bars'],
      dof: { focus: 14, range: 24, blur: 6 * smooth(seg(lt, 2.5, 5)) },
      fade: lerp(0, 0.86, smooth(seg(lt, 0.6, 2.0))) + 0.14 * smooth(seg(lt, D0 - 0.4, D0 + 0.4)),
      ov: {
        reply: { a: 1 - smooth(seg(lt, 0.4, 1.4)), n: Q.G, k: 1 },
        band: 1 - smooth(seg(lt, 0, 1)),
        end: { a1: smooth(seg(lt, 1.4, 2.4)) * (1 - smooth(seg(lt, D0 - 0.8, D0 - 0.2))), k1: smooth(seg(lt, 3.6, 4.6)) },
        site: segs.length ? {
          a: smooth(seg(lt, D0 - 0.1, D0 + 0.5)) * (1 - smooth(seg(t, T1 - 0.9, T1 - 0.05))),
          ct: capAt(Math.min(lt, FIN)),
          fin: easeInOut(seg(lt, FIN, FIN + 1.0)),
          finA: smooth(seg(lt, FIN + 0.5, FIN + 1.3)) * (1 - smooth(seg(t, T1 - 0.9, T1 - 0.05))),
          cursor: lt < FIN ? smooth(seg(lt, D0 + 0.3, D0 + 0.7)) * (1 - smooth(seg(lt, FIN - 0.4, FIN))) : 0,
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
  return { end, frame, subs, strips, chapters, cards, terms, chapterNames: CHAPTERS, progAt, endSite, events, sections, open: OPEN, statLine, bpm: BPM, shots: shots.map((s) => ({ name: s.name, t0: s.t0, t1: s.t1 })) };
}
