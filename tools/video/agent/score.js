// 智能体视频的分镜表：每一段的时间、桌面（编辑器 + 终端）回放到哪、镜头、讲解面板的状态、字幕、术语标签、配乐事件。
// 节拍 96 BPM（一拍 0.625 秒，一小节 2.5 秒），段落都卡在小节线上；compose.py 读同一张段落表和事件表。
//
// 任务：「sales.csv 里哪个城市的总销售额最高？写个 Python 脚本算出来。」（sales-top，Qwen3-4B-Instruct-2507 在 bwrap 沙箱里的真实录制）
// 8 圈、7 次工具调用，中间自己改了 3 次：python 找不到 → python3；缺 pandas → pip 装不上 → 改用自带的 csv 重写 → 跑通（杭州 9250）。
// 字幕和面板里的数字全部从数据里现取。
import { clamp, lerp, seg, smooth, smoother, easeInOut, cam2 } from './util.js';
import { fmtPct, esc } from '/public/js/ui.js';

export const BPM = 96, BEAT = 60 / BPM, BAR = 4 * BEAT;
const B = (bar, beat = 0) => bar * BAR + beat * BEAT;

// 段落：名字、小节数、能量（配乐用）
const PLAN = [['chat', 6, 0.1], ['work', 6, 0.55], ['loop', 10, 0.42], ['ctx', 10, 0.38], ['tok', 12, 0.34], ['fix', 12, 0.75], ['end', 18, 0.22]];
export const CHAPTERS = ['动手干活', '只会写字', '越滚越长', '逐词生成', '自己改错'];

const m = (s) => `<span class="m">${s}</span>`;
const q = (s) => `<q>${esc(s)}</q>`;
const num = (n) => Math.round(n).toLocaleString('en-US');

// 桌面镜头（桌面自己的坐标：1720×804；放到画面上的 (cx, cy)，缩放 s）
const DESK = { w: 1720, h: 804 };
const FULL = { x: 860, y: 402, s: 1, cx: 960, cy: 472 };
const ED = { x: 760, y: 290, s: 1.28, cx: 960, cy: 440 };
const TERM = { x: 1000, y: 640, s: 1.32, cx: 960, cy: 480 };
const SLOT = { x: 860, y: 402, s: 0.3, cx: 1578, cy: 492 };

export function buildScore(R, cap = null) {
  const subs = [], terms = [], events = [], sections = [], progs = [];
  const sub = (t0, t1, html) => subs.push({ t0, t1, html });
  const term = (t0, t1, html) => terms.push({ t0, t1, html });
  const ev = (t, type, o = {}) => events.push({ t, type, ...o });
  const prog = (t, i) => progs.push({ t, i });

  const SEC = {};
  let bb = 0;
  for (const [name, bars, energy] of PLAN) {
    SEC[name] = { t0: B(bb), t1: B(bb + bars) };
    sections.push({ name, t0: B(bb), t1: B(bb + bars), bars, energy });
    bb += bars;
  }
  const end = B(bb);

  // 真实数据
  const turns = R.turns;
  const acts = [];
  turns.forEach((T, t) => T.calls.forEach((c, ci) => acts.push({ t, ci, c })));
  const T3 = turns[3], T2 = turns[2];
  const toks = T3.gen.toks;                             // 第 4 圈：读到 “python: command not found” 以后写出的 23 个词元
  const jBash = toks.findIndex((x) => x.s === 'bash');
  const jPy = toks.findIndex((x) => x.s === 'python');
  const j3 = jPy + 1;                                   // 「3」
  const prev3 = T2.gen.toks[T2.gen.toks.findIndex((x) => x.s === 'python') + 1]; // 上一圈同一个位置
  const prev3p = prev3.top.find((x) => x[0] === '3')?.[1] ?? 0;
  const ctxN = turns.map((T) => T.ctx.n);
  const segSum = (T) => { const s = {}; for (const g of T.ctx.segs) s[g.k] = (s[g.k] || 0) + g.n; return s; };
  const S0 = segSum(turns[0]);
  const kv3 = T3.kv;
  const nErr = acts.filter((a) => a.c.name === 'bash' && /\[(退出码|exit code) [1-9]/.test(a.c.result)).length;
  const finalAct = acts.length - 1;

  /* ------------------------------------------------------------ 开场：聊天页 → 打字 → 发送 → 回复变成一个“动作” → 推进动作卡片 → 桌面 */
  // 打字：中文用拼音输入（敲完拼音、候选一闪、上屏），英文和标点直接敲
  const WORDS = [
    ['sales.csv'], [' '],
    ['里', 'li', ['里', '理', '离', '李', '力']],
    ['哪个', "na'ge", ['哪个', '那个', '拿个', '哪']],
    ['城市的', "cheng'shi'de", ['城市的', '城市', '成事的', '城']],
    ['总销售额', "zong'xiao'shou'e", ['总销售额', '总销售', '总', '纵']],
    ['最高', "zui'gao", ['最高', '最好', '醉', '最']],
    ['？'],
    ['写个', "xie'ge", ['写个', '写', '些个', '鞋']],
    [' Python '],
    ['脚本', "jiao'ben", ['脚本', '叫', '角', '较']],
    ['算出来', "suan'chu'lai", ['算出来', '算出', '算', '酸']],
    ['。'],
  ];
  const jit = (i) => (((i * 7919) % 13) / 13 - 0.5) * 0.024;
  let tk = 0.95, kn = 0;
  const typing = WORDS.map(([s, py, cands], k) => {
    if (!py) {
      const keys = [...s].map((ch, j) => { const t = tk; tk += 0.062 + jit(k * 10 + j) + (ch === ' ' ? 0.05 : 0); return t; });
      tk += 0.1;
      return { s, py: null, keys, commit: keys[keys.length - 1], chars: true };
    }
    const segd = py, n = py.replace(/'/g, '').length;
    const keys = Array.from({ length: n }, (_, j) => tk + j * 0.06 + jit(k * 10 + j));
    const commit = keys[n - 1] + 0.11;
    tk = commit + 0.14;
    return { s, py: segd.replace(/'/g, ''), seg: segd, cands, keys, commit };
  });
  const OPEN = {
    typing,
    sendT: B(3, 2),
    boxY0: 560, boxY1: 912,
  };
  OPEN.bubT = OPEN.sendT + 0.06;
  OPEN.botT = OPEN.sendT + 0.4;
  OPEN.toolT = OPEN.sendT + 1.2;          // 助手的回复不是一句话，而是一个“动作”：读文件
  OPEN.push0 = OPEN.sendT + 1.55; OPEN.push1 = B(4, 3) + 0.1;   // 推进动作卡片
  OPEN.desk0 = B(4, 2); OPEN.desk1 = B(5) - 0.1;                // 桌面从卡片里展开
  OPEN.end = B(5) + 0.2;
  OPEN.warpT = OPEN.push0 + 0.2;
  const titleIn = B(5), titleOut0 = B(6) - 0.15, titleOut1 = B(6) + 0.45;
  for (const w of typing) for (const k of w.keys) ev(k, 'key', { i: kn++ });
  typing.forEach((w, i) => { if (w.py) ev(w.commit, 'type', { i }); });
  ev(OPEN.sendT, 'send');
  ev(OPEN.toolT, 'tick', { k: 0.6 });
  ev(OPEN.desk0 - 0.2, 'whoosh', { k: 0.7 });
  ev(titleIn, 'hit', { k: 1 });

  /* ------------------------------------------------------------ 桌面回放的时间表：第 i 个工具调用从 t0 演到 t1 */
  const DS = [];
  const run = (i, t0, t1) => { DS.push({ i, t0, t1 }); const c = acts[i].c; ev(t0, c.name === 'bash' ? 'cmd' : 'tick', { k: 0.5 }); if (c.name === 'bash') { const bad = !/\[(退出码|exit code) 0\]/.test(c.result); ev(lerp(t0, t1, bad ? 0.5 : 0.55), bad ? 'err' : 'ok', { k: 0.8 }); } };
  // 第 1 章：读数据 → 写脚本（pandas）→ python 找不到
  run(0, 15.2, 17.4);
  run(1, 18.0, 22.6);
  run(2, 23.0, 26.4);
  // 第 5 章：python3 → 缺 pandas → pip 装不上 → 用 csv 重写 → 跑通 → 最终回答
  run(3, 110.2, 113.6);
  run(4, 114.6, 119.2);
  run(5, 125.4, 130.4);
  run(6, 130.8, 134.0);
  const FINAL_T = 137.4;                     // 第 8 圈：最终回答写完，桌面显示“任务完成 · 独立检查”
  const deskAt = (t) => {
    let k = 0, cur = -1, p = 0;
    for (const d of DS) {
      if (t >= d.t1) k = Math.max(k, d.i + 1);
      else if (t >= d.t0) { cur = d.i; p = clamp((t - d.t0) / (d.t1 - d.t0)); }
    }
    return { k, cur, p, final: t >= FINAL_T };
  };
  // 每个时刻是第几圈（右上角）
  const lapAt = (t) => {
    if (t < 18.0) return 1; if (t < 23.0) return 2; if (t < 48.2) return 3;
    if (t < 114.6) return 4; if (t < 119.8) return 5; if (t < 130.8) return 6; if (t < 135.2) return 7; return 8;
  };

  /* ------------------------------------------------------------ 第 1 章 · 动手干活 */
  prog(SEC.work.t0, 0);
  sub(15.6, 17.9, '交给它一个任务，它就自己动手');
  term(15.7, 17.9, `read_file · 读文件 ${m('sales.csv')}`);
  sub(18.2, 22.5, '看完数据，写一个 Python 脚本');
  term(18.3, 22.5, `write_file · 写文件 ${m('analyze_sales.py')}`);
  sub(23.2, 26.8, '运行一下……报错了');
  term(23.3, 26.8, `bash · 执行命令 · 退出码 ${m('127')}`);
  sub(27.1, 29.9, '可是，模型自己根本碰不到电脑');
  term(27.2, 29.9, '大模型 · 只会输出文字');
  const deskCam1 = cam2([
    { t: 11.6, ...FULL }, { t: 18.3, ...FULL }, { t: 20.0, ...ED }, { t: 22.6, ...ED, y: 300 },
    { t: 23.6, ...TERM }, { t: 26.4, ...TERM, y: 652 }, { t: 28.6, ...TERM, x: 860, y: 600, s: 1.62 }, { t: 29.6, ...TERM, x: 860, y: 600, s: 1.64 },
  ]);

  /* ------------------------------------------------------------ 第 2 章 · 只会写字：镜头拉开，桌面缩进“沙箱”，左边是模型，中间是 harness */
  prog(SEC.loop.t0, 1);
  const L = SEC.loop.t0; // 30
  const deskCam2 = cam2([{ t: L, ...TERM, x: 860, y: 600, s: 1.64 }, { t: L + 3.6, ...SLOT }]);
  sub(L + 0.3, L + 3.8, '其实，模型能做的只有一件事');
  term(L + 0.4, L + 3.8, '模型 · 读一串字，写一串字');
  sub(L + 4.0, L + 8.9, '写出一段字：这就是它的“动作”');
  term(L + 4.1, L + 8.9, '工具调用 tool call');
  sub(L + 9.2, L + 14.3, '外面的程序读懂它，替它去执行');
  term(L + 9.3, L + 14.3, 'harness · 外面的程序');
  sub(L + 14.6, L + 19.3, '结果原样接回对话，再交给模型');
  term(L + 14.7, L + 19.3, '工具结果 tool result');
  sub(L + 19.6, L + 24.8, '一圈接一圈，直到它不再调用工具');
  term(L + 19.7, L + 24.8, 'agent 循环 · 模型 ＋ 工具 ＋ 循环');
  // 数据包：沿着边飞（from → to）
  const PK = [
    { t0: L + 9.2, t1: L + 10.4, e: 'm2h', cls: 'amber', html: '&lt;tool_call&gt; …' },
    { t0: L + 11.7, t1: L + 12.8, e: 'h2s', cls: 'cyan', html: esc(acts[2].c.args.command) },
    { t0: L + 14.7, t1: L + 15.9, e: 's2h', cls: 'rose', html: `command not found · 退出码 127` },
    { t0: L + 16.9, t1: L + 18.2, e: 'h2m', cls: 'violet', html: `整段上下文 · ${num(ctxN[3])} 词元` },
  ];
  for (const p of PK) ev(p.t0, 'packet', { k: 0.6 });
  // harness 正在做哪一步（ops：0 拼上下文 1 等模型写完 2 找 tool_call 3 沙箱执行 4 接回对话）
  const OPS = [
    { t: L + 4.0, op: 1 }, { t: L + 10.3, op: 2 }, { t: L + 11.6, op: 3 }, { t: L + 15.9, op: 4 }, { t: L + 16.8, op: 0 },
    { t: L + 19.6, op: 1 }, { t: L + 20.6, op: 2 }, { t: L + 21.4, op: 3 }, { t: L + 22.4, op: 4 }, { t: L + 23.2, op: 0 }, { t: L + 24.0, op: 1 },
  ];
  // 一圈一圈：光点在四条边上循环（第 20 秒以后）
  const loopFlow = (t) => (t < L + 19.6 ? -1 : (t - (L + 19.6)) / 2.6);
  ev(L + 18.2, 'lap', { k: 0.7 });
  for (let k = 0; k < 2; k++) ev(L + 19.6 + k * 2.6, 'packet', { k: 0.35 });

  /* ------------------------------------------------------------ 第 3 章 · 越滚越长：每一圈喂给模型的整段上下文 */
  prog(SEC.ctx.t0, 2);
  const C = SEC.ctx.t0; // 55
  const rowAt = (i) => (i === 0 ? C + 8.6 : C + 9.4 + (i - 1) * 0.9);
  sub(C + 2.6, C + 5.9, '开头：规矩和工具说明书');
  term(C + 2.7, C + 5.9, `系统提示 ${m(S0.sys)} · 工具说明 ${m(S0.tools)} 词元`);
  sub(C + 6.1, C + 8.9, '然后，才是你交代的任务');
  term(C + 6.2, C + 8.9, `任务 ${m(S0.user)} 词元 · 第 1 圈一共 ${m(num(ctxN[0]))}`);
  sub(C + 9.2, C + 12.4, '它写的话、命令的结果，全接在后面');
  term(C + 9.3, C + 12.4, '每一圈都把整段历史重新读一遍');
  sub(C + 12.6, C + 15.9, `越滚越长：${m(num(ctxN[0]))} → ${m(num(ctxN[ctxN.length - 1]))} 个词元`);
  term(C + 12.7, C + 15.9, `第 1 圈 → 第 ${turns.length} 圈`);
  sub(C + 16.2, C + 20.8, '读过的有缓存，只算新接上的');
  term(C + 16.3, C + 20.8, `KV 缓存 · 第 4 圈：复用 ${m(num(kv3.reused))} · 新算 ${m(num(kv3.computed))}`);
  sub(C + 21.2, C + 24.8, '第 4 圈，它刚读到这句报错');
  term(C + 21.3, C + 24.8, '上下文的最后几行');
  ev(C + 2.4, 'seg', { k: 0.5, i: 0 }); ev(C + 3.2, 'seg', { k: 0.5, i: 1 }); ev(C + 6.2, 'seg', { k: 0.6, i: 2 });
  for (let i = 0; i < turns.length; i++) ev(rowAt(i), 'row', { i, k: 0.5 });
  ev(C + 16.4, 'step', { k: 0.5 });
  ev(C + 21.2, 'err', { k: 0.35 });

  /* ------------------------------------------------------------ 第 4 章 · 逐词生成（全片唯一一次细看）：第 4 圈的工具调用一个词元一个词元写出来 */
  prog(SEC.tok.t0, 3);
  const K = SEC.tok.t0; // 80
  // 每个词元出现的时刻；关键处停下来看前 5 名候选
  const tokT = [];
  {
    let t = K + 1.6;
    toks.forEach((x, j) => {
      if (j === jBash) t = K + 3.2;
      if (j === jBash + 1) t = K + 7.6;
      if (j === jPy) t = K + 9.4;
      if (j === j3) t = K + 13.2;
      if (j === j3 + 1) t = K + 20.6;
      tokT.push(t);
      t += j < jBash ? 0.24 : j < jPy ? 0.2 : 0.22;
    });
  }
  const tEnd = tokT[toks.length - 1];
  const DEC = [{ j: jBash, t0: K + 3.2, t1: K + 7.4 }, { j: jPy, t0: K + 9.4, t1: K + 13.0 }, { j: j3, t0: K + 13.2, t1: K + 20.4 }];
  const CMP = { t0: K + 16.6, t1: K + 20.4 };
  const PARSE = { t0: K + 23.8, t1: K + 27.4 };
  toks.forEach((x, j) => ev(tokT[j], DEC.some((d) => d.j === j) ? (j === j3 ? 'climax' : 'dec') : 'tok', { j, k: j === j3 ? 1 : 0.5 }));
  ev(CMP.t0, 'step', { k: 0.4 });
  ev(PARSE.t0, 'packet', { k: 0.6 });
  sub(K + 0.4, K + 3.0, '接下来写什么？一次写一个词元');
  term(K + 0.5, K + 3.0, '逐词元生成 · 第 4 圈');
  sub(K + 3.8, K + 7.3, '先挑工具：执行命令，还是写文件');
  term(K + 3.9, K + 7.3, `前 5 名候选 · ${m('bash')} ${fmtPct(toks[jBash].p)} · ${m(esc(toks[jBash].top[1][0]))} ${fmtPct(toks[jBash].top[1][1])}`);
  sub(K + 9.6, K + 12.9, '命令还是以 python 开头……');
  term(K + 9.7, K + 12.9, toks[jPy].top.slice(0, 3).map(([s, p]) => `${m(esc(s))} ${fmtPct(p)}`).join(' · '));
  sub(K + 13.4, K + 16.4, '读过报错，它补上了一个「3」');
  term(K + 13.5, K + 16.4, `${q('3')} ${fmtPct(toks[j3].p)}`);
  sub(K + 16.8, K + 20.3, `上一圈，这里写「3」只有 ${m(fmtPct(prev3p))}`);
  term(K + 16.9, K + 20.3, '同一个模型 · 上下文变了，概率就变了');
  sub(K + 21.4, K + 23.5, '写完这一段，停笔');
  term(K + 21.5, K + 23.5, `结束符 ${m('&lt;|im_end|&gt;')}`);
  sub(PARSE.t0 + 0.1, PARSE.t1 - 0.1, '外面的程序解析出命令，拿去执行');
  term(PARSE.t0 + 0.2, PARSE.t1 - 0.1, 'harness · 解析 JSON → 沙箱执行');
  const deskCam4 = cam2([{ t: K + 27.4, ...SLOT }, { t: K + 30.2, ...TERM, y: 652 }]);

  /* ------------------------------------------------------------ 第 5 章 · 自己改错：快放剩下的几圈 */
  prog(SEC.fix.t0, 4);
  const F = SEC.fix.t0; // 110
  sub(F + 0.5, F + 4.2, '又报错：沙箱里没有 pandas');
  term(F + 0.6, F + 4.2, `第 4 圈 · 退出码 ${m('1')}`);
  sub(F + 4.8, F + 9.6, '想装上 pandas？pip 拒绝安装');
  term(F + 4.9, F + 9.6, `第 5 圈 · ${m('pip3 install pandas')} · 退出码 ${m('1')}`);
  sub(F + 10.2, F + 15.0, '它换了个思路：不用 pandas 了');
  term(F + 10.3, F + 15.0, '第 6 圈 · 模型自己写下的想法');
  sub(F + 15.6, F + 20.3, '只用 Python 自带的 csv 重写');
  term(F + 15.7, F + 20.3, `第 6 圈 · write_file · ${m(String(acts[5].c.args.content.split('\n').length))} 行`);
  sub(F + 21.0, F + 25.0, '这次跑通了：杭州，9,250 元');
  term(F + 21.1, F + 25.0, `第 7 圈 · 退出码 ${m('0')}`);
  sub(F + 25.4, F + 29.7, '不再调用工具，循环结束');
  term(F + 25.5, F + 29.7, '第 8 圈 · 最终回答');
  const SAY = { t0: F + 9.8, t1: F + 15.3, type0: F + 10.2, type1: F + 13.4, text: turns[5].say.trim() };
  const ANS = { t0: F + 25.2, t1: F + 30.0, type0: F + 25.5, type1: F + 27.6, text: turns[turns.length - 1].say.trim() };
  ev(SAY.type0, 'tick', { k: 0.4 }); ev(ANS.type0, 'tick', { k: 0.4 }); ev(FINAL_T, 'ok', { k: 1 });
  const deskCam5 = cam2([
    { t: F, ...TERM, y: 652 }, { t: F + 3.6, ...TERM, y: 660 }, { t: F + 4.6, ...TERM, y: 620 }, { t: F + 9.6, ...TERM, y: 640 },
    { t: F + 11.0, ...FULL, cy: 540 }, { t: F + 15.2, ...FULL, cy: 540 }, { t: F + 16.4, ...ED }, { t: F + 20.4, ...ED, y: 330 },
    { t: F + 21.4, ...TERM, y: 652 }, { t: F + 24.6, ...TERM, y: 660 }, { t: F + 26.0, ...FULL, cy: 500 }, { t: F + 30, ...FULL, cy: 500 },
  ]);

  /* ------------------------------------------------------------ 片尾：模型 + 工具 + 循环 → 智能体页的真实录屏 → 落版网址 + 二维码 */
  const E0 = SEC.end.t0; // 140
  sub(E0 + 1.0, E0 + 5.2, '像 Claude Code 这样的编程 agent');
  sub(E0 + 5.4, E0 + 9.7, '本质上都是这个模式');
  ev(E0 + 0.8, 'hit', { k: 0.55 });
  const statLine = `${turns.length} 圈 · ${acts.length} 次工具调用 · ${nErr} 次报错，自己改对 · 上下文 ${num(ctxN[0])} → ${num(ctxN[ctxN.length - 1])} 词元`;
  // 录屏的时间表：片尾的一段 [f0, f1]（相对片尾开头）对应录屏里的 [c0, c1]（可以变速；段与段之间直接剪接）
  const D0 = 10.4, FIN = 36.2;
  const A = cap?.actions || [];
  const at = (name, k = 0) => A.filter((a) => a.name === name)[k]?.t ?? 0;
  const segs = [];
  let f = D0;
  const add = (d, c0, c1, note) => { segs.push({ f0: f, f1: f + d, c0, c1, note }); f += d; };
  if (A.length) {
    add(1.6, at('card') - 0.9, at('card') + 0.5, 'pick');                 // 点任务卡片
    add(4.4, at('card') + 0.5, at('watchEnd'), 'watch');                  // 看它干活（快放）
    add(2.6, at('fab') - 0.5, at('fab') + 2.0, 'reveal');                 // 点“揭开内部”
    add(2.4, at('lap') - 0.5, at('lap') + 1.4, 'lap');                    // 点第 4 圈
    add(3.0, at('in', 0) - 0.4, at('in', 0) + 2.6, 'dive');              // ＋ 上下文
    add(FIN - f, at('in', 1) - 0.4, at('end'), 'dive');                   // ＋ 逐词元
  }
  const capAt = (lt) => { if (!segs.length) return 0; const s = segs.find((x) => lt < x.f1) || segs[segs.length - 1]; return lerp(s.c0, s.c1, clamp((lt - s.f0) / (s.f1 - s.f0))); };
  const filmOf = (ct) => { const s = segs.find((x) => ct >= x.c0 && ct <= x.c1); return s ? s.f0 + ((ct - s.c0) / (s.c1 - s.c0)) * (s.f1 - s.f0) : null; };
  for (const a of A) { if (a.x == null) continue; const lt = filmOf(a.t); if (lt != null) ev(E0 + lt, a.name === 'in' ? 'step' : 'tick', { k: a.name === 'in' ? 0.55 : 0.35 }); }
  ev(E0 + FIN, 'end', { k: 0.4 });
  if (segs.length) {
    const sg = (n, k = 0) => segs.filter((s) => s.note === n)[k];
    sub(E0 + D0 + 0.2, E0 + sg('watch').f1 - 0.1, '想自己一步步看？选一个任务');
    sub(E0 + sg('reveal').f0 + 0.1, E0 + FIN - 0.15, '每点一次 ＋，往里钻一层');
    term(E0 + sg('reveal').f0 + 0.2, E0 + FIN - 0.15, '屏幕 → 循环 → 上下文 → 逐词元生成');
  }
  const endSite = { segs, D0, FIN };

  /* ------------------------------------------------------------ 每一帧的状态 */
  const opsAt = (t) => { let op = -1; for (const o of OPS) if (t >= o.t) op = o.op; return op; };
  function frame(t) {
    t = clamp(t, 0, end - 1e-6);
    const S = { t, ov: {} };
    // 开场
    S.open = t < OPEN.end;
    S.title = { a: smooth(seg(t, titleIn, titleIn + 0.8)) * (1 - smooth(seg(t, titleOut0, titleOut1))), k: seg(t, titleIn + 0.05, titleIn + 2.6), blur: 7 * smooth(seg(t, titleOut0, titleOut1)) };
    // 桌面
    const d = deskAt(t);
    let cam = FULL, a = 1, blur = 0, dim = 0, focus = null, fk = 0, hl = null;
    if (t < SEC.work.t0) {
      // 从动作卡片里展开：先小、虚，再落定；片名在上面时压暗
      const k = easeInOut(seg(t, OPEN.desk0, OPEN.desk1));
      cam = { ...FULL, s: lerp(0.78, 1, k) };
      a = smooth(seg(t, OPEN.desk0, OPEN.desk0 + 0.6));
      blur = lerp(10, 0, k) + 5 * smooth(seg(t, titleIn - 0.4, titleIn + 0.4)) * (1 - smooth(seg(t, titleOut0, titleOut1)));
      dim = 0.55 * smooth(seg(t, titleIn - 0.4, titleIn + 0.4));
    } else if (t < SEC.loop.t0) {
      cam = deskCam1(t);
      dim = 0.55 * (1 - smooth(seg(t, titleOut0, titleOut1)));
      blur = 5 * (1 - smooth(seg(t, titleOut0, titleOut1)));
      if (t >= 18.2 && t < 22.9) { focus = 'ed'; fk = smooth(seg(t, 18.2, 18.8)) * (1 - smooth(seg(t, 22.4, 22.9))); }
      if (t >= 23.0) { focus = 'term'; fk = smooth(seg(t, 23.0, 23.6)); }
      if (t >= 25.0) hl = { act: 2, line: 0, k: smooth(seg(t, 25.0, 25.5)), cls: 'bad' };
    } else if (t < SEC.ctx.t0) {
      cam = deskCam2(t);
      focus = 'term'; fk = 1 - smooth(seg(t, L, L + 2));
      hl = { act: 2, line: 0, k: 1 - smooth(seg(t, L, L + 1.5)), cls: 'bad' };
    } else if (t < SEC.tok.t0 + 27.4) {
      cam = SLOT;
      dim = 0.85 * smooth(seg(t, C, C + 1.6));
      a = 1 - smooth(seg(t, C + 0.5, C + 2.4));
    } else if (t < SEC.fix.t0) {
      cam = deskCam4(t);
      a = smooth(seg(t, K + 27.4, K + 28.6));
      focus = 'term'; fk = smooth(seg(t, K + 28.6, K + 30));
    } else if (t < SEC.end.t0) {
      cam = deskCam5(t);
      const lt = t - F;
      if (lt < 9.8) { focus = 'term'; fk = 1; }
      else if (lt < 15.4) { focus = 'term'; fk = 1 - smooth(seg(lt, 9.8, 10.6)); }
      else if (lt < 20.6) { focus = 'ed'; fk = smooth(seg(lt, 15.4, 16.0)) * (1 - smooth(seg(lt, 20.2, 20.6))); }
      else if (lt < 25.2) { focus = 'term'; fk = smooth(seg(lt, 20.6, 21.2)) * (1 - smooth(seg(lt, 24.8, 25.4))); }
      if (lt >= 1.6 && lt < 4.4) hl = { act: 3, line: -1, k: smooth(seg(lt, 1.6, 2.1)) * (1 - smooth(seg(lt, 4.0, 4.4))), cls: 'bad' };
      if (lt >= 6.2 && lt < 9.6) hl = { act: 4, line: 0, k: smooth(seg(lt, 6.2, 6.7)) * (1 - smooth(seg(lt, 9.2, 9.6))), cls: 'bad' };
      if (lt >= 22.4 && lt < 25.2) hl = { act: 6, line: 0, k: smooth(seg(lt, 22.4, 22.9)) * (1 - smooth(seg(lt, 24.8, 25.2))), cls: 'ok' };
      dim = 0.4 * smooth(seg(lt, 9.8, 10.6)) * (1 - smooth(seg(lt, 15.0, 15.6))) + 0.35 * smooth(seg(lt, 25.0, 25.6));
      // 片尾前：桌面退场
      a = 1 - smooth(seg(t, SEC.end.t0 - 0.4, SEC.end.t0 + 0.6));
    } else {
      cam = FULL; a = Math.max(0, 1 - smooth(seg(t, SEC.end.t0 - 0.4, SEC.end.t0 + 0.6))); dim = 0.4; blur = 6;
    }
    S.desk = { ...d, cam, a, blur, dim, focus, fk, hl };
    S.lap = { n: lapAt(t), a: (t >= SEC.work.t0 + 0.4 && t < SEC.loop.t0 + 1) || (t >= SEC.fix.t0 - 0.6 && t < SEC.end.t0 - 0.3) ? 1 : 0 };
    // 第 2 章：循环
    if (t >= L - 0.5 && t < SEC.ctx.t0 + 3) {
      const nodesA = smooth(seg(t, L + 1.2, L + 3.4)) * (1 - smooth(seg(t, C - 0.2, C + 1.0)));
      S.loop = {
        a: nodesA, slide: easeInOut(seg(t, L + 1.0, L + 3.6)),
        out: { a: smooth(seg(t, L + 4.0, L + 4.5)), n: clamp((t - (L + 4.2)) / 2.2), hl: smooth(seg(t, L + 6.6, L + 7.2)) * (1 - smooth(seg(t, L + 9.0, L + 9.6))) },
        op: opsAt(t), lap: t < L + 18.2 ? 3 : 4,
        pk: PK.map((p) => ({ ...p, u: (t - p.t0) / (p.t1 - p.t0) })).filter((p) => p.u > -0.05 && p.u < 1.25),
        flow: loopFlow(t), edgeA: smooth(seg(t, L + 2.4, L + 3.6)),
        sbxFlash: Math.exp(-Math.max(0, t - (L + 12.8)) / 0.5) * (t >= L + 12.8 ? 1 : 0),
        modelIn: Math.exp(-Math.max(0, t - (L + 18.2)) / 0.6) * (t >= L + 18.2 ? 1 : 0),
      };
    }
    // 第 3 章：上下文
    if (t >= C && t < K + 1.6) {
      const showLap = t < C + 9.4 ? 0 : t < C + 16.0 ? clamp(Math.floor((t - (C + 9.4)) / 0.9) + 1, 0, turns.length - 1) : 3;
      S.ctx = {
        a: smooth(seg(t, C + 0.6, C + 2.2)) * (1 - smooth(seg(t, K, K + 1.2))),
        lap: showLap,
        build: { sys: smooth(seg(t, C + 2.4, C + 3.2)), tools: smooth(seg(t, C + 3.2, C + 4.2)), user: smooth(seg(t, C + 6.2, C + 6.8)) },
        hiSeg: t < C + 6.1 ? 'head' : t < C + 9.2 ? 'user' : t < C + 12.6 ? 'tail' : null,
        rows: Array.from({ length: turns.length }, (_, i) => smooth(seg(t, rowAt(i), rowAt(i) + 0.45))),
        kv: smooth(seg(t, C + 16.3, C + 17.2)),
        raw: smooth(seg(t, C + 21.0, C + 21.8)),
        rawHi: smooth(seg(t, C + 22.0, C + 22.6)),
      };
    }
    // 第 4 章：逐词元
    if (t >= K - 0.2 && t < K + 28.6) {
      let n = 0;
      for (let j = 0; j < toks.length; j++) if (t >= tokT[j]) n = j + 1;
      const dec = DEC.find((x) => t >= x.t0 && t < x.t1);
      S.tok = {
        a: smooth(seg(t, K, K + 1.2)) * (1 - smooth(seg(t, K + 27.0, K + 28.2))),
        n, cur: n - 1, kNew: n ? smooth(seg(t, tokT[n - 1], tokT[n - 1] + 0.18)) : 0,
        dec: dec ? { j: dec.j, a: smooth(seg(t, dec.t0, dec.t0 + 0.35)) * (1 - smooth(seg(t, dec.t1 - 0.3, dec.t1))), k: seg(t, dec.t0 + 0.2, dec.t0 + 1.2) } : null,
        cmp: smooth(seg(t, CMP.t0, CMP.t0 + 0.5)) * (1 - smooth(seg(t, CMP.t1 - 0.3, CMP.t1))),
        glow3: t >= tokT[j3] ? Math.exp(-(t - tokT[j3]) / 1.4) : 0,
        parse: smooth(seg(t, PARSE.t0, PARSE.t0 + 0.8)),
        parseK: seg(t, PARSE.t0 + 0.6, PARSE.t1 - 0.6),
        ctxHi: smooth(seg(t, K + 9.4, K + 9.9)) * (1 - smooth(seg(t, K + 16.4, K + 16.8))),
      };
    }
    // 第 5 章：模型自己写下的话（第 6 圈的想法、第 8 圈的最终回答）
    for (const [key, X] of [['say', SAY], ['ans', ANS]]) {
      if (t >= X.t0 - 0.1 && t < X.t1 + 0.1) S[key] = { a: smooth(seg(t, X.t0, X.t0 + 0.5)) * (1 - smooth(seg(t, X.t1 - 0.5, X.t1))), n: Math.floor(X.text.length * clamp((t - X.type0) / (X.type1 - X.type0))), text: X.text };
    }
    // 片尾
    if (t >= SEC.end.t0 - 0.5) {
      const lt = t - SEC.end.t0;
      S.sum = { a: smooth(seg(lt, 0.6, 1.6)) * (1 - smooth(seg(lt, D0 - 0.9, D0 - 0.2))), k: seg(lt, 0.6, 3.0), k2: smooth(seg(lt, 3.4, 4.4)) };
      S.site = segs.length ? {
        a: smooth(seg(lt, D0 - 0.1, D0 + 0.5)) * (1 - smooth(seg(t, end - 0.9, end - 0.05))),
        ct: capAt(Math.min(lt, FIN)),
        fin: easeInOut(seg(lt, FIN, FIN + 1.0)),
        finA: smooth(seg(lt, FIN + 0.5, FIN + 1.3)) * (1 - smooth(seg(t, end - 0.9, end - 0.05))),
        cursor: lt < FIN ? smooth(seg(lt, D0 + 0.3, D0 + 0.7)) * (1 - smooth(seg(lt, FIN - 0.4, FIN))) : 0,
      } : null;
    }
    S.band = t < OPEN.end ? smooth(seg(t, OPEN.desk0, OPEN.desk1)) : 1;
    S.fade = 1 - smooth(seg(t, 0, 0.5));
    return S;
  }

  progs.sort((a, b) => a.t - b.t);
  const progAt = (t) => {
    let i = -1;
    for (const p of progs) if (t >= p.t) i = p.i;
    const a = progs.length ? smooth(seg(t, progs[0].t, progs[0].t + 0.6)) * (1 - smooth(seg(t, SEC.end.t0 - 0.2, SEC.end.t0 + 0.8))) : 0;
    return { i, a: i < 0 ? 0 : a };
  };
  const dbg = { titleIn, deskCam1, DEC, tokT, j3, jPy, jBash, prev3p, PARSE, CMP };
  return { end, frame, subs, terms, progAt, events, sections, open: OPEN, statLine, bpm: BPM, endSite, acts, DS, toks, T3, kv3, ctxN, S0, dbg, SAY, ANS, FINAL_T, DESK };
}
