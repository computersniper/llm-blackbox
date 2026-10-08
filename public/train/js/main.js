// 训练页面入口：三章真实训练记录，调试器式地一层层揭开。
//   第一章（默认）：玻璃小模型——一台 3D 机器，2,928 个参数每个都是一个方块；看它一批批“吃”数据：进料、前向、反向、更新，
//                  一路拆到一块矩阵、一个参数的一生（数据 ./glass/，3D 舞台 ./glass3d/）；
//   第二章：放大到唐宋诗小模型（664 万参数，从零预训练）——也是一台 3D 机器（./tiny3d/）；第三章：真实的 Qwen3-0.6B 三步监督微调。
// 首屏只取玻璃小模型的几十 KB（配置、曲线、初始化的参数）；另外两章的首屏数据随后在后台取；
// 各层要用的数据分块在进入视图、播放、拖动时按需取，空闲时后台预取（见 data.js）。
import { Loader, loadTiny, loadQwen } from './data.js';
import { wrapTiny } from './run.js';
import { Timeline } from './timeline.js';
import { Controls, SPEEDS } from './controls.js';
import { Stage } from './stage.js';
import { INSIGHTS, INSIGHT_BY_ID } from './insights.js';
import { Pipeline } from './scenes/pipeline.js';
import { Loop } from './scenes/loop.js';
import { Tokens } from './scenes/tokens.js';
import { Grid } from './scenes/grid.js';
import { LossView } from './scenes/loss.js';
import { LayerView } from './scenes/layer.js';
import { AdamView } from './scenes/adam.js';
import { Background } from '../../js/bg.js';
import { sfx, setSound, soundOn } from '../../js/audio.js';
import { $, esc } from '../../js/ui.js';
import { initPanes } from '../../js/resize.js';
import { initLang, isEn, L } from './lang.js';
import { loadGlass, wrapGlass } from './glass/data.js';
import { GlassTimeline } from './glass/timeline.js';
import { GlassStage } from './glass3d/stage.js';
import { wrapTiny3d } from './tiny3d/data.js';
import { TinyTimeline } from './tiny3d/timeline.js';
import { TinyStage, planTiny, discoverTiny } from './tiny3d/stage.js';
import { updateDebugger } from './tiny3d/ui.js';
import { loadQwen3d, QwenTimeline, QwenStage, withQwen3d, planQwen3d, discoverQwen3d } from './qwen3d/index.js';
import { mountSiteActions } from '../../js/visits.js';

mountSiteActions('train');   // 右上角的 GitHub 和全站访问量；后台按页面记一次访问

initLang();

const KEY = 'blackbox:train';
function load() { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } }
const saved = load();
const store = { found: new Set(saved.found || []), speed: saved.speed || 1, hinted: !!saved.hinted };
function save() { try { localStorage.setItem(KEY, JSON.stringify({ found: [...store.found], speed: store.speed, sound: soundOn(), hinted: store.hinted })); } catch { /* 忽略 */ } }
if (saved.sound) setSound(true);

let bg, stage, controls, g3 = null, t3 = null, q3 = null;
let runs = {}, tls = {}, run = 'glass', tl = null;
let pendingRun = null, othersErr = null;
const ctx = { feat: 0, gsel: null };
const scenes = {};
let lastView = '', lastDepth = 0, lastStepKey = '';
const loader = new Loader();
// 数据分块的等待状态：since = 这一轮开始等的时刻（舞台时钟），waiting = 当前画面还有分块没到（给测量脚本看）
const wait = { since: 0, waiting: false };

const app = {
  get glass() { return runs.glass; },
  get tiny() { return runs.tiny; },
  get qwen() { return runs.qwen; },
  get R() { return runs[run]; },
  get ctx() { return ctx; },
  sfx: (n) => sfx[n]?.(),
  discover: (id) => discover(id),
  scrub: (k) => { if (!tl) return; tl.pause(); if (k !== tl.k) { tl.seekCk(k); sfx.tick(); } },
  setFeat: (f) => { ctx.feat = f; updateControls(); },
  enterRun: (r) => enterRun(r),
  seek: (pred) => { tl.pause(); tl.seek(pred); },
  into: () => into(),
  pickParam: (gi, keep) => pickParam(gi, keep),
  pickPos: (i) => pickPos(i),
  get tl() { return tl; },
};
// 当前这一章用的舞台：三章都是 3D（玻璃小模型 g3、唐宋诗小模型 t3、Qwen3 微调 q3）；流水线（D0）仍是 2D 画布
const curStage = () => (run === 'glass' && g3 ? g3 : run === 'tiny' && t3 ? t3 : run === 'qwen' && q3 ? q3 : stage);
const onFreeChange = (f) => { $('#btnFollow').hidden = !f; if (f) showHint(false); };

/* ---------------------------------------------------------------- 启动 */

async function boot() {
  bg = new Background($('#bg'));
  try {
    runs.glass = wrapGlass(await loadGlass(loader));
  } catch (e) {
    console.error(e);
    $('#loading').innerHTML = `<span style="color:var(--rose)">${L('数据加载失败：', 'Failed to load data: ')}${esc(e.message)}</span>`;
    return;
  }
  ctx.glass = runs.glass;
  loader.on((key) => onChunk(key));
  tls.glass = new GlassTimeline(runs.glass);
  tls.glass.speed = store.speed;
  tls.glass.on((type, t) => onTl(type, t));
  tl = tls.glass;
  controls = new (withQwen3d(Controls))({
    into: () => into(),
    out: () => out(),
    prev: () => { tl.pause(); tl.back(); sfx.tick(); },
    next: () => { tl.pause(); tl.next(); sfx.tick(); },
    toggle: () => toggle(),
    speed: (s) => setSpeed(s),
    seek: (i, kind) => { tl.pause(); if (kind === 'ck') tl.seekCk(i); else tl.seekIndex(i); },
    depth: (d) => setDepth(d),
    fold: () => updateInsets(),
  });
  stage = new Stage($('#gl'), $('#cv'), { onFrame, onHover, onPick, onFreeChange });
  try {
    g3 = new GlassStage($('#gl3'), $('#stage'), runs.glass.D, app, { state: frameGlass, onTip: showTip, onFreeChange, scrub: app.scrub });
  } catch (e) {
    console.error(e);
    $('#loading').innerHTML = `<span style="color:var(--rose)">${L('3D 舞台初始化失败（浏览器可能不支持 WebGL）：', 'Could not start the 3D stage (WebGL may be unavailable): ')}${esc(e.message)}</span>`;
    return;
  }
  setGlassMode(true);
  if (matchMedia('(max-width: 900px)').matches) { $('#dbg').classList.add('folded'); $('#btnDbgFold').textContent = '+'; }
  updateInsets();
  bindKeys();
  bindChrome();
  renderCodexCount();
  // 一开始停在玻璃小模型的初始化上（?chapter=tiny / qwen 直接进另外两章）
  updateControls();
  renderRuns();
  $('#loading').hidden = true;
  const qs = new URLSearchParams(location.search);
  loadOthers(qs.get('chapter'));
  startBackground();
  if (!store.hinted && !matchMedia('(max-width: 900px)').matches) { store.hinted = true; save(); setTimeout(() => showHint(true), 2200); }
  if (qs.has('autoplay')) tl.play();
}

// 另外两章（唐宋诗小模型、Qwen3）的首屏数据：玻璃小模型画出来以后再取
async function loadOthers(want) {
  try {
    const [t, q] = await Promise.all([loadTiny(loader), loadQwen(loader)]);
    runs.tiny = wrapTiny3d(wrapTiny(t), loader);
    runs.qwen = await loadQwen3d(loader, q);
  } catch (e) {
    console.error(e);
    othersErr = e;
    renderRuns();
    return;
  }
  ctx.tiny = runs.tiny;
  ctx.qwen = runs.qwen;
  tls.tiny = new TinyTimeline(runs.tiny);
  tls.qwen = new QwenTimeline(runs.qwen);
  for (const r of ['tiny', 'qwen']) {
    tls[r].speed = store.speed;
    tls[r].on((type, t) => onTl(type, t));
  }
  scenes.pipeline = new Pipeline(app);
  renderRuns();
  startBackground();
  const go = pendingRun || (want === 'tiny' || want === 'qwen' ? want : null);
  pendingRun = null;
  if (go) enterRun(go);
}

function sceneFor(view) {
  if (scenes[view]) return scenes[view];
  const key = `${view}:${run}`;
  if (scenes[key]) return scenes[key];
  const R = app.R;
  let sc;
  switch (view) {
    case 'loop': sc = new Loop(app, R); break;
    case 'batch': sc = new Tokens(app, R); break;
    case 'fwd': sc = new Grid(app, R, 'fwd'); break;
    case 'bwd': sc = new Grid(app, R, 'bwd'); break;
    case 'loss': case 'ce': sc = scenes[`loss:${run}`] || new LossView(app, R); scenes[`loss:${run}`] = sc; scenes[`ce:${run}`] = sc; return sc;
    case 'layer': sc = new LayerView(app, R); break;
    case 'adam': case 'bits': sc = scenes[`adam:${run}`] || new AdamView(app, R); scenes[`adam:${run}`] = sc; scenes[`bits:${run}`] = sc; return sc;
    default: sc = scenes.pipeline;
  }
  scenes[key] = sc;
  return sc;
}

function onTl(type, t) {
  if (t !== tl) return;
  if (type === 'step' || type === 'ck') { updateControls(); discoverFor(); }
  if (type === 'play') controls.updatePlay(tl);
  if (type === 'end') controls.updatePlay(tl);
}

/* ---------------------------------------------------------------- 深度 / 两段训练之间切换 */

function enterRun(r) {
  if (r === run) return;
  if (!runs[r]) {
    // 另外两章的数据还在路上：记下来，到了再进
    pendingRun = othersErr ? null : r;
    renderRuns();
    if (othersErr) flashBtn(`#runs [data-run="${r}"]`);
    return;
  }
  const prevDepth = tl.depth, fromGlass = run === 'glass' || run === 'tiny';
  curStage().exitFree();
  run = r;
  setGlassMode(r === 'glass');
  tl.pause();
  tl = tls[r];
  tl.speed = store.speed;
  if (r === 'glass' || r === 'tiny' || r === 'qwen') { updateControls(); if (r === 'qwen' && !fromGlass && prevDepth === 0) tl.play(); }
  else if (fromGlass || prevDepth === 0) {
    // 从玻璃小模型或流水线进来：停在这一段训练的全程（D1）
    if (tl.depth !== 1) { tl.depth = 0; tl.setDepth(1); } else updateControls();
    if (!fromGlass) tl.play();
  } else {
    // 两段训练之间切换：保持同样的深度（从头开始那一段）
    if (tl.depth !== prevDepth) { if (tl.depth === 0) { tl.depth = 0; } tl.setDepth(prevDepth); }
    else updateControls();
  }
  ctx.feat = 0;
  renderRuns();
  sfx.dive();
}

function renderRuns() {
  document.querySelectorAll('#runs button').forEach((b) => {
    const r = b.dataset.run;
    const on = r === run && (r === 'glass' || tl.depth > 0);
    b.classList.toggle('on', on);
    b.setAttribute('aria-selected', on);
    b.classList.toggle('loading', r !== 'glass' && !runs[r] && pendingRun === r);
  });
}

// 玻璃小模型这一章换成 3D 舞台：2D 画布藏起来，引擎开始渲染
function setGlassMode(on) {
  document.body.classList.toggle('g3d', on);
  if (g3) g3.active = on;
  setTinyMode(run === 'tiny');
  setQwenMode(run === 'qwen');
  updateInsets();
}

// Qwen3 微调这一章也是 3D 舞台（./qwen3d/）：第一次进来时才建（多一个 WebGL 上下文）
function setQwenMode(on) {
  if (on && !q3 && runs.qwen) {
    try { q3 = new QwenStage($('#q3'), $('#stage'), runs.qwen, app, { state: frameQwen, onTip: showTip, onFreeChange }); }
    catch (e) { console.error(e); }
  }
  document.body.classList.toggle('q3d', on && !!q3);
  if (q3) q3.active = on;
}

// 唐宋诗小模型这一章的 3D 舞台：第一次进来时才建（另一个 WebGL 画布）
function setTinyMode(on) {
  if (on && !t3) {
    try {
      t3 = new TinyStage($('#gl3t'), $('#stage'), runs.tiny, app, { state: frameTiny, onTip: showTip, onFreeChange, scrub: app.scrub });
      updateInsets();
    } catch (e) { console.error(e); }
  }
  document.body.classList.toggle('t3d', on && !!t3);
  if (t3) t3.active = on;
}

// 点了一根预测柱：跳到“一块权重”那一层的这个位置的损失
function pickPos(i) {
  if (run !== 'tiny') return;
  tl.pause();
  curStage().exitFree();
  sfx.dive();
  tl.jump(4, (b) => b.ph === 'loss' && b.i === i);
}

// 点了某个参数方块：跳到“一个参数”（D5），停在更新这一段的第一小步
function pickParam(gi, keep = false) {
  if (run !== 'glass' || !runs.glass) return;
  const D = runs.glass.D, lc = D.locate(gi);
  if (!lc) return;
  ctx.gsel = gi;
  tl.pause();
  curStage().exitFree();
  const s = tl.step;
  if (keep && tl.depth === 5 && s.t) {
    // 已经在“一个参数”里：换成看这个参数；换了张量就跳到那个张量的同一小步（前向的“它乘了谁”、反向的“梯度从哪来”，或更新那段的同一项）
    if (s.t !== lc.p.name) tl.jump(5, (b) => b.t === lc.p.name && b.ph === s.ph && (s.ph !== 'upd' || b.mi === s.mi));
    else updateControls();
    sfx.click();
    return;
  }
  sfx.dive();
  tl.jump(5, (b) => b.ph === 'upd' && b.t === lc.p.name && b.mi === 'g');
}

function into() {
  if (!tl) return;
  if (run === 'glass' || run === 'tiny' || run === 'qwen') {
    if (!tl.canInto()) { flashBtn('#btnIn'); return; }
    sfx.dive();
    curStage().exitFree();
    if (tl.depth === 0) { tl.setDepth(1); tl.play(); } else tl.into();
    return;
  }
  if (tl.depth === 0) {
    const st = tl.step.st;
    if (st === 2) { flashBtn('#btnIn'); return; }
    stage.exitFree();
    enterRun(st === 0 ? 'tiny' : 'qwen');
    return;
  }
  if (!tl.canInto()) { flashBtn('#btnIn'); return; }
  sfx.dive();
  stage.exitFree();
  tl.into();
}

function out() {
  if (!tl || tl.depth <= (tl.minDepth ?? 0)) { if (tl && run === 'qwen') flashBtn('#btnOut'); return; }
  sfx.rise();
  curStage().exitFree();
  if (run === 'glass' || run === 'tiny' || run === 'qwen') { tl.pause(); tl.setDepth(tl.depth - 1); return; }
  if (tl.depth === 1) {
    tl.pause();
    const st = run === 'tiny' ? 0 : 1;
    tl.toPipeline(st);
    renderRuns();
    return;
  }
  tl.setDepth(tl.depth - 1);
}

function setDepth(d) {
  if (!tl) return;
  if (d === tl.depth) return;
  if (run === 'glass' || run === 'tiny' || run === 'qwen') { d > tl.depth ? sfx.dive() : sfx.rise(); curStage().exitFree(); tl.pause(); tl.setDepth(d); return; }
  if (d === 0) return out1to0();
  d > tl.depth ? sfx.dive() : sfx.rise();
  stage.exitFree();
  tl.setDepth(d);
}
function out1to0() { tl.pause(); stage.exitFree(); sfx.rise(); tl.toPipeline(run === 'tiny' ? 0 : 1); renderRuns(); }

function toggle() {
  if (!tl) return;
  if (tl.done && !tl.playing) {
    tl.done = false;
    if (tl.depth === 0) tl.seekIndex(0); else tl.seekCk(0, false);
  }
  tl.toggle();
}

function setSpeed(s) {
  store.speed = s;
  save();
  for (const t of Object.values(tls)) t.speed = s;
  controls.updatePlay(tl);
}

/* ---------------------------------------------------------------- 数据分块 */

const STEP_VIEWS = new Set(['loop', 'fwd', 'bwd', 'loss', 'ce', 'layer', 'adam', 'bits']);

// 当前画面用到的数据分块：
//   need —— 这一屏离不开它：没到时整屏先画最近一个已到的检查点（底部标注），一个都没有就画载入占位；
//   want —— 画面里的某一块要用：缺了那一块单独占位，或者先拿最近的检查点顶上（D1 的几张卡片）；
//   hold —— 播放时要等它到了才往下走（D1 的嵌入地图在相邻两个检查点之间插值，要用到下一个）；
//   soon —— 很可能马上要用：播放方向上后面几个检查点、相邻检查点、往下一层要用的，立刻排队预取。
function plan() {
  const p = { need: [], want: [], hold: [], soon: [] };
  if (!tl) return p;
  if (run === 'glass') return planGlass(p);
  if (run === 'tiny') return planTiny(tl, runs.tiny, p);
  if (run === 'qwen') return planQwen3d(p, app.R, tl);
  const T = runs.tiny.D;
  if (tl.depth === 0) { p.soon.push(T.key('ck', 0), T.key('ck', T.K - 1), T.key('ck', 1)); return p; }
  const R = app.R, D = R.D, K = R.K, k = tl.k, view = tl.view;
  const ck = (j) => (j >= 0 && j < K ? D.key('ck', j) : null);
  const stp = (j) => (j >= 0 && j < K ? D.key('st', j) : null);
  if (R.kind === 'tiny') {
    if (tl.depth === 1) {
      // 默认的注意力头、ΔW 的色标要用第一个和最后一个检查点
      p.want.push(ck(0), ck(K - 1), ck(k));
      p.hold.push(ck(0), ck(K - 1), ck(k), ck(k + 1));
      const ahead = tl.playing ? Math.min(6, 1 + Math.ceil(tl.speed)) : 1;
      for (let j = 2; j <= ahead; j++) p.soon.push(ck(k + j));
      p.soon.push(ck(k - 1), stp(k));
    } else {
      if (STEP_VIEWS.has(view)) p.need.push(stp(k));
      if (view === 'layer') p.want.push(ck(k));
      if (view === 'adam' || view === 'bits') p.want.push(D.key('feat'));
      p.hold.push(...p.need);
      // ck(k)：D4 层视图的梯度局部、退回 D1；feat：D3 更新的权重轨迹（一步之内的 5 个环节之一，提前取）
      p.soon.push(stp(k), stp(k + 1), stp(k - 1), ck(k), D.key('feat'));
    }
  } else {
    if (tl.depth >= 2) { p.need.push(stp(k)); p.hold.push(stp(k)); }
    for (let j = 0; j < K; j++) p.soon.push(stp(j));
  }
  for (const key of ['need', 'want', 'hold', 'soon']) p[key] = p[key].filter(Boolean);
  return p;
}

// 玻璃小模型：w = 权重块（每 9 帧一块），f = 每帧一块（梯度、Δw、激活），exact = 选中参数的 float32 原值。
// 3D 机器的训练全程（D1）每一拍都走一遍前向 / 反向 / 更新，所以 D1 起就要 w 和 f 两种
function planGlass(p) {
  const D = runs.glass.D, K = D.NF, k = tl.k;
  const w = (j) => (j >= 0 && j < K ? D.key('w', j) : null), f = (j) => (j >= 0 && j < K ? D.key('f', j) : null);
  const G = D.meta.chunks.group;
  if (tl.depth === 0) p.soon.push(w(0), f(0), w(K - 1));
  else if (tl.depth === 1) {
    // 一拍末尾跳过的步数要插值到下一帧
    p.want.push(w(k), f(k));
    p.hold.push(w(k), f(k), w(k + 1));
    if (tl.playing) for (let j = 1; j <= Math.min(6, 1 + Math.ceil(tl.speed)); j++) p.soon.push(f(k + j));
    if (tl.playing) p.soon.push(w(k + G));
    p.soon.push(w(K - 1), f(k + 1));
  } else {
    p.need.push(w(k), f(k));
    p.hold.push(w(k), f(k));
    p.soon.push(f(k + 1), f(k - 1), w(k + 1), w(K - 1));
    if (tl.depth === 5) {
      // “一个参数的一生”要用全部帧：低优先级全部排队（曲线随到随画）
      p.want.push(D.key('exact'));
      for (let j = 0; j < K; j += G) p.soon.push(w(j));
      for (let j = 0; j < K; j++) p.soon.push(f(j));
    }
  }
  for (const key of ['need', 'want', 'hold', 'soon']) p[key] = [...new Set(p[key].filter(Boolean))];
  return p;
}

function request(p) {
  for (const key of p.need) loader.want(key, 3);
  for (const key of p.want) loader.want(key, 3);
  for (const key of p.hold) loader.want(key, 2);
  for (const key of p.soon) loader.want(key, 1);
  loader.pump();
}

// 调试器里的讲解 / 变量要用的分块（D1 小模型的讲解要用这个检查点的逐字概率）
function controlsNeed() {
  if (!tl || tl.depth === 0) return [];
  const p = plan();
  if (run === 'glass') return tl.depth === 1 ? [runs.glass.D.key('w', tl.k)] : p.need;
  if (run === 'tiny' || run === 'qwen') return p.need;
  return tl.depth === 1 && app.R.kind === 'tiny' ? [...p.need, app.R.D.key('ck', tl.k)] : p.need;
}

function updateControls() {
  if (!controls || !tl) return;
  const need = controlsNeed();
  const ready = need.every((key) => loader.has(key));
  const err = ready ? null : need.map((key) => loader.error(key)).find(Boolean) || null;
  if (run === 'tiny') updateDebugger(controls, tl, runs.tiny, ctx, ready, err);
  else controls.update(tl, app.R, ctx, ready, err);
}

// 一块数据到了（或者失败了）：如果当前画面在等它，刷新调试器
function onChunk(key) {
  if (!tl) return;
  const p = plan();
  if (p.need.includes(key) || p.want.includes(key)) { updateControls(); discoverFor(); }
}

// 首屏画好以后开始后台预取：先 D1 的检查点（按播放顺序），再一步之内的、Qwen3 的、权重轨迹。
// 等首屏的字体先下完（最多等 3 秒），别跟它抢带宽；省流量模式 / 2G 网络不预取，只按需取
function startBackground() {
  // 先玻璃小模型（按播放顺序：每 9 帧的权重块后面跟着这 9 帧的每帧块，最后 exact），另外两章的数据到了再排在后面
  const G = runs.glass.D, order = [];
  for (let k = 0; k < G.NF; k++) { if (k % G.meta.chunks.group === 0) order.push(G.key('w', k)); order.push(G.key('f', k)); }
  order.push(G.key('exact'));
  const fontsReady = document.fonts?.ready ?? Promise.resolve();
  if (!runs.tiny) { Promise.race([fontsReady, new Promise((r) => setTimeout(r, 3000))]).then(() => loader.startBackground(order)); return; }
  const T = runs.tiny.D, Q = runs.qwen.D, K = T.K;
  order.push(T.key('ck', 0), T.key('ck', K - 1));
  for (let k = 1; k < K - 1; k++) order.push(T.key('ck', k));
  for (let k = 0; k < K; k++) order.push(T.key('st', k));
  for (let k = 0; k < Q.K; k++) order.push(Q.key('st', k));
  order.push(T.key('feat'));
  Promise.race([fontsReady, new Promise((r) => setTimeout(r, 3000))]).then(() => loader.startBackground(order));
}

// 离 k 最近的、一步之内的数据已经到了的检查点
function nearestStep(R, k) {
  for (let d = 1; d < R.K; d++) {
    if (k - d >= 0 && R.stepReady(k - d)) return k - d;
    if (k + d < R.K && R.stepReady(k + d)) return k + d;
  }
  return -1;
}

/* ---------------------------------------------------------------- 每帧 */

function onFrame(dt, clock) {
  if (!tl) return null;
  if (run === 'glass' || run === 'tiny' || run === 'qwen') { lastView = ''; return null; }   // 三章都由 3D 舞台（frameGlass / frameTiny / frameQwen）驱动
  // 播放中，这一屏（或下一个检查点）要等的分块还在路上：原地停一下，到了再走
  if (tl.playing && plan().hold.some((key) => loader.pending(key))) dt = 0;
  tl.tick(dt);
  const p = plan();
  request(p);
  const needMiss = p.need.filter((key) => !loader.has(key));
  const wantMiss = p.want.filter((key) => !loader.has(key));
  const waiting = needMiss.length + wantMiss.length > 0;
  if (waiting && !wait.waiting) wait.since = clock;
  wait.waiting = waiting;
  const err = [...needMiss, ...wantMiss].map((key) => loader.error(key)).find(Boolean) || null;
  const view = tl.view;
  const sc = sceneFor(view);
  if (view !== lastView || sc !== stage.scene) {
    const dir = tl.depth > lastDepth ? 1 : tl.depth < lastDepth ? -1 : 0;
    stage.setScene(sc, dir || (lastView ? 1 : 0));
    if (dir) bg.warp(dir > 0 ? 1 : -1);
    lastView = view;
    lastDepth = tl.depth;
  }
  lastDepth = tl.depth;
  const R = app.R;
  const st = { k: tl.k, p: tl.p, depth: tl.depth, step: tl.step, view, speed: tl.speed, playing: tl.playing, R, i: tl.i, wait: { since: wait.since, err } };
  // 整屏要用的分块还没到：先画最近一个已到的检查点的同一步，并在底部标注；一个都没有就画占位
  if (needMiss.length) {
    const kd = err ? -1 : nearestStep(R, tl.k);
    if (kd < 0) st.blocked = true;
    else {
      st.k = kd;
      st.step = { ...tl.step, k: kd };
      st.stale = { since: wait.since, label: L(`第 ${R.stepNo(tl.k)} 步的数据还在路上，先显示第 ${R.stepNo(kd)} 步`, `Data for step ${R.stepNo(tl.k)} is still on its way — showing step ${R.stepNo(kd)} for now`) };
    }
  }
  return st;
}

// 3D 舞台每帧调用：推进时间轴、按需取数据，返回给机器的当前步骤
function frameGlass(dt, clock) {
  if (!tl || run !== 'glass') return null;
  if (tl.playing && plan().hold.some((key) => loader.pending(key))) dt = 0;
  tl.tick(dt);
  const p = plan();
  request(p);
  const needMiss = p.need.filter((key) => !loader.has(key)), wantMiss = p.want.filter((key) => !loader.has(key));
  const waiting = needMiss.length + wantMiss.length > 0;
  if (waiting && !wait.waiting) wait.since = clock;
  wait.waiting = waiting;
  const err = [...needMiss, ...wantMiss].map((key) => loader.error(key)).find(Boolean) || null;
  waitPill(waiting && clock - wait.since > 0.25, err);
  return { k: tl.k, p: tl.p, depth: tl.depth, step: tl.step, view: tl.view, speed: tl.speed, playing: tl.playing, R: app.R, i: tl.i, wait: { since: wait.since, err } };
}

// 唐宋诗小模型的 3D 舞台每帧调用：同上（缺数据的提示由舞台自己画）
function frameTiny(dt, clock) {
  if (!tl || run !== 'tiny') return null;
  if (tl.playing && plan().hold.some((key) => loader.pending(key))) dt = 0;
  tl.tick(dt);
  const p = plan();
  request(p);
  const miss = [...p.need, ...p.want].filter((key) => !loader.has(key));
  const waiting = miss.length > 0;
  if (waiting && !wait.waiting) wait.since = clock;
  wait.waiting = waiting;
  const err = miss.map((key) => loader.error(key)).find(Boolean) || null;
  return { k: tl.k, p: tl.p, depth: tl.depth, step: tl.step, view: tl.view, speed: tl.speed, playing: tl.playing, R: app.R, i: tl.i, wait: { since: wait.since, err, waiting } };
}

// Qwen3 微调的 3D 舞台每帧调用（同上）
function frameQwen(dt, clock) {
  if (!tl || run !== 'qwen') return null;
  if (tl.playing && plan().hold.some((key) => loader.pending(key))) dt = 0;
  tl.tick(dt);
  const p = plan();
  request(p);
  const miss = [...p.need, ...p.want].filter((key) => !loader.has(key));
  if (miss.length && !wait.waiting) wait.since = clock;
  wait.waiting = miss.length > 0;
  const err = miss.map((key) => loader.error(key)).find(Boolean) || null;
  waitPill(wait.waiting && clock - wait.since > 0.25, err);
  return { k: tl.k, p: tl.p, depth: tl.depth, step: tl.step, view: tl.view, speed: tl.speed, playing: tl.playing, R: app.R, i: tl.i, wait: { since: wait.since, err } };
}

// 3D 舞台缺数据时顶部的小提示（机器先按已有的数据画）
let pillEl = null;
function waitPill(on, err) {
  if (!pillEl) {
    pillEl = document.createElement('div');
    pillEl.className = 'wait-pill';
    pillEl.setAttribute('role', 'status');
    $('#stage').append(pillEl);
  }
  const html = err ? esc(err.unsupported ? err.message : L('这一步的数据没有载入成功，稍后会自动重试…', 'This step’s data failed to load; retrying automatically…')) : `<span class="spin"></span>${L('正在载入这一步的真实记录…', 'Loading the real record of this step…')}`;
  if (pillEl._h !== html) { pillEl.innerHTML = html; pillEl._h = html; }
  pillEl.classList.toggle('on', !!on && (run === 'glass' || run === 'qwen'));
}

/* ---------------------------------------------------------------- 悬停 / 点击 */

const tip = $('#tooltip');
function onHover(h, e) {
  const info = h?.info;
  let html = null;
  if (info) html = info.tipAt ? info.tipAt(h.wx, h.wy) : typeof info.tip === 'function' ? info.tip() : info.tip;
  showTip(html, e);
}
function showTip(html, e) {
  if (!html || !e) { tip.classList.remove('on'); return; }
  tip.innerHTML = html;
  tip.classList.add('on');
  const r = tip.getBoundingClientRect();
  let x = e.clientX + 16, y = e.clientY + 16;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 16;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 12;
  tip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
}

function onPick(info, e, h) {
  tip.classList.remove('on');
  if (!info) return;
  if (info.act) { info.act(h); updateControls(); return; }
  if (info.fly) { stage.flyTo(info.fly); sfx.click(); }
}

/* ---------------------------------------------------------------- 知识碎片 */

function discoverFor() {
  if (run === 'glass') return discoverGlass();
  if (run === 'tiny') return discoverTiny(tl, runs.tiny, discover);
  if (run === 'qwen') return discoverQwen3d(tl, discover);
  const s = tl.step, d = tl.depth, R = app.R;
  if (d === 0 && s.st === 2) discover('dpo');
  if (d < 1 || !R) return;
  if (R.kind === 'tiny') {
    const t = R.D.meta.ckpts[s.k].t;
    if (d === 1 && t >= 46 && t < 400) discover('format');
    if (d === 1 && t >= 200 && t < 260) discover('warmup');
    if (d === 1 && s.k === R.K - 1) discover('cosine');
  } else if (d === 1 && s.k === R.K - 1) discover('memorize');
  if (s.ph === 'fwd' && s.sub && d >= 3) discover('teacher');
  if (s.ph === 'fwd' && s.sub && s.i > 0 && d >= 3) discover('lens');
  if (s.ph === 'loss' && s.mi === 'log') discover('ce');
  if (s.ph === 'bwd' && s.sub) discover('backprop');
  if (s.ph === 'bwd' && s.mi) discover('outer');
  if (s.ph === 'upd' && s.sub === 'clip') discover('clip');
  if (s.ph === 'upd' && s.sub === 'bc' && R.stepReady(s.k)) discover(R.adam(s.k, ctx.feat).t === 1 ? 'sign' : 'bias');
  if (s.ph === 'upd' && s.sub === 'dw') discover('decay');
  if (s.ph === 'upd' && s.mi === 'bf16') discover('fp32');
  if (s.ph === 'batch' && s.sub === 'mask') discover('sftmask');
  if (s.ph === 'batch' && s.sub === 'tpl') discover('template');
  if (s.ph === 'check') discover('onestep');
}

function discoverGlass() {
  const s = tl.step, d = tl.depth, D = runs.glass.D;
  if (d === 0) { if (s.sub === 'zero') discover('g-zero'); if (s.sub === 'big') discover('g-big'); return; }
  const t = D.FR[s.k];
  if (d === 1 && t >= 60) discover('g-lookback');
  if (d === 1 && t >= 120) discover('g-tied');
  if (s.ph === 'batch' && s.sub === 'shift') discover('teacher');
  if (s.ph === 'loss' && s.sub === 'pos') discover('ce');
  if (s.ph === 'bwd' && d >= 2) discover('backprop');
  if (s.ph === 'bwd' && s.sub === 'qkv' && t < 5) discover('g-qk');
  if (s.mi === 'chain') { discover('g-votes'); discover('outer'); }
  if (s.ph === 'upd' && s.sub === 'clip') discover('clip');
  if (s.ph === 'upd' && t === 0 && (s.mi === 'dw' || s.mi === 'write' || (d === 3 && s.sub === 't'))) discover('g-sign');
  if (s.mi === 'bc' && t > 0) discover('bias');
  if (s.mi === 'dw' && t > 0) discover('decay');
}

function discover(id) {
  if (store.found.has(id) || !INSIGHT_BY_ID.has(id)) return;
  store.found.add(id);
  save();
  sfx.discover();
  const ins = INSIGHT_BY_ID.get(id);
  const box = $('#toasts');
  const t = document.createElement('button');
  t.type = 'button';
  t.className = 'toast';
  t.innerHTML = `<span class="t-icon" aria-hidden="true">✦</span><span class="t-body"><span class="t-k">${L('发现知识碎片', 'INSIGHT FOUND')} · ${store.found.size}/${INSIGHTS.length}</span><span class="t-title">${esc(ins.title)}</span></span>`;
  t.addEventListener('click', () => { openCodex(); t.remove(); });
  box.prepend(t);
  const maxToasts = matchMedia('(max-width: 900px)').matches ? 1 : 3;
  while (box.children.length > maxToasts) box.lastElementChild.remove();
  setTimeout(() => t.classList.add('out'), 5000);
  setTimeout(() => t.remove(), 5700);
  renderCodexCount();
}

function renderCodexCount() {
  $('#codexCount').textContent = `${store.found.size}/${INSIGHTS.length}`;
  $('#btnCodex').style.setProperty('--p', store.found.size / INSIGHTS.length);
}

function openCodex() {
  const c = $('#codex');
  $('.cx-grid', c).innerHTML = INSIGHTS.map((x) => (store.found.has(x.id)
    ? `<article class="cx-card found"><div class="where">${esc(x.where)}</div><h4>✦ ${esc(x.title)}</h4><p>${esc(x.text)}</p></article>`
    : `<article class="cx-card locked"><div class="where">${esc(x.where)}</div><h4>${L('？？？', '???')}</h4><p>${L(`在「${esc(x.where)}」附近找找。`, `Look around “${esc(x.where)}”.`)}</p></article>`)).join('');
  $('.cx-progress b', c).textContent = `${store.found.size} / ${INSIGHTS.length}`;
  $('.cx-bar i', c).style.width = `${(store.found.size / INSIGHTS.length) * 100}%`;
  c.classList.add('on');
  c.setAttribute('aria-hidden', 'false');
}
function closeCodex() { const c = $('#codex'); c.classList.remove('on'); c.setAttribute('aria-hidden', 'true'); }

/* ---------------------------------------------------------------- 杂项 */

let hintTimer = 0;
function showHint(on = true) {
  const h = $('#stageHint');
  clearTimeout(hintTimer);
  h.classList.toggle('on', on);
  if (on) hintTimer = setTimeout(() => h.classList.remove('on'), 9000);
}

function updateInsets() {
  if (!stage) return;
  const small = matchMedia('(max-width: 900px)').matches;
  const dbg = $('#dbg');
  const folded = dbg.classList.contains('folded');
  const ctlH = $('#ctl').offsetHeight + 22;
  stage.setInsets(small || folded ? 0 : dbg.offsetWidth + 28, small ? 100 + dbg.offsetHeight + 8 : ctlH, 56);
  g3?.setInsets(small || folded ? 0 : dbg.offsetWidth + 28, small ? 100 + dbg.offsetHeight + 8 : ctlH);
  t3?.setInsets(small || folded ? 0 : dbg.offsetWidth + 28, small ? 100 + dbg.offsetHeight + 8 : ctlH);
  q3?.setInsets(small || folded ? 0 : dbg.offsetWidth + 28, small ? 100 + dbg.offsetHeight + 8 : ctlH);
}
addEventListener('resize', () => updateInsets());

function flashBtn(sel) {
  $(sel).animate([{ transform: 'translateX(0)' }, { transform: 'translateX(-4px)' }, { transform: 'translateX(4px)' }, { transform: 'translateX(0)' }], { duration: 260 });
}

function bindChrome() {
  $('#btnCodex').addEventListener('click', openCodex);
  $('.cx-close').addEventListener('click', closeCodex);
  $('#codex').addEventListener('click', (e) => { if (e.target.id === 'codex') closeCodex(); });
  $('#btnReset').addEventListener('click', () => { if (!confirm(L('清空已收集的知识碎片？', 'Clear all the insights you have collected?'))) return; store.found.clear(); save(); renderCodexCount(); closeCodex(); });
  const sb = $('#btnSound');
  const renderSound = () => { sb.classList.toggle('on', soundOn()); sb.setAttribute('aria-pressed', soundOn()); sb.title = soundOn() ? L('关闭声音', 'Turn sound off') : L('打开声音', 'Turn sound on'); };
  sb.addEventListener('click', () => { setSound(!soundOn()); renderSound(); save(); sfx.click(); });
  renderSound();
  $('#btnFollow').addEventListener('click', () => { curStage().exitFree(); sfx.click(); });
  $('#btnHint').addEventListener('click', () => showHint(!$('#stageHint').classList.contains('on')));
  $('#runs').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    enterRun(b.dataset.run);
  });
  new ResizeObserver(() => updateInsets()).observe($('#dbg'));
  initPanes('train', { right: { el: '#dbg', v: '--dbg-w', name: '调试器' }, reserve: 640 });
  if (isEn) enGrip();
}

// 调试器左边的拖动手柄：共用的 ../../js/resize.js 只写了中文的提示和读数，英文模式下在这里换掉
function enGrip() {
  const gr = $('#dbg .pane-grip');
  if (!gr) return;
  gr.setAttribute('aria-label', 'Resize the debugger');
  gr.title = 'Drag to resize, double-click to reset';
  const fix = () => { const v = gr.getAttribute('aria-valuetext'); if (v && v.includes('像素')) gr.setAttribute('aria-valuetext', v.replace(' 像素', ' px')); };
  fix();
  new MutationObserver(fix).observe(gr, { attributes: true, attributeFilter: ['aria-valuetext'] });
}

function bindKeys() {
  addEventListener('keydown', (e) => {
    if (e.target.closest('input, textarea')) return;
    if (e.key === 'Escape' && $('#codex').classList.contains('on')) return closeCodex();
    if (e.key === ' ') { e.preventDefault(); toggle(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); tl.pause(); tl.next(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); tl.pause(); tl.back(); }
    else if (e.key === '+' || e.key === '=' || e.key === 'ArrowDown') { e.preventDefault(); into(); }
    else if (e.key === '-' || e.key === '_' || e.key === 'ArrowUp' || e.key === 'Escape') { e.preventDefault(); if (curStage().free && e.key === 'Escape') curStage().exitFree(); else out(); }
    else if (e.key === 'f' || e.key === 'F') curStage().exitFree();
    else if (e.key === 'h' || e.key === 'H' || e.key === '?') showHint(!$('#stageHint').classList.contains('on'));
    else if (e.key >= '1' && e.key <= '6') setSpeed(SPEEDS[Number(e.key) - 1]);
  });
}

window.__train = { get tl() { return tl; }, get run() { return run; }, get stage() { return stage; }, get g3() { return g3; }, get t3() { return t3; }, get q3() { return q3; }, get waiting() { return wait.waiting; }, loader, app, into, out, enterRun, ctx, setSpeed };
boot();
