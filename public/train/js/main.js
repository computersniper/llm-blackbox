// 训练页面入口：载入两段真实训练记录，调试器式地一层层揭开。
import { loadTiny, loadQwen } from './data.js';
import { wrapTiny, wrapQwen } from './run.js';
import { Timeline } from './timeline.js';
import { Controls, SPEEDS } from './controls.js';
import { Stage } from './stage.js';
import { INSIGHTS, INSIGHT_BY_ID } from './insights.js';
import { Pipeline } from './scenes/pipeline.js';
import { Overview } from './scenes/overview.js';
import { OverviewQ } from './scenes/overviewQ.js';
import { Loop } from './scenes/loop.js';
import { Tokens } from './scenes/tokens.js';
import { Grid } from './scenes/grid.js';
import { LossView } from './scenes/loss.js';
import { LayerView } from './scenes/layer.js';
import { AdamView } from './scenes/adam.js';
import { Background } from '../../js/bg.js';
import { sfx, setSound, soundOn } from '../../js/audio.js';
import { $, esc } from '../../js/ui.js';

const KEY = 'blackbox:train';
function load() { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } }
const saved = load();
const store = { found: new Set(saved.found || []), speed: saved.speed || 1, hinted: !!saved.hinted };
function save() { try { localStorage.setItem(KEY, JSON.stringify({ found: [...store.found], speed: store.speed, sound: soundOn(), hinted: store.hinted })); } catch { /* 忽略 */ } }
if (saved.sound) setSound(true);

let bg, stage, controls;
let runs = {}, tls = {}, run = 'tiny', tl = null;
const ctx = { feat: 0 };
const scenes = {};
let lastView = '', lastDepth = 0, lastStepKey = '';

const app = {
  get tiny() { return runs.tiny; },
  get qwen() { return runs.qwen; },
  get R() { return runs[run]; },
  get ctx() { return ctx; },
  sfx: (n) => sfx[n]?.(),
  discover: (id) => discover(id),
  scrub: (k) => { if (!tl) return; tl.pause(); if (k !== tl.k) { tl.seekCk(k); sfx.tick(); } },
  setFeat: (f) => { ctx.feat = f; controls.update(tl, app.R, ctx); },
  enterRun: (r) => enterRun(r),
  seek: (pred) => { tl.pause(); tl.seek(pred); },
  into: () => into(),
  get tl() { return tl; },
};

/* ---------------------------------------------------------------- 启动 */

async function boot() {
  bg = new Background($('#bg'));
  try {
    const [t, q] = await Promise.all([loadTiny(), loadQwen()]);
    runs = { tiny: wrapTiny(t), qwen: wrapQwen(q) };
  } catch (e) {
    console.error(e);
    $('#loading').innerHTML = `<span style="color:var(--rose)">数据加载失败：${esc(e.message)}</span>`;
    return;
  }
  ctx.tiny = runs.tiny;
  ctx.qwen = runs.qwen;
  for (const r of ['tiny', 'qwen']) {
    tls[r] = new Timeline(runs[r]);
    tls[r].speed = store.speed;
    tls[r].on((type, t) => onTl(type, t));
  }
  tl = tls.tiny;
  controls = new Controls({
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
  scenes.pipeline = new Pipeline(app);
  scenes.overview = new Overview(app);
  scenes.overviewQ = new OverviewQ(app);
  stage = new Stage($('#gl'), $('#cv'), { onFrame, onHover, onPick, onFreeChange: (f) => { $('#btnFollow').hidden = !f; if (f) showHint(false); } });
  if (matchMedia('(max-width: 900px)').matches) { $('#dbg').classList.add('folded'); $('#btnDbgFold').textContent = '+'; }
  updateInsets();
  bindKeys();
  bindChrome();
  renderCodexCount();
  // 一开始停在流水线上
  tl.toPipeline(0);
  renderRuns();
  $('#loading').hidden = true;
  if (!store.hinted && !matchMedia('(max-width: 900px)').matches) { store.hinted = true; save(); setTimeout(() => showHint(true), 2200); }
  if (new URLSearchParams(location.search).has('autoplay')) tl.play();
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
  if (type === 'step' || type === 'ck') { controls.update(tl, app.R, ctx); discoverFor(); }
  if (type === 'play') controls.updatePlay(tl);
  if (type === 'end') controls.updatePlay(tl);
}

/* ---------------------------------------------------------------- 深度 / 两段训练之间切换 */

function enterRun(r) {
  if (!runs[r]) return;
  const from = tl.depth;
  run = r;
  tl.pause();
  const prevDepth = from;
  tl = tls[r];
  tl.speed = store.speed;
  if (prevDepth === 0) {
    if (tl.depth === 0 || tl.depth > 1) { tl.depth = 0; tl.setDepth(1); }
    else controls.update(tl, app.R, ctx);
    tl.play();
  } else {
    // 两段训练之间切换：保持同样的深度（从头开始那一段）
    if (tl.depth !== prevDepth) { if (tl.depth === 0) { tl.depth = 0; } tl.setDepth(prevDepth); }
    else controls.update(tl, app.R, ctx);
  }
  ctx.feat = 0;
  renderRuns();
  sfx.dive();
}

function renderRuns() {
  document.querySelectorAll('#runs button').forEach((b) => { const on = b.dataset.run === run && tl.depth > 0; b.classList.toggle('on', on); b.setAttribute('aria-selected', on); });
}

function into() {
  if (!tl) return;
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
  if (!tl || tl.depth === 0) return;
  sfx.rise();
  stage.exitFree();
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

/* ---------------------------------------------------------------- 每帧 */

function onFrame(dt) {
  if (!tl) return null;
  tl.tick(dt);
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
  return { k: tl.k, p: tl.p, depth: tl.depth, step: tl.step, view, speed: tl.speed, playing: tl.playing, R: app.R, i: tl.i };
}

/* ---------------------------------------------------------------- 悬停 / 点击 */

const tip = $('#tooltip');
function onHover(h, e) {
  const info = h?.info;
  let html = null;
  if (info) html = info.tipAt ? info.tipAt(h.wx, h.wy) : typeof info.tip === 'function' ? info.tip() : info.tip;
  if (!html) { tip.classList.remove('on'); return; }
  tip.innerHTML = html;
  tip.classList.add('on');
  const r = tip.getBoundingClientRect();
  let x = e.clientX + 16, y = e.clientY + 16;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 16;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 12;
  tip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
}

function onPick(info) {
  tip.classList.remove('on');
  if (!info) return;
  if (info.act) { info.act(); controls.update(tl, app.R, ctx); return; }
  if (info.fly) { stage.flyTo(info.fly); sfx.click(); }
}

/* ---------------------------------------------------------------- 知识碎片 */

function discoverFor() {
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
  if (s.ph === 'upd' && s.sub === 'bc') discover(R.adam(s.k, ctx.feat).t === 1 ? 'sign' : 'bias');
  if (s.ph === 'upd' && s.sub === 'dw') discover('decay');
  if (s.ph === 'upd' && s.mi === 'bf16') discover('fp32');
  if (s.ph === 'batch' && s.sub === 'mask') discover('sftmask');
  if (s.ph === 'batch' && s.sub === 'tpl') discover('template');
  if (s.ph === 'check') discover('onestep');
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
  t.innerHTML = `<span class="t-icon" aria-hidden="true">✦</span><span class="t-body"><span class="t-k">发现知识碎片 · ${store.found.size}/${INSIGHTS.length}</span><span class="t-title">${esc(ins.title)}</span></span>`;
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
    : `<article class="cx-card locked"><div class="where">${esc(x.where)}</div><h4>？？？</h4><p>在「${esc(x.where)}」附近找找。</p></article>`)).join('');
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
}
addEventListener('resize', () => updateInsets());

function flashBtn(sel) {
  $(sel).animate([{ transform: 'translateX(0)' }, { transform: 'translateX(-4px)' }, { transform: 'translateX(4px)' }, { transform: 'translateX(0)' }], { duration: 260 });
}

function bindChrome() {
  $('#btnCodex').addEventListener('click', openCodex);
  $('.cx-close').addEventListener('click', closeCodex);
  $('#codex').addEventListener('click', (e) => { if (e.target.id === 'codex') closeCodex(); });
  $('#btnReset').addEventListener('click', () => { if (!confirm('清空已收集的知识碎片？')) return; store.found.clear(); save(); renderCodexCount(); closeCodex(); });
  const sb = $('#btnSound');
  const renderSound = () => { sb.classList.toggle('on', soundOn()); sb.setAttribute('aria-pressed', soundOn()); sb.title = soundOn() ? '关闭声音' : '打开声音'; };
  sb.addEventListener('click', () => { setSound(!soundOn()); renderSound(); save(); sfx.click(); });
  renderSound();
  $('#btnFollow').addEventListener('click', () => { stage.exitFree(); sfx.click(); });
  $('#btnHint').addEventListener('click', () => showHint(!$('#stageHint').classList.contains('on')));
  $('#runs').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    stage.exitFree();
    enterRun(b.dataset.run);
  });
  new ResizeObserver(() => updateInsets()).observe($('#dbg'));
}

function bindKeys() {
  addEventListener('keydown', (e) => {
    if (e.target.closest('input, textarea')) return;
    if (e.key === 'Escape' && $('#codex').classList.contains('on')) return closeCodex();
    if (e.key === ' ') { e.preventDefault(); toggle(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); tl.pause(); tl.next(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); tl.pause(); tl.back(); }
    else if (e.key === '+' || e.key === '=' || e.key === 'ArrowDown') { e.preventDefault(); into(); }
    else if (e.key === '-' || e.key === '_' || e.key === 'ArrowUp' || e.key === 'Escape') { e.preventDefault(); if (stage.free && e.key === 'Escape') stage.exitFree(); else out(); }
    else if (e.key === 'f' || e.key === 'F') stage.exitFree();
    else if (e.key === 'h' || e.key === 'H' || e.key === '?') showHint(!$('#stageHint').classList.contains('on'));
    else if (e.key >= '1' && e.key <= '6') setSpeed(SPEEDS[Number(e.key) - 1]);
  });
}

window.__train = { get tl() { return tl; }, get run() { return run; }, get stage() { return stage; }, app, into, out, enterRun, ctx, setSpeed };
boot();
