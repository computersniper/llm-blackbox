// 世界模型页入口：载入真实权重 → 真实世界和梦并排跑（D1）→ ＋ 一层层揭开（循环 / V / M / 一次乘加）。
// 第二章「大模型里的世界地图」是另一个模块（./probe/probe.js），切过去时才载入；文件还没有时显示“即将上线”。
import { loadWorld } from './data.js';
import { Sim } from './sim.js';
import { Timeline, MAC_OPS } from './timeline.js';
import { Controls, SPEEDS } from './controls.js';
import { Stage } from './stage.js';
import { PlayView } from './scenes/play.js';
import { LoopView } from './scenes/loop.js';
import { VaeView } from './scenes/vae.js';
import { MemView } from './scenes/mem.js';
import { MacView } from './scenes/macview.js';
import { buildMac, defaultSel, renderBoard } from './mac.js';
import { INSIGHTS, INSIGHT_BY_ID } from './insights.js';
import { Background } from '../../js/bg.js';
import { sfx, setSound, soundOn } from '../../js/audio.js';
import { $, $$, esc } from '../../js/ui.js';
import { initPanes } from '../../js/resize.js';

const KEY = 'blackbox:world:v1';
function load() { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } }
const saved = load();
const store = { found: new Set(saved.found || []), speed: saved.speed || 1, hinted: !!saved.hinted };
function save() { try { localStorage.setItem(KEY, JSON.stringify({ found: [...store.found], speed: store.speed, sound: soundOn(), hinted: store.hinted })); } catch { /* 忽略 */ } }
if (saved.sound) setSound(true);

const qs = new URLSearchParams(location.search);
const small = () => matchMedia('(max-width: 900px)').matches;
let bg, stage, controls, sim, tl, meta, model;
let chapter = 'wm';
const scenes = {};
let lastView = '', lastDepth = 1;
const input = { key: 0, pad: 0 };       // 按着的方向：0 没按，1 左，2 右
let ctrlFrames = 0, playFrames = 0;
let mac = null, macKey = '';

const app = {
  get sim() { return sim; },
  get meta() { return meta; },
  get model() { return model; },
  get tl() { return tl; },
  get mac() { return mac; },
  sel: {},
  select(op, q) { app.sel[op] = q; mac = null; macKey = ''; sfx.click(); if (tl) refresh(); },
  dragStart() { app.wasPlaying = tl.playing; tl.pause(); },
  dragEnd() { if (app.wasPlaying) tl.play(); },
};

/* ---------------------------------------------------------------- 启动 */

async function boot() {
  bg = new Background($('#bg'));
  try {
    ({ meta, model } = await loadWorld());
  } catch (e) {
    console.error(e);
    $('#loading').innerHTML = `<span style="color:var(--rose)">模型载入失败：${esc(e.message)}</span>`;
    return;
  }
  sim = new Sim(model, { seed: Number(qs.get('seed')) || 20261002 });
  sim.keys = () => input.key || input.pad;
  sim.on((type, F) => {
    if (type === 'crash') sfx.rise();
    if (type === 'dreamCrash') discover('done');
    if (type === 'newgame') sfx.land();
  });
  tl = new Timeline(sim);
  tl.speed = store.speed;
  tl.on((type) => {
    if (type === 'step') { refresh(); discoverFor(); }
    if (type === 'frame') onFrameAdvance();
    if (type === 'play') controls.updatePlay(tl);
  });
  controls = new Controls({
    into: () => into(),
    out: () => out(),
    prev: () => { if (chapter !== 'wm') return; tl.pause(); tl.back(); sfx.tick(); },
    next: () => { if (chapter !== 'wm') return; tl.pause(); tl.next(); sfx.tick(); },
    toggle: () => toggle(),
    speed: (s) => setSpeed(s),
    seek: (i) => { tl.pause(); if (tl.depth === 1) { for (let k = 0; k < -i; k++) tl.back(); } else tl.seekIndex(i); },
    depth: (d) => setDepth(d),
    fold: () => updateInsets(),
  });
  scenes.play = new PlayView(app);
  scenes.loop = new LoopView(app);
  scenes.vae = new VaeView(app);
  scenes.mem = new MemView(app);
  scenes.mac = new MacView(app);
  stage = new Stage($('#gl'), $('#cv'), { onFrame, onHover, onPick, onFreeChange: (f) => { $('#btnFollow').hidden = !f; } });
  if (small()) controls.setFolded(true);
  bindKeys();
  bindChrome();
  bindPlayUi();
  renderCodexCount();
  $('#cxFoot').textContent = `模型：V（卷积 VAE，${meta.params.V.toLocaleString()} 个参数）+ M（LSTM 256 + 5 个高斯的混合密度输出，${meta.params.M.toLocaleString()} 个参数）${meta.params.C ? ` + C（线性，${meta.params.C} 个参数）` : ''}，结构照 Ha & Schmidhuber 2018《World Models》。在本地 ${meta.train.gpu || 'GPU'} 上用 ${meta.train.collect.frames.toLocaleString()} 帧真实游戏画面训练；网页下载的是它的真实权重（float16，约 ${(meta.params.total * 2 / 1048576).toFixed(1)} MB），每一帧的梦都是浏览器现场算出来的。`;
  updateInsets();
  refresh();
  $('#loading').hidden = true;
  if (!store.hinted && !small()) { store.hinted = true; save(); setTimeout(() => showHint(true), 1200); }
  if (qs.has('autoplay') || !qs.has('paused')) tl.play();
  if (qs.get('chapter') === 'probe') setChapter('probe');
}

/* ---------------------------------------------------------------- 调试器 */

let lastRefresh = 0;
function refresh(force = true) {
  if (!controls || chapter !== 'wm') return;
  // D1 播放时每秒十几帧：调试器的文字不用每帧都重写
  const now = performance.now();
  if (!force && now - lastRefresh < 180) return;
  lastRefresh = now;
  const s = tl.step;
  if (s.mi) ensureMac();
  const F = sim.cur;
  const ctx = { sim, meta, F, rec: F.rec, next: sim.next, det: tl.depth >= 3 && F.rec?.kind === 'step' ? sim.detail(F) : null, mac: s.mi ? mac : null };
  controls.update(tl, ctx);
  document.body.classList.toggle('w-play', tl.depth === 1);
  updateBoard(true);
}

function onFrameAdvance() {
  playFrames++;
  const F = sim.cur;
  if (F.rec?.by === 'ctrl' || sim.auto === 'ctrl') ctrlFrames++;
  if (tl.depth === 1) refresh(false);
  if (F.dreamAge >= 20) discover('dream');
  if (F.dreamAge >= 8 && sim.hist[sim.idx - 1]?.rec?.by === 'key') discover('causal');
  if (ctrlFrames >= 60) discover('ctrl');
  if (playFrames >= 240) discover('genie');
}

// 一次乘加：当前步骤的那个算子、选中的（或默认的）那个输出
function ensureMac() {
  const s = tl.step, F = sim.cur;
  if (!s.mi || !MAC_OPS.has(s.op) || F.rec?.kind !== 'step') { mac = null; return null; }
  const sel = app.sel[s.op] || defaultSel(s.op, sim, model, meta);
  const k = `${F.n}:${s.op}:${JSON.stringify(sel)}:${F.rec.z[0]}`;
  if (k !== macKey) { sim._mdnDim = app.sel.mdnDim; mac = buildMac(s.op, sel, sim, model, meta); macKey = k; }
  return mac;
}

let boardKey = '';
function updateBoard(force = false) {
  const b = $('#board');
  const s = tl.step;
  const on = chapter === 'wm' && !!s.mi && !!mac;
  if (b.hidden === on) { b.hidden = !on; document.body.classList.toggle('has-board', on); updateInsets(); }
  if (!on) return;
  const top = Math.min(12, mac.n);
  const rows = s.mi === 'mul' ? Math.ceil(top * Math.min(1, tl.p * 1.15 + 0.05)) : 0;
  const k = `${macKey}:${s.mi}:${rows}`;
  if (!force && k === boardKey) return;
  boardKey = k;
  const folded = b.classList.contains('folded');
  renderBoard(b, mac, s.mi, tl.p);
  b.querySelector('.bd-fold').textContent = folded ? '+' : '–';
}

/* ---------------------------------------------------------------- 深度 */

function into() {
  if (chapter === 'probe') return probeLayer(1);
  if (!tl.canInto()) { flashBtn('#btnIn'); return; }
  sfx.dive();
  stage.exitFree();
  tl.into();
  if (tl.depth === 2) discover('vmc');
}
function out() {
  if (chapter === 'probe') return probeLayer(-1);
  if (tl.depth <= 1) return;
  sfx.rise();
  stage.exitFree();
  tl.out();
}
function setDepth(d) {
  if (chapter !== 'wm') { if (d === 0) setChapter('wm'); return; }
  if (d === tl.depth) return;
  d > tl.depth ? sfx.dive() : sfx.rise();
  stage.exitFree();
  tl.setDepth(d);
}
function toggle() {
  if (chapter !== 'wm') return;
  tl.toggle();
}
function setSpeed(s) {
  store.speed = s;
  save();
  tl.speed = s;
  controls.updatePlay(tl);
}

/* ---------------------------------------------------------------- 每帧 */

function onFrame(dt) {
  if (!tl || chapter !== 'wm') return null;
  tl.tick(dt, 14);
  if (tl.step.mi) { ensureMac(); updateBoard(); }
  const view = tl.view;
  const sc = scenes[view];
  if (view !== lastView || sc !== stage.scene) {
    const dir = tl.depth > lastDepth ? 1 : tl.depth < lastDepth ? -1 : 0;
    stage.setScene(sc, dir || (lastView ? 1 : 0));
    if (dir) bg.warp?.(dir > 0 ? 1 : -1);
    lastView = view;
  }
  lastDepth = tl.depth;
  return { step: tl.step, p: tl.p, depth: tl.depth, view, playing: tl.playing };
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
function onPick(info, e, h) {
  tip.classList.remove('on');
  if (!info) return;
  if (info.clickAt) { info.clickAt(h.wx, h.wy); return; }
  if (info.act) { info.act(); sfx.click(); refresh(); return; }
  if (info.click && tl.canInto()) into();
}

/* ---------------------------------------------------------------- D1 的开关 */

function bindPlayUi() {
  const renderUi = () => {
    $$('#segEye button').forEach((b) => { const on = b.dataset.eye === sim.mode; b.classList.toggle('on', on); b.setAttribute('aria-checked', on); });
    $$('#segAuto button').forEach((b) => { const on = b.dataset.auto === sim.auto; b.classList.toggle('on', on); b.setAttribute('aria-checked', on); });
    $('#tauV').textContent = sim.tau.toFixed(2);
    $('#tau').value = sim.tau;
    const c = $('#segAuto [data-auto="ctrl"]');
    c.hidden = !meta.params.C;
  };
  sim.on((type) => { if (type === 'mode') renderUi(); });
  $('#segEye').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; setEye(b.dataset.eye); });
  $('#segAuto').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    sim.setAuto(b.dataset.auto);
    renderUi();
    sfx.click();
    if (b.dataset.auto !== 'off' && !tl.playing && tl.depth === 1) tl.play();
    refresh();
  });
  $('#tau').addEventListener('input', (e) => { sim.setTau(Number(e.target.value)); renderUi(); discover('tau'); });
  $('#btnSync').addEventListener('click', () => { sim.sync(); sfx.click(); refresh(); });
  $('#btnNew').addEventListener('click', () => { sim.newGame(); tl.rebuild(false); refresh(); });
  // 触屏方向键：按住转向；停着的时候按一下就开始跑
  for (const b of $$('#pad button')) {
    const a = Number(b.dataset.a);
    const up = (e) => { if (input.pad === a) input.pad = 0; b.classList.remove('on'); if (e && b.hasPointerCapture?.(e.pointerId)) b.releasePointerCapture(e.pointerId); };
    b.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      b.setPointerCapture(e.pointerId);
      input.pad = a;
      b.classList.add('on');
      if (chapter === 'wm' && tl.depth === 1 && !tl.playing) tl.play();
    });
    b.addEventListener('pointerup', up);
    b.addEventListener('pointercancel', up);
    b.addEventListener('lostpointercapture', () => up());
    b.addEventListener('contextmenu', (e) => e.preventDefault());
  }
  renderUi();
  app.renderUi = renderUi;
}

function setEye(m) {
  sim.setMode(m);
  sfx.click();
  app.renderUi?.();
  refresh();
}

/* ---------------------------------------------------------------- 第二章：大模型里的世界地图 */

let probeApi = null, probeTried = false, probeL = 1;
async function setChapter(ch) {
  if (ch === chapter) return;
  chapter = ch;
  document.body.classList.toggle('w-probe', ch === 'probe');
  $$('#chap button').forEach((b) => { const on = b.dataset.ch === ch; b.classList.toggle('on', on); b.setAttribute('aria-selected', on); });
  stage.exitFree();
  tip.classList.remove('on');
  $('#code').hidden = ch === 'probe';
  $('#dbgSub').textContent = ch === 'probe' ? 'probe()' : 'world_model()';
  if (ch === 'wm') {
    $('#probeHost').hidden = true;
    lastView = '';
    refresh();
    updateInsets();
    return;
  }
  tl.pause();
  $('#board').hidden = true;
  document.body.classList.remove('has-board');
  $('#probeHost').hidden = false;
  $('#watch').innerHTML = '';
  renderProbeChrome();
  if (!probeTried) {
    probeTried = true;
    try {
      const m = await import('./probe/probe.js');
      probeApi = await m.mountProbe($('#probeHost'), { onExplain: (html) => { if (chapter === 'probe') $('#explain').innerHTML = html; } });
      probeApi?.setLayer?.(probeL);
    } catch (e) {
      probeApi = null;
      $('#probeHost').innerHTML = `<div class="probe-soon"><div class="eyebrow">CHAPTER 2</div><h2>大模型里的世界地图</h2><p>一个只读过文字的大模型，脑子里有没有一张世界地图？把城市名喂给真实的开源模型，从它某一层的隐状态里，能不能用一个线性探针读出经纬度？</p><p class="soon">即将上线</p></div>`;
      $('#explain').innerHTML = '<div class="eyebrow" style="margin-bottom:4px">第二章</div><p>这一章正在制作中。先回到「能玩的世界模型」，在模型的梦里开开车。</p>';
    }
  }
  renderProbeChrome();
}

function probeMax() { return Number(probeApi?.maxLayer || probeApi?.layers?.length || probeApi?.levels || 4); }
function probeLayer(d) {
  if (!probeApi) { flashBtn(d > 0 ? '#btnIn' : '#btnOut'); return; }
  const n = Math.max(1, Math.min(probeMax(), probeL + d));
  if (n === probeL) { flashBtn(d > 0 ? '#btnIn' : '#btnOut'); return; }
  probeL = n;
  d > 0 ? sfx.dive() : sfx.rise();
  probeApi.setLayer?.(probeL);
  renderProbeChrome();
}
function renderProbeChrome() {
  if (chapter !== 'probe') return;
  $('#crumbs').innerHTML = `<button type="button" data-d="0"><span class="d">CH1</span>能玩的世界模型</button><span class="sep">›</span><button type="button" class="on" data-d="${probeL}"><span class="d">CH2</span>大模型里的世界地图 · 第 ${probeL} 层</button>`;
  $('#dname').innerHTML = `第 ${probeL} 层<small>DEPTH ${probeL} / ${probeApi ? probeMax() : '—'}</small>`;
  $('#btnIn').disabled = !probeApi || probeL >= probeMax();
  $('#btnOut').disabled = !probeApi || probeL <= 1;
}

/* ---------------------------------------------------------------- 知识碎片 */

function discoverFor() {
  const s = tl.step, d = tl.depth;
  if (d >= 2) discover('vmc');
  if (s.ph === 'sample' && !s.op) discover('tau');
  if (s.op === 'mu') discover('unused');
  if (s.op === 'z') discover('sweep');
  if (s.ph === 'enc' && /^e\d$/.test(s.op || '')) discover('conv');
  if (s.ph === 'dec' && /^d\d$/.test(s.op || '')) discover('deconv');
  if (s.op === 'd4' && !s.mi) discover('blur');
  if (s.op === 'gf') discover('forget');
  if (s.op === 'mdn' && !s.mi) discover('mdn');
  if (s.op === 'done' && !s.mi) discover('done');
  if (s.mi === 'sum') discover('mac');
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
  t.innerHTML = `<span class="t-icon" aria-hidden="true">✦</span><span class="t-body"><span class="t-k">发现知识碎片 · ${store.found.size}/${INSIGHTS.length}</span><span class="t-title">${esc(ins.title)}</span><span class="t-text">${esc(ins.text)}</span></span>`;
  t.addEventListener('click', () => { openCodex(); t.remove(); });
  box.prepend(t);
  const maxToasts = small() ? 1 : 2;
  while (box.children.length > maxToasts) box.lastElementChild.remove();
  setTimeout(() => t.classList.add('out'), 5200);
  setTimeout(() => t.remove(), 5900);
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

// 舞台被面板挡住的部分：右边调试器、下面控制条（和算式板）、上面章节切换和 D1 的开关
function updateInsets() {
  if (!stage) return;
  const dbg = $('#dbg'), folded = dbg.classList.contains('folded');
  const st = $('#stage').getBoundingClientRect();
  const ctl = $('#ctl').getBoundingClientRect();
  const board = $('#board');
  let bottom = st.bottom - ctl.top + 10;
  let top = $('.stage-top').getBoundingClientRect().bottom - st.top + 8;
  if (!$('#playUi').hidden && document.body.classList.contains('w-play') && chapter === 'wm') top = Math.max(top, $('#playUi').getBoundingClientRect().bottom - st.top + 6);
  if (small()) {
    bottom = st.bottom - Math.min(ctl.top, dbg.getBoundingClientRect().top) + 8;
    if (document.body.classList.contains('w-play')) bottom = Math.max(bottom, st.bottom - $('#pad').getBoundingClientRect().top + 6);
  }
  if (!board.hidden) {
    const br = board.getBoundingClientRect();
    if (small()) top = Math.max(top, br.bottom - st.top + 6);
    else bottom = Math.max(bottom, st.bottom - br.top + 8);
  }
  stage.setInsets(small() || folded ? 0 : dbg.offsetWidth + 28, Math.max(0, bottom), Math.max(0, top));
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
  $('#chap').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) { sfx.click(); setChapter(b.dataset.ch); } });
  $('#board').addEventListener('click', (e) => {
    if (e.target.closest('.bd-fold')) { $('#board').classList.toggle('folded'); updateBoard(true); updateInsets(); }
  });
  new ResizeObserver(() => updateInsets()).observe($('#dbg'));
  new ResizeObserver(() => updateInsets()).observe($('#board'));
  new ResizeObserver(() => updateInsets()).observe($('#ctl'));
  initPanes('world', { right: { el: '#dbg', v: '--dbg-w', name: '调试器' }, reserve: 640, onChange: updateInsets });
}

function bindKeys() {
  const steerKey = (e) => {
    const k = e.key.toLowerCase();
    if (k === 'a') return 1;
    if (k === 'd') return 2;
    // ← → 在 D1 播放时是方向键，其余时候是单步
    if (tl.depth === 1 && tl.playing && chapter === 'wm') { if (e.key === 'ArrowLeft') return 1; if (e.key === 'ArrowRight') return 2; }
    return 0;
  };
  addEventListener('keydown', (e) => {
    if (e.target.closest('input, textarea') && e.target.type !== 'range') return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === 'Escape' && $('#codex').classList.contains('on')) return closeCodex();
    const st = steerKey(e);
    if (st) {
      e.preventDefault();
      input.key = st;
      if (chapter === 'wm' && tl.depth === 1 && !tl.playing && (e.key === 'a' || e.key === 'd' || e.key === 'A' || e.key === 'D')) tl.play();
      return;
    }
    if (e.key === ' ') { e.preventDefault(); toggle(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); controls.h.next(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); controls.h.prev(); }
    else if (e.key === '+' || e.key === '=' || e.key === 'ArrowDown') { e.preventDefault(); into(); }
    else if (e.key === '-' || e.key === '_' || e.key === 'ArrowUp' || e.key === 'Escape') { e.preventDefault(); if (stage.free && e.key === 'Escape') stage.exitFree(); else out(); }
    else if (e.key === 'o' || e.key === 'O') setEye(sim.mode === 'open' ? 'closed' : 'open');
    else if (e.key === 'r' || e.key === 'R') $('#btnSync').click();
    else if (e.key === 'n' || e.key === 'N') $('#btnNew').click();
    else if (e.key === 'f' || e.key === 'F') stage.exitFree();
    else if (e.key === 'h' || e.key === 'H' || e.key === '?') showHint(!$('#stageHint').classList.contains('on'));
    else if (e.key >= '1' && e.key <= '6') setSpeed(SPEEDS[Number(e.key) - 1]);
  });
  addEventListener('keyup', (e) => {
    const k = e.key.toLowerCase();
    if ((k === 'a' && input.key === 1) || (k === 'd' && input.key === 2) || (e.key === 'ArrowLeft' && input.key === 1) || (e.key === 'ArrowRight' && input.key === 2)) input.key = 0;
  });
  addEventListener('blur', () => { input.key = 0; input.pad = 0; });
}

window.__world = { get tl() { return tl; }, get sim() { return sim; }, get stage() { return stage; }, get mac() { return mac; }, app, into, out, setEye, setChapter, setSpeed, input };
boot();
