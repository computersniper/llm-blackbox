// 智能体页面：选任务 → 正常速度看 agent 在屏幕上干活 → 点 ＋ 一层层揭开（循环 / 上下文 / 逐词元 / 模型内部）。
import { loadManifest, loadTask, isDecision } from './data.js';
import { Chat, md } from './chat.js';
import { Screen } from './screen.js';
import { LoopView, CtxView, TokView, FwdView } from './views.js';
import { Timeline } from './timeline.js';
import { Controls, SPEEDS } from './controls.js';
import { INSIGHTS, INSIGHT_BY_ID } from './insights.js';
import { Background } from './bg.js';
import { sfx, setSound, soundOn } from '../../js/audio.js';
import { $, esc, fmtNum } from '../../js/ui.js';
import { initPanes } from '../../js/resize.js';
import { splitExit } from './screen.js';
import { callArg } from './chat.js';

const KEY = 'blackbox:agent:v1';
function load() { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } }
const saved = load();
const store = { found: new Set(saved.found || []), speed: saved.speed || 1 };
function save() { try { localStorage.setItem(KEY, JSON.stringify({ found: [...store.found], speed: store.speed, sound: soundOn() })); } catch { /* 忽略 */ } }
if (saved.sound) setSound(true);
window.__sfx = sfx;

let M, chat, screen, controls, loopV, ctxV, tokV, fwdV;
let R = null, tl = null;
let mode = 'watch';
let lastView = '';
const small = () => matchMedia('(max-width: 900px)').matches;

/* ---------------------------------------------------------------- 启动 */

async function boot() {
  new Background($('#bg'));
  initPanes('agent', { left: { el: '#side', v: '--side-w', name: '对话栏', min: 300 }, right: { el: '#dbg', v: '--dbg-w', name: '调试器' }, reserve: 520 });
  try {
    M = await loadManifest();
  } catch (e) {
    $('#log').innerHTML = `<p style="color:var(--rose)">数据加载失败：${esc(e.message)}</p>`;
    return;
  }
  $('#modelName').textContent = M.model.name;
  $('#cxFoot').textContent = `模型：${M.model.name}（${(M.model.params / 1e8).toFixed(1)} 亿参数 · ${M.model.layers} 层 · 隐藏维度 ${M.model.hidden} · ${M.model.heads} 个查询头 / ${M.model.kvHeads} 个键值头）。每条轨迹都是这个模型在 bubblewrap 沙箱里的一次真实运行：上下文、词元概率、命令输出、文件改动和耗时都来自录制（${M.model.gpu}），网页只是回放。`;
  screen = new Screen();
  loopV = new LoopView();
  loopV.setModel(M.model);
  loopV.onLap = (t) => { if (!tl) return; tl.pause(); tl.seekTurn(t); sfx.click(); };
  ctxV = new CtxView();
  ctxV.onTurn = (t) => { if (!tl) return; tl.pause(); tl.seekTurn(t); sfx.click(); };
  tokV = new TokView();
  tokV.onTok = (j) => { if (!tl) return; tl.pause(); tl.seekTurn(tl.t, (s) => s.j === j); sfx.click(); };
  fwdV = new FwdView();
  controls = new Controls({
    into: () => into(),
    out: () => out(),
    prev: () => { if (!tl) return; tl.pause(); tl.back(); sfx.tick(); },
    next: () => { if (!tl) return; tl.pause(); tl.next(); sfx.tick(); },
    toggle: () => toggle(),
    speed: (s) => { store.speed = s; save(); if (tl && mode === 'inspect') { tl.speed = s; controls.updatePlay(tl); } },
    seek: (i) => { if (!tl) return; tl.pause(); tl.seekIndex(i); },
    depth: (d) => { if (!tl) return; d > tl.depth ? sfx.dive() : sfx.rise(); tl.setDepth(d); },
    fold: () => {},
  });
  chat = new Chat(M, { onSend, onPlus: () => into(), onSeek });
  bindKeys();
  bindChrome();
  renderCodexCount();
  addEventListener('resize', () => { lastView = ''; });
  requestAnimationFrame(frame);
  // 地址里带 #任务id 时直接开始（方便分享）
  const id = decodeURIComponent(location.hash.slice(1));
  const q = M.tasks.find((x) => x.id === id);
  if (q) chat.typeTask(q, true);
}

/* ---------------------------------------------------------------- 任务 */

async function onSend(q) {
  let r;
  try { r = await loadTask(q.id); } catch (e) { chat.log.insertAdjacentHTML('beforeend', `<p style="color:var(--rose)">数据加载失败：${esc(e.message)}</p>`); return; }
  r.M = M.model;
  R = r;
  document.body.classList.remove('idle');
  if (small()) document.body.classList.remove('chat-open');
  chat.startSession(q, R);
  chat.setBusy(true);
  screen.load(R);
  $('#btnPlus').disabled = false;
  const depth = tl && mode === 'inspect' ? tl.depth : 1;
  tl = new Timeline(R);
  tl.speed = mode === 'inspect' ? store.speed : 1;
  if (depth > 1) tl.setDepth(depth);
  tl.on((type) => {
    if (type === 'step') { if (mode === 'inspect') controls.update(tl); discoverFor(); }
    if (type === 'turn') sfx.tick();
    if (type === 'play' && mode === 'inspect') controls.updatePlay(tl);
    if (type === 'end') {
      chat.setBusy(false);
      $('#btnPlus').classList.add('glow');
      if (mode === 'inspect') controls.updatePlay(tl);
      discover('stop');
      sfx.land();
    }
  });
  lastView = '';
  if (mode === 'inspect') controls.update(tl);
  tl.play();
}

// 点对话记录里的某一次工具调用：跳过去
function onSeek(t, ci) {
  if (!tl) return;
  tl.pause();
  if (ci < 0) tl.seekTurn(t);
  else tl.seekTurn(t, (s) => s.c === ci && (s.k1 === `act${ci}`) && (s.p === undefined || s.p === 'exec'));
  if (small()) document.body.classList.remove('chat-open');
  sfx.click();
}

/* ---------------------------------------------------------------- 揭开 */

function enterInspect() {
  if (!tl) return;
  mode = 'inspect';
  document.body.classList.replace('mode-watch', 'mode-inspect');
  if (small()) { controls.setFolded(true); document.body.classList.remove('chat-open'); }
  if (tl.done) tl.restart();
  tl.speed = store.speed;
  tl.setDepth(2);
  controls.update(tl);
  tl.play();
  sfx.dive();
  lastView = '';
}

function exitInspect() {
  if (mode !== 'inspect') return;
  sfx.rise();
  mode = 'watch';
  document.body.classList.replace('mode-inspect', 'mode-watch');
  controls.renderCrumbs(null);
  if (tl) { tl.setDepth(1); tl.speed = 1; if (!tl.done) tl.play(); }
  lastView = '';
}

function into() {
  if (!tl) return;
  if (mode === 'watch') return enterInspect();
  if (!tl.canInto()) { flashBtn('#btnIn'); return; }
  sfx.dive();
  tl.into();
}

function out() {
  if (!tl) return;
  if (mode !== 'inspect') return;
  if (tl.depth <= 1) return exitInspect();
  sfx.rise();
  tl.out();
}

function toggle() {
  if (!tl) return;
  if (tl.done && !tl.playing) tl.restart();
  tl.toggle();
}

function flashBtn(sel) {
  $(sel).animate([{ transform: 'translateX(0)' }, { transform: 'translateX(-4px)' }, { transform: 'translateX(4px)' }, { transform: 'translateX(0)' }], { duration: 260 });
}

/* ---------------------------------------------------------------- 每帧 */

// 当前回放到哪：对话记录和屏幕都按它来画
function cursorOf() {
  const s = tl.step, T = R.turns[s.t], p = tl.p, L = T.say.length;
  const c = { t: s.t, say: 0, shown: 0, done: 0, fin: false, act: -1, ap: 0 };
  if (tl.depth === 1) {
    if (s.k1 === 'think') c.say = Math.floor(L * Math.min(1, p * 1.15));
    else if (s.k1 === 'answer') { c.say = Math.floor(L * Math.min(1, p * 1.2)); c.fin = c.say >= L; }
    else { c.say = L; c.shown = s.c + 1; c.done = s.c + (p >= 0.92 ? 1 : 0); c.act = s.c; c.ap = p; }
  } else {
    switch (s.p) {
      case 'gen': c.say = Math.min(L, s.j !== undefined ? T.gen.toks[s.j].b : Math.floor(T.gen.text.length * p)); break;
      case 'parse': c.say = L; break;
      case 'exec': c.say = L; c.shown = s.c + 1; c.done = s.c; c.act = s.c; c.ap = p; break;
      case 'append': c.say = L; c.shown = s.c + 1; c.done = s.c + 1; break;
      case 'done': c.say = L; c.fin = true; break;
    }
  }
  if (tl.done) { c.say = L; c.shown = c.done = T.calls.length; c.fin = T.final; c.act = -1; }
  return c;
}

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (tl && R) {
    tl.tick(dt);
    const view = mode === 'inspect' ? tl.view : 'screen';
    const views = $('#views');
    if (view !== lastView) {
      views.dataset.view = view;
      lastView = view;
      for (const [id, v] of [['vLoop', 'loop'], ['vCtx', 'ctx'], ['vTok', 'tok'], ['vFwd', 'fwd']]) $(`#${id}`).setAttribute('aria-hidden', String(v !== view));
      $('#desk').setAttribute('aria-hidden', String(view !== 'screen' && view !== 'loop'));
      if (view !== 'loop') resetDesk();
    }
    const cur = cursorOf();
    chat.render(cur);
    renderStrip(cur);
    if (view === 'screen' || view === 'loop') {
      const base = screen.acts.findIndex((a) => a.t >= cur.t);
      const b = base < 0 ? screen.acts.length : base;
      const k = b + cur.done;
      const ca = cur.act >= 0 && cur.act >= cur.done ? b + cur.act : -1;
      screen.show(k, ca, cur.ap, cur.fin);
    }
    const st = { R, s: tl.step, p: tl.p, depth: tl.depth };
    if (view === 'loop') { loopV.update(st); placeDesk(); }
    else if (view === 'ctx') ctxV.update(st);
    else if (view === 'tok') tokV.update(st);
    else if (view === 'fwd') fwdV.update(st);
  }
  requestAnimationFrame(frame);
}

// 揭开到“循环”时，屏幕缩小，落进沙箱那个节点里（按节点的长宽比重新排版，再整体缩小）
let deskT = '';
function placeDesk() {
  const desk = $('#desk');
  const slot = loopV.slotRect();
  if (!slot.w || !slot.h) return;
  const sc = small() ? 0.5 : 0.62;
  const w = slot.w / sc, h = slot.h / sc;
  const tr = `translate(${(slot.x - desk.offsetLeft).toFixed(1)}px, ${(slot.y - desk.offsetTop).toFixed(1)}px) scale(${sc})`;
  const key = `${tr}|${w | 0}|${h | 0}`;
  if (key !== deskT) {
    deskT = key;
    desk.style.width = `${w.toFixed(1)}px`;
    desk.style.height = `${h.toFixed(1)}px`;
    desk.style.transform = tr;
  }
}

function resetDesk() {
  const desk = $('#desk');
  desk.style.transform = desk.style.width = desk.style.height = '';
  deskT = '';
}

function renderStrip(cur) {
  const T = R.turns[cur.t];
  let h = `<span class="k">第 ${cur.t + 1}/${R.T} 圈</span>`;
  if (cur.act >= 0) { const c = T.calls[cur.act]; h += `<b>${esc(c.name)}</b> ${esc(callArg(c).slice(0, 80))}`; }
  else if (T.say) h += md(T.say.slice(0, Math.max(cur.say, 1)).split('\n').filter(Boolean).pop() || '');
  else if (T.calls.length) h += `<b>${esc(T.calls[0].name)}</b> ${esc(callArg(T.calls[0]).slice(0, 80))}`;
  h += '<span style="float:right;color:var(--dim)">对话 ›</span>';
  const el = $('#strip');
  if (el.dataset.h !== h) { el.dataset.h = h; el.innerHTML = h; }
}

/* ---------------------------------------------------------------- 知识碎片 */

function discoverFor() {
  const s = tl.step, d = tl.depth, T = R.turns[s.t];
  if (d >= 2) discover('loop');
  if (d >= 2 && s.p === 'parse' && T.calls.length) discover('harness');
  if (d >= 2 && s.p === 'exec') discover('sandbox');
  if (s.p === 'done') discover('stop');
  if (d >= 3 && s.sub === 'ctx' && s.t === 0) discover('tooldefs');
  if (d >= 3 && s.sub === 'ctx' && s.t > 0) discover('refeed');
  if (d >= 3 && s.t >= 3 && s.p === 'prompt') discover('growth');
  if (d >= 3 && s.sub === 'kv' && s.t > 0) { discover('kvreuse'); setTimeout(() => { if (tl?.step?.sub === 'kv') discover('prefill'); }, 1600); }
  if (d >= 3 && s.p === 'append') discover('toolresp');
  if (d >= 4 && s.j !== undefined && !s.f) {
    const tk = T.gen.toks[s.j];
    if (isDecision(tk, s.j) && tk.top[0][1] < 0.75) discover('decision');
    if (T.calls.some((c) => s.j > c.j0 + 3 && s.j <= c.j1)) discover('json');
  }
  if (d >= 5 && s.f) discover('forward');
  // 上一圈的工具报错了，这一圈换了办法
  if (s.t > 0 && (s.p === 'exec' || (d === 1 && s.c != null))) {
    const prev = R.turns[s.t - 1];
    if (prev.calls.some((c) => !c.info?.ok || (c.name === 'bash' && splitExit(c.result).code !== 0))) setTimeout(() => discover('selfcorrect'), 1200);
  }
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
  while (box.children.length > 3) box.lastElementChild.remove();
  setTimeout(() => t.classList.add('out'), 6000);
  setTimeout(() => t.remove(), 6700);
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

function bindChrome() {
  $('#btnCodex').addEventListener('click', openCodex);
  $('.cx-close').addEventListener('click', closeCodex);
  $('#codex').addEventListener('click', (e) => { if (e.target.id === 'codex') closeCodex(); });
  $('#btnReset').addEventListener('click', () => { if (!confirm('清空已收集的知识碎片？')) return; store.found.clear(); save(); renderCodexCount(); closeCodex(); });
  const sb = $('#btnSound');
  const renderSound = () => { sb.classList.toggle('on', soundOn()); sb.setAttribute('aria-pressed', soundOn()); sb.title = soundOn() ? '关闭声音' : '打开声音'; };
  sb.addEventListener('click', () => { setSound(!soundOn()); renderSound(); save(); sfx.click(); });
  renderSound();
  $('#btnSideClose').addEventListener('click', () => document.body.classList.remove('chat-open'));
  $('#strip').addEventListener('click', () => document.body.classList.add('chat-open'));
  $('#btnFab').addEventListener('click', () => into());
}

function bindKeys() {
  addEventListener('keydown', (e) => {
    if (e.target.closest('input, textarea')) return;
    if (e.key === 'Escape' && $('#codex').classList.contains('on')) return closeCodex();
    if (!tl) return;
    if (e.key === '+' || e.key === '=' || (mode === 'inspect' && e.key === 'ArrowDown')) { e.preventDefault(); into(); return; }
    if (mode !== 'inspect') return;
    if (e.key === ' ') { e.preventDefault(); toggle(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); tl.pause(); tl.next(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); tl.pause(); tl.back(); }
    else if (e.key === '-' || e.key === '_' || e.key === 'ArrowUp' || e.key === 'Escape') { e.preventDefault(); out(); }
    else if (e.key >= '1' && e.key <= '6') { const s = SPEEDS[Number(e.key) - 1]; store.speed = s; save(); tl.speed = s; controls.updatePlay(tl); }
  });
}

window.__ag = { get tl() { return tl; }, get R() { return R; }, get mode() { return mode; }, into, out, fmtNum };
boot();
