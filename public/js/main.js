import { analyze, PRESETS } from './model.js';
import { Background } from './bg.js';
import { sfx, setSound, soundOn } from './audio.js';
import { INSIGHTS, INSIGHT_BY_ID } from './insights.js';
import { $, $$, esc, tokHTML, typeHTML, reveal, reducedMotion } from './ui.js';

import surface from './levels/l0-surface.js';
import tokens from './levels/l1-tokens.js';
import embed from './levels/l2-embed.js';
import tower from './levels/l3-tower.js';
import attention from './levels/l4-attention.js';
import mlp from './levels/l5-mlp.js';
import neuron from './levels/l6-neuron.js';
import bits from './levels/l7-bits.js';
import output from './levels/l8-output.js';

const LEVELS = [surface, tokens, embed, tower, attention, mlp, neuron, bits, output];
const OUT = 8;
const KEY = 'blackbox:v1';

/* ---------------- 持久化 ---------------- */

function load() {
  try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; }
}
function save() {
  try {
    localStorage.setItem(KEY, JSON.stringify({ prompt: state.prompt, basePrompt: state.basePrompt, max: state.max, found: [...state.found], sound: soundOn(), seenOut: state.seenOut }));
  } catch { /* 隐私模式等情况下忽略 */ }
}

const saved = load();
const state = {
  prompt: saved.prompt || PRESETS[0].prompt,
  basePrompt: saved.basePrompt || saved.prompt || PRESETS[0].prompt,
  depth: -1,
  max: saved.max ?? 0,
  seenOut: !!saved.seenOut,
  found: new Set(saved.found || []),
  focus: null,
  layer: 5,
  head: null,
  neuron: null,
  weight: null,
  visits: {},
};
let model = analyze(state.prompt);
state.focus = model.tokens.length - 1;
if (saved.sound) setSound(true);

/* ---------------- 外壳元素 ---------------- */

const stage = $('#stage');
const bg = new Background($('#bg'));
const narrator = $('#narrator');
const narText = $('.nar-text', narrator);
const btnNext = $('#btnNext');
const btnUp = $('#btnUp');
const tooltip = $('#tooltip');

let current = null;
let busy = false;
let cancelType = () => {};
let nextAction = null;

/* ---------------- 对外 API（给各层用） ---------------- */

const app = {
  state,
  levels: LEVELS,
  get model() { return model; },
  sfx,

  setPrompt(p, { base = false } = {}) {
    p = Array.from(p).slice(0, 80).join('');
    if (!p.trim()) return;
    state.prompt = p;
    if (base) { state.basePrompt = p; state.generated = 0; }
    model = analyze(p);
    state.focus = model.tokens.length - 1;
    state.neuron = null;
    state.weight = null;
    save();
    renderHudPrompt();
  },

  go,
  dive(origin) { return go(Math.min(state.depth + 1, 7), origin); },
  up(origin) { return go(Math.max(state.depth - 1, 0), origin); },

  say(html, opts = {}) {
    cancelType();
    cancelType = typeHTML(narText, html, opts.speed ?? 14);
    if (opts.flash) {
      narrator.classList.remove('flash');
      void narrator.offsetWidth;
      narrator.classList.add('flash');
    }
  },

  setNext(label, fn, opts = {}) {
    nextAction = fn;
    btnNext.innerHTML = label;
    btnNext.hidden = !label;
    btnNext.classList.toggle('pulse', !!opts.pulse);
  },

  discover(id) {
    if (state.found.has(id) || !INSIGHT_BY_ID.has(id)) return;
    state.found.add(id);
    save();
    sfx.discover();
    toastInsight(INSIGHT_BY_ID.get(id));
    renderCodexCount();
  },

  tip: {
    show(html, x, y) {
      tooltip.innerHTML = html;
      tooltip.classList.add('on');
      this.move(x, y);
    },
    move(x, y) {
      const r = tooltip.getBoundingClientRect();
      let tx = x + 16, ty = y + 18;
      if (tx + r.width > innerWidth - 8) tx = x - r.width - 16;
      if (ty + r.height > innerHeight - 8) ty = y - r.height - 14;
      tooltip.style.transform = `translate(${Math.max(8, tx)}px, ${Math.max(8, ty)}px)`;
    },
    hide() { tooltip.classList.remove('on'); },
    // 给元素绑定悬停提示，html 可以是函数
    bind(el, html) {
      el.addEventListener('pointerenter', (e) => app.tip.show(typeof html === 'function' ? html(e) : html, e.clientX, e.clientY));
      el.addEventListener('pointermove', (e) => app.tip.move(e.clientX, e.clientY));
      el.addEventListener('pointerleave', () => app.tip.hide());
    },
  },

  visits(key) { return state.visits[key] || 0; },
};
window.__bb = app; // 方便调试

/* ---------------- 转场 ---------------- */

function originPoint(origin) {
  if (origin && typeof origin.clientX === 'number' && (origin.clientX || origin.clientY)) return { x: origin.clientX, y: origin.clientY };
  if (origin instanceof Element) {
    const r = origin.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  }
  const r = stage.getBoundingClientRect();
  return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
}

async function go(target, origin) {
  if (busy || target === state.depth || !LEVELS[target]) return;
  if (target > state.max && target !== OUT && target > state.depth + 1) return;
  busy = true;
  nextAction = null; // 转场期间旧场景的"继续"不再生效
  app.tip.hide();
  const from = state.depth;
  const dir = target === OUT ? -1 : from === OUT ? 1 : Math.sign(target - from) || 1;
  const { x, y } = originPoint(origin);
  const sr = stage.getBoundingClientRect();
  const ox = x - sr.left, oy = y - sr.top;
  const L = LEVELS[target];

  if (from >= 0) { dir > 0 ? sfx.dive() : sfx.rise(); }
  bg.warp(dir, x, y);
  document.body.dataset.depth = target;
  document.body.classList.toggle('at-surface', target === 0);
  bg.setDepth(target);

  const fast = reducedMotion;
  if (current) {
    const old = current;
    old.el.style.transformOrigin = `${ox}px ${oy}px`;
    old.el.style.pointerEvents = 'none';
    const anim = old.el.animate(
      [
        { transform: 'scale(1)', opacity: 1, filter: 'blur(0px)' },
        { transform: `scale(${dir > 0 ? 3.2 : 0.35})`, opacity: 0, filter: 'blur(12px)' },
      ],
      { duration: fast ? 1 : 620, easing: 'cubic-bezier(.7,0,.25,1)', fill: 'forwards' },
    );
    if (target > 0) titleCard(L, target);
    await anim.finished;
    try { old.inst?.destroy?.(); } catch (e) { console.error(e); }
    old.el.remove();
  } else if (target > 0) {
    titleCard(L, target);
  }

  state.depth = target;
  if (target === OUT) state.seenOut = true;
  else state.max = Math.max(state.max, target);
  state.visits[L.key] = (state.visits[L.key] || 0) + 1;
  save();
  setHash(target);
  renderGauge();
  renderHudPrompt();
  btnUp.hidden = target === 0 || target === OUT;
  app.setNext('继续下潜 <b>↓</b>', () => app.dive());

  const el = document.createElement('section');
  el.className = `scene scene-${L.key}`;
  el.setAttribute('aria-label', `深度 ${target} · ${L.name}`);
  stage.append(el);
  let inst = null;
  try { inst = L.mount(el, app); } catch (e) { console.error(e); }
  reveal(el, fast ? 0 : 260);
  $$('input[type="range"]', el).forEach(fillRange);
  el.style.transformOrigin = `${ox}px ${oy}px`;
  el.animate(
    [
      { transform: `scale(${dir > 0 ? 0.3 : 2.4})`, opacity: 0, filter: 'blur(14px)' },
      { transform: 'scale(1)', opacity: 1, filter: 'blur(0px)' },
    ],
    { duration: fast ? 1 : 900, easing: 'cubic-bezier(.16,1,.3,1)' },
  );
  current = { el, inst, depth: target };
  el.scrollTop = 0;
  busy = false;
}

/* ---------------- 标题卡 ---------------- */

const tc = $('#titlecard');
let tcTimer = 0;
function titleCard(L, d) {
  const repeat = (state.visits[L.key] || 0) > 0;
  $('.tc-depth', tc).textContent = d === OUT ? 'ASCENT · 上浮' : `DEPTH −${String(d).padStart(2, '0')}`;
  $('.tc-name', tc).textContent = L.name;
  $('.tc-en', tc).textContent = L.en;
  $('.tc-scale', tc).textContent = L.scale;
  tc.classList.remove('on', 'quick');
  void tc.offsetWidth;
  tc.classList.add('on');
  if (repeat) tc.classList.add('quick');
  clearTimeout(tcTimer);
  tcTimer = setTimeout(() => tc.classList.remove('on'), repeat ? 900 : 1500);
}

/* ---------------- 深度仪表 ---------------- */

const gauge = $('#gauge');
function buildGauge() {
  const order = [OUT, 0, 1, 2, 3, 4, 5, 6, 7];
  gauge.innerHTML = `
    <div class="g-track"><div class="g-fill"></div></div>
    <div class="g-marker"></div>
    ${order.map((d) => `
      <button class="g-node ${d === OUT ? 'g-out' : ''}" data-d="${d}" type="button">
        <span class="g-dot"></span>
        <span class="g-num">${d === OUT ? '↑' : String(d).padStart(2, '0')}</span>
        <span class="g-name">${LEVELS[d].name}</span>
      </button>`).join('')}
    <div class="g-readout"><span class="g-read-d"></span><span class="g-read-s"></span></div>`;
  gauge.addEventListener('click', (e) => {
    const b = e.target.closest('.g-node');
    if (!b || b.dataset.locked === '1') return;
    sfx.click();
    go(Number(b.dataset.d), b);
  });
}
function renderGauge() {
  const d = state.depth;
  $$('.g-node', gauge).forEach((b) => {
    const n = Number(b.dataset.d);
    const unlocked = n === OUT ? state.seenOut : n <= Math.max(state.max, d);
    b.dataset.locked = unlocked ? '0' : '1';
    b.classList.toggle('on', n === d);
    b.title = unlocked ? `${LEVELS[n].name}` : '尚未抵达';
  });
  const on = $(`.g-node[data-d="${d}"]`, gauge);
  if (on) {
    const top = on.offsetTop + on.offsetHeight / 2;
    $('.g-marker', gauge).style.translate = `0 ${top}px`;
    const first = $('.g-node[data-d="0"]', gauge);
    const fillTop = first.offsetTop + first.offsetHeight / 2;
    $('.g-fill', gauge).style.height = d === OUT ? '0px' : `${Math.max(0, top - fillTop)}px`;
    $('.g-fill', gauge).style.top = `${fillTop}px`;
  }
  $('.g-read-d', gauge).textContent = d === OUT ? '上浮中' : `深度 −${String(Math.max(0, d)).padStart(2, '0')}`;
  $('.g-read-s', gauge).textContent = LEVELS[d]?.scale ?? '';
  $('#hudDepth').innerHTML = d === OUT ? '↑ <b>输出</b>' : `D${d} <b>${LEVELS[d]?.name ?? ''}</b>`;
}

/* ---------------- 顶部 HUD ---------------- */

function renderHudPrompt() {
  const el = $('#hudPrompt');
  const toks = model.tokens.slice(-14);
  el.innerHTML = `<span class="hp-k">输入</span>${model.tokens.length > 14 ? '<span class="hp-more">…</span>' : ''}${toks.map((t) => tokHTML(t, 'mini')).join('')}`;
  el.title = state.prompt;
}

function renderCodexCount() {
  $('#codexCount').textContent = `${state.found.size}/${INSIGHTS.length}`;
  $('#btnCodex').style.setProperty('--p', state.found.size / INSIGHTS.length);
}

/* ---------------- 知识碎片提示与图鉴 ---------------- */

function toastInsight(ins) {
  const box = $('#toasts');
  const t = document.createElement('button');
  t.type = 'button';
  t.className = 'toast';
  t.innerHTML = `
    <span class="t-icon" aria-hidden="true">✦</span>
    <span class="t-body"><span class="t-k">发现知识碎片 · ${state.found.size}/${INSIGHTS.length}</span><span class="t-title">${esc(ins.title)}</span><span class="t-text">${esc(ins.text)}</span></span>`;
  t.addEventListener('click', () => { openCodex(ins.id); t.remove(); });
  box.prepend(t);
  setTimeout(() => t.classList.add('out'), 6500);
  setTimeout(() => t.remove(), 7200);
  if (state.found.size === INSIGHTS.length) {
    setTimeout(() => app.say('✦ 你集齐了全部知识碎片。黑箱不再那么黑了——它只是数字、乘法与加法，被叠了一千亿次。', { flash: true }), 900);
  }
}

const codex = $('#codex');
function openCodex(focusId) {
  const byD = new Map();
  INSIGHTS.forEach((x) => { if (!byD.has(x.d)) byD.set(x.d, []); byD.get(x.d).push(x); });
  $('.cx-grid', codex).innerHTML = [...byD.entries()].map(([d, list]) => `
    <div class="cx-group">
      <div class="cx-gh"><span class="eyebrow">${d === OUT ? '上浮 · 输出' : `深度 −${String(d).padStart(2, '0')}`}</span><span>${LEVELS[d].name}</span></div>
      ${list.map((x) => state.found.has(x.id) ? `
        <article class="cx-card found ${x.id === focusId ? 'focus' : ''}"><h4>✦ ${esc(x.title)}</h4><p>${esc(x.text)}</p></article>` : `
        <article class="cx-card locked"><h4>？？？</h4><p class="hint">线索：${esc(x.hint)}</p></article>`).join('')}
    </div>`).join('');
  $('.cx-progress b', codex).textContent = `${state.found.size} / ${INSIGHTS.length}`;
  $('.cx-bar i', codex).style.width = `${(state.found.size / INSIGHTS.length) * 100}%`;
  codex.classList.add('on');
  codex.setAttribute('aria-hidden', 'false');
  setTimeout(() => $('.cx-card.focus', codex)?.scrollIntoView({ block: 'center', behavior: 'smooth' }), 80);
}
function closeCodex() {
  codex.classList.remove('on');
  codex.setAttribute('aria-hidden', 'true');
}

/* ---------------- 路由 ---------------- */

let hashLock = false;
function setHash(d) {
  hashLock = true;
  const h = d === 0 ? '#/' : `#/${d}`;
  if (location.hash !== h) history.pushState(null, '', h);
  setTimeout(() => { hashLock = false; }, 0);
}
function readHash() {
  const m = location.hash.match(/^#\/([0-8])$/);
  return m ? Number(m[1]) : 0;
}
function followHash() {
  const d = readHash();
  if (d === state.depth) return;
  if (busy) setTimeout(followHash, 300); // 转场中按了后退/前进：等转场结束再跟上
  else go(d);
}
addEventListener('popstate', () => { if (!hashLock) followHash(); });

/* ---------------- 事件 ---------------- */

function fillRange(r) {
  const min = Number(r.min || 0), max = Number(r.max || 100);
  r.style.setProperty('--fill', `${((Number(r.value) - min) / (max - min)) * 100}%`);
}
document.addEventListener('input', (e) => { if (e.target.matches?.('input[type="range"]')) fillRange(e.target); });

btnNext.addEventListener('click', (e) => { if (busy) return; sfx.click(); nextAction?.(e); });
btnUp.addEventListener('click', (e) => { sfx.click(); app.up(e); });
$('#btnCodex').addEventListener('click', () => { sfx.click(); openCodex(); });
$('.cx-close', codex).addEventListener('click', closeCodex);
codex.addEventListener('click', (e) => { if (e.target === codex) closeCodex(); });
$('#btnReset').addEventListener('click', () => {
  if (!confirm('清空已收集的知识碎片和下潜进度？')) return;
  state.found.clear();
  state.max = 0;
  state.seenOut = false;
  save();
  renderCodexCount();
  closeCodex();
  go(0);
});
$('.brand').addEventListener('click', (e) => { e.preventDefault(); go(0); });

const btnSound = $('#btnSound');
function renderSound() {
  btnSound.classList.toggle('on', soundOn());
  btnSound.setAttribute('aria-pressed', soundOn());
  btnSound.title = soundOn() ? '关闭声音' : '打开声音';
}
btnSound.addEventListener('click', () => { setSound(!soundOn()); renderSound(); save(); sfx.click(); });

addEventListener('keydown', (e) => {
  if (e.target.closest('input, textarea, select, [contenteditable]')) return;
  if (e.key === 'Escape' && codex.classList.contains('on')) return closeCodex();
  if (codex.classList.contains('on')) return;
  if (e.key === 'ArrowDown' || e.key === 'PageDown') { e.preventDefault(); if (!btnNext.hidden) btnNext.click(); }
  if (e.key === 'ArrowUp' || e.key === 'PageUp' || e.key === 'Escape') {
    e.preventDefault();
    if (state.depth > 0 && state.depth !== OUT) app.up();
  }
});

/* ---------------- 启动 ---------------- */

buildGauge();
renderCodexCount();
renderSound();
const start = readHash();
if (start > state.max && start !== OUT) state.max = start;
if (start === OUT) state.seenOut = true;
go(start);
addEventListener('resize', () => renderGauge());
