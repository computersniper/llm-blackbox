import { loadManifest, loadQuestion } from './data.js';
import { Chat } from './chat.js';
import { Timeline } from './timeline.js';
import { Controls, SPEEDS } from './controls.js';
import { interestingHead } from './explain.js';
import { INSIGHTS, INSIGHT_BY_ID } from './insights.js';
import { Background } from './bg.js';
import { sfx, setSound, soundOn } from './audio.js';
import { $, esc, tokPlain, tokHTML, fmtPct, fmtNum, sleep } from './ui.js';

const KEY = 'blackbox:v2';
const ROLE_NAME = { system: '系统提示', user: '你的问题', assistant: '模型的回答', tpl: '模板 / 特殊标记' };

function load() { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } }
const saved = load();
const store = { found: new Set(saved.found || []), speed: saved.speed || 1 };
function save() { try { localStorage.setItem(KEY, JSON.stringify({ found: [...store.found], speed: store.speed, sound: soundOn() })); } catch { /* 忽略 */ } }
if (saved.sound) setSound(true);
window.__sfx = sfx;

let manifest, chat, controls;
let engine = null, machine = null, THREE = null;
let tl = null, cur = null, Q = null;
let mode = 'chat';
let dAnim = 1, lastView = '', lastDepth = 1;
let streaming = null;
const ctx = { head: null };

/* ---------------------------------------------------------------- 启动 */

async function boot() {
  new Background($('#bg'));
  try {
    manifest = await loadManifest();
  } catch (e) {
    $('#chatLog').innerHTML = `<p style="color:var(--rose)">数据加载失败：${esc(e.message)}</p>`;
    return;
  }
  controls = new Controls({
    into: () => into(),
    out: () => out(),
    prev: () => { if (!tl) return; tl.pause(); tl.back(); sfx.tick(); },
    next: () => { if (!tl) return; tl.pause(); tl.next(); sfx.tick(); },
    toggle: () => toggle(),
    speed: (s) => { store.speed = s; save(); if (tl) { tl.speed = s; controls.updatePlay(tl); } },
    seek: (i) => { if (!tl) return; tl.pause(); tl.seekIndex(i); },
    depth: (d) => { if (!tl) return; if (d === 0) exitInspect(); else { d > tl.depth ? sfx.dive() : sfx.rise(); tl.setDepth(d); } },
    fold: () => updateInsets(),
    head: (h) => { ctx.head = h; if (tl) { controls.update(tl, ctx); checkSink(); } },
  });
  chat = new Chat(manifest, { onSend, onPeek: (m) => enterInspect(m), onPlus });
  bindKeys();
  bindChrome();
  renderCodexCount();
}

/* ---------------------------------------------------------------- 聊天 */

function onSend(q) {
  chat.addUser(q);
  const msg = chat.addBot(q);
  if (mode === 'inspect') attach(msg);
  else streamNormal(msg, 0);
  if (q.id === 'q03' || q.id === 'q04') setTimeout(() => discover('wrong'), 2500);
}

// 不揭开的时候，就像平常的 AI 一样流式输出
function streamNormal(msg, from) {
  stopStreaming();
  chat.setBusy(true);
  const s = (streaming = { msg, n: from, alive: true });
  (async () => {
    if (from === 0) await sleep(550);
    while (s.alive && s.n < msg.q.replyTokens.length) {
      s.n++;
      msg.render(s.n, { fresh: true });
      await sleep(38 + Math.random() * 40);
    }
    if (!s.alive) return;
    msg.finish();
    streaming = null;
    chat.setBusy(false);
    $('#btnPlus').disabled = false;
    $('#btnPlus').classList.add('glow');
  })();
}

function stopStreaming() {
  if (!streaming) return 0;
  streaming.alive = false;
  const n = streaming.n;
  streaming = null;
  chat.setBusy(false);
  return n;
}

function onPlus() {
  if (mode === 'inspect') return into();
  const last = [...chat.msgs].reverse()[0];
  if (last) enterInspect(last);
}

/* ---------------------------------------------------------------- 揭开 */

async function ensureEngine() {
  if (engine) return;
  const eng = await import('./stage/engine.js');
  const mach = await import('./stage/machine.js');
  THREE = eng.THREE;
  engine = new eng.Engine($('#gl'), { onFrame, onWheelDepth: (d) => (d > 0 ? into() : out()), onHover, onPick });
  machine = new mach.Machine(engine);
}

async function enterInspect(msg) {
  const from = streaming?.msg === msg ? stopStreaming() : null;
  if (mode !== 'inspect') {
    mode = 'inspect';
    document.body.classList.replace('mode-chat', 'mode-inspect');
    $('#loading').hidden = false;
    try { await ensureEngine(); } catch (e) { console.error(e); $('#loading').innerHTML = `3D 舞台初始化失败：${esc(e.message)}`; return; }
    engine.active = true;
    if (matchMedia('(max-width: 900px)').matches && !$('#dbg').classList.contains('folded')) { $('#dbg').classList.add('folded'); $('#btnDbgFold').textContent = '+'; }
    updateInsets();
    sfx.dive();
  }
  await attach(msg, from);
}

async function attach(msg, from = null) {
  $('#loading').hidden = false;
  if (tl) tl.pause();
  let q;
  try { q = await loadQuestion(msg.q.id, manifest); } catch (e) { $('#loading').innerHTML = `数据加载失败：${esc(e.message)}`; return; }
  $('#loading').hidden = true;
  if (cur && cur !== msg && !cur.done) streamNormal(cur, cur.shown < 0 ? 0 : cur.shown); // 之前那条还没说完的，让它自己说完
  const fresh = Q !== q;
  Q = q;
  cur = msg;
  chat.markInspected(msg);
  const prevDepth = tl ? tl.depth : 1;
  tl = new Timeline(Q, manifest.focusLayers);
  tl.speed = store.speed;
  if (prevDepth > 1) tl.setDepth(prevDepth);
  if (from != null && from > 0) tl.seekToken(Math.min(from, Q.G - 1));
  tl.on((type) => {
    if (type === 'step') { controls.update(tl, ctx); discoverFor(); }
    if (type === 'token') { cur.render(tl.g, { cur: true, fresh: true }); renderStrip(); }
    if (type === 'play') controls.updatePlay(tl);
    if (type === 'end') {
      cur.finish();
      controls.updatePlay(tl);
      if (Q.steps[Q.G - 1].chosenS === '<|im_end|>') discover('imend');
    }
  });
  if (fresh) {
    machine.load(Q);
    const far = machine.camera({ view: 'box', step: tl.step, g: tl.g });
    engine.setView(far.pos.clone().multiplyScalar(2.2), far.look, { snap: true });
    dAnim = tl.depth;
  }
  msg.render(tl.g, { cur: true });
  if (msg.done) { msg.done = false; }
  renderStrip();
  controls.update(tl, ctx);
  tl.play();
}

function exitInspect() {
  if (mode !== 'inspect') return;
  sfx.rise();
  mode = 'chat';
  document.body.classList.replace('mode-inspect', 'mode-chat');
  document.body.classList.remove('chat-open');
  controls.renderCrumbs(null);
  if (tl) tl.pause();
  if (cur) {
    const n = tl ? tl.g : 0;
    chat.markInspected(null);
    if (!tl?.done && n < cur.q.replyTokens.length) streamNormal(cur, n);
    else cur.finish();
  }
  setTimeout(() => { if (mode === 'chat' && engine) engine.active = false; }, 900);
  tl = null;
  cur = null;
}

function into() {
  if (!tl) return;
  if (!tl.canInto()) { flashBtn('#btnIn'); return; }
  sfx.dive();
  tl.into();
}

function out() {
  if (!tl) return;
  if (tl.depth <= 1) return exitInspect();
  sfx.rise();
  tl.out();
}

function toggle() {
  if (!tl) return;
  if (tl.done && !tl.playing) { tl.done = false; tl.seekToken(0); }
  tl.toggle();
}

function updateInsets() {
  if (!engine) return;
  const small = matchMedia('(max-width: 900px)').matches;
  const dbg = $('#dbg');
  const folded = dbg.classList.contains('folded');
  engine.setInsets(small || folded ? 0 : dbg.offsetWidth + 28, small ? (folded ? 120 : 200) : 90);
}
addEventListener('resize', () => updateInsets());

function flashBtn(sel) {
  const b = $(sel);
  b.animate([{ transform: 'translateX(0)' }, { transform: 'translateX(-4px)' }, { transform: 'translateX(4px)' }, { transform: 'translateX(0)' }], { duration: 260 });
}

/* ---------------------------------------------------------------- 每帧 */

function onFrame(dt, t) {
  if (!tl || !machine) return;
  tl.tick(dt);
  dAnim += (tl.depth - dAnim) * (1 - Math.exp(-dt * 2.6));
  const view = tl.view, s = tl.step;
  let head = null;
  if ((view === 'attn' || view === 'dot' || view === 'bits') && s.ph === 'layer') {
    if (ctx.head === 'avg') head = null;
    else if (typeof ctx.head === 'number') head = ctx.head;
    else head = view === 'dot' ? Q.dotAt(s.L, tl.g)?.head ?? interestingHead(Q, s.L, Q.row(tl.g)) : interestingHead(Q, s.L, Q.row(tl.g));
  }
  const st = { step: s, p: tl.p, g: tl.g, depth: tl.depth, dAnim, view, head };
  machine.update(st, dt, t);
  const cam = machine.camera(st);
  const same = view === lastView && tl.depth === lastDepth;
  engine.setView(cam.pos, cam.look, { keepOrbit: same, speed: same ? 2.4 : 2.0 });
  if (!same) { lastView = view; lastDepth = tl.depth; }
}

/* ---------------------------------------------------------------- 悬停 / 点击 3D 物体 */

const tip = $('#tooltip');
function showTip(html, e) {
  tip.innerHTML = html;
  tip.classList.add('on');
  const r = tip.getBoundingClientRect();
  let x = e.clientX + 16, y = e.clientY + 16;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 16;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 12;
  tip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
}
function hideTip() { tip.classList.remove('on'); }

function onHover(info, e) {
  if (!info || !Q) return hideTip();
  const T = (i) => esc(tokPlain(Q.tokens[i].s));
  let h = '';
  switch (info.type) {
    case 'tile': { const t = Q.tokens[info.i]; h = `<span class="k">词元 #${info.i} · ${ROLE_NAME[t.role] || ''}</span><b>${T(info.i)}</b>　编号 <span class="v">${t.id}</span>`; break; }
    case 'slab': h = `<span class="k">第 ${info.L} 层 · Transformer 块</span>RMSNorm → 注意力（16 Q / 8 KV）→ 残差 → RMSNorm → SwiGLU → 残差<br>${fmtNum(manifest.model.paramsPerLayer)} 个参数<br><span class="v">点击跳到这一层</span>`; break;
    case 'kv': h = `<span class="k">KV 缓存 · 第 ${info.L} 层</span>位置 ${info.j}「${T(info.j)}」的 ${info.kind}（8 组 × 128 维）<br>后面的词元会直接复用它`; break;
    case 'bar': h = info.id < 0 ? `<span class="k">其余十五万个词元</span>加起来 <span class="v">${fmtPct(info.p)}</span>` : `<span class="k">候选词元</span><b>${esc(tokPlain(info.str))}</b>　<span class="v">${fmtPct(info.p)}</span>${info.inPool ? '<br>进入了最终候选池' : ''}`; break;
    case 'seg': h = `<span class="k">候选池 · 最终概率</span><b>${esc(tokPlain(info.str))}</b>　<span class="v">${fmtPct(info.p)}</span>`; break;
    case 'beam': h = `<span class="k">第 ${info.L} 层 · ${info.head == null ? '16 头平均' : `第 ${info.head} 头`}</span>${T(Q.row(tl.g))} → ${T(info.j)}　<span class="v">${fmtPct(info.w)}</span>`; break;
    case 'bulb': h = `<span class="k">神经元 #${info.n}</span>SiLU(g)·u = <span class="v">${info.v.toFixed(3)}</span>${info.full ? '' : '<br><span style="color:var(--dim)">这一层只导出了最亮的 16 个</span>'}`; break;
    case 'pin': h = `<span class="k">输入第 ${info.dim} 维</span>x = <span class="v">${info.x.toFixed(4)}</span><br>w<sub>gate</sub> = ${info.wg}<br>w<sub>up</sub> = ${info.wu}<br><span class="v">点击，看这个权重的比特</span>`; break;
    case 'wire': h = `<span class="k">${info.which === 'gate' ? 'gate' : 'up'} 权重</span>x × w = ${info.x.toFixed(3)} × ${info.w.toFixed(4)} = <span class="v">${info.prod.toFixed(4)}</span>`; break;
    case 'key': h = `<span class="k">第 ${info.i} 位 · ${info.i === 0 ? '符号' : info.i <= 8 ? '指数' : '尾数'}</span>点击翻转`; break;
    default: return hideTip();
  }
  showTip(h, e);
}

function onPick(info) {
  if (!info || !tl) return;
  if (info.type === 'slab') {
    tl.pause();
    if (tl.depth < 3) tl.setDepth(3);
    tl.seek((st) => st.ph === 'layer' && st.L === info.L);
    sfx.click();
  } else if (info.type === 'key') {
    machine.detail.flip(info.i);
    sfx.bit();
    discover('bits');
  } else if (info.type === 'pin') {
    machine.detail.selectPin(info.k);
    sfx.click();
    if (tl.depth === 6 && tl.step.mi === 'mul') tl.into();
  }
}

/* ---------------------------------------------------------------- 知识碎片 */

function discoverFor() {
  const s = tl.step, d = tl.depth, v = tl.view;
  if (d >= 2 && s.ph === 'read' && s.g === 0) discover('template');
  if (d >= 2 && s.ph === 'embed' && s.g === 0) discover('prefill');
  if (d >= 2 && s.ph === 'embed' && s.g > 0) discover('kvcache');
  if (d >= 2 && s.ph === 'read' && s.g > 0) discover('autoregress');
  if (v === 'tower' && s.L >= 22) discover('lens');
  if (s.op === 'ln1' && d >= 4) discover('rmsnorm');
  if (s.op === 'add1' && d >= 4) discover('residual');
  if (s.op === 'attn' && d >= 4) discover('gqa');
  if (s.sub === 'qkv') discover('rope');
  if (s.sub === 'softmax') discover('causal');
  if (s.sub === 'act') discover('swiglu');
  if (v === 'mlp') setTimeout(() => discover('sparse'), 1500);
  if (v === 'neuron' && s.mi === 'gate') discover('neuron');
  if (s.ph === 'head' && (!s.sub || s.sub === 'unembed') && d >= 2) discover('tied');
  if (s.ph === 'sample' && d >= 2) discover('sampling');
  checkSink();
}

function checkSink() {
  if (!tl || (tl.view !== 'attn' && tl.view !== 'dot')) return;
  const s = tl.step, i = Q.row(tl.g);
  const h = typeof ctx.head === 'number' ? ctx.head : null;
  if (h == null) return;
  const top = Q.att(s.L, h, i)[0];
  if (top && top.j === 0 && top.w > 0.5) discover('sink');
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

function renderStrip() {
  if (!cur) return;
  const n = tl ? tl.g : cur.q.replyTokens.length;
  $('#strip').innerHTML = `<span class="eyebrow" style="margin-right:6px">回复</span>${cur.q.replyTokens.slice(0, n).map((t) => esc(t.s)).join('')}<span class="caret" style="display:inline-block;width:6px;height:1em;background:var(--amber);margin-left:2px"></span>`;
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
  $('#btnChatToggle').addEventListener('click', () => document.body.classList.remove('chat-open'));
  $('#strip').addEventListener('click', () => document.body.classList.add('chat-open'));
}

function bindKeys() {
  addEventListener('keydown', (e) => {
    if (e.target.closest('input, textarea')) return;
    if (e.key === 'Escape' && $('#codex').classList.contains('on')) return closeCodex();
    if (mode !== 'inspect') return;
    if (e.key === ' ') { e.preventDefault(); toggle(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); tl?.pause(); tl?.next(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); tl?.pause(); tl?.back(); }
    else if (e.key === '+' || e.key === '=' || e.key === 'ArrowDown') { e.preventDefault(); into(); }
    else if (e.key === '-' || e.key === '_' || e.key === 'ArrowUp' || e.key === 'Escape') { e.preventDefault(); out(); }
    else if (e.key >= '1' && e.key <= '6') { const s = SPEEDS[Number(e.key) - 1]; store.speed = s; if (tl) { tl.speed = s; controls.updatePlay(tl); } }
  });
}

window.__bb = { get tl() { return tl; }, get Q() { return Q; }, get machine() { return machine; }, get engine() { return engine; }, into, out, ctx };
boot();
