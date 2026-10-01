import { loadManifest, loadQuestion, loadVision } from './data.js';
import { Chat } from './chat.js';
import { Timeline } from './timeline.js';
import { Controls, SPEEDS } from './controls.js';
import { Monitor } from './monitor.js';
import { INSIGHTS, INSIGHT_BY_ID } from './insights.js';
import { Background } from '../../js/bg.js';
import { sfx, setSound, soundOn } from '../../js/audio.js';
import { $, esc, tokPlain, fmtPct, sleep } from '../../js/ui.js';
import { initPanes } from '../../js/resize.js';

const KEY = 'blackbox:mm:v1';
const ROLE_NAME = { system: '系统提示', user: '你的问题', assistant: '模型的回答', tpl: '模板 / 特殊标记', img: '视觉词元' };

function load() { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } }
const saved = load();
const store = { found: new Set(saved.found || []), speed: saved.speed || 1 };
function save() { try { localStorage.setItem(KEY, JSON.stringify({ found: [...store.found], speed: store.speed, sound: soundOn() })); } catch { /* 忽略 */ } }
if (saved.sound) setSound(true);
window.__sfx = sfx;

let manifest, chat, controls, monitor;
let engine = null, scene = null;
let tl = null, cur = null, Q = null;
let mode = 'chat';
let dAnim = 1, lastView = '', lastDepth = 1, lastKey = '';
let streaming = null;
const ctx = { vq: null };

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
    depth: (d) => { if (!tl) return; if (d === 0) exitInspect(); else { d > tl.depth ? sfx.dive() : sfx.rise(); engine?.exitFree(); tl.setDepth(d); } },
    fold: () => updateInsets(),
  }, manifest);
  monitor = new Monitor({
    onPickToken: (m) => { ctx.vq = m; if (tl) controls.update(tl, ctx); },
    onChange: () => { if (tl) { controls.update(tl, ctx); if (monitor.mode.lens) discover('lens'); } },
  });
  chat = new Chat(manifest, { onSend, onPeek: (m) => enterInspect(m), onPlus, heatFor });
  $('#cxFoot').textContent = footText();
  bindKeys();
  bindChrome();
  renderCodexCount();
}

function footText() {
  const m = manifest.model, v = m.vision, t = m.text;
  return `模型：Qwen3-VL-2B-Instruct（视觉编码器 ${v.depth} 层 · ${v.hidden} 维 · ${v.heads} 头 · 图块 ${v.patch}×${v.patch} · 2×2 合并；语言模型 ${t.layers} 层 · ${t.hidden} 维 · ${t.heads} Q / ${t.kvHeads} KV 头 · SwiGLU ${t.ffn} · 词表 ${t.vocab}）。回答、分词、概率、注意力、特征和权重都来自真实模型的一次离线运行（贪心解码）；为控制体积，ViT 的完整注意力只导出了 5 层，并合并到 2×2 词元的分辨率。`;
}

/* ---------------------------------------------------------------- 聊天 */

// 聊天里悬停回复的字：生成它时，对准层（16 头平均）在图片上的注意力
async function heatFor(q, i) {
  const QQ = await loadQuestion(q, manifest);
  const [a, b] = manifest.groundLayers;
  discover('hover');
  return { img: QQ.V.img, values: QQ.attAvg(i, [a, b]), rows: QQ.V.mh, cols: QQ.V.mw, label: `第 ${a}–${b} 层 · 16 头平均` };
}

function onSend(q, im) {
  const userEl = chat.addUser(q, im);
  const msg = chat.addBot(q, userEl);
  loadQuestion(q, manifest).catch(() => { /* 揭开时再报错 */ });
  if (mode === 'inspect') attach(msg);
  else streamNormal(msg, 0);
}

// 不揭开的时候，就像平常的 AI 一样流式输出
function streamNormal(msg, from) {
  stopStreaming();
  chat.setBusy(true);
  const s = (streaming = { msg, n: from, alive: true });
  (async () => {
    if (from === 0) await sleep(700);
    while (s.alive && s.n < msg.q.replyTokens.length) {
      s.n++;
      msg.render(s.n, { fresh: true });
      await sleep(40 + Math.random() * 45);
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
  const eng = await import('../../js/stage/engine.js');
  const sc = await import('./scene.js');
  engine = new eng.Engine($('#gl'), { onFrame, onHover, onPick, onFreeChange: (f) => { $('#btnFollow').hidden = !f; if (f) showHint(false); } });
  // 只让真正亮的东西泛一点光，不要“背景光源”似的辉光
  engine.bloom.strength = 0.22;
  engine.bloom.threshold = 0.8;
  // 左边还有监视器：共享的 Engine 只考虑右侧和底部的遮挡，这里在实例上补上左侧
  engine.insetL = 0;
  engine.applyInsets = function applyInsets() {
    const w = this.w, h = this.h;
    if (!w) return;
    this.camera.setViewOffset(w, h, ((this.insetR || 0) - (this.insetL || 0)) / 2, (this.insetB || 0) / 2, w, h);
    this.camera.updateProjectionMatrix();
  };
  engine.fitDistance = function fitDistance(width, height, margin = 1.15) {
    const vw = Math.max(200, this.w - (this.insetR || 0) - (this.insetL || 0)), vh = Math.max(200, this.h - (this.insetB || 0));
    const vf = (this.camera.fov * Math.PI) / 180;
    const tv = Math.tan(vf / 2) * (vh / this.h);
    const th = Math.tan(vf / 2) * (vw / this.h);
    return Math.max((height * margin) / 2 / tv, (width * margin) / 2 / th);
  };
  scene = new sc.Scene(engine);
}

async function enterInspect(msg) {
  const from = streaming?.msg === msg ? stopStreaming() : null;
  if (mode !== 'inspect') {
    mode = 'inspect';
    document.body.classList.replace('mode-chat', 'mode-inspect');
    $('#loading').hidden = false;
    try { await ensureEngine(); } catch (e) { console.error(e); $('#loading').innerHTML = `3D 舞台初始化失败：${esc(e.message)}`; return; }
    engine.active = true;
    if (!store.hinted) { store.hinted = true; setTimeout(() => showHint(true), 1800); }
    if (matchMedia('(max-width: 900px)').matches) {
      if (!$('#dbg').classList.contains('folded')) { $('#dbg').classList.add('folded'); $('#btnDbgFold').textContent = '+'; }
    }
    updateInsets();
    sfx.dive();
  }
  await attach(msg, from);
}

async function attach(msg, from = null) {
  $('#loading').hidden = false;
  if (tl) tl.pause();
  let q;
  try { q = await loadQuestion(msg.q, manifest); } catch (e) { $('#loading').innerHTML = `数据加载失败：${esc(e.message)}`; return; }
  $('#loading').hidden = true;
  if (cur && cur !== msg && !cur.done) streamNormal(cur, cur.shown < 0 ? 0 : cur.shown);
  const fresh = Q !== q;
  Q = q;
  cur = msg;
  chat.markInspected(msg);
  const prevDepth = tl ? tl.depth : 1;
  const v = manifest.model.vision;
  tl = new Timeline(Q, { NV: v.depth, NL: manifest.model.text.layers, focus: manifest.focusLayers, deep: v.deepstack.length });
  tl.speed = store.speed;
  if (prevDepth > 1) tl.setDepth(prevDepth);
  if (from != null && from > 0) tl.seekToken(Math.min(from, Q.G - 1));
  tl.on((type) => {
    if (type === 'step') { controls.update(tl, ctx); discoverFor(); }
    if (type === 'token') { cur.render(tl.g, { cur: true, fresh: true }); renderStrip(); }
    if (type === 'play') controls.updatePlay(tl);
    if (type === 'end') { cur.finish(); controls.updatePlay(tl); }
  });
  engine.exitFree();
  if (fresh) {
    ctx.vq = null;
    scene.load(Q);
    monitor.set(Q);
    const far = scene.camera({ view: 'box', step: tl.step, g: tl.g });
    engine.setView(far.pos.clone().multiplyScalar(1.6), far.look, { snap: true });
    dAnim = tl.depth;
  }
  msg.render(tl.g, { cur: true });
  if (msg.done) msg.done = false;
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
  monitor.tip(null);
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
  engine?.exitFree();
  tl.into();
}

function out() {
  if (!tl) return;
  if (tl.depth <= 1) return exitInspect();
  sfx.rise();
  engine?.exitFree();
  tl.out();
}

let hintTimer = 0;
function showHint(on = true) {
  const h = $('#stageHint');
  clearTimeout(hintTimer);
  h.classList.toggle('on', on);
  if (on) hintTimer = setTimeout(() => h.classList.remove('on'), 9000);
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
  const mon = $('#mon');
  engine.insetL = small || mon.classList.contains('folded') ? 0 : mon.offsetWidth + 14;
  engine.setInsets(small || folded ? 0 : dbg.offsetWidth + 28, small ? (folded ? 120 : 200) : 90);
}
addEventListener('resize', () => updateInsets());

function flashBtn(sel) {
  $(sel).animate([{ transform: 'translateX(0)' }, { transform: 'translateX(-4px)' }, { transform: 'translateX(4px)' }, { transform: 'translateX(0)' }], { duration: 260 });
}

/* ---------------------------------------------------------------- 每帧 */

function onFrame(dt, t) {
  if (!tl || !scene) return;
  tl.tick(dt);
  dAnim += (tl.depth - dAnim) * (1 - Math.exp(-dt * 2.6));
  const view = tl.view, s = tl.step;
  const st = { step: s, p: tl.p, g: tl.g, depth: tl.depth, dAnim, view, vq: ctx.vq };
  scene.update(st, dt, t);
  monitor.draw(st);
  const cam = scene.camera(st);
  const key = `${view}|${tl.depth}|${s.L}|${s.sub}`;
  const same = view === lastView && tl.depth === lastDepth;
  engine.setView(cam.pos, cam.look, { speed: same ? 2.6 : 2.0 });
  if (key !== lastKey) { lastView = view; lastDepth = tl.depth; lastKey = key; }
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
  const V = Q.V;
  let h = '';
  switch (info.type) {
    case 'tile': { const t = Q.tokens[info.i]; const p = Q.pos(info.i); h = `<span class="k">词元 #${info.i} · ${ROLE_NAME[t.role] || ''}</span><b>${esc(tokPlain(t.s))}</b>　编号 <span class="v">${t.id}</span><br>位置 (t, h, w) = (${p.join(', ')})`; break; }
    case 'vcell': { const i = V.vs + info.m, p = Q.pos(i); h = `<span class="k">视觉词元 #${i} · 第 ${Math.floor(info.m / V.mw)} 行第 ${info.m % V.mw} 列</span>&lt;|image_pad|&gt; 的位置上换成了合并器的输出<br>位置 (t, h, w) = <span class="v">(${p.join(', ')})</span>`; break; }
    case 'slab': h = `<span class="k">语言模型第 ${info.L} 层</span>生成当前这个字时把 <span class="v">${fmtPct(Q.mass(tl.g, info.L))}</span> 的注意力给了图片<br><span class="v">点击跳到这一层</span>`; break;
    case 'vlevel': {
      const r = Math.floor(info.i / V.gw), c = info.i % V.gw;
      h = `<span class="k">ViT ${info.k === 0 ? '图块嵌入' : `第 ${info.k - 1} 层之后`} · 图块 (${r}, ${c})</span>颜色 = 这一级特征的前三个主成分<br><span class="v">点击跳到这一层</span>`;
      break;
    }
    case 'photo': h = `<span class="k">模型看到的图片</span>${V.gw * 16}×${V.gh * 16} 像素 · ${V.gh}×${V.gw} 个图块`; break;
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
  } else if (info.type === 'vlevel') {
    tl.pause();
    if (tl.g !== 0) tl.seekToken(0);
    if (tl.depth < 3) tl.setDepth(3);
    if (info.k === 0) tl.seek((st) => st.ph === 'vit' && st.sub === 'pos');
    else tl.seek((st) => st.ph === 'vit' && st.L === info.k - 1);
    const r = Math.floor(info.i / Q.V.gw), c = info.i % Q.V.gw;
    ctx.vq = Q.V.tokOfPatch(r, c);
    controls.update(tl, ctx);
    sfx.click();
  }
}

/* ---------------------------------------------------------------- 知识碎片 */

function discoverFor() {
  const s = tl.step, d = tl.depth, v = tl.view;
  if (s.ph === 'prep' && s.sub === 'resize') discover('resize');
  if (s.ph === 'prep' && (s.sub === 'patch' || d === 2)) discover('patch');
  if (s.ph === 'prep' && s.sub === 'norm') discover('frames');
  if (v === 'micro' && s.mi === 'sum') discover('conv');
  if (s.ph === 'vit' && s.op === 'attn') discover('noncausal');
  if (s.ph === 'vit' && s.L >= 18 && !s.op) discover('semantic');
  if (s.ph === 'merge' && (s.sub === 'group' || d === 2)) discover('merge');
  if ((s.ph === 'merge' && s.sub === 'deep') || s.op === 'deep') discover('deepstack');
  if (s.ph === 'splice' && (s.sub === 'template' || d === 2)) discover('imgpad');
  if (s.ph === 'splice' && s.sub === 'template') setTimeout(() => discover('imgfirst'), 1600);
  if (s.ph === 'splice' && s.sub === 'mrope') discover('mrope');
  if (s.ph === 'read') discover('kvimg');
  if (s.ph === 'layer' && !s.op && s.L >= manifest.groundLayers[0] && d >= 3) discover('ground');
  if (v === 'heads') discover('heads');
  if (s.ph === 'head' && s.sub === 'pick') discover('greedy');
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
  $('#btnFollow').addEventListener('click', () => { engine?.exitFree(); sfx.click(); });
  $('#btnHint').addEventListener('click', () => showHint(!$('#stageHint').classList.contains('on')));
  $('#btnMonFold').addEventListener('click', () => updateInsets());
  initPanes('mm', { left: { el: '#chat', v: '--chat-w', name: '聊天栏' }, right: { el: '#dbg', v: '--dbg-w', name: '调试器' }, reserve: 600, onChange: updateInsets });
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
    else if (e.key === 'f' || e.key === 'F') { engine?.exitFree(); }
    else if (e.key === 'h' || e.key === 'H' || e.key === '?') { showHint(!$('#stageHint').classList.contains('on')); }
    else if (e.key >= '1' && e.key <= '6') { const s = SPEEDS[Number(e.key) - 1]; store.speed = s; save(); if (tl) { tl.speed = s; controls.updatePlay(tl); } }
  });
}

window.__mm = { get tl() { return tl; }, get Q() { return Q; }, get scene() { return scene; }, get engine() { return engine; }, get monitor() { return monitor; }, into, out, ctx, loadVision };
boot();
