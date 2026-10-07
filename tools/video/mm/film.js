// 多模态电影模式：用多模态页的 3D 舞台（public/multimodal/js/scene.js）和 Qwen3-VL-2B 的真实导出数据，
// 按脚本化的时间线驱动舞台和摄影机，HTML 叠加字幕和讲解面板。window.__film.renderAt(t) 渲染第 t 秒，逐帧调用即可确定地出片。
//
// 片子讲的是「桌上的苹果」+「绿色的苹果是第几个？」这一次真实运行（贪心解码，回答「绿色的苹果是第3个。」）：
//   图片怎么被缩放、切成 16×16 的图块（一个图块 × 卷积核 → 嵌入的一个数，全片唯一一次逐数计算），穿过 ViT 24 层，
//   2×2 合并成 140 个视觉词元（外加 DeepStack），带着 M-RoPE 三维位置插进对话；语言模型生成「3」时，
//   第 16–26 层的注意力对准了绿苹果，逻辑透镜先读出「三个」；图片位置的透镜读数是「五个」「红色」「苹果」。
import { FilmEngine, THREE } from '../lib/engine.js';
import { clamp, lerp, seg, smooth, smoother, easeOut, easeInOut } from '../lib/cam.js';
import { el, CueLayer } from '../lib/overlay.js';
import { buildScore, G } from './score.js';
import { Opening } from './opening.js';
import { Monitor, CalcBoard, LensCard } from './panel.js';
import { loadManifest, loadQuestion } from '/public/multimodal/js/data.js';
import { tokPlain, shortSpecial, esc } from '/public/js/ui.js';

const params = new URLSearchParams(location.search);
const FPS = Number(params.get('fps') || 30);
const QID = params.get('q') || 'apples-2';
const PREVIEW = params.has('preview');
const $ = (s) => document.querySelector(s);

let E, S, Q, MAN, SC, OPENING, QR_SVG = '', CAP = null;
const SITE = new Map();
const OV = {};

/* ================================================================ 启动 */

async function boot() {
  await Promise.all([
    document.fonts.load('900 100px "Film Serif"', 'AI 是怎么看图的'),
    document.fonts.load('600 40px "Film Serif"', '绿色的苹果是第几个'),
    document.fonts.load('400 38px "Film Sans"', '绿色的苹果是第几个？'),
    document.fonts.load('600 38px "Film Sans"', 'Qwen'),
    document.fonts.load('600 76px "PingFang SC"', '绿色苹果'),
    document.fonts.load('600 76px "Noto Sans SC"', '绿色'),
    document.fonts.load('400 20px "JetBrains Mono"', '0123'),
    document.fonts.load('700 20px "JetBrains Mono"', '0123'),
  ]);
  QR_SVG = await (await fetch('/tools/video/mm/qr-multimodal.svg')).text().catch(() => '');
  MAN = await loadManifest();
  const spec = MAN.images.flatMap((im) => im.questions).find((q) => q.id === QID);
  Q = await loadQuestion(spec, MAN);
  CAP = await fetch('/ext/mm/sitecap/meta.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
  SC = buildScore(Q, CAP);
  if (CAP && SC.endSite) await preloadSite();

  E = new FilmEngine($('#gl'), { pixelRatio: Number(params.get('pr') || 1.25) });
  E.shiftY = 56; // 主体整体上移，给字幕留位置
  E.applyOffset();
  // 只让真正亮的东西泛一点光（ViT 塔上的主成分颜色很鲜艳，阈值低了会糊成一团白）
  E.bloom.strength = 0.26;
  E.bloom.threshold = 0.78;
  const { Scene } = await import('/public/multimodal/js/scene.js');
  S = new Scene(E);
  S.load(Q);
  tameScene();
  buildOverlays();
  OPENING = new Opening({ E, S, Q, frame: $('#frame'), T: SC.open, imgId: Q.V.spec.id });
  window.__film = { S, E, Q, SC, O: OPENING, probe, poster, ready: true, fps: FPS, duration: SC.end, renderAt, seek, events: () => SC.events.slice().sort((a, b) => a.t - b.t), score: { sections: SC.sections, shots: SC.shots, bpm: SC.bpm, end: SC.end, subs: SC.subs, terms: SC.terms } };
  if (PREVIEW) startPreview();
}

// 舞台在电影的灯光 / 泛光下太亮的地方调暗一些
function tameScene() {
  S.root.traverse((o) => {
    const mt = o.material;
    if (mt && mt.isMeshStandardMaterial && mt.roughness < 0.5 && !o.parent?.isGroup) { mt.roughness = Math.max(mt.roughness, 0.6); }
  });
  for (const o of E.scene.children) {
    if (o.isMesh && o.geometry?.type === 'CircleGeometry') o.material.opacity = 0.5;
    if (o.isPoints) { o.material.opacity = 0.16; o.material.size = 0.035; }
    if (o.type === 'GridHelper') o.material.opacity = 0.2;
  }
  // 舞台里的 3D 标签按网站的小字排版：片子里一律收起，讲解交给字幕和面板（个别镜头单独打开）
  for (const sl of S.slabs) { sl.material.color.setHex(0x2b3f73); sl.material.roughness = 0.75; sl.material.metalness = 0.05; }
  S.labels = { photo: S.photoLbl, vit: S.vitLbl, lvl: S.lvlLbl, merge: S.mergeLbl, llm: S.llmLbl, layer: S.layerLbl, out: S.outLbl, ds: S.dsLbl };
}

/* ================================================================ 叠加层 */

function buildOverlays() {
  const ov = $('#ov');
  OV.band = el('div', 'band', '', ov);
  OV.subs = new CueLayer(ov, 'sub', SC.subs, { rise: 8 });
  OV.terms = new CueLayer(ov, 'term', SC.terms, { rise: 6, fin: 0.3, fout: 0.3 });
  OV.prog = el('div', 'prog', SC.chapterNames.map((c, i) => `${i ? '<span class="ln"></span>' : ''}<span class="it"><i>${i + 1}</i>${c}</span>`).join(''), ov);
  OV.progIt = [...OV.prog.querySelectorAll('.it')];
  OV.progKey = '';
  const mm = MAN.model;
  OV.title = el('div', 'title', `<div class="eb">INSIDE A VISION-LANGUAGE MODEL</div><h1><span>AI 是怎么看图的</span></h1><div class="rule"></div><div class="st">一张图，怎么变成模型读得懂的“词”</div><div class="spec">QWEN3-VL-2B · VISION ${mm.vision.depth} + LANGUAGE ${mm.text.layers} LAYERS · ${mm.params.total.toLocaleString('en-US')} PARAMETERS</div>`, ov);
  OV.reply = el('div', 'reply', '', ov);
  OV.replyKey = '';
  OV.mon = new Monitor(ov, Q, SC.F);
  OV.calc = new CalcBoard(ov, Q);
  OV.lens3 = new LensCard(ov, Q, SC.F);
  OV.dims = el('div', 'dims', '<div class="dw"><span></span></div><div class="dh"><span></span></div>', ov);
  // 片尾：多模态页录屏放在一个干净的窗口框里（不模仿任何浏览器），鼠标和点击是叠加上去的；最后落版网址 + 二维码
  OV.win = el('div', 'win', `<div class="bar"><i></i><i></i><i></i><span class="url">caijiechao.com/blackbox/multimodal/</span></div><canvas width="1920" height="1080"></canvas>`, ov);
  OV.winCtx = OV.win.querySelector('canvas').getContext('2d');
  OV.winIdx = -1;
  OV.cur = el('div', 'cur', `<span class="rip"></span><svg viewBox="0 0 24 24" width="30" height="30"><path d="M5 2.5v17.2l4.6-4.3 3 6.6 2.9-1.3-3-6.5h6.2z" fill="#fff" stroke="#05080f" stroke-width="1.4" stroke-linejoin="round"/></svg>`, ov);
  OV.fin = el('div', 'fin7', `<div class="url">caijiechao.com/blackbox/multimodal/</div><div class="qr">${QR_SVG}</div><div class="hint">扫码打开多模态页，亲手看它怎么看图</div>`, ov);
}

function updateProg(t) {
  const p = SC.progAt(t);
  OV.prog.style.display = p.a > 0.001 ? 'flex' : 'none';
  if (p.a <= 0.001) return;
  OV.prog.style.opacity = p.a.toFixed(3);
  const key = String(p.i);
  if (key === OV.progKey) return;
  OV.progKey = key;
  OV.progIt.forEach((e, i) => { e.className = `it${i === p.i ? ' on' : i < p.i ? ' done' : ''}`; });
}

const toScreen = (v) => { const p = v.clone().project(E.camera); return { x: ((p.x + 1) / 2) * 1920, y: ((1 - p.y) / 2) * 1080 }; };

// 片名：落在黑箱正面
function updateTitle(o) {
  const ta = o.title ?? 0;
  OV.title.style.display = ta > 0.001 ? 'block' : 'none';
  if (ta <= 0.001) return;
  const k = o.titleK ?? 1;
  OV.title.style.opacity = ta.toFixed(3);
  const h1 = OV.title.querySelector('h1');
  h1.style.letterSpacing = `${(0.2 - 0.12 * easeOut(k)).toFixed(4)}em`;
  h1.style.paddingLeft = h1.style.letterSpacing;
  OV.title.style.filter = o.titleBlur ? `blur(${o.titleBlur.toFixed(2)}px)` : '';
  E.camera.updateMatrixWorld();
  const fr = S.cFront;
  const Wf = fr.geometry.parameters.width;
  // 正面板右边那大半截的中心（左边是取景窗）
  const c = fr.localToWorld(new THREE.Vector3(Wf * 0.12, fr.geometry.parameters.height * 0.06, 0.05));
  const p = toScreen(c);
  OV.title.style.left = `${p.x.toFixed(1)}px`;
  OV.title.style.top = `${p.y.toFixed(1)}px`;
  const l = toScreen(fr.localToWorld(new THREE.Vector3(-Wf * 0.22, 0, 0))), r = toScreen(fr.localToWorld(new THREE.Vector3(Wf * 0.46, 0, 0)));
  const fit = Math.min(1, (0.9 * Math.hypot(r.x - l.x, r.y - l.y)) / Math.max(1, OV.title.offsetWidth));
  OV.title.style.transform = `translate(-50%, -50%) scale(${fit.toFixed(4)})`;
  OV.title.querySelector('.rule').style.width = `${(560 * easeInOut(seg(k, 0.15, 0.8))).toFixed(1)}px`;
  OV.title.querySelector('.st').style.opacity = smooth(seg(k, 0.35, 0.8)).toFixed(3);
  OV.title.querySelector('.spec').style.opacity = (0.9 * smooth(seg(k, 0.55, 1))).toFixed(3);
  OV.title.querySelector('.eb').style.opacity = smooth(seg(k, 0.0, 0.5)).toFixed(3);
}

// 照片的尺寸标注：缩放时宽高数字变化，切块时变成“28 块 / 20 块”
function updateDims(D) {
  const on = D && D.a > 0.001;
  OV.dims.style.display = on ? 'block' : 'none';
  if (!on) return;
  const V = Q.V, pw = V.gw * G.PS, ph = V.gh * G.PS;
  E.camera.updateMatrixWorld();
  const tl = toScreen(new THREE.Vector3(G.PX - pw / 2, G.PY + ph / 2, 0)), br = toScreen(new THREE.Vector3(G.PX + pw / 2, G.PY - ph / 2, 0));
  const gap = 1 + 0.18 * (D.cut ?? 0);
  const cx = (tl.x + br.x) / 2, cy = (tl.y + br.y) / 2;
  const w = (br.x - tl.x) * gap, h = (br.y - tl.y) * gap;
  OV.dims.style.opacity = D.a.toFixed(3);
  const dw = OV.dims.querySelector('.dw'), dh = OV.dims.querySelector('.dh');
  dw.style.left = `${(cx - w / 2).toFixed(1)}px`; dw.style.width = `${w.toFixed(1)}px`; dw.style.top = `${(cy - h / 2 - 54).toFixed(1)}px`;
  dh.style.top = `${(cy - h / 2).toFixed(1)}px`; dh.style.height = `${h.toFixed(1)}px`; dh.style.left = `${(cx - w / 2 - 54).toFixed(1)}px`;
  const [W0, H0] = V.spec.orig, W1 = V.gw * 16, H1 = V.gh * 16;
  const r = smooth(D.resize ?? 1);
  const cut = (D.cut ?? 0) > 0.02;
  const tw = cut ? `<b>${V.gw}</b> 块` : `<b>${Math.round(lerp(W0, W1, r))}</b> 像素`;
  const th = cut ? `<b>${V.gh}</b> 块` : `<b>${Math.round(lerp(H0, H1, r))}</b> 像素`;
  dw.firstChild.innerHTML = tw;
  dh.firstChild.innerHTML = th;
}

// 回答一个字一个字长出来；hl = 正在讲的那个字（琥珀色）
function updateReply(R) {
  OV.reply.style.display = R && R.a > 0.001 ? 'block' : 'none';
  if (!R || R.a <= 0.001) return;
  OV.reply.style.opacity = R.a.toFixed(3);
  const key = `${R.n}|${R.hl}|${R.big ? 1 : 0}`;
  if (key === OV.replyKey) return;
  OV.replyKey = key;
  OV.reply.classList.toggle('big', !!R.big);
  const toks = Q.steps.slice(0, R.n).map((st, g) => {
    const sp = /^<\|/.test(st.chosenS);
    return `<span class="t${g === R.hl ? ' hl' : ''}${sp ? ' sp' : ''}${g === R.n - 1 && R.hl < 0 ? ' new' : ''}">${sp ? esc(shortSpecial(st.chosenS)) : esc(tokPlain(st.chosenS))}</span>`;
  });
  OV.reply.innerHTML = `<span class="q">绿色的苹果是第几个？</span><span class="a">${toks.join('')}${R.n < Q.G ? '<span class="caret"></span>' : ''}</span>`;
}

// 片尾录屏：按时间表换帧；窗口最后缩到左边，右边落版网址 + 二维码
async function preloadSite() {
  const need = new Set();
  for (const sg of SC.endSite.segs) for (let f = sg.f0; f <= sg.f1 + 1e-6; f += 1 / 60) need.add(Math.min(CAP.n - 1, Math.max(0, Math.round(lerp(sg.c0, sg.c1, (f - sg.f0) / (sg.f1 - sg.f0)) * CAP.fps))));
  await Promise.all([...need].map((k) => new Promise((res) => { const im = new Image(); im.onload = () => res(); im.onerror = () => res(); im.src = `/ext/mm/sitecap/f${String(k).padStart(4, '0')}.jpg`; SITE.set(k, im); })));
}
function cursorAt(ct) {
  const acts = CAP.actions.filter((a) => a.x != null);
  let prev = acts[0], next = null;
  for (const a of acts) { if (a.t <= ct) prev = a; else { next = a; break; } }
  if (next && next.t - ct < 0.55 && prev !== next) { const k = easeInOut(1 - (next.t - ct) / 0.55); return { x: lerp(prev.x, next.x, k), y: lerp(prev.y, next.y, k), click: 0 }; }
  const click = ct >= prev.t && ct < prev.t + 0.45 && prev.click !== false ? 1 - (ct - prev.t) / 0.45 : 0;
  return { x: prev.x, y: prev.y, click };
}
function updateSite(Sx) {
  const on = Sx && Sx.a > 0.001 && CAP;
  OV.win.style.display = on ? 'block' : 'none';
  OV.cur.style.display = on && Sx.cursor > 0.01 ? 'block' : 'none';
  OV.fin.style.display = on && Sx.finA > 0.001 ? 'flex' : 'none';
  if (!on) return;
  const idx = Math.min(CAP.n - 1, Math.max(0, Math.round(Sx.ct * CAP.fps)));
  if (idx !== OV.winIdx && SITE.get(idx)?.complete) { OV.winCtx.drawImage(SITE.get(idx), 0, 0, 1920, 1080); OV.winIdx = idx; }
  const k = Sx.fin ?? 0;
  let cx = lerp(960, 600, k), cy = lerp(474, 482, k), sc = lerp(1, 0.6, k);
  // 推近：把录屏里的某块区域（视口坐标）放大到 z 倍、移到画面中间
  const Z = Sx.zoom;
  if (Z && Z.k > 0) {
    const W = 1440, H = 810, BAR = 40;
    const lx = -W / 2 + (Z.x / CAP.viewport.w) * W, ly = -(H + BAR) / 2 + BAR + (Z.y / CAP.viewport.h) * H;
    const s2 = sc * Z.z;
    const cx2 = 960 - lx * s2, cy2 = 470 - ly * s2;
    cx = lerp(cx, cx2, Z.k); cy = lerp(cy, cy2, Z.k); sc = lerp(sc, s2, Z.k);
  }
  OV.win.style.opacity = Sx.a.toFixed(3);
  OV.win.style.left = `${cx.toFixed(1)}px`;
  OV.win.style.top = `${cy.toFixed(1)}px`;
  OV.win.style.transform = `translate(-50%, -50%) scale(${sc.toFixed(4)}) translateY(${((1 - easeOut(Math.min(1, Sx.a * 1.2))) * 18).toFixed(1)}px)`;
  if (Sx.cursor > 0.01) {
    const c = cursorAt(Sx.ct);
    const W = 1440, H = 810, BAR = 40;
    const x = cx + (-W / 2 + (c.x / CAP.viewport.w) * W) * sc, y = cy + (-(H + BAR) / 2 + BAR + (c.y / CAP.viewport.h) * H) * sc;
    OV.cur.style.left = `${x.toFixed(1)}px`;
    OV.cur.style.top = `${y.toFixed(1)}px`;
    OV.cur.style.opacity = Sx.cursor.toFixed(3);
    const rip = OV.cur.firstChild;
    rip.style.opacity = (c.click * 0.9).toFixed(3);
    rip.style.transform = `translate(-50%, -50%) scale(${(0.4 + 1.2 * (1 - c.click)).toFixed(3)})`;
  }
  if (Sx.finA > 0.001) { OV.fin.style.opacity = Sx.finA.toFixed(3); OV.fin.style.transform = `translateY(-50%) translateY(${((1 - easeOut(Sx.finA)) * 14).toFixed(1)}px)`; }
}

function updateOverlays(t, f) {
  const o = f.ov || {};
  OV.subs.update(t);
  OV.terms.update(t);
  OV.band.style.opacity = (o.band ?? 1).toFixed(3);
  updateProg(t);
  updateTitle(o);
  updateDims(f.dims);
  updateReply(f.reply);
  OV.mon.update(f.panel);
  OV.calc.update(f.calc);
  OV.lens3.update(f.lens3);
  updateSite(o.site);
}

/* ================================================================ 颗粒 */

let grainImg = null;
function drawGrain(frame) {
  const ctx = $('#grain').getContext('2d');
  if (!grainImg) grainImg = ctx.createImageData(480, 270);
  const d = grainImg.data;
  let s = (frame * 7919 + 13) >>> 0;
  for (let i = 0; i < d.length; i += 4) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const v = s >>> 24;
    d[i] = d[i + 1] = d[i + 2] = v; d[i + 3] = 255;
  }
  ctx.putImageData(grainImg, 0, 0);
}

/* ================================================================ 舞台的逐帧调整 */

// 抽出来演示乘加的那个图块：从照片上升起、转向镜头、放大
const tmpM = new THREE.Matrix4(), tmpP = new THREE.Vector3(), tmpQ = new THREE.Quaternion(), tmpS = new THREE.Vector3();
function liftPatch(k) {
  const V = Q.V, hp = SC.hotPatch;
  if (k <= 0.0001) { if (S._lifted) { S.photoKey = ''; S._lifted = false; } return; }
  S._lifted = true;
  S.photoKey = '';
  const r = hp.r, c = hp.c;
  const gap = 1.18;
  tmpP.set(G.PX + (c - (V.gw - 1) / 2) * G.PS * gap, G.PY - (r - (V.gh - 1) / 2) * G.PS * gap, 0);
  const e = easeInOut(k);
  tmpP.lerp(new THREE.Vector3(G.PX - 0.25, G.PY + 0.25, 2.2), e);
  tmpQ.identity();
  tmpS.setScalar(1 + 5 * e);
  tmpM.compose(tmpP, tmpQ, tmpS);
  S.photo.setMatrixAt(hp.i, tmpM);
  S.photo.instanceMatrix.needsUpdate = true;
}

function adjustStage(f) {
  // 3D 标签：默认全收起；镜头里要的单独打开
  const keep = new Set(f.labels || []);
  for (const [k, lb] of Object.entries(S.labels)) if (!keep.has(k)) lb.el.classList.add('hide');
  // 语言模型的玻璃层板在电影灯光下太亮：整体压一压
  for (const sl of S.slabs) { sl.material.opacity *= 0.4; sl.material.emissiveIntensity *= 0.5; sl.edge.material.opacity *= 0.6; }
  if (f.hideHeat) { S.heat.visible = false; S.beams.visible = false; }
  if (f.photoA === 0) { S.photo.visible = false; S.photoFrame.visible = false; } else { S.photo.visible = true; S.photoFrame.visible = true; }
  // 黑箱正面的型号字：片名在正面的那几秒先不亮
  const fa = f.boxLabel ?? 1;
  for (const c of S.cFront.children) { if (!c.isMesh || !c.material.map) continue; c.material.opacity = Math.min(c.material.opacity, fa); }
  liftPatch(f.lift ?? 0);
}

/* ================================================================ 每帧 */

let lastT = null, lastF = null;
function probe() {
  const v = (o) => (o ? [o.x, o.y, o.z].map((n) => +n.toFixed(3)) : null);
  return { shot: lastF?.shot, view: lastF?.st?.view, cam: { pos: v(E.camera.position) }, bx: +S.bx.toFixed(3), bcx: +S.bcx.toFixed(3), xEnd: +S.xEnd.toFixed(3), box: S.box };
}
function renderAt(t, { render = true } = {}) {
  const dt = lastT == null || t <= lastT || t - lastT > 0.5 ? 1 / FPS : t - lastT;
  lastT = t;
  const f = (lastF = SC.frame(t, { S, E, Q }));
  S.update(f.st, dt, t);
  adjustStage(f);
  OPENING.update(t, SC.open.camAt);
  const sx = f.shiftX ?? (f.panel && f.panel.a > 0.01 ? -330 * smooth(f.panel.a) : 0);
  if (Math.abs(sx - E.shiftX) > 0.5) { E.shiftX = sx; E.applyOffset(); }
  const cam = f.cam();
  E.setCamera(cam.pos, cam.look, cam.fov ?? 32, cam.roll ?? 0);
  if (f.dof) E.setDof(f.dof.focus ?? cam.pos.distanceTo(cam.look), f.dof.range ?? 3, f.dof.blur ?? 0); else E.setDof(0, 1, 0);
  E.renderer.toneMappingExposure = f.exposure ?? 0.86;
  $('#fade').style.opacity = (f.fade ?? 0).toFixed(4);
  updateOverlays(t, f);
  if (!render) return;
  drawGrain(Math.round(t * FPS));
  E.render();
}

// 跳到 t：先从 t - pre 秒起不出图地模拟一遍
function seek(t, pre = 4) {
  lastT = null;
  const t0 = Math.max(0, t - pre);
  for (let x = t0; x < t - 1e-6; x += 1 / FPS) renderAt(x, { render: false });
  renderAt(t);
}

// 封面：片名落版那一刻（去掉字幕）
function poster(t) {
  seek(t ?? SC.open.titleIn + 2.4);
  for (const sel of ['.sub', '.term', '.prog']) document.querySelectorAll(sel).forEach((e) => { e.style.display = 'none'; });
  E.render();
}

function startPreview() {
  const bar = $('#scrub');
  bar.hidden = false;
  const r = $('#scrubber');
  r.max = SC.end;
  let playing = true, t = Number(params.get('t') || 0), last = performance.now();
  seek(t);
  r.addEventListener('input', () => { t = Number(r.value); seek(t, 2); });
  addEventListener('keydown', (e) => { if (e.key === ' ') playing = !playing; });
  const loop = (now) => {
    const d = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (playing) { t += d; if (t > SC.end) t = 0; renderAt(t); r.value = t; }
    $('#scrubT').textContent = t.toFixed(2);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

boot().catch((e) => { console.error(e); document.body.insertAdjacentHTML('beforeend', `<pre style="color:#f66;position:fixed;top:0;left:0">${esc(e.stack || e.message)}</pre>`); window.__film = { error: String(e.stack || e) }; });
