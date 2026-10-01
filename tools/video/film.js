// 电影模式：用网站的 3D 舞台（public/js/stage/）和真实的 Qwen3-0.6B 数据，按脚本化的时间线
// 驱动机器和摄影机，HTML 叠加字幕。window.__film.renderAt(t) 渲染第 t 秒的画面，逐帧调用即可确定地出片。
//
// 片子讲的是「天空为什么是蓝色的？」这一次真实推理：
//   第 1 个词「天空」完整走一遍（分词、嵌入、28 层；逻辑透镜在第 21 层说「因为」，第 24 层改口成「天空」），
//   停在第 24 层拆开（第 6 头把 78% 的注意力投向问题里的「天空」），一直拆到乘加和 bf16 比特；
//   再拉回输出头和采样，自回归写完整句回答，中间放慢看关键词「散」是怎么在最后一层才定下来的。
import { FilmEngine, THREE } from './lib/engine.js';
import { path, blendCam, orbit, handheld, clamp, lerp, seg, smooth, smoother, easeOut, easeIn, easeInOut, v3, pchip } from './lib/cam.js';
import { el, vis, CueLayer } from './lib/overlay.js';
import { buildScore } from './score.js';
import { loadManifest, loadQuestion, loadThumbs } from '/public/js/data.js';
import { viewOf } from '/public/js/timeline.js';
import { label } from '/public/js/stage/engine.js';
import { tokPlain, shortSpecial, fmtPct, esc } from '/public/js/ui.js';
import { bf16Bits } from '/public/js/num.js';

const params = new URLSearchParams(location.search);
const FPS = Number(params.get('fps') || 30);
const QID = params.get('q') || 'q01';
const PREVIEW = params.has('preview');
const $ = (s) => document.querySelector(s);

let E, M, Q, MAN, SC;
const extras = {};
const EVENTS = []; // 给配乐 / 音效用的事件（时间点），渲染脚本会把它导出成 events.json

/* ================================================================ 小工具 */

const pct = (p) => fmtPct(p);
const tk = (s) => esc(tokPlain(s));
const q = (s) => `「${tk(s)}」`;
const m = (s) => `<span class="m">${s}</span>`;
const num = (v, d = 2) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
function hash01(n) { let x = (n + 1) * 2654435761 >>> 0; x ^= x >>> 15; x = Math.imul(x, 2246822507) >>> 0; x ^= x >>> 13; return (x >>> 0) / 4294967296; }

// 机器的状态：和网站调试器里的“步骤 + 进度”一样，交给 Machine.update
function mst(depth, g, step, p, o = {}) {
  const s = { g, ...step };
  return { step: s, p: clamp(p), g, depth, dAnim: o.dAnim ?? depth, view: o.view ?? viewOf(depth, s), head: o.head ?? null };
}

// 分段时间表：[{ key, t0, t1 }] → 当前段和段内进度
function schedule(list, t) {
  for (let i = 0; i < list.length; i++) {
    const a = list[i];
    if (t < a.t1 || i === list.length - 1) return { ...a, i, p: clamp((t - a.t0) / (a.t1 - a.t0)) };
  }
  return null;
}

/* ================================================================ 启动 */

async function boot() {
  await Promise.all([
    document.fonts.load('900 100px "Film Serif"', '揭开黑箱法国'),
    document.fonts.load('600 40px "Film Serif"', '法国的首都是哪里'),
    document.fonts.load('600 76px "PingFang SC"', '法国的首都'),
    document.fonts.load('600 76px "Noto Sans SC"', '法国'),
    document.fonts.load('400 20px "JetBrains Mono"', '0123'),
    document.fonts.load('700 20px "JetBrains Mono"', '0123'),
  ]);
  MAN = await loadManifest();
  Q = await loadQuestion(QID, MAN);
  await Promise.all(Array.from({ length: Q.NL }, (_, L) => Q.ensureMicro(L)));
  const thumbs = await loadThumbs();
  SC = buildScore(Q);

  E = new FilmEngine($('#gl'), { pixelRatio: Number(params.get('pr') || 1.5) });
  E.shiftY = 56; // 主体整体上移，给字幕留位置
  E.applyOffset();
  const { Machine } = await import('/public/js/stage/machine.js');
  M = new Machine(E);
  M.brightness = 0.82;
  M.mats.setThumbs(thumbs, MAN.thumbs.index);
  M.load(Q);
  // 算式板（网站 D6 / D7 的 HTML 浮层）：放到字幕层下面；飞行的乘积用的是实时动画，逐帧渲染时关掉
  $('#frame').insertBefore(M.board.el, $('#ov'));
  M.board.fly = () => {};
  tameScene();
  buildExtras();
  buildOverlays();
  window.__film = { M, E, Q, probe, poster, ready: true, fps: FPS, duration: SC.end, renderAt, seek, events: () => SC.events.slice().sort((a, b) => a.t - b.t), score: { sections: SC.sections, shots: SC.shots, bpm: SC.bpm, end: SC.end } };
  if (PREVIEW) startPreview();
}

// 场景的“背景光”压暗：地面的径向光晕、漂浮微粒都调淡（用户不要背景光源一样的辉光）
function tameScene() {
  // 玻璃层板、外壳太光滑：主光的镜面高光叠在一起会形成一大团泛光（像背景里有个光源），把粗糙度调高
  M.root.traverse((o) => {
    const mt = o.material;
    if (mt && mt.isMeshStandardMaterial && mt.roughness < 0.5) { mt.roughness = 0.72; mt.metalness = Math.min(mt.metalness, 0.25); }
  });
  M.pulses.material.color.setHex(0x8fd9e6);
  // silu(g)⊙u 那根向量原来是纯白发光，太刺眼：换成偏冷的灰蓝
  M.mats.mlp.din.segs.forEach((m) => { m.material.emissive.setHex(0x8fb4c8); m.material.color.setHex(0x1c2a36); });
  // ⊕ 和 RMSNorm 的圆环闪一下时是纯白，泛光之后像一盏灯：换成偏冷的浅灰蓝，亮度也压一点
  for (const o of [M.exAdd1, M.exAdd2]) o.mat.color.setHex(0xa9c4d6);
  for (const o of [M.exRing1, M.exRing2]) o.material.color.setHex(0x86c6d8);
  for (const o of E.scene.children) {
    if (o.isMesh && o.geometry?.type === 'CircleGeometry') o.material.opacity = 0.32;
    if (o.isPoints) { o.material.opacity = 0.2; o.material.size = 0.04; }
    if (o.type === 'GridHelper') o.material.opacity = 0.22;
  }
}

/* ================================================================ 场景里额外的东西 */

function buildExtras() {
  // 1. 托盘方块前面的真实词元编号
  extras.ids = Q.tokens.map((t, i) => {
    const lb = label(String(t.id), `lbl tid${t.role === 'user' && !t.sp ? ' u' : ''}`);
    lb.position.set(0, -0.12, 0.3);
    lb.center.set(0.5, -0.15);
    M.tiles[i].add(lb);
    return lb;
  });

  // 2. 嵌入表：立在机器背后的一整面墙。宽 = 1024 列，高 = 151936 行（从上往下按编号排）。
  //    每个词元按真实编号在墙上亮起自己那一行，再从这一行落下一颗光点，落进托盘上的方块
  const g = (extras.emb = new THREE.Group());
  const X0 = M.x(0) - 0.5, X1 = M.x(Q.P - 1) + 0.5, Y1 = 8.6, TH = 8.0, Z = -1.12;
  const TW = X1 - X0, XC = (X0 + X1) / 2;
  const mk = (mesh, base) => { mesh.userData.base = base; g.add(mesh); return mesh; };
  const glass = mk(new THREE.Mesh(new THREE.PlaneGeometry(TW, TH), new THREE.MeshBasicMaterial({ color: 0x0a1a2c, transparent: true, opacity: 0, depthWrite: false })), 0.55);
  glass.position.set(XC, Y1 - TH / 2, Z);
  const edge = mk(new THREE.LineSegments(new THREE.EdgesGeometry(glass.geometry), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0 })), 0.5);
  edge.position.copy(glass.position);
  // 细格线：表格的“行列”示意（不代表数值）
  const lines = [];
  for (let k = 1; k < 32; k++) { const y = Y1 - (TH * k) / 32; lines.push(v3(X0, y, Z + 0.001), v3(X1, y, Z + 0.001)); }
  for (let k = 1; k < 16; k++) { const x = X0 + (TW * k) / 16; lines.push(v3(x, Y1 - TH, Z + 0.001), v3(x, Y1, Z + 0.001)); }
  mk(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(lines), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0 })), 0.07);
  const head = label(`嵌入表<small>151936 行 × 1024 列 · 每个词元按编号取一行</small>`, 'lbl part');
  head.position.set(X0, Y1 + 0.05, Z);
  head.center.set(0, 1.25);
  g.add(head);
  extras.embHead = head;
  const ax = label(`第 0 行`, 'lbl hint'); ax.position.set(X1 + 0.1, Y1, Z); ax.center.set(0, 0.5); g.add(ax);
  const ax2 = label(`第 151935 行`, 'lbl hint'); ax2.position.set(X1 + 0.1, Y1 - TH, Z); ax2.center.set(0, 0.5); g.add(ax2);
  extras.embAx = [ax, ax2];
  const roleCol = { system: 0x8fa6d6, user: 0x5ee4f0, assistant: 0xffb65c, tpl: 0xb39dff };
  extras.rows = Q.tokens.slice(0, Q.P).map((t, i) => {
    const y = Y1 - TH * (t.id / 151936);
    const c = roleCol[t.role] || roleCol.tpl;
    const row = new THREE.Mesh(new THREE.PlaneGeometry(TW, 0.03), new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0, depthWrite: false }));
    row.position.set(XC, y, Z + 0.004);
    g.add(row);
    const a = v3(M.x(i), y, Z + 0.02), b = v3(M.x(i), 0.42, 0.02);
    const curve = new THREE.QuadraticBezierCurve3(a, v3(M.x(i), (a.y + b.y) / 2, 0.55), b);
    const pk = new THREE.Mesh(new THREE.SphereGeometry(0.05, 12, 8), new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.95 }));
    g.add(pk);
    const dotA = new THREE.Mesh(new THREE.CircleGeometry(0.07, 16), new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0, depthWrite: false }));
    dotA.position.copy(a);
    g.add(dotA);
    let lb = null;
    if (t.role === 'user' && !t.sp && t.s !== 'user' && t.s !== '\n') {
      lb = label(`${t.id} <span style="color:var(--ink)">${tk(t.s)}</span>`, 'lbl emb');
      lb.position.set(M.x(i), y + 0.02, Z + 0.02);
      lb.center.set(0.5, 1.35);
      g.add(lb);
    }
    return { row, pk, dotA, curve, lb, i };
  });
  g.visible = false;
  M.root.add(g);

  // 3. 「从这里接着写」的指示
  extras.next = label('↓ 下一个词从这里接着写', 'lbl hint');
  extras.next.el.style.color = 'var(--amber)';
  extras.next.center.set(0.5, 1.2);
  M.root.add(extras.next);
  extras.next.visible = false;
}

function updateExtras(t, f) {
  const ex = f.extras || {};
  // 词元编号
  const ia = ex.ids ?? 0;
  extras.ids.forEach((lb, i) => {
    const tile = M.tiles[i];
    const on = ia > 0.01 && tile.visible && tile.position.y < 0.2;
    lb.visible = on;
    if (on) {
      const tok = Q.tokens[i];
      const isUser = tok.role === 'user' && !tok.sp && tok.s !== 'user' && tok.s !== '\n';
      lb.el.style.opacity = (ia * (ex.idsFocus ? (isUser ? 1 : 0.25) : 1)).toFixed(3);
    }
  });
  // 嵌入表
  const ea = ex.emb ?? 0;
  extras.emb.visible = ea > 0.01;
  if (extras.emb.visible) {
    for (const o of extras.emb.children) if (o.material && o.userData.base != null) o.material.opacity = o.userData.base * ea;
    extras.embHead.el.style.opacity = ea.toFixed(3);
    extras.embAx.forEach((a) => { a.el.style.opacity = (ea * 0.8).toFixed(3); });
    const k = ex.embK ?? 0; // 0 → 1：取向量的进度
    for (const r of extras.rows) {
      const t0 = (r.i / Q.P) * 0.5, local = clamp((k - t0) / 0.5);
      const lit = local > 0 ? 1 : 0;
      r.row.material.opacity = ea * lit * (0.14 + 0.7 * (1 - smooth(seg(local, 0.15, 0.7))));
      r.dotA.material.opacity = ea * lit * 0.9;
      r.pk.visible = local > 0.02 && local < 1;
      if (r.pk.visible) r.pk.position.copy(r.curve.getPoint(easeIn(local) * 0.4 + easeInOut(local) * 0.6));
      if (r.lb) { r.lb.visible = local > 0.01; r.lb.el.style.opacity = (ea * smooth(local * 4)).toFixed(3); }
    }
  }
  // 指示
  const na = ex.next ?? 0;
  extras.next.visible = na > 0.01;
  if (extras.next.visible) {
    const i = Q.row(f.st?.g ?? 0);
    extras.next.position.set(M.x(i), 0.62, 0.1);
    extras.next.el.style.opacity = na.toFixed(3);
  }
}

// 聚焦：深入一层时，把无关的光柱、层板、KV 缓存再压暗一些，画面更干净（在 Machine.update 之后乘上去）
function isolate(k, st) {
  M.kvK.material.transparent = M.kvV.material.transparent = true;
  M.kvK.material.opacity = M.kvV.material.opacity = 1 - 0.85 * k;
  if (k <= 0.001) return;
  const focus = Q.row(st.g);
  M.columns.forEach((c, i) => { if (i !== focus && c.visible) c.material.opacity *= 1 - 0.8 * k; });
  M.slabs.forEach((sl, L) => {
    if (L === M.explodeL) return;
    sl.material.opacity *= 1 - 0.75 * k;
    sl.edge.material.opacity *= 1 - 0.7 * k;
  });
}

/* ================================================================ 叠加层 */

const OV = {};

function buildOverlays() {
  const ov = $('#ov');
  OV.band = el('div', 'band', '', ov);
  OV.subs = new CueLayer(ov, 'sub', SC.subs, { rise: 8 });
  OV.strips = new CueLayer(ov, 'strip', SC.strips, { rise: 6 });
  OV.chapters = new CueLayer(ov, 'chapter', SC.chapters, { rise: 0, fin: 0.6, fout: 0.6 });
  OV.cards = new CueLayer(ov, 'card', SC.cards, { rise: 8 });
  OV.corner = el('div', 'corner', '', ov);
  OV.cornerKey = '';

  // 冷开场：问题的词元
  OV.qline = el('div', 'qline', '', ov);
  const userToks = Q.tokens.filter((t) => t.role === 'user' && !t.sp && t.s !== 'user' && t.s !== '\n');
  OV.qtoks = userToks.map((t) => { const d = el('div', 'qtok', `<span class="s">${tk(t.s)}</span><span class="id">${t.id}</span>`, OV.qline); return d; });
  OV.qcaret = el('div', 'qcaret', '', OV.qline);

  // 片名
  OV.title = el('div', 'title', `<div class="eb">INSIDE A LANGUAGE MODEL</div><h1>揭开黑箱</h1><div class="rule"></div><div class="st">一个词，是怎样被算出来的</div><div class="spec">QWEN3-0.6B · 28 LAYERS · ${MAN.model.params.toLocaleString('en-US')} PARAMETERS · BF16</div>`, ov);

  // 逻辑透镜面板
  OV.lens = el('div', 'lenspanel', '', ov);
  OV.lensKey = '';
  // 注意力矩阵
  OV.att = el('div', 'attgrid', '', ov);
  OV.attKey = '';
  // 回答
  OV.reply = el('div', 'reply', '', ov);
  OV.replyKey = '';
  // 片尾
  OV.end = el('div', 'endcard', '', ov);
  OV.end2 = el('div', 'endcard', '', ov);
}

function updateOverlays(t, f) {
  const o = f.ov || {};
  OV.subs.update(t);
  OV.strips.update(t);
  OV.chapters.update(t);
  OV.cards.update(t);
  OV.band.style.opacity = (o.band ?? 1).toFixed(3);

  // 右上角：现在在第几个词、第几层
  const c = o.corner;
  const ck = c ? `${c.g}|${c.L ?? ''}|${c.what ?? ''}` : '';
  if (ck !== OV.cornerKey) {
    OV.cornerKey = ck;
    OV.corner.innerHTML = c ? `QWEN3-0.6B · BF16<br><b>词元 ${c.g + 1} / ${Q.G}</b>${c.L != null ? `<br><b>第 ${String(c.L).padStart(2, '0')} 层 / 28</b>` : ''}${c.what ? `<br>${c.what}` : ''}` : '';
  }
  OV.corner.style.opacity = (o.cornerA ?? 0).toFixed(3);

  // 冷开场的问题
  const qa = o.qline ?? 0;
  OV.qline.style.display = qa > 0.001 ? 'flex' : 'none';
  if (qa > 0.001) {
    OV.qline.style.opacity = qa.toFixed(3);
    const s = o.qScale ?? 1;
    OV.qline.style.transform = `translate(-50%, -50%) scale(${s.toFixed(4)})`;
    OV.qline.style.filter = o.qBlur ? `blur(${o.qBlur.toFixed(2)}px)` : '';
    SC.qTimes.forEach((tt, i) => {
      const k = clamp((t - tt) / 0.35);
      const d = OV.qtoks[i];
      d.style.display = k > 0 ? 'flex' : 'none';
      d.style.opacity = smooth(k).toFixed(3);
      d.style.transform = `translateY(${((1 - easeOut(k)) * 14).toFixed(2)}px)`;
      d.style.filter = k < 1 ? `blur(${((1 - k) * 6).toFixed(2)}px)` : '';
    });
    const typing = t < SC.qTimes[SC.qTimes.length - 1] + 0.8;
    OV.qcaret.style.opacity = typing || Math.floor(t * 1.6) % 2 === 0 ? (o.caret ?? 1).toFixed(2) : '0';
  }

  // 片名
  const ta = o.title ?? 0;
  OV.title.style.display = ta > 0.001 ? 'block' : 'none';
  if (ta > 0.001) {
    const k = o.titleK ?? 1;
    OV.title.style.opacity = ta.toFixed(3);
    const h1 = OV.title.querySelector('h1');
    h1.style.letterSpacing = `${(0.42 - 0.24 * easeOut(k)).toFixed(4)}em`;
    h1.style.paddingLeft = h1.style.letterSpacing;
    OV.title.style.filter = o.titleBlur ? `blur(${o.titleBlur.toFixed(2)}px)` : '';
    OV.title.querySelector('.rule').style.width = `${(520 * easeInOut(seg(k, 0.15, 0.8))).toFixed(1)}px`;
    OV.title.querySelector('.st').style.opacity = smooth(seg(k, 0.35, 0.8)).toFixed(3);
    OV.title.querySelector('.spec').style.opacity = (0.9 * smooth(seg(k, 0.55, 1))).toFixed(3);
    OV.title.querySelector('.eb').style.opacity = smooth(seg(k, 0.0, 0.5)).toFixed(3);
  }

  updateLens(o.lens);
  updateAtt(o.att);
  updateReply(o.reply);
  updateEnd(o.end);
}

// 逻辑透镜：每层输出直接接最终归一化 + 输出头，读出两个候选词的概率（真实数据：选中词的概率 + 每层前 3 名）
function updateLens(L) {
  OV.lens.style.display = L && L.a > 0.001 ? 'block' : 'none';
  if (!L || L.a <= 0.001) return;
  OV.lens.style.opacity = L.a.toFixed(3);
  OV.lens.style.left = L.side === 'left' ? '72px' : 'auto';
  OV.lens.style.right = L.side === 'left' ? 'auto' : '72px';
  const g = L.g, upto = L.upto; // upto：已经走过的层（可以是小数）
  const W = 382, H = 168, x = (l) => (l / 27) * W, y = (p) => H - p * H;
  const key = `${g}|${upto.toFixed(3)}`;
  if (key === OV.lensKey) return;
  OV.lensKey = key;
  const chosen = Q.steps[g].chosenS;
  const other = L.other;
  const pts = (fn) => {
    const out = [];
    for (let l = 0; l < 28; l++) { const v = fn(l); if (v != null) out.push([l, v]); }
    return out;
  };
  const pc = pts((l) => Q.lensAt(g, l).pc);
  const po = pts((l) => { const r = Q.lensAt(g, l).top.find((x) => x[2] === other); return r ? r[1] : null; });
  const clip = Math.min(27, upto);
  const line = (arr, col) => {
    const seen = arr.filter(([l]) => l <= clip + 1e-6);
    if (!seen.length) return '';
    let d = '';
    // 只连相邻层（中间缺数据就断开）
    seen.forEach(([l, v], k) => { d += `${k && seen[k - 1][0] === l - 1 ? 'L' : 'M'}${x(l).toFixed(1)},${y(v).toFixed(1)} `; });
    const dots = seen.map(([l, v]) => `<circle cx="${x(l).toFixed(1)}" cy="${y(v).toFixed(1)}" r="${l === Math.floor(clip) ? 4.5 : 2.4}" fill="${col}"/>`).join('');
    return `<path d="${d}" fill="none" stroke="${col}" stroke-width="2.2" stroke-linejoin="round"/>${dots}`;
  };
  const cur = Math.floor(clip);
  const vc = pc.find(([l]) => l === cur)?.[1] ?? 0;
  const vo = po.find(([l]) => l === cur)?.[1];
  const grid = [0, 0.5, 1].map((p) => `<line x1="0" x2="${W}" y1="${y(p)}" y2="${y(p)}" stroke="rgba(150,180,230,${p === 0 ? 0.25 : 0.1})"/><text class="ax" x="-8" y="${y(p) + 4}" text-anchor="end">${p * 100}%</text>`).join('');
  const ticks = [0, 7, 14, 21, 27].map((l) => `<text class="ax" x="${x(l)}" y="${H + 18}" text-anchor="middle">L${String(l).padStart(2, '0')}</text>`).join('');
  const cursor = `<line x1="${x(clip)}" x2="${x(clip)}" y1="0" y2="${H}" stroke="rgba(255,182,92,.35)" stroke-dasharray="3 4"/>`;
  OV.lens.innerHTML = `<div class="h">逻辑透镜 · 第 ${g + 1} 个词<small>每层直接接输出头：此刻开口会说什么</small></div>
    <svg width="${W}" height="${H + 24}" viewBox="0 0 ${W} ${H + 24}">${grid}${ticks}${cursor}${line(po, '#5ef0d4')}${line(pc, '#ffb65c')}</svg>
    <div style="display:flex;gap:22px;margin-top:6px;font-size:19px">
      <span style="color:var(--amber)">${q(chosen)} <span class="m" style="font-size:16px">${pct(vc)}</span></span>
      <span style="color:var(--cyan)">${q(other)} <span class="m" style="font-size:16px">${vo != null ? pct(vo) : '—'}</span></span>
      <span style="color:var(--dim);font-size:16px;margin-left:auto">榜首 ${q(Q.lensAt(g, cur).top[0][2])}</span>
    </div>`;
}

// 16 个头 × 前面所有位置的注意力（真实数据：每行只导出了前 4 名，其余视为 0）
function updateAtt(A) {
  OV.att.style.display = A && A.a > 0.001 ? 'block' : 'none';
  if (!A || A.a <= 0.001) return;
  OV.att.style.opacity = A.a.toFixed(3);
  OV.att.style.left = A.side === 'right' ? 'auto' : '72px';
  OV.att.style.right = A.side === 'right' ? '72px' : 'auto';
  const key = `${A.L}|${A.g}|${A.head}|${A.reveal.toFixed(2)}`;
  if (key === OV.attKey) return;
  OV.attKey = key;
  const i = Q.row(A.g), n = i + 1;
  const isUser = (t) => t.role === 'user' && !t.sp && t.s !== 'user' && t.s !== '\n';
  const rows = [];
  for (let h = 0; h < Q.H; h++) {
    const w = new Array(n).fill(0);
    for (const r of Q.att(A.L, h, i)) w[r.j] = r.w;
    const on = h === A.head;
    const shown = h / Q.H < A.reveal;
    rows.push(`<div class="row${on ? ' on' : ''}"><span class="hn">${h}</span>${w.map((v) => {
      const a = shown ? Math.pow(v, 0.6) : 0;
      const col = on ? `rgba(255,182,92,${(0.07 + 0.93 * a).toFixed(3)})` : `rgba(94,240,212,${(0.045 + 0.9 * a).toFixed(3)})`;
      return `<i style="background:${col}"></i>`;
    }).join('')}<span class="pk">${on && Q.att(A.L, h, i)[0] ? `${tk(Q.tokens[Q.att(A.L, h, i)[0].j].s)} ${pct(Q.att(A.L, h, i)[0].w)}` : ''}</span></div>`);
  }
  // 列标：开头的标记、问题那几个词元（用一个括号框起来）、最后一个位置自己
  const us = Q.tokens.slice(0, n).map((t, j) => (isUser(t) ? j : -1)).filter((j) => j >= 0);
  const cx = (j) => j * 15 + 6.5;
  const marks = `<span style="left:${cx(0)}px">开头</span>`
    + (us.length ? `<b style="left:${cx(us[0]) - 6}px;width:${cx(us[us.length - 1]) - cx(us[0]) + 12}px"></b><span class="u" style="left:${(cx(us[0]) + cx(us[us.length - 1])) / 2}px">问题「${us.map((j) => tk(Q.tokens[j].s)).join('')}」</span>` : '')
    + `<span style="left:${cx(i)}px">自己</span>`;
  OV.att.innerHTML = `<div class="h">第 ${A.L} 层 · 16 个头的注意力<small>最后一个位置在看谁 · 每行一个头</small></div>${rows.join('')}<div class="cols">${marks}</div>`;
}

// 回答一个词一个词长出来：每个词元下面一条细线，长度 = 它被选中时的真实概率；最新的那个是琥珀色
function updateReply(R) {
  OV.reply.style.display = R && R.a > 0.001 ? 'block' : 'none';
  if (!R || R.a <= 0.001) return;
  OV.reply.style.opacity = R.a.toFixed(3);
  OV.reply.style.transform = `translateX(-50%) translateY(${(R.dy ?? 0).toFixed(1)}px)`;
  const key = `${R.n}|${(R.k ?? 1).toFixed(2)}|${R.hl ?? ''}`;
  if (key === OV.replyKey) return;
  OV.replyKey = key;
  // 标点和前一个词元排在同一个不换行的块里，避免一行以「，」开头
  const groups = [];
  Q.steps.slice(0, R.n).forEach((st, g) => {
    const sp = st.chosenS === '<|im_end|>';
    const newest = g === R.n - 1;
    const k = newest ? R.k ?? 1 : 1;
    const low = st.chosenRank > 0;
    const html = `<span class="t${newest ? ' new' : ''}${sp ? ' sp' : ''}${g === R.hl ? ' hl' : ''}" style="opacity:${(0.2 + 0.8 * k).toFixed(3)}"><span class="s">${sp ? esc(shortSpecial(st.chosenS)) : tk(st.chosenS)}</span><i class="${low ? 'low' : ''}" style="width:${Math.max(6, st.chosenP1 * 100).toFixed(1)}%"></i></span>`;
    if ((sp || /^[，。、！？；：,.!?]$/.test(st.chosenS)) && groups.length) groups[groups.length - 1].push(html);
    else groups.push([html]);
  });
  OV.reply.innerHTML = groups.map((g) => `<span class="c">${g.join('')}</span>`).join('') + (R.tag ? `<span class="tag">${R.tag}</span>` : '');
}

function updateEnd(D) {
  const a1 = D?.a1 ?? 0, a2 = D?.a2 ?? 0;
  OV.end.style.display = a1 > 0.001 ? 'block' : 'none';
  OV.end2.style.display = a2 > 0.001 ? 'block' : 'none';
  if (a1 > 0.001) {
    if (!OV.end.innerHTML) {
      // 回答按逗号断成几行（每行不超过 22 个字），不在词中间折行
      const parts = Q.manifest.questions.find((x) => x.id === QID).reply.split(/(?<=[，。！？])/);
      const lines = [];
      for (const p of parts) { if (lines.length && (lines[lines.length - 1] + p).length <= 22) lines[lines.length - 1] += p; else lines.push(p); }
      OV.end.innerHTML = `<div class="ans">${lines.map((l) => `<div>${esc(l)}</div>`).join('')}</div><div class="stat">${SC.statLine}</div>`;
    }
    OV.end.style.opacity = a1.toFixed(3);
    OV.end.querySelector('.stat').style.opacity = (D.k1 ?? 1).toFixed(3);
  }
  if (a2 > 0.001) {
    if (!OV.end2.innerHTML) OV.end2.innerHTML = `<div class="brand">揭开黑箱</div><div class="url">caijiechao.com/blackbox/</div><div class="cred">在网页里，你可以亲手把这台机器一层层拆开<br><span class="m" style="font-size:15px;letter-spacing:.12em">Qwen3-0.6B 真实离线运行数据 · 画面与音乐均由程序生成</span></div>`;
    OV.end2.style.opacity = a2.toFixed(3);
    OV.end2.style.transform = `translate(-50%, -50%) translateY(${((1 - easeOut(D.k2 ?? 1)) * 12).toFixed(2)}px)`;
  }
}

/* ================================================================ 算式板 */

// 网站上的算式板有一些操作提示（点一行、点键帽……），片子里用不上：去掉
const STRIP = [/<span class="dim">点一行[^<]*<\/span>/g, /<b>点任意一位<\/b>把它翻过来，看看会发生什么。/g, /试试看：/g];
function cleanBoard(a) {
  const el = M.board.el;
  if (el.hidden) return;
  el.style.opacity = a.toFixed(3);
  for (const n of el.querySelectorAll('.bd-cap, .bd-eff p')) {
    if (n._c === n.innerHTML) continue;
    let h = n.innerHTML;
    for (const r of STRIP) h = h.replace(r, '');
    if (h !== n.innerHTML) n.innerHTML = h;
    n._c = n.innerHTML;
  }
}

/* ================================================================ 颗粒 */

const grainCtx = () => $('#grain').getContext('2d');
let grainImg = null;
function drawGrain(frame) {
  const ctx = grainCtx();
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

/* ================================================================ 每帧 */

let lastT = null, lastPanel = '', lastF = null;

// 调试：当前机器里几个关键物体的世界坐标（调机位用）
function probe() {
  const st = lastF?.st;
  const v = (o) => (o ? [o.x, o.y, o.z].map((n) => +n.toFixed(3)) : null);
  const W = (o) => v(o.getWorldPosition(new THREE.Vector3()));
  const mc = st ? M.camera(st) : null;
  return { shot: lastF?.shot, view: st?.view, cam: { pos: v(E.camera.position) }, xf: st ? M.x(Q.row(st.g)) : null, e: +M.e.toFixed(3), explodeL: M.explodeL, yL: M.explodeL >= 0 ? +M.yL(M.explodeL).toFixed(3) : null, yTop: +M.yTop.toFixed(3), mcam: mc && { pos: v(mc.pos), look: v(mc.look) }, attn: W(M.mats.attn), mlp: W(M.mats.mlp), panel: st ? v(M.detail.panelPos(st)) : null, rig: v(M.detail.rigCenter), dot: v(M.detail.dotCenter), bits: v(M.detail.bitsCenter), headX: st ? M.headX(st) : null };
}
function renderAt(t, { render = true } = {}) {
  const dt = lastT == null || t <= lastT || t - lastT > 0.5 ? 1 / FPS : t - lastT;
  lastT = t;
  const f = (lastF = SC.frame(t, { M, E, Q }));
  if (f.st) {
    // 比特：脚本里演示翻转某一位（网站上是点键帽），算式板会按真实公式重算后果
    const D = M.detail;
    if (f.flipBit != null && D.baseBits && D.bitsKey) { const cur = D.baseBits.slice(); cur[f.flipBit] ^= 1; D.flips.set(D.bitsKey, cur); } else if (D.flips.size) D.flips.clear();
    M.update(f.st, dt, t);
    cleanBoard(f.boardA ?? 1);
    if (f.after) f.after(M);
    isolate(f.iso ?? 0, f.st);
    // 层板右端的逻辑透镜读数：机器太宽（78 个位置），挪到当前词元的光柱旁边
    const lx = M.x(Q.row(f.st.g)) + (f.lensDx ?? 0.75);
    for (const sl of M.slabs) sl.lbl.position.x = lx;
    // 镜头需要的临时隐藏 / 压暗（每帧都重新赋值，不会残留）
    const hide = new Set(f.hide || []);
    if (hide.has('attnMats')) M.mats.attn.visible = false;
    if (hide.has('mlpMats')) M.mats.mlp.visible = false;
    if (hide.has('mlpVecs')) for (const o of [M.mats.mlp.x, M.mats.mlp.x2, M.mats.mlp.go, M.mats.mlp.uo]) o.visible = false;
    for (const o of [M.exAdd1, M.exAdd2, M.exRing1, M.exRing2, M.exUnit]) o.visible = !hide.has('exDeco');
    if (f.lensWin != null && f.st.step.L != null) M.slabs.forEach((sl, L) => { if (L < f.st.step.L - f.lensWin || L > f.st.step.L) sl.lbl.visible = false; });
    if (f.slabDim) M.slabs.forEach((sl) => { sl.material.opacity *= 1 - f.slabDim; sl.edge.material.opacity *= 1 - f.slabDim; sl.lbl.visible = false; });
    // 神经元阵列点亮的先后顺序原来用 Math.random：换成确定的哈希
    if (M.detail.panelKey && M.detail.panelKey !== lastPanel) {
      lastPanel = M.detail.panelKey;
      M.detail.order = Float32Array.from({ length: M.detail.order.length }, (_, n) => hash01(n));
      M.detail.litState = -1;
    }
  }
  updateExtras(t, f);
  const sx = f.shiftX ?? 0;
  if (Math.abs(sx - E.shiftX) > 0.01) { E.shiftX = sx; E.applyOffset(); }
  const cam = typeof f.cam === 'function' ? f.cam() : f.cam;
  E.setCamera(cam.pos, cam.look, cam.fov ?? 34, cam.roll ?? 0);
  if (f.dof) E.setDof(f.dof.focus ?? cam.pos.distanceTo(cam.look), f.dof.range ?? 3, f.dof.blur ?? 0); else E.setDof(0, 1, 0);
  E.renderer.toneMappingExposure = f.exposure ?? 0.84;
  $('#fade').style.opacity = (f.fade ?? 0).toFixed(4);
  updateOverlays(t, f);
  if (!render) return;
  drawGrain(Math.round(t * FPS));
  E.render();
}

// 跳到 t：先从 t - 8 秒起不出图地模拟一遍，让机器里各种平滑动画收敛到连续播放时的状态
function seek(t, pre = 8) {
  lastT = null;
  const t0 = Math.max(0, t - pre);
  for (let x = t0; x < t - 1e-6; x += 1 / FPS) renderAt(x, { render: false });
  renderAt(t);
}

/* ================================================================ 封面 */

// 封面：渲染第 t 秒的画面，去掉字幕和数据条，叠上片名
function poster(t = 58.4) {
  seek(t);
  for (const sel of ['.sub', '.strip', '.chapter', '.card', '.corner', '.lenspanel', '.attgrid', '.band']) document.querySelectorAll(sel).forEach((e) => { e.style.display = 'none'; });
  const T = OV.title;
  T.style.display = 'block';
  T.style.opacity = '1';
  T.style.top = '30%';
  T.style.filter = '';
  const h1 = T.querySelector('h1');
  h1.style.letterSpacing = h1.style.paddingLeft = '0.18em';
  T.querySelector('.rule').style.width = '520px';
  for (const c of ['.st', '.spec', '.eb']) T.querySelector(c).style.opacity = '1';
  $('#fade').style.opacity = '0.25';
  E.render();
}

/* ================================================================ 预览 */

function startPreview() {
  const bar = $('#scrub');
  bar.hidden = false;
  const r = $('#scrubber');
  r.max = SC.end;
  let playing = true, t = Number(params.get('t') || 0), last = performance.now();
  seek(t);
  r.addEventListener('input', () => { t = Number(r.value); seek(t, 4); });
  addEventListener('keydown', (e) => { if (e.key === ' ') playing = !playing; });
  const loop = (now) => {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (playing) { t += dt; if (t > SC.end) t = 0; renderAt(t); r.value = t; }
    $('#scrubT').textContent = t.toFixed(2);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

export { EVENTS };
boot().catch((e) => { console.error(e); document.body.insertAdjacentHTML('beforeend', `<pre style="color:#f66;position:fixed;top:0;left:0">${esc(e.stack || e.message)}</pre>`); window.__film = { error: String(e.stack || e) }; });
