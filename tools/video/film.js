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
import { Opening } from './lib/opening.js';
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

let E, M, Q, MAN, SC, OPENING, QR_SVG = '';
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
    document.fonts.load('900 100px "Film Serif"', 'AI 的一个字是怎么思考出来的'),
    document.fonts.load('400 38px "Film Sans"', '天空为什么是蓝色的？'),
    document.fonts.load('600 38px "Film Sans"', 'Qwen'),
    document.fonts.load('600 40px "Film Serif"', '法国的首都是哪里'),
    document.fonts.load('600 76px "PingFang SC"', '法国的首都'),
    document.fonts.load('600 76px "Noto Sans SC"', '法国'),
    document.fonts.load('400 20px "JetBrains Mono"', '0123'),
    document.fonts.load('700 20px "JetBrains Mono"', '0123'),
  ]);
  QR_SVG = await (await fetch('/tools/video/qr-blackbox.svg')).text();
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
  OPENING = new Opening({ E, M, Q, frame: $('#frame'), T: SC.open });
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
  OV.title = el('div', 'title', `<div class="eb">INSIDE A LANGUAGE MODEL</div><h1><span>AI 的一个字</span><span>是怎么思考出来的</span></h1><div class="rule"></div><div class="st">走进大模型推理的“黑箱”</div><div class="spec">QWEN3-0.6B · 28 LAYERS · ${MAN.model.params.toLocaleString('en-US')} PARAMETERS · BF16</div>`, ov);

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
  // 继续探索：一句引子 → 站内另外几页 → 延伸学习 + 二维码（二维码由 make_qr.py 在本地生成）
  OV.end3 = el('div', 'endcard', '<div class="lead">这只是一个字的旅程。想继续看？</div>', ov);
  OV.end4 = el('div', 'endcard', `<div class="more">${[['训练', '模型是怎么学会的'], ['多模态', '它怎么看图'], ['智能体', '它怎么动手干活'], ['世界模型', '它脑子里的世界']].map(([a, b]) => `<b>${a}</b><span>${b}</span>`).join('')}</div>`, ov);
  OV.end5 = el('div', 'endcard fin', `<div class="qr">${QR_SVG}</div><div class="fin-t"><div class="k">延伸学习</div><div class="url">caijiechao.com/blackbox/learn/</div><div class="hint">扫码打开网站 · caijiechao.com/blackbox/</div><div class="cred">Qwen3-0.6B 真实离线运行数据 · 画面与音乐均由程序生成</div></div>`, ov);

  // v5 讲解层：章节进度、术语标签、注意力公式、6 个头的小图、指示环
  OV.terms = new CueLayer(ov, 'term', SC.terms, { rise: 6, fin: 0.3, fout: 0.3 });
  OV.prog = el('div', 'prog', SC.chapterNames.map((c, i) => `${i ? '<span class="ln"></span>' : ''}<span class="it"><i>${i + 1}</i>${c}</span>`).join(''), ov);
  OV.progIt = [...OV.prog.querySelectorAll('.it')];
  OV.progKey = '';
  OV.formula = el('div', 'formula', `<div class="eq">
      <span class="t" data-k="lhs">Attention(<span class="q">Q</span>, <span class="k">K</span>, <span class="v">V</span>) =</span>
      <span class="t" data-k="smL">softmax(</span>
      <span class="frac"><span class="t" data-k="qk"><span class="q">Q</span><span class="k">K</span><sup>T</sup></span><span class="bar" data-k="sd"></span><span class="t" data-k="sd">√<span style="text-decoration:overline">d</span></span></span>
      <span class="t" data-k="smR">)</span>
      <span class="t" data-k="v">· <span class="v">V</span></span>
    </div><div class="cap"></div>`, ov);
  OV.fTerms = [...OV.formula.querySelectorAll('[data-k]')];
  OV.fCap = OV.formula.querySelector('.cap');
  OV.fKey = '';
  OV.heads6 = el('div', 'heads6', '', ov);
  OV.h6Key = '';
  OV.ffn = el('div', 'ffnmap', `<div class="row">
      <div class="b in" data-k="in"><b>1024</b><span>这个词的理解</span></div><i class="ar">→</i>
      <div class="stack"><div class="b g" data-k="gate"><b>门 gate</b><span>3072 个：放不放行、放多少</span></div><div class="b u" data-k="up"><b>内容 up</b><span>3072 份内容</span></div></div><i class="ar">→</i>
      <div class="b mul" data-k="mul"><b>×</b></div><i class="ar">→</i>
      <div class="b d" data-k="down"><b>down</b><span>收回 1024</span></div><i class="ar">→</i>
      <div class="b add" data-k="add"><b>⊕</b><span>加回去</span></div>
    </div><div class="cap"></div>`, ov);
  OV.ffnCap = OV.ffn.querySelector('.cap');
  OV.ffnKey = '';
  OV.rings = [];
}

// 章节进度：当前一章亮
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

// 注意力公式：show = 已经讲到的项，hi = 正在讲的项（'sm' 表示 softmax 的左右括号一起亮），cap = 下面一行真实数字
function updateFormula(F) {
  OV.formula.style.display = F && F.a > 0.001 ? 'block' : 'none';
  if (!F || F.a <= 0.001) return;
  OV.formula.style.opacity = F.a.toFixed(3);
  OV.formula.style.transform = `translateX(-50%) translateY(${((1 - easeOut(Math.min(1, F.a * 1.2))) * -10).toFixed(1)}px)`;
  const key = `${F.show.join(',')}|${F.hi}|${F.cap}`;
  if (key === OV.fKey) return;
  OV.fKey = key;
  const grp = (k) => (k === 'smL' || k === 'smR' ? 'sm' : k);
  for (const e of OV.fTerms) {
    const k = e.dataset.k;
    const shown = F.show.includes(grp(k)) || F.show.includes(k);
    e.style.opacity = shown ? '1' : '0.13';
    if (e.classList.contains('t')) e.className = `t${F.hi !== 'all' && F.hi === grp(k) ? ' on' : shown ? ' done' : ''}`;
  }
  OV.formula.classList.toggle('all', F.hi === 'all');
  OV.fCap.innerHTML = F.cap || '';
}

// 6 个头的小图：每个头前 3 名（真实数据）。「天空」那一行用琥珀色
const tokShort = (s) => ({ '<|im_start|>': '开头', '\n\n': '换行', '</think>': '思考结束', '\n': '换行' }[s] ?? tokPlain(s));
function updateHeads6(H) {
  OV.heads6.style.display = H && H.a > 0.001 ? 'grid' : 'none';
  if (!H || H.a <= 0.001) return;
  OV.heads6.style.opacity = H.a.toFixed(3);
  if (OV.h6Key !== H.list.join(',')) {
    OV.h6Key = H.list.join(',');
    const row = Q.row(H.g ?? 0);
    OV.heads6.innerHTML = H.list.map((h) => {
      const top = Q.att(H.L, h, row).slice(0, 3);
      return `<div class="hd" data-h="${h}"><div class="hn">第 ${h} 个头<small>${esc(H.notes?.[h] ?? '')}</small></div>${top.map((r) => `<div class="r${Q.tokens[r.j].s === H.sky ? ' sky' : ''}"><span>「${esc(tokShort(Q.tokens[r.j].s))}」</span><span class="bar" style="width:${Math.max(3, r.w * 100).toFixed(1)}%"></span><span class="p">${pct(r.w)}</span></div>`).join('')}</div>`;
    }).join('');
  }
  OV.heads6.querySelectorAll('.hd').forEach((d, k) => {
    const a = smooth(clamp((H.k ?? 1) * H.list.length - k));
    d.style.opacity = a.toFixed(3);
    d.style.transform = `translateY(${((1 - a) * 14).toFixed(1)}px)`;
    d.classList.toggle('on', Number(d.dataset.h) === H.on);
  });
}

// 指示环：at() 给世界坐标，w / h 是要圈住的世界尺寸
const toScreen = (v) => { const p = v.clone().project(E.camera); return { x: ((p.x + 1) / 2) * 1920, y: ((1 - p.y) / 2) * 1080 }; };
function updateRings(R) {
  const list = (R || []).filter((r) => r.a > 0.001);
  while (OV.rings.length < list.length) OV.rings.push(el('div', 'ring', '<span class="rl"></span>', $('#ov')));
  if (list.length) E.camera.updateMatrixWorld();
  OV.rings.forEach((e, k) => {
    const r = list[k];
    e.style.display = r ? 'block' : 'none';
    if (!r) return;
    const c = typeof r.at === 'function' ? r.at() : r.at;
    if (!c) { e.style.display = 'none'; return; }
    const a = toScreen(c), bx = toScreen(c.clone().add(new THREE.Vector3((r.w ?? 0.3) / 2, 0, 0))), by = toScreen(c.clone().add(new THREE.Vector3(0, (r.h ?? r.w ?? 0.3) / 2, 0)));
    const w = Math.max(r.min ?? 60, Math.hypot(bx.x - a.x, bx.y - a.y) * 2 + (r.pad ?? 36)), h = Math.max(r.min ?? 60, Math.hypot(by.x - a.x, by.y - a.y) * 2 + (r.pad ?? 36));
    const pop = 1 + 0.25 * (1 - easeOut(Math.min(1, r.a * 1.4)));
    e.className = `ring${r.cls ? ` ${r.cls}` : ''}${r.top ? ' top' : ''}`;
    e.style.left = `${(a.x - (w * pop) / 2).toFixed(1)}px`;
    e.style.top = `${(a.y - (h * pop) / 2).toFixed(1)}px`;
    e.style.width = `${(w * pop).toFixed(1)}px`;
    e.style.height = `${(h * pop).toFixed(1)}px`;
    e.style.opacity = r.a.toFixed(3);
    const lb = e.firstChild;
    if (lb.innerHTML !== (r.label || '')) lb.innerHTML = r.label || '';
  });
}

// 前馈的结构图：1024 → 门 / 内容（各 3072）→ 相乘 → down → 1024 → 加回去。hi = 正在讲的部分，cap = 下面一行（文字或 SiLU 曲线）
const SILU_SVG = (() => {
  const W = 340, H = 120, x0 = -6, x1 = 4, y0 = -0.6, y1 = 4;
  const X = (x) => ((x - x0) / (x1 - x0)) * W, Y = (y) => H - ((y - y0) / (y1 - y0)) * H;
  let d = '';
  for (let i = 0; i <= 80; i++) { const x = x0 + ((x1 - x0) * i) / 80; d += `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(x / (1 + Math.exp(-x))).toFixed(1)} `; }
  return `<svg width="${W}" height="${H + 6}" viewBox="0 -3 ${W} ${H + 6}"><line x1="0" x2="${W}" y1="${Y(0)}" y2="${Y(0)}" stroke="rgba(150,180,230,.35)"/><line x1="${X(0)}" x2="${X(0)}" y1="0" y2="${H}" stroke="rgba(150,180,230,.35)"/><path d="${d}" fill="none" stroke="#5ef0d4" stroke-width="3"/></svg>`;
})();
function updateFfnMap(F) {
  OV.ffn.style.display = F && F.a > 0.001 ? 'block' : 'none';
  if (!F || F.a <= 0.001) return;
  OV.ffn.style.opacity = F.a.toFixed(3);
  const key = `${F.hi}|${F.cap}`;
  if (key === OV.ffnKey) return;
  OV.ffnKey = key;
  const LIT = { shape: ['in', 'gate', 'up', 'mul', 'down', 'add'], gate: ['gate'], up: ['up'], mul: ['gate', 'up', 'mul'], silu: ['gate'], act: ['mul'], down: ['down', 'add'] }[F.hi] || [];
  OV.ffn.querySelectorAll('[data-k]').forEach((e) => e.classList.toggle('on', LIT.includes(e.dataset.k)));
  OV.ffnCap.innerHTML = F.cap === 'silu' ? `<span class="l">小于 0：基本关上</span>${SILU_SVG}<span class="r">大于 0：照常通过</span>` : (F.cap || '');
}

// 前馈的三块矩阵：讲到哪块，哪块亮；'din' = 门 × 内容之后的那根向量
function focusMlp(which, k = 1) {
  const P = M.mats.mlp;
  if (!P || !P.visible) return;
  const D = { g: { g: 1, u: 0.12, d: 0.12, din: 0.12 }, u: { g: 0.12, u: 1, d: 0.12, din: 0.12 }, din: { g: 0.4, u: 0.4, d: 0.12, din: 1 }, d: { g: 0.12, u: 0.12, d: 1, din: 0.5 } }[which];
  const dim = (key) => (D ? 1 - (1 - D[key]) * k : 1);
  for (const [key, pn, vec] of [['g', P.g, P.go], ['u', P.u, P.uo], ['d', P.d, P.dout]]) {
    const a = dim(key);
    pn.mat.opacity *= a;
    pn.edge.material.opacity *= a;
    pn.scan.material.opacity *= a;
    vec.segs.forEach((s) => { s.material.opacity = 0.95 * a; });
  }
  P.din.segs.forEach((s) => { s.material.opacity = 0.95 * dim('din'); });
}

// 注意力的三块矩阵：讲到哪个，哪个亮，另外两块和它们的标签压暗 / 收起
function focusAttn(which, k = 1) {
  const A = M.mats.attn;
  if (!A || !A.visible) return;
  A.gqaL.visible = false;
  A.gqa.visible = false;
  for (const [key, pn, vec] of [['q', A.q, A.qo], ['k', A.k, A.ko], ['v', A.v, A.vo]]) {
    const on = !which || which === key;
    const dim = on ? 1 : 1 - 0.88 * k;
    pn.mat.opacity *= dim;
    pn.edge.material.opacity = 0.7 * dim;
    pn.scan.material.opacity *= dim;
    vec.segs.forEach((s) => { s.material.opacity = 0.95 * dim; });
    const base = key === 'q' ? A.q.lb.visible : true; // W_q 的标签在微观视图里由 mats.js 收起
    pn.lb.visible = base && on;
    if (vec.lbl) vec.lbl.visible = on && !which;
  }
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
    h1.style.letterSpacing = `${(0.16 - 0.11 * easeOut(k)).toFixed(4)}em`;
    h1.style.paddingLeft = h1.style.letterSpacing;
    OV.title.style.filter = o.titleBlur ? `blur(${o.titleBlur.toFixed(2)}px)` : '';
    // 片名落在黑箱正面：跟着前面板中心在屏幕上的位置走
    if (o.titleOnBox) {
      E.camera.updateMatrixWorld();
      const p = M.cFront.getWorldPosition(new THREE.Vector3()).project(E.camera);
      OV.title.style.left = `${(((p.x + 1) / 2) * 1920).toFixed(1)}px`;
      OV.title.style.top = `${(((1 - p.y) / 2) * 1080).toFixed(1)}px`;
      // 黑箱还远的时候前面板在屏幕上很窄：片名跟着缩小，始终收在发光边框里
      const W = M.cFront.geometry.parameters.width;
      const l = toScreen(M.cFront.localToWorld(new THREE.Vector3(-W / 2, 0, 0))), r = toScreen(M.cFront.localToWorld(new THREE.Vector3(W / 2, 0, 0)));
      const fit = Math.min(1, (0.86 * Math.hypot(r.x - l.x, r.y - l.y)) / Math.max(1, OV.title.offsetWidth));
      OV.title.style.transform = `translate(-50%, -50%) scale(${fit.toFixed(4)})`;
    } else { OV.title.style.left = ''; OV.title.style.top = ''; OV.title.style.transform = ''; }
    OV.title.querySelector('.rule').style.width = `${(520 * easeInOut(seg(k, 0.15, 0.8))).toFixed(1)}px`;
    OV.title.querySelector('.st').style.opacity = smooth(seg(k, 0.35, 0.8)).toFixed(3);
    OV.title.querySelector('.spec').style.opacity = (0.9 * smooth(seg(k, 0.55, 1))).toFixed(3);
    OV.title.querySelector('.eb').style.opacity = smooth(seg(k, 0.0, 0.5)).toFixed(3);
  }

  updateLens(o.lens);
  updateAtt(o.att);
  updateReply(o.reply);
  updateEnd(o.end);
  OV.terms.update(t);
  updateProg(t);
  updateFormula(o.formula);
  updateHeads6(o.heads6);
  updateFfnMap(o.ffnmap);
  updateRings(o.rings);
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
    if (!OV.end2.innerHTML) OV.end2.innerHTML = `<div class="brand">在线体验</div><div class="url">caijiechao.com/blackbox/</div>`;
    OV.end2.style.opacity = a2.toFixed(3);
    OV.end2.style.transform = `translate(-50%, -50%) translateY(${((1 - easeOut(D.k2 ?? 1)) * 12).toFixed(2)}px)`;
  }
  // 继续探索
  const a3 = D?.a3 ?? 0, a4 = D?.a4 ?? 0, a5 = D?.a5 ?? 0;
  for (const [e, a] of [[OV.end3, a3], [OV.end4, a4], [OV.end5, a5]]) {
    e.style.display = a > 0.001 ? (e === OV.end5 ? 'flex' : 'block') : 'none';
    if (a > 0.001) { e.style.opacity = a.toFixed(3); e.style.transform = `translate(-50%, -50%) translateY(${((1 - easeOut(Math.min(1, a * 1.3))) * 14).toFixed(2)}px)`; }
  }
  if (a4 > 0.001) [...OV.end4.querySelectorAll('.more > *')].forEach((c, i) => { const k = smooth(clamp((D.k4 ?? 1) * 4 - Math.floor(i / 2) * 0.9)); c.style.opacity = k.toFixed(3); });
  if (a5 > 0.001) { const k = D.k5 ?? 1; OV.end5.querySelector('.qr').style.opacity = smooth(seg(k, 0, 0.5)).toFixed(3); OV.end5.querySelector('.hint').style.opacity = smooth(seg(k, 0.4, 0.9)).toFixed(3); OV.end5.querySelector('.cred').style.opacity = (0.8 * smooth(seg(k, 0.6, 1))).toFixed(3); }
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
    // v5：讲到 Q / K / V 中的哪一个，哪块矩阵亮；打分那一步先不显示弧线上的权重（softmax 之后才有）
    focusAttn(f.attnFocus ?? null, f.attnFocusK ?? 1);
    focusMlp(f.mlpFocus ?? null, f.mlpFocusK ?? 1);
    for (const b of M.beamList || []) b.m.children.forEach((c) => { if (c.el) c.visible = f.beamLabels !== false; });
    // 层板展开时的部件标签：只留正在讲的那一个（exOnly），其余收起
    if (f.exOnly !== undefined) for (const [k, lb] of Object.entries(M.exLabels)) if (k !== f.exOnly) lb.visible = false;
    if (f.hideMlpLabels) {
      const P = M.mats.mlp;
      for (const l of [P.xl, P.x2l, P.go.lbl, P.uo.lbl, P.dinL, P.dout.lbl, P.g.lb, P.u.lb, P.d.lb]) if (l) l.visible = false;
    }
    M.detail.panelTitle.visible = !f.hideMlpLabels;
    isolate(f.iso ?? 0, f.st);
    // 层板右端的逻辑透镜读数：机器太宽（78 个位置），挪到当前词元的光柱旁边
    const lx = M.x(Q.row(f.st.g)) + (f.lensDx ?? 0.75);
    for (const sl of M.slabs) sl.lbl.position.x = lx;
    // 镜头需要的临时隐藏 / 压暗（每帧都重新赋值，不会残留）
    const hide = new Set(f.hide || []);
    if (hide.has('attnMats')) M.mats.attn.visible = false;
    if (hide.has('mlpMats')) M.mats.mlp.visible = false;
    if (hide.has('bars')) { for (const b of M.bars) { b.visible = false; b.lbl.visible = false; } M.strip.visible = false; M.ball.visible = false; }
    if (hide.has('mlpVecs')) for (const o of [M.mats.mlp.x, M.mats.mlp.x2, M.mats.mlp.go, M.mats.mlp.uo]) o.visible = false;
    for (const o of [M.exAdd1, M.exAdd2, M.exRing1, M.exRing2, M.exUnit]) o.visible = !hide.has('exDeco');
    if (f.lensWin != null && f.st.step.L != null) M.slabs.forEach((sl, L) => { if (L < f.st.step.L - f.lensWin || L > f.st.step.L) sl.lbl.visible = false; });
    if (f.slabDim) M.slabs.forEach((sl) => { sl.material.opacity *= 1 - f.slabDim; sl.edge.material.opacity *= 1 - f.slabDim; sl.lbl.visible = false; });
    // 开场：对话框、飞行的词元、托盘上方块的落下时间
    OPENING.update(t, SC.open.camAt);
    // 片名落在黑箱正面的那几秒，正面的型号字不亮；片名淡出以后再亮（分镜里给出 boxLabel）
    const fa = f.boxLabel ?? 1;
    for (const c of M.cFront.children) { if (!c.isMesh) continue; if (c.material.map) c.material.opacity = fa; else c.scale.setScalar(Math.max(1e-3, fa)); c.visible = fa > 0.01; }
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

// 封面：片名落版那一刻的画面（黑箱正面的发光边框 + 片名卡 + 下面飞行的词元），去掉字幕等叠加层
function poster(t) {
  seek(t ?? SC.open.titleIn + 2.55);
  for (const sel of ['.sub', '.strip', '.chapter', '.card', '.corner', '.lenspanel', '.attgrid']) document.querySelectorAll(sel).forEach((e) => { e.style.display = 'none'; });
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
