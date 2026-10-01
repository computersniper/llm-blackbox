// 电影模式：用网站的 3D 舞台（public/js/stage/）和真实的 Qwen3-0.6B 数据，按脚本化的时间线
// 驱动机器和摄影机，HTML 叠加字幕。window.__film.renderAt(t) 渲染第 t 秒的画面，逐帧调用即可确定地出片。
//
// 片子讲的是「法国的首都是哪里？」→「法国的首都是巴黎。」这一次真实推理：
//   第 1 个词「法国」完整走一遍（分词、嵌入、28 层、输出头、采样），自回归加速到第 5 个词「巴黎」，
//   再走一遍 28 层，停在第 20 层（第 13 头把 82% 的注意力投向「法国」），拆到乘加和比特，最后收尾。
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
const QID = params.get('q') || 'q12';
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
  tameScene();
  buildExtras();
  buildOverlays();
  window.__film = { ready: true, fps: FPS, duration: SC.end, renderAt, seek, events: () => SC.events.slice().sort((a, b) => a.t - b.t), score: { sections: SC.sections, shots: SC.shots, bpm: SC.bpm, end: SC.end } };
  if (PREVIEW) startPreview();
}

// 场景的“背景光”压暗：地面的径向光晕、漂浮微粒都调淡（用户不要背景光源一样的辉光）
function tameScene() {
  for (const o of E.scene.children) {
    if (o.isMesh && o.geometry?.type === 'CircleGeometry') o.material.opacity = 0.32;
    if (o.isPoints) { o.material.opacity = 0.2; o.material.size = 0.04; }
    if (o.type === 'GridHelper') o.material.opacity = 0.22;
  }
}

/* ================================================================ 场景里额外的东西 */

function buildExtras() {
  // 1. 托盘方块下面的真实词元编号
  extras.ids = Q.tokens.map((t, i) => {
    const lb = label(String(t.id), `lbl tid${t.role === 'user' ? ' u' : ''}`);
    lb.position.set(0, -0.02, 0.24);
    lb.center.set(0.5, -0.25);
    M.tiles[i].add(lb);
    return lb;
  });

  // 2. 嵌入表：151936 行 × 1024 列，每个词元按真实编号在表里的位置亮起一行，取出的向量飞向托盘
  const g = (extras.emb = new THREE.Group());
  const TW = 1.5, TH = 7.6, X = M.x(0) - 3.4, Y1 = 8.0;
  extras.embGeo = { TW, TH, X, Y1 };
  const glass = new THREE.Mesh(new THREE.PlaneGeometry(TW, TH), new THREE.MeshBasicMaterial({ color: 0x0b1a2e, transparent: true, opacity: 0.55, depthWrite: false }));
  glass.position.set(X, Y1 - TH / 2, 0);
  g.add(glass);
  const edge = new THREE.LineSegments(new THREE.EdgesGeometry(glass.geometry), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.45 }));
  edge.position.copy(glass.position);
  g.add(edge);
  // 细横线：只是表格的“行”的示意（不代表数值）
  const lines = [];
  for (let k = 1; k < 40; k++) { const y = Y1 - (TH * k) / 40; lines.push(v3(X - TW / 2, y, 0.001), v3(X + TW / 2, y, 0.001)); }
  const grid = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(lines), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.06 }));
  g.add(grid);
  const head = label(`嵌入表<small>151936 × 1024</small>`, 'lbl part');
  head.position.set(X, Y1 + 0.1, 0);
  head.center.set(0.5, 1.2);
  g.add(head);
  const roleCol = { system: 0x8fa6d6, user: 0x5ee4f0, assistant: 0xffb65c, tpl: 0xb39dff };
  extras.rows = Q.tokens.slice(0, Q.P).map((t, i) => {
    const y = Y1 - TH * (t.id / 151936);
    const c = roleCol[t.role] || roleCol.tpl;
    const row = new THREE.Mesh(new THREE.PlaneGeometry(TW, 0.022), new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0, depthWrite: false }));
    row.position.set(X, y, 0.004);
    g.add(row);
    const a = v3(X + TW / 2, y, 0.02), b = v3(M.x(i), 0.42, 0.05);
    const ctrl = v3((a.x + b.x) / 2, Math.max(a.y, b.y) + 1.6 + 0.02 * i, 0.6);
    const curve = new THREE.QuadraticBezierCurve3(a, ctrl, b);
    const pk = new THREE.Mesh(new THREE.SphereGeometry(0.045, 12, 8), new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.95 }));
    g.add(pk);
    const trail = new THREE.Mesh(new THREE.TubeGeometry(curve, 40, 0.006, 4, false), new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.0, depthWrite: false, blending: THREE.AdditiveBlending }));
    g.add(trail);
    let lb = null;
    if (t.role === 'user' && !t.sp && i >= 24) {
      lb = label(`${t.id} <span style="color:var(--ink)">${tk(t.s)}</span>`, 'lbl emb');
      lb.position.set(X + TW / 2 + 0.08, y, 0.02);
      lb.center.set(0, 0.5);
      g.add(lb);
    }
    return { row, pk, trail, curve, lb, i };
  });
  // 编号离得近的用户词元标签错开一点，免得叠在一起
  const users = extras.rows.filter((r) => r.lb).sort((a, b) => b.row.position.y - a.row.position.y);
  let lastY = Infinity;
  for (const r of users) { const y = Math.min(r.row.position.y, lastY - 0.24); r.lb.position.y = y; lastY = y; }
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
    if (on) lb.el.style.opacity = (ia * (ex.idsFocus ? (Q.tokens[i].role === 'user' && !Q.tokens[i].sp ? 1 : 0.35) : 1)).toFixed(3);
  });
  // 嵌入表
  const ea = ex.emb ?? 0;
  extras.emb.visible = ea > 0.01;
  if (extras.emb.visible) {
    extras.emb.traverse((o) => { if (o.material && o.userData.base == null) o.userData.base = o.material.opacity; });
    const k = ex.embK ?? 0; // 0 → 1：取向量的进度
    for (const r of extras.rows) {
      const t0 = r.i / Q.P * 0.55, local = clamp((k - t0) / 0.45);
      r.row.material.opacity = ea * (local > 0 ? 0.95 * (1 - 0.6 * smooth(seg(local, 0.6, 1))) : 0.15);
      r.pk.visible = local > 0 && local < 1;
      if (r.pk.visible) r.pk.position.copy(r.curve.getPoint(easeInOut(local)));
      r.trail.material.opacity = ea * (local > 0 ? 0.22 * (1 - smooth(seg(local, 0.7, 1.4))) + 0.05 : 0);
      if (r.lb) { r.lb.visible = local > 0.05; r.lb.el.style.opacity = (ea * smooth(local * 3)).toFixed(3); }
    }
    for (const o of [extras.emb.children[0], extras.emb.children[1], extras.emb.children[2]]) o.material.opacity = o.userData.base * ea;
    extras.emb.children[3].el.style.opacity = ea.toFixed(3);
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
  const key = `${A.L}|${A.g}|${A.head}|${A.reveal.toFixed(2)}`;
  if (key === OV.attKey) return;
  OV.attKey = key;
  const i = Q.row(A.g), n = i + 1;
  const rows = [];
  for (let h = 0; h < Q.H; h++) {
    const w = new Array(n).fill(0);
    for (const r of Q.att(A.L, h, i)) w[r.j] = r.w;
    const on = h === A.head;
    const shown = h / Q.H < A.reveal;
    rows.push(`<div class="row${on ? ' on' : ''}"><span class="hn">H${String(h).padStart(2, '0')}</span>${w.map((v, j) => {
      const a = shown ? Math.pow(v, 0.6) : 0;
      const col = on ? `rgba(255,182,92,${(0.08 + 0.92 * a).toFixed(3)})` : `rgba(94,240,212,${(0.05 + 0.9 * a).toFixed(3)})`;
      return `<i style="background:${col}${on && v > 0.5 ? ';outline:1.5px solid #fff' : ''}"></i>`;
    }).join('')}</div>`);
  }
  const cols = Q.tokens.slice(0, n).map((t, j) => `<span class="${t.role === 'user' && !t.sp ? 'u' : ''}">${t.sp ? '·' : tk(t.s).slice(0, 3)}</span>`).join('');
  OV.att.innerHTML = `<div class="h">第 ${A.L} 层 · 16 个头的注意力<small>查询：${q(Q.tokens[i].s)}（第 ${i} 位）· 每行 = 一个头</small></div>${rows.join('')}<div class="cols">${cols}</div>`;
}

// 回答一个词一个词长出来，下面是它当时的概率和名次
function updateReply(R) {
  OV.reply.style.display = R && R.a > 0.001 ? 'flex' : 'none';
  if (!R || R.a <= 0.001) return;
  OV.reply.style.opacity = R.a.toFixed(3);
  const key = `${R.n}|${(R.k ?? 1).toFixed(2)}`;
  if (key === OV.replyKey) return;
  OV.replyKey = key;
  OV.reply.innerHTML = Q.steps.slice(0, R.n).map((st, g) => {
    const sp = st.chosenS === '<|im_end|>';
    const newest = g === R.n - 1;
    const k = newest ? R.k ?? 1 : 1;
    return `<div class="c${newest ? ' new' : ''}${sp ? ' sp' : ''}" style="opacity:${(0.25 + 0.75 * k).toFixed(3)};width:${sp ? 150 : 40 + 52 * tokPlain(st.chosenS).length}px">
      <span class="s">${sp ? esc(shortSpecial(st.chosenS)) : tk(st.chosenS)}</span>
      <span class="bar"><i style="width:${(st.chosenP1 * 100).toFixed(1)}%"></i></span>
      <span class="p">${pct(st.chosenP1)}</span></div>`;
  }).join('');
}

function updateEnd(D) {
  const a1 = D?.a1 ?? 0, a2 = D?.a2 ?? 0;
  OV.end.style.display = a1 > 0.001 ? 'block' : 'none';
  OV.end2.style.display = a2 > 0.001 ? 'block' : 'none';
  if (a1 > 0.001) {
    if (!OV.end.innerHTML) OV.end.innerHTML = `<div class="ans">${esc(Q.manifest.questions.find((x) => x.id === QID).reply)}</div><div class="stat">${SC.statLine}</div>`;
    OV.end.style.opacity = a1.toFixed(3);
    OV.end.querySelector('.stat').style.opacity = (D.k1 ?? 1).toFixed(3);
  }
  if (a2 > 0.001) {
    if (!OV.end2.innerHTML) OV.end2.innerHTML = `<div class="brand">揭开黑箱</div><div class="url">caijiechao.com/blackbox/</div><div class="cred">在网页里，你可以亲手把这台机器一层层拆开<br><span class="m" style="font-size:15px;letter-spacing:.12em">Qwen3-0.6B 真实离线运行数据 · 画面与音乐均由程序生成</span></div>`;
    OV.end2.style.opacity = a2.toFixed(3);
    OV.end2.style.transform = `translate(-50%, -50%) translateY(${((1 - easeOut(D.k2 ?? 1)) * 12).toFixed(2)}px)`;
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

let lastT = null, lastPanel = '';
function renderAt(t, { render = true } = {}) {
  const dt = lastT == null || t <= lastT || t - lastT > 0.5 ? 1 / FPS : t - lastT;
  lastT = t;
  const f = SC.frame(t, { M, E, Q });
  if (f.st) {
    M.update(f.st, dt, t);
    if (f.after) f.after(M);
    // 神经元阵列点亮的先后顺序原来用 Math.random：换成确定的哈希
    if (M.detail.panelKey && M.detail.panelKey !== lastPanel) {
      lastPanel = M.detail.panelKey;
      M.detail.order = Float32Array.from({ length: M.detail.order.length }, (_, n) => hash01(n));
      M.detail.litState = -1;
    }
  }
  updateExtras(t, f);
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
