// 训练视频（电影模式）：用训练页第一章的 3D 机器（public/train/js/glass3d/machine.js）和真实训练记录，
// 按 score.js 的时间线驱动机器和摄影机，HTML 叠加字幕和讲解。window.__film.renderAt(t) 渲染第 t 秒的画面，逐帧调用即可确定地出片。
// 渲染器、运镜、叠加层的小工具和推理视频共用 tools/video/lib/（只引用、不修改）。
import { FilmEngine, THREE } from '../lib/engine.js';
import { clamp, lerp, seg, smooth, easeOut, easeInOut } from '../lib/cam.js';
import { el, CueLayer } from '../lib/overlay.js';
import { loadAll } from './data.js';
import { buildScore } from './score.js';
import { GlassMachine } from '/public/train/js/glass3d/machine.js';
import { LIN } from '/public/train/js/glass3d/palette.js';
import { adamAt } from '/public/train/js/glass/math.js';

const params = new URLSearchParams(location.search);
const FPS = Number(params.get('fps') || 30);
const PREVIEW = params.has('preview');
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

let E, M, D, SC, QR_SVG = '', CAP = null;
const CAP_DIR = 'sitecap-train';
const SITE = new Map();
const OV = {};
// 快进时相邻两帧之间的插值（score 每帧给出 { k, f }）
const FRAC = { k: -1, f: 0 };

/* ================================================================ 启动 */

async function boot() {
  await Promise.all([
    document.fonts.load('900 100px "Film Serif"', 'AI 是怎么学会的'),
    document.fonts.load('600 40px "Film Serif"', '床前明月光疑是地上霜举头望低思故乡'),
    document.fonts.load('600 76px "PingFang SC"', '床前明月光'),
    document.fonts.load('600 76px "Noto Sans SC"', '床前'),
    document.fonts.load('400 38px "Film Sans"', '训练'),
    document.fonts.load('400 20px "JetBrains Mono"', '0123'),
    document.fonts.load('700 20px "JetBrains Mono"', '0123'),
  ]);
  QR_SVG = await (await fetch('/tools/video/qr-blackbox-train.svg')).text();
  D = await loadAll();
  CAP = await fetch(`/ext/${CAP_DIR}/meta.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  SC = buildScore(D, CAP);
  if (CAP && SC.endSite?.segs.length) await preloadSite();

  E = new FilmEngine($('#gl'), { pixelRatio: Number(params.get('pr') || 1.5) });
  // 机器的取景函数（fitBox）要这几个量：片子里没有侧边面板
  E.padB = 0; E.padT = 0; E.insetR = 0;
  E.shiftY = 40;
  E.applyOffset();
  M = new GlassMachine(E, D, { ctx: { gsel: null } });
  patchMachine();
  tameScene();
  // Eᵀ 的上标 ᵀ 在思源宋体里没有，换成普通的上标
  for (const o of Object.values(M.lw)) { o.short = o.short.replace('Eᵀ', 'E<sup>T</sup>'); o.full = o.full.replace('Eᵀ', 'E<sup>T</sup>'); }
  buildOverlays();
  window.__film = { M, E, D, SC, ready: true, fps: FPS, duration: SC.end, renderAt, seek, poster, probe, adamAt, events: () => SC.events.slice().sort((a, b) => a.t - b.t), score: { sections: SC.sections, shots: SC.shots, bpm: SC.bpm, end: SC.end, subs: SC.subs, terms: SC.terms, facts: SC.facts } };
  if (PREVIEW) startPreview();
}

// 快进：机器的流程（F）可以由分镜直接给；权重、概率、色标在相邻两帧之间插值，图案连续地长出来
function patchMachine() {
  const flow0 = M.flow.bind(M);
  M.flow = (st, p) => st.F || flow0(st, p);
  const probs0 = D.probs;
  D.probs = (k, i, v) => {
    const a = probs0(k, i, v);
    return FRAC.k === k && FRAC.f > 0 && k < D.NF - 1 ? a + (probs0(k + 1, i, v) - a) * FRAC.f : a;
  };
  const att0 = D.att;
  D.att = (k, h, i, j) => {
    const a = att0(k, h, i, j);
    return FRAC.k === k && FRAC.f > 0 && k < D.NF - 1 ? a + (att0(k + 1, h, i, j) - a) * FRAC.f : a;
  };
  const cs0 = M.cs.bind(M);
  M.cs = (k, W) => {
    const a = cs0(k, W);
    if (FRAC.k !== k || FRAC.f <= 0 || k >= D.NF - 1) return a;
    const b = cs0(k + 1, D.W(k + 1));
    return { mat: lerp(a.mat, b.mat, FRAC.f), g: lerp(a.g, b.g, FRAC.f) };
  };
}

// 背景的地面网格、漂浮微粒调淡；漂浮微粒的转动改成按时间算（每个渲染进程都一样）
function tameScene() {
  for (const o of E.scene.children) {
    if (o.isPoints) { o.material.opacity = 0.18; o.material.size = 0.04; }
    if (o.type === 'GridHelper') o.material.opacity = 0.2;
  }
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

  // 开场：一句问题（一个字一个字打出来）+ 一句回答
  OV.q = el('div', 'tq', `<div class="l1"></div><div class="l2">靠的是：<b>训练</b></div>`, ov);
  OV.qChars = [...SC.Q1];
  OV.qKey = -1;
  // 片名
  OV.title = el('div', 'title', `<div class="eb">INSIDE A TRAINING RUN</div><h1>AI 是怎么学会的</h1><div class="rule"></div><div class="st">看一个小模型从零学会背《静夜思》</div><div class="spec">${D.P.toLocaleString('en-US')} PARAMETERS · ${D.S} STEPS · REAL TRAINING RECORD</div>`, ov);

  // 初始值的直方图（真实的 2,880 个矩阵参数；γ 一律从 1 开始，不算在里面）
  OV.hist = el('div', 'tcard hist', '', ov);
  {
    const vals = [];
    for (const p of D.params) if (!p.norm) for (let i = p.off; i < p.off + p.n; i++) vals.push(D.w0[i]);
    const NB = 41, lo = -0.082, hi = 0.082, w = (hi - lo) / NB;
    const cnt = new Array(NB).fill(0);
    for (const v of vals) cnt[Math.min(NB - 1, Math.max(0, Math.floor((v - lo) / w)))]++;
    const mx = Math.max(...cnt);
    const W = 520, H = 190;
    const bars = cnt.map((c, i) => `<rect class="b" data-i="${i}" x="${(i / NB) * W + 1}" width="${W / NB - 2}" y="${H - (c / mx) * H}" height="${(c / mx) * H}" fill="${i < NB / 2 - 0.5 ? '#6b9bff' : i > NB / 2 - 0.5 ? '#ffb65c' : '#8592ad'}"/>`).join('');
    const ax = [-0.06, -0.03, 0, 0.03, 0.06].map((v) => `<text x="${((v - lo) / (hi - lo)) * W}" y="${H + 26}" text-anchor="middle">${v === 0 ? '0' : (v > 0 ? '+' : '−') + Math.abs(v).toFixed(2)}</text>`).join('');
    OV.hist.innerHTML = `<div class="h">出厂时的 ${vals.length.toLocaleString('en-US')} 个参数<small>真实初始值 · 每根柱子 = 这个范围里有几个</small></div><svg width="${W}" height="${H + 34}" viewBox="0 0 ${W} ${H + 34}"><line x1="0" x2="${W}" y1="${H}" y2="${H}" stroke="rgba(150,180,230,.25)"/>${bars}${ax}</svg>`;
    OV.histBars = [...OV.hist.querySelectorAll('rect.b')];
    OV.histH = cnt.map((c) => (c / mx) * H);
    OV.histHmax = H;
  }

  // 唯一一次逐数计算：一个参数的 AdamW（真实数字，逐行出现）
  {
    const A = SC.calc.A1, lc = SC.facts.LC;
    const n4 = (v) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(5);
    const ratio = A.mh / Math.sqrt(A.vh);
    const r2 = n4(ratio).replace(/(\.\d\d)\d+$/, '$1');
    OV.calc = el('div', 'tcard calc', `
      <div class="h">一个参数的一次更新<small>嵌入表 E[「${esc(D.ch(lc.i))}」, ${lc.j}] · 第 1 步 · 真实记录</small></div>
      <div class="ln" data-l="0"><span class="k">参数</span><span class="v">w = <b class="cw">${A.w.toFixed(5)}</b></span></div>
      <div class="ln" data-l="1"><span class="k">梯度</span><span class="v">g = <b class="cg">${n4(A.g)}</b><small>负数：把 w 调大，损失会变小</small></span></div>
      <div class="ln" data-l="2"><span class="k">AdamW</span><span class="v">m̂ = <b>${n4(A.mh)}</b>　√v̂ = <b>${Math.sqrt(A.vh).toFixed(5)}</b>　m̂ ÷ √v̂ = <b class="cy">${r2}</b><small>第一步，步子只看方向</small></span></div>
      <div class="ln" data-l="3"><span class="k">步子</span><span class="v">Δw = −${A.lr.toFixed(3)} × (${r2}) − ${A.lr.toFixed(3)} × ${A.wd} × w = <b class="cy">${A.dw >= 0 ? '+' : '−'}${Math.abs(A.dw).toFixed(6)}</b></span></div>
      <div class="ln" data-l="4"><span class="k">写回</span><span class="v">w：<b class="cw">${A.w.toFixed(5)}</b> → <b class="cy">${A.w1.toFixed(5)}</b></span></div>`, ov);
    OV.calcLn = [...OV.calc.querySelectorAll('.ln')];
  }

  // 损失曲线（全部 25 段上的平均损失，每一步都有）+ 全零对照
  OV.chart = el('div', 'tcard chart', '', ov);
  {
    const W = 520, H = 210, S = D.S;
    const x = (s) => ((s - 1) / (S - 1)) * W, y = (v) => H - (v / 3.2) * H;
    OV.chartXY = { W, H, x, y };
    const pts = Array.from({ length: S }, (_, i) => `${x(i + 1).toFixed(1)},${y(D.evalLoss[i]).toFixed(1)}`);
    const zp = Array.from({ length: S }, (_, i) => `${x(i + 1).toFixed(1)},${y(D.zLoss[i]).toFixed(1)}`);
    const grid = [0, 1, 2, 3].map((v) => `<line x1="0" x2="${W}" y1="${y(v)}" y2="${y(v)}" stroke="rgba(150,180,230,${v ? 0.1 : 0.25})"/><text class="ax" x="-10" y="${y(v) + 5}" text-anchor="end">${v}</text>`).join('');
    const xt = [1, 50, 100, 150, 200].map((s) => `<text class="ax" x="${x(s)}" y="${H + 26}" text-anchor="middle">${s}</text>`).join('');
    OV.chart.innerHTML = `<div class="h">损失<small>全部 25 段的平均 · 越低越好</small><span class="val"></span></div>
      <svg width="${W}" height="${H + 34}" viewBox="0 0 ${W} ${H + 34}" style="overflow:visible">${grid}${xt}
        <line x1="0" x2="${W}" y1="${y(SC.facts.LN20)}" y2="${y(SC.facts.LN20)}" stroke="rgba(255,107,147,.45)" stroke-dasharray="5 6"/><text class="ax" x="${W - 4}" y="${y(SC.facts.LN20) - 8}" text-anchor="end" fill="rgba(255,140,170,.8)">瞎猜 ln 20</text>
        <polyline class="zl" points="${zp.join(' ')}" fill="none" stroke="#ff6b93" stroke-width="3"/>
        <text class="zt" x="${W * 0.5}" y="${y(D.zLoss[0]) - 14}" text-anchor="middle">全零初始化：一直 ${D.zLoss[S - 1].toFixed(3)}</text>
        <polyline class="ln" points="${pts.join(' ')}" fill="none" stroke="#5ef0d4" stroke-width="3"/>
        <circle class="dot" r="6" fill="#5ef0d4"/>
      </svg><div class="foot"><span class="step"></span></div>`;
    OV.chartLine = OV.chart.querySelector('polyline.ln');
    OV.chartZ = [OV.chart.querySelector('polyline.zl'), OV.chart.querySelector('text.zt')];
    OV.chartDot = OV.chart.querySelector('circle.dot');
    OV.chartVal = OV.chart.querySelector('.val');
    OV.chartStep = OV.chart.querySelector('.step');
    OV.chartLen = OV.chartLine.getTotalLength?.() || 0;
  }

  // 固定样例 8 个位置：正确答案的概率（快进时）
  OV.preds = el('div', 'tcard preds', `<div class="h">固定样例：每个位置猜对的概率</div><div class="row">${Array.from({ length: D.T }, (_, p) => `<div class="c"><div class="io">${esc(D.ch(D.fixed[p]))}<i>→</i><b>${esc(D.ch(D.fixed[p + 1]))}</b></div><div class="bar"><i></i></div><div class="pv"></div></div>`).join('')}</div>`, ov);
  OV.predC = [...OV.preds.querySelectorAll('.c')];

  // 「月」后面接什么：诗里的两处
  OV.call = el('div', 'tcard call', `<div class="h">诗里的两个「月」</div><div class="pl">床前明<b>月</b><span class="g">光</span>，疑是地上霜。</div><div class="pl">举头望明<b>月</b><span class="g">，</span>低头思故乡。</div>`, ov);
  // 「月」这个位置，第一个注意力头在看谁（真实注意力，快进时插值）
  OV.att = el('div', 'tcard attrow', `<div class="h">注意力：「月」在看哪个字<small>第 1 个头</small></div><div class="row">${Array.from({ length: SC.facts.PM + 1 }, (_, j) => `<div class="c"><div class="bar"><i></i></div><div class="pv"></div><div class="ch">${esc(D.ch(D.fixed[j]))}</div></div>`).join('')}</div><div class="foot"></div>`, ov);
  OV.attC = [...OV.att.querySelectorAll('.c')];
  OV.attFoot = OV.att.querySelector('.foot');
  // 概率架上「月」那一行的读数（挂在画面右侧）
  OV.rowtag = el('div', 'tcard rowtag', '', ov);

  // 一个参数的一生（exact 记录：每一步的 float32 原值）
  OV.life = el('div', 'tcard life', '', ov);
  {
    const life = SC.facts.life, S = D.S, W = 560, H = 220;
    const lo = Math.min(...life), hi = Math.max(...life);
    const x = (s) => (s / S) * W, y = (v) => H - ((v - lo) / (hi - lo)) * H;
    const pts = life.map((v, s) => `${x(s).toFixed(1)},${y(v).toFixed(1)}`);
    const lc = SC.facts.LC;
    OV.lifeXY = { x, y };
    OV.life.innerHTML = `<div class="h">E[「${esc(D.ch(lc.i))}」, ${lc.j}] 的一生<small>每一步的真实值</small><span class="val"></span></div>
      <svg width="${W}" height="${H + 34}" viewBox="0 0 ${W} ${H + 34}" style="overflow:visible">
        <line x1="0" x2="${W}" y1="${H}" y2="${H}" stroke="rgba(150,180,230,.25)"/>
        ${[0, 50, 100, 150, 200].map((s) => `<text class="ax" x="${x(s)}" y="${H + 26}" text-anchor="middle">${s}</text>`).join('')}
        <text class="ax" x="-10" y="${y(lo) + 5}" text-anchor="end">${lo.toFixed(3)}</text><text class="ax" x="-10" y="${y(hi) + 5}" text-anchor="end">${hi.toFixed(2)}</text>
        <polyline class="ln" points="${pts.join(' ')}" fill="none" stroke="#ffb65c" stroke-width="3"/>
        <circle class="dot" r="6" fill="#ffd9a6"/>
      </svg>`;
    OV.lifeLine = OV.life.querySelector('polyline.ln');
    OV.lifeDot = OV.life.querySelector('circle.dot');
    OV.lifeVal = OV.life.querySelector('.val');
    OV.lifeLen = OV.lifeLine.getTotalLength?.() || 0;
  }

  // 片尾：训练页录屏放在一个干净的窗口框里（不模仿任何浏览器），鼠标和点击是叠加上去的；最后落版网址 + 二维码
  OV.win = el('div', 'win', `<div class="bar"><i></i><i></i><i></i><span class="url">caijiechao.com/blackbox/train/</span></div><canvas width="1920" height="1080"></canvas>`, ov);
  OV.winCtx = OV.win.querySelector('canvas').getContext('2d');
  OV.winIdx = -1;
  OV.cur = el('div', 'cur', `<span class="rip"></span><svg viewBox="0 0 24 24" width="30" height="30"><path d="M5 2.5v17.2l4.6-4.3 3 6.6 2.9-1.3-3-6.5h6.2z" fill="#fff" stroke="#05080f" stroke-width="1.4" stroke-linejoin="round"/></svg>`, ov);
  OV.fin = el('div', 'fin7', `<div class="url">caijiechao.com/blackbox/train/</div><div class="qr">${QR_SVG}</div><div class="hint">扫码打开训练页，亲手一步步看</div>`, ov);
}

const show = (e, a, disp = 'block') => { const on = a > 0.001; e.style.display = on ? disp : 'none'; if (on) e.style.opacity = a.toFixed(3); return on; };

function updateOverlays(t, f) {
  const o = f.ov || {};
  OV.subs.update(t);
  OV.terms.update(t);
  OV.band.style.opacity = (o.band ?? 1).toFixed(3);

  // 章节进度
  const pg = SC.progAt(t);
  OV.prog.style.opacity = pg.a.toFixed(3);
  OV.prog.style.display = pg.a > 0.001 ? 'flex' : 'none';
  const pk = `${pg.i}`;
  if (pk !== OV.progKey) { OV.progKey = pk; OV.progIt.forEach((e, i) => { e.classList.toggle('on', i === pg.i); e.classList.toggle('done', i < pg.i); }); }

  // 开场的问题
  if (show(OV.q, o.q?.a ?? 0)) {
    const n = Math.max(0, Math.min(OV.qChars.length, o.q.n));
    if (n !== OV.qKey) { OV.qKey = n; OV.q.firstChild.innerHTML = `${esc(OV.qChars.slice(0, n).join(''))}<span class="caret"${n >= OV.qChars.length ? ' style="opacity:0"' : ''}></span>`; }
    OV.q.lastChild.style.opacity = (o.q.ans ?? 0).toFixed(3);
    OV.q.lastChild.style.transform = `translateY(${((1 - easeOut(o.q.ans ?? 0)) * 12).toFixed(1)}px)`;
  }
  if (show(OV.title, o.title?.a ?? 0)) OV.title.style.transform = `translate(-50%, -50%) translateY(${((1 - easeOut(o.title.a)) * 14).toFixed(1)}px)`;

  // 直方图：柱子从 0 长出来
  if (show(OV.hist, o.hist?.a ?? 0)) {
    const k = o.hist.k ?? 1, H = OV.histHmax;
    OV.histBars.forEach((b, i) => { const h = OV.histH[i] * clamp(k * 1.25 - Math.abs(i - 20) / 40); b.setAttribute('y', (H - h).toFixed(1)); b.setAttribute('height', h.toFixed(1)); });
  }

  // AdamW 算式：逐行出现，正在讲的那一行高亮
  if (show(OV.calc, o.calc?.a ?? 0)) {
    const L = SC.calc.L;
    OV.calcLn.forEach((e, i) => {
      const a = smooth(seg(o.calc.t, L[i] - 0.1, L[i] + 0.4));
      e.style.opacity = a.toFixed(3);
      e.classList.toggle('on', o.calc.t >= L[i] && (i === L.length - 1 || o.calc.t < L[i + 1]));
    });
  }

  // 损失曲线
  if (show(OV.chart, o.chart?.a ?? 0)) {
    const s = o.chart.s, { x, y } = OV.chartXY;
    const i = Math.max(0, Math.min(D.S - 1, Math.floor(s) - 1)), fr = s - Math.floor(s);
    const v = i < D.S - 1 ? lerp(D.evalLoss[i], D.evalLoss[i + 1], fr) : D.evalLoss[i];
    const L = OV.chartLen;
    if (L) { OV.chartLine.style.strokeDasharray = `${L}`; OV.chartLine.style.strokeDashoffset = `${(L * (1 - (s - 1) / (D.S - 1))).toFixed(1)}`; }
    OV.chartDot.setAttribute('cx', x(s).toFixed(1));
    OV.chartDot.setAttribute('cy', y(v).toFixed(1));
    OV.chartVal.textContent = (s >= D.S - 0.01 ? SC.facts.evalLast : v).toFixed(3);
    OV.chartStep.innerHTML = `第 <b>${Math.round(s)}</b> 步 / ${D.S}`;
    const z = o.chart.zero ?? 0;
    OV.chartZ.forEach((e) => { e.style.opacity = z.toFixed(3); });
    OV.chart.classList.toggle('z', z > 0.01);
  }

  // 固定样例的预测
  if (show(OV.preds, o.preds?.a ?? 0)) {
    const { k } = o.preds.frac;
    OV.predC.forEach((c, p) => {
      const pr = D.probs(k, p, D.fixed[p + 1]);
      c.querySelector('.bar i').style.width = `${(pr * 100).toFixed(1)}%`;
      c.querySelector('.pv').textContent = `${(pr * 100).toFixed(0)}%`;
      c.classList.toggle('ok', pr > 0.5);
    });
  }

  if (show(OV.call, o.call?.a ?? 0)) OV.call.style.transform = `translateY(${((1 - easeOut(o.call.a)) * 12).toFixed(1)}px)`;

  // 注意力：「月」看谁
  if (show(OV.att, o.att?.a ?? 0)) {
    const { k } = o.att.frac;
    let best = 0;
    const vals = OV.attC.map((_, j) => D.att(k, 0, SC.facts.PM, j));
    vals.forEach((v, j) => { if (v > vals[best]) best = j; });
    OV.attC.forEach((c, j) => {
      c.querySelector('.bar i').style.height = `${(vals[j] * 100).toFixed(1)}%`;
      c.querySelector('.pv').textContent = `${(vals[j] * 100).toFixed(0)}%`;
      c.classList.toggle('top', j === best);
    });
    OV.attFoot.innerHTML = `走完 <b>${D.FR[k] + Math.round(o.att.frac.f * ((D.FR[Math.min(D.NF - 1, k + 1)] - D.FR[k]) || 0))}</b> 步`;
  }
  if (show(OV.rowtag, o.rowtag?.a ?? 0)) {
    const { k } = o.rowtag.frac, PM = SC.facts.PM;
    const pa = D.probs(k, PM, D.fixed[PM + 1]), pg = D.probs(k, PM, D.chars.indexOf('光'));
    OV.rowtag.innerHTML = `<div class="h">看到「举头望明月」，下一个字</div><div class="r ok"><b>，</b><span class="bar"><i style="width:${(pa * 100).toFixed(1)}%"></i></span><span class="pv">${(pa * 100).toFixed(0)}%</span></div><div class="r no"><b>光</b><span class="bar"><i style="width:${(pg * 100).toFixed(1)}%"></i></span><span class="pv">${(pg * 100).toFixed(0)}%</span></div>`;
  }

  // 一个参数的一生
  if (show(OV.life, o.life?.a ?? 0)) {
    const s = o.life.s, life = SC.facts.life, { x, y } = OV.lifeXY;
    const i = Math.max(0, Math.min(D.S, Math.floor(s))), fr = s - Math.floor(s);
    const v = i < D.S ? lerp(life[i], life[i + 1], fr) : life[D.S];
    const L = OV.lifeLen;
    if (L) { OV.lifeLine.style.strokeDasharray = `${L}`; OV.lifeLine.style.strokeDashoffset = `${(L * (1 - s / D.S)).toFixed(1)}`; }
    OV.lifeDot.setAttribute('cx', x(s).toFixed(1));
    OV.lifeDot.setAttribute('cy', y(v).toFixed(1));
    OV.lifeVal.innerHTML = `第 ${Math.round(s)} 步　w = <b>${v.toFixed(4)}</b>`;
  }

  updateSite(o.site);
}

/* ---------------------------------------------------------------- 片尾录屏 */

async function preloadSite() {
  const need = new Set();
  for (const sg of SC.endSite.segs) for (let f = sg.f0; f <= sg.f1 + 1e-6; f += 1 / 60) need.add(Math.min(CAP.n - 1, Math.max(0, Math.round(lerp(sg.c0, sg.c1, (f - sg.f0) / (sg.f1 - sg.f0)) * CAP.fps))));
  await Promise.all([...need].map((k) => new Promise((res) => { const im = new Image(); im.onload = () => res(); im.onerror = () => res(); im.src = `/ext/${CAP_DIR}/f${String(k).padStart(4, '0')}.jpg`; SITE.set(k, im); })));
}
function cursorAt(ct) {
  const acts = CAP.actions.filter((a) => a.x != null);
  let prev = acts[0], next = null;
  for (const a of acts) { if (a.t <= ct) prev = a; else { next = a; break; } }
  if (next && next.t - ct < 0.55 && prev !== next) { const k = easeInOut(1 - (next.t - ct) / 0.55); return { x: lerp(prev.x, next.x, k), y: lerp(prev.y, next.y, k), click: 0 }; }
  const click = ct >= prev.t && ct < prev.t + 0.45 ? 1 - (ct - prev.t) / 0.45 : 0;
  return { x: prev.x, y: prev.y, click };
}
function updateSite(S) {
  const on = S && S.a > 0.001 && CAP;
  OV.win.style.display = on ? 'block' : 'none';
  OV.cur.style.display = on && S.cursor > 0.01 ? 'block' : 'none';
  OV.fin.style.display = on && S.finA > 0.001 ? 'flex' : 'none';
  if (!on) return;
  const idx = Math.min(CAP.n - 1, Math.max(0, Math.round(S.ct * CAP.fps)));
  if (idx !== OV.winIdx && SITE.get(idx)?.complete) { OV.winCtx.drawImage(SITE.get(idx), 0, 0, 1920, 1080); OV.winIdx = idx; }
  const k = S.fin ?? 0;
  const cx = lerp(960, 600, k), cy = lerp(474, 482, k), sc = lerp(1, 0.6, k);
  OV.win.style.opacity = S.a.toFixed(3);
  OV.win.style.left = `${cx.toFixed(1)}px`;
  OV.win.style.top = `${cy.toFixed(1)}px`;
  OV.win.style.transform = `translate(-50%, -50%) scale(${sc.toFixed(4)}) translateY(${((1 - easeOut(Math.min(1, S.a * 1.2))) * 18).toFixed(1)}px)`;
  if (S.cursor > 0.01) {
    const c = cursorAt(S.ct);
    const W = 1440, H = 810, BAR = 40;
    const x = cx + (-W / 2 + (c.x / CAP.viewport.w) * W) * sc, y = cy + (-(H + BAR) / 2 + BAR + (c.y / CAP.viewport.h) * H) * sc;
    OV.cur.style.left = `${x.toFixed(1)}px`;
    OV.cur.style.top = `${y.toFixed(1)}px`;
    OV.cur.style.opacity = S.cursor.toFixed(3);
    const rip = OV.cur.firstChild;
    rip.style.opacity = (c.click * 0.9).toFixed(3);
    rip.style.transform = `translate(-50%, -50%) scale(${(0.4 + 1.2 * (1 - c.click)).toFixed(3)})`;
  }
  if (S.finA > 0.001) { OV.fin.style.opacity = S.finA.toFixed(3); OV.fin.style.transform = `translateY(-50%) translateY(${((1 - easeOut(S.finA)) * 14).toFixed(1)}px)`; }
}

/* ================================================================ 机器上的标签 */

// 分镜给一个标签方案：只留讲到的那些（其余收起，画面干净）
function applyLabels(mode, f) {
  const all = [...Object.values(M.lw), ...Object.values(M.la), ...Object.values(M.lr), M.lGauge, M.lClip, M.lGap, M.lWheel, M.lTray, M.lShelf, M.lFactory];
  const keep = new Set();
  const add = (...xs) => xs.forEach((x) => x && keep.add(x));
  if (mode === 'weights') add(...Object.values(M.lw).filter((o) => o.visible));
  if (mode === 'batch') add(M.lWheel);
  if (mode === 'fwd' || mode === 'bwd') add(...Object.values(M.lw).filter((o) => o.visible));
  if (mode === 'loss') add(M.lGauge, M.lShelf);
  for (const o of all) if (!keep.has(o)) o.visible = false;
  // 选中的方块：除非分镜要，不显示切片框
  if (!f.ioMarks) { M.mIn.visible = false; M.mOut.visible = false; }
}

/* ================================================================ 颗粒 */

const grainCtx = $('#grain').getContext('2d');
const grainImg = grainCtx.createImageData(480, 270);
function drawGrain(frame) {
  const d = grainImg.data;
  let s = (frame * 7919 + 13) >>> 0;
  for (let i = 0; i < d.length; i += 4) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const v = s >>> 24;
    d[i] = d[i + 1] = d[i + 2] = v; d[i + 3] = 255;
  }
  grainCtx.putImageData(grainImg, 0, 0);
}

/* ================================================================ 每帧 */

let lastT = null, lastF = null;

function probe() {
  const c = E.camera.position, f = lastF;
  return { shot: f?.shot, view: f?.st?.view, cam: [c.x, c.y, c.z].map((v) => +v.toFixed(3)), k: f?.st?.k, frac: f?.frac };
}

function renderAt(t, { render = true } = {}) {
  const dt = lastT == null || t <= lastT || t - lastT > 0.5 ? 1 / FPS : t - lastT;
  lastT = t;
  const f = (lastF = SC.frame(t, { M, E, D }));
  const st = f.st;
  if (f.frac) { FRAC.k = f.frac.k; FRAC.f = f.frac.f; } else { FRAC.k = -1; FRAC.f = 0; }
  if (f.sel != null) st.view = 'g-param';
  M.setSel(f.sel ?? null);
  M.update(st, dt, t);
  M.selLabel(f.selLbl ?? null);
  applyLabels(f.lbl ?? 'none', f);
  // 概率架：只看「月」那一行时，其余行压暗
  if (f.rowFocus != null && f.rowFocus >= 0) {
    const bc = M.bars.instanceColor.array;
    for (let p = 0; p < D.T; p++) if (p !== f.rowFocus) for (let v = 0; v < D.V; v++) { const c = (p * D.V + v) * 3; bc[c] *= 0.22; bc[c + 1] *= 0.22; bc[c + 2] *= 0.22; }
    M.bars.instanceColor.needsUpdate = true;
  }
  M.dust.rotation.y = t * 0.008;
  E.bloom.strength = f.bloom ?? 0.3;
  const cam = typeof f.cam === 'function' ? f.cam() : f.cam;
  E.setCamera(cam.pos, cam.look, cam.fov ?? 34, cam.roll ?? 0);
  if (f.dof) E.setDof(f.dof.focus ?? cam.pos.distanceTo(cam.look), f.dof.range ?? 3, f.dof.blur ?? 0); else E.setDof(0, 1, 0);
  E.renderer.toneMappingExposure = f.exposure ?? 0.94;
  $('#fade').style.opacity = (f.fade ?? 0).toFixed(4);
  updateOverlays(t, f);
  if (!render) return;
  drawGrain(Math.round(t * FPS));
  E.render();
}

// 跳到 t：先从 t - pre 秒起不出图地模拟一遍，让机器里的平滑动画收敛到连续播放时的状态
function seek(t, pre = 6) {
  lastT = null;
  const t0 = Math.max(0, t - pre);
  for (let x = t0; x < t - 1e-6; x += 1 / FPS) renderAt(x, { render: false });
  renderAt(t);
}

// 封面：片名落版那一刻（去掉字幕）
function poster(t) {
  seek(t ?? SC.open.title + 2.4);
  for (const sel of ['.sub', '.term']) document.querySelectorAll(sel).forEach((e) => { e.style.display = 'none'; });
  E.render();
}

function startPreview() {
  const bar = $('#scrub');
  bar.hidden = false;
  const r = $('#scrubber');
  r.max = SC.end;
  let playing = true, t = Number(params.get('t') || 0), last = performance.now();
  seek(t);
  r.addEventListener('input', () => { t = Number(r.value); seek(t, 3); });
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

boot().catch((e) => { console.error(e); document.body.insertAdjacentHTML('beforeend', `<pre style="color:#f66;position:fixed;top:0;left:0;z-index:99">${esc(e.stack || e.message)}</pre>`); window.__film = { error: String(e.stack || e) }; });
