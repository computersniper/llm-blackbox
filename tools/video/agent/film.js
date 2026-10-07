// 智能体视频的电影模式：用智能体页的真实录制数据（public/agent/data/sales-top.json）按分镜（score.js）
// 驱动桌面（文件树 / 编辑器 / 终端）、循环图、上下文面板、逐词元面板和讲解层。全是 DOM + 2D canvas，没有 3D。
// window.__film.renderAt(t) 确定地渲染第 t 秒，tools/video/agent/render.mjs 逐帧截图出片。
//
// 片子讲的是：模型本身只会写字，“动手”是外面的程序（harness）解析工具调用后执行的；每一圈都把整段历史重新喂给模型，
// 上下文越来越长；它会犯错，看到报错再改（python → python3；缺 pandas → pip 装不上 → 改用自带的 csv）。
import { clamp, lerp, seg, smooth, easeOut, easeInOut, el, vis, rng, bez } from './util.js';
import { buildScore, CHAPTERS } from './score.js';
import { Opening } from './opening.js';
import { loadManifest, loadTask } from '/public/agent/js/data.js';
import { lineDiff, splitExit, normPath } from '/public/agent/js/screen.js';
import { esc, fmtPct } from '/public/js/ui.js';

const params = new URLSearchParams(location.search);
const FPS = Number(params.get('fps') || 30);
const PREVIEW = params.has('preview');
const TASK = params.get('task') || 'sales-top';
const $ = (s) => document.querySelector(s);
const num = (n) => Math.round(n).toLocaleString('en-US');

let MAN, R, SC, OPENING, QR_SVG = '', CAP = null;
const SITE = new Map();
const TITLE = '只会写字的 AI，怎么自己动手干活';
const SUBTITLE = '走进编程智能体的“黑箱”';
const URL_SHOW = 'caijiechao.com/blackbox/agent/';

/* ================================================================ 启动 */

async function boot() {
  await Promise.all([
    document.fonts.load('900 74px "Film Serif"', '只会写字的怎么自己动手干活'),
    document.fonts.load('600 40px "Film Serif"', '交给它一个任务'),
    document.fonts.load('400 30px "Film Sans"', '有什么要我做的'),
    document.fonts.load('600 30px "Film Sans"', 'Qwen'),
    document.fonts.load('400 22px "JetBrains Mono"', '0123'),
    document.fonts.load('700 22px "JetBrains Mono"', '0123'),
  ]);
  QR_SVG = await (await fetch('/tools/video/agent/qr-agent.svg')).text().catch(() => '');
  MAN = await loadManifest();
  R = await loadTask(TASK);
  CAP = await fetch('/ext/sitecap/meta.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
  SC = buildScore(R, CAP);
  if (CAP && SC.endSite.segs.length) await preloadSite();
  // 所有字（包括片尾、面板里的）先让浏览器排一次版，把要用的字形都加载好
  await document.fonts.ready;

  buildBg();
  DESK = new Desk($('#desk'), R, SC.acts);
  LOOP = new Loop($('#loop'));
  CTX = new CtxPanel($('#panels'));
  TOK = new TokPanel($('#panels'));
  buildOverlays();
  const c0 = SC.acts[0].c;
  OPENING = new Opening({ frame: $('#frame'), T: SC.open, prompt: R.prompt, model: MAN.model.name, title: TITLE, sub: SUBTITLE, firstCall: { name: c0.name, arg: normPath(c0.args.path) } });
  // 先把每个时刻都排一遍版（字体、布局都就位），之后逐帧渲染就不会有第一次出现的跳动
  for (const t of [0, 10, 20, 40, 60, 75, 90, 100, 105, 112, 117, 122, 128, 136, 145, 160, 180]) renderAt(t, { render: false });
  await document.fonts.ready;
  window.__film = { ready: true, fps: FPS, duration: SC.end, renderAt, seek, poster, events: () => SC.events.slice().sort((a, b) => a.t - b.t), score: { sections: SC.sections, bpm: SC.bpm, end: SC.end, subs: SC.subs, terms: SC.terms }, SC, R };
  if (PREVIEW) startPreview();
}

/* ================================================================ 背景：缓慢上浮的微粒（闭式，不积累状态）；开场推进时拉成光线 */

let BG = null;
function buildBg() {
  const r = rng(20261007);
  BG = {
    g: $('#bg').getContext('2d'),
    snow: Array.from({ length: 170 }, () => { const z = 0.2 + r() * 0.8; return { x0: r() * 1920, y0: r() * 1104, z, vy: -(4 + 14 * z) * 1.3, ph: r() * Math.PI * 2, w: 0.3 + 0.4 * z, rad: (0.5 + z * 1.4) * 1.3 }; }),
  };
}
function warpAt(t) {
  const w0 = SC.open.warpT, D = 1.6;
  if (t <= w0) return { k: 0, I: 0 };
  const kf = (u) => (u > 0 && u < 1 ? Math.sin(Math.PI * Math.pow(u, 0.6)) : 0);
  const u = (t - w0) / D;
  let I = 0;
  const n = 48, uu = Math.min(1, u);
  for (let i = 0; i < n; i++) I += kf(((i + 0.5) / n) * uu) * (uu / n) * D;
  return { k: kf(u), I };
}
function snowPos(p, t, W) {
  const span = 1104;
  let y = (((p.y0 + p.vy * t) % span) + span) % span - 12;
  let x = p.x0 + ((6 * p.z) / p.w) * (Math.cos(p.ph) - Math.cos(p.ph + p.w * t)) * 1.4;
  if (W.I > 0) { const f = Math.exp(4.5 * (0.4 + p.z) * W.I); x = 960 + (x - 960) * f; y = 500 + (y - 500) * f; }
  return [x, y];
}
function drawBg(t, a = 1) {
  const g = BG.g;
  const grad = g.createRadialGradient(960, 520, 0, 960, 520, 1400);
  grad.addColorStop(0, 'rgb(6,13,26)');
  grad.addColorStop(0.6, 'rgb(5,11,22)');
  grad.addColorStop(1, 'rgb(3,7,15)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 1920, 1080);
  const W = warpAt(t), Wp = warpAt(t - 1 / 60);
  for (const p of BG.snow) {
    const [x, y] = snowPos(p, t, W);
    const al = (0.1 + p.z * 0.38) * a;
    if (W.k > 0.05) {
      const [px, py] = snowPos(p, t - 1 / 60, Wp);
      g.strokeStyle = `rgba(120,230,220,${(al * (0.6 + W.k)).toFixed(3)})`;
      g.lineWidth = p.rad;
      g.beginPath(); g.moveTo(px, py); g.lineTo(x + (x - px) * 3 * W.k, y + (y - py) * 3 * W.k); g.stroke();
    } else {
      g.fillStyle = `rgba(120,230,220,${al.toFixed(3)})`;
      g.beginPath(); g.arc(x, y, p.rad, 0, Math.PI * 2); g.fill();
    }
  }
}

/* ================================================================ 桌面：文件树 + 编辑器 + 终端（按真实工具调用回放） */

const PS1 = '<span class="pr">agent@sandbox<i>:/work</i>$</span> ';
const KW = /\b(def|return|if|elif|else|for|while|in|import|from|class|with|as|not|and|or|None|True|False|try|except|lambda|print)\b/;
// 极简语法高亮（Python）：和网站 screen.js 的一样
function hl(line, ext) {
  if (ext !== 'py') return esc(line);
  const re = /(#.*$)|("""[^]*?"""|f?"(?:[^"\\]|\\.)*"|f?'(?:[^'\\]|\\.)*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_]\w*)(?=\()|([A-Za-z_]\w*)/g;
  let out = '', last = 0, m;
  while ((m = re.exec(line))) {
    out += esc(line.slice(last, m.index));
    const s = esc(m[0]);
    if (m[1]) out += `<span class="hl-cm">${s}</span>`;
    else if (m[2]) out += `<span class="hl-str">${s}</span>`;
    else if (m[3]) out += `<span class="hl-num">${s}</span>`;
    else if (m[4]) out += KW.test(m[4]) && m[4] !== 'print' ? `<span class="hl-kw">${s}</span>` : `<span class="hl-fn">${s}</span>`;
    else out += KW.test(m[5]) ? `<span class="hl-kw">${s}</span>` : s;
    last = re.lastIndex;
  }
  return out + esc(line.slice(last));
}
const extOf = (p) => (p.match(/\.(\w+)$/) || [])[1] || '';
const outCls = (line) => (/^(OK|\.+)$|^OK |passed|最高的城市/.test(line) ? 'ok' : /Error|FAIL|Traceback|not found|^error|×/.test(line) ? 'bad' : '');

let DESK, LOOP, CTX, TOK;
class Desk {
  constructor(root, R, acts) {
    this.root = root; this.R = R; this.acts = acts;
    root.innerHTML = `<div class="dwin"><div class="dbar"><i></i><i></i><i></i><span class="tt">agent@sandbox:/work</span><span class="bd">沙箱 · 无网络</span></div>
      <div class="dbody"><nav class="tree"></nav><div class="ed"><div class="tabs"></div><div class="code"><div class="in"></div></div></div>
      <div class="tm"><div class="th">终端<span class="st"></span></div><div class="tb"><div class="in"></div></div></div></div>
      <div class="dstat"></div></div><div class="ddone"></div><div class="dim"></div>`;
    const q = (s) => root.querySelector(s);
    Object.assign(this, { tree: q('.tree'), tabs: q('.tabs'), code: q('.code'), codeIn: q('.code .in'), ed: q('.ed'), tm: q('.tm'), tb: q('.tb'), tbIn: q('.tb .in'), st: q('.tm .th .st'), stat: q('.dstat'), done: q('.ddone'), dim: q('.dim') });
    this.dbody = q('.dbody');
    this.dbody.style.gridTemplateRows = 'minmax(0, 1fr) 272px';
    this.LH = 35; this.TLH = 31;
    this.key = '';
    // 第 5 个调用（pip 的长报错）：前 3 行先出来停一会儿，再把剩下的刷出来，免得第一行“error: …”一闪就滚走
    this.reveal = { 4: (u) => (u < 0.14 ? (u / 0.14) * (3 / 20) : u < 0.58 ? 3 / 20 : 3 / 20 + ((u - 0.58) / 0.42) * (17 / 20)) };
  }

  edPath(c) {
    if (c.name === 'read_file' || c.name === 'write_file' || c.name === 'edit_file') return normPath(c.args?.path);
    if (c.name === 'bash' && c.diffs.length) return c.diffs[0].path;
    return null;
  }

  update(D) {
    const root = this.root;
    root.style.display = D.a > 0.002 ? 'block' : 'none';
    if (D.a <= 0.002) return;
    const c = D.cam;
    root.style.transform = `translate(${(c.cx - c.s * c.x).toFixed(2)}px, ${(c.cy - c.s * c.y).toFixed(2)}px) scale(${c.s.toFixed(5)})`;
    root.style.opacity = D.a.toFixed(3);
    root.style.filter = D.blur > 0.05 ? `blur(${D.blur.toFixed(2)}px)` : '';
    this.dim.style.opacity = (D.dim || 0).toFixed(3);
    const { k, cur, p } = D;
    const acts = this.acts;
    const curAct = cur >= 0 ? acts[cur] : null;
    // ---- 文件树、标签页
    const files = curAct ? curAct.c.filesBefore : k > 0 ? acts[k - 1].c.filesAfter : this.R.files0;
    const filesShown = curAct && curAct.c.diffs.length && p >= 0.5 ? curAct.c.filesAfter : files;
    let lastPath = null;
    const opened = [];
    for (let i = 0; i < k; i++) { const pth = this.edPath(acts[i].c); if (pth) { lastPath = pth; if (!opened.includes(pth)) opened.push(pth); } }
    const curPath = curAct ? this.edPath(curAct.c) : null;
    if (curPath && !opened.includes(curPath)) opened.push(curPath);
    const active = curPath || lastPath;
    const f0 = this.R.files0;
    const treeKey = `${Object.keys(filesShown).join(',')}|${active}|${Object.keys(filesShown).map((x) => (x in f0 ? (f0[x] === filesShown[x] ? 0 : 1) : 2)).join('')}`;
    if (treeKey !== this.treeKey) {
      this.treeKey = treeKey;
      this.tree.innerHTML = '<div class="th">/work</div>' + Object.keys(filesShown).sort().map((pth) => {
        const b = !(pth in f0) ? '<span class="b n">新</span>' : f0[pth] !== filesShown[pth] ? '<span class="b m">M</span>' : '';
        return `<div class="f ${pth === active ? 'on' : ''}">${esc(pth)}${b}</div>`;
      }).join('');
      this.tabs.innerHTML = opened.map((pth) => `<span class="tab ${pth === active ? 'on' : ''}">${!(pth in f0) || f0[pth] !== filesShown[pth] ? '<i class="d"></i>' : ''}${esc(pth.split('/').pop())}</span>`).join('');
    }
    // ---- 编辑器
    this.renderEditor(curAct, curPath, p, lastPath, k);
    // ---- 终端
    this.renderTerm(k, cur, p, D.hl);
    // ---- 状态栏
    let stat;
    if (curAct && p < 0.9) {
      const arg = curAct.c.name === 'bash' ? curAct.c.args.command : curAct.c.args.path;
      stat = `<span class="op">▶ ${esc(curAct.c.name)}</span><span>${esc(String(arg).split('\n')[0])}</span><span>· 在沙箱里执行</span>`;
    } else {
      const last = curAct ? curAct.c : k > 0 ? acts[k - 1].c : null;
      stat = last ? this.statusDone(last) : '<span>沙箱已就绪 · 只读系统目录 · 只能写 /work · 无网络</span>';
    }
    if (stat !== this.statKey) { this.statKey = stat; this.stat.innerHTML = stat; }
    // ---- 任务完成
    this.done.style.display = D.final ? 'block' : 'none';
    if (D.final && !this.done.innerHTML) this.done.innerHTML = `<b>✓ 任务完成</b>沙箱外的独立检查：${esc(this.R.check.detail)}`;
    // ---- 聚焦：讲到终端，编辑器和文件树压暗；讲到编辑器，终端和文件树压暗
    const fk = D.fk || 0;
    this.tree.style.opacity = (1 - 0.62 * fk * (D.focus ? 1 : 0)).toFixed(3);
    this.ed.style.opacity = (D.focus === 'term' ? 1 - 0.62 * fk : 1).toFixed(3);
    this.tm.style.opacity = (D.focus === 'ed' ? 1 - 0.62 * fk : 1).toFixed(3);
    this.stat.style.opacity = (1 - 0.7 * fk * (D.focus ? 1 : 0)).toFixed(3);
  }

  statusDone(c) {
    const code = c.name === 'bash' ? splitExit(c.result).code : null;
    const ok = c.info?.ok !== false && !(c.name === 'bash' && code !== 0);
    const ms = c.info?.ms != null ? ` · 实测 ${c.info.ms} ms` : '';
    const what = c.name === 'bash' ? `退出码 ${code}` : c.name === 'read_file' ? `读取了 ${c.result.split('\n').length} 行` : c.result.split('\n')[0];
    return `<span class="${ok ? 'ok' : 'bad'}">${ok ? '✓' : '✗'} ${esc(c.name)}</span><span>${esc(what)}${ms}</span>`;
  }

  renderEditor(curAct, curPath, p, lastPath, k) {
    const LH = this.LH, viewH = this.code.clientHeight - 20;
    let html = '', focusLine = 0;
    if (curAct && curPath) {
      const c = curAct.c, ext = extOf(curPath);
      const d = c.diffs.find((x) => x.path === curPath);
      if (c.name === 'read_file' || !d) {
        const text = c.filesBefore[curPath] ?? '';
        const lines = text.replace(/\n$/, '').split('\n');
        const scanF = Math.min(lines.length - 0.001, p * lines.length * 1.1);
        const scan = p < 1 ? Math.floor(scanF) : -1;
        html = lines.map((l, i) => `<div class="ln ${i === scan ? 'scan' : ''}"><span class="no">${i + 1}</span><span class="tx">${hl(l, ext)}</span></div>`).join('');
        focusLine = p < 1 ? scanF : 0;
      } else {
        // 改动：删掉的行标红，新写的行一个字一个字地打出来
        const ops = lineDiff(d.before, d.after);
        const adds = ops.filter((o) => o.op === 'add');
        const total = adds.reduce((a, o) => a + o.text.length + 1, 0) || 1;
        const tp = clamp((p - 0.12) / 0.76);
        let budget = total * tp, row = 0;
        for (const o of ops) {
          if (o.op === 'eq') html += `<div class="ln"><span class="no">${o.nb}</span><span class="tx">${hl(o.text, ext)}</span></div>`;
          else if (o.op === 'del') html += `<div class="ln del"><span class="no">−</span><span class="tx">${esc(o.text)}</span></div>`;
          else {
            if (budget <= 0) { row++; continue; }
            const len = o.text.length + 1;
            const take = Math.min(o.text.length, Math.floor(budget));
            const partial = budget < len && tp < 1;
            if (partial) focusLine = row + budget / len;
            html += `<div class="ln add"><span class="no">${o.nb}</span><span class="tx">${partial ? esc(o.text.slice(0, take)) + '<span class="cur"></span>' : hl(o.text, ext)}</span></div>`;
            budget -= len;
          }
          row++;
        }
        if (tp >= 1) focusLine = row;
      }
    } else if (lastPath) {
      const files = k > 0 ? this.acts[k - 1].c.filesAfter : this.R.files0;
      const text = files[lastPath] ?? '';
      html = text.replace(/\n$/, '').split('\n').map((l, i) => `<div class="ln"><span class="no">${i + 1}</span><span class="tx">${hl(l, extOf(lastPath))}</span></div>`).join('');
    } else {
      html = '<div class="empty"><span>编辑器</span><span>agent 读写文件时会在这里打开</span></div>';
    }
    if (html !== this.edKey) { this.edKey = html; this.codeIn.innerHTML = html; }
    const off = Math.max(0, (focusLine + 0.5) * LH - viewH * 0.62);
    const maxOff = Math.max(0, this.codeIn.scrollHeight - viewH);
    this.codeIn.style.transform = `translateY(${(-Math.min(off, maxOff)).toFixed(1)}px)`;
  }

  // 终端：之前所有 bash 调用的完整记录 + 正在执行的这一条
  renderTerm(k, cur, p, H) {
    const acts = this.acts;
    const rows = []; // { html, cls }
    let lastCode = null;
    const add = (i, c, pp) => {
      const cmd = String(c.args?.command ?? '');
      const { body, code } = splitExit(c.result);
      const tp = Math.min(1, pp / 0.3);
      const typing = pp < 0.33;
      rows.push({ html: PS1 + `<span class="cmd">${esc(cmd.slice(0, Math.ceil(cmd.length * tp)))}</span>${typing ? '<span class="cur"></span>' : ''}`, f: 1 });
      if (typing) return;
      const lines = body ? body.split('\n') : [];
      const u = clamp((pp - 0.33) / 0.55);
      const fr = this.reveal[i] ? this.reveal[i](u) : u;
      const shown = lines.length * fr;
      lines.forEach((l, j) => {
        if (j >= Math.ceil(shown - 1e-6)) return;
        const isHl = H && H.act === i && (H.line === j || (H.line === -1 && j === lines.length - 1));
        const cls = outCls(l);
        rows.push({ html: esc(l) || ' ', cls: `${cls}${isHl ? ` hl ${H.cls}` : ''}`, k: isHl ? H.k : 0, f: clamp(shown - j) });
      });
      if (pp >= 0.9) { rows.push({ html: `<span class="ex ${code === 0 ? 'ok' : 'bad'}">退出码 ${code}</span>`, f: smooth(seg(pp, 0.9, 0.95)) }); lastCode = code; }
      if (pp >= 1) rows.push({ html: ' ', f: 1 });
    };
    for (let i = 0; i < k; i++) if (acts[i].c.name === 'bash') add(i, acts[i].c, 1);
    const curBash = cur >= 0 && acts[cur].c.name === 'bash';
    if (curBash) add(cur, acts[cur].c, p);
    else rows.push({ html: PS1 + '<span class="cur"></span>', f: 1 });
    const html = rows.map((r) => `<div class="tl ${r.cls || ''}"${r.k ? ` style="--k:${(r.k * 0.18).toFixed(3)}"` : ''}>${r.html}</div>`).join('');
    if (html !== this.tKey) { this.tKey = html; this.tbIn.innerHTML = html; }
    // 滚到底：按“出现了几行”（可以是小数）连续地滚，不会一跳一跳
    const total = rows.reduce((a, r) => a + r.f, 0);
    const viewH = this.tb.clientHeight - 16;
    const off = Math.max(0, total * this.TLH - viewH);
    this.tbIn.style.transform = `translateY(${(-off).toFixed(1)}px)`;
    // 终端标题栏：上一条命令的退出码
    let code = lastCode;
    if (code == null) for (let i = k - 1; i >= 0; i--) if (acts[i].c.name === 'bash') { code = splitExit(acts[i].c.result).code; break; }
    const st = code == null ? '' : `上一条命令 · 退出码 ${code}`;
    if (st !== this.stKey) { this.stKey = st; this.st.textContent = st; this.st.className = `st${code == null ? '' : code === 0 ? ' ok' : ' bad'}`; }
  }
}

/* ================================================================ 第 2 章：循环（模型 ⇄ harness ⇄ 沙箱） */

const PATHS = {
  m2h: { a: { x: 330, y: 600 }, c: { x: 640, y: 150 }, b: { x: 946, y: 470 } },
  h2s: { a: { x: 946, y: 560 }, c: { x: 1262, y: 150 }, b: { x: 1578, y: 470 } },
  s2h: { a: { x: 1578, y: 640 }, c: { x: 1262, y: 930 }, b: { x: 946, y: 660 } },
  h2m: { a: { x: 946, y: 690 }, c: { x: 640, y: 930 }, b: { x: 330, y: 640 } },
};
const FLOW = ['m2h', 'h2s', 's2h', 'h2m'];

class Loop {
  constructor(root) {
    this.root = root;
    const M = MAN.model;
    const T2 = R.turns[2];
    this.outText = T2.gen.text;
    root.innerHTML = `
      <svg class="edges" viewBox="0 0 1920 1080">
        <path d="M576 404 L700 404" /><path d="M690 396 L702 404 L690 412" />
        <path d="M700 616 L576 616" /><path d="M588 608 L576 616 L588 624" />
        <path d="M1192 404 L1290 404" /><path d="M1280 396 L1292 404 L1280 412" />
        <path d="M1290 616 L1192 616" /><path d="M1202 608 L1190 616 L1202 624" />
        <text x="638" y="388" text-anchor="middle">文字</text><text x="638" y="648" text-anchor="middle">上下文</text>
        <text x="1241" y="388" text-anchor="middle">执行</text><text x="1241" y="648" text-anchor="middle">结果</text>
        <g class="flow"></g>
      </svg>
      <div class="node model"><div class="k">模型</div><div class="t">Qwen3-4B</div><div class="s">${(M.params / 1e8).toFixed(1)} 亿参数 · ${M.layers} 层</div>
        <div class="d">只会做一件事：<br><b>读一串字，写一串字</b></div><div class="out"></div></div>
      <div class="node harness"><div class="k">HARNESS · 外面的程序</div><div class="t">循环 · 解析 · 执行</div><span class="lap"></span>
        <ol>${['拼出整段上下文', '等模型写完', '找出 &lt;tool_call&gt;', '在沙箱里执行', '结果接回对话'].map((s, i) => `<li><i>${i + 1}</i>${s}</li>`).join('')}</ol></div>
      <div class="node sbx"><div class="k">沙箱 · /WORK</div><div class="s">只读系统目录 · 只能写 /work · 无网络</div><div class="note">命令在这里真的执行：bubblewrap 沙箱</div></div>`;
    const q = (s) => root.querySelector(s);
    Object.assign(this, { svg: q('.edges'), flow: q('.flow'), model: q('.model'), harness: q('.harness'), sbx: q('.sbx'), out: q('.out'), lap: q('.lap'), ops: [...root.querySelectorAll('.harness li')] });
    this.pks = [];
  }

  update(S) {
    const L = S;
    this.root.style.display = L && L.a > 0.002 ? 'block' : 'none';
    if (!L || L.a <= 0.002) return;
    this.root.style.opacity = L.a.toFixed(3);
    const sl = L.slide;
    this.model.style.transform = `translateX(${((1 - sl) * -160).toFixed(1)}px)`;
    this.harness.style.transform = `translateY(${((1 - sl) * 30).toFixed(1)}px)`;
    const dimOf = (name) => (L.focus && L.focus !== name ? 1 - 0.6 * L.fk : 1);
    this.model.style.opacity = dimOf('model').toFixed(3);
    this.harness.style.opacity = dimOf('harness').toFixed(3);
    this.sbx.style.opacity = (sl * dimOf('sbx')).toFixed(3);
    this.svg.style.opacity = L.edgeA.toFixed(3);
    // 模型写出的文字（第 3 圈那条 python 命令）
    const n = Math.floor(this.outText.length * L.out.n);
    const shown = this.outText.slice(0, n);
    const oh = esc(shown).replace(/&lt;\/?tool_call&gt;/g, (s) => `<span class="tg">${s}</span>`);
    if (oh !== this.outKey) { this.outKey = oh; this.out.innerHTML = oh || '<span style="color:var(--faint)">（等待输入）</span>'; }
    this.out.classList.toggle('hi', L.out.hl > 0.5);
    this.out.style.opacity = (0.35 + 0.65 * L.out.a).toFixed(3);
    this.ops.forEach((li, i) => li.classList.toggle('on', i === L.op));
    this.lap.textContent = `第 ${L.lap} 圈`;
    this.sbx.style.boxShadow = L.sbxFlash > 0.01 ? `0 0 ${(60 * L.sbxFlash).toFixed(1)}px rgba(111, 227, 161, ${(0.6 * L.sbxFlash).toFixed(3)})` : '';
    this.model.style.boxShadow = L.modelIn > 0.01 ? `0 0 ${(60 * L.modelIn).toFixed(1)}px rgba(255, 182, 92, ${(0.55 * L.modelIn).toFixed(3)})` : '';
    // 数据包
    while (this.pks.length < L.pk.length) this.pks.push(el('div', 'pk', '', this.root));
    this.pks.forEach((e, i) => {
      const p = L.pk[i];
      e.style.display = p ? 'block' : 'none';
      if (!p) return;
      const P = PATHS[p.e];
      const u = easeInOut(clamp(p.u));
      const pt = bez(P.a, P.c, P.b, u);
      e.className = `pk ${p.cls}`;
      if (e.innerHTML !== p.html) e.innerHTML = p.html;
      e.style.left = `${pt.x.toFixed(1)}px`;
      e.style.top = `${pt.y.toFixed(1)}px`;
      e.style.opacity = (smooth(p.u / 0.12) * (1 - smooth((p.u - 0.95) / 0.15))).toFixed(3);
    });
    // 一圈一圈：光点在四条路径上接力
    if (L.flow >= 0) {
      const ph = L.flow % 1, which = Math.floor(ph * 4), u = (ph * 4) % 1;
      const P = PATHS[FLOW[which]];
      let h = '';
      const col = { m2h: '#ffb65c', h2s: '#5ef0d4', s2h: '#6fe3a1', h2m: '#b39dff' }[FLOW[which]];
      for (let k = 0; k < 9; k++) { const uu = u - k * 0.03; if (uu < 0) continue; const pt = bez(P.a, P.c, P.b, easeInOut(uu)); h += `<circle cx="${pt.x.toFixed(1)}" cy="${pt.y.toFixed(1)}" r="${(14 - k * 1.4).toFixed(1)}" fill="${col}" opacity="${(1 - k * 0.1).toFixed(2)}" style="filter:drop-shadow(0 0 10px ${col})" />`; }
      this.flow.innerHTML = h;
    } else if (this.flow.innerHTML) this.flow.innerHTML = '';
  }
}

/* ================================================================ 第 3 章：上下文面板 */

const SEG_NAME = { sys: '系统提示', tools: '工具说明', user: '任务', asst: '模型写的', tool: '工具结果', gen: '轮到模型' };
class CtxPanel {
  constructor(parent) {
    const root = (this.root = el('div', 'ctxp', '', parent));
    this.maxN = Math.max(...R.turns.map((t) => t.ctx.n));
    this.W = 1600; this.RW = 1300;
    root.innerHTML = `<div class="hd"><span class="t"></span><span class="n"></span><span class="u">词元</span></div>
      <div class="cbar"></div><div class="legend"></div>
      <div class="stairs"><div class="h">每一圈喂给模型的长度</div>${R.turns.map((T, i) => `<div class="srow"><span class="l">第 ${i + 1} 圈</span><div class="bar">${this.rowSegs(T)}</div><span class="n">${num(T.ctx.n)}</span></div>`).join('')}</div>
      <div class="raw"></div>`;
    const q = (s) => root.querySelector(s);
    Object.assign(this, { t: q('.hd .t'), n: q('.hd .n'), bar: q('.cbar'), legend: q('.legend'), rows: [...root.querySelectorAll('.srow')], raw: q('.raw'), stairs: q('.stairs') });
    // 第 4 圈上下文的最后几行（聊天模板渲染后的原文）
    const P = R.turns[3].prompt;
    const i0 = P.lastIndexOf('<|im_start|>user');
    const tail = P.slice(i0).replace(/\n$/, '');
    this.raw.innerHTML = `<div class="cap">第 4 圈 · 上下文的最后几行（聊天模板原文）</div>` + tail.split('\n').map((l) => {
      let h = esc(l).replace(/&lt;\|?\/?[a-z_]+\|?&gt;/g, (s) => `<span class="sp">${s}</span>`);
      if (/command not found/.test(l)) h = `<span class="er">${h}</span>`;
      return `<div>${h}</div>`;
    }).join('');
    this.raw.style.left = '780px';
    this.raw.style.top = '440px';
    this.rawEr = this.raw.querySelector('.er');
  }

  rowSegs(T) {
    const sc = this.RW / this.maxN;
    let x = 0, h = '';
    for (const g of T.ctx.segs) { const w = g.n * sc; h += `<i style="left:${x.toFixed(1)}px;width:${Math.max(1, w).toFixed(1)}px;background:var(--k-${g.k})"></i>`; x += w; }
    return h;
  }

  update(C) {
    this.root.style.display = C && C.a > 0.002 ? 'block' : 'none';
    if (!C || C.a <= 0.002) return;
    this.root.style.opacity = C.a.toFixed(3);
    this.root.style.transform = `translateY(${((1 - easeOut(C.a)) * 16).toFixed(1)}px)`;
    const T = R.turns[C.lap];
    const sc = this.W / this.maxN;
    // 当前这一圈的整段上下文：第 1 圈按片段一段段搭出来
    const build = C.lap === 0 ? C.build : null;
    const hiKinds = C.hiSeg === 'head' ? ['sys', 'tools'] : C.hiSeg === 'user' ? ['user'] : C.hiSeg === 'tail' ? ['asst', 'tool'] : null;
    let x = 0, h = '', count = 0;
    for (const g of T.ctx.segs) {
      const f = build ? (g.k === 'sys' ? build.sys : g.k === 'tools' ? build.tools : build.user) : 1;
      const w = g.n * sc * f;
      count += g.n * f;
      if (w > 0.3) h += `<div class="sg ${g.k}${hiKinds && !hiKinds.includes(g.k) ? ' off' : ''}" style="left:${x.toFixed(1)}px;width:${w.toFixed(1)}px">${w > 64 ? g.n : ''}</div>`;
      x += w;
    }
    // KV 缓存：第 4 圈前面 1166 个词元复用、只新算 33 个
    if (C.kv > 0.01 && C.lap === 3) {
      const kv = T.kv;
      const wc = kv.reused * sc, wn = kv.computed * sc;
      h += `<div class="kvc" style="width:${wc.toFixed(1)}px;opacity:${C.kv.toFixed(3)}"></div>`;
      h += `<div class="kvn" style="left:${(wc - 4).toFixed(1)}px;width:${(wn + 8).toFixed(1)}px;opacity:${C.kv.toFixed(3)}"></div>`;
      h += `<div class="kvl" style="left:0;opacity:${C.kv.toFixed(3)};color:var(--ink2)">读过的 <b>${num(kv.reused)}</b> 个：有 KV 缓存，不用重算</div>`;
      h += `<div class="kvl" style="left:${(wc + wn + 18).toFixed(1)}px;top:18px;opacity:${C.kv.toFixed(3)};color:var(--amber)">新算 <b>${num(kv.computed)}</b> 个</div>`;
    }
    if (h !== this.barKey) { this.barKey = h; this.bar.innerHTML = h; }
    const tt = `第 ${C.lap + 1} 圈 · 喂给模型的整段上下文`;
    if (this.t.textContent !== tt) this.t.textContent = tt;
    const nn = build && count < 1 ? "" : num(build ? count : T.ctx.n);
    if (this.n.textContent !== nn) this.n.textContent = nn;
    // 图例：这一圈每类片段的真实词元数
    const sum = {};
    for (const g of T.ctx.segs) sum[g.k] = (sum[g.k] || 0) + g.n;
    const lg = ['sys', 'tools', 'user', 'asst', 'tool'].map((k) => `<span class="${(hiKinds && !hiKinds.includes(k)) || !sum[k] ? 'off' : ''}" style="--c:var(--k-${k})">${SEG_NAME[k]} <b>${sum[k] ? num(sum[k]) : 0}</b></span>`).join('');
    if (lg !== this.lgKey) { this.lgKey = lg; this.legend.innerHTML = lg; }
    this.legend.style.opacity = (1 - C.kv).toFixed(3);
    // 阶梯：一圈一行
    this.rows.forEach((r, i) => {
      const a = C.rows[i];
      r.style.opacity = (a * (C.raw > 0.01 ? 1 - 0.6 * C.raw : 1)).toFixed(3);
      r.style.transform = `translateX(${((1 - easeOut(a)) * -24).toFixed(1)}px)`;
      r.classList.toggle('on', (C.kv > 0.01 && i === 3) || (C.kv <= 0.01 && i === C.lap && C.rows[i] > 0.5 && C.hiSeg === null));
    });
    this.stairs.style.opacity = C.rows[0] > 0.001 ? '1' : '0';
    // 原文：第 4 圈上下文的最后几行
    this.raw.style.display = C.raw > 0.002 ? 'block' : 'none';
    this.raw.style.opacity = C.raw.toFixed(3);
    this.raw.style.transform = `translateY(${((1 - easeOut(C.raw)) * 20).toFixed(1)}px)`;
    if (this.rawEr) this.rawEr.classList.toggle('on', C.rawHi > 0.5);
  }
}

/* ================================================================ 第 4 章：逐词元面板 */

const tokShow = (x) => (x.end ? '⟨im_end⟩' : x.s === '\n' ? '↵' : x.s.replace(/\n/g, '↵').replace(/^ /, '␣'));
const candShow = (s) => (s === '<|im_end|>' ? '⟨im_end⟩' : s.replace(/\n/g, '↵').replace(/^ /, '␣'));
class TokPanel {
  constructor(parent) {
    const root = (this.root = el('div', 'tokp', '', parent));
    this.toks = SC.toks;
    const P = R.turns[3].prompt;
    const tail = P.slice(P.lastIndexOf('<tool_response>')).replace(/\n$/, '');
    root.innerHTML = `<div class="raw ctxt" style="position:relative;font-size:24px;line-height:35px;padding:14px 22px"><div class="cap">它刚读到的：第 4 圈上下文的最后几行</div>${tail.split('\n').map((l) => {
      let h = esc(l).replace(/&lt;\|?\/?[a-z_]+\|?&gt;/g, (s) => `<span class="sp">${s}</span>`);
      if (/command not found/.test(l)) h = `<span class="er">${h}</span>`;
      return `<div>${h}</div>`;
    }).join('')}</div>
      <div class="gen"><div class="cap">第 4 圈 · 模型一个词元一个词元写出来的</div><div class="chips">${this.toks.map((x, j) => `<span class="chip${x.end || x.s === '<tool_call>' || x.s === '</tool_call>' ? ' sp' : ''}${SC.dbg.DEC.some((d) => d.j === j) ? ' dec' : ''}" data-j="${j}">${esc(tokShow(x))}</span>`).join('')}</div></div>
      <div class="cands"><div class="cand"><div class="cap"></div><div class="rows"></div></div><div class="cmp"></div></div>
      <div class="parse"></div>`;
    const q = (s) => root.querySelector(s);
    Object.assign(this, { chips: [...root.querySelectorAll('.chip')], cands: q('.cands'), cand: q('.cand'), candCap: q('.cand .cap'), candRows: q('.cand .rows'), cmp: q('.cmp'), parse: q('.parse'), er: q('.ctxt .er'), gen: q('.gen') });
    // 对比卡：上一圈同一个位置 vs 这一圈
    const T2 = R.turns[2];
    const pj = T2.gen.toks.findIndex((x) => x.s === 'python') + 1;
    const prev = T2.gen.toks[pj];
    const now = this.toks[SC.dbg.j3];
    const bar = (p) => `<span class="bar" style="width:${Math.max(4, p * 300).toFixed(0)}px"></span>`;
    const p3 = prev.top.find((x) => x[0] === '3')?.[1] ?? 0;
    this.cmp.innerHTML = `<div class="r"><div class="k">上一圈（第 3 圈），同一个位置：</div>
        <div class="v"><span style="width:150px">${esc(candShow(prev.top[0][0]))}</span>${bar(prev.top[0][1])}<span class="p">${fmtPct(prev.top[0][1])}</span></div>
        <div class="v"><span style="width:150px">3</span>${bar(p3)}<span class="p">${fmtPct(p3)}</span></div></div>
      <div class="r now"><div class="k">这一圈（第 4 圈），读过报错以后：</div>
        <div class="v"><span style="width:150px">3</span>${bar(now.p)}<span class="p">${fmtPct(now.p)}</span></div></div>`;
    const call = R.turns[3].calls[0];
    this.parse.innerHTML = `<div class="box"><div><span class="k">工具：</span><span class="v">${esc(call.name)}</span></div><div><span class="k">命令：</span><span class="v">${esc(call.args.command)}</span></div></div>
      <span class="arr">→</span><div class="run"><span class="pr">agent@sandbox:/work$</span> ${esc(call.args.command)}</div>`;
  }

  update(K) {
    this.root.style.display = K && K.a > 0.002 ? 'block' : 'none';
    if (!K || K.a <= 0.002) return;
    this.root.style.opacity = K.a.toFixed(3);
    this.chips.forEach((c, j) => {
      const on = j < K.n;
      c.style.visibility = on ? 'visible' : 'hidden';
      if (!on) return;
      const isNew = j === K.cur && K.parse < 0.01;
      c.classList.toggle('new', isNew);
      c.classList.toggle('three', j === SC.dbg.j3 && !isNew);
      const pop = isNew ? 1 + 0.18 * (1 - easeOut(K.kNew)) : 1;
      c.style.transform = pop !== 1 ? `scale(${pop.toFixed(3)})` : '';
      c.style.boxShadow = j === SC.dbg.j3 && K.glow3 > 0.02 ? `0 0 ${(40 * K.glow3).toFixed(1)}px rgba(255, 182, 92, ${(0.85 * K.glow3).toFixed(3)})` : '';
      c.style.opacity = K.parse > 0.01 && (c.classList.contains('sp')) ? (1 - 0.6 * K.parse).toFixed(3) : '';
    });
    if (this.er) this.er.classList.toggle('on', K.ctxHi > 0.5 || K.n === 0);
    // 候选面板：关键处停下来看前 5 名的真实概率
    const D = K.dec;
    this.cands.style.display = K.parse > 0.99 ? 'none' : 'flex';
    this.cand.style.opacity = D ? D.a.toFixed(3) : '0';
    if (D) {
      const x = this.toks[D.j];
      const key = `${D.j}|${D.k.toFixed(2)}`;
      if (key !== this.cKey) {
        this.cKey = key;
        this.candCap.textContent = `第 ${D.j + 1} 个词元 · 前 5 名候选（真实概率）`;
        const g = easeOut(D.k);
        this.candRows.innerHTML = x.top.slice(0, 5).map(([s, p]) => `<div class="crow${s === x.s ? ' win' : ''}"><span>${esc(candShow(s))}</span><span class="bar" style="width:${Math.max(0.6, p * 100 * g).toFixed(1)}%"></span><span class="p">${fmtPct(p)}</span></div>`).join('');
      }
    }
    this.cmp.style.opacity = K.cmp.toFixed(3);
    this.cmp.style.transform = `translateX(${((1 - easeOut(K.cmp)) * 24).toFixed(1)}px)`;
    // 解析：harness 认出 <tool_call> 里的 JSON，拿出工具名和命令
    this.parse.style.display = K.parse > 0.002 ? 'flex' : 'none';
    this.parse.style.opacity = K.parse.toFixed(3);
    const run = this.parse.querySelector('.run');
    run.style.opacity = smooth(seg(K.parseK, 0.45, 0.8)).toFixed(3);
    this.gen.style.boxShadow = K.parse > 0.01 ? `0 0 0 ${(2 * K.parse).toFixed(2)}px rgba(94, 240, 212, ${(0.6 * K.parse).toFixed(3)})` : '';
  }
}

/* ================================================================ 讲解层 */

class CueLayer {
  constructor(parent, cls, cues, { fin = 0.35, fout = 0.3, rise = 8 } = {}) {
    this.parent = parent; this.cls = cls; this.cues = cues.slice().sort((a, b) => a.t0 - b.t0); this.fin = fin; this.fout = fout; this.rise = rise; this.live = new Map();
  }
  update(t) {
    this.cues.forEach((c, i) => {
      const a = vis(t, c.t0, c.t1, this.fin, this.fout);
      let e = this.live.get(i);
      if (a <= 0.001) { if (e) { e.remove(); this.live.delete(i); } return; }
      if (!e) { e = el('div', this.cls, c.html, this.parent); this.live.set(i, e); }
      const dy = (1 - easeOut(clamp((t - c.t0) / (this.fin * 1.6)))) * this.rise;
      e.style.opacity = a.toFixed(4);
      e.style.transform = `translateX(-50%) translateY(${dy.toFixed(2)}px)`;
    });
  }
}

const OV = {};
function buildOverlays() {
  const ov = $('#ov');
  OV.band = el('div', 'band', '', ov);
  OV.say = el('div', 'say', '<div class="who"><i></i><span></span></div><div class="tx"></div>', ov);
  OV.subs = new CueLayer(ov, 'sub', SC.subs, { rise: 8 });
  OV.tags = new CueLayer(ov, 'tag', SC.terms, { rise: 6, fin: 0.3, fout: 0.3 });
  OV.prog = el('div', 'prog', CHAPTERS.map((c, i) => `${i ? '<span class="ln"></span>' : ''}<span class="it"><i>${i + 1}</i>${c}</span>`).join(''), ov);
  OV.progIt = [...OV.prog.querySelectorAll('.it')];
  OV.lapc = el('div', 'lapc', '', ov);
  const M = MAN.model;
  OV.title = el('div', 'title', `<div class="eb">INSIDE A CODING AGENT</div><h1><span>只会写字的 AI</span><span>怎么自己动手干活</span></h1><div class="rule"></div><div class="st">${SUBTITLE}</div><div class="spec">${esc(M.name.toUpperCase())} · 真实沙箱 · 离线录制</div>`, ov);
  OV.sum = el('div', 'sum', `<div class="eb">MODEL + TOOLS + LOOP</div><div class="big"><span class="w1">模型</span><span class="p">+</span><span class="w2">工具</span><span class="p">+</span><span class="w3">循环</span></div>
    <div class="stat">${esc(SC.statLine)}</div><div class="src">${esc(M.name)} · BUBBLEWRAP 沙箱 · 全程实测 ${(R.totalMs / 1000).toFixed(1)} 秒</div>`, ov);
  OV.win = el('div', 'sitewin', `<div class="bar"><i></i><i></i><i></i><span class="url">${URL_SHOW}</span></div><canvas width="1920" height="1080"></canvas>`, ov);
  OV.winCtx = OV.win.querySelector('canvas').getContext('2d');
  OV.winIdx = -1;
  OV.cur = el('div', 'cur', `<span class="rip"></span><svg viewBox="0 0 24 24" width="30" height="30"><path d="M5 2.5v17.2l4.6-4.3 3 6.6 2.9-1.3-3-6.5h6.2z" fill="#fff" stroke="#05080f" stroke-width="1.4" stroke-linejoin="round"/></svg>`, ov);
  OV.fin = el('div', 'fin', `<div class="url">${URL_SHOW}</div><div class="qr">${QR_SVG}</div><div class="hint">扫码打开智能体页，亲手一层层看</div>`, ov);
}

function updateOverlays(t, S) {
  OV.band.style.opacity = (S.band ?? 1).toFixed(3);
  OV.subs.update(t);
  OV.tags.update(t);
  // 章节进度
  const p = SC.progAt(t);
  OV.prog.style.display = p.a > 0.001 ? 'flex' : 'none';
  if (p.a > 0.001) {
    OV.prog.style.opacity = p.a.toFixed(3);
    if (OV.progKey !== p.i) { OV.progKey = p.i; OV.progIt.forEach((e, i) => { e.className = `it${i === p.i ? ' on' : i < p.i ? ' done' : ''}`; }); }
  }
  // 右上角：第几圈
  OV.lapc.style.opacity = S.lap.a.toFixed(3);
  const lh = `第 <b>${S.lap.n}</b> / ${R.turns.length} 圈`;
  if (OV.lapc.innerHTML !== lh) OV.lapc.innerHTML = lh;
  // 片名
  const ta = S.title.a;
  OV.title.style.display = ta > 0.001 ? 'block' : 'none';
  if (ta > 0.001) {
    const k = S.title.k;
    OV.title.style.opacity = ta.toFixed(3);
    OV.title.style.filter = S.title.blur > 0.05 ? `blur(${S.title.blur.toFixed(2)}px)` : '';
    const h1 = OV.title.querySelector('h1');
    h1.style.letterSpacing = `${(0.14 - 0.1 * easeOut(k)).toFixed(4)}em`;
    OV.title.querySelector('.rule').style.width = `${(560 * easeInOut(seg(k, 0.15, 0.8))).toFixed(1)}px`;
    OV.title.querySelector('.st').style.opacity = smooth(seg(k, 0.35, 0.8)).toFixed(3);
    OV.title.querySelector('.spec').style.opacity = (0.9 * smooth(seg(k, 0.55, 1))).toFixed(3);
    OV.title.querySelector('.eb').style.opacity = smooth(seg(k, 0, 0.5)).toFixed(3);
  }
  // 模型自己写下的话
  const X = S.say || S.ans;
  OV.say.style.display = X && X.a > 0.002 ? 'block' : 'none';
  if (X && X.a > 0.002) {
    OV.say.style.opacity = X.a.toFixed(3);
    OV.say.style.transform = `translateX(-50%) translateY(${((1 - easeOut(X.a)) * 16).toFixed(1)}px)`;
    const who = S.say ? `${MAN.model.name} · 第 6 圈 · 调用工具之前先写下的话` : `${MAN.model.name} · 第 8 圈 · 最终回答（没有工具调用）`;
    const wEl = OV.say.querySelector('.who span');
    if (wEl.textContent !== who) wEl.textContent = who;
    const tx = esc(X.text.slice(0, X.n)).replace(/`([^`]*)`/g, '$1') + (X.n < X.text.length ? '<span class="c"></span>' : '');
    const txEl = OV.say.querySelector('.tx');
    if (txEl.innerHTML !== tx) txEl.innerHTML = tx;
  }
  // 片尾：模型 + 工具 + 循环
  const sm = S.sum;
  OV.sum.style.display = sm && sm.a > 0.002 ? 'block' : 'none';
  if (sm && sm.a > 0.002) {
    OV.sum.style.opacity = sm.a.toFixed(3);
    const w = OV.sum.querySelectorAll('.big > span');
    w.forEach((e, i) => { const a = smooth(seg(sm.k, i * 0.16, i * 0.16 + 0.3)); e.style.opacity = a.toFixed(3); e.style.transform = `translateY(${((1 - easeOut(a)) * 18).toFixed(1)}px)`; });
    OV.sum.querySelector('.stat').style.opacity = sm.k2.toFixed(3);
    OV.sum.querySelector('.src').style.opacity = (0.9 * sm.k2).toFixed(3);
  }
  updateSite(S.site);
}

/* ================================================================ 片尾：智能体页的真实录屏 */

async function preloadSite() {
  const need = new Set();
  for (const sg of SC.endSite.segs) for (let f = sg.f0; f <= sg.f1 + 1e-6; f += 1 / 60) need.add(Math.min(CAP.n - 1, Math.max(0, Math.round(lerp(sg.c0, sg.c1, (f - sg.f0) / (sg.f1 - sg.f0)) * CAP.fps))));
  await Promise.all([...need].map((k) => new Promise((res) => { const im = new Image(); im.onload = () => res(); im.onerror = () => res(); im.src = `/ext/sitecap/f${String(k).padStart(4, '0')}.jpg`; SITE.set(k, im); })));
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
  const cx = lerp(960, 600, k), cy = lerp(470, 482, k), sc = lerp(1, 0.6, k);
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

/* ================================================================ 每帧 */

let lastS = null;
function renderAt(t, { render = true } = {}) {
  const S = (lastS = SC.frame(t));
  OPENING.update(t);
  DESK.update(S.desk);
  LOOP.update(S.loop);
  CTX.update(S.ctx);
  TOK.update(S.tok);
  updateOverlays(t, S);
  $('#fade').style.opacity = (S.fade ?? 0).toFixed(4);
  if (!render) return;
  drawBg(t, S.site && S.site.a > 0.5 ? 0.5 : 1);
  drawGrain(Math.round(t * FPS));
}

// 没有需要预滚收敛的状态（全是闭式），seek 直接渲染；pre 参数留着和 render.mjs 的接口一致
function seek(t) { renderAt(t); }

function poster(t) {
  seek(t ?? SC.dbg.titleIn + 2.6);
  for (const sel of ['.sub', '.tag', '.lapc']) document.querySelectorAll(sel).forEach((e) => { e.style.display = 'none'; });
}

function startPreview() {
  const bar = $('#scrub');
  bar.hidden = false;
  const r = $('#scrubber');
  r.max = SC.end;
  let playing = true, t = Number(params.get('t') || 0), last = performance.now();
  seek(t);
  r.addEventListener('input', () => { t = Number(r.value); seek(t); });
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
