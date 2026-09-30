// 训练 / 多模态 / 智能体三个页面共用的调试器：和推理页同一套玩法。
//
//   一棵“步骤树”：第 1 层是最粗的步子，每个节点可以有更细的子步骤。
//   ＋ = 单步进入（换到更细的一层，停在当前步骤的第一个子步骤）
//   − = 跳出（回到包含当前步骤的那一步）
//   ⏮ / ⏭ = 在当前深度逐步执行；▶ = 按倍速自动播放
//
// 页面提供：roots（第 1 层节点）、每个节点的 kids()、深度名、伪代码，以及 render / explain。
// 节点：{ key, label, dur?, kids?: () => Node[], ...页面自己的字段 }

import { $, $$, esc } from '../ui.js';
import { Background } from '../bg.js';

export const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];

function kidsOf(n) {
  if (!n.kids) return null;
  if (!n._kids) n._kids = n.kids() || [];
  return n._kids.length ? n._kids : null;
}

// 把 root 往下展开 levels 层，得到这一深度上的步骤序列（每个元素是从 root 出发的路径）
function expand(path, levels) {
  const n = path[path.length - 1];
  const ks = levels > 0 ? kidsOf(n) : null;
  if (!ks) return [path];
  return ks.flatMap((k) => expand([...path, k], levels - 1));
}

const isPrefix = (a, b) => a.length <= b.length && a.every((n, i) => n === b[i]);

export class Tree {
  constructor(roots, maxDepth) {
    this.roots = roots;
    this.maxDepth = maxDepth;
    this.depth = 1;
    this.r = 0;
    this.i = 0;
    this.p = 0;
    this.speed = 1;
    this.playing = false;
    this.done = false;
    this.listeners = new Set();
    this.list = this.build(1, 0);
  }

  // 深度 1：所有根节点排成一列；更深：只展开当前根节点（和推理页“每个词元一组步骤”一样）
  build(depth, r) {
    if (depth <= 1) return this.roots.map((n) => [n]);
    return expand([this.roots[r]], depth - 1);
  }

  get path() { return this.list[this.i]; }
  get node() { const p = this.path; return p[p.length - 1]; }
  get root() { return this.roots[this.r]; }
  get level() { return this.path.length; } // 当前节点实际所在的层（可能比 depth 浅：没有更细的步子了）
  get dur() { return this.node.dur ?? 1.6; }
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(t) { for (const fn of this.listeners) fn(t, this); }

  canInto() { return this.depth < this.maxDepth && !!kidsOf(this.node) && this.level === this.depth; }

  setDepth(d) {
    d = Math.max(1, Math.min(this.maxDepth, d));
    if (d === this.depth) return;
    const cur = this.path;
    const list = this.build(d, this.r);
    const idx = d > this.depth ? list.findIndex((b) => isPrefix(cur, b)) : list.findIndex((a) => isPrefix(a, cur));
    this.depth = d;
    this.list = list;
    this.i = Math.max(0, idx);
    this.p = 0;
    this.done = false;
    this.emit('step');
  }

  into() { if (this.canInto()) this.setDepth(this.depth + 1); }
  out() { if (this.depth > 1) this.setDepth(this.depth - 1); }

  advance() {
    this.done = false;
    if (this.depth === 1) {
      if (this.i < this.list.length - 1) { this.i++; this.r = this.i; this.p = 0; this.emit('root'); this.emit('step'); return true; }
    } else {
      if (this.i < this.list.length - 1) { this.i++; this.p = 0; this.emit('step'); return true; }
      if (this.r < this.roots.length - 1) {
        this.r++;
        this.list = this.build(this.depth, this.r);
        this.i = 0;
        this.p = 0;
        this.emit('root');
        this.emit('step');
        return true;
      }
    }
    this.p = 1;
    if (!this.done) { this.done = true; this.emit('end'); }
    return false;
  }

  back() {
    this.done = false;
    if (this.p > 0.15) { this.p = 0; this.emit('step'); return; }
    if (this.depth === 1) {
      if (this.i > 0) { this.i--; this.r = this.i; this.p = 0; this.emit('root'); this.emit('step'); }
      return;
    }
    if (this.i > 0) { this.i--; this.p = 0; this.emit('step'); return; }
    if (this.r > 0) {
      this.r--;
      this.list = this.build(this.depth, this.r);
      this.i = this.list.length - 1;
      this.p = 0;
      this.emit('root');
      this.emit('step');
    }
  }

  seekIndex(i) {
    this.i = Math.max(0, Math.min(this.list.length - 1, i));
    if (this.depth === 1) { this.r = this.i; this.emit('root'); }
    this.p = 0;
    this.done = false;
    this.emit('step');
  }

  // 跳到某个根节点（例如在左侧点了某个检查点 / 某个字）
  seekRoot(r, depth = this.depth) {
    r = Math.max(0, Math.min(this.roots.length - 1, r));
    this.r = r;
    this.depth = depth;
    this.list = this.build(depth, r);
    this.i = depth === 1 ? r : 0;
    this.p = 0;
    this.done = false;
    this.emit('root');
    this.emit('step');
  }

  // 跳到当前根节点下满足条件的节点，深度设为它自然所在的那一层
  seekWhere(pred) {
    for (let d = 2; d <= this.maxDepth; d++) {
      const list = this.build(d, this.r);
      const i = list.findIndex((p) => p.length === d && pred(p[p.length - 1], p));
      if (i >= 0) { this.depth = d; this.list = list; this.seekIndex(i); return true; }
    }
    return false;
  }

  play() { if (this.done) { this.done = false; } this.playing = true; this.emit('play'); }
  pause() { this.playing = false; this.emit('play'); }
  toggle() { this.playing ? this.pause() : this.play(); }
  next() { this.p = 0; this.advance(); }

  tick(dt) {
    if (!this.playing) return;
    this.p += (dt * this.speed) / this.dur;
    let guard = 0;
    while (this.p >= 1 && guard++ < 50) {
      const carry = this.p - 1;
      if (!this.advance()) { this.playing = false; this.emit('play'); break; }
      this.p = Math.min(carry, 0.99);
    }
  }
}

// ---------------------------------------------------------------- 伪代码

const KW = /\b(for|in|if|else|return|def|with|while|and|or|not|None|True|False|import|from)\b/g;

function hl(line) {
  const ci = line.indexOf('#');
  const code = ci >= 0 ? line.slice(0, ci) : line;
  const cm = ci >= 0 ? line.slice(ci) : '';
  let h = esc(code)
    .replace(/\b(\d+(?:\.\d+)?(?:e-?\d+)?)\b/g, '<span class="nu">$1</span>')
    .replace(KW, '<span class="kw">$1</span>')
    .replace(/\b([a-zA-Z_][\w.]*)(?=\()/g, '<span class="fn">$1</span>');
  if (cm) h += `<span class="cm">${esc(cm)}</span>`;
  return h;
}

export function renderCode(el, lines) {
  el.innerHTML = lines.map((l, i) => `<span class="ln"><i class="bp"></i><span class="no">${i + 1}</span>${hl(l)}</span>`).join('');
}

// ---------------------------------------------------------------- 页面外壳

export class Lab {
  /**
   * @param {object} o
   *   roots, maxDepth, depthNames: ['总览', ...]（下标 = 深度）
   *   code: 伪代码行
   *   explain(node, path, tree) → { lines:[行号], shape, html, watch:[[k, v] | '— 标题'] }
   *   render(node, path, tree, changed)  每一步调用；frame(node, p, dt) 每帧调用（可选）
   *   posText(tree) → 右下角的位置文字
   *   onExit()  在深度 1 按 − 时调用
   */
  constructor(o) {
    this.o = o;
    this.tree = new Tree(o.roots, o.maxDepth);
    this.code = $('#code');
    renderCode(this.code, o.code);
    this.lines = $$('.ln', this.code);
    if ($('#bg')) this.bg = new Background($('#bg'));
    this.bind();
    this.tree.on((t) => {
      if (t === 'step' || t === 'root') this.update(t);
      if (t === 'play') this.updatePlay();
    });
    this.update('root');
    let last = performance.now();
    const loop = (now) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      this.tree.tick(dt);
      o.frame?.(this.tree.node, this.tree.p, dt, this.tree);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  bind() {
    const t = this.tree;
    $('#btnIn').addEventListener('click', () => { t.pause(); t.into(); });
    $('#btnOut').addEventListener('click', () => this.out());
    $('#btnPrev').addEventListener('click', () => { t.pause(); t.back(); });
    $('#btnNextStep').addEventListener('click', () => { t.pause(); t.next(); });
    $('#btnPlay').addEventListener('click', () => t.toggle());
    $('#btnDbgFold')?.addEventListener('click', () => {
      const d = $('#dbg');
      d.classList.toggle('folded');
      $('#btnDbgFold').textContent = d.classList.contains('folded') ? '+' : '–';
    });
    const sp = $('#speeds');
    sp.innerHTML = SPEEDS.map((s) => `<button type="button" data-s="${s}" role="radio">${s}×</button>`).join('');
    sp.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) { t.speed = Number(b.dataset.s); this.updatePlay(); } });
    $('#track').addEventListener('click', (e) => { const i = e.target.closest('i'); if (i) { t.pause(); t.seekIndex(Number(i.dataset.i)); } });
    $('#track').addEventListener('pointerover', (e) => { const i = e.target.closest('i'); if (i) i.title = i.dataset.l; });
    $('#crumbs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) { t.pause(); t.setDepth(Number(b.dataset.d)); } });
    addEventListener('keydown', (e) => {
      if (e.target.closest('input, textarea, select')) return;
      if (!document.body.classList.contains('mode-inspect')) return;
      if (e.key === ' ') { e.preventDefault(); t.toggle(); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); t.pause(); t.next(); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); t.pause(); t.back(); }
      else if (e.key === '+' || e.key === '=' || e.key === 'ArrowDown') { e.preventDefault(); t.pause(); t.into(); }
      else if (e.key === '-' || e.key === '_' || e.key === 'ArrowUp' || e.key === 'Escape') { e.preventDefault(); this.out(); }
      else if (e.key >= '1' && e.key <= '6') { t.speed = SPEEDS[Number(e.key) - 1]; this.updatePlay(); }
    });
  }

  out() {
    const t = this.tree;
    t.pause();
    if (t.depth <= 1) this.o.onExit?.();
    else t.out();
  }

  crumbs() {
    const t = this.tree, path = t.path, names = this.o.depthNames;
    const items = [];
    for (let d = 1; d <= t.depth; d++) {
      const n = path[Math.min(d, path.length) - 1];
      items.push({ d, label: d <= path.length ? n.crumb || n.label : names[d] });
    }
    $('#crumbs').innerHTML = items.map((c, k) => `${k ? '<span class="sep">›</span>' : ''}<button type="button" data-d="${c.d}" class="${c.d === t.depth ? 'on' : ''}"><span class="d">D${c.d}</span>${esc(c.label)}</button>`).join('');
  }

  update(kind) {
    const t = this.tree, n = t.node, o = this.o;
    this.crumbs();
    $('#dname').innerHTML = `${esc(o.depthNames[t.depth])}<small>DEPTH ${t.depth} / ${o.maxDepth}</small>`;
    $('#btnIn').disabled = !t.canInto();
    $('#btnOut').title = t.depth <= 1 ? '回到选择页（−）' : '退回上一层（−）';
    this.bg?.setDepth(t.depth - 1);
    const ex = o.explain(n, t.path, t) || {};
    const cur = new Set(ex.lines || []);
    this.lines.forEach((l, i) => l.classList.toggle('cur', cur.has(i + 1)));
    const first = this.lines[Math.min(...cur) - 1];
    if (first) {
      const box = this.code.parentElement;
      if (first.offsetTop < box.scrollTop + 20 || first.offsetTop > box.scrollTop + box.clientHeight * 0.55) {
        box.scrollTo({ top: first.offsetTop - 60, behavior: 'smooth' });
      }
    }
    $('#explain').innerHTML = `<div class="eyebrow" style="margin-bottom:4px">这一步 · ${esc(n.label)}</div>${ex.shape ? `<div class="shape">${ex.shape}</div>` : ''}${ex.html || ''}`;
    $('#watch').innerHTML = (ex.watch || []).map((w) => (typeof w === 'string'
      ? `<span class="wh">${esc(w)}</span>`
      : `<span class="k">${esc(w[0])}</span><span class="v" title="${esc(String(w[1]).replace(/<[^>]+>/g, ''))}">${w[2] ? w[1] : esc(w[1])}</span>`)).join('');
    this.track();
    $('#pos').innerHTML = o.posText ? o.posText(t) : `步骤 <b>${t.i + 1}</b>/${t.list.length}`;
    this.updatePlay();
    o.render(n, t.path, t, kind);
  }

  track() {
    const t = this.tree, n = t.list.length, tr = $('#track');
    const a = n <= 90 ? 0 : Math.max(0, Math.min(n - 80, t.i - 40)), b = n <= 90 ? n : a + 80;
    let h = '';
    for (let i = a; i < b; i++) {
      const p = t.list[i];
      h += `<i data-i="${i}" data-l="${esc(p[p.length - 1].label)}" class="${i < t.i ? 'done' : i === t.i ? 'cur' : ''}"></i>`;
    }
    tr.innerHTML = h;
  }

  updatePlay() {
    const t = this.tree;
    $('#btnPlay').textContent = t.playing ? '❚❚' : '▶';
    $('#btnPlay').title = t.playing ? '暂停（空格）' : '播放（空格）';
    $$('#speeds button').forEach((b) => { const on = Number(b.dataset.s) === t.speed; b.classList.toggle('on', on); b.setAttribute('aria-checked', on); });
  }
}

// ---------------------------------------------------------------- 数据

// 预先 gzip 过的 json（服务器只压缩 HTML），浏览器里用 DecompressionStream 解开
export async function fetchJSON(url) {
  if (typeof DecompressionStream !== 'undefined') {
    try {
      const r = await fetch(`${url}.gz`);
      if (r.ok && r.body) return JSON.parse(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).text());
    } catch { /* 退回未压缩 */ }
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} ${r.status}`);
  return r.json();
}

// ---------------------------------------------------------------- 小工具

export const fmt = (v, d = 3) => {
  if (v == null || Number.isNaN(v)) return '—';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2).replace('e', '×10^').replace(/\^\+?(-?\d+)/, (m, e) => `<sup>${e}</sup>`);
  return v.toFixed(d);
};
export const fmtP = (p) => (p >= 0.995 ? '100%' : p >= 0.1 ? `${(p * 100).toFixed(0)}%` : p >= 0.001 ? `${(p * 100).toFixed(1)}%` : p > 0 ? '<0.1%' : '0%');

// 数值 → 颜色（正：琥珀，负：蓝）
export function heat(v, max) {
  const t = Math.min(1, Math.abs(v) / (max || 1));
  return v >= 0 ? `rgba(255,182,92,${0.08 + 0.85 * t})` : `rgba(107,155,255,${0.08 + 0.85 * t})`;
}
export const cyan = (t) => `rgba(94,240,212,${0.06 + 0.9 * Math.min(1, Math.max(0, t))})`;

// 简单的折线（SVG path）
export function linePath(xs, ys, x0, x1, y0, y1, W, H) {
  let d = '';
  for (let i = 0; i < xs.length; i++) {
    const x = ((xs[i] - x0) / (x1 - x0)) * W, y = H - ((ys[i] - y0) / (y1 - y0)) * H;
    d += `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
  }
  return d;
}
