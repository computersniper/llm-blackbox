// D2–D5 的舞台：agent 循环、上下文窗口、逐词元生成、模型内部。全部用录下来的真实数值。
import { $, esc, fmtPct, fmtNum, tokPlain } from '../../js/ui.js';
import { isDecision } from './data.js';
import { normPath, splitExit } from './screen.js';
import { callArg } from './chat.js';

export const KIND_NAME = { sys: '系统提示', tools: '工具定义', user: '用户任务', asst: '模型的输出', tool: '工具结果', gen: '轮到模型（生成提示）' };
const KIND_ORDER = ['sys', 'tools', 'user', 'asst', 'tool', 'gen'];
const SPECIAL = /(<\|im_start\|>|<\|im_end\|>|<\/?tool_call>|<\/?tool_response>|<\/?tools>)/g;

const pct = (n, max) => `${Math.max(0, (n / max) * 100).toFixed(3)}%`;
const svgNS = 'http://www.w3.org/2000/svg';

// 原始上下文里的特殊标记画成小块
function rawHTML(text) {
  return esc(text).replace(/(&lt;\|im_start\|&gt;|&lt;\|im_end\|&gt;|&lt;\/?tool_call&gt;|&lt;\/?tool_response&gt;|&lt;\/?tools&gt;)/g, '<span class="sp">$1</span>');
}

// 词元的可见写法
function tokShow(s) {
  if (s === '') return '';
  if (/^<\|.*\|>$|^<\/?tool_call>$/.test(s)) return s;
  return s;
}

/* ================================================================ D2 循环 */

export class LoopView {
  constructor() {
    this.el = $('#vLoop');
    this.laps = $('#laps');
    this.svg = $('#edges');
    this.mOut = $('#nModelOut');
    this.sOut = $('#nSbxOut');
    this.ops = [...document.querySelectorAll('#hOps li')];
    this.key = '';
    this.laps.addEventListener('click', (e) => { const b = e.target.closest('[data-t]'); if (b) this.onLap?.(Number(b.dataset.t)); });
  }

  setModel(M) {
    $('#nModelName').textContent = M.name.replace('-Instruct-2507', '');
    $('#nModelSpec').innerHTML = `${(M.params / 1e8).toFixed(1)} 亿参数 · ${M.layers} 层<br>只会做一件事：读一串词元，写下一个词元`;
    const st = document.createElement('div');
    st.className = 'stack';
    st.innerHTML = '<i></i>'.repeat(12);
    $('#nModelSpec').after(st);
    this.stack = [...st.children];
  }

  slotRect() {
    const v = this.el.getBoundingClientRect(), s = $('#sbxSlot').getBoundingClientRect();
    return { x: s.left - v.left, y: s.top - v.top, w: s.width, h: s.height };
  }

  update(st) {
    const { R, s, p } = st;
    const T = R.turns[s.t];
    const key = `${R.id}|${s.t}|${s.k2}`;
    if (key !== this.key) { this.key = key; this.renderStatic(R, s, T); }
    // 模型在写：已经写出的文字（工具调用部分标紫）
    if (s.p === 'gen' || s.p === 'parse') {
      const n = s.p === 'parse' ? T.gen.text.length : Math.floor(T.gen.text.length * Math.min(1, p * 1.05));
      const txt = T.gen.text.slice(0, n);
      const a = T.calls.length ? T.calls[0].a : txt.length;
      this.mOut.innerHTML = `<span class="a">${esc(txt.slice(0, a).slice(-160))}</span><span class="v">${esc(txt.slice(a).slice(-200))}</span>`;
      const k = Math.floor(p * 40) % 12;
      this.stack.forEach((el, i) => el.classList.toggle('lit', s.p === 'gen' && (i === k || i === (k + 6) % 12)));
    } else this.stack?.forEach((el) => el.classList.remove('lit'));
    this.drawEdges(st, T);
  }

  renderStatic(R, s, T) {
    // 圈数
    let h = `<span class="lk">第 <b>${s.t + 1}</b> / ${R.T} 圈</span>`;
    R.turns.forEach((x, t) => {
      const bad = x.calls.some((c) => !c.info?.ok || (c.name === 'bash' && splitExit(c.result).code !== 0));
      const lab = x.final ? '✓ 回答' : x.calls.map((c) => c.name || '?').join(' + ');
      h += `<button type="button" class="lap ${t < s.t ? 'done' : ''} ${t === s.t ? 'cur' : ''} ${bad ? 'bad' : ''} ${x.final ? 'fin' : ''}" data-t="${t}" title="第 ${t + 1} 圈：${esc(lab)}">${t + 1}<span>${esc(lab.replace('_file', ''))}</span></button>`;
    });
    this.laps.innerHTML = h;
    // harness 的步骤
    const order = ['prompt', 'gen', 'parse', 'exec', 'append', 'done'];
    const at = order.indexOf(s.p);
    this.ops.forEach((li, i) => {
      const op = li.dataset.op;
      const skip = T.final ? op === 'exec' || op === 'append' : op === 'done';
      li.className = `${i < at && !skip ? 'past' : ''} ${i === at ? 'on' : ''} ${skip ? 'skip' : ''}`;
    });
    const act = { prompt: 'nHarness', gen: 'nModel', parse: 'nHarness', exec: 'nSbx', append: 'nHarness', done: 'nHarness' }[s.p];
    for (const id of ['nModel', 'nHarness', 'nSbx']) $(`#${id}`).classList.toggle('on', id === act);
    if (s.p === 'prompt') this.mOut.innerHTML = `<span style="color:var(--dim)">等待输入：整段上下文 ${fmtNum(T.ctx.n)} 个词元</span>`;
    if (s.p === 'exec' || s.p === 'append') {
      const c = T.calls[s.c];
      const r = splitExit(c.result);
      const ok = c.info?.ok !== false && (r.code == null || r.code === 0);
      this.sOut.innerHTML = `<span class="v">${esc(c.name || '?')}</span> ${esc(callArg(c).slice(0, 80))}\n<span class="${ok ? 'g' : 'r'}">→ ${esc((r.body || c.result).split('\n').filter(Boolean).slice(-2).join('\n').slice(0, 160))}</span>`;
    } else if (s.p === 'prompt' && s.t > 0) {
      const prev = R.turns[s.t - 1];
      this.sOut.innerHTML = prev.calls.length ? `<span style="color:var(--dim)">上一圈：${esc(prev.calls.map((c) => c.name).join(' + '))} 已执行完</span>` : '';
    } else if (s.p === 'done') this.sOut.innerHTML = `<span class="g">${R.check.ok ? '✓ 独立检查通过' : '检查未通过'}</span>`;
    else if (s.t === 0 && s.p === 'prompt') this.sOut.innerHTML = '<span style="color:var(--dim)">沙箱已就绪</span>';
  }

  drawEdges(st, T) {
    const { s, p } = st;
    const v = this.el.getBoundingClientRect();
    if (!v.width || getComputedStyle(this.svg).display === 'none') return;
    const r = (id) => { const b = $(`#${id}`).getBoundingClientRect(); return { l: b.left - v.left, r: b.right - v.left, t: b.top - v.top, b: b.bottom - v.top, cx: (b.left + b.right) / 2 - v.left, h: b.height }; };
    const M = r('nModel'), H = r('nHarness'), S = r('nSbx');
    const c = T.calls[s.c];
    // 模型 ⇄ harness：左右两条；harness ⇄ 沙箱：上下两条
    const yA = Math.min(M.t, H.t) + Math.min(M.h, H.h) * 0.36, yB = yA + 46;
    const gx = (M.r + H.l) / 2;
    const xR = H.cx - 28, xB = H.cx + 28, my = (H.b + S.t) / 2;
    const edges = [
      { id: 'in', d: `M${H.l - 2},${yA} C${gx + 20},${yA - 10} ${gx - 20},${yA - 10} ${M.r + 6},${yA}`, on: s.p === 'prompt', label: `上下文 ${fmtNum(T.ctx.n)} 词元`, lx: gx, ly: yA - 16, anchor: 'middle' },
      { id: 'out', d: `M${M.r + 2},${yB} C${gx - 20},${yB + 10} ${gx + 20},${yB + 10} ${H.l - 6},${yB}`, on: s.p === 'gen' || s.p === 'parse', label: `输出 ${T.gen.n} 词元`, lx: gx, ly: yB + 24, anchor: 'middle' },
      { id: 'run', d: `M${xR},${H.b + 2} C${xR - 8},${my} ${xR - 8},${my} ${xR},${S.t - 6}`, on: s.p === 'exec', label: c ? `执行 ${c.name}` : '执行工具', lx: xR - 14, ly: my + 4, anchor: 'end' },
      { id: 'back', d: `M${xB},${S.t - 2} C${xB + 8},${my} ${xB + 8},${my} ${xB},${H.b + 6}`, on: s.p === 'append', label: c ? `结果 ${c.resultTokens} 词元` : '结果', lx: xB + 14, ly: my + 4, anchor: 'start' },
    ];
    const key = `${Math.round(v.width)}|${Math.round(v.height)}|${M.r | 0}|${H.b | 0}|${S.t | 0}|${s.t}|${s.k2}`;
    if (key !== this.edgeKey) {
      this.edgeKey = key;
      const mk = (id, on) => `<marker id="${id}" class="${on ? 'on' : ''}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z"/></marker>`;
      this.svg.innerHTML = `<defs>${mk('ah', false)}${mk('ahOn', true)}</defs>` + edges.map((e) => `<path id="e-${e.id}" class="${e.on ? 'on' : ''}" d="${e.d}" marker-end="url(#${e.on ? 'ahOn' : 'ah'})"/><text class="${e.on ? 'on' : ''}" x="${e.lx}" y="${e.ly}" text-anchor="${e.anchor}">${esc(e.label)}</text>`).join('') + '<circle class="pk" r="3.5" cx="-20" cy="-20"/>';
    }
    const on = edges.find((e) => e.on);
    const pk = this.svg.querySelector('.pk');
    if (on && pk) {
      const path = this.svg.querySelector(`#e-${on.id}`);
      const L = path.getTotalLength();
      const pt = path.getPointAtLength(L * ((p * 1.6) % 1));
      pk.setAttribute('cx', pt.x);
      pk.setAttribute('cy', pt.y);
      pk.style.opacity = 1;
    } else if (pk) pk.style.opacity = 0;
  }
}

/* ================================================================ D3 上下文窗口 */

export class CtxView {
  constructor() {
    this.el = $('#vCtx');
    this.head = $('#ctxHead');
    this.bar = $('#ctxBar');
    this.legend = $('#ctxLegend');
    this.stairs = $('#ctxStairs');
    this.raw = $('#ctxRaw');
    this.key = '';
    this.stairs.addEventListener('click', (e) => { const r = e.target.closest('[data-t]'); if (r) this.onTurn?.(Number(r.dataset.t)); });
    this.bar.addEventListener('click', (e) => {
      const sg = e.target.closest('[data-a]');
      if (!sg) return;
      const el = this.raw.querySelector(`[data-a="${sg.dataset.a}"]`);
      if (el) this.raw.scrollTop = el.offsetTop - 10;
    });
  }

  // 这一步时上下文里有哪些段
  segsAt(R, s, p) {
    const T = R.turns[s.t];
    const segs = T.ctx.segs.map((g) => ({ ...g }));
    const extra = [];
    const after = ['gen', 'parse', 'exec', 'append', 'done'].includes(s.p);
    if (after) {
      const n = s.p === 'gen' ? (s.j !== undefined ? s.j + 1 : Math.round(T.gen.n * p)) : T.gen.n;
      extra.push({ k: 'asst', n, grow: s.p === 'gen', t: s.t, out: true });
    }
    if (s.p === 'exec' || s.p === 'append') {
      const next = R.turns[s.t + 1];
      T.calls.forEach((c, ci) => {
        if (ci > s.c) return;
        const nTok = next ? next.ctx.segs.filter((g) => g.k === 'tool' && g.t === s.t).reduce((a, g) => a + g.n, 0) / T.calls.length : c.resultTokens + 5;
        const pending = ci === s.c && s.p === 'exec';
        extra.push({ k: 'tool', n: Math.round(nTok), ghost: pending, t: s.t, out: true });
      });
    }
    return { segs, extra };
  }

  update(st) {
    const { R, s, p } = st;
    const T = R.turns[s.t];
    const key = `${R.id}|${s.t}|${s.k2}|${s.k3}|${s.j ?? ''}`;
    const grow = s.p === 'gen' && s.j === undefined;
    if (key !== this.key) { this.key = key; this.renderStatic(R, s, T, p); }
    else if (grow) this.renderBar(R, s, T, p);
  }

  renderStatic(R, s, T, p) {
    const max = R.maxCtx;
    const M = R.M;
    const kv = s.sub === 'kv';
    const total = T.ctx.n;
    let note = '';
    if (s.p === 'prompt' && s.sub !== 'kv') note = s.t === 0 ? '第一圈：系统提示、工具定义、用户任务' : `比上一圈多了 <b>${fmtNum(total - R.turns[s.t - 1].ctx.n)}</b> 个词元`;
    if (kv) note = `复用 <b>${fmtNum(T.kv.reused)}</b> · 新算 <b>${fmtNum(T.kv.computed)}</b> · 预填充实测 <b>${T.ms.prefill} ms</b>`;
    if (s.p === 'gen' || s.p === 'parse') note = `模型往后接着写：这一圈一共写了 <b>${T.gen.n}</b> 个词元，实测 <b>${(T.ms.decode / 1000).toFixed(1)} s</b>`;
    if (s.p === 'exec') note = '工具在沙箱里执行，结果还没接回来';
    if (s.p === 'append') note = `工具结果包进 <b>&lt;tool_response&gt;</b>，以 user 的身份接到末尾`;
    if (s.p === 'done') note = `任务结束时，整段对话一共 <b>${fmtNum(total + T.gen.n)}</b> 个词元`;
    this.head.innerHTML = `<h3>第 ${s.t + 1} 圈 · 喂给模型的整段上下文</h3><span class="big">${fmtNum(total)}<small>词元</small></span><span class="note">${note}</span><span class="note">上限 <b>${fmtNum(M.ctx)}</b> · 只用了 ${((total / M.ctx) * 100).toFixed(2)}%</span>`;
    this.renderBar(R, s, T, p);
    // 阶梯：每一圈的上下文长度
    let h = '<div class="sh">每一圈喂进去的长度</div>';
    R.turns.forEach((x, t) => {
      h += `<div class="st-row ${t === s.t ? 'cur' : ''} ${t > s.t ? 'fut' : ''}" data-t="${t}" title="第 ${t + 1} 圈：${fmtNum(x.ctx.n)} 词元，其中复用 KV 缓存 ${fmtNum(x.kv.reused)}，新算 ${fmtNum(x.kv.computed)}；生成 ${x.gen.n}"><span>#${t + 1}</span><span class="bar"><i class="reuse" style="width:${pct(x.kv.reused, max)}"></i><i class="newc" style="width:${pct(x.kv.computed, max)}"></i><i class="gen" style="width:${pct(x.gen.n, max)}"></i></span><span>${fmtNum(x.ctx.n)}</span></div>`;
    });
    this.stairs.innerHTML = h;
    // 原始文本
    this.renderRaw(R, s, T, p);
  }

  renderBar(R, s, T, p) {
    const max = R.maxCtx;
    const { segs, extra } = this.segsAt(R, s, p);
    const prev = s.t > 0 ? R.turns[s.t - 1] : null;
    const prevLen = prev ? prev.ctx.n + prev.gen.n : 0;
    let acc = 0, h = '';
    const fresh = s.p === 'prompt' && s.sub !== 'kv';
    for (const g of segs) {
      const isNew = fresh && acc + g.n > prevLen && s.t > 0 && g.k !== 'sys' && g.k !== 'tools';
      h += `<div class="sg ${g.k} ${isNew ? 'new' : ''}" data-a="${g.a}" style="width:${pct(g.n, max)}" title="${KIND_NAME[g.k]}${g.t != null ? `（第 ${g.t + 1} 圈）` : ''} · ${g.n} 词元">${g.n / max > 0.045 ? `<span>${g.n}</span>` : ''}</div>`;
      acc += g.n;
    }
    for (const g of extra) h += `<div class="sg ${g.k} ${g.grow ? 'grow' : ''} ${g.ghost ? 'ghost' : ''}" style="width:${pct(g.n, max)}" title="${KIND_NAME[g.k]} · ${g.n} 词元">${g.n / max > 0.04 ? `<span>${g.ghost ? '…' : g.n}</span>` : ''}</div>`;
    if (s.sub === 'kv') h += `<div class="kv" style="width:${pct(T.kv.reused, max)}"><b>KV 缓存复用 ${fmtNum(T.kv.reused)}</b></div>`;
    this.bar.innerHTML = h;
    // 图例：各类合计
    const sum = {};
    for (const g of segs.concat(extra)) sum[g.k] = (sum[g.k] || 0) + g.n;
    this.legend.innerHTML = KIND_ORDER.filter((k) => sum[k]).map((k) => `<span><i style="background:var(--k-${k})"></i>${KIND_NAME[k]} <b>${fmtNum(sum[k])}</b></span>`).join('');
  }

  renderRaw(R, s, T, p) {
    const text = T.prompt;
    const newFrom = s.p === 'prompt' ? T.ctx.keep : text.length;
    let h = '';
    for (const g of T.ctx.segs) {
      const a = g.a, b = g.b;
      if (newFrom > a && newFrom < b) {
        h += `<span class="r ${g.k}" data-a="${a}">${rawHTML(text.slice(a, newFrom))}</span><span class="r ${g.k} newp">${rawHTML(text.slice(newFrom, b))}</span>`;
      } else h += `<span class="r ${g.k} ${a >= newFrom ? 'newp' : ''}" data-a="${a}">${rawHTML(text.slice(a, b))}</span>`;
    }
    if (['gen', 'parse', 'exec', 'append', 'done'].includes(s.p)) {
      const n = s.p === 'gen' && s.j !== undefined ? T.gen.toks[s.j].b : T.gen.text.length;
      h += `<span class="out">${rawHTML(T.gen.text.slice(0, n))}</span>`;
      if (s.p !== 'gen') h += `<span class="sp">&lt;|im_end|&gt;</span>`;
      if (s.p === 'exec' || s.p === 'append') {
        T.calls.forEach((c, ci) => {
          if (ci > s.c) return;
          if (ci === s.c && s.p === 'exec') h += '\n<span class="pend">（工具正在沙箱里执行……）</span>';
          else h += `<span class="r tool">${rawHTML(`${ci === 0 ? '\n<|im_start|>user' : ''}\n<tool_response>\n${c.result}\n</tool_response>`)}</span>`;
        });
      }
    }
    this.raw.innerHTML = h;
    // 滚到最新的部分
    const target = this.raw.querySelector('.newp') || this.raw.querySelector('.out');
    if (target && s.p === 'prompt') this.raw.scrollTop = Math.max(0, target.offsetTop - 30);
    else this.raw.scrollTop = this.raw.scrollHeight;
  }
}

/* ================================================================ D4 逐词元 */

export class TokView {
  constructor() {
    this.el = $('#vTok');
    this.head = $('#tokHead');
    this.stream = $('#tokStream');
    this.cands = $('#tokCands');
    this.turnKey = '';
    this.key = '';
    this.stream.addEventListener('click', (e) => { const k = e.target.closest('[data-j]'); if (k) this.onTok?.(Number(k.dataset.j)); });
  }

  buildStream(R, t) {
    const T = R.turns[t];
    const inCall = (j) => T.calls.some((c) => j >= c.j0 && j <= c.j1);
    this.stream.innerHTML = T.gen.toks.map((tk, j) => {
      const sp = /^<\/?tool_call>$/.test(tk.s);
      const txt = tk.end ? '<|im_end|>' : tk.s === '' ? '·' : tk.s;
      const cls = ['tk', inCall(j) ? 'call' : '', sp || tk.end ? 'sp' : '', tk.end ? 'end' : '', isDecision(tk, j) ? 'dec' : ''].join(' ');
      const shown = esc(txt).replace(/\n/g, '<i style="opacity:.4;font-style:normal">↵</i>\n');
      return `<span class="${cls}" data-j="${j}" title="${tk.s === '' && !tk.end ? '半个字：要和下一个词元拼起来才能显示' : `概率 ${fmtPct(tk.p)}`}">${shown}</span>`;
    }).join('');
    this.spans = [...this.stream.children];
  }

  update(st) {
    const { R, s } = st;
    const T = R.turns[s.t];
    const tkey = `${R.id}|${s.t}`;
    if (tkey !== this.turnKey) { this.turnKey = tkey; this.buildStream(R, s.t); }
    const key = `${tkey}|${s.k2}|${s.j ?? ''}`;
    if (key === this.key) return;
    this.key = key;
    const parse = s.p === 'parse';
    const j = parse ? T.gen.toks.length - 1 : s.j;
    this.spans.forEach((el, k) => {
      el.classList.toggle('fut', k > j);
      el.classList.toggle('cur', k === j && !parse);
      el.classList.toggle('parsed', parse && T.calls.some((c) => k >= c.j0 && k <= c.j1));
    });
    const curEl = this.spans[j];
    if (curEl) {
      const top = curEl.offsetTop - this.stream.clientHeight * 0.45;
      if (curEl.offsetTop < this.stream.scrollTop + 10 || curEl.offsetTop > this.stream.scrollTop + this.stream.clientHeight - 30) this.stream.scrollTop = top;
    }
    const rate = T.gen.n ? (T.gen.n / (T.ms.decode / 1000)).toFixed(0) : 0;
    this.head.innerHTML = parse
      ? `<h3>第 ${s.t + 1} 圈 · harness 解析模型写下的文字</h3><span>在输出里找 <b>&lt;tool_call&gt;…&lt;/tool_call&gt;</b>，把中间那段当 JSON 读</span>`
      : `<h3>第 ${s.t + 1} 圈 · 第 <b>${j + 1}</b> / ${T.gen.toks.length} 个词元</h3><span>实测：这一圈 ${T.gen.n} 个词元用了 <b>${(T.ms.decode / 1000).toFixed(2)} s</b>（约 ${rate} 词元/秒）</span><span>● 关键决策点</span>`;
    if (parse) this.renderParse(T);
    else this.renderCands(T, j);
  }

  renderCands(T, j) {
    const tk = T.gen.toks[j];
    const dec = isDecision(tk, j);
    const shown = tk.end ? '<|im_end|>' : tk.s === '' ? '（半个字）' : tokPlain(tk.s);
    const max = Math.max(...tk.top.map((x) => x[1]), 1e-9);
    const chosen = tk.top.findIndex((x) => Math.abs(x[1] - tk.p) < 1e-9);
    const rows = tk.top.map(([w, pr], i) => `<div class="tc-row ${i === chosen ? 'win' : ''}"><span class="w">${esc(tokPlain(w))}</span><span class="bar"><i style="width:${(pr / max) * 100}%"></i></span><span class="p">${fmtPct(pr)}</span></div>`).join('');
    const why = j === 0 ? (T.calls.length ? '每一圈的第一个词元：先说句话，还是直接调用工具？' : '每一圈的第一个词元：继续调用工具，还是开始回答？')
      : tk.end ? '说完了吗？选中 <|im_end|>，这一圈就结束，控制权交回 harness。'
        : '第一名不到 75%：模型在这里真的有得选。';
    this.cands.innerHTML = `
      <div class="tc-big">
        <span class="k">选中的词元</span>
        <span class="w">${esc(shown)}</span>
        <span class="p">概率 ${fmtPct(tk.p)}${chosen < 0 ? '（不在前 5 名里，是抽签抽中的）' : ''}</span>
        ${dec ? '<span class="dec">● 关键决策</span>' : ''}
      </div>
      <div class="tc-bars">${rows}
        <div class="tc-note">${dec ? esc(why) + ' ' : ''}前 5 名是温度 1 的原始概率。真正抽签时温度 0.7、top-k 20、top-p 0.8，这一步的候选池剩 <b>${tk.pool}</b> 个。</div>
      </div>`;
  }

  renderParse(T) {
    if (!T.calls.length) {
      this.cands.innerHTML = `<div class="tc-big"><span class="k">解析结果</span><span class="w" style="font-size:16px">没有 &lt;tool_call&gt;</span></div><div class="tc-bars"><div class="tc-note">模型这一圈只写了文字。对 harness 来说，这就是最终回答：循环结束，把这段话交给用户。</div></div>`;
      return;
    }
    const cards = T.calls.map((c) => {
      if (!c.ok) return `<div class="json-card bad">json.loads 失败：${esc(c.error || '')}</div>`;
      const args = Object.entries(c.args || {}).map(([k, v]) => `    <span class="k">"${esc(k)}"</span>: <span class="s">${esc(JSON.stringify(v))}</span>`).join(',\n');
      return `<div class="json-card">{\n  <span class="k">"name"</span>: <span class="s">"${esc(c.name)}"</span>,\n  <span class="k">"arguments"</span>: {\n${args}\n  }\n}</div>`;
    }).join('');
    this.cands.innerHTML = `<div class="tc-big"><span class="k">解析结果</span><span class="w" style="font-size:16px">${T.calls.length} 个工具调用</span><span class="p">${T.calls.map((c) => esc(c.name || '?')).join(' + ')}</span></div><div class="tc-bars">${cards}<div class="tc-note">这只是模型写下的一段文字。是 harness 把它当成命令去执行的。</div></div>`;
  }
}

/* ================================================================ D5 模型内部 */

export class FwdView {
  constructor() {
    this.el = $('#vFwd');
    this.col = $('#fwdCol');
    this.info = $('#fwdInfo');
    this.key = '';
    this.layers = [];
  }

  update(st) {
    const { R, s, p } = st;
    const T = R.turns[s.t];
    const M = R.M;
    const tk = T.gen.toks[s.j];
    const key = `${R.id}|${s.t}|${s.j}|${s.f}`;
    if (key !== this.key) {
      this.key = key;
      const prevTok = s.j > 0 ? T.gen.toks[s.j - 1] : null;
      const ctxLen = T.ctx.n + s.j;
      const input = prevTok ? (prevTok.s === '' ? '（半个字）' : tokPlain(prevTok.s)) : '<|im_start|>assistant\\n';
      const shown = tk.end ? '<|im_end|>' : tk.s === '' ? '（半个字）' : tokPlain(tk.s);
      let lab = '';
      for (const L of [0, 9, 18, 27, 35]) lab += `<span class="lab" style="bottom:${((L + 0.5) / M.layers) * 100}%">L${L}</span>`;
      this.col.innerHTML = `
        <div class="fwd-io"><span class="k">输出 · 下一个词元</span>${s.f === 'pick' ? `抽中 <b>${esc(shown)}</b>（${fmtPct(tk.p)}）` : '15 万个词元的概率……'}</div>
        <div class="fwd-layers" style="padding-right:30px">${'<i></i>'.repeat(M.layers)}${lab}</div>
        <div class="fwd-io"><span class="k">输入</span>${s.j === 0 ? '整段上下文' : `上一个词元 <b>${esc(input)}</b>`} · KV 缓存里已有 <b>${fmtNum(ctxLen)}</b> 个</div>`;
      this.layers = [...this.col.querySelectorAll('.fwd-layers i')];
      const perTok = T.gen.n ? T.ms.decode / T.gen.n : 0;
      const pool = this.poolProbs(tk);
      this.info.innerHTML = s.f === 'fwd' ? `
        <h3>第 ${s.j + 1} 个词元：一次完整的前向计算</h3>
        <div>为了写出这一个词元，模型要让它穿过全部 <b>${M.layers}</b> 层 Transformer：每层先做注意力（回头看 KV 缓存里的 ${fmtNum(ctxLen)} 个词元），再过一个前馈网络。最后接输出头，算出词表里 ${fmtNum(M.vocab)} 个词元各自的分数。</div>
        <div class="fwd-spec">
          <div><b>${(M.params / 1e8).toFixed(1)} 亿</b>参数（bf16）</div>
          <div><b>${M.layers}</b> 层</div>
          <div><b>${M.hidden}</b> 隐藏维度</div>
          <div><b>${M.heads} / ${M.kvHeads}</b> Q 头 / KV 头</div>
          <div><b>${M.ffn}</b> SwiGLU 宽度</div>
          <div><b>${perTok.toFixed(0)} ms</b> 实测每个词元</div>
        </div>
        <div>每个参数大约参与一次乘加：写一个词元 ≈ <b>${(M.params / 1e8).toFixed(0)} 亿次乘加</b>。agent 这一整趟任务一共生成了 ${fmtNum(R.genTotal)} 个词元，再加上每一圈的预填充。</div>
        <div style="color:var(--dim)">这一层我们没有录下内部数值。同一家族的小模型 Qwen3-0.6B（28 层）每一层里的真实计算，在“推理”页面可以一路拆到权重的比特：</div>
        <a class="fwd-link" href="../">到「推理」页，看一层里面的真实计算 →</a>` : `
        <h3>从概率到一个词元：抽签</h3>
        <div>输出头给出的是 ${fmtNum(M.vocab)} 个词元的概率。agent 用的采样参数是温度 <b>${M.sampling.temperature}</b>、top-k <b>${M.sampling.top_k}</b>、top-p <b>${M.sampling.top_p}</b>：先放大强者，只留前 20 名，再按累计概率截到 80%，最后掷骰子。</div>
        ${pool}
        <div style="color:var(--dim)">选中的词元会被接到上下文末尾，作为下一步的输入。等它选中 &lt;|im_end|&gt;，这一圈就结束了。</div>`;
    }
    if (s.f === 'fwd') {
      const n = Math.floor(p * 1.1 * this.layers.length);
      this.layers.forEach((el, i) => { el.classList.toggle('lit', i < n); el.classList.toggle('now', i === n); });
    } else this.layers.forEach((el) => { el.classList.add('lit'); el.classList.remove('now'); });
  }

  // 候选池：前 5 名按温度重新归一化（池子不超过 5 个时是精确的）
  poolProbs(tk) {
    const T = 0.7;
    const top = tk.top.slice(0, Math.min(tk.pool, tk.top.length));
    const w = top.map(([, pr]) => Math.pow(Math.max(pr, 1e-12), 1 / T));
    const sum = w.reduce((a, b) => a + b, 0);
    const chosen = tk.top.findIndex((x) => Math.abs(x[1] - tk.p) < 1e-9);
    const bars = top.map(([wd], i) => `<i class="${i === chosen ? 'win' : ''}" style="width:${(w[i] / sum) * 100}%" title="${esc(tokPlain(wd))} ${fmtPct(w[i] / sum)}"></i>`).join('');
    const list = top.map(([wd], i) => `<span style="font-family:var(--mono);font-size:12px;${i === chosen ? 'color:var(--amber)' : ''}">${esc(tokPlain(wd))} ${fmtPct(w[i] / sum)}</span>`).join('　');
    return `<div>候选池里剩 <b>${tk.pool}</b> 个${tk.pool === 1 ? '：没什么可抽的，就是它' : ''}${tk.pool > 5 ? '（下面只画出前 5 个）' : ''}：</div><div class="draw">${bars}</div><div>${list}</div>`;
  }
}
