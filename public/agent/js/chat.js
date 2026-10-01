// 左栏：任务库 + 点选输入法 + agent 的对话记录。
// 输入法的候选是任务句子的真实分词（Qwen3 的词元），只能拼出真的录制过的任务。
import { $, $$, esc, tokHTML, tokInner, sleep, fmtNum } from '../../js/ui.js';
import { normPath, splitExit } from './screen.js';

const ICON = { bug: '✕', log: '≡', chart: '▤', rename: '⇄' };
const ckey = (c) => c.ids.join(',');

export function md(text) {
  return esc(text).replace(/`([^`\n]+)`/g, '<code>$1</code>');
}

export function callArg(c) {
  if (!c.ok) return 'JSON 解析失败';
  const a = c.args || {};
  if (c.name === 'bash') return String(a.command ?? '');
  if (c.name === 'edit_file') return `${normPath(a.path)}  “${String(a.old_string ?? '').split('\n')[0].trim().slice(0, 40)}” → …`;
  return normPath(a.path);
}

export function callResult(c) {
  if (!c.info?.ok) return { cls: 'bad', text: c.result.split('\n')[0] };
  if (c.name === 'bash') {
    const { body, code } = splitExit(c.result);
    const n = body ? body.split('\n').length : 0;
    return { cls: code === 0 ? 'ok' : 'bad', text: `退出码 ${code}${n ? ` · ${n} 行输出` : ''}` };
  }
  if (c.name === 'read_file') return { cls: '', text: `读到 ${c.result.split('\n').length} 行` };
  return { cls: 'ok', text: c.result.split('\n')[0] };
}

export class Chat {
  constructor(M, h) {
    this.M = M;
    this.h = h;
    this.log = $('#log');
    this.inputEl = $('#cmpInput');
    this.candsEl = $('#cands');
    this.sendBtn = $('#btnSend');
    this.busy = false;
    this.buildTrie();
    this.chosen = [];
    this.node = this.trie;
    this.renderIntro();
    this.renderInput();
    this.renderCands();
    $('#btnBack').addEventListener('click', () => this.pop());
    this.sendBtn.addEventListener('click', () => this.send());
    $('#btnPlus').addEventListener('click', () => h.onPlus?.());
    addEventListener('keydown', (e) => {
      if (e.target.closest('input, textarea')) return;
      if (e.key === 'Backspace' && !document.body.classList.contains('mode-inspect')) { e.preventDefault(); this.pop(); }
      if (e.key === 'Enter' && !this.sendBtn.disabled) { e.preventDefault(); this.send(); }
    });
  }

  buildTrie() {
    this.trie = { kids: new Map(), end: null };
    for (const q of this.M.tasks) {
      let n = this.trie;
      for (const c of q.chips) {
        if (!n.kids.has(ckey(c))) n.kids.set(ckey(c), { chip: c, kids: new Map(), end: null });
        n = n.kids.get(ckey(c));
      }
      n.end = q;
    }
  }

  renderIntro() {
    const m = this.M.model;
    const el = document.createElement('div');
    el.className = 'intro-card';
    el.innerHTML = `
      <h1>一个 agent 是怎么干活的</h1>
      像 Claude Code 这样的编程 agent，本质是同一个模式：<b>模型 + 工具 + 循环</b>。模型只会写字；外面的程序把它写出的“工具调用”拿去真的执行，再把结果喂回给它，一圈一圈，直到它说“做完了”。<br>
      这里没有用任何商业模型。下面每一步都是真实录下来的：开源模型 <b>${esc(m.name)}</b> 在一个真实的 Linux 沙箱里干活（没有网络），命令输出、文件改动、每个词元的概率，都来自那次运行。
      <span class="chips-hint">用下方的词元拼出一个任务，或者直接点一张任务卡片。看它干完，再点 <span class="key">＋</span> 一层层揭开。</span>
      <div class="tasks">${this.M.tasks.map((q, i) => `
        <button type="button" class="task-card" data-q="${i}">
          <span class="ico" aria-hidden="true">${ICON[q.icon] || '•'}</span>
          <b>${esc(q.title)}</b>
          <span>${esc(q.blurb)}</span>
          <span class="meta">${q.turns} 轮 · ${q.calls} 次工具调用 · ${fmtNum(q.ctxEnd)} 词元</span>
        </button>`).join('')}</div>
      <div class="spec"><span><b>${(m.params / 1e8).toFixed(1)}</b> 亿参数</span><span><b>${m.layers}</b> 层</span><span>工具 <b>bash · read_file · write_file · edit_file</b></span></div>`;
    this.log.append(el);
    this.intro = el;
    $$('.task-card', el).forEach((b) => b.addEventListener('click', () => this.typeTask(this.M.tasks[Number(b.dataset.q)], true)));
  }

  /* ---------------- 输入法 ---------------- */

  renderInput() {
    this.inputEl.innerHTML = this.chosen.length
      ? this.chosen.map((c) => tokHTML(c.s, 'r-user')).join('') + '<span class="cur"></span>'
      : '<span class="ph">点下面的词元，拼出一个任务</span>';
    const ready = !!this.node.end && !this.busy;
    this.sendBtn.disabled = !ready;
    this.sendBtn.classList.toggle('glow', ready);
  }

  renderCands() {
    const kids = [...this.node.kids.values()];
    let html = `<span class="lab">${this.chosen.length ? '接下来' : '候选词元'}</span>`;
    html += kids.map((k, i) => `<button type="button" class="cand" data-k="${k.chip.ids.join(',')}" style="animation-delay:${i * 30}ms">${tokInner(k.chip.s)}<i>${k.chip.ids.join('+')}</i></button>`).join('');
    if (this.node.end && !kids.length) html += '<span class="cand done" aria-hidden="true">✓ 拼好了，按 ↑ 交给 agent</span>';
    this.candsEl.innerHTML = html;
    $$('.cand[data-k]', this.candsEl).forEach((b) => b.addEventListener('click', () => this.push(b.dataset.k)));
  }

  push(k) {
    const n = this.node.kids.get(k);
    if (!n) return;
    this.chosen.push(n.chip);
    this.node = n;
    this.renderInput();
    this.renderCands();
    window.__sfx?.tick();
  }

  pop() {
    if (!this.chosen.length) return;
    this.chosen.pop();
    this.node = this.trie;
    for (const c of this.chosen) this.node = this.node.kids.get(ckey(c));
    this.renderInput();
    this.renderCands();
  }

  async typeTask(q, autoSend = false) {
    if (this.typing) return;
    this.typing = true;
    this.chosen = [];
    this.node = this.trie;
    this.renderInput();
    for (const c of q.chips) {
      this.push(ckey(c));
      await sleep(70);
    }
    this.typing = false;
    if (autoSend) { await sleep(260); this.send(); }
  }

  send() {
    if (!this.node.end || this.busy) return;
    const q = this.node.end;
    this.chosen = [];
    this.node = this.trie;
    this.renderInput();
    this.renderCands();
    this.h.onSend?.(q);
  }

  setBusy(b) { this.busy = b; this.renderInput(); $('#avatar').classList.toggle('busy', b); }

  /* ---------------- 对话记录 ---------------- */

  startSession(q, R) {
    this.R = R;
    this.q = q;
    $$('.task-card', this.intro).forEach((b) => b.classList.toggle('cur', this.M.tasks[Number(b.dataset.q)] === q));
    this.log.querySelectorAll('.ag-session').forEach((s) => s.remove());
    const s = document.createElement('div');
    s.className = 'ag-session';
    s.style.cssText = 'display:flex;flex-direction:column;gap:14px';
    s.innerHTML = `<div class="ag-sep">— 新任务 · 沙箱已重置 —</div><div class="ag-user ag-msg">${md(q.prompt)}</div>`;
    this.log.append(s);
    this.sess = s;
    this.blocks = [];
    this.cursorKey = '';
    this.scroll();
  }

  // cursor: { t, say: 已显示的字数, shown: 已出现的调用数, done: 已完成的调用数, fin: 最终回答已说完 }
  render(cur) {
    if (!this.R) return;
    const R = this.R;
    while (this.blocks.length > cur.t + 1) this.blocks.pop().el.remove();
    for (let t = 0; t <= cur.t; t++) {
      if (!this.blocks[t]) this.blocks[t] = this.makeBlock(t);
      const full = t < cur.t;
      this.fillBlock(t, full ? { say: Infinity, shown: Infinity, done: Infinity, fin: true } : cur);
    }
    this.blocks.forEach((b, t) => b.el.classList.toggle('cur', t === cur.t && !cur.fin));
    this.scroll(true);
  }

  makeBlock(t) {
    const T = this.R.turns[t];
    const el = document.createElement('div');
    el.className = 'ag-turn ag-msg';
    el.dataset.n = t + 1;
    let h = '';
    if (T.final) h += '<div class="ag-final"></div>';
    else h += '<div class="ag-say"></div>';
    T.calls.forEach((c, ci) => {
      h += `<button type="button" class="ag-call" data-ci="${ci}" hidden><span class="c-h"><span class="c-name">${esc(c.name || '?')}</span><span class="c-arg">${esc(callArg(c))}</span></span><span class="c-r"></span></button>`;
    });
    el.innerHTML = h;
    el.querySelectorAll('.ag-call').forEach((b) => b.addEventListener('click', () => this.h.onSeek?.(t, Number(b.dataset.ci))));
    if (T.final) el.addEventListener('click', (e) => { if (!e.target.closest('button')) this.h.onSeek?.(t, -1); });
    this.sess.append(el);
    return { el, T, sayKey: '', state: '' };
  }

  fillBlock(t, cur) {
    const b = this.blocks[t], T = b.T;
    const say = T.say;
    const n = Math.max(0, Math.min(say.length, cur.say));
    const fin = T.final && cur.fin && n >= say.length;
    const key = `${n}|${fin}`;
    if (key !== b.sayKey) {
      b.sayKey = key;
      const box = b.el.querySelector(T.final ? '.ag-final' : '.ag-say');
      box.hidden = n <= 0;
      box.innerHTML = md(say.slice(0, n)) + (n < say.length && n > 0 ? '<span class="caret"></span>' : '') + (fin ? this.metaHTML() : '');
      box.querySelector('.peek')?.addEventListener('click', (e) => { e.stopPropagation(); this.h.onPlus?.(); });
    }
    const state = `${Math.min(cur.shown, 99)}|${Math.min(cur.done, 99)}`;
    if (state !== b.state) {
      b.state = state;
      b.el.querySelectorAll('.ag-call').forEach((el, ci) => {
        el.hidden = ci >= cur.shown;
        const run = ci < cur.shown && ci >= cur.done;
        el.classList.toggle('run', run);
        const r = el.querySelector('.c-r');
        if (run) { r.className = 'c-r'; r.textContent = '在沙箱里执行'; }
        else if (ci < cur.done) { const x = callResult(T.calls[ci]); r.className = `c-r ${x.cls}`; r.textContent = x.text; }
      });
    }
  }

  metaHTML() {
    const R = this.R, q = this.q;
    const calls = R.turns.reduce((a, t) => a + t.calls.length, 0);
    const failed = q.attempts.filter((a) => !a.ok);
    return `<div class="ag-meta">
        <button type="button" class="peek" title="单步进入：看 agent 内部怎么运转"><b>＋</b>揭开这次任务</button>
        <span class="stat">${R.T} 轮 · ${calls} 次工具调用 · 生成 ${fmtNum(R.genTotal)} 词元 · 上下文 ${fmtNum(R.maxCtx)} 词元</span>
      </div>
      <div class="ag-check" style="margin-top:8px"><b>${R.check.ok ? '✓ 独立检查通过' : '✗ 独立检查没通过'}</b>　${esc(R.check.detail)}</div>
      ${failed.length ? `<div class="ag-check tries" style="margin-top:6px">这是第 ${q.attempts.length} 次录制（随机种子 ${R.seed}）。前 ${failed.length} 次没通过检查：${failed.map((a) => esc(a.detail)).join('；')}。</div>` : `<div class="ag-check tries" style="margin-top:6px">第一次录制（随机种子 ${R.seed}）就通过了检查。</div>`}`;
  }

  scroll(soft = false) {
    const nearBottom = this.log.scrollHeight - this.log.scrollTop - this.log.clientHeight < 200;
    if (!soft || nearBottom) this.log.scrollTop = this.log.scrollHeight;
  }
}
