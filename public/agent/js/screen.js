// D1 屏幕：沙箱里的文件树、编辑器、终端。全部按录下来的真实工具调用回放：
// 命令和输出来自沙箱里的真实执行，文件内容和 diff 来自每次工具调用前后的真实快照。
import { $, esc } from '../../js/ui.js';

const PS1 = '<span class="pr">agent@sandbox<i>:/work</i>$</span> ';

export const normPath = (p) => String(p || '').trim().replace(/^\/work\/?/, '').replace(/^\.\//, '');

// 工具输出拆成：正文 + 退出码（harness 在末尾加了一行 [退出码 N]）
export function splitExit(result) {
  const m = result.match(/\n?\[退出码 (\d+)\]$/);
  if (!m) return { body: result, code: null };
  return { body: result.slice(0, m.index), code: Number(m[1]) };
}

// 一行 shell 输出的颜色
function outCls(line) {
  if (/^(OK|\.+)$|^OK |passed/.test(line)) return 'o-ok';
  if (/Error|FAIL|Traceback|not found|^error|^E+\.?$|^\.?F|×/.test(line)) return 'o-bad';
  return '';
}

function verdict(body, code) {
  const m = body.match(/Ran (\d+) tests?/);
  if (!m) return '';
  return code === 0 ? `<span class="verdict ok">✓ ${m[1]} 个测试全部通过</span>` : `<span class="verdict bad">✗ 测试没通过（共 ${m[1]} 个）</span>`;
}

// 极简语法高亮（Python 为主）
const KW = /\b(def|return|if|elif|else|for|while|in|import|from|class|with|as|not|and|or|None|True|False|try|except|lambda|print)\b/;
function hl(line, ext) {
  if (ext !== 'py') return esc(line);
  const re = /(#.*$)|("""[^]*?"""|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|f"(?:[^"\\]|\\.)*")|(\b\d+(?:\.\d+)?\b)|([A-Za-z_]\w*)(?=\()|([A-Za-z_]\w*)/g;
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

// 行级 diff（LCS），文件都很小
export function lineDiff(a, b) {
  const A = a == null ? [] : a.replace(/\n$/, '').split('\n');
  const B = b == null ? [] : b.replace(/\n$/, '').split('\n');
  const n = A.length, m = B.length;
  const L = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const ops = [];
  let i = 0, j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && A[i] === B[j]) { ops.push({ op: 'eq', text: A[i], na: i + 1, nb: j + 1 }); i++; j++; }
    else if (i < n && (j >= m || L[i + 1][j] >= L[i][j + 1])) { ops.push({ op: 'del', text: A[i], na: i + 1 }); i++; }
    else { ops.push({ op: 'add', text: B[j], nb: j + 1 }); j++; }
  }
  return ops;
}

const extOf = (p) => (p.match(/\.(\w+)$/) || [])[1] || '';

export class Screen {
  constructor() {
    this.tree = $('#tree');
    this.tabs = $('#tabs');
    this.ed = $('#edBody');
    this.term = $('#termBody');
    this.termSt = $('#termSt');
    this.status = $('#winStatus');
    this.done = $('#deskDone');
    this.R = null;
    this.key = '';
    this.idle();
  }

  idle() {
    this.R = null;
    this.key = '';
    this.tree.innerHTML = '<div class="th">/work</div>';
    this.tabs.innerHTML = '';
    this.ed.innerHTML = '<div class="ed-empty"><span>还没有任务</span><span>在左边选一个任务，agent 会在这里打开、修改文件</span></div>';
    this.term.innerHTML = `${PS1}<span class="cur"></span>`;
    this.termSt.textContent = '';
    this.termSt.className = 'term-st';
    this.status.innerHTML = '<span>等待任务</span>';
    this.done.hidden = true;
  }

  load(R) {
    this.R = R;
    this.key = '';
    // 所有工具调用摊平成一条时间线
    this.acts = [];
    R.turns.forEach((T, t) => T.calls.forEach((c, ci) => this.acts.push({ t, ci, c })));
  }

  // 第 t 轮第 ci 个调用在摊平后的序号
  actIndex(t, ci) { return this.acts.findIndex((a) => a.t === t && a.ci === ci); }

  // k = 已经做完的调用数；cur = 正在做的调用（-1 表示没有）；p = 进度；final = 已给出最终回答
  show(k, cur, p, final = false) {
    if (!this.R) return;
    const key = `${k}|${cur}|${final}`;
    if (key !== this.key) {
      this.key = key;
      this.renderStatic(k, cur, final);
    }
    if (cur >= 0) this.renderPartial(this.acts[cur], p);
  }

  filesAt(k) { return k > 0 ? this.acts[k - 1].c.filesAfter : this.R.files0; }

  renderStatic(k, cur, final) {
    const R = this.R;
    this.treeAfter = false;
    // 终端：之前所有 bash 调用的完整记录
    let h = '';
    for (let i = 0; i < k; i++) if (this.acts[i].c.name === 'bash') h += this.bashHTML(this.acts[i].c, 1);
    this.termStatic = h;
    if (cur < 0 || this.acts[cur].c.name !== 'bash') {
      this.term.innerHTML = h + `${PS1}<span class="cur"></span>`;
      this.term.scrollTop = this.term.scrollHeight;
    }
    const lastBash = [...this.acts.slice(0, k)].reverse().find((a) => a.c.name === 'bash');
    this.setTermSt(lastBash ? splitExit(lastBash.c.result).code : null);
    // 编辑器：最近一次碰过文件的调用
    this.opened = [];
    let lastEd = null;
    for (let i = 0; i < k; i++) {
      const a = this.acts[i];
      const path = this.edPath(a.c);
      if (path) { lastEd = a; this.opened = this.opened.filter((x) => x !== path).concat(path); }
    }
    const curAct = cur >= 0 ? this.acts[cur] : null;
    const curPath = curAct ? this.edPath(curAct.c) : null;
    if (curPath) this.opened = this.opened.filter((x) => x !== curPath).concat(curPath);
    // 文件树
    this.renderTree(curAct ? curAct.c.filesBefore : this.filesAt(k), curPath || (lastEd && this.edPath(lastEd.c)));
    this.renderTabs(curPath || (lastEd && this.edPath(lastEd.c)));
    if (!curPath) {
      if (lastEd) this.renderEditor(lastEd.c, 1);
      else this.ed.innerHTML = '<div class="ed-empty"><span>编辑器</span><span>agent 读写文件时会在这里打开</span></div>';
    }
    // 状态栏
    if (!curAct) {
      const prev = k > 0 ? this.acts[k - 1].c : null;
      this.status.innerHTML = prev ? this.statusDone(prev) : '<span>沙箱已就绪 · 只读系统目录 · 只能写 /work · 无网络</span>';
    }
    this.done.hidden = !final;
    if (final) this.done.innerHTML = `<b>${R.check.ok ? '✓ 任务完成' : '任务结束'}</b>独立检查：${esc(R.check.detail)}`;
  }

  setTermSt(code) {
    this.termSt.className = `term-st${code == null ? '' : code === 0 ? ' ok' : ' bad'}`;
    this.termSt.textContent = code == null ? '' : `上一条命令 · 退出码 ${code}`;
  }

  edPath(c) {
    if (c.name === 'read_file' || c.name === 'write_file' || c.name === 'edit_file') return normPath(c.args?.path);
    if (c.name === 'bash' && c.diffs.length) return c.diffs[0].path;
    return null;
  }

  bashHTML(c, p) {
    const cmd = String(c.args?.command ?? '');
    const { body, code } = splitExit(c.result);
    const tp = Math.min(1, p / 0.3);
    let h = PS1 + `<span class="cmd">${esc(cmd.slice(0, Math.ceil(cmd.length * tp)))}</span>`;
    if (p < 0.33) return h + '<span class="cur"></span>';
    h += '\n';
    const lines = body ? body.split('\n') : [];
    const op = Math.min(1, (p - 0.33) / 0.55);
    const nShow = Math.ceil(lines.length * op);
    h += lines.slice(0, nShow).map((l) => { const k = outCls(l); return k ? `<span class="${k}">${esc(l)}</span>` : esc(l); }).join('\n');
    if (nShow) h += '\n';
    if (p >= 0.9) h += `<span class="ex ${code === 0 ? 'ok' : 'bad'}">退出码 ${code}${c.info?.timeout ? ' · 超时' : ''}</span>\n` + verdict(body, code);
    return h;
  }

  renderPartial(a, p) {
    const c = a.c;
    if (c.name === 'bash') {
      this.term.innerHTML = this.termStatic + this.bashHTML(c, p);
      this.term.scrollTop = this.term.scrollHeight;
      if (p >= 0.9) this.setTermSt(splitExit(c.result).code);
    }
    if (this.edPath(c)) this.renderEditor(c, p);
    if (p >= 0.6 && c.diffs.length && !this.treeAfter) { this.treeAfter = true; this.renderTree(c.filesAfter, this.edPath(c), true); }
    if (p < 0.6 && this.treeAfter) { this.treeAfter = false; this.renderTree(c.filesBefore, this.edPath(c)); }
    const arg = c.name === 'bash' ? c.args?.command : c.args?.path;
    this.status.innerHTML = p < 0.9
      ? `<span class="op">▶ ${esc(c.name || '?')}</span><span>${esc(String(arg ?? '').split('\n')[0].slice(0, 90))}</span>`
      : this.statusDone(c);
  }

  statusDone(c) {
    const ok = c.info?.ok !== false && !(c.name === 'bash' && splitExit(c.result).code !== 0);
    const ms = c.info?.ms != null ? ` · 实测 ${c.info.ms} ms` : '';
    const what = c.name === 'bash' ? `退出码 ${splitExit(c.result).code}` : c.result.split('\n')[0].slice(0, 80);
    const label = c.name === 'read_file' && c.info?.ok ? `读取了 ${c.result.split('\n').length} 行` : what;
    return `<span class="${ok ? 'ok' : 'bad'}">${ok ? '✓' : '✗'} ${esc(c.name || '?')}</span><span>${esc(label)}${ms}</span>`;
  }

  renderTabs(active) {
    this.tabs.innerHTML = this.opened.slice(-4).map((p) => {
      const ch = this.R.files0[p] !== this.lastFiles?.[p];
      return `<span class="tab ${p === active ? 'on' : ''}">${ch ? '<i class="d"></i>' : ''}${esc(p.split('/').pop())}</span>`;
    }).join('');
  }

  renderTree(files, active, flash = false) {
    this.lastFiles = files;
    const f0 = this.R.files0;
    const paths = Object.keys(files).sort();
    let h = '<div class="th">/work</div>';
    const seen = new Set();
    for (const p of paths) {
      const parts = p.split('/');
      for (let d = 1; d < parts.length; d++) {
        const dir = parts.slice(0, d).join('/');
        if (!seen.has(dir)) { seen.add(dir); h += `<div class="f dir" style="padding-left:${12 + (d - 1) * 12}px">▾ ${esc(parts[d - 1])}/</div>`; }
      }
      const b = !(p in f0) ? '<span class="b n">+</span>' : f0[p] !== files[p] ? '<span class="b m">M</span>' : '';
      h += `<div class="f ${p === active ? 'on' : ''} ${flash && p === active ? 'flash' : ''}" style="padding-left:${12 + (parts.length - 1) * 12}px">${esc(parts[parts.length - 1])}${b}</div>`;
    }
    this.tree.innerHTML = h;
  }

  renderEditor(c, p) {
    const path = this.edPath(c);
    const ext = extOf(path);
    const ok = c.info?.ok !== false;
    if (!ok) {
      const before = c.filesBefore[path];
      this.ed.innerHTML = `<div class="ed-err">✗ ${esc(c.result)}</div>` + (before != null ? this.plain(before, ext, -1) : '');
      return;
    }
    const d = c.diffs.find((x) => x.path === path);
    if (c.name === 'read_file' || !d) {
      const text = c.filesBefore[path] ?? '';
      const n = text.replace(/\n$/, '').split('\n').length;
      this.ed.innerHTML = this.plain(text, ext, p < 1 ? Math.floor(p * n * 1.1) : -1);
      if (p < 1) this.follow();
      return;
    }
    // 改动：删掉的行标红，新写的行一个字一个字地打出来
    const ops = lineDiff(d.before, d.after);
    const adds = ops.filter((o) => o.op === 'add');
    const total = adds.reduce((a, o) => a + o.text.length + 1, 0) || 1;
    const tp = Math.max(0, Math.min(1, (p - 0.2) / 0.6));
    let budget = Math.ceil(total * tp);
    let h = '', typing = false;
    for (const o of ops) {
      if (o.op === 'eq') h += `<div class="ln"><span class="no">${o.nb}</span><span class="tx">${hl(o.text, ext)}</span></div>`;
      else if (o.op === 'del') h += `<div class="ln del"><span class="no">−</span><span class="tx">${esc(o.text)}</span></div>`;
      else {
        if (budget <= 0) continue;
        const take = Math.min(o.text.length, budget);
        budget -= o.text.length + 1;
        const partial = take < o.text.length || (budget <= 0 && tp < 1);
        if (partial) typing = true;
        h += `<div class="ln add ${partial ? 'typing' : ''}"><span class="no">${o.nb}</span><span class="tx">${partial ? esc(o.text.slice(0, take)) : hl(o.text, ext)}</span></div>`;
      }
    }
    this.ed.innerHTML = h;
    if (p < 1 || typing) this.follow(typing ? '.ln.typing' : '.ln.del, .ln.add');
  }

  plain(text, ext, scan) {
    const lines = text.replace(/\n$/, '').split('\n');
    return lines.map((l, i) => `<div class="ln ${i === scan ? 'scan' : ''}"><span class="no">${i + 1}</span><span class="tx">${hl(l, ext)}</span></div>`).join('');
  }

  follow(sel = '.ln.scan') {
    const el = this.ed.querySelector(sel);
    if (!el) return;
    const top = el.offsetTop - this.ed.clientHeight * 0.4;
    if (Math.abs(this.ed.scrollTop - top) > 4) this.ed.scrollTop = top;
  }
}
