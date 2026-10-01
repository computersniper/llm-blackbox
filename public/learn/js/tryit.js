// 动手试试：左边选片段，右边是完整代码（带一键复制）、运行环境、实测输出和讲解。
// 代码文件就放在 snippets/ 里，按需取来；高亮是一个很小的手写词法着色，只认注释、字符串、数字、关键字和函数名。

import { $, $$, esc } from '../../js/ui.js';

const PY_KW = new Set('import from as for in with if elif else not and or is break continue def return None True False lambda while try except finally class pass yield'.split(' '));
const SH_CMD = new Set('git cd pip python cp export'.split(' '));

export function highlightCode(code, lang) {
  const re = lang === 'bash'
    ? /(#[^\n]*)|("(?:[^"\\\n]|\\.)*"|'[^'\n]*')|(--?[A-Za-z][\w-]*)|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][\w./-]*)/g
    : /(#[^\n]*)|([fFrRbB]?"(?:[^"\\\n]|\\.)*"|[fFrRbB]?'(?:[^'\\\n]|\\.)*')|(\b\d+(?:\.\d+)?(?:e-?\d+)?\b)|([A-Za-z_]\w*)(?=\()|([A-Za-z_]\w*)/g;
  let out = '', last = 0, m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index));
    last = re.lastIndex;
    const [tok, com, str, a, b, c] = m;
    let cls = '';
    if (com) cls = 'c';
    else if (str) cls = 's';
    else if (lang === 'bash') cls = a ? 'f' : b ? 'n' : SH_CMD.has(c) ? 'k' : '';
    else cls = a ? 'n' : b ? 'f' : PY_KW.has(c) ? 'k' : '';
    out += cls ? `<span class="${cls}">${esc(tok)}</span>` : esc(tok);
  }
  return out + esc(code.slice(last));
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // 非安全上下文或没有权限时退回老办法
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.cssText = 'position:fixed;left:-9999px;top:0;opacity:0';
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch { ok = false; }
    ta.remove();
    return ok;
  }
}

export class TryIt {
  constructor(root, SN) {
    this.root = root;
    this.items = SN.items;
    this.cache = new Map();
    this.cur = 0;
    const desc = $('#tryDesc');
    desc.insertAdjacentHTML('afterend', `<p class="tested">${esc(SN.tested)}</p>`);
    $('#tryTabs').innerHTML = this.items.map((s, i) => `<button class="try-tab" type="button" role="tab" id="tab-${esc(s.id)}" aria-controls="tryPanel" aria-selected="false" tabindex="-1" data-i="${i}"><span class="n">${esc(s.no)}</span><b>${esc(s.title)}</b><small>${esc(s.env.slice(0, 2).join(' · '))}</small></button>`).join('');
    this.bind();
    this.show(0, false);
  }

  bind() {
    const tabs = $('#tryTabs');
    tabs.addEventListener('click', (e) => { const b = e.target.closest('.try-tab'); if (b) this.show(Number(b.dataset.i), true); });
    tabs.addEventListener('keydown', (e) => {
      const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (!d) return;
      e.preventDefault();
      this.show((this.cur + d + this.items.length) % this.items.length, true);
    });
    $('#tryPanel').addEventListener('click', async (e) => {
      const b = e.target.closest('.copy');
      if (!b) return;
      const text = this.cache.get(b.dataset.path);
      if (text == null) return;
      const ok = await copyText(text);
      b.classList.remove('ok', 'err');
      b.classList.add(ok ? 'ok' : 'err');
      $('span', b).textContent = ok ? '已复制' : '复制失败，请手动选择';
      clearTimeout(b._t);
      b._t = setTimeout(() => { b.classList.remove('ok', 'err'); $('span', b).textContent = '复制'; }, 1800);
    });
  }

  // 片段 id 或序号
  select(id) {
    const i = this.items.findIndex((s) => s.id === id);
    if (i >= 0) this.show(i, false);
  }

  async load(path) {
    if (this.cache.has(path)) return this.cache.get(path);
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    const t = await r.text();
    this.cache.set(path, t);
    return t;
  }

  async show(i, focus) {
    this.cur = i;
    const s = this.items[i];
    $$('.try-tab', this.root).forEach((b, k) => { b.setAttribute('aria-selected', String(k === i)); b.tabIndex = k === i ? 0 : -1; });
    if (focus) $(`#tab-${s.id}`).focus();
    const panel = $('#tryPanel');
    panel.setAttribute('aria-labelledby', `tab-${s.id}`);
    const env = s.env.map((e) => `<span class="${/CPU/.test(e) ? 'cpu' : /pip |install/.test(e) ? 'mono' : ''}">${esc(e)}</span>`).join('');
    const boxes = s.files.map((f) => `<div class="codebox" data-path="${esc(f.path)}"><div class="codebox-h"><span class="fn">${esc(f.path.split('/').pop())}</span><span class="lang">${esc(f.lang)}</span><button class="copy" type="button" data-path="${esc(f.path)}" aria-label="复制 ${esc(f.path.split('/').pop())}"><svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><rect x="5" y="5" width="9" height="9" rx="2" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M11 5V3.5A1.5 1.5 0 0 0 9.5 2h-6A1.5 1.5 0 0 0 2 3.5v6A1.5 1.5 0 0 0 3.5 11H5" fill="none" stroke="currentColor" stroke-width="1.4"/></svg><span>复制</span></button></div><pre><code>正在载入…</code></pre></div>`).join('');
    panel.innerHTML = `<h3>${esc(s.no)} ${esc(s.title)}</h3>
      <p class="try-goal">${esc(s.goal)}</p>
      <div class="env">${env}<span class="time">⏱ ${esc(s.time)}</span></div>
      ${s.note ? `<p class="try-note">${esc(s.note)}</p>` : ''}
      ${boxes}
      <div class="out"><div class="out-h">实测输出</div><pre>${esc(s.output)}</pre></div>
      <p class="try-explain">${esc(s.explain)}</p>
      <div class="try-site">对照本站：<a class="site-chip" href="${esc(s.site.href)}">${esc(s.site.label)}</a></div>`;
    // 重新触发面板的浮现动画
    panel.style.animation = 'none'; void panel.offsetWidth; panel.style.animation = '';
    for (const f of s.files) {
      const box = $(`.codebox[data-path="${CSS.escape(f.path)}"] code`, panel);
      try {
        const t = await this.load(f.path);
        if (this.cur !== i) return;
        box.innerHTML = highlightCode(t.replace(/\n$/, ''), f.lang);
      } catch (e) {
        box.textContent = `代码载入失败：${e.message}`;
      }
    }
  }
}
