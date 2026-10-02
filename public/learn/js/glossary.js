// 术语表：按主题分组，或者按英文名 A–Z 排；可以搜索中文、英文和解释。
// 英文模式：标题用英文名，解释用 def_en；中文名不显示（缩写这类不含汉字、又和英文名不同的写法除外，比如 Q / K / V）。

import { $, $$, esc } from '../../js/ui.js';
import { highlight, searchTerms } from './library.js';
import { loadGlMode, saveGlMode } from './store.js';
import { isEn, L, tx } from './lang.js';

const CJK = /[\u3400-\u9fff]/;
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const letterOf = (t) => (t.en.match(/[A-Za-z]/)?.[0] || '#').toUpperCase();

export class Glossary {
  constructor(root, GL, hues) {
    this.root = root;
    this.terms = GL.terms;
    this.groups = GL.groups;
    this.hues = hues;
    this.groupName = new Map(this.groups.map((g) => [g.id, tx(g, 'name')]));
    this.mode = loadGlMode();
    this.q = '';
    this.bind();
    this.render();
  }

  termHTML(t, i, terms) {
    const where = t.where.map((w) => `<a class="site-chip" href="${esc(w.href)}">${esc(tx(w, 'label'))}</a>`).join('');
    const alt = !CJK.test(t.term) && t.term.toLowerCase() !== t.en.toLowerCase() ? t.term : '';
    const head = isEn
      ? `<dfn>${highlight(t.en, terms)}</dfn>${alt ? `<span class="en">${highlight(alt, terms)}</span>` : ''}`
      : `<dfn>${highlight(t.term, terms)}</dfn><span class="en">${highlight(t.en, terms)}</span>`;
    return `<div class="term" style="--h:${this.hues.get(t.group) ?? 186};--i:${Math.min(i, 30)}">
      <div class="term-h">${head}${this.mode === 'az' ? `<span class="g">${esc(this.groupName.get(t.group) || '')}</span>` : ''}</div>
      <p>${highlight(tx(t, 'def'), terms)}</p>
      <div class="where">${where}</div>
    </div>`;
  }

  render() {
    const terms = searchTerms(this.q);
    const hits = this.terms.filter((t) => {
      const h = (isEn ? `${t.en} ${t.term} ${t.def_en}` : `${t.term} ${t.en} ${t.def}`).toLowerCase();
      return terms.every((k) => h.includes(k));
    });
    let i = 0;
    let html = '';
    const rail = $('#azRail');
    if (this.mode === 'az') {
      const sorted = [...hits].sort((a, b) => a.en.localeCompare(b.en, 'en', { sensitivity: 'base' }));
      const by = new Map();
      sorted.forEach((t) => { const L = letterOf(t); if (!by.has(L)) by.set(L, []); by.get(L).push(t); });
      html = [...by].map(([L, ts]) => `<section class="gl-group" id="gl-${L}" aria-label="${L}">
        <div class="gl-group-h az"><h3>${L}</h3><i>${ts.length}</i></div>
        <div class="gl-grid">${ts.map((t) => this.termHTML(t, i++, terms)).join('')}</div></section>`).join('');
      rail.innerHTML = LETTERS.map((L) => (by.has(L) ? `<a href="#gl-${L}">${L}</a>` : `<span aria-hidden="true">${L}</span>`)).join('');
      rail.hidden = false;
    } else {
      html = this.groups.map((g) => {
        const ts = hits.filter((t) => t.group === g.id);
        if (!ts.length) return '';
        return `<section class="gl-group" aria-label="${esc(tx(g, 'name'))}">
          <div class="gl-group-h"><h3>${esc(tx(g, 'name'))}</h3><i>${ts.length}</i></div>
          <div class="gl-grid">${ts.map((t) => this.termHTML(t, i++, terms)).join('')}</div></section>`;
      }).join('');
      rail.hidden = true;
    }
    $('#glossary-list').innerHTML = html || L('<p class="empty">没有找到这个术语。试试英文名，或者换个说法。</p>', '<p class="empty">No such term here. Try another word for it.</p>');
    $$('.seg button', this.root).forEach((b) => b.setAttribute('aria-checked', String(b.dataset.mode === this.mode)));
  }

  bind() {
    $$('.seg button', this.root).forEach((b) => b.addEventListener('click', () => {
      if (this.mode === b.dataset.mode) return;
      this.mode = b.dataset.mode;
      saveGlMode(this.mode);
      this.render();
    }));
    let tm = 0;
    $('#glSearch').addEventListener('input', (e) => {
      clearTimeout(tm);
      tm = setTimeout(() => { this.q = e.target.value; this.render(); }, 100);
    });
  }
}
