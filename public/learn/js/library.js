// 资源库：按主题分组的卡片，主题 / 类型 / 难度 / 语言四组筛选加全文搜索。
// 每个筛选按钮上的数字是“在其余条件不变时，选它会剩几条”，选了会变成 0 条的按钮直接置灰。
// 英文模式：标题、作者、“为什么值得看”用 *_en 字段；语言默认筛成 English（中文资源点“Chinese”或“All”就能看到）。

import { $, esc } from '../../js/ui.js';
import { isEn, L, t, tx } from './lang.js';

const TYPE_ORDER = { intro: 0, viz: 1, code: 2, paper: 3 };
const LANGS = isEn ? [{ id: 'en', name: 'English' }, { id: 'zh', name: 'Chinese' }] : [{ id: 'zh', name: '中文' }, { id: 'en', name: '英文' }];
const DEFAULTS = () => ({ topic: 'all', type: 'all', level: 0, lang: isEn ? 'en' : 'all', q: '' });

// 把文字里命中的关键词包上 <mark>（先切段再转义，不会弄坏 HTML）
export function highlight(text, terms) {
  if (!terms.length) return esc(text);
  const low = text.toLowerCase();
  const marks = new Array(text.length).fill(false);
  for (const t of terms) {
    let i = low.indexOf(t);
    while (t && i >= 0) { for (let k = i; k < i + t.length; k++) marks[k] = true; i = low.indexOf(t, i + t.length); }
  }
  let out = '', on = false;
  for (let i = 0; i < text.length; i++) {
    if (marks[i] !== on) { out += marks[i] ? '<mark>' : '</mark>'; on = marks[i]; }
    out += esc(text[i]);
  }
  return out + (on ? '</mark>' : '');
}

export const searchTerms = (q) => q.trim().toLowerCase().split(/\s+/).filter(Boolean);

export class Library {
  constructor(root, RES, { stationOf, focusStation } = {}) {
    this.root = root;
    this.R = RES;
    this.items = RES.items;
    this.topics = RES.topics;
    this.types = RES.types;
    this.levels = (isEn && RES.levels_en) || RES.levels;
    this.platforms = (isEn && RES.platforms_en) || {};
    this.topicById = new Map(this.topics.map((t) => [t.id, t]));
    this.typeById = new Map(this.types.map((t) => [t.id, t]));
    this.stationOf = stationOf || (() => null);
    this.focusStation = focusStation || (() => {});
    this.f = DEFAULTS();
    const topicName = (x) => this.topicById.get(x.topic), typeName = (x) => this.typeById.get(x.type);
    this.hay = new Map(this.items.map((x) => [x.id, (isEn
      ? [tx(x, 'title'), tx(x, 'author'), tx(x, 'why'), this.plat(x), x.year, tx(topicName(x), 'name'), tx(typeName(x), 'name'), x.title, x.author]
      : [x.title, x.author, x.why, x.platform, x.year, topicName(x)?.name, typeName(x)?.name]).join(' ').toLowerCase()]));
    this.bind();
    this.render();
  }

  plat(x) {
    return this.platforms[x.platform] || x.platform;
  }

  match(x, skip) {
    const f = this.f;
    if (skip !== 'topic' && f.topic !== 'all' && x.topic !== f.topic) return false;
    if (skip !== 'type' && f.type !== 'all' && x.type !== f.type) return false;
    if (skip !== 'level' && f.level && x.level !== f.level) return false;
    if (skip !== 'lang' && f.lang !== 'all' && x.lang !== f.lang) return false;
    const terms = searchTerms(f.q);
    if (terms.length) { const h = this.hay.get(x.id); if (!terms.every((t) => h.includes(t))) return false; }
    return true;
  }

  chips(dim, opts, cur, label) {
    const pool = this.items.filter((x) => this.match(x, dim));
    const all = `<button class="chip" type="button" data-dim="${dim}" data-v="${dim === 'level' ? 0 : 'all'}" aria-pressed="${cur === 'all' || cur === 0}">${L('全部', 'All')}<i>${pool.length}</i></button>`;
    const btns = opts.map((o) => {
      const n = pool.filter((x) => String(x[dim]) === String(o.id)).length;
      const on = String(cur) === String(o.id);
      const hue = o.hue != null ? ` style="--h:${o.hue}"` : '';
      return `<button class="chip" type="button" data-dim="${dim}" data-v="${esc(o.id)}" aria-pressed="${on}"${hue}${!n && !on ? ' disabled' : ''}>${o.hue != null ? '<span class="d"></span>' : ''}${esc(tx(o, 'name'))}<i>${n}</i></button>`;
    }).join('');
    return `<span class="lab">${label}</span>${all}${btns}`;
  }

  renderChips() {
    $('#fTopic').innerHTML = this.chips('topic', this.topics, this.f.topic, t('lp.fTopic'));
    $('#fType').innerHTML = this.chips('type', this.types, this.f.type, t('lp.fType'));
    $('#fLevel').innerHTML = this.chips('level', [1, 2, 3].map((k) => ({ id: k, name: this.levels[k] })), this.f.level, t('lp.fLevel'));
    $('#fLang').innerHTML = this.chips('lang', LANGS, this.f.lang, t('lp.fLang'));
  }

  card(x, i, terms) {
    const t = this.topicById.get(x.topic);
    const st = this.stationOf(x.id);
    const lv = [1, 2, 3].map((k) => `<b class="${k <= x.level ? 'on' : ''}"></b>`).join('');
    const by = [tx(x, 'author'), x.year].filter(Boolean).map((s) => highlight(String(s), terms)).join(' · ');
    const badge = x.lang === 'zh' ? L('中文', 'ZH') : 'EN';
    const badgeTitle = isEn ? ` title="${x.lang === 'zh' ? 'In Chinese' : 'In English'}"` : '';
    // 英文模式下中文资源的标题是译名，原标题放在链接的悬停提示里
    const orig = isEn && x.lang === 'zh' ? ` title="${esc(x.title)}"` : '';
    return `<article class="res" style="--h:${t?.hue ?? 186};--i:${Math.min(i, 24)}">
      <div class="res-meta"><span class="pf${x.lang === 'zh' ? ' zh' : ''}"${badgeTitle}>${badge}</span><span>${esc(this.plat(x))}</span><span class="ty">${esc(tx(this.typeById.get(x.type), 'name') || '')}</span><span class="lv" title="${L('难度：', 'Level: ')}${esc(this.levels[x.level])}">${lv}<span>${esc(this.levels[x.level])}</span></span></div>
      <h4><a href="${esc(x.url)}" target="_blank" rel="noopener"${orig}>${highlight(tx(x, 'title'), terms)}</a></h4>
      ${by ? `<div class="by">${by}</div>` : ''}
      <p class="why">${highlight(tx(x, 'why'), terms)}</p>
      ${st ? `<a class="on-route" href="#n-${esc(st.id)}" data-station="${esc(st.id)}">${L(`↑ 路线 · 第 ${String(st.no).padStart(2, '0')} 站 ${esc(st.title)}`, `↑ Route · Stop ${String(st.no).padStart(2, '0')} ${esc(tx(st, 'title'))}`)}</a>` : ''}
    </article>`;
  }

  render() {
    this.renderChips();
    const terms = searchTerms(this.f.q);
    const hits = this.items.filter((x) => this.match(x));
    // 英文模式选“All”时，同一组里英文资源排在中文前面
    const langKey = (x) => (isEn && x.lang !== 'en' ? 1 : 0);
    const groups = this.topics.map((t) => ({ t, xs: hits.filter((x) => x.topic === t.id).sort((a, b) => langKey(a) - langKey(b) || a.level - b.level || TYPE_ORDER[a.type] - TYPE_ORDER[b.type]) })).filter((g) => g.xs.length);
    let i = 0;
    $('#library-list').innerHTML = groups.length
      ? groups.map((g) => `<section class="lib-group" style="--h:${g.t.hue}" aria-label="${esc(tx(g.t, 'name'))}">
          <div class="lib-group-h"><span class="d"></span><h3>${esc(tx(g.t, 'name'))}</h3><i>${L(`${g.xs.length} 条`, `${g.xs.length}`)}</i></div>
          <div class="lib-grid">${g.xs.map((x) => this.card(x, i++, terms)).join('')}</div>
        </section>`).join('')
      : L(`<p class="empty">没有符合条件的资料。换个关键词，或者<button class="linkish" type="button" data-clear>清除筛选</button>。</p>`,
        `<p class="empty">Nothing matches. Try another keyword, or <button class="linkish" type="button" data-clear>clear the filters</button>.</p>`);
    const d = DEFAULTS();
    const active = this.f.topic !== d.topic || this.f.type !== d.type || this.f.level !== d.level || this.f.lang !== d.lang || this.f.q.trim();
    $('#libCount').innerHTML = L(`显示 <b>${hits.length}</b> / ${this.items.length} 条`, `Showing <b>${hits.length}</b> of ${this.items.length}`);
    $('#libClear').hidden = !active;
  }

  bind() {
    $('#libCtl').addEventListener('click', (e) => {
      const b = e.target.closest('.chip');
      if (!b || b.disabled) return;
      const dim = b.dataset.dim;
      let v = b.dataset.v;
      if (dim === 'level') v = Number(v);
      // 再点一次已选中的按钮 = 取消
      this.f[dim] = this.f[dim] === v && v !== 'all' && v !== 0 ? (dim === 'level' ? 0 : 'all') : v;
      this.render();
    });
    let tm = 0;
    $('#libSearch').addEventListener('input', (e) => {
      clearTimeout(tm);
      tm = setTimeout(() => { this.f.q = e.target.value; this.render(); }, 120);
    });
    const clear = () => {
      this.f = DEFAULTS();
      $('#libSearch').value = '';
      this.render();
    };
    $('#libClear').addEventListener('click', clear);
    this.root.addEventListener('click', (e) => {
      if (e.target.closest('[data-clear]')) clear();
      const st = e.target.closest('[data-station]');
      if (st) { e.preventDefault(); this.focusStation(st.dataset.station); }
    });
  }
}
