// 中英文切换（全站共用）。
// 语言怎么定：网址 ?lang=zh|en → 上次手动选的（localStorage）→ 浏览器语言（中文浏览器用中文，其余用英文）。
// 用法：
//   addDict({ zh: {...}, en: {...} })   页面注册自己的文案
//   t('key', { n: 3 })                  取当前语言的文案，{n} 这类占位会被替换；没有英文时回退中文
//   L('中文', 'English')                 一次性的短文案，直接写两种语言
//   applyDom(root)                      把 [data-i18n]（文本）/ [data-i18n-html] / [data-i18n-attr="title:key;aria-label:key2"] 换成当前语言
//   mountLangSwitch(el)                 在顶栏放一个“中 / EN”切换；切换时记住选择并带 ?lang= 重新载入
const KEY = 'blackbox:lang';

function detect() {
  try {
    const q = new URLSearchParams(location.search).get('lang');
    if (q === 'zh' || q === 'en') {
      try { localStorage.setItem(KEY, q); } catch {}
      return q;
    }
  } catch {}
  try {
    const s = localStorage.getItem(KEY);
    if (s === 'zh' || s === 'en') return s;
  } catch {}
  return /^zh\b/i.test(navigator.language || '') ? 'zh' : 'en';
}

export const lang = detect();
export const isEn = lang === 'en';
document.documentElement.lang = isEn ? 'en' : 'zh-CN';

const dict = {
  zh: { 'nav.infer': '推理', 'nav.train': '训练', 'nav.mm': '多模态', 'nav.agent': '智能体', 'nav.world': '世界模型', 'nav.learn': '学习', 'nav.label': '站点' },
  en: { 'nav.infer': 'Inference', 'nav.train': 'Training', 'nav.mm': 'Multimodal', 'nav.agent': 'Agents', 'nav.world': 'World Models', 'nav.learn': 'Learn', 'nav.label': 'Site' },
};

export function addDict(d) {
  Object.assign(dict.zh, d.zh || {});
  Object.assign(dict.en, d.en || {});
}

export function t(key, vars) {
  let s = dict[lang][key] ?? dict.zh[key] ?? key;
  if (vars) s = String(s).replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m));
  return s;
}

export const L = (zh, en) => (isEn ? en : zh);

export function applyDom(root = document) {
  document.documentElement.classList.remove('i18n-wait');   // 页面 <head> 里英文模式先藏起来的，这里换好文案后显示
  root.querySelectorAll('[data-i18n]').forEach((el) => { el.textContent = t(el.dataset.i18n); });
  root.querySelectorAll('[data-i18n-html]').forEach((el) => { el.innerHTML = t(el.dataset.i18nHtml); });
  root.querySelectorAll('[data-i18n-attr]').forEach((el) => {
    for (const pair of el.dataset.i18nAttr.split(';')) {
      const [a, k] = pair.split(':').map((s) => s.trim());
      if (a && k) el.setAttribute(a, t(k));
    }
  });
  scrollNavToCurrent();
}

// 手机上导航可以横向滑：文案换成英文（变长）、顶栏放进语言开关（变挤）之后，把当前页的链接滚回可见区。
// 下一帧再滚一次，兜住稍后才出现的顶栏元素（比如访问量胶囊）。
export function scrollNavToCurrent() {
  const go = () => {
    const nav = document.querySelector('.site-nav'), cur = nav && nav.querySelector('[aria-current]');
    if (!cur || nav.scrollWidth <= nav.clientWidth) return;
    const c = cur.getBoundingClientRect(), n = nav.getBoundingClientRect();   // 按导航自己的坐标算（offsetLeft 是相对顶栏的，会把品牌宽度算进去）
    nav.scrollLeft += c.left - n.left - (n.width - c.width) / 2;
  };
  go();
  requestAnimationFrame(go);
}

export function setLang(l) {
  try { localStorage.setItem(KEY, l); } catch {}
  const u = new URL(location.href);
  u.searchParams.set('lang', l);
  location.replace(u.toString());
}

export function mountLangSwitch(el) {
  if (!el) return;
  el.classList.add('lang-switch');
  el.setAttribute('role', 'group');
  el.setAttribute('aria-label', isEn ? 'Language' : '语言');
  el.title = isEn ? '切换到中文 / Language' : 'Switch to English / 语言';
  // 宽屏写全称「中文 | English」，窄屏（app.css ≤560px）只留「中 | EN」
  el.innerHTML = `<svg class="ls-globe" viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M3 12h18M12 3c2.6 2.8 3.9 5.8 3.9 9s-1.3 6.2-3.9 9c-2.6-2.8-3.9-5.8-3.9-9S9.4 5.8 12 3z" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>`
    + `<button type="button" data-l="zh" aria-pressed="${!isEn}" lang="zh-CN"><span class="ls-long">中文</span><span class="ls-short">中</span></button>`
    + `<button type="button" data-l="en" aria-pressed="${isEn}" lang="en"><span class="ls-long">English</span><span class="ls-short">EN</span></button>`;
  el.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-l]');
    if (b && b.dataset.l !== lang) setLang(b.dataset.l);
  });
  scrollNavToCurrent();
}
