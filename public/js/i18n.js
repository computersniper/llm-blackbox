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
  root.querySelectorAll('[data-i18n]').forEach((el) => { el.textContent = t(el.dataset.i18n); });
  root.querySelectorAll('[data-i18n-html]').forEach((el) => { el.innerHTML = t(el.dataset.i18nHtml); });
  root.querySelectorAll('[data-i18n-attr]').forEach((el) => {
    for (const pair of el.dataset.i18nAttr.split(';')) {
      const [a, k] = pair.split(':').map((s) => s.trim());
      if (a && k) el.setAttribute(a, t(k));
    }
  });
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
  el.innerHTML = `<button type="button" data-l="zh" aria-pressed="${!isEn}" lang="zh-CN">中</button><button type="button" data-l="en" aria-pressed="${isEn}" lang="en">EN</button>`;
  el.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-l]');
    if (b && b.dataset.l !== lang) setLang(b.dataset.l);
  });
}
