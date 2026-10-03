// 学习页的中英文文案。机制在 ../../js/i18n.js；这里只登记本页的字典，再给数据里的 *_en 字段一个取值的小工具。
// 中文的值必须和 index.html / 原来代码里的文字一字不差：中文模式要和加英文之前完全一样。

import { addDict, applyDom, isEn, mountLangSwitch, t } from '../../js/i18n.js';

export { isEn, t, L } from '../../js/i18n.js';

// 数据字段：英文模式下有 xxx_en 就用它，否则用中文原文
export const tx = (o, k) => (isEn && o && o[`${k}_en`] != null ? o[`${k}_en`] : o?.[k]);

addDict({
  zh: {
    'lp.title': '延伸学习 · 揭开黑箱',
    'lp.desc': '看完揭开黑箱，接着往下学：一条从“大模型是什么”到“读论文、自己动手”的学习路线，80 条逐个核对过的中英文资料，54 个术语的白话解释，以及 6 段能在自己电脑上跑起来的代码。',
    'lp.brandLabel': '揭开黑箱 · 延伸学习',
    'lp.brand': '延伸学习',
    'lp.jump': '本页',
    'lp.j.route': '路线图', 'lp.j.lib': '资源库', 'lp.j.gl': '术语表', 'lp.j.try': '动手试试',
    'lp.progTitle': '学习路线的进度（存在这台设备的浏览器里）',
    'lp.progTitleN': '学习路线：已学完 {k} / {n} 站（进度存在这台设备的浏览器里）',
    'lp.heroEyebrow': 'LEARN · 延伸学习',
    'lp.heroTitle': '从看懂，到动手',
    'lp.lede': '这个网站把真实的模型一层层拆开给你看。看完想接着往下学，就从这里出发：一条从“大模型是什么”走到“读论文、自己动手”的路线，一份逐条核对过的资料清单，一张白话术语表，和几段能在自己电脑上跑起来的代码。',
    'lp.gauge': '路线的六段',
    'lp.routeEyebrow': 'ROUTE · 学习路线',
    'lp.routeTitle': '一条往下走的路',
    'lp.routeDesc': '从水面一直走到深处。每一站写着这一步要弄懂什么、在本站哪一页的哪一层能亲眼看到它，再挂几份最值得看的资料。学完一站就点一下“学完了”，进度只存在这台设备的浏览器里。',
    'lp.reset': '清空进度',
    'lp.resetConfirm': '再点一次，清空全部进度',
    'lp.loadingRoute': '正在载入路线…',
    'lp.libEyebrow': 'LIBRARY · 资源库',
    'lp.libTitle': '按主题找资料',
    'lp.libSearchPh': '搜索标题、作者、关键词…',
    'lp.libSearch': '搜索资源',
    'lp.fTopic': '主题', 'lp.fType': '类型', 'lp.fLevel': '难度', 'lp.fLang': '语言',
    'lp.clear': '清除筛选',
    'lp.glEyebrow': 'GLOSSARY · 术语表',
    'lp.glTitle': '术语，说人话',
    'lp.glDesc': '每个词一两句白话，后面标着在本站哪里能亲眼看到它。数字都是按本站用的真实模型写的。',
    'lp.glSearchPh': '搜索术语，中英文都行…',
    'lp.glSearch': '搜索术语',
    'lp.glSort': '排列方式',
    'lp.glByGroup': '按主题',
    'lp.azRail': '字母索引',
    'lp.tryEyebrow': 'HANDS-ON · 动手试试',
    'lp.tryTitle': '在自己电脑上跑起来',
    'lp.tryDesc': '每一段都是能直接复制运行的完整代码，下面的输出是原样跑出来的。',
    'lp.tryTabs': '代码片段',
    'lp.foot': '资料清单由本站维护，每条外部链接都用 <code>tools/learn/check_links.py</code> 逐个打开检查过；外面的网页随时可能改版或下线，以打开时看到的内容为准。',
    'lp.top': '回到顶部 ↑',
    'lp.loadFail': '数据载入失败：{msg}',
  },
  en: {
    'lp.title': 'Keep Learning · Opening the Black Box',
    'lp.desc': 'Finished opening the black box? Keep going: a learning route from "what is a large language model?" to reading papers and building things yourself, a checked list of resources, a plain-English glossary of 54 terms, and 6 code snippets you can run on your own computer.',
    'lp.brandLabel': 'Opening the Black Box · Keep Learning',
    'lp.brand': 'Keep Learning',
    'lp.jump': 'On this page',
    'lp.j.route': 'Route', 'lp.j.lib': 'Library', 'lp.j.gl': 'Glossary', 'lp.j.try': 'Hands-on',
    'lp.progTitle': 'Progress along the learning route (saved in this browser)',
    'lp.progTitleN': 'Learning route: {k} / {n} stops done (progress is saved in this browser)',
    'lp.heroEyebrow': 'LEARN · KEEP GOING',
    'lp.heroTitle': 'From seeing it to doing it',
    'lp.lede': 'This site takes real models apart, layer by layer. When you want to keep going, start here: a route from "what is a large language model?" all the way to reading papers and building things yourself, a resource list checked item by item, a plain-English glossary, and a few code snippets you can run on your own computer.',
    'lp.gauge': 'The six stages of the route',
    'lp.routeEyebrow': 'ROUTE · LEARNING PATH',
    'lp.routeTitle': 'A path that goes deeper',
    'lp.routeDesc': 'From the surface down to the depths. Each stop says what to understand at this step and where on this site — which page, which layer — you can see it for yourself, plus a few of the best resources. Mark a stop as done when you finish it; progress is saved only in this browser.',
    'lp.reset': 'Reset progress',
    'lp.resetConfirm': 'Click again to clear all progress',
    'lp.loadingRoute': 'Loading the route…',
    'lp.libEyebrow': 'LIBRARY · RESOURCES',
    'lp.libTitle': 'Find resources by topic',
    'lp.libSearchPh': 'Search titles, authors, keywords…',
    'lp.libSearch': 'Search resources',
    'lp.fTopic': 'Topic', 'lp.fType': 'Type', 'lp.fLevel': 'Level', 'lp.fLang': 'Language',
    'lp.clear': 'Clear filters',
    'lp.glEyebrow': 'GLOSSARY · TERMS',
    'lp.glTitle': 'Jargon, in plain words',
    'lp.glDesc': 'One or two plain sentences per term, followed by where on this site you can see it for yourself. The numbers are those of the real models this site uses.',
    'lp.glSearchPh': 'Search terms…',
    'lp.glSearch': 'Search terms',
    'lp.glSort': 'Sort by',
    'lp.glByGroup': 'By topic',
    'lp.azRail': 'Alphabetical index',
    'lp.tryEyebrow': 'HANDS-ON · RUN IT YOURSELF',
    'lp.tryTitle': 'Run it on your own computer',
    'lp.tryDesc': 'Each snippet is complete code you can copy and run as is; the output under it is exactly what it printed.',
    'lp.tryTabs': 'Code snippets',
    'lp.foot': 'This list is maintained by the site, and every external link was opened and checked one by one with <code>tools/learn/check_links.py</code>. Outside pages can change or go offline at any time — what you see when you open them is what counts.',
    'lp.top': 'Back to top ↑',
    'lp.loadFail': "Couldn't load the data: {msg}",
  },
});

// 启动：页面上带 data-i18n 的静态文字、标题和描述、顶栏的中 / EN 开关
export function initLang() {
  applyDom();
  document.title = t('lp.title');
  document.querySelector('meta[name="description"]')?.setAttribute('content', t('lp.desc'));
  mountLangSwitch(document.getElementById('langSwitch'));
}
