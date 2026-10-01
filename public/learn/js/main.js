// 延伸学习页：载入 data/*.json，搭起路线图、资源库、术语表和动手试试四块。

import { $, $$, esc } from '../../js/ui.js';
import { Background } from '../../js/bg.js';
import { Roadmap } from './roadmap.js';
import { Library } from './library.js';
import { Glossary } from './glossary.js';
import { TryIt } from './tryit.js';

const getJSON = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`); return r.json(); });

// 背景：沿用其他页面的“海雪”粒子；往下滚得越深，底色越往深处偏（和揭开一层层的意思一样）
const bg = new Background($('#bg'));
let depthQ = 0;
addEventListener('scroll', () => {
  if (depthQ) return;
  depthQ = requestAnimationFrame(() => {
    depthQ = 0;
    const max = document.documentElement.scrollHeight - innerHeight;
    bg.setDepth(Math.round(Math.min(1, scrollY / Math.max(1, max)) * 6));
  });
}, { passive: true });

function setProgress(k, n) {
  const p = n ? k / n : 0;
  const top = $('#topProg');
  top.style.setProperty('--p', p);
  $('b', top).textContent = k;
  $('.lp-prog-txt', top).lastChild.textContent = `/${n}`;
  top.title = `学习路线：已学完 ${k} / ${n} 站（进度存在这台设备的浏览器里）`;
  $('.rm-bar i').style.width = `${p * 100}%`;
  $('.rm-count').textContent = `${k} / ${n}`;
}

function heroStats(RM, RES, GL, SN) {
  const nodes = RM.stages.reduce((a, s) => a + s.nodes.length, 0);
  const zh = RES.items.filter((x) => x.lang === 'zh').length;
  $('#heroStats').innerHTML = `
    <div><dt>路线</dt><dd><b>${RM.stages.length}</b> 段 <b>${nodes}</b> 站</dd></div>
    <div><dt>资源</dt><dd><b>${RES.items.length}</b> 条，其中中文 <b>${zh}</b> 条</dd></div>
    <div><dt>术语</dt><dd><b>${GL.terms.length}</b> 个</dd></div>
    <div><dt>代码</dt><dd><b>${SN.items.length}</b> 段，都能用 CPU 跑</dd></div>
    <div class="checked">外部链接最近一次逐个检查：${esc(RES.checked)}</div>`;
  $('#libDesc').textContent = `${RES.items.length} 条，每一条都实际打开核对过（${RES.checked}），标题、作者和年份照页面上写的抄。优先收中文讲解；论文和代码大多只有英文原版。`;
}

// 顶栏的本页目录：屏幕中线落在哪一节，就点亮哪一节（还在首屏时一个都不亮）
function watchSections() {
  const links = $$('.lp-jump a').map((a) => [document.getElementById(a.getAttribute('href').slice(1)), a]).filter(([s]) => s);
  let q = 0;
  const update = () => {
    q = 0;
    const mid = innerHeight * 0.45;
    let cur = null;
    links.forEach(([s]) => { if (s.getBoundingClientRect().top <= mid) cur = s; });
    links.forEach(([s, a]) => a.classList.toggle('on', s === cur));
  };
  addEventListener('scroll', () => { if (!q) q = requestAnimationFrame(update); }, { passive: true });
  update();
}

async function main() {
  let RM, RES, GL, SN;
  try {
    [RM, RES, GL, SN] = await Promise.all(['data/roadmap.json', 'data/resources.json', 'data/glossary.json', 'data/snippets.json'].map(getJSON));
  } catch (e) {
    $('.lp-loading').textContent = `数据载入失败：${e.message}`;
    return;
  }
  heroStats(RM, RES, GL, SN);
  const rm = new Roadmap($('#roadmap'), RM, RES, { onProgress: setProgress });
  new Library($('#library'), RES, { stationOf: (id) => rm.stationOf(id), focusStation: (id) => rm.focus(id) });
  new Glossary($('#glossary'), GL, new Map(RES.topics.map((t) => [t.id, t.hue])));
  const tryit = new TryIt($('#try'), SN);
  watchSections();

  // #try-<片段 id>：选中那一段再滚过去（术语表里的“本页 · 动手试试 ③”用的就是它）
  const goTry = (hash) => {
    const m = /^#try-(.+)$/.exec(hash);
    if (!m) return false;
    tryit.select(m[1]);
    $('#try').scrollIntoView({ block: 'start' });
    return true;
  };
  document.addEventListener('click', (e) => {
    const a = e.target.closest('a[href^="#try-"]');
    if (a && goTry(a.getAttribute('href'))) e.preventDefault();
  });
  if (!goTry(location.hash) && location.hash.length > 1) {
    // 数据是异步渲染的，浏览器自己的锚点跳转那时还找不到目标，这里补一次
    const t = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (t) t.scrollIntoView({ block: 'start' });
  }
}

main();
