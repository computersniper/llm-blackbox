// 这一章的调试器界面：和 ../controls.js 的 Controls.update 同一套 DOM（面包屑、深度、代码高亮、讲解、变量、步骤轨道），
// 只是内容换成 ./explain.js。共享的 controls.js 不用改：页面在这一章里调用 updateDebugger(controls, …)。
import { $, $$, esc } from '../../../js/ui.js';
import * as TX from './explain.js';
import { L } from '../lang.js';

export function updateDebugger(controls, tl, X, ctx, ready = true, err = null) {
  const s = tl.step;
  // 代码：换章时重画一次（Controls 自己的缓存键也改掉，切回别的章节它会重画）
  if (controls.codeKey !== 'tiny3d') {
    controls.codeKey = 'tiny3d';
    controls.code.innerHTML = TX.codeFor(X).map((l, i) => `<span class="ln" data-n="${i + 1}"><span class="bp"></span><span class="no">${i + 1}</span>${l}</span>`).join('');
    controls.lines = $$('.ln', controls.code);
  }
  const cr = TX.crumbs(tl.depth, s, X);
  $('#crumbs').innerHTML = cr.map((c, k) => `${k ? '<span class="sep">›</span>' : ''}<button type="button" data-d="${c.d}" class="${c.d === tl.depth ? 'on' : ''}"><span class="d">D${c.d}</span>${esc(c.label)}</button>`).join('');
  $('#dname').innerHTML = `${tl.depthNames[tl.depth]}<small>DEPTH ${tl.depth} / ${tl.maxDepth}</small>`;
  $('#btnIn').disabled = !tl.canInto();
  $('#btnOut').disabled = tl.depth <= tl.minDepth;
  $('#dbgSub').textContent = 'pretrain()';
  const cur = new Set(TX.linesFor(s));
  controls.lines.forEach((l, i) => l.classList.toggle('cur', cur.has(i + 1)));
  const first = controls.lines[Math.min(...cur) - 1];
  if (first) {
    const box = controls.code.parentElement;
    if (first.offsetTop < box.scrollTop || first.offsetTop > box.scrollTop + box.clientHeight * 0.6) box.scrollTo({ top: first.offsetTop - box.clientHeight * 0.2, behavior: 'smooth' });
  }
  const ex = $('#explain'), wt = $('#watch');
  const head = `<div class="eyebrow" style="margin-bottom:4px">${L('这一步', 'This step')} · ${TX.stepLabel(s, X)}</div>`;
  let stale = false;
  if (ready) {
    const shape = TX.shapeOf(s, X);
    ex.innerHTML = `${head}${shape ? `<div class="shape">${shape}</div>` : ''}${TX.explain(s, X, ctx, tl.depth)}`;
    wt.innerHTML = TX.watch(s, X, ctx, tl.depth).map(([k, v]) => (k === 'wh' ? `<div class="wh">${esc(v)}</div>` : `<span class="k">${esc(k)}</span><span class="v" title="${esc(String(v))}">${esc(String(v))}</span>`)).join('');
    controls.shown = `${tl.depth}:tiny3d`;
  } else if (err || controls.shown !== `${tl.depth}:tiny3d`) {
    const msg = err ? (err.unsupported ? err.message : L('这一步的数据没有载入成功，稍后会自动重试…', 'This step’s data failed to load; retrying automatically…')) : L('正在载入这一步的真实记录…', 'Loading the real record of this step…');
    ex.innerHTML = `${head}<span class="${err ? 'err' : 'dimmed'}">${esc(msg)}</span>`;
    wt.innerHTML = '';
    controls.shown = '';
  } else stale = true;
  ex.classList.toggle('stale', stale);
  wt.classList.toggle('stale', stale);
  renderTrack(tl, X);
  controls.updatePlay(tl);
}

// 步骤轨道：D1 每个检查点一格（标注训练的第几步），其余深度是这个检查点里的步骤
function renderTrack(tl, X) {
  let h = '';
  if (tl.depth === 1) {
    for (let k = 0; k < tl.K; k++) h += `<i data-i="${k}" data-kind="ck" data-l="${L(`第 ${X.step(k) + 1} 步`, `Step ${X.step(k) + 1}`)}" class="${k < tl.k ? 'done' : k === tl.k ? 'cur' : ''}"></i>`;
  } else {
    const n = tl.list.length;
    const a = n <= 90 ? 0 : Math.max(0, Math.min(n - 80, tl.i - 40)), b = n <= 90 ? n : a + 80;
    for (let i = a; i < b; i++) h += `<i data-i="${i}" data-l="${esc(TX.stepLabel(tl.list[i], X).replace(/<[^>]+>/g, ''))}" class="${i < tl.i ? 'done' : i === tl.i ? 'cur' : ''}"></i>`;
  }
  $('#track').innerHTML = h;
  const t = X.step(tl.k) + 1;
  $('#pos').innerHTML = L(`第 <b>${t}</b>/${X.S} 步<br>${tl.depth === 1 ? `检查点 <b>${tl.k + 1}</b>/${tl.K}` : `步骤 <b>${tl.i + 1}</b>/${tl.list.length}`}`, `Step <b>${t}</b>/${X.S}<br>${tl.depth === 1 ? `Checkpoint <b>${tl.k + 1}</b>/${tl.K}` : `Substep <b>${tl.i + 1}</b>/${tl.list.length}`}`);
}
