// 调试器界面：面包屑、深度、播放控制、倍速、步骤轨道、代码高亮、讲解、变量监视。
import { $, $$, esc } from '../../js/ui.js';
import { DEPTH_NAMES, MAX_DEPTH } from './timeline.js';
import { codeFor, linesFor, explain, watch, stepLabel, crumbs, shapeOf } from './explain.js';

export const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];

export class Controls {
  constructor(h) {
    this.h = h;
    this.code = $('#code');
    this.codeKey = '';
    $('#btnIn').addEventListener('click', () => h.into());
    $('#btnOut').addEventListener('click', () => h.out());
    $('#btnPrev').addEventListener('click', () => h.prev());
    $('#btnNextStep').addEventListener('click', () => h.next());
    $('#btnPlay').addEventListener('click', () => h.toggle());
    $('#btnDbgFold').addEventListener('click', () => {
      const d = $('#dbg');
      d.classList.toggle('folded');
      $('#btnDbgFold').textContent = d.classList.contains('folded') ? '+' : '–';
      h.fold?.();
    });
    const sp = $('#speeds');
    sp.innerHTML = SPEEDS.map((s) => `<button type="button" data-s="${s}" role="radio">${s}×</button>`).join('');
    sp.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) h.speed(Number(b.dataset.s)); });
    $('#track').addEventListener('click', (e) => { const i = e.target.closest('i'); if (i) h.seek(Number(i.dataset.i), i.dataset.kind); });
    $('#track').addEventListener('pointerover', (e) => { const i = e.target.closest('i'); if (i) i.title = i.dataset.l; });
    $('#crumbs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) h.depth(Number(b.dataset.d)); });
  }

  renderCode(tl, R) {
    const key = tl.depth === 0 ? 'pipe' : R.kind;
    if (key === this.codeKey) return;
    this.codeKey = key;
    this.code.innerHTML = codeFor(key, R).map((l, i) => `<span class="ln" data-n="${i + 1}"><span class="bp"></span><span class="no">${i + 1}</span>${l}</span>`).join('');
    this.lines = $$('.ln', this.code);
  }

  // 每换一步调用一次
  update(tl, R, ctx) {
    const s = tl.step;
    this.renderCode(tl, R);
    const cr = crumbs(tl.depth, s, R);
    $('#crumbs').innerHTML = cr.map((c, k) => `${k ? '<span class="sep">›</span>' : ''}<button type="button" data-d="${c.d}" class="${c.d === tl.depth ? 'on' : ''}"><span class="d">D${c.d}</span>${esc(c.label)}</button>`).join('');
    $('#dname').innerHTML = `${DEPTH_NAMES[tl.depth]}<small>DEPTH ${tl.depth} / ${MAX_DEPTH}</small>`;
    $('#btnIn').disabled = !tl.canInto();
    $('#btnOut').disabled = tl.depth === 0;
    $('#dbgSub').textContent = tl.depth === 0 ? 'pipeline()' : R.kind === 'tiny' ? 'pretrain()' : 'sft()';
    const cur = new Set(linesFor(s, R, tl.depth));
    this.lines.forEach((l, i) => l.classList.toggle('cur', cur.has(i + 1)));
    const first = this.lines[Math.min(...cur) - 1];
    if (first) {
      const box = this.code.parentElement;
      if (first.offsetTop < box.scrollTop || first.offsetTop > box.scrollTop + box.clientHeight * 0.6) box.scrollTo({ top: first.offsetTop - box.clientHeight * 0.2, behavior: 'smooth' });
    }
    const shape = shapeOf(s, R, tl.depth);
    $('#explain').innerHTML = `<div class="eyebrow" style="margin-bottom:4px">这一步 · ${esc(stepLabel(s, R, tl.depth))}</div>${shape ? `<div class="shape">${shape}</div>` : ''}${explain(s, R, ctx, tl.depth)}`;
    $('#watch').innerHTML = watch(s, R, ctx, tl.depth).map(([k, v]) => (k === 'wh' ? `<div class="wh">${esc(v)}</div>` : `<span class="k">${esc(k)}</span><span class="v" title="${esc(String(v))}">${esc(String(v))}</span>`)).join('');
    this.renderTrack(tl, R);
    this.updatePlay(tl);
  }

  // 步骤轨道：D1 显示所有检查点；其余深度显示当前检查点里的步骤
  renderTrack(tl, R) {
    const tr = $('#track');
    let h = '';
    if (tl.depth === 1) {
      for (let k = 0; k < tl.K; k++) h += `<i data-i="${k}" data-kind="ck" data-l="第 ${R.stepNo(k)} 步" class="${k < tl.k ? 'done' : k === tl.k ? 'cur' : ''}"></i>`;
    } else {
      const n = tl.list.length;
      const a = n <= 90 ? 0 : Math.max(0, Math.min(n - 80, tl.i - 40)), b = n <= 90 ? n : a + 80;
      for (let i = a; i < b; i++) h += `<i data-i="${i}" data-l="${esc(stepLabel(tl.list[i], R, tl.depth))}" class="${i < tl.i ? 'done' : i === tl.i ? 'cur' : ''}"></i>`;
    }
    tr.innerHTML = h;
    $('#pos').innerHTML = tl.depth === 0 ? `阶段 <b>${tl.i + 1}</b>/3` : `${R.kind === 'tiny' ? '检查点' : '第'} <b>${tl.k + 1}</b>/${tl.K}${R.kind === 'tiny' ? '' : ' 步'}<br>步骤 <b>${tl.i + 1}</b>/${tl.list.length}`;
  }

  updatePlay(tl) {
    $('#btnPlay').textContent = tl.playing ? '❚❚' : '▶';
    $('#btnPlay').title = tl.playing ? '暂停（空格）' : tl.done ? '播放完了，再按一次从头开始' : '播放（空格）';
    $$('#speeds button').forEach((b) => { b.classList.toggle('on', Number(b.dataset.s) === tl.speed); b.setAttribute('aria-checked', Number(b.dataset.s) === tl.speed); });
  }
}
