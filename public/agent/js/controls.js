// 调试器界面：面包屑、深度、播放控制、倍速、步骤轨道、伪代码高亮、讲解、变量监视。
// 从 ../../js/controls.js 改来：步骤与讲解换成 agent 循环的版本。
import { $, $$, esc } from '../../js/ui.js';
import { DEPTH_NAMES, MAX_DEPTH } from './timeline.js';
import { renderCode, linesFor, explain, watch, renderWatch, stepLabel, crumbs } from './explain.js';
import { L } from '../../js/i18n.js';

export const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];

export class Controls {
  constructor(h) {
    this.h = h;
    this.code = $('#code');
    renderCode(this.code);
    this.lines = $$('.ln', this.code);
    $('#btnIn').addEventListener('click', () => h.into());
    $('#btnOut').addEventListener('click', () => h.out());
    $('#btnPrev').addEventListener('click', () => h.prev());
    $('#btnNextStep').addEventListener('click', () => h.next());
    $('#btnPlay').addEventListener('click', () => h.toggle());
    $('#btnDbgFold').addEventListener('click', () => {
      const folded = !document.body.classList.contains('dbg-folded');
      this.setFolded(folded);
      h.fold?.();
    });
    const sp = $('#speeds');
    sp.innerHTML = SPEEDS.map((s) => `<button type="button" data-s="${s}" role="radio">${s}×</button>`).join('');
    sp.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) h.speed(Number(b.dataset.s)); });
    $('#track').addEventListener('click', (e) => { const i = e.target.closest('i'); if (i) h.seek(Number(i.dataset.i)); });
    $('#track').addEventListener('pointerover', (e) => { const i = e.target.closest('i'); if (i) i.title = i.dataset.l; });
    $('#crumbs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) h.depth(Number(b.dataset.d)); });
  }

  setFolded(f) {
    document.body.classList.toggle('dbg-folded', f);
    $('#btnDbgFold').textContent = f ? '+' : '–';
  }

  renderCrumbs(tl) {
    const el = $('#crumbs');
    if (!tl) { el.innerHTML = ''; return; }
    const list = crumbs(tl.depth, tl.step, tl.R);
    el.innerHTML = list.map((c, k) => `${k ? '<span class="sep">›</span>' : ''}<button type="button" data-d="${c.d}" class="${c.d === tl.depth ? 'on' : ''}"><span class="d">D${c.d}</span>${esc(c.label)}</button>`).join('');
    el.scrollLeft = el.scrollWidth;   // 窄屏放不下时，露出最深的那一级
  }

  // 每换一步调用一次
  update(tl) {
    const s = tl.step, R = tl.R;
    this.renderCrumbs(tl);
    $('#dname').innerHTML = `${DEPTH_NAMES[tl.depth]}<small>DEPTH ${tl.depth} / ${MAX_DEPTH}</small>`;
    $('#btnIn').disabled = !tl.canInto();
    $('#btnOut').title = tl.depth <= 1 ? L('收起调试器（−）', 'Close the debugger (−)') : L('退回上一层（−）', 'Step out (−)');
    const cur = new Set(linesFor(s, tl.depth));
    this.lines.forEach((l, i) => l.classList.toggle('cur', cur.has(i + 1)));
    const first = this.lines[Math.min(...cur) - 1];
    if (first) {
      const box = this.code.parentElement;
      if (first.offsetTop < box.scrollTop || first.offsetTop > box.scrollTop + box.clientHeight * 0.6) box.scrollTo({ top: first.offsetTop - box.clientHeight * 0.25, behavior: 'smooth' });
    }
    $('#explain').innerHTML = `<div class="eyebrow" style="margin-bottom:4px">${L('这一步', 'This step')} · ${esc(stepLabel(s, R))}</div>${explain(s, tl.depth, R)}`;
    renderWatch($('#watch'), watch(s, tl.depth, R));
    const n = tl.list.length;
    const a = n <= 90 ? 0 : Math.max(0, Math.min(n - 80, tl.i - 40)), b = n <= 90 ? n : a + 80;
    let h = '';
    for (let i = a; i < b; i++) h += `<i data-i="${i}" data-l="${esc(stepLabel(tl.list[i], R))}" class="${i < tl.i ? 'done' : i === tl.i ? 'cur' : ''}"></i>`;
    $('#track').innerHTML = h;
    $('#pos').innerHTML = L(`第 <b>${tl.t + 1}</b>/${tl.T} 圈<br>步骤 <b>${tl.i + 1}</b>/${n}`, `turn <b>${tl.t + 1}</b>/${tl.T}<br>step <b>${tl.i + 1}</b>/${n}`);
    this.updatePlay(tl);
  }

  updatePlay(tl) {
    $('#btnPlay').textContent = tl.playing ? '❚❚' : '▶';
    $('#btnPlay').title = tl.playing ? L('暂停（空格）', 'Pause (Space)') : tl.done ? L('从头再放（空格）', 'Replay from the start (Space)') : L('播放（空格）', 'Play (Space)');
    $$('#speeds button').forEach((b) => { b.classList.toggle('on', Number(b.dataset.s) === tl.speed); b.setAttribute('aria-checked', Number(b.dataset.s) === tl.speed); });
  }
}
