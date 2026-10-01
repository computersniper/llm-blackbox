// 调试器界面：面包屑、深度、播放控制、倍速、步骤轨道、代码高亮、讲解、变量监视。
import { $, $$, esc } from '../../js/ui.js';
import { DEPTH_NAMES, MAX_DEPTH } from './timeline.js';
import { renderCode, linesFor, explain, watch, renderWatch, stepLabel, crumbs, shapeOf } from './explain.js';

export const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];

export class Controls {
  constructor(h, manifest) {
    this.h = h;
    this.M = manifest;
    this.code = $('#code');
    this.codeKey = '';
    this.setCode(null);
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
    $('#track').addEventListener('click', (e) => { const i = e.target.closest('i'); if (i) h.seek(Number(i.dataset.i)); });
    $('#track').addEventListener('pointerover', (e) => { const i = e.target.closest('i'); if (i) i.title = i.dataset.l; });
    $('#crumbs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) h.depth(Number(b.dataset.d)); });
  }

  // 伪代码里的注释带着这张图的真实尺寸，换图时重画
  setCode(V) {
    const key = V ? V.id : '';
    if (key === this.codeKey && this.lines) return;
    this.codeKey = key;
    renderCode(this.code, this.M, V);
    this.lines = $$('.ln', this.code);
  }

  renderCrumbs(tl) {
    const el = $('#crumbs');
    if (!tl) { el.innerHTML = ''; return; }
    const list = crumbs(tl.depth, tl.step);
    el.innerHTML = list.map((c, k) => `${k ? '<span class="sep">›</span>' : ''}<button type="button" data-d="${c.d}" class="${c.d === tl.depth ? 'on' : ''}"><span class="d">D${c.d}</span>${esc(c.label)}</button>`).join('');
  }

  // 每换一步调用一次
  update(tl, ctx) {
    const s = tl.step, Q = tl.Q;
    this.setCode(Q.V);
    this.renderCrumbs(tl);
    $('#dname').innerHTML = `${DEPTH_NAMES[tl.depth]}<small>DEPTH ${tl.depth} / ${MAX_DEPTH}</small>`;
    $('#btnIn').disabled = !tl.canInto();
    $('#btnOut').title = tl.depth <= 1 ? '回到聊天（−）' : '退回上一层（−）';
    const cur = new Set(linesFor(s, Q));
    this.lines.forEach((l, i) => l.classList.toggle('cur', cur.has(i + 1)));
    const first = this.lines[Math.min(...cur) - 1];
    if (first) {
      const box = this.code.parentElement;
      const top = first.offsetTop - box.clientHeight * 0.25;
      if (first.offsetTop < box.scrollTop || first.offsetTop > box.scrollTop + box.clientHeight * 0.6) box.scrollTo({ top, behavior: 'smooth' });
    }
    const shape = shapeOf(s, Q);
    $('#explain').innerHTML = `<div class="eyebrow" style="margin-bottom:4px">这一步 · ${esc(stepLabel(s))}</div>${shape ? `<div class="shape">${shape}</div>` : ''}${explain(s, Q, ctx)}`;
    renderWatch($('#watch'), watch(s, Q, ctx));
    const n = tl.list.length;
    const a = n <= 90 ? 0 : Math.max(0, Math.min(n - 80, tl.i - 40)), b = n <= 90 ? n : a + 80;
    let h = '';
    for (let i = a; i < b; i++) h += `<i data-i="${i}" data-l="${esc(stepLabel(tl.list[i]))}" class="${i < tl.i ? 'done' : i === tl.i ? 'cur' : ''}${tl.list[i].ph === 'vit' || tl.list[i].ph === 'prep' || tl.list[i].ph === 'merge' || tl.list[i].ph === 'splice' || tl.list[i].ph === 'see' ? ' vis' : ''}"></i>`;
    $('#track').innerHTML = h;
    $('#pos').innerHTML = `词元 <b>${tl.g + 1}</b>/${tl.G}<br>步骤 <b>${tl.i + 1}</b>/${n}`;
    this.updatePlay(tl);
  }

  updatePlay(tl) {
    $('#btnPlay').textContent = tl.playing ? '❚❚' : '▶';
    $('#btnPlay').title = tl.playing ? '暂停（空格）' : tl.done ? '已经生成完毕' : '播放（空格）';
    $$('#speeds button').forEach((b) => { b.classList.toggle('on', Number(b.dataset.s) === tl.speed); b.setAttribute('aria-checked', Number(b.dataset.s) === tl.speed); });
  }
}
