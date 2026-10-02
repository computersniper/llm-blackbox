// 调试器界面：面包屑、深度、播放控制、倍速、步骤轨道、伪代码高亮、讲解、变量监视。
// 从 ../../js/controls.js 改来：步骤换成世界模型的一帧。
import { $, $$, esc } from '../../js/ui.js';
import { L } from '../../js/i18n.js';
import { DEPTH_NAMES, MAX_DEPTH } from './timeline.js';
import { renderCode, linesFor, explain, watch, renderWatch, stepLabel, crumbs, shapeOf } from './explain.js';

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
    $('#btnDbgFold').addEventListener('click', () => this.setFolded(!$('#dbg').classList.contains('folded')));
    const sp = $('#speeds');
    sp.innerHTML = SPEEDS.map((s) => `<button type="button" data-s="${s}" role="radio">${s}×</button>`).join('');
    sp.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      const s = Number(b.dataset.s);
      // 窄屏只露出当前倍速：点它就换到下一档
      if (b.classList.contains('on') && matchMedia('(max-width: 900px)').matches) h.speed(SPEEDS[(SPEEDS.indexOf(s) + 1) % SPEEDS.length]);
      else h.speed(s);
    });
    $('#track').addEventListener('click', (e) => { const i = e.target.closest('i'); if (i) h.seek(Number(i.dataset.i)); });
    $('#track').addEventListener('pointerover', (e) => { const i = e.target.closest('i'); if (i) i.title = i.dataset.l; });
    $('#crumbs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) h.depth(Number(b.dataset.d)); });
    this.lastFrame = -1;
  }

  setFolded(f) {
    $('#dbg').classList.toggle('folded', f);
    document.body.classList.toggle('dbg-folded', f);
    $('#btnDbgFold').textContent = f ? '+' : '–';
    this.h.fold?.();
  }

  renderCrumbs(tl, extra = null) {
    const el = $('#crumbs');
    if (!tl) { el.innerHTML = extra || ''; return; }
    const list = crumbs(tl);
    el.innerHTML = list.map((c, k) => `${k ? '<span class="sep">›</span>' : ''}<button type="button" data-d="${c.d}" class="${c.d === tl.depth ? 'on' : ''}"><span class="d">D${c.d}</span>${esc(c.label)}</button>`).join('');
    el.scrollLeft = el.scrollWidth;
  }

  // 每换一步调用一次（D1 播放时每一帧都会来，讲解只在帧号变了、或者停下来时刷新）
  update(tl, ctx) {
    const s = tl.step, F = ctx.F;
    this.renderCrumbs(tl);
    $('#dname').innerHTML = `${DEPTH_NAMES[tl.depth]}<small>DEPTH ${tl.depth} / ${MAX_DEPTH}</small>`;
    $('#btnIn').disabled = !tl.canInto();
    $('#btnOut').disabled = tl.depth <= 1;
    const cur = new Set(linesFor(s, F));
    this.lines.forEach((l, i) => l.classList.toggle('cur', cur.has(i + 1)));
    const first = this.lines[Math.min(...cur) - 1];
    if (first) {
      const box = this.code.parentElement;
      if (first.offsetTop < box.scrollTop || first.offsetTop > box.scrollTop + box.clientHeight * 0.6) box.scrollTo({ top: first.offsetTop - box.clientHeight * 0.2, behavior: tl.depth === 1 ? 'auto' : 'smooth' });
    }
    const shape = shapeOf(s);
    $('#explain').innerHTML = `<div class="eyebrow" style="margin-bottom:4px">${L('这一步', 'This step')} · ${esc(stepLabel(s, F))}</div>${shape ? `<div class="shape">${shape}</div>` : ''}${explain(s, ctx)}`;
    renderWatch($('#watch'), watch(s, ctx));
    this.renderTrack(tl, F);
    this.updatePlay(tl);
  }

  renderTrack(tl, F) {
    const tr = $('#track');
    let h = '';
    if (tl.depth === 1) {
      // D1：最近 60 帧，每帧一格
      const sim = tl.sim, a = Math.max(0, sim.idx - 59);
      for (let i = a; i <= sim.idx; i++) h += `<i data-i="${i - sim.idx}" data-l="${L(`第 ${sim.hist[i].t} 帧`, `Frame ${sim.hist[i].t}`)}" class="${i < sim.idx ? 'done' : 'cur'}"></i>`;
    } else {
      const n = tl.list.length;
      const a = n <= 90 ? 0 : Math.max(0, Math.min(n - 80, tl.i - 40)), b = n <= 90 ? n : a + 80;
      for (let i = a; i < b; i++) h += `<i data-i="${i}" data-l="${esc(stepLabel(tl.list[i], F))}" class="${i < tl.i ? 'done' : i === tl.i ? 'cur' : ''}"></i>`;
    }
    tr.innerHTML = h;
    $('#pos').innerHTML = L(tl.depth === 1 ? `第 <b>${F.gameNo}</b> 局<br>里程 <b>${F.t}</b>` : `里程 <b>${F.t}</b><br>步骤 <b>${tl.i + 1}</b>/${tl.list.length}`,
      tl.depth === 1 ? `Game <b>${F.gameNo}</b><br>Dist <b>${F.t}</b>` : `Dist <b>${F.t}</b><br>Step <b>${tl.i + 1}</b>/${tl.list.length}`);
  }

  updatePlay(tl) {
    $('#btnPlay').textContent = tl.playing ? '❚❚' : '▶';
    $('#btnPlay').title = tl.playing ? L('暂停（空格）', 'Pause (Space)') : L('播放（空格）', 'Play (Space)');
    $$('#speeds button').forEach((b) => { b.classList.toggle('on', Number(b.dataset.s) === tl.speed); b.setAttribute('aria-checked', Number(b.dataset.s) === tl.speed); });
  }
}
