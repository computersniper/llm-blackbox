// 调试器界面：面包屑、深度、播放控制、倍速、步骤轨道、代码高亮、讲解、变量监视。
import { $, $$, esc } from '../../js/ui.js';
import { DEPTH_NAMES, MAX_DEPTH } from './timeline.js';
import { codeFor, linesFor, explain, watch, stepLabel, crumbs, shapeOf } from './explain.js';
import * as GX from './glass/explain.js';
import { L } from './lang.js';

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
    sp.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      const s = Number(b.dataset.s);
      // 窄屏只露出当前倍速：点它就换到下一档
      if (b.classList.contains('on') && matchMedia('(max-width: 900px)').matches) h.speed(SPEEDS[(SPEEDS.indexOf(s) + 1) % SPEEDS.length]);
      else h.speed(s);
    });
    $('#track').addEventListener('click', (e) => { const i = e.target.closest('i'); if (i) h.seek(Number(i.dataset.i), i.dataset.kind); });
    $('#track').addEventListener('pointerover', (e) => { const i = e.target.closest('i'); if (i) i.title = i.dataset.l; });
    $('#crumbs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) h.depth(Number(b.dataset.d)); });
  }

  renderCode(tl, R) {
    const key = R.kind === 'glass' ? 'glass' : tl.depth === 0 ? 'pipe' : R.kind;
    if (key === this.codeKey) return;
    this.codeKey = key;
    this.code.innerHTML = (key === 'glass' ? GX.codeFor(R) : codeFor(key, R)).map((l, i) => `<span class="ln" data-n="${i + 1}"><span class="bp"></span><span class="no">${i + 1}</span>${l}</span>`).join('');
    this.lines = $$('.ln', this.code);
  }

  // 每换一步调用一次。ready = 讲解 / 变量要用的数据分块都到了；没到时先保留上一步的内容并调暗（CSS 里延迟一点再暗，免得一闪），
  // 到了以后页面会再调用一次
  update(tl, R, ctx, ready = true, err = null) {
    const s = tl.step;
    this.renderCode(tl, R);
    // 玻璃小模型那一章有自己的一套讲解（./glass/explain.js），接口相同
    const G = R.kind === 'glass';
    const X = G ? GX : { crumbs, linesFor, stepLabel, shapeOf, explain, watch };
    const DN = tl.depthNames || DEPTH_NAMES, MD = tl.maxDepth ?? MAX_DEPTH;
    const cr = X.crumbs(tl.depth, s, R, ctx);
    $('#crumbs').innerHTML = cr.map((c, k) => `${k ? '<span class="sep">›</span>' : ''}<button type="button" data-d="${c.d}" class="${c.d === tl.depth ? 'on' : ''}"><span class="d">D${c.d}</span>${esc(c.label)}</button>`).join('');
    $('#dname').innerHTML = `${DN[tl.depth]}<small>DEPTH ${tl.depth} / ${MD}</small>`;
    $('#btnIn').disabled = !tl.canInto();
    $('#btnOut').disabled = tl.depth === 0;
    $('#dbgSub').textContent = G ? 'glass_train()' : tl.depth === 0 ? 'pipeline()' : R.kind === 'tiny' ? 'pretrain()' : 'sft()';
    const cur = new Set(X.linesFor(s, R, tl.depth));
    this.lines.forEach((l, i) => l.classList.toggle('cur', cur.has(i + 1)));
    const first = this.lines[Math.min(...cur) - 1];
    if (first) {
      const box = this.code.parentElement;
      if (first.offsetTop < box.scrollTop || first.offsetTop > box.scrollTop + box.clientHeight * 0.6) box.scrollTo({ top: first.offsetTop - box.clientHeight * 0.2, behavior: 'smooth' });
    }
    const ex = $('#explain'), wt = $('#watch');
    const head = `<div class="eyebrow" style="margin-bottom:4px">${L('这一步', 'This step')} · ${G ? X.stepLabel(s, R, tl.depth, ctx) : esc(stepLabel(s, R, tl.depth))}</div>`;
    let stale = false;
    if (ready) {
      const shape = X.shapeOf(s, R, tl.depth);
      ex.innerHTML = `${head}${shape ? `<div class="shape">${shape}</div>` : ''}${X.explain(s, R, ctx, tl.depth)}`;
      wt.innerHTML = X.watch(s, R, ctx, tl.depth).map(([k, v]) => (k === 'wh' ? `<div class="wh">${esc(v)}</div>` : `<span class="k">${esc(k)}</span><span class="v" title="${esc(String(v))}">${esc(String(v))}</span>`)).join('');
      this.shown = `${tl.depth}:${R.kind}`;
    } else if (err || this.shown !== `${tl.depth}:${R.kind}`) {
      // 上一次显示的是别的深度 / 别的训练，留着没意义：直接说正在载入（或者为什么载入不了）
      const msg = err ? (err.unsupported ? err.message : L('这一步的数据没有载入成功，稍后会自动重试…', 'This step’s data failed to load; retrying automatically…')) : L('正在载入这一步的真实记录…', 'Loading the real record of this step…');
      ex.innerHTML = `${head}<span class="${err ? 'err' : 'dimmed'}">${esc(msg)}</span>`;
      wt.innerHTML = '';
      this.shown = '';
    } else stale = true;
    ex.classList.toggle('stale', stale);
    wt.classList.toggle('stale', stale);
    this.renderTrack(tl, R, X, ctx);
    this.updatePlay(tl);
  }

  // 步骤轨道：D1 显示所有检查点；其余深度显示当前检查点里的步骤
  renderTrack(tl, R, X = { stepLabel }, ctx = null) {
    const tr = $('#track');
    let h = '';
    if (R.kind === 'glass') return this.renderTrackGlass(tl, R, X, ctx);
    if (tl.depth === 1) {
      for (let k = 0; k < tl.K; k++) h += `<i data-i="${k}" data-kind="ck" data-l="${L(`第 ${R.stepNo(k)} 步`, `Step ${R.stepNo(k)}`)}" class="${k < tl.k ? 'done' : k === tl.k ? 'cur' : ''}"></i>`;
    } else {
      const n = tl.list.length;
      const a = n <= 90 ? 0 : Math.max(0, Math.min(n - 80, tl.i - 40)), b = n <= 90 ? n : a + 80;
      for (let i = a; i < b; i++) h += `<i data-i="${i}" data-l="${esc(stepLabel(tl.list[i], R, tl.depth))}" class="${i < tl.i ? 'done' : i === tl.i ? 'cur' : ''}"></i>`;
    }
    tr.innerHTML = h;
    $('#pos').innerHTML = L(tl.depth === 0 ? `阶段 <b>${tl.i + 1}</b>/3` : `${R.kind === 'tiny' ? '检查点' : '第'} <b>${tl.k + 1}</b>/${tl.K}${R.kind === 'tiny' ? '' : ' 步'}<br>步骤 <b>${tl.i + 1}</b>/${tl.list.length}`,
      tl.depth === 0 ? `Stage <b>${tl.i + 1}</b>/3` : `${R.kind === 'tiny' ? 'Checkpoint' : 'SFT step'} <b>${tl.k + 1}</b>/${tl.K}<br>Substep <b>${tl.i + 1}</b>/${tl.list.length}`);
  }

  // 玻璃小模型：D1 每一帧一格（标注训练的第几步），其余深度是这一帧里的步骤
  renderTrackGlass(tl, R, X, ctx) {
    const tr = $('#track'), D = R.D;
    let h = '';
    if (tl.depth === 1) {
      for (let k = 0; k < tl.K; k++) h += `<i data-i="${k}" data-kind="ck" data-l="${L(`第 ${D.FR[k] + 1} 步`, `Step ${D.FR[k] + 1}`)}" class="${k < tl.k ? 'done' : k === tl.k ? 'cur' : ''}"></i>`;
    } else {
      const n = tl.list.length;
      const a = n <= 90 ? 0 : Math.max(0, Math.min(n - 80, tl.i - 40)), b = n <= 90 ? n : a + 80;
      for (let i = a; i < b; i++) h += `<i data-i="${i}" data-l="${esc(X.stepLabel(tl.list[i], R, tl.depth, ctx).replace(/<[^>]+>/g, ''))}" class="${i < tl.i ? 'done' : i === tl.i ? 'cur' : ''}"></i>`;
    }
    tr.innerHTML = h;
    const t = D.FR[tl.k] + 1;
    $('#pos').innerHTML = tl.depth === 0 ? L(`初始化 <b>${tl.i + 1}</b>/${tl.list.length}`, `Init <b>${tl.i + 1}</b>/${tl.list.length}`)
      : L(`第 <b>${t}</b>/${D.S} 步<br>${tl.depth === 1 ? `帧 <b>${tl.k + 1}</b>/${tl.K}` : `步骤 <b>${tl.i + 1}</b>/${tl.list.length}`}`, `Step <b>${t}</b>/${D.S}<br>${tl.depth === 1 ? `Frame <b>${tl.k + 1}</b>/${tl.K}` : `Substep <b>${tl.i + 1}</b>/${tl.list.length}`}`);
  }

  updatePlay(tl) {
    $('#btnPlay').textContent = tl.playing ? '❚❚' : '▶';
    $('#btnPlay').title = tl.playing ? L('暂停（空格）', 'Pause (Space)') : tl.done ? L('播放完了，再按一次从头开始', 'Finished — press again to start over') : L('播放（空格）', 'Play (Space)');
    $$('#speeds button').forEach((b) => { b.classList.toggle('on', Number(b.dataset.s) === tl.speed); b.setAttribute('aria-checked', Number(b.dataset.s) === tl.speed); });
  }
}
