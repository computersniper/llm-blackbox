// 调试器界面：面包屑、深度、播放控制、倍速、步骤轨道、代码高亮、讲解、变量监视、注意力头选择。
import { $, $$, esc } from './ui.js';
import { DEPTH_NAMES, MAX_DEPTH } from './timeline.js';
import { isEn, L as tr } from './i18n.js';
import { renderCode, linesFor, explain, watch, renderWatch, stepLabel, crumbs, interestingHead, shapeOf } from './explain.js';

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
    this.headsEl = $('#heads');
    this.headsEl.addEventListener('click', (e) => {
      const b = e.target.closest('[data-h]');
      if (b) h.head(b.dataset.h === 'avg' ? 'avg' : Number(b.dataset.h));
    });
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
    this.renderCrumbs(tl);
    $('#dname').innerHTML = `${DEPTH_NAMES[tl.depth]}<small>DEPTH ${tl.depth} / ${MAX_DEPTH}</small>`;
    $('#btnIn').disabled = !tl.canInto();
    $('#btnOut').title = tl.depth <= 1 ? tr('回到聊天（−）', 'Back to chat (−)') : tr('退回上一层（−）', 'Step out one level (−)');
    // 代码高亮
    const cur = new Set(linesFor(s, Q));
    this.lines.forEach((l, i) => l.classList.toggle('cur', cur.has(i + 1)));
    const first = this.lines[Math.min(...cur) - 1];
    if (first) {
      const box = this.code.parentElement;
      const top = first.offsetTop - box.clientHeight * 0.25;
      if (first.offsetTop < box.scrollTop || first.offsetTop > box.scrollTop + box.clientHeight * 0.6) box.scrollTo({ top, behavior: 'smooth' });
    }
    // 讲解要知道当前是不是比特视图（同一个“逐项相乘”步骤，在 D7 讲的是比特）
    const cx = { ...ctx, view: tl.view };
    const shape = shapeOf(s, Q, cx);
    const label = stepLabel(s) + (tl.view === 'bits' ? tr(' · 比特', ' · bits') : '');
    $('#explain').innerHTML = `<div class="eyebrow" style="margin-bottom:4px">${tr('这一步', 'This step')} · ${esc(label)}</div>${shape ? `<div class="shape">${shape}</div>` : ''}${explain(s, Q, cx)}`;
    renderWatch($('#watch'), watch(s, Q, ctx));
    // 步骤轨道（当前深度、当前词元的所有步骤）
    const n = tl.list.length;
    const show = n <= 90 ? tl.list.map((st, i) => i) : null;
    const track = $('#track');
    if (show) track.innerHTML = show.map((i) => `<i data-i="${i}" data-l="${esc(stepLabel(tl.list[i]))}" class="${i < tl.i ? 'done' : i === tl.i ? 'cur' : ''}"></i>`).join('');
    else {
      // 太多就只显示当前附近
      const a = Math.max(0, tl.i - 40), b = Math.min(n, a + 80);
      let h = '';
      for (let i = a; i < b; i++) h += `<i data-i="${i}" data-l="${esc(stepLabel(tl.list[i]))}" class="${i < tl.i ? 'done' : i === tl.i ? 'cur' : ''}"></i>`;
      track.innerHTML = h;
    }
    $('#pos').innerHTML = isEn ? `token <b>${tl.g + 1}</b>/${tl.G}<br>step <b>${tl.i + 1}</b>/${n}` : `词元 <b>${tl.g + 1}</b>/${tl.G}<br>步骤 <b>${tl.i + 1}</b>/${n}`;
    this.updatePlay(tl);
    this.updateHeads(tl, ctx);
  }

  updatePlay(tl) {
    $('#btnPlay').textContent = tl.playing ? '❚❚' : '▶';
    $('#btnPlay').title = tl.playing ? tr('暂停（空格）', 'Pause (Space)') : tl.done ? tr('已经生成完毕', 'Generation finished') : tr('播放（空格）', 'Play (Space)');
    $$('#speeds button').forEach((b) => { b.classList.toggle('on', Number(b.dataset.s) === tl.speed); b.setAttribute('aria-checked', Number(b.dataset.s) === tl.speed); });
  }

  updateHeads(tl, ctx) {
    const s = tl.step;
    const show = (tl.view === 'attn' || tl.view === 'dot') && s.ph === 'layer';
    this.headsEl.hidden = !show;
    if (!show) return;
    const Q = tl.Q, i = Q.row(tl.g);
    const auto = interestingHead(Q, s.L, i);
    const sel = ctx.head === 'avg' ? 'avg' : ctx.head ?? auto;
    let html = `<span class="eyebrow">${tr(`第 ${s.L} 层 · 16 个查询头（每 2 个共用一组 KV）`, `Layer ${s.L} · 16 query heads (every 2 share one KV group)`)}</span><div class="heads-grid">`;
    for (let kv = 0; kv < 8; kv++) {
      html += '<div class="hgrp">';
      for (const h of [kv * 2, kv * 2 + 1]) {
        const row = Q.att(s.L, h, i);
        const sink = row.find((r) => r.j === 0)?.w || 0;
        html += `<button type="button" class="hbtn ${sel === h ? 'on' : ''}" data-h="${h}" title="${tr(`第 ${h} 头 · 看开头的权重 ${(sink * 100).toFixed(0)}%`, `Head ${h} · weight on the first token ${(sink * 100).toFixed(0)}%`)}">H${h}<i style="width:${Math.round(sink * 100)}%"></i></button>`;
      }
      html += `<span>KV${kv}</span></div>`;
    }
    html += `</div><button type="button" class="avg ${sel === 'avg' ? 'on' : ''}" data-h="avg">${tr('16 个头平均', 'Average of 16 heads')}</button>`;
    this.headsEl.innerHTML = html;
  }
}
