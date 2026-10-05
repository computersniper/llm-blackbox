// 调试器界面接上第三章：在共用的 ../controls.js 的 Controls 上包一层，R.kind === 'qwen3d' 时换成这一章自己的讲解（./explain.js）
// 和步骤轨道；其余章节原样交给父类。
import { $, esc } from '../../../js/ui.js';
import * as QX from './explain.js';
import { L } from './lang.js';

export function withQwen3d(Base) {
  return class extends Base {
    renderCode(tl, R) {
      if (R.kind !== 'qwen3d') return super.renderCode(tl, R);
      if (this.codeKey === 'qwen3d') return;
      this.codeKey = 'qwen3d';
      this.code.innerHTML = QX.codeFor(R).map((l, i) => `<span class="ln" data-n="${i + 1}"><span class="bp"></span><span class="no">${i + 1}</span>${l}</span>`).join('');
      this.lines = [...this.code.querySelectorAll('.ln')];
    }

    update(tl, R, ctx, ready = true, err = null) {
      if (R.kind !== 'qwen3d') return super.update(tl, R, ctx, ready, err);
      ctx.qL = tl.L;
      const s = tl.step;
      this.renderCode(tl, R);
      const cr = QX.crumbs(tl.depth, s, R, ctx);
      $('#crumbs').innerHTML = cr.map((c, k) => `${k ? '<span class="sep">›</span>' : ''}<button type="button" data-d="${c.d}" class="${c.d === tl.depth && k === cr.length - 1 ? 'on' : ''}"><span class="d">D${c.d}</span>${esc(c.label)}</button>`).join('');
      $('#dname').innerHTML = `${tl.depthNames[tl.depth]}<small>DEPTH ${tl.depth} / ${tl.maxDepth}</small>`;
      $('#btnIn').disabled = !tl.canInto();
      $('#btnOut').disabled = tl.depth <= (tl.minDepth ?? 1);
      $('#dbgSub').textContent = 'sft()';
      const cur = new Set(QX.linesFor(s, R, tl.depth));
      this.lines.forEach((l, i) => l.classList.toggle('cur', cur.has(i + 1)));
      const first = this.lines[Math.min(...cur) - 1];
      if (first) {
        const box = this.code.parentElement;
        if (first.offsetTop < box.scrollTop || first.offsetTop > box.scrollTop + box.clientHeight * 0.6) box.scrollTo({ top: first.offsetTop - box.clientHeight * 0.2, behavior: 'smooth' });
      }
      const ex = $('#explain'), wt = $('#watch');
      const head = `<div class="eyebrow" style="margin-bottom:4px">${L('这一步', 'This step')} · ${QX.stepLabel(s, R, tl.depth, ctx)}</div>`;
      if (ready || err) {
        const shape = QX.shapeOf(s, R, tl.depth, ctx);
        ex.innerHTML = err && !ready ? `${head}<span class="err">${esc(err.unsupported ? err.message : L('这一步的数据没有载入成功，稍后会自动重试…', 'This step’s data failed to load; retrying automatically…'))}</span>`
          : `${head}${shape ? `<div class="shape">${shape}</div>` : ''}${QX.explain(s, R, ctx, tl.depth)}`;
        wt.innerHTML = ready ? QX.watch(s, R, ctx, tl.depth).map(([k, v]) => (k === 'wh' ? `<div class="wh">${esc(v)}</div>` : `<span class="k">${esc(k)}</span><span class="v" title="${esc(String(v))}">${esc(String(v))}</span>`)).join('') : '';
      } else {
        // 分块还在路上：讲解里能用首屏数据的部分照常写，缺的地方 explain 自己会说“正在载入”
        ex.innerHTML = `${head}${QX.explain(s, R, ctx, tl.depth)}`;
        wt.innerHTML = QX.watch(s, R, ctx, tl.depth).map(([k, v]) => (k === 'wh' ? `<div class="wh">${esc(v)}</div>` : `<span class="k">${esc(k)}</span><span class="v">${esc(String(v))}</span>`)).join('');
      }
      ex.classList.remove('stale');
      wt.classList.remove('stale');
      this.shown = '';
      this.renderTrackQ(tl, R, ctx);
      this.updatePlay(tl);
    }

    renderTrack(tl, R, X, ctx) {
      if (R.kind !== 'qwen3d') return super.renderTrack(tl, R, X, ctx);
      return this.renderTrackQ(tl, R, ctx);
    }

    // D1：三步 + 结果，一格一拍；其余深度是这一步里的步骤
    renderTrackQ(tl, R, ctx) {
      const tr = $('#track');
      let h = '';
      if (tl.depth === 1) {
        const at = tl.step.ph === 'end' ? R.K : tl.k;
        for (let k = 0; k <= R.K; k++) h += `<i data-i="${k}" data-kind="ck" data-l="${k === R.K ? L('3 步之后', 'After 3 steps') : L(`第 ${k + 1} 步`, `Step ${k + 1}`)}" class="${k < at ? 'done' : k === at ? 'cur' : ''}"></i>`;
      } else {
        const n = tl.list.length;
        const a = n <= 90 ? 0 : Math.max(0, Math.min(n - 80, tl.i - 40)), b = n <= 90 ? n : a + 80;
        for (let i = a; i < b; i++) h += `<i data-i="${i}" data-l="${esc(QX.stepLabel(tl.list[i], R, tl.depth, ctx).replace(/<[^>]+>/g, ''))}" class="${i < tl.i ? 'done' : i === tl.i ? 'cur' : ''}"></i>`;
      }
      tr.innerHTML = h;
      $('#pos').innerHTML = tl.depth === 1
        ? L(`第 <b>${Math.min(R.K, tl.k + 1)}</b>/${R.K} 步${tl.step.ph === 'end' ? '<br>结果' : ''}`, `Step <b>${Math.min(R.K, tl.k + 1)}</b>/${R.K}${tl.step.ph === 'end' ? '<br>Result' : ''}`)
        : L(`第 <b>${tl.k + 1}</b>/${R.K} 步<br>步骤 <b>${tl.i + 1}</b>/${tl.list.length}`, `Step <b>${tl.k + 1}</b>/${R.K}<br>Substep <b>${tl.i + 1}</b>/${tl.list.length}`);
    }
  };
}
