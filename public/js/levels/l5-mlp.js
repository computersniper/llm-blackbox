import { CFG, neuronPre, ACTS, CONCEPT_BY_ID } from '../model.js';
import { $, $$, esc, tokHTML, tokPlain, hueOf, fitCanvas, onResize, reducedMotion } from '../ui.js';

const N = CFG.dFF;
const COLS = 64, ROWS = N / COLS;
const POLY = 1460;

function cellColor(a) {
  if (a < 0) return `rgb(${Math.round(22 + a * 40)},${Math.round(28 + a * 10)},${Math.round(46 - a * 150)})`;
  if (a < 0.2) return 'rgb(22,28,46)';
  if (a < 1.5) {
    const k = (a - 0.2) / 1.3;
    return `rgb(${Math.round(80 + 175 * k)},${Math.round(52 + 130 * k)},${Math.round(24 + 68 * k)})`;
  }
  const k = Math.min(1, (a - 1.5) / 1.8);
  return `rgb(255,${Math.round(182 + 66 * k)},${Math.round(92 + 138 * k)})`;
}

const labelOf = (j) => CONCEPT_BY_ID.get(j)?.label ?? null;

export default {
  key: 'mlp',
  name: '前馈网络',
  en: 'NEURONS',
  scale: '3072 个神经元',

  mount(el, app) {
    const st = app.state;
    const { tokens } = app.model;
    const T = tokens.length;
    if (st.focus == null || st.focus >= T) st.focus = T - 1;
    const L = st.layer;
    const actsOf = (i) => Array.from(neuronPre(tokens[i], L), ACTS.GELU);
    let acts = actsOf(st.focus);
    let shown = acts.slice();
    let hover = -1;
    const order = Float32Array.from({ length: N }, () => Math.random());
    let reveal = reducedMotion ? 1 : 0;

    el.innerHTML = `
      <div class="lv-head rv">
        <div>
          <div class="eyebrow">Depth −05 · MLP neurons · Layer ${L}</div>
          <h2>3072 个神经元</h2>
          <p>注意力负责“交流”，前馈网络负责“消化”。每个词元的向量先被展开到 3072 维，每一维就是一个<b>神经元</b>：它只对某种模式敏感，看到了就亮起来；然后再压回 768 维，加回残差流。</p>
        </div>
        <div class="mp-stats">
          <div class="stat"><b class="mp-count acc">0</b><span>明显激活（&gt; 0.5）</span></div>
          <div class="stat"><b class="mp-frac">0%</b><span>占比</span></div>
        </div>
      </div>

      <div class="mp-flow rv" aria-hidden="true">
        <div class="mp-col c768"><span>768</span></div>
        <div class="mp-link"><i></i><em>W<sub>in</sub> · 768×3072</em></div>
        <div class="mp-col c3072"><span>3072</span><b>GELU</b></div>
        <div class="mp-link"><i></i><em>W<sub>out</sub> · 3072×768</em></div>
        <div class="mp-col c768"><span>768</span></div>
        <div class="mp-link out"><i></i><em>⊕ 加回残差流</em></div>
      </div>

      <div class="mp-toks rv"><span class="eyebrow">当前经过的词元</span>${tokens.map((t, i) => `<button type="button" class="tok mini mp-tok ${i === st.focus ? 'sel' : ''}" style="--h:${hueOf(t.text)}" data-i="${i}">${esc(tokPlain(t.text))}</button>`).join('')}</div>

      <div class="mp-grid">
        <section class="panel mp-field rv">
          <div class="panel-h"><h3>神经元阵列 <small class="mono">64 × 48</small></h3><span class="eyebrow">点击一个亮点，钻进去</span></div>
          <div class="mp-canvas-wrap"><canvas class="mp-canvas"></canvas></div>
          <div class="mp-legend hint-line"><span class="sw neg"></span>负（被抑制）<span class="sw zero"></span>≈ 0（沉默）<span class="sw pos"></span>正（激活）<span class="sw hot"></span>强烈激活</div>
        </section>
        <aside class="mp-side">
          <section class="panel mp-top rv">
            <div class="panel-h"><h3>最亮的神经元</h3></div>
            <div class="mp-list"></div>
            <p class="hint-line mp-disclaim">标签是示意：真实模型里，研究者要靠大量实验才能猜出一个神经元“在管什么”。</p>
          </section>
          <section class="panel mp-poly rv" tabindex="0">
            <div class="panel-h"><h3>奇怪的 #${POLY}</h3><span class="eyebrow">多义？</span></div>
            <div class="mp-poly-bars"></div>
            <p class="mp-poly-text">它在整句话里的激活。它似乎同时关心好几件毫不相干的事……</p>
          </section>
        </aside>
      </div>`;

    const canvas = $('.mp-canvas', el);
    let geom = { cs: 8, ox: 0, oy: 0 };

    const draw = () => {
      const { g, w, h } = fitCanvas(canvas);
      const cs = Math.max(4, Math.floor(Math.min(w / COLS, h / ROWS)));
      const ox = Math.floor((w - cs * COLS) / 2), oy = Math.floor((h - cs * ROWS) / 2);
      geom = { cs, ox, oy };
      g.clearRect(0, 0, w, h);
      const gap = cs > 6 ? 1.5 : 1;
      // 光晕
      for (let j = 0; j < N; j++) {
        const a = shown[j];
        if (a < 1.2 || order[j] > reveal) continue;
        const x = ox + (j % COLS) * cs, y = oy + Math.floor(j / COLS) * cs;
        g.fillStyle = `rgba(255,182,92,${Math.min(0.28, (a - 1.2) * 0.12)})`;
        g.fillRect(x - cs, y - cs, cs * 3, cs * 3);
      }
      for (let j = 0; j < N; j++) {
        const x = ox + (j % COLS) * cs, y = oy + Math.floor(j / COLS) * cs;
        g.fillStyle = order[j] > reveal ? 'rgb(16,21,36)' : cellColor(shown[j]);
        g.fillRect(x + gap / 2, y + gap / 2, cs - gap, cs - gap);
      }
      if (hover >= 0) {
        g.strokeStyle = '#fff';
        g.lineWidth = 1.5;
        g.strokeRect(ox + (hover % COLS) * cs - 1, oy + Math.floor(hover / COLS) * cs - 1, cs + 2, cs + 2);
      }
      // 标出多义神经元的位置（很淡）
      g.strokeStyle = 'rgba(179,157,255,.45)';
      g.lineWidth = 1;
      g.setLineDash([2, 2]);
      g.strokeRect(ox + (POLY % COLS) * cs - 2.5, oy + Math.floor(POLY / COLS) * cs - 2.5, cs + 5, cs + 5);
      g.setLineDash([]);
    };

    // 入场：神经元随机亮起
    let raf = 0;
    const t0 = performance.now();
    const intro = (now) => {
      reveal = Math.min(1, (now - t0 - 500) / 900);
      draw();
      if (reveal < 1) raf = requestAnimationFrame(intro);
    };
    if (!reducedMotion) raf = requestAnimationFrame(intro);

    const tween = (to) => {
      const from = shown.slice();
      const s0 = performance.now();
      cancelAnimationFrame(raf);
      const step = (now) => {
        const k = Math.min(1, (now - s0) / 450);
        const e = 1 - Math.pow(1 - k, 3);
        for (let j = 0; j < N; j++) shown[j] = from[j] + (to[j] - from[j]) * e;
        draw();
        if (k < 1) raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    };

    const renderSide = () => {
      const top = acts.map((a, j) => ({ a, j })).sort((x, y) => y.a - x.a).slice(0, 8);
      const max = top[0].a;
      $('.mp-list', el).innerHTML = top.map(({ a, j }) => `
        <button type="button" class="mp-row" data-j="${j}">
          <span class="mono">#${String(j).padStart(4, '0')}</span>
          <span class="mp-lbl">${labelOf(j) ? esc(labelOf(j)) : '<span class="dimmed">（说不清）</span>'}</span>
          <span class="mp-bar"><i style="width:${((a / max) * 100).toFixed(1)}%"></i></span>
          <span class="mono mp-v">${a.toFixed(2)}</span>
        </button>`).join('');
      $$('.mp-row', el).forEach((r) => {
        r.addEventListener('click', (e) => { st.neuron = Number(r.dataset.j); app.sfx.click(); app.go(6, e.currentTarget); });
        r.addEventListener('pointerenter', () => { hover = Number(r.dataset.j); draw(); if (hover === POLY) app.discover('poly'); });
        r.addEventListener('pointerleave', () => { hover = -1; draw(); });
      });
      const strong = acts.filter((a) => a > 0.5).length;
      $('.mp-count', el).textContent = strong;
      $('.mp-frac', el).textContent = `${((strong / N) * 100).toFixed(1)}%`;
      return { strong, top };
    };

    const polyActs = tokens.map((_, i) => ACTS.GELU(neuronPre(tokens[i], L)[POLY]));
    const pmax = Math.max(1, ...polyActs);
    $('.mp-poly-bars', el).innerHTML = tokens.map((t, i) => `
      <div class="mp-pb"><span class="mp-pb-bar"><i style="height:${Math.max(2, (polyActs[i] / pmax) * 100).toFixed(1)}%"></i></span>${tokHTML(t, 'mini')}</div>`).join('');
    const polyCard = $('.mp-poly', el);
    const revealPoly = () => {
      app.discover('poly');
      $('.mp-poly-text', el).innerHTML = '研究者给它的标签是：<b>“月亮 / 法国 / 数字”</b>。一个神经元同时响应互不相干的概念，这叫<b>多义神经元</b>。概念太多、神经元太少，只好“兼职”。';
    };
    polyCard.addEventListener('pointerenter', () => { hover = POLY; draw(); revealPoly(); });
    polyCard.addEventListener('pointerleave', () => { hover = -1; draw(); });
    polyCard.addEventListener('focus', revealPoly);

    canvas.addEventListener('pointermove', (e) => {
      const r = canvas.getBoundingClientRect();
      const { cs, ox, oy } = geom;
      const c = Math.floor((e.clientX - r.left - ox) / cs), rr = Math.floor((e.clientY - r.top - oy) / cs);
      if (c < 0 || rr < 0 || c >= COLS || rr >= ROWS) { hover = -1; draw(); app.tip.hide(); return; }
      const j = rr * COLS + c;
      if (j !== hover) { hover = j; draw(); }
      if (j === POLY) revealPoly();
      const lbl = labelOf(j);
      app.tip.show(`<span class="k">神经元 #${j}</span>激活 <span class="v">${acts[j].toFixed(3)}</span><br>${lbl ? `可能在响应：<b>${esc(lbl)}</b>` : '<span class="dimmed">没有明确的含义</span>'}<br><span class="dimmed">点击进入 ↓</span>`, e.clientX, e.clientY);
      canvas.style.cursor = 'zoom-in';
    });
    canvas.addEventListener('pointerleave', () => { hover = -1; draw(); app.tip.hide(); });
    const cellAt = (e) => {
      const r = canvas.getBoundingClientRect();
      const { cs, ox, oy } = geom;
      const c = Math.floor((e.clientX - r.left - ox) / cs), rr = Math.floor((e.clientY - r.top - oy) / cs);
      return c < 0 || rr < 0 || c >= COLS || rr >= ROWS ? -1 : rr * COLS + c;
    };
    canvas.addEventListener('click', (e) => {
      const j = cellAt(e); // 触屏上没有 pointermove，直接按点击位置算
      if (j < 0) return;
      st.neuron = j;
      app.sfx.click();
      app.go(6, e);
    });

    $$('.mp-tok', el).forEach((b) => b.addEventListener('click', () => {
      st.focus = Number(b.dataset.i);
      $$('.mp-tok', el).forEach((x) => x.classList.toggle('sel', x === b));
      acts = actsOf(st.focus);
      reveal = 1;
      tween(acts);
      ({ top } = renderSide());
      app.sfx.click();
    }));

    let { strong, top } = renderSide();
    const off = onResize($('.mp-canvas-wrap', el), draw);
    const timer = setTimeout(() => app.discover('sparse'), 2400);

    app.say(`现在经过的是 ${tokHTML(tokens[st.focus], 'mini')}。3072 个神经元里，只有 <b>${strong} 个</b>明显亮了，其余大多在沉默。<em>换一个词元</em>看看哪些神经元被点亮，或者<em>点击一个亮点</em>钻进去。`);
    app.setNext('进入最亮的神经元 <b>↓</b>', () => { st.neuron = top[0].j; app.go(6, canvas); });

    return { destroy() { cancelAnimationFrame(raf); off(); clearTimeout(timer); } };
  },
};
