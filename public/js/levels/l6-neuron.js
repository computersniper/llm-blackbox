import { CFG, neuronPre, ACTS, CONCEPT_BY_ID, embedding, rngFor, gaussian } from '../model.js';
import { $, $$, esc, tokHTML, diverge, signed, svgEl, reducedMotion } from '../ui.js';

const K = 12; // 展示出来的输入个数
const FN_NOTE = {
  GELU: 'GELU：GPT-2、BERT 用的激活函数，负数被压到接近 0，但不是一刀切。',
  ReLU: 'ReLU：最简单的弯折，负数全部归零。',
  SiLU: 'SiLU（Swish）：Llama、Qwen 等模型在门控结构 SwiGLU 里用它。',
  Linear: '没有激活函数：输出就是 z 本身。这样的网络叠多少层，都只等于一次矩阵乘法。',
};
const FNS = { ...ACTS, Linear: (x) => x };

// 激活函数图
const PX0 = -4, PX1 = 4.5, PY0 = -1, PY1 = 4.5;
const PW = 420, PH = 300, PAD = 34;
const sx = (x) => PAD + ((x - PX0) / (PX1 - PX0)) * (PW - PAD - 12);
const sy = (y) => PH - PAD - ((y - PY0) / (PY1 - PY0)) * (PH - PAD - 14);
const curvePath = (f) => {
  let d = '';
  for (let i = 0; i <= 160; i++) {
    const x = PX0 + ((PX1 - PX0) * i) / 160;
    const y = Math.max(PY0 - 0.4, Math.min(PY1 + 0.4, f(x)));
    d += `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`;
  }
  return d;
};

export default {
  key: 'neuron',
  name: '神经元内部',
  en: 'ARITHMETIC',
  scale: '768 次乘加',

  mount(el, app) {
    const st = app.state;
    const { tokens } = app.model;
    if (st.focus == null || st.focus >= tokens.length) st.focus = tokens.length - 1;
    const t = tokens[st.focus];
    const L = st.layer;
    const pre = neuronPre(t, L);
    if (st.neuron == null) {
      let best = 0;
      for (let j = 1; j < pre.length; j++) if (pre[j] > pre[best]) best = j;
      st.neuron = best;
    }
    const j = st.neuron;
    const concept = CONCEPT_BY_ID.get(j);
    const zTarget = pre[j];
    const x = Array.from(embedding(t).slice(0, K));
    const r = rngFor('w', L, j);
    const w0 = x.map((xi) => +(0.11 * gaussian(r) + (zTarget > 0.5 ? 0.07 * Math.sign(xi) : -0.03 * Math.sign(xi))).toFixed(3));
    const w = w0.slice();
    const b = +(-0.12 + 0.06 * gaussian(r)).toFixed(3);
    const rest = zTarget - w0.reduce((s, wi, i) => s + wi * x[i], 0) - b;
    let fn = 'GELU';

    el.innerHTML = `
      <div class="lv-head rv">
        <div>
          <div class="eyebrow">Depth −06 · Inside neuron #${j} · Layer ${L}</div>
          <h2>一个神经元，只做一件事</h2>
          <p>把 768 个输入各乘一个<b>权重</b>，全部加起来，加上<b>偏置</b>，再过一个弯曲的<b>激活函数</b>。就这样。整个大模型，就是几十亿次这样的运算。</p>
        </div>
        <div class="nr-who">${tokHTML(t)} <span class="dimmed">经过</span> <span class="mono acc">#${j}</span>${concept ? `<span class="nr-concept">${esc(concept.label)}</span>` : ''}</div>
      </div>
      <div class="nr-grid">
        <section class="panel nr-calc rv">
          <div class="panel-h"><h3>逐项相乘，再相加</h3><span class="eyebrow">拖动权重试试</span></div>
          <div class="nr-head mono"><span>输入 x</span><span></span><span>权重 w（可拖动）</span><span></span><span>乘积 x·w</span></div>
          <div class="nr-rows">${x.map((xi, i) => `
            <div class="nr-row" data-i="${i}" style="--i:${i}">
              <span class="nr-x mono"><sub>${i}</sub><b style="color:${diverge(xi, 1.2)}">${signed(xi, 3)}</b></span>
              <span class="nr-op">×</span>
              <span class="nr-w">
                <input type="range" min="-0.4" max="0.4" step="0.001" value="${w[i]}" aria-label="权重 ${i}">
                <button type="button" class="nr-wv mono portal" title="潜入这个数字的比特">${signed(w[i], 3)}</button>
              </span>
              <span class="nr-op">=</span>
              <span class="nr-prod"><i></i></span>
              <span class="nr-pv mono"></span>
            </div>`).join('')}
          </div>
          <div class="nr-sum">
            <div class="nr-srow"><span>其余 ${CFG.dModel - K} 项之和</span><b class="mono nr-rest">${signed(rest, 3)}</b></div>
            <div class="nr-srow"><span>偏置 b</span><b class="mono">${signed(b, 3)}</b></div>
            <div class="nr-srow total"><span>z = Σ x·w + b</span><b class="mono nr-z">0.000</b></div>
          </div>
        </section>
        <section class="panel nr-act rv">
          <div class="panel-h"><h3>再弯一下</h3>
            <div class="nr-fns">${Object.keys(FN_NOTE).map((k) => `<button type="button" class="chip nr-fn ${k === fn ? 'active' : ''}" data-f="${k}">${k === 'Linear' ? '不弯' : k}</button>`).join('')}</div>
          </div>
          <svg class="nr-plot" viewBox="0 0 ${PW} ${PH}" role="img" aria-label="激活函数曲线"></svg>
          <p class="nr-fnote hint-line"></p>
          <div class="nr-out">
            <div><span class="eyebrow">输出 a = f(z)</span><b class="mono nr-a">0.000</b></div>
            <p>这个数会乘上 W<sub>out</sub> 里的一行（768 个权重），写回残差流。<span class="nr-verdict"></span></p>
          </div>
        </section>
      </div>`;

    /* ---------- 图 ---------- */
    const svg = $('.nr-plot', el);
    const grid = svgEl('g', { class: 'nr-gridlines' });
    for (let gx = -4; gx <= 4; gx++) grid.append(svgEl('line', { x1: sx(gx), y1: sy(PY0), x2: sx(gx), y2: sy(PY1), class: gx === 0 ? 'axis' : '' }));
    for (let gy = -1; gy <= 4; gy++) grid.append(svgEl('line', { x1: sx(PX0), y1: sy(gy), x2: sx(PX1), y2: sy(gy), class: gy === 0 ? 'axis' : '' }));
    for (let gx = -4; gx <= 4; gx += 2) { const tx = svgEl('text', { x: sx(gx), y: sy(PY0) + 16, 'text-anchor': 'middle' }); tx.textContent = gx; grid.append(tx); }
    for (let gy = 0; gy <= 4; gy += 2) { const ty = svgEl('text', { x: sx(PX0) - 8, y: sy(gy) + 4, 'text-anchor': 'end' }); ty.textContent = gy; grid.append(ty); }
    svg.append(grid);
    const curves = {};
    for (const k of Object.keys(FNS)) {
      curves[k] = svgEl('path', { d: curvePath(FNS[k]), class: `nr-curve ${k === fn ? 'on' : ''}` });
      svg.append(curves[k]);
    }
    const guideV = svgEl('line', { class: 'nr-guide' });
    const guideH = svgEl('line', { class: 'nr-guide' });
    const dot = svgEl('circle', { r: 6, class: 'nr-dot' });
    const dotLbl = svgEl('text', { class: 'nr-dotlbl' });
    svg.append(guideV, guideH, dot, dotLbl);

    const zOf = () => w.reduce((s, wi, i) => s + wi * x[i], 0) + rest + b;
    const placeDot = (z) => {
      const a = FNS[fn](z);
      const zc = Math.max(PX0, Math.min(PX1, z)), ac = Math.max(PY0, Math.min(PY1, a));
      const X = sx(zc), Y = sy(ac);
      dot.setAttribute('cx', X); dot.setAttribute('cy', Y);
      guideV.setAttribute('x1', X); guideV.setAttribute('x2', X); guideV.setAttribute('y1', sy(0)); guideV.setAttribute('y2', Y);
      guideH.setAttribute('x1', sx(0)); guideH.setAttribute('x2', X); guideH.setAttribute('y1', Y); guideH.setAttribute('y2', Y);
      dotLbl.setAttribute('x', X + (X > PW - 150 ? -12 : 12));
      dotLbl.setAttribute('y', Y - 10);
      dotLbl.setAttribute('text-anchor', X > PW - 150 ? 'end' : 'start');
      dotLbl.textContent = `z=${z.toFixed(2)} → a=${a.toFixed(2)}`;
      return a;
    };

    const render = (zShown = zOf()) => {
      $$('.nr-row', el).forEach((row, i) => {
        const p = x[i] * w[i];
        const bar = $('.nr-prod i', row);
        const pct = Math.min(50, Math.abs(p) / 0.25 * 50);
        bar.style.width = `${pct}%`;
        bar.style.left = p >= 0 ? '50%' : `${50 - pct}%`;
        bar.style.background = diverge(p, 0.12);
        $('.nr-pv', row).textContent = signed(p, 3);
        $('.nr-wv', row).textContent = signed(w[i], 3);
      });
      $('.nr-z', el).textContent = signed(zShown, 3);
      const a = placeDot(zShown);
      const aEl = $('.nr-a', el);
      aEl.textContent = signed(a, 3);
      aEl.style.color = a > 0.5 ? 'var(--amber)' : a < 0 ? 'var(--blue)' : 'var(--ink2)';
      $('.nr-verdict', el).textContent = a > 1 ? ' 它被强烈激活了。' : a > 0.2 ? ' 它微微亮起。' : ' 它几乎沉默。';
      $('.nr-fnote', el).textContent = FN_NOTE[fn];
    };

    // 入场：逐项累加
    let raf = 0;
    const timers = [];
    if (!reducedMotion) {
      el.classList.add('nr-intro');
      const zEnd = zOf();
      const t0 = performance.now() + 700;
      const step = (now) => {
        const k = Math.max(0, Math.min(1, (now - t0) / 1800));
        const e = 1 - Math.pow(1 - k, 3);
        render(zEnd * e);
        if (k < 1) raf = requestAnimationFrame(step);
        else el.classList.remove('nr-intro');
      };
      raf = requestAnimationFrame(step);
      $$('.nr-row', el).forEach((row, i) => timers.push(setTimeout(() => { row.classList.add('lit'); if (i % 3 === 0) app.sfx.tick(); }, 700 + i * 110)));
    } else render();

    $$('.nr-row', el).forEach((row, i) => {
      const input = $('input', row);
      input.addEventListener('input', () => {
        w[i] = Number(input.value);
        cancelAnimationFrame(raf);
        render();
      });
      $('.nr-wv', row).addEventListener('click', (e) => {
        st.weight = { value: w[i], k: i, neuron: j };
        app.sfx.click();
        app.go(7, e.currentTarget);
      });
    });

    $$('.nr-fn', el).forEach((btn) => btn.addEventListener('click', () => {
      fn = btn.dataset.f;
      $$('.nr-fn', el).forEach((b2) => b2.classList.toggle('active', b2 === btn));
      Object.entries(curves).forEach(([k, c]) => c.classList.toggle('on', k === fn));
      cancelAnimationFrame(raf);
      render();
      app.sfx.click();
      app.discover('nonlinear');
      if (fn === 'Linear') app.say('去掉弯曲，输出就等于 z。<b>线性函数套线性函数还是线性函数</b>：没有这个小小的弯，一百层网络的表达能力和一层没有区别。');
    }));

    app.say(`这是神经元 <b>#${j}</b> 的全部工作：<b>768 次乘法、一次求和、一个弯曲</b>。<em>拖动权重</em>看输出怎么变，<em>切换激活函数</em>看看“弯”的作用；<em>点击任意一个权重的数值</em>，继续往下潜。`);
    app.setNext('潜入一个权重 <b>↓</b>', () => {
      st.weight = { value: w[0], k: 0, neuron: j };
      app.go(7, $('.nr-wv', el));
    });

    return { destroy() { cancelAnimationFrame(raf); timers.forEach(clearTimeout); } };
  },
};
