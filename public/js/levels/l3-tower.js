import { CFG, attention, headType } from '../model.js';
import { $, $$, esc, tokHTML, tokPlain, hueOf, fmtPct, fmtNum, svgEl, reducedMotion } from '../ui.js';

const NL = CFG.layers;
const X0 = 58, W = 400, SK = 104, DEP = 62, SP = 45, BASE = 616;
const VW = 640, VH = 720;
const yOf = (L) => BASE - L * SP;
const LANE_TOP = 34, LANE_BOT = 668;

export default {
  key: 'tower',
  name: '层塔',
  en: 'THE TOWER',
  scale: '12 层 · 残差流',

  mount(el, app) {
    const { tokens, lens } = app.model;
    const st = app.state;
    const T = tokens.length;
    const laneX = (i) => X0 + SK / 2 + ((i + 0.5) * W) / T;
    let lensL = 0;
    const seen = new Set();

    el.innerHTML = `
      <div class="lv-head rv">
        <div>
          <div class="eyebrow">Depth −03 · The Tower</div>
          <h2>十二层，一样的结构</h2>
          <p>向量被送进 12 个结构相同的<b>变换器块</b>。每个词元沿着自己的竖线往上走，这条线叫<b>残差流</b>：每经过一层，就被加上一点新信息。</p>
        </div>
        <div class="tw-stats">
          <div class="stat"><b>${NL}</b><span>层</span></div>
          <div class="stat"><b>${fmtNum(CFG.paramsPerLayer)}</b><span>每层参数</span></div>
          <div class="stat"><b>${T}</b><span>条残差流</span></div>
        </div>
      </div>
      <div class="tw-grid">
        <section class="panel tw-stage rv">
          <svg class="tw-svg" viewBox="0 0 ${VW} ${VH}" role="img" aria-label="12 层变换器堆叠示意图，点击任意一层进入"></svg>
          <div class="tw-caption hint-line">悬停一层可以预览透镜；<b>点击任意一层</b>，钻进去看看。</div>
        </section>
        <section class="panel tw-lens rv">
          <div class="panel-h"><h3>逻辑透镜</h3><span class="eyebrow">Logit lens</span></div>
          <div class="tw-lens-body">
            <p class="tw-intro">如果让模型在<b>第 <span class="tw-L">0</span> 层</b>就停下，直接开口，它会说：</p>
            <div class="tw-guess"><span class="tw-guess-tok"></span><span class="tw-guess-p mono"></span></div>
            <div class="tw-bars"></div>
            <label class="tw-slider">
              <span class="eyebrow">扫描层</span>
              <input type="range" class="tw-range" min="0" max="${NL - 1}" step="1" value="0" aria-label="选择透镜所在的层">
              <span class="tw-ticks mono"><span>0</span><span>6</span><span>11</span></span>
            </label>
            <div class="tw-strip">${lens.map((row, L) => `
              <button type="button" class="tw-srow" data-l="${L}" style="--p:${row[0].p}">
                <span class="mono">L${String(L).padStart(2, '0')}</span>${tokHTML(row[0].text, 'mini')}<i></i><span class="mono">${fmtPct(row[0].p)}</span>
              </button>`).reverse().join('')}
            </div>
          </div>
        </section>
      </div>`;

    /* ---------- 塔 ---------- */
    const svg = $('.tw-svg', el);
    const gSlabs = svgEl('g');
    const gLanes = svgEl('g');
    const gArcs = svgEl('g', { class: 'tw-arcs' });
    const gNodes = svgEl('g');
    const gParts = svgEl('g');
    const gLabels = svgEl('g');
    svg.append(gSlabs, gLanes, gArcs, gNodes, gParts, gLabels);

    // 每层：最后一个词元对其他词元的平均注意力
    const mix = [];
    for (let L = 0; L < NL; L++) {
      const acc = new Array(T).fill(0);
      for (let h = 0; h < CFG.heads; h++) {
        const { W: A } = attention(tokens, L, h);
        A[T - 1].forEach((v, j) => { acc[j] += v / CFG.heads; });
      }
      mix.push(acc);
    }

    const slabs = [];
    for (let L = 0; L < NL; L++) {
      const y = yOf(L);
      const g = svgEl('g', { class: 'tw-slab', 'data-l': L, tabindex: 0, role: 'button', 'aria-label': `第 ${L} 层` });
      g.append(svgEl('polygon', { class: 'tw-side', points: `${X0},${y} ${X0 + W},${y} ${X0 + W},${y + 7} ${X0},${y + 7}` }));
      g.append(svgEl('polygon', { class: 'tw-side r', points: `${X0 + W},${y} ${X0 + W + SK},${y - DEP} ${X0 + W + SK},${y - DEP + 7} ${X0 + W},${y + 7}` }));
      g.append(svgEl('polygon', { class: 'tw-top', points: `${X0},${y} ${X0 + W},${y} ${X0 + W + SK},${y - DEP} ${X0 + SK},${y - DEP}` }));
      const lbl = svgEl('text', { x: X0 + W + SK + 14, y: y - DEP / 2 + 5, class: 'tw-lbl' });
      lbl.textContent = `L${String(L).padStart(2, '0')}`;
      g.append(lbl);
      gSlabs.append(g);
      slabs.push(g);
    }

    tokens.forEach((t, i) => {
      const x = laneX(i);
      const last = i === T - 1;
      const lane = svgEl('line', { x1: x, y1: LANE_BOT, x2: x, y2: LANE_TOP + (last ? 0 : 14), class: `tw-lane ${last ? 'last' : ''}`, style: `--h:${hueOf(t.text)}` });
      gLanes.append(lane);
      const hit = svgEl('line', { x1: x, y1: LANE_BOT, x2: x, y2: LANE_TOP, class: 'tw-lane-hit' });
      gLanes.append(hit);
      app.tip.bind(hit, () => {
        app.discover('residual');
        return `<span class="k">残差流 · 词元 #${i}</span><b>${esc(tokPlain(t.text))}</b> 的 768 维向量从底部进入，每层读取它、再加上一点东西。${last ? '<br><b>它是最后一个词元</b>：它在顶层的状态，决定下一个词。' : ''}`;
      });
      for (let L = 0; L < NL; L++) {
        gNodes.append(svgEl('circle', { cx: x, cy: yOf(L) - DEP / 2, r: last ? 3.6 : 2.6, class: `tw-node ${last ? 'last' : ''}`, 'data-l': L, style: `--h:${hueOf(t.text)}` }));
      }
      const every = Math.ceil(T / 14);
      if (i % every !== 0 && i !== T - 1) return;
      const lab = svgEl('text', {
        x, y: LANE_BOT + 18, class: 'tw-tok', style: `--h:${hueOf(t.text)}`,
        'text-anchor': T > 8 ? 'end' : 'middle',
        transform: T > 8 ? `rotate(-40 ${x} ${LANE_BOT + 18})` : '',
      });
      lab.textContent = tokPlain(t.text);
      gLabels.append(lab);
    });
    const cap = svgEl('text', { x: laneX(T - 1), y: LANE_TOP - 10, class: 'tw-cap', 'text-anchor': 'middle' });
    cap.textContent = '→ 下一个词';
    gLabels.append(cap);

    // 注意力弧线：最后一个词元在每一层向谁"取材"
    for (let L = 0; L < NL; L++) {
      const yy = yOf(L) - DEP / 2;
      const xs = laneX(T - 1);
      mix[L].map((w, j) => ({ w, j })).filter((o) => o.j !== T - 1).sort((a, b) => b.w - a.w).slice(0, 2).forEach(({ w, j }) => {
        const xe = laneX(j);
        const bend = Math.min(26, 8 + Math.abs(xs - xe) * 0.12);
        gArcs.append(svgEl('path', { d: `M${xs},${yy} Q${(xs + xe) / 2},${yy - bend} ${xe},${yy}`, class: 'tw-arc', 'data-l': L, style: `--w:${Math.min(1, w * 2.2).toFixed(3)}` }));
      });
    }

    // 粒子沿残差流上升
    const parts = [];
    tokens.forEach((t, i) => {
      for (let k = 0; k < 2; k++) {
        const c = svgEl('circle', { cx: laneX(i), cy: LANE_BOT, r: i === T - 1 ? 2.6 : 1.8, class: 'tw-part', style: `--h:${hueOf(t.text)}` });
        gParts.append(c);
        parts.push({ c, phase: (i * 0.137 + k * 0.5) % 1, speed: 0.16 + (i % 3) * 0.02 });
      }
    });
    let raf = 0;
    const t0 = performance.now();
    const tick = (now) => {
      const s = (now - t0) / 1000;
      for (const p of parts) {
        const f = (p.phase + s * p.speed) % 1;
        p.c.setAttribute('cy', LANE_BOT - f * (LANE_BOT - LANE_TOP));
        p.c.style.opacity = Math.sin(f * Math.PI).toFixed(3);
      }
      raf = requestAnimationFrame(tick);
    };
    if (!reducedMotion) raf = requestAnimationFrame(tick);

    /* ---------- 透镜 ---------- */
    const rangeEl = $('.tw-range', el);
    const renderLens = (L, preview = false) => {
      const row = lens[L];
      $('.tw-L', el).textContent = L;
      $('.tw-guess-tok', el).innerHTML = tokHTML(row[0].text, 'big');
      $('.tw-guess-p', el).textContent = fmtPct(row[0].p);
      $('.tw-bars', el).innerHTML = row.map((c, k) => `
        <div class="tw-bar ${k === 0 ? 'top' : ''}">${tokHTML(c.text, 'mini')}<span class="tw-bar-t"><i style="width:${(c.p * 100).toFixed(1)}%"></i></span><span class="mono">${fmtPct(c.p)}</span></div>`).join('');
      $$('.tw-srow', el).forEach((r) => r.classList.toggle('on', Number(r.dataset.l) === L));
      slabs.forEach((s, k) => s.classList.toggle('lens', k === L));
      $$('.tw-arc', el).forEach((a) => a.classList.toggle('on', Number(a.dataset.l) === L));
      $$('.tw-node', el).forEach((n) => n.classList.toggle('on', Number(n.dataset.l) === L));
      if (!preview) {
        seen.add(L);
        if (seen.size >= 9 && seen.has(0) && seen.has(NL - 1)) {
          app.discover('lens');
          if (!st.lensSaid) {
            st.lensSaid = true;
            const early = lens[1][0].text, late = lens[NL - 1][0].text;
            app.say(`看到了吗？浅层还在复述眼前的「${esc(tokPlain(early))}」，到了深层，才锁定「${esc(tokPlain(late))}」。<b>答案是一层层“长”出来的</b>，而不是某一层突然想到的。<em>点击任意一层</em>，钻进去看看。`);
          }
        }
      }
    };
    rangeEl.addEventListener('input', () => { lensL = Number(rangeEl.value); renderLens(lensL); app.sfx.tick(); });
    $$('.tw-srow', el).forEach((r) => r.addEventListener('click', () => {
      lensL = Number(r.dataset.l);
      rangeEl.value = lensL;
      renderLens(lensL);
      app.sfx.tick();
    }));

    slabs.forEach((g, L) => {
      g.addEventListener('pointerenter', (e) => {
        renderLens(L, true);
        const types = Array.from({ length: CFG.heads }, (_, h) => headType(L, h));
        const uniq = [...new Set(types)].length;
        app.tip.show(`<span class="k">第 ${L} 层 · Transformer Block</span>12 个注意力头（${uniq} 种“性格”）+ 3072 个神经元<br>${fmtNum(CFG.paramsPerLayer)} 个参数<br><span class="v">点击进入 ↓</span>`, e.clientX, e.clientY);
      });
      g.addEventListener('pointermove', (e) => app.tip.move(e.clientX, e.clientY));
      g.addEventListener('pointerleave', () => { app.tip.hide(); renderLens(lensL, true); });
      const enter = (e) => { st.layer = L; st.head = null; app.sfx.click(); app.go(4, e); };
      g.addEventListener('click', enter);
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter') enter(e); });
    });

    // 入场：一次前向传播的光波从底层扫到顶层
    const timers = [];
    slabs.forEach((s, L) => timers.push(setTimeout(() => { s.classList.add('lit'); setTimeout(() => s.classList.remove('lit'), 600); }, reducedMotion ? 0 : 900 + L * 110)));
    timers.push(setTimeout(() => renderLens(lensL, true), 10));

    app.say(`向量沿着 12 层一路往上。每层里都有<b>注意力</b>（词元互相交流）和<b>前馈网络</b>（各自消化）。<em>拖动右侧的透镜</em>，从底扫到顶，看模型的“想法”怎样成形。`);
    app.setNext(`进入第 ${st.layer} 层 <b>↓</b>`, () => app.go(4, slabs[st.layer]));

    return { destroy() { cancelAnimationFrame(raf); timers.forEach(clearTimeout); } };
  },
};
