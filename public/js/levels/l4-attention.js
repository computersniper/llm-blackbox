import { CFG, attention, headType, findHead, HEAD_TYPES } from '../model.js';
import { $, $$, esc, tokHTML, tokPlain, hueOf, fmtPct, svgEl, fitCanvas, onResize } from '../ui.js';

const REPEAT_EXAMPLE = 'Harry Potter 是一个巫师。Harry';

function heatColor(w) {
  // 深蓝黑 → 青 → 白
  const t = Math.pow(Math.min(1, w), 0.6);
  const a = [14, 20, 36], b = [94, 240, 212], c = [255, 255, 255];
  const mix = (p, q, k) => p.map((v, i) => Math.round(v + (q[i] - v) * k));
  return t < 0.75 ? mix(a, b, t / 0.75) : mix(b, c, (t - 0.75) / 0.25);
}

function paintHeat(canvas, W, { query = -1, hover = null, mini = false } = {}) {
  const { g, w, h } = fitCanvas(canvas);
  const T = W.length;
  const cs = Math.min(w, h) / T;
  g.clearRect(0, 0, w, h);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      const x = j * cs, y = i * cs;
      if (j > i) {
        if (!mini) {
          g.fillStyle = 'rgba(150,170,210,.035)';
          g.fillRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
        }
        continue;
      }
      const [r, gg, b] = heatColor(W[i][j]);
      g.fillStyle = `rgb(${r},${gg},${b})`;
      g.fillRect(x + (mini ? 0 : 0.5), y + (mini ? 0 : 0.5), cs - (mini ? 0 : 1), cs - (mini ? 0 : 1));
    }
  }
  if (!mini) {
    // 未来区域的斜线
    g.save();
    g.beginPath();
    g.moveTo(cs, 0); g.lineTo(T * cs, 0); g.lineTo(T * cs, (T - 1) * cs); g.closePath();
    g.clip();
    g.strokeStyle = 'rgba(150,170,210,.1)';
    g.lineWidth = 1;
    for (let k = -T * cs; k < T * cs * 2; k += 7) { g.beginPath(); g.moveTo(k, 0); g.lineTo(k + T * cs, T * cs); g.stroke(); }
    g.restore();
    if (query >= 0) {
      g.strokeStyle = 'rgba(255,255,255,.85)';
      g.lineWidth = 1.5;
      g.strokeRect(0.75, query * cs + 0.75, (query + 1) * cs - 1.5, cs - 1.5);
    }
    if (hover) {
      g.strokeStyle = '#ffb65c';
      g.lineWidth = 2;
      g.strokeRect(hover[1] * cs + 1, hover[0] * cs + 1, cs - 2, cs - 2);
    }
  }
  return cs;
}

export default {
  key: 'attention',
  name: '注意力',
  en: 'ATTENTION',
  scale: '12 个注意力头',

  mount(el, app) {
    const st = app.state;
    let query = null;
    let preview = null;
    let offResize = () => {};

    const build = () => {
      const { tokens, hasRepeat } = app.model;
      const T = tokens.length;
      const L = st.layer;
      if (st.head == null) {
        const pick = hasRepeat ? findHead(L, 'induct') : findHead(L, 'sem');
        st.head = pick >= 0 ? pick : Math.max(0, findHead(L, 'prev'));
      }
      if (query == null || query >= T) query = T - 1;
      const heads = Array.from({ length: CFG.heads }, (_, h) => attention(tokens, L, h));

      el.innerHTML = `
        <div class="lv-head rv">
          <div>
            <div class="eyebrow">Depth −04 · Attention · Layer ${L}</div>
            <h2>词元之间的对话</h2>
            <p>在第 ${L} 层，每个词元都会回头看前面的词：<b>谁和我有关？</b>答案是一组加起来等于 100% 的权重，决定它从谁那里汲取信息。12 个头各看各的。</p>
          </div>
          <div class="at-layer">
            <button type="button" class="btn small at-prev" aria-label="上一层" ${L === 0 ? 'disabled' : ''}>◀</button>
            <span class="mono">第 <b>${L}</b> 层 / 12</span>
            <button type="button" class="btn small at-next" aria-label="下一层" ${L === CFG.layers - 1 ? 'disabled' : ''}>▶</button>
          </div>
        </div>

        <div class="at-anat rv" aria-label="一个变换器块的结构">
          <span class="an-n">残差流</span><i></i>
          <span class="an-n">LayerNorm</span><i></i>
          <span class="an-n on">多头注意力</span><i></i>
          <span class="an-n plus">⊕</span><i></i>
          <span class="an-n">LayerNorm</span><i></i>
          <button type="button" class="an-n an-mlp portal">前馈网络 <b>↓</b></button><i></i>
          <span class="an-n plus">⊕</span><i></i>
          <span class="an-n">第 ${L + 1 < 12 ? L + 1 : '输出'}${L + 1 < 12 ? ' 层' : ''}</span>
        </div>

        <div class="at-grid">
          <section class="panel at-main rv">
            <div class="panel-h"><h3 class="at-title"></h3><span class="eyebrow">悬停 / 点击词元，切换“提问者”</span></div>
            <div class="at-arcs">
              <svg class="at-svg"></svg>
              <div class="at-toks">${tokens.map((t, i) => `<button type="button" class="tok at-tok" style="--h:${hueOf(t.text)}" data-i="${i}">${esc(tokPlain(t.text))}</button>`).join('')}</div>
            </div>
            <div class="at-read"></div>
            <div class="at-qkv">
              <div><b class="q">Q</b> 查询：我在找什么？</div>
              <div><b class="k">K</b> 键：我身上有什么？</div>
              <div><b class="v">V</b> 值：我能提供什么？</div>
              <div class="at-formula mono">权重 = softmax(Q·Kᵀ / √64)　输出 = Σ 权重 × V</div>
            </div>
          </section>
          <section class="panel at-side rv">
            <div class="panel-h"><h3>注意力矩阵</h3><span class="eyebrow">行 = 提问者 · 列 = 被看的</span></div>
            <div class="at-heat-wrap"><canvas class="at-heat"></canvas></div>
            <div class="at-headinfo"></div>
          </section>
        </div>

        <section class="panel at-heads rv">
          <div class="panel-h"><h3>第 ${L} 层的 12 个头</h3><span class="eyebrow">每个头学会了一种“看”的方式</span></div>
          <div class="at-hgrid">${heads.map((hd, h) => `
            <button type="button" class="at-hcard ${h === st.head ? 'on' : ''}" data-h="${h}">
              <canvas class="at-mini"></canvas>
              <span class="at-hname"><span class="mono">H${h}</span> ${HEAD_TYPES[hd.type].name}</span>
            </button>`).join('')}
          </div>
        </section>`;

      const svg = $('.at-svg', el);
      const wrap = $('.at-arcs', el);
      const chips = $$('.at-tok', el);
      const heat = $('.at-heat', el);
      let hoverCell = null;

      const cur = () => heads[st.head];
      const qi = () => (preview != null ? preview : query);

      const drawArcs = () => {
        const W = cur().W;
        const q = qi();
        const toks = $('.at-toks', el);
        const width = Math.max(toks.scrollWidth, wrap.clientWidth);
        const H = 150;
        svg.style.width = `${width}px`;
        svg.style.height = `${H}px`;
        svg.setAttribute('viewBox', `0 0 ${width} ${H}`);
        svg.innerHTML = '';
        const centers = chips.map((c) => c.offsetLeft - toks.offsetLeft + c.offsetWidth / 2);
        chips.forEach((c, i) => {
          c.classList.toggle('q', i === q);
          c.classList.toggle('future', i > q);
          const w = i <= q ? W[q][i] : 0;
          c.style.setProperty('--w', w.toFixed(3));
        });
        const xq = centers[q];
        const order = W[q].map((w, j) => ({ w, j })).filter((o) => o.j <= q).sort((a, b) => a.w - b.w);
        for (const { w, j } of order) {
          if (w < 0.015) continue;
          const xk = centers[j];
          const y = H - 2;
          let d;
          if (j === q) d = `M${xq - 7},${y} C${xq - 26},${y - 46} ${xq + 26},${y - 46} ${xq + 7},${y}`;
          else {
            const hgt = Math.min(H - 16, 26 + Math.abs(xq - xk) * 0.38);
            d = `M${xq},${y} C${xq},${y - hgt} ${xk},${y - hgt} ${xk},${y}`;
          }
          svg.append(svgEl('path', { d, class: 'at-arc', style: `--w:${w.toFixed(3)}` }));
          if (w >= 0.1) {
            const hgt = j === q ? 36 : Math.min(H - 16, 26 + Math.abs(xq - xk) * 0.38) * 0.75;
            const lbl = svgEl('text', { x: j === q ? xq : (xq + xk) / 2, y: y - hgt - 4, class: 'at-pct', 'text-anchor': 'middle' });
            lbl.textContent = fmtPct(w);
            svg.append(lbl);
          }
        }
        const best = W[q].map((w, j) => ({ w, j })).filter((o) => o.j <= q).sort((a, b) => b.w - a.w)[0];
        $('.at-read', el).innerHTML = `<span class="eyebrow">解读</span> 在 <b>H${st.head} ${HEAD_TYPES[cur().type].name}</b> 眼里，${tokHTML(tokens[q], 'mini')} 把 <b class="acc">${fmtPct(best.w)}</b> 的注意力给了 ${tokHTML(tokens[best.j], 'mini')}${best.j === q ? '（它自己）' : best.j === 0 && q > 0 ? '（开头的词元）' : ''}。`;
      };

      const drawHeat = () => {
        const wrapEl = $('.at-heat-wrap', el);
        const size = Math.min(wrapEl.clientWidth, 340);
        heat.style.width = heat.style.height = `${size}px`;
        paintHeat(heat, cur().W, { query: qi(), hover: hoverCell });
      };

      const drawInfo = () => {
        const ht = HEAD_TYPES[cur().type];
        $('.at-title', el).innerHTML = `H${st.head} · ${ht.name} <small class="mono">${ht.en}</small>`;
        let extra = '';
        if (cur().type === 'induct' && !hasRepeat) {
          extra = `<div class="at-note">这句话里没有重复出现的词，归纳头暂时“无事可做”，只好停在开头。<button type="button" class="btn small at-repeat">换一句有重复的话</button></div>`;
        }
        $('.at-headinfo', el).innerHTML = `<p><b>${ht.name}</b>：${ht.desc}</p>${extra}`;
        $('.at-repeat', el)?.addEventListener('click', () => {
          app.setPrompt(REPEAT_EXAMPLE, { base: true });
          query = null;
          app.sfx.click();
          build();
          app.say('新句子里 <b>Harry</b> 出现了两次。看最后一个 Harry：归纳头找到上一次的 Harry，然后看向紧跟其后的 <b>Potter</b>，于是模型会预测“Potter”。');
        });
      };

      const drawMinis = () => $$('.at-mini', el).forEach((c, h) => paintHeat(c, heads[h].W, { mini: true }));

      const selectHead = (h) => {
        st.head = h;
        $$('.at-hcard', el).forEach((c) => c.classList.toggle('on', Number(c.dataset.h) === h));
        drawArcs(); drawHeat(); drawInfo();
        const type = heads[h].type;
        if (type === 'sink') {
          app.discover('sink');
          app.say('这是一个<b>汇聚头</b>：几乎所有词元都把注意力“停”在第一个词元上。它不是在读开头，而是<b>没有值得看的东西时</b>，总得把 100% 的权重放在某个地方。');
        } else if (type === 'induct' && hasRepeat) {
          app.discover('induction');
          app.say('<b>归纳头</b>在找“上一次出现当前这个词时，后面跟着什么”。它让模型能<b>照着上文依葫芦画瓢</b>，是上下文学习的关键。');
        } else if (type === 'prev') {
          app.say('<b>前一词头</b>：每个词元都只看紧挨着的前一个。简单，但很重要，很多更复杂的回路都要靠它提供“上一个词是什么”。');
        } else if (type === 'sem') {
          app.say('<b>语义头</b>：注意力落在意思相关的词上。试试悬停不同的词元，看看它们各自在找谁。');
        } else {
          app.say(`<b>${HEAD_TYPES[type].name}</b>：${HEAD_TYPES[type].desc}`);
        }
      };

      chips.forEach((c, i) => {
        c.addEventListener('pointerenter', () => { preview = i; drawArcs(); drawHeat(); });
        c.addEventListener('pointerleave', () => { preview = null; drawArcs(); drawHeat(); });
        c.addEventListener('click', () => { query = i; preview = null; app.sfx.click(); drawArcs(); drawHeat(); });
      });

      heat.addEventListener('pointermove', (e) => {
        const r = heat.getBoundingClientRect();
        const cs = r.width / T;
        const j = Math.floor((e.clientX - r.left) / cs), i = Math.floor((e.clientY - r.top) / cs);
        if (i < 0 || j < 0 || i >= T || j >= T) return;
        hoverCell = [i, j];
        drawHeat();
        if (j > i) {
          app.discover('causal');
          app.tip.show(`<span class="k">未来 · 不可见</span>${esc(tokPlain(tokens[i].text))} 不能看 ${esc(tokPlain(tokens[j].text))}：它在后面，生成时还不存在。`, e.clientX, e.clientY);
        } else {
          app.tip.show(`<span class="k">第 ${i} 行 · 第 ${j} 列</span><b>${esc(tokPlain(tokens[i].text))}</b> → <b>${esc(tokPlain(tokens[j].text))}</b>　<span class="v">${fmtPct(cur().W[i][j])}</span>`, e.clientX, e.clientY);
        }
      });
      heat.addEventListener('pointerleave', () => { hoverCell = null; drawHeat(); app.tip.hide(); });
      heat.addEventListener('click', (e) => {
        const r = heat.getBoundingClientRect();
        const i = Math.floor((e.clientY - r.top) / (r.width / T));
        if (i >= 0 && i < T) { query = i; drawArcs(); drawHeat(); }
      });

      $$('.at-hcard', el).forEach((c) => c.addEventListener('click', () => { app.sfx.click(); selectHead(Number(c.dataset.h)); }));
      $('.at-prev', el).addEventListener('click', () => { st.layer = Math.max(0, L - 1); app.sfx.click(); build(); });
      $('.at-next', el).addEventListener('click', () => { st.layer = Math.min(CFG.layers - 1, L + 1); app.sfx.click(); build(); });
      $('.an-mlp', el).addEventListener('click', (e) => { app.sfx.click(); app.go(5, e.currentTarget); });

      const redraw = () => { drawArcs(); drawHeat(); drawMinis(); };
      // 长句子：先把"提问者"（最后一个词元）滚进视野
      requestAnimationFrame(() => { wrap.scrollLeft = wrap.scrollWidth; });
      offResize();
      offResize = onResize($('.at-main', el), redraw);
      drawInfo();
      requestAnimationFrame(redraw);
      app.setNext('进入前馈网络 <b>↓</b>', () => app.go(5, $('.an-mlp', el)));
    };

    build();
    app.say(`这里是第 ${st.layer} 层的<b>注意力</b>。弧线越粗，关注越多。<em>悬停不同的词元</em>看它们在找谁，<em>在下方切换 12 个头</em>：有的看前一个词，有的盯着开头，有的找意思相近的词。`);

    return { destroy() { offResize(); } };
  },
};
