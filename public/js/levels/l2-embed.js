import { embedding, posEnc, mapPos, MAP_WORDS, tokenize, CATS, CFG } from '../model.js';
import { $, $$, esc, tokHTML, tokPlain, hueOf, divergeRGB, fitCanvas, onResize, svgEl, signed } from '../ui.js';

const D = CFG.dModel;
const VW = 1000, VH = 700;
const toXY = ([x, y]) => [VW / 2 + x * 440, VH / 2 - y * 310];

function vecFor(t, withPos) {
  const e = embedding(t);
  if (!withPos) return e;
  const v = new Float32Array(D);
  for (let i = 0; i < D; i++) v[i] = e[i] + 0.8 * posEnc(t.i, i);
  return v;
}

// 把一个向量画成一条 768 像素宽的条形码（再缩放到画布大小）
function paintStrip(canvas, vec) {
  const { g, w, h } = fitCanvas(canvas);
  const off = document.createElement('canvas');
  off.width = vec.length; off.height = 1;
  const og = off.getContext('2d');
  const img = og.createImageData(vec.length, 1);
  for (let i = 0; i < vec.length; i++) {
    const [r, gg, b] = divergeRGB(vec[i], 1.8);
    img.data.set([r, gg, b, 255], i * 4);
  }
  og.putImageData(img, 0, 0);
  g.imageSmoothingEnabled = false;
  g.clearRect(0, 0, w, h);
  g.drawImage(off, 0, 0, w, h);
}

function paintGrid(canvas, vec, cols = 32) {
  const rows = Math.ceil(vec.length / cols);
  const { g, w, h } = fitCanvas(canvas);
  const cw = w / cols, ch = h / rows;
  g.clearRect(0, 0, w, h);
  for (let i = 0; i < vec.length; i++) {
    const [r, gg, b] = divergeRGB(vec[i], 1.8);
    g.fillStyle = `rgb(${r},${gg},${b})`;
    g.fillRect((i % cols) * cw + 0.5, Math.floor(i / cols) * ch + 0.5, cw - 1, ch - 1);
  }
}

function paintPE(canvas, T) {
  const { g, w, h } = fitCanvas(canvas);
  const dims = 96;
  const cw = w / dims, ch = h / Math.max(T, 1);
  for (let p = 0; p < T; p++) {
    for (let i = 0; i < dims; i++) {
      const [r, gg, b] = divergeRGB(posEnc(p, i * 2, 192), 1);
      g.fillStyle = `rgb(${r},${gg},${b})`;
      g.fillRect(i * cw, p * ch, cw + 0.5, ch - 1);
    }
  }
}

export default {
  key: 'embed',
  name: '嵌入',
  en: 'EMBED',
  scale: '768 维向量',

  mount(el, app) {
    const { tokens } = app.model;
    const st = app.state;
    if (st.focus == null || st.focus >= tokens.length) st.focus = tokens.length - 1;
    let withPos = false;
    const hovered = new Set();

    el.innerHTML = `
      <div class="lv-head rv">
        <div>
          <div class="eyebrow">Depth −02 · Embed</div>
          <h2>每个词元，变成 768 个数字</h2>
          <p>编号本身没有意义。模型拿着编号，去一张 50257 × 768 的<b>嵌入矩阵</b>里取出对应的一行：768 个数字，合起来就是这个词元的“意思”。</p>
        </div>
        <div class="legend-bar"><span>负</span><i></i><span>正</span></div>
      </div>
      <div class="em-grid">
        <section class="panel em-list rv">
          <div class="panel-h"><h3>词元 → 向量</h3><span class="eyebrow">每一条 = 768 个数</span></div>
          <div class="em-rows">${tokens.map((t, i) => `
            <button type="button" class="em-row" data-i="${i}">
              <span class="em-tok">${tokHTML(t, 'mini')}</span>
              <canvas class="em-strip" data-i="${i}"></canvas>
              <span class="em-id mono">#${t.id}</span>
            </button>`).join('')}
          </div>
          <div class="em-detail">
            <div class="em-detail-h"><span class="em-detail-tok"></span><span class="eyebrow">展开成 24 × 32 的网格</span></div>
            <div class="em-detail-body">
              <canvas class="em-cells"></canvas>
              <div class="em-nums mono"></div>
            </div>
          </div>
          <div class="em-pe" hidden>
            <div class="em-detail-h"><span class="eyebrow">位置编码：每个位置一种独特的波纹（前 96 维）</span></div>
            <canvas class="em-pe-c"></canvas>
          </div>
        </section>
        <section class="panel em-map rv">
          <div class="panel-h">
            <h3>语义地图</h3>
            <div class="em-tools">
              <button type="button" class="btn small em-analogy">国王 − 男人 + 女人 = ?</button>
              <button type="button" class="btn small em-pos" aria-pressed="false">位置编码</button>
            </div>
          </div>
          <div class="em-map-wrap"><svg class="em-svg" viewBox="0 0 ${VW} ${VH}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="词向量压缩到二维后的语义地图"></svg><div class="em-eq serif"></div></div>
          <p class="hint-line em-map-note">把 768 维压到 2 维后的示意图。离得越近，意思越像。你的词元是亮点。</p>
        </section>
      </div>`;

    /* ---------- 左侧：条形码 ---------- */
    const rowsEl = $('.em-rows', el);
    const drawStrips = () => $$('.em-strip', el).forEach((c) => paintStrip(c, vecFor(tokens[Number(c.dataset.i)], withPos)));
    const drawDetail = () => {
      const t = tokens[st.focus];
      const v = vecFor(t, withPos);
      $('.em-detail-tok', el).innerHTML = `${tokHTML(t)} <span class="dimmed mono">第 ${t.i} 个词元${withPos ? ' · 已叠加位置编码' : ''}</span>`;
      paintGrid($('.em-cells', el), v);
      $('.em-nums', el).innerHTML = `[${Array.from(v.slice(0, 6)).map((x) => `<span style="color:rgb(${divergeRGB(x, 1.2).join(',')})">${signed(x, 3)}</span>`).join(', ')}, <span class="dimmed">… 还有 762 个</span>]`;
      $$('.em-row', el).forEach((r) => r.classList.toggle('sel', Number(r.dataset.i) === st.focus));
      markMap();
    };
    rowsEl.addEventListener('click', (e) => {
      const r = e.target.closest('.em-row');
      if (!r) return;
      st.focus = Number(r.dataset.i);
      app.sfx.click();
      drawDetail();
    });
    $$('.em-strip', el).forEach((c) => {
      c.addEventListener('pointermove', (e) => {
        const rect = c.getBoundingClientRect();
        const k = Math.max(0, Math.min(D - 1, Math.floor(((e.clientX - rect.left) / rect.width) * D)));
        const v = vecFor(tokens[Number(c.dataset.i)], withPos)[k];
        app.tip.show(`<span class="k">第 ${k} 维</span><span class="v">${signed(v, 4)}</span>`, e.clientX, e.clientY);
      });
      c.addEventListener('pointerleave', () => app.tip.hide());
    });

    /* ---------- 右侧：语义地图 ---------- */
    const svg = $('.em-svg', el);
    const layers = {};
    ['cats', 'bg', 'path', 'mine', 'analogy'].forEach((k) => { layers[k] = svgEl('g', { class: `em-l-${k}` }); svg.append(layers[k]); });

    for (const [key, c] of Object.entries(CATS)) {
      if (key === 'misc' || key === 'byte') continue;
      const [x, y] = toXY(c.c);
      const t = svgEl('text', { x, y: y - 26, class: 'em-cat', 'text-anchor': 'middle' });
      t.textContent = c.name;
      layers.cats.append(t);
    }

    const promptTexts = new Set(tokens.map((t) => t.text.trim()));
    const bgPoints = [];
    MAP_WORDS.forEach((w) => {
      if (promptTexts.has(w)) return;
      const t = { ...tokenize(w)[0], text: w, i: 0 };
      const [x, y] = toXY(mapPos(t));
      const g = svgEl('g', { class: 'em-pt', transform: `translate(${x},${y})`, 'data-w': w });
      g.append(svgEl('circle', { r: 3.2 }));
      const label = svgEl('text', { x: 7, y: 4 });
      label.textContent = w;
      g.append(label);
      layers.bg.append(g);
      bgPoints.push({ w, x, y, cat: t.cat, g });
      g.addEventListener('pointerenter', (e) => {
        hovered.add(w);
        app.tip.show(`<b>${esc(w)}</b> <span class="dimmed">· ${CATS[t.cat]?.name ?? ''}</span>`, e.clientX, e.clientY);
        if (hovered.size >= 3) app.discover('meaning');
      });
      g.addEventListener('pointerleave', () => app.tip.hide());
    });

    const mine = tokens.map((t) => ({ t, xy: toXY(mapPos(t)) }));
    const pts = mine.map((m) => m.xy.join(',')).join(' ');
    layers.path.append(svgEl('polyline', { points: pts, class: 'em-path' }));
    mine.forEach(({ t, xy: [x, y] }, i) => {
      const g = svgEl('g', { class: 'em-me', transform: `translate(${x},${y})`, style: `--h:${hueOf(t.text)}; --d:${i * 90}ms`, 'data-i': i });
      g.append(svgEl('circle', { r: 16, class: 'em-ring' }));
      g.append(svgEl('circle', { r: 6.5, class: 'em-dot' }));
      const label = svgEl('text', { x: 11, y: -10 });
      label.textContent = tokPlain(t.text);
      g.append(label);
      layers.mine.append(g);
      g.addEventListener('pointerenter', (e) => {
        hovered.add(t.text);
        const near = bgPoints.map((p) => ({ ...p, d: Math.hypot(p.x - x, p.y - y) })).sort((a, b) => a.d - b.d).slice(0, 3).map((p) => p.w);
        app.tip.show(`<span class="k">你的词元 #${t.i}</span><b>${esc(tokPlain(t.text))}</b> · ${CATS[t.cat]?.name ?? ''}<br>最近的邻居：${near.map(esc).join('、')}`, e.clientX, e.clientY);
        if (hovered.size >= 3) app.discover('meaning');
      });
      g.addEventListener('pointerleave', () => app.tip.hide());
      g.addEventListener('click', () => { st.focus = t.i; app.sfx.click(); drawDetail(); });
    });

    function markMap() {
      $$('.em-me', svg).forEach((g) => g.classList.toggle('focus', Number(g.dataset.i) === st.focus));
    }

    /* ---------- 向量算术 ---------- */
    const eq = $('.em-eq', el);
    let analogyTimers = [];
    const P = (w) => bgPoints.find((p) => p.w === w) || (() => { const [x, y] = toXY(mapPos({ ...tokenize(w)[0], text: w })); return { x, y }; })();
    const arrow = (a, b, cls) => {
      const len = Math.hypot(b.x - a.x, b.y - a.y);
      const line = svgEl('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: `em-arrow ${cls}`, 'marker-end': 'url(#emHead)', style: `--len:${len}` });
      layers.analogy.append(line);
      return line;
    };
    const defs = svgEl('defs');
    defs.innerHTML = '<marker id="emHead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#ffb65c"/></marker>';
    svg.prepend(defs);

    $('.em-analogy', el).addEventListener('click', () => {
      analogyTimers.forEach(clearTimeout);
      layers.analogy.innerHTML = '';
      app.sfx.click();
      const man = P('男人'), woman = P('女人'), king = P('国王'), queen = P('女王');
      [man, woman, king, queen].forEach((p) => p.g?.classList.add('hl'));
      el.classList.add('analogy-on');
      eq.innerHTML = '<span>国王</span>';
      const res = { x: king.x + (woman.x - man.x), y: king.y + (woman.y - man.y) };
      analogyTimers.push(setTimeout(() => { arrow(man, woman, 'a1'); eq.innerHTML = '<span>国王</span> − <span>男人</span> + <span>女人</span>'; app.sfx.tick(); }, 350));
      analogyTimers.push(setTimeout(() => { arrow(king, res, 'a2'); app.sfx.tick(); }, 1300));
      analogyTimers.push(setTimeout(() => {
        const ring = svgEl('circle', { cx: res.x, cy: res.y, r: 20, class: 'em-hit' });
        layers.analogy.append(ring);
        eq.innerHTML = '<span>国王</span> − <span>男人</span> + <span>女人</span> ≈ <b>女王</b>';
        app.sfx.land();
        app.discover('analogy');
        app.say('从“男人”指向“女人”的箭头，平移到“国王”身上，正好落在“女王”旁边。<b>方向本身也携带意义</b>：这里是“性别”，别的方向可能是“时态”“单复数”“国家→首都”。');
      }, 2300));
    });

    /* ---------- 位置编码 ---------- */
    const posBtn = $('.em-pos', el);
    posBtn.addEventListener('click', () => {
      withPos = !withPos;
      posBtn.setAttribute('aria-pressed', withPos);
      $('.em-pe', el).hidden = !withPos;
      app.sfx.click();
      drawStrips();
      drawDetail();
      if (withPos) {
        paintPE($('.em-pe-c', el), Math.max(tokens.length, 8));
        app.discover('position');
        app.say('每个位置都被叠加了一段不同频率的正弦波。注意同一个字出现在不同位置时，条形码不再完全相同，<b>模型因此分得清先后</b>。');
      }
    });

    const redraw = () => { drawStrips(); drawDetail(); if (withPos) paintPE($('.em-pe-c', el), Math.max(tokens.length, 8)); };
    const off = onResize($('.em-list', el), redraw);
    requestAnimationFrame(redraw);

    app.say(`每个词元都换成了<b>一条 768 个数字的向量</b>。意思相近的词，向量也相近，在右边的地图上挨在一起。<em>把鼠标移到地图上逛逛</em>，再试试“向量算术”。`);
    app.setNext('送入层塔 <b>↓</b>', () => app.go(3, $('.em-rows', el)));

    return { destroy() { off(); analogyTimers.forEach(clearTimeout); } };
  },
};
