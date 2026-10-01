// 图像监视器：舞台左上角的一块 2D 画布，随调试步骤显示这张图在模型里“现在是什么样子”：
// 网格、主成分颜色、ViT 注意力、合并后的词元、M-RoPE 坐标、生成每个字时的注意力热力图、图片词元的逻辑透镜读数。
// 全部是导出的真实数据；热力图只做了按最大值归一化，方便看清。
import { $, esc, fmtPct } from '../../js/ui.js';
import { drawHeat, drawGrid, gridCanvas, normMax, heatRGBA } from './paint.js';
import { argmax, avgMass } from './explain.js';

const HEAT_VIEWS = new Set(['box', 'tower', 'layer', 'layerop', 'head', 'tray']);

export class Monitor {
  constructor({ onPickToken, onChange }) {
    this.el = $('#mon');
    this.cv = $('#monCv');
    this.g = this.cv.getContext('2d');
    this.cap = $('#monCap');
    this.title = $('#monTitle');
    this.tools = $('#monTools');
    this.onPickToken = onPickToken;
    this.onChange = onChange;
    this.mode = { heat: 'ground', vit: 'pca', lens: false };
    this.Q = null;
    this.key = '';
    this.hover = null;
    $('#btnMonFold').addEventListener('click', () => {
      this.el.classList.toggle('folded');
      $('#btnMonFold').textContent = this.el.classList.contains('folded') ? '+' : '–';
    });
    this.tools.addEventListener('click', (e) => {
      const b = e.target.closest('button[data-k]');
      if (!b) return;
      const k = b.dataset.k, v = b.dataset.v;
      this.mode[k] = k === 'lens' ? !this.mode.lens : v;
      this.key = '';
      this.onChange?.();
      window.__sfx?.click();
    });
    this.cv.addEventListener('pointermove', (e) => this.onMove(e));
    this.cv.addEventListener('pointerleave', () => { this.hover = null; this.tip(null); });
    this.cv.addEventListener('click', (e) => {
      const c = this.cellAt(e);
      if (c && this.clickable) { this.onPickToken?.(c.m); window.__sfx?.click(); }
    });
    new ResizeObserver(() => { this.key = ''; this.onChange?.(); }).observe(this.cv.parentElement);
  }

  set(Q) { this.Q = Q; this.key = ''; }

  // 画布尺寸跟着图片比例走
  fit() {
    const V = this.Q.V;
    const box = this.cv.parentElement;
    const w = Math.max(80, box.clientWidth);
    const h = Math.round((w * V.gh) / V.gw);
    const dpr = Math.min(2, devicePixelRatio || 1);
    if (this.cv.width !== Math.round(w * dpr) || this.cv.height !== Math.round(h * dpr)) {
      this.cv.width = Math.round(w * dpr);
      this.cv.height = Math.round(h * dpr);
      this.cv.style.height = `${h}px`;
    }
    this.W = this.cv.width; this.H = this.cv.height; this.dpr = dpr;
  }

  cellAt(e) {
    if (!this.Q) return null;
    const V = this.Q.V, r = this.cv.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
    if (x < 0 || y < 0 || x >= 1 || y >= 1) return null;
    const row = Math.floor(y * V.mh), col = Math.floor(x * V.mw);
    return { row, col, m: row * V.mw + col, pr: Math.floor(y * V.gh), pc: Math.floor(x * V.gw) };
  }

  onMove(e) {
    const c = this.cellAt(e);
    if (!c || !this.cellInfo) return this.tip(null);
    this.tip(this.cellInfo(c), e);
  }

  tip(html, e) {
    const tip = $('#tooltip');
    if (!html) { tip.classList.remove('on'); return; }
    tip.innerHTML = html;
    tip.classList.add('on');
    const r = tip.getBoundingClientRect();
    let x = e.clientX + 16, y = e.clientY + 16;
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 16;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 12;
    tip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
  }

  // st = { step, view, g, p, depth, vq }
  draw(st) {
    if (!this.Q) return;
    const s = st.step, view = st.view;
    const anim = this.animated(st);
    const pk = anim ? Math.round(st.p * 40) : 0;
    const key = [this.Q.id, view, s.ph, s.sub, s.L, s.op, s.mi, st.g, st.vq, pk, this.mode.heat, this.mode.vit, this.mode.lens].join('|');
    if (key === this.key) return;
    this.key = key;
    this.fit();
    const V = this.Q.V, g = this.g, W = this.W, H = this.H;
    g.save();
    g.clearRect(0, 0, W, H);
    this.clickable = false;
    this.cellInfo = null;
    let title = '监视器', cap = '', tools = '';
    const p = st.p;
    const heatTools = (layerOK) => `<div class="seg">${layerOK ? `<button type="button" data-k="heat" data-v="cur" class="${this.mode.heat === 'cur' ? 'on' : ''}">本层</button>` : ''}<button type="button" data-k="heat" data-v="ground" class="${this.mode.heat === 'ground' || (!layerOK && this.mode.heat === 'cur') ? 'on' : ''}">对准层平均</button><button type="button" data-k="heat" data-v="all" class="${this.mode.heat === 'all' ? 'on' : ''}">全部 28 层</button></div>${layerOK ? `<button type="button" data-k="lens" class="tog ${this.mode.lens ? 'on' : ''}">读数</button>` : ''}`;

    if (view === 'image' || (view === 'box' && s.ph === 'see')) {
      g.drawImage(V.img, 0, 0, W, H);
      if (s.sub === 'resize') {
        title = '缩放';
        const [ow, oh] = V.spec.orig;
        // 原图比例的虚线框（按同样的高度画）
        const sw = (ow / oh) * H;
        g.setLineDash([6 * this.dpr, 5 * this.dpr]);
        g.strokeStyle = 'rgba(255, 182, 92, .8)';
        g.lineWidth = 1.5 * this.dpr;
        g.strokeRect((W - sw) / 2, 1, sw, H - 2);
        g.setLineDash([]);
        cap = `原图 <b>${ow}×${oh}</b> → 模型看到 <b>${V.gw * 16}×${V.gh * 16}</b>（每边取整到 32 的倍数）`;
      } else if (s.sub === 'norm') {
        title = '归一化';
        this.shade(0.35);
        const mic = V.micro, ps = W / V.gw;
        drawGrid(g, W, H, V.gh, V.gw, 'rgba(200,220,255,.12)');
        g.strokeStyle = '#ffb65c'; g.lineWidth = 2 * this.dpr;
        g.strokeRect(mic.col * ps, mic.row * ps, ps, ps);
        cap = `像素 p → (p/255 − 0.5)/0.5 ∈ [−1, 1]。例：图块 (${mic.row}, ${mic.col}) 左上角 RGB = ${V.pix[0]}, ${V.pix[1]}, ${V.pix[2]} → ${V.px[0].toFixed(2)}, ${V.px[256].toFixed(2)}, ${V.px[512].toFixed(2)}`;
      } else {
        const k = s.sub === 'patch' ? Math.min(1, p * 1.6) : s.ph === 'see' ? p : view === 'image' ? Math.min(1, p * 1.4) : 1;
        title = s.sub === 'patch' ? '切成图块' : s.ph === 'see' ? '看图' : '预处理';
        const rows = Math.max(1, Math.round(V.gh * k));
        g.save();
        g.beginPath(); g.rect(0, 0, W, (rows / V.gh) * H); g.clip();
        drawGrid(g, W, H, V.gh, V.gw, 'rgba(210,225,255,.30)');
        drawGrid(g, W, H, V.mh, V.mw, 'rgba(255,182,92,.55)', 1.4 * this.dpr);
        g.restore();
        cap = `${V.gh} × ${V.gw} = <b>${V.Np}</b> 个 16×16 图块（细线）；每 2×2 合并成一个词元，共 <b>${V.Nv}</b> 个（橙线）`;
      }
      this.cellInfo = (c) => `<span class="k">图块 (${c.pr}, ${c.pc}) · 词元 (${c.row}, ${c.col})</span>像素 ${c.pc * 16}–${c.pc * 16 + 15}, ${c.pr * 16}–${c.pr * 16 + 15}`;
    } else if (view === 'embed' || view === 'vit' || view === 'vitlayer' || view === 'vitop') {
      const L = s.L;
      const lvl = view === 'embed' ? 0 : view === 'vit' ? Math.min(24, Math.floor(p * 25)) : (s.op === 'ln1' || s.op === 'attn' ? L : L + 1);
      const showAttn = (view === 'vitop' && s.op === 'attn') || (view === 'vitlayer' && this.mode.vit === 'attn');
      const has = L !== undefined && V.vitAttnLayers.includes(L);
      const q = st.vq ?? V.micro.token;
      if (showAttn && has) {
        title = `ViT 第 ${L} 层 · 注意力`;
        const a = V.vattn(L, q);
        drawHeat(g, V.img, W, H, a, V.mh, V.mw, { dim: 0.42, gamma: 0.7 });
        this.mark(q, '#5ef0d4');
        const top = argmax(a);
        cap = `从 ${cellS(V, q)} 出发（青框），16 头平均。看自己 ${fmtPct(V.vattnSelf(L, q))}，最多看 ${cellS(V, top)} ${fmtPct(a[top])}。<b>点任意一格换查询位置。</b>`;
        this.clickable = true;
        this.cellInfo = (c) => `<span class="k">词元 (${c.row}, ${c.col})</span>被 ${cellS(V, q)} 看了 <span class="v">${fmtPct(a[c.m])}</span><br><span class="v">点击：从这里出发</span>`;
      } else if (showAttn) {
        title = `ViT 第 ${L} 层 · 注意距离`;
        drawHeat(g, V.img, W, H, null, 1, 1, { dim: 0.28 });
        this.distChart(V.vitDist[L]);
        cap = `这一层没有导出完整注意力图（只导出了第 ${V.vitAttnLayers.join('、')} 层）。柱子是 16 个头各自的<b>平均注意距离</b>（单位：图块），真实计算自 ${V.Np}×${V.Np} 的注意力矩阵。`;
      } else {
        title = view === 'embed' ? (s.sub === 'pos' ? '加上位置嵌入' : '图块嵌入') : `ViT ${lvl === 0 ? '输入' : `第 ${lvl - 1} 层之后`} · 特征`;
        const a = view === 'embed' ? Math.min(1, p * 1.5) : 0.92;
        this.pcaCells(V.pca(lvl), V.gh, V.gw, a);
        const v = V.pcaVar[lvl];
        cap = `每个图块的 ${1024} 维特征投到前 3 个主成分上，当作红绿蓝（解释了 ${((v[0] + v[1] + v[2]) * 100).toFixed(0)}% 的方差）。颜色相近 = 模型觉得它们相似。${view === 'vit' ? `正在播放第 0 → 24 级。` : ''}`;
        if (view === 'vitlayer' && has) tools = `<div class="seg"><button type="button" data-k="vit" data-v="pca" class="${this.mode.vit === 'pca' ? 'on' : ''}">特征</button><button type="button" data-k="vit" data-v="attn" class="${this.mode.vit === 'attn' ? 'on' : ''}">注意力</button></div>`;
        this.cellInfo = (c) => `<span class="k">图块 (${c.pr}, ${c.pc})</span>第 ${lvl} 级特征的主成分颜色`;
      }
    } else if (view === 'conv') {
      title = '卷积核（1024 个里的 16 个）';
      this.bank();
      cap = `对图块 (${V.micro.row}, ${V.micro.col}) 反应最强的 16 个输出通道的卷积核（两帧相加，灰色 = 0）。真实权重，每个 3×16×16。`;
    } else if (view === 'micro') {
      title = `一次乘加 · 第 ${V.micro.ch} 维`;
      this.microView(s, p);
      const mic = V.micro;
      cap = s.mi === 'sum'
        ? `Σ ${1536} 项 + b = <b>${mic.conv.toFixed(4)}</b>，+ 位置嵌入 ${mic.pos >= 0 ? '+' : '−'}${Math.abs(mic.pos).toFixed(4)} = <b>${mic.after.toFixed(4)}</b>`
        : s.mi === 'mul' ? `左：图块的真实像素；右：第 ${mic.ch} 个卷积核（第 0 帧 / 第 1 帧）。圈出的是贡献最大的几项。` : `图块 (${mic.row}, ${mic.col}) 放大成 16×16 个像素`;
    } else if (view === 'merge') {
      title = s.sub === 'deep' ? 'DeepStack · 三路中间特征' : '合并成词元';
      if (s.sub === 'deep') this.deepView();
      else {
        const k = s.sub === 'group' ? p : 1;
        if (k < 0.5) this.pcaCells(V.pca(24), V.gh, V.gw, 0.9);
        else this.pcaCells(V.mergePca, V.mh, V.mw, 0.9);
        drawGrid(g, W, H, V.mh, V.mw, 'rgba(255,182,92,.5)', 1.2 * this.dpr);
        cap = s.sub === 'group' ? `2×2 个图块（${V.Np} 个）→ 1 个词元（${V.Nv} 个）` : `合并器输出：${V.Nv} 个 ${2048} 维向量的主成分颜色。这就是语言模型“看到”的图片。`;
      }
      this.cellInfo = (c) => `<span class="k">视觉词元 (${c.row}, ${c.col})</span>第 ${c.m} 个 &lt;|image_pad|&gt;`;
    } else if (view === 'splice') {
      title = s.sub === 'mrope' ? 'M-RoPE 位置 (t, h, w)' : '插进对话';
      drawHeat(g, V.img, W, H, null, 1, 1, { dim: 0.45 });
      drawGrid(g, W, H, V.mh, V.mw, 'rgba(255,182,92,.45)', 1.2 * this.dpr);
      this.coords(s.sub === 'mrope' ? 'pos' : 'idx');
      const p0 = this.Q.pos(V.vs);
      cap = s.sub === 'mrope'
        ? `所有图片词元 t = ${p0[0]}；h = ${p0[1]} + 行，w = ${p0[2]} + 列。格子里是 (h, w)。`
        : `每格对应序列里的一个 &lt;|image_pad|&gt;，编号 #${V.vs} 到 #${V.vs + V.Nv - 1}（按行展开）`;
      this.cellInfo = (c) => { const pp = this.Q.pos(V.vs + c.m); return `<span class="k">词元 #${V.vs + c.m}</span>位置 (t, h, w) = <span class="v">(${pp[0]}, ${pp[1]}, ${pp[2]})</span>`; };
    } else if (view === 'heads') {
      title = `第 ${s.L} 层 · 16 个头`;
      this.headsView(st);
      cap = `每个小图是一个头在图片上的注意力（各自归一化），左上角是它分给图片的比例。`;
    } else if (HEAT_VIEWS.has(view)) {
      const Q = this.Q, gg = st.g;
      const layerOK = view === 'layer' || view === 'layerop';
      const mode = !layerOK && this.mode.heat === 'cur' ? 'ground' : this.mode.heat;
      const [ga, gb] = Q.manifest.groundLayers;
      let vals, lab;
      if (mode === 'cur' && layerOK) { vals = Q.attImg(gg, s.L); lab = `第 ${s.L} 层`; }
      else if (mode === 'all') { vals = Q.attAvg(gg, [0, Q.NL - 1]); lab = `全部 ${Q.NL} 层平均`; }
      else { vals = Q.attAvg(gg, [ga, gb]); lab = `第 ${ga}–${gb} 层平均`; }
      drawHeat(g, V.img, W, H, vals, V.mh, V.mw, { dim: 0.45 });
      if (this.mode.lens && layerOK) this.lensWords(s.L);
      const tokS = Q.steps[gg].chosenS;
      title = `生成「${tokS.replace(/\n/g, '↵')}」时在看哪`;
      const mass = mode === 'cur' && layerOK ? Q.mass(gg, s.L) : avgMass(Q, gg, mode === 'all' ? 0 : ga, mode === 'all' ? Q.NL - 1 : gb);
      cap = `真实注意力（16 头平均，${lab}），按最大值归一化。图片一共分到 ${fmtPct(mass)} 的注意力。`;
      tools = heatTools(layerOK);
      this.cellInfo = (c) => {
        let h = `<span class="k">视觉词元 (${c.row}, ${c.col})</span>注意力 <span class="v">${fmtPct(vals[c.m])}</span>`;
        if (layerOK) { const w = V.ilens(s.L + 1, c.m); h += `<br>第 ${s.L} 层读数：<b>${esc(w.s.replace(/\n/g, '↵'))}</b> ${fmtPct(w.p)}`; }
        return h;
      };
    }
    g.restore();
    this.title.textContent = title;
    this.cap.innerHTML = cap;
    if (this.tools.dataset.html !== tools) { this.tools.innerHTML = tools; this.tools.dataset.html = tools; }
    this.cv.style.cursor = this.clickable ? 'crosshair' : '';
  }

  animated(st) {
    const v = st.view, s = st.step;
    return v === 'vit' || v === 'embed' || (v === 'image' && s.sub !== 'resize' && s.sub !== 'norm') || (v === 'box' && s.ph === 'see') || (v === 'merge' && s.sub === 'group') || (v === 'micro' && s.mi === 'mul');
  }

  shade(k) { this.g.fillStyle = `rgba(5,10,20,${k})`; this.g.fillRect(0, 0, this.W, this.H); }

  mark(m, color) {
    const V = this.Q.V, cw = this.W / V.mw, ch = this.H / V.mh;
    this.g.strokeStyle = color;
    this.g.lineWidth = 2 * this.dpr;
    this.g.strokeRect((m % V.mw) * cw + 1, Math.floor(m / V.mw) * ch + 1, cw - 2, ch - 2);
  }

  pcaCells(cols, rows, ncol, alpha) {
    const V = this.Q.V, g = this.g;
    g.drawImage(V.img, 0, 0, this.W, this.H);
    this.shade(0.55);
    const c = gridCanvas(new Float32Array(rows * ncol), rows, ncol, (_, i) => [cols[i * 3] * 0.88, cols[i * 3 + 1] * 0.88, cols[i * 3 + 2] * 0.88, alpha]);
    g.imageSmoothingEnabled = false;
    g.drawImage(c, 0, 0, this.W, this.H);
    g.imageSmoothingEnabled = true;
  }

  distChart(dist) {
    const g = this.g, W = this.W, H = this.H, n = dist.length;
    const mx = Math.max(...dist, 1);
    const bw = W / n;
    g.font = `${10 * this.dpr}px "JetBrains Mono", monospace`;
    g.textAlign = 'center';
    for (let h = 0; h < n; h++) {
      const bh = (dist[h] / mx) * H * 0.7;
      g.fillStyle = 'rgba(94, 240, 212, .55)';
      g.fillRect(h * bw + bw * 0.18, H - 16 * this.dpr - bh, bw * 0.64, bh);
      g.fillStyle = 'rgba(200, 215, 240, .7)';
      if (bw > 14 * this.dpr) g.fillText(String(h), h * bw + bw / 2, H - 4 * this.dpr);
    }
  }

  bank() {
    const V = this.Q.V, g = this.g, W = this.W, H = this.H;
    g.fillStyle = '#070d1a'; g.fillRect(0, 0, W, H);
    const n = 4, pad = 6 * this.dpr;
    const sz = Math.min((W - pad * (n + 1)) / n, (H - pad * (n + 1)) / n);
    const ox = (W - (sz * n + pad * (n - 1))) / 2, oy = (H - (sz * n + pad * (n - 1))) / 2;
    const c = document.createElement('canvas'); c.width = 16; c.height = 16;
    const cg = c.getContext('2d');
    for (let k = 0; k < 16; k++) {
      const im = cg.createImageData(16, 16);
      for (let i = 0; i < 256; i++) { for (let ch = 0; ch < 3; ch++) im.data[i * 4 + ch] = V.bank[k * 768 + i * 3 + ch]; im.data[i * 4 + 3] = 255; }
      cg.putImageData(im, 0, 0);
      g.imageSmoothingEnabled = false;
      const x = ox + (k % n) * (sz + pad), y = oy + Math.floor(k / n) * (sz + pad);
      g.drawImage(c, x, y, sz, sz);
      g.fillStyle = 'rgba(200,215,240,.75)';
      g.font = `${9 * this.dpr}px "JetBrains Mono", monospace`;
      g.fillText(`#${V.micro.bankCh[k]}`, x + 2 * this.dpr, y + 10 * this.dpr);
    }
    g.imageSmoothingEnabled = true;
  }

  microView(s, p) {
    const V = this.Q.V, g = this.g, W = this.W, H = this.H, mic = V.micro;
    g.fillStyle = '#070d1a'; g.fillRect(0, 0, W, H);
    const gap = 8 * this.dpr;
    const sz = Math.min((W - gap * 4) / 3, H - gap * 3 - 12 * this.dpr);
    const y = (H - sz) / 2 + 6 * this.dpr;
    const xs = [gap, gap * 2 + sz, gap * 3 + sz * 2];
    const c = document.createElement('canvas'); c.width = 16; c.height = 16;
    const cg = c.getContext('2d');
    const put = (fn) => { const im = cg.createImageData(16, 16); for (let i = 0; i < 256; i++) { const [r, gg, b] = fn(i); im.data[i * 4] = r; im.data[i * 4 + 1] = gg; im.data[i * 4 + 2] = b; im.data[i * 4 + 3] = 255; } cg.putImageData(im, 0, 0); };
    g.imageSmoothingEnabled = false;
    put((i) => [V.pix[i * 3], V.pix[i * 3 + 1], V.pix[i * 3 + 2]]);
    g.drawImage(c, xs[0], y, sz, sz);
    const show = s.mi !== 'pick';
    for (let t = 0; t < 2; t++) {
      put((i) => [V.kern[(0 * 2 + t) * 256 + i], V.kern[(1 * 2 + t) * 256 + i], V.kern[(2 * 2 + t) * 256 + i]]);
      g.globalAlpha = show ? 1 : 0.25;
      g.drawImage(c, xs[1 + t], y, sz, sz);
      g.globalAlpha = 1;
    }
    g.imageSmoothingEnabled = true;
    g.font = `${10 * this.dpr}px "JetBrains Mono", monospace`;
    g.fillStyle = 'rgba(200,215,240,.75)';
    g.fillText('像素', xs[0], y - 4 * this.dpr);
    g.fillText('核·第0帧', xs[1], y - 4 * this.dpr);
    g.fillText('核·第1帧', xs[2], y - 4 * this.dpr);
    if (show) {
      const n = s.mi === 'mul' ? Math.max(1, Math.ceil(p * mic.top.length)) : mic.top.length;
      const cell = sz / 16;
      mic.top.slice(0, n).forEach((e) => {
        g.strokeStyle = e.prod >= 0 ? '#ffb65c' : '#6b9bff';
        g.lineWidth = 1.5 * this.dpr;
        g.strokeRect(xs[0] + e.x * cell, y + e.y * cell, cell, cell);
        g.strokeRect(xs[1 + e.t] + e.x * cell, y + e.y * cell, cell, cell);
      });
    }
  }

  deepView() {
    const V = this.Q.V, g = this.g, W = this.W, H = this.H;
    g.fillStyle = '#070d1a'; g.fillRect(0, 0, W, H);
    const gap = 6 * this.dpr, w = (W - gap * 4) / 3, h = (w * V.mh) / V.mw;
    const y = (H - h) / 2 + 6 * this.dpr;
    for (let d = 0; d < 3; d++) {
      const cols = V.dsPca(d);
      const c = gridCanvas(new Float32Array(V.Nv), V.mh, V.mw, (_, i) => [cols[i * 3] * 0.88, cols[i * 3 + 1] * 0.88, cols[i * 3 + 2] * 0.88, 1]);
      g.imageSmoothingEnabled = false;
      g.drawImage(c, gap + d * (w + gap), y, w, h);
      g.fillStyle = 'rgba(200,215,240,.8)';
      g.font = `${10 * this.dpr}px "JetBrains Mono", monospace`;
      g.fillText(`ViT ${this.Q.manifest.model.vision.deepstack[d]} → LLM ${d}`, gap + d * (w + gap), y - 5 * this.dpr);
    }
    g.imageSmoothingEnabled = true;
    this.cap.innerHTML = '';
  }

  coords(kind) {
    const V = this.Q.V, g = this.g, cw = this.W / V.mw, ch = this.H / V.mh;
    const fs = Math.min(cw / 3.6, ch / 2.2, 11 * this.dpr);
    if (fs < 5.5 * this.dpr) return;
    g.font = `${fs}px "JetBrains Mono", monospace`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillStyle = 'rgba(233, 239, 249, .85)';
    for (let m = 0; m < V.Nv; m++) {
      const r = Math.floor(m / V.mw), c = m % V.mw;
      const txt = kind === 'pos' ? (() => { const pp = this.Q.pos(V.vs + m); return `${pp[1]},${pp[2]}`; })() : String(V.vs + m);
      g.fillText(txt, (c + 0.5) * cw, (r + 0.5) * ch);
    }
  }

  lensWords(L) {
    const V = this.Q.V, g = this.g, cw = this.W / V.mw, ch = this.H / V.mh;
    const fs = Math.min(cw / 2.2, ch / 1.6, 13 * this.dpr);
    g.font = `600 ${fs}px "PingFang SC","Microsoft YaHei","Noto Sans SC",sans-serif`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    for (let m = 0; m < V.Nv; m++) {
      const { s, p } = V.ilens(L + 1, m);
      const w = s.trim().replace(/\n/g, '');
      if (!w || p < 0.05 || /<\|/.test(w)) continue;
      const r = Math.floor(m / V.mw), c = m % V.mw;
      g.fillStyle = `rgba(233, 239, 249, ${0.35 + 0.6 * Math.min(1, p * 2)})`;
      g.fillText(w.length > 3 ? w.slice(0, 3) : w, (c + 0.5) * cw, (r + 0.5) * ch);
    }
  }

  headsView(st) {
    const Q = this.Q, V = Q.V, g = this.g, W = this.W, H = this.H;
    const f = Q.manifest.focusLayers.indexOf(st.step.L);
    g.fillStyle = '#070d1a'; g.fillRect(0, 0, W, H);
    const n = 4, gap = 3 * this.dpr;
    const w = (W - gap * (n + 1)) / n, h = (H - gap * (n + 1)) / n;
    for (let hh = 0; hh < 16; hh++) {
      const x = gap + (hh % n) * (w + gap), y = gap + Math.floor(hh / n) * (h + gap);
      g.save();
      g.translate(x, y);
      drawHeat(g, V.img, w, h, Q.attHead(st.g, f, hh), V.mh, V.mw, { dim: 0.4 });
      g.fillStyle = 'rgba(5,10,20,.7)';
      g.fillRect(0, 0, w, 13 * this.dpr);
      g.fillStyle = 'rgba(233,239,249,.9)';
      g.font = `${9 * this.dpr}px "JetBrains Mono", monospace`;
      g.textBaseline = 'top';
      g.fillText(`H${hh} ${fmtPct(Q.headMass(st.g, st.step.L, hh))}`, 2 * this.dpr, 2 * this.dpr);
      g.restore();
    }
  }
}

const cellS = (V, m) => `(${Math.floor(m / V.mw)}, ${m % V.mw})`;
export { heatRGBA, normMax };
