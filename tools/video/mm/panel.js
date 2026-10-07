// 片子里的 2D 讲解面板：全部按 1080p 排版、按时间 t 画（没有 CSS 过渡），数据都是导出的真实数据。
//   Monitor：左侧一块大“监视器”，画这张图在模型里现在是什么样子（主成分颜色、2×2 合并、M-RoPE 坐标、热力图、图片词元的透镜读数）
//   CalcBoard：全片唯一一次逐数计算——一个图块的 1536 个数 × 第 c 个卷积核 + 偏置 = 嵌入的一个数
//   LensCard：说出「3」之前，逻辑透镜每层读出的第一名
import { drawHeat, drawGrid, gridCanvas, normMax } from '/public/multimodal/js/paint.js';
import { tokPlain, esc } from '/public/js/ui.js';
import { clamp, seg, smooth, easeOut, lerp } from '../lib/cam.js';

const el = (tag, cls, html, parent) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; if (parent) parent.append(e); return e; };
const pct = (p) => `${(p * 100).toFixed(p >= 0.995 ? 0 : p < 0.1 ? 1 : 0)}%`;
const num = (v, d = 3) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
const sgn = (v, d = 5) => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(d);

/* ================================================================ 监视器 */

export class Monitor {
  constructor(parent, Q, F) {
    this.Q = Q; this.V = Q.V; this.F = F;
    const V = this.V;
    this.W = 660; this.H = Math.round((this.W * V.gh) / V.gw);
    this.el = el('div', 'mon', `<div class="h"><b></b><small></small></div><div class="cv"><canvas width="${this.W * 1.5}" height="${this.H * 1.5}" style="width:${this.W}px;height:${this.H}px"></canvas><div class="ax"></div></div><div class="foot"></div>`, parent);
    this.cv = this.el.querySelector('canvas');
    this.g = this.cv.getContext('2d');
    this.hT = this.el.querySelector('.h b');
    this.hS = this.el.querySelector('.h small');
    this.ax = this.el.querySelector('.ax');
    this.foot = this.el.querySelector('.foot');
    this.key = '';
  }

  update(P) {
    const on = P && P.a > 0.001;
    this.el.style.display = on ? 'block' : 'none';
    if (!on) return;
    this.el.style.opacity = P.a.toFixed(3);
    this.el.style.transform = `${P.center ? 'translateX(-50%) scale(1.22) ' : ''}translateY(${((1 - easeOut(Math.min(1, P.a * 1.3))) * 16).toFixed(1)}px)`;
    const key = JSON.stringify(P, (k, v) => (k === 'a' || k === 'enr' ? undefined : typeof v === 'number' ? +v.toFixed(3) : v));
    if (key === this.key) return;
    this.key = key;
    const g = this.g, V = this.V, Q = this.Q;
    const W = this.cv.width, H = this.cv.height;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, W, H);
    this.ax.innerHTML = '';
    this.foot.innerHTML = '';
    this.el.className = `mon k-${P.kind}${P.center ? ' center' : ''}`;
    const k = this[P.kind];
    if (k) k.call(this, P, g, W, H);
  }

  shade(g, W, H, a) { g.fillStyle = `rgba(5,10,20,${a})`; g.fillRect(0, 0, W, H); }

  pcaGrid(g, W, H, cols, rows, ncol, alpha = 0.92, dim = 0.55) {
    g.drawImage(this.V.img, 0, 0, W, H);
    this.shade(g, W, H, dim);
    const c = gridCanvas(new Float32Array(rows * ncol), rows, ncol, (_, i) => [cols[i * 3] * 0.9, cols[i * 3 + 1] * 0.9, cols[i * 3 + 2] * 0.9, alpha]);
    g.imageSmoothingEnabled = false;
    g.drawImage(c, 0, 0, W, H);
    g.imageSmoothingEnabled = true;
  }

  // ② ViT：爬塔时画当前这一级的主成分颜色；之后对比“输入”和第 cmpL 级（交叉淡化，整张大图）
  // 级的编号和网站一致：第 0 级 = 图块嵌入 + 位置（输入），第 k 级 = 第 k−1 层（从 0 数）之后
  pca(P, g, W, H) {
    const V = this.V;
    const lv = Math.max(0, Math.min(24, P.lv));
    const cmp = P.cmp ?? 0, ck = P.cmpK ?? 0, cl = P.cmpL ?? 6;
    const name = (k) => (k === 0 ? '输入（图块嵌入 + 位置）' : `第 ${k - 1} 层之后`);
    this.hT.textContent = cmp > 0.5 ? (ck < 0.5 ? name(0) : name(cl)) : `视觉编码器 · ${name(lv)}`;
    this.hS.textContent = '颜色相近 = 模型觉得这两块相似';
    if (cmp < 0.001) { this.pcaGrid(g, W, H, V.pca(lv), V.gh, V.gw); return; }
    this.pcaGrid(g, W, H, V.pca(lv), V.gh, V.gw);
    g.globalAlpha = cmp;
    this.pcaGrid(g, W, H, V.pca(0), V.gh, V.gw);
    g.globalAlpha = cmp * ck;
    this.pcaGrid(g, W, H, V.pca(cl), V.gh, V.gw);
    g.globalAlpha = 1;
  }

  // ③ 2×2 合并：图上细线是 16×16 的图块，橙线是合并后的词元；绿苹果那一格的 4 块框出来
  merge(P, g, W, H) {
    const V = this.V, F = this.F;
    const k = P.k ?? 0;
    this.hT.textContent = k < 0.5 ? `${V.Np} 个图块` : `${V.Nv} 个视觉词元`;
    this.hS.textContent = '每 2×2 块合成一个';
    drawHeat(g, V.img, W, H, null, 1, 1, { dim: 0.62 });
    drawGrid(g, W, H, V.gh, V.gw, `rgba(210,225,255,${(0.32 * (1 - 0.7 * smooth(seg(k, 0.4, 0.8)))).toFixed(3)})`, 1.5);
    drawGrid(g, W, H, V.mh, V.mw, `rgba(255,182,92,${(0.85 * smooth(seg(k, 0.1, 0.5))).toFixed(3)})`, 3.5);
    const r = Math.floor(F.apple / V.mw), c = F.apple % V.mw, cw = W / V.mw, ch = H / V.mh;
    const a = smooth(seg(k, 0.0, 0.3));
    g.strokeStyle = `rgba(94,240,212,${a.toFixed(3)})`; g.lineWidth = 6;
    g.strokeRect(c * cw + 3, r * ch + 3, cw - 6, ch - 6);
    if (k > 0.55) { g.fillStyle = `rgba(94,240,212,${(0.25 * smooth(seg(k, 0.55, 0.9))).toFixed(3)})`; g.fillRect(c * cw, r * ch, cw, ch); }
    const lab = k < 0.55 ? '4 个图块' : '→ 1 个视觉词元';
    this.ax.innerHTML = `<div class="tag" style="left:${((c + 1) * this.W) / V.mw + 12}px;top:${((r + 0.5) * this.H) / V.mh}px;opacity:${a.toFixed(3)}">${lab}</div>`;
  }

  // ④ M-RoPE：行号 h、列号 w，绿苹果那一格写出 (t, h, w)
  mrope(P, g, W, H) {
    const V = this.V, Q = this.Q, F = this.F;
    this.hT.textContent = '每个视觉词元的位置 (t, h, w)';
    this.hS.textContent = `t 都是 ${F.p0[0]}`;
    drawHeat(g, V.img, W, H, null, 1, 1, { dim: 0.5 });
    drawGrid(g, W, H, V.mh, V.mw, 'rgba(255,182,92,.45)', 2);
    const cw = W / V.mw, ch = H / V.mh;
    const hot = F.apple, hr = Math.floor(hot / V.mw), hc = hot % V.mw;
    if (P.hl === 'hot') {
      g.strokeStyle = '#5ef0d4'; g.lineWidth = 5;
      g.strokeRect(hc * cw + 2, hr * ch + 2, cw - 4, ch - 4);
      g.fillStyle = 'rgba(94,240,212,.22)'; g.fillRect(hc * cw, hr * ch, cw, ch);
      g.fillStyle = 'rgba(94,240,212,.1)'; g.fillRect(0, hr * ch, W, ch); g.fillRect(hc * cw, 0, cw, H);
    }
    // 轴：上面写列号（w），左边写行号（h）——HTML 字，手机上看得清
    const p0 = F.p0;
    const cols = Array.from({ length: V.mw }, (_, c) => `<span class="cx${P.hl === 'hot' && c === hc ? ' on' : ''}" style="left:${((c + 0.5) * this.W) / V.mw}px">${p0[2] + c}</span>`).join('');
    const rows = Array.from({ length: V.mh }, (_, r) => `<span class="ry${P.hl === 'hot' && r === hr ? ' on' : ''}" style="top:${((r + 0.5) * this.H) / V.mh}px">${p0[1] + r}</span>`).join('');
    this.ax.innerHTML = `<div class="axc">${cols}</div><div class="axr">${rows}</div><div class="axl hw">h↓ w→</div>${P.hl === 'hot' ? `<div class="tag" style="left:${((hc + 1) * this.W) / V.mw + 10}px;top:${((hr + 0.5) * this.H) / V.mh}px">(${F.pApple.join(', ')})</div>` : ''}`;
    // 下面一行：图片后面的文字词元接着往下数
    const after = Q.tokens.slice(V.vs + V.Nv, V.vs + V.Nv + 5).map((t, j) => ({ s: t.sp ? (t.s === '<|vision_end|>' ? '图片结束' : t.s) : tokPlain(t.s), p: Q.pos(V.vs + V.Nv + j)[0] }));
    if (P.rows) this.foot.innerHTML = `<span class="seq">${after.map((x) => `<i>${esc(x.s)}<b>${x.p}</b></i>`).join('')}<i>…</i></span>`;
  }

  // ⑤ 热力图：生成第 g 个字时第 L 层的注意力；下面是 28 层的“对准倍数”
  heat(P, g, W, H) {
    const V = this.V, Q = this.Q, F = this.F;
    const L = P.L;
    const vals = P.avg ? Q.attAvg(P.g, [F.ga, F.gb]) : Q.attImg(P.g, L);
    const tok = Q.steps[P.g].chosenS;
    this.hT.innerHTML = P.avg ? `生成「${esc(tokPlain(tok))}」时 · 第 ${F.ga}–${F.gb} 层平均` : `生成「${esc(tokPlain(tok))}」时 · 第 ${L} 层在看哪`;
    this.hS.textContent = '16 个头平均 · 按最大值归一化';
    drawHeat(g, V.img, W, H, vals, V.mh, V.mw, { dim: 0.42, gamma: 0.85 });
    // 绿苹果的框（虚线）
    const b = F.box, cw = W / V.mw, ch = H / V.mh;
    g.setLineDash([12, 9]);
    g.strokeStyle = 'rgba(94,240,212,.85)'; g.lineWidth = 3;
    g.strokeRect(b.c0 * cw, b.r0 * ch, (b.c1 - b.c0 + 1) * cw, (b.r1 - b.r0 + 1) * ch);
    g.setLineDash([]);
    // 每层的对准倍数（柱子），已经走过的层才画
    const up = P.upto ?? L + 1;
    const enr = F.enr, mx = Math.max(6, ...enr);
    const bars = enr.map((e, l) => {
      const shown = l < up;
      const hi = l >= F.ga && l <= F.gb;
      const h = shown ? Math.max(2, (e / mx) * 92) : 0;
      return `<i class="${hi ? 'hi' : ''}${l === L ? ' cur' : ''}" style="height:${h.toFixed(1)}px"></i>`;
    }).join('');
    const e = P.avg ? F.enrHi : enr[L];
    this.foot.innerHTML = `<div class="enr"><div class="bars">${bars}<span class="one" style="bottom:${((1 / mx) * 92).toFixed(1)}px"></span></div><div class="lab"><span>对准倍数（看绿苹果的比例 ÷ 它占的面积）</span><b>${P.avg ? `第 ${F.ga}–${F.gb} 层平均` : `第 ${L} 层`} ${e.toFixed(1)}×</b></div><div class="lx">${[0, F.ga, F.gb, 27].map((l) => `<span style="left:${((l + 0.5) / 28) * 100}%">L${l}</span>`).join('')}</div></div>`;
  }

  // ⑤′ 图片词元的逻辑透镜：每格写出读数最高的中文词
  lens(P, g, W, H) {
    const V = this.V;
    const L = P.L;
    this.hT.textContent = `图片的每个位置 · 第 ${L} 层“读”出的词`;
    this.hS.textContent = '逻辑透镜：直接接到输出头上';
    drawHeat(g, V.img, W, H, null, 1, 1, { dim: 0.32 });
    drawGrid(g, W, H, V.mh, V.mw, 'rgba(200,220,255,.14)', 1.5);
    const cw = this.W / V.mw, chh = this.H / V.mh;
    const k = P.k ?? 1;
    const words = [];
    for (let mm = 0; mm < V.Nv; mm++) {
      const { s, p } = V.ilens(L, mm);
      const w = s.trim();
      if (p < 0.08 || !/^[\u4e00-\u9fff]{2}$/.test(w)) continue;   // 只写两个字的中文词（一格放得下，不互相压）
      const r = Math.floor(mm / V.mw), c = mm % V.mw;
      const order = ((r * 7 + c * 3) % 17) / 17;
      const a = smooth(seg(k, order * 0.7, order * 0.7 + 0.3));
      if (a <= 0) continue;
      const big = ['苹果', '五个', '红色', '绿色'].includes(w);
      words.push(`<span class="${big ? 'b' : ''}" style="left:${(c + 0.5) * cw}px;top:${(r + 0.5) * chh}px;opacity:${(a * (0.45 + 0.55 * Math.min(1, p * 2))).toFixed(3)}">${esc(w)}</span>`);
    }
    this.ax.innerHTML = `<div class="words">${words.join('')}</div>`;
  }

  // ⑤″ 回答的每个字：对准层平均的热力图
  hover(P, g, W, H) {
    const V = this.V, Q = this.Q, F = this.F;
    const tok = Q.steps[P.g].chosenS;
    const sp = /^<\|/.test(tok);
    this.foot.innerHTML = '';
    this.hT.innerHTML = sp ? '写完：结束标记' : `生成「${esc(tokPlain(tok))}」时在看哪`;
    this.hS.textContent = `第 ${F.ga}–${F.gb} 层平均`;
    drawHeat(g, V.img, W, H, Q.attAvg(P.g, [F.ga, F.gb]), V.mh, V.mw, { dim: 0.42, gamma: 0.85 });
  }
}

/* ================================================================ 一次乘加 */

export class CalcBoard {
  constructor(parent, Q) {
    this.Q = Q; this.V = Q.V;
    const V = this.V, mic = V.micro, mv = Q.manifest.model.vision;
    this.el = el('div', 'calc', '', parent);
    const pixCv = this.mk16((i) => [V.pix[i * 3], V.pix[i * 3 + 1], V.pix[i * 3 + 2]]);
    const kernCv = [0, 1].map((t) => this.mk16((i) => [V.kern[(0 * 2 + t) * 256 + i], V.kern[(1 * 2 + t) * 256 + i], V.kern[(2 * 2 + t) * 256 + i]]));
    // 三个最大的乘积（真实数据，按 |乘积| 排序取前几项里正负各有的）
    const top = mic.top;
    this.rows = [top[0], top[2], top[4]];
    const cname = ['红', '绿', '蓝'];
    this.el.innerHTML = `
      <div class="col cpix"><div class="cap">这一块的像素</div><div class="stack"><div class="fr f1"></div><div class="fr f0"></div></div><div class="csub"><b>${mv.patch}×${mv.patch}</b> × 红绿蓝 <b>3</b> × <b>${mv.temporal}</b> 帧 = <b class="cx">${mv.inDim}</b> 个数</div><div class="note">单张图复制一份，当成两帧一样的视频</div></div>
      <div class="op ox">×</div>
      <div class="col ckern"><div class="cap">卷积核 #${mic.ch}（权重）</div><div class="kpair"><div class="fr k0"></div><div class="fr k1"></div></div><div class="csub">同样 <b class="cw">${mv.inDim}</b> 个权重 · 灰色 = 0</div></div>
      <div class="op oeq">=</div>
      <div class="col cres"><div class="cap">逐项相乘</div>
        <div class="rows">${this.rows.map((e, j) => `<div class="r" data-j="${j}"><span class="cx">${num(e.px)}</span><span class="o">×</span><span class="cw">${sgn(e.w, 5)}</span><span class="o">=</span><span class="cp">${sgn(e.prod, 5)}</span><small>${cname[e.c]} · 第${e.t}帧</small></div>`).join('')}
          <div class="r dots">… 一共 ${mv.inDim} 项</div></div>
        <div class="sum"><div class="l s1"><span>全部加起来</span><b class="cp">${sgn(mic.mine - mic.bias, 3)}</b></div><div class="l s2"><span>+ 偏置</span><b>${num(mic.bias)}</b></div><div class="l s3"><span>=</span><b class="cy">${num(mic.mine)}</b></div></div>
      </div>
      <div class="many"><div class="vec"></div><div class="t">第 ${mic.ch} 维 = <b class="cy">${num(mic.mine)}</b><span>${mv.hidden} 个卷积核 → 这一块的 ${mv.hidden} 个数</span></div></div>`;
    const put = (sel, cv) => { const host = this.el.querySelector(sel); cv.className = 'px'; host.append(cv); };
    put('.f0', pixCv);
    put('.f1', this.mk16((i) => [V.pix[i * 3], V.pix[i * 3 + 1], V.pix[i * 3 + 2]]));
    put('.k0', kernCv[0]);
    put('.k1', kernCv[1]);
    // 两帧上圈出参与上面三行乘积的格子
    this.marks = this.rows.map((e) => {
      const mk = (host, cls) => { const d = el('div', `mk ${cls}`, '', this.el.querySelector(host)); d.style.left = `${(e.x / 16) * 100}%`; d.style.top = `${(e.y / 16) * 100}%`; return d; };
      return [mk('.f0', e.prod >= 0 ? 'pos' : 'neg'), mk(e.t ? '.k1' : '.k0', e.prod >= 0 ? 'pos' : 'neg')];
    });
    // 1024 维向量：一条细长的格子带，第 c 维亮
    const vec = this.el.querySelector('.vec');
    vec.innerHTML = Array.from({ length: 64 }, (_, j) => `<i class="${j === Math.floor((mic.ch / mv.hidden) * 64) ? 'on' : ''}"></i>`).join('');
    this.q = (s) => this.el.querySelector(s);
  }

  mk16(fn) {
    const c = document.createElement('canvas');
    c.width = 16; c.height = 16;
    const g = c.getContext('2d');
    const im = g.createImageData(16, 16);
    for (let i = 0; i < 256; i++) { const [r, gg, b] = fn(i); im.data[i * 4] = r; im.data[i * 4 + 1] = gg; im.data[i * 4 + 2] = b; im.data[i * 4 + 3] = 255; }
    g.putImageData(im, 0, 0);
    return c;
  }

  update(P) {
    const on = P && P.a > 0.001;
    this.el.style.display = on ? 'grid' : 'none';
    if (!on) return;
    const { t, C } = P;
    this.el.style.opacity = P.a.toFixed(3);
    const A = (x0, d = 0.5) => smooth(seg(t, x0, x0 + d));
    const set = (sel, a, dy = 12) => { const e = this.q(sel); e.style.opacity = a.toFixed(3); e.style.transform = `translateY(${((1 - a) * dy).toFixed(1)}px)`; };
    set('.cpix', A(C.pix));
    set(".cpix .csub", A(C.frames, 0.6));
    set('.cpix .note', A(C.frames + 0.5, 0.6));
    // 第二帧从第一帧后面错开滑出
    const fk = smooth(seg(t, C.frames, C.frames + 0.7));
    this.q('.f1').style.transform = `translate(${(26 * fk).toFixed(1)}px, ${(-26 * fk).toFixed(1)}px)`;
    this.q('.f1').style.opacity = fk.toFixed(3);
    set('.ox', A(C.kern - 0.2));
    set('.ckern', A(C.kern));
    set('.oeq', A(C.prods[0] - 0.3));
    set('.cres', A(C.prods[0] - 0.2));
    this.rows.forEach((_, j) => {
      const tj = lerp(C.prods[0], C.prods[1], j / 3);
      set(`.r[data-j="${j}"]`, A(tj, 0.4), 8);
      const hot = t >= tj && t < (j < 2 ? lerp(C.prods[0], C.prods[1], (j + 1) / 3) : C.sum);
      this.marks[j].forEach((mk) => { mk.style.opacity = (A(tj, 0.3) * (hot ? 1 : 0.35)).toFixed(3); mk.classList.toggle('hot', hot); });
    });
    set('.r.dots', A(C.prods[1], 0.4), 8);
    set('.s1', A(C.sum), 8);
    set('.s2', A(C.bias), 8);
    set('.s3', A(C.res), 8);
    this.q('.s3').classList.toggle('glow', t >= C.res && t < C.res + 1.6);
    set('.many', A(C.many, 0.7));
  }
}

/* ================================================================ 说出「3」之前：每层的逻辑透镜 */

export class LensCard {
  constructor(parent, Q, F) {
    this.Q = Q; this.F = F;
    this.el = el('div', 'lens3', '', parent);
    const Ls = [12, 15, 18, 21, 24, 27];
    this.Ls = Ls;
    this.el.innerHTML = `<div class="h">说出「${esc(tokPlain(F.stA.chosenS))}」之前<small>逻辑透镜：每层直接接输出头，此刻会说什么</small></div>${Ls.map((L) => {
      const [s, p] = F.lens[L];
      const w = s.trim() || s;
      const junk = !/[一-鿿0-9]/.test(w) || p < 0.15;
      return `<div class="r${junk ? ' junk' : ''}" data-l="${L}"><span class="l">L${String(L).padStart(2, '0')}</span><span class="w">${esc(junk ? w.replace(/[^\x20-\x7e一-鿿０-９]/g, '·') : w)}</span><span class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></span><span class="p">${pct(p)}</span></div>`;
    }).join('')}`;
  }

  update(P) {
    const on = P && P.a > 0.001;
    this.el.style.display = on ? 'block' : 'none';
    if (!on) return;
    this.el.style.opacity = P.a.toFixed(3);
    this.el.querySelectorAll('.r').forEach((r, j) => {
      const a = smooth(clamp((P.k ?? 1) * this.Ls.length - j));
      r.style.opacity = a.toFixed(3);
      r.style.transform = `translateX(${((1 - a) * 14).toFixed(1)}px)`;
      const L = Number(r.dataset.l);
      const think = this.F.thinkL;
      r.classList.toggle('on', (L === think && (P.fin ?? 0) < 0.5) || (L === 27 && (P.fin ?? 0) >= 0.5));
    });
  }
}
