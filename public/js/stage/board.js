// 算式板：D6「一次乘加」和 D7「比特」的主角。
// 把正在算的这一个数写成一行从左到右的算式 y[j] = Σ x[i] × W[i, j]：
// 输入 x（蓝）、权重 W（紫）、乘积（琥珀）、结果 y（青）四种颜色和 3D 舞台、右侧讲解一致；
// 每乘出一项就飞进累加器，最后写明结果落进输出向量的哪一格、它是什么意思。数字全部来自真实模型。
import { esc, tokPlain, fmtPct } from '../ui.js';
import { bf16Value, fmtSci } from '../num.js';
import { termAt, termsDone } from './micro.js';
import { neuronId, sums } from './fields.js';
import { isEn, L as tr } from '../i18n.js';

const small = () => matchMedia('(max-width: 900px)').matches;
const Q_ID = (M) => M.Q?.id ?? '';
const easeOut = (t) => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);
const MINUS = '−';
// 带符号的数：+1.234 / −1.234
const sg = (v, d = 3) => (Number.isFinite(v) ? `${v >= 0 ? '+' : MINUS}${Math.abs(v).toFixed(d)}` : Number.isNaN(v) ? 'NaN' : v > 0 ? '+∞' : `${MINUS}∞`);
// 权重很小，按有效数字显示
const sw = (v) => (Math.abs(v) >= 0.01 || v === 0 ? sg(v, 4) : `${v >= 0 ? '+' : MINUS}${Math.abs(v).toPrecision(3)}`);
// 极大 / 极小 / 无穷都能写的数
// 科学计数法的指数写成上标：× 10^−21 → × 10<sup>−21</sup>
const sci = (v) => fmtSci(v).replace(/10\^([−-]?\d+)/, '10<sup>$1</sup>');
const sx = (v, d = 3) => (Number.isFinite(v) && Math.abs(v) < 1e5 && (Math.abs(v) >= 1e-3 || v === 0) ? sg(v, d) : (v > 0 ? '+' : '') + sci(v));
const pct = (v) => (Number.isFinite(v) ? fmtPct(v) : '—');

const STEPS = isEn ? {
  mm: [['pick', 'Pick a column'], ['mul', 'Multiply'], ['sum', 'Add up']],
  q: [['pick', 'Pick a column'], ['mul', 'Multiply'], ['sum', 'Add up'], ['rope', 'Norm + rotate']],
  dot: [['mul', 'Multiply'], ['sum', 'Add up'], ['scale', '÷√128 → weight']],
  neuron: [['silu', 'SiLU'], ['gate', 'Times u']],
} : {
  mm: [['pick', '选一列'], ['mul', '逐项相乘'], ['sum', '全部加起来']],
  q: [['pick', '选一列'], ['mul', '逐项相乘'], ['sum', '全部加起来'], ['rope', '归一化 + 旋转']],
  dot: [['mul', '逐维相乘'], ['sum', '加起来'], ['scale', '÷√128 → 权重']],
  neuron: [['silu', 'SiLU 激活'], ['gate', '乘上 u']],
};

export class Board {
  constructor(E, M) {
    this.E = E;
    this.M = M;
    const el = (this.el = document.createElement('section'));
    el.className = 'board';
    el.hidden = true;
    el.setAttribute('aria-label', tr('算式板', 'Formula board'));
    E.host.parentElement.append(el);
    this.key = '';
    this.fk = '';
    this.folded = false;
    el.addEventListener('click', (e) => this.onClick(e));
    new ResizeObserver(() => this.measure()).observe(el);
    addEventListener('resize', () => { this.key = ''; });
  }

  /* ------------------------------------------------------------ 每帧 */

  update(st) {
    const v = st.view, s = st.step;
    let mode = null;
    if (v === 'mm') mode = 'mm';
    else if (v === 'dot') mode = 'dot';
    else if (v === 'neuron') mode = 'neuron';
    else if (v === 'bits') mode = 'bits';
    if (!mode || st.dAnim < 4.6) return this.hide();
    const data = this.dataFor(mode, st);
    const key = data ? `${Q_ID(this.M)}|${mode}|${st.g}|${s.ph}|${s.L}|${s.sub}|${data.id}|${small()}` : `wait|${s.L}`;
    if (key !== this.key) {
      this.key = key;
      this.fk = '';
      this.last = null;
      this.data = data;
      this.mode = data ? mode : 'wait';
      this.el.innerHTML = data ? this.render(mode, data, st) : `<p class="bd-wait"><span class="spin"></span>${tr(`正在载入第 ${s.L} 层的乘加数据…`, `Loading the multiply-add data for layer ${s.L}…`)}</p>`;
      this.el.classList.toggle('folded', this.folded);
      this.show();
    }
    if (data) this.animate(st);
  }

  show() {
    if (!this.el.hidden) return;
    this.el.hidden = false;
    document.body.classList.add('has-board');
    this.measure();
  }

  hide() {
    if (this.el.hidden) return;
    this.el.hidden = true;
    this.key = '';
    document.body.classList.remove('has-board');
    this.E.setOverlay(0, 0);
  }

  // 算式板挡住的那一块画面告诉 Engine，3D 就在剩下的区域里取景
  measure() {
    if (this.el.hidden) return;
    const r = this.el.getBoundingClientRect(), h = this.E.host.getBoundingClientRect();
    if (!r.height) return;
    const atTop = r.top + r.height / 2 < h.top + h.height / 2;
    if (atTop) this.E.setOverlay(Math.max(0, r.bottom - h.top + 8), 0);
    else this.E.setOverlay(0, Math.max(0, h.bottom - r.top + 8));
  }

  dataFor(mode, st) {
    const M = this.M, Q = M.Q, s = st.step;
    if (!Q) return null;
    if (mode === 'mm') {
      const src = M.micro.source(st);
      return src && { id: `${src.kind}${src.e.j}`, src };
    }
    if (mode === 'dot') {
      const d = Q.dotAt(s.L, st.g);
      return d && { id: `dot${d.head}|${d.key}`, d, sum: sums(d, 128) };
    }
    if (mode === 'neuron') {
      const n = Q.neuronAt(s.L, st.g);
      return n && { id: `n${neuronId(n)}`, n };
    }
    const tg = M.detail.bitsTarget(st);
    return tg && { id: `b${tg.kind}|${tg.k}|${tg.label || ''}`, tg };
  }

  /* ------------------------------------------------------------ 搭建 */

  head(where, steps) {
    return `<header class="bd-h">
      <span class="bd-where">${where}</span>
      ${steps ? `<ol class="bd-steps">${steps.map(([mi, t], i) => `<li data-mi="${mi}"><b>${i + 1}</b>${t}</li>`).join('')}</ol>` : ''}
      <button class="bd-fold" type="button" title="${tr('收起 / 展开算式板', 'Collapse / expand the board')}" aria-label="${tr('收起 / 展开算式板', 'Collapse / expand the board')}">${this.folded ? '+' : '–'}</button>
    </header>`;
  }

  render(mode, data, st) {
    if (mode === 'mm') return this.renderMM(data.src, st);
    if (mode === 'dot') return this.renderDot(data, st);
    if (mode === 'neuron') return this.renderNeuron(data.n, st);
    return this.renderBits(data.tg, st);
  }

  // 一张“乘积清单”：前 R 项逐行列出，其余的合成一行，加起来正好等于 total
  table(rows, { xh, wh, ph, restN, rest, maxP }) {
    const bar = (v) => {
      const w = Math.min(100, (Math.abs(v) / maxP) * 100);
      return `<span class="bar"><i class="${v >= 0 ? 'pos' : 'neg'}" style="width:${(w / 2).toFixed(1)}%"></i></span>`;
    };
    const restPN = tr('<small>每一项都很小</small>', '<small>each one tiny</small>');
    return `<div class="bd-tbl">
      <div class="bd-tr bd-th"><span class="i">i</span><span class="cx">${xh}</span><span class="o"></span><span class="cw">${wh}</span><span class="o"></span><span class="cp">${ph}</span><span class="bar"></span></div>
      ${rows.map((r) => `<div class="bd-tr" data-k="${r.k}" data-d="${r.d}" title="${tr('点一下：之后按 ＋ 看这个权重的比特', "Click, then press ＋ to see this weight's bits")}"><span class="i">${r.d}</span><span class="cx">${sg(r.x, r.xd ?? 3)}</span><span class="o">×</span><span class="cw">${r.wf ? r.wf(r.w) : sw(r.w)}</span><span class="o">=</span><span class="cp" data-p>${sg(r.p, r.pd ?? 3)}</span>${bar(r.p)}</div>`).join('')}
      <div class="bd-tr bd-rest"><span class="i">…</span><span class="rest-t">${tr(`其余 ${restN.toLocaleString('zh-CN')} 项 ${restPN}`, `the other ${restN.toLocaleString('en-US')} terms ${restPN}`)}</span><span class="cp" data-p>${sg(rest, 3)}</span>${bar(rest)}</div>
    </div>`;
  }

  // 累加器 + 正负两堆 + 累计曲线 + 结果落点
  accHTML(yHtml, pn) {
    return `<div class="bd-acc">
        <div class="bd-acc-k"><span data-acck>${tr('累加器', 'Accumulator')}</span></div>
        <div class="bd-acc-v"><span class="cy" data-accy>${yHtml}</span> <span class="eq" data-acceq>≈</span> <b class="cp" data-accv>0</b></div>
        <div class="bd-meter"><i class="zero"></i><b data-meter></b></div>
      </div>
      ${pn ? `<div class="bd-pn" data-pn><span>${tr('所有正乘积', 'All positive products')} <b class="cp">${sg(pn[0], 2)}</b></span><span>${tr('所有负乘积', 'All negative products')} <b class="cp">${sg(pn[1], 2)}</b></span><span class="bd-pn-bar"><i class="neg" style="width:${(Math.abs(pn[1]) / (Math.abs(pn[0]) + Math.abs(pn[1])) * 100).toFixed(1)}%"></i><i class="pos" style="width:${(pn[0] / (Math.abs(pn[0]) + Math.abs(pn[1])) * 100).toFixed(1)}%"></i></span><em>${tr('正负大量抵消，剩下的才是结果', "Positives and negatives mostly cancel; what's left is the result")}</em></div>` : ''}`;
  }

  // 累计和随项数的变化（按 |乘积| 从大到小）。有导出的 cum 就画真实曲线，没有就画前几项 + 虚线连到总和
  chart(prods, total, n, cum, R) {
    const W = 248, H = 78, pad = 6;
    let pts = [];
    let run = 0;
    if (cum) pts = cum.map(([k, v]) => [k, v]);
    else { prods.forEach((p, k) => { run += p; pts.push([k + 1, run]); }); }
    const all = pts.map((p) => p[1]).concat([0, total]);
    const lo = Math.min(...all), hi = Math.max(...all), span = hi - lo || 1;
    const X = (k) => pad + (Math.log2(Math.max(1, k)) / Math.log2(Math.max(2, n))) * (W - pad * 2);
    const Y = (v) => H - pad - ((v - lo) / span) * (H - pad * 2);
    const line = pts.map(([k, v]) => `${X(k).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
    const tail = cum ? '' : `<line x1="${X(R)}" y1="${Y(run)}" x2="${X(n)}" y2="${Y(total)}" class="ch-dash"/>`;
    return `<figure class="bd-chart"><svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" preserveAspectRatio="none" aria-hidden="true">
        <clipPath id="bdclip"><rect data-clip x="0" y="0" width="0" height="${H}"/></clipPath>
        <line x1="${pad}" x2="${W - pad}" y1="${Y(0)}" y2="${Y(0)}" class="ch-zero"/>
        <line x1="${pad}" x2="${W - pad}" y1="${Y(total)}" y2="${Y(total)}" class="ch-total"/>
        <line x1="${X(R)}" x2="${X(R)}" y1="${pad - 2}" y2="${H - pad + 2}" class="ch-r"/>
        <g clip-path="url(#bdclip)"><polyline points="${line}" class="ch-line"/>${tail}</g>
        <circle cx="${X(n)}" cy="${Y(total)}" r="3" class="ch-end"/>
      </svg><figcaption>${tr(`累计和：按贡献从大到小一项项加（横轴是项数，对数刻度）。竖线是上面列出的前 ${R} 项，横线是最终结果`, `Running total, adding terms from the biggest contribution down (x axis: number of terms, log scale). The vertical line marks the ${R} terms listed above; the horizontal line is the final result`)}</figcaption></figure>`;
  }

  // 输出向量：一条长条，目标那一格亮起来
  vecHTML(src) {
    const segs = src.ySegs ? Array.from({ length: src.ySegs }, (_, i) => `<i class="${i === src.ySeg ? 'hot' : ''}"></i>`).join('') : '<i></i>';
    const at = ((src.e.j + 0.5) / src.yLen) * 100;
    return `<div class="bd-vec" title="${tr(`${src.y.replace(/<[^>]+>/g, '')} 一共 ${src.yLen} 个数`, `${src.y.replace(/<[^>]+>/g, '')} has ${src.yLen} numbers`)}"><span class="segs">${segs}</span><b style="left:${at.toFixed(2)}%"></b></div>`;
  }

  renderMM(src, st) {
    const e = src.e, R = small() ? 6 : 12;
    const rows = e.dims.slice(0, R).map((d, k) => ({ k, d, x: e.x[k], w: e.w[k], p: e.x[k] * e.w[k] }));
    const shownR = rows.reduce((a, r) => a + r.p, 0);
    const rest = e.total - shownR;
    const n = src.sum.n;
    const pn = src.sum.pos != null && src.sum.neg != null ? [src.sum.pos, src.sum.neg] : null;
    const maxP = Math.max(...rows.map((r) => Math.abs(r.p)), Math.abs(rest), 1e-9);
    this.calc = { rows, R, rest, total: e.total, n, shownR, scale: Math.max(Math.abs(e.total), Math.abs(shownR), ...(src.sum.cum || []).map((c) => Math.abs(c[1])), ...rows.map((_, k) => Math.abs(rows.slice(0, k + 1).reduce((a, r) => a + r.p, 0)))) * 1.15 || 1 };
    const yHtml = `${src.y}[${e.j}]`;
    const xi = `<span class="cx">${src.x}[i]</span>`, wi = `<span class="cw">${src.wIdx('i')}</span>`;
    this.cap = isEn ? {
      pick: `To compute <b class="cy">${yHtml}</b>, you only need <b>column ${e.j}</b> of <b class="cw">${src.w}</b>: its ${n.toLocaleString('en-US')} weights pair up one-to-one with the ${n.toLocaleString('en-US')} numbers of the input <b class="cx">${src.x}</b>.<span class="dim"> <b class="cx">${src.x}</b>: ${esc(src.xDesc)}; <b class="cw">${src.w}</b>: ${esc(src.wDesc)}.</span>`,
      mul: `Multiply term by term: the i-th input number × the i-th weight of this column. Below are the ${R} products with the largest absolute values; each one drops into the accumulator as soon as it is computed.<span class="dim"> Click a row, then press ＋ to see that weight's bits.</span>`,
      sum: `Add it all up: each of the other ${(n - R).toLocaleString('en-US')} terms is tiny, but there are a lot of them, so together they can't be ignored. Once they're all in, that's <b class="cy">${yHtml}</b>.`,
      rope: `Two more steps: the 128 numbers of the same head are <b>normalized</b> together (q_norm), then <b>rotated</b> according to the token's position (RoPE). Only then is it the q that gets compared with K.`,
    } : {
      pick: `要算 <b class="cy">${yHtml}</b>，只需要 <b class="cw">${src.w}</b> 的<b>第 ${e.j} 列</b>：这一列的 ${n.toLocaleString('zh-CN')} 个权重，正好和输入 <b class="cx">${src.x}</b> 的 ${n.toLocaleString('zh-CN')} 个数一一配对。<span class="dim"><b class="cx">${src.x}</b>：${esc(src.xDesc)}；<b class="cw">${src.w}</b>：${esc(src.wDesc)}。</span>`,
      mul: `逐项相乘：输入的第 i 个数 × 这一列的第 i 个权重。下面列出乘积绝对值最大的 ${R} 项，每乘出一项就加进累加器。<span class="dim">点一行，再按 ＋ 看那个权重的比特。</span>`,
      sum: `全部加起来：其余 ${(n - R).toLocaleString('zh-CN')} 项每一项都很小，但数量多，合起来也不能忽略。加完就是 <b class="cy">${yHtml}</b>。`,
      rope: `还要再加工两步：同一个头的 128 个数一起<b>归一化</b>（q_norm），再按词元的位置<b>旋转</b>（RoPE），才是拿去和 K 比较的 q。`,
    };
    const rope = src.rope ? this.ropeHTML(e, src) : '';
    return `${this.head(esc(src.where), STEPS[src.rope ? 'q' : 'mm'])}
      <div class="bd-scroll">
      <p class="bd-cap" data-cap></p>
      <div class="bd-body">
        <div class="bd-main">
          <div class="bd-fx"><span class="cy">${yHtml}</span> = <span class="op">Σ</span><sub class="lim">i</sub> ${xi} × ${wi}<small>${tr(`i 从 0 到 ${n - 1}，一共 ${n.toLocaleString('zh-CN')} 项`, `i from 0 to ${n - 1}, ${n.toLocaleString('en-US')} terms in all`)}</small></div>
          ${this.table(rows, { xh: `${src.x}[i]`, wh: src.wIdx('i'), ph: tr('乘积', 'product'), restN: n - R, rest, maxP })}
          ${rope}
        </div>
        <div class="bd-side">
          ${this.accHTML(yHtml, pn)}
          ${this.chart(rows.map((r) => r.p), e.total, n, src.sum.cum, R)}
          <div class="bd-dest" data-dest>
            <div class="bd-dest-h">${tr(`结果落进 <b class="cy">${src.y}</b> 的第 ${e.j} 格`, `The result lands in slot ${e.j} of <b class="cy">${src.y}</b>`)} · ${esc(src.yDesc)}</div>
            ${this.vecHTML(src)}
            <p>${src.dest}</p>
          </div>
        </div>
      </div>
      </div>`;
  }

  ropeHTML(e, src) {
    const pos = src.ropePos;
    const cos = Math.cos(e.angle), sin = Math.sin(e.angle);
    const lower = e.dim < 64;
    const deg = (e.angle * 180) / Math.PI;
    const turns = Math.floor(deg / 360);
    return `<div class="bd-rope" data-rope>
      <div class="bd-rp">${tr(`<b>1 · 归一化（q_norm）</b>第 ${e.head} 号头的 128 个数一起除以它们的均方根，再乘一个训练好的缩放 γ：`, `<b>1 · Normalize (q_norm)</b> All 128 numbers of head ${e.head} are divided by their root mean square, then multiplied by a trained scale γ:`)}
        <code><span class="cy">${sg(e.total)}</span> ÷ ${e.rms.toFixed(3)} × ${e.qnW.toFixed(4)} = <span class="cy">${sg(e.qn, 4)}</span></code></div>
      <div class="bd-rp bd-rp2">${dial(e)}<div>${isEn
        ? `<b>2 · Rotate (RoPE)</b> Dimensions ${e.dim} and ${e.partner} form a pair, seen as a point in a plane. This token is at position ${pos}, so it turns by θ = ${pos} ÷ 10⁶<sup>${2 * e.freq}/128</sup> = ${e.angle.toFixed(3)} radians${turns > 0 ? ` (${turns} full turn${turns > 1 ? 's' : ''} plus ${(deg - turns * 360).toFixed(0)}°)` : ` (${deg.toFixed(0)}°)`}:`
        : `<b>2 · 旋转（RoPE）</b>第 ${e.dim} 维和第 ${e.partner} 维配成一对，看成平面上的一个点；这个词元在第 ${pos} 位，就转 θ = ${pos} ÷ 10⁶<sup>${2 * e.freq}/128</sup> = ${e.angle.toFixed(3)} 弧度${turns > 0 ? `（${turns} 圈多 ${(deg - turns * 360).toFixed(0)}°）` : `（${deg.toFixed(0)}°）`}：`}
        <code>${sg(e.qn, 4)} × cos θ ${lower ? MINUS : '+'} (${sg(e.qnP, 4)}) × sin θ = <span class="cy">${sg(lower ? e.qn * cos - e.qnP * sin : e.qn * cos + e.qnP * sin, 4)}</span></code>
        <small>${tr('位置越靠后转得越多；两个词元的点积因此只取决于它们相隔多远。', 'Later positions turn further, so the dot product of two tokens depends only on how far apart they are.')}</small></div></div>
    </div>`;
  }

  renderDot(D, st) {
    const d = D.d, Q = this.M.Q, R = small() ? 6 : 12;
    const tok = esc(tokPlain(Q.tokens[d.key]?.s ?? ''));
    const rows = d.dims.slice(0, R).map((dim, k) => ({ k, d: dim, x: d.q[k], w: d.k[k], p: d.q[k] * d.k[k], xd: 3, wf: (v) => sg(v, 3), pd: 2 }));
    const shownR = rows.reduce((a, r) => a + r.p, 0);
    const n = D.sum.n, rest = d.sum - shownR;
    const pn = D.sum.pos != null && D.sum.neg != null ? [D.sum.pos, D.sum.neg] : null;
    const maxP = Math.max(...rows.map((r) => Math.abs(r.p)), Math.abs(rest), 1e-9);
    this.calc = { rows, R, rest, total: d.sum, n, shownR, scale: Math.max(Math.abs(d.sum), ...(D.sum.cum || []).map((c) => Math.abs(c[1])), ...rows.map((_, k) => Math.abs(rows.slice(0, k + 1).reduce((a, r) => a + r.p, 0)))) * 1.15 || 1 };
    const ctxN = Q.P + st.g;
    this.cap = isEn ? {
      mul: `Head ${d.head} takes the current token's <b class="cx">q</b> (128 numbers, already through q_norm and RoPE) and the <b class="cw">k</b> of “${tok}” (128 numbers, kept in the KV cache), and multiplies them dimension by dimension. Below are the ${R} dimensions with the largest products.<span class="dim"> Click a row, then press ＋ to see the bits of that number.</span>`,
      sum: `Adding up all 128 dimensions gives q·k: the more the two vectors point the same way, the bigger this number, meaning “${tok}” is a better match for the current position.`,
      scale: `Divide by √128 ≈ 11.31 to keep the numbers from getting too big, then softmax it together with the scores of all ${ctxN} tokens so far to get attention weights.`,
    } : {
      mul: `第 ${d.head} 号头拿当前词元的 <b class="cx">q</b>（128 个数，已经过 q_norm 和 RoPE）和「${tok}」的 <b class="cw">k</b>（128 个数，存在 KV 缓存里）逐维相乘。下面是乘积最大的 ${R} 维。<span class="dim">点一行，再按 ＋ 看那个数的比特。</span>`,
      sum: `128 维全部加起来就是 q·k：两个向量越“同向”，这个数越大，说明「${tok}」和当前位置越对得上。`,
      scale: `除以 √128 ≈ 11.31，防止数值太大；再和前面全部 ${ctxN} 个词元的打分一起做 softmax，变成注意力权重。`,
    };
    const yHtml = 'q·k';
    return `${this.head(tr(`第 ${st.step.L} 层 · 注意力 · 第 ${d.head} 号头给「${tok}」打分`, `Layer ${st.step.L} · Attention · head ${d.head} scores “${tok}”`), STEPS.dot)}
      <div class="bd-scroll">
      <p class="bd-cap" data-cap></p>
      <div class="bd-body">
        <div class="bd-main">
          <div class="bd-fx"><span class="cy">q·k</span> = <span class="op">Σ</span><sub class="lim">d</sub> <span class="cx">q[d]</span> × <span class="cw">k[d]</span><small>${tr('d 从 0 到 127，一共 128 维', 'd from 0 to 127, 128 dims in all')}</small></div>
          ${this.table(rows, { xh: 'q[d]', wh: 'k[d]', ph: tr('乘积', 'product'), restN: n - R, rest, maxP }).replace(/title="[^"]*"/g, `title="${tr('点一下：之后按 ＋ 看这个数的比特', 'Click, then press ＋ to see the bits of this number')}"`)}
        </div>
        <div class="bd-side">
          ${this.accHTML(yHtml, pn)}
          ${this.chart(rows.map((r) => r.p), d.sum, n, D.sum.cum, R)}
          <div class="bd-dest" data-dest>
            <div class="bd-dest-h">${tr('打分', 'score')} = <span class="cy">q·k</span> ÷ √128</div>
            <div class="bd-big"><span class="cy">${sg(d.sum, 2)}</span> ÷ 11.314 = <b class="cy">${sg(d.score, 3)}</b></div>
            <div class="bd-big">softmax → <b class="cy">${pct(d.w)}</b></div>
            <p>${tr(`第 ${d.head} 号头把 <b>${pct(d.w)}</b> 的注意力给了「${tok}」：之后它会按这个比例，从「${tok}」的 V 里取信息。`, `Head ${d.head} gives <b>${pct(d.w)}</b> of its attention to “${tok}”: it will then take information from the V of “${tok}” in that proportion.`)}</p>
          </div>
        </div>
      </div>
      </div>`;
  }

  renderNeuron(n, st) {
    const id = neuronId(n);
    const sig = n.silu / n.gz;
    const a = n.silu * n.uz;
    this.cap = isEn ? {
      silu: `<b class="cx">g</b> first goes through <b>SiLU</b>: negative numbers are squashed to nearly 0, positive ones pass through almost unchanged. It decides how far this neuron "opens".`,
      gate: `Then multiply by <b class="cw">u</b>: SiLU(g) acts like a valve that decides how much of u gets through. The product <b class="cy">a</b> is the output of neuron #${id}.`,
    } : {
      silu: `<b class="cx">g</b> 先过 <b>SiLU</b>：负数被压到接近 0，正数几乎原样通过。它决定这个神经元“开多大”。`,
      gate: `再乘上 <b class="cw">u</b>：SiLU(g) 像一道阀门，决定放多少 u 通过。乘出来的 <b class="cy">a</b> 就是神经元 #${id} 的输出。`,
    };
    this.calc = null;
    return `${this.head(tr(`第 ${st.step.L} 层 · 前馈 · 神经元 #${id}`, `Layer ${st.step.L} · Feed-forward · neuron #${id}`), STEPS.neuron)}
      <div class="bd-scroll">
      <p class="bd-cap" data-cap></p>
      <div class="bd-body">
        <div class="bd-main">
          <div class="bd-ins">
            <div><span class="cx">g[${id}]</span> = <b class="cx">${sg(n.gz)}</b><small>${tr(`上一步 W<sub>gate</sub> 第 ${id} 列乘加出来的`, `from column ${id} of W<sub>gate</sub> (previous step)`)}</small></div>
            <div><span class="cw">u[${id}]</span> = <b class="cw">${sg(n.uz)}</b><small>${tr('同一个神经元在 W<sub>up</sub> 里那一列乘加出来的', "from the same neuron's column in W<sub>up</sub>")}</small></div>
          </div>
          ${siluSVG(n.gz)}
        </div>
        <div class="bd-side">
          <div class="bd-eqs">
            <div data-eq="silu"><small>${tr('SiLU(g) = g × σ(g)，σ 是 0 到 1 之间的 S 形曲线', 'SiLU(g) = g × σ(g), where σ is an S-shaped curve between 0 and 1')}</small>
              <code>SiLU(${sg(n.gz)}) = ${sg(n.gz)} × ${sig.toFixed(3)} = <b class="cx">${sg(n.silu)}</b></code></div>
            <div data-eq="gate"><small>a = SiLU(g) × u</small>
              <code><span class="cy">a[${id}]</span> = ${sg(n.silu)} × ${sg(n.uz)} = <b class="cy">${sg(a)}</b></code></div>
          </div>
          <div class="bd-dest" data-dest>
            <div class="bd-dest-h">${tr(`结果落进 <b class="cy">a</b> 的第 ${id} 格`, `The result lands in slot ${id} of <b class="cy">a</b>`)}</div>
            <div class="bd-vec"><span class="segs">${Array.from({ length: 12 }, (_, i) => `<i class="${i === Math.floor(id / 256) ? 'hot' : ''}"></i>`).join('')}</span><b style="left:${(((id + 0.5) / 3072) * 100).toFixed(2)}%"></b></div>
            <p>${isEn
              ? `Each of the 3072 neurons computes one number like this; together they form a (1 × 3072), which goes on to the down-projection matrix W<sub>down</sub>.${a > 1 ? ' This neuron clearly "lit up" this time.' : a < 0.05 && a > -0.05 ? ' This neuron barely lit up this time.' : ''}`
              : `3072 个神经元各自这样算出一个数，组成 a（1 × 3072），下一步交给降维矩阵 W<sub>down</sub>。${a > 1 ? '这个神经元这次被明显“点亮”了。' : a < 0.05 && a > -0.05 ? '这个神经元这次几乎没亮。' : ''}`}</p>
          </div>
        </div>
      </div>
      </div>`;
  }

  renderBits(tg, st) {
    const name = tg.kind === 'q' ? `q[${tg.d.dims[tg.k]}]` : esc(tg.label);
    const what = isEn
      ? (tg.kind === 'q' ? `dimension ${tg.d.dims[tg.k]} of the current token's query q in head ${tg.d.head}` : `${tg.wHtml}: one of the weights used for ${esc(tg.yDesc)}`)
      : (tg.kind === 'q' ? `当前词元在第 ${tg.d.head} 号头的查询 q 的第 ${tg.d.dims[tg.k]} 维` : `${tg.wHtml}：${esc(tg.yDesc)} 用到的一个权重`);
    const bits = Array.from({ length: 16 }, (_, i) => `<button type="button" class="bd-bit ${i === 0 ? 's' : i <= 8 ? 'e' : 'm'}" data-i="${i}" aria-label="${tr(`第 ${i} 位`, `bit ${i}`)}">0</button>`);
    return `${this.head(`${tr('比特', 'Bits')} · ${name}`, null)}
      <div class="bd-scroll">
      <p class="bd-cap">${isEn
        ? `${tg.kind === 'q' ? 'Activations are' : 'Every weight in the model is'} stored in memory as <b>16 zeros and ones</b> (bfloat16 format). <b>Click any bit</b> to flip it and see what happens.<span class="dim"> ${what}</span>`
        : `${tg.kind === 'q' ? '激活值' : '模型里的每个权重'}在显存里都是 <b>16 个 0/1</b>（bfloat16 格式）。<b>点任意一位</b>把它翻过来，看看会发生什么。<span class="dim">${what}</span>`}</p>
      <div class="bd-bitrow">
        <div class="bd-bits"><span class="g s">${bits[0]}</span><span class="g e">${bits.slice(1, 9).join('')}</span><span class="g m">${bits.slice(9).join('')}</span></div>
        <div class="bd-bitk"><span class="s">${tr('符号', 'Sign')}</span><span class="e">${tr('指数 · 8 位', 'Exponent · 8 bits')}</span><span class="m">${tr('尾数 · 7 位', 'Mantissa · 7 bits')}</span></div>
      </div>
      <div class="bd-body bd-bits-body">
        <div class="bd-main">
          <div class="bd-dec" data-dec></div>
          <div class="bd-fx bd-bfx" data-bfx></div>
        </div>
        <div class="bd-side">
          <div class="bd-eff" data-eff></div>
        </div>
      </div>
      </div>`;
  }

  /* ------------------------------------------------------------ 动画 */

  animate(st) {
    const s = st.step, p = st.p, mi = s.mi;
    if (this.mode === 'bits') return this.animateBits(st);
    const order = (STEPS[this.mode === 'mm' ? (this.data.src.rope ? 'q' : 'mm') : this.mode] || []).map((x) => x[0]);
    const at = order.indexOf(mi);
    const c = this.calc;
    let doneN = 0, val = 0, sumP = 0, cur = -1;
    if (c) {
      const mulAt = order.indexOf('mul'), sumAt = order.indexOf('sum');
      doneN = at === mulAt ? termsDone(p, c.R) : at > mulAt ? c.R : 0;
      if (at === mulAt) for (let k = 0; k < c.R; k++) { const t = termAt(p, k, c.R); if (t > 0 && t < 0.62) { cur = k; break; } }
      sumP = at === sumAt ? easeOut(seg01(p, 0.1, 0.9)) : at > sumAt ? 1 : 0;
      val = c.rows.slice(0, doneN).reduce((a, r) => a + r.p, 0) + c.rest * sumP;
    }
    const fk = `${mi}|${doneN}|${cur}|${Math.round(sumP * 60)}|${mi === 'silu' ? Math.round(p * 40) : ''}|${this.folded}`;
    if (fk === this.fk) return;
    this.fk = fk;
    const el = this.el;
    el.querySelectorAll('.bd-steps li').forEach((li) => {
      const k = order.indexOf(li.dataset.mi);
      li.classList.toggle('on', k === at);
      li.classList.toggle('done', k < at);
    });
    const cap = el.querySelector('[data-cap]');
    if (cap && this.cap[mi] && cap._mi !== mi) { cap.innerHTML = this.cap[mi]; cap._mi = mi; }
    if (this.mode === 'neuron') return this.animateNeuron(st, at);
    // 清单：已经乘完的亮起来，正在乘的那一行高亮
    const sel = this.selectedK();
    el.querySelectorAll('.bd-tr[data-k]').forEach((tr) => {
      const k = Number(tr.dataset.k);
      tr.classList.toggle('done', k < doneN);
      tr.classList.toggle('cur', k === cur);
      tr.classList.toggle('pend', at === order.indexOf('pick') || (k >= doneN && k !== cur));
      tr.classList.toggle('sel', k === sel);
    });
    const restRow = el.querySelector('.bd-rest');
    restRow?.classList.toggle('done', sumP > 0.02);
    // 累加器
    const accv = el.querySelector('[data-accv]'), acck = el.querySelector('[data-acck]'), eq = el.querySelector('[data-acceq]');
    if (accv) {
      const final = sumP >= 0.999;
      accv.textContent = at === 0 && this.mode === 'mm' ? '?' : sg(val, this.mode === 'dot' ? 2 : 3);
      eq.textContent = final ? '=' : '≈';
      acck.textContent = isEn
        ? (at === 0 && this.mode === 'mm' ? 'Accumulator: not started' : sumP > 0 ? (final ? `all ${c.n.toLocaleString('en-US')} terms added` : `adding the other ${(c.n - c.R).toLocaleString('en-US')} terms…`) : `Accumulator · sum of the first ${doneN} terms`)
        : (at === 0 && this.mode === 'mm' ? '累加器：还没开始' : sumP > 0 ? (final ? `${c.n.toLocaleString('zh-CN')} 项全部加完` : `再加上其余 ${(c.n - c.R).toLocaleString('zh-CN')} 项…`) : `累加器 · 前 ${doneN} 项之和`);
      const m = el.querySelector('[data-meter]');
      const f = Math.max(-1, Math.min(1, val / c.scale));
      m.style.left = f >= 0 ? '50%' : `${50 + f * 50}%`;
      m.style.width = `${Math.abs(f) * 50}%`;
      el.querySelector('.bd-acc').classList.toggle('final', final);
    }
    el.querySelector('[data-pn]')?.classList.toggle('on', sumP > 0.3);
    // 累计曲线：只露出已经加到的部分
    const clip = el.querySelector('[data-clip]');
    if (clip && c) {
      const W = 248, pad = 6, L2 = Math.log2(Math.max(2, c.n));
      const rank = sumP > 0 ? c.R * Math.pow(c.n / c.R, sumP) : Math.max(doneN, 0);
      const x = rank <= 0 ? 0 : pad + (Math.log2(Math.max(1, rank)) / L2) * (W - pad * 2) + 3;
      clip.setAttribute('width', x.toFixed(1));
    }
    el.querySelector('[data-dest]')?.classList.toggle('on', this.mode === 'dot' ? at === 2 : sumP > 0.6 || at > order.indexOf('sum'));
    el.querySelectorAll('.bd-dest .bd-big').forEach((b, i) => b.classList.toggle('on', at === 2 && st.p > 0.15 + i * 0.35));
    // 旋转那一步：清单收起，换成 q_norm + RoPE
    const rope = el.querySelector('[data-rope]');
    if (rope) {
      const on = mi === 'rope';
      rope.classList.toggle('on', on);
      el.querySelector('.bd-tbl')?.classList.toggle('gone', on);
      el.querySelector('.bd-chart')?.classList.toggle('gone', on);
      el.querySelector('[data-pn]')?.classList.toggle('gone', on);
    }
    // 乘出来的一项飞进累加器（只在顺着播放时）
    const last = this.last;
    if (last && last.mi === mi && doneN === last.doneN + 1 && !reduced) this.fly(el.querySelector(`.bd-tr[data-k="${doneN - 1}"] [data-p]`));
    if (last && last.mi === mi && sumP > 0 && last.sumP === 0 && !reduced) this.fly(el.querySelector('.bd-rest [data-p]'));
    this.last = { mi, doneN, sumP };
  }

  animateNeuron(st, at) {
    const el = this.el, n = this.data.n, p = st.p;
    const sp = at === 0 ? easeOut(p) : 1;
    // SiLU 曲线上的小球从左边滑到 g
    const ball = el.querySelector('[data-ball]');
    if (ball) {
      const g = Math.max(-6, Math.min(6, n.gz));
      const x = -6 + (g + 6) * sp, y = x / (1 + Math.exp(-x));
      ball.setAttribute('cx', sx2(x)); ball.setAttribute('cy', sy2(y));
      el.querySelector('[data-guide]')?.classList.toggle('on', sp > 0.95);
    }
    el.querySelector('[data-eq="silu"]')?.classList.toggle('on', at >= 0 && sp > 0.5);
    el.querySelector('[data-eq="gate"]')?.classList.toggle('on', at >= 1);
    el.querySelector('[data-dest]')?.classList.toggle('on', at >= 1 && p > 0.4);
  }

  animateBits(st) {
    const D = this.M.detail, tg = this.data.tg;
    const flips = D.flips.get(D.bitsKey);
    const bits = flips || D.baseBits;
    if (!bits) return;
    const fk = `${bits.join('')}|${this.folded}`;
    if (fk === this.fk) return;
    this.fk = fk;
    const el = this.el;
    const base = D.baseBits;
    el.querySelectorAll('.bd-bit').forEach((b, i) => {
      b.textContent = bits[i];
      b.classList.toggle('one', bits[i] === 1);
      b.classList.toggle('flip', bits[i] !== base[i]);
    });
    const val = bf16Value(bits), old = bf16Value(base);
    const E = val.e - 127;
    const ebin = bits.slice(1, 9).join(''), mbin = bits.slice(9).join('');
    const special = isEn
      ? (val.e === 255 ? (val.m ? ' (exponent all 1s, mantissa not 0: NaN, "not a number")' : ' (exponent all 1s: infinity)') : val.e === 0 ? ' (exponent all 0s: a subnormal number; the 1 + in the formula becomes 0 +)' : '')
      : (val.e === 255 ? (val.m ? '（指数全 1、尾数不为 0：NaN，“不是一个数”）' : '（指数全 1：无穷大）') : val.e === 0 ? '（指数全 0：非规格化数，公式里的 1 + 要换成 0 +）' : '');
    el.querySelector('[data-dec]').innerHTML = `
      <span class="s">s = ${val.s} → ${val.s ? tr('负', 'negative') : tr('正', 'positive')}</span>
      <span class="e">e = ${ebin}₂ = ${val.e} → 2<sup>${val.e}−127</sup> = 2<sup>${E}</sup></span>
      <span class="m">m = ${mbin}₂ = ${val.m} → 1 + ${val.m}/128 = ${(1 + val.m / 128).toFixed(4)}</span>`;
    el.querySelector('[data-bfx]').innerHTML = `${tr('值', 'value')} = (−1)<sup class="s">${val.s}</sup> × 2<sup class="e">${val.e}−127</sup> × (1 + <span class="m">${val.m}</span>/128) = <b class="cw">${sx(val.value, 6)}</b><small>${special}</small>`;
    // 翻转的后果：只改了这一个数，下游那一个输出元素怎么变（精确重算：总和里只有这一项变了）
    const flipped = !!flips && bits.some((b, i) => b !== base[i]);
    const ov = old.value, nv = val.value, dv = nv - ov;
    const row = (k, a, b, extra = '') => `<div class="bd-ef"><span class="k">${k}</span><span class="v"><span class="a">${a}</span><span class="to">→</span><b>${b}</b>${extra}</span></div>`;
    let h = '';
    if (tg.kind === 'q') {
      const d = tg.d, kd = d.k[tg.k];
      const s2 = d.sum + kd * dv, sc2 = s2 / Math.sqrt(128);
      const w2 = 1 / (1 + (1 / d.w - 1) * Math.exp(d.score - sc2));
      h = flipped
        ? row(`q[${d.dims[tg.k]}]`, sx(ov, 4), sx(nv, 4), ratio(ov, nv)) + row('q·k', sg(d.sum, 2), sx(s2, 2)) + row(tr('打分', 'score'), sg(d.score, 3), sx(sc2, 3)) + row(tr('注意力权重', 'attention weight'), pct(d.w), pct(w2))
          + (isEn
            ? `<p>Only this one term of q·k changed: new q·k = old ${sg(d.sum, 2)} + k[${d.dims[tg.k]}] × (new − old) = ${sg(d.sum, 2)} + ${par(sg(kd, 3))} × ${par(sx(dv, 4))}.</p>`
            : `<p>q·k 里只有这一项变了：新的 q·k = 原来的 ${sg(d.sum, 2)} + k[${d.dims[tg.k]}] × (新 − 旧) = ${sg(d.sum, 2)} + ${par(sg(kd, 3))} × ${par(sx(dv, 4))}。</p>`)
        : (isEn
          ? `<p>It is multiplied by k[${d.dims[tg.k]}] = ${sg(kd, 3)} of “${esc(tokPlain(this.M.Q.tokens[d.key]?.s ?? ''))}”, contributing <b class="cp">${sg(ov * kd, 2)}</b>.</p>`
          : `<p>它和「${esc(tokPlain(this.M.Q.tokens[d.key]?.s ?? ''))}」的 k[${d.dims[tg.k]}] = ${sg(kd, 3)} 相乘，贡献了 <b class="cp">${sg(ov * kd, 2)}</b>。</p>`);
    } else {
      const t2 = tg.total + tg.x * dv;
      h = flipped ? row(tr('这个权重', 'this weight'), sx(ov, 6), sx(nv, 6), ratio(ov, nv)) + row(tg.out, sg(tg.total), sx(t2), `<small>${diff(t2 - tg.total)}</small>`) : '';
      if (flipped && tg.e2) {
        const silu = (z) => (Number.isFinite(z) ? z / (1 + Math.exp(-z)) : z > 0 ? Infinity : 0);
        const u = tg.e2.total;
        h += row(tr('神经元输出 a', 'neuron output a'), sg(silu(tg.total) * u), sx(silu(t2) * u));
      }
      if (flipped && tg.head) {
        const stp = this.M.Q.steps[st.g];
        const p1 = stp.top.find((x) => x[0] === tg.head.j)?.[1];
        if (p1) h += row(tr(`「${esc(tokPlain(tg.head.token))}」的概率`, `probability of “${esc(tokPlain(tg.head.token))}”`), pct(p1), pct(1 / (1 + (1 / p1 - 1) * Math.exp(tg.total - t2))));
      }
      h += isEn
        ? (flipped
          ? `<p>Only this one term of ${tg.out} changed: new value = old ${sg(tg.total)} + ${tg.xHtml} × (new weight − old weight) = ${sg(tg.total)} + ${par(sg(tg.x))} × ${par(sx(dv, 5))}.</p>`
          : `<p>This weight multiplies <span class="cx">${tg.xHtml} = ${sg(tg.x)}</span>, contributing <b class="cp">${sg(tg.x * ov, 4)}</b>, one of the largest terms in ${tg.out}.</p>`)
        : (flipped
          ? `<p>${tg.out} 里只有这一项变了：新值 = 原来的 ${sg(tg.total)} + ${tg.xHtml} × (新权重 − 旧权重) = ${sg(tg.total)} + ${par(sg(tg.x))} × ${par(sx(dv, 5))}。</p>`
          : `<p>这个权重乘的是 <span class="cx">${tg.xHtml} = ${sg(tg.x)}</span>，贡献了 <b class="cp">${sg(tg.x * ov, 4)}</b>，是 ${tg.out} 里最大的几项之一。</p>`);
    }
    if (!flipped) h += tr('<p class="dim">试试看：翻<b class="e">指数</b>位，数会成倍地变大变小；翻<b class="m">尾数</b>位，只改一点点；翻<b class="s">符号</b>位，正负颠倒。</p>', '<p class="dim">Try it: flip an <b class="e">exponent</b> bit and the number grows or shrinks by powers of two; flip a <b class="m">mantissa</b> bit and it changes only a little; flip the <b class="s">sign</b> bit and it changes sign.</p>');
    else h += `<button type="button" class="bd-reset">${tr('还原', 'Reset')}</button>`;
    el.querySelector('[data-eff]').innerHTML = `<div class="bd-eff-h">${flipped ? tr('翻转之后', 'After the flip') : tr('它在算式里', 'In the formula')}</div>${h}`;
  }

  // 飞行的乘积：从清单里那一格飞到累加器的数字上
  fly(from) {
    const to = this.el.querySelector('[data-accv]');
    if (!from || !to || this.folded) return;
    const a = from.getBoundingClientRect(), b = to.getBoundingClientRect(), o = this.el.getBoundingClientRect();
    if (!a.width || !b.width) return;
    const chip = document.createElement('span');
    chip.className = 'bd-fly';
    chip.textContent = from.textContent;
    chip.style.left = `${a.left - o.left + this.el.scrollLeft}px`;
    chip.style.top = `${a.top - o.top}px`;
    this.el.append(chip);
    const dx = b.left - a.left + (b.width - a.width) / 2, dy = b.top - a.top;
    chip.animate([
      { transform: 'translate(0, 0) scale(1)', opacity: 1 },
      { transform: `translate(${dx * 0.5}px, ${dy * 0.5 - 18}px) scale(1.15)`, opacity: 1, offset: 0.5 },
      { transform: `translate(${dx}px, ${dy}px) scale(0.7)`, opacity: 0 },
    ], { duration: 520, easing: 'cubic-bezier(.4,0,.2,1)' }).onfinish = () => chip.remove();
    to.animate([{ transform: 'scale(1.18)', color: '#fff' }, { transform: 'scale(1)' }], { duration: 380, delay: 380, easing: 'ease-out' });
  }

  selectedK() {
    if (this.mode === 'dot') return this.M.detail.dotSel ?? 0;
    const e = this.data?.src?.e;
    if (!e) return 0;
    return Math.max(0, e.dims.indexOf(this.M.micro.selD ?? e.dims[0]));
  }

  /* ------------------------------------------------------------ 点击 */

  onClick(e) {
    const t = e.target;
    if (t.closest('.bd-fold')) {
      this.folded = !this.folded;
      this.el.classList.toggle('folded', this.folded);
      t.closest('.bd-fold').textContent = this.folded ? '+' : '–';
      this.fk = '';
      requestAnimationFrame(() => this.measure());
      return;
    }
    const bit = t.closest('.bd-bit');
    if (bit) { this.E.onPick?.({ type: 'key', i: Number(bit.dataset.i), click: true }); this.fk = ''; return; }
    if (t.closest('.bd-reset')) { this.M.detail.flips.delete(this.M.detail.bitsKey); this.fk = ''; return; }
    const tr = t.closest('.bd-tr[data-k]');
    if (tr && (this.mode === 'mm' || this.mode === 'dot')) {
      const k = Number(tr.dataset.k), d = Number(tr.dataset.d);
      // 和点 3D 格子走同一条路：记下选中的那一项，乘法这一步里直接进入比特层。
      // 打分里选的是 q 的某一维，不要动矩阵那边已经选好的格子
      if (this.mode === 'dot') this.M.detail.dotSel = k;
      this.E.onPick?.({ type: 'mmcell', d: this.mode === 'dot' ? this.M.micro.selD : d, click: true });
      this.fk = '';
    }
  }
}

const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const seg01 = (p, a, b) => Math.min(1, Math.max(0, (p - a) / (b - a)));
const ratio = (a, b) => {
  if (!Number.isFinite(b) || !a) return '';
  const r = b / a;
  if (r === 1) return '';
  return `<small>${r < 0 ? tr('变号，', 'sign flipped, ') : ''}×${Math.abs(r) >= 100 || Math.abs(r) < 0.01 ? sci(Math.abs(r)) : Math.abs(r).toFixed(Math.abs(r) >= 10 ? 1 : 3)}</small>`;
};
const diff = (d) => (Number.isFinite(d) ? `${tr('变化', 'change')} ${sx(d)}` : '');
// 算式里的带符号数：负数加括号，免得出现“+ −2.1”
const par = (s) => (s.startsWith(MINUS) ? `(${s})` : s);

// SiLU 曲线（x ∈ [−6, 6]）
const sx2 = (x) => (12 + ((x + 6) / 12) * 216).toFixed(1);
const sy2 = (y) => (96 - ((y + 0.6) / 6.8) * 88).toFixed(1);
function siluSVG(g) {
  const pts = [];
  for (let i = 0; i <= 96; i++) { const x = -6 + i / 8; pts.push(`${sx2(x)},${sy2(x / (1 + Math.exp(-x)))}`); }
  const gx = Math.max(-6, Math.min(6, g)), gy = gx / (1 + Math.exp(-gx));
  return `<figure class="bd-silu"><svg viewBox="0 0 240 104" width="100%" height="104" aria-hidden="true">
      <line x1="12" x2="228" y1="${sy2(0)}" y2="${sy2(0)}" class="ch-zero"/><line x1="${sx2(0)}" x2="${sx2(0)}" y1="6" y2="98" class="ch-zero"/>
      <polyline points="${pts.join(' ')}" class="ch-silu"/>
      <g data-guide class="guide"><line x1="${sx2(gx)}" x2="${sx2(gx)}" y1="${sy2(0)}" y2="${sy2(gy)}"/><line x1="${sx2(0)}" x2="${sx2(gx)}" y1="${sy2(gy)}" y2="${sy2(gy)}"/></g>
      <circle data-ball r="4.5" cx="${sx2(-6)}" cy="${sy2(0)}" class="ch-ball"/>
      <text x="16" y="14">SiLU(g)</text><text x="200" y="${Number(sy2(0)) + 12}">g</text>
    </svg><figcaption>${tr('负的 g 被压到接近 0（阀门关上）；正的 g 几乎原样通过（阀门打开）', 'A negative g is squashed to nearly 0 (valve closed); a positive g passes through almost unchanged (valve open)')}</figcaption></figure>`;
}

// RoPE 表盘：这一对数 (d, d±64) 在平面上转过的角度（q_norm 之后、旋转前后）
function dial(e) {
  const R = 40, cx = 50, cy = 50;
  const mx = Math.max(Math.hypot(e.qn, e.qnP), Math.hypot(e.qr, e.qrP), 1e-6);
  const p = (a, b) => [cx + (a / mx) * R, cy - (b / mx) * R];
  const [x0, y0] = p(e.qn, e.qnP), [x1, y1] = p(e.qr, e.qrP);
  return `<svg class="bd-dial" width="100" height="100" viewBox="0 0 100 100" aria-hidden="true">
    <circle cx="${cx}" cy="${cy}" r="${R}" class="ch-zero" fill="none"/>
    <line x1="${cx - R - 4}" y1="${cy}" x2="${cx + R + 4}" y2="${cy}" class="ch-zero"/>
    <line x1="${cx}" y1="${cy - R - 4}" x2="${cx}" y2="${cy + R + 4}" class="ch-zero"/>
    <line x1="${cx}" y1="${cy}" x2="${x0}" y2="${y0}" class="dl-before"/>
    <line x1="${cx}" y1="${cy}" x2="${x1}" y2="${y1}" class="dl-after"/>
    <circle cx="${x1}" cy="${y1}" r="3.5" class="dl-dot"/>
    <text x="3" y="97">${tr('虚线 旋转前 · 实线 旋转后', '- - before · — after')}</text>
  </svg>`;
}
