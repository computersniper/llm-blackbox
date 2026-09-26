import { floatBits, bitsToValue, quantize, FORMATS, QUANT_SCALE, rngFor } from '../model.js';
import { $, $$, fmtNum, countUp, sleep, fitCanvas, reducedMotion } from '../ui.js';

const ORDER = ['FP32', 'BF16', 'FP16', 'INT8', 'INT4'];
const BYTES = { FP32: 4, BF16: 2, FP16: 2, INT8: 1, INT4: 0.5 };
const MODELS = [
  { name: 'GPT-2 Small', n: 124439808 },
  { name: '7B 模型', n: 7e9 },
  { name: '70B 模型', n: 70e9 },
];
const MONTAGE = [
  { R: 16, n: 768, sub: '一个神经元的全部输入权重' },
  { R: 24, n: 589824, sub: '一个 768 × 768 的注意力投影矩阵' },
  { R: 32, n: 7087872, sub: '一整层变换器块' },
  { R: 40, n: 124439808, sub: '整个 GPT-2 Small' },
  { R: 52, n: 70e9, sub: '一个 700 亿参数的模型（如 Llama 3 70B）' },
];

const fmtVal = (v) => {
  if (Number.isNaN(v)) return 'NaN';
  if (!Number.isFinite(v)) return v > 0 ? '+∞' : '−∞';
  if (v === 0) return '0';
  const a = Math.abs(v);
  const s = a >= 1e5 || a < 1e-4 ? a.toExponential(4).replace('e', ' × 10^') : a.toPrecision(7);
  return (v < 0 ? '−' : '') + s;
};
const fmtBytes = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(b >= 1e10 ? 0 : 1)} GB` : `${(b / 1e6).toFixed(0)} MB`);

function encode(x, fmt) {
  return fmt === 'INT8' || fmt === 'INT4' ? quantize(x, fmt) : floatBits(x, fmt);
}

export default {
  key: 'bits',
  name: '比特',
  en: 'BITS · SEAFLOOR',
  scale: '16 个比特',

  mount(el, app) {
    const st = app.state;
    const original = st.weight?.value ?? 0.0734;
    let fmt = 'BF16';
    let bits = encode(original, fmt);
    let flips = 0;

    el.innerHTML = `
      <div class="lv-head rv">
        <div>
          <div class="eyebrow">Depth −07 · Bits · 海底</div>
          <h2>一个数字，一串比特</h2>
          <p>你抵达了黑箱的最底层。刚才那个权重 <b class="mono">${original >= 0 ? '+' : '−'}${Math.abs(original).toFixed(3)}</b>，在显存里就是下面这串 0 和 1。再往下，就只剩晶体管里的电荷了。</p>
        </div>
        <div class="bt-fmts" role="tablist">${ORDER.map((f) => `<button type="button" class="chip bt-fmt ${f === fmt ? 'active' : ''}" data-f="${f}" role="tab">${f}</button>`).join('')}</div>
      </div>

      <section class="panel bt-main rv">
        <div class="bt-top">
          <div>
            <div class="eyebrow bt-fname"></div>
            <div class="bt-value mono">0</div>
            <div class="bt-diff hint-line"></div>
          </div>
          <button type="button" class="btn small bt-reset">恢复原值</button>
        </div>
        <div class="bt-bits" aria-label="比特，点击可以翻转"></div>
        <div class="bt-groups"></div>
        <div class="bt-formula mono"></div>
      </section>

      <div class="bt-grid">
        <section class="panel bt-mem rv">
          <div class="panel-h"><h3>用多少比特，决定模型有多大</h3><span class="eyebrow">参数 × 每个参数的字节数</span></div>
          <div class="bt-table"></div>
        </section>
        <section class="panel bt-scale rv">
          <div class="panel-h"><h3>有多少个这样的数？</h3></div>
          <p>你手里这一个，只是模型里的沧海一粟。</p>
          <button type="button" class="btn primary bt-zoom">从这一个数，回望整个模型</button>
        </section>
      </div>
      <div class="bt-montage" hidden>
        <canvas></canvas>
        <div class="bt-m-label"><b class="mono">1</b><span>一个参数</span></div>
        <div class="bt-m-end" hidden>
          <p class="serif">如果每秒数一个，要数 <b>2,220 年</b>。</p>
          <p class="hint-line">而模型每生成一个词元，都要把这些参数全部用上一遍。</p>
          <div class="bt-m-actions"><button type="button" class="btn ghost bt-m-close">留在海底</button><button type="button" class="btn primary bt-m-up">上浮 · 看它开口 <b>↑</b></button></div>
        </div>
      </div>`;

    const render = () => {
      const F = FORMATS[fmt];
      const isInt = fmt === 'INT8' || fmt === 'INT4';
      const dec = bitsToValue(bits, fmt);
      const v = dec.value;
      $('.bt-fname', el).textContent = `${fmt} · ${F.name} · ${F.note}`;
      const valEl = $('.bt-value', el);
      valEl.textContent = fmtVal(v);
      const changed = flips > 0;
      valEl.classList.toggle('changed', changed);
      let diff = '';
      if (!Number.isFinite(v)) diff = '指数位全是 1：它不再是一个正常的数。模型里只要混进一个这样的值，结果就会全部崩坏。';
      else if (changed) {
        const ratio = Math.abs(v / original);
        diff = ratio > 8 || ratio < 1 / 8 ? `和原值相差 <b>${ratio >= 1 ? fmtNum(ratio) : `1/${fmtNum(1 / ratio)}`} 倍</b>，只因为翻转了一个比特。` : `原值 ${fmtVal(original)}，现在 ${fmtVal(v)}。`;
      } else {
        const err = Math.abs(v - original) / Math.abs(original || 1);
        diff = `原值 ${fmtVal(original)} · 用 ${fmt} 存储后的误差 <b>${(err * 100).toFixed(err < 0.001 ? 4 : 2)}%</b>`;
      }
      $('.bt-diff', el).innerHTML = diff;

      const role = (i) => (isInt ? (i === 0 ? 'sign' : 'man') : i === 0 ? 'sign' : i <= F.exp ? 'exp' : 'man');
      $('.bt-bits', el).className = `bt-bits n${bits.length}`;
      $('.bt-bits', el).innerHTML = bits.map((b, i) => `<button type="button" class="bt-bit ${role(i)} ${b ? 'one' : ''}" data-i="${i}" aria-label="第 ${i} 位：${b}">${b}</button>`).join('');
      $('.bt-groups', el).innerHTML = isInt
        ? `<span class="g sign" style="flex:1">符号（补码）</span><span class="g man" style="flex:${bits.length - 1}">整数部分 · ${bits.length - 1} 位</span>`
        : `<span class="g sign" style="flex:1">符号</span><span class="g exp" style="flex:${F.exp}">指数 · ${F.exp} 位</span><span class="g man" style="flex:${F.man}">尾数 · ${F.man} 位</span>`;

      let formula;
      if (isInt) {
        formula = `值 = q / ${dec.qmax} × ${QUANT_SCALE}（缩放系数）= <b>${dec.q}</b> / ${dec.qmax} × ${QUANT_SCALE} = <b class="acc">${fmtVal(v)}</b><br><span class="dimmed">量化：用一个小整数 × 一个整行共享的缩放系数来近似。${fmt} 只有 ${2 ** bits.length} 种可能的取值。</span>`;
      } else if (dec.e === (1 << F.exp) - 1) {
        formula = `指数 = ${dec.e}（全为 1）→ ${dec.m === 0 ? '无穷大' : 'NaN（不是一个数）'}`;
      } else if (dec.e === 0) {
        formula = `(−1)<sup>${dec.s}</sup> × 2<sup>1 − ${dec.bias}</sup> × (${dec.m} / 2<sup>${dec.man}</sup>) = <b class="acc">${fmtVal(v)}</b> <span class="dimmed">（指数为 0：非规格化数）</span>`;
      } else {
        formula = `(−1)<sup class="s">${dec.s}</sup> × 2<sup class="e">${dec.e} − ${dec.bias}</sup> × (1 + <span class="m">${dec.m}</span> / 2<sup>${dec.man}</sup>) = ${dec.s ? '−' : ''}${Math.pow(2, dec.e - dec.bias)} × ${(1 + dec.m / Math.pow(2, dec.man)).toFixed(5)} = <b class="acc">${fmtVal(v)}</b>`;
      }
      $('.bt-formula', el).innerHTML = formula;

      $('.bt-table', el).innerHTML = `
        <div class="bt-trow head"><span></span>${ORDER.map((f) => `<span class="${f === fmt ? 'on' : ''}">${f}</span>`).join('')}</div>
        ${MODELS.map((m) => `
          <div class="bt-trow"><span>${m.name}</span>${ORDER.map((f) => {
            const bytes = m.n * BYTES[f];
            return `<span class="${f === fmt ? 'on' : ''}"><i style="width:${(BYTES[f] / 4) * 100}%"></i>${fmtBytes(bytes)}</span>`;
          }).join('')}</div>`).join('')}
        <p class="hint-line">7B 模型用 INT4 只要约 3.5 GB，一台普通笔记本就装得下；用 FP32 则要 28 GB。</p>`;
    };

    $('.bt-bits', el).addEventListener('click', (e) => {
      const b = e.target.closest('.bt-bit');
      if (!b) return;
      const i = Number(b.dataset.i);
      bits[i] ^= 1;
      flips++;
      app.sfx.bit();
      render();
      const nb = $(`.bt-bit[data-i="${i}"]`, el);
      nb.classList.add('flip');
      app.discover('float');
    });
    $('.bt-reset', el).addEventListener('click', () => { bits = encode(original, fmt); flips = 0; render(); app.sfx.click(); });
    $$('.bt-fmt', el).forEach((btn) => btn.addEventListener('click', () => {
      fmt = btn.dataset.f;
      $$('.bt-fmt', el).forEach((b2) => b2.classList.toggle('active', b2 === btn));
      bits = encode(original, fmt);
      flips = 0;
      render();
      app.sfx.click();
      if (fmt === 'INT4') {
        app.discover('quant');
        app.say('INT4 只有 16 种取值，这个权重被“吸”到了最近的一格上。精度损失看起来不小，但几十亿个误差会互相抵消，<b>模型体积却只剩 FP32 的八分之一</b>。');
      }
    }));

    /* ---------- 蒙太奇：从一个数拉远到整个模型 ---------- */
    const mont = $('.bt-montage', el);
    const mc = $('canvas', mont);
    let raf = 0;
    let alive = true;
    const drawGrid = (R, prevR, scale, stageIdx) => {
      const { g, w, h } = fitCanvas(mc);
      g.clearRect(0, 0, w, h);
      const size = Math.min(w, h) * 0.78;
      const cell = (size / R) * scale;
      const cx = w / 2, cy = h / 2;
      const mid = Math.floor(R / 2);
      const r = rngFor('mont', stageIdx);
      for (let i = 0; i < R; i++) {
        for (let j = 0; j < R; j++) {
          const x = cx + (j - mid - 0.5) * cell, y = cy + (i - mid - 0.5) * cell;
          const v = r() * 2 - 1;
          if (x + cell < 0 || y + cell < 0 || x > w || y > h) continue;
          if (i === mid && j === mid) {
            g.strokeStyle = '#ffb65c';
            g.lineWidth = Math.max(1, Math.min(3, cell / 20));
            g.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
            if (prevR && cell > 40) {
              const sub = cell / prevR;
              const r2 = rngFor('mont', stageIdx - 1);
              for (let a = 0; a < prevR; a++) for (let b = 0; b < prevR; b++) {
                const v2 = r2() * 2 - 1;
                g.fillStyle = v2 > 0 ? `rgba(255,182,92,${0.25 + v2 * 0.6})` : `rgba(92,142,255,${0.25 - v2 * 0.6})`;
                g.beginPath(); g.arc(x + (b + 0.5) * sub, y + (a + 0.5) * sub, Math.max(0.6, sub * 0.28), 0, Math.PI * 2); g.fill();
              }
            } else {
              g.fillStyle = '#ffe2b8';
              g.beginPath(); g.arc(x + cell / 2, y + cell / 2, Math.max(1.2, cell * 0.3), 0, Math.PI * 2); g.fill();
            }
            continue;
          }
          g.fillStyle = v > 0 ? `rgba(255,182,92,${0.2 + v * 0.55})` : `rgba(92,142,255,${0.2 - v * 0.55})`;
          g.beginPath(); g.arc(x + cell / 2, y + cell / 2, Math.max(0.6, cell * 0.28), 0, Math.PI * 2); g.fill();
        }
      }
    };
    const animateScale = (R, prevR, idx, dur) => new Promise((res) => {
      const t0 = performance.now();
      const from = R * 0.9;
      const step = (now) => {
        if (!alive) return res();
        const k = Math.min(1, (now - t0) / dur);
        const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
        drawGrid(R, prevR, from * Math.pow(1 / from, e), idx);
        if (k < 1) raf = requestAnimationFrame(step); else res();
      };
      raf = requestAnimationFrame(step);
    });
    $('.bt-zoom', el).addEventListener('click', async () => {
      mont.hidden = false;
      el.scrollTop = 0;
      el.style.overflow = 'hidden';
      $('.bt-m-end', mont).hidden = true;
      const lbl = $('.bt-m-label', mont);
      lbl.hidden = false;
      $('b', lbl).textContent = '1';
      $('span', lbl).textContent = '一个参数：你刚才翻转的那个数';
      app.sfx.rise();
      requestAnimationFrame(() => mont.classList.add('on'));
      await sleep(reducedMotion ? 0 : 1100);
      for (let s = 0; s < MONTAGE.length && alive; s++) {
        const m = MONTAGE[s];
        const prev = s === 0 ? 0 : MONTAGE[s - 1].R;
        $('span', lbl).textContent = m.sub;
        countUp($('b', lbl), m.n, { dur: reducedMotion ? 1 : 1300, from: s === 0 ? 1 : MONTAGE[s - 1].n });
        app.sfx.tick();
        await animateScale(m.R, prev, s, reducedMotion ? 1 : 1500);
        await sleep(reducedMotion ? 0 : 650);
      }
      if (!alive) return;
      $('.bt-m-end', mont).hidden = false;
      app.sfx.land();
      app.discover('scale');
    });
    $('.bt-m-close', mont).addEventListener('click', () => { mont.classList.remove('on'); el.style.overflow = ''; setTimeout(() => { mont.hidden = true; }, 400); });
    $('.bt-m-up', mont).addEventListener('click', (e) => app.go(8, e.currentTarget));

    render();
    app.say('这里是<b>海底</b>：一个参数，就是一串比特。<em>点击任意一个比特</em>把它翻过来，看看数值怎么变；<em>切换存储格式</em>，看看精度和体积怎么取舍。');
    app.setNext('上浮 · 看它说出下一个词 <b>↑</b>', () => app.go(8));

    return { destroy() { alive = false; cancelAnimationFrame(raf); } };
  },
};
