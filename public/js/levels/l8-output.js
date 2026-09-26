import { CFG, withTemperature, sampleFrom } from '../model.js';
import { $, $$, esc, tokHTML, tokPlain, hueOf, fmtPct, fmtNum, svgEl, sleep, reducedMotion } from '../ui.js';

const CX = 150, CY = 150, R = 132;
const pt = (deg, r) => [CX + r * Math.sin((deg * Math.PI) / 180), CY - r * Math.cos((deg * Math.PI) / 180)];

function segPath(a0, a1) {
  if (a1 - a0 >= 359.99) return `M${CX},${CY - R} A${R},${R} 0 1 1 ${CX - 0.01},${CY - R} Z`;
  const [x0, y0] = pt(a0, R), [x1, y1] = pt(a1, R);
  return `M${CX},${CY} L${x0},${y0} A${R},${R} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1},${y1} Z`;
}

const tempWord = (T) => (T < 0.06 ? '贪心：永远选第一名' : T < 0.6 ? '保守' : T < 1.15 ? '平衡' : T < 1.5 ? '大胆' : '开始胡言乱语');

export default {
  key: 'output',
  name: '输出',
  en: 'OUTPUT',
  scale: `${fmtNum(CFG.vocab)} 个候选`,

  mount(el, app) {
    const st = app.state;
    if (st.temp == null) st.temp = 1;
    if (!st.basePrompt) st.basePrompt = st.prompt;
    st.generated = st.generated || 0;
    let rot = 0;
    let spinning = false;
    let auto = false;
    let autoRun = 0;
    let alive = true;
    let tempTouched = false;

    el.innerHTML = `
      <div class="lv-head rv">
        <div>
          <div class="eyebrow">Ascent · Output</div>
          <h2>从数字，回到文字</h2>
          <p>最后一个词元的向量爬完 12 层后，乘上一个 768 × 50257 的矩阵，得到<b>每个候选词元的分数</b>。softmax 把分数变成概率，然后——<b>掷骰子</b>。</p>
        </div>
      </div>

      <section class="panel op-sent rv">
        <div class="op-line"></div>
        <div class="op-pipe" aria-hidden="true">
          <span class="op-step">最后一个向量 · 768</span><i></i>
          <span class="op-step">× 反嵌入矩阵 768 × 50257</span><i></i>
          <span class="op-step">${fmtNum(CFG.vocab)} 个分数</span><i></i>
          <span class="op-step">softmax → 概率</span><i></i>
          <span class="op-step">采样</span>
        </div>
      </section>

      <div class="op-grid">
        <section class="panel op-dist rv">
          <div class="panel-h"><h3>下一个词元的概率</h3><span class="eyebrow op-src"></span></div>
          <div class="op-rows"></div>
          <div class="op-temp">
            <div class="op-temp-h"><span>温度 <b class="mono op-tv">1.00</b></span><span class="op-tw"></span></div>
            <input type="range" class="op-range" min="0" max="2" step="0.01" aria-label="温度">
            <div class="op-temp-q">
              <button type="button" class="chip" data-t="0">贪心 T=0</button>
              <button type="button" class="chip" data-t="0.7">0.7</button>
              <button type="button" class="chip" data-t="1">1.0</button>
              <button type="button" class="chip" data-t="1.6">1.6</button>
            </div>
          </div>
        </section>
        <section class="panel op-wheel rv">
          <div class="panel-h"><h3>掷骰子</h3><span class="eyebrow">扇区大小 = 概率</span></div>
          <div class="op-wheel-wrap">
            <svg class="op-svg" viewBox="0 0 300 300" role="img" aria-label="按概率划分的转盘">
              <g class="op-rot"></g>
              <circle cx="150" cy="150" r="46" class="op-hub"></circle>
              <path d="M150,6 L140,-10 L160,-10 Z" class="op-pointer" transform="translate(0,14)"></path>
            </svg>
            <button type="button" class="op-spin">采样</button>
          </div>
          <div class="op-actions">
            <button type="button" class="btn small op-auto">自动生成 × 6</button>
            <button type="button" class="btn small ghost op-reset">重置句子</button>
          </div>
        </section>
      </div>`;

    const lineEl = $('.op-line', el);
    const rotG = $('.op-rot', el);
    const range = $('.op-range', el);
    range.value = st.temp;

    const setFill = () => range.style.setProperty('--fill', `${(range.value / 2) * 100}%`);

    const renderLine = (fresh = -1) => {
      const toks = app.model.tokens;
      lineEl.innerHTML = `${toks.map((t, i) => tokHTML(t, `op-t ${i === fresh ? 'fresh' : ''}`)).join('')}<span class="op-slot" aria-label="下一个词元">?</span>`;
    };

    let segs = [];
    const renderDist = () => {
      const T = st.temp;
      const d = withTemperature(app.model.dist, T);
      $('.op-tv', el).textContent = T.toFixed(2);
      $('.op-tw', el).textContent = tempWord(T);
      $('.op-src', el).textContent = app.model.dist.source === 'curated' ? '示意分布' : '示意分布 · 启发式';
      const rows = d.cands.map((c) => ({ text: c.text, q: c.q, z: c.logit + 11 }));
      $('.op-rows', el).innerHTML = rows.map((r, k) => `
        <div class="op-row ${k === 0 ? 'top' : ''}" data-k="${k}">
          ${tokHTML(r.text, 'mini')}
          <span class="op-z mono" title="分数（logit）">${r.z.toFixed(2)}</span>
          <span class="op-bar"><i style="width:${(r.q * 100).toFixed(2)}%; --h:${hueOf(r.text)}"></i></span>
          <span class="op-p mono">${fmtPct(r.q)}</span>
        </div>`).join('') + `
        <div class="op-row other" data-k="-1">
          <span class="op-other">其他 ${fmtNum(CFG.vocab - rows.length)} 个</span>
          <span class="op-z mono">…</span>
          <span class="op-bar"><i style="width:${(d.other * 100).toFixed(2)}%"></i></span>
          <span class="op-p mono">${fmtPct(d.other)}</span>
        </div>`;

      // 转盘
      segs = [];
      let a = 0;
      let rest = d.other;
      d.cands.forEach((c, k) => {
        if (c.q < 0.006) { rest += c.q; return; }
        segs.push({ k, text: c.text, a0: a, a1: a + c.q * 360, h: hueOf(c.text) });
        a += c.q * 360;
      });
      if (rest > 0.0005) segs.push({ k: -1, text: '其他', a0: a, a1: 360, other: true });
      else if (segs.length) segs[segs.length - 1].a1 = 360;
      rotG.innerHTML = '';
      segs.forEach((s) => {
        const p = svgEl('path', { d: segPath(s.a0, s.a1), class: `op-seg ${s.other ? 'other' : ''}`, style: s.other ? '' : `--h:${s.h}`, 'data-k': s.k });
        rotG.append(p);
        if (s.a1 - s.a0 > 16) {
          const mid = (s.a0 + s.a1) / 2;
          const [x, y] = pt(mid, R * 0.7);
          const t = svgEl('text', { x, y, class: 'op-seg-t', 'text-anchor': 'middle', 'dominant-baseline': 'middle', transform: `rotate(${mid > 90 && mid < 270 ? mid + 180 : mid} ${x} ${y})` });
          t.textContent = s.other ? '其他' : tokPlain(s.text).slice(0, 6);
          rotG.append(t);
        }
      });
      return d;
    };

    const runPipe = async () => {
      const steps = $$('.op-step', el);
      for (const s of steps) {
        if (!alive) return;
        s.classList.add('run');
        await sleep(reducedMotion ? 0 : auto ? 70 : 140);
      }
      await sleep(reducedMotion ? 0 : 200);
      steps.forEach((s) => s.classList.remove('run'));
    };

    const spin = async () => {
      if (spinning || !alive) return;
      if (Array.from(app.state.prompt).length >= 68) {
        app.say('句子已经够长了。点“重置句子”重新开始，或者带着这句话<b>再潜一次</b>。');
        auto = false;
        return;
      }
      spinning = true;
      el.classList.add('spinning');
      range.disabled = true;
      const d = withTemperature(app.model.dist, st.temp);
      const pick = sampleFrom(d);
      let seg = segs.find((s) => s.k === pick.k) || segs.find((s) => s.other) || segs[0];
      const span = seg.a1 - seg.a0;
      const target = seg.a0 + span / 2 + (Math.random() - 0.5) * span * 0.6;
      const turns = auto ? 2 : 4;
      const delta = (((-target - rot) % 360) + 360) % 360;
      rot += turns * 360 + delta;
      const dur = reducedMotion ? 10 : auto ? 1300 : 2800;
      rotG.style.transition = `transform ${dur}ms cubic-bezier(.12,.75,.18,1)`;
      rotG.style.transform = `rotate(${rot}deg)`;
      // 指针拨过扇区的“咔哒”声
      const ticks = auto ? 8 : 16;
      for (let i = 0; i < ticks; i++) setTimeout(() => alive && app.sfx.tick(), dur * (1 - Math.pow(1 - i / ticks, 2.2)));
      await sleep(dur + 60);
      if (!alive) return;
      $$('.op-seg', el).forEach((p) => p.classList.toggle('hit', Number(p.dataset.k) === seg.k));
      app.sfx.land();

      // 词元飞进句子
      const slot = $('.op-slot', el);
      const hub = $('.op-hub', el).getBoundingClientRect();
      const sr = slot.getBoundingClientRect();
      const fly = document.createElement('span');
      fly.innerHTML = tokHTML(pick.text, 'op-fly');
      const flyEl = fly.firstElementChild;
      flyEl.style.left = `${hub.left + hub.width / 2}px`;
      flyEl.style.top = `${hub.top + hub.height / 2}px`;
      document.body.append(flyEl);
      if (!reducedMotion) {
        await flyEl.animate(
          [
            { transform: 'translate(-50%,-50%) scale(1.6)', opacity: 1 },
            { transform: `translate(calc(-50% + ${sr.left + sr.width / 2 - hub.left - hub.width / 2}px), calc(-50% + ${sr.top + sr.height / 2 - hub.top - hub.height / 2}px)) scale(1)`, opacity: 1 },
          ],
          { duration: auto ? 420 : 700, easing: 'cubic-bezier(.5,0,.2,1)' },
        ).finished.catch(() => {});
      }
      flyEl.remove();
      if (!alive) return;

      const n = app.model.tokens.length;
      app.setPrompt(app.state.prompt + pick.text);
      st.generated++;
      renderLine(n);
      if (st.generated >= 3) app.discover('autoregress');
      if (pick.k === -1 || seg.other) app.say(`掷到了长尾里的「${esc(tokPlain(pick.text))}」！五万个低概率词元加起来也不少，<b>温度越高，越容易掷到这种怪词</b>。`);
      else if (!auto) app.say(`掷出了「${esc(tokPlain(pick.text))}」，接到句子末尾。然后整个模型<b>从分词开始，再完整跑一遍</b>，只为了下一个词元。再掷一次试试？`);
      await runPipe();
      renderDist();
      $$('.op-seg', el).forEach((p) => p.classList.remove('hit'));
      el.classList.remove('spinning');
      range.disabled = false;
      spinning = false;
    };

    range.addEventListener('input', () => {
      if (spinning) { range.value = st.temp; return; } // 转盘转动时不改分布
      st.temp = Number(range.value);
      setFill();
      renderDist();
      if (!tempTouched) { tempTouched = true; }
      app.discover('temperature');
      if (st.temp < 0.06) {
        app.discover('greedy');
        app.say('温度为 0：永远选概率最高的那个。<b>输出完全确定</b>，同一句话每次都得到同样的续写。转盘只剩一个扇区了。');
      } else if (st.temp > 1.45) {
        app.say('温度很高时，第一名的优势被抹平，<b>长尾里五万个怪词</b>一起涨潮。掷一次试试看？');
      }
    });
    $$('.op-temp-q .chip', el).forEach((b) => b.addEventListener('click', () => {
      range.value = b.dataset.t;
      range.dispatchEvent(new Event('input'));
      app.sfx.click();
    }));

    $('.op-spin', el).addEventListener('click', () => { app.sfx.click(); spin(); });
    $('.op-auto', el).addEventListener('click', async (e) => {
      const btn = e.currentTarget;
      if (auto) { auto = false; btn.textContent = '自动生成 × 6'; return; }
      auto = true;
      const run = ++autoRun;
      btn.textContent = '停止';
      for (let i = 0; i < 6 && auto && alive && run === autoRun; i++) {
        while (spinning && alive && run === autoRun) await sleep(80); // 等上一次转完
        if (!auto || run !== autoRun) break;
        await spin();
        await sleep(reducedMotion ? 0 : 150);
      }
      if (run !== autoRun) return;
      auto = false;
      if (alive) btn.textContent = '自动生成 × 6';
      if (alive) app.say(`这就是大模型“说话”的方式：<b>一次一个词元，每次都把整句话从头算一遍</b>。你刚刚看着它生成了 ${st.generated} 个。`);
    });
    $('.op-reset', el).addEventListener('click', () => {
      if (spinning) return;
      app.setPrompt(st.basePrompt);
      renderLine();
      renderDist();
      app.sfx.click();
    });

    setFill();
    renderLine();
    renderDist();

    const top = withTemperature(app.model.dist, 1).cands[0];
    app.say(`模型给「${esc(tokPlain(top.text))}」打了最高分，概率 <b>${fmtPct(top.q)}</b>。但它并不“确定”，只是在掷一个<b>灌了铅的骰子</b>。<em>点“采样”掷一次</em>，或者<em>拖动温度</em>看看骰子怎么变。`);
    app.setNext('带着新句子，再潜一次 <b>↓</b>', (e) => app.go(1, e?.currentTarget));

    return { destroy() { alive = false; auto = false; $$('.op-fly').forEach((f) => f.remove()); } };
  },
};
