import { PRESETS } from '../model.js';
import { $, $$, esc, countUp, sleep, reducedMotion } from '../ui.js';

const ORBIT = ['token', 'embedding', 'attention', 'softmax', 'GELU', 'residual', 'logits', 'Q·K', 'MLP', 'bf16', 'LayerNorm', 'sampling'];
const FACES = ['front', 'back', 'left', 'right', 'top', 'bottom'];

function faceNumbers(seed) {
  let s = seed * 9301 + 49297;
  const rnd = () => ((s = (s * 9301 + 49297) % 233280) / 233280);
  return Array.from({ length: 26 }, () => Array.from({ length: 3 }, () => {
    const v = (rnd() - 0.5) * 0.4;
    return (v < 0 ? '−' : ' ') + Math.abs(v).toFixed(4);
  }).join(' ')).join('\n');
}

export default {
  key: 'surface',
  name: '地表',
  en: 'SURFACE',
  scale: '一句话',

  mount(el, app) {
    const resume = app.state.max > 0 ? `<button type="button" class="linkish sf-resume">或者，回到上次抵达的 深度 −${String(app.state.max).padStart(2, '0')} · ${app.levels[app.state.max].name}</button>` : '';
    el.innerHTML = `
      <div class="sf-wrap">
        <div class="sf-eyebrow eyebrow rv">Depth 00 · 一次可交互的大模型下潜</div>
        <div class="sf-cube-zone rv" aria-hidden="true">
          <div class="sf-tilt">
            <div class="sf-orbit">${ORBIT.map((w, i) => `<span style="--i:${i}">${w}</span>`).join('')}</div>
            <div class="cube">
              <div class="cube-core"></div>
              ${FACES.map((f, i) => `<div class="face ${f}"><pre>${faceNumbers(i + 1)}</pre></div>`).join('')}
            </div>
          </div>
          <div class="cube-shadow"></div>
        </div>
        <h1 class="sf-title rv"><span>潜</span><span>入</span><span>黑</span><span>箱</span></h1>
        <p class="sf-sub rv">大模型常被称作“黑箱”。这一次，从你写下的一句话出发，<br class="br-wide">一层层往下潜，直到看见组成它的最小单位：<em>一个比特</em>。</p>
        <form class="sf-form rv" autocomplete="off">
          <label class="sf-input">
            <span class="sf-k" aria-hidden="true">›</span>
            <input id="sfInput" maxlength="60" spellcheck="false" placeholder="对黑箱说一句话…" aria-label="输入一句话">
            <button class="btn primary sf-go" type="submit">下潜 <b>↓</b></button>
          </label>
          <div class="sf-presets">
            <span class="eyebrow">试试</span>
            ${PRESETS.map((p, i) => `<button type="button" class="chip" data-i="${i}">${esc(p.label)}</button>`).join('')}
          </div>
          ${resume}
        </form>
        <div class="sf-stats rv">
          <div><b data-n="12">0</b><span>层</span></div>
          <div><b data-n="144">0</b><span>个注意力头</span></div>
          <div><b data-n="36864">0</b><span>个 MLP 神经元</span></div>
          <div><b data-n="124439808">0</b><span>个参数</span></div>
        </div>
        <p class="sf-note rv">结构取自 GPT-2 Small · 画面中的数值为示意 · 键盘 ↓ 下潜 / ↑ 上浮</p>
      </div>`;

    const input = $('#sfInput', el);
    const form = $('.sf-form', el);
    const cube = $('.cube', el);
    const zone = $('.sf-cube-zone', el);
    const tilt = $('.sf-tilt', el);
    input.value = app.state.prompt;
    let launching = false;
    let alive = true;
    const timers = [];

    // 统计数字滚动
    timers.push(setTimeout(() => $$('.sf-stats b', el).forEach((b, i) => {
      setTimeout(() => countUp(b, Number(b.dataset.n), { dur: 1400 + i * 250 }), i * 120);
    }), 900));

    // 鼠标视差
    const onMove = (e) => {
      const r = zone.getBoundingClientRect();
      const dx = (e.clientX - (r.left + r.width / 2)) / innerWidth;
      const dy = (e.clientY - (r.top + r.height / 2)) / innerHeight;
      tilt.style.setProperty('--rx', `${-dy * 24}deg`);
      tilt.style.setProperty('--ry', `${dx * 30}deg`);
    };
    addEventListener('pointermove', onMove);

    // 测量输入框里某个位置的横坐标
    const mirror = document.createElement('span');
    mirror.className = 'sf-mirror';
    el.append(mirror);
    const caretX = (upto) => {
      mirror.textContent = input.value.slice(0, upto);
      const ir = input.getBoundingClientRect();
      return Math.min(ir.right - 8, ir.left + mirror.offsetWidth - input.scrollLeft);
    };
    const cubeCenter = () => {
      const r = cube.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    };

    const fly = (ch, x, y, delay = 0, dur = 650) => {
      if (reducedMotion || !ch.trim()) return;
      const s = document.createElement('span');
      s.className = 'sf-fly';
      s.textContent = ch;
      s.style.left = `${x}px`;
      s.style.top = `${y}px`;
      document.body.append(s);
      const c = cubeCenter();
      const dx = c.x - x, dy = c.y - y;
      const a = s.animate(
        [
          { transform: 'translate(-50%,-50%) scale(1)', opacity: 1 },
          { transform: `translate(calc(-50% + ${dx * 0.35}px), calc(-50% + ${dy * 0.55 - 40}px)) scale(.9)`, opacity: 1, offset: 0.45 },
          { transform: `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px)) scale(.15)`, opacity: 0 },
        ],
        { duration: dur, delay, easing: 'cubic-bezier(.55,0,.7,.4)', fill: 'backwards' },
      );
      a.onfinish = () => { s.remove(); cube.classList.remove('gulp'); void cube.offsetWidth; cube.classList.add('gulp'); };
    };

    input.addEventListener('input', (e) => {
      if (!e.data || launching) return;
      const ir = input.getBoundingClientRect();
      const pos = input.selectionStart ?? input.value.length;
      const chars = Array.from(e.data);
      chars.forEach((ch, i) => fly(ch, caretX(pos - chars.length + i + 1) - 6, ir.top + ir.height / 2, i * 40, 560));
      app.sfx.tick();
    });
    input.addEventListener('focus', () => { zone.classList.add('listening'); input.select(); });
    input.addEventListener('blur', () => zone.classList.remove('listening'));

    $$('.chip', el).forEach((b) => b.addEventListener('click', () => {
      input.value = PRESETS[Number(b.dataset.i)].prompt;
      app.sfx.click();
      input.focus();
      $$('.chip', el).forEach((c) => c.classList.toggle('active', c === b));
    }));

    $('.sf-resume', el)?.addEventListener('click', (e) => app.go(app.state.max, e.currentTarget));

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (launching) return;
      const text = input.value.trim() || PRESETS[0].prompt;
      launching = true;
      input.blur();
      app.setPrompt(text, { base: true });
      app.sfx.click();
      const chars = Array.from(input.value).slice(0, 48);
      const ir = input.getBoundingClientRect();
      chars.forEach((ch, i) => fly(ch, caretX(i + 1) - 6, ir.top + ir.height / 2, i * 28, 720));
      const drain = setInterval(() => { input.value = input.value.slice(1); }, 28);
      timers.push(drain);
      await sleep(reducedMotion ? 0 : Math.min(1500, chars.length * 28 + 650));
      clearInterval(drain);
      if (!alive) return;
      zone.classList.add('charged');
      el.classList.add('launch');
      app.discover('enter');
      await sleep(reducedMotion ? 0 : 520);
      if (alive) app.go(1, cube);
    });

    app.setNext('下潜 <b>↓</b>', () => form.requestSubmit());

    return {
      destroy() {
        alive = false;
        removeEventListener('pointermove', onMove);
        timers.forEach((t) => { clearTimeout(t); clearInterval(t); });
        $$('.sf-fly').forEach((s) => s.remove());
      },
    };
  },
};
