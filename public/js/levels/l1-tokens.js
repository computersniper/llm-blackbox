import { tokenize, CFG } from '../model.js';
import { $, $$, esc, tokHTML, tokInner, hueOf, countUp, fmtNum, reducedMotion } from '../ui.js';

const KIND = { zh: '中文词', en: '英文（子）词', num: '数字', punct: '标点', space: '空白', byte: 'UTF-8 字节', other: '字符' };
const TRY = [
  { label: '一串数字', text: '圆周率是3.14159' },
  { label: 'emoji 🤖', text: '你好🤖，黑箱！' },
  { label: '长单词', text: 'unbelievable tokenization' },
  { label: '中英混排', text: '我在学习 transformer 模型' },
];

function cellHTML(t, i) {
  const cls = `clickable portal ${t.kind === 'byte' ? 'byte' : ''}`;
  return `<div class="tk-cell" style="--i:${i}">
      <span class="tok big ${cls}" style="--h:${hueOf(t.text)}" data-i="${i}" tabindex="0" role="button" aria-label="词元 ${esc(t.text)}，编号 ${t.id}">${tokInner(t.text)}</span>
      <span class="tk-id" data-id="${t.id}">0</span>
    </div>`;
}

function tipHTML(t) {
  const lead = t.text.startsWith(' ') ? '<br>开头带一个空格：在英文里，“ word”和“word”是两个不同的词元。' : '';
  const byte = t.kind === 'byte' ? `<br>字符 <b>${esc(t.of)}</b> 不在词表里，被拆成了 UTF-8 字节。` : '';
  return `<span class="k">词元 #${t.i} · ${KIND[t.kind] ?? ''}</span><b>${esc(t.text.replace('\n', '↵'))}</b> → 编号 <span class="v">${t.id}</span>${lead}${byte}<br><span class="dimmed">点击，下潜到它的向量</span>`;
}

export default {
  key: 'tokens',
  name: '分词',
  en: 'TOKENIZE',
  scale: '词元 · token',

  mount(el, app) {
    const { tokens, prompt } = app.model;
    const chars = Array.from(prompt).length;
    el.innerHTML = `
      <div class="lv-head rv">
        <div>
          <div class="eyebrow">Depth −01 · Tokenize</div>
          <h2>文字被切成词元</h2>
          <p>模型不认识“字”。分词器先把你的话切成一块块<b>词元</b>（token），再把每块换成词表里的一个编号。从这一刻起，模型只和数字打交道。</p>
        </div>
        <div class="tk-stats">
          <div class="stat"><b>${chars}</b><span>字符</span></div>
          <div class="stat"><b class="acc">${tokens.length}</b><span>词元</span></div>
          <div class="stat"><b>${(chars / Math.max(1, tokens.length)).toFixed(2)}</b><span>字符 / 词元</span></div>
          <div class="stat"><b>${fmtNum(CFG.vocab)}</b><span>词表大小</span></div>
        </div>
      </div>

      <section class="panel tk-main rv">
        <div class="tk-scan" aria-hidden="true"></div>
        <div class="tk-label eyebrow">你的输入</div>
        <div class="tk-row joined">${tokens.map(cellHTML).join('')}</div>
        <div class="tk-arrow" aria-hidden="true"><span></span>查词表<span></span></div>
        <div class="tk-label eyebrow">模型真正看到的</div>
        <div class="tk-ids mono" aria-label="词元编号序列">[${tokens.map((t, i) => `<span style="--i:${i}">${t.id}</span>`).join('<i>, </i>')}]</div>
      </section>

      <div class="tk-bottom">
        <section class="panel tk-lab rv">
          <div class="panel-h"><h3>分词实验室</h3><span class="eyebrow">试着骗过分词器</span></div>
          <div class="tk-lab-body">
            <input class="tk-lab-in" maxlength="60" spellcheck="false" placeholder="随便输入点什么，实时看它怎么切…" aria-label="分词实验输入">
            <div class="tk-lab-try">${TRY.map((x, i) => `<button type="button" class="chip" data-i="${i}">${esc(x.label)}</button>`).join('')}</div>
            <div class="tk-lab-out"></div>
            <div class="tk-lab-foot"><span class="tk-lab-count hint-line"></span><button type="button" class="btn small tk-adopt">用这句话下潜 ↓</button></div>
          </div>
        </section>
        <section class="panel tk-facts rv">
          <div class="panel-h"><h3>为什么要切？</h3></div>
          <ul>
            <li><b>按字切</b>，序列太长，模型要处理的步数翻倍。</li>
            <li><b>按词切</b>，词表会大到装不下，新词也没法处理。</li>
            <li><b>BPE</b> 等算法取中间：把训练语料里最常一起出现的片段合并成词元。常见词一块，生僻词拆成几块。</li>
          </ul>
          <p class="hint-line">颜色只是为了区分边界，同一个词元永远是同一种颜色。</p>
        </section>
      </div>`;

    const row = $('.tk-row', el);
    const timers = [];
    const later = (fn, ms) => timers.push(setTimeout(fn, reducedMotion ? 0 : ms));

    // 1. 扫描线 → 2. 裂开 → 3. 编号滚动 → 4. 编号序列
    later(() => $('.tk-main', el).classList.add('scanning'), 700);
    later(() => { row.classList.remove('joined'); app.sfx.tick(); }, 1500);
    later(() => {
      row.classList.add('ided');
      $$('.tk-id', el).forEach((s, i) => setTimeout(() => countUp(s, Number(s.dataset.id), { dur: 700, fmt: (v) => String(Math.round(v)) }), i * 50));
    }, 1900);
    later(() => $('.tk-ids', el).classList.add('on'), 2300);
    later(() => app.discover('tokens'), 3200);

    $$('.tk-row .tok', el).forEach((chip) => {
      const t = tokens[Number(chip.dataset.i)];
      app.tip.bind(chip, () => tipHTML(t));
      const go = (e) => {
        app.state.focus = t.i;
        app.sfx.click();
        app.go(2, e.currentTarget);
      };
      chip.addEventListener('click', go);
      chip.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(e); } });
    });

    // 实验室
    const labIn = $('.tk-lab-in', el);
    const labOut = $('.tk-lab-out', el);
    const labCount = $('.tk-lab-count', el);
    const renderLab = () => {
      const text = labIn.value;
      const toks = tokenize(text);
      labOut.innerHTML = toks.length
        ? toks.map((t) => `<span class="tk-lab-cell">${tokHTML(t, t.kind === 'byte' ? 'byte' : '')}<i>${t.id}</i></span>`).join('')
        : '<span class="hint-line">（空）</span>';
      labCount.textContent = text ? `${Array.from(text).length} 个字符 → ${toks.length} 个词元` : '';
      if (toks.filter((t) => t.kind === 'num').length >= 3) app.discover('digits');
      if (toks.some((t) => t.kind === 'byte')) app.discover('bytes');
      if (toks.some((t) => t.sub)) app.discover('subword');
      if (toks.length > 1 && toks.length !== tokens.length) app.state.labTried = true;
    };
    labIn.addEventListener('input', renderLab);
    $$('.tk-lab-try .chip', el).forEach((b) => b.addEventListener('click', () => {
      labIn.value = TRY[Number(b.dataset.i)].text;
      app.sfx.click();
      renderLab();
    }));
    $('.tk-adopt', el).addEventListener('click', (e) => {
      const v = labIn.value.trim();
      if (!v) { labIn.focus(); return; }
      app.setPrompt(v, { base: true });
      app.state.focus = app.model.tokens.length - 1;
      app.go(2, e.currentTarget);
    });
    labIn.value = '';
    renderLab();

    const bytes = tokens.filter((t) => t.kind === 'byte').length;
    const digits = tokens.filter((t) => t.kind === 'num').length;
    let extra = '';
    if (bytes) extra = ` 注意那 ${bytes} 个 <b>&lt;0x..&gt;</b>：词表外的字符被拆成了字节。`;
    else if (digits >= 3) extra = ' 注意数字是<b>一位一位</b>切开的。';
    app.say(`你的话被切成了 <b>${tokens.length} 个词元</b>，每个都换成了一个编号。${extra} <em>点击任意一个词元</em>，看看编号背后藏着什么。`);
    app.setNext('进入嵌入层 <b>↓</b>', (e) => { app.state.focus = tokens.length - 1; app.go(2, e?.currentTarget); });

    return { destroy() { timers.forEach(clearTimeout); } };
  },
};

