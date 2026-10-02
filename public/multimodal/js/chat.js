// 聊天面板：先选一张图，再用点选输入法拼问题。候选就是这张图预设问题的真实分词（Qwen 的词元），
// 按前缀一块块弹进输入框，所以只能拼出模型真的回答过的问题。
// 回答完以后，把鼠标放在回复里的某个字上，就能在你发的那张图上看到：生成这个字时模型在看哪里（真实注意力）。
import { $, $$, esc, tokHTML, tokInner, sleep } from '../../js/ui.js';
import { drawHeat } from './paint.js';
import { isEn, L } from '../../js/i18n.js';

export class Chat {
  constructor(manifest, { onSend, onPeek, onPlus, heatFor, onPickImage }) {
    this.M = manifest;
    this.onSend = onSend;
    this.onPickImage = onPickImage;
    this.onPeek = onPeek;
    this.onPlus = onPlus;
    this.heatFor = heatFor;
    this.log = $('#chatLog');
    this.inputEl = $('#cmpInput');
    this.candsEl = $('#cands');
    this.picsEl = $('#pics');
    this.sendBtn = $('#btnSend');
    this.plusBtn = $('#btnPlus');
    this.attachBtn = $('#btnAttach');
    this.msgs = [];
    this.busy = false;
    this.showList = false;
    this.img = null;
    this.chosen = [];
    this.trie = null;
    this.node = null;
    this.renderIntro();
    this.renderPics();
    this.renderInput();
    this.renderCands();
    $('#btnBack').addEventListener('click', () => this.pop());
    this.sendBtn.addEventListener('click', () => this.send());
    this.plusBtn.addEventListener('click', () => this.onPlus?.());
    this.attachBtn.addEventListener('click', () => { this.picsEl.classList.toggle('open'); });
    addEventListener('keydown', (e) => {
      if (e.target.closest('input, textarea')) return;
      if (e.key === 'Backspace' && document.body.classList.contains('mode-chat')) { e.preventDefault(); this.pop(); }
      if (e.key === 'Enter' && !this.sendBtn.disabled) { e.preventDefault(); this.send(); }
    });
  }

  renderIntro() {
    const m = this.M.model, v = m.vision, t = m.text;
    const el = document.createElement('div');
    el.className = 'msg intro-card';
    el.innerHTML = isEn ? `
      <h1>How it sees an image</h1>
      This is <b>Qwen3-VL-2B</b>, an open vision-language model from Alibaba. Its “eyes” are a 24-layer vision encoder; its “mouth” is a 28-layer language model from the same family as Qwen3. Every answer and every number below comes from one real run of this model (recorded offline, so nothing lags and nothing costs money).<br>
      First <b>pick an image</b>, then build a question from the <b>tokens</b> below. To see how it looks at the image and comes up with each word, press the <span class="key">＋</span> next to a message: each press goes one level deeper, stepping through it like a debugger.
      <div class="spec"><span><b>${v.patch}×${v.patch}</b>-pixel patches</span><span>vision encoder <b>${v.depth}</b> layers · <b>${v.hidden}</b> dims</span><span><b>${v.merge}×${v.merge}</b> merged into one token</span><span>language model <b>${t.layers}</b> layers · <b>${t.hidden}</b> dims</span><span>vocabulary <b>${t.vocab.toLocaleString('en-US')}</b></span><span><b>${(m.params.total / 1e9).toFixed(2)}</b>B parameters</span></div>` : `
      <h1>它怎么看图</h1>
      这是阿里开源的视觉语言模型 <b>Qwen3-VL-2B</b>。它的“眼睛”是一个 24 层的视觉编码器，“嘴”是和 Qwen3 同一家族的 28 层语言模型。下面每个回答、每个数字，都来自这个真实模型的一次运行（离线录制，所以不会卡，也不花钱）。<br>
      先<b>选一张图</b>，再用下方的<b>词元</b>拼出问题。想看它是怎么看图、怎么想出每一个字的，就点消息旁的 <span class="key">＋</span>：每按一次，往里钻一层，像调试程序一样单步执行。
      <div class="spec"><span>图块 <b>${v.patch}×${v.patch}</b> 像素</span><span>视觉编码器 <b>${v.depth}</b> 层 · <b>${v.hidden}</b> 维</span><span><b>${v.merge}×${v.merge}</b> 合并成一个词元</span><span>语言模型 <b>${t.layers}</b> 层 · <b>${t.hidden}</b> 维</span><span>词表 <b>${t.vocab.toLocaleString('zh-CN')}</b></span><span><b>${(m.params.total / 1e8).toFixed(1)}</b> 亿参数</span></div>`;
    this.log.append(el);
  }

  /* ---------------- 选图 ---------------- */

  renderPics() {
    this.picsEl.innerHTML = `<span class="lab">${L('先选一张图', 'Pick an image')}</span><div class="pic-row">${this.M.images.map((im, i) => `
      <button type="button" class="pic ${this.img === im ? 'on' : ''}" data-i="${i}" style="animation-delay:${i * 40}ms" title="${esc(im.title)}">
        <img src="data/${im.file}" alt="${esc(im.title)}" loading="lazy" width="${im.size[0]}" height="${im.size[1]}">
        <span>${esc(im.title)}</span>
      </button>`).join('')}</div>`;
    this.picsEl.classList.toggle('open', !this.img);
    $$('.pic', this.picsEl).forEach((b) => b.addEventListener('click', () => this.pickImage(this.M.images[Number(b.dataset.i)])));
  }

  pickImage(im, { keep = false } = {}) {
    if (this.img !== im) {
      this.img = im;
      this.buildTrie();
      if (!keep) { this.chosen = []; this.node = this.trie; }
      this.onPickImage?.(im);
    }
    this.attachBtn.innerHTML = `<img src="data/${im.file}" alt="">`;
    this.attachBtn.classList.add('has');
    this.attachBtn.title = L(`已选：${im.title}（点击换一张图）`, `Selected: ${im.title} (click to pick another image)`);
    this.renderPics();
    this.picsEl.classList.remove('open');
    this.renderInput();
    this.renderCands();
    window.__sfx?.click();
  }

  buildTrie() {
    this.trie = { kids: new Map(), end: null };
    for (const q of this.img.questions) {
      let n = this.trie;
      for (const c of q.chips) {
        if (!n.kids.has(c.id)) n.kids.set(c.id, { chip: c, kids: new Map(), end: null });
        n = n.kids.get(c.id);
      }
      n.end = q;
    }
  }

  /* ---------------- 输入法 ---------------- */

  renderInput() {
    this.inputEl.innerHTML = this.chosen.length
      ? this.chosen.map((c) => tokHTML(c.s, 'r-user')).join('') + '<span class="cur"></span>'
      : `<span class="ph">${this.img ? L('点下面的词元，拼出你的问题', 'Tap the tokens below to build your question') : L('先在下面选一张图', 'First pick an image below')}</span>`;
    const ready = !!this.node?.end && !this.busy;
    this.sendBtn.disabled = !ready;
    this.sendBtn.classList.toggle('glow', ready);
  }

  renderCands() {
    const total = this.M.images.reduce((a, im) => a + im.questions.length, 0);
    let html = '';
    if (!this.img) html += `<span class="lab">${L('还没有选图', 'No image yet')}</span>`;
    else {
      const kids = [...this.node.kids.values()];
      html += `<span class="lab">${this.chosen.length ? L('接下来', 'Next') : L('候选词元', 'Candidate tokens')}</span>`;
      html += kids.map((k, i) => `<button type="button" class="cand" data-id="${k.chip.id}" style="animation-delay:${i * 30}ms">${tokInner(k.chip.s)}<i>${k.chip.id}</i></button>`).join('');
      if (this.node.end && !kids.length) html += `<span class="cand done" aria-hidden="true">${L('✓ 拼好了，按 ↑ 发送', '✓ Done — press ↑ to send')}</span>`;
      else if (this.node.end) html += `<span class="lab" style="color:var(--acc)">${L('✓ 已是完整问题', '✓ Already a complete question')}</span>`;
    }
    html += `<button type="button" class="cand-more">${this.showList ? L('收起问题库', 'Hide question list') : L(`问题库（${total}）`, `Question list (${total})`)}</button>`;
    if (this.showList) {
      html += `<div class="qlist">${this.M.images.map((im, ii) => im.questions.map((q, qi) => `<button type="button" data-im="${ii}" data-q="${qi}"><img src="data/${im.file}" alt="">${esc(q.text)}</button>`).join('')).join('')}</div>`;
    }
    this.candsEl.innerHTML = html;
    $$('.cand[data-id]', this.candsEl).forEach((b) => b.addEventListener('click', () => this.push(Number(b.dataset.id))));
    $('.cand-more', this.candsEl).addEventListener('click', () => { this.showList = !this.showList; this.renderCands(); });
    $$('.qlist button', this.candsEl).forEach((b) => b.addEventListener('click', () => {
      const im = this.M.images[Number(b.dataset.im)];
      this.typeQuestion(im, im.questions[Number(b.dataset.q)]);
    }));
  }

  push(id) {
    const k = this.node?.kids.get(id);
    if (!k) return;
    this.chosen.push(k.chip);
    this.node = k;
    this.renderInput();
    this.renderCands();
    window.__sfx?.tick();
  }

  pop() {
    if (!this.chosen.length) return;
    this.chosen.pop();
    this.node = this.trie;
    for (const c of this.chosen) this.node = this.node.kids.get(c.id);
    this.renderInput();
    this.renderCands();
  }

  async typeQuestion(im, q) {
    this.showList = false;
    this.pickImage(im);
    this.chosen = [];
    this.node = this.trie;
    this.renderInput();
    this.renderCands();
    for (const c of q.chips) {
      this.push(c.id);
      await sleep(95);
    }
  }

  send() {
    if (!this.node?.end || this.busy) return;
    const q = this.node.end;
    this.chosen = [];
    this.node = this.trie;
    this.renderInput();
    this.renderCands();
    this.onSend?.(q, this.img);
  }

  setBusy(b) { this.busy = b; this.renderInput(); }

  /* ---------------- 消息 ---------------- */

  addUser(q, im) {
    const el = document.createElement('div');
    el.className = 'msg user';
    el.innerHTML = `
      <div class="ub-img" style="aspect-ratio:${im.size[0]} / ${im.size[1]}">
        <img src="data/${im.file}" alt="${esc(im.title)}" width="${im.size[0]}" height="${im.size[1]}">
        <canvas class="heat" width="${im.size[0]}" height="${im.size[1]}"></canvas>
        <span class="heat-cap"></span>
      </div>
      ${im.source ? `<a class="ub-src" href="${esc(im.source.url)}" target="_blank" rel="noopener">${esc(im.source.name)} · ${esc(im.source.author)} · ${esc(im.source.license)}</a>` : ''}
      <div class="bubble">${q.chips.map((c) => tokHTML(c.s, 'r-user')).join('')}</div>`;
    this.log.append(el);
    this.scroll();
    return el;
  }

  addBot(q, userEl) {
    const el = document.createElement('div');
    el.className = 'msg bot';
    el.innerHTML = `
      <span class="avatar eye busy"><i></i></span>
      <div class="bot-body">
        <div class="bot-text"><span class="typing"><i></i><i></i><i></i></span></div>
        <div class="bot-meta" hidden>
          <button type="button" class="peek" title="${L('单步进入：看它怎么看图、怎么想出这句话', 'Step into it: see how it looks at the image and comes up with this reply')}"><b>＋</b>${L('揭开这条回复', 'Open up this reply')}</button>
          <span class="stat">${L(`${q.replyTokens.length} 个词元`, `${q.replyTokens.length} token${q.replyTokens.length === 1 ? "" : "s"}`)}</span>
          <span class="hover-tip">${L('把鼠标放在字上：看它生成这个字时在看图的哪里', 'Hover over a word: see where it was looking in the image when it wrote it')}</span>
        </div>
      </div>`;
    this.log.append(el);
    const text = $('.bot-text', el);
    const msg = {
      q, el, text, userEl, shown: -1, done: false,
      render: (n, { cur = false, fresh = false } = {}) => {
        if (msg.shown === -1) text.innerHTML = q.replyTokens.map((t, i) => `<span class="rt" data-i="${i}">${esc(t.s)}</span>`).join('') + '<span class="caret"></span>';
        const spans = text.children;
        for (let i = 0; i < q.replyTokens.length; i++) {
          const sp = spans[i];
          sp.hidden = i >= n;
          sp.classList.toggle('cur', cur && i === n - 1);
          if (fresh && i === n - 1) { sp.classList.remove('fresh'); void sp.offsetWidth; sp.classList.add('fresh'); }
        }
        msg.shown = n;
        const caret = text.lastElementChild;
        caret.hidden = n >= q.replyTokens.length && msg.done;
        this.scroll(true);
      },
      finish: () => {
        msg.done = true;
        $('.avatar', el).classList.remove('busy');
        $('.bot-meta', el).hidden = false;
        msg.render(q.replyTokens.length);
        el.classList.add('hoverable');
      },
    };
    // 悬停 / 点按回复里的字：在用户发的那张图上画出真实的注意力热力图
    const show = (i) => this.showHeat(msg, i);
    text.addEventListener('pointerover', (e) => { const sp = e.target.closest('.rt'); if (sp && msg.done) show(Number(sp.dataset.i)); });
    text.addEventListener('pointerleave', () => { if (!msg.pinned) this.showHeat(msg, null); });
    text.addEventListener('click', (e) => {
      const sp = e.target.closest('.rt');
      if (!sp || !msg.done) return;
      const i = Number(sp.dataset.i);
      msg.pinned = msg.pinned === i ? null : i;
      show(msg.pinned);
    });
    $('.peek', el).addEventListener('click', () => this.onPeek?.(msg));
    this.msgs.push(msg);
    this.scroll();
    return msg;
  }

  async showHeat(msg, i) {
    const box = $('.ub-img', msg.userEl);
    const cv = $('.heat', box), cap = $('.heat-cap', box);
    $$('.rt', msg.text).forEach((s) => s.classList.toggle('lit', i != null && Number(s.dataset.i) === i));
    if (i == null) { box.classList.remove('on'); return; }
    msg.want = i;
    const h = await this.heatFor?.(msg.q, i);
    if (!h || msg.want !== i) return;
    const g = cv.getContext('2d');
    drawHeat(g, h.img, cv.width, cv.height, h.values, h.rows, h.cols, { dim: 0.5 });
    cap.innerHTML = isEn ? `Where it looked while writing “${esc(msg.q.replyTokens[i].s.trim() || msg.q.replyTokens[i].s)}” <small>${esc(h.label)}</small>` : `生成「${esc(msg.q.replyTokens[i].s)}」时在看哪 <small>${esc(h.label)}</small>`;
    box.classList.add('on');
  }

  markInspected(msg) {
    this.msgs.forEach((m) => m.el.classList.toggle('inspected', m === msg));
    if (msg) { $('.bot-meta', msg.el).hidden = false; $('.avatar', msg.el).classList.add('busy'); }
  }

  scroll(soft = false) {
    const nearBottom = this.log.scrollHeight - this.log.scrollTop - this.log.clientHeight < 160;
    if (!soft || nearBottom) this.log.scrollTop = this.log.scrollHeight;
  }
}
