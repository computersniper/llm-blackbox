// 聊天面板 + 点选输入法。输入法的候选就是问题的真实分词结果（Qwen3 的词元），
// 按前缀一块块弹进输入框，所以只能拼出模型真的回答过的问题。
import { $, $$, esc, tokHTML, tokInner, sleep } from './ui.js';
import { isEn, L as tr } from './i18n.js';

export class Chat {
  constructor(manifest, { onSend, onPeek, onPlus }) {
    this.M = manifest;
    this.onSend = onSend;
    this.onPeek = onPeek;
    this.onPlus = onPlus;
    this.log = $('#chatLog');
    this.inputEl = $('#cmpInput');
    this.candsEl = $('#cands');
    this.sendBtn = $('#btnSend');
    this.plusBtn = $('#btnPlus');
    this.msgs = [];
    this.busy = false;
    this.showList = false;
    this.buildTrie();
    this.chosen = [];
    this.node = this.trie;
    this.renderIntro();
    this.renderInput();
    this.renderCands();
    $('#btnBack').addEventListener('click', () => this.pop());
    this.sendBtn.addEventListener('click', () => this.send());
    this.plusBtn.addEventListener('click', () => this.onPlus?.());
    addEventListener('keydown', (e) => {
      if (e.target.closest('input, textarea')) return;
      if (e.key === 'Backspace' && document.body.classList.contains('mode-chat')) { e.preventDefault(); this.pop(); }
      if (e.key === 'Enter' && !this.sendBtn.disabled) { e.preventDefault(); this.send(); }
    });
  }

  buildTrie() {
    this.trie = { kids: new Map(), end: null };
    for (const q of this.M.questions) {
      let n = this.trie;
      for (const c of q.chips) {
        if (!n.kids.has(c.id)) n.kids.set(c.id, { chip: c, kids: new Map(), end: null, count: 0 });
        n = n.kids.get(c.id);
        n.count++;
      }
      n.end = q;
    }
  }

  renderIntro() {
    const m = this.M.model;
    const el = document.createElement('div');
    el.className = 'msg intro-card';
    el.innerHTML = isEn ? `
      <h1>Inside the Black Box</h1>
      This is <b>Qwen3-0.6B</b>, an open-source model from Alibaba. Every answer and every number below comes from one real run of this model (recorded offline, so it never lags and costs nothing). It has only 0.6 billion parameters and sometimes says wrong things with a straight face; that is part of the real thing too.<br>
      Build a question out of the <b>tokens</b> below and send it. To see how it comes up with each word, press the <span class="key">＋</span> next to its reply: every press takes you one level deeper into the black box, stepping through it like a program in a debugger.
      <div class="spec"><span><b>${m.layers}</b> layers</span><span>hidden size <b>${m.hidden}</b></span><span><b>${m.heads}</b> Q heads / <b>${m.kvHeads}</b> KV heads</span><span>SwiGLU <b>${m.ffn}</b></span><span>vocab <b>${m.vocab.toLocaleString('en-US')}</b></span><span><b>${(m.params / 1e6).toFixed(0)}M</b> parameters</span></div>` : `
      <h1>揭开黑箱</h1>
      这是阿里开源的 <b>Qwen3-0.6B</b>。下面的每个回答、每个数字，都来自这个真实模型的一次运行（离线录制，所以不会卡，也不花钱）。它只有 6 亿参数，偶尔会一本正经地说错话，这也是真实的一部分。<br>
      用下方的<b>词元</b>拼出一个问题发给它。想看它是怎么想出每一个字的，就点消息旁的 <span class="key">＋</span>：每按一次，往黑箱里钻一层，像调试程序一样单步执行。
      <div class="spec"><span><b>${m.layers}</b> 层</span><span>隐藏维度 <b>${m.hidden}</b></span><span><b>${m.heads}</b> Q 头 / <b>${m.kvHeads}</b> KV 头</span><span>SwiGLU <b>${m.ffn}</b></span><span>词表 <b>${m.vocab.toLocaleString('zh-CN')}</b></span><span><b>${(m.params / 1e8).toFixed(2)}</b> 亿参数</span></div>`;
    this.log.append(el);
  }

  /* ---------------- 输入法 ---------------- */

  renderInput() {
    this.inputEl.innerHTML = this.chosen.length
      ? this.chosen.map((c) => tokHTML(c.s, 'r-user')).join('') + '<span class="cur"></span>'
      : `<span class="ph">${tr('点下面的词元，拼出你的问题', 'Tap the tokens below to build your question')}</span>`;
    const ready = !!this.node.end && !this.busy;
    this.sendBtn.disabled = !ready;
    this.sendBtn.classList.toggle('glow', ready);
  }

  renderCands() {
    const kids = [...this.node.kids.values()];
    const head = this.chosen.length ? tr('接下来', 'Next') : tr('候选词元', 'Tokens');
    let html = `<span class="lab">${head}</span>`;
    html += kids.map((k, i) => `<button type="button" class="cand" data-id="${k.chip.id}" style="animation-delay:${i * 30}ms">${tokInner(k.chip.s)}<i>${k.chip.id}</i></button>`).join('');
    if (this.node.end && !kids.length) html += `<span class="cand done" aria-hidden="true">${tr('✓ 拼好了，按 ↑ 发送', '✓ Done! Press ↑ to send')}</span>`;
    else if (this.node.end) html += `<span class="lab" style="color:var(--acc)">${tr('✓ 已是完整问题', '✓ Complete question')}</span>`;
    html += `<button type="button" class="cand-more">${this.showList ? tr('收起问题库', 'Hide questions') : tr(`问题库（${this.M.questions.length}）`, `All questions (${this.M.questions.length})`)}</button>`;
    if (this.showList) html += `<div class="qlist">${this.M.questions.map((q, i) => `<button type="button" data-q="${i}">${esc(q.text)}</button>`).join('')}</div>`;
    this.candsEl.innerHTML = html;
    $$('.cand[data-id]', this.candsEl).forEach((b) => b.addEventListener('click', () => this.push(Number(b.dataset.id))));
    $('.cand-more', this.candsEl).addEventListener('click', () => { this.showList = !this.showList; this.renderCands(); });
    $$('.qlist button', this.candsEl).forEach((b) => b.addEventListener('click', () => this.typeQuestion(this.M.questions[Number(b.dataset.q)])));
  }

  push(id) {
    const k = this.node.kids.get(id);
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

  async typeQuestion(q) {
    this.showList = false;
    this.chosen = [];
    this.node = this.trie;
    this.renderInput();
    for (const c of q.chips) {
      this.push(c.id);
      await sleep(95);
    }
  }

  send() {
    if (!this.node.end || this.busy) return;
    const q = this.node.end;
    this.chosen = [];
    this.node = this.trie;
    this.renderInput();
    this.renderCands();
    this.onSend?.(q);
  }

  setBusy(b) { this.busy = b; this.renderInput(); }

  /* ---------------- 消息 ---------------- */

  addUser(q) {
    const el = document.createElement('div');
    el.className = 'msg user';
    el.innerHTML = `<div class="bubble">${q.chips.map((c) => tokHTML(c.s, 'r-user')).join('')}</div>`;
    this.log.append(el);
    this.scroll();
  }

  addBot(q) {
    const el = document.createElement('div');
    el.className = 'msg bot';
    el.innerHTML = `
      <span class="avatar busy"><i></i></span>
      <div class="bot-body">
        <div class="bot-text"><span class="typing"><i></i><i></i><i></i></span></div>
        <div class="bot-meta" hidden>
          <button type="button" class="peek" title="${tr('单步进入：看它怎么想出这句话', 'Step into it: see how it came up with this reply')}"><b>＋</b>${tr('揭开这条回复', 'Open up this reply')}</button>
          <span class="stat">${tr(`${q.replyTokens.length} 个词元`, `${q.replyTokens.length} tokens`)}</span>
        </div>
      </div>`;
    this.log.append(el);
    const text = $('.bot-text', el);
    const msg = {
      q, el, text, shown: -1, done: false,
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
      },
    };
    $('.peek', el).addEventListener('click', () => this.onPeek?.(msg));
    this.msgs.push(msg);
    this.scroll();
    return msg;
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
