// 开场：一个干净的 AI 聊天页（沿用推理视频 v7 开场的样子：深色、网站的配色和字体；不模仿任何真实产品）。
// 在输入框里打出任务（中文走拼音输入法，英文和标点直接敲）→ 按发送 → 任务变成消息气泡、助手“正在思考…”
// → 助手的回复不是一句话，而是一个“动作”卡片（read_file sales.csv）→ 镜头推进卡片，桌面（编辑器 + 终端）从里面展开。
// 由 lib/opening.js 的聊天页部分改写而来（那边还连着 3D 词元，这里不需要）；全部由时间 t 决定。
import { clamp, lerp, seg, smooth, easeOut, easeInOut, el } from './util.js';
import { esc } from '/public/js/ui.js';

const SEND_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5.5M5.8 11.4L12 5.2l6.2 6.2" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const mix = (a, b, k) => a.map((x, i) => Math.round(lerp(x, b[i], k)));
const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a.toFixed(3)})`;
const DIM = [122, 133, 158], ACC = [94, 240, 212];

// 拼音按音节加分隔符（cheng'shi'de）；只显示已经敲出来的字母
function pyPrefix(segd, n) {
  let out = '', k = 0;
  for (const c of segd) {
    if (c === "'") { if (k < n) out += c; continue; }
    if (k >= n) break;
    out += c; k++;
  }
  return out.replace(/'$/, '');
}

export class Opening {
  constructor({ frame, T, prompt, model, title, sub, firstCall }) {
    this.T = T;
    const wrap = (this.wrap = el('div', 'cpw'));
    frame.insertBefore(wrap, frame.querySelector('#desk'));
    const page = (this.page = el('div', 'cpage', '', wrap));
    page.innerHTML = `
      <div class="cp-model"><span class="cp-av"><i></i></span><b>${esc(model)}</b></div>
      <header class="cp-head"><div class="t">${title}</div><div class="s">${sub}</div></header>
      <div class="cp-rule"></div>
      <div class="cp-hello"><span class="cp-av"><i></i></span><div class="h">有什么要我做的？</div><div class="s">${esc(model)} · 能在沙箱里读写文件、执行命令</div></div>
      <div class="cp-user"><div class="cp-bub">${esc(prompt)}</div></div>
      <div class="cp-bot"><span class="cp-av"><i></i></span><div><div class="n">${esc(model)}</div><div class="th"><span class="tx">正在思考</span><span class="dots"><i></i><i></i><i></i></span></div>
        <div class="cp-act"><span class="op">▶ ${esc(firstCall.name)}</span><span class="arg">${esc(firstCall.arg)}</span><span class="st">在沙箱里执行…</span></div></div></div>
      <div class="cp-box"><div class="cp-in"><div class="cp-line"><span class="cp-txt"></span><span class="cp-comp"></span><span class="cp-caret"></span><span class="cp-ph">给 ${esc(model.split('-Instruct')[0])} 布置一个任务</span></div></div><span class="cp-send">${SEND_SVG}</span></div>
      <div class="cp-hint">每一步都来自真实模型在沙箱里的离线录制 · Enter 发送</div>
      <div class="cp-ime"><div class="py"></div><div class="cs"></div></div>`;
    const q = (s) => page.querySelector(s);
    Object.assign(this, {
      model: q('.cp-model'), head: q('.cp-head'), rule: q('.cp-rule'), hello: q('.cp-hello'),
      userEl: q('.cp-user'), bub: q('.cp-bub'), bot: q('.cp-bot'), botAv: q('.cp-bot .cp-av i'), th: q('.cp-bot .th'), thTx: q('.cp-bot .tx'), dots: [...page.querySelectorAll('.cp-bot .dots i')],
      act: q('.cp-act'), actSt: q('.cp-act .st'),
      box: q('.cp-box'), inp: q('.cp-in'), line: q('.cp-line'), txt: q('.cp-txt'), comp: q('.cp-comp'), caret: q('.cp-caret'), ph: q('.cp-ph'), send: q('.cp-send'),
      hint: q('.cp-hint'), ime: q('.cp-ime'), imePy: q('.cp-ime .py'), imeCs: q('.cp-ime .cs'),
      avs: [q('.cp-model .cp-av i'), q('.cp-hello .cp-av i')],
    });
    this.imeKey = '';
  }

  typingAt(t) {
    const T = this.T;
    if (t >= T.sendT) return { txt: '', comp: '', ime: null, last: T.sendT };
    let txt = '', comp = '', ime = null, last = -1;
    for (const w of T.typing) {
      for (const k of w.keys) if (t >= k) last = Math.max(last, k);
      if (t >= w.commit) { txt += w.s; last = Math.max(last, w.commit); continue; }
      const n = w.keys.filter((k) => t >= k).length;
      if (w.chars) { txt += w.s.slice(0, n); break; }
      if (n > 0 && w.py) {
        comp = pyPrefix(w.seg, n);
        const kl = w.keys[w.keys.length - 1];
        if (t >= kl - 0.03) ime = { w, a: smooth(seg(t, kl - 0.03, kl + 0.05)) };
      }
      break;
    }
    return { txt, comp, ime, last };
  }

  // 动作卡片的中心（排版坐标，不含推进的缩放）
  actCenter() {
    const b = this.bot, a = this.act;
    return { x: b.offsetLeft + a.parentElement.offsetLeft + a.offsetLeft + a.offsetWidth / 2, y: b.offsetTop + a.parentElement.offsetTop + a.offsetTop + a.offsetHeight / 2 };
  }

  update(t) {
    const T = this.T;
    const on = t < T.end;
    this.wrap.style.display = on ? 'block' : 'none';
    if (!on) return;
    const sent = t >= T.sendT;

    // ---- 输入框
    const ty = this.typingAt(t);
    this.txt.textContent = ty.txt;
    this.comp.textContent = ty.comp;
    this.ph.style.display = ty.txt || ty.comp ? 'none' : '';
    // 输入框里放不下整句：和真的输入框一样往左滚，光标始终看得见
    const over = Math.max(0, this.line.scrollWidth - this.inp.clientWidth + 12);
    this.line.style.transform = over ? `translateX(${(-over).toFixed(1)}px)` : '';
    const typingNow = !sent && ty.last >= 0 && t - ty.last < 0.55;
    const idle = sent ? t - T.sendT : t - Math.max(0, ty.last);
    this.caret.style.opacity = typingNow || (idle % 1.06) < 0.53 ? '1' : '0';
    const ready = !sent && !!ty.txt && !ty.comp;
    this.send.classList.toggle('on', ready || (t >= T.sendT && t < T.sendT + 0.12));
    const press = smooth(seg(t, T.sendT - 0.14, T.sendT - 0.03)) * (1 - smooth(seg(t, T.sendT + 0.03, T.sendT + 0.28)));
    const flash = Math.exp(-Math.max(0, t - T.sendT) / 0.25) * (t >= T.sendT ? 1 : 0);
    this.send.style.transform = `scale(${(1 - 0.1 * press).toFixed(4)})`;
    this.send.style.boxShadow = ready || flash > 0.02 ? `0 0 ${(28 + 40 * flash).toFixed(1)}px rgba(94, 240, 212, ${(0.4 + 0.45 * flash).toFixed(3)})` : 'none';
    const mv = easeInOut(seg(t, T.sendT + 0.05, T.sendT + 0.65));
    const boxTop = lerp(T.boxY0, T.boxY1, mv);
    this.box.style.top = `${boxTop.toFixed(1)}px`;
    this.hint.style.top = `${(boxTop + 140).toFixed(1)}px`;
    this.hint.style.opacity = (1 - smooth(seg(t, T.sendT, T.sendT + 0.3))).toFixed(3);
    if (ty.ime && ty.ime.a > 0.01) {
      const w = ty.ime.w;
      if (this.imeKey !== w.s) {
        this.imeKey = w.s;
        this.imePy.textContent = w.seg;
        this.imeCs.innerHTML = w.cands.map((c, i) => `<span class="${i ? '' : 'on'}"><b>${i + 1}</b>${esc(c)}</span>`).join('');
      }
      this.ime.style.display = 'block';
      this.ime.style.opacity = ty.ime.a.toFixed(3);
      this.ime.style.left = `${Math.min(1480, this.box.offsetLeft + this.inp.offsetLeft + this.comp.offsetLeft - over - 16).toFixed(1)}px`;
      this.ime.style.top = `${(boxTop + 96 + 6 * (1 - ty.ime.a)).toFixed(1)}px`;
    } else this.ime.style.display = 'none';

    // ---- 页眉、欢迎语
    const headOut = smooth(seg(t, T.push0, T.push0 + 0.9));
    for (const e of [this.head, this.model, this.rule]) e.style.opacity = (1 - headOut).toFixed(3);
    const hi = smooth(seg(t, 0.25, 1.0)) * (1 - smooth(seg(t, T.sendT, T.sendT + 0.35)));
    this.hello.style.display = hi > 0.001 ? '' : 'none';
    this.hello.style.opacity = hi.toFixed(3);
    this.hello.style.transform = `translateX(-50%) translateY(${(-24 * smooth(seg(t, T.sendT, T.sendT + 0.35))).toFixed(1)}px)`;
    const breathe = (p) => 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / p);
    for (const i of this.avs) { const b = breathe(3); i.style.transform = `scale(${(1 - 0.4 * b).toFixed(3)})`; i.style.opacity = (1 - 0.4 * b).toFixed(3); }
    { const b = breathe(0.9); this.botAv.style.transform = `scale(${(1 - 0.45 * b).toFixed(3)})`; this.botAv.style.opacity = (1 - 0.4 * b).toFixed(3); }

    // ---- 消息：任务成为气泡；助手先“正在思考…”，然后回复一个动作
    const ub = seg(t, T.bubT, T.bubT + 0.45), bb = seg(t, T.botT, T.botT + 0.45);
    this.userEl.style.display = t >= T.bubT ? '' : 'none';
    this.userEl.style.opacity = smooth(ub).toFixed(3);
    this.userEl.style.transform = `translateY(${(18 * (1 - easeOut(ub))).toFixed(1)}px)`;
    this.bot.style.display = t >= T.botT ? '' : 'none';
    this.bot.style.transform = `translateY(${(18 * (1 - easeOut(bb))).toFixed(1)}px)`;
    this.bot.style.opacity = smooth(bb).toFixed(3);
    const sh = (((t - T.botT) / 1.7) % 1 + 1) % 1;
    this.thTx.style.backgroundPosition = `${(100 - 200 * sh).toFixed(1)}% 0`;
    this.dots.forEach((d, i) => {
      const ph = t - T.botT - 0.16 * i;
      const k = ph < 0 ? 0 : Math.sin(Math.PI * Math.min(1, (ph % 1.1) / 0.55)) ** 2 * ((ph % 1.1) < 0.55 ? 1 : 0);
      d.style.transform = `translateY(${(-7 * k).toFixed(2)}px)`;
      d.style.background = rgb(mix(DIM, ACC, k));
    });
    const ak = seg(t, T.toolT, T.toolT + 0.4);
    this.act.style.display = t >= T.toolT ? '' : 'none';
    this.act.style.opacity = smooth(ak).toFixed(3);
    this.act.style.transform = `translateY(${(14 * (1 - easeOut(ak))).toFixed(1)}px)`;
    const glow = smooth(seg(t, T.push0 + 0.4, T.push1));
    this.act.style.boxShadow = `0 0 ${(10 + 50 * glow).toFixed(1)}px rgba(94, 240, 212, ${(0.15 + 0.5 * glow).toFixed(3)})`;
    this.act.style.borderColor = rgb(ACC, 0.45 + 0.5 * glow);
    this.actSt.style.opacity = (0.5 + 0.5 * Math.sin(t * 7) ** 2).toFixed(3);
    this.th.style.opacity = (1 - 0.6 * smooth(seg(t, T.toolT, T.toolT + 0.4))).toFixed(3);

    // ---- 推进动作卡片：周围的界面虚掉，最后整页散去
    const f = easeInOut(seg(t, T.push0, T.push1));
    const Z = lerp(1, 2.6, f) * (1 + 1.6 * smooth(seg(t, T.push1 - 0.5, T.end)));
    const ac = this.actCenter();
    const F = { x: lerp(960, ac.x, f), y: lerp(540, ac.y, f) };
    this.page.style.transform = Z > 1.0001 ? `translate(${(960 - Z * F.x).toFixed(2)}px, ${(lerp(540, 470, f) - Z * F.y).toFixed(2)}px) scale(${Z.toFixed(5)})` : '';
    const dof = smooth(seg(t, T.push0 + 0.3, T.push1));
    for (const e of [this.box, this.userEl, this.hello]) e.style.filter = dof > 0.01 ? `blur(${(4 * dof).toFixed(2)}px)` : '';
    this.page.style.opacity = (1 - smooth(seg(t, T.desk0 + 0.1, T.desk1 - 0.2))).toFixed(3);
    this.wrap.style.opacity = smooth(seg(t, 0.05, 0.8)).toFixed(3);
  }
}
