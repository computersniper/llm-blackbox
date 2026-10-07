// 开场（从 ../lib/opening.js 改来）：一个干净的 AI 聊天页（网站的配色和字体气质，不模仿任何真实产品，助手就叫 Qwen3-VL-2B）。
// 点「图片」→ 选中桌上的苹果那张 → 用拼音逐词打出问题（输入法候选一闪而过）→ 按发送 → 图和问题变成消息、助手“正在看图…”
// → 镜头推近消息，文字按真实分词裂成词元块 → 界面散去，图片换成 3D 的照片、词元块换成 3D 方块，镜头跟着它们飞向远处的黑箱，
// 从黑箱正面的取景窗推进去。全部由时间 t 决定（score.js 里的 OPEN 时间表），逐帧渲染完全确定。
import { THREE } from '../lib/engine.js';
import { clamp, lerp, seg, smooth, easeOut, easeInOut, v3 } from '../lib/cam.js';
import { textTexture } from '/public/js/stage/engine.js';
import { RoundedBoxGeometry } from '/public/js/vendor/three/addons/geometries/RoundedBoxGeometry.js';
import { tokPlain, esc } from '/public/js/ui.js';

const el = (tag, cls, html, parent) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; if (parent) parent.append(e); return e; };
const TILE_GEO = new RoundedBoxGeometry(0.36, 0.24, 0.36, 3, 0.045);
const FACE_GEO = new THREE.PlaneGeometry(0.33, 0.21);
const SEND_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5.5M5.8 11.4L12 5.2l6.2 6.2" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const PIC_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3.2" y="4.6" width="17.6" height="14.8" rx="3" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="9" cy="10" r="1.9" fill="currentColor"/><path d="M4.5 17.6l5.2-5 3.6 3.3 2.6-2.3 3.8 4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>';
const CUR_SVG = '<svg viewBox="0 0 24 24" width="34" height="34"><path d="M5 2.5v17.2l4.6-4.3 3 6.6 2.9-1.3-3-6.5h6.2z" fill="#fff" stroke="#05080f" stroke-width="1.4" stroke-linejoin="round"/></svg>';
const mix = (a, b, k) => a.map((x, i) => Math.round(lerp(x, b[i], k)));
const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a.toFixed(3)})`;
const INK = [233, 239, 249], TOK = [196, 244, 249], DIM = [122, 133, 158], ACC = [94, 240, 212];

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
  constructor({ E, S, Q, frame, T, imgId = 'apples' }) {
    this.E = E; this.S = S; this.Q = Q; this.T = T;
    this.user = Q.tokens.map((t, i) => ({ ...t, i })).filter((t) => t.role === 'user' && !t.sp && t.s !== 'user' && t.s !== '\n');
    this.imgs = Q.manifest.images;
    this.pickIdx = this.imgs.findIndex((im) => im.id === imgId);
    this.fog0 = { near: E.scene.fog.near, far: E.scene.fog.far };
    this.buildPage(frame);
    this.buildSnow();
    this.buildTiles();
    this.buildStreaks();
  }

  /* ------------------------------------------------------------ 聊天页（DOM，按 1080p 直接排版） */

  buildPage(frame) {
    const wrap = (this.wrap = el('div', 'cpw'));
    frame.insertBefore(wrap, frame.querySelector('#vignette'));
    this.bg = el('canvas', 'cp-bg', '', wrap);
    this.bg.width = 1920; this.bg.height = 1080;
    const page = (this.page = el('div', 'cpage', '', wrap));
    const toks = this.user.map((t) => `<span class="tk">${esc(tokPlain(t.s))}</span>`).join('');
    const m = this.Q.manifest.model;
    const im = this.imgs[this.pickIdx];
    page.innerHTML = `
      <div class="cp-model"><span class="cp-av"><i></i></span><b>Qwen3-VL-2B</b></div>
      <header class="cp-head"><div class="t">AI 是怎么看图的</div><div class="s">走进多模态大模型的“黑箱”</div></header>
      <div class="cp-rule"></div>
      <div class="cp-hello"><span class="cp-av"><i></i></span><div class="h">有什么想问的？</div><div class="s">Qwen3-VL-2B · 阿里开源的 ${(m.params.total / 1e8).toFixed(0)} 亿参数视觉语言模型</div></div>
      <div class="cp-user"><div class="cp-uimg"><img src="data/${im.file}" alt=""></div><div class="cp-bub">${toks}</div></div>
      <div class="cp-bot"><span class="cp-av"><i></i></span><div><div class="n">Qwen3-VL-2B</div><div class="th"><span class="tx">正在看图</span><span class="dots"><i></i><i></i><i></i></span></div></div></div>
      <div class="cp-pick"><div class="lab">选一张图</div><div class="row">${this.imgs.map((x, i) => `<div class="pc${i === this.pickIdx ? ' it' : ''}"><img src="data/${x.file}" alt=""><span>${esc(x.title)}</span></div>`).join('')}</div></div>
      <div class="cp-box"><span class="cp-att">${PIC_SVG}<img src="data/${im.file}" alt=""></span><div class="cp-in"><span class="cp-txt"></span><span class="cp-comp"></span><span class="cp-caret"></span><span class="cp-ph">先选一张图，再提问</span></div><span class="cp-send">${SEND_SVG}</span></div>
      <div class="cp-hint">回答来自真实模型的离线运行 · Enter 发送</div>
      <div class="cp-ime"><div class="py"></div><div class="cs"></div></div>
      <div class="cp-cur"><span class="rip"></span>${CUR_SVG}</div>`;
    const q = (s) => page.querySelector(s);
    Object.assign(this, {
      model: q('.cp-model'), head: q('.cp-head'), rule: q('.cp-rule'), hello: q('.cp-hello'),
      userEl: q('.cp-user'), uimg: q('.cp-uimg'), bub: q('.cp-bub'), bot: q('.cp-bot'), botAv: q('.cp-bot .cp-av i'), thTx: q('.cp-bot .tx'), dots: [...page.querySelectorAll('.cp-bot .dots i')],
      box: q('.cp-box'), att: q('.cp-att'), attImg: q('.cp-att img'), attSvg: q('.cp-att svg'), inp: q('.cp-in'), txt: q('.cp-txt'), comp: q('.cp-comp'), caret: q('.cp-caret'), ph: q('.cp-ph'), send: q('.cp-send'),
      hint: q('.cp-hint'), ime: q('.cp-ime'), imePy: q('.cp-ime .py'), imeCs: q('.cp-ime .cs'),
      pick: q('.cp-pick'), pcs: [...page.querySelectorAll('.cp-pick .pc')], cur: q('.cp-cur'), rip: q('.cp-cur .rip'),
      avs: [q('.cp-model .cp-av i'), q('.cp-hello .cp-av i')],
    });
    this.toks = [...this.bub.querySelectorAll('.tk')];
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
      if (n > 0 && w.py) {
        comp = pyPrefix(w.seg, n);
        const kl = w.keys[w.keys.length - 1];
        if (t >= kl - 0.03) ime = { w, a: smooth(seg(t, kl - 0.03, kl + 0.05)) };
      }
      break;
    }
    return { txt, comp, ime, last };
  }

  // 你的消息（图 + 文字）的中心（排版坐标，不含镜头推进的缩放）
  msgCenter() {
    const u = this.userEl;
    return { x: u.offsetLeft + u.offsetWidth / 2, y: u.offsetTop + u.offsetHeight / 2 };
  }

  // 鼠标：点「图片」按钮 → 点中苹果那张（只在选图那几秒出现）
  cursorAt(t) {
    const T = this.T;
    const a = { x: this.box.offsetLeft + this.att.offsetLeft + this.att.offsetWidth / 2, y: T.boxY0 + this.box.offsetHeight / 2 };
    const pc = this.pcs[this.pickIdx];
    const b = { x: this.pick.offsetLeft + pc.offsetLeft + pc.offsetWidth / 2, y: this.pick.offsetTop + pc.offsetTop + pc.offsetHeight / 2 - 10 };
    const s = { x: 1240, y: 930 };
    let p;
    if (t < T.attachT) p = { x: lerp(s.x, a.x, easeInOut(seg(t, T.attachT - 0.9, T.attachT - 0.08))), y: lerp(s.y, a.y, easeInOut(seg(t, T.attachT - 0.9, T.attachT - 0.08))) };
    else p = { x: lerp(a.x, b.x, easeInOut(seg(t, T.attachT + 0.35, T.pickT - 0.08))), y: lerp(a.y, b.y, easeInOut(seg(t, T.attachT + 0.35, T.pickT - 0.08))) };
    const click = Math.max(t >= T.attachT ? Math.max(0, 1 - (t - T.attachT) / 0.45) : 0, t >= T.pickT ? Math.max(0, 1 - (t - T.pickT) / 0.45) : 0);
    const alpha = smooth(seg(t, T.attachT - 1.1, T.attachT - 0.7)) * (1 - smooth(seg(t, T.pickT + 0.5, T.pickT + 0.9)));
    return { ...p, click, alpha };
  }

  updatePage(t) {
    const T = this.T;
    const on = t < T.swapT + 0.3;
    this.wrap.style.display = on ? 'block' : 'none';
    if (!on) return;
    this.wrap.style.opacity = smooth(seg(t, 0.05, 0.8)).toFixed(3);
    const sent = t >= T.sendT;

    // ---- 选图：图片按钮 → 弹出一排缩略图 → 点中苹果 → 缩略图挂到输入框左边
    const picked = t >= T.pickT;
    const pk = smooth(seg(t, T.attachT, T.attachT + 0.3)) * (1 - smooth(seg(t, T.pickT + 0.15, T.pickT + 0.45)));
    this.pick.style.display = pk > 0.001 ? 'block' : 'none';
    this.pick.style.opacity = pk.toFixed(3);
    this.pick.style.transform = `translateX(-50%) translateY(${(10 * (1 - pk)).toFixed(1)}px)`;
    const hov = smooth(seg(t, T.pickT - 0.5, T.pickT - 0.2));
    this.pcs.forEach((p, i) => {
      const me = i === this.pickIdx;
      const on2 = me && (hov > 0 || picked);
      p.classList.toggle('on', on2);
      p.style.transform = me ? `translateY(${(-6 * hov * (1 - smooth(seg(t, T.pickT, T.pickT + 0.2)) * 0.5)).toFixed(1)}px)` : '';
    });
    const ai = smooth(seg(t, T.pickT + 0.05, T.pickT + 0.35));
    this.attImg.style.opacity = ai.toFixed(3);
    this.attSvg.style.opacity = (1 - ai).toFixed(3);
    this.att.classList.toggle('has', picked);
    this.att.classList.toggle('press', t >= T.attachT - 0.12 && t < T.attachT + 0.15);
    const c = this.cursorAt(t);
    this.cur.style.display = c.alpha > 0.001 ? 'block' : 'none';
    if (c.alpha > 0.001) {
      this.cur.style.left = `${c.x.toFixed(1)}px`;
      this.cur.style.top = `${c.y.toFixed(1)}px`;
      this.cur.style.opacity = c.alpha.toFixed(3);
      this.rip.style.opacity = (c.click * 0.9).toFixed(3);
      this.rip.style.transform = `translate(-50%, -50%) scale(${(0.4 + 1.2 * (1 - c.click)).toFixed(3)})`;
    }

    // ---- 输入框
    const ty = this.typingAt(t);
    this.txt.textContent = ty.txt;
    this.comp.textContent = ty.comp;
    this.ph.style.display = ty.txt || ty.comp ? 'none' : '';
    this.ph.textContent = picked ? '问点什么吧' : '先选一张图，再提问';
    const typingNow = !sent && ty.last >= 0 && t - ty.last < 0.55;
    const idle = sent ? t - T.sendT : t - Math.max(0, ty.last);
    this.caret.style.opacity = typingNow || (idle % 1.06) < 0.53 ? '1' : '0';
    const ready = !sent && !!ty.txt && !ty.comp && picked && t >= T.typing[T.typing.length - 1].commit;
    this.send.classList.toggle('on', ready || (t >= T.sendT && t < T.sendT + 0.12));
    const press = smooth(seg(t, T.sendT - 0.14, T.sendT - 0.03)) * (1 - smooth(seg(t, T.sendT + 0.03, T.sendT + 0.28)));
    const flash = Math.exp(-Math.max(0, t - T.sendT) / 0.25) * (t >= T.sendT ? 1 : 0);
    this.send.style.transform = `scale(${(1 - 0.1 * press).toFixed(4)})`;
    this.send.style.boxShadow = ready || flash > 0.02 ? `0 0 ${(28 + 40 * flash).toFixed(1)}px rgba(94, 240, 212, ${(0.4 + 0.45 * flash).toFixed(3)})` : 'none';
    // 发送以后附图从输入框里消失（它跟着消息走了）
    if (sent) { this.attImg.style.opacity = (1 - smooth(seg(t, T.sendT, T.sendT + 0.2))).toFixed(3); this.attSvg.style.opacity = smooth(seg(t, T.sendT + 0.1, T.sendT + 0.4)).toFixed(3); this.att.classList.remove('has'); }
    const mv = easeInOut(seg(t, T.sendT + 0.05, T.sendT + 0.65));
    const boxTop = lerp(T.boxY0, T.boxY1, mv);
    this.box.style.top = `${boxTop.toFixed(1)}px`;
    this.hint.style.top = `${(boxTop + 140).toFixed(1)}px`;
    this.hint.style.opacity = (1 - smooth(seg(t, T.sendT, T.sendT + 0.3))).toFixed(3);
    this.pick.style.top = `${(boxTop - 230).toFixed(1)}px`;
    if (ty.ime && ty.ime.a > 0.01) {
      const w = ty.ime.w;
      if (this.imeKey !== w.s) {
        this.imeKey = w.s;
        this.imePy.textContent = w.seg;
        this.imeCs.innerHTML = w.cands.map((cc, i) => `<span class="${i ? '' : 'on'}"><b>${i + 1}</b>${esc(cc)}</span>`).join('');
      }
      this.ime.style.display = 'block';
      this.ime.style.opacity = ty.ime.a.toFixed(3);
      this.ime.style.left = `${(this.box.offsetLeft + this.inp.offsetLeft + this.comp.offsetLeft - 16).toFixed(1)}px`;
      this.ime.style.top = `${(boxTop + 96 + 6 * (1 - ty.ime.a)).toFixed(1)}px`;
    } else this.ime.style.display = 'none';

    // ---- 页眉、欢迎语
    const headOut = smooth(seg(t, T.push0, T.push0 + 0.9));
    for (const e of [this.head, this.model, this.rule]) e.style.opacity = (1 - headOut).toFixed(3);
    const hi = smooth(seg(t, 0.25, 1.0)) * (1 - smooth(seg(t, T.attachT, T.attachT + 0.35)));
    this.hello.style.display = hi > 0.001 ? '' : 'none';
    this.hello.style.opacity = hi.toFixed(3);
    this.hello.style.transform = `translateX(-50%) translateY(${(-24 * smooth(seg(t, T.attachT, T.attachT + 0.35))).toFixed(1)}px)`;
    const breathe = (p) => 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / p);
    for (const i of this.avs) { const b = breathe(3); i.style.transform = `scale(${(1 - 0.4 * b).toFixed(3)})`; i.style.opacity = (1 - 0.4 * b).toFixed(3); }
    { const b = breathe(0.9); this.botAv.style.transform = `scale(${(1 - 0.45 * b).toFixed(3)})`; this.botAv.style.opacity = (1 - 0.4 * b).toFixed(3); }

    // ---- 消息
    const ub = seg(t, T.bubT, T.bubT + 0.45), bb = seg(t, T.botT, T.botT + 0.45);
    this.userEl.style.display = t >= T.bubT ? '' : 'none';
    this.userEl.style.opacity = smooth(ub).toFixed(3);
    this.userEl.style.transform = `translateY(${(18 * (1 - easeOut(ub))).toFixed(1)}px)`;
    this.bot.style.display = t >= T.botT ? '' : 'none';
    this.bot.style.transform = `translateY(${(18 * (1 - easeOut(bb))).toFixed(1)}px)`;
    const sh = (((t - T.botT) / 1.7) % 1 + 1) % 1;
    this.thTx.style.backgroundPosition = `${(100 - 200 * sh).toFixed(1)}% 0`;
    this.dots.forEach((d, i) => {
      const ph = t - T.botT - 0.16 * i;
      const k = ph < 0 ? 0 : Math.sin(Math.PI * Math.min(1, (ph % 1.1) / 0.55)) ** 2 * ((ph % 1.1) < 0.55 ? 1 : 0);
      d.style.transform = `translateY(${(-7 * k).toFixed(2)}px)`;
      d.style.background = rgb(mix(DIM, ACC, k));
    });

    // ---- 文字裂成词元块；图片的边框亮一下
    this.toks.forEach((s, k) => {
      const a = T.split0 + k * T.splitGap;
      const e = easeInOut(seg(t, a, a + T.splitD));
      const g = smooth(seg(t, T.swapT - 0.6, T.swapT - 0.05));
      const gone = smooth(seg(t, T.swapT, T.swapT + 0.14));
      s.style.marginLeft = k ? `${(18 * e).toFixed(2)}px` : '0';
      s.style.padding = `${(5 * e).toFixed(2)}px ${(14 * e).toFixed(2)}px`;
      s.style.borderColor = rgb([94, 228, 240], 0.55 * e + 0.35 * g);
      s.style.background = rgb([94, 228, 240], 0.14 * e + 0.08 * g);
      s.style.color = rgb(mix(INK, TOK, e));
      s.style.boxShadow = g > 0.01 ? `0 0 ${(26 * g).toFixed(1)}px rgba(94, 228, 240, ${(0.45 * g).toFixed(3)})` : 'none';
      s.style.transform = `translateY(${(Math.sin(Math.PI * e) * (k % 2 ? -5 : 5)).toFixed(2)}px)`;
      s.style.opacity = (1 - gone).toFixed(3);
    });
    const ig = smooth(seg(t, T.split0, T.split0 + 0.6));
    this.uimg.style.boxShadow = ig > 0.01 ? `0 0 ${(30 * ig).toFixed(1)}px rgba(94, 228, 240, ${(0.35 * ig).toFixed(3)})` : '';
    this.uimg.style.borderColor = rgb([94, 228, 240], 0.3 + 0.4 * ig);
    this.uimg.style.opacity = (1 - smooth(seg(t, T.swapT, T.swapT + 0.14))).toFixed(3);
    const bubOut = smooth(seg(t, T.split0 + 0.3, T.split0 + 0.3 + T.splitD + T.splitGap * this.toks.length));
    this.bub.style.background = rgb([94, 228, 240], 0.1 * (1 - bubOut));
    this.bub.style.borderColor = rgb([94, 228, 240], 0.3 * (1 - bubOut));

    // ---- 镜头推近消息；周围的界面虚掉，最后散去，露出后面的 3D 空间
    const dof = smooth(seg(t, T.push0 + 0.5, T.push1));
    const diss = smooth(seg(t, T.diss0, T.diss1));
    for (const e of [this.box, this.bot]) {
      e.style.filter = dof + diss > 0.01 ? `blur(${(2.6 * dof + 9 * diss).toFixed(2)}px)` : '';
      e.style.opacity = (e === this.bot ? smooth(bb) : 1) * (1 - diss);
    }
    this.bg.style.opacity = (1 - diss).toFixed(3);
    const f = easeInOut(seg(t, T.push0, T.push1));
    const Z = lerp(1, T.zoom, f) * (1 + T.zoom2 * smooth(seg(t, T.push1, T.swapT + 0.3)));
    const bc = this.msgCenter();
    const Fp = { x: lerp(960, bc.x, f), y: lerp(540, bc.y, f) };
    const Sp = { x: 960, y: lerp(540, T.focusY, f) };
    this.page.style.transform = Z > 1.0001 ? `translate(${(Sp.x - Z * Fp.x).toFixed(2)}px, ${(Sp.y - Z * Fp.y).toFixed(2)}px) scale(${Z.toFixed(5)})` : '';
    this.drawSnow(t);
  }

  /* ------------------------------------------------------------ 背景：缓慢上浮的“海雪”（闭式，不积累状态） */

  buildSnow() {
    let s = 20240;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    this.snow = Array.from({ length: 190 }, () => {
      const z = 0.2 + rnd() * 0.8;
      return { x0: rnd() * 1920, y0: rnd() * 1104, z, vy: -(4 + 14 * z) * 1.4, ph: rnd() * Math.PI * 2, w: 0.3 + 0.4 * z, r: (0.5 + z * 1.4) * 1.35 };
    });
    this.g = this.bg.getContext('2d');
  }

  warpAt(t) {
    const w0 = this.T.warpT, D = 1.3;
    if (t <= w0) return { k: 0, I: 0 };
    const kf = (u) => (u > 0 && u < 1 ? Math.sin(Math.PI * Math.pow(u, 0.6)) : 0);
    const u = (t - w0) / D;
    let I = 0;
    const n = 48, uu = Math.min(1, u);
    for (let i = 0; i < n; i++) I += kf(((i + 0.5) / n) * uu) * (uu / n) * D;
    return { k: kf(u), I };
  }

  snowPos(p, t, W) {
    const span = 1104;
    let y = (((p.y0 + p.vy * t) % span) + span) % span - 12;
    let x = p.x0 + ((6 * p.z) / p.w) * (Math.cos(p.ph) - Math.cos(p.ph + p.w * t)) * 1.4;
    if (W.I > 0) {
      const f = Math.exp(5.5 * (0.4 + p.z) * W.I);
      x = 960 + (x - 960) * f; y = 520 + (y - 520) * f;
    }
    return [x, y];
  }

  drawSnow(t) {
    const g = this.g;
    const grad = g.createRadialGradient(960, 540, 0, 960, 540, 1440);
    grad.addColorStop(0, 'rgb(6,13,26)');
    grad.addColorStop(0.6, 'rgb(6,13,26)');
    grad.addColorStop(1, 'rgb(4,9,19)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 1920, 1080);
    const W = this.warpAt(t), Wp = this.warpAt(t - 1 / 60);
    for (const p of this.snow) {
      const [x, y] = this.snowPos(p, t, W);
      const a = 0.12 + p.z * 0.45;
      if (W.k > 0.05) {
        const [px, py] = this.snowPos(p, t - 1 / 60, Wp);
        g.strokeStyle = `rgba(120,230,220,${(a * (0.6 + W.k)).toFixed(3)})`;
        g.lineWidth = p.r;
        g.beginPath();
        g.moveTo(px, py);
        g.lineTo(x + (x - px) * 3 * W.k, y + (y - py) * 3 * W.k);
        g.stroke();
      } else {
        g.fillStyle = `rgba(120,230,220,${a.toFixed(3)})`;
        g.beginPath();
        g.arc(x, y, p.r, 0, Math.PI * 2);
        g.fill();
      }
    }
  }

  /* ------------------------------------------------------------ 飞行的照片和词元（3D） */

  buildTiles() {
    const S = this.S;
    const mat = (this.tileMat = S.roleMat.user.clone());
    mat.emissiveIntensity = 0.07;
    this.tiles = this.user.map((t) => {
      const g = new THREE.Group();
      g.add(new THREE.Mesh(TILE_GEO, mat));
      const face = new THREE.Mesh(FACE_GEO, new THREE.MeshBasicMaterial({ map: textTexture(tokPlain(t.s), { color: '#5ee4f0', w: 512, h: 320, font: '600 152px "PingFang SC","Noto Sans SC",sans-serif' }), transparent: true }));
      face.position.z = 0.181;
      g.add(face);
      g.visible = false;
      S.root.add(g);
      return g;
    });
    // 照片：一整张平面（交接时盖住聊天页上的图，飞到取景窗前再交给舞台里由图块拼成的照片）
    const V = this.Q.V;
    this.photoW = V.gw * 0.1; this.photoH = V.gh * 0.1;
    this.photo = new THREE.Mesh(new THREE.PlaneGeometry(this.photoW, this.photoH), new THREE.MeshBasicMaterial({ map: S.imgTex, transparent: true, color: new THREE.Color(0.82, 0.82, 0.82) }));
    this.photo.visible = false;
    S.root.add(this.photo);
    this.photoEdge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(this.photoW + 0.04, this.photoH + 0.04)), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.6 }));
    this.photo.add(this.photoEdge);
  }

  buildStreaks() {
    let s = 12345;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    const n = 700, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos[i * 3] = (rnd() - 0.5) * 60 - 6;
      pos[i * 3 + 1] = rnd() * 22 - 4;
      pos[i * 3 + 2] = 16 + rnd() * 120;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.streaks = new THREE.Points(g, new THREE.PointsMaterial({ color: 0x9fe8ff, size: 0.07, transparent: true, opacity: 0, depthWrite: false }));
    this.E.scene.add(this.streaks);
  }

  // 交接那一刻：屏幕上的图和每个词元块 → 镜头前 d 处的位置（镜头坐标系）和缩放
  swapLocal(cam0, t) {
    if (this.local) return this.local;
    this.updatePage(this.T.swapT);
    const fr = document.querySelector('#frame').getBoundingClientRect();
    const rectOf = (e) => { const r = e.getBoundingClientRect(); return { x: r.left - fr.left + r.width / 2, y: r.top - fr.top + r.height / 2, w: r.width, h: r.height }; };
    const rects = this.toks.map(rectOf);
    const ir = rectOf(this.uimg);
    this.updatePage(t);
    const c = new THREE.PerspectiveCamera(cam0.fov, this.E.w / this.E.h, 0.01, 400);
    if (this.E.shiftY || this.E.shiftX) c.setViewOffset(this.E.w, this.E.h, this.E.shiftX || 0, this.E.shiftY || 0, this.E.w, this.E.h);
    c.position.copy(cam0.pos);
    c.lookAt(cam0.look);
    c.updateMatrixWorld();
    c.updateProjectionMatrix();
    const d = this.T.depth;
    const fwd = cam0.look.clone().sub(cam0.pos).normalize();
    const inv = c.matrixWorld.clone().invert();
    const worldPerPx = (2 * d * Math.tan((cam0.fov * Math.PI) / 360)) / this.E.h;
    const toLocal = (r) => {
      const ndc = v3((r.x / this.E.w) * 2 - 1, -((r.y / this.E.h) * 2 - 1), 0.5).unproject(c);
      const dir = ndc.sub(cam0.pos).normalize();
      return cam0.pos.clone().add(dir.multiplyScalar(d / dir.dot(fwd))).applyMatrix4(inv);
    };
    this.local = rects.map((r) => ({ local: toLocal(r), scale: Math.min((r.h * 1.08) / 0.24, (r.w * 1.0) / 0.36) * worldPerPx }));
    this.localImg = { local: toLocal(ir), scale: (ir.w * worldPerPx) / this.photoW };
    return this.local;
  }

  static basis(cam) {
    const m = new THREE.Matrix4().lookAt(cam.pos, cam.look, new THREE.Vector3(0, 1, 0));
    return m.setPosition(cam.pos);
  }

  update(t, camAt) {
    const T = this.T;
    this.updatePage(t);
    const inOpening = t < T.end;
    const fk = inOpening ? smooth(seg(t, T.fog0, T.fog1)) : 1;
    this.E.scene.fog.near = lerp(9, this.fog0.near, fk);
    this.E.scene.fog.far = lerp(30, this.fog0.far, fk);
    const sa = smooth(seg(t, T.swapT, T.swapT + 0.8)) * (1 - smooth(seg(t, T.dock0 - 0.8, T.enter0 - 0.4)));
    this.streaks.visible = sa > 0.01;
    this.streaks.material.opacity = 0.45 * sa;
    const show = t >= T.swapT - 0.02 && t < T.enter0;
    this.tiles.forEach((g) => { g.visible = false; });
    this.photo.visible = false;
    if (!show) return;
    this.tileMat.emissiveIntensity = 0.07 + 0.2 * (1 - smooth(seg(t, T.swapT, T.swapT + 0.6)));
    const L = this.swapLocal(camAt(T.swapT), t);
    const LI = this.localImg;
    const dock0 = T.dock0;   // 离开编队、飞向取景窗
    const n = this.user.length;
    // 跟随编队（镜头坐标系）：照片在上、词元在下
    const formAt = (loc0, target, tt, k) => {
      const c = easeInOut(Math.min(1, seg(tt, T.swapT, dock0) * 3));
      const l = loc0.clone().lerp(target, c);
      l.y += Math.sin(tt * 2.1 + k * 1.3) * 0.04 * c;
      return l.applyMatrix4(Opening.basis(camAt(tt)));
    };
    const imgTarget = v3(0, 0.55, -6.2);
    const fly = seg(t, T.swapT, dock0);
    // 照片
    {
      const a = t < dock0 ? formAt(LI.local, imgTarget, t, 0) : null;
      let p, sc;
      const s0 = lerp(LI.scale, 0.95, easeInOut(Math.min(1, fly * 3)));
      if (t < dock0) { p = a; sc = s0; }
      else {
        const from = formAt(LI.local, imgTarget, dock0, 0);
        const u = seg(t, dock0, T.enter0), q = 1 - (1 - u) ** 2.2;   // 先快后慢：始终跑在镜头前面
        p = from.clone().lerp(T.photoEnd, q);
        p.y += Math.sin(q * Math.PI) * 0.4;
        sc = lerp(0.95, 1, q);
      }
      this.photo.position.copy(p);
      this.photo.scale.setScalar(sc);
      const cm = camAt(t);
      // 面向镜头，进窗前转正
      const q2 = easeInOut(seg(t, dock0, T.enter0));
      const look = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(cm.pos, p, new THREE.Vector3(0, 1, 0)));
      this.photo.quaternion.copy(look).slerp(new THREE.Quaternion(), q2);
      this.photo.visible = true;
    }
    // 词元：编队里排在照片下面；进窗时跟着照片钻进去、缩小
    this.tiles.forEach((g, k) => {
      const tgt = v3((k - (n - 1) / 2) * 0.48, -0.95, -6.0);
      let p, sc = lerp(L[k].scale, 0.85, easeInOut(Math.min(1, fly * 3)));
      if (t < dock0 + k * 0.05) p = formAt(L[k].local, tgt, t, k + 1);
      else {
        const t0 = dock0 + k * 0.05;
        const from = formAt(L[k].local, tgt, t0, k + 1);
        const u = seg(t, t0, T.enter0 - 0.1), q = 1 - (1 - u) ** 2.2;
        const to = T.photoEnd.clone().add(v3((k - (n - 1) / 2) * 0.2, -0.6, -0.4));
        p = from.clone().lerp(to, q);
        sc *= 1 - 0.75 * q;
      }
      g.position.copy(p);
      g.scale.setScalar(sc);
      const wob = smooth(seg(t, T.swapT, T.swapT + 0.8)) * (1 - seg(t, dock0, T.enter0));
      g.quaternion.copy(this.photo.quaternion);
      g.rotateY(Math.sin(t * 1.7 + k) * 0.18 * wob);
      g.visible = t < T.enter0 - 0.1;
    });
  }
}
