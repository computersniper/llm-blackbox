// 开场：一个干净的 AI 聊天页（深色，沿用网站的配色和字体气质；不模仿任何真实产品，助手就叫 Qwen3-0.6B）。
// 在输入框里用拼音逐字打出问题（输入法候选一闪而过）→ 按发送 → 问题变成消息气泡、助手“正在思考…”
// → 镜头推近气泡，文字按真实分词裂成词元块 → 界面散去，词元块换成 3D 方块，镜头跟着它们飞向远处的黑箱；
// 黑箱打开，它们落进词元托盘，聊天模板的其余词元从两边一圈圈落下来，把问题包在中间。
// 全部由时间 t 决定（score.js 里的 OPEN 时间表），逐帧渲染完全确定。
import { THREE } from './engine.js';
import { clamp, lerp, seg, smooth, easeOut, easeInOut, v3 } from './cam.js';
import { textTexture } from '/public/js/stage/engine.js';
import { RoundedBoxGeometry } from '/public/js/vendor/three/addons/geometries/RoundedBoxGeometry.js';
import { tokPlain, esc } from '/public/js/ui.js';

const el = (tag, cls, html, parent) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; if (parent) parent.append(e); return e; };
const TILE_GEO = new RoundedBoxGeometry(0.36, 0.24, 0.36, 3, 0.045);
const FACE_GEO = new THREE.PlaneGeometry(0.33, 0.21);
const SEND_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5.5M5.8 11.4L12 5.2l6.2 6.2" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const mix = (a, b, k) => a.map((x, i) => Math.round(lerp(x, b[i], k)));
const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a.toFixed(3)})`;
const INK = [233, 239, 249], TOK = [196, 244, 249], DIM = [122, 133, 158], ACC = [94, 240, 212];

// 拼音按音节加分隔符（tian'kong）；只显示已经敲出来的字母
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
  constructor({ E, M, Q, frame, T }) {
    this.E = E; this.M = M; this.Q = Q; this.T = T;
    this.user = Q.tokens.map((t, i) => ({ ...t, i })).filter((t) => t.role === 'user' && !t.sp && t.s !== 'user' && t.s !== '\n');
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
    page.innerHTML = `
      <div class="cp-model"><span class="cp-av"><i></i></span><b>Qwen3-0.6B</b></div>
      <header class="cp-head"><div class="t">AI 的一个字是怎么思考出来的</div><div class="s">走进大模型推理的“黑箱”</div></header>
      <div class="cp-rule"></div>
      <div class="cp-hello"><span class="cp-av"><i></i></span><div class="h">有什么想问的？</div><div class="s">Qwen3-0.6B · 阿里开源的 6 亿参数模型</div></div>
      <div class="cp-user"><div class="cp-bub">${toks}</div></div>
      <div class="cp-bot"><span class="cp-av"><i></i></span><div><div class="n">Qwen3-0.6B</div><div class="th"><span class="tx">正在思考</span><span class="dots"><i></i><i></i><i></i></span></div></div></div>
      <div class="cp-box"><div class="cp-in"><span class="cp-txt"></span><span class="cp-comp"></span><span class="cp-caret"></span><span class="cp-ph">给 Qwen3-0.6B 发消息</span></div><span class="cp-send">${SEND_SVG}</span></div>
      <div class="cp-hint">回答来自真实模型的离线运行 · Enter 发送</div>
      <div class="cp-ime"><div class="py"></div><div class="cs"></div></div>`;
    const q = (s) => page.querySelector(s);
    Object.assign(this, {
      model: q('.cp-model'), head: q('.cp-head'), rule: q('.cp-rule'), hello: q('.cp-hello'),
      userEl: q('.cp-user'), bub: q('.cp-bub'), bot: q('.cp-bot'), botAv: q('.cp-bot .cp-av i'), thTx: q('.cp-bot .tx'), dots: [...page.querySelectorAll('.cp-bot .dots i')],
      box: q('.cp-box'), inp: q('.cp-in'), txt: q('.cp-txt'), comp: q('.cp-comp'), caret: q('.cp-caret'), ph: q('.cp-ph'), send: q('.cp-send'),
      hint: q('.cp-hint'), ime: q('.cp-ime'), imePy: q('.cp-ime .py'), imeCs: q('.cp-ime .cs'),
      avs: [q('.cp-model .cp-av i'), q('.cp-hello .cp-av i')],
    });
    this.toks = [...this.bub.querySelectorAll('.tk')];
    this.imeKey = '';
  }

  // 输入框此刻的内容：已上屏的字 + 正在拼的拼音 + 输入法候选
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
        // 拼音敲完那一下，候选条一闪：选中第一个，上屏
        if (t >= kl - 0.03) ime = { w, a: smooth(seg(t, kl - 0.03, kl + 0.05)) };
      }
      break;
    }
    return { txt, comp, ime, last };
  }

  // 气泡中心（排版坐标，不含镜头推进的缩放）
  bubbleCenter() {
    const u = this.userEl, b = this.bub;
    return { x: u.offsetLeft + b.offsetLeft + b.offsetWidth / 2, y: u.offsetTop + b.offsetTop + b.offsetHeight / 2 };
  }

  updatePage(t) {
    const T = this.T;
    const on = t < T.swapT + 0.3;
    this.wrap.style.display = on ? 'block' : 'none';
    if (!on) return;
    this.wrap.style.opacity = smooth(seg(t, 0.05, 0.8)).toFixed(3);
    const sent = t >= T.sendT;

    // ---- 输入框
    const ty = this.typingAt(t);
    this.txt.textContent = ty.txt;
    this.comp.textContent = ty.comp;
    this.ph.style.display = ty.txt || ty.comp ? 'none' : '';
    const typingNow = !sent && ty.last >= 0 && t - ty.last < 0.55;
    const idle = sent ? t - T.sendT : t - Math.max(0, ty.last);
    this.caret.style.opacity = typingNow || (idle % 1.06) < 0.53 ? '1' : '0';
    const ready = !sent && !!ty.txt && !ty.comp;
    this.send.classList.toggle('on', ready || (t >= T.sendT && t < T.sendT + 0.12));
    const press = smooth(seg(t, T.sendT - 0.14, T.sendT - 0.03)) * (1 - smooth(seg(t, T.sendT + 0.03, T.sendT + 0.28)));
    const flash = Math.exp(-Math.max(0, t - T.sendT) / 0.25) * (t >= T.sendT ? 1 : 0);
    this.send.style.transform = `scale(${(1 - 0.1 * press).toFixed(4)})`;
    this.send.style.boxShadow = ready || flash > 0.02 ? `0 0 ${(28 + 40 * flash).toFixed(1)}px rgba(94, 240, 212, ${(0.4 + 0.45 * flash).toFixed(3)})` : 'none';
    // 发送后输入框从屏幕中间滑到底部（和常见的聊天页一样：开始对话以后输入框贴底）
    const mv = easeInOut(seg(t, T.sendT + 0.05, T.sendT + 0.65));
    const boxTop = lerp(T.boxY0, T.boxY1, mv);
    this.box.style.top = `${boxTop.toFixed(1)}px`;
    this.hint.style.top = `${(boxTop + 140).toFixed(1)}px`;
    this.hint.style.opacity = (1 - smooth(seg(t, T.sendT, T.sendT + 0.3))).toFixed(3);
    // 输入法候选条：贴在拼音下面
    if (ty.ime && ty.ime.a > 0.01) {
      const w = ty.ime.w;
      if (this.imeKey !== w.s) {
        this.imeKey = w.s;
        this.imePy.textContent = w.seg;
        this.imeCs.innerHTML = w.cands.map((c, i) => `<span class="${i ? '' : 'on'}"><b>${i + 1}</b>${esc(c)}</span>`).join('');
      }
      this.ime.style.display = 'block';
      this.ime.style.opacity = ty.ime.a.toFixed(3);
      this.ime.style.left = `${(this.box.offsetLeft + this.inp.offsetLeft + this.comp.offsetLeft - 16).toFixed(1)}px`;
      this.ime.style.top = `${(boxTop + 96 + 6 * (1 - ty.ime.a)).toFixed(1)}px`;
    } else this.ime.style.display = 'none';

    // ---- 页眉、欢迎语：发送后欢迎语让位；镜头开始推进时页眉淡出
    const headOut = smooth(seg(t, T.push0, T.push0 + 0.9));
    for (const e of [this.head, this.model, this.rule]) e.style.opacity = (1 - headOut).toFixed(3);
    const hi = smooth(seg(t, 0.25, 1.0)) * (1 - smooth(seg(t, T.sendT, T.sendT + 0.35)));
    this.hello.style.display = hi > 0.001 ? '' : 'none';
    this.hello.style.opacity = hi.toFixed(3);
    this.hello.style.transform = `translateX(-50%) translateY(${(-24 * smooth(seg(t, T.sendT, T.sendT + 0.35))).toFixed(1)}px)`;
    // 头像里的小灯在呼吸；助手思考时呼吸变快
    const breathe = (p) => 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / p);
    for (const i of this.avs) { const b = breathe(3); i.style.transform = `scale(${(1 - 0.4 * b).toFixed(3)})`; i.style.opacity = (1 - 0.4 * b).toFixed(3); }
    { const b = breathe(0.9); this.botAv.style.transform = `scale(${(1 - 0.45 * b).toFixed(3)})`; this.botAv.style.opacity = (1 - 0.4 * b).toFixed(3); }

    // ---- 消息：你的问题成为气泡；助手“正在思考…”
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

    // ---- 文字裂成词元块：字间拉开、每块长出边框；气泡本身慢慢化掉，只剩下词元
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
    const bubOut = smooth(seg(t, T.split0 + 0.3, T.split0 + 0.3 + T.splitD + T.splitGap * this.toks.length));
    this.bub.style.background = rgb([94, 228, 240], 0.1 * (1 - bubOut));
    this.bub.style.borderColor = rgb([94, 228, 240], 0.3 * (1 - bubOut));

    // ---- 镜头推近气泡；周围的界面虚掉（景深），最后散去，露出后面的 3D 空间
    const dof = smooth(seg(t, T.push0 + 0.5, T.push1));
    const diss = smooth(seg(t, T.diss0, T.diss1));
    for (const e of [this.box, this.bot]) {
      e.style.filter = dof + diss > 0.01 ? `blur(${(2.6 * dof + 9 * diss).toFixed(2)}px)` : '';
      e.style.opacity = (e === this.bot ? smooth(bb) : 1) * (1 - diss);
    }
    this.bg.style.opacity = (1 - diss).toFixed(3);
    const f = easeInOut(seg(t, T.push0, T.push1));
    const Z = lerp(1, T.zoom, f) * (1 + T.zoom2 * smooth(seg(t, T.push1, T.swapT + 0.3)));
    const bc = this.bubbleCenter();
    const F = { x: lerp(960, bc.x, f), y: lerp(540, bc.y, f) };
    const S = { x: 960, y: lerp(540, T.focusY, f) };
    this.page.style.transform = Z > 1.0001 ? `translate(${(S.x - Z * F.x).toFixed(2)}px, ${(S.y - Z * F.y).toFixed(2)}px) scale(${Z.toFixed(5)})` : '';
    this.drawSnow(t);
  }

  /* ------------------------------------------------------------ 背景：网站上那种缓慢上浮的“海雪”（闭式，不积累状态） */

  buildSnow() {
    let s = 20240;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    this.snow = Array.from({ length: 190 }, () => {
      const z = 0.2 + rnd() * 0.8;
      return { x0: rnd() * 1920, y0: rnd() * 1104, z, vy: -(4 + 14 * z) * 1.4, ph: rnd() * Math.PI * 2, w: 0.3 + 0.4 * z, r: (0.5 + z * 1.4) * 1.35 };
    });
    this.g = this.bg.getContext('2d');
  }

  // 发送以后的“跃迁”：粒子沿径向拉成光线（网站揭开黑箱时的同款效果）
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

  /* ------------------------------------------------------------ 飞行的词元（3D） */

  buildTiles() {
    const mat = (this.tileMat = this.M.roleMat.user.clone());
    mat.emissiveIntensity = 0.22;
    this.tiles = this.user.map((t) => {
      const g = new THREE.Group();
      g.add(new THREE.Mesh(TILE_GEO, mat));
      const face = new THREE.Mesh(FACE_GEO, new THREE.MeshBasicMaterial({ map: textTexture(tokPlain(t.s), { color: '#5ee4f0' }), transparent: true }));
      face.position.z = 0.181;
      g.add(face);
      const top = face.clone();
      top.rotation.x = -Math.PI / 2;
      top.position.set(0, 0.121, 0);
      g.add(top);
      g.visible = false;
      this.M.root.add(g);
      return g;
    });
  }

  // 一路上掠过的细碎光点：给穿越的速度感（位置固定，确定的伪随机）
  buildStreaks() {
    let s = 12345;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    const n = 700, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos[i * 3] = (rnd() - 0.5) * 60;
      pos[i * 3 + 1] = rnd() * 22 - 4;
      pos[i * 3 + 2] = 16 + rnd() * 120;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.streaks = new THREE.Points(g, new THREE.PointsMaterial({ color: 0x9fe8ff, size: 0.07, transparent: true, opacity: 0, depthWrite: false }));
    this.E.scene.add(this.streaks);
  }

  // 交接那一刻：屏幕上第 k 个词元块 → 镜头前 d 处的位置（镜头坐标系）和缩放，让 3D 方块正好盖住它
  swapLocal(cam0, t) {
    if (this.local) return this.local;
    this.updatePage(this.T.swapT);
    const fr = document.querySelector('#frame').getBoundingClientRect();
    const rects = this.toks.map((s) => { const r = s.getBoundingClientRect(); return { x: r.left - fr.left + r.width / 2, y: r.top - fr.top + r.height / 2, w: r.width, h: r.height }; });
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
    this.local = rects.map((r) => {
      const ndc = v3((r.x / this.E.w) * 2 - 1, -((r.y / this.E.h) * 2 - 1), 0.5).unproject(c);
      const dir = ndc.sub(cam0.pos).normalize();
      const p = cam0.pos.clone().add(dir.multiplyScalar(d / dir.dot(fwd)));
      return { local: p.applyMatrix4(inv), scale: Math.min((r.h * 1.08) / 0.24, (r.w * 1.0) / 0.36) * worldPerPx };
    });
    return this.local;
  }

  // 镜头坐标系（右、上、后）→ 世界
  static basis(cam) {
    const m = new THREE.Matrix4().lookAt(cam.pos, cam.look, new THREE.Vector3(0, 1, 0));
    return m.setPosition(cam.pos);
  }

  update(t, camAt) {
    const T = this.T;
    this.updatePage(t);
    const M = this.M;
    const inOpening = t < T.end;
    // 雾：界面散去之前，远处的黑箱藏在黑暗里；词元飞起来以后再慢慢显出来
    const fk = inOpening ? smooth(seg(t, T.fog0, T.fog1)) : 1;
    this.E.scene.fog.near = lerp(9, this.fog0.near, fk);
    this.E.scene.fog.far = lerp(30, this.fog0.far, fk);
    // 词元托盘上的方块：自己的落下时间到了才出现（机器本身这时处在“读入”这一步、所有方块都已就位）
    if (inOpening && t >= T.boxOpen0) {
      M.tiles.forEach((tile, i) => {
        const t1 = T.landAt[i];
        if (t1 == null) return;
        const k = clamp((t - (t1 - T.dropD)) / T.dropD);
        const user = this.user.some((u) => u.i === i);
        if (user) { tile.visible = t >= t1; if (tile.visible) { const b = seg(t, t1, t1 + 0.25); tile.scale.set(1, 1 - 0.25 * Math.sin(Math.PI * b) * (1 - b), 1); tile.position.y = 0.12; } return; }
        tile.visible = k > 0;
        if (k > 0) { const e = easeOut(k); tile.position.y = 0.12 + (1 - e) * 1.3; tile.scale.setScalar(0.35 + 0.65 * e); }
      });
    }
    // 速度光点
    const sa = smooth(seg(t, T.swapT, T.swapT + 0.8)) * (1 - smooth(seg(t, T.boxOpen0, T.boxOpen0 + 1.6)));
    this.streaks.visible = sa > 0.01;
    this.streaks.material.opacity = 0.45 * sa;
    // 飞行的 6 个词元
    const show = t >= T.swapT - 0.02 && t < T.landAt[this.user[this.user.length - 1].i] + 0.05;
    this.tiles.forEach((g) => { g.visible = false; });
    if (!show) return;
    // 交接时亮一下（泛光），之后回到正常亮度
    this.tileMat.emissiveIntensity = 0.22 + 0.3 * (1 - smooth(seg(t, T.swapT, T.swapT + 0.6)));
    const L = this.swapLocal(camAt(T.swapT), t);
    const fly = seg(t, T.swapT, T.landStart0);
    this.tiles.forEach((g, k) => {
      const u = this.user[k];
      const tl0 = T.landStart0 + k * T.landGap, tl1 = T.landAt[u.i];
      if (t >= tl1) return;
      // 跟随编队：在镜头坐标系里慢慢收拢、稍微往前，各自有一点上下浮动和摆动
      const formAt = (tt) => {
        const c = easeInOut(Math.min(1, seg(tt, T.swapT, T.landStart0) * 3));
        const n = this.user.length;
        const l = L[k].local.clone().lerp(v3((k - (n - 1) / 2) * 0.55, -0.75, -5.6), c);
        l.y += Math.sin(tt * 2.1 + k * 1.3) * 0.045 * c;
        l.x += Math.sin(tt * 1.4 + k * 0.7) * 0.02 * c;
        return l.applyMatrix4(Opening.basis(camAt(tt)));
      };
      let p;
      if (t < tl0) p = formAt(t);
      else {
        // 落进托盘：从编队里的位置画一道弧线落到自己的格子
        const a = formAt(tl0), b = v3(M.x(u.i), 0.12, 0);
        const ctrl = v3((a.x + b.x) / 2, Math.max(a.y, b.y) + 0.8, (a.z + b.z) / 2);
        const q = easeInOut(seg(t, tl0, tl1));
        p = new THREE.QuadraticBezierCurve3(a, ctrl, b).getPoint(q);
      }
      g.position.copy(p);
      const sc = lerp(L[k].scale, 1, easeInOut(Math.min(1, fly * 3)));
      g.scale.setScalar(sc);
      const wob = smooth(seg(t, T.swapT, T.swapT + 0.8)) * (1 - seg(t, tl0, tl1));
      g.rotation.set(0, Math.sin(t * 1.7 + k) * 0.18 * wob, Math.sin(t * 1.3 + k * 2) * 0.06 * wob);
      g.visible = true;
    });
  }
}
