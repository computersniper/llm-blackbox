// 开场：一个和网站推理页同款的 AI 对话输入框。问题一个词元一个词元地点进输入框，按下发送，
// 输入框里的词元块升起来、变成 3D 方块，镜头跟着它们穿过界面、飞向远处的黑箱；
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

export class Opening {
  constructor({ E, M, Q, ov, T }) {
    this.E = E; this.M = M; this.Q = Q; this.T = T;
    this.user = Q.tokens.map((t, i) => ({ ...t, i })).filter((t) => t.role === 'user' && !t.sp && t.s !== 'user' && t.s !== '\n');
    this.buildDOM(ov);
    this.buildTiles();
    this.buildStreaks();
  }

  /* ------------------------------------------------------------ 对话界面（DOM） */

  buildDOM(ov) {
    const ui = (this.ui = el('div', 'chatui', '', ov));
    ui.innerHTML = `
      <div class="ch-head"><span class="ch-av"><i></i></span><div class="ch-t"><b>Qwen3-0.6B</b><span>阿里开源的 6 亿参数模型 · 真实输出，离线录制</span></div></div>
      <div class="ch-hello">有什么想问的？用下面的词元拼出你的问题。</div>
      <div class="ch-row">
        <span class="ch-plus">＋</span>
        <div class="ch-input"><span class="ch-ph">点选下面的词元，拼出一个问题…</span></div>
        <span class="ch-back">⌫</span>
        <span class="ch-send">↑</span>
      </div>
      <div class="ch-cands"><span class="lab">候选</span></div>`;
    this.input = ui.querySelector('.ch-input');
    this.ph = ui.querySelector('.ch-ph');
    this.send = ui.querySelector('.ch-send');
    const cands = ui.querySelector('.ch-cands');
    // 候选：问题里的 6 个词元（真实编号）+ 几个别的问题里的开头，像网站的点选输入法
    // 干扰项取自其他预设问题的第一个词元（真实编号）
    const decoy = ['q07', 'q12', 'q15', 'q13'].map((id) => this.Q.manifest.questions.find((x) => x.id === id)?.chips[0]).filter(Boolean).map((c) => [c.s, c.id]);
    const list = [...this.user.map((t) => [t.s, t.id, true]), ...decoy.map(([s, id]) => [s, id, false])];
    const order = [0, 6, 1, 7, 2, 3, 8, 4, 5, 9].filter((k) => k < list.length);
    this.cands = [];
    for (const k of order) {
      const [s, id, real] = list[k];
      const c = el('span', 'ch-cand', `${esc(tokPlain(s))}<i>${id}</i>`, cands);
      if (real) this.cands[k] = c;
    }
    this.chips = this.user.map((t) => {
      const c = el('span', 'ch-tok', esc(tokPlain(t.s)), this.input);
      c.style.display = 'none';
      return c;
    });
    this.caret = el('span', 'ch-caret', '', this.input);
    // 量好每个词元块在发送那一刻的位置：之后 3D 方块从同一个位置接过去
    for (const c of this.chips) c.style.display = 'inline-flex';
    this.ph.style.display = 'none';
    const frame = document.querySelector('#frame').getBoundingClientRect();
    this.chipRects = this.chips.map((c) => { const r = c.getBoundingClientRect(); return { x: r.left - frame.left + r.width / 2, y: r.top - frame.top + r.height / 2, w: r.width, h: r.height }; });
    for (const c of this.chips) c.style.display = 'none';
    this.ph.style.display = '';
  }

  updateDOM(t) {
    const T = this.T;
    const a = smooth(seg(t, 0.15, 0.9));
    const out = smooth(seg(t, T.uiOut0, T.uiOut1)); // 镜头穿过界面
    this.ui.style.display = a > 0.001 && out < 0.999 ? 'block' : 'none';
    if (this.ui.style.display === 'none') return;
    this.ui.style.opacity = (a * (1 - out)).toFixed(3);
    this.ui.style.transform = `translate(-50%, -50%) translateY(${((1 - easeOut(seg(t, 0.15, 1.2))) * 18).toFixed(1)}px) scale(${(1 + 0.9 * out * out).toFixed(4)})`;
    this.ui.style.filter = out > 0.01 ? `blur(${(out * 10).toFixed(2)}px)` : '';
    const n = T.popT.filter((x) => t >= x).length;
    this.ph.style.display = n ? 'none' : '';
    this.chips.forEach((c, k) => {
      const k0 = T.popT[k];
      const on = t >= k0;
      c.style.display = on ? 'inline-flex' : 'none';
      if (!on) return;
      const p = clamp((t - k0) / 0.35);
      // 弹入：和网站一样从上面落下、略微回弹
      const y = p < 0.6 ? lerp(-14, 2, easeOut(p / 0.6)) : lerp(2, 0, (p - 0.6) / 0.4);
      const sc = p < 0.6 ? lerp(0.6, 1.06, easeOut(p / 0.6)) : lerp(1.06, 1, (p - 0.6) / 0.4);
      // 发送后升起；交给 3D 方块的那一刻淡出
      const lift = easeInOut(seg(t, T.liftT0, T.swapT));
      const gone = smooth(seg(t, T.swapT - 0.02, T.swapT + 0.16));
      c.style.opacity = (smooth(p * 1.6) * (1 - gone)).toFixed(3);
      c.style.transform = `translateY(${(y - lift * T.lift).toFixed(2)}px) scale(${(sc * (1 + 0.08 * lift)).toFixed(4)})`;
      c.classList.toggle('lit', t >= T.sendT);
    });
    // 候选：被点中的那一下亮起来，之后标成已用
    this.cands.forEach((c, k) => {
      if (!c) return;
      const k0 = T.popT[k];
      const press = smooth(seg(t, k0 - 0.18, k0 - 0.05)) * (1 - smooth(seg(t, k0 + 0.05, k0 + 0.3)));
      c.classList.toggle('done', t >= k0);
      c.style.transform = `translateY(${(-2 * press).toFixed(2)}px) scale(${(1 - 0.06 * press).toFixed(4)})`;
      c.style.background = press > 0.02 ? `rgba(94,240,212,${(0.22 * press).toFixed(3)})` : '';
    });
    const typing = t < T.popT[T.popT.length - 1] + 0.6;
    this.caret.style.opacity = t > T.liftT0 ? '0' : typing || Math.floor(t * 1.7) % 2 === 0 ? '1' : '0';
    // 发送键：问题拼完后亮起，按下时压一下
    const ready = smooth(seg(t, T.popT[T.popT.length - 1] + 0.2, T.popT[T.popT.length - 1] + 0.6));
    const press = smooth(seg(t, T.sendT - 0.16, T.sendT - 0.02)) * (1 - smooth(seg(t, T.sendT + 0.04, T.sendT + 0.3)));
    this.send.style.opacity = (0.22 + 0.78 * ready).toFixed(3);
    this.send.style.transform = `scale(${(1 - 0.12 * press).toFixed(4)})`;
    this.send.style.boxShadow = `0 0 ${(14 + 30 * press + 10 * ready).toFixed(1)}px rgba(94, 240, 212, ${(0.15 + 0.35 * ready + 0.4 * press).toFixed(3)})`;
  }

  /* ------------------------------------------------------------ 飞行的词元（3D） */

  buildTiles() {
    const mat = this.M.roleMat.user.clone();
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

  // 发送那一刻：屏幕上第 k 个词元块的中心 → 镜头前 d 处的世界坐标，和镜头坐标系下的偏移
  swapLocal(cam0) {
    if (this.local) return this.local;
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
    this.local = this.chipRects.map((r) => {
      const ndc = v3((r.x / this.E.w) * 2 - 1, -(((r.y - this.T.lift) / this.E.h) * 2 - 1), 0.5).unproject(c);
      const dir = ndc.sub(cam0.pos).normalize();
      const p = cam0.pos.clone().add(dir.multiplyScalar(d / dir.dot(fwd)));
      return { local: p.applyMatrix4(inv), scale: (r.h * 1.15 * worldPerPx) / 0.24 };
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
    this.updateDOM(t);
    const M = this.M;
    // 词元托盘上的方块：自己的落下时间到了才出现（机器本身这时处在“读入”这一步、所有方块都已就位）
    const inOpening = t < T.end;
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
    const L = this.swapLocal(camAt(T.swapT));
    const fly = seg(t, T.swapT, T.landStart0);
    this.tiles.forEach((g, k) => {
      const u = this.user[k];
      const tl0 = T.landStart0 + k * T.landGap, tl1 = T.landAt[u.i];
      if (t >= tl1) return;
      // 跟随编队：在镜头坐标系里慢慢收拢、稍微往前，各自有一点上下浮动和摆动
      const formAt = (tt) => {
        const c = easeInOut(Math.min(1, seg(tt, T.swapT, T.landStart0) * 1.5));
        const n = this.user.length;
        const l = L[k].local.clone().lerp(v3((k - (n - 1) / 2) * 0.5, -0.75, -5.6), c);
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
      const sc = lerp(L[k].scale, 1, smooth(fly));
      g.scale.setScalar(sc);
      g.rotation.set(0, Math.sin(t * 1.7 + k) * 0.18 * (1 - seg(t, tl0, tl1)), Math.sin(t * 1.3 + k * 2) * 0.06 * (1 - seg(t, tl0, tl1)));
      g.visible = true;
    });
  }
}
