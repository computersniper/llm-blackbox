// 最深的几层：SwiGLU 的 3072 个神经元、单个神经元的真实乘加、一次 Q·K 打分、权重的 bf16 比特。
import { THREE, label, textTexture, easeOut, seg } from './engine.js';
import { RoundedBoxGeometry } from '../vendor/three/addons/geometries/RoundedBoxGeometry.js';
import { esc } from '../ui.js';
import { bf16Bits } from '../num.js';
import { neuronId } from './fields.js';
import { termAt, ROLE } from './micro.js';
import { L as tr } from '../i18n.js';

const COLS = 64, ROWS = 48, CELL = 0.05;
const AMBER = new THREE.Color(0xffb65c), BLUE = new THREE.Color(0x6b9bff), CYAN = new THREE.Color(0x5ef0d4), VIOLET = new THREE.Color(0xb39dff);
const DIM = new THREE.Color(0x0d1424);
const ROLE_X = new THREE.Color(ROLE.x), ROLE_W = new THREE.Color(ROLE.w);
// 比特的三组：符号（玫红）/ 指数（琥珀）/ 尾数（青），和算式板一致
const BIT_C = [new THREE.Color(0xff6b93), AMBER, CYAN];
const WHITE = new THREE.Color(0xffffff);
const tmpM = new THREE.Matrix4();
const tmpC = new THREE.Color();
const v3 = (x, y, z) => new THREE.Vector3(x, y, z);

function dispose(o) {
  o.traverse((c) => {
    c.geometry?.dispose();
    if (c.material) (Array.isArray(c.material) ? c.material : [c.material]).forEach((m) => m.dispose());
    if (c.el) c.el.remove();
  });
}

function signColor(v, mag) {
  const c = (v >= 0 ? AMBER : BLUE).clone();
  return c.multiplyScalar(0.25 + 0.75 * Math.min(1, mag));
}

export class Detail {
  constructor(E) {
    this.E = E;
    this.root = new THREE.Group();
    E.scene.add(this.root);
    this.bitsIndex = 0;
    this.flips = new Map();
  }

  load(Q, M) {
    dispose(this.root);
    this.root.clear();
    this.Q = Q;
    this.M = M;
    this.panelKey = this.rigKey = this.dotKey = this.bitsKey = '';
    this.flips.clear();

    // 神经元阵列
    const panel = (this.panel = new THREE.Group());
    this.root.add(panel);
    const back = new THREE.Mesh(new THREE.PlaneGeometry(COLS * CELL + 0.3, ROWS * CELL + 0.3), new THREE.MeshStandardMaterial({ color: 0x060b16, roughness: 0.6, metalness: 0.3, transparent: true, opacity: 0.92 }));
    back.position.z = -0.03;
    panel.add(back);
    panel.add(new THREE.LineSegments(new THREE.EdgesGeometry(back.geometry), new THREE.LineBasicMaterial({ color: 0xffb65c, transparent: true, opacity: 0.5 })));
    this.bulbs = new THREE.InstancedMesh(new THREE.SphereGeometry(0.016, 8, 6), new THREE.MeshBasicMaterial({ color: 0xffffff }), COLS * ROWS);
    for (let n = 0; n < COLS * ROWS; n++) {
      tmpM.makeTranslation((n % COLS - (COLS - 1) / 2) * CELL, ((ROWS - 1) / 2 - Math.floor(n / COLS)) * CELL, 0);
      this.bulbs.setMatrixAt(n, tmpM);
      this.bulbs.setColorAt(n, DIM);
    }
    this.bulbs.userData.pick = (hit) => ({ type: 'bulb', n: hit.instanceId, v: this.acts ? this.acts[hit.instanceId] : 0, full: this.fullLayer });
    this.E.pickables.push(this.bulbs);
    panel.add(this.bulbs);
    this.panelTitle = label('', 'lbl part');
    this.panelTitle.position.set(-(COLS * CELL) / 2, (ROWS * CELL) / 2 + 0.16, 0);
    this.panelTitle.center.set(0, 0.5);
    panel.add(this.panelTitle);
    this.hot = new THREE.Mesh(new THREE.RingGeometry(0.03, 0.04, 24), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true }));
    panel.add(this.hot);
    this.fanG = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }));
    this.fanU = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: 0xb39dff, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }));
    this.root.add(this.fanG, this.fanU);
    panel.visible = this.fanG.visible = this.fanU.visible = false;

    this.rig = new THREE.Group();
    this.dot = new THREE.Group();
    this.bits = new THREE.Group();
    this.root.add(this.rig, this.dot, this.bits);
  }

  /* ------------------------------------------------------------ 位置 */

  base() { return 0.95 + this.M.explodeL * 0.26; }
  panelPos(st) { return v3(this.M.xFocus(st), this.base() + 1.78 * this.M.e + 1.45, -1.1); }
  bulbPos(st, n) {
    const p = this.panelPos(st);
    return p.add(v3((n % COLS - (COLS - 1) / 2) * CELL, ((ROWS - 1) / 2 - Math.floor(n / COLS)) * CELL, 0));
  }

  /* ------------------------------------------------------------ 每帧 */

  update(st, dt, t) {
    const v = st.view, s = st.step;
    const inMlp = s.ph === 'layer' && s.op === 'mlp' && (v === 'mlp' || v === 'neuron');
    this.updatePanel(st, inMlp, t);
    this.updateRig(st, v === 'neuron', t);
    this.updateDot(st, v === 'dot' || (v === 'bits' && s.sub === 'score'), t);
    this.updateBits(st, v === 'bits');
  }

  updatePanel(st, show, t) {
    const Q = this.Q, s = st.step;
    this.panel.visible = show && this.M.e > 0.5;
    this.fanG.visible = this.fanU.visible = false;
    if (!this.panel.visible) return;
    const L = s.L, g = st.g;
    const pp = this.panelPos(st);
    this.panel.position.copy(pp);
    const key = `${g}|${L}`;
    if (key !== this.panelKey) {
      this.panelKey = key;
      this.fullLayer = L === Q.manifest.fullMlpLayer;
      const acts = new Float32Array(COLS * ROWS);
      if (this.fullLayer) acts.set(Q.mlpFull(g));
      else for (const { n, v } of Q.mlpTop(g, L)) acts[n] = v;
      this.acts = acts;
      let mx = 0;
      for (const a of acts) mx = Math.max(mx, Math.abs(a));
      this.maxAct = mx || 1;
      this.order = Float32Array.from({ length: acts.length }, () => Math.random());
      this.topN = acts.reduce((bi, a, i) => (a > acts[bi] ? i : bi), 0);
      this.panelTitle.el.innerHTML = tr(`第 ${L} 层 · SwiGLU 的 3072 个神经元<small>${this.fullLayer ? '完整导出' : '只导出了最亮的 16 个'} · 明显激活 ${Q.mlpCount(g, L)} 个</small>`, `Layer ${L} · the 3072 SwiGLU neurons<small>${this.fullLayer ? 'fully exported' : 'only the 16 brightest exported'} · ${Q.mlpCount(g, L)} clearly active</small>`);
      this.litState = -1;
    }
    // 升维：两把“扇子”从残差流展开到阵列；激活：灯泡逐个点亮；降维：光收回残差流
    const sub = s.sub || 'act', p = st.p;
    let lit = 1, fan = 0;
    if (sub === 'up') { lit = 0; fan = easeOut(p); }
    else if (sub === 'act') lit = s.mi ? 1 : easeOut(seg(p, 0.05, 0.85));
    else if (sub === 'down') { lit = 1 - 0.7 * easeOut(p); fan = 1 - easeOut(p); }
    const dimK = st.view === 'mlp' ? 1 : 0.08;
    const litKey = Math.round(lit * 60) + (dimK < 1 ? 1000 : 0);
    this.dimK = dimK;
    if (litKey !== this.litState) {
      this.litState = litKey;
      for (let n = 0; n < this.acts.length; n++) {
        const a = this.acts[n], m = Math.abs(a) / this.maxAct;
        if (this.order[n] > lit || m < 0.02) { this.bulbs.setColorAt(n, DIM); continue; }
        tmpC.copy(a >= 0 ? AMBER : BLUE).lerp(new THREE.Color(0xffffff), Math.max(0, m - 0.6)).multiplyScalar((0.12 + Math.pow(m, 0.7) * 1.45) * this.dimK);
        this.bulbs.setColorAt(n, tmpC);
      }
      this.bulbs.instanceColor.needsUpdate = true;
    }
    const hp = this.bulbPos(st, this.topN).sub(pp);
    this.hot.position.set(hp.x, hp.y, 0.01);
    this.hot.material.opacity = lit > 0.9 ? 0.6 + 0.4 * Math.sin(t * 5) : 0;
    if (fan > 0.01 && false) { // 投影已经由 W_gate / W_up 面板表示，扇形光不再画
      const src = v3(this.M.xFocus(st), this.base() + 1.78 * this.M.e + 0.11, 0);
      const w = (COLS * CELL) / 2, h = (ROWS * CELL) / 2;
      const mk = (mesh, yOff) => {
        const pts = [src, pp.clone().add(v3(-w * fan, h * fan + yOff, 0)), pp.clone().add(v3(w * fan, h * fan + yOff, 0)), pp.clone().add(v3(w * fan, -h * fan + yOff, 0)), pp.clone().add(v3(-w * fan, -h * fan + yOff, 0))];
        const pos = [];
        for (let k = 1; k <= 4; k++) { const a = pts[k], b = pts[k === 4 ? 1 : k + 1]; pos.push(src.x, src.y, src.z, a.x, a.y, a.z, b.x, b.y, b.z); }
        mesh.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        mesh.geometry.computeBoundingSphere();
        mesh.visible = true;
      };
      mk(this.fanG, 0.02);
      mk(this.fanU, -0.02);
    }
  }

  /* ------------------------------------------------------------ 单个神经元（SwiGLU） */

  buildRig(n) {
    dispose(this.rig);
    this.rig.clear();
    const R = this.rig;
    const maxC = Math.max(...n.x.map((x, k) => Math.abs(x * n.wg[k])));
    this.pins = [];
    this.wires = [];
    const gateP = v3(0, 0.8, 0), upP = v3(0, -0.8, 0);
    n.x.forEach((x, k) => {
      const y = 1.65 - k * 0.3;
      const pin = new THREE.Mesh(new RoundedBoxGeometry(0.2, 0.15, 0.15, 2, 0.03), new THREE.MeshStandardMaterial({ color: 0x0b1222, emissive: x >= 0 ? AMBER : BLUE, emissiveIntensity: 0.2 + Math.min(1, Math.abs(x) / 3) * 0.9 }));
      pin.position.set(-2.1, y, 0);
      pin.userData.pick = { type: 'pin', k, dim: n.dims[k], x, wg: n.wg[k], wu: n.wu[k], click: true };
      this.E.pickables.push(pin);
      R.add(pin);
      pin.visible = false;
      const wire = (to, w, c) => {
        const prod = x * w;
        const curve = new THREE.QuadraticBezierCurve3(v3(-2.0, y, 0), v3(-1.0, (y + to.y) / 2, 0.15), to);
        const geo = new THREE.TubeGeometry(curve, 20, 0.004 + Math.min(0.03, Math.abs(w) * 0.4), 5, false);
        const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: signColor(prod, Math.abs(prod) / maxC), transparent: true, opacity: 0.2 + 0.8 * Math.min(1, Math.abs(prod) / maxC), blending: THREE.AdditiveBlending, depthWrite: false }));
        m.userData.pick = { type: 'wire', k, w, x, prod, which: c };
        this.E.pickables.push(m);
        R.add(m);
        return m;
      };
      this.wires.push({ g: wire(gateP, n.wg[k], 'gate'), u: wire(upP, n.wu[k], 'up'), k });
      this.pins.push(pin);
    });
    const node = (p, c, txt) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.13, 24, 16), new THREE.MeshStandardMaterial({ color: 0x0b1222, emissive: c, emissiveIntensity: 0.6 }));
      m.position.copy(p);
      R.add(m);
      const lb = label(txt, 'lbl part');
      lb.position.copy(p).add(v3(0, 0.25, 0));
      R.add(lb);
      return m;
    };
    this.gateNode = node(gateP, ROLE_X, '<span class="cx">g</span> = Σ h·w<sub>gate</sub>');
    this.upNode = node(upP, ROLE_W, '<span class="cw">u</span> = Σ h·w<sub>up</sub>');
    // 两个“水箱”：累加的结果有多大，就灌多满
    const tank = (p, c) => {
      const glass = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 1.1, 24, 1, true), new THREE.MeshStandardMaterial({ color: 0x88aaff, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }));
      glass.position.copy(p);
      const fillGeo = new THREE.CylinderGeometry(0.145, 0.145, 1, 24);
      fillGeo.translate(0, 0.5, 0);
      const fill = new THREE.Mesh(fillGeo, new THREE.MeshStandardMaterial({ color: c.clone().multiplyScalar(0.4), emissive: c, emissiveIntensity: 0.5, transparent: true, opacity: 0.9 }));
      fill.position.copy(p).add(v3(0, -0.55, 0));
      fill.scale.y = 0.001;
      const lb = label('', 'lbl big');
      lb.position.copy(p).add(v3(0, 0.75, 0));
      R.add(glass, fill, lb);
      return { glass, fill, lb };
    };
    this.tG = tank(v3(0.95, 0.8, 0), ROLE_X);
    this.tU = tank(v3(0.95, -0.8, 0), ROLE_W);
    // SiLU 曲线：真实的 x·σ(x)
    const px = (x) => 1.75 + ((x + 5) / 10) * 1.4, py = (y) => 0.35 + (y / 5) * 1.2 + 0.18;
    const pts = [];
    for (let i = 0; i <= 80; i++) { const x = -5 + i / 8; pts.push(v3(px(x), py(x / (1 + Math.exp(-x))), 0)); }
    const curve = new THREE.CatmullRomCurve3(pts);
    R.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 120, 0.018, 6, false), new THREE.MeshBasicMaterial({ color: 0x5ef0d4 })));
    const axis = new THREE.BufferGeometry().setFromPoints([v3(px(-5), py(0), 0), v3(px(5), py(0), 0), v3(px(0), py(-0.3), 0), v3(px(0), py(5), 0)]);
    R.add(new THREE.LineSegments(axis, new THREE.LineBasicMaterial({ color: 0x4b5572 })));
    const sl = label('SiLU(g) = g·σ(g)', 'lbl hint');
    sl.position.set(px(0), py(5) + 0.1, 0);
    R.add(sl);
    this.siluPx = px; this.siluPy = py;
    this.ballS = new THREE.Mesh(new THREE.SphereGeometry(0.075, 20, 14), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    R.add(this.ballS);
    this.ballLbl = label('', 'lbl num pos');
    this.ballLbl.center.set(0.5, 1.5);
    this.ballS.add(this.ballLbl);
    // 乘法阀门：SiLU(g) × u
    this.mul = new THREE.Mesh(new THREE.TorusGeometry(0.16, 0.035, 12, 40), new THREE.MeshStandardMaterial({ color: 0x111, emissive: AMBER, emissiveIntensity: 0.5 }));
    this.mul.position.set(3.55, 0, 0);
    R.add(this.mul);
    const x1 = label('×', 'lbl title');
    x1.position.copy(this.mul.position);
    R.add(x1);
    this.outLbl = label('', 'lbl big');
    this.outLbl.position.set(3.55, -0.42, 0);
    this.outLbl.center.set(0.5, 0);
    R.add(this.outLbl);
    const link = (a, b) => new THREE.Mesh(new THREE.TubeGeometry(new THREE.LineCurve3(a, b), 1, 0.012, 5), new THREE.MeshBasicMaterial({ color: 0x7a859e, transparent: true, opacity: 0.6 }));
    this.linkS = link(v3(3.2, 1.2, 0), v3(3.5, 0.15, 0));
    this.linkU = link(v3(1.15, -0.8, 0), v3(3.45, -0.1, 0));
    R.add(this.linkS, this.linkU);
    const title = label(`${tr('神经元', 'Neuron')} #${neuronId(n)}`, 'lbl title');
    title.position.set(-0.3, 2.05, 0);
    title.center.set(0, 0.5);
    R.add(title);
  }

  updateRig(st, show, t) {
    const s = st.step;
    this.rig.visible = show && this.M.e > 0.5;
    if (!this.rig.visible) return;
    const n = this.Q.neuronAt(s.L, st.g);
    if (!n) { this.rig.visible = false; return; }
    const key = `${st.g}|${s.L}`;
    if (key !== this.rigKey) { this.rigKey = key; this.buildRig(n); this.bitsIndex = 0; this.flips.clear(); }
    const bp = this.bulbPos(st, neuronId(n));
    this.rig.position.copy(bp).add(v3(0.35, 0.05, 0.55));
    this.rig.scale.setScalar(0.3);
    this.rigCenter = this.rig.position.clone().add(v3(1.8 * 0.3, 0.3 * 0.3, 0));
    if (st.view === 'bits') this.rig.visible = false; // 看比特时，把神经元装置藏起来，只留键帽
    const mi = s.mi || 'gate', p = st.p;
    const done = (m) => ['mul', 'sum', 'silu', 'gate'].indexOf(mi) > ['mul', 'sum', 'silu', 'gate'].indexOf(m);
    // 输入的 12 根引线属于上一步（升维），这里收起来，只看 g、u、SiLU 和阀门
    this.wires.forEach(({ g, u }) => { g.visible = u.visible = false; });
    const fillP = mi === 'sum' ? easeOut(p) : done('sum') ? 1 : 0;
    const mx = Math.max(Math.abs(n.gz), Math.abs(n.uz), 3);
    const fill = (tk, v) => {
      tk.fill.scale.y = Math.max(0.001, (Math.abs(v) / mx) * 1.1 * fillP);
      tk.fill.material.emissive.copy(tk === this.tG ? ROLE_X : ROLE_W);
      tk.lb.el.textContent = fillP > 0 ? `${tk === this.tG ? 'g' : 'u'} = ${(v * fillP).toFixed(3)}` : '';
      tk.lb.visible = fillP > 0;
    };
    fill(this.tG, n.gz);
    fill(this.tU, n.uz);
    const sp = mi === 'silu' ? easeOut(p) : done('silu') ? 1 : 0;
    this.ballS.visible = sp > 0;
    if (sp > 0) {
      const gx = Math.max(-5, Math.min(5, n.gz));
      const x = -5 + (gx + 5) * sp;
      const y = x / (1 + Math.exp(-x));
      this.ballS.position.set(this.siluPx(x), this.siluPy(y) + 0.08, 0.02);
      this.ballLbl.el.textContent = `${y.toFixed(3)}`;
    }
    const gp = mi === 'gate' ? easeOut(p) : 0;
    this.mul.material.emissiveIntensity = 0.5 + gp * 2.2;
    this.outLbl.visible = gp > 0.3;
    this.outLbl.el.textContent = `a = ${(n.silu * n.uz).toFixed(3)}`;
    this.linkS.material.opacity = this.linkU.material.opacity = 0.2 + gp * 0.8;
  }

  /* ------------------------------------------------------------ 一次 Q·K 打分 */

  buildDot(d) {
    dispose(this.dot);
    this.dot.clear();
    const R = this.dot;
    const mxv = Math.max(...d.q.map(Math.abs), ...d.k.map(Math.abs));
    const prods = d.q.map((q, i) => q * d.k[i]);
    const mxp = Math.max(...prods.map(Math.abs));
    this.dotBars = [];
    d.q.forEach((q, i) => {
      const x = -1.9 + i * 0.32;
      const bar = (v, y, c, scale) => {
        const h = Math.max(0.01, (Math.abs(v) / scale) * 0.7);
        const m = new THREE.Mesh(new THREE.BoxGeometry(0.2, h, 0.2), new THREE.MeshStandardMaterial({ color: c.clone().multiplyScalar(0.5), emissive: c, emissiveIntensity: 0.22, roughness: 0.4 }));
        m.position.set(x, y + (v >= 0 ? h / 2 : -h / 2), 0);
        R.add(m);
        return m;
      };
      const qb = bar(q, 0.95, ROLE_X, mxv), kb = bar(d.k[i], -0.95, ROLE_W, mxv);
      const pb = bar(prods[i], 0, AMBER, mxp);
      this.dotBars.push({ pb, qb, kb, i });
    });
    const L = (html, x, y, cls) => { const o = label(html, cls); o.position.set(x, y, 0); o.center.set(1, 0.5); R.add(o); this.dotTags.push(o); return o; };
    this.dotTags = [];
    L(`<span class="cx">q</span> · ${tr(`第 ${d.head} 号头`, `head ${d.head}`)}`, -2.15, 0.95, 'lbl tag x');
    L(`<span class="cw">k</span> · ${tr(`「${esc(this.Q.tokens[d.key].s.replace(/\n/g, '↵'))}」`, `“${esc(this.Q.tokens[d.key].s.replace(/\n/g, '↵'))}”`)}`, -2.15, -0.95, 'lbl tag w');
    L('<span class="cp">q × k</span>', -2.15, 0, 'lbl tag p');
    const hint = label(tr(`128 维里乘积最大的 12 维`, 'the 12 largest of the 128 products'), 'lbl hint');
    hint.position.set(0, 1.85, 0);
    R.add(hint);
    this.dotTags.push(hint);
  }

  updateDot(st, show) {
    const s = st.step;
    this.dot.visible = show && this.M.e > 0.5;
    if (!this.dot.visible) return;
    const d = this.Q.dotAt(s.L, st.g);
    if (!d) { this.dot.visible = false; return; }
    const key = `${st.g}|${s.L}`;
    if (key !== this.dotKey) { this.dotKey = key; this.buildDot(d); }
    const xf = this.M.xFocus(st);
    this.dot.position.set(xf - 1.1, this.base() + 0.62 * this.M.e + 0.95, 0.7);
    this.dot.scale.setScalar(0.3);
    this.dotCenter = this.dot.position.clone().add(v3(0.4 * 0.3, 0, 0));
    // 乘法：12 维轮流出场（和算式板同一个节拍），乘积柱从中间长出来
    const mi = s.mi || 'scale', p = st.p;
    for (const o of this.dotTags) o.visible = st.view !== 'bits';
    const sel = this.dotSel ?? 0;
    this.dotBars.forEach(({ pb, qb, kb, i }) => {
      const ph = mi === 'mul' ? termAt(p, i, 12) : 1;
      pb.visible = ph > 0.3;
      const hot = mi === 'mul' && ph > 0 && ph < 0.7;
      const bits = st.view === 'bits';
      qb.material.emissiveIntensity = hot ? 0.75 : bits ? (i === sel ? 0.6 : 0.05) : 0.22;
      kb.material.emissiveIntensity = hot ? 0.75 : bits ? 0.05 : 0.22;
      pb.material.emissiveIntensity = hot ? 0.8 : bits ? 0.05 : 0.3;
    });
  }

  /* ------------------------------------------------------------ 比特 */

  // 16 个键帽：符号 1 位 / 指数 8 位 / 尾数 7 位，三组之间留缝，正面朝向镜头
  buildBits() {
    dispose(this.bits);
    this.bits.clear();
    const R = this.bits;
    this.keys = [];
    const kx = (i) => (i - 7.5) * 0.25 + (i > 0 ? 0.14 : 0) + (i > 8 ? 0.14 : 0) - 0.14;
    for (let i = 0; i < 16; i++) {
      const c = BIT_C[i === 0 ? 0 : i <= 8 ? 1 : 2];
      const m = new THREE.Mesh(new RoundedBoxGeometry(0.21, 0.12, 0.21, 3, 0.035), new THREE.MeshStandardMaterial({ color: 0x0b1222, emissive: c, emissiveIntensity: 0.1, roughness: 0.3, metalness: 0.2 }));
      m.position.set(kx(i), 0, 0);
      const face = new THREE.Mesh(new THREE.PlaneGeometry(0.16, 0.16), new THREE.MeshBasicMaterial({ transparent: true }));
      face.rotation.x = -Math.PI / 2;
      face.position.y = 0.061;
      m.add(face);
      m.face = face;
      m.col = c;
      m.userData.pick = { type: 'key', i, click: true };
      this.E.pickables.push(m);
      R.add(m);
      this.keys.push(m);
    }
    const grp = (txt, a, b, cls) => { const o = label(txt, `lbl tag ${cls}`); o.position.set((kx(a) + kx(b)) / 2, 0, 0.2); o.center.set(0.5, 0); R.add(o); };
    grp(tr('符号', 'sign'), 0, 0, 's');
    grp(tr('指数 · 8 位', 'exponent · 8 bits'), 1, 8, 'e');
    grp(tr('尾数 · 7 位', 'mantissa · 7 bits'), 9, 15, 'm');
    this.bits.rotation.x = 1.2; // 键帽顶面大致朝向镜头，0 / 1 才看得清
  }

  // 当前要看的那个数：矩阵里被选中的权重，或者打分里 q 的某一维（都可以在算式板上点选）
  bitsTarget(st) {
    const s = st.step;
    if (s.sub === 'score') {
      const d = this.Q.dotAt(s.L, st.g);
      if (!d) return null;
      const k = Math.min(d.q.length - 1, this.dotSel ?? 0);
      return { kind: 'q', v: d.q[k], k, d, anchor: this.dotBars?.[k]?.qb };
    }
    const b = this.M.micro.bitsSource(st);
    return b ? { kind: 'mm', ...b } : null;
  }

  flip(i) {
    const key = this.bitsKey;
    if (!this.baseBits) return;
    const cur = this.flips.get(key) || this.baseBits.slice();
    cur[i] ^= 1;
    this.flips.set(key, cur);
  }

  selectPin(k) { this.bitsIndex = k; }

  // 数值、公式和翻转的后果都写在算式板上；这里只管键帽本身
  updateBits(st, show) {
    this.bits.visible = show;
    if (!show) return;
    const tg = this.bitsTarget(st);
    if (!tg) { this.bits.visible = false; return; }
    const key = `${st.g}|${st.step.L}|${st.step.sub}|${tg.kind}|${tg.k}|${tg.label || ''}`;
    if (key !== this.bitsKey) {
      this.bitsKey = key;
      if (!this.keys) this.buildBits();
      this.baseBits = bf16Bits(tg.v);
    }
    const bits = this.flips.get(key) || this.baseBits;
    this.keys.forEach((m, i) => {
      const on = bits[i] === 1, changed = bits[i] !== this.baseBits[i];
      m.position.y += ((on ? -0.035 : 0) - m.position.y) * 0.3;
      m.material.emissiveIntensity = on ? 0.7 : 0.06;
      m.material.emissive.copy(changed ? WHITE : m.col);
      const tex = textTexture(on ? '1' : '0', { color: on ? '#ffffff' : '#7a859e', font: '700 120px "JetBrains Mono",monospace', w: 128, h: 128 });
      if (m.face.material.map !== tex) { m.face.material.map = tex; m.face.material.needsUpdate = true; }
    });
    const anchor = tg.anchor ? tg.anchor.getWorldPosition(v3(0, 0, 0)) : (this.rigCenter || this.dotCenter || v3(0, 0, 0));
    this.bits.position.copy(anchor).add(v3(0, tg.kind === 'q' ? 0.12 : 0.06, 0.16));
    this.bits.scale.setScalar(0.11);
    this.bitsCenter = this.bits.position.clone();
  }

  /* ------------------------------------------------------------ 相机 */

  camera(st) {
    switch (st.view) {
      case 'mlp': {
        // 神经元阵列在焦点列后面，W_gate / W_up / W_down 在右边：两者都框进来
        const pp = this.panelPos(st);
        const x0 = pp.x - 1.7, x1 = this.M.mats.extentX(st);
        const d = Math.max(4.2, this.E.fitDistance(x1 - x0, 2.9, 1.05));
        const look = v3((x0 + x1) / 2, pp.y - 0.3, -0.2);
        return { pos: look.clone().add(v3(0.4, 0.5, d)), look };
      }
      case 'neuron': { const look = (this.rigCenter || this.panelPos(st)).clone(); const d = this.E.fitDistance(1.4, 1.15, 1.1); return { pos: look.clone().add(v3(0.06, 0.08, d)), look }; }
      case 'dot': { const look = (this.dotCenter || this.panelPos(st)).clone(); const d = this.E.fitDistance(1.9, 1.35, 1.12); return { pos: look.clone().add(v3(0.08, 0.12, d)), look }; }
      case 'bits': { const look = (this.bitsCenter || this.panelPos(st)).clone().add(v3(0, -0.005, 0)); const d = this.E.fitDistance(0.56, 0.2, 1.08); return { pos: look.clone().add(v3(0, d * 0.12, d)), look }; }
    }
    return { pos: v3(10, 8, 20), look: v3(0, 4, 0) };
  }
}
