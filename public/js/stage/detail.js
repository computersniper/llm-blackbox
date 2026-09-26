// 最深的几层：SwiGLU 的 3072 个神经元、单个神经元的真实乘加、一次 Q·K 打分、权重的 bf16 比特。
import { THREE, label, textTexture, easeOut, seg } from './engine.js';
import { RoundedBoxGeometry } from '../vendor/three/addons/geometries/RoundedBoxGeometry.js';
import { esc, fmtPct } from '../ui.js';
import { bf16Bits, bf16Value, fmtSci } from '../num.js';

const COLS = 64, ROWS = 48, CELL = 0.05;
const AMBER = new THREE.Color(0xffb65c), BLUE = new THREE.Color(0x6b9bff), CYAN = new THREE.Color(0x5ef0d4), VIOLET = new THREE.Color(0xb39dff);
const DIM = new THREE.Color(0x0d1424);
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
    this.bulbs = new THREE.InstancedMesh(new THREE.SphereGeometry(0.017, 8, 6), new THREE.MeshBasicMaterial({ color: 0xffffff }), COLS * ROWS);
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
    const inMlp = s.ph === 'layer' && s.op === 'mlp' && (v === 'mlp' || v === 'neuron' || v === 'bits');
    this.updatePanel(st, inMlp, t);
    this.updateRig(st, v === 'neuron' || (v === 'bits' && s.op === 'mlp'), t);
    this.updateDot(st, v === 'dot' || (v === 'bits' && s.op === 'attn'), t);
    this.updateBits(st, v === 'bits', t);
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
      this.panelTitle.el.innerHTML = `第 ${L} 层 · SwiGLU 的 3072 个神经元<small>${this.fullLayer ? '完整导出' : '只导出了最亮的 16 个'} · 明显激活 ${Q.mlpCount(g, L)} 个</small>`;
      this.litState = -1;
    }
    // 升维：两把“扇子”从残差流展开到阵列；激活：灯泡逐个点亮；降维：光收回残差流
    const sub = s.sub || 'act', p = st.p;
    let lit = 1, fan = 0;
    if (sub === 'up') { lit = 0; fan = easeOut(p); }
    else if (sub === 'act') lit = s.mi ? 1 : easeOut(seg(p, 0.05, 0.85));
    else if (sub === 'down') { lit = 1 - 0.7 * easeOut(p); fan = 1 - easeOut(p); }
    const dimK = st.view === 'mlp' ? 1 : 0.18;
    const litKey = Math.round(lit * 60) + (dimK < 1 ? 1000 : 0);
    this.dimK = dimK;
    if (litKey !== this.litState) {
      this.litState = litKey;
      for (let n = 0; n < this.acts.length; n++) {
        const a = this.acts[n], m = Math.abs(a) / this.maxAct;
        if (this.order[n] > lit || m < 0.02) { this.bulbs.setColorAt(n, DIM); continue; }
        tmpC.copy(a >= 0 ? AMBER : BLUE).lerp(new THREE.Color(0xffffff), Math.max(0, m - 0.6)).multiplyScalar((0.15 + Math.pow(m, 0.6) * 2.4) * this.dimK);
        this.bulbs.setColorAt(n, tmpC);
      }
      this.bulbs.instanceColor.needsUpdate = true;
    }
    const hp = this.bulbPos(st, this.topN).sub(pp);
    this.hot.position.set(hp.x, hp.y, 0.01);
    this.hot.material.opacity = lit > 0.9 ? 0.6 + 0.4 * Math.sin(t * 5) : 0;
    if (fan > 0.01) {
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
      const lb = label(`x<sub>${n.dims[k]}</sub> ${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(3)}`, 'lbl num');
      lb.position.set(-2.25, y, 0);
      lb.center.set(1, 0.5);
      R.add(lb);
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
    const rest = label(`… 其余 ${1024 - n.x.length} 项`, 'lbl hint');
    rest.position.set(-2.1, 1.65 - 12 * 0.3, 0);
    R.add(rest);
    const node = (p, c, txt) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.13, 24, 16), new THREE.MeshStandardMaterial({ color: 0x0b1222, emissive: c, emissiveIntensity: 0.6 }));
      m.position.copy(p);
      R.add(m);
      const lb = label(txt, 'lbl part');
      lb.position.copy(p).add(v3(0, 0.25, 0));
      R.add(lb);
      return m;
    };
    this.gateNode = node(gateP, CYAN, 'Σ x·w<sub>gate</sub>');
    this.upNode = node(upP, VIOLET, 'Σ x·w<sub>up</sub>');
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
    this.tG = tank(v3(0.95, 0.8, 0), CYAN);
    this.tU = tank(v3(0.95, -0.8, 0), VIOLET);
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
    const title = label(`神经元 #${n.n}`, 'lbl title');
    title.position.set(-2.1, 2.05, 0);
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
    const bp = this.bulbPos(st, n.n);
    this.rig.position.copy(bp).add(v3(0.35, 0.05, 0.55));
    this.rig.scale.setScalar(0.3);
    this.rigCenter = this.rig.position.clone().add(v3(0.7 * 0.3, 0.05, 0));
    if (st.view === 'bits') this.rig.visible = false; // 看比特时，把神经元装置藏起来，只留键帽
    const mi = s.mi || 'gate', p = st.p;
    const done = (m) => ['mul', 'sum', 'silu', 'gate'].indexOf(mi) > ['mul', 'sum', 'silu', 'gate'].indexOf(m);
    const wireP = mi === 'mul' ? p * 1.15 : 1;
    this.wires.forEach(({ g, u, k }) => {
      const on = k / 12 < wireP;
      g.visible = u.visible = on;
    });
    const fillP = mi === 'sum' ? easeOut(p) : done('sum') ? 1 : 0;
    const mx = Math.max(Math.abs(n.gz), Math.abs(n.uz), 3);
    const fill = (tk, v) => {
      tk.fill.scale.y = Math.max(0.001, (Math.abs(v) / mx) * 1.1 * fillP);
      tk.fill.material.emissive.copy(v >= 0 ? (tk === this.tG ? CYAN : VIOLET) : BLUE);
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
      const qb = bar(q, 0.95, CYAN, mxv), kb = bar(d.k[i], -0.95, AMBER, mxv);
      const pb = bar(prods[i], 0, prods[i] >= 0 ? AMBER : BLUE, mxp);
      const lb = label(`${prods[i] >= 0 ? '+' : '−'}${Math.abs(prods[i]).toFixed(1)}`, `lbl num ${prods[i] >= 0 ? 'pos' : 'neg'}`);
      lb.position.set(x, -0.08, 0.15);
      lb.center.set(0.5, 1);
      R.add(lb);
      this.dotBars.push({ pb, lb, i });
    });
    const L = (html, x, y, cls = 'lbl part') => { const o = label(html, cls); o.position.set(x, y, 0); o.center.set(1, 0.5); R.add(o); return o; };
    L(`Q · 第 ${d.head} 头`, -2.15, 1.35);
    L(`K · 第 ${d.kv} 组（${esc(this.Q.tokens[d.key].s.replace(/\n/g, '↵'))}）`, -2.15, -1.35);
    L('q<sub>i</sub>·k<sub>i</sub>', -2.15, 0);
    this.dotSum = label('', 'lbl big');
    this.dotSum.position.set(2.45, 0.3, 0);
    R.add(this.dotSum);
    this.dotScore = label('', 'lbl big');
    this.dotScore.position.set(2.45, -0.3, 0);
    R.add(this.dotScore);
    const hint = label(`只画出了 128 维里乘积最大的 12 维`, 'lbl hint');
    hint.position.set(0, 1.85, 0);
    R.add(hint);
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
    const mi = s.mi || 'scale', p = st.p;
    const mulP = mi === 'mul' ? p * 1.15 : 1;
    this.dotBars.forEach(({ pb, lb, i }) => { pb.visible = i / 12 < mulP; lb.visible = pb.visible; });
    const sumP = mi === 'sum' ? easeOut(p) : mi === 'scale' ? 1 : 0;
    this.dotSum.visible = sumP > 0;
    this.dotSum.el.textContent = `Σ = ${(d.sum * sumP).toFixed(2)}`;
    const scP = mi === 'scale' ? easeOut(p) : 0;
    this.dotScore.visible = scP > 0.2;
    this.dotScore.el.innerHTML = `÷ √128 = ${d.score.toFixed(3)} → 权重 ${fmtPct(d.w)}`;
  }

  /* ------------------------------------------------------------ 比特 */

  buildBits() {
    dispose(this.bits);
    this.bits.clear();
    const R = this.bits;
    this.keys = [];
    for (let i = 0; i < 16; i++) {
      const c = i === 0 ? new THREE.Color(0xff6b93) : i <= 8 ? AMBER : CYAN;
      const m = new THREE.Mesh(new RoundedBoxGeometry(0.21, 0.12, 0.21, 3, 0.035), new THREE.MeshStandardMaterial({ color: 0x0b1222, emissive: c, emissiveIntensity: 0.1, roughness: 0.3, metalness: 0.2 }));
      m.position.set((i - 7.5) * 0.25, 0, 0);
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
    const grp = (txt, a, b, cls) => { const o = label(txt, `lbl ${cls}`); o.position.set(((a + b) / 2 - 7.5) * 0.25, -0.2, 0.1); o.center.set(0.5, 0); R.add(o); };
    grp('符号', 0, 0, 'num neg');
    grp('指数 · 8 位', 1, 8, 'num pos');
    grp('尾数 · 7 位', 9, 15, 'num');
    this.bitsVal = label('', 'lbl big');
    this.bitsVal.position.set(0, 0.42, 0);
    this.bitsVal.center.set(0.5, 1);
    R.add(this.bitsVal);
    this.bitsFx = label('', 'lbl hint');
    this.bitsFx.position.set(0, 0.2, 0);
    this.bitsFx.center.set(0.5, 1);
    R.add(this.bitsFx);
  }

  // 当前要看的那个数：神经元里某个输入的 gate 权重，或者打分里 Q 的某一维
  bitsTarget(st) {
    const s = st.step;
    if (s.op === 'mlp') {
      const n = this.Q.neuronAt(s.L, st.g);
      if (!n) return null;
      const k = Math.min(this.bitsIndex, n.wg.length - 1);
      return { kind: 'w', v: n.wg[k], k, n, anchor: this.pins?.[k] };
    }
    const d = this.Q.dotAt(s.L, st.g);
    return d ? { kind: 'q', v: d.q[0], k: 0, d, anchor: this.dotBars?.[0]?.pb } : null;
  }

  flip(i) {
    const key = this.bitsKey;
    const cur = this.flips.get(key) || this.baseBits.slice();
    cur[i] ^= 1;
    this.flips.set(key, cur);
  }

  selectPin(k) { this.bitsIndex = k; }

  updateBits(st, show, t) {
    this.bits.visible = show;
    if (!show) return;
    const tg = this.bitsTarget(st);
    if (!tg) { this.bits.visible = false; return; }
    const key = `${st.g}|${st.step.L}|${tg.kind}|${tg.k}`;
    if (key !== this.bitsKey) {
      this.bitsKey = key;
      if (!this.keys) this.buildBits();
      this.baseBits = bf16Bits(tg.v);
    }
    const bits = this.flips.get(key) || this.baseBits;
    const val = bf16Value(bits);
    this.keys.forEach((m, i) => {
      const on = bits[i] === 1;
      m.position.y += ((on ? -0.035 : 0) - m.position.y) * 0.3;
      m.material.emissiveIntensity = on ? 0.75 : 0.06;
      const tex = textTexture(on ? '1' : '0', { color: on ? '#ffffff' : '#4b5572', font: '700 120px "JetBrains Mono",monospace', w: 128, h: 128 });
      if (m.face.material.map !== tex) { m.face.material.map = tex; m.face.material.needsUpdate = true; }
    });
    const E = val.e - 127, frac = 1 + val.m / 128;
    const what = tg.kind === 'w' ? `w<sub>gate</sub>[${tg.n.dims[tg.k]}]` : 'q[0]';
    this.bitsVal.el.innerHTML = `${what} = ${fmtSci(val.value)}<br><small style="font-size:11px;color:var(--dim)">(−1)<sup>${val.s}</sup> × 2<sup>${val.e}−127</sup> × (1 + ${val.m}/128) = ${val.s ? '−' : ''}2<sup>${E}</sup> × ${frac.toFixed(4)}</small>`;
    if (tg.kind === 'w') {
      const n = tg.n;
      const gz2 = n.gz + n.x[tg.k] * (val.value - tg.v);
      const a2 = (gz2 / (1 + Math.exp(-gz2))) * n.uz;
      this.bitsFx.el.innerHTML = this.flips.has(key)
        ? `翻转之后：g ${n.gz.toFixed(3)} → <b>${Number.isFinite(gz2) ? gz2.toFixed(3) : fmtSci(gz2)}</b>，神经元输出 ${(n.silu * n.uz).toFixed(3)} → <b>${Number.isFinite(a2) ? a2.toFixed(3) : fmtSci(a2)}</b>`
        : '点击任意一个键帽，翻转这个比特';
    } else this.bitsFx.el.innerHTML = this.flips.has(key) ? '激活值也是 bf16：翻转指数位，数值会成倍地变' : '点击任意一个键帽，翻转这个比特';
    const anchor = tg.anchor ? tg.anchor.getWorldPosition(v3(0, 0, 0)) : (this.rigCenter || this.dotCenter || v3(0, 0, 0));
    this.bits.position.copy(anchor).add(v3(0.02, 0.09, 0.2));
    this.bits.scale.setScalar(0.11);
    this.bitsCenter = this.bits.position.clone();
  }

  /* ------------------------------------------------------------ 相机 */

  camera(st) {
    switch (st.view) {
      case 'mlp': { const look = this.panelPos(st).add(v3(0, -0.15, 0)); return { pos: look.clone().add(v3(0.7, 0.45, 3.9)), look }; }
      case 'neuron': { const look = (this.rigCenter || this.panelPos(st)).clone(); const d = this.E.fitDistance(2.05, 1.45, 1.12); return { pos: look.clone().add(v3(0.1, 0.12, d)), look }; }
      case 'dot': { const look = (this.dotCenter || this.panelPos(st)).clone(); const d = this.E.fitDistance(1.9, 1.35, 1.12); return { pos: look.clone().add(v3(0.08, 0.12, d)), look }; }
      case 'bits': { const look = (this.bitsCenter || this.panelPos(st)).clone().add(v3(0, 0.02, 0)); const d = this.E.fitDistance(0.5, 0.25, 1.1); return { pos: look.clone().add(v3(0, d * 0.3, d)), look }; }
    }
    return { pos: v3(10, 8, 20), look: v3(0, 4, 0) };
  }
}
