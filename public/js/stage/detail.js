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
    const inMlp = s.ph === 'layer' && s.op === 'mlp' && (v === 'mlp' || v === 'neuron' || v === 'bits');
    this.updatePanel(st, inMlp, t);
    this.updateRig(st, v === 'neuron', t);
    this.updateDot(st, v === 'dot' || (v === 'bits' && s.sub === 'score'), t);
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

    // 标题
    const title = label(`神经元 #${n.n} · SwiGLU 公式板`, 'lbl title');
    title.position.set(-2.1, 1.85, 0);
    title.center.set(0, 0.5);
    R.add(title);

    // 左侧：输入向量的 12 个显著维度（垂直排列）
    const inputTitle = label('输入 x', 'lbl part');
    inputTitle.position.set(-2.4, 1.45, 0);
    inputTitle.center.set(0.5, 0.5);
    R.add(inputTitle);

    n.x.forEach((x, k) => {
      const y = 1.3 - k * 0.22;
      const pin = new THREE.Mesh(
        new RoundedBoxGeometry(0.18, 0.13, 0.08, 2, 0.025),
        new THREE.MeshStandardMaterial({
          color: 0x0d1424,
          emissive: x >= 0 ? AMBER : BLUE,
          emissiveIntensity: 0.2 + Math.min(0.8, Math.abs(x) / 3),
          roughness: 0.35
        })
      );
      pin.position.set(-2.1, y, 0);
      pin.userData.pick = { type: 'pin', k, dim: n.dims[k], x, wg: n.wg[k], wu: n.wu[k], click: true };
      this.E.pickables.push(pin);
      R.add(pin);
      const lb = label(`x<sub>${n.dims[k]}</sub>=${x >= 0 ? '+' : ''}${x.toFixed(2)}`, 'lbl num');
      lb.position.set(-2.35, y, 0);
      lb.center.set(1, 0.5);
      R.add(lb);
      this.pins.push({ mesh: pin, k, y });
    });

    const rest = label(`… 其余 ${1024 - n.x.length} 项`, 'lbl hint');
    rest.position.set(-2.1, 1.3 - 12 * 0.22 - 0.05, 0);
    rest.center.set(0.5, 0.5);
    R.add(rest);

    // 中间上：gate路径 (x · W_gate → g → SiLU(g))
    const gateLabel = label('W<sub>gate</sub> 路径', 'lbl part');
    gateLabel.position.set(-0.2, 1.45, 0);
    gateLabel.center.set(0.5, 0.5);
    R.add(gateLabel);

    this.gateBox = new THREE.Mesh(
      new RoundedBoxGeometry(0.32, 0.28, 0.12, 3, 0.035),
      new THREE.MeshStandardMaterial({ color: 0x0d1424, emissive: CYAN, emissiveIntensity: 0.5, roughness: 0.3 })
    );
    this.gateBox.position.set(-0.2, 0.85, 0);
    R.add(this.gateBox);
    this.gateLbl = label('', 'lbl big');
    this.gateLbl.position.set(-0.2, 0.55, 0);
    this.gateLbl.center.set(0.5, 0.5);
    R.add(this.gateLbl);

    // SiLU激活
    this.siluBox = new THREE.Mesh(
      new RoundedBoxGeometry(0.32, 0.28, 0.12, 3, 0.035),
      new THREE.MeshStandardMaterial({ color: 0x0d1424, emissive: CYAN, emissiveIntensity: 0.55, roughness: 0.3 })
    );
    this.siluBox.position.set(0.65, 0.85, 0);
    R.add(this.siluBox);
    this.siluLbl = label('', 'lbl big');
    this.siluLbl.position.set(0.65, 0.55, 0);
    this.siluLbl.center.set(0.5, 0.5);
    R.add(this.siluLbl);

    const arrow1 = label('→<small>SiLU</small>', 'lbl hint');
    arrow1.position.set(0.2, 0.85, 0);
    arrow1.center.set(0.5, 0.5);
    R.add(arrow1);

    // 中间下：up路径 (x · W_up → u)
    const upLabel = label('W<sub>up</sub> 路径', 'lbl part');
    upLabel.position.set(-0.2, -0.3, 0);
    upLabel.center.set(0.5, 0.5);
    R.add(upLabel);

    this.upBox = new THREE.Mesh(
      new RoundedBoxGeometry(0.32, 0.28, 0.12, 3, 0.035),
      new THREE.MeshStandardMaterial({ color: 0x0d1424, emissive: VIOLET, emissiveIntensity: 0.5, roughness: 0.3 })
    );
    this.upBox.position.set(-0.2, -0.7, 0);
    R.add(this.upBox);
    this.upLbl = label('', 'lbl big');
    this.upLbl.position.set(-0.2, -1.0, 0);
    this.upLbl.center.set(0.5, 0.5);
    R.add(this.upLbl);

    // 右侧：乘法与输出 (SiLU(g) × u → a)
    const mulBox = new THREE.Mesh(
      new THREE.TorusGeometry(0.18, 0.04, 12, 32),
      new THREE.MeshStandardMaterial({ color: 0x111, emissive: AMBER, emissiveIntensity: 0.6, roughness: 0.3 })
    );
    mulBox.position.set(1.55, 0.1, 0);
    R.add(mulBox);
    const mulLabel = label('×', 'lbl title');
    mulLabel.position.set(1.55, 0.1, 0);
    R.add(mulLabel);

    this.outputBox = new THREE.Mesh(
      new RoundedBoxGeometry(0.38, 0.32, 0.14, 3, 0.04),
      new THREE.MeshStandardMaterial({ color: 0x0d1424, emissive: AMBER, emissiveIntensity: 0.65, roughness: 0.3 })
    );
    this.outputBox.position.set(2.5, 0.1, 0);
    R.add(this.outputBox);
    this.outLbl = label('', 'lbl big');
    this.outLbl.position.set(2.5, -0.28, 0);
    this.outLbl.center.set(0.5, 0.5);
    R.add(this.outLbl);

    const arrow2 = label('→', 'lbl hint');
    arrow2.position.set(2.0, 0.1, 0);
    arrow2.center.set(0.5, 0.5);
    R.add(arrow2);

    // 连接线（简化的线条，从输入到各个计算节点）
    const line = (from, to, color, opacity = 0.3) => {
      const geo = new THREE.BufferGeometry().setFromPoints([from, to]);
      const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
      const l = new THREE.Line(geo, mat);
      R.add(l);
      return l;
    };

    this.linkToGate = line(v3(-1.9, 0.7, 0), v3(-0.5, 0.85, 0), 0x5ef0d4, 0.25);
    this.linkToUp = line(v3(-1.9, 0, 0), v3(-0.5, -0.7, 0), 0xb39dff, 0.25);
    this.linkSiluToMul = line(v3(0.8, 0.85, 0), v3(1.4, 0.2, 0), 0x5ef0d4, 0.35);
    this.linkUpToMul = line(v3(0.1, -0.7, 0), v3(1.4, 0, 0), 0xb39dff, 0.35);

    // 小型SiLU曲线参考（可选，放在右下角）
    const siluHint = label('SiLU(x)=x·σ(x)', 'lbl hint');
    siluHint.position.set(2.5, -0.75, 0);
    siluHint.center.set(0.5, 0.5);
    R.add(siluHint);
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
    this.rig.position.copy(bp).add(v3(0.4, 0.05, 0.55));
    this.rig.scale.setScalar(0.32);
    this.rigCenter = this.rig.position.clone().add(v3(0.8 * 0.32, 0.05, 0));
    if (st.view === 'bits') this.rig.visible = false;

    const mi = s.mi || 'gate', p = st.p;
    const done = (m) => ['mul', 'sum', 'silu', 'gate'].indexOf(mi) > ['mul', 'sum', 'silu', 'gate'].indexOf(m);

    // 输入引脚高亮
    const wireP = mi === 'mul' ? p * 1.15 : done('mul') ? 1 : 0;
    this.pins.forEach(({ mesh, k }) => {
      const on = k / 12 < wireP;
      mesh.material.emissiveIntensity = on ? 0.5 + 0.3 * Math.sin(t * 5) : 0.2;
    });

    // 连接线透明度
    const linkOpacity = wireP > 0.5 ? 0.35 + 0.15 * Math.sin(t * 4) : 0.15;
    this.linkToGate.material.opacity = linkOpacity;
    this.linkToUp.material.opacity = linkOpacity;

    // gate 累加结果
    const sumP = mi === 'sum' ? easeOut(p) : done('sum') ? 1 : 0;
    this.gateBox.material.emissiveIntensity = sumP > 0 ? 0.5 + 0.3 * Math.sin(t * 5) : 0.3;
    this.gateLbl.visible = sumP > 0.2;
    if (this.gateLbl.visible) {
      this.gateLbl.el.innerHTML = `<b>g = ${(n.gz * sumP).toFixed(2)}</b><br><small>Σ x·w<sub>gate</sub></small>`;
    }

    this.upBox.material.emissiveIntensity = sumP > 0 ? 0.5 + 0.3 * Math.sin(t * 5) : 0.3;
    this.upLbl.visible = sumP > 0.2;
    if (this.upLbl.visible) {
      this.upLbl.el.innerHTML = `<b>u = ${(n.uz * sumP).toFixed(2)}</b><br><small>Σ x·w<sub>up</sub></small>`;
    }

    // SiLU 激活
    const siluP = mi === 'silu' ? easeOut(p) : done('silu') ? 1 : 0;
    this.siluBox.visible = siluP > 0.1;
    if (this.siluBox.visible) {
      this.siluBox.material.emissiveIntensity = 0.55 + 0.35 * Math.sin(t * 5);
      this.siluLbl.visible = true;
      const siluVal = n.silu * siluP;
      this.siluLbl.el.innerHTML = `<b>${siluVal.toFixed(3)}</b><br><small>SiLU(g)</small>`;
    } else {
      this.siluBox.visible = false;
      this.siluLbl.visible = false;
    }

    // 最终乘法与输出
    const gateP = mi === 'gate' ? easeOut(p) : 0;
    this.linkSiluToMul.material.opacity = gateP > 0.3 ? 0.5 + 0.3 * Math.sin(t * 4) : 0.2;
    this.linkUpToMul.material.opacity = gateP > 0.3 ? 0.5 + 0.3 * Math.sin(t * 4) : 0.2;
    this.outputBox.material.emissiveIntensity = gateP > 0 ? 0.65 + 0.45 * gateP : 0.4;
    this.outLbl.visible = gateP > 0.3;
    if (this.outLbl.visible) {
      const outVal = n.silu * n.uz * gateP;
      this.outLbl.el.innerHTML = `<b>a = ${outVal.toFixed(3)}</b><br><small>SiLU(g) × u</small>`;
    }
  }

  /* ------------------------------------------------------------ 一次 Q·K 打分 */

  buildDot(d) {
    dispose(this.dot);
    this.dot.clear();
    const R = this.dot;
    const prods = d.q.map((q, i) => q * d.k[i]);
    const mxp = Math.max(...prods.map(Math.abs));
    this.dotBars = [];

    // 标题：从左到右的算式板
    const title = label(`Q[头${d.head}] · K[${esc(this.Q.tokens[d.key].s.replace(/\n/g, '↵'))}]`, 'lbl part');
    title.position.set(-2.4, 0.85, 0);
    title.center.set(0, 0.5);
    R.add(title);

    // 从左到右排列：q[i] × k[i] = 乘积
    d.q.forEach((q, i) => {
      const x0 = -2.1 + i * 0.7;  // 更宽的间距
      const y = 0.15;

      // q[i] 值（青色方块）
      const qBox = new THREE.Mesh(
        new RoundedBoxGeometry(0.16, 0.16, 0.08, 2, 0.02),
        new THREE.MeshStandardMaterial({ color: 0x0d1424, emissive: CYAN, emissiveIntensity: 0.45, roughness: 0.35 })
      );
      qBox.position.set(x0, y, 0);
      R.add(qBox);
      const qLbl = label(`${q >= 0 ? '+' : ''}${q.toFixed(2)}`, 'lbl num');
      qLbl.position.set(x0, y + 0.22, 0);
      qLbl.center.set(0.5, 0.5);
      R.add(qLbl);

      // × 符号
      const times = label('×', 'lbl hint');
      times.position.set(x0 + 0.14, y, 0);
      times.center.set(0.5, 0.5);
      R.add(times);

      // k[i] 值（琥珀色方块）
      const kBox = new THREE.Mesh(
        new RoundedBoxGeometry(0.16, 0.16, 0.08, 2, 0.02),
        new THREE.MeshStandardMaterial({ color: 0x0d1424, emissive: AMBER, emissiveIntensity: 0.45, roughness: 0.35 })
      );
      kBox.position.set(x0 + 0.26, y, 0);
      R.add(kBox);
      const kLbl = label(`${d.k[i] >= 0 ? '+' : ''}${d.k[i].toFixed(2)}`, 'lbl num');
      kLbl.position.set(x0 + 0.26, y + 0.22, 0);
      kLbl.center.set(0.5, 0.5);
      R.add(kLbl);

      // = 符号
      const eq = label('=', 'lbl hint');
      eq.position.set(x0 + 0.4, y, 0);
      eq.center.set(0.5, 0.5);
      R.add(eq);

      // 乘积结果（根据正负显示颜色）
      const prod = prods[i];
      const prodBox = new THREE.Mesh(
        new RoundedBoxGeometry(0.18, 0.18, 0.1, 2, 0.025),
        new THREE.MeshStandardMaterial({
          color: 0x0d1424,
          emissive: prod >= 0 ? AMBER : BLUE,
          emissiveIntensity: 0.3 + Math.min(0.65, Math.abs(prod) / mxp * 0.7),
          roughness: 0.3
        })
      );
      prodBox.position.set(x0 + 0.56, y, 0);
      R.add(prodBox);

      const prodLbl = label(`<b>${prod >= 0 ? '+' : '−'}${Math.abs(prod).toFixed(1)}</b>`, `lbl num ${prod >= 0 ? 'pos' : 'neg'}`);
      prodLbl.position.set(x0 + 0.56, y - 0.24, 0);
      prodLbl.center.set(0.5, 0.5);
      R.add(prodLbl);

      this.dotBars.push({ qBox, kBox, prodBox, qLbl, kLbl, prodLbl, i });
    });

    // 右侧：求和与最终得分
    this.dotSum = label('', 'lbl big');
    this.dotSum.position.set(6.5, 0.4, 0);
    this.dotSum.center.set(0, 0.5);
    R.add(this.dotSum);

    this.dotScore = label('', 'lbl big');
    this.dotScore.position.set(6.5, -0.15, 0);
    this.dotScore.center.set(0, 0.5);
    R.add(this.dotScore);

    const hint = label(`只画出了 128 维里乘积最大的 12 维`, 'lbl hint');
    hint.position.set(2.2, 0.85, 0);
    hint.center.set(0, 0.5);
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
    this.dot.position.set(xf - 2.2, this.base() + 0.62 * this.M.e + 0.95, 0.7);
    this.dot.scale.setScalar(0.26);
    this.dotCenter = this.dot.position.clone().add(v3(1.2 * 0.26, 0, 0));
    const mi = s.mi || 'scale', p = st.p;
    const mulP = mi === 'mul' ? p * 1.15 : 1;
    this.dotBars.forEach(({ qBox, kBox, prodBox, qLbl, kLbl, prodLbl, i }) => {
      const visible = i / 12 < mulP;
      qBox.visible = kBox.visible = prodBox.visible = visible;
      qLbl.visible = kLbl.visible = prodLbl.visible = visible;
    });
    const sumP = mi === 'sum' ? easeOut(p) : mi === 'scale' ? 1 : 0;
    this.dotSum.visible = sumP > 0;
    this.dotSum.el.innerHTML = `<b>Σ 12项 = ${(d.sum * sumP).toFixed(2)}</b><br><small style="font-size:11px">完整128维求和</small>`;
    const scP = mi === 'scale' ? easeOut(p) : 0;
    this.dotScore.visible = scP > 0.2;
    this.dotScore.el.innerHTML = `<b>÷ √128 = ${d.score.toFixed(3)}</b><br><small style="font-size:11px">→ softmax后权重 ${fmtPct(d.w)}</small>`;
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
    if (s.sub === 'score') {
      const d = this.Q.dotAt(s.L, st.g);
      return d ? { kind: 'q', v: d.q[0], k: 0, d, anchor: this.dotBars?.[0]?.pb } : null;
    }
    const b = this.M.micro.bitsSource(st);
    return b ? { kind: 'mm', ...b } : null;
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
    const key = `${st.g}|${st.step.L}|${st.step.sub}|${tg.kind}|${tg.k}|${tg.label || ''}`;
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
    const what = tg.kind === 'mm' ? tg.label : 'q[0]';
    this.bitsVal.el.innerHTML = `${what} = ${fmtSci(val.value)}<br><small style="font-size:11px;color:var(--dim)">(−1)<sup>${val.s}</sup> × 2<sup>${val.e}−127</sup> × (1 + ${val.m}/128) = ${val.s ? '−' : ''}2<sup>${E}</sup> × ${frac.toFixed(4)}</small>`;
    if (tg.kind === 'mm') {
      const t2 = tg.total + tg.x * (val.value - tg.v);
      const fmt = (v) => (Number.isFinite(v) ? v.toFixed(3) : fmtSci(v));
      let extra = '';
      if (tg.kind2 === 'gate' || tg.e2) {
        const u = tg.e2.total;
        extra = `，神经元输出 silu(g)·u ${fmt((tg.total / (1 + Math.exp(-tg.total))) * u)} → <b>${fmt((t2 / (1 + Math.exp(-t2))) * u)}</b>`;
      }
      this.bitsFx.el.innerHTML = this.flips.has(key) ? `翻转之后：${tg.out} ${fmt(tg.total)} → <b>${fmt(t2)}</b>${extra}` : `${tg.out} = … + x × 这个权重 + …（x = ${tg.x.toFixed(3)}）· 点击任意一个键帽翻转`;
    } else this.bitsFx.el.innerHTML = this.flips.has(key) ? '激活值也是 bf16：翻转指数位，数值会成倍地变' : '点击任意一个键帽，翻转这个比特';
    const anchor = tg.anchor ? tg.anchor.getWorldPosition(v3(0, 0, 0)) : (this.rigCenter || this.dotCenter || v3(0, 0, 0));
    this.bits.position.copy(anchor).add(v3(0.02, 0.09, 0.2));
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
      case 'neuron': {
        const look = (this.rigCenter || this.panelPos(st)).clone();
        const d = this.E.fitDistance(2.8, 1.55, 1.08);
        return { pos: look.clone().add(v3(0.15, 0.15, d)), look };
      }
      case 'dot': {
        const look = (this.dotCenter || this.panelPos(st)).clone();
        const d = this.E.fitDistance(5.2, 1.45, 1.08);
        return { pos: look.clone().add(v3(0.12, 0.15, d)), look };
      }
      case 'bits': {
        const look = (this.bitsCenter || this.panelPos(st)).clone().add(v3(0, 0.02, 0));
        const d = this.E.fitDistance(0.5, 0.25, 1.1);
        return { pos: look.clone().add(v3(0, d * 0.3, d)), look };
      }
    }
    return { pos: v3(10, 8, 20), look: v3(0, 4, 0) };
  }
}
