// 可以被“揭开”的机器：外壳、词元托盘、28 层玻璃层板、残差流光柱、KV 缓存、注意力光束、
// 逻辑透镜读数、拆开的单层结构、输出头、采样轨道和自回归回路。
// 一切都由 update(state) 根据当前调试步骤计算出来，所以暂停、单步、回退都能正确显示。
import { THREE, textTexture, label, easeOut, easeInOut, seg } from './engine.js';
import { RoundedBoxGeometry } from '../vendor/three/addons/geometries/RoundedBoxGeometry.js';
import { tokPlain, shortSpecial, fmtPct, hueOf, esc } from '../ui.js';
import { Detail } from './detail.js';

export const ROLE = { system: 0x8fa6d6, user: 0x5ee4f0, assistant: 0xffb65c, tpl: 0xb39dff };
const S = 0.42;        // 词元间距
const Y0 = 0.95;       // 第 0 层高度
const GAP = 0.26;      // 层间距
const EXP = 2.7;       // 拆开一层时撑开的空间
const DEPTH = 1.8;     // 机器进深
const SUB = { ln1: 0.25, attn: 0.62, add1: 1.08, ln2: 1.36, mlp: 1.78, add2: 2.34 };

const col = (hex) => new THREE.Color(hex);
const tmpM = new THREE.Matrix4();
const tmpV = new THREE.Vector3();
const ZERO = new THREE.Matrix4().makeScale(0, 0, 0);

function disposeTree(o) {
  o.traverse((c) => {
    if (c.geometry && !c.geometry.userData?.shared) c.geometry.dispose();
    if (c.material && !c.material.userData?.shared) (Array.isArray(c.material) ? c.material : [c.material]).forEach((m) => m.dispose());
    if (c.el) c.el.remove();
  });
}

const GEO = {
  tile: new RoundedBoxGeometry(0.36, 0.24, 0.36, 3, 0.045),
  face: new THREE.PlaneGeometry(0.33, 0.21),
  column: (() => { const g = new THREE.CylinderGeometry(0.026, 0.026, 1, 8, 1, true); g.translate(0, 0.5, 0); return g; })(),
  ring: new THREE.TorusGeometry(0.075, 0.012, 8, 32),
  kv: new THREE.BoxGeometry(0.07, 0.07, 0.07),
  bar: (() => { const g = new THREE.BoxGeometry(0.34, 1, 0.34); g.translate(0, 0.5, 0); return g; })(),
  ball: new THREE.SphereGeometry(0.07, 20, 14),
};
Object.values(GEO).forEach((g) => { g.userData.shared = true; });

export class Machine {
  constructor(E) {
    this.E = E;
    this.scene = E.scene;
    this.detail = new Detail(E);
    this.buildStatic();
    this.root = null;
  }

  /* ------------------------------------------------------------ 静态环境 */

  buildStatic() {
    const floorTex = (() => {
      const c = document.createElement('canvas');
      c.width = c.height = 512;
      const g = c.getContext('2d');
      const gr = g.createRadialGradient(256, 256, 0, 256, 256, 256);
      gr.addColorStop(0, 'rgba(34,78,105,0.42)');
      gr.addColorStop(0.35, 'rgba(14,30,52,0.35)');
      gr.addColorStop(1, 'rgba(5,11,23,0)');
      g.fillStyle = gr;
      g.fillRect(0, 0, 512, 512);
      const t = new THREE.CanvasTexture(c);
      t.colorSpace = THREE.SRGBColorSpace;
      return t;
    })();
    const floor = new THREE.Mesh(new THREE.CircleGeometry(40, 64), new THREE.MeshBasicMaterial({ map: floorTex, transparent: true, depthWrite: false }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.3;
    this.scene.add(floor);
    const grid = new THREE.GridHelper(80, 80, 0x1c3350, 0x10203a);
    grid.position.y = -0.29;
    grid.material.transparent = true;
    grid.material.opacity = 0.35;
    this.scene.add(grid);
    // 漂浮的微粒
    const n = 700, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { pos[i * 3] = (Math.random() - 0.5) * 70; pos[i * 3 + 1] = Math.random() * 26 - 2; pos[i * 3 + 2] = (Math.random() - 0.5) * 50; }
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.dust = new THREE.Points(pg, new THREE.PointsMaterial({ color: 0x7fd8e8, size: 0.05, transparent: true, opacity: 0.32, depthWrite: false }));
    this.scene.add(this.dust);
  }

  /* ------------------------------------------------------------ 为一条回复搭建机器 */

  load(Q) {
    if (this.root) { disposeTree(this.root); this.scene.remove(this.root); }
    this.E.pickables.length = 0;
    this.Q = Q;
    const T = (this.T = Q.T);
    const NL = (this.NL = Q.NL);
    this.W = T * S + 1.4;
    const root = (this.root = new THREE.Group());
    this.scene.add(root);
    this.x = (i) => (i - (T - 1) / 2) * S;
    this.explodeL = -1;
    this.e = 0;
    this.open = 0;
    this.beamKey = '';
    this.fade = 0;

    // 托盘
    const tray = new THREE.Mesh(new THREE.BoxGeometry(this.W, 0.06, 0.8), new THREE.MeshStandardMaterial({ color: 0x0c1629, metalness: 0.6, roughness: 0.35 }));
    tray.position.y = -0.03;
    root.add(tray);
    const trayEdge = new THREE.LineSegments(new THREE.EdgesGeometry(tray.geometry), new THREE.LineBasicMaterial({ color: 0x2c6b8a, transparent: true, opacity: 0.8 }));
    trayEdge.position.copy(tray.position);
    root.add(trayEdge);

    // 词元方块 + 残差流光柱
    this.roleMat = {};
    for (const [r, c] of Object.entries(ROLE)) {
      this.roleMat[r] = new THREE.MeshStandardMaterial({ color: col(c).multiplyScalar(0.22), emissive: col(c), emissiveIntensity: 0.08, metalness: 0.25, roughness: 0.4 });
      this.roleMat[r].userData.shared = false;
    }
    this.tiles = [];
    this.columns = [];
    Q.tokens.forEach((t, i) => {
      const g = new THREE.Group();
      const m = new THREE.Mesh(GEO.tile, this.roleMat[t.role] || this.roleMat.tpl);
      g.add(m);
      const txt = t.sp ? shortSpecial(t.s) : tokPlain(t.s);
      const face = new THREE.Mesh(GEO.face, new THREE.MeshBasicMaterial({ map: textTexture(txt, { color: '#' + col(ROLE[t.role] || ROLE.tpl).getHexString(), font: t.sp ? '600 44px "JetBrains Mono",monospace' : undefined }), transparent: true }));
      face.position.z = 0.181;
      g.add(face);
      const top = face.clone();
      top.rotation.x = -Math.PI / 2;
      top.position.set(0, 0.121, 0);
      g.add(top);
      g.position.set(this.x(i), 0.12, 0);
      m.userData.pick = { type: 'tile', i };
      this.E.pickables.push(m);
      root.add(g);
      this.tiles.push(g);
      const cm = new THREE.MeshBasicMaterial({ color: col(ROLE[t.role] || ROLE.tpl).multiplyScalar(0.72), transparent: true, opacity: 0.5, depthWrite: false });
      const c = new THREE.Mesh(GEO.column, cm);
      c.position.set(this.x(i), 0.25, 0);
      c.scale.y = 0.001;
      root.add(c);
      this.columns.push(c);
    });

    // 28 层玻璃层板
    this.slabs = [];
    const slabGeo = new THREE.BoxGeometry(this.W - 0.3, 0.03, DEPTH);
    const edgeGeo = new THREE.EdgesGeometry(slabGeo);
    for (let L = 0; L < NL; L++) {
      const mat = new THREE.MeshStandardMaterial({ color: 0x6d8cff, emissive: 0x5ef0d4, emissiveIntensity: 0, transparent: true, opacity: 0.1, roughness: 0.15, metalness: 0.2, depthWrite: false });
      const s = new THREE.Mesh(slabGeo, mat);
      const e = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.22 }));
      s.add(e);
      s.edge = e;
      s.userData.pick = { type: 'slab', L, click: true };
      this.E.pickables.push(s);
      const lb = label(`<span class="l">L${String(L).padStart(2, '0')}</span>`, 'lbl lens');
      lb.position.set(this.W / 2 - 0.1, 0.05, DEPTH / 2);
      lb.center.set(0, 0.5);
      s.add(lb);
      s.lbl = lb;
      root.add(s);
      this.slabs.push(s);
    }

    // KV 缓存：每层、每个位置一对小方块（K 琥珀色、V 紫色）
    const kvMat = (c) => new THREE.MeshStandardMaterial({ color: col(c).multiplyScalar(0.35), emissive: col(c), emissiveIntensity: 0.22, roughness: 0.4 });
    this.kvK = new THREE.InstancedMesh(GEO.kv, kvMat(0xffb65c), T * NL);
    this.kvV = new THREE.InstancedMesh(GEO.kv, kvMat(0xb39dff), T * NL);
    this.kvK.frustumCulled = this.kvV.frustumCulled = false;
    root.add(this.kvK, this.kvV);
    this.kvK.userData.pick = (hit) => ({ type: 'kv', kind: 'K', L: Math.floor(hit.instanceId / T), j: hit.instanceId % T });
    this.kvV.userData.pick = (hit) => ({ type: 'kv', kind: 'V', L: Math.floor(hit.instanceId / T), j: hit.instanceId % T });
    this.E.pickables.push(this.kvK, this.kvV);
    this.kvState = '';

    // 残差流上奔跑的光环
    this.pulses = new THREE.InstancedMesh(GEO.ring, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 }), T);
    this.pulses.frustumCulled = false;
    root.add(this.pulses);

    this.beams = new THREE.Group();
    root.add(this.beams);

    this.buildExploded();
    this.buildHead();
    this.buildCasing();
    this.detail.load(Q, this);
  }

  /* ------------------------------------------------------------ 拆开的一层 */

  buildExploded() {
    const g = (this.ex = new THREE.Group());
    this.root.add(g);
    const plate = (color, op) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(this.W - 0.3, 0.022, DEPTH), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.05, transparent: true, opacity: op, depthWrite: false, roughness: 0.2 }));
      m.add(new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.35 })));
      g.add(m);
      return m;
    };
    this.exAttn = plate(0x5ef0d4, 0.08);
    this.exMlp = plate(0xffb65c, 0.07);
    const ringMat = () => new THREE.MeshBasicMaterial({ color: 0x9fe8ff, transparent: true, opacity: 0.8 });
    this.exRing1 = new THREE.Mesh(GEO.ring, ringMat());
    this.exRing2 = new THREE.Mesh(GEO.ring, ringMat());
    this.exRing1.rotation.x = this.exRing2.rotation.x = Math.PI / 2;
    this.exRing1.scale.setScalar(1.6);
    this.exRing2.scale.setScalar(1.6);
    const plus = () => {
      const p = new THREE.Group();
      const m = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 });
      p.add(new THREE.Mesh(new THREE.TorusGeometry(0.09, 0.012, 8, 32), m));
      p.add(new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.018, 0.018), m));
      p.add(new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.11, 0.018), m));
      p.mat = m;
      return p;
    };
    this.exAdd1 = plus();
    this.exAdd2 = plus();
    this.exUnit = new THREE.Mesh(new RoundedBoxGeometry(0.28, 0.2, 0.28, 3, 0.04), new THREE.MeshStandardMaterial({ color: 0x3a2a10, emissive: 0xffb65c, emissiveIntensity: 0.2, roughness: 0.35 }));
    g.add(this.exRing1, this.exRing2, this.exAdd1, this.exAdd2, this.exUnit);
    const L = (html, sub) => { const o = label(`${html}${sub ? `<small>${sub}</small>` : ''}`, 'lbl part'); o.center.set(0, 0.5); g.add(o); return o; };
    this.exLabels = {
      ln1: L('RMSNorm', 'input_layernorm'),
      attn: L('注意力', 'GQA · 16 Q 头 / 8 KV 头'),
      add1: L('⊕ 残差'),
      ln2: L('RMSNorm', 'post_attention'),
      mlp: L('SwiGLU 前馈', '1024 → 3072 → 1024'),
      add2: L('⊕ 残差'),
    };
    g.visible = false;
  }

  /* ------------------------------------------------------------ 输出头与采样 */

  buildHead() {
    const g = (this.head = new THREE.Group());
    this.root.add(g);
    this.normRing = new THREE.Mesh(GEO.ring, new THREE.MeshBasicMaterial({ color: 0x9fe8ff, transparent: true, opacity: 0.8 }));
    this.normRing.rotation.x = Math.PI / 2;
    this.normRing.scale.setScalar(1.8);
    g.add(this.normRing);
    this.lm = new THREE.Mesh(new THREE.BoxGeometry(this.W - 0.3, 0.05, DEPTH), new THREE.MeshStandardMaterial({ color: 0xb39dff, emissive: 0xb39dff, emissiveIntensity: 0.04, transparent: true, opacity: 0.1, depthWrite: false }));
    this.lm.add(new THREE.LineSegments(new THREE.EdgesGeometry(this.lm.geometry), new THREE.LineBasicMaterial({ color: 0xb39dff, transparent: true, opacity: 0.5 })));
    g.add(this.lm);
    this.lmLabel = label('lm_head<small>1024 × 151936 · 与嵌入表共享</small>', 'lbl part');
    this.lmLabel.center.set(0, 0.5);
    g.add(this.lmLabel);

    this.bars = [];
    for (let k = 0; k < 13; k++) {
      const m = new THREE.Mesh(GEO.bar, new THREE.MeshStandardMaterial({ color: 0x223, emissive: 0xffffff, emissiveIntensity: 0.25, roughness: 0.35, metalness: 0.1, transparent: true, opacity: 0.95 }));
      m.scale.y = 0.001;
      const lb = label('', 'lbl bar');
      lb.center.set(0.5, 1);
      m.lbl = lb;
      g.add(m, lb);
      m.userData.pick = { type: 'bar', k };
      this.E.pickables.push(m);
      this.bars.push(m);
    }
    this.strip = new THREE.Group();
    g.add(this.strip);
    this.ball = new THREE.Mesh(GEO.ball, new THREE.MeshBasicMaterial({ color: 0xffffff }));
    g.add(this.ball);
    this.uLabel = label('', 'lbl num pos');
    this.uLabel.center.set(0.5, 1.6);
    this.ball.add(this.uLabel);
    this.stripKey = '';

    // 飞出的词元与回路轨道
    this.flyer = new THREE.Group();
    this.flyerTile = new THREE.Mesh(GEO.tile, this.roleMat.assistant);
    this.flyerFace = new THREE.Mesh(GEO.face, new THREE.MeshBasicMaterial({ transparent: true }));
    this.flyerFace.position.z = 0.181;
    this.flyer.add(this.flyerTile, this.flyerFace);
    this.flyer.scale.setScalar(1.3);
    this.root.add(this.flyer);
    this.loopLine = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: 0xffb65c, transparent: true, opacity: 0.18, depthWrite: false }));
    this.root.add(this.loopLine);
    this.loopKey = '';
  }

  /* ------------------------------------------------------------ 外壳 */

  buildCasing() {
    const W = this.W + 0.7, H = Y0 + (this.NL - 1) * GAP + 4.2, D = DEPTH + 0.9;
    const g = (this.casing = new THREE.Group());
    this.root.add(g);
    const mat = () => new THREE.MeshStandardMaterial({ color: 0x08101f, metalness: 0.7, roughness: 0.22, transparent: true, opacity: 0.94 });
    const edge = () => new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.9 });
    const panel = (w, h, d) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat());
      m.add(new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), edge()));
      return m;
    };
    this.cBack = panel(W, H, 0.06);
    this.cBack.position.set(0, H / 2 - 0.3, -D / 2);
    // 前面板绕底边铰链倒下
    this.cFrontPivot = new THREE.Group();
    this.cFrontPivot.position.set(0, -0.3, D / 2);
    this.cFront = panel(W, H, 0.06);
    this.cFront.position.set(0, H / 2, 0);
    this.cFrontPivot.add(this.cFront);
    const logo = new THREE.Mesh(new THREE.PlaneGeometry(Math.min(W * 0.5, 7), Math.min(W * 0.5, 7) * 0.3), new THREE.MeshBasicMaterial({ map: textTexture('Qwen3-0.6B', { color: '#5ef0d4', font: '700 110px "JetBrains Mono",monospace', w: 1024, h: 300 }), transparent: true }));
    logo.position.set(0, H * 0.12, 0.035);
    this.cFront.add(logo);
    this.cLight = new THREE.Mesh(new THREE.CircleGeometry(0.22, 32), new THREE.MeshBasicMaterial({ color: 0x5ef0d4 }));
    this.cLight.position.set(0, -H * 0.12, 0.035);
    this.cFront.add(this.cLight);
    const sub = new THREE.Mesh(new THREE.PlaneGeometry(W * 0.6, W * 0.6 * 0.08), new THREE.MeshBasicMaterial({ map: textTexture('28 层 · 5.96 亿参数 · bfloat16', { color: '#7a859e', font: '500 60px "PingFang SC",sans-serif', w: 1400, h: 112 }), transparent: true }));
    sub.position.set(0, H * 0.02, 0.035);
    this.cFront.add(sub);
    this.cTop = panel(W, 0.06, D);
    this.cTop.position.set(0, H - 0.3, 0);
    this.cLeft = panel(0.06, H, D);
    this.cLeft.position.set(-W / 2, H / 2 - 0.3, 0);
    this.cRight = panel(0.06, H, D);
    this.cRight.position.set(W / 2, H / 2 - 0.3, 0);
    g.add(this.cBack, this.cFrontPivot, this.cTop, this.cLeft, this.cRight);
    this.casingH = H;
  }

  /* ------------------------------------------------------------ 布局 */

  yL(L) { return Y0 + L * GAP + (L > this.explodeL && this.explodeL >= 0 ? this.e * EXP : 0); }
  get yTop() { return this.yL(this.NL - 1) + 0.4; }
  xFocus(st) { return this.x(this.Q.row(st.g)); }
  headX(st) {
    const half = this.W / 2 - 3.6;
    return Math.max(-half, Math.min(half, this.xFocus(st) - 2.4));
  }

  // 当前计算推进到了哪里（以“层”为单位）：-1 = 刚嵌入，NL = 28 层全部结束
  flow(st) {
    const s = st.step, p = st.p;
    switch (s.ph) {
      case 'pass': return -1 + p * (this.NL + 2);
      case 'read': return -1.5;
      case 'embed': return -1 + p;
      case 'layers': return p * this.NL;
      case 'layer': {
        if (!s.op) return s.L + p;
        const oi = ['ln1', 'attn', 'add1', 'ln2', 'mlp', 'add2'].indexOf(s.op);
        let f = p;
        if (s.sub) { const subs = s.op === 'attn' ? ['qkv', 'score', 'softmax', 'mix'] : ['up', 'act', 'down']; f = (subs.indexOf(s.sub) + (s.mi ? 0.5 : p)) / subs.length; }
        return s.L + (oi + f) / 6;
      }
      case 'head': return this.NL + 0.2;
      case 'sample': return this.NL + 0.6;
    }
    return 0;
  }

  /* ------------------------------------------------------------ 每帧更新 */

  update(st, dt, t) {
    if (!this.Q) return;
    const Q = this.Q, T = this.T, NL = this.NL;
    const s = st.step, p = st.p, g = st.g;
    const ctxLen = Q.P + g;          // 这一步输入的词元数
    const focus = Q.row(g);          // 最后一个位置：它的输出决定下一个词
    const flow = this.flow(st);
    const da = st.dAnim;
    const view = st.view;

    // 外壳：深度 1 → 2 之间揭开
    this.open += (Math.min(1, Math.max(0, da - 1)) - this.open) * Math.min(1, dt * 4);
    const o = easeInOut(this.open);
    this.cFrontPivot.rotation.x = o * Math.PI * 0.5;
    this.cTop.position.y = this.casingH - 0.3 + o * 5;
    this.cTop.material.opacity = 0.94 * (1 - o);
    this.cLeft.position.x = -(this.W + 0.7) / 2 - o * 4;
    this.cRight.position.x = (this.W + 0.7) / 2 + o * 4;
    for (const m of [this.cLeft, this.cRight, this.cFront]) m.material.opacity = 0.94 * (1 - o * 0.95);
    for (const m of [this.cTop, this.cLeft, this.cRight, this.cFront]) m.children[0].material.opacity = 0.9 * (1 - o);
    this.cBack.material.opacity = 0.94 - o * 0.5;
    this.casing.visible = o < 0.999;
    const think = s.ph === 'pass' ? Math.sin(p * Math.PI) : 0;
    this.cLight.material.color.setHSL(0.47, 0.8, 0.45 + think * 0.35 + 0.08 * Math.sin(t * 3));
    const inside = Math.max(0.15, o);

    // 聚焦：越深，越把无关的层板、光柱、KV 缓存淡出
    const FADE = { layer: 0.35, attn: 0.8, dot: 0.9, mlp: 0.85, neuron: 0.95, bits: 0.97 };
    this.fade += ((FADE[view] || 0) - (this.fade || 0)) * Math.min(1, dt * 3);
    const fade = this.fade;
    const BLOOM = { box: 0.46, machine: 0.42, tray: 0.38, tower: 0.34, layer: 0.32, attn: 0.36, head: 0.34, mlp: 0.28, neuron: 0.16, dot: 0.12, bits: 0.14 };
    this.E.bloom.strength += ((BLOOM[view] ?? 0.36) * (this.brightness ?? 1) - this.E.bloom.strength) * Math.min(1, dt * 3);

    // 拆开的层
    const wantEx = view === 'layer' || view === 'attn' || view === 'mlp' || view === 'neuron' || view === 'dot' || view === 'bits';
    const exL = s.ph === 'layer' ? s.L : -1;
    if (wantEx && exL !== this.explodeL && this.e < 0.02) this.explodeL = exL;
    const eT = wantEx && exL === this.explodeL ? 1 : 0;
    this.e += (eT - this.e) * Math.min(1, dt * 3.5);
    if (this.e < 0.005 && !wantEx) this.explodeL = -1;

    // 词元方块
    this.tiles.forEach((tile, i) => {
      let vis = i < ctxLen, y = 0.12, a = 1;
      if (s.ph === 'read' && g === 0) {
        const k = seg(p, (i / Math.max(1, Q.P)) * 0.7, (i / Math.max(1, Q.P)) * 0.7 + 0.3);
        y = 0.12 + (1 - easeOut(k)) * 2.5;
        a = k;
      }
      if (s.ph === 'read' && g > 0 && i === ctxLen - 1) vis = p > 0.97; // 新词元正沿回路落下
      if (s.ph === 'pass' && i === ctxLen - 1 && g > 0) y = 0.12 + (1 - easeOut(Math.min(1, p * 4))) * 1.5;
      tile.visible = vis && a > 0.01;
      tile.position.y = y;
      const hl = i === focus && s.ph !== 'read' ? 1 : 0;
      tile.scale.setScalar(1 + hl * 0.12);
    });

    // 光柱：已缓存的位置是完整的；正在算的位置长到当前层
    const active = (i) => (g === 0 ? i < Q.P : i === focus);
    this.columns.forEach((c, i) => {
      const vis = i < ctxLen && !(s.ph === 'read' && (g === 0 || i === focus));
      c.visible = vis;
      if (!vis) return;
      const isA = active(i);
      const top = this.yTop + (i === focus ? 0.9 : 0);
      let h = top - 0.25;
      if (isA) {
        const f = Math.max(-1, flow);
        h = f < 0 ? (Y0 - 0.25) * (1 + f) : f >= NL ? h : this.yL(Math.min(NL - 1, f)) - 0.25 + (f % 1) * GAP;
        h = Math.max(0.001, h);
      }
      c.scale.y = h;
      const nL = Math.min(NL - 1, Math.max(0, Math.floor(flow)));
      const bright = isA ? 0.68 : 0.14 + 0.12 * Math.min(1, Q.norm(nL, i) / 60);
      c.material.opacity = bright * inside * (i === focus ? 1 - fade * 0.4 : 1 - fade * 0.92);
      c.scale.x = c.scale.z = i === focus ? 1.35 : isA ? 1.1 : 1;
    });

    // 光环
    let pc = 0;
    if (flow > -1 && flow < NL + 0.5 && da >= 1.5) {
      const y = flow < 0 ? 0.25 + (Y0 - 0.25) * (1 + flow) : flow >= NL ? this.yTop : this.yL(Math.floor(flow)) + (flow % 1) * (this.e > 0.3 && Math.floor(flow) === this.explodeL ? EXP * this.e : GAP);
      for (let i = 0; i < ctxLen; i++) {
        if (!active(i)) continue;
        tmpM.makeRotationX(Math.PI / 2).setPosition(this.x(i), y, 0);
        if (i === focus) tmpM.scale(tmpV.set(1.5, 1.5, 1.5));
        this.pulses.setMatrixAt(pc++, tmpM);
      }
    }
    this.pulses.count = pc;
    this.pulses.instanceMatrix.needsUpdate = true;
    this.pulses.material.opacity = 0.45 + 0.2 * Math.sin(t * 10);

    // 层板
    const cur = s.ph === 'layer' ? s.L : s.ph === 'layers' ? Math.floor(flow) : -1;
    this.slabs.forEach((sl, L) => {
      sl.position.y = this.yL(L);
      const passed = flow >= L + 1 || s.ph === 'head' || s.ph === 'sample';
      const on = L === cur || (s.ph === 'layers' && Math.floor(flow) === L);
      const glow = on ? 0.26 : passed ? 0.05 : 0;
      sl.material.emissiveIntensity += (glow - sl.material.emissiveIntensity) * Math.min(1, dt * 8);
      sl.material.opacity = ((0.08 + (on ? 0.12 : 0)) * inside + (L === this.explodeL ? -0.05 * this.e : 0)) * (1 - fade * 0.85);
      sl.edge.material.opacity = (on ? 0.9 : passed ? 0.35 : 0.2) * inside * (1 - fade * 0.8);
      // 层号 / 逻辑透镜读数
      const showLbl = da >= 2.6 && (view === 'tower' || (view === 'layer' && L === cur));
      sl.lbl.visible = showLbl && (flow >= L + 0.35 || L === cur || view !== 'tower');
      if (sl.lbl.visible) {
        const lens = Q.lensAt(g, L).top[0];
        const txt = flow >= L + 0.5 || (view === 'layer' && L === cur) ? `<span class="l">L${String(L).padStart(2, '0')}</span>${esc(tokPlain(lens[2]))}<span class="p">${fmtPct(lens[1])}</span>` : `<span class="l">L${String(L).padStart(2, '0')}</span>`;
        if (sl.lbl.el._t !== txt) { sl.lbl.el.innerHTML = txt; sl.lbl.el._t = txt; }
        sl.lbl.el.classList.toggle('on', L === cur);
      }
    });

    // KV 缓存
    const showKV = da >= 2.6;
    const onlyL = fade > 0.3 ? this.explodeL : -1;
    const kvKey = `${g}|${Math.floor(flow * 6)}|${showKV}|${this.explodeL}|${this.e.toFixed(2)}|${onlyL}|${fade > 0.9}`;
    if (kvKey !== this.kvState) {
      this.kvState = kvKey;
      let n = 0;
      for (let L = 0; L < NL; L++) {
        const y = this.yL(L) + 0.06;
        for (let j = 0; j < T; j++) {
          const has = showKV && j < ctxLen && (!active(j) || flow >= L + 0.3) && (onlyL < 0 || L === onlyL) && fade < 0.9;
          const id = L * T + j;
          if (has) {
            tmpM.makeTranslation(this.x(j) - 0.055, y, -0.55);
            this.kvK.setMatrixAt(id, tmpM);
            tmpM.makeTranslation(this.x(j) + 0.055, y, -0.55);
            this.kvV.setMatrixAt(id, tmpM);
            n++;
          } else { this.kvK.setMatrixAt(id, ZERO); this.kvV.setMatrixAt(id, ZERO); }
        }
      }
      this.kvK.instanceMatrix.needsUpdate = this.kvV.instanceMatrix.needsUpdate = true;
    }

    // 注意力光束
    this.updateBeams(st, flow, focus);

    // 拆开的层内部
    this.updateExploded(st, focus);

    // 输出头
    this.updateHead(st, flow, focus, dt, t);

    // 细节（神经元阵列、单个神经元、比特……）
    this.detail.update(st, dt, t);

    this.dust.rotation.y += dt * 0.01;
  }

  updateBeams(st, flow, focus) {
    const s = st.step, Q = this.Q;
    let L = -1, mode = '', grow = 0, head = null;
    const view = st.view;
    if (view === 'tower' && s.ph === 'layer') { L = s.L; mode = 'flat'; grow = seg(st.p, 0.05, 0.6); }
    else if (view === 'layer' && s.op === 'attn') { L = s.L; mode = 'arch'; grow = seg(st.p, 0, 0.7); }
    else if (view === 'layer' && ['add1', 'ln2', 'mlp', 'add2'].includes(s.op)) { L = s.L; mode = 'arch'; grow = 1; }
    else if (view === 'attn' || view === 'dot') {
      L = s.L; mode = 'arch'; head = st.head;
      grow = s.sub === 'qkv' ? 0 : s.sub === 'score' ? seg(st.p, 0, 0.8) : 1;
    }
    const key = `${L}|${mode}|${head}|${st.g}|${this.explodeL}|${view === 'attn' ? s.sub : ''}`;
    if (key !== this.beamKey) {
      this.beamKey = key;
      disposeTree(this.beams);
      this.beams.clear();
      if (L >= 0) {
        const list = head == null ? Q.attMean(L, focus, 5) : Q.att(L, head, focus);
        const xq = this.x(focus);
        this.beamList = list.map(({ j, w }) => {
          const xk = this.x(j);
          const dx = Math.abs(xq - xk);
          const y0 = mode === 'flat' ? this.yL(L) + 0.03 : this.yL(L) + SUB.attn * this.e + 0.02;
          const ctrl = mode === 'flat'
            ? new THREE.Vector3((xq + xk) / 2, y0 + 0.04, 0.35 + dx * 0.04)
            : new THREE.Vector3((xq + xk) / 2, y0 + Math.min(1.5, 0.2 + dx * 0.16), 0);
          const curve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(xq, y0, 0), ctrl, new THREE.Vector3(xk, y0, j === focus ? 0.25 : 0));
          const geo = new THREE.TubeGeometry(curve, 48, mode === 'arch' ? 0.012 + w * 0.05 : 0.006 + w * 0.03, 8, false);
          const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x4fd8c0, transparent: true, opacity: 0.18 + 0.6 * w, blending: THREE.AdditiveBlending, depthWrite: false }));
          m.userData.pick = { type: 'beam', L, j, w, head };
          this.beams.add(m);
          if (mode === 'arch' && w >= 0.06) {
            const lb = label(`${esc(tokPlain(Q.tokens[j].s))} <b>${fmtPct(w)}</b>`, 'lbl num pos');
            lb.position.copy(curve.getPoint(0.5));
            lb.center.set(0.5, 1.2);
            m.add(lb);
          }
          return { m, geo, j, w };
        });
      } else this.beamList = [];
    }
    const hideBeams = ['mlp', 'neuron', 'bits'].includes(view);
    this.beams.visible = !hideBeams;
    for (const b of this.beamList || []) {
      b.geo.setDrawRange(0, Math.floor(b.geo.index.count * grow / 6) * 6);
      b.m.children.forEach((c) => { if (c.el) c.el.classList.toggle('hide', grow < 0.95); });
    }
  }

  updateExploded(st, focus) {
    const e = this.e, g = this.ex;
    g.visible = e > 0.02 && this.explodeL >= 0;
    if (!g.visible) return;
    const s = st.step;
    const base = Y0 + this.explodeL * GAP;
    const xf = this.x(focus);
    const at = (k) => base + SUB[k] * e;
    this.exAttn.position.set(0, at('attn'), 0);
    this.exMlp.position.set(0, at('mlp'), 0);
    this.exAttn.material.opacity = 0.1 * e;
    this.exMlp.material.opacity = 0.09 * e;
    this.exRing1.position.set(xf, at('ln1'), 0);
    this.exRing2.position.set(xf, at('ln2'), 0);
    this.exAdd1.position.set(xf, at('add1'), 0);
    this.exAdd2.position.set(xf, at('add2'), 0);
    this.exUnit.position.set(xf, at('mlp') + 0.11, 0);
    const opOn = (k) => (s.op === k ? 1 : 0);
    const flash = (k) => opOn(k) * (0.5 + 0.5 * Math.sin(st.p * Math.PI));
    this.exRing1.material.opacity = (0.3 + 0.5 * flash('ln1')) * e;
    this.exRing2.material.opacity = (0.3 + 0.5 * flash('ln2')) * e;
    this.exRing1.scale.setScalar(1.6 + flash('ln1') * 0.6);
    this.exRing2.scale.setScalar(1.6 + flash('ln2') * 0.6);
    this.exAdd1.mat.opacity = (0.35 + 0.45 * flash('add1')) * e;
    this.exAdd2.mat.opacity = (0.35 + 0.45 * flash('add2')) * e;
    this.exAdd1.scale.setScalar(1 + flash('add1') * 0.5);
    this.exAdd2.scale.setScalar(1 + flash('add2') * 0.5);
    this.exAttn.material.emissiveIntensity = 0.04 + opOn('attn') * 0.16;
    this.exMlp.material.emissiveIntensity = 0.04 + opOn('mlp') * 0.14;
    this.exUnit.material.emissiveIntensity = 0.15 + flash('mlp') * 0.9;
    const show = st.view === 'layer';
    for (const [k, lb] of Object.entries(this.exLabels)) {
      lb.position.set(xf + 0.32, at(k) + (k === 'mlp' ? 0.1 : 0), 0.25);
      lb.visible = e > 0.6 && (show || k === s.op);
      lb.el.style.borderColor = s.op === k ? 'var(--amber)' : '';
    }
  }

  updateHead(st, flow, focus, dt, t) {
    const Q = this.Q, s = st.step, p = st.p, g = st.g;
    const stp = Q.steps[g];
    const yTop = this.yTop;
    const xf = this.x(focus);
    const hx = this.headX(st);
    const vis = st.dAnim >= 1.6;
    this.head.visible = vis;
    if (!vis) { this.flyer.visible = this.dAnimFly(st, t); return; }
    this.normRing.position.set(xf, yTop + 0.25, 0);
    this.lm.position.set(0, yTop + 0.62, 0);
    this.lmLabel.position.set(-this.W / 2 + 0.3, yTop + 0.62, DEPTH / 2);
    this.lmLabel.visible = st.view === 'head' || st.view === 'machine';
    const inHead = s.ph === 'head' || s.ph === 'sample';
    this.normRing.material.opacity = inHead && (!s.sub || s.sub === 'norm') ? 0.9 : 0.25;
    this.lm.material.emissiveIntensity = inHead && (!s.sub || s.sub === 'unembed') ? 0.16 : 0.04;

    // 概率柱：输出头阶段按 T=1 升起；采样阶段按温度 0.7 变形，再被 top-k / top-p 截断
    const barsY = yTop + 1.2;
    let rise = 0, temp = 0, cut = 0;
    if (s.ph === 'head') rise = !s.sub ? easeOut(seg(p, 0.3, 1)) : s.sub === 'softmax' ? easeOut(p) : 0;
    if (s.ph === 'sample') {
      rise = 1;
      temp = !s.sub ? seg(p, 0, 0.25) : s.sub === 'temp' ? easeOut(p) : 1;
      cut = !s.sub ? seg(p, 0.25, 0.45) : s.sub === 'topp' ? easeOut(p) : s.sub === 'draw' ? 1 : 0;
    }
    if (st.view === 'machine' && s.ph === 'pass') rise = seg(p, 0.7, 0.9);
    const poolIds = new Set(stp.pool.map((x) => x[0]));
    const t07 = stp.temps['0.7'];
    const other1 = Math.max(0, 1 - stp.top.reduce((a, b) => a + b[1], 0));
    const items = stp.top.map(([id, p1, str], k) => ({ id, str, p1, p7: t07[k] })).concat([{ id: -1, str: '其他', p1: other1, p7: t07[12] }]);
    items.forEach((it, k) => {
      const bar = this.bars[k];
      const pv = it.p1 + (it.p7 - it.p1) * temp;
      const inPool = it.id >= 0 && poolIds.has(it.id);
      const alive = 1 - cut * (inPool ? 0 : 0.8);
      bar.visible = vis;
      bar.position.set(hx + (k - 6) * 0.5, barsY, -0.25);
      bar.scale.y = Math.max(0.001, pv * 3.2 * rise);
      const hue = it.id < 0 ? 0.62 : hueOf(it.str) / 360;
      const won = s.ph === 'sample' && (!s.sub || s.sub === 'draw') && p > 0.8 && it.id === stp.chosen;
      bar.material.emissive.setHSL(hue, it.id < 0 ? 0.1 : 0.75, won ? 0.7 : 0.45);
      bar.material.emissiveIntensity = (0.16 + (won ? 0.55 : 0)) * alive;
      bar.material.opacity = 0.25 + 0.7 * alive;
      bar.lbl.position.set(bar.position.x, barsY - 0.06, 0.05);
      bar.lbl.visible = rise > 0.2 && (st.view === 'head' || st.view === 'machine') && (k < 8 || it.id < 0);
      const html = `<span class="t">${it.id < 0 ? '其他' : esc(tokPlain(it.str))}</span><span class="p">${fmtPct(pv)}</span>`;
      if (bar.lbl.el._t !== html) { bar.lbl.el.innerHTML = html; bar.lbl.el._t = html; }
      bar.lbl.el.classList.toggle('win', won);
      bar.lbl.el.classList.toggle('cut', cut > 0.5 && !inPool);
      bar.userData.pick = { type: 'bar', str: it.str, p: pv, id: it.id, inPool };
    });

    // 采样轨道：候选池按最终概率铺成一条线段，随机数 u 决定小球停在哪
    const stripY = barsY - 0.34, L = 5.2, x0 = hx - L / 2;
    const showStrip = s.ph === 'sample' && (!s.sub || s.sub === 'draw' || s.sub === 'topp');
    const key = `${g}`;
    if (key !== this.stripKey) {
      this.stripKey = key;
      disposeTree(this.strip);
      this.strip.clear();
      let acc = 0;
      stp.pool.forEach(([id, pr, str]) => {
        const w = pr * L;
        const m = new THREE.Mesh(new THREE.BoxGeometry(Math.max(0.01, w - 0.02), 0.06, 0.16), new THREE.MeshStandardMaterial({ color: 0x111, emissive: new THREE.Color().setHSL(hueOf(str) / 360, 0.7, 0.5), emissiveIntensity: 0.35 }));
        m.position.set(x0 + acc + w / 2, stripY, 0.75);
        m.userData.pick = { type: 'seg', str, p: pr };
        m.chosen = id === stp.chosen;
        this.E.pickables.push(m);
        this.strip.add(m);
        if (w > 0.45) { const lb = label(esc(tokPlain(str)), 'lbl hint'); lb.position.set(0, -0.12, 0.1); lb.center.set(0.5, 0); m.add(lb); }
        acc += w;
      });
    }
    this.strip.visible = showStrip;
    const drawP = s.ph === 'sample' ? (!s.sub ? seg(p, 0.45, 0.85) : s.sub === 'draw' ? seg(p, 0.05, 0.75) : 0) : 0;
    this.ball.visible = showStrip && drawP > 0;
    if (this.ball.visible) {
      const k = easeOut(drawP);
      const bounce = Math.abs(Math.sin(k * Math.PI * 3)) * (1 - k) * 0.5;
      this.ball.position.set(x0 + stp.u * L * k, stripY + 0.1 + bounce, 0.75);
      this.uLabel.el.textContent = `u = ${(stp.u * k).toFixed(3)}`;
    }
    this.strip.children.forEach((m) => { m.material.emissiveIntensity = m.chosen && drawP >= 1 ? 0.8 : 0.25; });

    // 回路轨道：输出 → 机器右侧 → 落回托盘末端
    const nextSlot = Math.min(this.T - 1, Q.P + g);
    const lk = `${g}|${this.explodeL}|${this.e > 0.5}`;
    if (lk !== this.loopKey) {
      this.loopKey = lk;
      const a = new THREE.Vector3(hx + 3.5, barsY + 1.0, 0);
      const pts = [a, new THREE.Vector3(this.W / 2 + 0.6, barsY + 0.6, 0.4), new THREE.Vector3(this.W / 2 + 0.9, yTop * 0.5, 0.6), new THREE.Vector3(this.W / 2 + 0.6, 0.9, 0.7), new THREE.Vector3(this.x(nextSlot), 0.55, 0.45)];
      this.loop = new THREE.CatmullRomCurve3(pts);
      this.loopLine.geometry.dispose();
      this.loopLine.geometry = new THREE.TubeGeometry(this.loop, 80, 0.012, 6, false);
    }
    this.loopLine.visible = st.dAnim >= 1.8;

    // 飞行的词元：采样结束时从胜出的柱子升起；下一步“读入”时沿回路落进托盘
    let fly = null;
    if (s.ph === 'sample' && (!s.sub || s.sub === 'draw') && p > 0.8 && stp.chosenS !== '<|im_end|>') {
      const k = easeOut(seg(p, 0.8, 1));
      const bi = stp.top.findIndex((x) => x[0] === stp.chosen);
      const bx = bi >= 0 ? hx + (bi - 6) * 0.5 : hx;
      fly = { pos: new THREE.Vector3(bx + (hx + 3.5 - bx) * k, barsY + 3.4 * (bi >= 0 ? stp.top[bi][1] : 0.1) + 0.3 + k * 0.6, 0), str: stp.chosenS };
    } else if (s.ph === 'read' && g > 0) {
      const k = easeInOut(p);
      fly = { pos: this.loop ? this.loop.getPoint(Math.min(1, k)) : new THREE.Vector3(), str: Q.tokens[Q.P + g - 1].s };
    }
    this.flyer.visible = !!fly;
    if (fly) {
      this.flyer.position.copy(fly.pos);
      this.flyer.rotation.y = Math.sin(t * 3) * 0.2;
      if (this.flyer._s !== fly.str) { this.flyerFace.material.map = textTexture(tokPlain(fly.str), { color: '#ffb65c' }); this.flyerFace.material.needsUpdate = true; this.flyer._s = fly.str; }
    }
  }

  // 黑箱合着的时候：每想完一个词，就从顶部吐出一块
  dAnimFly(st, t) {
    const s = st.step, stp = this.Q.steps[st.g];
    if (s.ph !== 'pass' || st.p < 0.75 || stp.chosenS === '<|im_end|>') return false;
    const k = easeOut(seg(st.p, 0.75, 1));
    this.flyer.position.set(0, this.casingH + k * 1.6, 0.4);
    this.flyer.rotation.y = t;
    if (this.flyer._s !== stp.chosenS) { this.flyerFace.material.map = textTexture(tokPlain(stp.chosenS), { color: '#ffb65c' }); this.flyerFace.material.needsUpdate = true; this.flyer._s = stp.chosenS; }
    return true;
  }

  /* ------------------------------------------------------------ 相机 */

  camera(st) {
    const E = this.E, W = this.W;
    const v3 = (x, y, z) => new THREE.Vector3(x, y, z);
    const focus = this.Q.row(st.g);
    const xf = this.x(focus);
    const yTop = this.yTop;
    switch (st.view) {
      case 'box': {
        const d = E.fitDistance(W + 1.5, this.casingH + 1, 1.25);
        return { pos: v3(W * 0.12, this.casingH * 0.62, d), look: v3(0, this.casingH * 0.45, 0) };
      }
      case 'machine': {
        const d = E.fitDistance(W + 2, yTop + 5, 1.12);
        return { pos: v3(W * 0.1, yTop * 0.7, d), look: v3(0, yTop * 0.52, 0) };
      }
      case 'tray': { const look = v3(xf - 2.2, 0.9, 0); return { pos: look.clone().add(v3(1.4, 2.0, 7.2)), look }; }
      case 'tower': {
        const f = Math.max(0, Math.min(this.NL - 1, st.step.L ?? 0));
        const look = v3(xf - 2.4, Math.max(2.2, this.yL(f) + 0.3), 0);
        return { pos: look.clone().add(v3(2.2, 2.4, 10.5)), look };
      }
      case 'layer': { const look = v3(xf - 1.1, Y0 + this.explodeL * GAP + 1.25, 0); return { pos: look.clone().add(v3(2.2, 1.2, 6.8)), look }; }
      case 'attn': {
        const list = st.head == null ? this.Q.attMean(this.explodeL, focus, 5) : this.Q.att(this.explodeL, st.head, focus);
        // “看开头”的那一束会伸到最左边，取景时不算它，免得整体缩得太小
        const xs = list.filter((a) => a.j !== 0 || list.length === 1).map((a) => this.x(a.j)).concat([xf]);
        const x0 = Math.min(...xs), x1 = Math.max(...xs);
        const d = Math.max(4.5, Math.min(16, E.fitDistance(x1 - x0 + 2.2, 3.2, 1.05)));
        const look = v3((x0 + x1) / 2, Y0 + this.explodeL * GAP + SUB.attn + 0.6, 0);
        return { pos: look.clone().add(v3(0.6, d * 0.32, d)), look };
      }
      case 'head': { const look = v3(this.headX(st), yTop + 1.9, 0); return { pos: look.clone().add(v3(1.6, 1.0, 8.2)), look }; }
      default: return this.detail.camera(st);
    }
  }
}
