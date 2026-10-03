// 玻璃小模型的 3D 机器：2,928 个参数每个都是一个方块（颜色 = 正负和大小，厚度 = 绝对值），按真实矩阵形状摆成面板；
// 左边的残差流是 8 根光柱（第 0 段的 8 个位置）；每块面板旁边是它的输入 / 输出激活（真实数值）。
// 一步训练在机器上是这样走的（全部来自记录）：
//   进料：这一步的 8 段从语料轮上切下来，字块飞进托盘（第 0 段在最前面，正好在光柱底下）；
//   前向：光脉冲沿光柱往上走，每经过一块面板，它的输出激活一列列亮起来；最上面 20 个字的概率柱升起，绿框是正确答案；
//   损失：每个位置 −ln p，平均成这一步的损失；
//   反向：玫红色的脉冲倒流回去，激活换成它们的梯度，每块面板按真实的 |∂L/∂w| 发光；
//   更新：每个方块按真实的 Δw 往前顶出 / 往后沉下（紫色），再落到新值。
// 和推理页的机器一样，一切都由 update(st) 根据当前步骤算出来：暂停、单步、回退都能正确显示。
import { THREE, label, easeOut, easeInOut, seg } from '../../../js/stage/engine.js';
import { FACE, xPos, SPINE, Y, BLOCKS, BLOCK, cellX, cellY, SHELF, shelfX, shelfZ, GAUGE, WHEEL, TRAY, BOUNDS, OP_RECT, TENSOR_PARTS, unionRect } from './layout.js';
import { divInto, gradInto, seqInto, LIN } from './palette.js';
import { G_PHASES, FWD_OPS, BWD_OPS, FWD_TENSORS, BWD_TENSORS, UPD_TENSORS, ADAM_SUBS } from '../glass/timeline.js';
import { TENSOR_LABEL } from '../glass/data.js';
import { absQuantile, fnum, paramName } from '../glass/heat.js';
import { fmtP } from '../draw.js';
import { L } from '../lang.js';
import { esc } from '../../../js/ui.js';

const HS_MAT = 0.55, HS_G = 0.85;          // 方块厚度的满格：矩阵 |w| = 0.55、γ 的 |γ − 1| = 0.85（训练结束时的量级，看得出越练越厚）
const H0 = 0.012, HK = 0.36;                // 方块厚度 = H0 + HK × (|w| / 满格)^0.85
const INIT_CS = 0.06;                      // 初始化时的色标（±3 个标准差）
const POP = 0.15;                          // Δw = 一个 lr 时方块顶出 / 沉下的距离

// 每块在前向 / 反向里什么时候亮：[第几个算子, 这个算子进度的开始, 结束]
const FW = {
  h0: [0, 0, 1], E: [0, 0, 1], g1: [1, 0, 1], n1: [1, 0, 1],
  Wq: [2, 0, 1 / 3], q: [2, 0, 1 / 3], Wk: [2, 1 / 3, 2 / 3], k: [2, 1 / 3, 2 / 3], Wv: [2, 2 / 3, 1], v: [2, 2 / 3, 1],
  att0: [3, 0, 0.45], att1: [3, 0.2, 0.65], ao: [3, 0.6, 1],
  aoT: [4, 0, 0.25], Wo: [4, 0.25, 0.85], o: [4, 0.25, 0.85], h1: [4, 0.85, 1],
  g2: [5, 0, 1], n2: [5, 0, 1],
  Wg: [6, 0, 0.45], gate: [6, 0, 0.45], Wu: [6, 0.5, 0.9], up: [6, 0.5, 0.9], act: [6, 0.85, 1],
  actT: [7, 0, 0.2], Wd: [7, 0.2, 0.85], f: [7, 0.2, 0.85], h2: [7, 0.85, 1],
  gf: [8, 0, 1], nf: [8, 0, 1], ET: [9, 0, 0.9], logits: [9, 0, 0.9],
};
const BW = {
  logits: [0, 0, 0.5], ET: [0, 0.3, 0.9], nf: [0, 0.5, 1],
  gf: [1, 0, 0.6], h2: [1, 0.3, 1],
  f: [2, 0, 0.25], Wd: [2, 0.15, 0.7], actT: [2, 0.6, 1], act: [2, 0.7, 1],
  gate: [3, 0, 0.25], Wg: [3, 0.1, 0.5], up: [3, 0.5, 0.75], Wu: [3, 0.55, 0.95], n2: [3, 0.8, 1],
  g2: [4, 0, 0.6], h1: [4, 0.3, 1],
  o: [5, 0, 0.25], Wo: [5, 0.15, 0.7], aoT: [5, 0.6, 1], ao: [5, 0.7, 1],
  att0: [6, 0, 0.5], att1: [6, 0, 0.5], v: [6, 0.3, 0.7], q: [6, 0.5, 1], k: [6, 0.5, 1],
  Wq: [7, 0, 1 / 3], Wk: [7, 1 / 3, 2 / 3], Wv: [7, 2 / 3, 1], n1: [7, 0.8, 1],
  g1: [8, 0, 0.6], h0: [8, 0.3, 1], E: [9, 0, 1],
};
const win = (w, x) => (!w ? 0 : Math.min(1, Math.max(0, (x - w[0] - w[1]) / (w[2] - w[1]))));
// 激活块用的数据：哪个激活、转置没有、梯度用哪个
const ACT_SRC = {
  h0: ['h0', true], h1: ['h1', true], h2: ['h2', true], n1: ['n1', true], n2: ['n2', true], nf: ['nf', true],
  aoT: ['ao', true], actT: ['act', true],
  q: ['q'], k: ['k'], v: ['v'], ao: ['ao'], o: ['o', false, 'h1'], gate: ['gate'], up: ['up'], act: ['act'], f: ['f', false, 'h2'], logits: ['logits'],
};
// 每块参数的输入 / 输出激活（D5 里框出这个参数乘到的那一行 / 那一列）
const IO = { Wq: ['n1', 'q'], Wk: ['n1', 'k'], Wv: ['n1', 'v'], Wo: ['aoT', 'o'], Wg: ['n2', 'gate'], Wu: ['n2', 'up'], Wd: ['actT', 'f'], g1: [null, 'n1'], g2: [null, 'n2'], gf: [null, 'nf'] };
const GROUP_COL = { E: 0xb39dff, g1: 0x6b9bff, g2: 0x6b9bff, gf: 0x6b9bff, Wq: 0x5ef0d4, Wk: 0x5ef0d4, Wv: 0x5ef0d4, Wo: 0x5ef0d4, Wg: 0xffb65c, Wu: 0xffb65c, Wd: 0xffb65c };

const SHAPE = (t, D) => { const p = D.pIndex.get(t); return p.norm ? `[${p.rows}]` : `[${p.rows} × ${p.cols}]`; };
const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

function disposeTree(o) {
  o.traverse((c) => {
    c.geometry?.dispose?.();
    if (c.material && !c.material.userData?.shared) (Array.isArray(c.material) ? c.material : [c.material]).forEach((m) => m.dispose());
    if (c.el) c.el.remove();
  });
}

// 方块的材质：光照下的颜色 + 一点自发光（实例颜色乘 k），暗面也看得出正负
function cubeMaterial(k = 0.35) {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0.08 });
  m.onBeforeCompile = (sh) => {
    sh.fragmentShader = sh.fragmentShader.replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>\n#ifdef USE_INSTANCING_COLOR\n\ttotalEmissiveRadiance += vColor * ${k.toFixed(2)};\n#endif`);
  };
  return m;
}

export class GlassMachine {
  constructor(E, D, app) {
    this.E = E;
    this.D = D;
    this.app = app;
    this.scene = E.scene;
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.hover = null;
    this.csCache = new Map();
    this.gqCache = new Map();
    this.aqCache = new Map();
    this.pk = '';
    this.pt0 = 0;
    this.buildEnv();
    this.buildWeights();
    this.buildActs();
    this.buildSpine();
    this.buildPipes();
    this.buildFeed();
    this.buildHead();
    this.buildLabels();
    this.buildMarks();
    this.fixBounds();
  }

  /* ================================================================ 搭机器 */

  buildEnv() {
    const grid = new THREE.GridHelper(80, 80, 0x1c3350, 0x10203a);
    grid.position.y = -0.32;
    grid.material.transparent = true;
    grid.material.opacity = 0.3;
    this.scene.add(grid);
    const n = 600, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { pos[i * 3] = (Math.random() - 0.5) * 70 + 4; pos[i * 3 + 1] = Math.random() * 30 - 2; pos[i * 3 + 2] = (Math.random() - 0.5) * 40 - 6; }
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.dust = new THREE.Points(pg, new THREE.PointsMaterial({ color: 0x7fd8e8, size: 0.05, transparent: true, opacity: 0.28, depthWrite: false }));
    this.scene.add(this.dust);
  }

  // 一块面板的底板和边框
  plate(b, color, op = 0.9) {
    const g = new THREE.Group();
    const w = b.w + 0.04, h = b.h + 0.04;
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, 0.02), new THREE.MeshStandardMaterial({ color: 0x0a1222, roughness: 0.6, metalness: 0.2, transparent: true, opacity: op }));
    m.position.set(b.x0 + b.w / 2, b.yBot + b.h / 2, -0.012);
    m.userData.noFocus = false;
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(w + 0.02, h + 0.02)), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.4 }));
    e.position.set(b.x0 + b.w / 2, b.yBot + b.h / 2, 0.001);
    g.add(m, e);
    g.edge = e;
    this.root.add(g);
    return g;
  }

  buildWeights() {
    const D = this.D;
    const wb = BLOCKS.filter((b) => b.kind === 'w');
    let n = 0;
    for (const b of wb) n += b.rows * b.cols;
    this.nW = n;
    this.wGi = new Int16Array(n);        // 实例 → 参数下标
    this.wBlk = [];                      // 实例 → 块
    this.wR = new Int16Array(n);
    this.wC = new Int16Array(n);
    this.wX = new Float32Array(n);
    this.wY = new Float32Array(n);
    let i = 0;
    for (const b of wb) {
      const p = D.pIndex.get(b.t);
      b.i0 = i;
      for (let r = 0; r < b.rows; r++) for (let c = 0; c < b.cols; c++) {
        this.wGi[i] = b.mirror ? p.off + c * p.cols + r : p.norm ? p.off + r : p.off + r * p.cols + c;
        this.wBlk[i] = b;
        this.wR[i] = r; this.wC[i] = c;
        this.wX[i] = cellX(b, c); this.wY[i] = cellY(b, r);
        i++;
      }
      b.plate = this.plate(b, GROUP_COL[b.t], b.mirror ? 0.55 : 0.9);
    }
    const geo = new THREE.BoxGeometry(FACE, FACE, 1);
    geo.translate(0, 0, 0.5);
    this.cubes = new THREE.InstancedMesh(geo, cubeMaterial(0.34), n);
    this.cubes.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
    this.cubes.frustumCulled = false;
    this.cubes.userData.pick = (hit) => this.pickW(hit.instanceId);
    const gg = new THREE.BoxGeometry(FACE * 1.32, FACE * 1.32, 1);
    gg.translate(0, 0, 0.5);
    this.glow = new THREE.InstancedMesh(gg, new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }), n);
    this.glow.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
    this.glow.frustumCulled = false;
    this.glow.userData.noFocus = true;
    this.glow.raycast = () => {};
    for (const m of [this.cubes, this.glow]) {
      const a = m.instanceMatrix.array;
      for (let j = 0; j < n; j++) { a[j * 16 + 15] = 1; }
      m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      m.instanceColor.setUsage(THREE.DynamicDrawUsage);
    }
    this.root.add(this.cubes, this.glow);
    this.E.pickables.push(this.cubes);
    // 扫描线：前向时“这一列正在和输入做点积”
    this.scan = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0xbff8ff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.scan.raycast = () => {};
    this.root.add(this.scan);
  }

  buildActs() {
    const ab = BLOCKS.filter((b) => b.kind === 'a');
    let n = 0;
    for (const b of ab) n += b.rows * b.cols;
    this.nA = n;
    this.aBlk = [];
    this.aR = new Int16Array(n);
    this.aC = new Int16Array(n);
    const geo = new THREE.PlaneGeometry(1, 1);
    this.acts = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({ color: 0xffffff }), n);
    this.acts.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
    this.acts.frustumCulled = false;
    const a = this.acts.instanceMatrix.array;
    let i = 0;
    for (const b of ab) {
      b.i0 = i;
      const sx = b.px * (b.spine ? 0.84 : 0.86), sy = b.py * 0.86;
      for (let r = 0; r < b.rows; r++) for (let c = 0; c < b.cols; c++) {
        this.aBlk[i] = b; this.aR[i] = r; this.aC[i] = c;
        const o = i * 16;
        a[o] = sx; a[o + 5] = sy; a[o + 10] = 1; a[o + 12] = cellX(b, c); a[o + 13] = cellY(b, r); a[o + 14] = b.spine ? 0.26 : 0.012; a[o + 15] = 1;
        i++;
      }
      if (!b.spine) b.plate = this.plate(b, 0x2c4a70, 0.75);
    }
    this.acts.instanceMatrix.needsUpdate = true;
    this.acts.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.acts.userData.pick = (hit) => this.pickA(hit.instanceId);
    this.root.add(this.acts);
    this.E.pickables.push(this.acts);
  }

  // 残差流：8 根光柱 + 玻璃外壳 + 三个 RMSNorm 环 + 两个 ⊕
  buildSpine() {
    const g = (this.spine = new THREE.Group());
    this.root.add(g);
    const h = SPINE.top - 0.32;
    const cg = new THREE.CylinderGeometry(0.02, 0.02, 1, 8, 1, true);
    cg.translate(0, 0.5, 0);
    this.cols = [];
    for (let p = 0; p < 8; p++) {
      const m = new THREE.Mesh(cg, new THREE.MeshBasicMaterial({ color: 0x4fd8c0, transparent: true, opacity: 0.4, depthWrite: false }));
      m.position.set(xPos(p), 0.32, 0.12);
      m.scale.y = h;
      m.raycast = () => {};
      g.add(m);
      this.cols.push(m);
    }
    const W = SPINE.x1 - SPINE.x0 + 0.16;
    const shell = new THREE.Mesh(new THREE.BoxGeometry(W, h, 0.5), new THREE.MeshStandardMaterial({ color: 0x1a2c50, emissive: 0x5ef0d4, emissiveIntensity: 0.02, transparent: true, opacity: 0.06, depthWrite: false, roughness: 0.3 }));
    shell.position.set((SPINE.x0 + SPINE.x1) / 2, 0.32 + h / 2, 0.12);
    shell.raycast = () => {};
    const edge = new THREE.LineSegments(new THREE.EdgesGeometry(shell.geometry), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.16 }));
    edge.position.copy(shell.position);
    g.add(shell, edge);
    // 环：扁的椭圆环套在 8 根光柱外面
    const ring = (y, color) => {
      const m = new THREE.Mesh(new THREE.TorusGeometry(1, 0.03, 8, 48), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.5, depthWrite: false }));
      m.rotation.x = Math.PI / 2;
      m.scale.set(W / 2 + 0.06, 0.34, 1);
      m.position.set((SPINE.x0 + SPINE.x1) / 2, y, 0.12);
      m.raycast = () => {};
      g.add(m);
      return m;
    };
    this.rings = { norm1: ring(Y.ring1, 0x6b9bff), norm2: ring(Y.ring2, 0x6b9bff), normf: ring(Y.ringf, 0x6b9bff), add1: ring(Y.add1, 0xe9eff9), add2: ring(Y.add2, 0xe9eff9) };
    // 脉冲：前向（青）往上，反向（玫红）往下
    const pg = new THREE.TorusGeometry(0.06, 0.016, 6, 20);
    const mk = (color) => { const m = new THREE.InstancedMesh(pg, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthWrite: false }), 8); m.frustumCulled = false; m.raycast = () => {}; g.add(m); return m; };
    this.pulseF = mk(0x9ffcff);
    this.pulseB = mk(0xff7aa0);
    // 每一层底下一块很淡的玻璃地板（和推理页的层板一样只靠边线勾出来），看得出机器分成几层
    const floor = (y, x1) => {
      const w = x1 - SPINE.x0 + 0.5, d = 1.7;
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, 0.025, d), new THREE.MeshStandardMaterial({ color: 0x22346a, transparent: true, opacity: 0.05, depthWrite: false, roughness: 0.4 }));
      m.position.set(SPINE.x0 - 0.25 + w / 2, y, -0.25);
      m.raycast = () => {};
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.12 }));
      e.position.copy(m.position);
      e.raycast = () => {};
      this.root.add(m, e);
    };
    floor(Y.ring1 + 0.06, BLOCK.Wo.x1 + 0.2);
    floor(Y.ring2 + 0.1, BLOCK.Wd.x1 + 0.2);
    floor(Y.ringf + 0.1, SHELF.x0 + 20 * SHELF.px + 0.6);
  }

  tube(curve, color, r = 0.014) {
    const m = new THREE.Mesh(new THREE.TubeGeometry(curve, 28, r, 5, false), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.16, depthWrite: false }));
    m.raycast = () => {};
    this.root.add(m);
    return m;
  }

  // 连接管：残差 → γ（分出去）、o / f → ⊕（加回来）、ao → aoᵀ / act → actᵀ（转置一下当下一块矩阵的输入）、查表的 8 束光
  buildPipes() {
    const V = (x, y, z = 0) => new THREE.Vector3(x, y, z);
    const branch = (y, yb) => this.tube(new THREE.QuadraticBezierCurve3(V(SPINE.x1 + 0.12, y, 0.12), V(0.36, y, 0.08), V(0.36, yb - 0.02, 0)), 0x6b9bff);
    this.pipes = {
      norm1: branch(Y.ring1, BLOCK.g1.yBot), norm2: branch(Y.ring2, BLOCK.g2.yBot), normf: branch(Y.ringf, BLOCK.gf.yBot),
      add1: this.tube(new THREE.CatmullRomCurve3([V(BLOCK.o.x0 + 0.2, Y.add1, -0.04), V(BLOCK.o.x0 - 0.5, Y.add1, -0.6), V(0.4, Y.add1, -0.6), V(SPINE.x1 + 0.12, Y.add1, 0.12)]), 0xe9eff9),
      add2: this.tube(new THREE.CatmullRomCurve3([V(BLOCK.f.x0 + 0.2, Y.add2, -0.04), V(BLOCK.f.x0 - 0.5, Y.add2, -0.6), V(0.4, Y.add2, -0.6), V(SPINE.x1 + 0.12, Y.add2, 0.12)]), 0xe9eff9),
      aoT: this.tube(new THREE.QuadraticBezierCurve3(V(BLOCK.ao.x1 + 0.04, (BLOCK.ao.yTop + BLOCK.ao.yBot) / 2), V(BLOCK.aoT.x0 + 0.48, (BLOCK.ao.yTop + BLOCK.ao.yBot) / 2), V(BLOCK.aoT.x0 + 0.48, BLOCK.aoT.yTop + 0.04)), 0x8fb8ff),
      actT: this.tube(new THREE.QuadraticBezierCurve3(V(BLOCK.act.x1 + 0.04, (BLOCK.act.yTop + BLOCK.act.yBot) / 2), V(BLOCK.actT.x0 + 0.48, (BLOCK.act.yTop + BLOCK.act.yBot) / 2), V(BLOCK.actT.x0 + 0.48, BLOCK.actT.yTop + 0.04)), 0x8fb8ff),
    };
    // 查表：第 0 段每个位置的字块 → E 里那一行（固定样例每一步都一样，形状不变）
    const D = this.D, E = BLOCK.E;
    this.look = [];
    for (let p = 0; p < 8; p++) {
      const r = D.fixed[p];
      const a = V(xPos(p), Y.tray + 0.14, 0.14), b = V(E.x0 - 0.03, cellY(E, r), 0.03);
      const m = this.tube(new THREE.QuadraticBezierCurve3(a, V((a.x + b.x) / 2 + 0.4, Math.max(a.y, b.y) + 0.35, 0.5), b), 0x5ef0d4, 0.01);
      m.material.opacity = 0;
      this.look.push(m);
    }
  }

  /* ---------------------------------------------------------------- 进料：语料轮、托盘、字块 */

  // 字块：轮子上 25 块 + 托盘里 64 块 + 第 0 段的 8 个答案，方块和字面各用一个 InstancedMesh（字面查一张字表贴图）
  buildFeed() {
    const D = this.D, NS = D.stream.length, B = D.B, T = D.T;
    const g = (this.feed = new THREE.Group());
    this.root.add(g);
    // 字表：20 个字 × 4 种颜色（轮子 / 第 0 段 / 其余 7 段 / 答案）
    const COLS = ['#b4bed2', '#ffd9a6', '#b8cdf5', '#bff8ec'], CELL = 96, V = D.V;
    const cv = document.createElement('canvas');
    cv.width = CELL * V; cv.height = CELL * COLS.length;
    const cx = cv.getContext('2d');
    cx.font = `600 ${Math.round(CELL * 0.62)}px "PingFang SC","Microsoft YaHei","Noto Sans SC",sans-serif`;
    cx.textAlign = 'center'; cx.textBaseline = 'middle';
    COLS.forEach((c, r) => { cx.fillStyle = c; for (let v = 0; v < V; v++) cx.fillText(D.ch(v), CELL * v + CELL / 2, CELL * r + CELL * 0.54); });
    const atlas = new THREE.CanvasTexture(cv);
    atlas.colorSpace = THREE.SRGBColorSpace;
    atlas.anisotropy = 4;
    this.nTok = NS + B * T + T;
    this.tok0 = { wheel: 0, tray: NS, tgt: NS + B * T };
    const bodyGeo = new THREE.BoxGeometry(0.26, 0.22, 0.2);
    this.tokBody = new THREE.InstancedMesh(bodyGeo, cubeMaterial(0.3), this.nTok);
    this.tokBody.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(this.nTok * 3), 3);
    const faceGeo = new THREE.PlaneGeometry(0.22, 0.19);
    this.tokUv = new THREE.InstancedBufferAttribute(new Float32Array(this.nTok * 2), 2);
    faceGeo.setAttribute('aUv', this.tokUv);
    const fm = new THREE.MeshBasicMaterial({ map: atlas, transparent: true, depthWrite: false });
    fm.onBeforeCompile = (sh) => {
      sh.vertexShader = sh.vertexShader.replace('#include <uv_pars_vertex>', '#include <uv_pars_vertex>\nattribute vec2 aUv;')
        .replace('#include <uv_vertex>', `#include <uv_vertex>\n\tvMapUv = vec2((aUv.x + vMapUv.x) / ${V}.0, 1.0 - (aUv.y + 1.0 - vMapUv.y) / ${COLS.length}.0);`);
    };
    this.tokFace = new THREE.InstancedMesh(faceGeo, fm, this.nTok);
    for (const m of [this.tokBody, this.tokFace]) { m.frustumCulled = false; m.instanceMatrix.setUsage(THREE.DynamicDrawUsage); g.add(m); }
    this.tokFace.raycast = () => {};
    const bc = this.tokBody.instanceColor.array;
    const setCol = (i, c, k) => { const col = new THREE.Color(c); bc[i * 3] = col.r * k; bc[i * 3 + 1] = col.g * k; bc[i * 3 + 2] = col.b * k; };
    for (let i = 0; i < NS; i++) { setCol(i, 0x8fa6d6, 0.32); this.tokUv.setXY(i, D.stream[i], 0); }
    for (let b = 0; b < B; b++) for (let i = 0; i < T; i++) setCol(NS + b * T + i, b === 0 ? 0xffb65c : 0x6b9bff, b === 0 ? 0.42 : 0.26);
    for (let i = 0; i < T; i++) setCol(NS + B * T + i, 0x5ef0d4, 0.4);
    this.tokBody.instanceColor.needsUpdate = true;
    this.tokBody.userData.pick = (hit) => {
      const i = hit.instanceId;
      if (i < NS) return { type: 'wheel', i };
      if (i < NS + B * T) return { type: 'tok', b: Math.floor((i - NS) / T), i: (i - NS) % T };
      return { type: 'tgt', i: i - NS - B * T };
    };
    this.E.pickables.push(this.tokBody);
    // 语料轮
    const wheel = (this.wheel = new THREE.Group());
    wheel.position.set(WHEEL.x, WHEEL.y, WHEEL.z);
    g.add(wheel);
    const rim = new THREE.Mesh(new THREE.TorusGeometry(WHEEL.r, 0.012, 6, 64), new THREE.MeshBasicMaterial({ color: 0x2c4a70, transparent: true, opacity: 0.6 }));
    rim.raycast = () => {};
    const hub = new THREE.Mesh(new THREE.CircleGeometry(WHEEL.r - 0.24, 40), new THREE.MeshBasicMaterial({ color: 0x0a1222, transparent: true, opacity: 0.55, depthWrite: false }));
    hub.position.z = -0.06;
    hub.raycast = () => {};
    wheel.add(rim, hub);
    this.wheelArcs = new THREE.Group();
    wheel.add(this.wheelArcs);
    this.arcKey = '';
    // 托盘
    const tray = new THREE.Mesh(new THREE.BoxGeometry(SPINE.x1 - SPINE.x0 + 0.4, 0.05, 2.9), new THREE.MeshStandardMaterial({ color: 0x0c1629, metalness: 0.6, roughness: 0.35 }));
    tray.position.set((SPINE.x0 + SPINE.x1) / 2, -0.02, -0.95);
    const te = new THREE.LineSegments(new THREE.EdgesGeometry(tray.geometry), new THREE.LineBasicMaterial({ color: 0x2c6b8a, transparent: true, opacity: 0.6 }));
    te.position.copy(tray.position);
    g.add(tray, te);
    this.batchT = -1;
  }

  // 一个位置在托盘里的坐标（第 b 段往后排，一排比一排高一点，像阶梯座位）
  slot(b, i) { return new THREE.Vector3(xPos(i), TRAY.y + b * 0.085, -b * TRAY.dz); }
  wheelPos(si) {
    const NS = this.D.stream.length, a = Math.PI / 2 - (((si % NS) + NS) % NS / NS) * Math.PI * 2;
    return new THREE.Vector3(WHEEL.x + Math.cos(a) * WHEEL.r, WHEEL.y + Math.sin(a) * WHEEL.r, WHEEL.z);
  }

  setBatch(t) {
    if (t === this.batchT) return;
    this.batchT = t;
    const D = this.D, bt = D.batch(t), o = this.tok0;
    for (let b = 0; b < D.B; b++) for (let i = 0; i < D.T; i++) this.tokUv.setXY(o.tray + b * D.T + i, bt[b][i], b === 0 ? 1 : 2);
    for (let i = 0; i < D.T; i++) this.tokUv.setXY(o.tgt + i, bt[0][i + 1], 3);
    this.tokUv.needsUpdate = true;
  }

  // 一个字块放到 (x, y, z)，缩放 s，绕 y 转 ry；s = 0 藏起来
  placeTok(i, pos, s, ry = 0) {
    const m = this.tmpM || (this.tmpM = new THREE.Matrix4()), q = this.tmpQ || (this.tmpQ = new THREE.Quaternion()), sc = this.tmpS || (this.tmpS = new THREE.Vector3());
    q.setFromAxisAngle(this.yAxis || (this.yAxis = new THREE.Vector3(0, 1, 0)), ry);
    m.compose(pos, q, sc.set(s, s, s));
    this.tokBody.setMatrixAt(i, m);
    // 字面贴在方块正面
    const f = this.tmpF || (this.tmpF = new THREE.Vector3());
    f.set(0, 0, 0.102 * s).applyQuaternion(q).add(pos);
    m.compose(f, q, sc);
    this.tokFace.setMatrixAt(i, m);
  }

  /* ---------------------------------------------------------------- 输出：概率架和损失 */

  buildHead() {
    const D = this.D, T = D.T, V = D.V;
    const g = (this.head = new THREE.Group());
    this.root.add(g);
    const w = V * SHELF.px + 0.2, d = T * SHELF.pz + 0.2;
    const shelf = new THREE.Mesh(new THREE.BoxGeometry(w, 0.04, d), new THREE.MeshStandardMaterial({ color: 0x0c1629, metalness: 0.5, roughness: 0.4, transparent: true, opacity: 0.92 }));
    shelf.position.set(SHELF.x0 + V * SHELF.px / 2, SHELF.y - 0.02, SHELF.z0 + (T - 1) * SHELF.pz / 2);
    const se = new THREE.LineSegments(new THREE.EdgesGeometry(shelf.geometry), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.3 }));
    se.position.copy(shelf.position);
    g.add(shelf, se);
    const bg = new THREE.BoxGeometry(SHELF.px * 0.66, 1, SHELF.pz * 0.66);
    bg.translate(0, 0.5, 0);
    this.bars = new THREE.InstancedMesh(bg, cubeMaterial(0.45), T * V);
    this.bars.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(T * V * 3), 3);
    this.bars.frustumCulled = false;
    const a = this.bars.instanceMatrix.array;
    for (let i = 0; i < T * V; i++) a[i * 16 + 15] = 1;
    this.bars.userData.pick = (hit) => ({ type: 'bar', p: Math.floor(hit.instanceId / V), v: hit.instanceId % V });
    this.E.pickables.push(this.bars);
    g.add(this.bars);
    // 正确答案：架子上的绿框
    const mg = new THREE.RingGeometry(SHELF.px * 0.4, SHELF.px * 0.48, 4, 1, Math.PI / 4);
    mg.rotateX(-Math.PI / 2);
    this.marks = new THREE.InstancedMesh(mg, new THREE.MeshBasicMaterial({ color: 0x8ee07a, transparent: true, opacity: 0.85, depthWrite: false }), T);
    this.marks.frustumCulled = false;
    this.marks.raycast = () => {};
    g.add(this.marks);
    // 每个位置的 −ln p（架子右边一排）和整批的损失（一根粗管）
    const lg = new THREE.BoxGeometry(0.16, 1, 0.16);
    lg.translate(0, 0.5, 0);
    this.nll = new THREE.InstancedMesh(lg, cubeMaterial(0.5), T);
    this.nll.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(T * 3), 3);
    this.nll.frustumCulled = false;
    const na = this.nll.instanceMatrix.array;
    for (let i = 0; i < T; i++) na[i * 16 + 15] = 1;
    this.nll.userData.pick = (hit) => ({ type: 'nll', p: hit.instanceId });
    this.E.pickables.push(this.nll);
    g.add(this.nll);
    const glass = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, GAUGE.hMax, 24, 1, true), new THREE.MeshStandardMaterial({ color: 0x1a2c50, transparent: true, opacity: 0.18, depthWrite: false, side: THREE.DoubleSide }));
    glass.position.set(GAUGE.x, GAUGE.y + GAUGE.hMax / 2, GAUGE.z);
    glass.raycast = () => {};
    const fg = new THREE.CylinderGeometry(0.16, 0.16, 1, 24);
    fg.translate(0, 0.5, 0);
    this.gauge = new THREE.Mesh(fg, new THREE.MeshStandardMaterial({ color: 0x3a2a10, emissive: 0xffb65c, emissiveIntensity: 0.35, roughness: 0.4 }));
    this.gauge.position.set(GAUGE.x, GAUGE.y, GAUGE.z);
    this.gauge.scale.y = 0.001;
    this.gauge.userData.pick = { type: 'gauge' };
    this.E.pickables.push(this.gauge);
    const ln = new THREE.Mesh(new THREE.TorusGeometry(0.25, 0.012, 6, 32), new THREE.MeshBasicMaterial({ color: 0xff6b93, transparent: true, opacity: 0.7 }));
    ln.rotation.x = Math.PI / 2;
    ln.position.set(GAUGE.x, GAUGE.y + GAUGE.hMax * (GAUGE.lnV / 3.2), GAUGE.z);
    g.add(glass, this.gauge, ln);
    // 字的刻度（架子前沿）和每一行“看到的字 → 正确答案”（架子左边），画成贴图
    const chars = D.chars.map((c) => c);
    const cv = document.createElement('canvas');
    cv.width = 40 * V; cv.height = 64;
    const cx = cv.getContext('2d');
    cx.font = '600 27px "PingFang SC","Microsoft YaHei","Noto Sans SC",sans-serif';
    cx.textAlign = 'center'; cx.textBaseline = 'middle'; cx.fillStyle = '#9aa6bf';
    chars.forEach((c, v) => cx.fillText(c, 40 * v + 20, 34));
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    const strip = new THREE.Mesh(new THREE.PlaneGeometry(V * SHELF.px, V * SHELF.px * 64 / (40 * V)), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
    strip.position.set(SHELF.x0 + V * SHELF.px / 2, SHELF.y + 0.005, SHELF.z0 + (T - 0.35) * SHELF.pz);
    strip.rotation.x = -Math.PI / 2;
    strip.raycast = () => {};
    g.add(strip);
    const rc = document.createElement('canvas');
    rc.width = 160; rc.height = 64 * T;
    const rx = rc.getContext('2d');
    rx.font = '600 38px "PingFang SC","Microsoft YaHei","Noto Sans SC",sans-serif';
    rx.textBaseline = 'middle';
    for (let p = 0; p < T; p++) {
      rx.textAlign = 'right'; rx.fillStyle = '#7a859e'; rx.fillText(D.ch(D.fixed[p]), 62, 64 * p + 34);
      rx.textAlign = 'center'; rx.fillStyle = '#4b5572'; rx.fillText('→', 88, 64 * p + 32);
      rx.textAlign = 'left'; rx.fillStyle = '#8ee07a'; rx.fillText(D.ch(D.fixed[p + 1]), 110, 64 * p + 34);
    }
    const rt = new THREE.CanvasTexture(rc);
    rt.colorSpace = THREE.SRGBColorSpace;
    rt.anisotropy = 4;
    const rows = new THREE.Mesh(new THREE.PlaneGeometry(T * SHELF.pz * 160 / (64 * T), T * SHELF.pz), new THREE.MeshBasicMaterial({ map: rt, transparent: true, depthWrite: false }));
    rows.rotation.x = -Math.PI / 2;
    rows.position.set(SHELF.x0 - 0.42, SHELF.y + 0.005, SHELF.z0 + (T - 1) * SHELF.pz / 2);
    rows.raycast = () => {};
    g.add(rows);
  }

  /* ---------------------------------------------------------------- 标签 */

  buildLabels() {
    const D = this.D;
    const add = (html, cls, x, y, z, cx, cy) => { const o = label(html, `lbl ${cls}`); o.position.set(x, y, z); o.center.set(cx, cy); this.root.add(o); return o; };
    this.lw = {};
    for (const b of BLOCKS.filter((x) => x.kind === 'w')) {
      const nm = b.mirror ? 'Eᵀ' : TENSOR_LABEL[b.t];
      const sh = b.mirror ? '[16 × 20]' : SHAPE(b.t, D);
      const sub = b.mirror ? L('和最下面的 E 是同一张表', 'same table as E at the bottom') : '';
      const isG = b.t.startsWith('g');
      const o = add(`${nm}<small>${sh}${sub ? ` · ${sub}` : ''}</small>`, 'part gw', isG ? b.x0 + 0.06 : b.x0, b.yBot - 0.06, 0.05, isG ? 0.5 : 0, 0);
      o.short = nm;
      o.full = o.el.innerHTML;
      this.lw[b.id] = o;
    }
    const AL = { h0: 'h₀', h1: 'h₁', h2: 'h₂', n1: 'n₁', n2: 'n₂', nf: 'n_f', q: 'q', k: 'k', v: 'v', att0: L('注意力 头 0', 'attn head 0'), att1: L('头 1', 'head 1'), ao: 'ao', aoT: 'aoᵀ', o: 'o', gate: 'gate', up: 'up', act: 'silu(gate) ⊙ up', actT: 'actᵀ', f: 'f', logits: 'logits' };
    this.la = {};
    for (const b of BLOCKS.filter((x) => x.kind === 'a')) {
      const shp = `[${b.rows} × ${b.cols}]`;
      let o;
      if (b.spine) o = add(`${AL[b.id]}<small>${L('残差', 'residual')} ${shp}</small>`, 'num ga', SPINE.x0 - 0.12, (b.yTop + b.yBot) / 2, 0.2, 1, 0.5);
      else if (b.id === 'aoT' || b.id === 'actT') o = add(`${AL[b.id]}`, 'num ga', b.x0, b.yBot - 0.06, 0.05, 0, 0);
      else if (b.id === 'logits') o = add(`${AL[b.id]}<small>${shp}</small>`, 'num ga', b.x1 + 0.08, (b.yTop + b.yBot) / 2, 0.05, 0, 0.5);
      else if (b.id.startsWith('att')) o = add(`${AL[b.id]}`, 'num ga', b.x0, b.yTop + 0.05, 0.05, 0, 1);
      else o = add(`${AL[b.id]}<small>${shp}</small>`, 'num ga', b.x0, b.yTop + 0.05, 0.05, 0, 1);
      this.la[b.id] = o;
    }
    this.lr = {
      norm1: add('RMSNorm', 'hint gr', SPINE.x1 + 0.2, Y.ring1 - 0.04, 0.2, 0, 0),
      norm2: add('RMSNorm', 'hint gr', SPINE.x1 + 0.2, Y.ring2 - 0.04, 0.2, 0, 0),
      normf: add('RMSNorm', 'hint gr', SPINE.x1 + 0.2, Y.ringf - 0.04, 0.2, 0, 0),
      add1: add(L('⊕ 加回残差', '⊕ add to residual'), 'hint gr', SPINE.x0 - 0.12, Y.add1, 0.2, 1, 0.5),
      add2: add(L('⊕ 加回残差', '⊕ add to residual'), 'hint gr', SPINE.x0 - 0.12, Y.add2, 0.2, 1, 0.5),
    };
    this.lGauge = add('', 'big gg', GAUGE.x + 0.3, GAUGE.y + 0.6, 0, 0, 0.5);
    this.lClip = add('', 'num gc', GAUGE.x + 0.3, GAUGE.y - 0.3, 0.2, 0, 0);
    this.lGap = add('', 'num gt', WHEEL.x, WHEEL.y - WHEEL.r - 0.25, 0.3, 0.5, 0);
    this.lWheel = add(L('《静夜思》首尾相接 · 25 个字', '静夜思 looped · 25 characters'), 'hint gw2', WHEEL.x, WHEEL.y + WHEEL.r + 0.2, 0.3, 0.5, 1);
    this.lTray = add(L('这一步的批次：8 段 × 8 个字（最前一排是第 0 段）', 'This step’s batch: 8 windows × 8 characters (front row = window 0)'), 'hint gw2', (SPINE.x0 + SPINE.x1) / 2, -0.12, 0.6, 0.5, 0);
    this.lShelf = add(L('20 个字的概率 · 绿框 = 正确答案', 'Probabilities over 20 characters · green = answer'), 'hint gs', SHELF.x0, SHELF.y + 0.02, SHELF.z0 - 0.3, 0, 1);
    this.lSel = add('', 'num gsel', 0, 0, 0, 0, 1.25);
    this.lFactory = add(L('出厂状态 · 还没训练', 'Factory state · untrained'), 'title gf2', (BOUNDS.x0 + BOUNDS.x1) / 2, BOUNDS.y1 - 0.4, 0, 0.5, 1);
  }

  // 选中参数的白框、悬停的框、D5 里的输入 / 输出切片框
  buildMarks() {
    const box = (color) => {
      const m = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95, depthTest: false }));
      m.renderOrder = 10;
      m.visible = false;
      m.raycast = () => {};
      this.root.add(m);
      return m;
    };
    this.mHover = box(0xffffff);
    this.mSel = box(0xffd18a);
    this.mSel2 = box(0xffd18a);
    this.mIn = box(0x7cc4ff);
    this.mOut = box(0x5ef0d4);
  }

  // 拾取用的包围球：装得下整台机器（实例每帧都在动，不必每帧重算）
  fixBounds() {
    const sp = new THREE.Sphere(new THREE.Vector3((BOUNDS.x0 + BOUNDS.x1) / 2, (BOUNDS.y0 + BOUNDS.y1) / 2, 0), 32);
    for (const m of [this.cubes, this.acts, this.bars, this.nll, this.tokBody]) m.boundingSphere = sp.clone();
  }

  /* ================================================================ 状态 */

  // 暂停时单步过来：这一步的动画自己播一遍再停住；播放时跟着时间轴走
  animP(st, t) {
    const key = `${st.depth}:${st.k}:${st.i}`;
    if (this.pk !== key) { this.pk = key; this.pt0 = t; }
    const dur = { 'g-init': 1.6, 'g-run': 3.2, 'g-step': 3, 'g-op': 1.5, 'g-mat': 2.2, 'g-param': 1.6 }[st.view] || 1.5;
    return st.playing ? st.p : Math.max(st.p, Math.min(1, (t - this.pt0) / dur));
  }

  // 这一刻机器走到了哪：进料、前向 / 反向进度（以算子为单位，连续）、损失、每个张量的更新
  flow(st, p) {
    const s = st.step;
    const F = { feed: -1, shift: 0, fwd: -1, bwd: -1, loss: 0, lossUpTo: 0, lossPos: -1, lossMean: 0, clip: 0, upd: {}, next: 0 };
    if (s.ph === 'init') return F;
    const all = (dw, done) => { for (const t of UPD_TENSORS) F.upd[t] = { dw, done }; };
    if (s.ph === 'run') {
      F.feed = seg(p, 0, 0.1); F.shift = seg(p, 0.06, 0.12);
      F.fwd = 10 * seg(p, 0.12, 0.42);
      F.loss = seg(p, 0.4, 0.48); F.lossUpTo = 8 * F.loss; F.lossMean = F.loss;
      F.bwd = 10 * seg(p, 0.5, 0.74);
      all(seg(p, 0.76, 0.84), seg(p, 0.86, 0.95));
      F.next = seg(p, 0.95, 1);
      return F;
    }
    const PH = G_PHASES.indexOf(s.ph);
    F.feed = 1; F.shift = 1;
    if (PH >= 2) { F.fwd = 10; F.loss = 1; F.lossUpTo = 8; F.lossMean = 1; }
    if (PH >= 4) { F.bwd = 10; F.clip = 1; }
    if (!s.sub) {
      if (s.ph === 'batch') { F.feed = easeOut(seg(p, 0, 0.6)); F.shift = easeOut(seg(p, 0.62, 0.92)); }
      else if (s.ph === 'fwd') F.fwd = Math.min(10, p * 10.6);
      else if (s.ph === 'loss') { F.lossUpTo = 8 * seg(p, 0, 0.5); F.lossMean = easeOut(seg(p, 0.5, 0.85)); }
      else if (s.ph === 'bwd') F.bwd = Math.min(10, p * 10.6);
      else if (s.ph === 'upd') { F.clip = seg(p, 0, 0.15); all(seg(p, 0.18, 0.5), seg(p, 0.56, 0.92)); }
      return F;
    }
    switch (s.ph) {
      case 'batch':
        if (s.sub === 'pick') { F.feed = easeOut(p); F.shift = 0; } else F.shift = easeOut(seg(p, 0.1, 0.7));
        break;
      case 'fwd': case 'bwd': {
        const ops = s.ph === 'fwd' ? FWD_OPS : BWD_OPS, oi = ops.indexOf(s.sub);
        let v;
        if (s.mi) v = oi + 1;
        else if (s.t) {
          const ts = s.ph === 'fwd' ? FWD_TENSORS[s.sub] : BWD_TENSORS[s.sub];
          v = oi + (ts.indexOf(s.t) + easeInOut(seg(p, 0, 0.8))) / ts.length;
        } else v = oi + easeInOut(seg(p, 0, 0.75));
        if (s.ph === 'fwd') F.fwd = v; else F.bwd = v;
        break;
      }
      case 'loss':
        if (s.sub === 'pos') { F.lossPos = s.i; F.lossUpTo = s.i + easeOut(seg(p, 0, 0.5)); F.lossMean = 0; }
        else F.lossMean = easeOut(seg(p, 0, 0.6));
        break;
      case 'upd': {
        if (s.sub === 'clip') { F.clip = seg(p, 0, 0.6); break; }
        const ti = UPD_TENSORS.indexOf(s.t);
        for (let j = 0; j < ti; j++) F.upd[UPD_TENSORS[j]] = { dw: 1, done: 1 };
        if (s.mi) {
          const mi = ADAM_SUBS.indexOf(s.mi);
          F.upd[s.t] = mi < 4 ? { dw: 0, done: 0 } : mi === 4 ? { dw: easeOut(p), done: 0 } : { dw: 1, done: easeInOut(seg(p, 0.1, 0.8)) };
        } else F.upd[s.t] = { dw: seg(p, 0, 0.45), done: seg(p, 0.5, 0.95) };
        break;
      }
    }
    return F;
  }

  // 第 k 帧的色标：矩阵 / γ 各一个（此刻 |w| 的 98.5% 分位）
  cs(k, W) {
    if (this.csCache.has(k)) return this.csCache.get(k);
    const D = this.D, mats = [], gs = [];
    for (const p of D.params) for (let i = p.off; i < p.off + p.n; i++) (p.norm ? gs : mats).push(Math.abs(p.norm ? W[i] - 1 : W[i]));
    const q = (a) => { a.sort((x, y) => x - y); return a[Math.floor(0.985 * (a.length - 1))]; };
    const r = { mat: Math.max(0.02, q(mats)), g: Math.max(0.02, q(gs)) };
    this.csCache.set(k, r);
    return r;
  }
  // 第 k 帧每个张量 |∂L/∂w| 的 99% 分位（梯度发光的满格）
  gq(k, G) {
    if (this.gqCache.has(k)) return this.gqCache.get(k);
    const out = {};
    for (const p of this.D.params) out[p.name] = absQuantile(G, 0.99, p.off, p.off + p.n) || 1e-12;
    this.gqCache.set(k, out);
    if (this.gqCache.size > 12) this.gqCache.delete(this.gqCache.keys().next().value);
    return out;
  }
  aq(k, key, arr) {
    const ck = `${k}:${key}`;
    if (this.aqCache.has(ck)) return this.aqCache.get(ck);
    const v = absQuantile(arr, 0.99) || 1e-9;
    this.aqCache.set(ck, v);
    if (this.aqCache.size > 400) this.aqCache.delete(this.aqCache.keys().next().value);
    return v;
  }

  // 最近的一个已经载入的权重帧（还没到时先顶上，D1 拖得很快时用）
  nearW(k) {
    const D = this.D;
    if (D.has('w', k)) return k;
    for (let d = 1; d < D.NF; d++) { if (k - d >= 0 && D.has('w', k - d)) return k - d; if (k + d < D.NF && D.has('w', k + d)) return k + d; }
    return -1;
  }

  /* ================================================================ 每帧 */

  update(st, dt, t) {
    const D = this.D, s = st.step;
    this.small = this.E.w < 700;
    const p = this.animP(st, t);
    const F = (this.F = this.flow(st, p));
    const k = st.k, tStep = D.FR[k];
    const init = s.ph === 'init';
    const kw = init ? -1 : this.nearW(k);
    const haveF = !init && D.has('f', k);
    this.st = st; this.p = p;
    if (!init) this.setBatch(tStep);
    this.updWeights(st, F, kw, haveF, p, t);
    this.updActs(st, F, haveF);
    this.updSpine(st, F, haveF, t);
    this.updFeed(st, F);
    this.updHead(st, F, kw);
    this.updLabels(st, F);
    this.updMarks(st);
    const close = st.view === 'g-mat' || st.view === 'g-param';
    const bl = close ? 0.18 : st.view === 'g-op' ? 0.24 : 0.3;
    this.E.bloom.strength += (bl - this.E.bloom.strength) * Math.min(1, dt * 3);
    this.dust.rotation.y += dt * 0.008;
  }

  // 2,928 个方块（+ Eᵀ 镜像）：值 → 颜色 / 厚度；梯度 → 玫红光；Δw → 往前顶 / 往后沉 + 紫光
  updWeights(st, F, kw, haveF, p, t) {
    const D = this.D, s = st.step, k = st.k;
    const ma = this.cubes.instanceMatrix.array, ca = this.cubes.instanceColor.array;
    const ga = this.glow.instanceMatrix.array, gca = this.glow.instanceColor.array;
    let W, W1 = null, G = null, DW = null, lr = 0.01, csM, csG, gqs = null, initMode = null, reveal = 1;
    if (s.ph === 'init') {
      initMode = s.sub === 'zero' ? 'zero' : s.sub === 'big' ? 'big' : 'normal';
      W = initMode === 'big' ? D.bW0 : D.w0;
      csM = INIT_CS; csG = 0.05;
      if (s.sub === 'hist') reveal = easeOut(Math.min(1, p * 1.5));
      // 换初始化的时候，方块从上一种慢慢变过去
      if (this.initMode !== initMode) { this.initFrom = this.initMode || initMode; this.initMode = initMode; this.initT0 = t; }
      this.initMix = Math.min(1, (t - (this.initT0 ?? t)) / 0.9);
    } else {
      this.initMode = null;
      if (kw < 0) W = D.w0;
      else {
        W = D.W(kw);
        const c = this.cs(kw, W);
        csM = c.mat; csG = c.g;
        if (F.next > 0 && kw === k && k < D.NF - 1 && D.has('w', k + 1)) W1 = D.W(k + 1);
      }
      if (kw < 0) { csM = INIT_CS; csG = 0.05; }
      if (haveF) { G = D.G(k); DW = D.DW(k); gqs = this.gq(k, G); }
      lr = D.lr[D.FR[k]];
    }
    const hash01 = (i) => { let x = (i + 1) * 2654435761 >>> 0; x ^= x >>> 15; x = Math.imul(x, 2246822519) >>> 0; x ^= x >>> 13; return (x >>> 0) / 4294967296; };
    // 每块：前向扫描到哪、反向发光多少、更新到哪
    const fx = F.fwd, bx = F.bwd;
    const scanOn = { blk: null, f: 0 };
    // 梯度光的强度：远景亮一点才看得出；拆到一块矩阵、一个参数时收着点，免得盖住方块本来的颜色
    const gK = { 'g-run': 0.42, 'g-step': 0.4, 'g-op': 0.34, 'g-mat': 0.32, 'g-param': 0.2 }[st.view] ?? 0.4;
    // 更新那一段：还没轮到的张量，梯度光再暗一半（注意力在正在更新的那一块）
    const updOn = s.ph === 'upd' && !!s.sub && s.sub !== 'clip';
    const blkState = new Map();
    for (const b of BLOCKS) {
      if (b.kind !== 'w') continue;
      const fwin = win(FW[b.id], fx), bwin = win(BW[b.id], bx);
      const u = F.upd[b.t] || { dw: 0, done: 0 };
      const gOn = s.ph === 'init' ? 0 : bwin * (1 - Math.min(1, u.dw * 2.5)) * (updOn && b.t !== s.t ? 0.5 : 1);
      blkState.set(b, { fwin, bwin, u, gOn });
      if (fwin > 0 && fwin < 1 && !b.t.startsWith('g') && b.id !== 'E') scanOn.blk = b, scanOn.f = fwin;
    }
    const lookRow = new Set();
    if (fx > 0 && fx < 1) for (let q = 0; q < 8; q++) if (fx * 8 > q && fx * 8 < q + 1.6) lookRow.add(D.fixed[q]);
    const selGi = st.view === 'g-param' ? this.selGi : null;
    let anyGlow = false;
    for (let i = 0; i < this.nW; i++) {
      const b = this.wBlk[i], gi = this.wGi[i], bs = blkState.get(b);
      const norm = b.t.charCodeAt(0) === 103; // 'g'
      let v;
      if (initMode) {
        const val = (mode) => (norm ? 0 : mode === 'zero' ? 0 : mode === 'big' ? D.bW0[gi] : D.w0[gi]);
        v = this.initMix < 1 ? val(this.initFrom) + (val(initMode) - val(this.initFrom)) * easeInOut(this.initMix) : val(initMode);
        if (reveal < 1 && hash01(gi) > reveal) v = 0;
      } else {
        const w = W[gi] - (norm ? 1 : 0);
        const dwv = DW ? DW[gi] : 0;
        v = w + dwv * bs.u.done;
        if (W1) v += (W1[gi] - (norm ? 1 : 0) - v) * F.next;
      }
      const cs = norm ? csG : csM, hs = norm ? HS_G : HS_MAT;
      const av = Math.abs(v);
      const h = H0 + HK * Math.pow(Math.min(1, av / hs), 0.85);
      // Δw：顶出去再收回来
      let pop = 0, popA = 0;
      if (DW && bs.u.dw > 0) { popA = Math.max(0, bs.u.dw - bs.u.done); pop = (DW[gi] / lr) * POP * popA; }
      const o = i * 16;
      ma[o] = 1; ma[o + 5] = 1; ma[o + 10] = h;
      ma[o + 12] = this.wX[i]; ma[o + 13] = this.wY[i]; ma[o + 14] = pop;
      divInto(ca, i * 3, v / cs, b.mirror ? 0.8 : 1);
      // 发光层
      let gr = 0, gg = 0, gb = 0;
      if (G && bs.gOn > 0) {
        const I = Math.pow(Math.min(1, Math.abs(G[gi]) / gqs[b.t]), 1.6) * bs.gOn * gK;
        gr += LIN.rose[0] * I; gg += LIN.rose[1] * I; gb += LIN.rose[2] * I;
      }
      if (popA > 0) {
        const I = Math.min(1, Math.abs(DW[gi]) / lr) * popA * 0.32;
        gr += LIN.violet[0] * I; gg += LIN.violet[1] * I; gb += LIN.violet[2] * I;
      }
      if (b.id === 'E' && lookRow.size && lookRow.has(this.wR[i])) { gr += 0.1; gg += 0.28; gb += 0.26; }
      if (selGi != null && gi === selGi) { const I = 0.2 + 0.1 * Math.sin(t * 5); gr += I; gg += I * 0.8; gb += I * 0.5; }
      ga[o] = 1; ga[o + 5] = 1; ga[o + 10] = h + 0.03; ga[o + 12] = this.wX[i]; ga[o + 13] = this.wY[i]; ga[o + 14] = pop - 0.015;
      gca[i * 3] = gr; gca[i * 3 + 1] = gg; gca[i * 3 + 2] = gb;
      if (gr + gg + gb > 0.003) anyGlow = true;
    }
    this.glow.visible = anyGlow;
    this.cubes.instanceMatrix.needsUpdate = true;
    this.cubes.instanceColor.needsUpdate = true;
    this.glow.instanceMatrix.needsUpdate = true;
    this.glow.instanceColor.needsUpdate = true;
    // 面板边框：正在算 / 梯度到了 / 更新中 换颜色
    for (const [b, bs] of blkState) {
      const e = b.plate.edge.material;
      const active = (bs.fwin > 0 && bs.fwin < 1) || (bs.bwin > 0 && bs.bwin < 1) || (bs.u.dw > 0 && bs.u.done < 1);
      e.opacity = active ? 0.95 : 0.38;
      e.color.setHex(bs.u.dw > 0 && bs.u.done < 1 ? 0xb39dff : bs.bwin > 0 && bs.bwin < 1 ? 0xff6b93 : GROUP_COL[b.t]);
    }
    // 扫描线
    const sb = scanOn.blk;
    this.scan.material.opacity = sb ? 0.5 : 0;
    if (sb) {
      const c = Math.min(sb.cols - 1, Math.floor(scanOn.f * sb.cols));
      this.scan.position.set(cellX(sb, c), (sb.yTop + sb.yBot) / 2, 0.34);
      this.scan.scale.set(sb.px * 0.9, sb.h + 0.06, 1);
    }
  }

  // 激活：前向时一列列亮起（真实数值），反向时换成梯度（玫红 / 紫）
  updActs(st, F, haveF) {
    const D = this.D, k = st.k, T = D.T;
    const ca = this.acts.instanceColor.array;
    const fx = F.fwd, bx = F.bwd;
    const cache = {};
    const src = (id, grad) => {
      const key = `${id}:${grad ? 1 : 0}`;
      if (key in cache) return cache[key];
      let r = null;
      if (haveF) {
        if (id === 'att0' || id === 'att1') {
          const h = id === 'att0' ? 0 : 1;
          if (grad) { const a = D.grad(k, 'att'); r = { get: (rr, cc) => a[(h * T + rr) * T + cc], sc: this.aq(k, 'gatt', a) }; }
          else r = { get: (rr, cc) => D.att(k, h, rr, cc), seq: true };
        } else {
          const [name, tr, gname] = ACT_SRC[id];
          const a = grad ? D.grad(k, gname || name) : D.act(k, name);
          if (a) {
            const n = a.length / T;
            r = { get: tr ? (rr, cc) => a[cc * n + rr] : (rr, cc) => a[rr * n + cc], sc: this.aq(k, `${grad ? 'g' : 'a'}${gname && grad ? gname : name}`, a) };
          }
        }
      }
      cache[key] = r;
      return r;
    };
    for (const b of BLOCKS) {
      if (b.kind !== 'a') continue;
      const fw = win(FW[b.id], fx), bw = win(BW[b.id], bx);
      const V = fw > 0 ? src(b.id, false) : null, Gd = bw > 0 ? src(b.id, true) : null;
      const byRow = b.id.startsWith('att');
      const n = byRow ? b.rows : b.cols;
      for (let r = 0; r < b.rows; r++) for (let c = 0; c < b.cols; c++) {
        const i = b.i0 + r * b.cols + c, o = i * 3;
        const idx = byRow ? r : c;
        if (Gd && idx >= n * (1 - bw) - 1e-6) gradInto(ca, o, Gd.get(r, c) / Gd.sc, 0.95);
        else if (V && idx < n * fw) { if (V.seq) seqInto(ca, o, V.get(r, c), 0.95); else divInto(ca, o, V.get(r, c) / V.sc, 0.92); }
        else { ca[o] = LIN.idle[0]; ca[o + 1] = LIN.idle[1]; ca[o + 2] = LIN.idle[2]; }
      }
    }
    this.acts.instanceColor.needsUpdate = true;
  }

  // 光柱上的脉冲：前向往上，反向往下；环和连接管在用到时亮起
  updSpine(st, F, haveF, t) {
    const D = this.D, k = st.k;
    const lerpK = (x, ks) => { if (x <= ks[0][0]) return ks[0][1]; for (let j = 1; j < ks.length; j++) if (x <= ks[j][0]) { const [x0, y0] = ks[j - 1], [x1, y1] = ks[j]; return y0 + (y1 - y0) * (x - x0) / (x1 - x0); } return ks[ks.length - 1][1]; };
    const FK = [[0, 0.4], [1, Y.ring1], [4, Y.ring1], [5, Y.add1], [5.6, Y.ring2], [7, Y.ring2], [8, Y.add2], [9, Y.ringf]];
    const BKk = [[0, Y.ringf], [1, Y.ringf], [2, Y.add2], [4, Y.ring2], [5, Y.add1], [7, Y.ring1], [8, Y.ring1], [9, 1.8], [10, 0.4]];
    const m = new THREE.Matrix4();
    const fOn = F.fwd > 0 && F.fwd < 10;
    let n = 0;
    if (fOn) {
      const y = lerpK(F.fwd, FK);
      for (let q = 0; q < 8; q++) { m.makeRotationX(Math.PI / 2).setPosition(xPos(q), y, 0.12); this.pulseF.setMatrixAt(n++, m); }
    }
    this.pulseF.count = n;
    this.pulseF.instanceMatrix.needsUpdate = true;
    this.pulseF.material.opacity = 0.55 + 0.25 * Math.sin(t * 9);
    n = 0;
    const bOn = F.bwd > 0 && F.bwd < 10;
    if (bOn) {
      const y = lerpK(F.bwd, BKk);
      // 每个位置的脉冲大小 ∝ 这里 ∂L/∂h 的长度（真实）
      const gname = F.bwd < 4 ? 'h2' : F.bwd < 8 ? 'h1' : 'h0';
      const g = haveF ? D.grad(k, gname) : null;
      let mx = 1e-12;
      const nr = [];
      for (let q = 0; q < 8; q++) { let s2 = 0; if (g) for (let j = 0; j < 16; j++) s2 += g[q * 16 + j] ** 2; nr.push(Math.sqrt(s2)); mx = Math.max(mx, nr[q]); }
      for (let q = 0; q < 8; q++) { const sc = g ? 0.6 + 0.9 * nr[q] / mx : 1; m.makeRotationX(Math.PI / 2).scale(new THREE.Vector3(sc, sc, sc)).setPosition(xPos(q), y, 0.12); this.pulseB.setMatrixAt(n++, m); }
    }
    this.pulseB.count = n;
    this.pulseB.instanceMatrix.needsUpdate = true;
    this.pulseB.material.opacity = 0.55 + 0.25 * Math.sin(t * 9);
    // 光柱：有数据流过时亮一点
    for (const c of this.cols) c.material.opacity = (F.fwd >= 0 ? 0.32 : 0.16) + (fOn ? 0.15 : 0);
    // 环 / 连接管：前向或反向正经过这个算子时亮
    const opOn = (name) => { const fi = FWD_OPS.indexOf(name), bi = BWD_OPS.indexOf(name); return (F.fwd > fi && F.fwd < fi + 1) || (F.bwd > bi && F.bwd < bi + 1); };
    const lit = (o, on, base = 0.16, hi = 0.75) => { o.material.opacity += ((on ? hi : base) - o.material.opacity) * 0.25; };
    lit(this.rings.norm1, opOn('norm1'), 0.35, 0.95); lit(this.rings.norm2, opOn('norm2'), 0.35, 0.95); lit(this.rings.normf, opOn('normf'), 0.35, 0.95);
    lit(this.rings.add1, (F.fwd > 4.8 && F.fwd < 5.3) || opOn('wo') && F.bwd >= 0, 0.3, 0.95);
    lit(this.rings.add2, (F.fwd > 7.8 && F.fwd < 8.3) || opOn('wd') && F.bwd >= 0, 0.3, 0.95);
    lit(this.pipes.norm1, opOn('norm1')); lit(this.pipes.norm2, opOn('norm2')); lit(this.pipes.normf, opOn('normf'));
    lit(this.pipes.add1, F.fwd > 4.7 && F.fwd < 5.2 || opOn('wo') && F.bwd >= 0);
    lit(this.pipes.add2, F.fwd > 7.7 && F.fwd < 8.2 || opOn('wd') && F.bwd >= 0);
    lit(this.pipes.aoT, F.fwd > 3.9 && F.fwd < 4.35 || (F.bwd > 5.5 && F.bwd < 6.1));
    lit(this.pipes.actT, F.fwd > 6.9 && F.fwd < 7.3 || (F.bwd > 2.5 && F.bwd < 3.1));
    const fe = F.fwd > 0 && F.fwd < 1 ? F.fwd : -1;
    this.look.forEach((m2, q) => {
      const on = fe >= 0 && fe * 8 > q - 0.2;
      m2.material.opacity = on ? (fe * 8 < q + 1.6 ? 0.75 : 0.18) : 0;
      m2.geometry.setDrawRange(0, on ? Math.floor(m2.geometry.index.count * Math.min(1, (fe * 8 - q + 0.2) / 0.8) / 3) * 3 : 0);
    });
  }

  // 语料轮 → 托盘：这一步的 8 段从轮子上切下来，字块飞进托盘；第 0 段错开一位的答案排在最前面
  updFeed(st, F) {
    const D = this.D, o = this.tok0;
    const tStep = D.FR[st.k];
    const show = F.feed >= 0;
    // 轮子上这一步的 8 个切口
    const ak = show ? `${tStep}` : 'none';
    if (ak !== this.arcKey) {
      this.arcKey = ak;
      disposeTree(this.wheelArcs);
      this.wheelArcs.clear();
      if (show) {
        const NS = D.stream.length;
        for (let b = D.B - 1; b >= 0; b--) {
          const off = D.offs(tStep, b), r = WHEEL.r + 0.2 + b * 0.04;
          const a0 = Math.PI / 2 - ((off - 0.45) / NS) * Math.PI * 2, a1 = Math.PI / 2 - ((off + D.T + 0.45) / NS) * Math.PI * 2;
          const pts = [];
          for (let j = 0; j <= 24; j++) { const a = a0 + (a1 - a0) * (j / 24); pts.push(new THREE.Vector3(Math.cos(a) * r, Math.sin(a) * r, 0)); }
          const m = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 24, b === 0 ? 0.022 : 0.012, 5, false), new THREE.MeshBasicMaterial({ color: b === 0 ? 0xffb65c : 0x6b9bff, transparent: true, opacity: b === 0 ? 0.85 : 0.45, depthWrite: false }));
          m.raycast = () => {};
          this.wheelArcs.add(m);
        }
      }
    }
    this.wheelArcs.visible = !!show;
    const NS = D.stream.length, T = D.T, B = D.B;
    for (let i = 0; i < NS; i++) this.placeTok(o.wheel + i, this.wheelPos(i), 0.9);
    const tmp = new THREE.Vector3();
    for (let b = 0; b < B; b++) for (let i = 0; i < T; i++) {
      const idx = o.tray + b * T + i;
      const order = b * T + i, d0 = (order / (B * T)) * 0.62, f = show ? seg(F.feed, d0, d0 + 0.38) : 0;
      if (f <= 0) { this.placeTok(idx, tmp.set(0, -50, 0), 0); continue; }
      const from = this.wheelPos(D.offs(tStep, b) + i), to = this.slot(b, i);
      const e = easeInOut(f);
      tmp.copy(from).lerp(to, e);
      tmp.y += Math.sin(e * Math.PI) * (0.9 + b * 0.05);
      tmp.z += Math.sin(e * Math.PI) * 0.4;
      const hi = F.lossPos >= 0 && b === 0 && i === F.lossPos;
      this.placeTok(idx, tmp, b === 0 ? (hi ? 1.18 : 1.06) : 0.92, (1 - e) * 0.6);
    }
    for (let i = 0; i < T; i++) {
      const f = show ? F.shift : 0, idx = o.tgt + i;
      if (f <= 0.01) { this.placeTok(idx, tmp.set(0, -50, 0), 0); continue; }
      const e = easeInOut(f);
      tmp.set(xPos(i + 1), TRAY.y, 0).lerp(new THREE.Vector3(xPos(i), TRAY.y, TRAY.zTarget), e);
      tmp.y += Math.sin(e * Math.PI) * 0.25;
      this.placeTok(idx, tmp, 0.86);
    }
    this.tokBody.instanceMatrix.needsUpdate = true;
    this.tokFace.instanceMatrix.needsUpdate = true;
  }

  // 概率架：前向到最后升起来（真实概率），正确答案绿框；损失：每个位置 −ln p、整批平均
  updHead(st, F, kw) {
    const D = this.D, k = st.k, T = D.T, V = D.V;
    const have = kw >= 0 && D.has('w', k) && st.step.ph !== 'init';
    const rise = have ? easeOut(seg(F.fwd, 9.25, 10)) : 0;
    const ba = this.bars.instanceMatrix.array, bc = this.bars.instanceColor.array;
    for (let p = 0; p < T; p++) {
      const ans = D.fixed[p + 1];
      let best = 0;
      if (have) for (let v = 1; v < V; v++) if (D.probs(k, p, v) > D.probs(k, p, best)) best = v;
      const dimRow = F.lossPos >= 0 && F.lossPos !== p;
      for (let v = 0; v < V; v++) {
        const i = p * V + v, o = i * 16;
        const pr = have ? D.probs(k, p, v) : 0;
        ba[o] = 1; ba[o + 5] = Math.max(0.002, pr * SHELF.hMax * rise); ba[o + 10] = 1;
        ba[o + 12] = shelfX(v); ba[o + 13] = SHELF.y; ba[o + 14] = shelfZ(p);
        const c = i * 3, kk = dimRow ? 0.35 : 1;
        if (v === ans) { bc[c] = LIN.green[0] * kk; bc[c + 1] = LIN.green[1] * kk; bc[c + 2] = LIN.green[2] * kk; }
        else if (v === best && pr > 0.12) { bc[c] = LIN.rose[0] * 0.8 * kk; bc[c + 1] = LIN.rose[1] * 0.8 * kk; bc[c + 2] = LIN.rose[2] * 0.8 * kk; }
        else seqInto(bc, c, 0.25 + pr * 0.8, 0.8 * kk);
      }
    }
    this.bars.instanceMatrix.needsUpdate = true;
    this.bars.instanceColor.needsUpdate = true;
    const m = new THREE.Matrix4();
    for (let p = 0; p < T; p++) { m.makeTranslation(shelfX(D.fixed[p + 1]), SHELF.y + 0.004, shelfZ(p)); this.marks.setMatrixAt(p, m); }
    this.marks.instanceMatrix.needsUpdate = true;
    this.marks.material.opacity = 0.85 * rise;
    // −ln p
    const na = this.nll.instanceMatrix.array, nc = this.nll.instanceColor.array;
    for (let p = 0; p < T; p++) {
      const pr = have ? D.probs(k, p, D.fixed[p + 1]) : 1, nl = -Math.log(Math.max(pr, 1e-9));
      const f = clamp01(F.lossUpTo - p);
      const o = p * 16;
      na[o] = 1; na[o + 5] = Math.max(0.002, Math.min(1, nl / 3.2) * GAUGE.hMax * 0.6 * f); na[o + 10] = 1;
      na[o + 12] = SHELF.x0 + V * SHELF.px + 0.32; na[o + 13] = SHELF.y; na[o + 14] = shelfZ(p);
      const hl = F.lossPos === p ? 1.25 : 1;
      const col = nl > 1 ? LIN.rose : nl > 0.3 ? LIN.amber : LIN.cyan;
      nc[p * 3] = col[0] * 0.8 * hl; nc[p * 3 + 1] = col[1] * 0.8 * hl; nc[p * 3 + 2] = col[2] * 0.8 * hl;
    }
    this.nll.instanceMatrix.needsUpdate = true;
    this.nll.instanceColor.needsUpdate = true;
    const L0 = have ? D.loss[D.FR[k]] : 0;
    this.gauge.scale.y = Math.max(0.001, Math.min(1, L0 / 3.2) * GAUGE.hMax * F.lossMean);
  }

  updLabels(st, F) {
    const D = this.D, v = st.view, s = st.step, k = st.k;
    const far = v === 'g-init' || v === 'g-run' || v === 'g-step' || (v === 'g-op' && s.ph === 'upd' && s.sub === 'clip');
    const focusT = s.t ? s.t : null;
    // 参数块：远景只留名字，近景加形状；γ 条太小，远景不标
    for (const [id, o] of Object.entries(this.lw)) {
      const b = BLOCK[id], isG = b.t.startsWith('g');
      const mine = focusT && (focusT === b.t) && (v === 'g-mat' || v === 'g-param');
      // 窄屏的远景里标签相对更大：同一排并列的几块只标第一块（W_q 代表 q k v，W_gate 代表 gate up）
      let show = (!isG || !far) && !(far && this.small && (id === 'Wk' || id === 'Wv' || id === 'Wu'));
      if (v === 'g-param') show = mine;
      if (v === 'g-mat') show = mine || this.near(b);
      if (v === 'g-op') show = this.inOp(b) || !isG;
      o.visible = !!show;
      const html = far ? `${o.short}` : o.full;
      if (o.el._h !== html) { o.el.innerHTML = html; o.el._h = html; }
      o.el.classList.toggle('on', !!mine || (v === 'g-op' && this.inOp(b) && s.ph !== 'loss'));
    }
    for (const [id, o] of Object.entries(this.la)) {
      const b = BLOCK[id];
      let show = false;
      if (v === 'g-op') show = this.inOp(b) && !(b.spine && this.small);
      else if (v === 'g-mat') show = this.near(b);
      else if (v === 'g-step') show = (s.ph === 'fwd' || s.ph === 'bwd') && ((b.spine && !this.small) || this.inPhaseOp(b, F));
      o.visible = !!show;
    }
    // 残差窗口和 ⊕ 的标签挂在光柱左边，窄屏上会出画，不标
    for (const [id, o] of Object.entries(this.lr)) o.visible = !!((v === 'g-step' && (s.ph === 'fwd' || s.ph === 'bwd')) || (v === 'g-op' && this.inOpName(id))) && !(this.small && id.startsWith('add'));
    const t = D.FR[k];
    // 损失
    const lossOn = F.lossMean > 0.05 && s.ph !== 'init' && (v !== 'g-mat' && v !== 'g-param');
    this.lGauge.visible = !!lossOn;
    if (lossOn) {
      const html = `L = ${D.loss[t].toFixed(3)}<small>${L('整批 64 个位置平均', 'mean of 64 positions')}</small>`;
      if (this.lGauge.el._h !== html) { this.lGauge.el.innerHTML = html; this.lGauge.el._h = html; }
      this.lGauge.position.y = GAUGE.y + Math.min(1, D.loss[t] / 3.2) * GAUGE.hMax * F.lossMean + 0.1;
    }
    const clipOn = s.ph === 'upd' && (s.sub === 'clip' || (!s.sub && F.clip < 1)) && v !== 'g-param';
    this.lClip.visible = !!clipOn;
    if (clipOn) {
      const c = D.clip[t], html = `‖g‖ = ${D.gnorm[t].toFixed(3)}${c < 1 ? L(` > 1 → 全部梯度 × ${c.toFixed(3)}`, ` > 1 → all gradients × ${c.toFixed(3)}`) : L(' ≤ 1 · 不裁剪', ' ≤ 1 · no clipping')}`;
      if (this.lClip.el._h !== html) { this.lClip.el.innerHTML = html; this.lClip.el._h = html; }
    }
    // 跳过的步数（第 50 步以后帧之间隔 5 / 10 步）
    const gap = k > 0 ? D.FR[k] - D.FR[k - 1] : 1;
    this.lGap.visible = s.ph !== 'init' && (v === 'g-run' || v === 'g-step') && gap > 1;
    if (this.lGap.visible) {
      const html = L(`第 ${t + 1} 步 · 和上一帧隔 ${gap} 步`, `Step ${t + 1} · ${gap} steps after the last frame`);
      if (this.lGap.el._h !== html) { this.lGap.el.innerHTML = html; this.lGap.el._h = html; }
    }
    this.lWheel.visible = far || (v === 'g-op' && s.ph === 'batch');
    this.lTray.visible = (v === 'g-step' && s.ph === 'batch') || (v === 'g-op' && s.ph === 'batch');
    this.lShelf.visible = !!(F.fwd >= 9.3 && (v === 'g-step' || v === 'g-op') && (s.ph === 'loss' || s.sub === 'logits'));
    this.lFactory.visible = s.ph === 'init';
    if (s.ph === 'init') {
      const html = { model: L('出厂状态 · 还没训练 · 2,928 个参数', 'Factory state · untrained · 2,928 parameters'), hist: L('出厂状态 · 每个参数是一个很小的随机数', 'Factory state · each parameter a small random number'), zero: L('对照：全部设成 0（γ 仍为 1）', 'Control: everything set to 0 (γ still 1)'), big: L('对照：放大 50 倍 N(0, 1²)', 'Control: 50× larger N(0, 1²)') }[s.sub];
      if (this.lFactory.el._h !== html) { this.lFactory.el.innerHTML = html; this.lFactory.el._h = html; }
    }
  }

  // 这一块是不是当前算子用到的（D3 标签、D2 高亮）
  inOp(b) {
    const s = this.st?.step;
    if (!s || !s.sub) return false;
    const op = s.ph === 'fwd' || s.ph === 'bwd' ? s.sub : null;
    if (s.ph === 'upd' && s.t) return b.t === s.t && b.kind === 'w';
    if (!op) return false;
    const fw = FW[b.id], o = FWD_OPS.indexOf(op);
    if (fw && fw[0] === o) return true;
    const bw = BW[b.id];
    return s.ph === 'bwd' && bw && bw[0] === BWD_OPS.indexOf(op);
  }
  // 一步之内（D2）前向 / 反向正经过的那个算子用到的激活块
  inPhaseOp(b, F) {
    const s = this.st?.step;
    if (!s || s.sub) return false;
    if (s.ph === 'fwd' && F.fwd > 0 && F.fwd < 10) { const w = FW[b.id]; return !!w && w[0] === Math.floor(F.fwd); }
    if (s.ph === 'bwd' && F.bwd > 0 && F.bwd < 10) { const w = BW[b.id]; return !!w && w[0] === Math.floor(F.bwd); }
    return false;
  }
  inOpName(id) {
    const s = this.st?.step;
    if (!s || !s.sub) return false;
    if (id === 'add1') return s.sub === 'wo';
    if (id === 'add2') return s.sub === 'wd';
    return s.sub === id;
  }
  // D4：这一块是不是焦点矩阵的输入 / 输出
  near(b) {
    const s = this.st?.step;
    if (!s?.t) return false;
    const key = s.t === 'E' && s.sub === 'logits' ? 'ET' : s.t;
    return (TENSOR_PARTS[key] || []).includes(b.id);
  }

  // 选中的参数：方块外面的框；D5 再框出它乘到的输入那一行、输出那一列
  updMarks(st) {
    const D = this.D, s = st.step;
    const place = (m, x, y, z, w, h, d) => { m.position.set(x, y, z); m.scale.set(w, h, d); m.visible = true; };
    // 悬停
    const hv = this.hover;
    if (hv && hv.type === 'w') {
      const i = hv.inst, h = this.cubes.instanceMatrix.array[i * 16 + 10], z0 = this.cubes.instanceMatrix.array[i * 16 + 14];
      place(this.mHover, this.wX[i], this.wY[i], z0 + h / 2, FACE * 1.25, FACE * 1.25, h + 0.04);
    } else if (hv && hv.type === 'a') {
      const b = this.aBlk[hv.inst];
      place(this.mHover, cellX(b, this.aR[hv.inst]) * 0 + cellX(b, this.aC[hv.inst]), cellY(b, this.aR[hv.inst]), 0.03, b.px * 0.95, b.py * 0.95, 0.05);
    } else this.mHover.visible = false;
    this.mSel.visible = this.mSel2.visible = this.mIn.visible = this.mOut.visible = false;
    this.lSel.visible = false;
    const gi = this.selGi;
    if (gi == null || st.view !== 'g-param') return;
    const lc = D.locate(gi);
    const marks = [this.mSel, this.mSel2];
    let mi = 0;
    for (let i = 0; i < this.nW && mi < 2; i++) {
      if (this.wGi[i] !== gi) continue;
      const h = this.cubes.instanceMatrix.array[i * 16 + 10], z0 = this.cubes.instanceMatrix.array[i * 16 + 14];
      place(marks[mi], this.wX[i], this.wY[i], z0 + h / 2, FACE * 1.45, FACE * 1.45, h + 0.08);
      if (mi === 0 || (s.sub === 'logits' && this.wBlk[i].mirror)) {
        this.lSel.position.set(this.wX[i], this.wY[i] + FACE, z0 + h);
        this.selInst = i;
      }
      mi++;
    }
    // 输入那一行 / 输出那一列
    const t = lc.p.name, io = t === 'E' ? (s.sub === 'logits' ? ['nf', 'logits'] : [null, 'h0']) : IO[t];
    if (!io || s.ph === 'upd') return;
    const [inId, outId] = io;
    const slice = (m, id, row, idx) => {
      const b = BLOCK[id];
      if (row) place(m, b.x0 + b.w / 2, cellY(b, idx), 0.04, b.w + 0.04, b.py, 0.06);
      else place(m, cellX(b, idx), b.yBot + b.h / 2, 0.04, b.px, b.h + 0.04, 0.06);
    };
    if (t === 'E') {
      if (s.sub === 'logits') { slice(this.mIn, 'nf', true, lc.j); slice(this.mOut, 'logits', false, lc.i); }
      else slice(this.mOut, 'h0', true, lc.j);
    } else if (lc.p.norm) slice(this.mOut, outId, true, lc.i);
    else { slice(this.mIn, inId, true, lc.i); slice(this.mOut, outId, false, lc.j); }
  }

  /* ================================================================ 拾取 / 提示 */

  pickW(inst) {
    const gi = this.wGi[inst];
    return { type: 'w', inst, gi, click: true };
  }
  pickA(inst) { return { type: 'a', inst }; }

  // 悬停提示（真实数值）
  tip(info) {
    const D = this.D, st = this.st;
    if (!info || !st) return null;
    const k = st.k, init = st.step.ph === 'init', t = D.FR[k];
    if (info.type === 'w') {
      const gi = info.gi, lc = D.locate(gi), norm = lc.p.norm;
      const nm = esc(paramName(D, gi));
      if (init) {
        const v = norm ? 1 : this.initMode === 'zero' ? 0 : this.initMode === 'big' ? D.bW0[gi] : D.w0[gi];
        return `<span class="k">${nm} · ${L('初始值', 'initial value')}</span>w <span class="v">${fnum(v, 4)}</span>${norm ? L('<br>RMSNorm 的 γ 一律从 1 开始', '<br>RMSNorm’s γ always starts at 1') : ''}<br><span style="color:var(--dim)">${L('点一下看这个参数的一生', 'Click to see this parameter’s life')}</span>`;
      }
      const kw = this.nearW(k);
      const w = kw >= 0 ? D.W(kw)[gi] : D.w0[gi];
      let extra = '';
      if (D.has('f', k)) {
        const g = D.G(k)[gi], dw = D.DW(k)[gi], m = D.Mo(k)[gi], sv = Math.sqrt(D.Vo(k)[gi]);
        extra = `<br>∂L/∂w <span class="a">${fnum(g, 3)}</span>　Δw <span class="v">${fnum(dw, 3)}</span><br>m ${fnum(m, 3)}　√v ${fnum(sv, 3)}`;
      }
      const mir = this.wBlk[info.inst].mirror ? L('<br>Eᵀ：和最下面的 E 是同一个数', '<br>Eᵀ: the same number as in E at the bottom') : '';
      return `<span class="k">${nm} · ${L(`第 ${t + 1} 步`, `step ${t + 1}`)}</span>w <span class="v">${fnum(w, 4)}</span>${norm ? `（γ）` : ''}${extra}${mir}<br><span style="color:var(--dim)">${L('点一下看这个参数的一生', 'Click to see this parameter’s life')}</span>`;
    }
    if (info.type === 'a') {
      if (init || !D.has('f', k)) return null;
      const b = this.aBlk[info.inst], r = this.aR[info.inst], c = this.aC[info.inst];
      const T = D.T;
      const posOf = (q) => L(`位置 ${q}「${esc(D.ch(D.fixed[q]))}」`, `position ${q} “${esc(D.ch(D.fixed[q]))}”`);
      const F = this.F;
      const fwOn = win(FW[b.id], F.fwd) > 0, bwOn = win(BW[b.id], F.bwd) > 0;
      if (b.id.startsWith('att')) {
        const h = b.id === 'att0' ? 0 : 1, a = D.att(k, h, r, c);
        const g = bwOn ? D.grad(k, 'att')[(h * T + r) * T + c] : null;
        return `<span class="k">${L(`注意力 · 头 ${h}`, `attention · head ${h}`)}</span>${posOf(r)} → ${posOf(c)}：<span class="v">${fmtP(a)}</span>${c > r ? L('<br>因果遮罩：看不到后面的字', '<br>causal mask: can’t see later characters') : ''}${g != null ? `<br>∂L/∂a <span class="a">${fnum(g, 3)}</span>` : ''}`;
      }
      const [name, tr, gname] = ACT_SRC[b.id];
      const pos = tr ? c : r, dim = tr ? r : c;
      const a = D.act(k, name), n = a.length / T;
      const val = a[pos * n + dim];
      const gA = D.grad(k, gname || name);
      const g = bwOn && gA ? gA[pos * n + dim] : null;
      const NM = { h0: 'h₀', h1: 'h₁', h2: 'h₂', n1: 'n₁', n2: 'n₂', nf: 'n_f', aoT: 'ao', actT: 'act' }[b.id] || b.id;
      const what = b.id === 'logits' ? L(`给「${esc(D.ch(dim))}」打的分`, `score for “${esc(D.ch(dim))}”`) : L(`第 ${dim} 维`, `dim ${dim}`);
      return `<span class="k">${NM} · ${posOf(pos)}</span>${what}：<span class="v">${fwOn || !bwOn ? fnum(val, 4) : '—'}</span>${g != null ? `<br>∂L/∂${NM === 'o' ? 'h₁' : NM === 'f' ? 'h₂' : NM} <span class="a">${fnum(g, 3)}</span>` : ''}`;
    }
    if (info.type === 'bar') {
      if (init || !D.has('w', k)) return null;
      const pr = D.probs(k, info.p, info.v), ans = D.fixed[info.p + 1];
      return `<span class="k">${L(`位置 ${info.p}：看到“${esc(D.fixed.slice(0, info.p + 1).map(D.ch).join(''))}”`, `position ${info.p}: has seen “${esc(D.fixed.slice(0, info.p + 1).map(D.ch).join(''))}”`)}</span>${L('下一个字是', 'next is')}「${esc(D.ch(info.v))}」<span class="v">${fmtP(pr)}</span>${info.v === ans ? L('　← 正确答案', '　← the answer') : ''}`;
    }
    if (info.type === 'nll') {
      if (init || !D.has('w', k)) return null;
      const pr = D.probs(k, info.p, D.fixed[info.p + 1]);
      return `<span class="k">${L(`位置 ${info.p} 的损失`, `loss at position ${info.p}`)}</span>${L('正确答案', 'answer')}「${esc(D.ch(D.fixed[info.p + 1]))}」p = <span class="v">${fmtP(pr)}</span><br>−ln p = <span class="a">${(-Math.log(Math.max(pr, 1e-9))).toFixed(4)}</span>`;
    }
    if (info.type === 'gauge') {
      if (init) return null;
      return `<span class="k">${L(`第 ${t + 1} 步的损失`, `loss at step ${t + 1}`)}</span>${L('这一批 64 个位置的 −ln p 平均', 'mean −ln p over this batch’s 64 positions')}：<span class="v">${D.loss[t].toFixed(4)}</span><br>${L('瞎猜', 'blind guess')} ln 20 = ${Math.log(20).toFixed(3)}（${L('玫红环', 'rose ring')}）`;
    }
    if (info.type === 'tok') {
      if (init) return null;
      const bt = D.batch(t)[info.b];
      return `<span class="k">${L(`第 ${info.b} 段 · 位置 ${info.i}`, `window ${info.b} · position ${info.i}`)}${info.b === 0 ? L(' · 固定样例', ' · fixed sample') : ''}</span>「${esc(D.ch(bt[info.i]))}」${L('，要猜的下一个字', ', next to guess')}「${esc(D.ch(bt[info.i + 1]))}」<br>${L(`这一段从语料第 ${D.offs(t, info.b) + 1} 个字开始`, `this window starts at character ${D.offs(t, info.b) + 1}`)}`;
    }
    if (info.type === 'tgt') return `<span class="k">${L(`答案 · 位置 ${info.i}`, `answer · position ${info.i}`)}</span>「${esc(D.ch(D.fixed[info.i + 1]))}」${L('（输入错开一位）', ' (the input shifted by one)')}`;
    if (info.type === 'wheel') return `<span class="k">${L(`语料第 ${info.i + 1} 个字`, `corpus character ${info.i + 1}`)}</span>「${esc(D.ch(D.stream[info.i]))}」`;
    return null;
  }

  // 选中参数的读数（D5 方块上方的小标签）
  setSel(gi) { this.selGi = gi; }
  selLabel(html) {
    if (!this.lSel || this.selGi == null || this.st?.view !== 'g-param') { if (this.lSel) this.lSel.visible = false; return; }
    this.lSel.visible = !!html;
    if (html && this.lSel.el._h !== html) { this.lSel.el.innerHTML = html; this.lSel.el._h = html; }
  }

  /* ================================================================ 镜头 */

  // 让一个盒子刚好装进没被面板挡住的那块画面：按真实透视把 8 个角投影出来，二分出距离，再把盒子挪到画面正中
  fitBox(b, dir, margin = 1.06, minD = 2) {
    const E = this.E;
    const key = `${b.join(',')}|${dir.x},${dir.y},${dir.z}|${margin}|${E.w}x${E.h}|${E.insetR | 0}|${E.padB | 0}|${E.padT | 0}`;
    if (this.fitKey === key) return { pos: this.fitRes.pos.clone(), look: this.fitRes.look.clone() };
    const cam = this.fitCam || (this.fitCam = new THREE.PerspectiveCamera());
    cam.copy(E.camera);
    cam.updateProjectionMatrix();
    const [x0, x1, y0, y1, z0 = -0.7, z1 = 0.6] = b;
    const pts = [];
    for (const x of [x0, x1]) for (const y of [y0, y1]) for (const z of [z0, z1]) pts.push(new THREE.Vector3(x, y, z));
    const w = E.w || 1, h = E.h || 1;
    const rx0 = -1, rx1 = 1 - (2 * (E.insetR || 0)) / w, ry0 = -1 + (2 * E.padB) / h, ry1 = 1 - (2 * E.padT) / h;
    const cx = (rx0 + rx1) / 2, cy = (ry0 + ry1) / 2, hw = (rx1 - rx0) / 2 / margin, hh = (ry1 - ry0) / 2 / margin;
    const d0 = dir.clone().normalize();
    const look = new THREE.Vector3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
    const v = new THREE.Vector3();
    const test = (dd) => {
      cam.position.copy(look).addScaledVector(d0, dd);
      cam.lookAt(look);
      cam.updateMatrixWorld();
      let mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity, behind = false;
      for (const p of pts) {
        v.copy(p).applyMatrix4(cam.matrixWorldInverse);
        if (v.z > -0.05) behind = true;
        v.applyMatrix4(cam.projectionMatrix);
        mnx = Math.min(mnx, v.x); mxx = Math.max(mxx, v.x); mny = Math.min(mny, v.y); mxy = Math.max(mxy, v.y);
      }
      return { mnx, mxx, mny, mxy, behind };
    };
    let d = minD;
    const right = new THREE.Vector3(), up = new THREE.Vector3();
    for (let it = 0; it < 3; it++) {
      let lo = minD, hi = 400;
      for (let j = 0; j < 24; j++) {
        const mid = (lo + hi) / 2, r = test(mid);
        if (!r.behind && (r.mxx - r.mnx) / 2 <= hw && (r.mxy - r.mny) / 2 <= hh) hi = mid; else lo = mid;
      }
      d = hi;
      const r = test(d);
      right.setFromMatrixColumn(cam.matrixWorld, 0);
      up.setFromMatrixColumn(cam.matrixWorld, 1);
      const halfH = d * Math.tan((cam.fov * Math.PI) / 360), halfW = halfH * cam.aspect;
      look.addScaledVector(right, ((r.mnx + r.mxx) / 2 - cx) * halfW).addScaledVector(up, ((r.mny + r.mxy) / 2 - cy) * halfH);
    }
    const res = { pos: look.clone().addScaledVector(d0, d), look: look.clone() };
    this.fitKey = key;
    this.fitRes = res;
    return { pos: res.pos.clone(), look: res.look.clone() };
  }

  camera(st) {
    const s = st.step, v = st.view, F = this.F || this.flow(st, st.p);
    const V3 = (x, y, z) => new THREE.Vector3(x, y, z);
    const frame = (r, dir, margin = 1.06, minD = 2) => this.fitBox(r, dir, margin, minD);
    // 远景从右前上方斜看（看得出方块的厚度），越近越正
    const WIDE = V3(0.5, 0.32, 1), MID = V3(0.36, 0.26, 1), NEAR = V3(0.2, 0.14, 1), TOP = V3(0.25, 0.85, 1);
    const whole = [BOUNDS.x0, BOUNDS.x1, BOUNDS.y0, BOUNDS.y1, -2.2, 1.4];
    if (v === 'g-init' || v === 'g-run') return frame(whole, WIDE, 1.02);
    if (v === 'g-step') {
      if (s.ph === 'batch') return frame([...OP_RECT.batch, -2.4, 0.8], MID, 1.06);
      if (s.ph === 'loss') return frame([...OP_RECT.loss, -1.3, 1.3], TOP, 1.06);
      if (s.ph === 'upd') return frame(whole, WIDE, 1.02);
      // 前向 / 反向：镜头跟着正在算的那一层走
      const x = s.ph === 'fwd' ? F.fwd : F.bwd;
      const op = s.ph === 'fwd' ? FWD_OPS[Math.min(9, Math.max(0, Math.floor(x)))] : BWD_OPS[Math.min(9, Math.max(0, Math.floor(x)))];
      const r = OP_RECT[op];
      const yc = (r[2] + r[3]) / 2;
      return frame([BOUNDS.x0 + 2.6, BOUNDS.x1, yc - 4.6, yc + 4.6], MID, 1.02);
    }
    if (v === 'g-op') {
      if (s.ph === 'batch') return frame([...OP_RECT.batch, -2.4, 0.8], MID, 1.06);
      if (s.ph === 'loss') return frame([...OP_RECT.loss, -1.3, 1.3], TOP, 1.06);
      if (s.ph === 'upd') {
        if (s.sub === 'clip') return frame(whole, WIDE, 1.02);
        return frame(unionRect(TENSOR_PARTS[s.t], 0.8), MID, 1.08);
      }
      return frame(OP_RECT[s.sub], MID, 1.04);
    }
    if (v === 'g-mat') {
      const key = s.t === 'E' && s.sub === 'logits' ? 'ET' : s.t;
      return frame([...unionRect(TENSOR_PARTS[key], 0.3), -0.2, 0.4], NEAR, 1.04);
    }
    // 一个参数：推到方块跟前（底部有算式板，取景区域会让出来）
    const i = this.selInst ?? 0;
    const b = this.wBlk[i];
    const x = this.wX[i], y = this.wY[i];
    const r = b && b.t.startsWith('g') ? [x - 1.2, x + 1.8, y - 1.0, y + 1.0] : [x - 1.4, x + 1.4, y - 1.0, y + 1.0];
    return frame([...r, -0.1, 0.4], V3(0.22, 0.16, 1), 1.04, 0.8);
  }
}
