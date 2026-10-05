// 第三章的 3D 机器：推理页那台 Qwen3-0.6B（同样的托盘、词元方块、28 块玻璃层板、残差流光柱、RMSNorm 环、lm_head），
// 上面叠加一步真实的监督微调（数据全部来自记录）：
//   进料：聊天模板渲染出的 24 个真实词元落进托盘；labels 那一排把提示部分换成 −100（暗、不算损失），只剩回答的 8 个；再错开一位；
//   前向：青色光环沿 23 根光柱往上走；逻辑透镜在回答的 8 根柱子上、每一层旁边读出“正确答案的概率”；顶上是每个位置的真实概率；
//   损失：只在回答的 8 个位置立起 −ln p，平均成这一步的损失（预训练式“全都算”的对照画成虚影）；
//   反向：玫红光环从第 27 层倒流回嵌入，环的大小 = 那里真实的 ‖∂L/∂h‖，层板亮度 / 左边的梯度条 = 这一层真实的梯度长度；
//   更新：每层按真实的 ‖Δw‖ 闪紫光；拆开的那一层里每块权重面板贴着真实的梯度 / Δw 缩略图。
// 和推理页的机器一样，一切都由 update(st) 根据当前步骤算出来：暂停、单步、回退都能正确显示。
import { THREE, label, textTexture, easeOut, easeInOut, seg } from '../../../js/stage/engine.js';
import { RoundedBoxGeometry } from '../../../js/vendor/three/addons/geometries/RoundedBoxGeometry.js';
import { tokPlain, shortSpecial, fmtPct, esc } from '../../../js/ui.js';
import { seqInto, LIN } from '../glass3d/palette.js';
import { LAYER_OPS, UPD_TENSORS, ADAM_SUBS, fwdOps, OP_TENSORS, OP_TENSORS_BWD } from './timeline.js';
import { tname, TSHAPE, isVec, MAT_T } from './data.js';
import { t as T_, L } from './lang.js';

export const ROLE = { user: 0x5ee4f0, answer: 0xffb65c, tpl: 0xb39dff };
const S = 0.42;          // 词元间距（和推理页一样）
const Y0 = 0.95;         // 第 0 层高度
const GAP = 0.26;        // 层间距
const DEPTH = 1.8;       // 机器进深
const EXP = 4.75;        // 拆开一层时撑开的空间
// 拆开的一层里各部分的高度（相对这一层的底，× 拆开程度）
const SUB = { ln1: 0.24, attnB: 0.5, attnT: 1.86, add1: 2.06, ln2: 2.32, mlpB: 2.58, mlpT: 4.44, add2: 4.6 };
const OPY = { ln1: SUB.ln1, qkv: 0.86, attn: 1.42, o: SUB.add1, ln2: SUB.ln2, ffn: 3.1, down: SUB.add2 };
const U = 0.62 / 1024;   // 权重面板：每一维的长度
const PZ = -1.2;         // 权重面板在光柱后面
// 面板：x0、底边高度（相对层底）、宽 = 输出维、高 = 输入维；γ 是细条
const PANEL = {
  q: { x0: -2.75, y: SUB.attnB, w: 2048 * U, h: 1024 * U, col: 0x5ef0d4 },
  k: { x0: -1.2, y: SUB.attnB, w: 1024 * U, h: 1024 * U, col: 0xffb65c },
  v: { x0: -0.3, y: SUB.attnB, w: 1024 * U, h: 1024 * U, col: 0xb39dff },
  o: { x0: 0.75, y: SUB.attnB, w: 1024 * U, h: 2048 * U, col: 0x8fb8ff },
  gate: { x0: -3.1, y: SUB.mlpB, w: 3072 * U, h: 1024 * U, col: 0x5ef0d4 },
  up: { x0: -0.95, y: SUB.mlpB, w: 3072 * U, h: 1024 * U, col: 0xb39dff },
  down: { x0: 1.3, y: SUB.mlpB, w: 1024 * U, h: 3072 * U, col: 0xffb65c },
  qn: { x0: -2.75, y: SUB.attnB + 1024 * U + 0.08, w: 0.5, h: 0.045, col: 0x6b9bff, vec: true },
  kn: { x0: -1.2, y: SUB.attnB + 1024 * U + 0.08, w: 0.5, h: 0.045, col: 0x6b9bff, vec: true },
  ln1: { x0: -2.75, y: SUB.ln1 - 0.02, w: 1024 * U, h: 0.045, col: 0x6b9bff, vec: true },
  ln2: { x0: -3.1, y: SUB.ln2 - 0.02, w: 1024 * U, h: 0.045, col: 0x6b9bff, vec: true },
};
const EMB = { x0: -2.2, x1: 2.2, y0: 0.08, y1: 0.86, z: PZ };   // 嵌入表（托盘后面）

const col = (hex) => new THREE.Color(hex);
const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
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
  bar: (() => { const g = new THREE.BoxGeometry(0.3, 1, 0.3); g.translate(0, 0.5, 0); return g; })(),
  thin: (() => { const g = new THREE.BoxGeometry(0.12, 1, 0.12); g.translate(0, 0.5, 0); return g; })(),
  disc: new THREE.CylinderGeometry(0.075, 0.075, 0.016, 16),
  meter: (() => { const g = new THREE.BoxGeometry(1, 0.05, 0.12); g.translate(-0.5, 0, 0); return g; })(),
  unitBox: new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
};
Object.values(GEO).forEach((g) => { g.userData.shared = true; });

// 灰阶数据 → 一张带透明度的贴图（缩略图：权重 / 梯度 / Δw）
function heatTexture(data, w, h, color, { flip = false, gamma = 1, base = 0 } = {}) {
  const c = col(color), out = new Uint8Array(w * h * 4);
  for (let r = 0; r < h; r++) for (let x = 0; x < w; x++) {
    const src = flip ? (h - 1 - r) * w + x : r * w + x;
    const v = Math.pow(data[src] / 255, gamma), o = (r * w + x) * 4;
    out[o] = Math.round(255 * (base + (1 - base) * c.r * v)); out[o + 1] = Math.round(255 * (base + (1 - base) * c.g * v)); out[o + 2] = Math.round(255 * (base + (1 - base) * c.b * v));
    out[o + 3] = Math.round(255 * Math.min(1, 0.08 + v));
  }
  const t = new THREE.DataTexture(out, w, h, THREE.RGBAFormat);
  t.magFilter = THREE.NearestFilter;
  t.minFilter = THREE.LinearFilter;
  t.colorSpace = THREE.SRGBColorSpace;
  t.needsUpdate = true;
  return t;
}

export class QwenMachine {
  constructor(E, R, app) {
    this.E = E;
    this.R = R;
    this.app = app;
    this.scene = E.scene;
    this.NL = R.NL;
    this.T = R.N1;           // 23 个输入位置
    this.W = (this.T + 1) * S + 1.0;
    this.x = (i) => (i - (this.T - 1) / 2) * S;
    this.xL = R.NL - 1;      // 拆开的层
    this.e = 0;              // 拆开程度
    this.fade = 0;
    this.pk = '';
    this.texCache = new Map();
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.buildEnv();
    this.buildTray();
    this.buildTower();
    this.buildExploded();
    this.buildHead();
    this.buildLabels();
    this.buildMarks();
  }

  /* ================================================================ 搭机器 */

  buildEnv() {
    const floorTex = (() => {
      const c = document.createElement('canvas');
      c.width = c.height = 512;
      const g = c.getContext('2d');
      const gr = g.createRadialGradient(256, 256, 0, 256, 256, 256);
      gr.addColorStop(0, 'rgba(18,40,60,0.16)');
      gr.addColorStop(0.35, 'rgba(10,22,40,0.12)');
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
    floor.userData.noFocus = true;
    this.scene.add(floor);
    const grid = new THREE.GridHelper(80, 80, 0x1c3350, 0x10203a);
    grid.position.y = -0.29;
    grid.material.transparent = true;
    grid.material.opacity = 0.35;
    this.scene.add(grid);
    const n = 700, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { pos[i * 3] = (Math.random() - 0.5) * 70; pos[i * 3 + 1] = Math.random() * 30 - 2; pos[i * 3 + 2] = (Math.random() - 0.5) * 50; }
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.dust = new THREE.Points(pg, new THREE.PointsMaterial({ color: 0x7fd8e8, size: 0.05, transparent: true, opacity: 0.3, depthWrite: false }));
    this.scene.add(this.dust);
  }

  tileMat(role) {
    const c = ROLE[role] ?? ROLE.tpl;
    return new THREE.MeshStandardMaterial({ color: col(c).multiplyScalar(0.22), emissive: col(c), emissiveIntensity: 0.08, metalness: 0.25, roughness: 0.4, transparent: true, opacity: 1 });
  }

  tokTex(s, color, small = false) {
    const sp = /^<\|.*\|>$|^<\/?think>$/.test(s);
    const txt = sp ? shortSpecial(s) : tokPlain(s);
    return textTexture(txt, { color, font: sp || small ? '600 44px "JetBrains Mono",monospace' : undefined });
  }

  // 托盘：24 个词元（聊天模板渲染后的整段）+ labels 那一排 + 嵌入表
  buildTray() {
    const R = this.R, m = R.D.meta, N = m.ids.length;
    const tray = new THREE.Mesh(new THREE.BoxGeometry(this.W, 0.06, 0.8), new THREE.MeshStandardMaterial({ color: 0x0c1629, metalness: 0.6, roughness: 0.35 }));
    tray.position.y = -0.03;
    const te = new THREE.LineSegments(new THREE.EdgesGeometry(tray.geometry), new THREE.LineBasicMaterial({ color: 0x2c6b8a, transparent: true, opacity: 0.8 }));
    te.position.copy(tray.position);
    this.root.add(tray, te);
    // labels 的托盘：在前面一排
    const lt = new THREE.Mesh(new THREE.BoxGeometry(this.W, 0.04, 0.5), new THREE.MeshStandardMaterial({ color: 0x0a1222, metalness: 0.5, roughness: 0.4, transparent: true, opacity: 0.9 }));
    lt.position.set(0, -0.03, 0.78);
    const lte = new THREE.LineSegments(new THREE.EdgesGeometry(lt.geometry), new THREE.LineBasicMaterial({ color: 0x3a3060, transparent: true, opacity: 0.7 }));
    lte.position.copy(lt.position);
    this.labTray = [lt, lte];
    this.root.add(lt, lte);
    this.tiles = [];
    for (let j = 0; j < N; j++) {
      const role = m.roles[j];
      const g = new THREE.Group();
      const mat = this.tileMat(role);
      const body = new THREE.Mesh(GEO.tile, mat);
      const c = '#' + col(ROLE[role] ?? ROLE.tpl).getHexString();
      const face = new THREE.Mesh(GEO.face, new THREE.MeshBasicMaterial({ map: this.tokTex(R.tok(m.ids[j]), c), transparent: true }));
      face.position.z = 0.181;
      const top = face.clone();
      top.rotation.x = -Math.PI / 2;
      top.position.set(0, 0.121, 0);
      g.add(body, face, top);
      g.position.set(this.x(j), 0.12, 0);
      body.userData.pick = { type: 'tile', j, click: j < this.T && R.ans.includes(j) };
      this.E.pickables.push(body);
      g.mat = mat;
      top.material = face.material.clone();
      g.faces = [face.material, top.material];
      this.root.add(g);
      this.tiles.push(g);
    }
    // labels：和 ids 一样长，提示部分换成 −100；错开一位后位置 i 的答案是 labels[i+1]
    this.labs = [];
    const negTex = textTexture('−100', { color: '#5d6680', font: '600 44px "JetBrains Mono",monospace' });
    for (let j = 0; j < N; j++) {
      const counted = j >= 1 && R.D.meta.sftMask[j - 1];
      const g = new THREE.Group();
      const mat = this.tileMat(counted ? 'answer' : m.roles[j]);
      const body = new THREE.Mesh(GEO.tile, mat);
      const c = '#' + col(counted ? ROLE.answer : ROLE[m.roles[j]] ?? ROLE.tpl).getHexString();
      const front = new THREE.Mesh(GEO.face, new THREE.MeshBasicMaterial({ map: this.tokTex(R.tok(m.ids[j]), c), transparent: true }));
      front.position.z = 0.181;
      const back = new THREE.Mesh(GEO.face, new THREE.MeshBasicMaterial({ map: negTex, transparent: true }));
      back.position.z = -0.181;
      back.rotation.y = Math.PI;
      g.add(body, front, back);
      g.scale.setScalar(0.82);
      body.userData.pick = { type: 'lab', j, counted };
      this.E.pickables.push(body);
      g.mat = mat;
      g.counted = counted;
      g.visible = false;
      this.root.add(g);
      this.labs.push(g);
    }
    // 嵌入表（和 lm_head 是同一块权重）：一行一个词元，按编号从下往上排；这条对话用到的行画成亮线
    const ew = EMB.x1 - EMB.x0, eh = EMB.y1 - EMB.y0;
    const emb = new THREE.Mesh(new THREE.PlaneGeometry(ew, eh), new THREE.MeshBasicMaterial({ color: 0x1a1430, transparent: true, opacity: 0.55, depthWrite: false, side: THREE.DoubleSide }));
    emb.position.set((EMB.x0 + EMB.x1) / 2, (EMB.y0 + EMB.y1) / 2, EMB.z);
    emb.userData.pick = { type: 'emb', click: true };
    this.E.pickables.push(emb);
    const ee = new THREE.LineSegments(new THREE.EdgesGeometry(emb.geometry), new THREE.LineBasicMaterial({ color: 0xb39dff, transparent: true, opacity: 0.45 }));
    ee.position.copy(emb.position);
    this.embPlate = emb;
    this.embEdge = ee;
    this.root.add(emb, ee);
    const rows = new THREE.Group();
    this.embRowY = (id) => EMB.y0 + (id / 151936) * eh;
    const seen = new Set();
    for (let j = 0; j < this.T; j++) {
      const id = m.ids[j];
      if (seen.has(id)) continue;
      seen.add(id);
      const r = new THREE.Mesh(new THREE.PlaneGeometry(ew, 0.008), new THREE.MeshBasicMaterial({ color: ROLE[m.roles[j]] ?? ROLE.tpl, transparent: true, opacity: 0.7, depthWrite: false }));
      r.position.set((EMB.x0 + EMB.x1) / 2, this.embRowY(id), EMB.z + 0.003);
      r.raycast = () => {};
      rows.add(r);
    }
    this.embRows = rows;
    this.root.add(rows);
    // 查表的光：每个输入词元 → 它那一行
    this.look = [];
    for (let j = 0; j < this.T; j++) {
      const id = m.ids[j];
      const a = new THREE.Vector3(this.x(j), 0.25, -0.12), b = new THREE.Vector3(Math.max(EMB.x0 + 0.1, Math.min(EMB.x1 - 0.1, this.x(j) * 0.45)), this.embRowY(id), EMB.z + 0.01);
      const curve = new THREE.QuadraticBezierCurve3(a, new THREE.Vector3((a.x + b.x) / 2, Math.max(a.y, b.y) + 0.35, (a.z + b.z) / 2), b);
      const mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, 20, 0.008, 5, false), new THREE.MeshBasicMaterial({ color: 0xb39dff, transparent: true, opacity: 0, depthWrite: false }));
      mesh.raycast = () => {};
      this.root.add(mesh);
      this.look.push(mesh);
    }
  }

  // 28 块层板 + 23 根光柱 + 逻辑透镜的小圆片 + 左边的梯度条
  buildTower() {
    const R = this.R, T = this.T, NL = this.NL, m = R.D.meta;
    this.columns = [];
    for (let i = 0; i < T; i++) {
      const isAns = R.ans.includes(i);
      const c = isAns ? 0xffb65c : ROLE[m.roles[i]] ?? ROLE.tpl;
      const cm = new THREE.MeshBasicMaterial({ color: col(c).multiplyScalar(0.72), transparent: true, opacity: 0.4, depthWrite: false });
      const cyl = new THREE.Mesh(GEO.column, cm);
      cyl.position.set(this.x(i), 0.25, 0);
      cyl.scale.y = 0.001;
      cyl.raycast = () => {};
      this.root.add(cyl);
      cyl.isAns = isAns;
      this.columns.push(cyl);
    }
    this.slabs = [];
    const slabGeo = new THREE.BoxGeometry(this.W - 0.3, 0.03, DEPTH);
    const edgeGeo = new THREE.EdgesGeometry(slabGeo);
    for (let l = 0; l < NL; l++) {
      const mat = new THREE.MeshStandardMaterial({ color: 0x22346a, emissive: 0x5ef0d4, emissiveIntensity: 0, transparent: true, opacity: 0.05, roughness: 0.4, metalness: 0.1, depthWrite: false, envMap: this.scene.environment, envMapIntensity: 0.12 });
      const s = new THREE.Mesh(slabGeo, mat);
      const e = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.22 }));
      s.add(e);
      s.edge = e;
      s.userData.pick = { type: 'slab', L: l, click: true };
      this.E.pickables.push(s);
      const lb = label('', 'lbl lens');
      lb.position.set(this.W / 2 - 0.1, 0.05, DEPTH / 2);
      lb.center.set(0, 0.5);
      s.add(lb);
      s.lbl = lb;
      this.root.add(s);
      this.slabs.push(s);
    }
    // 逻辑透镜：回答的 8 根光柱 × 28 层，每层一片（亮 = 这一层就读得出正确答案）
    const A = R.ans.length;
    this.discs = new THREE.InstancedMesh(GEO.disc, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95 }), NL * A);
    this.discs.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(NL * A * 3), 3);
    this.discs.frustumCulled = false;
    this.discs.userData.pick = (hit) => ({ type: 'lens', L: Math.floor(hit.instanceId / A), a: hit.instanceId % A });
    this.E.pickables.push(this.discs);
    this.root.add(this.discs);
    // 梯度条（玫红）/ 更新量（紫）：每层一根，挂在层板左边
    const mk = (color, op) => {
      const im = new THREE.InstancedMesh(GEO.meter, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: op, depthWrite: false }), NL);
      im.frustumCulled = false;
      this.root.add(im);
      return im;
    };
    this.gMeter = mk(0xff6b93, 0.75);
    this.dMeter = mk(0xb39dff, 0.7);
    this.gMeter.userData.pick = (hit) => ({ type: 'gmeter', L: hit.instanceId });
    this.E.pickables.push(this.gMeter);
    // 光环：前向（青）往上，反向（玫红）往下
    const ring = (color) => { const im = new THREE.InstancedMesh(GEO.ring, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85, depthWrite: false }), T); im.frustumCulled = false; im.raycast = () => {}; this.root.add(im); return im; };
    this.pulseF = ring(0x9ffcff);
    this.pulseB = ring(0xff7aa0);
  }

  // 拆开的一层：RMSNorm 环、注意力的 W_q / W_k / W_v / W_o、⊕、RMSNorm、前馈的 W_gate / W_up / W_down、⊕
  buildExploded() {
    const g = (this.ex = new THREE.Group());
    this.root.add(g);
    const W = this.W;
    const ell = (color, sx = W / 2 - 0.1) => {
      const m = new THREE.Mesh(new THREE.TorusGeometry(1, 0.016, 8, 64), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.5, depthWrite: false }));
      m.rotation.x = Math.PI / 2;
      m.scale.set(sx, 0.34, 1);
      m.raycast = () => {};
      g.add(m);
      return m;
    };
    this.exRing = { ln1: ell(0x6b9bff), ln2: ell(0x6b9bff), add1: ell(0xe9eff9), add2: ell(0xe9eff9) };
    const plate = (color, op) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(W - 0.3, 0.02, DEPTH + 0.8), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.04, transparent: true, opacity: op, depthWrite: false, roughness: 0.2 }));
      m.add(new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.3 })));
      m.raycast = () => {};
      g.add(m);
      return m;
    };
    this.exAttn = plate(0x5ef0d4, 0.05);
    this.exMlp = plate(0xffb65c, 0.045);
    // 权重面板：底板（真实的权重分布，到了再贴）+ 梯度 / Δw 两层叠加
    this.panels = {};
    for (const [t, P] of Object.entries(PANEL)) {
      const grp = new THREE.Group();
      const base = new THREE.Mesh(new THREE.PlaneGeometry(P.w, P.h), new THREE.MeshBasicMaterial({ color: P.vec ? col(P.col).multiplyScalar(0.5) : 0xffffff, map: P.vec ? null : this.placeholderTex(P.col, t), transparent: true, opacity: 0.85, depthWrite: false, side: THREE.DoubleSide }));
      base.position.set(P.w / 2, P.h / 2, 0);
      base.userData.pick = { type: 'panel', t, click: true };
      this.E.pickables.push(base);
      const over = (c2) => {
        const m = new THREE.Mesh(new THREE.PlaneGeometry(P.w, P.h), new THREE.MeshBasicMaterial({ color: c2, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
        m.position.set(P.w / 2, P.h / 2, 0.004);
        m.raycast = () => {};
        return m;
      };
      const gOver = over(0xff6b93), dOver = over(0xb39dff);
      dOver.position.z = 0.008;
      const edge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(P.w + 0.02, P.h + 0.02)), new THREE.LineBasicMaterial({ color: P.col, transparent: true, opacity: 0.55 }));
      edge.position.set(P.w / 2, P.h / 2, 0.006);
      const scan = new THREE.Mesh(new THREE.PlaneGeometry(0.014, P.h), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, depthWrite: false }));
      scan.position.set(0, P.h / 2, 0.01);
      scan.raycast = () => {};
      grp.add(base, gOver, dOver, edge, scan);
      Object.assign(grp, { base, gOver, dOver, edge, scan, P, t });
      g.add(grp);
      this.panels[t] = grp;
    }
    // 注意力光束、每个位置“加回残差”的小柱（注意力 / 前馈各一排）
    this.beams = new THREE.Group();
    g.add(this.beams);
    this.beamKey = '';
    const deltas = (color) => { const im = new THREE.InstancedMesh(GEO.thin, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8, depthWrite: false }), this.T); im.frustumCulled = false; g.add(im); return im; };
    this.dAttn = deltas(0x5ef0d4);
    this.dMlp = deltas(0xffb65c);
    this.dAttn.userData.pick = (hit) => ({ type: 'delta', kind: 'attn', i: hit.instanceId });
    this.dMlp.userData.pick = (hit) => ({ type: 'delta', kind: 'mlp', i: hit.instanceId });
    this.E.pickables.push(this.dAttn, this.dMlp);
    g.visible = false;
  }

  // 没拿到真实缩略图之前的面板：很暗的格子
  placeholderTex(color, seed) {
    const c = document.createElement('canvas');
    c.width = c.height = 64;
    const g = c.getContext('2d');
    const base = col(color);
    g.fillStyle = 'rgba(10,16,30,1)';
    g.fillRect(0, 0, 64, 64);
    g.strokeStyle = `rgba(${Math.round(base.r * 255)},${Math.round(base.g * 255)},${Math.round(base.b * 255)},0.18)`;
    for (let i = 0; i <= 64; i += 8) { g.beginPath(); g.moveTo(i, 0); g.lineTo(i, 64); g.stroke(); g.beginPath(); g.moveTo(0, i); g.lineTo(64, i); g.stroke(); }
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }

  // 输出：最后的 RMSNorm 环 + γ 条、lm_head、每个位置的概率柱（+ 更新后的虚影）、−ln p、损失量筒
  buildHead() {
    const g = (this.head = new THREE.Group());
    this.root.add(g);
    const W = this.W, T = this.T;
    const ring = new THREE.Mesh(new THREE.TorusGeometry(1, 0.016, 8, 64), new THREE.MeshBasicMaterial({ color: 0x6b9bff, transparent: true, opacity: 0.5, depthWrite: false }));
    ring.rotation.x = Math.PI / 2;
    ring.scale.set(W / 2 - 0.1, 0.34, 1);
    ring.raycast = () => {};
    this.normRing = ring;
    const nst = new THREE.Mesh(new THREE.PlaneGeometry(1024 * U, 0.045), new THREE.MeshBasicMaterial({ color: col(0x6b9bff).multiplyScalar(0.6), transparent: true, opacity: 0.9, depthWrite: false, side: THREE.DoubleSide }));
    nst.userData.pick = { type: 'panel', t: 'norm', click: true };
    this.E.pickables.push(nst);
    this.normStrip = nst;
    this.lm = new THREE.Mesh(new THREE.BoxGeometry(W - 0.3, 0.05, DEPTH), new THREE.MeshStandardMaterial({ color: 0xb39dff, emissive: 0xb39dff, emissiveIntensity: 0.04, transparent: true, opacity: 0.1, depthWrite: false }));
    this.lm.add(new THREE.LineSegments(new THREE.EdgesGeometry(this.lm.geometry), new THREE.LineBasicMaterial({ color: 0xb39dff, transparent: true, opacity: 0.5 })));
    this.lm.userData.pick = { type: 'lm', click: true };
    this.E.pickables.push(this.lm);
    g.add(ring, nst, this.lm);
    // 概率柱：每个位置一根，高度 = 正确答案的真实概率
    const mk = (geo, mat, n) => { const im = new THREE.InstancedMesh(geo, mat, n); im.frustumCulled = false; g.add(im); return im; };
    // 颜色直接取实例颜色（不受灯光影响，暗处也看得出红 / 琥珀 / 绿）
    this.bars = mk(GEO.bar, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.88 }), T);
    this.bars.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(T * 3), 3);
    this.bars.userData.pick = (hit) => ({ type: 'bar', i: hit.instanceId, click: this.R.ans.includes(hit.instanceId) });
    this.E.pickables.push(this.bars);
    // 不算损失的位置：同样是真实概率，画成很暗的半透明柱子
    this.barsM = mk(GEO.bar, new THREE.MeshBasicMaterial({ color: 0x2a3858, transparent: true, opacity: 0.32, depthWrite: false }), T);
    this.barsM.userData.pick = (hit) => ({ type: 'bar', i: hit.instanceId });
    this.E.pickables.push(this.barsM);
    this.ghost = new THREE.InstancedMesh(new THREE.BoxGeometry(0.3, 1, 0.3).translate(0, 0.5, 0), new THREE.MeshBasicMaterial({ color: 0x8ee07a, transparent: true, opacity: 0.16, depthWrite: false }), T);
    this.ghost.frustumCulled = false;
    this.ghost.raycast = () => {};
    g.add(this.ghost);
    this.ghostTop = mk(new THREE.BoxGeometry(0.34, 0.02, 0.34), new THREE.MeshBasicMaterial({ color: 0xbff8c0, transparent: true, opacity: 0.85, depthWrite: false }), T);
    this.ghostTop.raycast = () => {};
    // −ln p：回答的位置实心（玫红），其余位置（预训练式也会算的）是虚影
    this.nll = mk(GEO.thin, new THREE.MeshBasicMaterial({ color: 0xff6b93, transparent: true, opacity: 0.85, depthWrite: false }), T);
    this.nll.userData.pick = (hit) => ({ type: 'nll', i: hit.instanceId });
    this.E.pickables.push(this.nll);
    this.nllPre = mk(GEO.thin, new THREE.MeshBasicMaterial({ color: 0x8fa6d6, transparent: true, opacity: 0.22, depthWrite: false }), T);
    this.nllPre.raycast = () => {};
    // 损失量筒
    const glass = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 2.4, 24, 1, true), new THREE.MeshStandardMaterial({ color: 0x1a2c50, transparent: true, opacity: 0.18, depthWrite: false, side: THREE.DoubleSide }));
    glass.raycast = () => {};
    const fg = new THREE.CylinderGeometry(0.17, 0.17, 1, 24);
    fg.translate(0, 0.5, 0);
    this.gauge = new THREE.Mesh(fg, new THREE.MeshStandardMaterial({ color: 0x3a1020, emissive: 0xff6b93, emissiveIntensity: 0.3, roughness: 0.4 }));
    this.gauge.userData.pick = { type: 'gauge' };
    this.E.pickables.push(this.gauge);
    this.gaugePre = new THREE.Mesh(new THREE.TorusGeometry(0.26, 0.012, 6, 32), new THREE.MeshBasicMaterial({ color: 0x8fa6d6, transparent: true, opacity: 0.7 }));
    this.gaugePre.rotation.x = Math.PI / 2;
    this.gaugePre.raycast = () => {};
    this.gaugeGlass = glass;
    g.add(glass, this.gauge, this.gaugePre);
  }

  buildLabels() {
    const add = (html, cls, cx = 0, cy = 0.5) => { const o = label(html, `lbl ${cls}`); o.center.set(cx, cy); this.root.add(o); return o; };
    this.lTray = add('', 'num qr', 1, 0.5);
    this.lLabs = add('', 'num qr', 1, 0.5);
    this.lMaskL = add('', 'num qm', 0.5, 1);
    this.lMaskR = add('', 'num qa', 0.5, 1);
    this.lPre = add('', 'num qp', 0.5, 1);
    this.lEmb = add('', 'part', 0, 0.5);
    this.lLm = add('', 'part', 1, 0.5);
    this.lNorm = add('', 'part', 0, 0.5);
    this.lGauge = add('', 'big', 1, 0.5);
    this.lGaugePre = add('', 'num qp', 1, 0.5);
    this.lClip = add('', 'num gc', 0.6, 0);
    this.lMeter = add('', 'num qg', 1, 0.5);
    this.lDmeter = add('', 'num gc', 1, 0.5);
    this.lTitle = add('', 'title', 0.5, 1);
    this.lSel = add('', 'num gsel', 0.5, 1.25);
    this.lAns = this.R.ans.map(() => add('', 'num qb', 0.5, 1));
    this.lCands = add('', 'num qc', 0, 0.5);
    this.lPanel = {};
    for (const t of Object.keys(PANEL)) this.lPanel[t] = add('', 'part', 0, 1);
    this.lEx = { ln1: add('RMSNorm', 'hint', 0, 0.5), ln2: add('RMSNorm', 'hint', 0, 0.5), add1: add(T_('q3.addRes'), 'hint', 0, 0.5), add2: add(T_('q3.addRes'), 'hint', 0, 0.5) };
    this.lDelta = add('', 'num', 0.5, 1);
  }

  buildMarks() {
    const box = (color) => {
      const m = new THREE.LineSegments(GEO.unitBox, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95, depthTest: false }));
      m.renderOrder = 10;
      m.visible = false;
      m.raycast = () => {};
      this.root.add(m);
      return m;
    };
    this.mHover = box(0xffffff);
    this.mSel = box(0xffd18a);
    this.mFocus = box(0xffd18a);
  }

  /* ================================================================ 布局 */

  yL(l) { return Y0 + l * GAP + (l > this.xL && this.xL >= 0 ? this.e * EXP : 0); }
  get yTop() { return this.yL(this.NL - 1) + 0.4; }
  get barsY() { return this.yTop + 1.05; }
  yS(key) { return Y0 + this.xL * GAP + SUB[key] * this.e; }
  // 前向 / 反向进度 f（以“层”为单位：−1 = 托盘，L..L+1 = 第 L 层里面，NL = 28 层结束，NL+1 = 概率）→ 光环的高度
  yAt(f) {
    const NL = this.NL;
    if (f < 0) return 0.25 + (Y0 - 0.25) * Math.max(0, 1 + f);
    if (f >= NL) return this.yTop + Math.min(1, f - NL) * 0.75;
    const l = Math.floor(f), fr = f - l;
    if (l === this.xL && this.e > 0.05) {
      const base = Y0 + l * GAP;
      const ks = [[0, 0], ...LAYER_OPS.map((op, oi) => [(oi + 0.5) / 7, OPY[op] * this.e]), [1, GAP + EXP * this.e]];
      for (let j = 1; j < ks.length; j++) if (fr <= ks[j][0]) { const [x0, y0] = ks[j - 1], [x1, y1] = ks[j]; return base + y0 + (y1 - y0) * (fr - x0) / (x1 - x0); }
      return base + GAP + EXP * this.e;
    }
    return this.yL(l) + fr * GAP;
  }
  panelPos(t) {
    if (t === 'norm') return { x0: -2.75, y0: this.yTop + 0.23, w: 1024 * U, h: 0.045, z: PZ };
    if (t === 'embed') return { x0: EMB.x0, y0: EMB.y0, w: EMB.x1 - EMB.x0, h: EMB.y1 - EMB.y0, z: EMB.z };
    const P = PANEL[t];
    return { x0: P.x0, y0: Y0 + this.xL * GAP + P.y * this.e, w: P.w, h: P.h, z: PZ };
  }

  /* ================================================================ 状态 */

  // 暂停时单步过来：这一步的动画自己播一遍再停住
  animP(st, t) {
    const key = `${st.depth}:${st.k}:${st.i}:${this.app.tl?.L}`;
    if (this.pk !== key) { this.pk = key; this.pt0 = t; }
    const dur = { 'q-run': 4.5, 'q-end': 2, 'q-step': 3.2, 'q-op': 1.8, 'q-mat': 1.8, 'q-param': 1.6 }[st.view] || 1.6;
    return st.playing ? st.p : Math.max(st.p, Math.min(1, (t - this.pt0) / dur));
  }

  // 这一刻机器走到了哪
  flow(st, p) {
    const s = st.step, NL = this.NL, Lx = this.app.tl?.L ?? NL - 1;
    const F = { tpl: 1, lab: 1, mask: 1, shift: 1, fwd: -2, lossUpTo: 0, lossMean: 0, lossPos: -1, pre: 0, bwd: NL + 2, bDone: false, clip: 0, upd: 0, updT: null, after: 0, next: 0, state: st.k, emb: 0 };
    const fwdDone = () => { F.fwd = NL + 1; };
    if (s.ph === 'end') { F.fwd = NL + 1; F.state = this.R.K; F.lossUpTo = 8; F.lossMean = 1; return F; }
    if (s.ph === 'run') {
      if (st.k === 0) { F.tpl = seg(p, 0, 0.05); F.lab = F.mask = seg(p, 0.05, 0.08); F.shift = seg(p, 0.08, 0.1); }
      F.fwd = -1 + (NL + 2) * seg(p, 0.1, 0.42);
      F.lossUpTo = 8 * seg(p, 0.42, 0.48); F.lossMean = seg(p, 0.46, 0.52);
      F.bwd = NL + 1 - (NL + 2) * seg(p, 0.53, 0.77); F.bDone = p >= 0.77;
      F.clip = seg(p, 0.77, 0.8); F.upd = seg(p, 0.8, 0.9); F.after = seg(p, 0.88, 0.94); F.next = seg(p, 0.93, 1);
      return F;
    }
    const PH = ['batch', 'fwd', 'loss', 'bwd', 'upd'].indexOf(s.ph);
    if (PH >= 2) { fwdDone(); }
    if (PH >= 3) { F.lossUpTo = 8; F.lossMean = 1; }
    if (PH >= 4) { F.bwd = -1; F.bDone = true; }
    if (!s.sub) {
      if (s.ph === 'batch') { F.tpl = easeOut(seg(p, 0, 0.35)); F.lab = seg(p, 0.35, 0.5); F.mask = seg(p, 0.5, 0.75); F.shift = easeInOut(seg(p, 0.78, 1)); F.pre = seg(p, 0.36, 0.5) * (1 - seg(p, 0.6, 0.75)); }
      else if (s.ph === 'fwd') F.fwd = Math.min(NL + 1, -1 + p * (NL + 2.4));
      else if (s.ph === 'loss') { F.lossUpTo = 8 * seg(p, 0, 0.55); F.lossMean = easeOut(seg(p, 0.55, 0.8)); F.pre = seg(p, 0.8, 1); }
      else if (s.ph === 'bwd') { F.bwd = Math.max(-1, NL + 1 - p * (NL + 2.4)); F.bDone = p > 0.98; }
      else if (s.ph === 'upd') { F.clip = seg(p, 0, 0.2); F.upd = seg(p, 0.22, 0.65); F.after = seg(p, 0.7, 1); }
      return F;
    }
    switch (s.ph) {
      case 'batch':
        if (s.sub === 'tpl') { F.tpl = easeOut(p); F.lab = F.mask = F.shift = 0; }
        else if (s.sub === 'mask') { F.lab = seg(p, 0, 0.25); F.mask = seg(p, 0.35, 0.9); F.shift = 0; F.pre = seg(p, 0.05, 0.25) * (1 - seg(p, 0.4, 0.6)); }
        else F.shift = easeInOut(seg(p, 0.1, 0.8));
        break;
      case 'fwd': case 'bwd': {
        const ops = fwdOps(Lx, NL);
        const op = s.sub;
        let v;
        const pp = s.mi ? 1 : s.t ? (() => { const ts = (s.ph === 'fwd' ? OP_TENSORS : OP_TENSORS_BWD)[op]; return (ts.indexOf(s.t) + easeInOut(seg(p, 0, 0.8))) / ts.length; })() : easeInOut(seg(p, 0, 0.85));
        if (op === 'emb') v = -1 + pp;
        else if (op === 'lo') v = pp * Lx;
        else if (op === 'hi') v = Lx + 1 + pp * (NL - 1 - Lx);
        else if (op === 'head') v = NL + pp;
        else v = Lx + (LAYER_OPS.indexOf(op) + pp) / 7;
        if (s.ph === 'fwd') { F.fwd = v; F.emb = op === 'emb' ? 1 : 0; }
        else {
          // 反向：把前向的位置倒过来走（v 是“前向走到哪”，反向从 v 的终点往起点退）
          const ops2 = { emb: [0, -1], lo: [Lx, 0], hi: [NL, Lx + 1], head: [NL + 1, NL] };
          if (ops2[op]) { const [a, b] = ops2[op]; F.bwd = a + (b - a) * pp; }
          else { const oi = LAYER_OPS.indexOf(op); F.bwd = Lx + (oi + 1 - pp) / 7; }
          F.emb = op === 'emb' ? 1 : 0;
        }
        void ops;
        break;
      }
      case 'loss':
        if (s.sub === 'pos') { const a = this.R.ans.indexOf(s.i); F.lossPos = s.i; F.lossUpTo = a + easeOut(seg(p, 0, 0.5)); }
        else { F.lossUpTo = 8; F.lossMean = easeOut(seg(p, 0, 0.55)); F.pre = seg(p, 0.5, 0.9); }
        break;
      case 'upd': {
        if (s.sub === 'clip') { F.clip = seg(p, 0, 0.6); break; }
        if (s.sub === 'bf16') { F.clip = 1; F.upd = 1; F.after = 1; break; }
        F.clip = 1;
        if (!s.t) { F.upd = seg(p, 0, 0.7); F.after = seg(p, 0.72, 1); break; }
        F.updT = {};
        const ti = UPD_TENSORS.indexOf(s.t);
        for (let j = 0; j < ti; j++) F.updT[UPD_TENSORS[j]] = 1;
        if (s.mi) { const mi = ADAM_SUBS.indexOf(s.mi); F.updT[s.t] = mi < 4 ? 0 : mi === 4 ? easeOut(p) * 0.999 : 1; }
        else F.updT[s.t] = seg(p, 0.1, 0.8);
        F.upd = ti / UPD_TENSORS.length;
        break;
      }
    }
    return F;
  }

  /* ================================================================ 每帧 */

  update(st, dt, t) {
    const R = this.R, s = st.step, NL = this.NL, T = this.T;
    this.small = this.E.w < 700;
    const p = this.animP(st, t);
    const F = (this.F = this.flow(st, p));
    this.st = st;
    this.p = p;
    const k = st.k, view = st.view;
    const haveSt = R.hasSt(k), have3 = R.has3(k);
    const Lx = this.app.tl?.L ?? NL - 1;
    const focus = F.lossPos >= 0 ? F.lossPos : this.app.ctx.qPos ?? R.ans[1];
    this.focus = focus;

    // 拆开的层：D3 往下的前向 / 反向 / 更新（不含整段换层带过的 lo / hi、嵌入、输出头）
    const inLayer = st.depth >= 3 && ((s.ph === 'fwd' || s.ph === 'bwd') ? LAYER_OPS.includes(s.sub) : s.ph === 'upd' && s.sub === 'adam' && (!s.t || !['embed', 'norm'].includes(s.t)));
    const wantEx = inLayer || (st.depth >= 3 && (s.ph === 'fwd' || s.ph === 'bwd') && (s.sub === 'lo' || s.sub === 'hi') && false);
    if (wantEx && Lx !== this.xL && this.e < 0.02) this.xL = Lx;
    const eT = wantEx && Lx === this.xL ? 1 : 0;
    this.e += (eT - this.e) * Math.min(1, dt * 3.2);
    if (this.e < 0.003) this.e = 0;

    // 越深越把无关的东西淡出
    const FADE = { 'q-run': 0, 'q-end': 0, 'q-step': 0, 'q-op': 0.2, 'q-mat': 0.55, 'q-param': 0.62 };
    this.fade += ((FADE[view] ?? 0) - this.fade) * Math.min(1, dt * 3);
    // 进料时层板、光柱先退到后面（看托盘）
    const tf = s.ph === 'batch' && st.depth >= 2 ? 0.7 : 0;
    this.towerFade = (this.towerFade || 0) + (tf - (this.towerFade || 0)) * Math.min(1, dt * 3);
    const fade = this.fade;
    const BLOOM = { 'q-run': 0.3, 'q-end': 0.3, 'q-step': 0.28, 'q-op': 0.24, 'q-mat': 0.18, 'q-param': 0.16 };
    this.E.bloom.strength += ((BLOOM[view] ?? 0.26) - this.E.bloom.strength) * Math.min(1, dt * 3);

    this.updTray(st, F, focus);
    this.updTower(st, F, haveSt, have3, focus, t, dt);
    this.updExploded(st, F, have3, focus, t);
    this.updHead(st, F, focus);
    this.updLabels(st, F, haveSt, have3, focus);
    this.updMarks(st, F, have3);
    this.dust.rotation.y += dt * 0.008;
  }

  // 词元方块（落进托盘）、labels 那一排（−100 翻面、错开一位）、嵌入表和查表的光
  updTray(st, F, focus) {
    const R = this.R, N = this.tiles.length, T = this.T;
    const fadeTray = this.fade > 0.4 ? 0.35 : 1;
    // 推到嵌入表跟前时，托盘上的词元挡在前面：变得几乎透明
    const embClose = (st.view === 'q-mat' || st.view === 'q-param') && this.selT === 'embed';
    this.tiles.forEach((g, j) => {
      const k = seg(F.tpl, (j / N) * 0.7, (j / N) * 0.7 + 0.3);
      g.visible = k > 0.01;
      g.position.y = 0.12 + (1 - easeOut(k)) * 2.2;
      // 第 24 个词元（<|im_end|> 后面的换行）只当答案用、不当输入：错开一位以后淡下去
      const notIn = j >= T;
      g.mat.opacity = (notIn ? 1 - 0.75 * F.shift : 1) * (embClose ? 0.12 : 1);
      g.mat.depthWrite = !embClose;
      for (const fm of g.faces) fm.opacity = embClose ? 0.15 : 1;
      const hl = j === focus && F.fwd > -2 && st.view !== 'q-run' ? 1.12 : 1;
      g.scale.setScalar(hl);
      g.mat.emissiveIntensity = (0.08 + (j === focus ? 0.12 : 0)) * fadeTray;
    });
    const labOn = F.lab > 0.01 && !embClose;
    this.labTray.forEach((m) => { m.visible = labOn; });
    this.labs.forEach((g, j) => {
      g.visible = labOn && !(j === 0 && F.shift > 0.6);
      if (!g.visible) return;
      const from = this.x(j), to = this.x(j - 1);
      const e = easeInOut(F.shift);
      g.position.set(from + (to - from) * e, 0.1 + (1 - easeOut(F.lab)) * 0.5 + Math.sin(e * Math.PI) * 0.18, 0.78);
      // 不算损失的翻到背面（−100）
      const flip = g.counted ? 0 : easeInOut(seg(F.mask, (j / N) * 0.6, (j / N) * 0.6 + 0.4));
      g.rotation.y = flip * Math.PI;
      g.mat.opacity = g.counted ? 1 : 1 - 0.35 * flip;
      g.mat.emissiveIntensity = g.counted ? 0.14 + (j - 1 === focus ? 0.18 : 0) : 0.04;
      g.scale.setScalar(0.82 * (g.counted && j - 1 === focus && F.shift > 0.9 ? 1.12 : 1));
    });
    // 嵌入表：查表时、看嵌入张量时亮
    const embOn = F.emb > 0 || this.selT === 'embed';
    const embK = st.view === 'q-run' || st.view === 'q-end' || st.view === 'q-step' ? 0.35 : 0.6;
    this.embPlate.material.opacity = (embOn ? 0.8 : 0.4) * embK;
    this.embEdge.material.opacity = embOn ? 0.8 : 0.28;
    this.embRows.visible = true;
    const fe = F.emb > 0 && st.step.ph === 'fwd' ? clamp01(F.fwd + 1) : F.emb > 0 && st.step.ph === 'bwd' ? clamp01(-F.bwd) : -1;
    this.look.forEach((m, j) => {
      const on = fe >= 0 && fe * 1.3 > j / T;
      m.material.opacity = on ? 0.55 : 0;
      m.material.color.setHex(st.step.ph === 'bwd' ? 0xff6b93 : 0xb39dff);
    });
  }

  // 光柱、层板、光环、逻辑透镜、梯度条
  updTower(st, F, haveSt, have3, focus, t, dt) {
    const R = this.R, NL = this.NL, T = this.T, k = st.k, s = st.step;
    const inside = 1;
    const fade = Math.max(this.fade, this.towerFade || 0);
    const fwdOn = F.fwd > -1.5;
    // 光柱：前向走到哪长到哪
    this.columns.forEach((c, i) => {
      c.visible = F.tpl > 0.9;
      let top = this.yTop + 0.55;
      if (F.fwd < NL + 1 && F.fwd > -1.99) top = Math.min(top, this.yAt(F.fwd));
      if (F.fwd <= -1.99) top = 0.26;
      c.scale.y = Math.max(0.001, top - 0.25);
      const ans = c.isAns;
      c.material.opacity = (ans ? 0.62 : 0.3) * (i === focus ? 1.25 : 1) * (1 - fade * (i === focus ? 0.2 : 0.75)) * (this.fade > 0.45 ? 0.25 : 1);
      c.scale.x = c.scale.z = i === focus ? 1.5 : ans ? 1.15 : 1;
    });
    // 前向光环
    let n = 0;
    if (F.fwd > -1 && F.fwd < NL + 1) {
      const y = this.yAt(F.fwd);
      for (let i = 0; i < T; i++) {
        tmpM.makeRotationX(Math.PI / 2).setPosition(this.x(i), y, 0);
        if (i === focus) tmpM.scale(tmpV.set(1.4, 1.4, 1.4));
        this.pulseF.setMatrixAt(n++, tmpM);
      }
    }
    this.pulseF.count = n;
    this.pulseF.instanceMatrix.needsUpdate = true;
    this.pulseF.material.opacity = 0.5 + 0.2 * Math.sin(t * 9);
    // 反向光环：大小 = 这里真实的 ‖∂L/∂h‖（这一步所有层、所有位置里的最大值 = 满格）
    n = 0;
    if (F.bwd < NL + 1 && F.bwd > -1 && !F.bDone) {
      const y = this.yAt(F.bwd);
      const b = Math.max(0, Math.min(NL, Math.ceil(F.bwd)));
      const mx = this.gMax(k);
      for (let i = 0; i < T; i++) {
        const gv = haveSt ? R.residGrad(k, b, i) : 0;
        const sc = haveSt ? 0.25 + 1.9 * Math.sqrt(gv / mx) : 1;
        tmpM.makeRotationX(Math.PI / 2).scale(tmpV.set(sc, sc, sc)).setPosition(this.x(i), y, 0);
        this.pulseB.setMatrixAt(n++, tmpM);
      }
    }
    this.pulseB.count = n;
    this.pulseB.instanceMatrix.needsUpdate = true;
    this.pulseB.material.opacity = 0.55 + 0.2 * Math.sin(t * 9);
    // 层板：前向经过时青光，反向经过后留着玫红（亮度 = 这一层真实的梯度长度），更新时紫光（= 真实的 ‖Δw‖）
    const gL = this.layerG(k), dL = this.layerD(k);
    const cur = s.ph === 'fwd' || s.ph === 'run' ? Math.floor(F.fwd) : s.ph === 'bwd' ? Math.floor(F.bwd) : -9;
    this.slabs.forEach((sl, l) => {
      sl.position.y = this.yL(l);
      const passedF = F.fwd >= l + 1, passedB = F.bwd <= l, onF = Math.floor(F.fwd) === l && F.fwd < NL, onB = Math.floor(F.bwd) === l && !F.bDone;
      const gi = gL ? gL[l] : 0, di = dL ? dL[l] : 0;
      let em = 0, emCol = 0x5ef0d4;
      if (F.upd > 0 && F.upd < 1 && dL) { em = 0.08 + 0.32 * di * Math.sin(Math.PI * clamp01(F.upd * 1.1)); emCol = 0xb39dff; }
      else if (passedB && gL && F.bwd < NL + 1) { em = (onB ? 0.4 : 0.06 + 0.22 * gi) * (F.next > 0 ? 1 - F.next : 1); emCol = 0xff6b93; }
      else if (onF) em = 0.26;
      else if (passedF && fwdOn) em = 0.04;
      sl.material.emissive.setHex(emCol);
      sl.material.emissiveIntensity += (em - sl.material.emissiveIntensity) * Math.min(1, dt * 8);
      const on = l === cur || (l === this.xL && this.e > 0.3);
      sl.material.opacity = Math.max(0, (0.045 + (on ? 0.08 : 0)) * inside - (l === this.xL ? 0.035 * this.e : 0)) * (1 - fade * 0.7);
      sl.edge.material.opacity = (on ? 0.8 : passedF || passedB ? 0.19 : 0.12) * (1 - fade * 0.6);
      sl.edge.material.color.setHex(passedB && F.bwd < NL + 1 && !(F.upd > 0) ? 0xff8fb0 : 0x5ef0d4);
    });
    // 逻辑透镜的小圆片：前向经过这一层之后亮起（颜色 = 这一层读出正确答案的概率）
    const A = R.ans.length;
    const da = this.discs.instanceColor.array;
    const showDiscs = fwdOn && haveSt && fade < 0.5;
    for (let l = 0; l < NL; l++) {
      for (let a = 0; a < A; a++) {
        const id = l * A + a, i = R.ans[a];
        if (!showDiscs || F.fwd < l + 1) { this.discs.setMatrixAt(id, ZERO); continue; }
        const pv = R.lensP(k, l + 1, i);
        tmpM.makeTranslation(this.x(i), this.yL(l) + 0.045, 0);
        if (i === focus) tmpM.scale(tmpV.set(1.25, 1, 1.25));
        this.discs.setMatrixAt(id, tmpM);
        seqInto(da, id * 3, 0.12 + 0.88 * Math.pow(pv, 0.6), 0.25 + 0.75 * Math.pow(pv, 0.5));
      }
    }
    this.discs.instanceMatrix.needsUpdate = true;
    this.discs.instanceColor.needsUpdate = true;
    // 梯度条 / 更新量条（左边）
    const showG = gL && F.bwd <= NL && fade < 0.5 && st.view !== 'q-end';
    const showD = dL && F.upd > 0.02 && fade < 0.5;
    const x0 = -this.W / 2 + 0.1;
    for (let l = 0; l < NL; l++) {
      const y = this.yL(l);
      if (showG && F.bwd <= l + 1) { tmpM.makeScale(0.12 + 1.5 * gL[l], 1, 1).setPosition(x0, y, 0.55); this.gMeter.setMatrixAt(l, tmpM); } else this.gMeter.setMatrixAt(l, ZERO);
      if (showD) { tmpM.makeScale((0.12 + 1.5 * dL[l]) * Math.min(1, F.upd * 1.6), 1, 1).setPosition(x0, y + 0.06, 0.62); this.dMeter.setMatrixAt(l, tmpM); } else this.dMeter.setMatrixAt(l, ZERO);
    }
    this.gMeter.instanceMatrix.needsUpdate = this.dMeter.instanceMatrix.needsUpdate = true;
    this.gMeter.material.opacity = 0.7 * (F.next > 0 ? 1 - F.next : 1);
  }

  // 第 k 步每层梯度长度 / 更新量（归一化到这一步各层的最大值）
  layerG(k) {
    if (!this.R.hasSt(k)) return null;
    this.gCache ||= {};
    if (this.gCache[k]) return this.gCache[k];
    const v = []; for (let l = 0; l < this.NL; l++) v.push(this.R.layerGrad(k, l));
    const mx = Math.max(...v);
    return (this.gCache[k] = v.map((x) => x / mx));
  }
  layerD(k) {
    if (!this.R.has3(k)) return null;
    this.dCache ||= {};
    if (this.dCache[k]) return this.dCache[k];
    const v = []; for (let l = 0; l < this.NL; l++) v.push(this.R.layerDw(k, l));
    // 第 1 步各层几乎一样长（Adam 第一步每个权重都挪 lr）：按最大值归一化，看得出“都差不多”
    const mx = Math.max(...v);
    return (this.dCache[k] = v.map((x) => x / mx));
  }
  gMax(k) {
    this.gmCache ||= {};
    if (this.gmCache[k]) return this.gmCache[k];
    let mx = 1e-9;
    if (this.R.hasSt(k)) for (let b = 0; b <= this.NL; b++) for (let i = 0; i < this.T; i++) mx = Math.max(mx, this.R.residGrad(k, b, i));
    return (this.gmCache[k] = mx);
  }

  // 拆开的一层
  updExploded(st, F, have3, focus, t) {
    const g = this.ex, e = this.e, R = this.R, s = st.step, k = st.k, l = this.xL;
    g.visible = e > 0.02;
    this.beams.visible = false;
    // 父级藏起来时引擎的拾取看不出来：自己关掉
    const pickOn = (on) => { for (const grp of Object.values(this.panels)) grp.base.userData.pickOn = on && grp.visible; for (const b of this.beams.children) b.userData.pickOn = on && this.beams.visible; };
    if (!g.visible) { this.dAttn.count = this.dMlp.count = 0; pickOn(false); return; }
    const base = Y0 + l * GAP;
    const at = (key) => base + SUB[key] * e;
    for (const key of ['ln1', 'ln2', 'add1', 'add2']) this.exRing[key].position.set(0, at(key), 0);
    this.exAttn.position.set(0, at('attnB') - 0.08, -0.3);
    this.exMlp.position.set(0, at('mlpB') - 0.08, -0.3);
    this.exAttn.material.opacity = 0.05 * e;
    this.exMlp.material.opacity = 0.045 * e;
    const op = s.ph === 'fwd' || s.ph === 'bwd' ? s.sub : null;
    const opOn = (o) => (op === o ? 1 : 0);
    const flash = (o) => opOn(o) * (0.5 + 0.5 * Math.sin(this.p * Math.PI));
    this.exRing.ln1.material.opacity = (0.3 + 0.55 * flash('ln1')) * e;
    this.exRing.ln2.material.opacity = (0.3 + 0.55 * flash('ln2')) * e;
    this.exRing.add1.material.opacity = (0.25 + 0.6 * flash('o')) * e;
    this.exRing.add2.material.opacity = (0.25 + 0.6 * flash('down')) * e;
    // 面板：真实的权重分布（推理页导出的缩略图）+ 梯度 / Δw 缩略图
    const th = R.thumbs;
    if (th && this.thumbL !== l) this.applyThumbs(l, th);
    const tk = `${k}:${l}`;
    if (have3 && this.overKey !== tk) this.applyOver(k, l);
    const bwdOn = s.ph === 'bwd' || s.ph === 'upd' || (s.ph === 'run' && F.bwd <= l + 1);
    const gNorm = this.tensorG(k, l);
    for (const [tn, grp] of Object.entries(this.panels)) {
      const pp = this.panelPos(tn);
      grp.position.set(pp.x0, pp.y0, PZ);
      grp.scale.setScalar(1);
      grp.visible = e > 0.3;
      const mine = this.selT === tn;
      // 前向：这块正在算 → 扫描线；反向：梯度叠加的亮度 = 这个张量的梯度长度（这一层 11 个里最大的 = 满格）
      const fwdOp = s.ph === 'fwd' && OP_TENSORS[op]?.includes(tn);
      const bwdOp = s.ph === 'bwd' && OP_TENSORS_BWD[op]?.includes(tn);
      const passedB = s.ph === 'bwd' ? F.bwd <= this.tensorAt(tn) : s.ph === 'upd';
      const gi = gNorm ? gNorm[tn] : 0;
      grp.gOver.material.opacity = (passedB ? 0.25 + 0.6 * gi : 0) * (bwdOp || mine ? 1.3 : 1) * (s.ph === 'upd' ? 0.5 : 1);
      let dOp = 0;
      if (s.ph === 'upd') dOp = F.updT ? (F.updT[tn] ?? 0) : F.upd;
      grp.dOver.material.opacity = dOp > 0 ? 0.25 + 0.55 * Math.sin(Math.PI * Math.min(1, dOp)) + (dOp >= 1 ? 0.15 : 0) : 0;
      grp.position.z = PZ + (dOp > 0 && dOp < 1 ? 0.08 * Math.sin(Math.PI * dOp) : 0);
      const active = fwdOp || bwdOp || mine || (s.ph === 'upd' && F.updT && F.updT[tn] > 0 && F.updT[tn] < 1);
      grp.edge.material.opacity = active ? 0.95 : 0.45;
      grp.edge.material.color.setHex(bwdOp ? 0xff6b93 : s.ph === 'upd' && dOp > 0 && dOp < 1 ? 0xb39dff : mine ? 0xffd18a : grp.P.col);
      grp.base.material.opacity = (0.55 + (active ? 0.35 : 0)) * (this.fade > 0.4 && !mine && !active ? 0.55 : 1);
      // 扫描线：前向时“这一列正在和输入做点积”
      if (fwdOp && !grp.P.vec) {
        const ts = OP_TENSORS[op], ti = ts.indexOf(tn);
        const f = s.t ? (s.t === tn ? seg(this.p, 0, 0.85) : -1) : seg(this.p, ti / ts.length * 0.85, (ti + 1) / ts.length * 0.85);
        grp.scan.material.opacity = f > 0 && f < 1 ? 0.75 : 0;
        grp.scan.position.x = grp.P.w * Math.max(0, f);
      } else grp.scan.material.opacity = 0;
    }
    // 注意力光束（真实的注意力，16 个头平均）：从选中的位置伸向它看的位置
    const showBeams = op === 'attn' && have3 && st.view === 'q-op';
    this.beams.visible = showBeams;
    if (showBeams) this.updBeams(k, l, focus, s.ph === 'fwd' ? seg(this.p, 0.05, 0.7) : 1, s.ph === 'bwd');
    // 加回残差：每个位置注意力 / 前馈的输出有多长（真实），反向时换成中间那一点残差的梯度
    const showD = have3 && e > 0.6 && st.view === 'q-op' && (op === 'o' || op === 'down' || op === 'attn' || op === 'ffn' || op === 'ln2');
    const T = this.T;
    let mxA = 1e-9, mxM = 1e-9;
    if (showD) for (let i = 0; i < T; i++) { mxA = Math.max(mxA, R.attnOut(k, l, i)); mxM = Math.max(mxM, R.mlpOut(k, l, i)); }
    const growA = op === 'o' && s.ph === 'fwd' ? seg(this.p, 0.3, 0.9) : 1, growM = op === 'down' && s.ph === 'fwd' ? seg(this.p, 0.3, 0.9) : 1;
    let na = 0, nm = 0;
    for (let i = 0; i < T; i++) {
      if (showD && (op === 'o' || (s.ph === 'fwd' && ['ln2', 'ffn', 'down'].includes(op)) || (s.ph === 'bwd' && ['o', 'attn'].includes(op)))) {
        const h = 0.06 + 0.5 * R.attnOut(k, l, i) / mxA;
        tmpM.makeScale(1, h * growA, 1).setPosition(this.x(i), at('add1') + 0.02, 0.18);
        this.dAttn.setMatrixAt(na++, tmpM);
      }
      if (showD && op === 'down') {
        const h = 0.06 + 0.5 * R.mlpOut(k, l, i) / mxM;
        tmpM.makeScale(1, h * growM, 1).setPosition(this.x(i), at('add2') + 0.02, 0.18);
        this.dMlp.setMatrixAt(nm++, tmpM);
      }
    }
    this.dAttn.count = na; this.dMlp.count = nm;
    this.dAttn.instanceMatrix.needsUpdate = this.dMlp.instanceMatrix.needsUpdate = true;
    this.dAttn.material.color.setHex(s.ph === 'bwd' ? 0x7fe8d8 : 0x5ef0d4);
    pickOn(true);
  }

  // 反向在拆开的那一层里经过每个张量时的进度位置（以“层”为单位）
  tensorAt(tn) {
    const opOf = { ln1: 'ln1', q: 'qkv', qn: 'qkv', k: 'qkv', kn: 'qkv', v: 'qkv', o: 'o', ln2: 'ln2', gate: 'ffn', up: 'ffn', down: 'down' }[tn];
    const oi = LAYER_OPS.indexOf(opOf);
    return this.xL + (oi + 0.5) / 7;
  }

  tensorG(k, l) {
    if (!this.R.hasSt(k)) return null;
    const key = `${k}:${l}`;
    if (this.tgKey === key) return this.tgVal;
    const out = {};
    let mx = 1e-12;
    for (const tn of Object.keys(PANEL)) { out[tn] = this.R.gradOf(k, tname(l, tn)); mx = Math.max(mx, out[tn]); }
    for (const tn of Object.keys(out)) out[tn] = Math.sqrt(out[tn] / mx);
    this.tgKey = key;
    this.tgVal = out;
    return out;
  }

  applyThumbs(l, th) {
    this.thumbL = l;
    for (const tn of MAT_T) {
      const spec = th.index[`${l}:${tn}`], grp = this.panels[tn];
      if (!spec) continue;
      const key = `w:${l}:${tn}`;
      let tex = this.texCache.get(key);
      if (!tex) {
        tex = heatTexture(th.bytes.subarray(spec.offset, spec.offset + spec.w * spec.h), spec.w, spec.h, grp.P.col, { flip: true, gamma: 0.7, base: 0.12 });
        this.texCache.set(key, tex);
        if (this.texCache.size > 200) { const [k0, v0] = this.texCache.entries().next().value; v0.dispose(); this.texCache.delete(k0); }
      }
      grp.base.material.map = tex;
      grp.base.material.needsUpdate = true;
      grp.real = true;
    }
  }

  applyOver(k, l) {
    this.overKey = `${k}:${l}`;
    for (const tn of MAT_T) {
      const grp = this.panels[tn];
      for (const [kind, mesh, c] of [['g', grp.gOver, 0xff6b93], ['d', grp.dOver, 0xb39dff]]) {
        const th = this.R.thumb(k, l, tn, kind);
        if (!th) continue;
        const key = `${kind}:${k}:${l}:${tn}`;
        let tex = this.texCache.get(key);
        if (!tex) {
          // 缩略图第 0 行 = 输入第 0 维，贴图第 0 行在下面：和权重缩略图一样“输入 0 在底部”
          tex = heatTexture(th.data, th.w, th.h, c, { gamma: 1.6 });
          this.texCache.set(key, tex);
        }
        mesh.material.map = tex;
        mesh.material.color.setHex(0xffffff);
        mesh.material.needsUpdate = true;
      }
    }
  }

  updBeams(k, l, focus, grow, bwd) {
    const R = this.R, T = this.T;
    const key = `${k}|${l}|${focus}`;
    if (key !== this.beamKey) {
      this.beamKey = key;
      disposeTree(this.beams);
      this.beams.clear();
      const list = [];
      for (let j = 0; j <= focus; j++) list.push({ j, w: R.att(k, l, focus, j) });
      list.sort((a, b) => b.w - a.w);
      const top = list.slice(0, 6).filter((a) => a.w >= 0.02);
      const y0 = Y0 + l * GAP + OPY.attn * this.e + 0.02;
      const xq = this.x(focus);
      this.beamList = top.map(({ j, w }) => {
        const xk = this.x(j), dx = Math.abs(xq - xk);
        const curve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(xq, y0, 0.05), new THREE.Vector3((xq + xk) / 2, y0 + Math.min(1.3, 0.2 + dx * 0.14), 0.25), new THREE.Vector3(xk, y0, j === focus ? 0.3 : 0.05));
        const geo = new THREE.TubeGeometry(curve, 40, 0.01 + w * 0.045, 8, false);
        const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x4fd8c0, transparent: true, opacity: 0.2 + 0.6 * w, blending: THREE.AdditiveBlending, depthWrite: false }));
        m.userData.pick = { type: 'beam', j, w };
        this.E.pickables.push(m);
        this.beams.add(m);
        const lb = label(`${esc(tokPlain(R.inStr(j)))} <b>${fmtPct(w)}</b>`, 'lbl num');
        lb.position.copy(curve.getPoint(0.5));
        lb.center.set(0.5, 1.2);
        m.add(lb);
        return { m, geo, lb };
      });
      // 拾取列表里清掉已经销毁的光束
      this.E.pickables = this.E.pickables.filter((o) => o.userData.pick?.type !== 'beam' || this.beams.children.includes(o));
    }
    for (const b of this.beamList || []) {
      b.geo.setDrawRange(0, Math.floor(b.geo.index.count * grow / 6) * 6);
      b.m.material.color.setHex(bwd ? 0xff8fb0 : 0x4fd8c0);
      b.lb.visible = grow > 0.95 && !this.small;
    }
  }

  // 输出头：概率柱（真实）、更新后的虚影、−ln p、损失量筒、top-5 候选
  updHead(st, F, focus) {
    const R = this.R, T = this.T, NL = this.NL, s = st.step;
    const yTop = this.yTop, by = this.barsY;
    this.normRing.position.set(0, yTop + 0.25, 0);
    const np = this.panelPos('norm');
    this.normStrip.position.set(np.x0 + np.w / 2, np.y0 + np.h / 2, PZ);
    this.lm.position.set(0, yTop + 0.62, 0);
    const inHead = (s.ph === 'fwd' && s.sub === 'head') || (s.ph === 'run' && F.fwd > NL && F.fwd < NL + 1);
    this.normRing.material.opacity = inHead ? 0.9 : 0.35;
    this.lm.material.emissiveIntensity = inHead ? 0.16 : (s.ph === 'bwd' && s.sub === 'head') ? 0.14 : 0.04;
    this.lm.material.emissive.setHex(s.ph === 'bwd' && (s.sub === 'head' || !s.sub && F.bwd > NL) ? 0xff6b93 : 0xb39dff);
    this.normStrip.material.color.setHex(this.selT === 'norm' ? 0xffd18a : 0x3a5a99);
    // 概率：这一步开始时的状态；更新后（F.next）变到下一个状态
    const ks = Math.min(R.K, F.state), kn = Math.min(R.K, ks + 1);
    const rise = easeOut(seg(F.fwd, NL + 0.35, NL + 1));
    const ba = this.bars.instanceColor.array;
    const fade = this.fade;
    for (let i = 0; i < T; i++) {
      const ans = R.ans.includes(i);
      let pv = R.pState(ks, i);
      if (F.next > 0 && kn <= R.K) pv += (R.pState(kn, i) - pv) * easeInOut(F.next);
      const h = Math.max(0.002, pv * 2.6 * rise);
      tmpM.makeScale(1, h, 1).setPosition(this.x(i), by, -0.25);
      this.bars.setMatrixAt(i, ans ? tmpM : ZERO);
      this.barsM.setMatrixAt(i, ans ? ZERO : tmpM);
      const dim = (F.lossPos >= 0 && F.lossPos !== i ? 0.26 : 0.6) * (1 - fade * 0.4);
      if (ans) { const c = pv > 0.5 ? LIN.green : pv > 0.1 ? LIN.amber : LIN.rose; ba[i * 3] = c[0] * dim; ba[i * 3 + 1] = c[1] * dim; ba[i * 3 + 2] = c[2] * dim; }
      // 更新后的虚影（下一个状态的真实概率）
      if (F.after > 0 && kn <= R.K && ans && F.next < 1) {
        const p2 = R.pState(kn, i), h2 = Math.max(0.002, p2 * 2.6);
        tmpM.makeScale(1, h2 * easeOut(F.after), 1).setPosition(this.x(i), by, -0.25);
        this.ghost.setMatrixAt(i, tmpM);
        tmpM.makeTranslation(this.x(i), by + h2 * easeOut(F.after), -0.25);
        this.ghostTop.setMatrixAt(i, tmpM);
      } else { this.ghost.setMatrixAt(i, ZERO); this.ghostTop.setMatrixAt(i, ZERO); }
      // −ln p
      const nl = R.nllState(ks, i), f = ans ? clamp01(F.lossUpTo - R.ans.indexOf(i)) : 0;
      if (ans && f > 0) { tmpM.makeScale(1, Math.max(0.002, Math.min(12, nl) * 0.2 * f), 1).setPosition(this.x(i), by, 0.32); this.nll.setMatrixAt(i, tmpM); } else this.nll.setMatrixAt(i, ZERO);
      if (!ans && F.pre > 0) { tmpM.makeScale(1, Math.max(0.002, Math.min(12, nl) * 0.2 * F.pre), 1).setPosition(this.x(i), by, 0.32); this.nllPre.setMatrixAt(i, tmpM); } else this.nllPre.setMatrixAt(i, ZERO);
    }
    this.bars.instanceMatrix.needsUpdate = this.bars.instanceColor.needsUpdate = this.barsM.instanceMatrix.needsUpdate = true;
    // 推到一块面板 / 一个权重跟前时，输出头上的柱子会挡住拆开的第 27 层：收起来
    const close = st.view === 'q-mat' || st.view === 'q-param';
    this.head.visible = !close || this.selT === 'norm';
    this.barsM.visible = rise > 0.001 && !close;
    this.barsM.material.opacity = (F.lossPos >= 0 ? 0.16 : 0.32) * (1 - fade * 0.6);
    this.ghost.instanceMatrix.needsUpdate = this.ghostTop.instanceMatrix.needsUpdate = true;
    this.nll.instanceMatrix.needsUpdate = this.nllPre.instanceMatrix.needsUpdate = true;
    this.bars.visible = rise > 0.001 && !close;
    // 看最后那个 RMSNorm 的 γ 时只留环和 γ 条
    for (const o of [this.ghost, this.ghostTop, this.nll, this.nllPre]) o.visible = !close;
    for (const o of [this.bars, this.barsM, this.nll, this.gauge, this.lm, this.normStrip]) o.userData.pickOn = this.head.visible && o.visible;
    this.bars.material.opacity = 0.92 * (1 - fade * 0.5);
    // 损失量筒
    const gx = this.W / 2 + 0.9;
    this.gaugeGlass.position.set(gx, by + 1.2, 0);
    this.gauge.position.set(gx, by, 0);
    let L0 = R.lossState(ks);
    if (F.next > 0 && kn <= R.K) L0 += (R.lossState(kn) - L0) * easeInOut(F.next);
    this.gauge.scale.y = Math.max(0.001, Math.min(6, L0) * 0.4 * F.lossMean);
    this.gaugePre.position.set(gx, by + Math.min(6, R.lossPreState(ks)) * 0.4, 0);
    this.gaugePre.visible = F.pre > 0.3 && !close;
    this.gaugeGlass.visible = this.gauge.visible = (F.lossMean > 0.01 || st.view === 'q-end') && !close;
  }

  /* ================================================================ 标签 */

  setL(o, on, html) {
    o.visible = !!on;
    if (on && html != null && o.el._h !== html) { o.el.innerHTML = html; o.el._h = html; }
  }

  updLabels(st, F, haveSt, have3, focus) {
    const R = this.R, s = st.step, v = st.view, k = st.k, NL = this.NL, m = R.D.meta;
    const far = v === 'q-run' || v === 'q-end' || v === 'q-step';
    const sm = this.small;
    const ks = Math.min(R.K, F.state);
    // 托盘：两排的“行首”（输入 ids / labels）
    this.lTray.position.set(-this.W / 2 - 0.05, 0.12, 0);
    this.setL(this.lTray, (s.ph === 'batch' && (v === 'q-step' || v === 'q-op')), T_('q3.trayCap', { n: m.ids.length }));
    this.lLabs.position.set(-this.W / 2 - 0.05, 0.1, 0.78);
    this.setL(this.lLabs, F.lab > 0.5 && s.ph === 'batch', T_('q3.labCap'));
    // 遮罩的标注：提示部分 / 回答部分
    const showMask = s.ph === 'batch' && (s.sub === 'mask' || s.sub === 'shift' || !s.sub) && F.mask > 0.5;
    const a0 = R.ans[0], a1 = R.ans[R.ans.length - 1];
    const shiftX = (j) => this.x(j) + (this.x(j - 1) - this.x(j)) * easeInOut(F.shift);
    this.lMaskL.position.set((shiftX(1) + shiftX(a0)) / 2, 0.45, 0.78);
    this.setL(this.lMaskL, showMask, T_('q3.maskL', { n: a0 + 1 }));
    this.lMaskR.position.set((shiftX(a0 + 1) + shiftX(a1 + 1)) / 2, 0.45, 0.78);
    this.setL(this.lMaskR, showMask, T_('q3.maskR', { n: R.ans.length }));
    this.lPre.position.set(0, 0.62, 0.78);
    this.setL(this.lPre, s.ph === 'batch' && F.pre > 0.3, T_('q3.preAll', { n: R.N1 }));
    // 嵌入表 / lm_head / 最后的 RMSNorm
    this.lEmb.position.set(EMB.x1 + 0.1, EMB.y1, EMB.z);
    this.setL(this.lEmb, (F.emb > 0 || this.selT === 'embed') && !sm, `${T_('q3.emb')}<small>151936 × 1024</small>`);
    this.lLm.position.set(this.W / 2 - 0.2, this.yTop + 0.62, DEPTH / 2);
    this.setL(this.lLm, (v === 'q-step' && (s.ph === 'fwd' || s.ph === 'loss')) || (s.sub === 'head' && v !== 'q-param') || this.selT === 'embed' && s.sub === 'head', `lm_head<small>${T_('q3.lmSub')}</small>`);
    const np = this.panelPos('norm');
    this.lNorm.position.set(np.x0 + np.w + 0.08, np.y0, PZ);
    this.setL(this.lNorm, (s.sub === 'head' && !far) || this.selT === 'norm', `${T_('q3.normF')}<small>γ [1024]</small>`);
    // 损失量筒
    const lossOn = F.lossMean > 0.3 && v !== 'q-mat' && v !== 'q-param';
    const gx = this.W / 2 + 0.9, by = this.barsY;
    let L0 = R.lossState(ks);
    const kn = Math.min(R.K, ks + 1);
    if (F.next > 0.5) L0 = R.lossState(kn);
    // 标签放在量筒左边（右边是调试器）；预训练式“全都算”的对照写在同一个标签里
    this.lGauge.position.set(gx - 0.3, by + Math.min(6, L0) * 0.4 * F.lossMean, 0);
    const after = F.after > 0.5 && F.next < 0.5 && kn <= R.K ? ` → ${R.lossState(kn).toFixed(3)}` : '';
    const pre = F.pre > 0.3 && !sm ? `<small>${T_('q3.lossPre', { v: R.lossPreState(ks).toFixed(3), n: R.N1 })}</small>` : '';
    this.setL(this.lGauge, lossOn, `L = ${L0.toFixed(3)}${after}<small>${F.next > 0.5 ? T_('q3.lossAfter') : T_('q3.lossMean', { n: R.ans.length })}</small>${pre}`);
    this.lGaugePre.visible = false;
    // 裁剪
    const clipOn = s.ph === 'upd' && (s.sub === 'clip' || (!s.sub && F.clip > 0 && F.clip < 1)) && v !== 'q-param';
    this.lClip.position.set(gx, by - 0.12, 0);
    this.setL(this.lClip, clipOn, T_('q3.clip', { g: m.steps[k].gradNorm.toFixed(1), c: m.steps[k].clip.toFixed(5) }));
    // 梯度条 / 更新量条
    const gl = this.layerG(k);
    const x0 = -this.W / 2 + 0.1;
    this.lMeter.position.set(x0 - 1.7, this.yL(NL - 1) + 0.3, 0.55);
    this.setL(this.lMeter, gl && F.bwd <= 0 && F.upd <= 0 && (v === 'q-step' || v === 'q-run') && s.ph !== 'loss' && !sm, T_('q3.meterG'));
    this.lDmeter.position.set(x0 - 1.7, this.yL(NL - 1) + 0.3, 0.62);
    this.setL(this.lDmeter, F.upd > 0.3 && F.upd <= 1 && (v === 'q-step' || v === 'q-run') && s.ph === 'upd' && !sm, T_('q3.meterD'));
    // 层号 + 逻辑透镜读数（选中的回答位置：模型此刻最想说的词、正确答案的概率）
    const ansF = R.ans.includes(focus) ? focus : R.ans[1];
    // 一拍带过很多层时，层间距在屏幕上只有十几个像素：隔一层（窄屏隔两层）标一次，正在经过的那层一定标
    const stride = sm || this.E.h < 820 ? 3 : 2, curL = Math.floor(F.fwd);
    const thin = (l) => l === curL || l === this.NL - 1 || (this.NL - 1 - l) % stride === 0;
    const lensVis = (l) => {
      if (!haveSt) return false;
      if (v === 'q-step' && s.ph === 'fwd') return F.fwd >= l + 0.5 && Math.abs(F.fwd - (l + 1)) < (sm ? 3 : 6) && thin(l);
      if (v === 'q-op' && s.ph === 'fwd' && (s.sub === 'lo' || s.sub === 'hi')) return F.fwd >= l + 0.5 && Math.abs(F.fwd - (l + 1)) < (sm ? 4 : 8) && thin(l);
      if ((v === 'q-op' || v === 'q-mat') && l === this.xL && (s.ph === 'fwd' || s.ph === 'bwd')) return true;
      return false;
    };
    this.slabs.forEach((sl, l) => {
      const on = lensVis(l);
      sl.lbl.visible = on;
      if (!on) return;
      const top = R.lensTop(k, l + 1, ansF), pc = R.lensP(k, l + 1, ansF);
      const done = s.ph !== 'fwd' || F.fwd >= l + 1;
      const html = done ? `<span class="l">L${String(l).padStart(2, '0')}</span>${esc(tokPlain(top.s))}<span class="p">${esc(tokPlain(R.tgtStr(ansF)))} ${fmtPct(pc)}</span>` : `<span class="l">L${String(l).padStart(2, '0')}</span>`;
      if (sl.lbl.el._t !== html) { sl.lbl.el.innerHTML = html; sl.lbl.el._t = html; }
      sl.lbl.el.classList.toggle('on', l === Math.floor(F.fwd) || l === this.xL && st.depth >= 3);
    });
    // 回答 8 个位置的概率
    const rise = easeOut(seg(F.fwd, NL + 0.5, NL + 1));
    const showAns = rise > 0.5 && (far || (s.ph === 'loss') || s.sub === 'head') && !(sm && v === 'q-run') && v !== 'q-param' && v !== 'q-mat';
    R.ans.forEach((i, a) => {
      const o = this.lAns[a];
      let pv = R.pState(ks, i);
      if (F.next > 0) pv += (R.pState(kn, i) - pv) * easeInOut(F.next);
      // 看损失的时候 8 个都标，错开两排；其余远景里 8 个标签挤在一起，只标选中的那个（8 个的概率在左上角的监视器里）
      const lossView = s.ph === 'loss' && (v === 'q-step' || (v === 'q-op' && s.sub === 'mean')) && !sm;
      o.position.set(this.x(i), by + pv * 2.6 * rise + 0.08 + (lossView && a % 2 ? 0.34 : 0), -0.25);
      const show = showAns && (lossView || (far ? i === ansF : !sm || a % 2 === 0 || F.lossPos === i));
      this.setL(o, show, `${esc(tokPlain(R.tgtStr(i)))}<b>${fmtPct(pv)}</b>`);
      o.el.classList.toggle('on', F.lossPos === i);
    });
    // top-5 候选：逐个位置看损失时，挂在那根概率柱旁边（绿 = 正确答案）
    if (F.lossPos >= 0) {
      const top = R.topState(ks, F.lossPos), tg = R.tgt(F.lossPos);
      const rows = top.map((tc) => `<span class="${tc.id === tg ? 'ok' : ''}">${esc(tokPlain(tc.s))}<b>${fmtPct(tc.p)}</b></span>`).join('');
      const pv = R.pState(ks, F.lossPos);
      this.lCands.position.set(this.x(F.lossPos) + 0.3, by + Math.max(0.6, pv * 2.6 * 0.6), -0.25);
      this.setL(this.lCands, seg(this.p, 0.1, 0.6) > 0.5, `<i>${T_('q3.candHead')}</i>${rows}`);
    } else this.lCands.visible = false;
    // 面板标签
    // 面板标签：一层之内只写名字（正在算的那块再加上它的梯度 / 更新量），看一个张量时写全
    const opT = s.ph === 'fwd' ? OP_TENSORS[s.sub] : s.ph === 'bwd' ? OP_TENSORS_BWD[s.sub] : null;
    for (const [tn, o] of Object.entries(this.lPanel)) {
      const grp = this.panels[tn], vec = PANEL[tn].vec;
      const on = this.e > 0.7 && grp.visible && (v === 'q-op' ? (!vec || s.sub === tn) : this.selT === tn) && !(sm && v === 'q-op' && vec);
      if (!on) { o.visible = false; continue; }
      const pp = this.panelPos(tn);
      o.position.set(pp.x0, pp.y0 + pp.h + 0.02, PZ);
      const nm = T_(`q3.t.${tn}`), sh = TSHAPE[tn];
      const tsx = this.tensorText(k, this.xL, tn, have3, s);
      const active = opT?.includes(tn) || (s.ph === 'upd' && s.sub === 'adam');
      const html = v === 'q-op' ? `${nm}${active && tsx && !sm ? `<small>${tsx.replace(/^ · /, '')}</small>` : ''}` : `${nm}<small>${sh.length === 2 ? `${sh[0]} × ${sh[1]}` : `[${sh[0]}]`}${tsx}</small>`;
      this.setL(o, true, html);
      o.el.classList.toggle('on', this.selT === tn || (v === 'q-op' && !!opT?.includes(tn)));
    }
    for (const [key, o] of Object.entries(this.lEx)) {
      o.position.set(this.W / 2 - 0.05, this.yS(key), 0.4);
      o.visible = this.e > 0.7 && v === 'q-op' && !sm;
    }
    // 标题
    const title = this.titleFor(st, F);
    this.lTitle.position.set(0, (v === 'q-end' ? this.barsY + 3.2 : this.barsY + 3.0), 0);
    this.setL(this.lTitle, !!title && far && !sm, title);
    this.lDelta.visible = false;
  }

  tensorText(k, l, tn, have3, s) {
    if (!this.R.hasSt(k)) return '';
    const ts = this.R.tensor(k, l, tn);
    if (s.ph === 'upd' && have3 && ts.dw != null) return ` · ‖Δw‖ ${ts.dw.toExponential(2)}`;
    if (s.ph === 'bwd' || s.ph === 'upd') return ` · ‖∇‖ ${ts.g >= 1 ? ts.g.toFixed(2) : ts.g.toPrecision(3)}`;
    return '';
  }

  titleFor(st, F) {
    const s = st.step, R = this.R;
    if (s.ph === 'end') return T_('q3.titleEnd', { a: R.lossState(0).toFixed(2), b: R.lossState(R.K).toFixed(3) });
    if (st.view !== 'q-run') return '';
    const n = st.k + 1;
    if (F.next > 0.2) return T_('q3.titleAfter', { n, v: R.lossState(st.k + 1).toFixed(3) });
    if (F.upd > 0 || F.clip > 0) return T_('q3.titleUpd', { n });
    if (F.bwd < this.NL + 1) return T_('q3.titleBwd', { n });
    if (F.lossMean > 0) return T_('q3.titleLoss', { n, v: R.lossState(st.k).toFixed(3) });
    if (F.fwd > -1) return T_('q3.titleFwd', { n });
    return T_('q3.titleFeed', { n });
  }

  /* ================================================================ 选中 / 框 */

  updMarks(st, F, have3) {
    const s = st.step, place = (m, x, y, z, w, h, d) => { m.position.set(x, y, z); m.scale.set(w, h, d); m.visible = true; };
    const hv = this.hover;
    this.mHover.visible = false;
    if (hv?.type === 'slab' && st.depth >= 2) place(this.mHover, 0, this.yL(hv.L), 0, this.W - 0.2, 0.08, DEPTH + 0.1);
    else if (hv?.type === 'panel' && this.e > 0.5 && PANEL[hv.t]) { const pp = this.panelPos(hv.t); place(this.mHover, pp.x0 + pp.w / 2, pp.y0 + pp.h / 2, PZ, pp.w + 0.06, pp.h + 0.06, 0.04); }
    // 选中的权重（D5）：框出它在面板里的位置
    this.mSel.visible = false;
    this.lSel.visible = false;
    if (st.view === 'q-param' && s.t) {
      const pos = this.weightPos(s.t);
      if (pos) {
        place(this.mSel, pos.x, pos.y, pos.z + 0.02, pos.s, pos.s, 0.05);
        // γ 条很细，上面是它的名字：读数挂在下面
        const vec = isVec(s.t);
        this.lSel.position.set(pos.x, vec ? pos.y - pos.s : pos.y + pos.s, pos.z + 0.05);
        this.lSel.center.set(0.5, vec ? -0.25 : 1.25);
        const a = this.R.adamT(st.k, this.xL, s.t);
        if (a) {
          const html = s.ph === 'upd' && (s.mi === 'write' || s.mi === 'dw' || s.mi === 'bits') ? `w ${a.w0.toPrecision(5)} → <b>${a.w1.toPrecision(6)}</b>` : s.ph === 'bwd' ? `∂L/∂w = <b>${a.gRaw.toPrecision(4)}</b>` : `w = <b>${a.w0.toPrecision(5)}</b>`;
          this.setL(this.lSel, true, html);
        }
      }
    }
    // 选中的回答位置
    this.mFocus.visible = false;
  }

  // 跟踪的那个权重在机器上的位置
  weightPos(t) {
    const R = this.R, idx = R.trackedIndex(this.xL, t);
    if (!idx) return null;
    const pp = this.panelPos(t);
    if (t === 'embed') return { x: pp.x0 + ((idx[1] + 0.5) / 1024) * pp.w, y: this.embRowY(idx[0]), z: pp.z, s: 0.06 };
    if (isVec(t)) return { x: pp.x0 + ((idx[0] + 0.5) / TSHAPE[t][0]) * pp.w, y: pp.y0 + pp.h / 2, z: pp.z, s: 0.06 };
    // torch [输出, 输入]：x = 输出维，y = 输入维（输入 0 在底部）
    return { x: pp.x0 + (idx[0] + 0.5) * U, y: pp.y0 + (idx[1] + 0.5) * U, z: pp.z, s: 0.05 };
  }

  setSel(t) { this.selT = t; }

  /* ================================================================ 提示 */

  tip(info) {
    const R = this.R, st = this.st;
    if (!info || !st) return null;
    const k = st.k, m = R.D.meta, ks = Math.min(R.K, this.F?.state ?? k);
    const pos = (i) => T_('q3.tipPos', { i, s: esc(tokPlain(R.inStr(i))) });
    switch (info.type) {
      case 'tile': {
        const j = info.j, role = m.roles[j];
        const what = T_(`q3.role.${role}`);
        return `<span class="k">${T_('q3.tipTok', { j })} · ${what}</span>「${esc(tokPlain(R.tok(m.ids[j])))}」 <span class="v">#${m.ids[j]}</span>${j < R.N1 ? `<br>${T_('q3.tipPred', { s: esc(tokPlain(R.tgtStr(j))) })}${m.sftMask[j] ? T_('q3.tipCounted') : T_('q3.tipMasked')}` : `<br>${T_('q3.tipLast')}`}`;
      }
      case 'lab': return `<span class="k">labels[${info.j}]</span>${info.counted ? `「${esc(tokPlain(R.tok(m.ids[info.j])))}」 ${T_('q3.tipLabOn')}` : `−100 · ${T_('q3.tipLabOff')}`}`;
      case 'slab': {
        if (!R.hasSt(k)) return null;
        const g = R.layerGrad(k, info.L), d = R.has3(k) ? R.layerDw(k, info.L) : null;
        return `<span class="k">${T_('q3.layerN', { l: info.L })}</span>${T_('q3.tipLayerG', { g: g.toFixed(2) })}${d != null ? `<br>${T_('q3.tipLayerD', { d: d.toExponential(2) })}` : ''}<br><span style="color:var(--dim)">${T_('q3.tipLayerClick')}</span>`;
      }
      case 'lens': {
        if (!R.hasSt(k)) return null;
        const i = R.ans[info.a], top = R.lensTop(k, info.L + 1, i);
        return `<span class="k">${T_('q3.tipLens', { l: info.L })} · ${pos(i)}</span>${T_('q3.tipLensBody', { top: esc(tokPlain(top.s)), t: esc(tokPlain(R.tgtStr(i))), p: fmtPct(R.lensP(k, info.L + 1, i)) })}`;
      }
      case 'gmeter': return R.hasSt(k) ? `<span class="k">${T_('q3.layerN', { l: info.L })}</span>${T_('q3.tipLayerG', { g: R.layerGrad(k, info.L).toFixed(2) })}` : null;
      case 'bar': {
        const i = info.i, pv = R.pState(ks, i), ans = R.ans.includes(i);
        const top = R.topState(ks, i).slice(0, 3).map((c) => `${esc(tokPlain(c.s))} ${fmtPct(c.p)}`).join('　');
        return `<span class="k">${pos(i)}${ans ? '' : ` · ${T_('q3.tipMasked2')}`}</span>${T_('q3.tipBar', { t: esc(tokPlain(R.tgtStr(i))), p: fmtPct(pv) })}<br><span style="color:var(--dim)">${T_('q3.tipTop3')}</span> ${top}`;
      }
      case 'nll': { const i = info.i; return `<span class="k">${pos(i)}</span>−ln p = <span class="a">${R.nllState(ks, i).toFixed(4)}</span>（p = ${fmtPct(R.pState(ks, i))}）`; }
      case 'cand': { const c = R.topState(ks, this.F.lossPos)[info.c]; return c ? `<span class="k">${T_('q3.tipCand', { n: info.c + 1 })}</span>「${esc(tokPlain(c.s))}」 <span class="v">${fmtPct(c.p)}</span>${c.id === R.tgt(this.F.lossPos) ? T_('q3.tipIsAns') : ''}` : null; }
      case 'gauge': return `<span class="k">${T_('q3.tipGauge', { n: ks })}</span>${T_('q3.tipGaugeBody', { a: R.lossState(ks).toFixed(4), b: R.lossPreState(ks).toFixed(4), n: R.ans.length, m: R.N1 })}`;
      case 'panel': {
        const tn = info.t, l = tn === 'norm' ? -1 : this.xL;
        if (!R.hasSt(k)) return `<span class="k">${T_(`q3.t.${tn}`)}</span>`;
        const ts = R.tensor(k, l, tn);
        return `<span class="k">${tn === 'norm' ? '' : T_('q3.layerN', { l: this.xL }) + ' · '}${T_(`q3.t.${tn}`)}</span>${T_('q3.tipTensor', { g: ts.g.toPrecision(4), dw: ts.dw != null ? ts.dw.toExponential(2) : '—' })}<br><span style="color:var(--dim)">${T_('q3.tipTensorClick')}</span>`;
      }
      case 'emb': return `<span class="k">${T_('q3.emb')} · 151936 × 1024</span>${T_('q3.tipEmb')}<br><span style="color:var(--dim)">${T_('q3.tipTensorClick')}</span>`;
      case 'lm': return `<span class="k">lm_head</span>${T_('q3.lmSub')}`;
      case 'beam': return `<span class="k">${T_('q3.tipBeam', { l: this.xL })}</span>${pos(this.focus)} → ${pos(info.j)}：<span class="v">${fmtPct(info.w)}</span>`;
      case 'delta': {
        const i = info.i;
        if (!R.has3(k)) return null;
        return `<span class="k">${pos(i)}</span>${info.kind === 'attn' ? T_('q3.tipDA', { v: R.attnOut(k, this.xL, i).toFixed(2) }) : T_('q3.tipDM', { v: R.mlpOut(k, this.xL, i).toFixed(2) })}<br>${T_('q3.tipMid', { v: R.midGrad(k, this.xL, i).toPrecision(3) })}`;
      }
    }
    return null;
  }

  /* ================================================================ 镜头 */

  fitBox(b, dir, margin = 1.06, minD = 2) {
    const E = this.E;
    const key = `${b.map((v) => v.toFixed(2)).join(',')}|${dir.x},${dir.y},${dir.z}|${margin}|${E.w}x${E.h}|${E.insetR | 0}|${E.padB | 0}|${E.padT | 0}`;
    if (this.fitKey === key) return { pos: this.fitRes.pos.clone(), look: this.fitRes.look.clone() };
    const cam = this.fitCam || (this.fitCam = new THREE.PerspectiveCamera());
    cam.copy(E.camera);
    cam.updateProjectionMatrix();
    const [x0, x1, y0, y1, z0 = -0.9, z1 = 0.9] = b;
    const pts = [];
    for (const x of [x0, x1]) for (const y of [y0, y1]) for (const z of [z0, z1]) pts.push(new THREE.Vector3(x, y, z));
    const w = E.w || 1, h = E.h || 1;
    // 顶栏 + 章节按钮盖住画布最上面一条（引擎不知道），取景时也让出来
    const top = Math.max(E.padT, this.small ? 104 : 118);
    const rx0 = -1, rx1 = 1 - (2 * (E.insetR || 0)) / w, ry0 = -1 + (2 * E.padB) / h, ry1 = 1 - (2 * top) / h;
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
    const s = st.step, v = st.view, F = this.F || this.flow(st, st.p), NL = this.NL, W = this.W;
    const V3 = (x, y, z) => new THREE.Vector3(x, y, z);
    const frame = (r, dir, margin = 1.05, minD = 2) => this.fitBox(r, dir, margin, minD);
    const WIDE = V3(0.42, 0.3, 1), MID = V3(0.32, 0.22, 1), NEAR = V3(0.18, 0.12, 1), TOP = V3(0.22, 0.42, 1), TRAY = V3(0.1, 0.4, 1);
    const xa = -W / 2 - (this.small ? 0.3 : 2.0), xb = W / 2 + 1.6;
    const whole = [xa, xb, -0.3, this.barsY + 3.0, -1.4, 1.2];
    if (v === 'q-run' || v === 'q-end') return frame(whole, WIDE, 1.02);
    const trayBox = [-W / 2 - (this.small ? 0.2 : 1.9), W / 2, -0.2, 0.7, -0.4, 1.15];
    const topBox = [-W / 2, W / 2 + 1.4, this.yTop - 0.2, this.barsY + 2.9, -1.0, 0.8];
    const follow = (f) => { const y = this.yAt(Math.max(-1, Math.min(NL + 1, f))); return frame([xa, xb, y - 2.6, y + 2.6, -1.2, 1.0], MID, 1.02); };
    if (v === 'q-step') {
      if (s.ph === 'batch') return frame(trayBox, TRAY, 1.04);
      if (s.ph === 'loss') return frame(topBox, TOP, 1.04);
      if (s.ph === 'upd') return frame(whole, WIDE, 1.02);
      return follow(s.ph === 'fwd' ? F.fwd : F.bwd);
    }
    const exBox = () => { const b0 = Y0 + this.xL * GAP; return [-W / 2 + 0.2, W / 2 + 0.2, b0 - 0.3, b0 + (SUB.add2 + 0.35) * Math.max(0.3, this.e), -1.4, 0.8]; };
    const panelBox = (t, pad = 0.35) => { const pp = this.panelPos(t); return [pp.x0 - pad, pp.x0 + pp.w + pad, pp.y0 - pad, pp.y0 + pp.h + pad, PZ - 0.1, PZ + 0.3]; };
    if (v === 'q-op') {
      if (s.ph === 'batch') return frame(trayBox, TRAY, 1.04);
      if (s.ph === 'loss') {
        if (s.sub === 'pos') { const x = this.x(s.i); return frame([x - 2.4, x + 2.4, this.barsY - 0.3, this.barsY + 2.9, -1.1, 0.6], TOP, 1.04); }
        return frame(topBox, TOP, 1.04);
      }
      if (s.ph === 'upd') return s.sub === 'adam' && this.e > 0.1 ? frame(exBox(), MID, 1.04) : frame(whole, WIDE, 1.02);
      if (s.sub === 'emb') return frame([-W / 2, W / 2, -0.3, 1.8, -1.4, 0.9], MID, 1.04);
      if (s.sub === 'head') return frame([-W / 2, W / 2 + 1.4, this.yTop - 0.4, this.barsY + 2.8, -1.4, 0.8], MID, 1.04);
      if (s.sub === 'lo' || s.sub === 'hi') return follow(s.ph === 'fwd' ? F.fwd : F.bwd);
      return frame(exBox(), MID, 1.04);
    }
    if (v === 'q-mat' || v === 'q-param') {
      const t = s.t;
      if (v === 'q-param') {
        const pos = this.weightPos(t);
        if (pos) { const r = t === 'embed' || isVec(t) ? 1.1 : 1.05; return frame([pos.x - r, pos.x + r, pos.y - r * 0.7, pos.y + r * 0.7, PZ - 0.1, PZ + 0.3], NEAR, 1.04, 0.6); }
      }
      if (t === 'embed') return frame([EMB.x0 - 0.3, EMB.x1 + 0.3, EMB.y0 - 0.3, EMB.y1 + 0.5, PZ - 0.1, 0.4], NEAR, 1.04);
      if (t === 'norm') return frame([-3.1, 0.6, this.yTop - 0.3, this.yTop + 0.9, PZ - 0.1, 0.6], NEAR, 1.04);
      if (PANEL[t]?.vec) return frame(panelBox(t, 0.8), NEAR, 1.04);
      return frame(panelBox(t), NEAR, 1.04);
    }
    return frame(whole, WIDE, 1.02);
  }
}
