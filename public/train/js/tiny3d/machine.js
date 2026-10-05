// 唐宋诗小模型（Qwen3 同构，6 层 · 256 维 · 4 查询头 / 2 KV 头 · 664 万参数）的 3D 机器。
// 664 万个参数没法一个个画：每块权重是一块按真实形状等比例的面板；有导出真实局部的三块（嵌入、第 2 层 W_q、第 4 层 W_down）
// 角上贴 48 × 48 的真实数值，其余面板只显示统计量（反向时按这个张量真实的梯度范数发光，更新时按到下一个检查点的 RMS(ΔW) 发光）。
// 一步训练在机器上（全部来自记录）：
//   进料：批次第 0 行的 64 个字（真实的字）飞进托盘；其余 63 行没有记录，画成托盘下面一叠空白的薄片；
//   前向：光脉冲沿 64 根光纤一层层往上，每层旁边的逻辑透镜屏写出这一层此刻最想写的字；顶上的柱子升起 p（正确答案的概率）；
//   损失：每个位置 −ln p，整批平均；
//   反向：玫红脉冲倒流，每根光纤上的环大小 = 这里的 ‖∂L/∂h‖，面板按 ‖∇W‖ 发光，有局部的面板换成梯度热力；
//   更新：局部按真实 ΔW 变色（到下一个检查点），其余面板按 RMS(ΔW) 发紫光。
// 一切都由 update(st) 根据当前步骤算出来，暂停、单步、回退都能正确显示。
import { THREE, label, easeOut, easeInOut, seg } from '../../../js/stage/engine.js';
import * as LY from './layout.js';
import { divInto, gradInto, seqInto, LIN } from '../glass3d/palette.js';
import { FWD, BWD, PHASES, opIndex, ADAM_SUBS, FEAT_CROP } from './timeline.js';
import { tensorName, KIND_LABEL, kindShape, kindCount, CROPS, cropOf } from './data.js';
import { fmtP, sciSup } from '../draw.js';
import { esc } from '../../../js/ui.js';
import { t, isEn } from './lang.js';

const { U, NP, NC, NR, FS, fiberX, fiberZ, BUNDLE, BCX, BCZ, yBand, yBound, SUB, PANEL, LAYER_KINDS, HEAD, headX, EMB, embZ, EMB_Z1, NF, YTOP, YT, YP, PH, TRAY, TGT_DZ, LENS, LENS_W, LENS_H, LOSSB, lossX, GAUGE, PRINTER, CORPUS, PCA, LAND, MACHINE, ALL } = LY;
const NLAY = LY.NL;
const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const lerp = (a, b, f) => a + (b - a) * f;
const FONT = '"PingFang SC","Microsoft YaHei","Noto Sans SC","Noto Sans CJK SC",sans-serif';
const C48 = 48, MAG = { cell: 0.05, face: 0.042 };

// 发散色（sRGB 字节）：负 = 蓝，正 = 琥珀，零 = 暗；梯度：正 = 玫红，负 = 紫；ΔW：正 = 青，负 = 紫
function divBytes(t, pos = [255, 182, 92], neg = [107, 155, 255], zero = [12, 20, 36]) {
  t = t > 1 ? 1 : t < -1 ? -1 : t;
  const e = Math.pow(Math.abs(t), 0.75), c = t >= 0 ? pos : neg;
  return [zero[0] + (c[0] - zero[0]) * e, zero[1] + (c[1] - zero[1]) * e, zero[2] + (c[2] - zero[2]) * e];
}
const GPOS = [255, 107, 147], GNEG = [150, 128, 255], DPOS = [94, 240, 212], DNEG = [179, 157, 255];

function disposeTree(o) {
  o.traverse((c) => {
    c.geometry?.dispose?.();
    if (c.material && !c.material.userData?.shared) (Array.isArray(c.material) ? c.material : [c.material]).forEach((m) => { m.map?.dispose?.(); m.dispose(); });
    if (c.el) c.el.remove();
  });
}

function cubeMaterial(k = 0.35) {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0.08 });
  m.onBeforeCompile = (sh) => {
    sh.fragmentShader = sh.fragmentShader.replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>\n#ifdef USE_INSTANCING_COLOR\n\ttotalEmissiveRadiance += vColor * ${k.toFixed(2)};\n#endif`);
  };
  return m;
}

// 面板的网格贴图（规则的格线，每 32 维一格；只是框架，不是数据）
function gridTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  g.strokeStyle = 'rgba(140,180,235,0.22)';
  g.lineWidth = 2;
  g.strokeRect(0, 0, 64, 64);
  g.strokeStyle = 'rgba(140,180,235,0.07)';
  g.lineWidth = 1;
  g.beginPath(); g.moveTo(32, 0); g.lineTo(32, 64); g.moveTo(0, 32); g.lineTo(64, 32); g.stroke();
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

// 字表：cols × rows 个格子，每格一个字（白字透明底，材质颜色再染色）
function atlas(cols, rows, cell = 64) {
  const cv = document.createElement('canvas');
  cv.width = cols * cell; cv.height = rows * cell;
  const g = cv.getContext('2d');
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return {
    cv, g, tex, cols, rows, cell,
    draw(list) {
      g.clearRect(0, 0, cv.width, cv.height);
      g.font = `600 ${Math.round(cell * 0.7)}px ${FONT}`;
      g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillStyle = '#fff';
      list.forEach((s, i) => { if (s) g.fillText(s, (i % cols) * cell + cell / 2, Math.floor(i / cols) * cell + cell * 0.54); });
      tex.needsUpdate = true;
    },
  };
}

// 给 InstancedMesh 的平面贴字表里的第 n 格
function atlasFaceMaterial(at, color = 0xffffff) {
  const m = new THREE.MeshBasicMaterial({ map: at.tex, transparent: true, depthWrite: false, color });
  m.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader.replace('#include <uv_pars_vertex>', '#include <uv_pars_vertex>\nattribute float aCell;')
      .replace('#include <uv_vertex>', `#include <uv_vertex>\n\tfloat cx = mod(aCell, ${at.cols}.0), cy = floor(aCell / ${at.cols}.0);\n\tvMapUv = vec2((cx + vMapUv.x) / ${at.cols}.0, 1.0 - (cy + 1.0 - vMapUv.y) / ${at.rows}.0);`);
  };
  return m;
}

export class TinyMachine {
  constructor(E, X, app) {
    this.E = E;
    this.X = X;
    this.app = app;
    this.scene = E.scene;
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.hover = null;
    this.pk = '';
    this.pt0 = 0;
    this.baked = { st: -1, ck: -1, dw: -1, row: -1 };
    this.tmpM = new THREE.Matrix4();
    this.tmpV = new THREE.Vector3();
    this.tmpQ = new THREE.Quaternion();
    this.tmpS = new THREE.Vector3();
    this.gridTex = gridTexture();
    this.buildEnv();
    this.buildBundle();
    this.buildTray();
    this.buildEmb();
    this.buildLayers();
    this.buildLens();
    this.buildHead();
    this.buildPrinter();
    this.buildExhibits();
    this.buildMagnifier();
    this.buildMarks();
    this.buildLabels();
    const sp = new THREE.Sphere(new THREE.Vector3(5, 12, -5), 40);
    for (const m of [this.fibers, this.tiles, this.tgts, this.pillars, this.lossBars, this.cubes, this.tops]) if (m) m.boundingSphere = sp.clone();
  }

  /* ================================================================ 搭机器 */

  buildEnv() {
    const grid = new THREE.GridHelper(120, 120, 0x1c3350, 0x10203a);
    grid.position.y = -0.34;
    grid.material.transparent = true;
    grid.material.opacity = 0.28;
    this.scene.add(grid);
    const n = 700, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { pos[i * 3] = (Math.random() - 0.5) * 80 + 4; pos[i * 3 + 1] = Math.random() * 34 - 2; pos[i * 3 + 2] = (Math.random() - 0.5) * 50 - 8; }
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.dust = new THREE.Points(pg, new THREE.PointsMaterial({ color: 0x7fd8e8, size: 0.05, transparent: true, opacity: 0.26, depthWrite: false }));
    this.scene.add(this.dust);
    // 每层底下一块很淡的玻璃地板（脉冲经过时边框闪一下）
    this.floors = [];
    for (let l = 0; l <= NLAY; l++) {
      const y = l === NLAY ? YTOP + 0.05 : yBand(l) - 0.02, x0 = LY.LENS.x0 - 0.2, x1 = l === NLAY ? 4.9 : 7.95, w = x1 - x0, d = 1.5;
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, 0.02, d), new THREE.MeshStandardMaterial({ color: 0x22346a, transparent: true, opacity: 0.045, depthWrite: false, roughness: 0.4 }));
      m.position.set(x0 + w / 2, y, -0.2);
      m.raycast = () => {};
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.1 }));
      e.position.copy(m.position);
      e.raycast = () => {};
      this.root.add(m, e);
      this.floors.push({ y, e });
    }
  }

  // 残差流：64 根光纤（16 × 4）+ 玻璃外壳 + 每层 4 个环（RMSNorm、⊕）+ 最后的 RMSNorm
  buildBundle() {
    const h = YT - 0.25;
    const cg = new THREE.CylinderGeometry(0.011, 0.011, 1, 6, 1, true);
    cg.translate(0, 0.5, 0);
    this.fibers = new THREE.InstancedMesh(cg, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7, depthWrite: false }), NP);
    this.fibers.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(NP * 3), 3);
    const M = this.tmpM;
    for (let i = 0; i < NP; i++) { M.makeScale(1, h, 1).setPosition(fiberX(i), 0.25, fiberZ(i)); this.fibers.setMatrixAt(i, M); }
    this.fibers.raycast = () => {};
    this.root.add(this.fibers);
    const W = BUNDLE.x1 - BUNDLE.x0, D = BUNDLE.z1 - BUNDLE.z0;
    const shell = new THREE.Mesh(new THREE.BoxGeometry(W, h, D), new THREE.MeshStandardMaterial({ color: 0x1a2c50, emissive: 0x5ef0d4, emissiveIntensity: 0.02, transparent: true, opacity: 0.05, depthWrite: false, roughness: 0.3 }));
    shell.position.set(BCX, 0.25 + h / 2, BCZ);
    shell.raycast = () => {};
    const edge = new THREE.LineSegments(new THREE.EdgesGeometry(shell.geometry), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.13 }));
    edge.position.copy(shell.position);
    edge.raycast = () => {};
    this.root.add(shell, edge);
    const ring = (y, color) => {
      const m = new THREE.Mesh(new THREE.TorusGeometry(1, 0.022, 6, 48), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.35, depthWrite: false }));
      m.rotation.x = Math.PI / 2;
      m.scale.set(W / 2 + 0.12, D / 2 + 0.12, 1);
      m.position.set(BCX, y, BCZ);
      m.raycast = () => {};
      this.root.add(m);
      return m;
    };
    this.rings = [];
    for (let l = 0; l < NLAY; l++) {
      const yb = yBand(l);
      this.rings.push({ norm1: ring(yb + SUB.ring1, 0x6b9bff), add1: ring(yb + SUB.add1, 0xe9eff9), norm2: ring(yb + SUB.ring2, 0x6b9bff), add2: ring(yBound(l + 1), 0xe9eff9) });
    }
    this.ringF = ring(YTOP + 0.18, 0x6b9bff);
    // 脉冲：前向（青）往上、反向（玫红）往下，每根光纤一个小环
    const pg = new THREE.TorusGeometry(0.04, 0.009, 5, 14);
    const mk = (color) => { const m = new THREE.InstancedMesh(pg, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85, depthWrite: false }), NP); m.frustumCulled = false; m.raycast = () => {}; this.root.add(m); return m; };
    this.pulseF = mk(0x9ffcff);
    this.pulseB = mk(0xff7aa0);
    const slab = (color) => { const m = new THREE.Mesh(new THREE.BoxGeometry(W + 0.12, 0.035, D + 0.12), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false })); m.position.set(BCX, 0, BCZ); m.raycast = () => {}; this.root.add(m); return m; };
    this.slabF = slab(0x5ef0d4);
    this.slabB = slab(0xff6b93);
    // 分支管：残差 → 这一半层的面板（RMSNorm 之后），这一半层的输出 → ⊕ 加回残差
    this.pipes = [];
    const tube = (pts, color) => {
      const m = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 20, 0.012, 5, false), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.14, depthWrite: false }));
      m.raycast = () => {};
      this.root.add(m);
      return m;
    };
    const V = (x, y, z = 0) => new THREE.Vector3(x, y, z);
    for (let l = 0; l < NLAY; l++) {
      const yb = yBand(l), x1 = BUNDLE.x1;
      this.pipes.push({
        a: tube([V(x1, yb + SUB.ring1, BCZ), V(x1 + 0.3, yb + SUB.ring1 + 0.05, 0.02), V(0.04, yb + SUB.attn0 + 0.02, 0.02)], 0x6b9bff),
        aOut: tube([V(PANEL.o.x0 + 0.5, yb + SUB.attn1 + 0.12, 0.02), V(PANEL.o.x0 + 0.5, yb + SUB.add1, -0.45), V(0.3, yb + SUB.add1, -0.45), V(x1, yb + SUB.add1, BCZ)], 0xe9eff9),
        f: tube([V(x1, yb + SUB.ring2, BCZ), V(x1 + 0.3, yb + SUB.ring2 + 0.05, 0.02), V(0.04, yb + SUB.ffn0 + 0.02, 0.02)], 0x6b9bff),
        fOut: tube([V(PANEL.down.x0 + 0.5, yb + SUB.attn0 + 3.06, 0.02), V(PANEL.down.x0 + 0.5, yBound(l + 1), -0.55), V(0.3, yBound(l + 1), -0.55), V(x1, yBound(l + 1), BCZ)], 0xe9eff9),
        act: tube([V(PANEL.up.x0 + 3.0, yb + SUB.ffn1 + 0.06, 0.02), V(PANEL.down.x0 - 0.15, yb + SUB.ffn1 + 0.06, 0.02), V(PANEL.down.x0 - 0.15, yb + SUB.attn0 + 2.9, 0.02)], 0x8fb8ff),
      });
    }
  }

  // 托盘：64 个字块（第 0 行），答案字块（错开一位），其余 63 行的空白薄片，左边的语料（示意）
  buildTray() {
    const g = (this.trayG = new THREE.Group());
    this.root.add(g);
    this.rowAtlas = atlas(16, 5);
    const T = TRAY.tile;
    const body = new THREE.BoxGeometry(T, T * 0.6, T);
    const face = new THREE.PlaneGeometry(T * 0.92, T * 0.92);
    face.rotateX(-Math.PI / 2);
    const mk = (color, bodyCol, k) => {
      const b = new THREE.InstancedMesh(body, cubeMaterial(k), NP);
      b.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(NP * 3), 3);
      const c = new THREE.Color(bodyCol);
      for (let i = 0; i < NP; i++) b.setColorAt(i, c);
      const fg = face.clone();
      const cell = new Float32Array(NP);
      fg.setAttribute('aCell', new THREE.InstancedBufferAttribute(cell, 1));
      const f = new THREE.InstancedMesh(fg, atlasFaceMaterial(this.rowAtlas, color), NP);
      f.raycast = () => {};
      for (const m of [b, f]) { m.frustumCulled = false; m.instanceMatrix.setUsage(THREE.DynamicDrawUsage); g.add(m); }
      return { b, f, cell };
    };
    const tl = mk(0xffd9a6, 0x3a2a14, 0.25);
    this.tiles = tl.b; this.tileFaces = tl.f;
    for (let i = 0; i < NP; i++) tl.cell[i] = i;
    const tg = mk(0xbff8ec, 0x0f3a33, 0.3);
    this.tgts = tg.b; this.tgtFaces = tg.f;
    for (let i = 0; i < NP; i++) tg.cell[i] = i + 1;
    this.tiles.userData.pick = (hit) => ({ type: 'tile', i: hit.instanceId });
    this.tgts.userData.pick = (hit) => ({ type: 'tgt', i: hit.instanceId });
    this.E.pickables.push(this.tiles, this.tgts);
    // 托盘底板 + 下面一叠薄片（其余 63 行，没有记录）
    const W = BUNDLE.x1 - BUNDLE.x0 + 0.16, D = BUNDLE.z1 - BUNDLE.z0 + 0.16;
    const plate = new THREE.Mesh(new THREE.BoxGeometry(W, 0.03, D), new THREE.MeshStandardMaterial({ color: 0x0c1629, metalness: 0.6, roughness: 0.35 }));
    plate.position.set(BCX, TRAY.y - 0.05, BCZ);
    const sc = document.createElement('canvas');
    sc.width = 8; sc.height = 128;
    const sg = sc.getContext('2d');
    for (let i = 0; i < 64; i++) { sg.fillStyle = i % 2 ? 'rgba(60,90,140,0.55)' : 'rgba(10,18,34,0.3)'; sg.fillRect(0, i * 2, 8, 2); }
    const st = new THREE.CanvasTexture(sc);
    st.colorSpace = THREE.SRGBColorSpace;
    const stack = new THREE.Mesh(new THREE.BoxGeometry(W - 0.04, 0.36, D - 0.04), new THREE.MeshBasicMaterial({ map: st, transparent: true, opacity: 0.55, depthWrite: false }));
    stack.position.set(BCX, TRAY.y - 0.25, BCZ);
    stack.userData.pick = { type: 'stack' };
    this.E.pickables.push(stack);
    const se = new THREE.LineSegments(new THREE.EdgesGeometry(stack.geometry), new THREE.LineBasicMaterial({ color: 0x2c6b8a, transparent: true, opacity: 0.5 }));
    se.position.copy(stack.position);
    g.add(plate, stack, se);
    // 语料（示意）：一条从左边伸过来的长带子，上面是淡淡的格子
    const cw = CORPUS.x1 - CORPUS.x0;
    const cc = document.createElement('canvas');
    cc.width = 1024; cc.height = 32;
    const cg = cc.getContext('2d');
    for (let x = 0; x < 1024; x += 16) { cg.fillStyle = `rgba(150,180,230,${0.08 + 0.1 * Math.abs(Math.sin(x * 0.37))})`; cg.fillRect(x + 2, 6, 12, 20); }
    const ct = new THREE.CanvasTexture(cc);
    ct.colorSpace = THREE.SRGBColorSpace;
    ct.wrapS = THREE.RepeatWrapping;
    this.corpusTex = ct;
    const rib = new THREE.Mesh(new THREE.PlaneGeometry(cw, 0.28), new THREE.MeshBasicMaterial({ map: ct, transparent: true, opacity: 0.8, depthWrite: false, side: THREE.DoubleSide }));
    rib.rotation.x = -Math.PI / 2;
    rib.position.set(CORPUS.x0 + cw / 2, CORPUS.y, CORPUS.z);
    rib.userData.pick = { type: 'corpus' };
    this.E.pickables.push(rib);
    const re = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(cw, 0.28)), new THREE.LineBasicMaterial({ color: 0x2c4a70, transparent: true, opacity: 0.6 }));
    re.rotation.x = -Math.PI / 2;
    re.position.copy(rib.position);
    g.add(rib, re);
  }

  // 嵌入表 E：7478 行 × 256 维，平躺在地上往后伸；角上贴真实的 48 × 48 局部；查表的 64 束光
  buildEmb() {
    const len = EMB.z0 - EMB_Z1;
    this.embP = this.makePanel({ l: -1, kind: 'E', w: 1, h: len, flat: true, x0: EMB.x0, y: EMB.y, z0: EMB.z0 });
    this.ghostP = this.makePanel({ l: -1, kind: 'ET', w: 1, h: len, flat: true, x0: EMB.x0, y: YT, z0: EMB.z0, ghost: true });
    this.nfP = this.makePanel({ l: -1, kind: 'nf', w: NF.w, h: NF.h, x0: NF.x0, y0: NF.y });
    // 查表：字块 → E 里那一行（64 条曲线，每个检查点重画）
    this.beamGeo = new THREE.BufferGeometry();
    this.beamGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NP * 16 * 2 * 3), 3));
    this.beams = new THREE.LineSegments(this.beamGeo, new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.beams.frustumCulled = false;
    this.beams.raycast = () => {};
    this.root.add(this.beams);
    // 输出：残差 → Eᵀ（64 条竖直的光，前向到输出头时亮）
    this.headGeo = new THREE.BufferGeometry();
    this.headGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NP * 16 * 2 * 3), 3));
    this.headBeams = new THREE.LineSegments(this.headGeo, new THREE.LineBasicMaterial({ color: 0x9ffcff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.headBeams.frustumCulled = false;
    this.headBeams.raycast = () => {};
    this.root.add(this.headBeams);
  }

  // 一块面板：底色（发光用）+ 规则格线 + 边框；可以带一块真实局部（DataTexture）
  makePanel(o) {
    const g = new THREE.Group();
    const geo = new THREE.PlaneGeometry(o.w, o.h);
    const uv = geo.attributes.uv;
    for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * Math.max(1, o.w / (32 * U)), uv.getY(i) * Math.max(1, o.h / (32 * U)));
    const back = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x0c1830, transparent: true, opacity: o.ghost ? 0.18 : 0.82, depthWrite: false, side: THREE.DoubleSide }));
    const grid = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: this.gridTex, transparent: true, opacity: o.ghost ? 0.35 : 0.9, depthWrite: false, side: THREE.DoubleSide }));
    grid.raycast = () => {};
    grid.position.z = 0.001;
    const edge = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: o.ghost ? 0.25 : 0.45 }));
    edge.raycast = () => {};
    g.add(back, grid, edge);
    if (o.flat) {
      g.rotation.x = -Math.PI / 2;
      g.position.set(o.x0 + o.w / 2, o.y, o.z0 - o.h / 2);
    } else g.position.set(o.x0 + o.w / 2, o.y0 + o.h / 2, 0);
    this.root.add(g);
    const P = { ...o, g, back, grid, edge, crop: o.kind === 'ET' ? -1 : cropOf(o.l, o.kind), glowG: 0, glowU: 0 };
    back.userData.pick = { type: 'panel', P };
    this.E.pickables.push(back);
    // 有真实局部的面板：左上角（嵌入：最前、最左）贴 48 × 48
    if (P.crop >= 0 && !o.ghost) {
      const data = new Uint8Array(C48 * C48 * 4);
      const tex = new THREE.DataTexture(data, C48, C48, THREE.RGBAFormat);
      tex.magFilter = THREE.NearestFilter;
      tex.minFilter = THREE.LinearFilter;
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.needsUpdate = true;
      const s = C48 * U;
      const pm = new THREE.Mesh(new THREE.PlaneGeometry(s, s), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
      // 平躺的嵌入表：局部的第 0 行（1 号字，0 号是分隔符）在最前面 = 平面的下边
      pm.position.set(-o.w / 2 + s / 2, o.flat ? -o.h / 2 + U + s / 2 : o.h / 2 - s / 2, 0.003);
      pm.userData.pick = (hit) => ({ type: 'patch', P, uv: hit.uv });
      this.E.pickables.push(pm);
      const pe = new THREE.LineSegments(new THREE.EdgesGeometry(pm.geometry), new THREE.LineBasicMaterial({ color: 0xffd18a, transparent: true, opacity: 0.8 }));
      pe.position.copy(pm.position);
      pe.raycast = () => {};
      g.add(pm, pe);
      Object.assign(P, { patch: pm, patchEdge: pe, patchData: data, patchTex: tex, patchKey: '' });
    }
    (this.panels ||= []).push(P);
    return P;
  }

  buildLayers() {
    this.layerP = [];
    this.heads = [];
    this.outs = [];
    for (let l = 0; l < NLAY; l++) {
      const row = {};
      const yb = yBand(l);
      for (const kind of LAYER_KINDS) {
        const p = PANEL[kind];
        row[kind] = this.makePanel({ l, kind, w: p.w, h: p.h, x0: p.x0, y0: yb + p.y0 });
      }
      this.layerP.push(row);
      // 4 个查询头的注意力图案（25 × 25 的下三角）
      const hs = [];
      for (let h = 0; h < 4; h++) {
        const data = new Uint8Array(25 * 25 * 4);
        const tex = new THREE.DataTexture(data, 25, 25, THREE.RGBAFormat);
        tex.magFilter = THREE.NearestFilter;
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.needsUpdate = true;
        const m = new THREE.Mesh(new THREE.PlaneGeometry(HEAD.size, HEAD.size), new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.9, depthWrite: false }));
        m.position.set(headX(h) + HEAD.size / 2, yb + SUB.heads0 + HEAD.size / 2, 0.01);
        m.userData.pick = (hit) => ({ type: 'head', l, h, uv: hit.uv });
        this.E.pickables.push(m);
        const e = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color: h < 2 ? 0x5ef0d4 : 0xb39dff, transparent: true, opacity: 0.45 }));
        e.position.copy(m.position);
        e.raycast = () => {};
        this.root.add(m, e);
        hs.push({ m, e, data, tex });
      }
      this.heads.push(hs);
      // 输出条：q / k / v、o、gate / up、down（填满表示“算出来了”，不代表具体数值）
      const bar = (x0, w, y) => {
        const m = new THREE.Mesh(new THREE.BoxGeometry(1, 0.03, 0.03), new THREE.MeshBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.0, depthWrite: false }));
        m.position.set(x0, y, 0.02);
        m.userData = { x0, w };
        m.raycast = () => {};
        this.root.add(m);
        return m;
      };
      this.outs.push({
        q: bar(PANEL.q.x0, 1, yb + SUB.qkvOut), k: bar(PANEL.k.x0, 0.5, yb + SUB.qkvOut), v: bar(PANEL.v.x0, 0.5, yb + SUB.qkvOut),
        o: bar(PANEL.o.x0, 1, yb + SUB.qkvOut), gate: bar(PANEL.gate.x0, 3, yb + SUB.ffn1 + 0.06), up: bar(PANEL.up.x0, 3, yb + SUB.ffn1 + 0.06), down: bar(PANEL.down.x0, 1, yb + SUB.attn0 + 3.06),
      });
    }
    // 扫描线：前向时“这一列正在和输入做点积”（同时最多三块面板在算）
    this.scans = [];
    for (let i = 0; i < 3; i++) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0xbff8ff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
      m.raycast = () => {};
      this.root.add(m);
      this.scans.push(m);
    }
  }

  // 逻辑透镜屏：每个层边界一块，16 × 4 个字（这一层的残差直接接最后的 RMSNorm + Eᵀ 最想写的字；绿 = 正好是答案）
  buildLens() {
    this.lens = [];
    for (let b = 0; b <= NLAY; b++) {
      const cv = document.createElement('canvas');
      cv.width = 1024; cv.height = 320;
      const tex = new THREE.CanvasTexture(cv);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = 4;
      const y = yBound(b);
      const m = new THREE.Mesh(new THREE.PlaneGeometry(LENS_W, LENS_H), new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.25, depthWrite: false }));
      m.position.set(LENS.x0 + LENS_W / 2, y, 0.05);
      m.userData.pick = (hit) => ({ type: 'lens', b, uv: hit.uv });
      this.E.pickables.push(m);
      const back = new THREE.Mesh(new THREE.PlaneGeometry(LENS_W + 0.08, LENS_H + 0.08), new THREE.MeshBasicMaterial({ color: 0x08101f, transparent: true, opacity: 0.8, depthWrite: false }));
      back.position.set(m.position.x, y, 0.03);
      back.raycast = () => {};
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(back.geometry), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.25 }));
      e.position.copy(back.position);
      e.raycast = () => {};
      // 一根细线把屏连到光纤束的这个边界
      const lg = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(LENS.x0 + LENS_W + 0.04, y, 0.04), new THREE.Vector3(BUNDLE.x0, y, BCZ)]);
      const ln = new THREE.Line(lg, new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.2 }));
      ln.raycast = () => {};
      this.root.add(back, e, m, ln);
      this.lens.push({ cv, g: cv.getContext('2d'), tex, m, back, e, ln, key: '' });
    }
  }

  // 输出：64 根预测柱（高 = 正确答案的概率，顶上是它最想写的字）、−ln p、整批的损失管、正在看的位置的前 5 名
  buildHead() {
    const g = (this.headG = new THREE.Group());
    this.root.add(g);
    const bg = new THREE.BoxGeometry(FS * 0.7, 1, FS * 0.7);
    bg.translate(0, 0.5, 0);
    this.pillars = new THREE.InstancedMesh(bg, cubeMaterial(0.45), NP);
    this.pillars.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(NP * 3), 3);
    this.pillars.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.pillars.userData.pick = (hit) => ({ type: 'pillar', i: hit.instanceId, click: true });
    this.E.pickables.push(this.pillars);
    g.add(this.pillars);
    this.topAtlas = atlas(16, 4);
    const fg = new THREE.PlaneGeometry(FS * 0.95, FS * 0.95);
    const cell = new Float32Array(NP);
    for (let i = 0; i < NP; i++) cell[i] = i;
    fg.setAttribute('aCell', new THREE.InstancedBufferAttribute(cell, 1));
    this.tops = new THREE.InstancedMesh(fg, atlasFaceMaterial(this.topAtlas), NP);
    this.tops.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(NP * 3), 3);
    this.tops.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.tops.frustumCulled = false;
    this.tops.raycast = () => {};
    g.add(this.tops);
    // 底座
    const W = BUNDLE.x1 - BUNDLE.x0 + 0.16, D = BUNDLE.z1 - BUNDLE.z0 + 0.16;
    const base = new THREE.Mesh(new THREE.BoxGeometry(W, 0.03, D), new THREE.MeshStandardMaterial({ color: 0x0c1629, metalness: 0.5, roughness: 0.4, transparent: true, opacity: 0.9 }));
    base.position.set(BCX, YP - 0.02, BCZ);
    const be = new THREE.LineSegments(new THREE.EdgesGeometry(base.geometry), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.3 }));
    be.position.copy(base.position);
    g.add(base, be);
    // −ln p
    const lg = new THREE.BoxGeometry(FS * 0.6, 1, FS * 0.6);
    lg.translate(0, 0.5, 0);
    this.lossBars = new THREE.InstancedMesh(lg, cubeMaterial(0.5), NP);
    this.lossBars.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(NP * 3), 3);
    this.lossBars.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.lossBars.userData.pick = (hit) => ({ type: 'nll', i: hit.instanceId, click: true });
    this.E.pickables.push(this.lossBars);
    g.add(this.lossBars);
    const lb = new THREE.Mesh(new THREE.BoxGeometry(NC * FS + 0.1, 0.03, D), base.material);
    lb.position.set(LOSSB.x0 + (NC - 1) * FS / 2, YP - 0.02, BCZ);
    g.add(lb);
    // 整批的损失管
    const glass = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, GAUGE.hMax, 24, 1, true), new THREE.MeshStandardMaterial({ color: 0x1a2c50, transparent: true, opacity: 0.18, depthWrite: false, side: THREE.DoubleSide }));
    glass.position.set(GAUGE.x, GAUGE.y + GAUGE.hMax / 2, GAUGE.z);
    glass.raycast = () => {};
    const fgeo = new THREE.CylinderGeometry(0.16, 0.16, 1, 24);
    fgeo.translate(0, 0.5, 0);
    this.gauge = new THREE.Mesh(fgeo, new THREE.MeshStandardMaterial({ color: 0x3a2a10, emissive: 0xffb65c, emissiveIntensity: 0.35, roughness: 0.4 }));
    this.gauge.position.set(GAUGE.x, GAUGE.y, GAUGE.z);
    this.gauge.scale.y = 0.001;
    this.gauge.userData.pick = { type: 'gauge' };
    this.E.pickables.push(this.gauge);
    const ln = new THREE.Mesh(new THREE.TorusGeometry(0.25, 0.012, 6, 32), new THREE.MeshBasicMaterial({ color: 0xff6b93, transparent: true, opacity: 0.7 }));
    ln.rotation.x = Math.PI / 2;
    ln.position.set(GAUGE.x, GAUGE.y + GAUGE.hMax * (Math.log(this.X.vocab) / GAUGE.max), GAUGE.z);
    g.add(glass, this.gauge, ln);
    // 正在看的那个位置：前 5 名立在 Eᵀ 的影子上（按字的编号排在 7478 行里的真实位置）
    const tb = new THREE.BoxGeometry(0.05, 1, 0.05);
    tb.translate(0, 0.5, 0);
    this.top5 = [];
    for (let j = 0; j < 5; j++) {
      const m = new THREE.Mesh(tb, new THREE.MeshStandardMaterial({ color: 0x0a1a1a, emissive: 0x5ef0d4, emissiveIntensity: 0.6 }));
      m.visible = false;
      m.raycast = () => {};
      g.add(m);
      this.top5.push(m);
    }
  }

  // 打印机：每个检查点用同一组随机数、同样的开头写 4 首诗
  buildPrinter() {
    const g = (this.printer = new THREE.Group());
    g.position.set(PRINTER.x, PRINTER.y, PRINTER.z);
    this.root.add(g);
    const body = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.5, 0.8), new THREE.MeshStandardMaterial({ color: 0x142036, metalness: 0.5, roughness: 0.4 }));
    body.position.y = 0.25;
    body.userData.pick = { type: 'printer' };
    this.E.pickables.push(body);
    const be = new THREE.LineSegments(new THREE.EdgesGeometry(body.geometry), new THREE.LineBasicMaterial({ color: 0xffb65c, transparent: true, opacity: 0.45 }));
    be.position.copy(body.position);
    const slot = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.03, 0.06), new THREE.MeshBasicMaterial({ color: 0xffb65c, transparent: true, opacity: 0.6 }));
    slot.position.set(0, 0.5, 0.1);
    this.printLed = slot;
    g.add(body, be, slot);
    this.paperCv = document.createElement('canvas');
    this.paperCv.width = 512; this.paperCv.height = 640;
    this.paperTex = new THREE.CanvasTexture(this.paperCv);
    this.paperTex.colorSpace = THREE.SRGBColorSpace;
    this.paperTex.anisotropy = 4;
    const pg = new THREE.PlaneGeometry(1.1, 1.375);
    pg.translate(0, 0.6875, 0);
    this.paper = new THREE.Mesh(pg, new THREE.MeshBasicMaterial({ map: this.paperTex, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
    this.paper.position.set(0, 0.5, 0.1);
    this.paper.rotation.x = -0.18;
    this.paper.userData.pick = { type: 'printer' };
    this.E.pickables.push(this.paper);
    g.add(this.paper);
    this.paperK = -1;
  }

  // 右边的两件展品：嵌入的地图（PCA）、损失地形
  buildExhibits() {
    const X = this.X, D = X.D, m = X.meta;
    // 嵌入的地图：一块竖着的玻璃板，316 个字的点
    const pw = PCA.x1 - PCA.x0, ph = PCA.y1 - PCA.y0;
    const plate = new THREE.Mesh(new THREE.PlaneGeometry(pw, ph), new THREE.MeshBasicMaterial({ color: 0x08101f, transparent: true, opacity: 0.75, depthWrite: false }));
    plate.position.set(PCA.x0 + pw / 2, PCA.y0 + ph / 2, PCA.z - 0.01);
    plate.userData.pick = (hit) => ({ type: 'pca', p: hit.point });
    this.E.pickables.push(plate);
    const pe = new THREE.LineSegments(new THREE.EdgesGeometry(plate.geometry), new THREE.LineBasicMaterial({ color: 0x8ee07a, transparent: true, opacity: 0.35 }));
    pe.position.copy(plate.position);
    pe.raycast = () => {};
    this.root.add(plate, pe);
    const N = D.NP;
    this.pcaN = N;
    const GROUPS = [['，。⏎', 0xffb65c], ['一二三四五六七八九十百千万', 0xff6b93], ['春夏秋冬', 0x8ee07a], ['红白青黄绿紫碧金', 0xb39dff], ['东南西北', 0x6b9bff]];
    const groupOf = new Map();
    GROUPS.forEach(([s, c], gi) => { for (const ch of s) groupOf.set(ch, gi); });
    const chars = D.pcaChars.map((c) => (c === '<|endoftext|>' ? '⏎' : c));
    this.pcaChars = chars;
    const lab = [], dots = [];
    chars.forEach((c, n) => { if (groupOf.has(c) || n <= 12) lab.push(n); else dots.push(n); });
    this.pcaLab = lab; this.pcaDots = dots;
    const dg = new THREE.BufferGeometry();
    dg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(dots.length * 3), 3));
    this.pcaPts = new THREE.Points(dg, new THREE.PointsMaterial({ color: 0xb4bed2, size: 0.05, transparent: true, opacity: 0.55, depthWrite: false }));
    this.pcaPts.frustumCulled = false;
    this.pcaPts.raycast = () => {};
    this.root.add(this.pcaPts);
    this.pcaAtlas = atlas(8, Math.ceil(lab.length / 8));
    this.pcaAtlas.draw(lab.map((n) => chars[n]));
    const fg = new THREE.PlaneGeometry(0.2, 0.2);
    const cell = new Float32Array(lab.length);
    lab.forEach((n, i) => { cell[i] = i; });
    fg.setAttribute('aCell', new THREE.InstancedBufferAttribute(cell, 1));
    this.pcaFaces = new THREE.InstancedMesh(fg, atlasFaceMaterial(this.pcaAtlas), lab.length);
    this.pcaFaces.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(lab.length * 3), 3);
    lab.forEach((n, i) => { const gi = groupOf.get(chars[n]); this.pcaFaces.setColorAt(i, new THREE.Color(gi != null ? GROUPS[gi][1] : 0xe9eff9)); });
    this.pcaFaces.frustumCulled = false;
    this.pcaFaces.raycast = () => {};
    this.root.add(this.pcaFaces);
    // 损失地形：41 × 41 个真实算出来的验证损失（主成分平面上），高度和颜色都是损失；轨迹 + 小球
    const Ld = m.land, G = Ld.G;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < G * G; i++) { lo = Math.min(lo, D.A.land[i]); hi = Math.max(hi, D.A.land[i]); }
    this.landLo = lo; this.landHi = hi;
    const lw = LAND.x1 - LAND.x0, ld = LAND.z1 - LAND.z0;
    const hOf = (v) => LAND.y0 + LAND.hMax * Math.pow((Math.log(v - lo + 0.05) - Math.log(0.05)) / (Math.log(hi - lo + 0.05) - Math.log(0.05)), 1.0);
    this.landH = hOf;
    const geo = new THREE.PlaneGeometry(lw, ld, G - 1, G - 1);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position, col = new Float32Array(pos.count * 3);
    for (let iy = 0; iy < G; iy++) for (let ix = 0; ix < G; ix++) {
      // 平面网格：第 iy 行在 z 方向从后往前（远 = y 大）
      const vi = (G - 1 - iy) * G + ix, v = D.land(iy, ix);
      pos.setY(vi, hOf(v) - LAND.y0);
      const tt = (Math.log(v - lo + 0.05) - Math.log(0.05)) / (Math.log(hi - lo + 0.05) - Math.log(0.05));
      seqInto(col, vi * 3, 1 - Math.floor(tt * 14) / 14, 0.32);
    }
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.computeVertexNormals();
    const terr = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8, metalness: 0.05, transparent: true, opacity: 0.92, side: THREE.DoubleSide }));
    terr.position.set(LAND.x0 + lw / 2, LAND.y0, LAND.z0 + ld / 2);
    terr.userData.pick = (hit) => ({ type: 'land', p: hit.point });
    this.E.pickables.push(terr);
    const wire = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x5ef0d4, wireframe: true, transparent: true, opacity: 0.035, depthWrite: false }));
    wire.position.copy(terr.position);
    wire.raycast = () => {};
    this.root.add(terr, wire);
    this.landXZ = (a, b) => [LAND.x0 + ((a - Ld.x[0]) / (Ld.x[1] - Ld.x[0])) * lw, LAND.z1 - ((b - Ld.y[0]) / (Ld.y[1] - Ld.y[0])) * ld];
    const path = Ld.path.map(([a, b], k) => { const [x, z] = this.landXZ(a, b); return new THREE.Vector3(x, hOf(Ld.pathLoss[k]) + 0.06, z); });
    this.landPath = path;
    const plg = new THREE.BufferGeometry().setFromPoints(path);
    this.landLine = new THREE.Line(plg, new THREE.LineBasicMaterial({ color: 0xffb65c, transparent: true, opacity: 0.9 }));
    this.landLine.raycast = () => {};
    const ghost = new THREE.Line(plg.clone(), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.15 }));
    ghost.raycast = () => {};
    this.ball = new THREE.Mesh(new THREE.SphereGeometry(0.11, 18, 12), new THREE.MeshStandardMaterial({ color: 0x3a2a10, emissive: 0xffb65c, emissiveIntensity: 0.8 }));
    this.ball.raycast = () => {};
    this.root.add(ghost, this.landLine, this.ball);
  }

  // D4：一块权重局部放大成 48 × 48 个方块（颜色 = 正负和大小，高度 = |w|）
  buildMagnifier() {
    const g = (this.mag = new THREE.Group());
    g.visible = false;
    this.root.add(g);
    const n = C48 * C48;
    const geo = new THREE.BoxGeometry(MAG.face, MAG.face, 1);
    geo.translate(0, 0, 0.5);
    this.cubes = new THREE.InstancedMesh(geo, cubeMaterial(0.34), n);
    this.cubes.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
    this.cubes.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.cubes.frustumCulled = false;
    this.cubes.userData.pick = (hit) => ({ type: 'cube', n: hit.instanceId });
    this.E.pickables.push(this.cubes);
    const gg = new THREE.BoxGeometry(MAG.face * 1.3, MAG.face * 1.3, 1);
    gg.translate(0, 0, 0.5);
    this.cubeGlow = new THREE.InstancedMesh(gg, new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }), n);
    this.cubeGlow.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
    this.cubeGlow.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.cubeGlow.frustumCulled = false;
    this.cubeGlow.raycast = () => {};
    const S = C48 * MAG.cell;
    const plate = new THREE.Mesh(new THREE.BoxGeometry(S + 0.08, S + 0.08, 0.02), new THREE.MeshStandardMaterial({ color: 0x0a1222, roughness: 0.6, metalness: 0.2, transparent: true, opacity: 0.92 }));
    plate.position.set(S / 2, -S / 2, -0.012);
    plate.raycast = () => {};
    const pe = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(S + 0.1, S + 0.1)), new THREE.LineBasicMaterial({ color: 0xffd18a, transparent: true, opacity: 0.6 }));
    pe.position.set(S / 2, -S / 2, 0.001);
    pe.raycast = () => {};
    this.magScan = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0xbff8ff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.magScan.raycast = () => {};
    g.add(plate, pe, this.cubes, this.cubeGlow, this.magScan);
    // 嵌入的局部：每一行是哪个字
    this.magRows = atlas(8, 6);
    this.magRows.draw(Array.from({ length: 48 }, (_, i) => this.X.ch(i + 1)));
    const rg = new THREE.PlaneGeometry(MAG.cell * 0.9, MAG.cell * 0.9);
    const cell = new Float32Array(48);
    for (let i = 0; i < 48; i++) cell[i] = i;
    rg.setAttribute('aCell', new THREE.InstancedBufferAttribute(cell, 1));
    this.magRowFaces = new THREE.InstancedMesh(rg, atlasFaceMaterial(this.magRows, 0xb4bed2), 48);
    for (let i = 0; i < 48; i++) { this.tmpM.makeTranslation(-MAG.cell * 0.7, -(i + 0.5) * MAG.cell, 0.01); this.magRowFaces.setMatrixAt(i, this.tmpM); }
    this.magRowFaces.raycast = () => {};
    g.add(this.magRowFaces);
    // 从面板角上的小块连到放大的方块阵：四条光线
    this.magLines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffd18a, transparent: true, opacity: 0.35, depthWrite: false }));
    this.magLines.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(8 * 3), 3));
    this.magLines.frustumCulled = false;
    this.magLines.visible = false;
    this.magLines.raycast = () => {};
    this.root.add(this.magLines);
    this.magC = -1;
  }

  // D5：跟踪的那个权重在面板上的位置（方块 + 十字线），以及悬停 / 选中的框
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
    this.featCube = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0x3a2a10, emissive: 0xffb65c, emissiveIntensity: 0.9 }));
    this.featCube.visible = false;
    this.featCube.raycast = () => {};
    this.root.add(this.featCube);
    this.cross = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffd18a, transparent: true, opacity: 0.5, depthWrite: false }));
    this.cross.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(4 * 3), 3));
    this.cross.frustumCulled = false;
    this.cross.visible = false;
    this.cross.raycast = () => {};
    this.root.add(this.cross);
  }

  buildLabels() {
    const add = (html, cls, x, y, z, cx, cy) => { const o = label(html, `lbl ${cls}`); o.position.set(x, y, z); o.center.set(cx, cy); this.root.add(o); return o; };
    this.lPanel = new Map();
    for (const P of this.panels) {
      if (P.kind === 'qn' || P.kind === 'kn') continue;
      const nm = P.kind === 'ET' ? 'Eᵀ' : KIND_LABEL[P.kind];
      const sh = P.kind === 'ET' ? '[256 × 7478]' : `[${kindShape(P.kind).join(' × ')}]`;
      const note = P.kind === 'ET' ? t('t3.tied') : P.crop >= 0 ? t('t3.realCrop') : t('t3.statsOnly');
      let o;
      if (P.flat) o = add(`${nm}<small>${sh} · ${note}</small>`, 'part t3p', P.x0 + 1.08, P.y + 0.02, P.z0 - 0.4, 0, 0.5);
      else if (P.kind === 'ln1' || P.kind === 'ln2' || P.kind === 'nf') o = add(`${nm}<small>${sh}</small>`, 'part t3p', P.x0 + P.w / 2, P.y0 + P.h + 0.02, 0.05, 0.5, 1.1);
      else if (P.kind === 'k' || P.kind === 'v') o = add(`${nm}<small>${kindShape(P.kind).join('×')}</small>`, 'part t3p', P.x0 + P.w / 2, P.y0 - 0.03, 0.05, 0.5, 0);
      else o = add(`${nm}<small>${sh} · ${note}</small>`, 'part t3p', P.x0, P.y0 - 0.03, 0.05, 0, 0);
      o.short = nm;
      o.full = o.el.innerHTML;
      o.visible = false;
      this.lPanel.set(P, o);
    }
    this.lLayer = [];
    for (let l = 0; l < NLAY; l++) this.lLayer.push(add(t('t3.layer', { l }), 'title t3l', LENS.x0 - 0.15, yBand(l) + HB_MID, 0, 1, 0.5));
    this.lLens = [];
    for (let b = 0; b <= NLAY; b++) this.lLens.push(add(b === 0 ? t('t3.lens0') : t('t3.lensL', { l: b - 1 }), 'hint t3lens', LENS.x0, yBound(b) + LENS_H / 2 + 0.04, 0.06, 0, 1));
    this.lHeads = [];
    for (let l = 0; l < NLAY; l++) this.lHeads.push(add(t('t3.heads'), 'hint', headX(0), yBand(l) + SUB.heads1 + 0.04, 0.05, 0, 1));
    this.lBundle = add(t('t3.bundle'), 'hint t3w', BCX, 0.55, BUNDLE.z1 + 0.15, 0.5, 0);
    this.lTray = add(t('t3.tray'), 'hint t3w', BCX, -0.12, BUNDLE.z1 + 0.5, 0.5, 0);
    this.lStack = add(t('t3.stack'), 'num t3dim', BUNDLE.x0 - 0.05, TRAY.y - 0.25, BUNDLE.z1, 1, 0.5);
    this.lCorpus = add(t('t3.corpus'), 'hint t3w', (CORPUS.x0 + CORPUS.x1) / 2, CORPUS.y + 0.12, CORPUS.z, 0.5, 1);
    this.lPillars = add(t('t3.pillars'), 'hint t3w', BCX, YP + PH + 0.3, BCZ, 0.5, 1);
    this.lLoss = add(t('t3.lossBars'), 'hint t3w', LOSSB.x0 + NC * FS / 2, YP + 1.45, BCZ, 0.5, 1);
    this.lGauge = add('', 'big t3g', GAUGE.x + 0.3, GAUGE.y + 0.6, 0, 0, 0.5);
    this.lClip = add('', 'num gc', GAUGE.x + 0.3, GAUGE.y - 0.3, 0.2, 0, 0);
    this.lGap = add('', 'num gt', BCX, -0.5, BUNDLE.z1 + 0.9, 0.5, 0);
    this.lPrinter = add(t('t3.printer'), 'hint t3w', PRINTER.x, PRINTER.y - 0.06, PRINTER.z + 0.45, 0.5, 0);
    this.lPca = add(t('t3.pca'), 'hint t3w', PCA.x0, PCA.y1 + 0.08, PCA.z, 0, 1);
    this.lLand = add(t('t3.land'), 'hint t3w', LAND.x0, LAND.y0 + LAND.hMax + 0.5, LAND.z0, 0, 1);
    this.lMag = add('', 'part t3mag', 0, 0, 0, 0, 1.1);
    this.lFeat = add('', 'num gsel', 0, 0, 0, 0.5, 1.3);
    this.lTop5 = [];
    for (let j = 0; j < 5; j++) this.lTop5.push(add('', 'num t3t5', 0, 0, 0, 0.5, 1.1));
    this.lUpdNote = add('', 'num t3dim', BCX, YP + PH + 0.8, BCZ, 0.5, 1);
  }

  /* ================================================================ 每个检查点的数据 */

  // 数据分块到了（或换了检查点）：重画字表、透镜屏、局部、注意力图案、查表的光、打印机的纸
  bake(k) {
    const X = this.X;
    if (this.baked.row !== k) {
      this.baked.row = k;
      this.rowAtlas.draw(Array.from({ length: NP + 1 }, (_, i) => X.ch(X.row(k, i))));
      this.bakeBeams(k);
      this.lens.forEach((ls) => { ls.key = ''; });
    }
    if (X.has('st', k) && this.baked.st !== k) {
      this.baked.st = k;
      this.topAtlas.draw(Array.from({ length: NP }, (_, i) => X.ch(X.rowTop(k, i)[0].id)));
      this.lens.forEach((ls) => { ls.key = ''; });
    }
    if (X.has('ck', k) && this.baked.ck !== k) {
      this.baked.ck = k;
      this.bakeHeads(k);
      for (const P of this.panels) if (P.patch) P.patchKey = '';
    }
    if (X.has('dw', k) && this.baked.dw !== k) {
      this.baked.dw = k;
      for (const P of this.panels) if (P.patch) P.patchKey = '';
    }
    if (this.paperK !== k) { this.paperK = k; this.drawPaper(k); }
  }

  bakeBeams(k) {
    const X = this.X, a = this.beamGeo.attributes.position.array;
    let o = 0;
    const v0 = new THREE.Vector3(), v1 = new THREE.Vector3(), c = new THREE.Vector3(), p = new THREE.Vector3(), q = new THREE.Vector3();
    for (let i = 0; i < NP; i++) {
      const id = X.row(k, i);
      v0.set(fiberX(i), TRAY.y + 0.05, fiberZ(i));
      v1.set(EMB.x0 + 0.5, EMB.y + 0.01, embZ(id));
      c.set((v0.x + v1.x) / 2, 0.55 + Math.min(1.6, Math.abs(v1.z - v0.z) * 0.08), (v0.z + v1.z) / 2);
      for (let s = 0; s < 16; s++) {
        const t0 = s / 16, t1 = (s + 1) / 16;
        p.copy(v0).multiplyScalar((1 - t0) ** 2).addScaledVector(c, 2 * (1 - t0) * t0).addScaledVector(v1, t0 * t0);
        q.copy(v0).multiplyScalar((1 - t1) ** 2).addScaledVector(c, 2 * (1 - t1) * t1).addScaledVector(v1, t1 * t1);
        a[o++] = p.x; a[o++] = p.y; a[o++] = p.z; a[o++] = q.x; a[o++] = q.y; a[o++] = q.z;
      }
    }
    this.beamGeo.attributes.position.needsUpdate = true;
    // 输出头：每根光纤顶上一条竖线下到 Eᵀ 的影子上方（示意：和 7478 行逐一点积）
    const h = this.headGeo.attributes.position.array;
    o = 0;
    for (let i = 0; i < NP; i++) {
      const top = X.has('st', k) ? X.rowTop(k, i)[0].id : 0;
      v0.set(fiberX(i), YTOP + 0.25, fiberZ(i));
      v1.set(EMB.x0 + 0.5, YT + 0.01, embZ(top));
      c.set((v0.x + v1.x) / 2, YT + 0.6, (v0.z + v1.z) / 2);
      for (let s = 0; s < 16; s++) {
        const t0 = s / 16, t1 = (s + 1) / 16;
        p.copy(v0).multiplyScalar((1 - t0) ** 2).addScaledVector(c, 2 * (1 - t0) * t0).addScaledVector(v1, t0 * t0);
        q.copy(v0).multiplyScalar((1 - t1) ** 2).addScaledVector(c, 2 * (1 - t1) * t1).addScaledVector(v1, t1 * t1);
        h[o++] = p.x; h[o++] = p.y; h[o++] = p.z; h[o++] = q.x; h[o++] = q.y; h[o++] = q.z;
      }
    }
    this.headGeo.attributes.position.needsUpdate = true;
  }

  bakeHeads(k) {
    const X = this.X;
    for (let l = 0; l < NLAY; l++) for (let h = 0; h < 4; h++) {
      const { data, tex } = this.heads[l][h];
      for (let i = 0; i < 25; i++) for (let j = 0; j < 25; j++) {
        const o = ((24 - i) * 25 + j) * 4;   // 贴图第 0 行在下面：第 i 个字画在上面第 i 行
        if (j > i) { data[o] = 10; data[o + 1] = 16; data[o + 2] = 30; data[o + 3] = 200; continue; }
        const a = Math.min(1, Math.pow(X.attn(k, l, h, i, j), 0.6));
        data[o] = 14 + a * (179 - 14); data[o + 1] = 20 + a * (157 - 20); data[o + 2] = 38 + a * (255 - 38); data[o + 3] = 235;
      }
      tex.needsUpdate = true;
    }
  }

  // 打印机的纸：4 首诗（开头琥珀色），每首后面标格律对不对
  drawPaper(k) {
    const X = this.X, m = X.meta, c = this.paperCv, g = c.getContext('2d');
    g.clearRect(0, 0, c.width, c.height);
    g.fillStyle = 'rgba(150,146,134,0.92)';
    g.fillRect(0, 0, c.width, c.height);
    g.fillStyle = 'rgba(30,26,22,0.8)';
    g.font = `600 22px ${FONT}`;
    g.textBaseline = 'top';
    g.fillText(t('t3.paperHead', { t: X.step(k) + 1 }), 22, 18);
    let y = 58;
    const size = 30;
    m.ckpts[k].samples.forEach((s, i) => {
      const pf = m.prefixes[i];
      let x = 22, line = 0;
      g.font = `600 ${size}px ${FONT}`;
      for (let j = 0; j < s.length && line < 3; j++) {
        const ch = s[j];
        const w = g.measureText(ch).width;
        if (x + w > c.width - 22) { x = 22; y += size + 6; line++; if (line >= 3) break; }
        g.fillStyle = j < pf.length ? '#7a4206' : ch === '，' || ch === '。' ? 'rgba(30,26,22,0.5)' : '#10141e';
        g.fillText(ch, x, y);
        x += w;
      }
      y += size + 22;
    });
    this.paperTex.needsUpdate = true;
  }

  // 一块透镜屏：第 b 个边界的 64 个字，绿 = 正好是答案；字下面一道细条 = 给答案的概率
  drawLens(b, k, mode) {
    const X = this.X, ls = this.lens[b], key = `${k}:${mode}:${X.has('st', k)}`;
    if (ls.key === key) return;
    ls.key = key;
    const g = ls.g, W = 1024, cw = 64, rh = 80;
    g.clearRect(0, 0, W, 320);
    if (!X.has('st', k)) return void (ls.tex.needsUpdate = true);
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.font = `600 44px ${FONT}`;
    for (let i = 0; i < NP; i++) {
      const id = X.lensTop(k, b, i), tg = X.tgt(k, i), ok = id === tg, p = X.lensP(k, b, i);
      const cx = (i % NC) * cw + cw / 2, cy = Math.floor(i / NC) * rh + 36;
      g.fillStyle = ok ? '#8ee07a' : 'rgba(170,182,206,0.82)';
      g.fillText(X.ch(id), cx, cy);
      g.fillStyle = 'rgba(94,240,212,0.18)';
      g.fillRect(cx - 26, cy + 30, 52, 5);
      g.fillStyle = 'rgba(94,240,212,0.85)';
      g.fillRect(cx - 26, cy + 30, Math.max(1, 52 * Math.min(1, Math.sqrt(p))), 5);
    }
    ls.tex.needsUpdate = true;
  }

  // 一块局部的贴图：mode = w（权重）、g（梯度）、d（ΔW），f = 混合
  drawPatch(P, k, st) {
    const X = this.X, c = P.crop;
    const key = `${k}:${st.mode}:${st.f.toFixed(2)}:${st.done.toFixed(2)}:${X.has('ck', k)}:${X.has('dw', k)}`;
    if (P.patchKey === key) return;
    P.patchKey = key;
    const d = P.patchData;
    if (!X.has('ck', k)) { d.fill(0); P.patchTex.needsUpdate = true; return; }
    const sw = this.cropScale(k, c), sg = this.gScale(k, c), sd = this.dwScale(k, c);
    for (let r = 0; r < C48; r++) for (let q = 0; q < C48; q++) {
      // 显示的第 r 行、第 q 列 → PyTorch 下标
      const [i, j] = c === 0 ? [r, q] : [q, r];
      const w0 = X.wcrop(k, c, i, j), dw = X.dwcrop(k, c, i, j);
      const w = w0 + dw * st.done;
      let col = divBytes(w / sw);
      if (st.mode === 'g' && st.f > 0) col = mix3(col, divBytes(X.gcrop(k, c, i, j) / sg, GPOS, GNEG, [14, 18, 34]), st.f);
      if (st.mode === 'd' && st.f > 0 && sd > 0) col = mix3(col, divBytes(dw / sd, DPOS, DNEG, [14, 18, 34]), st.f * (1 - st.done));
      const o = ((P.flat ? r : C48 - 1 - r) * C48 + q) * 4;
      d[o] = col[0]; d[o + 1] = col[1]; d[o + 2] = col[2]; d[o + 3] = 255;
    }
    P.patchTex.needsUpdate = true;
  }
  cropScale(k, c) { return this.X.meta.ckpts[k].wScale[c] * 127 * 0.8; }
  gScale(k, c) {
    const ck = this.gsc || (this.gsc = new Map()), key = `${k}:${c}`;
    if (ck.has(key)) return ck.get(key);
    const a = [];
    for (let i = 0; i < C48; i++) for (let j = 0; j < C48; j++) a.push(Math.abs(this.X.gcrop(k, c, i, j)));
    a.sort((x, y) => x - y);
    const v = a[Math.floor(a.length * 0.99)] || 1e-12;
    ck.set(key, v);
    return v;
  }
  dwScale(k, c) {
    const X = this.X;
    if (!X.hasDw(k)) return 0;
    const ck = this.dwsc || (this.dwsc = new Map()), key = `${k}:${c}`;
    if (ck.has(key)) return ck.get(key);
    let mx = 0;
    for (let i = 0; i < C48; i++) for (let j = 0; j < C48; j++) mx = Math.max(mx, Math.abs(X.dwcrop(k, c, i, j)));
    ck.set(key, mx || 1e-12);
    return mx || 1e-12;
  }

  /* ================================================================ 状态 */

  animP(st, t) {
    const key = `${st.depth}:${st.k}:${st.i}`;
    if (this.pk !== key) { this.pk = key; this.pt0 = t; }
    const dur = { 't-run': 3.4, 't-step': 3, 't-op': 1.6, 't-crop': 2.4, 't-pos': 0.9, 't-param': 1.6 }[st.view] || 1.5;
    return st.playing ? st.p : Math.max(st.p, Math.min(1, (t - this.pt0) / dur));
  }

  // 这一刻机器走到了哪
  flow(st, p) {
    const s = st.step;
    const F = { feed: 0, shift: 0, ids: 0, fwd: -1, lossUpTo: 0, lossPos: -1, lossMean: 0, bwd: -1, clip: 0, upd: 0, done: 0, cu: [{ dw: 0, done: 0 }, { dw: 0, done: 0 }, { dw: 0, done: 0 }], print: 1 };
    const allCrops = (dw, done) => { for (const c of F.cu) { c.dw = dw; c.done = done; } };
    if (s.ph === 'run') {
      F.print = seg(p, 0, 0.4);
      F.feed = seg(p, 0, 0.09); F.shift = seg(p, 0.07, 0.12);
      F.fwd = 14 * seg(p, 0.12, 0.44);
      F.lossUpTo = 64 * seg(p, 0.44, 0.5); F.lossMean = seg(p, 0.48, 0.53);
      F.bwd = 14 * seg(p, 0.54, 0.78);
      F.clip = seg(p, 0.78, 0.8);
      F.upd = seg(p, 0.8, 0.88); F.done = seg(p, 0.88, 0.97);
      allCrops(F.upd, F.done);
      return F;
    }
    const PHI = PHASES.indexOf(s.ph);
    F.feed = 1; F.shift = 1; F.ids = 1;
    if (PHI >= 2) { F.fwd = 14; F.lossUpTo = 64; F.lossMean = 1; }
    if (PHI >= 4) { F.bwd = 14; F.clip = 1; }
    if (!s.sub) {
      if (s.ph === 'batch') { F.feed = easeOut(seg(p, 0, 0.6)); F.shift = easeOut(seg(p, 0.62, 0.92)); F.ids = 0; }
      else if (s.ph === 'fwd') F.fwd = Math.min(14, p * 14.8);
      else if (s.ph === 'loss') { F.lossUpTo = 64 * seg(p, 0, 0.55); F.lossMean = easeOut(seg(p, 0.55, 0.9)); }
      else if (s.ph === 'bwd') F.bwd = Math.min(14, p * 14.8);
      else if (s.ph === 'upd') { F.clip = seg(p, 0, 0.15); F.upd = seg(p, 0.18, 0.5); F.done = seg(p, 0.55, 0.92); allCrops(F.upd, F.done); }
      return F;
    }
    switch (s.ph) {
      case 'batch':
        if (s.sub === 'row') { F.feed = easeOut(p); F.shift = 0; F.ids = 0; }
        else if (s.sub === 'ids') { F.shift = 0; F.ids = easeOut(seg(p, 0, 0.7)); }
        else F.shift = easeOut(seg(p, 0.1, 0.7));
        break;
      case 'fwd': case 'bwd': {
        const list = s.ph === 'fwd' ? FWD : BWD, oi = opIndex(list, s);
        const v = oi + easeInOut(seg(p, 0, 0.82));
        if (s.ph === 'fwd') F.fwd = v; else F.bwd = v;
        break;
      }
      case 'loss':
        if (s.sub === 'pos' && s.i == null) { F.lossUpTo = 64 * seg(p, 0, 0.85); F.lossMean = 0; }
        else if (s.sub === 'pos') { F.lossPos = s.i; F.lossUpTo = s.i + easeOut(seg(p, 0, 0.5)); F.lossMean = 0; }
        else F.lossMean = easeOut(seg(p, 0, 0.6));
        break;
      case 'upd': {
        if (s.sub === 'clip') { F.clip = seg(p, 0, 0.6); break; }
        if (s.c == null && s.f == null) { F.upd = seg(p, 0, 0.45); F.done = seg(p, 0.5, 0.95); allCrops(F.upd, F.done); break; }
        // D4 / D5：一块一块地看。其余面板当作已经更新完；这块之前的局部更新完，之后的还没动
        F.upd = 1; F.done = 1;
        const cur = s.c ?? -1;
        for (let c = 0; c < 3; c++) {
          const before = s.f != null ? (FEAT_CROP.indexOf(c) < s.f) : c < cur;
          F.cu[c] = before ? { dw: 1, done: 1 } : { dw: 0, done: 0 };
        }
        if (cur >= 0) {
          if (s.mi) {
            const mi = ADAM_SUBS.indexOf(s.mi);
            F.cu[cur] = mi < 4 ? { dw: 0, done: 0 } : mi === 4 ? { dw: easeOut(p), done: 0 } : { dw: 1, done: easeInOut(seg(p, 0.1, 0.8)) };
          } else F.cu[cur] = { dw: seg(p, 0, 0.45), done: seg(p, 0.5, 0.95) };
        }
        break;
      }
    }
    return F;
  }

  /* ================================================================ 每帧 */

  update(st, dt, t) {
    const X = this.X, s = st.step, k = st.k;
    this.small = this.E.w < 700;
    this.bake(k);
    const p = this.animP(st, t);
    const F = (this.F = this.flow(st, p));
    this.st = st; this.p = p;
    const haveSt = X.has('st', k), haveCk = X.has('ck', k);
    this.updBundle(st, F, haveSt, t);
    this.updTray(st, F);
    this.updPanels(st, F, haveSt);
    this.updLens(st, F);
    this.updHead(st, F, haveSt, t);
    this.updPrinter(st, F);
    this.updExhibits(st, F, haveCk);
    this.updMagnifier(st, F, haveCk, t);
    this.updMarks(st, t);
    this.updLabels(st, F);
    const close = st.view === 't-crop' || st.view === 't-param';
    const bl = close ? 0.18 : st.view === 't-op' ? 0.22 : 0.24;
    this.E.bloom.strength += (bl - this.E.bloom.strength) * Math.min(1, dt * 3);
    this.dust.rotation.y += dt * 0.006;
    this.corpusTex.offset.x = (t * 0.02) % 1;
  }

  // 光纤 + 脉冲 + 环 + 分支管
  updBundle(st, F, haveSt, t) {
    const X = this.X, k = st.k, M = this.tmpM;
    const fx = F.fwd, bx = F.bwd;
    const fOn = fx >= 0 && fx < 14, bOn = bx >= 0 && bx < 14;
    // 前向脉冲的高度：0 嵌入、1 + 2l 注意力、2 + 2l 前馈、13 输出头
    const yF = (x) => {
      const o = Math.min(13, Math.floor(x)), f = x - o;
      if (o === 0) return lerp(0.25, yBound(0), f);
      if (o === 13) return lerp(yBound(NLAY), YTOP + 0.4, f);
      const l = Math.floor((o - 1) / 2), yb = yBand(l);
      return (o - 1) % 2 === 0 ? lerp(yBound(l), yb + SUB.add1, f) : lerp(yb + SUB.add1, yBound(l + 1), f);
    };
    const yB = (x) => {
      const o = Math.min(13, Math.floor(x)), f = x - o;
      if (o === 0) return lerp(YTOP + 0.4, yBound(NLAY), f);
      if (o === 13) return lerp(yBound(0), 0.25, f);
      const l = NLAY - 1 - Math.floor((o - 1) / 2), yb = yBand(l);
      return (o - 1) % 2 === 0 ? lerp(yBound(l + 1), yb + SUB.add1, f) : lerp(yb + SUB.add1, yBound(l), f);
    };
    // 前向：每个位置的亮度 = 这里残差 ‖h‖（真实，按这一步所有边界取对数归一）
    let n = 0;
    if (fOn) {
      const y = yF(fx), b = Math.max(0, Math.min(NLAY, Math.round((y - yBound(0)) / 3.5)));
      let lo = Infinity, hi = -Infinity;
      if (haveSt) for (let i = 0; i < NP; i++) { const v = Math.log(X.residNorm(k, b, i) + 1e-6); lo = Math.min(lo, v); hi = Math.max(hi, v); }
      for (let i = 0; i < NP; i++) {
        const sc = haveSt ? 0.7 + 0.6 * ((Math.log(X.residNorm(k, b, i) + 1e-6) - lo) / (hi - lo || 1)) : 1;
        M.makeRotationX(Math.PI / 2).scale(this.tmpS.set(sc, sc, sc)).setPosition(fiberX(i), y, fiberZ(i));
        this.pulseF.setMatrixAt(n++, M);
      }
    }
    this.pulseF.count = n;
    this.pulseF.instanceMatrix.needsUpdate = true;
    this.pulseF.material.opacity = 0.55 + 0.25 * Math.sin(t * 9);
    const yf = fOn ? yF(fx) : -99;
    this.slabF.position.y = yf;
    this.slabF.material.opacity = fOn ? 0.32 : 0;
    // 反向：每个位置的环大小 = 这里 ‖∂L/∂h‖（真实；在相邻两个层边界之间插值，按这一步的最大值归一）
    n = 0;
    if (bOn) {
      const y = yB(bx);
      const bf = Math.max(0, Math.min(NLAY, (y - yBound(0)) / 3.5)), b0 = Math.floor(bf), b1 = Math.min(NLAY, b0 + 1), fr = bf - b0;
      if (!this.rgMax || this.rgMax.k !== k || this.rgMax.have !== haveSt) {
        let mx = 1e-12;
        if (haveSt) for (let b = 0; b <= NLAY; b++) for (let i = 0; i < NP; i++) mx = Math.max(mx, Math.log1p(X.residGrad(k, b, i) * 1e5));
        this.rgMax = { k, have: haveSt, mx };
      }
      for (let i = 0; i < NP; i++) {
        let sc = 1;
        if (haveSt) { const g = lerp(X.residGrad(k, b0, i), X.residGrad(k, b1, i), fr); sc = 0.35 + 1.5 * Math.log1p(g * 1e5) / this.rgMax.mx; }
        M.makeRotationX(Math.PI / 2).scale(this.tmpS.set(sc, sc, sc)).setPosition(fiberX(i), y, fiberZ(i));
        this.pulseB.setMatrixAt(n++, M);
      }
    }
    this.pulseB.count = n;
    this.pulseB.instanceMatrix.needsUpdate = true;
    this.pulseB.material.opacity = 0.55 + 0.25 * Math.sin(t * 9);
    const yb2 = bOn ? yB(bx) : -99;
    this.slabB.position.y = yb2;
    this.slabB.material.opacity = bOn ? 0.3 : 0;
    for (const fl of this.floors) {
      const a = Math.exp(-Math.abs(yf - fl.y) * 2.2), b = Math.exp(-Math.abs(yb2 - fl.y) * 2.2);
      fl.e.material.opacity = 0.1 + 0.5 * Math.max(a, b);
      fl.e.material.color.setHex(b > a ? 0xff6b93 : 0x5ef0d4);
    }
    // 光纤：有数据流过时亮，进料前暗
    const ca = this.fibers.instanceColor.array, base = F.feed > 0.5 ? 0.42 : 0.15;
    for (let i = 0; i < NP; i++) {
      let r = 0.18 * base, g2 = 0.75 * base, b2 = 0.68 * base;
      if (F.lossPos === i) { r = 1; g2 = 0.8; b2 = 0.45; }
      ca[i * 3] = r; ca[i * 3 + 1] = g2; ca[i * 3 + 2] = b2;
    }
    this.fibers.instanceColor.needsUpdate = true;
    // 环和分支管：脉冲经过时亮
    const opF = (o) => fOn && fx >= o && fx < o + 1, opB = (o) => bOn && bx >= o && bx < o + 1;
    const lit = (m, on, lo = 0.3, hi = 0.95) => { m.material.opacity += ((on ? hi : lo) - m.material.opacity) * 0.25; };
    for (let l = 0; l < NLAY; l++) {
      const R = this.rings[l], P = this.pipes[l], fa = 1 + 2 * l, ff = 2 + 2 * l, ba = 2 + 2 * (NLAY - 1 - l), bf = 1 + 2 * (NLAY - 1 - l);
      const subF = (o, a, b) => opF(o) && fx - o >= a && fx - o < b, subB = (o, a, b) => opB(o) && bx - o >= a && bx - o < b;
      lit(R.norm1, subF(fa, 0, 0.2) || subB(ba, 0.75, 1));
      lit(R.add1, subF(fa, 0.88, 1) || subB(ba, 0, 0.12));
      lit(R.norm2, subF(ff, 0, 0.2) || subB(bf, 0.75, 1));
      lit(R.add2, subF(ff, 0.88, 1) || subB(bf, 0, 0.12));
      lit(P.a, subF(fa, 0, 0.25) || subB(ba, 0.7, 1), 0.12, 0.8);
      lit(P.aOut, subF(fa, 0.85, 1) || subB(ba, 0, 0.15), 0.12, 0.8);
      lit(P.f, subF(ff, 0, 0.25) || subB(bf, 0.7, 1), 0.12, 0.8);
      lit(P.fOut, subF(ff, 0.85, 1) || subB(bf, 0, 0.15), 0.12, 0.8);
      lit(P.act, subF(ff, 0.5, 0.62) || subB(bf, 0.35, 0.5), 0.1, 0.7);
    }
    lit(this.ringF, opF(13) && fx - 13 < 0.25 || opB(0) && bx > 0.7);
    // 查表的光：前向的嵌入这一步
    const look = fOn && fx < 1 ? fx : -1;
    this.beams.material.opacity = look >= 0 ? 0.55 * Math.min(1, look * 3) * (look < 0.85 ? 1 : (1 - look) / 0.15) : 0;
    this.beamGeo.setDrawRange(0, look >= 0 ? Math.floor(NP * 32 * Math.min(1, look * 1.4) / 2) * 2 : 0);
    const hd = fOn && fx >= 13 ? fx - 13 : -1;
    this.headBeams.material.opacity = hd >= 0 ? 0.35 * Math.sin(Math.min(1, hd / 0.6) * Math.PI) : 0;
  }

  // 托盘：语料 → 字块飞进来；“编号”时字块上浮出编号（调试器里写）；错开一位的答案排在前面
  updTray(st, F) {
    const T = TRAY.tile, M = this.tmpM, Q = this.tmpQ, S = this.tmpS, V = this.tmpV;
    const show = F.feed > 0;
    for (let i = 0; i < NP; i++) {
      const d0 = (i / NP) * 0.6, f = show ? clamp01((F.feed - d0) / 0.4) : 0;
      if (f <= 0) { M.makeScale(0, 0, 0); this.tiles.setMatrixAt(i, M); this.tileFaces.setMatrixAt(i, M); continue; }
      const e = easeInOut(f);
      const fromX = CORPUS.x1 - 0.3 - (NP - i) * 0.06, toX = fiberX(i), toZ = fiberZ(i);
      V.set(lerp(fromX, toX, e), TRAY.y + Math.sin(e * Math.PI) * 0.7, lerp(CORPUS.z, toZ, e));
      const hi = F.lossPos === i ? 1.25 : 1;
      Q.identity();
      M.compose(V, Q, S.set(hi, hi, hi));
      this.tiles.setMatrixAt(i, M);
      V.y += T * 0.3 * hi + 0.002;
      M.compose(V, Q, S.set(hi, hi, hi));
      this.tileFaces.setMatrixAt(i, M);
    }
    for (let i = 0; i < NP; i++) {
      const f = F.shift;
      if (f <= 0.01) { M.makeScale(0, 0, 0); this.tgts.setMatrixAt(i, M); this.tgtFaces.setMatrixAt(i, M); continue; }
      const e = easeInOut(f);
      // 从“下一个字”的位置滑到前面一排的同一列（错开一位）
      const j = Math.min(NP - 1, i + 1);
      V.set(lerp(fiberX(j), fiberX(i), e), TRAY.y + Math.sin(e * Math.PI) * 0.18, lerp(fiberZ(j), fiberZ(i) + TGT_DZ, e));
      const sc = 0.82;
      M.compose(V, Q.identity(), S.set(sc, sc, sc));
      this.tgts.setMatrixAt(i, M);
      V.y += T * 0.3 * sc + 0.002;
      M.compose(V, Q, S);
      this.tgtFaces.setMatrixAt(i, M);
    }
    for (const m of [this.tiles, this.tileFaces, this.tgts, this.tgtFaces]) m.instanceMatrix.needsUpdate = true;
  }

  // 面板：前向扫描、反向按 ‖∇W‖ 发光、更新按 RMS(ΔW) 发紫光；局部换成梯度 / ΔW 的热力
  updPanels(st, F, haveSt) {
    const X = this.X, k = st.k, s = st.step;
    const fx = F.fwd, bx = F.bwd;
    const gmax = haveSt ? X.tgradAll(k) : 1;
    const dmax = X.has('dw', k) ? X.dwRmsMax(k) : 0;
    // 每块面板在前向 / 反向的哪个算子、算子里的哪一段
    const fwin = (P) => {
      if (fx < 0) return 0;
      if (P.kind === 'E') return clamp01(fx / 0.8);
      if (P.kind === 'ET') return clamp01((fx - 13.2) / 0.5);
      if (P.kind === 'nf') return clamp01((fx - 13) / 0.2);
      const o = (P.kind === 'gate' || P.kind === 'up' || P.kind === 'down' || P.kind === 'ln2') ? 2 + 2 * P.l : 1 + 2 * P.l;
      const w = { ln1: [0, 0.15], q: [0.15, 0.5], k: [0.15, 0.5], v: [0.15, 0.5], qn: [0.45, 0.55], kn: [0.45, 0.55], o: [0.7, 0.92], ln2: [0, 0.15], gate: [0.15, 0.55], up: [0.15, 0.55], down: [0.55, 0.92] }[P.kind];
      return clamp01((fx - o - w[0]) / (w[1] - w[0]));
    };
    const bwin = (P) => {
      if (bx < 0) return 0;
      if (P.kind === 'E') return Math.max(clamp01((bx - 0.2) / 0.5), clamp01((bx - 13) / 0.8));
      if (P.kind === 'ET') return clamp01((bx - 0.2) / 0.5);
      if (P.kind === 'nf') return clamp01((bx - 0.6) / 0.4);
      const ffn = P.kind === 'gate' || P.kind === 'up' || P.kind === 'down' || P.kind === 'ln2';
      const o = ffn ? 1 + 2 * (NLAY - 1 - P.l) : 2 + 2 * (NLAY - 1 - P.l);
      const w = { down: [0.1, 0.45], gate: [0.4, 0.8], up: [0.4, 0.8], ln2: [0.75, 1], o: [0.1, 0.35], q: [0.45, 0.75], k: [0.45, 0.75], v: [0.45, 0.75], qn: [0.5, 0.7], kn: [0.5, 0.7], ln1: [0.75, 1] }[P.kind];
      return clamp01((bx - o - w[0]) / (w[1] - w[0]));
    };
    const scanList = [];
    const col = this.tmpCol || (this.tmpCol = new THREE.Color());
    const BASE = new THREE.Color(0x0c1830), STEEL = new THREE.Color(0x24406e), ROSE = new THREE.Color(0xc4486e), VIOL = new THREE.Color(0x8a72d8), CYAN = new THREE.Color(0x5ef0d4);
    for (const P of this.panels) {
      const kind = P.kind === 'ET' ? 'E' : P.kind;
      const fw = fwin(P), bw = bwin(P);
      // 梯度光：‖∇W‖ / 这一步最大的，开平方
      const gI = haveSt ? Math.sqrt(X.tgrad(k, P.l, kind) / gmax) : 0;
      const gOn = bw * (1 - Math.min(1, F.upd * 2));
      // 更新光：RMS(ΔW) / 这一步最大的（到下一个检查点），开平方
      const dI = dmax > 0 && X.dwRms(k, P.l, kind) != null ? Math.sqrt(X.dwRms(k, P.l, kind) / dmax) : 0;
      const uOn = Math.max(0, F.upd - F.done) * (P.crop >= 0 ? 0.6 : 1);
      const active = fw > 0 && fw < 1;
      col.copy(BASE);
      const wr = X.wRms(k, P.l, kind);
      if (wr != null) col.lerp(STEEL, kind === 'E' || P.kind === 'ET' || LY.MAT_KINDS.includes(kind) ? 0.5 * Math.pow(clamp01((wr - 0.018) / 0.12), 0.7) : 0.5 * clamp01(Math.abs(wr - 1) / 0.7));
      if (gOn > 0) col.lerp(ROSE, 0.42 * gI * gOn);
      if (uOn > 0) col.lerp(VIOL, 0.42 * dI * uOn);
      if (active) col.lerp(CYAN, 0.3);
      P.back.material.color.copy(col);
      P.edge.material.opacity = P.ghost ? (active || gOn > 0.3 ? 0.5 : 0.22) : active ? 0.95 : gOn > 0 && bw < 1 ? 0.9 : 0.42;
      P.edge.material.color.setHex(uOn > 0.05 ? 0xb39dff : gOn > 0 && bw < 1 ? 0xff6b93 : active ? 0xbff8ff : 0x5ef0d4);
      if (active && !P.flat && (P.kind === 'q' || P.kind === 'k' || P.kind === 'v' || P.kind === 'o' || P.kind === 'gate' || P.kind === 'up' || P.kind === 'down')) scanList.push([P, fw]);
      // 局部
      if (P.patch) {
        const cu = F.cu[P.crop];
        let mode = 'w', f = 0;
        if (cu.dw > 0 && cu.done < 1) { mode = 'd'; f = cu.dw; }
        else if (gOn > 0) { mode = 'g'; f = gOn; }
        this.drawPatch(P, k, { mode, f: Math.round(f * 10) / 10, done: cu.done });
      }
      P.gI = gI; P.dI = dI;
    }
    // 输出条
    for (let l = 0; l < NLAY; l++) {
      const R = this.layerP[l], O = this.outs[l];
      for (const kind of ['q', 'k', 'v', 'o', 'gate', 'up', 'down']) {
        const f = fwin(R[kind]), m = O[kind], w = m.userData.w * f;
        m.scale.x = Math.max(0.0001, w);
        m.position.x = m.userData.x0 + w / 2;
        m.material.opacity = f > 0 ? (fx >= 0 ? 0.55 : 0) : 0;
      }
    }
    // 扫描线
    this.scans.forEach((m, i) => {
      const e = scanList[i];
      m.material.opacity = e ? 0.5 : 0;
      if (!e) return;
      const [P, f] = e;
      m.position.set(P.x0 + P.w * f, P.y0 + P.h / 2, 0.02);
      m.scale.set(0.025, P.h + 0.04, 1);
    });
    // 注意力图案：前向经过时亮一下，平时淡
    for (let l = 0; l < NLAY; l++) {
      const o = 1 + 2 * l, a = fx >= o + 0.45 && fx < o + 0.75, b = bx >= 2 + 2 * (NLAY - 1 - l) + 0.3 && bx < 2 + 2 * (NLAY - 1 - l) + 0.5;
      for (const H of this.heads[l]) { H.m.material.opacity += (((a || b) ? 1 : X.has('ck', k) ? 0.78 : 0.15) - H.m.material.opacity) * 0.2; H.e.material.opacity = a ? 0.95 : b ? 0.9 : 0.4; H.e.material.color.setHex(b ? 0xff6b93 : 0x8fb8ff); }
    }
  }

  // 透镜屏：前向经过这个边界后亮起
  updLens(st, F) {
    const k = st.k;
    for (let b = 0; b <= NLAY; b++) {
      const end = b === 0 ? 1 : 3 + 2 * (b - 1);
      const on = F.fwd >= 0 ? clamp01((F.fwd - end + 0.3) / 0.3) : 0;
      this.drawLens(b, k, 'top');
      const ls = this.lens[b];
      ls.m.material.opacity = 0.1 + 0.9 * on;
      ls.ln.material.opacity = 0.08 + 0.3 * on;
      ls.e.material.opacity = on > 0 && on < 1 ? 0.9 : 0.25 + 0.15 * on;
    }
  }

  // 预测柱、顶上的字、−ln p、损失管、前 5 名
  updHead(st, F, haveSt) {
    const X = this.X, k = st.k, M = this.tmpM;
    const rise = haveSt ? easeOut(clamp01((F.fwd - 13.4) / 0.6)) : 0;
    const ca = this.pillars.instanceColor.array, ta = this.tops.instanceColor.array;
    for (let i = 0; i < NP; i++) {
      const p = haveSt ? X.rowP(k, i) : 0;
      const ok = haveSt && X.rowTop(k, i)[0].id === X.tgt(k, i);
      const dim = F.lossPos >= 0 && F.lossPos !== i ? 0.35 : 1;
      M.makeScale(1, Math.max(0.002, (0.02 + p * PH) * rise), 1).setPosition(fiberX(i), YP, fiberZ(i));
      this.pillars.setMatrixAt(i, M);
      seqInto(ca, i * 3, 0.25 + p * 0.75, 0.85 * dim);
      // 顶上的字：它最想写的那个字（绿 = 正好是答案，玫红 = 不是）
      const h = YP + (0.02 + p * PH) * rise + 0.075;
      M.makeTranslation(fiberX(i), h, fiberZ(i) + FS * 0.36);
      if (rise < 0.05) M.makeScale(0, 0, 0);
      this.tops.setMatrixAt(i, M);
      const c = ok ? LIN.green : LIN.rose, kk = (ok ? 1 : 0.55) * dim;
      ta[i * 3] = c[0] * kk; ta[i * 3 + 1] = c[1] * kk; ta[i * 3 + 2] = c[2] * kk;
    }
    this.pillars.instanceMatrix.needsUpdate = true;
    this.pillars.instanceColor.needsUpdate = true;
    this.tops.instanceMatrix.needsUpdate = true;
    this.tops.instanceColor.needsUpdate = true;
    // −ln p
    const la = this.lossBars.instanceColor.array;
    for (let i = 0; i < NP; i++) {
      const p = haveSt ? X.rowP(k, i) : 1, nl = -Math.log(Math.max(p, 1e-9)), f = clamp01(F.lossUpTo - i);
      M.makeScale(1, Math.max(0.002, Math.min(1, nl / LOSSB.lnMax) * LOSSB.hMax * f), 1).setPosition(lossX(i), YP, fiberZ(i));
      this.lossBars.setMatrixAt(i, M);
      const c = nl > 3 ? LIN.rose : nl > 1 ? LIN.amber : LIN.cyan, hl = F.lossPos === i ? 1.3 : F.lossPos >= 0 ? 0.4 : 0.85;
      la[i * 3] = c[0] * hl; la[i * 3 + 1] = c[1] * hl; la[i * 3 + 2] = c[2] * hl;
    }
    this.lossBars.instanceMatrix.needsUpdate = true;
    this.lossBars.instanceColor.needsUpdate = true;
    this.gauge.scale.y = Math.max(0.001, Math.min(1, X.loss(k) / GAUGE.max) * GAUGE.hMax * F.lossMean);
    // 前 5 名：正在看的位置（D4 损失的一个位置；或者悬停的柱子）
    const fp = F.lossPos >= 0 ? F.lossPos : this.hover && (this.hover.type === 'pillar' || this.hover.type === 'nll') ? this.hover.i : -1;
    const show5 = haveSt && fp >= 0 && rise > 0.5;
    const top = show5 ? X.rowTop(k, fp) : null;
    this.top5.forEach((m, j) => {
      m.visible = show5;
      const lb = this.lTop5[j];
      lb.visible = show5 && (st.view === 't-pos' || st.view === 't-op' || st.view === 't-step');
      if (!show5) return;
      const tp = top[j], z = embZ(tp.id), ok = tp.id === X.tgt(k, fp);
      m.position.set(EMB.x0 + 0.5 + (j - 2) * 0.09, YT, z);
      m.scale.y = Math.max(0.01, tp.p * 1.6);
      m.material.emissive.setHex(ok ? 0x8ee07a : 0x5ef0d4);
      lb.position.set(m.position.x, YT + tp.p * 1.6 + 0.05, z);
      const html = `${esc(X.ch(tp.id))} ${fmtP(tp.p)}<small>#${tp.id}</small>`;
      if (lb.el._h !== html) { lb.el.innerHTML = html; lb.el._h = html; }
      lb.el.classList.toggle('ok', ok);
    });
  }

  updPrinter(st, F) {
    const f = st.view === 't-run' ? easeOut(F.print) : 1;
    this.paper.scale.set(1, Math.max(0.001, f), 1);
    this.printLed.material.opacity = st.view === 't-run' && F.print < 1 ? 0.5 + 0.4 * Math.sin(this.E.clock * 12) : 0.45;
  }

  // 嵌入的地图（在相邻两个检查点之间插值）和地形上的小球
  updExhibits(st, F, haveCk) {
    const X = this.X, D = X.D, k = st.k;
    const kk = haveCk ? k : D.nearestCk(k);
    const vis = kk >= 0;
    this.pcaPts.visible = this.pcaFaces.visible = vis;
    if (vis) {
      const k2 = st.view === 't-run' && kk === k && k + 1 < X.K && D.has('ck', k + 1) ? k + 1 : kk;
      const f = k2 !== kk ? easeInOut(seg(st.p, 0.5, 1)) : 0;
      const pw = PCA.x1 - PCA.x0, ph = PCA.y1 - PCA.y0, sc = Math.min(pw, ph) / 5.4, cx = PCA.x0 + pw / 2, cy = PCA.y0 + ph / 2;
      const P = (n) => { const [ax, ay] = D.pca(kk, n), [bx, by] = D.pca(k2, n); return [cx + Math.max(-2.6, Math.min(2.6, lerp(ax, bx, f))) * sc, cy - Math.max(-2.6, Math.min(2.6, lerp(ay, by, f))) * sc]; };
      const pa = this.pcaPts.geometry.attributes.position.array;
      this.pcaDots.forEach((n, i) => { const [x, y] = P(n); pa[i * 3] = Math.max(PCA.x0 + 0.05, Math.min(PCA.x1 - 0.05, x)); pa[i * 3 + 1] = Math.max(PCA.y0 + 0.05, Math.min(PCA.y1 - 0.05, y)); pa[i * 3 + 2] = PCA.z; });
      this.pcaPts.geometry.attributes.position.needsUpdate = true;
      const M = this.tmpM;
      this.pcaLab.forEach((n, i) => { const [x, y] = P(n); M.makeTranslation(Math.max(PCA.x0 + 0.1, Math.min(PCA.x1 - 0.1, x)), Math.max(PCA.y0 + 0.1, Math.min(PCA.y1 - 0.1, y)), PCA.z + 0.01); this.pcaFaces.setMatrixAt(i, M); });
      this.pcaFaces.instanceMatrix.needsUpdate = true;
      this.pcaK = { kk, k2, f, sc, cx, cy };
    }
    // 地形：轨迹画到当前检查点（训练全程里在两个检查点之间插值）
    const f = st.view === 't-run' && k < X.K - 1 ? st.p : 0;
    const a = this.landPath[k], b = this.landPath[Math.min(X.K - 1, k + 1)];
    this.ball.position.copy(a).lerp(b, f);
    this.ball.position.y += 0.05;
    this.landLine.geometry.setDrawRange(0, k + 1);
  }

  // D4：放大的局部
  updMagnifier(st, F, haveCk, t) {
    const X = this.X, s = st.step, k = st.k;
    const on = st.view === 't-crop' && s.c != null;
    this.mag.visible = on;
    this.magLines.visible = on;
    for (const P of this.panels) if (P.patchEdge) P.patchEdge.material.opacity = on && P.crop === s.c ? 1 : 0.8;
    if (!on) { this.magC = -1; return; }
    const c = s.c, S = C48 * MAG.cell;
    // 放在源面板的前面：嵌入那块立在托盘右前方；矩阵的放在角上的正前方
    const src = this.cropSrc(c);
    const org = c === 0 ? new THREE.Vector3(-0.85, 2.85, 1.25) : new THREE.Vector3(src.x0 - S / 2 + 0.1, src.y1 + 0.4, 1.2);
    if (c === 2) org.set(src.x0 - S - 0.3, src.y1 + 0.35, 1.0);
    this.mag.position.copy(org);
    this.magOrg = org;
    this.magRowFaces.visible = c === 0;
    if (this.magC !== c) {
      this.magC = c;
      const la = this.magLines.geometry.attributes.position.array;
      const corners = c === 0
        ? [[src.x0, EMB.y, src.z0], [src.x1, EMB.y, src.z0], [src.x0, EMB.y, src.z1], [src.x1, EMB.y, src.z1]]
        : [[src.x0, src.y1, 0.01], [src.x1, src.y1, 0.01], [src.x0, src.y0, 0.01], [src.x1, src.y0, 0.01]];
      const mc = [[org.x, org.y, org.z], [org.x + S, org.y, org.z], [org.x, org.y - S, org.z], [org.x + S, org.y - S, org.z]];
      for (let j = 0; j < 4; j++) { la.set(corners[j], j * 6); la.set(mc[j], j * 6 + 3); }
      this.magLines.geometry.attributes.position.needsUpdate = true;
    }
    if (!haveCk) { this.cubes.count = 0; this.cubeGlow.count = 0; return; }
    this.cubes.count = this.cubeGlow.count = C48 * C48;
    const ma = this.cubes.instanceMatrix.array, cc = this.cubes.instanceColor.array, ga = this.cubeGlow.instanceMatrix.array, gc = this.cubeGlow.instanceColor.array;
    const sw = this.cropScale(k, c), sg = this.gScale(k, c), sd = this.dwScale(k, c);
    const ph = s.ph, cu = F.cu[c];
    // 前向：扫描线一列列扫过；反向：梯度光；更新：按 ΔW 顶出 / 沉下（一个最大的 ΔW 约 0.25），再落到新值
    const fwdF = ph === 'fwd' ? easeInOut(seg(this.p, 0.05, 0.85)) : -1;
    const gOn = ph === 'bwd' ? easeOut(seg(this.p, 0.1, 0.6)) : ph === 'loss' ? 0 : 0;
    const HMAX = 0.32;
    for (let r = 0; r < C48; r++) for (let q = 0; q < C48; q++) {
      const n = r * C48 + q, [i, j] = c === 0 ? [r, q] : [q, r];
      const w0 = X.wcrop(k, c, i, j), dw = X.dwcrop(k, c, i, j);
      const v = w0 + dw * cu.done;
      const h = 0.006 + HMAX * Math.pow(Math.min(1, Math.abs(v) / sw), 0.85);
      const popA = Math.max(0, cu.dw - cu.done), pop = sd > 0 ? (dw / sd) * 0.25 * popA : 0;
      const o = n * 16, x = (q + 0.5) * MAG.cell, y = -(r + 0.5) * MAG.cell;
      ma[o] = 1; ma[o + 1] = 0; ma[o + 2] = 0; ma[o + 4] = 0; ma[o + 5] = 1; ma[o + 6] = 0; ma[o + 8] = 0; ma[o + 9] = 0; ma[o + 10] = h; ma[o + 12] = x; ma[o + 13] = y; ma[o + 14] = pop; ma[o + 15] = 1;
      divInto(cc, n * 3, v / sw);
      let gr = 0, gg = 0, gb = 0;
      if (gOn > 0) {
        const gv = X.gcrop(k, c, i, j) / sg, I = Math.pow(Math.min(1, Math.abs(gv)), 1.3) * gOn * 0.55, col = gv >= 0 ? LIN.rose : LIN.violet;
        gr += col[0] * I; gg += col[1] * I; gb += col[2] * I;
      }
      if (popA > 0 && sd > 0) { const I = Math.min(1, Math.abs(dw) / sd) * popA * 0.45; gr += LIN.violet[0] * I; gg += LIN.violet[1] * I; gb += LIN.violet[2] * I; }
      if (fwdF >= 0 && Math.abs((q + 0.5) / C48 - fwdF) < 0.025) { gr += 0.08; gg += 0.22; gb += 0.24; }
      ga.set(ma.subarray(o, o + 16), o);
      ga[o + 10] = h + 0.01; ga[o + 14] = pop - 0.005;
      gc[n * 3] = gr; gc[n * 3 + 1] = gg; gc[n * 3 + 2] = gb;
    }
    this.cubes.instanceMatrix.needsUpdate = this.cubes.instanceColor.needsUpdate = true;
    this.cubeGlow.instanceMatrix.needsUpdate = this.cubeGlow.instanceColor.needsUpdate = true;
    this.magScan.material.opacity = fwdF >= 0 && fwdF < 1 ? 0.35 : 0;
    if (fwdF >= 0) { this.magScan.position.set(fwdF * S, -S / 2, HMAX + 0.02); this.magScan.scale.set(MAG.cell, S + 0.04, 1); }
  }

  // 局部在源面板上的范围
  cropSrc(c) {
    const s = C48 * U;
    if (c === 0) return { x0: EMB.x0, x1: EMB.x0 + s, z0: EMB.z0 - U, z1: EMB.z0 - 49 * U };
    const cr = CROPS[c], P = this.layerP[cr.l][cr.kind];
    const y1 = P.y0 + P.h;
    return { x0: P.x0, x1: P.x0 + s, y0: y1 - s, y1 };
  }

  // D5：跟踪的那个权重
  updMarks(st, t) {
    const X = this.X, s = st.step;
    const hv = this.hover;
    if (hv && hv.type === 'cube' && this.mag.visible) {
      const r = Math.floor(hv.n / C48), q = hv.n % C48, h = this.cubes.instanceMatrix.array[hv.n * 16 + 10], z = this.cubes.instanceMatrix.array[hv.n * 16 + 14];
      this.mHover.position.set(this.magOrg.x + (q + 0.5) * MAG.cell, this.magOrg.y - (r + 0.5) * MAG.cell, this.magOrg.z + z + h / 2);
      this.mHover.scale.set(MAG.cell * 1.05, MAG.cell * 1.05, h + 0.02);
      this.mHover.visible = true;
    } else this.mHover.visible = false;
    const on = st.view === 't-param' && s.f != null;
    this.featCube.visible = this.cross.visible = this.lFeat.visible = on;
    if (!on) return;
    const ft = X.feats[s.f], fp = LY.featPos(ft);
    this.featPosNow = fp;
    const sz = 0.026 + 0.006 * Math.sin(t * 5);
    this.featCube.position.set(fp.x, fp.y, fp.z + (fp.flat ? 0 : 0.01));
    this.featCube.scale.set(sz, fp.flat ? sz * 0.6 : sz, fp.flat ? sz : sz * 0.6);
    const ca = this.cross.geometry.attributes.position.array;
    if (fp.flat) {
      ca.set([EMB.x0, fp.y + 0.005, fp.z, EMB.x0 + 1, fp.y + 0.005, fp.z, fp.x, fp.y + 0.005, EMB.z0, fp.x, fp.y + 0.005, EMB.z0 - 0.6]);
    } else {
      const r = LY.panelRect(fp.l, fp.kind);
      ca.set([r[0], fp.y, 0.015, r[1], fp.y, 0.015, fp.x, r[2], 0.015, fp.x, r[3], 0.015]);
    }
    this.cross.geometry.attributes.position.needsUpdate = true;
    this.lFeat.position.set(fp.x, fp.y + (fp.flat ? 0.05 : 0.03), fp.z + 0.02);
    const a = X.has('st', st.k) ? X.adam(st.k, s.f) : null;
    let html = '';
    if (a) {
      const f4 = (v) => (v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e4 ? Number(v.toPrecision(4)).toString() : sciSup(v, 3));
      if (s.ph === 'fwd') html = `w = <b>${f4(a.w0)}</b>`;
      else if (s.ph === 'bwd') html = `∂L/∂w = <b>${f4(a.gRaw)}</b>`;
      else html = s.mi === 'write' || s.mi === 'dw' ? `w ${f4(a.w0)} → <b>${f4(a.w1)}</b>` : `w = ${f4(a.w0)}`;
    }
    if (this.lFeat.el._h !== html) { this.lFeat.el.innerHTML = html; this.lFeat.el._h = html; }
  }

  updLabels(st, F) {
    const X = this.X, v = st.view, s = st.step, k = st.k;
    const far = v === 't-run' || v === 't-step' && (s.ph === 'upd' || s.ph === 'batch');
    // 当前算子用到的面板
    const opPanels = new Set();
    if ((s.ph === 'fwd' || s.ph === 'bwd') && s.sub) {
      if (s.sub === 'emb') opPanels.add(this.embP);
      else if (s.sub === 'head') { opPanels.add(this.ghostP); opPanels.add(this.nfP); }
      else {
        const R = this.layerP[s.L];
        for (const kind of s.sub === 'attn' ? ['ln1', 'q', 'k', 'v', 'o'] : ['ln2', 'gate', 'up', 'down']) opPanels.add(R[kind]);
      }
    }
    if (v === 't-step' && (s.ph === 'fwd' || s.ph === 'bwd')) {
      // 一步之内：镜头跟着的那一层，标出它的主要面板
      const x = s.ph === 'fwd' ? F.fwd : F.bwd;
      const o = Math.min(13, Math.max(0, Math.floor(x)));
      const op = (s.ph === 'fwd' ? FWD : BWD)[o];
      if (op.L != null && !this.small) for (const kind of ['q', 'o', 'gate', 'down']) opPanels.add(this.layerP[op.L][kind]);
    }
    for (const [P, o] of this.lPanel) {
      let show = opPanels.has(P);
      if (v === 't-crop') show = P.crop === s.c && P.kind !== 'ET';
      if (v === 't-param') show = false;
      if (v === 't-run' && !this.small) show = P.kind === 'E' || P.kind === 'ET' || (P.l === 0 && (P.kind === 'q' || P.kind === 'gate' || P.kind === 'down'));
      if (v === 't-step' && s.ph === 'batch') show = P.kind === 'E';
      if (s.ph === 'upd' && v === 't-op' && s.sub === 'adam') show = P.crop >= 0 && P.kind !== 'ET';
      o.visible = !!show;
      const html = far && !opPanels.has(P) ? o.short : o.full;
      if (o.el._h !== html) { o.el.innerHTML = html; o.el._h = html; }
      o.el.classList.toggle('on', opPanels.has(P) && v !== 't-run');
    }
    const showLayers = v === 't-run' || v === 't-step';
    this.lLayer.forEach((o, l) => { o.visible = showLayers || ((s.ph === 'fwd' || s.ph === 'bwd') && s.L === l && v === 't-op'); });
    const lensOn = v === 't-step' || v === 't-op';
    this.lLens.forEach((o, b) => { o.visible = lensOn && !this.small && (s.ph !== 'fwd' && s.ph !== 'bwd' ? false : true) && this.lensNear(b, s, F); });
    this.lHeads.forEach((o, l) => { o.visible = v === 't-op' && s.sub === 'attn' && s.L === l; });
    this.lBundle.visible = (v === 't-step' || v === 't-op') && s.ph === 'batch' && s.sub !== 'row';
    this.lTray.visible = (v === 't-step' || v === 't-op') && s.ph === 'batch';
    this.lStack.visible = (v === 't-step' || v === 't-op') && s.ph === 'batch' && !this.small;
    this.lCorpus.visible = v === 't-run' || ((v === 't-step' || v === 't-op') && s.ph === 'batch' && (s.sub === 'row' || !s.sub));
    this.lPillars.visible = (v === 't-step' || v === 't-op') && (s.ph === 'loss' || s.sub === 'head');
    this.lLoss.visible = (v === 't-step' || v === 't-op') && s.ph === 'loss';
    const lossOn = F.lossMean > 0.05 && v !== 't-crop' && v !== 't-param';
    this.lGauge.visible = !!lossOn;
    if (lossOn) {
      const html = `L = ${X.loss(k).toFixed(3)}<small>${t('t3.gaugeSub')}</small>`;
      if (this.lGauge.el._h !== html) { this.lGauge.el.innerHTML = html; this.lGauge.el._h = html; }
      this.lGauge.position.y = GAUGE.y + Math.min(1, X.loss(k) / GAUGE.max) * GAUGE.hMax * F.lossMean + 0.1;
    }
    const clipOn = s.ph === 'upd' && (s.sub === 'clip' || (!s.sub && F.clip < 1)) && v !== 't-param';
    this.lClip.visible = !!clipOn;
    if (clipOn) {
      const c = X.clip(k), html = c < 1 ? t('t3.clipYes', { g: X.gnorm(k).toFixed(3), c: c.toFixed(3) }) : t('t3.clipNo', { g: X.gnorm(k).toFixed(3) });
      if (this.lClip.el._h !== html) { this.lClip.el.innerHTML = html; this.lClip.el._h = html; }
    }
    const gap = k > 0 ? X.step(k) - X.step(k - 1) : 1;
    this.lGap.visible = (v === 't-run' || (v === 't-step' && s.ph === 'batch')) && gap > 1 && !this.small;
    if (this.lGap.visible) {
      const html = t('t3.gap', { t: X.step(k) + 1, g: gap });
      if (this.lGap.el._h !== html) { this.lGap.el.innerHTML = html; this.lGap.el._h = html; }
    }
    this.lPrinter.visible = v === 't-run' && !this.small;
    this.lPca.visible = this.lLand.visible = v === 't-run' && !this.small;
    // 更新那一段：局部按“到下一个检查点”的 ΔW 变色——写清楚隔了几步
    const updNote = s.ph === 'upd' && (v === 't-step' || v === 't-op') && s.sub !== 'clip' || (v === 't-run' && F.upd > 0 && F.done < 1);
    this.lUpdNote.visible = !!updNote && !this.small;
    if (this.lUpdNote.visible) {
      const sp = X.dwSpan(k), html = sp ? (sp[1] - sp[0] === 1 ? t('t3.dwOne', { a: sp[0] + 1, b: sp[1] + 1 }) : t('t3.dwMany', { a: sp[0] + 1, b: sp[1] + 1, n: sp[1] - sp[0] })) : X.has('dw', k) ? t('t3.dwLast') : '';
      if (this.lUpdNote.el._h !== html) { this.lUpdNote.el.innerHTML = html; this.lUpdNote.el._h = html; }
      this.lUpdNote.position.set(v === 't-run' ? BCX : 3.2, v === 't-run' ? YP + PH + 0.8 : YP + PH + 0.8, BCZ);
    }
    // 放大的局部的标题
    this.lMag.visible = v === 't-crop' && s.c != null;
    if (this.lMag.visible) {
      const c = s.c, mode = s.ph === 'bwd' ? 'g' : s.ph === 'upd' ? 'd' : 'w';
      const html = `${t(`t3.crop${c}`)}<small>${t('t3.cropZoom')} · ${t(`t3.mode_${mode}`)}</small>`;
      if (this.lMag.el._h !== html) { this.lMag.el.innerHTML = html; this.lMag.el._h = html; }
      this.lMag.position.set(this.magOrg.x, this.magOrg.y + 0.08, this.magOrg.z);
    }
  }
  lensNear(b, s, F) {
    if (s.sub === 'emb') return b === 0;
    if (s.sub === 'head') return b === NLAY;
    if (s.L != null) return b === s.L || b === s.L + 1;
    const x = s.ph === 'fwd' ? F.fwd : F.bwd;
    const o = Math.min(13, Math.max(0, Math.floor(x)));
    const l = o === 0 ? 0 : o === 13 ? NLAY : Math.floor((o - 1) / 2) + (s.ph === 'fwd' ? 1 : 0);
    return Math.abs(b - l) <= 1;
  }

  /* ================================================================ 悬停提示（真实数值） */

  tip(info) {
    const X = this.X, st = this.st;
    if (!info || !st) return null;
    const k = st.k, haveSt = X.has('st', k), haveCk = X.has('ck', k);
    const step = X.step(k) + 1;
    const posCtx = (i) => { let s = ''; for (let j = Math.max(0, i - 7); j <= i; j++) s += X.ch(X.row(k, j)); return esc(s); };
    switch (info.type) {
      case 'panel': {
        const P = info.P, kind = P.kind === 'ET' ? 'E' : P.kind;
        const nm = P.kind === 'ET' ? 'Eᵀ' : KIND_LABEL[P.kind];
        const where = P.l >= 0 ? t('t3.layer', { l: P.l }) + ' · ' : '';
        const sh = P.kind === 'ET' ? '256 × 7478' : kindShape(kind).join(' × ');
        const n = kindCount(kind).toLocaleString('en-US');
        const g = haveSt ? X.tgrad(k, P.l, kind) : null, wr = X.wRms(k, P.l, kind), dr = X.dwRms(k, P.l, kind);
        let h = `<span class="k">${where}${nm} · [${sh}]</span>${t('t3.tipCount', { n })}`;
        if (P.kind === 'ET') h += `<br>${t('t3.tipTied')}`;
        if (g != null) h += `<br>‖∇W‖ <span class="a">${sciSup(g, 3)}</span>`;
        if (wr != null) h += `　RMS(W) <span class="v">${wr.toPrecision(3)}</span>`;
        if (dr != null) { const sp = X.dwSpan(k); h += `<br>RMS(ΔW) ${t('t3.tipDwTo', { t: sp[1] + 1 })} <span class="v">${dr.toPrecision(3)}</span>`; }
        h += `<br><span style="color:var(--dim)">${P.crop >= 0 && P.kind !== 'ET' ? t('t3.tipCrop') : t('t3.tipStats')}</span>`;
        return h;
      }
      case 'patch': case 'cube': {
        if (!haveCk) return null;
        let c, r, q;
        if (info.type === 'patch') { c = info.P.crop; q = Math.min(47, Math.floor(info.uv.x * 48)); r = Math.min(47, Math.floor((info.P.flat ? info.uv.y : 1 - info.uv.y) * 48)); }
        else { c = this.magC; r = Math.floor(info.n / C48); q = info.n % C48; }
        const [i, j] = c === 0 ? [r, q] : [q, r];
        const w = X.wcrop(k, c, i, j), g = X.gcrop(k, c, i, j), dw = X.has('dw', k) && X.hasDw(k) ? X.dwcrop(k, c, i, j) : null;
        const where = c === 0 ? t('t3.cellE', { ch: esc(X.ch(i + 1)), id: i + 1, d: j }) : t('t3.cellW', { in: r, out: q, name: c === 1 ? t('t3.crop1') : t('t3.crop2'), i, j });
        const sp = X.dwSpan(k);
        return `<span class="k">${where}</span>w <span class="v">${w.toPrecision(4)}</span>　∇ <span class="a">${sciSup(g, 2)}</span>${dw != null ? `<br>ΔW ${t('t3.tipDwTo', { t: sp[1] + 1 })} <span class="v">${sciSup(dw, 3)}</span>` : ''}<br><span style="color:var(--dim)">${t('t3.tipInt8')}</span>`;
      }
      case 'lens': {
        if (!haveSt) return null;
        const q = Math.min(15, Math.floor(info.uv.x * 16)), r = Math.min(3, Math.floor((1 - info.uv.y) * 4)), i = r * NC + q, b = info.b;
        const id = X.lensTop(k, b, i), tg = X.tgt(k, i);
        return `<span class="k">${b === 0 ? t('t3.lens0') : t('t3.lensL', { l: b - 1 })} · ${t('t3.pos', { i })}</span>${t('t3.tipCtx', { s: posCtx(i) })}<br>${t('t3.tipLens', { c: esc(X.ch(id)) })}${id === tg ? ' ✓' : ''}<br>${t('t3.tipAns', { c: esc(X.ch(tg)) })} <span class="v">${fmtP(X.lensP(k, b, i))}</span>`;
      }
      case 'pillar': case 'nll': {
        if (!haveSt) return null;
        const i = info.i, p = X.rowP(k, i), tg = X.tgt(k, i);
        const top = X.rowTop(k, i).map((tp) => `<tr><td class="${tp.id === tg ? 'tg' : ''}">${esc(X.ch(tp.id))}</td><td>${fmtP(tp.p)}</td></tr>`).join('');
        return `<span class="k">${t('t3.pos', { i })} · ${t('t3.tipCtx', { s: posCtx(i) })}</span>${t('t3.tipAns', { c: esc(X.ch(tg)) })} <span class="v">${fmtP(p)}</span>　−ln p = <span class="a">${(-Math.log(Math.max(p, 1e-12))).toFixed(3)}</span><table>${top}</table><span style="color:var(--dim)">${t('t3.tipClickPos')}</span>`;
      }
      case 'tile': case 'tgt': {
        const i = info.i, id = info.type === 'tile' ? X.row(k, i) : X.tgt(k, i);
        return `<span class="k">${info.type === 'tile' ? t('t3.tipTile', { i }) : t('t3.tipTgt', { i })}</span>「${esc(X.ch(id))}」 #${id}`;
      }
      case 'stack': return `<span class="k">${t('t3.stack')}</span>${t('t3.tipStack')}`;
      case 'corpus': return `<span class="k">${t('t3.corpus')}</span>${t('t3.tipCorpus')}`;
      case 'gauge': return `<span class="k">${t('t3.tipGaugeK', { t: step })}</span>${t('t3.tipGauge')} <span class="v">${X.loss(k).toFixed(4)}</span><br>${t('t3.tipRowLoss')} ${X.rowLoss(k).toFixed(4)}<br>${t('t3.tipBlind', { v: X.vocab, l: Math.log(X.vocab).toFixed(3) })}`;
      case 'head': {
        if (!haveCk) return null;
        const chars = '⏎' + X.held.text;
        const i = Math.min(24, Math.floor((1 - info.uv.y) * 25)), j = Math.min(24, Math.floor(info.uv.x * 25));
        const a = X.attn(k, info.l, info.h, i, j);
        return `<span class="k">${t('t3.tipHead', { l: info.l, h: info.h, kv: info.h >> 1 })}</span>${j > i ? t('t3.tipMask') : `「${esc(chars[i])}」→「${esc(chars[j])}」 <span class="v">${fmtP(a)}</span>`}<br><span style="color:var(--dim)">${t('t3.tipHeadNote')}</span>`;
      }
      case 'printer': return `<span class="k">${t('t3.printer')}</span>${t('t3.tipPrinter', { t: step, temp: X.meta.train.sampleTemp })}`;
      case 'pca': {
        const K = this.pcaK;
        if (!K) return null;
        let best = -1, bd = 1e9;
        for (let n = 0; n < this.pcaN; n++) {
          const [ax, ay] = X.D.pca(K.kk, n), [bx, by] = X.D.pca(K.k2, n);
          const x = K.cx + lerp(ax, bx, K.f) * K.sc, y = K.cy - lerp(ay, by, K.f) * K.sc;
          const d = Math.hypot(x - info.p.x, y - info.p.y);
          if (d < bd) { bd = d; best = n; }
        }
        if (best < 0 || bd > 0.25) return `<span class="k">${t('t3.pca')}</span>${t('t3.tipPca')}`;
        return `<span class="k">${t('t3.pca')}</span><b>${esc(this.pcaChars[best])}</b> · ${t('t3.tipPcaRank', { n: X.meta.pcaIds[best] })}`;
      }
      case 'land': {
        const Ld = X.meta.land, G = Ld.G, lw = LAND.x1 - LAND.x0, ld = LAND.z1 - LAND.z0;
        const ix = Math.max(0, Math.min(G - 1, Math.round(((info.p.x - LAND.x0) / lw) * (G - 1)))), iy = Math.max(0, Math.min(G - 1, Math.round(((LAND.z1 - info.p.z) / ld) * (G - 1))));
        return `<span class="k">${t('t3.land')}</span>${t('t3.tipLand', { v: X.D.land(iy, ix).toFixed(3), t: X.step(k) + 1, p: Ld.pathLoss[k].toFixed(3) })}`;
      }
    }
    return null;
  }

  /* ================================================================ 镜头 */

  fitBox(b, dir, margin = 1.06, minD = 2) {
    const E = this.E;
    const insetL = this.insetL || 0;
    const key = `${b.join(',')}|${dir.x},${dir.y},${dir.z}|${margin}|${E.w}x${E.h}|${E.insetR | 0}|${E.padB | 0}|${E.padT | 0}|${insetL}`;
    if (this.fitKey === key) return { pos: this.fitRes.pos.clone(), look: this.fitRes.look.clone() };
    const cam = this.fitCam || (this.fitCam = new THREE.PerspectiveCamera());
    cam.copy(E.camera);
    cam.updateProjectionMatrix();
    const [x0, x1, y0, y1, z0 = -0.7, z1 = 0.6] = b;
    const pts = [];
    for (const x of [x0, x1]) for (const y of [y0, y1]) for (const z of [z0, z1]) pts.push(new THREE.Vector3(x, y, z));
    const w = E.w || 1, h = E.h || 1;
    const rx0 = -1 + (2 * insetL) / w, rx1 = 1 - (2 * (E.insetR || 0)) / w, ry0 = -1 + (2 * E.padB) / h, ry1 = 1 - (2 * E.padT) / h;
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
      let lo = minD, hi = 600;
      for (let j = 0; j < 26; j++) {
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
    const WIDE = V3(0.55, 0.3, 1), MID = V3(0.4, 0.26, 1), NEAR = V3(0.22, 0.14, 1), TOP = V3(0.2, 0.9, 1), LOW = V3(0.3, 0.55, 1);
    const machine = [MACHINE.x0, MACHINE.x1, MACHINE.y0, MACHINE.y1, -1.4, 1.2];
    const all = [ALL.x0, ALL.x1, ALL.y0, ALL.y1, -1.6, 5.8];
    if (v === 't-run') return frame(this.small ? machine : all, WIDE, 1.02);
    const batchR = [CORPUS.x1 - 2.4, 1.6, -0.5, 1.4, -1.2, 1.6];
    const lossR = [BUNDLE.x0 - 0.2, GAUGE.x + 1.4, YT - 0.2, YP + PH + 0.6, -1.0, 1.0];
    if (v === 't-step') {
      if (s.ph === 'batch') return frame(batchR, LOW, 1.06);
      if (s.ph === 'loss') return frame(lossR, TOP, 1.06);
      if (s.ph === 'upd') return frame(machine, WIDE, 1.02);
      const x = s.ph === 'fwd' ? F.fwd : F.bwd;
      const o = Math.min(13, Math.max(0, x));
      const yc = s.ph === 'fwd' ? lerp(0.6, YP, o / 13.5) : lerp(YP, 0.6, o / 13.5);
      return frame([MACHINE.x0, MACHINE.x1, yc - 4.2, yc + 4.2, -0.8, 0.8], MID, 1.02);
    }
    if (v === 't-op') {
      if (s.ph === 'batch') return frame(s.sub === 'row' ? batchR : [BUNDLE.x0 - 0.3, BUNDLE.x1 + 0.3, -0.5, 0.7, BUNDLE.z0 - 0.3, BUNDLE.z1 + TGT_DZ + 0.3], s.sub === 'row' ? LOW : TOP, 1.06);
      if (s.ph === 'loss') return frame(lossR, TOP, 1.06);
      if (s.ph === 'upd') return frame(machine, WIDE, 1.02);
      return frame(LY.opRect(s), MID, 1.04);
    }
    if (v === 't-pos') {
      const i = s.i;
      return frame([fiberX(i) - 1.2, Math.max(EMB.x0 + 1.4, lossX(i) + 0.4), YT - 0.1, YP + PH + 0.3, fiberZ(i) - 1.6, fiberZ(i) + 0.6], TOP, 1.08, 1.2);
    }
    if (v === 't-crop' && this.magOrg) {
      const S = C48 * MAG.cell, o = this.magOrg, src = this.cropSrc(s.c);
      if (s.c === 0) return frame([o.x - 0.3, Math.max(o.x + S, src.x1) + 0.2, -0.1, o.y + 0.3, -0.6, o.z + 0.5], NEAR, 1.04, 1);
      return frame([Math.min(o.x, src.x0) - 0.3, Math.max(o.x + S, src.x1) + 0.3, Math.min(o.y - S, src.y0) - 0.2, Math.max(o.y, src.y1) + 0.3, -0.2, o.z + 0.5], NEAR, 1.04, 1);
    }
    if (v === 't-param' && s.f != null) {
      const fp = LY.featPos(this.X.feats[s.f]);
      if (fp.flat) return frame([fp.x - 0.8, fp.x + 0.8, fp.y - 0.1, fp.y + 0.6, fp.z - 0.7, fp.z + 0.5], V3(0.15, 0.7, 1), 1.04, 0.6);
      return frame([fp.x - 0.9, fp.x + 0.9, fp.y - 0.65, fp.y + 0.65, -0.1, 0.3], V3(0.18, 0.12, 1), 1.04, 0.6);
    }
    return frame(machine, WIDE, 1.02);
  }
}

const HB_MID = LY.HB / 2;
function mix3(a, b, f) { return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]; }
