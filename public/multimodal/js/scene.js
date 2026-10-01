// 多模态的 3D 舞台：照片 → 图块 → 视觉编码器（24 级特征网格叠成的塔）→ 合并器 → 插进词元序列 → 语言模型 28 层 → 输出。
// 塔上每一格的颜色是那一级真实特征的主成分；语言模型层板上的热力图是生成当前这个字时真实的注意力。
// 一切都由 update(state) 根据当前调试步骤算出来，所以暂停、单步、回退都能正确显示。
import { THREE, label, textTexture, easeOut, easeInOut, seg } from '../../js/stage/engine.js';
import { tokPlain, shortSpecial, esc, fmtPct } from '../../js/ui.js';
import { drawHeat } from './paint.js';
import { argmax } from './explain.js';

const PS = 0.1;            // 一个 16×16 图块在场景里的边长
const LVL = 0.16;          // ViT 相邻两级的间距
const VX = -3.7;           // ViT 塔的 x
const Y0 = 0.32;           // ViT 第 0 级的高度
const PX = -7.6;           // 照片的 x
const PY = 1.7;            // 照片中心高度
const S = 0.27;            // 语言模型里相邻词元的间距
const LY0 = 0.6, LG = 0.16; // LLM 第 0 层高度、层距
const X0 = -1.3;           // 词元序列起点
const MY = Y0 + 24 * LVL + 0.55; // 合并器高度

export const ROLE = { system: 0x8fa6d6, user: 0x5ee4f0, assistant: 0xffb65c, tpl: 0xb39dff, img: 0x9fb8ff };
const col = (hex) => new THREE.Color(hex);
const tmpM = new THREE.Matrix4();
const tmpQ = new THREE.Quaternion();
const tmpS = new THREE.Vector3();
const tmpP = new THREE.Vector3();
const FLAT = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0));
const UPRIGHT = new THREE.Quaternion();

function disposeTree(o) {
  o.traverse((c) => {
    if (c.geometry && !c.geometry.userData?.shared) c.geometry.dispose();
    if (c.material) (Array.isArray(c.material) ? c.material : [c.material]).forEach((m) => { if (!m.userData?.shared) { if (m.map && !m.map.userData?.shared) m.map.dispose(); m.dispose(); } });
    if (c.el) c.el.remove();
  });
}

const GEO = {
  cell: new THREE.BoxGeometry(PS * 0.86, 0.014, PS * 0.86),
  mcell: new THREE.BoxGeometry(PS * 1.72, 0.03, PS * 1.72),
  tile: new THREE.BoxGeometry(0.22, 0.07, 0.22),
  face: new THREE.PlaneGeometry(0.2, 0.2),
  plane: new THREE.PlaneGeometry(1, 1),
};
Object.values(GEO).forEach((g) => { g.userData.shared = true; });

function srgb(r, g, b, k = 1) { return new THREE.Color().setRGB((r / 255) * k, (g / 255) * k, (b / 255) * k, THREE.SRGBColorSpace); }

export class Scene {
  constructor(E) {
    this.E = E;
    this.scene = E.scene;
    this.root = null;
    this.buildStatic();
  }

  buildStatic() {
    const c = document.createElement('canvas');
    c.width = c.height = 256;
    const g = c.getContext('2d');
    const gr = g.createRadialGradient(128, 128, 0, 128, 128, 128);
    gr.addColorStop(0, 'rgba(30,64,92,0.32)');
    gr.addColorStop(0.4, 'rgba(14,28,50,0.25)');
    gr.addColorStop(1, 'rgba(5,11,23,0)');
    g.fillStyle = gr;
    g.fillRect(0, 0, 256, 256);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    const floor = new THREE.Mesh(new THREE.CircleGeometry(36, 48), new THREE.MeshBasicMaterial({ map: t, transparent: true, depthWrite: false }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(2, -0.1, 0);
    floor.userData.noFocus = true;
    this.scene.add(floor);
    const grid = new THREE.GridHelper(70, 70, 0x1a2e48, 0x0f1d33);
    grid.position.set(2, -0.09, 0);
    grid.material.transparent = true;
    grid.material.opacity = 0.3;
    this.scene.add(grid);
    const n = 500, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { pos[i * 3] = (Math.random() - 0.5) * 60; pos[i * 3 + 1] = Math.random() * 14 - 1; pos[i * 3 + 2] = (Math.random() - 0.5) * 30; }
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.dust = new THREE.Points(pg, new THREE.PointsMaterial({ color: 0x6fc4d8, size: 0.04, transparent: true, opacity: 0.25, depthWrite: false }));
    this.scene.add(this.dust);
  }

  /* ------------------------------------------------------------ 为一条回复搭场景 */

  load(Q) {
    if (this.root) { disposeTree(this.root); this.scene.remove(this.root); }
    this.E.pickables.length = 0;
    this.Q = Q;
    const V = (this.V = Q.V);
    const root = (this.root = new THREE.Group());
    this.scene.add(root);
    this.key = {};
    const { gh, gw, mh, mw, Np, Nv } = V;
    this.pw = gw * PS; this.ph = gh * PS;

    // 照片：模型真正看到的像素，由 Np 个图块拼成（每块一个实例，可以拆开、飞走）
    const tex = new THREE.Texture(V.img);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    this.imgTex = tex;
    const uvr = new Float32Array(Np * 4);
    for (let i = 0; i < Np; i++) {
      const r = Math.floor(i / gw), c = i % gw;
      uvr.set([c / gw, 1 - (r + 1) / gh, 1 / gw, 1 / gh], i * 4);
    }
    const tg = new THREE.PlaneGeometry(PS, PS);
    tg.setAttribute('uvr', new THREE.InstancedBufferAttribute(uvr, 4));
    this.photoMat = new THREE.ShaderMaterial({
      uniforms: { map: { value: tex }, bright: { value: 0.82 }, opacity: { value: 1 } },
      vertexShader: 'attribute vec4 uvr; varying vec2 vUv; void main(){ vUv = uvr.xy + uv * uvr.zw; gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0); }',
      fragmentShader: 'uniform sampler2D map; uniform float bright; uniform float opacity; varying vec2 vUv; void main(){ vec4 c = texture2D(map, vUv); gl_FragColor = vec4(c.rgb * bright, opacity); }',
      transparent: true,
      side: THREE.DoubleSide,
    });
    this.photo = new THREE.InstancedMesh(tg, this.photoMat, Np);
    this.photo.frustumCulled = false;
    this.photo.userData.pick = { type: 'photo' };
    this.E.pickables.push(this.photo);
    root.add(this.photo);
    this.photoFrame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(this.pw + 0.04, this.ph + 0.04)), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.45 }));
    root.add(this.photoFrame);
    this.photoLbl = label('', 'lbl part');
    this.photoLbl.center.set(0.5, 1.2);
    root.add(this.photoLbl);

    // 视觉编码器：25 级（第 0 级 = 图块嵌入，之后每个块一级），每级 gh×gw 个格子，颜色 = 那一级特征的主成分
    this.levels = [];
    for (let k = 0; k <= 24; k++) {
      const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9, depthWrite: false });
      const m = new THREE.InstancedMesh(GEO.cell, mat, Np);
      const cols = V.pca(k);
      for (let i = 0; i < Np; i++) {
        const r = Math.floor(i / gw), c = i % gw;
        tmpM.makeTranslation(VX + (c - (gw - 1) / 2) * PS, Y0 + k * LVL, (r - (gh - 1) / 2) * PS);
        m.setMatrixAt(i, tmpM);
        m.setColorAt(i, srgb(cols[i * 3], cols[i * 3 + 1], cols[i * 3 + 2]));
      }
      m.instanceColor.needsUpdate = true;
      m.frustumCulled = false;
      m.userData.pick = (hit) => ({ type: 'vlevel', k, i: hit.instanceId, click: true });
      this.E.pickables.push(m);
      root.add(m);
      this.levels.push(m);
    }
    const shaftGeo = new THREE.BoxGeometry(this.pw + 0.12, 24 * LVL + 0.1, this.ph + 0.12);
    this.shaft = new THREE.LineSegments(new THREE.EdgesGeometry(shaftGeo), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.16 }));
    this.shaft.position.set(VX, Y0 + 12 * LVL, 0);
    root.add(this.shaft);
    this.vitLbl = label(`视觉编码器 ViT<small>${Q.manifest.model.vision.depth} 层 · ${Q.manifest.model.vision.hidden} 维</small>`, 'lbl part');
    this.vitLbl.position.set(VX, Y0 + 24 * LVL + 0.15, -this.ph / 2);
    this.vitLbl.center.set(0.5, 1.6);
    root.add(this.vitLbl);
    this.lvlLbl = label('', 'lbl lens');
    this.lvlLbl.center.set(0, 0.5);
    root.add(this.lvlLbl);
    // 查询格子的标记环
    this.ring = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(PS * 2, 0.03, PS * 2)), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0 }));
    root.add(this.ring);

    // 合并器输出：mh×mw 个视觉词元
    this.mpos = [];
    const mcols = V.mergePca;
    this.merged = new THREE.InstancedMesh(GEO.mcell, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95, depthWrite: false }), Nv);
    for (let m = 0; m < Nv; m++) {
      const r = Math.floor(m / mw), c = m % mw;
      this.mpos.push(new THREE.Vector3(VX + (c - (mw - 1) / 2) * PS * 2, MY, (r - (mh - 1) / 2) * PS * 2));
      this.merged.setColorAt(m, srgb(mcols[m * 3], mcols[m * 3 + 1], mcols[m * 3 + 2]));
    }
    this.merged.instanceColor.needsUpdate = true;
    this.merged.frustumCulled = false;
    root.add(this.merged);
    this.mergeLbl = label(`合并器<small>2×2 → 1 · ${Q.manifest.model.vision.out} 维</small>`, 'lbl part');
    this.mergeLbl.position.set(VX + this.pw / 2 + 0.1, MY, 0);
    this.mergeLbl.center.set(0, 0.5);
    root.add(this.mergeLbl);

    // 词元序列：文字是一行小方块，图片是插在中间的 mh×mw 方阵（M-RoPE：二维位置嵌在一维序列里）
    const T = Q.T;
    this.tx = new Float32Array(T);
    this.tz = new Float32Array(T);
    let x = X0;
    for (let i = 0; i < T; i++) {
      if (Q.isImg(i)) {
        const m = i - V.vs, r = Math.floor(m / mw), c = m % mw;
        if (m === 0) this.bx = x;
        this.tx[i] = this.bx + c * S;
        this.tz[i] = (r - (mh - 1) / 2) * S;
        if (m === Nv - 1) x = this.bx + mw * S;
      } else { this.tx[i] = x; this.tz[i] = 0; x += S; }
    }
    this.xEnd = x - S;
    this.bw = mw * S; this.bd = mh * S;
    this.bcx = this.bx + (mw - 1) * S / 2;
    this.tiles = [];
    this.roleMat = {};
    for (const [r, c] of Object.entries(ROLE)) this.roleMat[r] = new THREE.MeshStandardMaterial({ color: col(c).multiplyScalar(0.2), emissive: col(c), emissiveIntensity: 0.08, metalness: 0.25, roughness: 0.45 });
    for (let i = 0; i < T; i++) {
      if (Q.isImg(i)) { this.tiles.push(null); continue; }
      const t = Q.tokens[i];
      const grp = new THREE.Group();
      const body = new THREE.Mesh(GEO.tile, this.roleMat[t.role] || this.roleMat.tpl);
      grp.add(body);
      const txt = t.sp ? shortSpecial(t.s) : tokPlain(t.s);
      const face = new THREE.Mesh(GEO.face, new THREE.MeshBasicMaterial({ map: textTexture(txt, { color: '#' + col(ROLE[t.role] || ROLE.tpl).getHexString(), font: t.sp ? '600 40px "JetBrains Mono",monospace' : undefined, w: 160, h: 160 }), transparent: true }));
      face.material.map.userData = { shared: true };
      face.rotation.x = -Math.PI / 2;
      face.position.y = 0.037;
      grp.add(face);
      grp.position.set(this.tx[i], 0.12, this.tz[i]);
      body.userData.pick = { type: 'tile', i };
      this.E.pickables.push(body);
      root.add(grp);
      this.tiles.push(grp);
    }
    // 序列里的图片：一块平放的图（就是模型看到的那张），上面是 mh×mw 个视觉词元格子
    const trayImg = new THREE.Mesh(GEO.plane, new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0, depthWrite: false, color: 0x9a9a9a }));
    trayImg.scale.set(this.bw, this.bd, 1);
    trayImg.quaternion.copy(FLAT);
    trayImg.position.set(this.bcx, 0.1, 0);
    trayImg.userData.noFocus = true;
    root.add(trayImg);
    this.trayImg = trayImg;
    this.vtok = new THREE.InstancedMesh(GEO.mcell, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9, depthWrite: false }), Nv);
    for (let m = 0; m < Nv; m++) this.vtok.setColorAt(m, srgb(mcols[m * 3], mcols[m * 3 + 1], mcols[m * 3 + 2]));
    this.vtok.instanceColor.needsUpdate = true;
    this.vtok.frustumCulled = false;
    this.vtok.userData.pick = (hit) => ({ type: 'vcell', m: hit.instanceId });
    this.E.pickables.push(this.vtok);
    root.add(this.vtok);
    this.vtokKey = '';
    const trayW = this.xEnd - X0 + 0.8;
    this.cx = (X0 + this.xEnd) / 2;
    const tray = new THREE.Mesh(new THREE.BoxGeometry(trayW, 0.05, this.bd + 0.6), new THREE.MeshStandardMaterial({ color: 0x0c1629, metalness: 0.6, roughness: 0.4 }));
    tray.position.set(this.cx, 0.04, 0);
    root.add(tray);
    const trayEdge = new THREE.LineSegments(new THREE.EdgesGeometry(tray.geometry), new THREE.LineBasicMaterial({ color: 0x2c6b8a, transparent: true, opacity: 0.7 }));
    trayEdge.position.copy(tray.position);
    root.add(trayEdge);

    // 语言模型的 28 层玻璃层板
    this.slabs = [];
    const slabGeo = new THREE.BoxGeometry(trayW - 0.2, 0.02, this.bd + 0.4);
    const slabEdge = new THREE.EdgesGeometry(slabGeo);
    for (let L = 0; L < Q.NL; L++) {
      const mat = new THREE.MeshStandardMaterial({ color: 0x6d8cff, emissive: 0x5ef0d4, emissiveIntensity: 0, transparent: true, opacity: 0.06, roughness: 0.2, metalness: 0.2, depthWrite: false });
      const s = new THREE.Mesh(slabGeo, mat);
      const e = new THREE.LineSegments(slabEdge, new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.16 }));
      s.add(e);
      s.edge = e;
      s.position.set(this.cx, LY0 + L * LG, 0);
      s.userData.pick = { type: 'slab', L, click: true };
      this.E.pickables.push(s);
      root.add(s);
      this.slabs.push(s);
    }
    this.llmLbl = label(`语言模型<small>${Q.NL} 层 · ${Q.M.hidden} 维</small>`, 'lbl part');
    this.llmLbl.position.set(this.cx, LY0 + Q.NL * LG + 0.1, -this.bd / 2 - 0.2);
    this.llmLbl.center.set(0.5, 1.4);
    root.add(this.llmLbl);
    this.layerLbl = label('', 'lbl lens');
    this.layerLbl.center.set(1, 0.5);
    root.add(this.layerLbl);

    // 层板上的热力图：生成当前这个字时，这一层对每个视觉词元的注意力（真实数据）
    this.heatCv = document.createElement('canvas');
    this.heatCv.width = mw * 24; this.heatCv.height = mh * 24;
    this.heatTex = new THREE.CanvasTexture(this.heatCv);
    this.heatTex.colorSpace = THREE.SRGBColorSpace;
    this.heat = new THREE.Mesh(GEO.plane, new THREE.MeshBasicMaterial({ map: this.heatTex, transparent: true, opacity: 0, depthWrite: false }));
    this.heat.scale.set(this.bw, this.bd, 1);
    this.heat.quaternion.copy(FLAT);
    this.heat.userData.noFocus = true;
    root.add(this.heat);
    this.heatKey = '';
    // 从当前词元射向图片 / 文字的光束
    const MAXB = 24;
    this.beamGeo = new THREE.BufferGeometry();
    this.beamGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAXB * 6), 3));
    this.beamGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(MAXB * 6), 3));
    this.beams = new THREE.LineSegments(this.beamGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false }));
    this.beams.frustumCulled = false;
    root.add(this.beams);
    this.beamKey = '';
    // 当前词元的光柱
    this.column = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 1, 8, 1, true), new THREE.MeshBasicMaterial({ color: 0xffb65c, transparent: true, opacity: 0.4, depthWrite: false }));
    root.add(this.column);
    this.pulse = new THREE.Mesh(new THREE.TorusGeometry(0.07, 0.012, 8, 24), new THREE.MeshBasicMaterial({ color: 0xffd9a6, transparent: true, opacity: 0.9 }));
    this.pulse.rotation.x = Math.PI / 2;
    root.add(this.pulse);
    // 输出：选中的词元浮在塔顶
    this.outLbl = label('', 'lbl big');
    this.outLbl.center.set(0.5, 1.2);
    root.add(this.outLbl);

    // DeepStack：ViT 第 5 / 11 / 17 层旁边伸出三条管道，接到语言模型前三层的图片位置
    this.tubes = [];
    Q.manifest.model.vision.deepstack.forEach((vl, d) => {
      const a = new THREE.Vector3(VX + this.pw / 2 + 0.05, Y0 + (vl + 1) * LVL, 0.15 * (d - 1));
      const b = new THREE.Vector3(this.bx - 0.12, LY0 + d * LG, 0.15 * (d - 1));
      const mid = a.clone().lerp(b, 0.5);
      mid.y = Math.max(a.y, b.y) + 0.35;
      const curve = new THREE.CatmullRomCurve3([a, new THREE.Vector3(a.x + 0.5, a.y + 0.1, a.z), mid, new THREE.Vector3(b.x - 0.5, b.y + 0.15, b.z), b]);
      const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 48, 0.014, 6), new THREE.MeshBasicMaterial({ color: 0xb39dff, transparent: true, opacity: 0.22, depthWrite: false }));
      tube.userData.noFocus = true;
      root.add(tube);
      this.tubes.push(tube);
    });
    this.dsLbl = label('DeepStack', 'lbl hint');
    this.dsLbl.position.set((VX + this.bx) / 2 + 0.6, LY0 + 1.6, 0);
    root.add(this.dsLbl);

    // 黑箱外壳（深度 1）：前面板左边开着一个“取景窗”，照片从这里塞进去；打开时前面板绕底边倒下
    const x0 = PX - this.pw / 2 - 0.6, x1 = this.xEnd + 0.7;
    const y1 = Math.max(MY, LY0 + Q.NL * LG) + 0.5;
    const z0 = Math.max(this.bd, this.ph) / 2 + 0.6;
    this.box = { x0, x1, y1, z0, cx: (x0 + x1) / 2 };
    this.buildCasing(root, Q);
    this.lastTiles = -1;
  }

  buildCasing(root, Q) {
    const { x0, x1, y1, z0, cx } = this.box;
    const W = x1 - x0, H = y1 + 0.2, D = z0 * 2;
    const g = (this.casing = new THREE.Group());
    root.add(g);
    const mat = () => new THREE.MeshStandardMaterial({ color: 0x08101f, metalness: 0.7, roughness: 0.24, transparent: true, opacity: 0.95 });
    const edge = () => new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0.8 });
    const panel = (w, h, d) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat());
      m.add(new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), edge()));
      return m;
    };
    this.cPanels = [];
    const back = panel(W, H, 0.06); back.position.set(cx, H / 2 - 0.1, -z0); g.add(back);
    const top = panel(W, 0.06, D); top.position.set(cx, H - 0.1, 0); g.add(top);
    const left = panel(0.06, H, D); left.position.set(x0, H / 2 - 0.1, 0); g.add(left);
    const right = panel(0.06, H, D); right.position.set(x1, H / 2 - 0.1, 0); g.add(right);
    this.cTop = top;
    this.cPanels.push(back, top, left, right);
    // 前面板：绕底边的铰链
    this.cPivot = new THREE.Group();
    this.cPivot.position.set(cx, -0.1, z0);
    g.add(this.cPivot);
    const front = panel(W, H, 0.06);
    front.position.set(0, H / 2, 0);
    this.cPivot.add(front);
    this.cFront = front;
    this.cPanels.push(front);
    // 取景窗：照片从这里进去
    const winW = this.pw + 0.3, winH = this.ph + 0.3;
    const win = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(winW, winH)), new THREE.LineBasicMaterial({ color: 0xffb65c, transparent: true, opacity: 0.7 }));
    win.position.set(PX - cx, PY + 0.1, 0.04);
    front.add(win);
    const winBg = new THREE.Mesh(new THREE.PlaneGeometry(winW, winH), new THREE.MeshBasicMaterial({ color: 0x02050c, transparent: true, opacity: 0.85 }));
    winBg.position.set(PX - cx, PY + 0.1, 0.035);
    front.add(winBg);
    this.cWin = win;
    const winLbl = new THREE.Mesh(new THREE.PlaneGeometry(winW, winW * 0.12), new THREE.MeshBasicMaterial({ map: textTexture('图片从这里进去', { color: '#c9a36b', font: '500 56px "PingFang SC","Microsoft YaHei",sans-serif', w: 900, h: 108 }), transparent: true }));
    winLbl.position.set(PX - cx, PY + 0.1 - winH / 2 - 0.16, 0.035);
    front.add(winLbl);
    const lw = Math.min(W * 0.3, 6);
    const logo = new THREE.Mesh(new THREE.PlaneGeometry(lw, lw * 0.22), new THREE.MeshBasicMaterial({ map: textTexture('Qwen3-VL-2B', { color: '#5ef0d4', font: '700 110px "JetBrains Mono",monospace', w: 1200, h: 264 }), transparent: true }));
    logo.position.set(x0 + W * 0.36 - cx, H * 0.58, 0.035);
    front.add(logo);
    const m = Q.manifest.model;
    const sub = new THREE.Mesh(new THREE.PlaneGeometry(lw, lw * 0.07), new THREE.MeshBasicMaterial({ map: textTexture(`视觉 ${m.vision.depth} 层 + 语言 ${m.text.layers} 层 · ${(m.params.total / 1e8).toFixed(1)} 亿参数`, { color: '#7a859e', font: '500 60px "PingFang SC","Microsoft YaHei",sans-serif', w: 1400, h: 98 }), transparent: true }));
    sub.position.set(logo.position.x, H * 0.4, 0.035);
    front.add(sub);
    this.cLight = new THREE.Mesh(new THREE.CircleGeometry(0.16, 32), new THREE.MeshBasicMaterial({ color: 0x5ef0d4 }));
    this.cLight.position.set(logo.position.x, H * 0.24, 0.035);
    front.add(this.cLight);
    this.cAll = [];
    g.traverse((o) => { if (o.material) this.cAll.push({ m: o.material, op: o.material.opacity }); });
  }

  updateCasing(open, st, t) {
    const e = easeInOut(open);
    this.cPivot.rotation.x = e * Math.PI * 0.5;
    this.cTop.position.y = this.box.y1 + 0.1 + e * 1.2;
    const fade = 1 - Math.min(1, open * 1.4);
    this.casing.visible = fade > 0.01;
    for (const { m, op } of this.cAll) m.opacity = op * fade;
    // 指示灯：在“想”的时候闪
    const busy = st.step.ph === 'pass' || st.step.ph === 'see';
    this.cLight.material.color.setHex(busy && Math.sin(t * 9) > 0 ? 0xffb65c : 0x5ef0d4);
    this.cWin.material.opacity = 0.7 * fade * (st.step.ph === 'see' ? 0.6 + 0.4 * Math.sin(t * 6) : 0.6);
  }

  /* ------------------------------------------------------------ 每帧 */

  update(st, dt, t) {
    const Q = this.Q, V = this.V, s = st.step, view = st.view, p = st.p;
    const d = st.dAnim;
    const open = Math.min(1, Math.max(0, d - 1));            // 0 = 黑箱，1 = 打开
    const g = st.g, row = Q.row(g);
    this.dust.rotation.y += dt * 0.004;

    // 外壳
    this.updateCasing(open, st, t);

    // 视觉阶段的进度（0..1）：决定照片、塔、合并器、序列里图片的状态
    const vis = this.visState(st);

    // 照片的图块
    this.updatePhoto(vis, st, t);

    // ViT 各级
    for (let k = 0; k <= 24; k++) {
      const m = this.levels[k];
      let a = 0, b = 0.32;
      if (vis.vit >= 0) {
        const reached = vis.vit * 24 >= k - 0.001;
        a = reached ? 1 : 0;
        if (k === vis.hl) { b = 0.95; }
        else if (vis.hl >= 0) b = 0.26;
        else b = 0.5;
      }
      if (vis.embed >= 0 && k === 0) a = Math.max(a, vis.embed > 0.8 ? 1 : 0);
      m.visible = a > 0 && open > 0.02;
      m.material.opacity = 0.85 * a * Math.max(0.15, open);
      m.material.color.setScalar(b);
    }
    this.shaft.material.opacity = 0.16 * open;
    // 当前 ViT 级的标签
    if (vis.hl >= 0 && open > 0.5) {
      const L = vis.hl - 1;
      const k = vis.hl;
      const dist = L >= 0 ? V.vitDist[L].reduce((a, b) => a + b, 0) / 16 : 0;
      const html = k === 0 ? `<span class="l">嵌入</span>图块嵌入 + 位置` : `<span class="l">L${String(L).padStart(2, '0')}</span>注意距离 <span class="p">${dist.toFixed(1)}</span>`;
      if (this.lvlLbl.el.innerHTML !== html) this.lvlLbl.el.innerHTML = html;
      this.lvlLbl.position.set(VX + this.pw / 2 + 0.12, Y0 + k * LVL, this.ph / 2);
      this.lvlLbl.el.classList.remove('hide');
    } else this.lvlLbl.el.classList.add('hide');
    // 查询格子（ViT 注意力从这里出发）
    const showRing = (view === 'vitop' && s.op === 'attn') || view === 'vitlayer';
    if (showRing && vis.hl >= 0) {
      const q = st.vq ?? V.micro.token;
      const r = Math.floor(q / V.mw), c = q % V.mw;
      this.ring.position.set(VX + (c * 2 + 0.5 - (V.gw - 1) / 2) * PS, Y0 + vis.hl * LVL + 0.02, (r * 2 + 0.5 - (V.gh - 1) / 2) * PS);
      this.ring.material.opacity = 0.9;
    } else this.ring.material.opacity = 0;

    // 合并器
    const mergeA = vis.merge;
    this.merged.visible = mergeA > 0 && open > 0.02;
    this.merged.material.opacity = 0.92 * Math.max(0, mergeA) * open;
    this.updateMerged(vis);
    this.mergeLbl.el.classList.toggle('hide', !(open > 0.5 && mergeA > 0.2));
    this.vitLbl.el.classList.toggle('hide', open < 0.5);

    // 序列里的视觉词元
    this.updateVtok(vis, open);
    this.trayImg.material.opacity = 0.55 * vis.land * open;

    // 文字词元：没生成的先藏起来
    const shown = vis.text ? row : -1;
    if (shown !== this.lastTiles) {
      this.lastTiles = shown;
      this.tiles.forEach((tl, i) => { if (tl) tl.visible = i <= shown; });
    }

    // 语言模型
    const lit = this.llmState(st);
    for (let L = 0; L < Q.NL; L++) {
      const sl = this.slabs[L];
      const on = L === lit.L;
      const passed = lit.upTo >= L;
      // 还在看图的时候，语言模型那边先暗着；逐层看的时候，当前层上面的层板几乎透明，免得挡住热力图
      const dimV = lit.upTo < 0 ? 0.22 : st.depth >= 3 && lit.L >= 0 && L > lit.L ? 0.18 : 1;
      sl.material.opacity = (on ? 0.16 : passed ? 0.07 : 0.04) * open * dimV;
      sl.material.emissiveIntensity = on ? 0.3 : passed ? 0.05 : 0;
      sl.edge.material.opacity = (on ? 0.6 : 0.14) * open * dimV;
      sl.visible = open > 0.02;
    }
    this.llmLbl.el.classList.toggle('hide', open < 0.5);
    // 光柱 + 光环
    const qx = this.tx[row], top = LY0 + (Q.NL - 1) * LG;
    const colOn = vis.text && open > 0.1 && lit.upTo >= 0;
    this.column.visible = colOn;
    this.pulse.visible = colOn;
    if (colOn) {
      const h = Math.max(0.05, LY0 + lit.upTo * LG - 0.1);
      this.column.scale.y = h;
      this.column.position.set(qx, 0.12 + h / 2, 0);
      this.pulse.position.set(qx, LY0 + Math.max(0, lit.y) * LG, 0);
    }
    // 热力图与光束
    this.updateHeat(st, lit, open);
    // 层标签
    if (lit.L >= 0 && open > 0.5) {
      const lens = Q.lensAt(g, lit.L)[0];
      const html = `<span class="l">L${String(lit.L).padStart(2, '0')}</span>看图 <span class="p">${fmtPct(Q.mass(g, lit.L))}</span>　透镜 <b>${esc(tokPlain(lens[0]))}</b>`;
      if (this.layerLbl.el.innerHTML !== html) this.layerLbl.el.innerHTML = html;
      this.layerLbl.position.set(this.bx - 0.25, LY0 + lit.L * LG, this.bd / 2);
      this.layerLbl.el.classList.remove('hide');
    } else this.layerLbl.el.classList.add('hide');
    // 输出
    const outOn = (s.ph === 'head' && open > 0.5) || (s.ph === 'pass' && p > 0.55);
    if (outOn) {
      const st2 = Q.steps[g];
      const html = `${esc(tokPlain(st2.chosenS))}<small style="margin-left:6px;color:var(--amber)">${fmtPct(st2.p)}</small>`;
      if (this.outLbl.el.innerHTML !== html) this.outLbl.el.innerHTML = html;
      if (open < 0.5) this.outLbl.position.set(this.box.x0 + (this.box.x1 - this.box.x0) * 0.45, this.box.y1 + 0.2 + 0.6 * easeOut((p - 0.55) / 0.45), this.box.z0);
      else this.outLbl.position.set(this.tx[Math.min(row + 1, Q.T - 1)] ?? qx + S, top + 0.35 + 0.25 * easeOut(s.ph === 'head' ? p : 1), 0);
    }
    this.outLbl.el.classList.toggle('hide', !outOn);
    // DeepStack 管道
    const dsOn = (s.ph === 'merge' && s.sub === 'deep') || (s.ph === 'layer' && s.op === 'deep');
    this.tubes.forEach((tb, k) => {
      const hot = dsOn && (s.ph === 'merge' || s.L === k);
      tb.material.opacity = (hot ? 0.75 : vis.merge > 0 || vis.land > 0 ? 0.2 : 0.05) * open;
    });
    this.dsLbl.el.classList.toggle('hide', !(open > 0.5 && (dsOn || view === 'vit' || view === 'merge')));
    // 照片标签
    const pl = `图片<small>${V.spec.orig[0]}×${V.spec.orig[1]} → ${V.gw * 16}×${V.gh * 16} · ${V.gh}×${V.gw} 块</small>`;
    if (this.photoLbl.el.innerHTML !== pl) this.photoLbl.el.innerHTML = pl;
    this.photoLbl.el.classList.toggle('hide', open < 0.5 || vis.photo < 1 || vis.fly > 0);
  }

  // 根据当前步骤，算出视觉流水线各部分的状态
  visState(st) {
    const s = st.step, p = st.p, g = st.g;
    const r = { photo: 1, split: 0, fly: 0, embed: -1, vit: -1, hl: -1, merge: 0, group: 0, land: 0, text: true };
    if (g > 0) { r.photo = 0.5; r.vit = 1; r.merge = 1; r.land = 1; return r; }
    switch (s.ph) {
      case 'see':
        r.photo = 1; r.text = false; r.enter = p; return r;
      case 'pass': case 'llm': case 'layer': case 'head': case 'read':
        r.photo = 0.5; r.vit = 1; r.merge = 1; r.land = 1; return r;
      case 'prep':
        r.text = false;
        if (!s.sub) { r.split = easeInOut(p); return r; }
        if (s.sub === 'resize') { r.resize = p; return r; }
        r.split = s.sub === 'patch' ? easeInOut(Math.min(1, p * 1.3)) : 1;
        return r;
      case 'vit':
        r.text = false; r.split = 1;
        if (s.sub === 'embed') { r.fly = s.mi ? 1 : easeInOut(p); r.embed = r.fly; r.hl = r.fly > 0.8 ? 0 : -1; if (s.mi) r.micro = true; return r; }
        if (s.sub === 'pos') { r.fly = 1; r.embed = 1; r.hl = 0; return r; }
        r.fly = 1; r.embed = 1;
        if (s.L === undefined) { r.vit = p; r.hl = Math.min(24, Math.floor(p * 25)); return r; }
        r.vit = (s.L + (s.op && s.op !== 'add2' ? 0 : 1)) / 24;
        r.hl = s.op === 'ln1' || s.op === 'attn' || s.op === 'add1' || s.op === 'ln2' || s.op === 'mlp' ? s.L : s.L + 1;
        return r;
      case 'merge':
        r.text = false; r.split = 1; r.fly = 1; r.embed = 1; r.vit = 1;
        if (s.sub === 'group' || !s.sub) { r.group = easeInOut(p); r.merge = r.group; r.hl = 24; return r; }
        r.merge = 1; r.group = 1; return r;
      case 'splice':
        r.split = 1; r.fly = 1; r.embed = 1; r.vit = 1; r.merge = 1; r.group = 1;
        if (s.sub === 'mrope') { r.land = 1; return r; }
        r.land = easeInOut(p); r.text = true; return r;
    }
    return r;
  }

  updatePhoto(vis, st, t) {
    const V = this.V, { gh, gw, Np } = V;
    const k = [vis.split, vis.fly, vis.enter ?? 1, vis.resize ?? 1, st.dAnim > 1.5 ? 1 : 0, vis.micro ? Math.min(1, st.p * 3) : 0].map((x) => x.toFixed(3)).join(',');
    const hot = vis.micro ? V.micro.row * gw + V.micro.col : -1;   // 微观视图里被放大的那个图块
    this.photoMat.uniforms.bright.value = 0.82 * (vis.photo >= 1 ? 1 : 0.55);
    const enter = vis.enter ?? 1;
    const ez = (1 - easeInOut(enter)) * (this.box.z0 + 2.6);   // 从外壳前面推进取景窗
    this.photoFrame.position.set(PX, PY, ez);
    this.photoFrame.material.opacity = 0.45 * (1 - vis.fly) * (st.dAnim < 1.5 || vis.photo > 0 ? 1 : 0);
    if (k === this.photoKey) return;
    this.photoKey = k;
    const split = vis.split, fly = vis.fly;
    const gap = 1 + 0.18 * split * (1 - fly);
    for (let i = 0; i < Np; i++) {
      const r = Math.floor(i / gw), c = i % gw;
      // 照片上的位置（竖着）
      tmpP.set(PX + (c - (gw - 1) / 2) * PS * gap, PY - (r - (gh - 1) / 2) * PS * gap, ez + split * (1 - fly) * 0.05 * Math.sin(r * 1.7 + c * 2.3));
      if (fly > 0) {
        // 飞到 ViT 第 0 级对应的格子（平放）
        const tgt = new THREE.Vector3(VX + (c - (gw - 1) / 2) * PS, Y0 - 0.012, (r - (gh - 1) / 2) * PS);
        const delay = ((r * gw + c) / Np) * 0.35;
        const f = easeInOut(Math.min(1, Math.max(0, (fly - delay) / 0.65)));
        tmpP.lerp(tgt, f);
        tmpP.y += Math.sin(f * Math.PI) * 0.8;
        tmpQ.copy(UPRIGHT).slerp(FLAT, f);
        const sc = 1 - 0.14 * f;
        tmpS.set(sc, sc, sc);
      } else { tmpQ.copy(UPRIGHT); tmpS.set(1, 1, 1); }
      if (i === hot) {
        // 从塔底的格子里升起来，转向镜头，放大
        const u = st.step.mi === 'pick' ? easeOut(Math.min(1, st.p * 2)) : 1;
        tmpP.y += 0.9 * u;
        tmpP.z += 0.6 * u;
        tmpQ.slerp(UPRIGHT, u);
        tmpS.setScalar(1 + 4 * u);
      }
      tmpM.compose(tmpP, tmpQ, tmpS);
      this.photo.setMatrixAt(i, tmpM);
    }
    this.photo.instanceMatrix.needsUpdate = true;
    // 已经飞进塔里之后，照片原处留一张淡淡的副本
    this.photo.visible = true;
  }

  updateMerged(vis) {
    const V = this.V, k = vis.group.toFixed(3) + (vis.merge > 0 ? 1 : 0);
    if (k === this.mergedKey) return;
    this.mergedKey = k;
    const top = Y0 + 24 * LVL;
    for (let m = 0; m < V.Nv; m++) {
      const pp = this.mpos[m];
      tmpP.set(pp.x, top + (pp.y - top) * easeOut(vis.group), pp.z);
      const sc = 0.4 + 0.6 * vis.group;
      tmpS.set(sc, 1, sc);
      tmpM.compose(tmpP, UPRIGHT, tmpS);
      this.merged.setMatrixAt(m, tmpM);
    }
    this.merged.instanceMatrix.needsUpdate = true;
  }

  updateVtok(vis, open) {
    const V = this.V, Q = this.Q;
    const land = vis.land;
    this.vtok.visible = land > 0.001 && open > 0.02;
    this.vtok.material.opacity = (land < 1 ? 0.9 : 0.35) * open;
    const k = land.toFixed(3);
    if (k === this.vtokKey) return;
    this.vtokKey = k;
    for (let m = 0; m < V.Nv; m++) {
      const a = this.mpos[m];
      const i = V.vs + m;
      const delay = (m / V.Nv) * 0.3;
      const f = easeInOut(Math.min(1, Math.max(0, (land - delay) / 0.7)));
      tmpP.set(a.x + (this.tx[i] - a.x) * f, a.y + (0.13 - a.y) * f + Math.sin(f * Math.PI) * 1.2, a.z + (this.tz[i] - a.z) * f);
      const sc = 1 + 0.25 * f;
      tmpS.set(sc, f > 0.98 ? 0.5 : 1, sc);
      tmpM.compose(tmpP, UPRIGHT, tmpS);
      this.vtok.setMatrixAt(m, tmpM);
    }
    this.vtok.instanceMatrix.needsUpdate = true;
  }

  // 语言模型此刻点亮到哪一层
  llmState(st) {
    const s = st.step, p = st.p, NL = this.Q.NL;
    if (s.ph === 'layer') return { L: s.L, upTo: s.L, y: s.L };
    if (s.ph === 'llm' || s.ph === 'pass') { const y = Math.min(NL - 1, p * NL); return { L: s.ph === 'llm' ? Math.floor(y) : -1, upTo: Math.floor(y), y }; }
    if (s.ph === 'head') return { L: -1, upTo: NL - 1, y: NL - 1 };
    return { L: -1, upTo: -1, y: 0 };
  }

  updateHeat(st, lit, open) {
    const Q = this.Q, V = this.V, g = st.g, s = st.step;
    const show = lit.L >= 0 && open > 0.3;
    this.heat.material.opacity = show ? 0.9 * open : 0;
    this.heat.visible = show;
    this.beams.visible = show;
    if (!show) return;
    const L = lit.L;
    this.heat.position.set(this.bcx, LY0 + L * LG + 0.015, 0);
    const key = `${Q.id}|${g}|${L}`;
    if (key !== this.heatKey) {
      this.heatKey = key;
      const a = Q.attImg(g, L);
      const hg = this.heatCv.getContext('2d');
      drawHeat(hg, V.img, this.heatCv.width, this.heatCv.height, a, V.mh, V.mw, { gamma: 0.85, dim: 0.32 });
      this.heatTex.needsUpdate = true;
      // 光束：图片里最受关注的 10 个词元 + 文字里最受关注的 4 个位置
      const pos = this.beamGeo.attributes.position.array, colA = this.beamGeo.attributes.color.array;
      pos.fill(0); colA.fill(0);
      const row = Q.row(g), y = LY0 + L * LG + 0.02;
      const order = [...a.keys()].sort((x, z) => a[z] - a[x]).slice(0, 10);
      const mx = a[order[0]] || 1;
      let n = 0;
      const push = (x1, z1, w, c) => {
        pos.set([this.tx[row], y, 0, x1, y, z1], n * 6);
        const k = Math.min(1, w);
        colA.set([c[0] * k, c[1] * k, c[2] * k, c[0] * k * 0.5, c[1] * k * 0.5, c[2] * k * 0.5], n * 6);
        n++;
      };
      for (const m of order) push(this.tx[V.vs + m], this.tz[V.vs + m], (a[m] / mx) * 0.9, [1, 0.71, 0.36]);
      for (const { j, w } of Q.txt(g, L)) if (!Q.isImg(j)) push(this.tx[j], this.tz[j], Math.sqrt(w) * 0.9, [0.37, 0.94, 0.83]);
      this.beamGeo.attributes.position.needsUpdate = true;
      this.beamGeo.attributes.color.needsUpdate = true;
      this.beamGeo.setDrawRange(0, n * 2);
    }
  }

  /* ------------------------------------------------------------ 相机 */

  camera(st) {
    const E = this.E, V = this.V, Q = this.Q, s = st.step, view = st.view;
    const v3 = (x, y, z) => new THREE.Vector3(x, y, z);
    const row = Q.row(st.g);
    const fromFront = (look, w, h, elev = 0.35, side = 0.0, margin = 1.15) => {
      const dist = E.fitDistance(w, h, margin);
      const dir = v3(side, elev, 1).normalize();
      return { pos: look.clone().addScaledVector(dir, dist), look };
    };
    switch (view) {
      case 'box': {
        const b = this.box;
        const w = (b.x1 - b.x0) * 0.6;
        return fromFront(v3(b.x0 + w / 2, b.y1 / 2, b.z0), w, b.y1 + 1.4, 0.4, s.ph === 'see' ? -0.22 : 0.1, 1.05);
      }
      case 'image': return fromFront(v3(PX, PY, 0), this.pw + 0.6, this.ph + 0.8, 0.08, 0, 1.25);
      case 'embed':
        return fromFront(v3((PX + VX) / 2, 1.0, 0.2), Math.abs(PX - VX) + this.pw + 1.2, 3.6, 0.45, 0.05);
      case 'conv': case 'micro': return fromFront(v3(VX, 0.9, 0.4), this.pw + 1.6, 2.6, 0.5, 0.15);
      case 'vit': return fromFront(v3(VX, Y0 + 12 * LVL, 0), this.pw + 2.4, 24 * LVL + 1.6, 0.42, 0.45);
      case 'vitlayer': case 'vitop': {
        const k = s.op === 'ln1' || s.op === 'attn' || s.op === 'add1' || s.op === 'ln2' || s.op === 'mlp' ? s.L : s.L + 1;
        return fromFront(v3(VX, Y0 + k * LVL, 0), this.pw + 1.4, this.ph + 0.6, 0.95, 0.35, 1.1);
      }
      case 'merge': return fromFront(v3(VX, MY - 0.2, 0), this.pw + 1.8, 1.8, 0.9, 0.3);
      case 'splice':
        if (s.sub === 'mrope') return fromFront(v3(this.bcx + 0.6, 0.15, 0), this.bw + 3.2, this.bd + 0.8, 1.25, 0.15, 1.1);
        return fromFront(v3((VX + this.bcx) / 2 + 0.6, 2.0, 0), this.bcx - VX + this.bw + 2.5, MY + 0.4, 0.35, 0.05);
      case 'tray': return fromFront(v3(this.tx[row] - 1.5, 0.3, 0), 6.5, 2.0, 0.55, 0.05);
      case 'tower': return fromFront(v3(this.cx, LY0 + 13 * LG, 0), this.xEnd - X0 + 1.4, Q.NL * LG + 1.4, 0.3, 0.12, 1.06);
      case 'layer': case 'layerop': case 'heads': {
        const y = LY0 + s.L * LG;
        const x1 = this.bx - 0.3, x2 = this.tx[row] + 0.6;
        return fromFront(v3((x1 + x2) / 2, y, 0), x2 - x1 + 0.8, this.bd + 0.8, 0.85, 0.08, 1.05);
      }
      case 'head': return fromFront(v3(this.tx[row] - 1.0, LY0 + (Q.NL - 1) * LG, 0), 6, 3, 0.35, 0.1);
    }
    return fromFront(v3(this.cx, 2, 0), 20, 6);
  }
}
