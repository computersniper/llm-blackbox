// 矩阵乘法的显微镜：y[j] = Σ_i x[i]·W[i, j]。
// 在真实形状的权重面板上高亮第 j 列；乘法这一步里，贡献最大的 12 项轮流出场：
// 输入的第 i 个数（蓝）沿第 i 行走到格子 (i, j)（紫），乘出来的积（琥珀）顺着这一列飞进输出向量的第 j 格（青）。
// 具体数字写在算式板（board.js）上，这里只负责“在哪里算、结果去了哪里”。数字全部来自真实模型。
import { THREE, label, easeOut, easeInOut, seg } from './engine.js';
import { esc, tokPlain } from '../ui.js';
import { sums, neuronMM, ropePos } from './fields.js';

const U = 0.7 / 1024;
// 四种角色的颜色，算式板和讲解面板用同一套（app.css 里的 --c-x / --c-w / --c-p / --c-y）
export const ROLE = { x: 0x7cc4ff, w: 0xc9a2ff, p: 0xffb65c, y: 0x5ef0d4 };
const C = Object.fromEntries(Object.entries(ROLE).map(([k, v]) => [k, new THREE.Color(v)]));

// 乘法这一步里 12 项轮流出场：第 k 项在 p ∈ [k/n, (k+1)/n)·0.94 这一段里走完（留一点尾巴给最后一项）
export function termAt(p, k, n) { return seg(p / 0.94, k / n, (k + 1) / n); }
// 已经乘完（飞进累加器）的项数
export function termsDone(p, n) { let c = 0; for (let k = 0; k < n; k++) if (termAt(p, k, n) >= 0.62) c++; return c; }

const sub = (n) => String(n).split('').map((c) => '₀₁₂₃₄₅₆₇₈₉'[c] || c).join('');

export class Micro {
  constructor(E, M) {
    this.E = E;
    this.M = M;
    this.root = new THREE.Group();
    this.key = '';
    this.head = null;
  }

  // 当前这一步要放大哪一块矩阵、用哪条真实数据，以及怎么称呼输入 / 权重 / 输出
  source(st) {
    const s = st.step, Q = this.M.Q, mats = this.M.mats;
    if (!Q) return null;
    if (s.ph === 'head' && s.sub === 'unembed' && s.mi) {
      const e = Q.headMMAt(st.g);
      if (!e) return null;
      const tok = esc(tokPlain(e.token));
      return {
        kind: 'head', e, sum: sums(e, 1024), panel: this.headPanel(st), inX: -0.14,
        where: '输出头 · 给候选词元打分',
        x: 'h', xDesc: '最后一层之后、最终 RMSNorm 之后的向量（1024 个数）',
        w: 'E', wDesc: `嵌入表里「${tok}」那一行（输出矩阵和嵌入表共用）`, wIdx: (i) => `E[${e.j}, ${i}]`,
        y: 'logit', yIdx: `logit[${e.j}]`, yDesc: `「${tok}」的分数`,
        yLen: Q.M.vocab, ySegs: 0, yUnit: '词元',
        dest: `「${tok}」在词表里的编号是 ${e.j}。模型给词表里全部 ${Q.M.vocab.toLocaleString('zh-CN')} 个词元都这样算一个分数，再用 softmax 变成概率。`,
      };
    }
    if (s.ph !== 'layer' || !s.mi) return null;
    const mm = Q.mmAt(s.L, st.g);
    const where = (t) => `第 ${s.L} 层 · ${t}`;
    if (s.sub === 'qkv') {
      if (!mm?.q) return null;
      const e = mm.q;
      return {
        kind: 'q', e, sum: sums(e, 1024), panel: mats.attn.q, outVec: mats.attn.qo, outSeg: e.head, inX: -0.14, rope: true,
        where: where('注意力 · 算出 q 的一个数'),
        x: 'h', xDesc: '这一层 RMSNorm 之后的向量（1024 个数）',
        w: 'W<sub>q</sub>', wDesc: '训练好的查询矩阵，1024 × 2048', wIdx: (i) => `W<sub>q</sub>[${i}, ${e.j}]`,
        y: 'q', yIdx: `q[${e.j}]`, yDesc: `第 ${e.head} 号头 · 第 ${e.dim} 维`,
        yLen: 2048, ySegs: 16, ySeg: e.head, yUnit: '头',
        dest: `q 一共 2048 个数 = 16 个头 × 每头 128 维；这是第 ${e.head} 号头的第 ${e.dim} 维。每一个都这样算一遍，k、v 也一样（换成 W<sub>k</sub>、W<sub>v</sub>）。`,
        ropePos: ropePos(e),
      };
    }
    if (s.sub === 'mix') {
      if (!mm?.o) return null;
      const e = mm.o;
      return {
        kind: 'o', e, sum: sums(e, 2048), panel: mats.attn.o, outVec: mats.attn.oo, outSeg: Math.floor(e.j / 128), inX: -0.14,
        where: where('注意力 · 输出投影'),
        x: 'z', xDesc: '16 个头各自加权求和的结果，拼成 2048 个数',
        w: 'W<sub>o</sub>', wDesc: '训练好的输出矩阵，2048 × 1024', wIdx: (i) => `W<sub>o</sub>[${i}, ${e.j}]`,
        y: 'Δx', yIdx: `Δx[${e.j}]`, yDesc: `注意力的输出 · 第 ${e.j} 维`,
        yLen: 1024, ySegs: 8, ySeg: Math.floor(e.j / 128), yUnit: '段',
        dest: `Δx 一共 1024 个数，会原样加回残差流（⊕）。这是其中第 ${e.j} 个。`,
      };
    }
    if (s.sub === 'down') {
      if (!mm?.down) return null;
      const e = mm.down;
      return {
        kind: 'down', e, sum: sums(e, 3072), panel: mats.mlp.d, outVec: mats.mlp.dout, outSeg: Math.floor(e.j / 128), inX: -0.08,
        where: where('前馈 · 降维'),
        x: 'a', xDesc: '3072 个神经元的输出 silu(g)·u',
        w: 'W<sub>down</sub>', wDesc: '训练好的降维矩阵，3072 × 1024', wIdx: (i) => `W<sub>down</sub>[${i}, ${e.j}]`,
        y: 'Δx', yIdx: `Δx[${e.j}]`, yDesc: `前馈网络的输出 · 第 ${e.j} 维`,
        yLen: 1024, ySegs: 8, ySeg: Math.floor(e.j / 128), yUnit: '段',
        dest: `Δx 一共 1024 个数，加回残差流（⊕）之后，第 ${s.L} 层就结束了。这是其中第 ${e.j} 个。`,
      };
    }
    if (s.sub === 'up') {
      const n = Q.neuronAt(s.L, st.g);
      if (!n) return null;
      const { g, u } = neuronMM(n);
      return {
        kind: 'gate', e: g, e2: u, sum: g.sum, panel: mats.mlp.g, panel2: mats.mlp.u, outVec: mats.mlp.go, outVec2: mats.mlp.uo, outSeg: Math.floor(g.j / 256), inX: -0.14,
        where: where(`前馈 · 升维（神经元 #${g.j}）`),
        x: 'h', xDesc: '这一层第二次 RMSNorm 之后的向量（1024 个数）',
        w: 'W<sub>gate</sub>', wDesc: '训练好的门控矩阵，1024 × 3072', wIdx: (i) => `W<sub>gate</sub>[${i}, ${g.j}]`,
        y: 'g', yIdx: `g[${g.j}]`, yDesc: `神经元 #${g.j} 的门控值`,
        yLen: 3072, ySegs: 12, ySeg: Math.floor(g.j / 256), yUnit: '段',
        dest: `3072 个神经元，每个都有自己的一列。同一个神经元在 W<sub>up</sub> 里也有一列，同样乘加得到 u[${g.j}]。`,
      };
    }
    return null;
  }

  // 输出头用的矩阵太长（151936 列），只画一截
  headPanel(st) {
    if (!this.head) {
      const g = new THREE.Group();
      const w = 3.2, h = 1024 * U;
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: 0xb39dff, transparent: true, opacity: 0.14, side: THREE.DoubleSide, depthWrite: false }));
      m.position.set(w / 2, h / 2, 0);
      g.add(m);
      const edge = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color: 0xb39dff, transparent: true, opacity: 0.5 }));
      edge.position.copy(m.position);
      g.add(edge);
      const lb = label('E<sup>T</sup><small>1024 × 151936 · 只画了其中一段</small>', 'lbl part');
      lb.position.set(0, -0.02, 0);
      lb.center.set(0, -0.2);
      g.add(lb);
      const vec = new THREE.Mesh(new THREE.BoxGeometry(0.05, h, 0.05), new THREE.MeshStandardMaterial({ color: 0x223, emissive: ROLE.x, emissiveIntensity: 0.4 }));
      vec.position.set(-0.14, h / 2, 0);
      g.add(vec);
      g.w = w; g.h = h;
      this.head = g;
      this.M.root.add(g);
    }
    const hx = this.M.headX(st);
    this.head.position.set(hx - 1.4, this.M.yTop + 0.95, 0.9);
    return this.head;
  }

  clear() {
    this.root.traverse((c) => { c.geometry?.dispose(); c.material?.dispose?.(); if (c.el) c.el.remove(); });
    this.root.clear();
    this.root.parent?.remove(this.root);
    this.dropUp();
    this.rows = null;
  }

  dropUp() {
    if (!this.up) return;
    this.up.traverse((c) => { c.geometry?.dispose(); c.material?.dispose?.(); if (c.el) c.el.remove(); });
    this.up.parent?.remove(this.up);
    this.up = null;
  }

  // 输出向量里第 j 个数在面板坐标系里的位置（面板和输出向量挂在同一个父节点下，x 方向一一对应）
  outY(src) {
    if (!src.outVec) return src.panel.h + 0.14;
    return src.outVec.position.y - src.panel.position.y;
  }

  build(src) {
    this.clear();
    const { e, panel } = src;
    panel.add(this.root);
    const R = this.root;
    const colX = src.kind === 'head' ? panel.w * 0.62 : e.j * U;
    this.colX = colX;
    const oy = this.outY(src);
    this.oy = oy;
    const mat = (c, op = 1) => new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: op, depthWrite: false });
    // 高亮的那一列（紫色 = 权重）
    this.colMesh = new THREE.Mesh(new THREE.PlaneGeometry(0.012, panel.h), mat(ROLE.w, 0.8));
    this.colMesh.position.set(colX, panel.h / 2, 0.003);
    // 列顶到输出格的一段导轨：乘积沿着它飞上去
    this.rail = new THREE.Mesh(new THREE.PlaneGeometry(0.004, Math.max(0.01, oy - panel.h)), mat(ROLE.p, 0.35));
    this.rail.position.set(colX, (panel.h + oy) / 2, 0.003);
    R.add(this.colMesh, this.rail);
    this.colLbl = label(src.kind === 'head' ? `<span class="cw">E<sup>T</sup></span> 的第 ${e.j} 列 = 嵌入表里「${esc(tokPlain(e.token))}」那一行` : `<span class="cw">${src.w}</span> 的第 ${e.j} 列`, 'lbl tag w');
    this.colLbl.position.set(colX, 0, 0.01);
    this.colLbl.center.set(0.5, -0.35);
    R.add(this.colLbl);
    // 输出格（青色 = 结果）
    this.outCell = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.07, 0.07), new THREE.MeshBasicMaterial({ color: ROLE.y, transparent: true, opacity: 0.9 }));
    this.outCell.position.set(colX, oy, 0.03);
    R.add(this.outCell);
    this.outLbl = label('', 'lbl tag y');
    this.outLbl.position.set(colX, oy + 0.04, 0.03);
    this.outLbl.center.set(0.5, 1.25);
    R.add(this.outLbl);
    this.inLbl = label(`输入 <span class="cx">${src.x}</span><small>${src.sum.n} 个数</small>`, 'lbl tag x');
    this.inLbl.position.set(src.inX - 0.04, panel.h / 2, 0.03);
    this.inLbl.center.set(1, 0.5);
    R.add(this.inLbl);
    // 12 项：输入的第 i 个数 → 格子 (i, j)
    const n = e.dims.length;
    this.rows = e.dims.map((d, k) => {
      const y = src.kind === 'head' ? ((d + 0.5) / 1024) * panel.h : (d + 0.5) * U;
      const x = e.x[k], w = e.w[k], prod = x * w;
      const cell = new THREE.Mesh(new THREE.PlaneGeometry(0.03, 0.03), mat(ROLE.w, 0.95));
      cell.position.set(colX, y, 0.005);
      const row = new THREE.Mesh(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(src.inX, y, 0.004), new THREE.Vector3(colX, y, 0.004)]), new THREE.LineBasicMaterial({ color: ROLE.x, transparent: true, opacity: 0 }));
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.012, 10, 8), new THREE.MeshBasicMaterial({ color: ROLE.x }));
      dot.position.set(src.inX, y, 0.03);
      R.add(cell, row, dot);
      cell.userData.pick = { type: 'mmcell', d, j: e.j, x, w, prod, click: true };
      this.E.pickables.push(cell);
      return { cell, row, dot, k, d, y, prod };
    });
    this.n = n;
    // 正在走的那一项：蓝色的输入沿行走过去，琥珀色的乘积顺着列飞上去
    this.runner = new THREE.Mesh(new THREE.SphereGeometry(0.016, 12, 10), new THREE.MeshBasicMaterial({ color: ROLE.x }));
    this.spark = new THREE.Mesh(new THREE.SphereGeometry(0.02, 12, 10), new THREE.MeshBasicMaterial({ color: ROLE.p }));
    this.halo = new THREE.Mesh(new THREE.RingGeometry(0.024, 0.03, 24), mat(0xffffff, 0));
    R.add(this.runner, this.spark, this.halo);
    // SwiGLU 的 up 那一路：同一个神经元在 W_up 里的那一列
    if (src.e2) {
      this.up = new THREE.Group();
      src.panel2.add(this.up);
      const c2 = new THREE.Mesh(new THREE.PlaneGeometry(0.012, src.panel2.h), mat(ROLE.w, 0.6));
      c2.position.set(colX, src.panel2.h / 2, 0.003);
      this.up.add(c2);
      this.upLbl = label(`<span class="cw">W<sub>up</sub></span> 的第 ${e.j} 列 → u[${e.j}]`, 'lbl tag w');
      this.upLbl.position.set(colX, 0, 0.01);
      this.upLbl.center.set(0.5, -0.35);
      this.up.add(this.upLbl);
      this.upCells = src.e2.dims.map((d) => {
        const c = new THREE.Mesh(new THREE.PlaneGeometry(0.03, 0.03), mat(ROLE.w, 0.9));
        c.position.set(colX, (d + 0.5) * U, 0.005);
        this.up.add(c);
        return c;
      });
    } else this.up = null;
    this.src = src;
  }

  update(st, dt, t) {
    const v = st.view;
    const bits = v === 'bits' && st.step.mi === 'mul' && st.step.sub !== 'score';
    const active = v === 'mm' || bits;
    if (this.head) this.head.visible = active && st.step.ph === 'head';
    if (!active) { if (this.root.parent) this.clear(); this.dropUp(); this.src = null; this.key = ''; return; }
    const src = this.source(st);
    if (!src) return;   // 数据分块还没载入：保持上一帧
    const key = `${st.g}|${st.step.L}|${st.step.sub}|${src.kind}|${st.step.ph}`;
    if (key !== this.key) { this.key = key; this.build(src); }
    const mi = st.step.mi, p = st.p;
    const at = ['pick', 'mul', 'sum', 'rope'].indexOf(mi);
    const n = this.n;
    const selK = this.rows.find((r) => r.d === this.selD)?.k ?? 0;
    // 比特视图：只留下被放大的那一个格子
    if (bits) {
      this.colMesh.material.opacity = 0.12;
      this.rail.visible = this.outCell.visible = this.runner.visible = this.spark.visible = false;
      this.colLbl.visible = this.outLbl.visible = this.inLbl.visible = false;
      this.halo.material.opacity = 0;
      this.rows.forEach((r) => { const on = r.k === selK; r.cell.visible = on; r.row.material.opacity = 0; r.dot.visible = false; r.cell.material.color.copy(C.w); r.cell.scale.setScalar(on ? 1.2 : 1); });
      if (this.up) this.up.visible = false;
      return;
    }
    this.rail.visible = this.outCell.visible = true;
    this.colLbl.visible = this.inLbl.visible = true;
    this.colMesh.material.opacity = 0.5 + 0.3 * Math.sin(t * 4);
    // 乘法：第 k 项的进度（0 → 0.55 输入走到格子，0.55 → 1 乘积飞到输出格）
    const phase = (k) => (at > 1 ? 1 : at < 1 ? 0 : termAt(p, k, n));
    let cur = -1;
    this.rows.forEach((r) => {
      const ph = phase(r.k);
      const done = ph >= 0.62;
      if (ph > 0 && ph < 1 && cur < 0) cur = r.k;
      r.row.material.opacity = at === 0 ? 0.28 : ph > 0 && ph < 1 ? 0.9 : done ? 0.3 : 0.1;
      r.dot.visible = true;
      r.cell.material.color.copy(done ? C.p : C.w);
      r.cell.scale.setScalar(ph > 0.45 && ph < 0.8 ? 1.6 : done ? 1.15 : 0.9);
    });
    // 行走的输入与飞行的乘积
    this.runner.visible = this.spark.visible = false;
    this.halo.material.opacity = 0;
    if (cur >= 0) {
      const r = this.rows[cur], ph = phase(cur);
      const a = easeInOut(seg(ph, 0, 0.5)), b = easeInOut(seg(ph, 0.55, 1));
      if (ph < 0.55) { this.runner.visible = true; this.runner.position.set(this.src.inX + (this.colX - this.src.inX) * a, r.y, 0.03); }
      else { this.spark.visible = true; this.spark.position.set(this.colX, r.y + (this.oy - r.y) * b, 0.035); }
      this.halo.position.set(this.colX, r.y, 0.03);
      this.halo.material.opacity = seg(ph, 0.4, 0.55) * (1 - seg(ph, 0.7, 0.9));
      this.halo.scale.setScalar(1 + seg(ph, 0.4, 0.9) * 1.5);
    }
    // 求和：其余的项一起顺着整列扫上去
    const sumP = mi === 'sum' ? easeOut(p) : at > 2 ? 1 : 0;
    if (mi === 'sum' && p < 0.85) {
      this.spark.visible = true;
      const k = seg(p, 0, 0.85);
      this.spark.position.set(this.colX, this.src.panel.h * k + (this.oy - this.src.panel.h) * seg(p, 0.6, 0.85), 0.035);
    }
    const e = this.src.e;
    const doneN = at === 1 ? termsDone(p, n) : at > 1 ? n : 0;
    const partial = this.rows.reduce((a, r) => a + (r.k < doneN ? r.prod : 0), 0);
    const val = sumP > 0 ? e.shown + (e.total - e.shown) * sumP : partial;
    const glow = at === 0 ? 0.25 : at === 1 ? 0.35 + 0.45 * (doneN / n) : 1;
    this.outCell.material.opacity = 0.35 + 0.6 * glow;
    this.outCell.scale.set(1, 1 + (mi === 'sum' ? 0.6 * Math.sin(p * Math.PI) : 0), 1);
    const name = this.src.yIdx;
    const txt = at === 0 ? `${name} = ?` : at === 3 ? `${name} = ${fmt(e.total)} → 旋转后 ${fmt(e.qr, 4)}` : `${name} ${sumP >= 1 ? '=' : '≈'} ${fmt(val)}`;
    if (this.outLbl.el._t !== txt) { this.outLbl.el.innerHTML = txt; this.outLbl.el._t = txt; }
    if (this.up) {
      this.up.visible = true;
      this.upCells.forEach((c, k) => { const done = phase(k) >= 0.62; c.material.color.copy(done ? C.p : C.w); c.visible = at > 0; });
    }
    // 输出向量里对应的那一段闪一下
    if (this.src.outVec) {
      const segs = this.src.outVec.segs;
      const hot = Math.min(segs.length - 1, this.src.outSeg % segs.length);
      segs.forEach((m, i) => { m.material.emissiveIntensity = i === hot ? 0.45 + (sumP > 0 ? 0.35 + 0.25 * Math.sin(t * 6) : 0) : 0.12; });
    }
  }

  // 相机：把输入向量、整块面板和输出向量都框进来（算式板占掉的那部分画面由 Engine 的边距扣掉）
  camera(st) {
    const src = this.src;
    if (!src || !this.root.parent) return null;
    const panel = src.panel;
    let y0 = -0.16, y1 = this.oy + 0.16;
    if (src.panel2) y0 = src.panel2.position.y - panel.position.y - 0.2;
    const a = panel.localToWorld(new THREE.Vector3(src.inX - 0.22, y0, 0));
    const b = panel.localToWorld(new THREE.Vector3(panel.w + 0.08, y1, 0));
    const look = a.clone().add(b).multiplyScalar(0.5);
    const d = this.E.fitDistance(b.x - a.x, b.y - a.y, 1.06);
    return { pos: look.clone().add(new THREE.Vector3(0.04, d * 0.05, d)), look };
  }

  // 比特层要看的那个权重：默认是贡献最大的那一项，也可以在上一层点选某个格子 / 某一行
  bitsSource(st) {
    const src = this.src || this.source(st);
    if (!src) return null;
    const e = src.e;
    const k = Math.max(0, e.dims.indexOf(this.selD ?? e.dims[0]));
    const row = this.rows?.find((r) => r.k === k);
    const d = e.dims[k];
    return {
      v: e.w[k], x: e.x[k], total: e.total, k, d,
      label: src.wIdx(d).replace(/<sub>(.*?)<\/sub>/g, '_$1').replace(/<[^>]+>/g, ''),
      wHtml: src.wIdx(d), xHtml: `${src.x}[${d}]`, out: src.yIdx, yDesc: src.yDesc,
      anchor: row?.cell, srcKind: src.kind, e2: src.e2, head: src.kind === 'head' ? e : null,
    };
  }
}

export const fmt = (v, d = 3) => (Number.isFinite(v) ? `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}` : String(v));
export { sub as subscript, esc };
