// 矩阵乘法的显微镜：y[j] = Σ_i x[i]·W[i, j]。
// 在真实形状的权重面板上高亮第 j 列，输入向量的第 i 个数沿着第 i 行走到格子 (i, j)，
// 逐项相乘，再沿着这一列加起来，落进输出向量的第 j 个位置。数字全部来自真实模型。
import { THREE, label, easeOut, seg } from './engine.js';
import { esc, tokPlain } from '../ui.js';

const U = 0.7 / 1024;
const SUBS = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
const sub = (n) => String(n).split('').map((c) => SUBS[c] || c).join('');
const f = (v, d = 3) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`;

function dial(e) {
  // RoPE 这一对数 (d, d±64) 在平面上转过的角度：真实的 q_norm 前后、旋转前后
  const R = 44, cx = 56, cy = 56;
  const mx = Math.max(Math.hypot(e.qn, e.qnP), Math.hypot(e.qr, e.qrP), 1e-6);
  const p = (a, b) => [cx + (a / mx) * R, cy - (b / mx) * R];
  const [x0, y0] = p(e.qn, e.qnP), [x1, y1] = p(e.qr, e.qrP);
  const deg = ((e.angle * 180) / Math.PI) % 360;
  return `<svg width="112" height="112" viewBox="0 0 112 112" style="display:block;margin:2px auto">
    <circle cx="${cx}" cy="${cy}" r="${R}" fill="none" stroke="rgba(150,180,230,.25)"/>
    <line x1="${cx - R - 4}" y1="${cy}" x2="${cx + R + 4}" y2="${cy}" stroke="rgba(150,180,230,.2)"/>
    <line x1="${cx}" y1="${cy - R - 4}" x2="${cx}" y2="${cy + R + 4}" stroke="rgba(150,180,230,.2)"/>
    <line x1="${cx}" y1="${cy}" x2="${x0}" y2="${y0}" stroke="#7a859e" stroke-width="2" stroke-dasharray="3 3"/>
    <line x1="${cx}" y1="${cy}" x2="${x1}" y2="${y1}" stroke="#5ef0d4" stroke-width="2.5"/>
    <circle cx="${x1}" cy="${y1}" r="3.5" fill="#5ef0d4"/>
    <text x="4" y="108" font-size="9" fill="#7a859e" font-family="JetBrains Mono,monospace">转过 ${deg.toFixed(1)}°</text>
  </svg>`;
}

export class Micro {
  constructor(E, M) {
    this.E = E;
    this.M = M;
    this.root = new THREE.Group();
    this.key = '';
    this.head = null;
  }

  // 当前这一步要放大哪一块矩阵、用哪条真实数据
  source(st) {
    const s = st.step, Q = this.M.Q, mats = this.M.mats;
    if (s.ph === 'head' && s.sub === 'unembed' && s.mi) {
      const e = Q.headMMAt(st.g);
      return e && { kind: 'head', e, inDim: 1024, panel: this.headPanel(st), inX: -0.14, name: 'Eᵀ', out: `logit[${e.j}]`, outName: `「${tokPlain(e.token)}」的分数`, col: `第 ${e.j} 列（「${tokPlain(e.token)}」在嵌入表里的那一行）` };
    }
    if (s.ph !== 'layer' || !s.mi) return null;
    const mm = Q.mmAt(s.L, st.g);
    if (s.sub === 'qkv' && mm) return { kind: 'q', inDim: 1024, e: mm.q, panel: mats.attn.q, outVec: mats.attn.qo, outSeg: mm.q.head, inX: -0.14, name: 'W<sub>q</sub>', out: `q[${mm.q.j}]`, outName: `第 ${mm.q.head} 头第 ${mm.q.dim} 维`, col: `第 ${mm.q.j} 列 = 第 ${mm.q.head} 头 × 128 + ${mm.q.dim}` };
    if (s.sub === 'mix' && mm) return { kind: 'o', inDim: 2048, e: mm.o, panel: mats.attn.o, outVec: mats.attn.oo, outSeg: Math.floor(mm.o.j / 128), inX: -0.14, name: 'W<sub>o</sub>', out: `Δx[${mm.o.j}]`, outName: `注意力输出的第 ${mm.o.j} 维`, col: `第 ${mm.o.j} 列` };
    if (s.sub === 'down' && mm) return { kind: 'down', inDim: 3072, e: mm.down, panel: mats.mlp.d, outVec: mats.mlp.dout, outSeg: Math.floor(mm.down.j / 128), inX: -0.08, name: 'W<sub>down</sub>', out: `Δx[${mm.down.j}]`, outName: `前馈输出的第 ${mm.down.j} 维`, col: `第 ${mm.down.j} 列` };
    if (s.sub === 'up') {
      const n = Q.neuronAt(s.L, st.g);
      if (!n) return null;
      const e = { j: n.j, dims: n.dims, x: n.x, w: n.wg, total: n.gz, shown: n.x.reduce((a, x, k) => a + x * n.wg[k], 0) };
      const e2 = { j: n.j, dims: n.dims, x: n.x, w: n.wu, total: n.uz, shown: n.x.reduce((a, x, k) => a + x * n.wu[k], 0) };
      return { kind: 'gate', inDim: 1024, e, e2, panel: mats.mlp.g, panel2: mats.mlp.u, outVec: mats.mlp.go, outVec2: mats.mlp.uo, outSeg: Math.floor(n.j / 256), inX: -0.14, name: 'W<sub>gate</sub>', out: `g[${n.j}]`, outName: `神经元 #${n.j} 的 g`, col: `第 ${n.j} 列 = 神经元 #${n.j}` };
    }
    return null;
  }

  // 输出头用的矩阵太长（151936 列），只画一截
  headPanel(st) {
    if (!this.head) {
      const g = new THREE.Group();
      const w = 3.2, h = 1024 * U;
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: 0xb39dff, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false }));
      m.position.set(w / 2, h / 2, 0);
      g.add(m);
      const edge = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color: 0xb39dff, transparent: true, opacity: 0.6 }));
      edge.position.copy(m.position);
      g.add(edge);
      const lb = label('E<sup>T</sup><small>1024 × 151936 · 只画了其中一段</small>', 'lbl part');
      lb.position.set(0, h + 0.02, 0);
      lb.center.set(0, 1.15);
      g.add(lb);
      const vec = new THREE.Mesh(new THREE.BoxGeometry(0.05, h, 0.05), new THREE.MeshStandardMaterial({ color: 0x223, emissive: 0x9fb8ff, emissiveIntensity: 0.4 }));
      vec.position.set(-0.14, h / 2, 0);
      g.add(vec);
      const vl = label('h<small>最终 RMSNorm 之后 · 1 × 1024</small>', 'lbl num');
      vl.position.set(-0.18, h / 2, 0);
      vl.center.set(1, 0.5);
      g.add(vl);
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
  }

  dropUp() {
    if (!this.up) return;
    this.up.traverse((c) => { c.geometry?.dispose(); c.material?.dispose?.(); if (c.el) c.el.remove(); });
    this.up.parent?.remove(this.up);
    this.up = null;
  }

  build(src) {
    this.clear();
    const { e, panel } = src;
    panel.add(this.root);
    const R = this.root;
    const colX = src.kind === 'head' ? panel.w * 0.62 : e.j * U;
    this.colX = colX;
    // 高亮的那一列
    this.colMesh = new THREE.Mesh(new THREE.PlaneGeometry(0.014, panel.h), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85, depthWrite: false }));
    this.colMesh.position.set(colX, panel.h / 2, 0.003);
    R.add(this.colMesh);
    this.colLbl = label(`${src.name} 的${esc(src.col)}`, 'lbl part');
    this.colLbl.position.set(colX, -0.02, 0);
    this.colLbl.center.set(0.5, -0.2);
    R.add(this.colLbl);
    // 12 行：输入的第 i 个数 → 格子 (i, j)
    const order = e.dims.map((d, k) => ({ d, k })).sort((a, b) => b.d - a.d);
    const listX = Math.max(colX + 0.25, (panel.w || 0) + 0.12);
    const n = order.length;
    this.rows = order.map(({ d, k }, r) => {
      const y = src.kind === 'head' ? ((d + 0.5) / 1024) * panel.h : (d + 0.5) * U;
      const x = e.x[k], w = e.w[k], prod = x * w;
      const pos = prod >= 0;
      const col = pos ? 0xffb65c : 0x6b9bff;
      const cell = new THREE.Mesh(new THREE.PlaneGeometry(0.03, 0.03), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.95, depthWrite: false }));
      cell.position.set(colX, y, 0.005);
      const row = new THREE.Mesh(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(src.inX, y, 0.004), new THREE.Vector3(colX, y, 0.004)]), new THREE.LineBasicMaterial({ color: 0x9fb8ff, transparent: true, opacity: 0 }));
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.013, 10, 8), new THREE.MeshBasicMaterial({ color: 0x9fe8ff }));
      dot.position.set(src.inX, y, 0.03);
      const sp = Math.max(panel.h / n, 0.085);
      const ly = panel.h / 2 + (n / 2 - r - 0.5) * sp;
      const lead = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(colX, y, 0.005), new THREE.Vector3(listX, ly, 0.005)]), new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0 }));
      const lb = label(`x${sub(d)} × W[${d}, ${e.j}] = ${f(x)} × ${f(w, 4)} = <b>${f(prod, 4)}</b>`, `lbl num ${pos ? 'pos' : 'neg'}`);
      lb.position.set(listX + 0.02, ly, 0.005);
      lb.center.set(0, 0.5);
      R.add(cell, row, dot, lead, lb);
      cell.userData.pick = { type: 'mmcell', d, j: e.j, x, w, prod, click: true };
      this.E.pickables.push(cell);
      return { cell, row, dot, lead, lb, k, r };
    });
    this.sumLbl = label('', 'lbl big');
    this.sumLbl.position.set(colX, panel.h + 0.05, 0.01);
    this.sumLbl.center.set(0.5, 1.4);
    R.add(this.sumLbl);
    if (src.kind === 'q') {
      this.ropeLbl = label('', 'lbl');
      this.ropeLbl.position.set(listX + 0.02, panel.h + 0.32, 0.01);
      this.ropeLbl.center.set(0, 1);
      this.ropeLbl.el.style.whiteSpace = 'normal';
      this.ropeLbl.el.style.width = '250px';
      R.add(this.ropeLbl);
    } else this.ropeLbl = null;
    // SwiGLU 的 up 那一路：同一个神经元在 W_up 里的那一列
    if (src.e2) {
      this.up = new THREE.Group();
      src.panel2.add(this.up);
      const c2 = new THREE.Mesh(new THREE.PlaneGeometry(0.014, src.panel2.h), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 }));
      c2.position.set(colX, src.panel2.h / 2, 0.003);
      this.up.add(c2);
      this.upLbl = label('', 'lbl big');
      this.upLbl.position.set(colX, -0.04, 0.01);
      this.upLbl.center.set(0.5, -0.3);
      this.up.add(this.upLbl);
      this.upCells = src.e2.dims.map((d, k) => {
        const c = new THREE.Mesh(new THREE.PlaneGeometry(0.03, 0.03), new THREE.MeshBasicMaterial({ color: src.e2.x[k] * src.e2.w[k] >= 0 ? 0xffb65c : 0x6b9bff, transparent: true }));
        c.position.set(colX, (d + 0.5) * U, 0.005);
        this.up.add(c);
        return c;
      });
    } else this.up = null;
    this.src = src;
  }

  update(st, dt, t) {
    const v = st.view;
    const active = v === 'mm' || (v === 'bits' && st.step.mi === 'mul' && (st.step.sub !== 'score'));
    if (this.head) this.head.visible = active && st.step.ph === 'head';
    if (!active) { if (this.root.parent) this.clear(); this.dropUp(); this.src = null; this.key = ''; return; }
    const src = this.source(st);
    if (!src) return;
    const key = `${st.g}|${st.step.L}|${st.step.sub}|${src.kind}`;
    if (key !== this.key) { this.key = key; this.build(src); }
    const mi = st.step.mi, p = st.p;
    const order = ['pick', 'mul', 'sum', 'rope'];
    const at = order.indexOf(mi);
    const mulP = mi === 'mul' ? p * 1.1 : at > 1 ? 1 : 0;
    const e = src.e;
    this.colMesh.material.opacity = 0.55 + 0.35 * Math.sin(t * 4);
    this.rows.forEach(({ cell, row, dot, lead, lb, r }) => {
      const on = (this.rows.length - 1 - r) / this.rows.length < mulP || at > 1;
      row.material.opacity = mi === 'pick' ? 0.15 : on ? 0.5 : 0.08;
      lead.material.opacity = on ? 0.55 : 0;
      lb.visible = on;
      cell.scale.setScalar(on ? 1.3 : 0.8);
      dot.visible = true;
    });
    const sumP = mi === 'sum' ? easeOut(p) : at > 2 ? 1 : 0;
    this.sumLbl.visible = sumP > 0;
    if (sumP > 0) {
      const rest = e.total - e.shown;
      this.sumLbl.el.innerHTML = `${src.out} = Σ<small>${src.inDim} 项</small> = ${f(e.total * sumP, 3)}<br><small style="font-size:10.5px;color:var(--dim)">前 12 项 ${f(e.shown, 3)} ＋ 其余 ${src.inDim - 12} 项 ${f(rest, 3)} · ${esc(src.outName)}</small>`;
    }
    if (this.up) {
      this.up.visible = true;
      this.upCells.forEach((c) => { c.visible = mulP > 0; });
      this.upLbl.visible = sumP > 0;
      if (sumP > 0) this.upLbl.el.innerHTML = `u[${src.e2.j}] = ${f(src.e2.total * sumP, 3)}<br><small style="font-size:10.5px;color:var(--dim)">同一个神经元在 W<sub>up</sub> 里的那一列</small>`;
    }
    if (this.ropeLbl) {
      const rp = mi === 'rope' ? easeOut(p) : 0;
      this.ropeLbl.visible = rp > 0;
      if (rp > 0) {
        this.ropeLbl.el.innerHTML = `<b style="color:var(--ink)">q_norm</b>：${f(e.total)} ÷ RMS ${e.rms.toFixed(3)} × γ ${e.qnW.toFixed(4)} = <b style="color:var(--cyan)">${f(e.qn, 4)}</b><br>
          <b style="color:var(--ink)">RoPE</b>：第 ${e.dim} 维和第 ${e.partner} 维配成一对，按位置 ${e.row} 旋转 θ = ${e.row} / 10⁶<sup>${(2 * e.freq / 128).toFixed(3)}</sup> = ${e.angle.toFixed(3)} 弧度
          ${dial(e)}
          旋转后 q[${e.j}] = <b style="color:var(--cyan)">${f(e.qr, 4)}</b>（这就是拿去和 K 做点积的数）`;
      }
    }
    // 输出向量里对应的那一段闪一下
    if (src.outVec && sumP > 0) {
      const segs = src.outVec.segs;
      const k = Math.min(segs.length - 1, Math.floor((src.outSeg / (segs.length)) * segs.length));
      segs.forEach((m, i) => { m.material.emissiveIntensity = i === Math.min(segs.length - 1, src.outSeg % segs.length) ? 0.6 + 0.5 * Math.sin(t * 6) : 0.2; });
      void k;
    }
  }

  // 相机：对准高亮的那一列，并把右边的乘积清单框进来
  camera(st) {
    const src = this.src;
    if (!src || !this.root.parent) return null;
    const panel = src.panel;
    const a = panel.localToWorld(new THREE.Vector3(Math.min(this.colX, 0) - 0.3, 0, 0));
    const listH = Math.max(panel.h, 12 * 0.085);
    const b = panel.localToWorld(new THREE.Vector3(Math.max(this.colX + 0.25, panel.w + 0.12) + 1.5, panel.h, 0));
    const look = a.clone().add(b).multiplyScalar(0.5);
    const rope = src.kind === 'q' && st.step.mi === 'rope';
    const h = Math.max(b.y - a.y, listH) + (rope ? 1.5 : 0.45);
    const d = this.E.fitDistance(b.x - a.x, h, 1.04);
    if (rope) look.y += 0.6;
    if (src.kind === 'gate') look.y -= 0.35;
    return { pos: look.clone().add(new THREE.Vector3(0.1, d * 0.08, d)), look };
  }

  // 比特层要看的那个权重：第一行（贡献最大）的 W[i, j]
  bitsSource(st) {
    const src = this.src || this.source(st);
    if (!src) return null;
    const e = src.e;
    const k = Math.max(0, e.dims.indexOf(this.selD ?? e.dims[0]));
    const row = this.rows?.find((r) => r.k === k);
    return { v: e.w[k], x: e.x[k], total: e.total, label: `${src.name.replace(/<[^>]+>/g, '')}[${e.dims[k]}, ${e.j}]`, out: src.out, anchor: row?.cell, srcKind: src.kind, e2: src.e2, k };
  }
}
