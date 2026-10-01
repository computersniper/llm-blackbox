// 权重矩阵的实物：按真实形状等比例缩放的面板（1024 维 = 0.7 个单位），
// 输入向量从左边进来，扫描线扫过矩阵，输出向量逐段填满。Q 分 16 段、K/V 各 8 段，直接看得出 GQA。
import { THREE, label, easeOut, seg } from './engine.js';

const U = 0.7 / 1024;           // 每一维的长度（1024 维 = 0.7 个单位）
const C = { q: 0x5ef0d4, k: 0xffb65c, v: 0xb39dff, o: 0x8fb8ff, gate: 0x5ef0d4, up: 0xb39dff, down: 0xffb65c };

function gridTexture(color, seed) {
  const c = document.createElement('canvas');
  c.width = 128; c.height = 128;
  const g = c.getContext('2d');
  const base = new THREE.Color(color);
  let s = seed * 9301 + 49297;
  const rnd = () => ((s = (s * 9301 + 49297) % 233280) / 233280);
  for (let y = 0; y < 32; y++) {
    for (let x = 0; x < 32; x++) {
      const v = rnd();
      const k = 0.06 + Math.pow(v, 3) * 0.5;
      g.fillStyle = `rgba(${Math.round(base.r * 255)},${Math.round(base.g * 255)},${Math.round(base.b * 255)},${k})`;
      g.fillRect(x * 4, y * 4, 4, 4);
    }
  }
  g.strokeStyle = 'rgba(255,255,255,0.06)';
  for (let i = 0; i <= 128; i += 16) { g.beginPath(); g.moveTo(i, 0); g.lineTo(i, 128); g.stroke(); g.beginPath(); g.moveTo(0, i); g.lineTo(128, i); g.stroke(); }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

// 一块权重矩阵：宽 = 输出维度，高 = 输入维度
function panel(name, inDim, outDim, color, seed) {
  const w = outDim * U, h = inDim * U;
  const g = new THREE.Group();
  const tex = gridTexture(color, seed);
  tex.repeat.set(Math.max(1, Math.round(outDim / 1024)), Math.max(1, Math.round(inDim / 1024)));
  const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.9, depthWrite: false, side: THREE.DoubleSide });
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
  m.position.set(w / 2, h / 2, 0);
  m.userData.noFocus = false;
  g.add(m);
  const edge = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 }));
  edge.position.copy(m.position);
  g.add(edge);
  // 扫描线：表示“这一列正在和输入向量做点积”
  const scan = new THREE.Mesh(new THREE.PlaneGeometry(0.012, h), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 }));
  scan.position.set(0, h / 2, 0.002);
  g.add(scan);
  const lb = label(`${name}<small>${inDim} × ${outDim} · 真实权重分布</small>`, 'lbl part');
  lb.position.set(0, h + 0.02, 0);
  lb.center.set(0, 1.15);
  g.add(lb);
  Object.assign(g, { w, h, mat, edge, scan, lb, color });
  return g;
}

// 一条向量：分成若干段（比如 16 个头），fill 控制填满几成
function vec(len, segs, color, vertical = false) {
  const g = new THREE.Group();
  const gap = 0.006, sl = (len - gap * (segs - 1)) / segs;
  g.segs = [];
  for (let i = 0; i < segs; i++) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(vertical ? 0.05 : sl, vertical ? sl : 0.05, 0.05), new THREE.MeshStandardMaterial({ color: new THREE.Color(color).multiplyScalar(0.35), emissive: color, emissiveIntensity: 0.35, transparent: true, opacity: 0.95 }));
    const o = i * (sl + gap) + sl / 2;
    if (vertical) m.position.set(0, o, 0); else m.position.set(o, 0, 0);
    g.add(m);
    g.segs.push(m);
  }
  g.len = len;
  g.setFill = (f, glow = 0.35) => g.segs.forEach((m, i) => { const on = i / segs < f; m.visible = on; m.material.emissiveIntensity = glow; });
  return g;
}

export class Mats {
  constructor(E, M) {
    this.E = E;
    this.M = M;
    this.texCache = new Map();
    this.root = new THREE.Group();
    E.scene.add(this.root);
  }

  load(Q) {
    this.root.traverse((c) => { c.geometry?.dispose(); c.material?.map?.dispose(); c.material?.dispose?.(); if (c.el) c.el.remove(); });
    this.root.clear();
    const Mo = Q.M;
    const H = Mo.hidden, qd = Mo.heads * Mo.headDim, kd = Mo.kvHeads * Mo.headDim, F = Mo.ffn;

    // 注意力：输入 h → W_q / W_k / W_v → q / k / v；再由 W_o 投影回去
    const A = (this.attn = new THREE.Group());
    A.x = vec(H * U, 8, 0x9fb8ff, true);
    A.xl = label('h<small>1 × 1024</small>', 'lbl num');
    A.xl.center.set(1.1, 0.5);
    A.xl.position.set(-0.03, (H * U) / 2, 0);
    A.x.add(A.xl);
    A.q = panel('W<sub>q</sub>', H, qd, C.q, 1);
    A.k = panel('W<sub>k</sub>', H, kd, C.k, 2);
    A.v = panel('W<sub>v</sub>', H, kd, C.v, 3);
    A.q.position.x = 0.14;
    A.k.position.x = A.q.position.x + A.q.w + 0.16;
    A.v.position.x = A.k.position.x + A.k.w + 0.16;
    A.qo = vec(qd * U, Mo.heads, C.q);
    A.ko = vec(kd * U, Mo.kvHeads, C.k);
    A.vo = vec(kd * U, Mo.kvHeads, C.v);
    const outY = H * U + 0.2;
    A.qo.position.set(A.q.position.x, outY, 0);
    A.ko.position.set(A.k.position.x, outY, 0);
    A.vo.position.set(A.v.position.x, outY, 0);
    const outL = (o, html) => { const l = label(html, 'lbl num'); l.position.set(o.len / 2, 0.06, 0); l.center.set(0.5, 1.1); o.add(l); o.lbl = l; };
    outL(A.qo, `q<small>1 × ${qd} = ${Mo.heads} 头 × ${Mo.headDim}</small>`);
    outL(A.ko, `k<small>${Mo.kvHeads} 头 × ${Mo.headDim} → 缓存</small>`);
    outL(A.vo, `v<small>${Mo.kvHeads} 头 × ${Mo.headDim} → 缓存</small>`);
    // GQA：每 2 个 Q 头连到同一个 KV 头
    const pts = [];
    const qs = (qd * U) / Mo.heads, ks = (kd * U) / Mo.kvHeads;
    for (let h = 0; h < Mo.heads; h++) {
      const kv = Math.floor(h / (Mo.heads / Mo.kvHeads));
      pts.push(new THREE.Vector3(A.q.position.x + (h + 0.5) * qs, outY + 0.05, 0), new THREE.Vector3(A.k.position.x + (kv + 0.5) * ks, outY + 0.05, 0));
    }
    A.gqa = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: 0 }));
    // 把连线画成向上拱的弧线，免得和向量重叠
    const arc = [];
    for (let i = 0; i < pts.length; i += 2) {
      const a = pts[i], b = pts[i + 1];
      const curve = new THREE.QuadraticBezierCurve3(a, new THREE.Vector3((a.x + b.x) / 2, a.y + 0.22 + Math.abs(a.x - b.x) * 0.15, 0), b);
      const cp = curve.getPoints(16);
      for (let j = 0; j < cp.length - 1; j++) arc.push(cp[j], cp[j + 1]);
    }
    A.gqa.geometry.dispose();
    A.gqa.geometry = new THREE.BufferGeometry().setFromPoints(arc);
    A.gqaL = label('GQA：每 2 个 Q 头共用 1 个 K/V 头', 'lbl hint');
    A.gqaL.position.set(A.q.position.x + A.q.w + 0.1, outY + 0.42, 0);
    A.gqaL.center.set(0.5, 1);
    A.add(A.x, A.q, A.k, A.v, A.qo, A.ko, A.vo, A.gqa, A.gqaL);
    // W_o：16 个头拼成 2048 维，投影回 1024
    A.o = panel('W<sub>o</sub>', qd, H, C.o, 4);
    A.o.visible = false;
    A.oo = vec(H * U, 8, C.o);
    A.oo.visible = false;
    outL(A.oo, 'Δx<small>1 × 1024 → 加回残差</small>');
    A.oin = vec(qd * U, Mo.heads, C.q, true);
    A.oin.visible = false;
    const oinL = label(`拼接<small>${Mo.heads} × ${Mo.headDim} = 1 × ${qd}</small>`, 'lbl num');
    oinL.position.set(-0.03, (qd * U) / 2, 0);
    oinL.center.set(1.1, 0.5);
    A.oin.add(oinL);
    A.oinL = oinL;
    A.add(A.o, A.oo, A.oin);
    this.root.add(A);

    // SwiGLU：h → W_gate / W_up（1024 → 3072）；silu(g)·u → W_down（3072 → 1024）
    const P = (this.mlp = new THREE.Group());
    P.x = vec(H * U, 8, 0x9fb8ff, true);
    const xl = label('h<small>1 × 1024</small>', 'lbl num');
    xl.position.set(-0.03, (H * U) / 2, 0);
    xl.center.set(1.1, 0.5);
    P.x.add(xl);
    P.xl = xl;
    P.g = panel('W<sub>gate</sub>', H, F, C.gate, 5);
    P.u = panel('W<sub>up</sub>', H, F, C.up, 6);
    P.g.position.x = 0.14;
    P.u.position.set(0.14, -(H * U) - 0.32, 0);
    P.x2 = P.x.clone();
    P.x2.position.set(0, P.u.position.y, 0);
    P.x2l = P.x2.children.find((c) => c.isCSS2DObject);
    P.go = vec(F * U, 12, C.gate);
    P.uo = vec(F * U, 12, C.up);
    P.go.position.set(0.14, H * U + 0.1, 0);
    P.uo.position.set(0.14, P.u.position.y - 0.1, 0);
    outL(P.go, `g<small>1 × ${F}</small>`);
    outL(P.uo, `u<small>1 × ${F}</small>`);
    P.uo.lbl.center.set(0.5, -0.2);
    P.d = panel('W<sub>down</sub>', F, H, C.down, 7);
    P.d.position.set(0.14 + F * U + 0.35, -(H * U) - 0.32, 0);
    P.din = vec(F * U, 12, 0xffffff, true);
    P.din.position.set(P.d.position.x - 0.08, P.d.position.y, 0);
    const dinL = label(`silu(g) ⊙ u<small>1 × ${F}</small>`, 'lbl num');
    dinL.position.set(-0.03, (F * U) / 2, 0);
    dinL.center.set(1.1, 0.5);
    P.din.add(dinL);
    P.dinL = dinL;
    P.dout = vec(H * U, 8, C.down);
    P.dout.position.set(P.d.position.x, P.d.position.y + F * U + 0.1, 0);
    outL(P.dout, 'Δx<small>1 × 1024 → 加回残差</small>');
    P.add(P.x, P.x2, P.g, P.u, P.go, P.uo, P.d, P.din, P.dout);
    this.root.add(P);
    this.root.visible = true;
    this.thumbL = -1;
  }

  update(st, dt, t) {
    const M = this.M, s = st.step, v = st.view, p = st.p;
    const A = this.attn, P = this.mlp;
    if (!A) return;
    const micro = v === 'mm' || (v === 'bits' && s.sub !== 'score');
    const inLayer = s.ph === 'layer' && M.e > 0.6 && (v === 'layer' || v === 'attn' || v === 'mlp' || micro);
    if (inLayer) this.applyThumbs(M.explodeL);
    // 微观视图里输入 / 输出另有带颜色的标签（micro.js），这些尺寸标签先收起来，免得叠在一起
    for (const l of [A.xl, A.qo.lbl, A.oo.lbl, A.oinL, P.xl, P.x2l, P.go.lbl, P.uo.lbl, P.dinL, P.dout.lbl, P.u.lb]) if (l) l.visible = !micro;
    for (const pn of [A.q, A.o, P.g, P.u, P.d]) pn.edge.material.opacity = micro ? 0.35 : 0.7;
    const xf = M.xFocus(st);
    const base = 0.95 + M.explodeL * 0.26;
    const e = M.e;
    // 注意力这一组
    const showA = inLayer && s.op === 'attn';
    A.visible = showA;
    if (showA) {
      A.position.set(xf + 0.62, base + 0.62 * e + 0.2, 0.62);
      const sub = s.sub || (v === 'layer' ? (p < 0.5 ? 'qkv' : 'mix') : 'qkv');
      const qkv = sub === 'qkv';
      const kP = s.sub ? (qkv ? p : 1) : v === 'layer' ? seg(p, 0, 0.5) : 1;
      for (const [pn, out] of [[A.q, A.qo], [A.k, A.ko], [A.v, A.vo]]) {
        pn.visible = out.visible = sub !== 'mix';
        const active = qkv && kP < 1;
        pn.scan.material.opacity = active ? 0.85 : 0;
        pn.scan.position.x = pn.w * easeOut(kP);
        pn.mat.opacity = qkv ? 0.9 : 0.35;
        out.setFill(easeOut(kP), qkv ? 0.55 : 0.25);
      }
      if (micro) {
        // 微观视图：只留下正在放大的那块矩阵（W_q），W_k / W_v 和 GQA 连线先收起来；面板压暗，好让高亮的那一列跳出来
        for (const pn of [A.k, A.v, A.ko, A.vo]) pn.visible = false;
        A.q.scan.material.opacity = 0;
        A.q.mat.opacity = v === 'bits' ? 0.06 : 0.55;
        A.qo.setFill(1, 0.12);
      } else for (const pn of [A.k, A.v, A.ko, A.vo]) pn.visible = sub !== 'mix';
      A.x.visible = sub !== 'mix';
      A.x.setFill(1, qkv ? 0.6 : 0.2);
      A.gqa.material.opacity = sub === 'qkv' ? 0.55 * seg(kP, 0.7, 1) : sub === 'score' ? 0.5 : 0.15;
      A.gqa.visible = A.gqaL.visible = sub !== 'mix' && !micro;
      const mix = sub === 'mix';
      A.o.visible = A.oo.visible = A.oin.visible = mix;
      if (mix) {
        const mP = s.sub ? p : seg(p, 0.5, 1);
        A.oin.position.set(0, 0, 0);
        A.o.position.set(0.14, 0, 0);
        A.oo.position.set(0.14, (A.o.h) + 0.12, 0);
        A.oin.setFill(1, 0.5);
        A.o.scan.material.opacity = mP < 1 ? 0.85 : 0;
        A.o.scan.position.x = A.o.w * easeOut(mP);
        A.oo.setFill(easeOut(mP), 0.55);
        if (micro) { A.o.scan.material.opacity = 0; A.oo.setFill(1, 0.12); A.o.mat.opacity = v === 'bits' ? 0.06 : 0.55; }
      }
    }
    // 前馈这一组
    const showP = inLayer && s.op === 'mlp';
    P.visible = showP;
    if (showP) {
      P.position.set(xf + 0.62, base + 1.78 * e + 0.72, 0.62);
      const sub = s.sub || (v === 'layer' ? (p < 0.5 ? 'up' : 'down') : 'up');
      const upP = sub === 'up' ? (s.sub ? p : seg(p, 0, 0.5)) : 1;
      const up = sub === 'up';
      for (const [pn, out] of [[P.g, P.go], [P.u, P.uo]]) {
        pn.scan.material.opacity = up && upP < 1 ? 0.85 : 0;
        pn.scan.position.x = pn.w * easeOut(upP);
        pn.mat.opacity = up ? 0.9 : 0.35;
        out.setFill(easeOut(upP), up ? 0.55 : 0.3);
      }
      P.x.setFill(1, up ? 0.6 : 0.2);
      P.x2.visible = true;
      const down = sub === 'down';
      const dP = down ? (s.sub ? p : seg(p, 0.5, 1)) : 0;
      P.d.mat.opacity = down ? 0.9 : 0.3;
      P.d.scan.material.opacity = down && dP < 1 ? 0.85 : 0;
      P.d.scan.position.x = P.d.w * easeOut(dP);
      P.din.setFill(sub === 'up' ? 0 : 1, sub === 'act' ? 0.7 : 0.4);
      P.dout.setFill(easeOut(dP), 0.55);
      if (micro) {
        // 升维时只留 W_gate / W_up，降维时只留 W_down 和它的输入输出
        for (const pn of [P.g, P.u, P.d]) pn.scan.material.opacity = 0;
        const hi = v === 'bits' ? 0.06 : 0.55;
        for (const o of [P.g, P.u, P.go, P.uo, P.x, P.x2]) o.visible = up;
        for (const o of [P.d, P.din, P.dout]) o.visible = down;
        P.g.mat.opacity = P.u.mat.opacity = P.d.mat.opacity = hi;
        P.go.setFill(1, 0.12); P.uo.setFill(1, 0.12); P.dout.setFill(1, 0.12); P.din.setFill(1, 0.25);
      } else for (const o of [P.g, P.u, P.go, P.uo, P.x, P.x2, P.d, P.din, P.dout]) o.visible = true;
    }
  }

  // 换成这一层真实的权重分布（每格 = 32×32 个权重的均方根，越亮越大）
  applyThumbs(L) {
    if (!this.thumbs || L < 0 || this.thumbL === L) return;
    this.thumbL = L;
    const idx = this.thumbIndex;
    const set = (pn, name) => {
      const spec = idx[`${L}:${name}`];
      if (!spec) return;
      const key = `${L}:${name}`;
      let tex = this.texCache.get(key);
      if (!tex) {
        const { w, h, offset } = spec;
        const c = new THREE.Color(pn.color);
        const data = new Uint8Array(w * h * 4);
        for (let r = 0; r < h; r++) {
          for (let x = 0; x < w; x++) {
            const v = this.thumbs[offset + (h - 1 - r) * w + x] / 255; // 导出时第 0 行是最后一维，这里翻回来
            const o = (r * w + x) * 4;
            const k = Math.pow(v, 0.7);
            data[o] = Math.round(40 + c.r * 215 * k); data[o + 1] = Math.round(40 + c.g * 215 * k); data[o + 2] = Math.round(50 + c.b * 205 * k);
            data[o + 3] = Math.round(40 + 215 * k);
          }
        }
        tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat);
        tex.magFilter = THREE.NearestFilter;
        tex.minFilter = THREE.LinearFilter;
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.needsUpdate = true;
        this.texCache.set(key, tex);
      }
      pn.mat.map = tex;
      pn.mat.needsUpdate = true;
      pn.real = true;
    };
    set(this.attn.q, 'q'); set(this.attn.k, 'k'); set(this.attn.v, 'v'); set(this.attn.o, 'o');
    set(this.mlp.g, 'gate'); set(this.mlp.u, 'up'); set(this.mlp.d, 'down');
  }

  setThumbs(bytes, index) { this.thumbs = bytes; this.thumbIndex = index; this.thumbL = -1; }

  // 让相机把矩阵也框进去：返回这一组在世界坐标里的右边界
  extentX(st) {
    const xf = this.M.xFocus(st);
    const s = st.step;
    if (s.op === 'mlp') return xf + 0.62 + 0.14 + (3072 + 1024) * U + 0.6;
    return xf + 0.62 + 0.14 + (2048 + 1024 + 1024) * U + 0.5;
  }

  // 注意力这一组的中心和尺寸，给 Q·K·V / 加权求和这两步取景用
  attnFrame(st) {
    const x0 = this.M.xFocus(st) + 0.3, base = 0.95 + this.M.explodeL * 0.26 + 0.62 * this.M.e + 0.2;
    if (st.step.sub === 'mix') return { x0: x0 - 0.9, x1: x0 + 1.9, y: base + 0.75, h: 2.1 };
    return { x0, x1: this.extentX(st), y: base + 0.55, h: 1.9 };
  }
}
