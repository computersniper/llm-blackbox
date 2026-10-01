// D4 · 反向 · 一层之内：这一层 7 个矩阵各分到多少梯度。矩阵按真实形状等比例画出（宽 = 输出维，高 = 输入维）。
import { COL, text, rr, card, hexA, clamp, sciSup, fmtInt, divColor, wrap, waitFade } from '../draw.js';
import { shapes, TENSOR_NAME } from '../run.js';

const ORDER = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'];
const TCOL = { q: COL.cyan, k: COL.amber, v: COL.violet, o: '#8fb8ff', gate: COL.cyan, up: COL.violet, down: COL.amber };

export class LayerView {
  constructor(app, R) { this.app = app; this.R = R; this.cache = new Map(); }

  layout(env) {
    this.portrait = env.portrait;
    const sh = shapes(this.R.model);
    const T = this.R.kind === 'tiny';
    const P = this.portrait;
    // 每一维多长
    const u = P ? (T ? 0.36 : 0.068) : T ? 0.42 : 0.105;
    this.u = u;
    const gap = P ? 12 : 22;
    const pos = {};
    if (!P) {
      let x = 0;
      const y = 180;
      for (const t of ORDER) {
        const [inD, outD] = sh[t];
        pos[t] = { x, y, w: outD * u, h: inD * u };
        x += outD * u + gap + (t === 'o' ? 30 : 0);
      }
      this.W = x;
    } else {
      // 竖屏：注意力一行、前馈一行
      let x = 0, y = 170;
      for (const t of ['q', 'k', 'v', 'o']) { const [inD, outD] = sh[t]; pos[t] = { x, y, w: outD * u, h: inD * u }; x += outD * u + gap; }
      y += Math.max(...['q', 'k', 'v', 'o'].map((t) => pos[t].h)) + 64;
      x = 0;
      for (const t of ['gate', 'up']) { const [inD, outD] = sh[t]; pos[t] = { x, y, w: outD * u, h: inD * u }; y += inD * u + 40; }
      const [inD, outD] = sh.down;
      pos.down = { x, y, w: outD * u, h: inD * u };
      this.W = 380;
    }
    this.pos = pos;
    this.H = Math.max(...Object.values(pos).map((p) => p.y + p.h)) + 160;
  }

  focus() { return { x: -16, y: -10, w: Math.max(this.W, 380) + 32, h: this.H + 10 }; }

  crop(k, c) {
    const key = `${k}-${c}`;
    if (this.cache.has(key)) return this.cache.get(key);
    const D = this.R.D, C = D.C;
    const cv = document.createElement('canvas');
    cv.width = C; cv.height = C;
    const x = cv.getContext('2d');
    const img = x.createImageData(C, C);
    const sc = D.meta.ckpts[k].gScale[c] * 127;
    for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) {
      // 权重是 [输出, 输入]；画面上宽 = 输出、高 = 输入
      const v = D.gcrop(k, c, i, j) / sc;
      const col = divColor(v).match(/\d+/g).map(Number);
      const o = (j * C + i) * 4;
      img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = 255;
    }
    x.putImageData(img, 0, 0);
    this.cache.set(key, cv);
    return cv;
  }

  draw(g, st, env) {
    const R = this.R, k = st.k, s = st.step, T = R.kind === 'tiny', P = this.portrait;
    const L = s.i - 1;
    const sh = shapes(R.model);
    text(g, `BACKWARD · LAYER ${L} · 7 个矩阵的梯度`, 0, 18, { size: 10, kind: 'mono', color: COL.dim });
    text(g, `第 ${L} 层：梯度分到每个矩阵`, 0, 52, { size: P ? 19 : 24, kind: 'serif', weight: 900, color: COL.ink });
    wrap(g, '一个矩阵的梯度 = 流进它的上游梯度 ⊗ 它的输入（外积），再把所有位置加起来。下面每块按真实形状等比例画出，越亮梯度越大。', 0, 80, P ? 380 : 900, 18, { size: 12, color: COL.ink2 });
    const vals = ORDER.map((t) => R.tgrad(k, L, t));
    const mx = Math.max(...vals);
    ORDER.forEach((t, j) => {
      const p = this.pos[t];
      const v = vals[j];
      const on = s.mi === t;
      const a = 0.08 + 0.55 * Math.sqrt(v / mx);
      g.fillStyle = hexA(TCOL[t], a);
      g.fillRect(p.x, p.y, p.w, p.h);
      g.lineWidth = on ? 2.2 : 1;
      g.strokeStyle = on ? COL.amber : hexA(TCOL[t], 0.6);
      g.strokeRect(p.x, p.y, p.w, p.h);
      // 小模型：两块有真实梯度局部（左上角 48×48）
      if (T) {
        const ci = L === 2 && t === 'q' ? 1 : L === 4 && t === 'down' ? 2 : -1;
        if (ci >= 0 && R.D.has('ck', k)) {
          const cs = 48 * this.u;
          g.imageSmoothingEnabled = false;
          g.drawImage(this.crop(k, ci), p.x, p.y, cs, cs);
          g.imageSmoothingEnabled = true;
          g.strokeStyle = COL.ink2;
          g.strokeRect(p.x, p.y, cs, cs);
          text(g, '真实梯度 48×48', p.x + cs + 4, p.y + 10, { size: 8.5, color: COL.ink2 });
        } else if (ci >= 0) {
          // 梯度局部在这个检查点的 D1 分块里，还没到
          g.globalAlpha = waitFade(env.t - st.wait.since);
          text(g, '真实梯度 48×48 · 载入中…', p.x + 4, p.y + 12, { size: 8.5, color: COL.ink2 });
          g.globalAlpha = 1;
        }
      }
      text(g, TENSOR_NAME[t], p.x, p.y - (P ? 22 : 26), { size: P ? 11 : 13, color: on ? COL.amber : COL.ink, weight: 600 });
      text(g, `${sh[t][0]}×${sh[t][1]}`, p.x, p.y - (P ? 9 : 10), { size: 9, kind: 'mono', color: COL.dim });
      text(g, sciSup(v, 2), p.x, p.y + p.h + 15, { size: P ? 10 : 11, kind: 'mono', color: on ? COL.amber : COL.ink2 });
      const per = v / Math.sqrt(sh[t][0] * sh[t][1]);
      env.hit(p.x, p.y, p.w, p.h, {
        click: true,
        act: () => this.app.seek((b) => b.ph === 'bwd' && b.i === s.i && b.mi === t),
        tip: `<span class="k">第 ${L} 层 ${TENSOR_NAME[t]} · ${sh[t][0]} × ${sh[t][1]}</span>梯度范数 ‖∇W‖ = <span class="v">${sciSup(v, 4)}</span><br>平均每个权重 ${sciSup(per, 3)}（均方根）<br>${fmtInt(sh[t][0] * sh[t][1])} 个梯度`,
      });
    });
    if (!P) {
      const a = this.pos.q, b = this.pos.o, c = this.pos.gate, d = this.pos.down;
      text(g, '注意力', a.x, a.y - 52, { size: 11, kind: 'mono', color: COL.dim });
      text(g, '前馈 SwiGLU', c.x, c.y - 52, { size: 11, kind: 'mono', color: COL.dim });
      g.strokeStyle = COL.line2;
      g.beginPath(); g.moveTo(a.x, a.y - 46); g.lineTo(b.x + b.w, a.y - 46); g.stroke();
      g.beginPath(); g.moveTo(c.x, c.y - 46); g.lineTo(d.x + d.w, c.y - 46); g.stroke();
    }
    // RMSNorm 的缩放 γ（向量）
    const ny = this.H - 120;
    text(g, '同一层里还有 4 个向量参数（RMSNorm 的缩放 γ）：', 0, ny, { size: 11, color: COL.dim });
    ['ln1', 'qn', 'kn', 'ln2'].forEach((t, j) => {
      const v = R.tgrad(k, L, t);
      const x = P ? (j % 2) * 190 : j * 230, y = ny + 22 + (P ? Math.floor(j / 2) * 38 : 0);
      text(g, TENSOR_NAME[t], x, y + 10, { size: 10.5, color: COL.ink2 });
      text(g, sciSup(v, 2), x, y + 26, { size: 10.5, kind: 'mono', color: COL.dim });
    });
  }
}
