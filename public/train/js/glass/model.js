// 玻璃小模型的“全景图”：把 11 个参数张量按结构摆开，每个都是真实数值的热力图（D0 初始化、D1 训练全程共用）。
//   E [20 × 16]（输入时取行，输出时乘 Eᵀ）→ γ₁ → W_q W_k W_v → 注意力 → W_o → γ₂ → W_gate W_up → SwiGLU → W_down → γ_f → Eᵀ
// 矩阵一律画成 [输入维 × 输出维]（y = x · W），RMSNorm 的 γ 画成一条 16 格的带子，对齐它缩放的那 16 维。
import { COL, text, hexA, arrow } from '../draw.js';
import { heat, cellAt, mark } from './heat.js';
import { L } from '../lang.js';
import { TENSOR_LABEL } from './data.js';

// 每个张量在全景图里的位置：x, y（左上角），cell（格子边长），o = 'v' | 'h'（γ 的方向）
function place(portrait) {
  if (!portrait) {
    const c = 8, y1 = 30, y2 = 228;
    return {
      W: 890, H: 500, c,
      at: {
        E: { x: 24, y: y1, cell: c }, gf: { x: 24, y: y1 + 20 * c + 36, cell: c, o: 'h' },
        g1: { x: 182, y: y1, cell: c, o: 'v' },
        Wq: { x: 196, y: y1, cell: c }, Wk: { x: 340, y: y1, cell: c }, Wv: { x: 484, y: y1, cell: c }, Wo: { x: 650, y: y1, cell: c },
        g2: { x: 182, y: y2, cell: c, o: 'v' },
        Wg: { x: 196, y: y2, cell: c }, Wu: { x: 468, y: y2, cell: c }, Wd: { x: 752, y: y2, cell: c },
      },
    };
  }
  const c = 9;
  return {
    W: 380, H: 1290, c,
    at: {
      E: { x: 26, y: 54, cell: c },
      g1: { x: 204, y: 70, cell: c, o: 'h' }, g2: { x: 204, y: 130, cell: c, o: 'h' }, gf: { x: 204, y: 190, cell: c, o: 'h' },
      Wq: { x: 26, y: 290, cell: c }, Wk: { x: 204, y: 290, cell: c },
      Wv: { x: 26, y: 470, cell: c }, Wo: { x: 204, y: 470, cell: c },
      Wg: { x: 26, y: 650, cell: c }, Wu: { x: 26, y: 830, cell: c },
      Wd: { x: 26, y: 1010, cell: c },
    },
  };
}

export class ModelMap {
  constructor(D) { this.D = D; }

  layout(portrait, ox = 0, oy = 0) {
    const pl = place(portrait);
    this.portrait = portrait;
    this.W = pl.W;
    this.H = pl.H;
    this.ox = ox;
    this.oy = oy;
    this.rects = {};
    for (const p of this.D.params) {
      const a = pl.at[p.name];
      const vert = p.norm && a.o === 'v';
      const rows = p.norm ? (vert ? p.rows : 1) : p.rows, cols = p.norm ? (vert ? 1 : p.rows) : p.cols;
      this.rects[p.name] = { p, x: ox + a.x, y: oy + a.y, rows, cols, cw: a.cell, ch: a.cell, w: cols * a.cell, h: rows * a.cell, vert };
    }
    return this;
  }

  bounds() { return { x: this.ox - 8, y: this.oy - 8, w: this.W + 16, h: this.H + 16 }; }

  // 第 r 行第 c 列 → 参数在全部参数里的下标
  gi(name, r, c) {
    const R = this.rects[name], p = R.p;
    if (p.norm) return p.off + (R.vert ? r : c);
    return p.off + r * p.cols + c;
  }

  // opts：value(gi) → 上色用的值；scale(name) → 色标；tip(name, gi) → 提示 html；pick(name, gi)；sel = 选中参数的下标；
  //       sub(name) → 标题下面的一行小字；frame(name) → 边框色（显示梯度 / Δw 时换色）；dim(name) → 0–1 的透明度
  draw(g, env, o) {
    const D = this.D, P = this.portrait;
    this.flow(g, env, o);
    for (const p of D.params) {
      const R = this.rects[p.name];
      const a = o.dim ? o.dim(p.name) : 1;
      const sc = o.scale(p.name);
      g.globalAlpha = a;
      heat(g, R.x, R.y, R.rows, R.cols, R.cw, R.ch, (r, c) => o.value(this.gi(p.name, r, c)), { scale: sc, s: env.s });
      const fc = o.frame?.(p.name);
      if (fc) { g.strokeStyle = fc; g.lineWidth = 1.6; g.strokeRect(R.x - 3, R.y - 3, R.w + 6, R.h + 6); }
      // 标题：名字 [形状]
      const shape = p.norm ? `[${p.rows}]` : `[${p.rows} × ${p.cols}]`;
      const ty = R.y - (p.norm && !R.vert ? 8 : 22);
      const lab = TENSOR_LABEL[p.name];
      if (p.norm && R.vert) {
        text(g, lab, R.x + R.w / 2, R.y - 8, { size: 11, kind: 'mono', weight: 600, color: COL.ink2, align: 'center' });
      } else {
        text(g, lab, R.x, ty, { size: P ? 11.5 : 12.5, kind: 'mono', weight: 600, color: COL.ink });
        text(g, shape, R.x + (P ? 6 : 7) * lab.length + 10, ty, { size: 9.5, kind: 'mono', color: COL.dim });
        const sub = o.sub?.(p.name);
        if (sub && !p.norm) text(g, sub, R.x, ty + 13, { size: 9.5, color: COL.dim, max: Math.max(R.w, 150) });
      }
      // 嵌入表：每行是哪个字
      if (p.name === 'E') {
        for (let r = 0; r < R.rows; r++) text(g, D.ch(r), R.x - 5, R.y + r * R.ch + R.ch * 0.82, { size: Math.min(11, R.ch * 1.05), color: COL.ink2, align: 'right' });
      }
      // 注意力的两个头：W_q / W_k / W_v 的输出维、W_o 的输入维前 8 维是头 0、后 8 维是头 1
      if (['Wq', 'Wk', 'Wv'].includes(p.name)) {
        g.strokeStyle = 'rgba(233,239,249,0.35)'; g.lineWidth = 1;
        g.beginPath(); g.moveTo(R.x + 8 * R.cw, R.y - 3); g.lineTo(R.x + 8 * R.cw, R.y + R.h + 3); g.stroke();
      } else if (p.name === 'Wo') {
        g.strokeStyle = 'rgba(233,239,249,0.35)'; g.lineWidth = 1;
        g.beginPath(); g.moveTo(R.x - 3, R.y + 8 * R.ch); g.lineTo(R.x + R.w + 3, R.y + 8 * R.ch); g.stroke();
      }
      g.globalAlpha = 1;
      if (o.sel != null) {
        const lc = D.locate(o.sel);
        if (lc && lc.p.name === p.name) {
          const r = p.norm ? (R.vert ? lc.i : 0) : lc.i, c = p.norm ? (R.vert ? 0 : lc.i) : lc.j;
          mark(g, R.x, R.y, r, c, R.cw, R.ch, '#ffffff', 2);
        }
      }
      if (o.tip || o.pick) {
        env.hit(R.x, R.y, R.w, R.h, {
          click: !!o.pick,
          tipAt: (wx, wy) => { const h = cellAt(wx, wy, R.x, R.y, R.rows, R.cols, R.cw, R.ch); return h && o.tip ? o.tip(p.name, this.gi(p.name, h.r, h.c)) : null; },
          // 舞台点击时把命中的位置（世界坐标 wx, wy）传给 act
          act: o.pick ? (h) => {
            const hc = h && cellAt(h.wx, h.wy, R.x, R.y, R.rows, R.cols, R.cw, R.ch);
            if (hc) o.pick(p.name, this.gi(p.name, hc.r, hc.c));
          } : null,
        });
      }
    }
  }

  // 数据流：残差主干和几个箭头（只是示意结构，不是数据）
  flow(g, env, o) {
    const R = this.rects, P = this.portrait;
    const col = 'rgba(150,180,230,0.28)';
    const lab = (s, x, y, al = 'left') => text(g, s, x, y, { size: 9.5, color: COL.dim, align: al });
    if (!P) {
      const E = R.E, q = R.Wq, wo = R.Wo, g2 = R.g2, wd = R.Wd, gf = R.gf, g1 = R.g1;
      const yA = q.y + q.h + 20, yB = R.Wg.y + R.Wg.h + 20;
      // 注意力一行：E → γ₁ → q k v → W_o
      arrow(g, E.x + E.w + 6, E.y + 64, g1.x - 5, E.y + 64, col, 1.2, 6);
      g.strokeStyle = col; g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(q.x, yA); g.lineTo(wo.x + wo.w, yA); g.stroke();
      lab(L('q·k → 注意力（两个头，RoPE 按位置旋转，没有参数）→ 加权求和 v → 乘 W_o，加回残差', 'q·k → attention (2 heads, RoPE rotates by position, no parameters) → weighted sum of v → × W_o, added to the residual'), q.x, yA + 13);
      // 第一行 → 第二行
      g.beginPath(); g.moveTo(wo.x + wo.w + 10, wo.y + wo.h / 2); g.lineTo(wo.x + wo.w + 26, wo.y + wo.h / 2); g.lineTo(wo.x + wo.w + 26, g2.y - 36); g.lineTo(g2.x - 12, g2.y - 36); g.stroke();
      arrow(g, g2.x - 12, g2.y - 36, g2.x - 12, g2.y + 30, col, 1.2, 6);
      // 前馈一行
      g.beginPath(); g.moveTo(R.Wg.x, yB); g.lineTo(wd.x - 14, yB); g.stroke();
      lab(L('SwiGLU：silu(n·W_gate) ⊙ (n·W_up)，再乘 W_down，加回残差', 'SwiGLU: silu(n·W_gate) ⊙ (n·W_up), then × W_down, added to the residual'), R.Wg.x, yB + 13);
      // 输出：最后的 RMSNorm → 乘 Eᵀ
      lab(L('最后的 RMSNorm，再乘 Eᵀ（输出层和输入共用 E）', 'final RMSNorm, then × Eᵀ (output shares E with the input)'), gf.x, gf.y + gf.h + 13);
    } else {
      lab(L('输入取 E 的行，输出乘 Eᵀ（共用）', 'Input reads rows of E; output multiplies by Eᵀ (shared)'), R.E.x, R.E.y + R.E.h + 22);
      lab(L('三个 RMSNorm 的缩放 γ', 'The three RMSNorm scales γ'), R.g1.x, R.g1.y - 26);
    }
  }
}
