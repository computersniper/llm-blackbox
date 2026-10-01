// D3 · 前向 / 反向：残差流的“层 × 位置”网格。
// 前向：每一层的逻辑透镜读数（它此刻会猜哪个字、猜中正确答案的概率），从下往上一层层亮起来；
// 反向：每一层每个位置上的梯度 ‖∂L/∂h‖，从上往下一层层流回去。
// 横屏：行 = 层、列 = 位置（小模型 64 个位置折成两段）；竖屏：转置成 行 = 位置、列 = 层。
import { COL, text, rr, hexA, fmtP, clamp, ease, seg, seqColor, heatColor, sciSup, arrow } from '../draw.js';
import { disp, short } from './row.js';
import { esc } from '../../../js/ui.js';

export class Grid {
  constructor(app, R, mode) { this.app = app; this.R = R; this.mode = mode; }

  layout(env) {
    const R = this.R, T = R.kind === 'tiny';
    this.portrait = env.portrait;
    const n = R.rowLen, NB = R.NB;
    if (!this.portrait) {
      this.per = T ? 32 : n;
      this.blocks = Math.ceil(n / this.per);
      this.cw = T ? 29 : 46;
      this.ch = T ? 30 : 19;
      this.gx = 104;
      this.gy = 100;
      this.gw = this.per * this.cw;
      this.blockH = NB * this.ch + 64;
      this.gh = this.blocks * this.blockH - 64;
      this.side = { x: this.gx + this.gw + 20, w: 170 };
      this.W = this.side.x + this.side.w;
    } else {
      this.per = n;
      this.blocks = 1;
      this.cw = T ? 40 : 10.5;
      this.ch = T ? 21 : 26;
      this.gx = T ? 54 : 70;
      this.gy = 110;
      this.gw = NB * this.cw;
      this.gh = n * this.ch;
      this.side = null;
      this.W = this.gx + this.gw + 8;
    }
    this.H = this.gy + this.gh + 80;
  }

  focus() {
    if (this.portrait) return { x: -6, y: -6, w: Math.max(392, this.W + 12), h: Math.min(this.H + 12, 600) };
    return { x: -12, y: -12, w: this.W + 24, h: this.H + 24 };
  }

  // 第 b 层、第 i 个位置的格子
  cell(b, i) {
    const R = this.R;
    if (!this.portrait) {
      const blk = Math.floor(i / this.per), col = i % this.per;
      return { x: this.gx + col * this.cw, y: this.gy + blk * this.blockH + (R.NB - 1 - b) * this.ch, w: this.cw, h: this.ch };
    }
    return { x: this.gx + b * this.cw, y: this.gy + i * this.ch, w: this.cw, h: this.ch };
  }

  draw(g, st, env) {
    const R = this.R, k = st.k, s = st.step, T = R.kind === 'tiny', P = this.portrait;
    const fwd = this.mode === 'fwd';
    const bi = s.i;
    const n = R.rowLen, NB = R.NB;
    text(g, fwd ? 'FORWARD · LOGIT LENS ON THE RESIDUAL STREAM' : 'BACKWARD · ‖∂L/∂h‖ ON THE RESIDUAL STREAM', 0, 18, { size: 10, kind: 'mono', color: COL.dim });
    text(g, fwd ? '前向：一层层算上去，每层都“读一下”它会猜什么' : '反向：梯度从输出头一层层流回来', 0, 52, { size: P ? 17 : 24, kind: 'serif', weight: 900, color: COL.ink });
    let gmax = 1e-30;
    if (!fwd) for (let b = 0; b < NB; b++) for (let i = 0; i < n; i++) gmax = Math.max(gmax, R.residGrad(k, b, i));
    const pr = ease(seg(st.p, 0, 0.6));
    for (let b = 0; b < NB; b++) {
      const shown = fwd ? b <= bi : b >= bi;
      const a = b === bi ? pr : shown ? 1 : 0;
      for (let i = 0; i < n; i++) {
        const c = this.cell(b, i);
        const counted = R.counted(i);
        if (a <= 0) {
          g.strokeStyle = COL.line;
          g.lineWidth = 1;
          g.strokeRect(c.x + 0.5, c.y + 0.5, c.w - 1, c.h - 1);
          continue;
        }
        g.globalAlpha = a * (counted ? 1 : 0.42);
        if (fwd) {
          const p = R.lensP(k, b, i), lt = R.lensTop(k, b, i);
          g.fillStyle = seqColor(Math.sqrt(p), 0.9);
          g.fillRect(c.x + 0.5, c.y + 0.5, c.w - 1, c.h - 1);
          if (lt.ok) { g.strokeStyle = COL.green; g.lineWidth = 1.3; g.strokeRect(c.x + 1.2, c.y + 1.2, c.w - 2.4, c.h - 2.4); }
          if (c.w >= 14) {
            const sz = T ? 13 : 10;
            text(g, short(lt.s), c.x + c.w / 2, c.y + c.h / 2 + sz * 0.36, { size: sz, color: p > 0.35 ? '#04121a' : COL.ink2, align: 'center', max: c.w - 2 });
          }
        } else {
          const v = Math.log10(R.residGrad(k, b, i) / gmax + 1e-30);
          g.fillStyle = heatColor(clamp(1 + v / 5, 0, 1), 0.92);
          g.fillRect(c.x + 0.5, c.y + 0.5, c.w - 1, c.h - 1);
        }
        g.globalAlpha = 1;
      }
      const lab = b === 0 ? '嵌入' : b === NB - 1 ? `第 ${b - 1} 层 → 输出` : `第 ${b - 1} 层`;
      for (let blk = 0; blk < this.blocks; blk++) {
        const c0 = this.cell(b, blk * this.per);
        if (!P) text(g, lab, this.gx - 10, c0.y + c0.h / 2 + 4, { size: T ? 11.5 : 9.5, color: b === bi ? COL.amber : COL.dim, align: 'right' });
        else if (T || b % 4 === 0 || b === NB - 1) {
          g.save();
          g.translate(c0.x + c0.w / 2 + 3, this.gy - 6);
          g.rotate(-Math.PI / 2.6);
          text(g, b === 0 ? '嵌入' : `L${b - 1}`, 0, 0, { size: 9.5, color: b === bi ? COL.amber : COL.dim });
          g.restore();
        }
        const rowRect = P ? { x: c0.x, y: this.gy, w: this.cw, h: this.gh } : { x: 0, y: c0.y, w: this.gx + this.gw, h: this.ch };
        const i0 = blk * this.per;
        env.hit(rowRect.x, rowRect.y, rowRect.w, rowRect.h, {
          click: true,
          act: !fwd && b >= 1 ? () => { this.app.seek((x) => x.ph === 'bwd' && x.i === b); this.app.into(); } : () => { this.app.seek((x) => x.ph === this.mode && x.i === b); },
          tipAt: (wx, wy) => {
            const i = P ? clamp(Math.floor((wy - this.gy) / this.ch), 0, n - 1) : clamp(i0 + Math.floor((wx - this.gx) / this.cw), 0, n - 1);
            if (P ? wy < this.gy : wx < this.gx) return `<span class="k">${lab}</span>${!fwd && b >= 1 ? '点一下进去看这一层的 7 个矩阵' : '点一下跳到这一层'}`;
            const tgt = disp(R.tok(k, i + 1)), inp = disp(R.tok(k, i));
            const note = R.counted(i) ? '' : `<br><span style="color:var(--dim)">这个位置不计入损失${fwd ? '' : '，但梯度照样流过'}</span>`;
            if (fwd) {
              const lt = R.lensTop(k, b, i);
              return `<span class="k">${lab} · 位置 ${i}</span>输入「${esc(inp)}」，正确答案「<b>${esc(tgt)}</b>」<br>此刻最想说「${esc(disp(lt.s))}」${lt.ok ? '（猜对了）' : ''}<br>正确答案的概率 <span class="v">${fmtP(R.lensP(k, b, i))}</span>　‖h‖ = ${R.residNorm(k, b, i).toFixed(2)}${note}`;
            }
            return `<span class="k">${lab} · 位置 ${i}</span>输入「${esc(inp)}」<br>‖∂L/∂h‖ = <span class="v">${sciSup(R.residGrad(k, b, i), 3)}</span>${b >= 1 ? '<br>点一下进去看这一层的 7 个矩阵' : ''}${note}`;
          },
        });
      }
    }
    // 输入 / 目标
    for (let i = 0; i < n; i++) {
      const cb = this.cell(0, i), ct = this.cell(NB - 1, i);
      const inp = short(R.tok(k, i)), tgt = short(R.tok(k, i + 1));
      const counted = R.counted(i);
      if (!P) {
        text(g, inp, cb.x + cb.w / 2, cb.y + cb.h + 17, { size: T ? 13 : 9.5, color: COL.ink2, align: 'center', max: cb.w });
        text(g, tgt, ct.x + ct.w / 2, ct.y - 8, { size: T ? 13 : 9.5, color: counted ? COL.amber : COL.faint, align: 'center', max: ct.w });
      } else {
        text(g, inp, this.gx - 6, cb.y + cb.h / 2 + 4, { size: T ? 12 : 9.5, color: COL.ink2, align: 'right', max: this.gx - 8 });
        text(g, `→${tgt}`, this.gx + this.gw + 4, cb.y + cb.h / 2 + 4, { size: T ? 11 : 8.5, color: counted ? COL.amber : COL.faint, max: 60 });
      }
    }
    if (!P) {
      for (let blk = 0; blk < this.blocks; blk++) {
        const c0 = this.cell(0, blk * this.per), c1 = this.cell(NB - 1, blk * this.per);
        text(g, '输入', this.gx - 10, c0.y + this.ch + 17, { size: 10, color: COL.dim, align: 'right' });
        text(g, '正确答案', this.gx - 10, c1.y - 8, { size: 10, color: COL.amber, align: 'right' });
        const ax = 14;
        arrow(g, ax, fwd ? c0.y + this.ch : c1.y, ax, fwd ? c1.y + 6 : c0.y + this.ch - 6, fwd ? hexA(COL.cyan, 0.6) : hexA(COL.rose, 0.6), 1.5, 8);
      }
    }
    if (this.side) this.drawSide(g, st, env, fwd);
    const ly = this.gy + this.gh + (P ? 26 : 40);
    text(g, fwd ? '格子 = 逻辑透镜在这一层读出的猜测；底色越亮，正确答案的概率越高；绿框 = 第一名就是正确答案' : '格子 = 梯度的大小（对数刻度，亮 = 大）；残差连接让梯度能直接抄近路流到底层', 0, ly, { size: 10.5, color: COL.dim, max: Math.max(360, this.W) });
    if (!T) text(g, '灰掉的列是提示部分：不计入损失，但前向照算、梯度也照样流过它们', 0, ly + 18, { size: 10.5, color: COL.faint, max: Math.max(360, this.W) });
  }

  drawSide(g, st, env, fwd) {
    const R = this.R, k = st.k, NB = R.NB, n = R.rowLen, S = this.side;
    text(g, fwd ? '这一层的平均 ‖h‖' : '这一层参数的梯度', S.x, this.gy - 10, { size: 10, color: COL.dim });
    const vals = [];
    for (let b = 0; b < NB; b++) {
      if (fwd) { let s = 0; for (let i = 0; i < n; i++) s += R.residNorm(k, b, i); vals.push(s / n); }
      else if (b === 0) vals.push(R.tgrad(k, -1, 'embed'));
      else vals.push(Math.sqrt(['q', 'k', 'v', 'o', 'gate', 'up', 'down', 'ln1', 'ln2', 'qn', 'kn'].reduce((a, t) => a + R.tgrad(k, b - 1, t) ** 2, 0)));
    }
    const mx = Math.max(...vals);
    const bi = st.step.i;
    vals.forEach((v, b) => {
      const c = this.cell(b, 0);
      const shown = fwd ? b <= bi : b >= bi;
      if (!shown) return;
      const w = Math.max(1, (S.w - 60) * (v / mx));
      rr(g, S.x, c.y + c.h * 0.2, w, c.h * 0.6, 2);
      g.fillStyle = b === bi ? (fwd ? COL.cyan : COL.rose) : hexA(fwd ? COL.cyan : COL.rose, 0.4);
      g.fill();
      text(g, fwd ? v.toFixed(1) : sciSup(v, 2), S.x + w + 4, c.y + c.h / 2 + 3.5, { size: 9, kind: 'mono', color: COL.dim });
    });
    if (!fwd) text(g, '最底下是嵌入（和输出矩阵共用一块）', S.x, this.cell(0, 0).y + this.ch + 17, { size: 9, color: COL.faint });
  }
}
