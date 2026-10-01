// D3 · 批次：小模型——取一行 / 字 → 编号 / 错开一位；Qwen3——套聊天模板 / SFT 遮罩 / 错开一位。
import { COL, text, rr, card, hexA, wrap, fmtInt, fmtP, clamp, ease, seg, arrow, measure, line, dot, sciSup } from '../draw.js';
import { rowLayout, tileBase, disp, short, ROLE_COL } from './row.js';
import { esc } from '../../../js/ui.js';

export class Tokens {
  constructor(app, R) { this.app = app; this.R = R; }

  layout(env) {
    this.portrait = env.portrait;
    this.W = this.portrait ? 380 : 1100;
    this.bounds = this.portrait ? { x: -6, y: -6, w: 392, h: 1000 } : { x: -12, y: -12, w: 1124, h: 680 };
  }

  focus() { return this.bounds; }

  draw(g, st, env) {
    if (this.R.kind === 'tiny') this.drawTiny(g, st, env);
    else this.drawQwen(g, st, env);
  }

  /* ---------------------------------------------------------------- 小模型 */

  drawTiny(g, st, env) {
    const R = this.R, k = st.k, s = st.step, m = R.D.meta, P = this.portrait, W = this.W;
    const sub = s.sub;
    const subs = ['row', 'ids', 'shift'];
    const si = subs.indexOf(sub);
    // 左上：批次张量的形状（第 0 行是真实内容，其余 63 行没有导出）
    const bx = 0, by = 0, bw = P ? 380 : 330, bh = P ? 170 : 236;
    card(g, bx, by, bw, bh, { eyebrow: `TENSOR · x [${m.train.batch} × ${m.train.seq + 1}]`, title: '一个批次', accent: COL.blue });
    const gx = bx + 14, gy = by + 56, gw = bw - 28, gh = bh - 90;
    const rows = m.train.batch, cols = m.train.seq + 1;
    const cw = gw / cols, ch = gh / rows;
    g.fillStyle = 'rgba(107,155,255,0.07)';
    g.fillRect(gx, gy, gw, gh);
    g.fillStyle = hexA(COL.amber, 0.85);
    g.fillRect(gx, gy, gw * (si === 0 ? ease(seg(st.p, 0, 0.8)) : 1), Math.max(2, ch));
    g.strokeStyle = COL.line2;
    g.strokeRect(gx, gy, gw, gh);
    text(g, '← 第 0 行（下面画出的就是它）', gx, gy + gh + 16, { size: 10, color: COL.amber });
    text(g, `其余 ${rows - 1} 行内容没有导出`, gx + gw, gy + gh + 16, { size: 10, color: COL.faint, align: 'right' });
    // 右上 / 下：讲解
    const ex = P ? 0 : 350, ey = P ? 186 : 0, ew = P ? 380 : W - 350;
    card(g, ex, ey, ew, P ? 130 : bh, { eyebrow: ['STEP 1 · SAMPLE', 'STEP 2 · TOKENIZE', 'STEP 3 · SHIFT'][si], title: ['从语料里取一行', '字 → 编号', '错开一位：输入和目标'][si], accent: COL.cyan });
    const lines = [
      `训练语料是一条 ${(m.corpus.chars.train / 1e4).toFixed(0)} 万字的长带子（诗与诗之间用 ⏎ 隔开）。每一步随机挑 ${m.train.batch} 个诗的开头，各取连续 ${cols} 个字。`,
      `字级分词：每个字换成它在词表里的编号。这个词表按字频排序，所以编号越小越常见：「，」= 1，「。」= 2。真实的 Qwen3 用 15 万个 BPE 词元。`,
      `输入是前 ${m.train.seq} 个字，目标是后 ${m.train.seq} 个字：位置 i 看到第 0…i 个字，要猜第 i+1 个。一行 ${m.train.seq} 道题，一批 ${fmtInt(m.train.tokensPerStep)} 道。`,
    ];
    wrap(g, lines[si], ex + 14, ey + 66, ew - 28, 20, { size: 12.5, color: COL.ink2 });
    // 主体：65 个字块
    const top = P ? 336 : 256;
    const tile = P ? 30 : 38;
    const rows2 = rowLayout(g, R, k, 0, top + 16, W, { tile, th: si === 1 ? tile + 16 : tile + 4, gap: 4, lineGap: si === 2 ? 30 : 12 });
    const reveal = si === 0 ? Math.floor(ease(seg(st.p, 0, 0.85)) * rows2.length) + 1 : rows2.length;
    for (const r of rows2) {
      if (r.j >= reveal) break;
      tileBase(g, { ...r, h: tile + 4 }, { size: P ? 14 : 16 });
      if (si === 1) {
        const a = clamp((st.p * rows2.length * 1.4 - r.j) / 3, 0, 1);
        g.globalAlpha = a;
        text(g, R.tokId(k, r.j), r.x + r.w / 2, r.y + tile + 16, { size: 9.5, kind: 'mono', color: COL.cyan, align: 'center' });
        g.globalAlpha = 1;
      }
      if (si === 2 && r.j < R.rowLen) {
        // 每个字下面：它要猜的那个字
        const a = clamp((st.p * rows2.length * 1.3 - r.j) / 3, 0, 1);
        g.globalAlpha = a;
        const nx = r.x + r.w / 2, ny = r.y + tile + 8;
        arrow(g, nx, ny, nx, ny + 7, COL.amber, 1, 4);
        text(g, disp(R.tok(k, r.j + 1)), nx, ny + 20, { size: 11, color: COL.amber, align: 'center' });
        g.globalAlpha = 1;
      }
      env.hit(r.x, r.y, r.w, tile + 4, { tip: `<span class="k">第 ${r.j} 个字</span><b>${esc(r.s)}</b>　编号 <span class="v">${R.tokId(k, r.j)}</span>${r.j < R.rowLen ? `<br>目标（下一个字）：<b>${esc(disp(R.tok(k, r.j + 1)))}</b>` : ''}` });
    }
    text(g, `第 0 行的前 ${R.rowLen + 1} 个字（一行共 ${cols} 个）${si === 2 ? '　·　每个字下面的橙色小字是它要猜的下一个字' : ''}`, 0, top, { size: 10.5, color: COL.dim });
  }

  /* ---------------------------------------------------------------- Qwen3 */

  drawQwen(g, st, env) {
    const R = this.R, k = st.k, s = st.step, m = R.D.meta, P = this.portrait, W = this.W;
    const si = ['tpl', 'mask', 'shift'].indexOf(s.sub);
    // 原始模板文本
    const tx = 0, ty = 0, tw = P ? 380 : 520, th = P ? 210 : 250;
    card(g, tx, ty, tw, th, { eyebrow: 'CHAT TEMPLATE · 非思考模式', title: '模型真正看到的文字', accent: COL.violet });
    const raw = m.text.replace(/\n/g, '↵\n');
    let yy = ty + 66;
    for (const ln of raw.split('\n')) {
      if (!ln) continue;
      text(g, ln, tx + 14, yy, { size: P ? 10.5 : 12, kind: 'mono', color: COL.ink2, max: tw - 28 });
      yy += P ? 15 : 18;
    }
    // 讲解卡
    const ex = P ? 0 : 540, ey = P ? 226 : 0, ew = P ? 380 : W - 540;
    const titles = ['套聊天模板 → 24 个词元', 'SFT 遮罩：只对回答算损失', '错开一位：输入和标签'];
    card(g, ex, ey, ew, th, { eyebrow: ['STEP 1 · TEMPLATE', 'STEP 2 · LOSS MASK', 'STEP 3 · SHIFT'][si], title: titles[si], accent: COL.amber });
    const st0 = m.states[k];
    if (si === 0) wrap(g, `apply_chat_template 把“用户问 / 助手答”拼成一串带特殊标记的文字，再用 Qwen3 的 BPE 分词器切成 ${m.ids.length} 个词元。紫色是模板标记，青色是问题，橙色是回答。非思考模式会自动插入一个空的 <think></think>。`, ex + 14, ey + 66, ew - 28, 20, { size: 12.5, color: COL.ink2 });
    else if (si === 1) this.maskCompare(g, ex + 14, ey + 60, ew - 28, th - 76, st0, m, k, env);
    else wrap(g, `输入是前 ${R.rowLen} 个词元，标签是后 ${R.rowLen} 个；提示部分的标签被换成 −100（PyTorch 的 cross_entropy 会跳过它们）。下面每个词元下方是它的标签。`, ex + 14, ey + 66, ew - 28, 20, { size: 12.5, color: COL.ink2 });
    // 词元
    const top = P ? 470 : th + 40;
    const rows = rowLayout(g, R, k, 0, top + 18, W, { tile: 36, th: 40, gap: 5, lineGap: si === 0 ? 28 : 40 });
    text(g, si === 0 ? '每个词元下面是它在词表里的编号（词表 151,936）' : si === 1 ? '青色下划线 = 这个词元是计入损失的预测目标' : '每个词元下面：它这个位置的标签（下一个词元，或 −100）', 0, top, { size: 10.5, color: COL.dim });
    for (const r of rows) {
      const counted = r.j >= 1 && m.sftMask[r.j - 1];
      tileBase(g, r, { fill: hexA(ROLE_COL[r.role], 0.08), stroke: hexA(ROLE_COL[r.role], 0.35) });
      if (si === 0) text(g, m.ids[r.j], r.x + r.w / 2, r.y + r.h + 14, { size: 9, kind: 'mono', color: COL.dim, align: 'center' });
      if (si === 1 && counted) { g.fillStyle = COL.cyan; g.fillRect(r.x + 2, r.y + r.h + 4, r.w - 4, 3); }
      if (si === 1 && r.j >= 1 && !counted) text(g, '不算', r.x + r.w / 2, r.y + r.h + 15, { size: 9, color: COL.faint, align: 'center' });
      if (si === 2 && r.j < R.rowLen) {
        const lab = m.sftMask[r.j] ? short(R.tok(k, r.j + 1)) : '−100';
        text(g, lab, r.x + r.w / 2, r.y + r.h + 16, { size: 10, color: m.sftMask[r.j] ? COL.amber : COL.faint, align: 'center', kind: lab === '−100' || lab.startsWith('⟨') ? 'mono' : 'sans', max: r.w });
      }
      env.hit(r.x, r.y, r.w, r.h, { tip: `<span class="k">第 ${r.j} 个词元 · ${{ user: '用户的问题', answer: '回答', tpl: '模板' }[r.role]}</span><b>${esc(disp(R.tok(k, r.j)))}</b>　编号 <span class="v">${m.ids[r.j]}</span>${r.j >= 1 ? `<br>${counted ? '是计入损失的目标' : '不计入损失'}，模型猜中它的概率 ${fmtP(st0.p[r.j - 1])}` : ''}` });
    }
  }

  // 两种算法的对比：损失、梯度（原模型上、第 1 步）
  maskCompare(g, x, y, w, h, st0, m, k, env) {
    const items = [['只算回答（SFT）', st0.lossSft, COL.cyan], ['全都算（像预训练）', st0.lossPre, COL.violet]];
    const mx = Math.max(st0.lossSft, st0.lossPre);
    items.forEach(([lab, v, c], i) => {
      const yy = y + 14 + i * 34;
      text(g, lab, x, yy, { size: 11.5, color: COL.ink2 });
      rr(g, x + 128, yy - 10, (w - 200) * (v / mx), 12, 3);
      g.fillStyle = c;
      g.fill();
      text(g, `损失 ${v.toFixed(3)}`, x + w, yy, { size: 11, kind: 'mono', color: COL.ink, align: 'right' });
    });
    if (k === 0) {
      // 每层的梯度范数：两种算法（都在原模型上）
      const pc = m.pretrainCompare.gradNorms, sf = m.steps[0].gradNorms;
      const NL = m.model.layers;
      const layerNorm = (gn, L) => Math.sqrt(Object.entries(gn).filter(([n]) => n.startsWith(`model.layers.${L}.`)).reduce((a, [, v]) => a + v * v, 0));
      const a = [], b = [];
      for (let L = 0; L < NL; L++) { a.push(layerNorm(sf, L)); b.push(layerNorm(pc, L)); }
      const top = Math.max(...a, ...b);
      const cy = y + 92, chh = h - 104;
      text(g, '每一层的梯度范数（原模型上）', x, cy - 6, { size: 10.5, color: COL.dim });
      const bw = w / NL;
      for (let L = 0; L < NL; L++) {
        const h1 = (a[L] / top) * chh, h2 = (b[L] / top) * chh;
        g.fillStyle = hexA(COL.violet, 0.55);
        g.fillRect(x + L * bw + bw * 0.5, cy + chh - h2 + 6, bw * 0.4, h2);
        g.fillStyle = COL.cyan;
        g.fillRect(x + L * bw + bw * 0.1, cy + chh - h1 + 6, bw * 0.4, h1);
        env.hit(x + L * bw, cy, bw, chh + 6, { tip: `<span class="k">第 ${L} 层</span>只算回答 <span class="v">${a[L].toFixed(2)}</span><br>全都算 <span class="a">${b[L].toFixed(2)}</span>` });
      }
      text(g, `合计 ${m.steps[0].gradNorm.toFixed(1)} vs ${m.pretrainCompare.total.toFixed(1)}`, x + w, cy - 6, { size: 10, kind: 'mono', color: COL.dim, align: 'right' });
    } else wrap(g, '遮罩决定了梯度从哪里来：只有回答位置的误差会反向传播回去。（第 1 步时两种算法的逐层梯度对比，可以退回第 1 步看。）', x, y + 92, w, 18, { size: 11.5, color: COL.dim });
  }
}
