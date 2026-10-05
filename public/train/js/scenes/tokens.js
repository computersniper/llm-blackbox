// D3 · 批次（唐宋诗小模型）：取一行 / 字 → 编号 / 错开一位。Qwen3 那一章的进料在 3D 机器里（../qwen3d/）。
import { COL, text, rr, card, hexA, wrap, fmtInt, fmtP, clamp, ease, seg, arrow, measure, line, dot, sciSup } from '../draw.js';
import { rowLayout, tileBase, disp } from './row.js';
import { esc } from '../../../js/ui.js';
import { isEn, L, compact } from '../lang.js';

export class Tokens {
  constructor(app, R) { this.app = app; this.R = R; }

  layout(env) {
    this.portrait = env.portrait;
    this.W = this.portrait ? 380 : 1100;
    this.bounds = this.portrait ? { x: -6, y: -6, w: 392, h: 1000 } : { x: -12, y: -12, w: 1124, h: 680 };
  }

  focus() {
    if (!this.portrait) return this.bounds;
    const y0 = this.R.kind === 'tiny' ? 176 : 216;
    const y1 = (this.bottom || 900) + 24;
    return { x: -6, y: y0, w: 392, h: y1 - y0 };
  }

  draw(g, st, env) {
    this.drawTiny(g, st, env);
  }

  /* ---------------------------------------------------------------- 小模型 */

  drawTiny(g, st, env) {
    const R = this.R, k = st.k, s = st.step, m = R.D.meta, P = this.portrait, W = this.W;
    const sub = s.sub;
    const subs = ['row', 'ids', 'shift'];
    const si = subs.indexOf(sub);
    // 左上：批次张量的形状（第 0 行是真实内容，其余 63 行没有导出）
    const bx = 0, by = 0, bw = P ? 380 : 330, bh = P ? 170 : 236;
    card(g, bx, by, bw, bh, { eyebrow: `TENSOR · x [${m.train.batch} × ${m.train.seq + 1}]`, title: L('一个批次', 'One batch'), accent: COL.blue });
    const gx = bx + 14, gy = by + 56, gw = bw - 28, gh = bh - 90;
    const rows = m.train.batch, cols = m.train.seq + 1;
    const cw = gw / cols, ch = gh / rows;
    g.fillStyle = 'rgba(107,155,255,0.07)';
    g.fillRect(gx, gy, gw, gh);
    g.fillStyle = hexA(COL.amber, 0.85);
    g.fillRect(gx, gy, gw * (si === 0 ? ease(seg(st.p, 0, 0.8)) : 1), Math.max(2, ch));
    g.strokeStyle = COL.line2;
    g.strokeRect(gx, gy, gw, gh);
    text(g, L('← 第 0 行（下面画出的就是它）', '← row 0 (drawn below)'), gx, gy + gh + 16, { size: 10, color: COL.amber });
    text(g, L(`其余 ${rows - 1} 行内容没有导出`, `other ${rows - 1} rows not exported`), gx + gw, gy + gh + 16, { size: 10, color: COL.faint, align: 'right' });
    // 右上 / 下：讲解
    const ex = P ? 0 : 350, ey = P ? 186 : 0, ew = P ? 380 : W - 350;
    card(g, ex, ey, ew, P ? 130 : bh, { eyebrow: ['STEP 1 · SAMPLE', 'STEP 2 · TOKENIZE', 'STEP 3 · SHIFT'][si], title: (isEn ? ['Take a row from the corpus', 'Characters → IDs', 'Shift by one: inputs and targets'] : ['从语料里取一行', '字 → 编号', '错开一位：输入和目标'])[si], accent: COL.cyan });
    const lines = isEn ? [
      `The training corpus is one long ribbon of ${compact(m.corpus.chars.train, 1)} characters (poems separated by ⏎). Each step picks ${m.train.batch} random poem openings and takes ${cols} consecutive characters from each.`,
      `Character-level tokenization: each character becomes its ID in the vocabulary. The vocabulary is sorted by frequency, so smaller IDs are more common: “，” = 1, “。” = 2. Real Qwen3 uses 150K BPE tokens.`,
      `The input is the first ${m.train.seq} characters and the target is the last ${m.train.seq}: position i sees characters 0…i and must guess character i+1. ${m.train.seq} questions per row, ${fmtInt(m.train.tokensPerStep)} per batch.`,
    ] : [
      `训练语料是一条 ${(m.corpus.chars.train / 1e4).toFixed(0)} 万字的长带子（诗与诗之间用 ⏎ 隔开）。每一步随机挑 ${m.train.batch} 个诗的开头，各取连续 ${cols} 个字。`,
      `字级分词：每个字换成它在词表里的编号。这个词表按字频排序，所以编号越小越常见：「，」= 1，「。」= 2。真实的 Qwen3 用 15 万个 BPE 词元。`,
      `输入是前 ${m.train.seq} 个字，目标是后 ${m.train.seq} 个字：位置 i 看到第 0…i 个字，要猜第 i+1 个。一行 ${m.train.seq} 道题，一批 ${fmtInt(m.train.tokensPerStep)} 道。`,
    ];
    if (isEn && P) wrap(g, lines[si], ex + 14, ey + 62, ew - 28, 16, { size: 11.5, color: COL.ink2, maxLines: 4 });
    else wrap(g, lines[si], ex + 14, ey + 66, ew - 28, 20, { size: 12.5, color: COL.ink2 });
    // 主体：65 个字块
    const top = P ? 336 : 256;
    const tile = P ? 30 : 38;
    const rows2 = rowLayout(g, R, k, 0, top + 16, W, { tile, th: si === 1 ? tile + 16 : tile + 4, gap: 4, lineGap: si === 2 ? 30 : 12 });
    this.bottom = top + 16 + rows2.height + (si === 2 ? 30 : 16);
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
      env.hit(r.x, r.y, r.w, tile + 4, { tip: L(`<span class="k">第 ${r.j} 个字</span><b>${esc(r.s)}</b>　编号 <span class="v">${R.tokId(k, r.j)}</span>${r.j < R.rowLen ? `<br>目标（下一个字）：<b>${esc(disp(R.tok(k, r.j + 1)))}</b>` : ''}`, `<span class="k">character ${r.j}</span><b>${esc(r.s)}</b> · ID <span class="v">${R.tokId(k, r.j)}</span>${r.j < R.rowLen ? `<br>target (next character): <b>${esc(disp(R.tok(k, r.j + 1)))}</b>` : ''}`) });
    }
    if (isEn) text(g, `First ${R.rowLen + 1} characters of row 0 (${cols} per row)${si === 2 ? ' · the small orange character under each one is the next character it must guess' : ''}`, 0, top, { size: 10.5, color: COL.dim, max: W });
    else text(g, `第 0 行的前 ${R.rowLen + 1} 个字（一行共 ${cols} 个）${si === 2 ? '　·　每个字下面的橙色小字是它要猜的下一个字' : ''}`, 0, top, { size: 10.5, color: COL.dim });
  }
}
