// 一行训练数据的“字块”排版：小模型每个字一样宽；Qwen3 的词元按文字宽度排。
// 第 j 块是输入里的第 j 个词元；j ≥ 1 时，它同时是位置 j−1 的预测目标。
import { COL, text, rr, measure, hexA } from '../draw.js';

export const disp = (s) => (s === '\n' ? '↵' : s === '\n\n' ? '↵↵' : String(s).replace(/\n/g, '↵'));
export const short = (s) => disp(s).replace(/^<\|(.+)\|>$/, '⟨$1⟩');
export const ROLE_COL = { user: '#5ee4f0', answer: COL.amber, tpl: COL.violet, char: COL.ink };

export function rowLayout(g, R, k, x0, y0, w, { tile = 34, gap = 4, th = 44, lineGap = 14, n = null } = {}) {
  const N = n ?? R.rowLen + 1;
  const out = [];
  let x = x0, y = y0;
  for (let j = 0; j < N; j++) {
    const s = short(R.tok(k, j));
    const tw = R.kind === 'tiny' ? tile : Math.max(tile * 0.8, measure(g, s, 13, /^⟨/.test(s) ? 'mono' : 'sans') + 14);
    if (x + tw > x0 + w + 0.5 && x > x0) { x = x0; y += th + lineGap; }
    out.push({ x, y, w: tw, h: th, s, j, role: R.kind === 'tiny' ? (R.isSep(k, j) ? 'tpl' : 'char') : R.role(j) });
    x += tw + gap;
  }
  out.height = y + th - y0;
  return out;
}

export function tileBase(g, r, { fill = 'rgba(255,255,255,0.03)', stroke = COL.line2, color = null, size = null, dim = false } = {}) {
  rr(g, r.x, r.y, r.w, r.h, 7);
  g.fillStyle = fill;
  g.fill();
  g.strokeStyle = stroke;
  g.lineWidth = 1;
  g.stroke();
  const c = color || (r.role === 'char' ? COL.ink : ROLE_COL[r.role]);
  const mono = /^⟨/.test(r.s);
  const sz = size || (mono ? 10 : r.s.length > 2 ? 12 : 15);
  text(g, r.s, r.x + r.w / 2, r.y + r.h / 2 + sz * 0.35 - 3, { size: sz, color: dim ? hexA(c.startsWith('#') ? c : '#e9eff9', 0.35) : c, align: 'center', kind: mono ? 'mono' : 'sans', weight: r.role === 'char' ? 500 : 400 });
}
