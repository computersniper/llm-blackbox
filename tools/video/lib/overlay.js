// HTML 叠加层：字幕、数据条、章节标记……全部按时间 t 计算出透明度和位移，不依赖 CSS 过渡。
import { clamp, smooth, easeOut } from './cam.js';

export function el(tag, cls, html, parent) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  if (parent) parent.append(e);
  return e;
}

// 一段时间 [t0, t1] 内的可见度：渐入 fin 秒、渐出 fout 秒
export function vis(t, t0, t1, fin = 0.35, fout = 0.35) {
  if (t < t0 || t > t1) return 0;
  return Math.min(fin > 0 ? smooth((t - t0) / fin) : 1, fout > 0 ? smooth((t1 - t) / fout) : 1);
}

// 一组互不重叠的提示（字幕 / 数据条 / 章节）：同一时刻最多两条在交叉淡化
export class CueLayer {
  constructor(parent, cls, cues, { fin = 0.35, fout = 0.3, rise = 10, blur = 0 } = {}) {
    this.parent = parent;
    this.cls = cls;
    this.cues = cues.slice().sort((a, b) => a.t0 - b.t0);
    this.fin = fin; this.fout = fout; this.rise = rise; this.blur = blur;
    this.live = new Map();
  }

  update(t) {
    this.cues.forEach((c, i) => {
      const a = vis(t, c.t0, c.t1, c.fin ?? this.fin, c.fout ?? this.fout);
      let e = this.live.get(i);
      if (a <= 0.001) { if (e) { e.remove(); this.live.delete(i); } return; }
      if (!e) { e = el('div', `${this.cls} ${c.cls || ''}`, c.html, this.parent); this.live.set(i, e); }
      const kIn = easeOut(clamp((t - c.t0) / ((c.fin ?? this.fin) * 1.6)));
      const dy = (1 - kIn) * this.rise;
      e.style.opacity = a.toFixed(4);
      const base = c.transform ?? (this.cls.includes('chapter') || this.cls.includes('corner') ? '' : 'translateX(-50%)');
      e.style.transform = `${base} translateY(${dy.toFixed(2)}px)`;
      if (this.blur) e.style.filter = a < 0.999 ? `blur(${((1 - a) * this.blur).toFixed(2)}px)` : '';
    });
  }
}
