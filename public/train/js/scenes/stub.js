// 临时占位视图（开发用）
import { COL, text, card } from '../draw.js';

export class Stub {
  constructor(app, R, name) { this.app = app; this.R = R; this.name = name; }
  layout() {}
  focus() { return { x: 0, y: 0, w: 800, h: 500 }; }
  draw(g, st) {
    card(g, 0, 0, 800, 500, { title: this.name, eyebrow: 'TODO' });
    text(g, JSON.stringify(st.step), 20, 80, { size: 13, color: COL.dim, kind: 'mono' });
  }
}
