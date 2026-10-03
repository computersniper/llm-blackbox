// 玻璃小模型这一章的 3D 舞台：引擎（和推理页共用 ../../../js/stage/engine.js：渲染、泛光、跟随 / 自由镜头、拾取、CSS2D 标签）
// + 机器（machine.js）+ 两块 2D 浮层（board.js）。每帧向页面要当前的调试步骤（frame 回调），机器和镜头都按它算。
import { Engine } from '../../../js/stage/engine.js';
import { GlassMachine } from './machine.js';
import { GlassBoard } from './board.js';
import { defaultParam, adamAt } from '../glass/math.js';
import { fnum } from '../glass/heat.js';
import { L } from '../lang.js';

export class GlassStage {
  constructor(host, stageEl, D, app, { state, onTip, onFreeChange, scrub }) {
    this.D = D;
    this.app = app;
    this.state = state;
    this.onTip = onTip;
    this.host = host;
    this.E = new Engine(host, { onFrame: (dt, t) => this.frame(dt, t), onHover: (info, e) => this.hover(info, e), onPick: (info) => this.pick(info), onFreeChange });
    this.M = new GlassMachine(this.E, D, app);
    this.B = new GlassBoard(stageEl, { onTip, scrub, E: this.E });
    this.lastView = '';
    this.lastDepth = -1;
    this.started = false;
  }

  get free() { return this.E.free; }
  exitFree() { this.E.exitFree(); }
  setInsets(r, b) { this.E.setInsets(r, b); }

  set active(on) {
    this.E.active = on;
    if (!on) { this.B.hide(); this.onTip(null); }
  }
  get active() { return this.E.active; }

  // 这一步选中的参数：点过的那个（同一个张量时），否则这个张量里默认的那个
  selOf(st) {
    const s = st.step, g = this.app.ctx.gsel, D = this.D;
    if (!s?.t) return null;
    return g != null && D.locate(g)?.p.name === s.t ? g : defaultParam(D, s.t);
  }

  frame(dt, t) {
    const st = this.state(dt, t);
    if (!st) return;
    const M = this.M, D = this.D;
    const sel = st.view === 'g-param' ? this.selOf(st) : null;
    M.setSel(sel);
    M.update(st, dt, t);
    this.B.update(st, D, this.app.ctx, t, M.p);
    // 选中参数的读数
    if (sel != null && D.has('w', st.k) && D.has('f', st.k)) {
      const s = st.step;
      let html;
      if (s.ph === 'upd') { const a = adamAt(D, st.k, sel); html = s.mi === 'write' || s.mi === 'dw' ? `w ${fnum(a.w, 4)} → <b>${fnum(a.w1, 5)}</b>` : `w = ${fnum(a.w, 4)}`; }
      else if (s.ph === 'bwd') html = `∂L/∂w = <b>${fnum(D.G(st.k)[sel], 3)}</b>`;
      else html = `w = <b>${fnum(D.W(st.k)[sel], 4)}</b>`;
      M.selLabel(html);
    } else M.selLabel(null);
    const cam = M.camera(st);
    if (!this.started) {
      // 第一次：从远处推进来
      this.started = true;
      this.E.setView(cam.pos.clone().sub(cam.look).multiplyScalar(1.8).add(cam.look), cam.look, { snap: true });
    }
    const same = st.view === this.lastView && st.depth === this.lastDepth;
    this.E.setView(cam.pos, cam.look, { speed: same ? 2.4 : 2.0 });
    if (!same) { this.lastView = st.view; this.lastDepth = st.depth; }
  }

  hover(info, e) {
    this.M.hover = info;
    if (!info) return this.onTip(null);
    this.onTip(this.M.tip(info), e);
  }

  pick(info) {
    this.onTip(null);
    if (info?.type === 'w') this.app.pickParam(info.gi, this.app.tl?.depth === 5);
  }
}

export const HINT_3D = () => L('在机器里自由走动', 'Walk around the machine');
