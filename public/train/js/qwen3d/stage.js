// 第三章的 3D 舞台：引擎（和推理页共用 ../../../js/stage/engine.js：渲染、泛光、跟随 / 自由镜头、拾取、CSS2D 标签）
// + 机器（machine.js）+ 浮层（board.js）。每帧向页面要当前的调试步骤（state 回调），机器和镜头都按它算。
// 点层板换层（进入“一层之内”），点权重面板看那个张量（“一个张量”），点回答的位置换焦点。
import { Engine } from '../../../js/stage/engine.js';
import { QwenMachine } from './machine.js';
import { QwenBoard } from './board.js';
import { OP_TENSORS, OP_TENSORS_BWD, UPD_TENSORS } from './timeline.js';

export class QwenStage {
  constructor(host, stageEl, R, app, { state, onTip, onFreeChange }) {
    this.R = R;
    this.app = app;
    this.state = state;
    this.onTip = onTip;
    this.E = new Engine(host, { onFrame: (dt, t) => this.frame(dt, t), onHover: (info, e) => this.hover(info, e), onPick: (info) => this.pick(info), onFreeChange });
    this.M = new QwenMachine(this.E, R, app);
    this.B = new QwenBoard(stageEl, { onTip, E: this.E, R, app });
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

  frame(dt, t) {
    const st = this.state(dt, t);
    if (!st) return;
    const s = st.step;
    this.M.setSel(s.t && (st.view === 'q-mat' || st.view === 'q-param') ? s.t : null);
    // 拆开一层要用的真实权重分布：进到一层之内再取（推理页导出的缩略图，约 380 KB）
    if (st.depth >= 3 && !this.R.thumbs) this.R.loadThumbs()?.catch(() => {});
    this.M.update(st, dt, t);
    this.B.update(st, t, this.M.F);
    const cam = this.M.camera(st);
    if (!this.started) {
      this.started = true;
      this.E.setView(cam.pos.clone().sub(cam.look).multiplyScalar(1.7).add(cam.look), cam.look, { snap: true });
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
    const tl = this.app.tl;
    if (!info || !tl) return;
    if (info.type === 'slab' || info.type === 'gmeter') return this.pickLayer(info.L);
    if (info.type === 'panel') return this.pickTensor(info.t);
    if (info.type === 'emb') return this.pickTensor('embed');
    if (info.type === 'lm') return this.pickTensor('embed', 'head');
    if ((info.type === 'bar' || info.type === 'tile') && this.R.ans.includes(info.i ?? info.j)) return this.pickPos(info.i ?? info.j);
  }

  // 点了一层：换成看这一层；还没到“一层之内”就进去（反向 / 更新时停在反向的第一个算子，否则停在前向的 RMSNorm）
  pickLayer(L) {
    const tl = this.app.tl;
    tl.pause();
    this.E.exitFree();
    this.app.sfx('dive');
    const ph = tl.step.ph;
    tl.setLayer(L);
    if (tl.depth < 3) {
      const back = ph === 'bwd' || ph === 'upd';
      tl.jump(3, (b) => (back ? b.ph === 'bwd' && b.sub === 'down' : b.ph === 'fwd' && b.sub === 'ln1'));
    }
  }

  // 点了一个张量：跳到“一个张量”，停在当前环节里用到它的那一步
  pickTensor(t, op) {
    const tl = this.app.tl, s = tl.step;
    tl.pause();
    this.E.exitFree();
    this.app.sfx('dive');
    let ph = s.ph === 'bwd' ? 'bwd' : s.ph === 'upd' ? 'upd' : 'fwd';
    if (ph === 'upd' && !UPD_TENSORS.includes(t)) ph = 'fwd';
    const d = tl.depth === 5 && ph === 'upd' ? 5 : 4;
    const pred = ph === 'upd' ? (b) => b.ph === 'upd' && b.t === t && (d < 5 || b.mi === (s.mi || 'g'))
      : (b) => b.ph === ph && b.t === t && (!op || b.sub === op) && (ph === 'fwd' ? OP_TENSORS : OP_TENSORS_BWD)[b.sub]?.includes(t);
    tl.jump(d, pred);
  }

  // 点了回答的一个位置：逻辑透镜读数、注意力光束都换成看它；在逐个位置看损失时直接跳过去
  pickPos(i) {
    const tl = this.app.tl;
    this.app.ctx.qPos = i;
    this.app.sfx('click');
    if (tl.depth >= 3 && tl.step.ph === 'loss') { tl.pause(); tl.seek((b) => b.ph === 'loss' && b.i === i); }
    else tl.emit('step');
  }
}
