// 训练视频的分镜表：每个镜头的时间、机器状态、机位、字幕、术语标签、叠加层、配乐事件。
// 节拍和推理视频一样：96 BPM（一拍 0.625 秒，一小节 2.5 秒），段落都卡在小节线上；配乐（compose_train.py）读同一张段落表和事件表。
// 字幕一屏一行（≲18 字），一个小号术语标签；数字全部从真实训练记录里现取（tools/train/glassbox.py 导出的 public/train/data/glass*）。
//
// 片子讲的是玻璃小模型（2,928 个参数）怎样用 200 步 AdamW 学会李白《静夜思》：
//   出厂状态（随机的小数，只会瞎猜）→ 喂一批 → 前向 → 损失 → 反向 → 更新（全片唯一一次逐数计算：一个参数的 AdamW）
//   → 重复 200 步（损失下降、预测变绿）→ 最难的「月→，」→ 为什么要随机初始化 → 一个参数的一生 → 训练页录屏 + 二维码
import { THREE } from '../lib/engine.js';
import { path, blendCam, orbit, handheld, clamp, lerp, seg, smooth, smoother, easeOut, easeInOut, v3 } from '../lib/cam.js';
import { gView } from '/public/train/js/glass/timeline.js';
import { BOUNDS, OP_RECT, BLOCK, cellX, cellY, SHELF, shelfX, shelfZ, GAUGE, WHEEL } from '/public/train/js/glass3d/layout.js';
import { adamAt, defaultParam } from '/public/train/js/glass/math.js';

export const BPM = 96, BEAT = 60 / BPM, BAR = 4 * BEAT;
const B = (bar, beat = 0) => bar * BAR + beat * BEAT;

// 段落：名字、小节数、能量（配乐用）
const PLAN = [
  ['open', 5, 0.12], ['init', 6, 0.3], ['batch', 4, 0.4], ['fwd', 5, 0.5], ['loss', 3, 0.42], ['bwd', 5, 0.55],
  ['upd', 9, 0.3], ['run', 10, 0.72], ['hard', 7, 0.85], ['zero', 3, 0.3], ['life', 5, 0.5], ['end', 9, 0.22],
];
// 顶部的章节进度条
const CHAPTERS = ['出厂状态', '喂一批', '前向', '损失', '反向', '更新', '重复 200 步'];

const m = (s) => `<span class="m">${s}</span>`;
const q = (s) => `<q>${s}</q>`;
const pc = (p, d = 0) => `${(p * 100).toFixed(d)}%`;
const sgn = (v, d) => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(d);
const num = (v, d) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
const V3 = (a) => (a.isVector3 ? a : v3(a[0], a[1], a[2]));
const cam = (p, l, fov = 34) => ({ pos: V3(p), look: V3(l), fov });
const farther = (c, k) => ({ ...c, pos: c.look.clone().add(c.pos.clone().sub(c.look).multiplyScalar(k)) });
const lift = (c, dx = 0, dy = 0, dz = 0) => ({ ...c, pos: c.pos.clone().add(v3(dx, dy, dz)), look: c.look.clone().add(v3(dx, dy, dz)) });

// 机器的状态：和训练页调试器里的“深度 + 步骤 + 进度”一样，交给 GlassMachine.update（playing = true：进度直接用 p）
function G(depth, step, p, o = {}) {
  return { step, p: clamp(p), depth, k: step.k ?? 0, i: 0, playing: true, view: o.view ?? gView(depth, step), ...o };
}

export function buildScore(D, cap = null) {
  const subs = [], terms = [], events = [], sections = [], progs = [];
  const sub = (t0, t1, html) => subs.push({ t0, t1, html });
  const term = (t0, t1, html) => terms.push({ t0, t1, html });
  const ev = (t, type, o = {}) => events.push({ t, type, ...o });
  const prog = (t, i) => progs.push({ t, i });

  const SEC = {};
  let bb = 0;
  for (const [name, bars, energy] of PLAN) {
    SEC[name] = { b0: bb, bars, t0: B(bb), t1: B(bb + bars) };
    sections.push({ name, t0: B(bb), t1: B(bb + bars), bars, energy });
    bb += bars;
  }
  const end = B(bb);
  const shots = [];
  const shot = (name, fn) => shots.push({ name, t0: SEC[name].t0, t1: SEC[name].t1, fn });
  const endCam = new Map();
  const prevCam = (name, fb) => endCam.get(name) || fb;

  /* ------------------------------------------------------------ 真实数字 */
  const T = D.T, V = D.V, S = D.S;
  const ch = (v) => D.ch(v);
  const LN20 = Math.log(V);
  const probsOK = (k) => Array.from({ length: T }, (_, p) => D.probs(k, p, D.fixed[p + 1]));
  const p0 = probsOK(0), pL = probsOK(D.NF - 1);
  const evalFirst = D.evalLoss[0], evalLast = D.meta.train.finalEval;
  // 「月→，」：固定样例的第 4 个位置（看到“举头望明月”，下一个是“，”）
  const PM = 4;
  const kHard = D.frameOf(80);                // 走完 80 步
  const pHard = { ans: D.probs(kHard, PM, D.fixed[PM + 1]), guang: D.probs(kHard, PM, D.chars.indexOf('光')) };
  const pHardEnd = D.probs(D.NF - 1, PM, D.fixed[PM + 1]);
  const attAt = (k, h, i) => Array.from({ length: i + 1 }, (_, j) => D.att(k, h, i, j));
  const att81 = attAt(kHard, 0, PM), attEnd = attAt(D.NF - 1, 0, PM);
  const argmax = (a) => a.reduce((bi, v, i) => (v > a[bi] ? i : bi), 0);
  const jEnd = argmax(attEnd), j81 = argmax(att81);
  // 唯一一次逐数计算：嵌入表里「月」那一行变化最大的那个数（exact 记录，每一步都有 float32 原值，能逐位对上）
  const GI = defaultParam(D, 'E');
  const LC = D.locate(GI);
  const A1 = adamAt(D, 0, GI);
  const EX = D.exact(), XJ = EX.pos.get(GI);
  const life = Array.from({ length: S + 1 }, (_, t) => EX.w[t * EX.n + XJ]);
  const zLoss = D.zLoss[S - 1];
  const STEPS_TOTAL = D.P * S;

  /* ------------------------------------------------------------ 机位工具 */
  const WIDE = [0.5, 0.32, 1], MID = [0.36, 0.26, 1], TOP = [0.25, 0.85, 1];
  const whole = [BOUNDS.x0, BOUNDS.x1, BOUNDS.y0, BOUNDS.y1 + 0.8, -2.2, 1.4];
  const fit = (M, rect, dir, margin = 1.04, minD = 2) => { const r = M.fitBox(rect, V3(dir), margin, minD); return { pos: r.pos, look: r.look, fov: 34 }; };
  // 选中的那个参数（「月」那一行的方块）在机器上的位置
  const EB = BLOCK.E;
  const SELP = v3(cellX(EB, LC.j), cellY(EB, LC.i), 0.03);

  // 快进：电影时间 → 训练第几步（连续）→ 帧号 + 帧内插值
  const kOfStep = (s) => {
    const t = clamp(s, 1, S) - 1;
    let k = 0;
    while (k < D.NF - 1 && D.FR[k + 1] <= t) k++;
    if (k >= D.NF - 1) return { k: D.NF - 1, f: 0 };
    return { k, f: (t - D.FR[k]) / (D.FR[k + 1] - D.FR[k]) };
  };
  // 快进时的机器：前向已经走完（概率架立着、损失管有读数），没有脉冲；权重在相邻两帧之间插值
  const FF = (next) => ({ feed: 1, shift: 1, fwd: 10, bwd: -1, loss: 1, lossUpTo: 8, lossPos: -1, lossMean: 1, clip: 0, upd: {}, next });
  const runState = (s) => {
    const { k, f } = kOfStep(s);
    return { st: G(1, { k, ph: 'run' }, 0.45, { F: FF(f) }), frac: { k, f } };
  };

  /* ------------------------------------------------------------ 开场 */
  const Q1 = 'AI 是怎么学会说话的？';
  const OPEN = { q0: 0.7, dt: 0.13, ans: 4.6, out: 7.0, title: 7.4, titleOut: 12.1 };
  {
    const n = [...Q1].length;
    for (let i = 0; i < n; i++) if (Q1[i] !== ' ') ev(OPEN.q0 + i * OPEN.dt, 'key', { k: i });
    ev(OPEN.ans, 'ans');
    ev(OPEN.title, 'hit');
    shot('open', (lt, t, { M }) => {
      const far = cam([-26, 30, 62], [3.6, 10.5, 0], 30);
      const near = fit(M, whole, [-0.25, 0.28, 1], 1.18);
      return {
        st: G(0, { ph: 'init', sub: 'model' }, 0),
        cam: () => handheld(blendCam(far, farther(near, 1.12), smoother(seg(t, 2.5, 10.5))), t, 0.006),
        fade: 1 - 0.62 * smooth(seg(t, 6.6, 9.2)),
        lbl: 'none',
        bloom: 0.3,
        ov: {
          q: { a: smooth(seg(t, OPEN.q0 - 0.3, OPEN.q0)) * (1 - smooth(seg(t, OPEN.out, OPEN.out + 0.5))), n: Math.floor((t - OPEN.q0) / OPEN.dt) + 1, ans: smooth(seg(t, OPEN.ans, OPEN.ans + 0.5)) },
          title: { a: smooth(seg(t, OPEN.title, OPEN.title + 0.8)) * (1 - smooth(seg(t, OPEN.titleOut, OPEN.titleOut + 0.4))) },
          band: 0,
        },
      };
    });
  }

  /* ------------------------------------------------------------ 出厂状态 */
  {
    const T0 = SEC.init.t0, T1 = SEC.init.t1;
    prog(T0 + 0.3, 0);
    sub(T0 + 0.3, T0 + 3.9, `这是一个很小的 AI：${m(D.P.toLocaleString('en-US'))} 个参数`);
    term(T0 + 0.4, T0 + 3.9, '每个方块 = 一个参数');
    sub(T0 + 4.1, T0 + 7.7, '每个参数就是一个数：颜色是正负');
    term(T0 + 4.2, T0 + 7.7, '蓝 = 负 · 琥珀 = 正 · 越厚越大');
    sub(T0 + 7.9, T0 + 11.5, '刚出厂时，全是随机的小数');
    term(T0 + 8.0, T0 + 11.5, `初始化 · 正态分布 N(0, 0.02${'²'})`);
    sub(T0 + 11.7, T1 - 0.2, '这时候，它什么都还不会');
    ev(T0 + 7.9, 'reveal');
    shot('init', (lt, t, { M }) => {
      const wide = fit(M, whole, WIDE, 1.03);
      // 第二句推近到注意力那一层的 W_q / W_k，看清方块的颜色和厚度
      const close = fit(M, [1.6, 8.2, 3.4, 5.8, -0.2, 0.5], [0.42, 0.34, 1], 1.02);
      const k1 = smoother(seg(lt, 3.6, 6.6)) * (1 - smoother(seg(lt, 7.4, 10.4)));
      const live = blendCam(orbit(wide, lerp(-4, 4, lt / 15), 0, 1), close, k1);
      return {
        st: G(0, { ph: 'init', sub: lt < 7.9 ? 'model' : 'hist' }, lt < 7.9 ? 1 : seg(lt, 7.9, 10.5)),
        cam: () => blendCam(prevCam('open', live), live, smoother(seg(lt, 0, 3.2))),
        fade: 0.38 * (1 - smooth(seg(lt, 0, 2.2))),
        lbl: lt > 3.4 && lt < 11 ? 'weights' : 'none',
        bloom: 0.3,
        ov: { hist: { a: smooth(seg(lt, 8.0, 8.8)) * (1 - smooth(seg(lt, 11.6, 12.3))), k: easeOut(seg(lt, 8.2, 10.4)) } },
      };
    });
  }

  /* ------------------------------------------------------------ 喂一批 */
  {
    const T0 = SEC.batch.t0, T1 = SEC.batch.t1;
    prog(T0 + 0.2, 1);
    sub(T0 + 0.3, T0 + 3.4, `教材只有一首诗：李白的《${D.meta.corpus.title}》`);
    term(T0 + 0.4, T0 + 3.4, `语料 · ${D.stream.length} 个字首尾相接`);
    sub(T0 + 3.6, T0 + 6.6, `每一步切下 ${m(D.B)} 段，每段 ${m(T)} 个字`);
    term(T0 + 3.7, T0 + 6.6, `一批 = ${D.B} 段 × ${T} 个字`);
    sub(T0 + 6.8, T1 - 0.2, '答案就是：每个字的下一个字');
    term(T0 + 6.9, T1 - 0.2, `${D.fixed.slice(0, T).map(ch).join('')} → ${D.fixed.slice(1).map(ch).join('')}`);
    ev(T0 + 0.6, 'whoosh');
    for (let b = 0; b < D.B; b++) ev(T0 + 0.9 + b * 0.42, 'drop', { b });
    ev(T0 + 6.9, 'shift');
    shot('batch', (lt, t, { M }) => {
      const p = lt < 4.4 ? 0.6 * seg(lt, 0.4, 4.4) : lt < 6.6 ? 0.6 : 0.6 + 0.4 * seg(lt, 6.8, 9.2);
      const c0 = fit(M, [-6.2, 0.4, -0.5, 3.4, -2.6, 0.8], [0.22, 0.24, 1], 1.12);
      const c1 = fit(M, [-3.2, 0.2, -0.4, 1.6, -2.6, 0.9], [0.18, 0.5, 1], 1.04);
      const live = blendCam(c0, c1, smoother(seg(lt, 6.2, 8.4)));
      return {
        st: G(2, { k: 0, ph: 'batch' }, p),
        cam: () => blendCam(prevCam('init', live), live, smoother(seg(lt, 0, 2.6))),
        lbl: 'batch',
        bloom: 0.3,
      };
    });
  }

  /* ------------------------------------------------------------ 前向 */
  {
    const T0 = SEC.fwd.t0, T1 = SEC.fwd.t1;
    prog(T0 + 0.2, 2);
    sub(T0 + 0.3, T0 + 3.9, '这 8 个字，一层层往上算');
    term(T0 + 0.4, T0 + 3.9, '前向传播');
    sub(T0 + 4.1, T0 + 7.9, `每个位置给 ${m(V)} 个字各打一个概率`);
    term(T0 + 4.2, T0 + 7.9, 'softmax · 20 个字的概率加起来 = 1');
    sub(T0 + 8.1, T1 - 0.2, `正确答案只有 ${m(pc(Math.min(...p0)))}–${m(pc(Math.max(...p0)))}：等于瞎猜`);
    term(T0 + 8.2, T1 - 0.2, `瞎猜 = 1 / ${V} = 5%`);
    const P0 = T0 + 0.4, P1 = T0 + 7.6;
    for (let o = 0; o < 10; o++) ev(P0 + ((o + 0.5) / 10.6) * (P1 - P0), 'layer', { o, dir: 1 });
    ev(T0 + 8.1, 'reveal');
    shot('fwd', (lt, t, { M }) => {
      const p = seg(t, P0, P1);
      const st = G(2, { k: 0, ph: 'fwd' }, p);
      return {
        st,
        cam: () => {
          const follow = M.camera(st);
          const fol = { pos: follow.pos, look: follow.look, fov: 34 };
          const shelf = fit(M, [...OP_RECT.loss, -1.3, 1.3], TOP, 1.05);
          const live = blendCam(fol, shelf, smoother(seg(t, P1 - 1.2, P1 + 1.4)));
          return blendCam(prevCam('batch', live), live, smoother(seg(lt, 0, 2.0)));
        },
        lbl: 'fwd',
        bloom: 0.3,
      };
    });
  }

  /* ------------------------------------------------------------ 损失 */
  {
    const T0 = SEC.loss.t0, T1 = SEC.loss.t1;
    prog(T0 + 0.2, 3);
    sub(T0 + 0.3, T0 + 3.6, '猜得有多差？算出损失');
    term(T0 + 0.4, T0 + 3.6, '交叉熵 · 每个位置 −ln p');
    sub(T0 + 3.8, T1 - 0.2, `损失 ${m(D.loss[0].toFixed(3))}：和瞎猜一样`);
    term(T0 + 3.9, T1 - 0.2, `瞎猜 = ln ${V} ≈ ${LN20.toFixed(3)}`);
    for (let p = 0; p < T; p++) ev(T0 + 0.4 + p * 0.22, 'drop', { b: p });
    ev(T0 + 3.0, 'reveal');
    shot('loss', (lt, t, { M }) => {
      const st = G(2, { k: 0, ph: 'loss' }, seg(lt, 0.3, 4.6));
      const shelf = fit(M, [...OP_RECT.loss, -1.3, 1.3], TOP, 1.05);
      const gauge = fit(M, [5.0, 10.4, 19.4, 23.4, -1.3, 1.3], [0.4, 0.45, 1], 1.05);
      const live = blendCam(shelf, gauge, smoother(seg(lt, 2.8, 5.0)));
      return {
        st,
        cam: () => blendCam(prevCam('fwd', live), live, smoother(seg(lt, 0, 1.5))),
        lbl: 'loss',
        bloom: 0.3,
      };
    });
  }

  /* ------------------------------------------------------------ 反向 */
  {
    const T0 = SEC.bwd.t0, T1 = SEC.bwd.t1;
    prog(T0 + 0.2, 4);
    sub(T0 + 0.3, T0 + 3.9, '再把误差倒着传回去');
    term(T0 + 0.4, T0 + 3.9, '反向传播');
    sub(T0 + 4.1, T0 + 7.9, '每个参数都算出：该往哪边改');
    term(T0 + 4.2, T0 + 7.9, '梯度 ∂L/∂w');
    sub(T0 + 8.1, T1 - 0.2, '越亮的，改它越管用');
    term(T0 + 8.2, T1 - 0.2, '亮度 = |梯度|');
    const P0 = T0 + 0.4, P1 = T0 + 8.0;
    for (let o = 0; o < 10; o++) ev(P0 + ((o + 0.5) / 10.6) * (P1 - P0), 'layer', { o, dir: -1 });
    shot('bwd', (lt, t, { M }) => {
      const st = G(2, { k: 0, ph: 'bwd' }, seg(t, P0, P1));
      return {
        st,
        cam: () => {
          const follow = M.camera(st);
          const fol = { pos: follow.pos, look: follow.look, fov: 34 };
          const wide = orbit(fit(M, whole, WIDE, 1.03), lerp(0, 6, seg(t, P1 - 1, T1)), 0, 1);
          const live = blendCam(fol, wide, smoother(seg(t, P1 - 1.6, P1 + 1.4)));
          return blendCam(prevCam('loss', live), live, smoother(seg(lt, 0, 2.0)));
        },
        lbl: 'bwd',
        bloom: 0.32,
      };
    });
  }

  /* ------------------------------------------------------------ 更新（全片唯一一次逐数计算） */
  const CALC = {};
  {
    const T0 = SEC.upd.t0, T1 = SEC.upd.t1;
    prog(T0 + 0.2, 5);
    const L = [T0 + 3.8, T0 + 7.6, T0 + 11.2, T0 + 15.0, T0 + 18.6];
    sub(T0 + 0.3, L[0] - 0.2, '最后，每个参数都要挪一小步');
    term(T0 + 0.4, L[0] - 0.2, '更新 · AdamW 优化器');
    sub(L[0], L[1] - 0.2, `放大一个：${q('月')}字向量里的一个数`);
    term(L[0] + 0.1, L[1] - 0.2, `嵌入表 · ${q('月')}那一行的第 ${LC.j} 个数`);
    sub(L[1], L[2] - 0.2, '梯度是负的：把它调大，损失会变小');
    term(L[1] + 0.1, L[2] - 0.2, '梯度 = 往哪边改');
    sub(L[2], L[3] - 0.2, 'AdamW 换算步子：第一步只看方向');
    term(L[2] + 0.1, L[3] - 0.2, '动量 m · 平方均值 v');
    sub(L[3], L[4] - 0.2, `乘上学习率 ${m(A1.lr.toFixed(3))}，往上挪一小步`);
    term(L[3] + 0.1, L[4] - 0.2, '学习率 · 每步挪多远');
    sub(L[4], T1 - 0.2, `${m(D.P.toLocaleString('en-US'))} 个参数一起，各挪一小步`);
    term(L[4] + 0.1, T1 - 0.2, '第 1 步训练完成');
    Object.assign(CALC, { L, A1 });
    ev(L[0], 'focus');
    ev(L[1], 'blip');
    ev(L[2], 'blip');
    ev(L[3] + 0.9, 'pop');
    ev(L[4] + 0.7, 'popAll');
    shot('upd', (lt, t, { M }) => {
      const close0 = cam([SELP.x + 1.3, SELP.y + 1.25, 3.4], [SELP.x + 0.25, SELP.y + 0.25, 0], 34);
      const close1 = cam([SELP.x + 0.75, SELP.y + 0.62, 1.9], [SELP.x + 0.12, SELP.y + 0.34, 0], 34);
      const wide = fit(M, whole, WIDE, 1.03);
      const kIn = smoother(seg(t, T0 + 0.4, L[0] + 0.6));
      const kOut = smoother(seg(t, L[4] - 0.2, L[4] + 2.4));
      let st, sel = null, selLbl = null;
      if (t < L[4]) {
        const mi = t < L[3] ? (t < L[2] ? 'g' : 'bc') : t < L[3] + 1.6 ? 'dw' : 'write';
        const p = mi === 'dw' ? seg(t, L[3] + 0.4, L[3] + 1.5) : mi === 'write' ? seg(t, L[3] + 1.6, L[3] + 3.0) : 0;
        st = G(5, { k: 0, ph: 'upd', sub: 't', t: 'E', mi }, p);
        sel = GI;
        selLbl = t < L[3] + 1.6 ? `w = <b>${A1.w.toFixed(4)}</b>` : `w ${A1.w.toFixed(4)} → <b>${A1.w1.toFixed(4)}</b>`;
      } else st = G(2, { k: 0, ph: 'upd' }, seg(t, L[4] + 0.2, T1 - 0.6));
      const near = blendCam(close0, close1, smoother(seg(t, L[0], L[1])));
      return {
        st, sel, selLbl,
        cam: () => {
          const live = blendCam(blendCam(prevCam('bwd', wide), near, kIn), wide, kOut);
          return handheld(live, t, 0.004);
        },
        dof: { focus: 2.0, range: 1.6, blur: 7 * kIn * (1 - kOut) },
        lbl: t < L[4] ? 'none' : 'none',
        bloom: 0.26,
        ov: { calc: { a: smooth(seg(t, L[0] - 0.2, L[0] + 0.4)) * (1 - smooth(seg(t, L[4] + 0.6, L[4] + 1.3))), t } },
      };
    });
  }

  /* ------------------------------------------------------------ 重复 200 步 */
  // 训练第几步 ↔ 电影时间：前面慢、后面快（前 50 步损失掉得最多）
  const RUN = { t0: SEC.run.t0 + 0.4, t1: SEC.run.t1 - 0.6 };
  const stepAtRun = (t) => 1 + (S - 1) * Math.pow(seg(t, RUN.t0, RUN.t1), 1.45);
  {
    const T0 = SEC.run.t0, T1 = SEC.run.t1;
    prog(T0 + 0.2, 6);
    const ev50 = D.evalLoss[50];
    sub(T0 + 0.3, T0 + 3.9, `把这一步，重复 ${m(S)} 次`);
    term(T0 + 0.4, T0 + 3.9, `每一步换一批新的 ${D.B} 段`);
    sub(T0 + 4.1, T0 + 8.4, '方块越练越厚，图案越来越清楚');
    term(T0 + 4.2, T0 + 8.4, '厚度 = |w|');
    sub(T0 + 8.6, T0 + 13.4, '损失一路往下掉');
    term(T0 + 8.7, T0 + 13.4, `${evalFirst.toFixed(3)} → ${ev50.toFixed(3)}（第 50 步）`);
    sub(T0 + 13.6, T0 + 18.4, `正确答案的概率，从 5% 涨到 ${m(pc(Math.min(...pL)))} 以上`);
    term(T0 + 13.7, T0 + 18.4, '绿色 = 正确答案');
    sub(T0 + 18.6, T1 - 0.2, '只有一个字，迟迟学不会');
    term(T0 + 18.7, T1 - 0.2, `${q('月')}后面接什么？`);
    // 每 10 步一个滴答（配乐）
    for (let s = 10; s <= S; s += 10) {
      const u = Math.pow((s - 1) / (S - 1), 1 / 1.45);
      ev(RUN.t0 + u * (RUN.t1 - RUN.t0), 'tick', { s });
    }
    ev(T0, 'whoosh');
    shot('run', (lt, t, { M }) => {
      const s = stepAtRun(t);
      const r = runState(s);
      const wide = fit(M, whole, WIDE, 1.03);
      const a = orbit(wide, 8, 2, 1.02), b = orbit(wide, -14, 9, 1.0);
      const live = handheld(blendCam(a, b, smoother(seg(lt, 0, 25))), t, 0.004);
      return {
        ...r,
        cam: () => blendCam(prevCam('upd', live), live, smoother(seg(lt, 0, 2.2))),
        lbl: 'run',
        bloom: 0.3,
        ov: { chart: { a: smooth(seg(lt, 0.4, 1.2)), s, step: s }, preds: { a: smooth(seg(lt, 13.4, 14.2)), frac: r.frac } },
      };
    });
  }

  /* ------------------------------------------------------------ 最难的「月→，」 */
  {
    const T0 = SEC.hard.t0, T1 = SEC.hard.t1;
    const H = [T0 + 0.3, T0 + 4.1, T0 + 7.9, T0 + 11.7];
    sub(H[0], H[1] - 0.2, `${q('月')}后面，诗里有两种接法`);
    term(H[0] + 0.1, H[1] - 0.2, '明月光 · 明月，');
    sub(H[1], H[2] - 0.2, `走完 80 步，它还押${q('光')}：${m(pc(pHard.guang))}`);
    term(H[1] + 0.1, H[2] - 0.2, `正确答案${q('，')}只有 ${pc(pHard.ans)}`);
    sub(H[2], H[3] - 0.2, `得往前看：是${q('举头望')}的那个明月`);
    term(H[2] + 0.1, H[3] - 0.2, '注意力 · 往回看前面的字');
    sub(H[3], T1 - 0.2, `学会回头看，${q('，')}涨到 ${m(pc(pHardEnd))}`);
    term(H[3] + 0.1, T1 - 0.2, `一个注意力头把 ${pc(attEnd[jEnd])} 给了${q(ch(D.fixed[jEnd]))}`);
    // 第 80 步停一下 → 快进到最后（「，」来回摇摆，最后稳住）
    const R0 = H[2] + 0.6, R1 = T1 - 2.2;
    const stepHard = D.FR[kHard] + 1;
    const stepAt = (t) => (t < R0 ? stepHard : stepHard + (S - stepHard) * easeInOut(seg(t, R0, R1)));
    ev(H[1], 'blip');
    // 「，」第一次稳稳超过「光」的那一刻：配乐起势
    let learnT = R1;
    for (let x = R0; x < R1; x += 1 / 30) {
      const { k, f } = kOfStep(stepAt(x));
      const pp = (kk) => D.probs(kk, PM, D.fixed[PM + 1]);
      const v = lerp(pp(k), pp(Math.min(D.NF - 1, k + 1)), f);
      if (v > 0.88) { learnT = x; break; }
    }
    ev(learnT, 'learn');
    ev(R0, 'rise', { dur: learnT - R0 });
    shot('hard', (lt, t, { M }) => {
      const s = stepAt(t);
      const r = runState(s);
      // 概率架上「月」那一行：从上往下看
      const row = cam([shelfX(8) + 0.3, SHELF.y + 3.1, shelfZ(PM) + 3.0], [shelfX(7.5), SHELF.y + 0.5, shelfZ(PM)], 34);
      const att = fit(M, [2.6, 5.3, 6.9, 8.4, -0.2, 0.5], [0.12, 0.2, 1], 1.15);
      const wide = fit(M, whole, WIDE, 1.03);
      let live = blendCam(row, att, smoother(seg(t, H[2] - 0.4, H[2] + 1.6)));
      // 学会以后：从上往下看整个概率架（其余几行压暗），8 行都长出了绿柱
      live = blendCam(live, fit(M, [...OP_RECT.loss, -1.3, 1.3], TOP, 1.05), smoother(seg(t, H[3] - 0.6, H[3] + 1.8)));
      live = blendCam(live, wide, smoother(seg(t, T1 - 2.4, T1)));
      return {
        ...r,
        cam: () => blendCam(prevCam('run', live), live, smoother(seg(lt, 0, 2.4))),
        lbl: 'hard',
        rowFocus: t < T1 - 2 ? PM : -1,
        bloom: 0.3,
        ov: {
          call: { a: smooth(seg(t, H[0], H[0] + 0.6)) * (1 - smooth(seg(t, H[1] - 0.4, H[1] + 0.2))) },
          att: { a: smooth(seg(t, H[2] - 0.2, H[2] + 0.5)) * (1 - smooth(seg(t, T1 - 1.6, T1 - 0.9))), frac: r.frac },
          rowtag: { a: smooth(seg(t, H[1], H[1] + 0.5)) * (1 - smooth(seg(t, T1 - 1.6, T1 - 0.9))), frac: r.frac },
          chart: { a: 0.55 * (1 - smooth(seg(lt, 0, 1))), s, step: s },
        },
      };
    });
  }

  /* ------------------------------------------------------------ 为什么要随机初始化 */
  {
    const T0 = SEC.zero.t0, T1 = SEC.zero.t1;
    sub(T0 + 0.3, T0 + 3.7, '如果一开始全设成 0 呢？');
    term(T0 + 0.4, T0 + 3.7, '全零初始化');
    sub(T0 + 3.9, T1 - 0.2, `误差传不回来，${m(S)} 步一动不动`);
    term(T0 + 4.0, T1 - 0.2, `损失一直是 ${zLoss.toFixed(3)}`);
    ev(T0 + 0.6, 'flat');
    shot('zero', (lt, t, { M }) => {
      const wide = fit(M, [BOUNDS.x0 - 9, BOUNDS.x1, BOUNDS.y0, BOUNDS.y1 + 0.8, -2.2, 1.4], WIDE, 1.0);
      const live = orbit(wide, lerp(-4, 0, lt / 7.5), 2, 1.0);
      return {
        st: G(0, { ph: 'init', sub: lt < 0.5 ? 'model' : 'zero' }, 1),
        cam: () => blendCam(prevCam('hard', live), live, smoother(seg(lt, 0, 1.6))),
        lbl: 'none',
        bloom: 0.28,
        ov: { chart: { a: smooth(seg(lt, 1.0, 1.8)) * (1 - smooth(seg(lt, 6.6, 7.4))), s: S, step: S, zero: smooth(seg(lt, 2.0, 4.0)) } },
      };
    });
  }

  /* ------------------------------------------------------------ 一个参数的一生 */
  {
    const T0 = SEC.life.t0, T1 = SEC.life.t1;
    const w0 = life[0], w1 = life[S];
    sub(T0 + 0.3, T0 + 4.1, `回到${q('月')}的那个数，看它的一生`);
    term(T0 + 0.4, T0 + 4.1, `第 1 步 → 第 ${S} 步`);
    sub(T0 + 4.3, T0 + 8.1, `从 ${m(w0.toFixed(3))} 一路长到 ${m(w1.toFixed(2))}`);
    term(T0 + 4.4, T0 + 8.1, '每一步都是真实记录');
    sub(T0 + 8.3, T1 - 0.2, `${m((STEPS_TOTAL / 1e4).toFixed(1))} 万次这样的小改动，就是“学会”`);
    term(T0 + 8.4, T1 - 0.2, `${D.P.toLocaleString('en-US')} 个参数 × ${S} 步`);
    const L0 = T0 + 1.2, L1 = T0 + 8.0;
    const stepAt = (t) => 1 + (S - 1) * easeInOut(seg(t, L0, L1));
    shot('life', (lt, t, { M }) => {
      const s = stepAt(t);
      const r = runState(s);
      // 方块放在画面左半边，右边留给曲线卡片
      const close = cam([SELP.x + 1.75, SELP.y + 1.05, 3.3], [SELP.x + 1.05, SELP.y + 0.42, 0], 34);
      const wide = fit(M, whole, WIDE, 1.03);
      const live = blendCam(close, orbit(close, 8, 3, 1.18), smoother(seg(lt, 0, 12.5)));
      const si = Math.max(0, Math.min(S, Math.floor(s))), sf = s - Math.floor(s);
      const wv = si < S ? lerp(life[si], life[si + 1], sf) : life[S];
      return {
        ...r,
        sel: GI,
        selLbl: `w = <b>${wv.toFixed(4)}</b>`,
        view: 'g-param',
        cam: () => blendCam(prevCam('zero', wide), live, smoother(seg(lt, 0, 2.2))),
        dof: { focus: live.pos.distanceTo(SELP), range: 2.2, blur: 5 * smooth(seg(lt, 1.0, 2.6)) },
        lbl: 'none',
        bloom: 0.26,
        ov: { life: { a: smooth(seg(lt, 0.8, 1.6)) * (1 - smooth(seg(lt, 11.8, 12.4))), s } },
      };
    });
  }

  /* ------------------------------------------------------------ 片尾：训练页录屏 + 网址 + 二维码 */
  let endSite = null;
  {
    const T0 = SEC.end.t0, T1 = SEC.end.t1;
    sub(T0 + 0.3, T0 + 3.4, '训练就是：猜、量误差、改一点，重复');
    const D0 = 3.6; // 录屏从这里开始
    // 录屏的剪辑：按 meta.json 里的点击时刻挑几段，变速接起来
    const segs = [];
    if (cap?.actions?.length) {
      const A = cap.actions;
      const at = (name, n = 0) => A.filter((a) => a.name === name)[n]?.t;
      const ins = A.filter((a) => a.name === 'in').map((a) => a.t);
      const add = (c0, c1, d) => { const f0 = segs.length ? segs[segs.length - 1].f1 : D0; segs.push({ c0, c1, f0, f1: f0 + d }); };
      // ① 首屏（出厂状态）→ 点 ＋ 进入训练全程
      add(Math.max(0, ins[0] - 1.6), ins[0] + 1.4, 2.6);
      // ② 一路往里钻：一步之内 → 一层之内 → 一块矩阵 → 一个参数
      add(ins[1] - 0.6, ins[ins.length - 1] + 1.5, 6.2);
      // ③ 播放 / 暂停 / 单步
      add(at('play') - 0.3, at('end') ?? at('step', 1) + 1.2, 4.2);
    }
    const FIN = segs.length ? segs[segs.length - 1].f1 : D0;
    if (segs.length) {
      sub(T0 + D0 + 0.3, T0 + D0 + 2.9, '想自己一层层打开看看？');
      sub(T0 + D0 + 3.1, T0 + segs[1].f1 - 0.2, '在训练页上，每按一次 ＋ 就钻深一层');
      term(T0 + D0 + 3.2, T0 + segs[1].f1 - 0.2, '训练全程 → 一步之内 → 一层之内 → 一块矩阵 → 一个参数');
      sub(T0 + segs[2].f0 + 0.1, T0 + FIN - 0.15, '播放、暂停、单步：看它一步步学');
      term(T0 + segs[2].f0 + 0.2, T0 + FIN - 0.15, '播放 · 单步');
      for (const a of cap.actions) if (a.x != null) {
        const sg = segs.find((g) => a.t >= g.c0 && a.t <= g.c1);
        if (sg) ev(T0 + sg.f0 + ((a.t - sg.c0) / (sg.c1 - sg.c0)) * (sg.f1 - sg.f0), 'click');
      }
    }
    ev(T0 + FIN + 0.4, 'final');
    const capAt = (lt) => {
      for (const sg of segs) if (lt <= sg.f1) return lerp(sg.c0, sg.c1, clamp((lt - sg.f0) / (sg.f1 - sg.f0)));
      return segs.length ? segs[segs.length - 1].c1 : 0;
    };
    shot('end', (lt, t, { M }) => {
      const wide = fit(M, whole, WIDE, 1.03);
      const live = orbit(wide, lerp(-6, 6, lt / 6), 4, lerp(1.0, 1.25, smooth(seg(lt, 0, 6))));
      const fin = runState(S);
      return {
        ...fin,
        cam: () => blendCam(prevCam('life', live), live, smoother(seg(lt, 0, 2.0))),
        lbl: 'none',
        bloom: 0.3,
        fade: smooth(seg(lt, D0 - 0.8, D0 + 0.3)) * 0.9,
        ov: {
          band: 1 - smooth(seg(lt, FIN - 0.5, FIN + 0.4)),
          site: segs.length ? {
            a: smooth(seg(lt, D0 - 0.1, D0 + 0.5)) * (1 - smooth(seg(t, T1 - 0.9, T1 - 0.05))),
            ct: capAt(Math.min(lt, FIN)),
            fin: easeInOut(seg(lt, FIN, FIN + 1.0)),
            finA: smooth(seg(lt, FIN + 0.5, FIN + 1.3)) * (1 - smooth(seg(t, T1 - 0.9, T1 - 0.05))),
            cursor: lt < FIN ? smooth(seg(lt, D0 + 0.3, D0 + 0.7)) * (1 - smooth(seg(lt, FIN - 0.4, FIN))) : 0,
          } : null,
        },
      };
    });
    endSite = { segs, D0, FIN };
  }

  function frame(t, ctx) {
    t = clamp(t, 0, end - 1e-6);
    const i = shots.findIndex((s) => t >= s.t0 && t < s.t1);
    const s = shots[i];
    const r = s.fn(t - s.t0, t, ctx);
    const camFn = typeof r.cam === 'function' ? r.cam : () => r.cam;
    r.cam = () => { const c = camFn(); endCam.set(s.name, c); return c; };
    r.shot = s.name;
    r.ov = r.ov || {};
    return r;
  }

  progs.sort((a, b) => a.t - b.t);
  const progAt = (t) => {
    let i = -1;
    for (const p of progs) if (t >= p.t) i = p.i;
    const a = progs.length ? smooth(seg(t, progs[0].t, progs[0].t + 0.6)) * (1 - smooth(seg(t, SEC.end.t0, SEC.end.t0 + 1.0))) : 0;
    return { i, a: i < 0 ? 0 : a };
  };
  const facts = { GI, LC, A1, life, kHard, pHard, pHardEnd, att81, attEnd, j81, jEnd, PM, p0, pL, evalFirst, evalLast, zLoss, STEPS_TOTAL, LN20 };
  return { end, frame, subs, terms, chapterNames: CHAPTERS, progAt, endSite, events, sections, open: OPEN, calc: CALC, facts, bpm: BPM, Q1, kOfStep, shots: shots.map((s) => ({ name: s.name, t0: s.t0, t1: s.t1 })) };
}
