// 3D 舞台旁边的 2D 卡片（画在 board.js 的小画布上），数字全部来自记录：
//   训练曲线（损失 / 学习率 / 梯度范数，D1 可以拖动换检查点）、打印机打出的诗、留出的《登鹳雀楼》逐字概率、
//   D5 一个权重：它的值 / 梯度、AdamW 算式（真实数字），以及它每 4 步一个点的一生。
import { COL, text, rr, card, line, dot, clamp, lerp, fmtP, sciSup, fmtInt, measure, hexA, badge, wrap, seqColor, waitBox } from '../draw.js';
import { esc } from '../../../js/ui.js';
import { isEn, L, featLabel, POEM_EN } from '../lang.js';
import { ADAM_SUBS } from './timeline.js';

const num = (v, d = 4) => (v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e4 ? Number(v.toPrecision(d)).toString().replace('-', '−') : sciSup(v, d));
const par = (v, d = 4) => (v < 0 ? `(${num(v, d)})` : num(v, d));

/* ---------------------------------------------------------------- 训练曲线 */

export function drawCurves(g, C, X, tNow, k, env, { drag, compact = false } = {}) {
  const m = X.meta, S = X.S;
  const ti = Math.min(S - 1, Math.max(0, Math.round(tNow)));
  card(g, C.x, C.y, C.w, C.h, { eyebrow: L(`LOSS · 第 ${fmtInt(ti + 1)} / ${fmtInt(S)} 步`, `LOSS · STEP ${fmtInt(ti + 1)} / ${fmtInt(S)}`), accent: COL.cyan });
  text(g, X.ema[ti].toFixed(3), C.x + C.w - 14, C.y + 19, { size: 12, kind: 'mono', weight: 700, color: COL.amber, align: 'right' });
  const xs = (t) => Math.log10(1 + t / 8) / Math.log10(1 + S / 8);
  const px = C.x + 34, pw = C.w - 48;
  const ly0 = C.y + 30, ly1 = C.y + C.h - (compact ? 20 : 74);
  const lo = 3.5, hi = 9.3;
  const Xp = (t) => px + xs(t) * pw, Y = (v) => ly1 - ((clamp(v, lo, hi) - lo) / (hi - lo)) * (ly1 - ly0);
  g.lineWidth = 1;
  for (const v of compact ? [4, 6, 8] : [4, 5, 6, 7, 8, 9]) { g.strokeStyle = COL.line; g.beginPath(); g.moveTo(px, Y(v)); g.lineTo(px + pw, Y(v)); g.stroke(); text(g, v, px - 5, Y(v) + 3, { size: 9, kind: 'mono', color: COL.faint, align: 'right' }); }
  const lnV = Math.log(m.model.vocab);
  g.setLineDash([2, 3]);
  line(g, [[px, Y(lnV)], [px + pw, Y(lnV)]], 'rgba(255,107,147,0.4)', 1);
  g.setLineDash([]);
  if (!compact) text(g, L(`瞎猜 ln ${m.model.vocab} = ${lnV.toFixed(2)}`, `blind guess ln ${m.model.vocab} = ${lnV.toFixed(2)}`), Xp(40), Y(lnV) - 4, { size: 9, color: 'rgba(255,107,147,0.75)' });
  const LS = X.D.loss, tEnd = Math.max(1, Math.round(tNow));
  const raw = [], sm = [];
  for (let i = 0; i <= tEnd && i < LS.length; i += i < 200 ? 1 : 3) { raw.push([Xp(i), Y(LS[i])]); sm.push([Xp(i), Y(X.ema[i])]); }
  line(g, raw, 'rgba(94,240,212,0.22)', 0.9);
  line(g, sm, COL.cyan, 1.7);
  const vpts = m.valCurve.filter(([tt]) => tt <= tNow).map(([tt, v]) => [Xp(tt), Y(v)]);
  if (!compact) { line(g, vpts, 'rgba(255,182,92,0.75)', 1.2); for (const [xx, yy] of vpts) dot(g, xx, yy, 1.5, COL.amber); }
  m.ckpts.forEach((c, i) => { g.fillStyle = i === k ? COL.amber : i < k ? 'rgba(94,240,212,0.5)' : COL.line3; g.fillRect(Xp(c.t) - 0.6, ly1 + 2, 1.2, i === k ? 6 : 4); });
  const hx = Xp(tNow);
  g.strokeStyle = 'rgba(255,182,92,0.55)';
  g.beginPath(); g.moveTo(hx, ly0 - 2); g.lineTo(hx, ly1); g.stroke();
  dot(g, hx, Y(X.ema[ti]), 3.4, COL.amber, '#fff');
  if (!compact) {
    const mini = (y0, h, arr, max, color, lab, fmt) => {
      text(g, lab, px - 5, y0 + h / 2 + 3, { size: 9, color: COL.dim, align: 'right' });
      const pts = [];
      for (let i = 0; i <= ti; i += i < 200 ? 1 : 3) pts.push([Xp(i), y0 + h - clamp(arr[i] / max, 0, 1) * h]);
      line(g, pts, color, 1.3);
      g.strokeStyle = COL.line; g.beginPath(); g.moveTo(px, y0 + h); g.lineTo(px + pw, y0 + h); g.stroke();
      text(g, fmt(arr[ti]), px + pw, y0 + 8, { size: 9.5, kind: 'mono', color, align: 'right' });
    };
    mini(ly1 + 14, 22, X.D.lr, m.train.peakLr * 1.05, COL.amber, 'lr', (v) => sciSup(v, 2));
    mini(ly1 + 44, 22, X.D.gnorm, 2.6, COL.violet, '‖g‖', (v) => v.toFixed(2));
    const gy = ly1 + 44 + 22 - (m.train.clip / 2.6) * 22;
    g.setLineDash([2, 3]); g.strokeStyle = 'rgba(179,157,255,0.35)'; g.beginPath(); g.moveTo(px, gy); g.lineTo(px + pw, gy); g.stroke(); g.setLineDash([]);
  }
  const toT = (wx) => 8 * (Math.pow(10, clamp((wx - px) / pw, 0, 1) * Math.log10(1 + S / 8)) - 1);
  env.hit(px - 4, ly0 - 8, pw + 8, C.y + C.h - ly0, {
    drag: drag ? (wx, wy, phase) => {
      if (phase === 'end') return;
      const tt = toT(wx);
      let best = 0;
      m.ckpts.forEach((c, i) => { if (Math.abs(xs(c.t) - xs(tt)) < Math.abs(xs(m.ckpts[best].t) - xs(tt))) best = i; });
      drag(best);
    } : null,
    tipAt: (wx) => {
      const tt = Math.round(clamp(toT(wx), 0, S - 1));
      return `<span class="k">${L(`第 ${fmtInt(tt + 1)} 步`, `Step ${fmtInt(tt + 1)}`)}</span>${L('损失', 'loss')} <span class="v">${LS[tt].toFixed(3)}</span>（${L('平滑', 'smoothed')} ${X.ema[tt].toFixed(3)}）<br>${L('学习率', 'learning rate')} <span class="a">${sciSup(X.D.lr[tt], 3)}</span>　‖g‖ ${X.D.gnorm[tt].toFixed(3)}${drag ? `<br><span style="color:var(--dim)">${L('按住拖动，跳到附近的检查点', 'Drag to jump to a nearby checkpoint')}</span>` : ''}`;
    },
  });
}

/* ---------------------------------------------------------------- 打印机打出的诗 */

export function splitPoem(s) {
  const out = [];
  let cur = '';
  for (const ch of s) { cur += ch; if (ch === '。') { out.push(cur); cur = ''; } }
  if (cur) out.push(cur);
  return out;
}
// 格律：几句、每句几个字、标点是否“，。”交替
export function checkForm(s) {
  if (!s) return null;
  const parts = s.split(/[，。]/);
  if (parts[parts.length - 1] === '') parts.pop();
  const n = parts[0]?.length, lines = parts.length;
  let alt = true, idx = 0;
  for (const ch of s) if (ch === '，' || ch === '。') { if (ch !== (idx % 2 === 0 ? '，' : '。')) alt = false; idx++; }
  const ok = (n === 5 || n === 7) && (lines === 4 || lines === 8) && parts.every((p) => p.length === n) && alt && s.endsWith('。');
  if (ok) return { ok, name: isEn ? `${n}-char ${lines === 4 ? 'quatrain' : 'regulated verse'}` : `${n === 5 ? '五' : '七'}言${lines === 4 ? '绝句' : '律诗'}` };
  return { ok: false, name: L('格式还不对', 'not a valid form yet') };
}

// age：这个检查点“打印”了多久（秒 × 倍速），决定露出几个字
export function drawReceipt(g, C, X, k, age, env) {
  const m = X.meta;
  card(g, C.x, C.y, C.w, C.h, { eyebrow: L(`PRINTER · 第 ${fmtInt(m.ckpts[k].t + 1)} 步的模型`, `PRINTER · MODEL AT STEP ${fmtInt(m.ckpts[k].t + 1)}`), title: L('它此刻写的诗', 'Poems it writes right now'), accent: COL.amber });
  text(g, L(`温度 ${m.train.sampleTemp} · 同一组随机数`, `temp ${m.train.sampleTemp} · same seed`), C.x + C.w - 14, C.y + 19, { size: 9, color: COL.dim, align: 'right' });
  // 出纸口
  rr(g, C.x + 14, C.y + 48, C.w - 28, 4, 2);
  g.fillStyle = 'rgba(255,182,92,0.35)';
  g.fill();
  const n = m.prefixes.length, top = C.y + 58, bh = (C.h - 64) / n, size = 13;
  m.ckpts[k].samples.forEach((s, i) => {
    const y = top + i * bh, pf = m.prefixes[i];
    text(g, pf ? L(`开头「${pf}」`, `opening 「${pf}」`) : L('不给开头', 'no opening'), C.x + 14, y + 10, { size: 9, color: COL.dim });
    const fm = checkForm(s);
    if (fm) badge(g, C.x + C.w - 14, y, fm.ok ? `✓ ${fm.name}` : fm.name, fm.ok ? COL.cyan : COL.faint, 'right', 9);
    const shown = s.slice(0, Math.max(pf.length, Math.floor(age)));
    let cx = C.x + 14, cy = y + 28, used = 1;
    const maxLines = Math.max(1, Math.floor((bh - 18) / (size + 4)));
    let j = 0;
    for (const ch of shown) {
      const cw = measure(g, ch, size);
      if (cx + cw > C.x + C.w - 14) { if (used >= maxLines) break; used++; cx = C.x + 14; cy += size + 4; }
      text(g, ch, cx, cy, { size, color: j < pf.length ? COL.amber : ch === '，' || ch === '。' ? COL.dim : COL.ink });
      cx += cw;
      j++;
    }
  });
  env.hit(C.x, C.y, C.w, C.h, { tip: L('每个检查点用同一组随机数、同样的 4 个开头生成（温度 0.8）：差别只来自模型本身。最开始是随机的字，几十步后先学会标点，再学会五言、七言。', 'At every checkpoint the same random numbers and the same 4 openings are used (temperature 0.8), so differences come only from the model. At first it is random characters; after a few dozen steps punctuation appears, then five- and seven-character lines.') });
}

/* ---------------------------------------------------------------- 留出的《登鹳雀楼》 */

export function drawHeld(g, C, X, k, env) {
  const D = X.D, m = X.meta;
  card(g, C.x, C.y, C.w, C.h, { eyebrow: L('HELD-OUT · 训练时从没见过', 'HELD-OUT · NEVER SEEN IN TRAINING'), title: L('《登鹳雀楼》每个字的概率', '“On the Stork Tower”: p per char'), accent: COL.cyan });
  if (!D.has('ck', k)) { waitBox(g, C.x + 14, C.y + 50, C.w - 28, C.h - 60, env, { since: 0 }, { label: L('正在载入这个检查点…', 'Loading this checkpoint…'), size: 11 }); return; }
  const textS = m.held.text + '⏎', Lv = D.Lv;
  const cols = 13, tw = Math.min(24, (C.w - 28) / cols - 2), th = tw + 11;
  const x0 = C.x + (C.w - cols * (tw + 2)) / 2, y0 = C.y + 52;
  let mean = 0;
  for (let i = 0; i < Lv; i++) {
    const r = Math.floor(i / cols), c = i % cols, x = x0 + c * (tw + 2), y = y0 + r * (th + 4);
    const p = D.valP(k, i);
    mean += -Math.log(Math.max(p, 1e-9));
    rr(g, x, y, tw, th, 4);
    g.fillStyle = seqColor(Math.pow(p, 0.5), 0.9);
    g.fill();
    text(g, textS[i], x + tw / 2, y + tw / 2 + 4, { size: tw * 0.52, color: p > 0.5 ? '#04121a' : COL.ink, align: 'center', weight: 600 });
    text(g, fmtP(p), x + tw / 2, y + th - 3, { size: 7.5, kind: 'mono', color: p > 0.5 ? '#04121a' : COL.dim, align: 'center' });
    env.hit(x, y, tw, th, { tip: () => {
      const top = D.valTop(k, i).map((tt) => `<tr><td class="${tt.id === m.held.ids[i + 1] ? 'tg' : ''}">${esc(X.ch(tt.id))}</td><td>${fmtP(tt.p)}</td></tr>`).join('');
      const ctx = m.held.text.slice(0, i) || L('（只有开头标记）', '(only the start marker)');
      return `<span class="k">${L(`看到「${esc(ctx.slice(-8))}」之后`, `after “${esc(ctx.slice(-8))}”`)}</span>${L('正确答案', 'answer')}「<b>${esc(textS[i])}</b>」 <span class="v">${fmtP(p)}</span><table>${top}</table>${isEn ? `<span style="color:var(--dim)">${POEM_EN}</span>` : ''}`;
    } });
  }
  text(g, L(`平均 −ln p = ${(mean / Lv).toFixed(2)}`, `mean −ln p = ${(mean / Lv).toFixed(2)}`), C.x + 14, C.y + C.h - 10, { size: 9.5, color: COL.dim });
}

/* ---------------------------------------------------------------- D5 一个权重 */

// 它在机器上的哪里、前向乘了谁（没有记录激活，只说清楚位置）、反向的梯度（裁剪前 / 后）
export function drawWhere(g, C, X, k, f, mi, p, env) {
  const ft = X.feats[f], a = X.adam(k, f), lr = X.lr(k);
  card(g, C.x, C.y, C.w, C.h, { eyebrow: mi === 'w' ? L('FORWARD · 它此刻的值', 'FORWARD · ITS VALUE NOW') : L('BACKWARD · 它这一步的梯度', 'BACKWARD · ITS GRADIENT THIS STEP'), title: featLabel(ft, 'tiny'), accent: mi === 'w' ? COL.violet : COL.rose, active: true });
  const x0 = C.x + 16, w = C.w - 32;
  let y = C.y + 64;
  const m = ft.name.match(/^model\.layers\.(\d+)\.(.+)$/), idx = ft.index;
  let where;
  if (ft.name === 'model.embed_tokens.weight') where = L(`嵌入表 E 第 ${idx[0]} 行（「${X.ch(idx[0])}」）的第 ${idx[1]} 维。前向时，每遇到「${X.ch(idx[0])}」就把这一行取出来当它的向量；输出层共用这张表，所以它也参与给「${X.ch(idx[0])}」打分。`, `Row ${idx[0]} of the embedding table E (“${X.ch(idx[0])}”), dim ${idx[1]}. In the forward pass this row is fetched whenever “${X.ch(idx[0])}” appears; the output layer shares the table, so it also helps score “${X.ch(idx[0])}”.`);
  else if (m[2].includes('layernorm')) where = L(`第 ${m[1]} 层注意力前 RMSNorm 的缩放 γ 的第 ${idx[0]} 维：把归一化后的残差第 ${idx[0]} 维乘以它，再送进 W_q / W_k / W_v。`, `Dim ${idx[0]} of the scale γ of layer ${m[1]}’s pre-attention RMSNorm: the normalized residual’s dim ${idx[0]} is multiplied by it before going into W_q / W_k / W_v.`);
  else {
    const nm = m[2].includes('q_proj') ? 'W_q' : 'W_down';
    where = L(`第 ${m[1]} 层 ${nm}：输入第 ${idx[1]} 维 → 输出第 ${idx[0]} 维（PyTorch 下标 [${idx[0]}, ${idx[1]}]）。前向时，每个位置的输入第 ${idx[1]} 维乘以它，加进输出第 ${idx[0]} 维（批次的激活没有记录，这里只说清楚它在哪）。`, `Layer ${m[1]} ${nm}: input dim ${idx[1]} → output dim ${idx[0]} (PyTorch index [${idx[0]}, ${idx[1]}]). In the forward pass, each position’s input dim ${idx[1]} is multiplied by it and added into output dim ${idx[0]} (the batch’s activations weren’t recorded, so this only says where it sits).`);
  }
  y += 4 + 14 * wrap(g, where, x0, y, w, 14, { size: 11, color: COL.ink2, maxLines: 5 });
  y += 10;
  const row = (k1, v, col, sub = '') => { text(g, k1, x0, y, { size: 11, color: COL.dim }); text(g, v, x0 + w, y, { size: 14, kind: 'mono', weight: 700, color: col, align: 'right' }); if (sub) { y += 15; text(g, sub, x0 + w, y, { size: 9.5, color: COL.faint, align: 'right' }); } y += 22; };
  if (mi === 'w') {
    row(L('这一步开始时 w', 'w at the start of this step'), num(a.w0, 6), COL.violet);
    row(L('到训练结束（第 4000 步）', 'at the end of training (step 4000)'), X.has('feat', 0) ? num(X.D.feat(f).w[X.D.feat(f).n - 1], 6) : '…', COL.ink2);
  } else {
    row(L('∂L/∂w（裁剪前）', '∂L/∂w (before clipping)'), num(a.gRaw, 5), COL.rose);
    row(L(`× 裁剪系数 ${X.clip(k) < 1 ? X.clip(k).toFixed(5) : '1（不裁剪）'}`, `× clip factor ${X.clip(k) < 1 ? X.clip(k).toFixed(5) : '1 (no clipping)'}`), num(a.g, 5), COL.amber, L(`全部梯度的长度 ‖g‖ = ${X.gnorm(k).toFixed(4)}`, `length of all gradients ‖g‖ = ${X.gnorm(k).toFixed(4)}`));
    wrap(g, L(`g ${a.g > 0 ? '> 0：w 增大一点，损失会上升，所以它要往小的方向挪' : '< 0：w 增大一点，损失会下降，所以它要往大的方向挪'}。这个数是批次 64 行 × 128 个位置各自贡献加起来的。`, `g ${a.g > 0 ? '> 0: nudging w up raises the loss, so it should move down' : '< 0: nudging w up lowers the loss, so it should move up'}. This number sums the contributions of all 64 rows × 128 positions in the batch.`), x0, y, w, 13, { size: 10, color: COL.dim, maxLines: 3 });
  }
}

// AdamW 算式：每一项都是这一步真实的数（导出时用 PyTorch 同样的 float32 运算核对过，最大误差 5 ulp）
export function drawAdam5(g, C, X, k, f, mi, p, env) {
  const a = X.adam(k, f), ft = X.feats[f], lr = X.lr(k), t = a.t, b1 = 0.9, b2 = 0.95, eps = 1e-8;
  card(g, C.x, C.y, C.w, C.h, { eyebrow: L(`ADAMW · 第 ${fmtInt(t)} 步 · 真实数字`, `ADAMW · STEP ${fmtInt(t)} · REAL NUMBERS`), title: featLabel(ft, 'tiny'), accent: COL.amber, active: true });
  const cur = ADAM_SUBS.indexOf(mi);
  const adam = (-lr * a.mh) / (Math.sqrt(a.vh) + eps), decay = -lr * a.wd * a.w0, dw = adam + decay;
  const n3 = (v) => num(v, 3), p3 = (v) => par(v, 3);
  const rows = [
    ['g', L('梯度（裁剪后）', 'gradient (clipped)'), `g = ${n3(a.gRaw)} × ${X.clip(k) < 1 ? X.clip(k).toFixed(4) : '1'} = ${n3(a.g)}`],
    ['m', L('一阶动量', '1st moment'), `m = 0.9 × ${p3(a.m0)} + 0.1 × ${p3(a.g)} = ${n3(a.m)}`],
    ['v', L('二阶动量', '2nd moment'), `v = 0.95 × ${n3(a.v0)} + 0.05 × ${p3(a.g)}² = ${n3(a.v)}`],
    ['bc', L('偏差校正', 'bias correction'), `m̂ = m / ${(1 - b1 ** t).toPrecision(4)} = ${n3(a.mh)}　v̂ = v / ${(1 - b2 ** t).toPrecision(4)} = ${n3(a.vh)}`],
    ['dw', L('更新量', 'update'), `Δw = −${sciSup(lr, 3)} × (${n3(a.mh)} / √${n3(a.vh)} + ${a.wd} × ${p3(a.w0)}) = ${sciSup(dw, 3)}`],
    ['write', L('写回', 'write back'), `w = ${num(a.w0, 6)} ${dw >= 0 ? '+' : '−'} ${sciSup(Math.abs(dw), 3)} = ${num(a.w0 + dw, 6)}`],
  ];
  const narrow = C.w < 600;
  const x0 = C.x + 16, w = C.w - 32;
  let y = C.y + 64;
  const lh = narrow ? 37 : 30;
  rows.forEach(([key, lab, f1], i) => {
    const on = i === cur, done = i < cur;
    const a1 = on ? clamp(p * 2, 0.35, 1) : done ? 0.75 : 0.32;
    g.globalAlpha = a1;
    if (on) { rr(g, C.x + 8, y - 14, C.w - 16, lh - 2, 8); g.fillStyle = 'rgba(255,182,92,0.08)'; g.fill(); }
    text(g, lab, x0, y, { size: 10, color: on ? COL.amber : COL.dim });
    if (narrow) text(g, f1, x0, y + 15, { size: 10.5, kind: 'mono', color: on ? COL.ink : COL.ink2, max: w });
    else text(g, f1, x0 + 92, y, { size: 11, kind: 'mono', color: on ? COL.ink : COL.ink2, max: w - 92 });
    g.globalAlpha = 1;
    y += lh;
  });
  y += 4;
  const d = a.w0 + dw - a.w1;
  const note = mi === 'bc' && t === 1 ? L('第 1 步：m̂ = g、v̂ = g²，更新量 = −lr × g/|g|，只剩下符号。', 'Step 1: m̂ = g and v̂ = g², so the update is −lr × g/|g| — only the sign is left.')
    : mi === 'dw' ? (a.wd ? L(`权重衰减 λ = ${a.wd}：每步把 w 往 0 拉 lr × λ × w = ${sciSup(Math.abs(decay), 2)}。m̂/√v̂ 通常在 1 左右，所以每个权重每步大约挪一个 lr。`, `Weight decay λ = ${a.wd}: each step pulls w toward 0 by lr × λ × w = ${sciSup(Math.abs(decay), 2)}. m̂/√v̂ is usually around 1, so each weight moves about one lr per step.`) : L('RMSNorm 的 γ 按惯例不做权重衰减（λ = 0）。', 'By convention RMSNorm’s γ gets no weight decay (λ = 0).'))
      : L(`记录的 w′ = ${num(a.w1, 7)}，按公式算出的差 ${Math.abs(d) < 1e-12 ? '0' : sciSup(Math.abs(d), 2)}（这里用双精度重算；导出时用 PyTorch 同样的 float32 运算核对全部 4000 步，最大误差 ${X.meta.train.adamCheckUlp} ulp）。`, `Recorded w′ = ${num(a.w1, 7)}; the formula differs by ${Math.abs(d) < 1e-12 ? '0' : sciSup(Math.abs(d), 2)} (recomputed here in double precision; the export checked all 4,000 steps with PyTorch’s own float32 ops, max error ${X.meta.train.adamCheckUlp} ulp).`);
  wrap(g, note, x0, y + 4, w, 13, { size: 9.5, color: COL.dim, maxLines: 3 });
  env.hit(C.x, C.y, C.w, C.h, { tip: L(`学习率 lr = ${sciSup(lr, 3)}（${X.step(k) < X.meta.train.warmup ? '预热中' : '余弦退火中'}），β₁ = 0.9，β₂ = 0.95，ε = 10⁻⁸`, `learning rate lr = ${sciSup(lr, 3)} (${X.step(k) < X.meta.train.warmup ? 'warming up' : 'cosine annealing'}), β₁ = 0.9, β₂ = 0.95, ε = 10⁻⁸`) });
}

// 一生：w、g、m、√v，每 4 步一个点（4000 步 → 1000 个点）
export function drawLife5(g, C, X, f, tNow, env, { wide = true } = {}) {
  const ft = X.feats[f], S = X.S;
  card(g, C.x, C.y, C.w, C.h, { eyebrow: L('HISTORY · 每 4 步一个点 · float32', 'HISTORY · ONE POINT EVERY 4 STEPS · FLOAT32'), title: L(`${featLabel(ft, 'tiny')} 的一生`, `The life of ${featLabel(ft, 'tiny')}`), accent: COL.cyan });
  if (!X.has('feat', 0)) { waitBox(g, C.x + 14, C.y + 50, C.w - 28, C.h - 60, env, { since: 0 }, { label: L('正在载入权重轨迹…', 'Loading the weight trajectory…'), size: 11 }); return; }
  const F = X.D.feat(f), n = F.n, st = F.stride;
  const sv = Array.from(F.v, (v) => Math.sqrt(v));
  const rowsDef = [['w', F.w, L('权重 w', 'weight w'), COL.violet], ['g', F.g, L('梯度 g（裁剪后）', 'gradient g (clipped)'), COL.rose], ['m', F.m, L('一阶动量 m', '1st moment m'), COL.amber], ['sv', sv, L('√v', '√v'), COL.blue]];
  const nr = rowsDef.length, gap = 14;
  const cw = wide ? (C.w - 28 - gap * (nr - 1)) / nr : C.w - 28;
  const top = C.y + 52, rh = wide ? C.h - 62 : (C.h - 62) / nr;
  const ci = Math.min(n - 1, Math.round(tNow / st));
  rowsDef.forEach(([key, arr, lab, col], ri) => {
    const x0 = C.x + 14 + (wide ? ri * (cw + gap) : 0), w = cw, y0 = wide ? top : top + ri * rh, h = rh - 24;
    const Xp = (i) => x0 + (i / (n - 1)) * w;
    text(g, lab, x0, y0 + 9, { size: 10, color: col, max: w - 60 });
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) { lo = Math.min(lo, arr[i]); hi = Math.max(hi, arr[i]); }
    if (key === 'w') { const pd = (hi - lo) * 0.1 || 0.01; lo -= pd; hi += pd; } else if (key === 'sv') lo = 0; else { const mm = Math.max(Math.abs(lo), Math.abs(hi)) || 1e-9; lo = -mm; hi = mm; }
    const Y = (v) => y0 + 14 + h - ((v - lo) / (hi - lo || 1)) * h;
    if (key === 'g' || key === 'm') { g.strokeStyle = COL.line; g.beginPath(); g.moveTo(x0, Y(0)); g.lineTo(x0 + w, Y(0)); g.stroke(); }
    const pts = [];
    for (let i = 0; i < n; i++) pts.push([Xp(i), Y(arr[i])]);
    line(g, pts, hexA(col, key === 'g' ? 0.55 : 0.95), key === 'g' ? 0.8 : 1.3);
    const cx = Xp(ci);
    g.strokeStyle = hexA(COL.amber, 0.5); g.beginPath(); g.moveTo(cx, y0 + 12); g.lineTo(cx, y0 + 14 + h); g.stroke();
    dot(g, cx, Y(arr[ci]), 3, COL.amber);
    text(g, num(arr[ci], 3), x0 + w, y0 + 9, { size: 9, kind: 'mono', color: COL.ink2, align: 'right' });
    g.strokeStyle = COL.line; g.strokeRect(x0, y0 + 14, w, h);
    env.hit(x0, y0, w, rh, { tipAt: (wx) => { const i = clamp(Math.round(((wx - x0) / w) * (n - 1)), 0, n - 1); return `<span class="k">${L(`第 ${fmtInt(i * st + 1)} 步`, `step ${fmtInt(i * st + 1)}`)}</span>${lab} = <span class="v">${num(arr[i], 5)}</span>`; } });
  });
}
