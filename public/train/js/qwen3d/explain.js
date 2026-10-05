// 第三章的调试器内容：伪代码（高亮当前行）、这一步的白话讲解 + 真实数字、变量监视。
// 接口和 ../glass/explain.js 一样：codeFor / linesFor / explain / watch / stepLabel / crumbs / shapeOf。
// 选中的层 L 由 controls 放在 ctx.qL，选中的回答位置在 ctx.qPos。
import { esc, tokPlain } from '../../../js/ui.js';
import { fmtP, sciSup, fmtInt } from '../draw.js';
import { t as T_, L, isEn } from './lang.js';
import { tname, TSHAPE, numel, LAYER_T } from './data.js';
import { bf16Round } from '../explain.js';

const K_ = (s) => `<span class="kw">${s}</span>`;
const F_ = (s) => `<span class="fn">${s}</span>`;
const C_ = (s) => `<span class="cm"># ${s}</span>`;
const N_ = (s) => `<span class="nu">${s}</span>`;
const tk = (s) => esc(tokPlain(s));
const n3 = (v) => (v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e4 ? Number(v.toPrecision(4)).toString() : sciSup(v, 3));
const n5 = (v) => (v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e4 ? Number(v.toPrecision(6)).toString() : sciSup(v, 5));
const LX = (ctx, R) => ctx?.qL ?? R.NL - 1;
const FOCUS = (ctx, R) => (R.ans.includes(ctx?.qPos) ? ctx.qPos : R.ans[1]);
const lt = (tn, L_) => (tn === 'embed' || tn === 'norm' ? -1 : L_);

export function codeFor(R) {
  const m = R.D.meta;
  return [
    `model = ${F_('from_pretrained')}(<span class="nu">"Qwen3-0.6B"</span>)  ${C_(T_('q3.code0'))}`,
    `ids = ${F_('apply_chat_template')}(chat)    ${C_(T_('q3.code1', { n: m.ids.length }))}`,
    `labels = ${F_('mask_prompt')}(ids, ${N_('-100')})  ${C_(T_('q3.code2', { n: R.ans.length }))}`,
    `opt = ${F_('AdamW')}(lr=${N_('1e-5')}, betas=(${N_('.9')}, ${N_('.95')}), wd=${N_('0.1')})`,
    `${K_('for')} step ${K_('in')} ${F_('range')}(${N_(R.K)}):         ${C_(T_('q3.code3'))}`,
    `    h = ${F_('embed')}(ids[:${N_('-1')}])          ${C_(T_('q3.code4'))}`,
    `    ${K_('for')} layer ${K_('in')} layers:          ${C_(T_('q3.code5'))}`,
    `        h = h + ${F_('attn')}(${F_('rms')}(h))     ${C_(T_('q3.code6'))}`,
    `        h = h + ${F_('mlp')}(${F_('rms')}(h))      ${C_(T_('q3.code7'))}`,
    `    logits = ${F_('lm_head')}(${F_('rms')}(h))     ${C_(T_('q3.code8'))}`,
    `    loss = ${F_('cross_entropy')}(logits, labels[${N_('1')}:])  ${C_(T_('q3.code9', { n: R.ans.length }))}`,
    `    loss.${F_('backward')}()               ${C_(T_('q3.code10'))}`,
    `    ${F_('clip_grad_norm_')}(params, ${N_('1.0')})`,
    `    ${K_('for')} w, g ${K_('in')} params:           ${C_(T_('q3.code11'))}`,
    `        m = ${N_('.9')}m + ${N_('.1')}g;  v = ${N_('.95')}v + ${N_('.05')}g²`,
    `        w −= lr·(m̂/(√v̂+ε) + λ·w)`,
  ];
}

export function linesFor(s) {
  switch (s.ph) {
    case 'run': case 'end': return [5];
    case 'batch': return !s.sub ? [2, 3] : { tpl: [2], mask: [3], shift: [6, 11] }[s.sub];
    case 'fwd': if (!s.sub) return [6, 7, 8, 9, 10]; return { emb: [6], lo: [7, 8, 9], hi: [7, 8, 9], ln1: [8], qkv: [8], attn: [8], o: [8], ln2: [9], ffn: [9], down: [9], head: [10] }[s.sub];
    case 'loss': return [11];
    case 'bwd': return [12];
    case 'upd':
      if (!s.sub) return [13, 14, 15, 16];
      if (s.sub === 'clip') return [13];
      if (s.sub === 'bf16') return [16];
      if (!s.mi) return [14, 15, 16];
      return { g: [14], m: [15], v: [15], bc: [16], dw: [16], write: [16], bits: [16] }[s.mi];
  }
  return [];
}

/* ---------------------------------------------------------------- 名字 */

function subName(s, R, ctx) {
  const Lx = LX(ctx, R);
  if (s.sub === 'lo') return T_('q3.sub.lo', { a: Lx - 1 });
  if (s.sub === 'hi') return T_('q3.sub.hi', { a: Lx + 1 });
  if (s.sub === 'pos') return T_('q3.sub.pos', { i: s.i, s: tokPlain(R.tgtStr(s.i)) });
  return T_(`q3.sub.${s.sub}`);
}

export function stepLabel(s, R, depth, ctx) {
  if (s.ph === 'run') return T_('q3.stepLbl', { n: s.k + 1 });
  if (s.ph === 'end') return T_('q3.endLbl');
  const ph = T_(`q3.ph.${s.ph}`);
  if (!s.sub) return ph;
  const Lx = LX(ctx, R);
  const inLayer = ['ln1', 'qkv', 'attn', 'o', 'ln2', 'ffn', 'down'].includes(s.sub);
  let out = `${ph} · ${inLayer ? `L${Lx} · ` : ''}${esc(subName(s, R, ctx))}`;
  if (s.t) out += ` · ${T_(`q3.t.${s.t}`)}`;
  if (s.mi) out += ` · ${s.ph === 'bwd' ? T_('q3.mi.wg') : T_(`q3.mi.${s.mi}`)}`;
  return out;
}

export function crumbs(depth, s, R, ctx) {
  const out = [{ d: 1, label: T_('q3.crumb1') }];
  if (depth === 1 && s.ph === 'end') out.push({ d: 1, label: T_('q3.crumbEnd') });
  if (depth >= 2) out.push({ d: 2, label: T_('q3.crumb2', { n: s.k + 1 }) });
  if (depth >= 3 && s.sub) {
    const inLayer = ['ln1', 'qkv', 'attn', 'o', 'ln2', 'ffn', 'down'].includes(s.sub);
    out.push({ d: 3, label: `${T_(`q3.ph.${s.ph}`)}${inLayer ? ` · ${T_('q3.layerN', { l: LX(ctx, R) })}` : ''}` });
  }
  if (depth >= 4 && s.t) out.push({ d: 4, label: T_(`q3.t.${s.t}`) });
  if (depth >= 5 && s.mi) { const idx = R.trackedIndex(lt(s.t, LX(ctx, R)), s.t); out.push({ d: 5, label: `[${idx.join(', ')}]` }); }
  return out;
}

export function shapeOf(s, R, depth, ctx) {
  if (s.ph === 'run' || s.ph === 'end') return '';
  const n = R.N1, M = R.model;
  if (s.t) {
    const sh = TSHAPE[s.t];
    const nm = T_(`q3.t.${s.t}`);
    if (s.ph === 'bwd') return `∇${nm} <b>${sh.length === 2 ? `[${sh[0]} × ${sh[1]}]` : `[${sh[0]}]`}</b>${L('（和权重同形状）', ' (same shape as the weight)')}`;
    if (s.ph === 'upd') return `m, v, Δw <b>${sh.length === 2 ? `[${sh[0]} × ${sh[1]}]` : `[${sh[0]}]`}</b> · fp32`;
    return `${nm} <b>${sh.length === 2 ? `[${sh[0]} × ${sh[1]}]` : `[${sh[0]}]`}</b> · ${fmtInt(numel(s.t))}`;
  }
  switch (s.ph) {
    case 'batch': return `ids <b>[1 × ${n + 1}]</b> → inp <b>[1 × ${n}]</b>，labels <b>[1 × ${n}]</b>`.replace('，', isEn ? ', ' : '，');
    case 'fwd':
      if (s.sub === 'qkv') return `x <b>[${n} × 1024]</b> → q <b>[${n} × 16 × 128]</b>, k, v <b>[${n} × 8 × 128]</b>`;
      if (s.sub === 'attn') return `${L('打分', 'scores')} <b>[16 × ${n} × ${n}]</b> → softmax → ${L('加权求和', 'weighted sum')} <b>[${n} × 2048]</b>`;
      if (s.sub === 'ffn') return `x <b>[${n} × 1024]</b> → gate, up <b>[${n} × 3072]</b>`;
      if (s.sub === 'head') return `h <b>[${n} × 1024]</b> → logits <b>[${n} × ${fmtInt(M.vocab)}]</b>`;
      return `h <b>[${n} × 1024]</b>`;
    case 'loss': return `logits <b>[${n} × ${fmtInt(M.vocab)}]</b> → ${R.ans.length} × −ln p → ${L('平均', 'mean')}`;
    case 'bwd': return `∂L/∂h <b>[${n} × 1024]</b> · ${L('每个参数一个同形状的梯度', 'one same-shaped gradient per parameter')} · <b>${fmtInt(M.params)}</b>`;
    case 'upd': return `m, v ${L('各', 'each')} <b>${fmtInt(M.params)}</b> × fp32（${((M.params * 8) / 2 ** 20).toFixed(0)} MB）`;
  }
  return '';
}

/* ---------------------------------------------------------------- 讲解 */

function lensFirst(R, k, i) {
  if (!R.hasSt(k)) return -1;
  for (let b = 1; b <= R.NL; b++) if (R.lensTop(k, b, i).ok) return b - 1;
  return -1;
}

export function explain(s, R, ctx, depth) {
  const k = s.k, m = R.D.meta, Lx = LX(ctx, R), f = FOCUS(ctx, R);
  const st = m.steps[k];
  const hasSt = R.hasSt(k), has3 = R.has3(k);
  const loading = `<span class="dimmed">${T_('q3.loadingStep')}</span>`;
  switch (s.ph) {
    case 'run': return T_('q3.x.run', { n: k + 1, a: R.lossState(k).toFixed(4), b: R.lossState(k + 1).toFixed(4), g: st.gradNorm.toFixed(1) });
    case 'end': { const i = R.ans[1]; return T_('q3.x.end', { a: R.lossState(0).toFixed(3), b: R.lossState(R.K).toFixed(3), p0: fmtP(R.pState(0, i)), p3: fmtP(R.pState(R.K, i)) }); }
  }
  if (!s.sub) {
    switch (s.ph) {
      case 'batch': return T_('q3.x.batch', { n: m.ids.length, np: m.nPrompt, na: R.ans.length });
      case 'fwd': {
        if (!hasSt) return loading;
        const lf = lensFirst(R, k, f);
        const lensTxt = lf >= 0 ? T_('q3.x.lensFirst', { l: lf }) : T_('q3.x.lensNever', { top: tk(R.lensTop(k, R.NL, f).s) });
        return T_('q3.x.fwd', { m: R.N1, t: tk(R.tgtStr(f)), i: f, lensTxt, p: fmtP(R.pState(k, f)) });
      }
      case 'loss': { let hi = R.ans[0]; for (const i of R.ans) if (R.nllState(k, i) > R.nllState(k, hi)) hi = i; return T_('q3.x.loss', { n: R.ans.length, loss: R.lossState(k).toFixed(4), ht: tk(R.tgtStr(hi)), hp: fmtP(R.pState(k, hi)), hn: R.nllState(k, hi).toFixed(3), m: R.N1, pre: R.lossPreState(k).toFixed(4) }); }
      case 'bwd': {
        if (!hasSt) return loading;
        let lmax = 0; for (let l = 1; l < R.NL; l++) if (R.layerGrad(k, l) > R.layerGrad(k, lmax)) lmax = l;
        return T_('q3.x.bwd', { lmax, gmax: R.layerGrad(k, lmax).toFixed(1) });
      }
      case 'upd': return T_('q3.x.upd', { g: st.gradNorm.toFixed(1), c: st.clip.toFixed(5), flat: k === 0 ? T_('q3.x.updFlat') : T_('q3.x.updVar'), a: R.lossState(k).toFixed(4), b: R.lossState(k + 1).toFixed(4) });
    }
  }
  if (s.mi) return explainWeight(s, R, ctx, Lx);
  if (s.t) return explainTensor(s, R, ctx, Lx);
  switch (s.ph) {
    case 'batch':
      if (s.sub === 'tpl') return T_('q3.x.tpl', { n: m.ids.length });
      if (s.sub === 'mask') return T_('q3.x.mask', { np: m.nPrompt, m: R.N1, na: R.ans.length });
      return T_('q3.x.shift', { a0: R.ans[0], a1: R.ans[R.ans.length - 1], t0: tk(R.tgtStr(R.ans[0])) });
    case 'fwd': {
      if (s.sub === 'emb') return T_('q3.x.emb');
      if (s.sub === 'lo') return T_('q3.x.lo', { a: Lx - 1, l: Lx });
      if (!hasSt) return loading;
      if (s.sub === 'hi') return T_('q3.x.hi', { a: Lx + 1, t: tk(R.tgtStr(f)), p: fmtP(R.lensP(k, R.NL, f)) });
      if (s.sub === 'head') { const top = R.topState(k, f)[0]; return T_('q3.x.head', { i: f, t: tk(R.tgtStr(f)), p: fmtP(R.pState(k, f)), top: tk(top.s), tp: fmtP(top.p) }); }
      if (s.sub === 'ln1') return T_('q3.x.ln1', { i: f, s: tk(R.inStr(f)), n: R.residNorm(k, Lx, f).toFixed(2) });
      if (s.sub === 'qkv') return T_('q3.x.qkv');
      if (s.sub === 'ln2') return T_('q3.x.ln2');
      if (s.sub === 'ffn') return T_('q3.x.ffn');
      if (!has3) return loading;
      if (s.sub === 'attn') {
        const list = []; for (let j = 0; j <= f; j++) list.push([j, R.att(k, Lx, f, j)]);
        list.sort((a, b) => b[1] - a[1]);
        return T_('q3.x.attn', { i: f, s: tk(R.inStr(f)), l: Lx, top: list.slice(0, 3).map(([j, w]) => `「${tk(R.inStr(j))}」${fmtP(w)}`).join(isEn ? ', ' : '、') });
      }
      if (s.sub === 'o') { let jm = 0; for (let j = 1; j < R.N1; j++) if (R.attnOut(k, Lx, j) > R.attnOut(k, Lx, jm)) jm = j; return T_('q3.x.o', { i: f, v: R.attnOut(k, Lx, f).toFixed(2), jm, vm: R.attnOut(k, Lx, jm).toFixed(2) }); }
      if (s.sub === 'down') return T_('q3.x.down', { i: f, v: R.mlpOut(k, Lx, f).toFixed(2), t: tk(R.tgtStr(f)), top: tk(R.lensTop(k, Lx + 1, f).s), p: fmtP(R.lensP(k, Lx + 1, f)) });
      return '';
    }
    case 'loss': {
      if (s.sub === 'mean') return T_('q3.x.mean', { loss: R.lossState(k).toFixed(4), m: R.N1, pre: R.lossPreState(k).toFixed(4) });
      const i = s.i, ctxs = [];
      for (let j = Math.max(0, i - 4); j <= i; j++) ctxs.push(tokPlain(R.inStr(j)));
      const top = R.topState(k, i).map((c) => `${tk(c.s)} ${fmtP(c.p)}`).join(isEn ? ', ' : '、');
      return T_('q3.x.pos', { i, ctx: esc(ctxs.join('')), t: tk(R.tgtStr(i)), p: fmtP(R.pState(k, i)), nll: R.nllState(k, i).toFixed(4), top });
    }
    case 'bwd': {
      if (!hasSt) return loading;
      const g = (t) => n3(R.gradOf(k, tname(Lx, t)));
      switch (s.sub) {
        case 'head': return T_('q3.x.bHead', { gn: n3(R.gradOf(k, tname(-1, 'norm'))) });
        case 'hi': return T_('q3.x.bHi', { a: Lx + 1 });
        case 'lo': { let lmax = 0; for (let l = 1; l < R.NL; l++) if (R.layerGrad(k, l) > R.layerGrad(k, lmax)) lmax = l; return T_('q3.x.bLo', { a: Lx - 1, lmax }); }
        case 'down': return T_('q3.x.bDown', { g: g('down') });
        case 'ffn': return T_('q3.x.bFfn', { g1: g('gate'), g2: g('up') });
        case 'ln2': return T_('q3.x.bLn2', { g: g('ln2'), i: f, m: has3 ? n3(R.midGrad(k, Lx, f)) : '…' });
        case 'o': return T_('q3.x.bO', { g: g('o') });
        case 'attn': return T_('q3.x.bAttn');
        case 'qkv': return T_('q3.x.bQkv', { gq: g('q'), gk: g('k'), gv: g('v'), gqn: g('qn'), gkn: g('kn') });
        case 'ln1': return T_('q3.x.bLn1', { g: g('ln1'), l: Lx, i: f, r: n3(R.residGrad(k, Lx, f)), lg: R.layerGrad(k, Lx).toFixed(2) });
        case 'emb': {
          const em = R.emb(k);
          if (!em) return loading;
          return T_('q3.x.bEmb', { top: em.top.slice(0, 5).map(([id, v]) => `「${tk(R.tok(id))}」${n3(v)}`).join(isEn ? ', ' : '、') });
        }
      }
      return '';
    }
    case 'upd': {
      if (s.sub === 'clip') return T_('q3.x.clip', { g: st.gradNorm.toFixed(2), c: st.clip.toFixed(5) });
      if (!has3) return loading;
      if (s.sub === 'adam') return T_('q3.x.adam', { l: Lx, d: R.layerDw(k, Lx).toExponential(3), a: R.lossState(k).toFixed(4), b: R.lossState(k + 1).toFixed(4) });
      const ch = R.changedAfter(k), rv = R.revertAfter(k);
      return T_('q3.x.bf16', { n: k + 1, c: fmtInt(ch), r: fmtInt(rv), p: ((rv / R.total) * 100).toFixed(1) });
    }
  }
  return '';
}

function tensorName(s, R, Lx) {
  const tn = s.t;
  return tn === 'embed' || tn === 'norm' ? T_(`q3.t.${tn}`) : `${T_('q3.layerN', { l: Lx })} ${T_(`q3.t.${tn}`)}`;
}

function explainTensor(s, R, ctx, Lx) {
  const k = s.k, tn = s.t, L_ = lt(tn, Lx);
  if (!R.hasSt(k) || (s.ph === 'upd' && !R.has3(k))) return `<span class="dimmed">${T_('q3.loadingStep')}</span>`;
  const ts = R.tensor(k, L_, tn), sh = TSHAPE[tn];
  const name = esc(tensorName(s, R, Lx)), shape = sh.length === 2 ? `[${sh[0]} × ${sh[1]}]` : `[${sh[0]}]`;
  if (s.ph === 'fwd') return T_('q3.x.tenF', { name, shape, n: fmtInt(ts.n), w: ts.w != null ? n3(ts.w) : '…' });
  if (s.ph === 'bwd') return T_('q3.x.tenB', { name, n: fmtInt(ts.n), k: k + 1, g: n3(ts.g), c: ts.clip.toPrecision(3), gc: n3(ts.g * ts.clip), pre: ts.gPre != null ? T_('q3.x.tenBPre', { v: n3(ts.gPre) }) : '' });
  return T_('q3.x.tenU', { name, dw: n3(ts.dw), mx: n3(ts.dwMax), rel: n3(ts.dw / ts.w), ch: fmtInt(ts.changed), rv: ((ts.revert / ts.n) * 100).toFixed(1) });
}

function explainWeight(s, R, ctx, Lx) {
  const k = s.k, tn = s.t, L_ = lt(tn, Lx);
  const a = R.adamT(k, L_, tn);
  if (!a) return `<span class="dimmed">${T_('q3.loadingStep')}</span>`;
  const idx = R.trackedIndex(L_, tn);
  const name = esc(tensorName(s, R, Lx));
  if (s.ph === 'bwd') return T_('q3.x.wg', { name, idx: `[${idx.join(', ')}]`, g: n5(a.gRaw) });
  const dw = a.w1 - a.w0;
  switch (s.mi) {
    case 'g': return T_('q3.x.aG', { gr: n5(a.gRaw), c: (a.g / a.gRaw).toPrecision(4), g: n5(a.g) });
    case 'm': return T_('q3.x.aM', { m0: n5(a.m0), g: n5(a.g), m: n5(a.m) });
    case 'v': return T_('q3.x.aV', { v0: n5(a.v0), v: n5(a.v) });
    case 'bc': return T_('q3.x.aBc', { mh: n5(a.mh), vh: n5(a.vh), sign: a.t === 1 ? T_('q3.x.aSign') : '' });
    case 'dw': return T_('q3.x.aDw', { dw: n5(dw), r: n5(a.mh / (Math.sqrt(a.vh) + a.eps)), wd: a.wd ? T_('q3.x.aDwWd', { v: n5(a.wd * a.w0) }) : T_('q3.x.aDwNo') });
    case 'write': return T_('q3.x.aW', { w0: n5(a.w0), w1: Number(a.w1.toPrecision(8)), rel: n3(dw / a.w0) });
    case 'bits': {
      const a0 = R.adamT(0, L_, tn), orig = a0 ? a0.w0 : a.w0, back = bf16Round(a.w1);
      return T_('q3.x.aBits', { w0: n5(orig), n: k + 1, w1: Number(a.w1.toPrecision(8)), back: n5(back), same: back === orig ? T_('q3.x.aBitsSame') : T_('q3.x.aBitsDiff') });
    }
  }
  return '';
}

/* ---------------------------------------------------------------- 变量监视 */

export function watch(s, R, ctx, depth) {
  const k = s.k, m = R.D.meta, st = m.steps[k], Lx = LX(ctx, R), f = FOCUS(ctx, R);
  const rows = [['wh', 'QWEN3-0.6B · SFT'], ['step', s.ph === 'end' ? `${R.K} / ${R.K}` : `${k + 1} / ${R.K}`], ['lr', '1e-5'], ['loss', R.lossState(s.ph === 'end' ? R.K : k).toFixed(4)], ['loss_all', R.lossPreState(s.ph === 'end' ? R.K : k).toFixed(4)]];
  if (s.ph !== 'end') rows.push(['‖g‖', st.gradNorm.toFixed(2)], ['clip', `×${st.clip.toFixed(5)}`]);
  if (depth >= 3) rows.push(['wh', L('选中', 'SELECTED')], [T_('q3.w.layer'), `L${Lx}`], [T_('q3.w.pos'), `${f}「${tokPlain(R.inStr(f))}」`], [T_('q3.w.tgt'), tokPlain(R.tgtStr(f))]);
  if (depth >= 2 && R.hasSt(k) && s.ph !== 'end') {
    rows.push(['wh', L(`位置 ${f}`, `POSITION ${f}`)], ['p', fmtP(R.pState(k, f))], ['−ln p', R.nllState(k, f).toFixed(4)], [T_('q3.w.lens'), `L${Lx}: ${tokPlain(R.lensTop(k, Lx + 1, f).s)} · ${fmtP(R.lensP(k, Lx + 1, f))}`]);
  }
  if (s.t && R.hasSt(k)) {
    const ts = R.tensor(k, lt(s.t, Lx), s.t);
    rows.push(['wh', T_(`q3.t.${s.t}`)], ['‖∇‖', n3(ts.g)]);
    if (ts.dw != null) rows.push(['‖Δw‖', n3(ts.dw)], ['‖w‖', n3(ts.w)]);
  }
  if (s.mi && s.ph === 'upd') {
    const a = R.adamT(k, lt(s.t, Lx), s.t);
    if (a) rows.push(['wh', 'ADAMW'], ['g', n5(a.g)], ['m', n5(a.m)], ['v', n5(a.v)], ['m̂', n5(a.mh)], ['v̂', n5(a.vh)], ['w', `${n5(a.w0)} → ${Number(a.w1.toPrecision(8))}`]);
  }
  return rows;
}

void LAYER_T;
