// 调试器里的“代码 / 这一步 / 变量”三块内容，全部用真实记录的数值填充。
import { esc } from '../../js/ui.js';
import { shapes, TENSOR_NAME } from './run.js';
import { fmtP, sciSup, fmtInt } from './draw.js';

const K_ = (s) => `<span class="kw">${s}</span>`;
const F_ = (s) => `<span class="fn">${s}</span>`;
const C_ = (s) => `<span class="cm"># ${s}</span>`;
const N_ = (s) => `<span class="nu">${s}</span>`;

export function codeFor(key, R) {
  if (key === 'pipe') return [
    `base = ${F_('pretrain')}(随机初始化, 海量文本)`,
    `chat = ${F_('sft')}(base, 对话)    ${C_('只算回答')}`,
    `final = ${F_('align')}(chat, 偏好) ${C_('DPO / RL')}`,
  ];
  if (key === 'tiny') {
    const m = R.D.meta;
    return [
      `model = ${F_('Qwen3ForCausalLM')}(cfg) ${C_(`${(m.model.params / 1e4).toFixed(0)} 万参数`)}`,
      `opt = ${F_('AdamW')}(lr=${N_('3e-3')}, wd=${N_('0.1')})`,
      `${K_('for')} step ${K_('in')} ${F_('range')}(${N_(m.train.steps)}):`,
      `    x = ${F_('next_batch')}()     ${C_(`${m.train.batch}×${m.train.seq + 1} 字`)}`,
      `    inp, tgt = x[:, :${N_('-1')}], x[:, ${N_('1')}:]`,
      `    logits = ${F_('model')}(inp)  ${C_('前向')}`,
      `    loss = ${F_('cross_entropy')}(logits, tgt)`,
      `    loss.${F_('backward')}()      ${C_('反向')}`,
      `    ${F_('clip_grad_norm_')}(params, ${N_('1.0')})`,
      `    lr = ${F_('warmup_cosine')}(step)`,
      `    ${K_('for')} w, g ${K_('in')} params: ${C_('AdamW')}`,
      `        m = β1·m + (${N_('1')}−β1)·g`,
      `        v = β2·v + (${N_('1')}−β2)·g²`,
      `        m̂, v̂ = m/(${N_('1')}−β1ᵗ), v/(${N_('1')}−β2ᵗ)`,
      `        w −= lr·(m̂/(√v̂+ε) + λ·w)`,
    ];
  }
  const m = R.D.meta;
  return [
    `model = ${F_('from_pretrained')}(<span class="nu">"Qwen3-0.6B"</span>)`,
    `ids = ${F_('apply_chat_template')}(对话) ${C_(`${m.ids.length} 个`)}`,
    `labels = ${F_('mask_prompt')}(ids)  ${C_('提示 → −100')}`,
    `${K_('for')} step ${K_('in')} ${F_('range')}(${N_(m.steps.length)}):  ${C_('同一条对话')}`,
    `    logits = ${F_('model')}(ids[:${N_('-1')}])  ${C_('前向')}`,
    `    loss = ${F_('cross_entropy')}(logits, labels[${N_('1')}:])`,
    `    loss.${F_('backward')}()      ${C_('反向')}`,
    `    ${F_('clip_grad_norm_')}(params, ${N_('1.0')})`,
    `    ${K_('for')} w, g ${K_('in')} params: ${C_('lr = 1e-5')}`,
    `        m = β1·m + (${N_('1')}−β1)·g`,
    `        v = β2·v + (${N_('1')}−β2)·g²`,
    `        m̂, v̂ = m/(${N_('1')}−β1ᵗ), v/(${N_('1')}−β2ᵗ)`,
    `        w −= lr·(m̂/(√v̂+ε) + λ·w)`,
  ];
}

export function linesFor(s, R, depth) {
  if (depth === 0) return [s.st + 1];
  const T = R.kind === 'tiny';
  const UPD = T ? { clip: [9], g: [11], m: [12], v: [13], bc: [14], dw: [10, 15], write: [15] } : { clip: [8], g: [9], m: [10], v: [11], bc: [12], dw: [13], write: [13] };
  switch (s.ph) {
    case 'ck': return T ? [3] : [4];
    case 'batch': return !s.sub ? (T ? [4, 5] : [2, 3]) : T ? { row: [4], ids: [4], shift: [5] }[s.sub] : { tpl: [2], mask: [3], shift: [5, 6] }[s.sub];
    case 'fwd': return T ? [6] : [5];
    case 'loss': return T ? [7] : [6];
    case 'bwd': return T ? [8] : [7];
    case 'upd': return s.sub ? UPD[s.sub] : T ? [9, 10, 11, 12, 13, 14, 15] : [8, 9, 10, 11, 12, 13];
    case 'check': return T ? [6, 7] : [5, 6];
  }
  return [];
}

/* ---------------------------------------------------------------- 名字 */

export const PH_NAME = { batch: '批次', fwd: '前向', loss: '损失', bwd: '反向', upd: '更新', check: '重新前向' };
const SUB_NAME = { row: '取一行', ids: '字 → 编号', shift: '错开一位', tpl: '套聊天模板', mask: 'SFT 遮罩', clip: '梯度裁剪', g: '梯度 g', m: '一阶动量 m', v: '二阶动量 v', bc: '偏差校正', dw: '算出 Δw', write: '写回' };
const CE_NAME = { softmax: 'softmax', pick: '取正确答案的概率', log: '−ln p' };
const BITS_NAME = { fp32a: '更新前的 fp32', fp32b: '更新后的 fp32', bf16: '如果只存 bf16' };
const PIPE = ['预训练', '监督微调', '偏好对齐 / 强化学习'];

const disp = (s) => (s === '\n' ? '↵' : s === '\n\n' ? '↵↵' : String(s).replace(/\n/g, '↵'));
export const tokSpan = (s, role = 'answer') => `<span class="tok r-${role}${/^<\|.*\|>$|^<\/?think>$/.test(s) ? ' sp' : ''}">${esc(disp(s))}</span>`;

function bName(R, i) {
  if (i === 0) return '嵌入';
  return i === R.NL ? `第 ${i - 1} 层（最后一层）` : `第 ${i - 1} 层`;
}

export function stepLabel(s, R, depth) {
  if (s.ph === 'pipe') return PIPE[s.st];
  if (s.ph === 'ck') return `第 ${fmtInt(R.stepNo(s.k))} 步`;
  if (!s.sub) return s.ph === 'check' ? '重新前向（演示）' : PH_NAME[s.ph];
  switch (s.ph) {
    case 'batch': return `批次 · ${SUB_NAME[s.sub]}`;
    case 'fwd': return `前向 · ${bName(R, s.i)}`;
    case 'bwd': return `反向 · ${bName(R, s.i)}${s.mi ? ` · ${TENSOR_NAME[s.mi]}` : ''}`;
    case 'loss': return `损失 · 位置 ${s.i}「${disp(R.tok(s.k, s.i + 1))}」${s.mi ? ` · ${CE_NAME[s.mi]}` : ''}`;
    case 'upd': return `更新 · ${SUB_NAME[s.sub]}${s.mi ? ` · ${BITS_NAME[s.mi]}` : ''}`;
  }
  return '';
}

export function crumbs(depth, s, R) {
  const out = [{ d: 0, label: '流水线' }];
  if (depth === 0) return out;
  out.push({ d: 1, label: R.kind === 'tiny' ? '预训练全程' : 'SFT 三步' });
  if (depth >= 2) out.push({ d: 2, label: `第 ${fmtInt(R.stepNo(s.k))} 步` });
  if (depth >= 3 && s.sub) out.push({ d: 3, label: PH_NAME[s.ph] });
  if (depth >= 4 && s.mi) {
    const lab = s.ph === 'loss' ? `位置 ${s.i}` : s.ph === 'bwd' ? bName(R, s.i) : '写回的比特';
    out.push({ d: 4, label: lab });
  }
  return out;
}

export function shapeOf(s, R, depth) {
  if (depth === 0 || s.ph === 'ck') return '';
  const T = R.kind === 'tiny';
  const M = R.model;
  const m = R.D.meta;
  const B = T ? m.train.batch : 1, L = T ? m.train.seq : R.rowLen;
  if (s.ph === 'batch') return T ? `x <b>[${B} × ${L + 1}]</b> → inp <b>[${B} × ${L}]</b>，tgt <b>[${B} × ${L}]</b>` : `ids <b>[1 × ${L + 1}]</b> → inp <b>[1 × ${L}]</b>，labels <b>[1 × ${L}]</b>（${m.nPrompt - 1} 个是 −100）`;
  if (s.ph === 'fwd' || s.ph === 'check') return `inp <b>[${B} × ${L}]</b> → 嵌入 <b>[${B} × ${L} × ${M.hidden}]</b> → ${M.layers} 层 → logits <b>[${B} × ${L} × ${fmtInt(M.vocab)}]</b>`;
  if (s.ph === 'loss') return `logits <b>[${B} × ${L} × ${fmtInt(M.vocab)}]</b> → 每个位置 −ln p → 平均成一个数`;
  if (s.ph === 'bwd') {
    if (s.mi) { const sh = shapes(M)[s.mi]; return `∇${TENSOR_NAME[s.mi]} <b>[${sh[0]} × ${sh[1]}]</b>（和权重同形状，${fmtInt(sh[0] * sh[1])} 个数）`; }
    return `∂L/∂h <b>[${B} × ${L} × ${M.hidden}]</b>；每个参数都得到一个同形状的梯度，共 <b>${fmtInt(M.params)}</b> 个数`;
  }
  if (s.ph === 'upd') return `m、v：和参数同形状，各 <b>${fmtInt(M.params)}</b> 个 fp32（${((M.params * 8) / 2 ** 20).toFixed(0)} MB）`;
  return '';
}

/* ---------------------------------------------------------------- 讲解 */

const P = (p) => `<em>${fmtP(p)}</em>`;
const f4 = (v) => (Math.abs(v) >= 1e-3 || v === 0 ? Number(v.toPrecision(5)).toString() : sciSup(v, 4));

function tinyNarrative(R, k) {
  const D = R.D, m = D.meta, t = m.ckpts[k].t;
  let punct = 0, np = 0, cont = 0, nc = 0;
  for (let i = 0; i < D.Lv; i++) {
    const c = m.held.text[i] ?? '⏎';
    const p = D.valP(k, i);
    if (c === '，' || c === '。') { punct += p; np++; } else if (c !== '⏎') { cont += p; nc++; }
  }
  punct /= np; cont /= nc;
  let s = '';
  if (t < 20) s = `刚刚随机初始化：模型对 ${fmtInt(m.model.vocab)} 个字几乎一视同仁，损失接近 ln ${m.model.vocab} = ${Math.log(m.model.vocab).toFixed(2)}，写出来的是随机的字。`;
  else if (punct < 0.6) s = '它先学会的是最常见的字和标点的大致频率——写出来的字开始“像诗里的字”，但还没有句子。';
  else if (cont < 0.08) s = `<b>格式先于内容</b>：《登鹳雀楼》里标点的平均概率已经到 ${P(punct)}，而正文的字只有 ${P(cont)}。它先学会了“五个字一个逗号”。`;
  else s = `正文的字也在变得可猜：《登鹳雀楼》正文平均 ${P(cont)}，标点 ${P(punct)}。它开始记住常见的搭配（比如“千”后面接“里”）。`;
  return s;
}

export function explain(s, R, ctx, depth) {
  if (s.ph === 'pipe') return pipeExplain(s.st, ctx);
  const T = R.kind === 'tiny';
  const m = R.D.meta;
  const k = s.k;
  const t = R.stepNo(k);
  if (s.ph === 'ck') {
    if (T) {
      const c = m.ckpts[k];
      const lrPhase = c.t < m.train.warmup ? `还在预热（前 ${m.train.warmup} 步线性升到 ${m.train.peakLr}）` : '在余弦退火中慢慢变小';
      return `第 <b>${fmtInt(t)}</b> 步（共 ${fmtInt(m.train.steps)} 步）。这一批的损失 <b>${c.loss.toFixed(3)}</b>，没训练过的验证集 <b>${c.valLoss.toFixed(3)}</b>；学习率 ${sciSup(c.lr, 3)}，${lrPhase}。<br>${tinyNarrative(R, k)}<br><span class="dimmed">按 ＋ 拆开这一步：批次 → 前向 → 损失 → 反向 → 更新。拖动损失曲线可以来回看。</span>`;
    }
    const a = m.states[k].lossSft, b = m.states[k + 1].lossSft;
    const hei = m.nPrompt + 1;
    return `第 <b>${t}</b> 步（共 ${m.steps.length} 步，都在同一条对话上）。回答部分的损失 <b>${a.toFixed(3)}</b> → <b>${b.toFixed(4)}</b>；回答里的「黑」在“我是”之后出现的概率 ${P(m.states[k].p[hei - 1])} → ${P(m.states[k + 1].p[hei - 1])}。<br>梯度范数 ${m.steps[k].gradNorm.toFixed(1)}，裁剪到 1.0（系数 ${m.steps[k].clip.toFixed(4)}）；学习率 ${m.train.lr}。<br><span class="dimmed">${m.train.note}。按 ＋ 拆开这一步。</span>`;
  }
  switch (s.ph) {
    case 'batch': return batchExplain(s, R);
    case 'fwd': return fwdExplain(s, R);
    case 'loss': return lossExplain(s, R);
    case 'bwd': return bwdExplain(s, R);
    case 'upd': return updExplain(s, R, ctx);
    case 'check': {
      const before = R.loss(k), after = R.lossAfter(k);
      return `<span class="demo">演示</span> 更新完以后，对<b>同一个批次</b>再前向一次：${T ? '这一批的' : '回答部分的'}损失 <b>${before.toFixed(4)}</b> → <b>${after.toFixed(4)}</b>（${after < before ? '降了' : '反而升了'} ${Math.abs(before - after).toFixed(4)}）。<br>${T ? '真实训练不会这样做——下一步直接换一批新的数据。这里只是为了看清“走一步”到底改变了什么。' : '下一步还是这条对话，所以这个“更新后”的模型就是下一步的起点。'}`;
    }
  }
  return '';
}

function pipeExplain(st, ctx) {
  const tm = ctx.tiny?.D.meta, qm = ctx.qwen?.D.meta;
  if (st === 0) return `<b>预训练</b>：把海量文本切成词元，让模型一遍遍“猜下一个词元”，<b>每个位置都算损失</b>。语言、知识、文风都是在这一步学到的。<br>本页实测：在 ${tm ? fmtInt(tm.corpus.poems.train) : ''} 首唐宋格律诗上，从随机初始化训练一个 Qwen3 同构的小模型（${tm ? (tm.model.params / 1e4).toFixed(0) : ''} 万参数），${tm ? fmtInt(tm.train.steps) : ''} 步，用时 ${tm ? tm.train.seconds : ''} 秒，验证损失 ${tm ? tm.ckpts[0].valLoss.toFixed(2) : ''} → ${tm ? tm.train.finalVal.toFixed(2) : ''}。<br><span class="dimmed">对照：Qwen3 的预训练用了约 36 万亿个词元、119 种语言（据 Qwen3 技术报告）。按 ＋ 看它从乱码学会写诗。</span>`;
  if (st === 1) return `<b>监督微调（SFT）</b>：用一问一答的对话继续训练。损失函数和预训练一模一样，区别只有一个：<b>提示部分的词元被遮掉，只对助手的回答算损失</b>。模型因此学会“轮到我时该怎么答”。<br>本页实测：Qwen3-0.6B 全参数、fp32，在一条“自我认知”对话上走 3 步，回答部分的损失 ${qm ? qm.states.map((x) => x.lossSft.toFixed(qm.states.indexOf(x) === 3 ? 3 : 2)).join(' → ') : ''}。<br><span class="dimmed">按 ＋ 看一步训练，一直细到单个权重的 AdamW 算式和比特。</span>`;
  const d = qm?.dpo;
  return `<b>偏好对齐 / 强化学习</b>：不再给标准答案，而是告诉模型“这个回答比那个好”（DPO、RLHF），或者用奖励直接强化（PPO、GRPO）。<span class="demo">流程为示意</span><br>${d ? `舞台上的 DPO 损失是真实算出来的一个例子：把“新回答”和原模型自己的回答，分别在微调后的模型和原模型上算对数概率，代入公式得到 ${d.loss.toFixed(4)}。这里只算了一次损失，没有真的做偏好训练。` : ''}<br><span class="dimmed">据 Qwen3 技术报告：旗舰模型的后训练分四段——长思维链冷启动、推理强化学习（GRPO）、思考模式融合、通用强化学习；0.6B 这样的小模型主要靠从大模型蒸馏。</span>`;
}

function batchExplain(s, R) {
  const T = R.kind === 'tiny', m = R.D.meta, k = s.k;
  if (T) {
    if (!s.sub) return `从 ${(m.corpus.chars.train / 1e4).toFixed(0)} 万字的训练语料里随机挑 ${m.train.batch} 个位置，每处从一首诗的开头起，连续取 ${m.train.seq + 1} 个字（会跨过好几首诗，诗与诗之间用 ⏎ 分隔）。<br>舞台上是第 0 行的前 ${R.rowLen} 个字。一个批次 = ${fmtInt(m.train.tokensPerStep)} 道“猜下一个字”的题。`;
    if (s.sub === 'row') return `第 0 行：从某首诗的开头开始，连续 ${m.train.seq + 1} 个字。<b>⏎</b> 是 &lt;|endoftext|&gt;，和 Qwen3 一样用它分隔文档。<br>训练数据就是这样一条很长的带子，模型并不知道“诗”是什么，只看到一串字。`;
    if (s.sub === 'ids') return `<b>字级分词</b>：每个字查词表换成一个编号。这个词表按字频排序：「，」是 1 号、「。」是 2 号……一共 ${fmtInt(m.model.vocab)} 个（含 &lt;|endoftext|&gt;）。<br><span class="dimmed">真实的 Qwen3 用 15 万词表的 BPE 分词，一个词元可能是半个字、一个字或一个词。</span>`;
    return `<b>错开一位</b>：输入是第 0–${m.train.seq - 1} 个字，目标是第 1–${m.train.seq} 个字。位置 i 只能看到前 i+1 个字，要猜第 i+1 个。<br>一行 ${m.train.seq} 道题、一批 ${fmtInt(m.train.tokensPerStep)} 道题，在一次前向里同时作答。`;
  }
  const st = m.states[k];
  if (!s.sub) return `SFT 的一个“批次”就是这一条对话：套上聊天模板，一共 ${m.ids.length} 个词元。其中只有回答的 ${R.D.sftPos.length} 个位置计入损失。`;
  if (s.sub === 'tpl') return `聊天模板把一问一答拼成 ${m.ids.length} 个词元：${tokSpan('<|im_start|>', 'tpl')} user … ${tokSpan('<|im_end|>', 'tpl')} ${tokSpan('<|im_start|>', 'tpl')} assistant ${tokSpan('<think>', 'tpl')}${tokSpan('</think>', 'tpl')} 回答 ${tokSpan('<|im_end|>', 'tpl')}。<br>非思考模式下，模板会在回答前塞一个空的思考块——推理时模型也会看到同样的格式。`;
  if (s.sub === 'mask') {
    const pc = m.pretrainCompare;
    return `<b>SFT 的关键</b>：labels 里提示部分（前 ${m.nPrompt} 个词元）设成 −100，不算损失。剩下 ${R.D.sftPos.length} 个位置：回答的 ${R.D.sftPos.length - 1} 个词元和结束标记 ${tokSpan('<|im_end|>', 'tpl')}。<br>此刻：只算回答 <b>${st.lossSft.toFixed(3)}</b>；如果像预训练那样全都算，是 <b>${st.lossPre.toFixed(3)}</b>。${k === 0 ? `<br>在原模型上，全都算时梯度范数是 ${pc.total.toFixed(1)}，只算回答是 ${m.steps[0].gradNorm.toFixed(1)}——遮罩决定了模型往哪个方向学。` : ''}`;
  }
  return `和预训练一样错开一位：位置 i 预测第 i+1 个词元。${R.rowLen} 个位置里只有 ${R.D.sftPos.length} 个的目标不是 −100。`;
}

function lensStats(R, k, b) {
  let ok = 0, n = 0, p = 0;
  for (let i = 0; i < R.rowLen; i++) {
    if (!R.counted(i)) continue;
    n++;
    if (R.lensTop(k, b, i).ok) ok++;
    p += R.lensP(k, b, i);
  }
  return { acc: ok / Math.max(1, n), p: p / Math.max(1, n), n };
}

function fwdExplain(s, R) {
  const T = R.kind === 'tiny', k = s.k, M = R.model;
  if (!s.sub) return `${T ? `${R.D.meta.train.batch} × ${R.D.meta.train.seq}` : `${R.rowLen}`} 个位置一起送进 ${M.layers} 层 Transformer，<b>每个位置同时预测它的下一个${T ? '字' : '词元'}</b>；因果遮罩保证它看不到后面。<br>这叫<b>教师强制</b>：训练时每个位置拿到的都是真实的上文，而不是模型自己写的字，所以所有位置可以并行。`;
  const st = lensStats(R, k, s.i);
  const what = `格子里是<b>逻辑透镜</b>的读数：把这一层的残差流直接接最终 RMSNorm + 输出矩阵，看它此刻会猜什么；颜色越亮，猜中正确答案的概率越高。`;
  if (s.i === 0) return `<b>嵌入</b>：每个${T ? '字' : '词元'}查表变成 ${M.hidden} 维向量。${what}<br>只靠嵌入就能猜中 ${P(st.acc)}——相当于只看“当前是哪个${T ? '字' : '词元'}”的统计规律。`;
  if (s.i === R.NL) return `<b>最后一层 → 输出头</b>：这一行就是模型真正的预测。猜中第一名的位置占 ${P(st.acc)}，正确答案的平均概率 ${P(st.p)}。${T ? '' : '（只统计计入损失的位置）'}`;
  return `<b>${bName(R, s.i)}</b>：注意力让每个位置看前面的${T ? '字' : '词元'}，前馈网络再加工，结果加回残差流。${what}<br>这一层读出来：猜中 ${P(st.acc)}，正确答案平均概率 ${P(st.p)}。`;
}

function lossExplain(s, R) {
  const T = R.kind === 'tiny', k = s.k, m = R.D.meta;
  if (!s.sub) {
    return T ? `每个位置的损失 = <b>−ln p(正确的下一个字)</b>。${fmtInt(m.train.tokensPerStep)} 个位置取平均：<b>${R.loss(k).toFixed(4)}</b>。<br>舞台上是第 0 行前 ${R.rowLen} 个位置各自的损失：标点、常见搭配几乎不扣分，诗的第一个字最难猜。`
      : `只在回答的 ${R.D.sftPos.length} 个位置上算 −ln p，取平均：<b>${R.loss(k).toFixed(4)}</b>。提示部分的位置也有预测（灰色），只是不计入。`;
  }
  const i = s.i, p = R.p(k, i), tgt = R.tok(k, i + 1);
  const ctxStr = Array.from({ length: Math.min(i + 1, T ? 10 : 6) }, (_, j) => R.tok(k, i - Math.min(i, T ? 9 : 5) + j)).map(disp).join(T ? '' : '·');
  const top = R.top(k, i).map((t) => `${esc(disp(t.s))} ${fmtP(t.p)}`).join('、');
  if (!s.mi) return `位置 <b>${i}</b>：上文「…${esc(ctxStr)}」，正确答案「<b>${esc(disp(tgt))}</b>」。<br>模型给它的概率 ${P(p)}，损失 −ln p = <b>${(-Math.log(Math.max(p, 1e-12))).toFixed(3)}</b>。<br>它最想说的前 5 名：${top}。`;
  if (s.mi === 'softmax') return `输出头给词表里 <b>${fmtInt(R.model.vocab)}</b> 个${T ? '字' : '词元'}各打一个分数（logit），softmax 把分数变成概率：先取 e 的指数，再除以总和，所有概率加起来等于 1。舞台上是概率最高的 5 个和“其余所有”。`;
  if (s.mi === 'pick') return `损失只看<b>正确答案那一项</b>：p(「${esc(disp(tgt))}」) = ${P(p)}。其他 ${fmtInt(R.model.vocab - 1)} 项不直接出现在损失里，但它们在 softmax 的分母里——梯度会把它们统统往下压一点，把正确答案往上抬。`;
  return `取负对数：损失 = −ln ${p < 1e-3 ? sciSup(p, 3) : p.toFixed(4)} = <b>${(-Math.log(Math.max(p, 1e-12))).toFixed(3)}</b>。p = 1 时损失为 0；p 越小损失越大，p → 0 时趋于无穷。<br>完全瞎猜（p = 1/${fmtInt(R.model.vocab)}）的损失是 ${Math.log(R.model.vocab).toFixed(2)}，也就是训练刚开始时的损失。`;
}

function layerGradSum(R, k, L) {
  return Math.sqrt(['q', 'k', 'v', 'o', 'gate', 'up', 'down', 'ln1', 'ln2', 'qn', 'kn'].reduce((a, t) => a + R.tgrad(k, L, t) ** 2, 0));
}

function residGradMean(R, k, b) {
  let s = 0;
  for (let i = 0; i < R.rowLen; i++) s += R.residGrad(k, b, i);
  return s / R.rowLen;
}

function bwdExplain(s, R) {
  const T = R.kind === 'tiny', k = s.k, M = R.model;
  if (!s.sub) return `<b>loss.backward()</b> 用链式法则，把损失对每个参数的偏导数（梯度）算出来：从输出头开始，一层一层往回传，和前向的方向正好相反。<br>所有 ${fmtInt(M.params)} 个梯度合起来的长度 ‖g‖ = <b>${R.gradNorm(k).toFixed(T ? 3 : 1)}</b>。`;
  const i = s.i;
  const gm = residGradMean(R, k, i);
  if (s.mi) {
    const L = i - 1, sh = shapes(M)[s.mi], gn = R.tgrad(k, L, s.mi);
    const all = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'].map((t) => R.tgrad(k, L, t));
    const rank = all.filter((v) => v > gn).length + 1;
    return `第 ${L} 层的 <b>${TENSOR_NAME[s.mi]}</b>（${sh[0]} × ${sh[1]}）：梯度范数 ‖∇W‖ = <b>${sciSup(gn, 3)}</b>，在这一层 7 个矩阵里排第 ${rank}。<br>每个矩阵的梯度 = 流进这个矩阵的“上游梯度”和它的“输入”做外积，再把所有位置加起来。`;
  }
  const extra = T ? '' : '<br>注意：提示部分的位置虽然不计入损失，梯度照样流过它们——回答里的词元在注意力里看过它们。';
  if (i === R.NL) return `梯度从输出头（最终 RMSNorm + 输出矩阵）流进<b>最后一层的残差流</b>。每个位置的 ‖∂L/∂h‖ 平均 ${sciSup(gm, 3)}。${extra}`;
  if (i === 0) return `梯度流到<b>嵌入</b>：每个位置的梯度会加到它那个${T ? '字' : '词元'}的嵌入行上。因为输出矩阵和嵌入是同一块权重，这一行的梯度还要加上输出头那边传来的一份。<br>嵌入的梯度范数 ${sciSup(R.tgrad(k, -1, 'embed'), 3)}。`;
  return `梯度流过 <b>第 ${i - 1} 层</b>：残差连接让它可以直接抄近路往下传，注意力和前馈网络再各自贡献一份。<br>这一层输入处 ‖∂L/∂h‖ 平均 ${sciSup(gm, 3)}；这一层所有参数的梯度合计 ${sciSup(layerGradSum(R, k, i - 1), 3)}。<span class="dimmed">按 ＋ 看 7 个矩阵各分到多少。</span>`;
}

function updExplain(s, R, ctx) {
  const T = R.kind === 'tiny', k = s.k, m = R.D.meta;
  const f = ctx.feat ?? 0;
  const ft = R.feats[f];
  const a = R.adam(k, f);
  const lr = R.lr(k), b1 = 0.9, b2 = 0.95, eps = 1e-8;
  const tt = a.t;
  const name = `<b>${esc(ft.label)}</b>`;
  if (!s.sub) return `AdamW 用梯度更新<b>每一个</b>权重，${fmtInt(R.model.params)} 个各算各的。学习率 ${sciSup(lr, 3)}${T ? `（${R.D.meta.ckpts[k].t < m.train.warmup ? '预热中' : '余弦退火中'}）` : ''}。<br>舞台上跟踪其中一个：${name}。按 ＋ 一项一项看它的算式。`;
  const step = m.ckpts ? '' : '';
  switch (s.sub) {
    case 'clip': return `所有梯度合起来的长度 ‖g‖ = <b>${R.gradNorm(k).toFixed(T ? 4 : 2)}</b>。${R.clip(k) < 1 ? `超过 1.0，整体乘以 ${R.clip(k).toFixed(5)} 缩到 1.0：` : '没超过 1.0，不用裁剪：'}这个权重的梯度 ${f4(a.gRaw)} → <b>${f4(a.g)}</b>。<br>裁剪只改长度、不改方向，防止某一批数据把模型“踹”得太远。${!T ? '<br><span class="dimmed">有意思的是：Adam 用 m/√v 做更新，梯度整体缩放几乎不影响结果——尤其是第 1 步。</span>' : step}`;
    case 'g': return `${name}的梯度 g = <b>${f4(a.g)}</b>：这个权重增大一点点，损失就会${a.g > 0 ? '上升' : '下降'}约 |g| 倍那么多。所以要往 ${a.g > 0 ? '减小' : '增大'}的方向挪。`;
    case 'm': return `<b>一阶动量</b>（梯度的滑动平均）：m = 0.9 × ${f4(a.m0)} + 0.1 × ${f4(a.g)} = <b>${f4(a.m)}</b>。<br>它记住过去的方向，抹平批次之间的噪声，像一个有惯性的小球。`;
    case 'v': return `<b>二阶动量</b>（梯度平方的滑动平均）：v = 0.95 × ${f4(a.v0)} + 0.05 × (${f4(a.g)})² = <b>${f4(a.v)}</b>。<br>√v 是这个权重梯度的“典型大小”，用来给它单独定步长。`;
    case 'bc': return `<b>偏差校正</b>：m、v 都从 0 开始累积，前几步会偏小，所以除以 (1 − βᵗ)，t = ${tt}：<br>m̂ = ${f4(a.m)} / ${(1 - b1 ** tt).toPrecision(4)} = <b>${f4(a.mh)}</b><br>v̂ = ${f4(a.v)} / ${(1 - b2 ** tt).toPrecision(4)} = <b>${f4(a.vh)}</b>${tt === 1 ? '<br>第 1 步时 m̂ 正好等于 g、v̂ 正好等于 g²：更新量 = lr × g/|g|，只剩下符号。' : ''}`;
    case 'dw': {
      const adam = -lr * a.mh / (Math.sqrt(a.vh) + eps), decay = -lr * a.wd * a.w0;
      return `Δw = −lr × m̂/(√v̂+ε) − lr × λ × w = ${sciSup(adam, 3)} ${decay >= 0 ? '+' : '−'} ${sciSup(Math.abs(decay), 3)} = <b>${sciSup(adam + decay, 3)}</b>。<br>m̂/√v̂ 的大小通常在 1 左右，所以每个权重每一步大约挪动 lr 那么远，不管它的梯度本身是大是小。${a.wd ? `权重衰减 λ = ${a.wd} 每步把权重往 0 拉一点点。` : '这个参数是 RMSNorm 的缩放 γ，按惯例不做权重衰减。'}`;
    }
    case 'write': return `写回：w = ${f4(a.w0)} + (${sciSup(a.w1 - a.w0, 3)}) = <b>${f4(a.w1)}</b>。<br>改动只有原值的 ${sciSup(Math.abs((a.w1 - a.w0) / a.w0), 2)}。${s.mi ? bitsExplain(s.mi, a) : '<span class="dimmed">按 ＋ 看这个数的比特。</span>'}`;
  }
  return '';
}

// float32 / bf16 的比特
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
export function f32Bits(x) { f32[0] = x; return u32[0] >>> 0; }
export function bf16Round(x) {
  f32[0] = x;
  const b = u32[0];
  const r = ((b + 0x7fff + ((b >>> 16) & 1)) >>> 16) << 16;
  u32[0] = r >>> 0;
  return f32[0];
}
export function f32Of(bits) { u32[0] = bits >>> 0; return f32[0]; }

function bitsExplain(mi, a) {
  const b0 = f32Bits(a.w0), b1 = f32Bits(a.w1);
  let diff = 0;
  for (let i = 0; i < 32; i++) if (((b0 ^ b1) >>> i) & 1) diff++;
  if (mi === 'fp32a') return '<br>fp32：1 位符号、8 位指数、23 位尾数。';
  if (mi === 'fp32b') return `<br>和更新前比，32 个比特里变了 <b>${diff}</b> 个，几乎都在尾数的末尾。`;
  const r0 = bf16Round(a.w0), r1 = bf16Round(a.w1);
  return r0 === r1 ? `<br>如果只用 bf16（7 位尾数）存权重，更新前后舍入成<b>同一个数</b> ${r0}——这一步更新被吞掉了。所以训练时要另存一份 fp32 的<b>主权重</b>。` : `<br>如果只用 bf16 存，更新前后是 ${r0} → ${r1}，这次幸运地跨过了一个 bf16 的间隔；但大多数权重没这么幸运。`;
}

/* ---------------------------------------------------------------- 变量监视 */

export function watch(s, R, ctx, depth) {
  if (s.ph === 'pipe') return [['wh', 'PIPELINE'], ['stage', ['pretrain', 'sft', 'align'][s.st]]];
  const T = R.kind === 'tiny', k = s.k, m = R.D.meta;
  const rows = [['wh', T ? 'TINY QWEN3 · 唐宋诗' : 'QWEN3-0.6B · SFT'], ['step', `${fmtInt(R.stepNo(k))} / ${fmtInt(T ? m.train.steps : m.steps.length)}`], ['lr', sciSup(R.lr(k), 3)], ['loss', R.loss(k).toFixed(4)], ['‖g‖', R.gradNorm(k).toFixed(T ? 4 : 2)], ['clip', R.clip(k) < 1 ? `×${R.clip(k).toFixed(5)}` : '—']];
  if (T) rows.push(['tokens', `${fmtInt(R.tokensSeen(k))}`]);
  if (s.ph === 'loss' && s.sub) {
    const p = R.p(k, s.i);
    rows.push(['wh', 'POSITION'], ['i', s.i], ['target', disp(R.tok(k, s.i + 1))], ['p', p.toPrecision(4)], ['−ln p', (-Math.log(Math.max(p, 1e-12))).toFixed(4)]);
  }
  if ((s.ph === 'fwd' || s.ph === 'bwd') && s.sub) {
    let nm = 0;
    for (let i = 0; i < R.rowLen; i++) nm += R.residNorm(k, s.i, i);
    rows.push(['wh', 'RESIDUAL'], ['boundary', bName(R, s.i)], ['mean ‖h‖', (nm / R.rowLen).toFixed(3)]);
    if (s.ph === 'bwd') rows.push(['mean ‖∂L/∂h‖', sciSup(residGradMean(R, k, s.i), 3)]);
    if (s.mi) rows.push(['‖∇W‖', sciSup(R.tgrad(k, s.i - 1, s.mi), 4)]);
  }
  if (s.ph === 'upd') {
    const f = ctx.feat ?? 0, a = R.adam(k, f);
    rows.push(['wh', 'ADAMW · 一个权重'], ['t', a.t], ['w', f4(a.w0)], ['g', f4(a.g)], ['m', f4(a.m)], ['v', f4(a.v)], ['m̂', f4(a.mh)], ['v̂', f4(a.vh)], ['Δw', sciSup(a.w1 - a.w0, 4)], ['w′', f4(a.w1)]);
  }
  if (s.ph === 'check') rows.push(['wh', 'AFTER'], ['loss′', R.lossAfter(k).toFixed(4)]);
  return rows;
}
