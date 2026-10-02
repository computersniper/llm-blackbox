// 调试器右侧：伪代码（高亮当前行）、这一步的讲解、变量监视。所有数字都来自浏览器里刚算出来的真实前向。
import { esc } from '../../js/ui.js';
import { ACTIONS } from './game.js';
import { ENC, DEC, Z, KMIX } from './nn.js';
import { isEn, L as Lx } from '../../js/i18n.js';

const kw = (s) => `<span class="kw">${s}</span>`;
const fn = (s) => `<span class="fn">${s}</span>`;
const cm = (s) => `<span class="cm">${s}</span>`;
const nu = (s) => `<span class="nu">${s}</span>`;

export const CODE = [
  cm(Lx('# V 看，M 记住并想象（Ha &amp; Schmidhuber 2018）', '# V sees, M remembers (Ha &amp; Schmidhuber 2018)')),
  `h, c = ${fn('zeros')}(${nu(256)}), ${fn('zeros')}(${nu(256)})`,
  `obs = env.${fn('reset')}()          ${cm('# 64×64×3')}`,
  `${kw('while')} True:`,
  Lx(`  ${kw('if')} 睁眼: z = V.${fn('encode')}(obs).μ`, `  ${kw('if')} eyes_open: z = V.${fn('encode')}(obs).μ`),
  Lx(`  ${kw('else')}:   z = ẑ           ${cm('# 闭眼')}`, `  ${kw('else')}:         z = ẑ      ${cm('# eyes closed')}`),
  Lx(`  a = 方向键 ${kw('or')} 自动驾驶 ${kw('or')} ${fn('C')}(z, h)`, `  a = keys ${kw('or')} autopilot ${kw('or')} ${fn('C')}(z, h)`),
  `  π, μ, σ, done, h, c = ${fn('M')}(z, a, h, c)`,
  `  ẑ = ${fn('sample')}(π, μ, σ, τ)   ${cm(Lx('# 下一帧的 z', '# next z'))}`,
  `  ô = V.${fn('decode')}(ẑ)         ${cm(Lx('# 梦见的下一帧', '# dreamed frame'))}`,
  `  obs = env.${fn('step')}(a)       ${cm(Lx('# 真实的下一帧', '# real frame'))}`,
  Lx(`  误差 = ${fn('mean')}((obs − ô)²)`, `  err = ${fn('mean')}((obs − ô)²)`),
  '',
  `${kw('def')} V.${fn('encode')}(obs):         ${cm(Lx('# 卷积 VAE', '# conv VAE'))}`,
  `  x = ${fn('relu')}(${fn('conv')}(obs, ${nu(16)}, ${nu(4)}, ${nu(2)})) ${cm('# 31²×16')}`,
  `  x = ${fn('relu')}(${fn('conv')}(x, ${nu(32)}, ${nu(4)}, ${nu(2)}))   ${cm('# 14²×32')}`,
  `  x = ${fn('relu')}(${fn('conv')}(x, ${nu(64)}, ${nu(4)}, ${nu(2)}))   ${cm('# 6²×64')}`,
  `  x = ${fn('relu')}(${fn('conv')}(x, ${nu(128)}, ${nu(4)}, ${nu(2)}))  ${cm('# 2²×128')}`,
  `  μ, logσ² = W<sub>μ</sub>·x+b, W<sub>σ</sub>·x+b ${cm('# 32, 32')}`,
  `  ${kw('return')} μ             ${cm(Lx('# 训练时 μ+σ·ε', '# train: μ+σ·ε'))}`,
  '',
  `${kw('def')} V.${fn('decode')}(z):`,
  `  x = W·z + b               ${cm('# 1×1×512')}`,
  `  x = ${fn('relu')}(${fn('deconv')}(x, ${nu(64)}, ${nu(5)}, ${nu(2)})) ${cm('# 5²×64')}`,
  `  x = ${fn('relu')}(${fn('deconv')}(x, ${nu(32)}, ${nu(5)}, ${nu(2)})) ${cm('# 13²×32')}`,
  `  x = ${fn('relu')}(${fn('deconv')}(x, ${nu(16)}, ${nu(6)}, ${nu(2)})) ${cm('# 30²×16')}`,
  `  ${kw('return')} σ(${fn('deconv')}(x, ${nu(3)}, ${nu(6)}, ${nu(2)}))  ${cm('# 64²×3')}`,
  '',
  `${kw('def')} M(z, a, h, c):         ${cm('# LSTM + MDN')}`,
  `  x = ${fn('concat')}(z, ${fn('onehot')}(a))  ${cm('# 35')}`,
  `  i = σ(W<sub>i</sub>·[x, h] + b<sub>i</sub>)     ${cm(Lx('# 输入门', '# input gate'))}`,
  `  f = σ(W<sub>f</sub>·[x, h] + b<sub>f</sub>)     ${cm(Lx('# 遗忘门', '# forget gate'))}`,
  `  g = ${fn('tanh')}(W<sub>g</sub>·[x, h] + b<sub>g</sub>)  ${cm(Lx('# 候选记忆', '# candidate'))}`,
  `  o = σ(W<sub>o</sub>·[x, h] + b<sub>o</sub>)     ${cm(Lx('# 输出门', '# output gate'))}`,
  `  c = f*c + i*g`,
  `  h = o*${fn('tanh')}(c)`,
  `  π, μ, logσ = W<sub>head</sub>·h + b ${cm(Lx('# 各 32×5', '# 32×5 each'))}`,
  `  done = σ(w<sub>done</sub>·h + b)   ${cm(Lx('# 撞车了吗', '# crashed?'))}`,
  '',
  `${kw('def')} ${fn('sample')}(π, μ, σ, τ):     ${cm(Lx('# 每一维', '# per dim'))}`,
  `  k ~ ${fn('softmax')}(log π / τ)    ${cm(Lx('# 挑一个高斯', '# pick a Gaussian'))}`,
  `  ${kw('return')} μ[k] + σ[k]·√τ·ε  ${cm('# ε~N(0,1)')}`,
];

export function renderCode(el) {
  el.innerHTML = CODE.map((l, i) => `<span class="ln"><span class="bp"></span><span class="no">${i + 1}</span>${l}</span>`).join('');
}

const OPLINE = { e1: 15, e2: 16, e3: 17, e4: 18, mu: 19, z: 20, dfc: 23, d1: 24, d2: 25, d3: 26, d4: 27, cat: 30, gi: 31, gf: 32, gg: 33, go: 34, cell: 35, hid: 36, mdn: 37, done: 38, pick: 41, draw: 42 };
export function linesFor(s, F) {
  switch (s.ph) {
    case 'frame': return [5, 6, 7, 8, 9, 10, 11, 12];
    case 'reset': return [2, 3];
    case 'obs': return [F && F.t === 0 ? 3 : 11];
    case 'enc': return s.op ? [OPLINE[s.op]] : [F?.rec?.src === 'dream' ? 6 : 5];
    case 'rnn': return s.op ? [OPLINE[s.op]] : [7, 8];
    case 'sample': return s.op ? [OPLINE[s.op]] : [9];
    case 'dec': return s.op ? [OPLINE[s.op]] : [10];
    case 'cmp': return [11, 12];
  }
  return [4];
}

export const STAGE_LABEL = Lx({ frame: '一帧', reset: '新的一局', obs: '看：真实画面', enc: '编码', rnn: '记忆与预测', sample: '采样', dec: '解码', cmp: '对照' },
  { frame: 'A frame', reset: 'New game', obs: 'Look: real frame', enc: 'Encode', rnn: 'Remember & predict', sample: 'Sample', dec: 'Decode', cmp: 'Compare' });
export const OP_LABEL = Lx({
  e1: '卷积 1', e2: '卷积 2', e3: '卷积 3', e4: '卷积 4', mu: 'μ 和 σ', z: 'z（32 个数）',
  dfc: '全连接', d1: '反卷积 1', d2: '反卷积 2', d3: '反卷积 3', d4: '反卷积 4',
  cat: '拼接输入', gi: '输入门', gf: '遗忘门', gg: '候选记忆', go: '输出门', cell: '细胞状态', hid: '隐状态', mdn: '混合密度', done: '撞车概率',
  pick: '挑一个高斯', draw: '加上噪声',
}, {
  e1: 'Conv 1', e2: 'Conv 2', e3: 'Conv 3', e4: 'Conv 4', mu: 'μ and σ', z: 'z (32 numbers)',
  dfc: 'Fully connected', d1: 'Deconv 1', d2: 'Deconv 2', d3: 'Deconv 3', d4: 'Deconv 4',
  cat: 'Concat input', gi: 'Input gate', gf: 'Forget gate', gg: 'Candidate memory', go: 'Output gate', cell: 'Cell state', hid: 'Hidden state', mdn: 'Mixture density', done: 'Crash probability',
  pick: 'Pick a Gaussian', draw: 'Add noise',
});
export const MI_LABEL = Lx({ pick: '选一个输出', mul: '逐项相乘', sum: '加起来', act: '激活' }, { pick: 'Pick an output', mul: 'Multiply term by term', sum: 'Add up', act: 'Activate' });

export function stepLabel(s, F) {
  if (s.ph === 'frame') return F ? Lx(`第 ${F.t} 帧`, `Frame ${F.t}`) : Lx('一帧', 'A frame');
  let l = STAGE_LABEL[s.ph];
  if (s.op) l = OP_LABEL[s.op];
  if (s.mi) l += ` · ${MI_LABEL[s.mi]}`;
  return l;
}

const DNAME = Lx(['', '玩', '循环', 'V 的内部', 'M 的内部', '一次乘加'], ['', 'Play', 'Loop', 'Inside V', 'Inside M', 'One multiply-add']);
export function crumbs(tl) {
  return tl.path().map((p) => ({ d: p.d, label: DNAME[p.d] }));
}

/* ---------------------------------------------------------------- 小工具 */

// 这一步是谁开的：（你按的）/（自动驾驶）/（C）
export const byTag = (by) => (by === 'key' ? Lx('（你按的）', ' (you)') : by === 'heur' ? Lx('（自动驾驶）', ' (autopilot)') : by === 'ctrl' ? Lx('（C）', ' (C)') : '');
export const fmtMSE = (v) => (v == null ? '—' : v < 0.0001 ? v.toExponential(1) : v.toFixed(4));
const f3 = (v) => (Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 10 ? v.toFixed(2) : v.toFixed(3));
const pct = (p) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;
// 动作名：game.js 要能在 Node 里直接 import（不碰 DOM），所以英文名放在这里
const ACTIONS_EN = ['straight', 'left', 'right'];
export const actName = (a) => (isEn ? ACTIONS_EN : ACTIONS)[a] || '—';

// 潜变量各维按 KL 从大到小排（KL≈0 的是没用上的维度）
let ORDER = null;
export function dimOrder(meta) {
  if (!ORDER) ORDER = meta.dims.map((d, i) => i).sort((a, b) => meta.dims[b].kl - meta.dims[a].kl);
  return ORDER;
}
// 一维“有没有用上”：KL（和标准正态的距离）大于 0.05
export const used = (meta, d) => meta.dims[d].kl > 0.05;
// 一维的含义：导出时把这一维从数据里的 1% 分位扫到 99% 分位、其余不动，解码出来量的（小车、三段路、抛锚车）
const SHORT = Lx({ car_x: '车位', road_near: '近路', road_mid: '中路', road_far: '远路', ob_x: '障碍↔', ob_y: '障碍↕', ob_amt: '有车' },
  { car_x: 'car', road_near: 'near', road_mid: 'mid', road_far: 'far', ob_x: 'obs↔', ob_y: 'obs↕', ob_amt: 'obs?' });
// 导出时写进 model.json 的中文含义（sweep.label）按量的是什么（sweep.feat）换成英文
const FEAT_EN = { car_x: 'car left/right', road_near: 'near road left/right', road_mid: 'mid road left/right', road_far: 'far road left/right', ob_x: 'stalled car left/right', ob_y: 'stalled car up/down', ob_amt: 'stalled car on/off' };
export function dimShort(meta, d) {
  const sw = meta.dims[d].sweep;
  return used(meta, d) && sw && (sw.feat === 'ob_amt' ? sw.px >= 0.5 : sw.px >= 2) ? SHORT[sw.feat] || '' : '';
}
export function dimMeaning(meta, d) {
  const sw = meta.dims[d].sweep;
  if (!used(meta, d) || !sw) return Lx('没用上', 'unused');
  if (sw.feat === 'ob_amt') return sw.px >= 0.5 ? Lx(`抛锚车出现 / 消失（约 ${sw.px.toFixed(1)} 辆）`, `stalled car on/off, ~${sw.px.toFixed(1)} cars`) : Lx('几乎看不出变化', 'barely any visible change');
  return sw.px >= 2 ? Lx(`${sw.label}，移动约 ${Math.round(sw.px)} 个像素`, `${FEAT_EN[sw.feat] || sw.feat}, moves ~${Math.round(sw.px)} px`) : Lx('变化很小', 'very little change');
}

function stats(a) {
  let mx = -Infinity, mn = Infinity, nz = 0, s = 0;
  for (let i = 0; i < a.length; i++) { const v = a[i]; if (v > mx) mx = v; if (v < mn) mn = v; if (v > 0) nz++; s += v; }
  return { mx, mn, nz, mean: s / a.length, n: a.length };
}
const norm = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0));

function shapeHTML(parts) { return parts.map((p) => (Array.isArray(p) ? `<b>${p[0]}</b>` : p)).join(''); }

export function shapeOf(s) {
  if (s.ph === 'frame') return shapeHTML(['obs ', ['[64×64×3]'], ' → V → z ', ['[32]'], ' → M → ẑ ', ['[32]'], ' → V → ô ', ['[64×64×3]']]);
  const L = [...ENC, ...DEC].find((l) => l.id === s.op);
  if (L) {
    const kind = L.id[0] === 'e' ? 'conv' : 'deconv';
    if (isEn) return shapeHTML([`[${L.inH}×${L.inH}×${L.cin}] → ${kind}, ${L.cout} kernels of ${L.k}×${L.k}, stride ${L.s} → `, [`[${L.outH}×${L.outH}×${L.cout}]`], ` · weights ${L.cin}×${L.cout}×${L.k}×${L.k} = ${(L.cin * L.cout * L.k * L.k).toLocaleString('en-US')}`]);
    return shapeHTML([`[${L.inH}×${L.inH}×${L.cin}] → ${kind} ${L.cout} 个 ${L.k}×${L.k} 核，步长 ${L.s} → `, [`[${L.outH}×${L.outH}×${L.cout}]`], ` · 权重 ${L.cin}×${L.cout}×${L.k}×${L.k} = ${(L.cin * L.cout * L.k * L.k).toLocaleString()}`]);
  }
  switch (s.op) {
    case 'mu': return shapeHTML(['x ', ['[512]'], ' @ W<sub>μ</sub> [512×32] → μ ', ['[32]'], Lx('；W<sub>σ</sub> 同形 → logσ² ', '; W<sub>σ</sub>, same shape → logσ² '), ['[32]']]);
    case 'z': return shapeHTML(['z = μ ', ['[32]'], Lx('（网页里不加噪声；训练时 z = μ + σ·ε）', ' (no noise on this page; in training z = μ + σ·ε)')]);
    case 'dfc': return shapeHTML(['z ', ['[32]'], ' @ W [32×512] → ', ['[1×1×512]']]);
    case 'cat': return shapeHTML(['concat(z ', ['[32]'], ', onehot(a) ', ['[3]'], ') → x ', ['[35]']]);
    case 'gi': case 'gf': case 'gg': case 'go': return shapeHTML(['W<sub>ih</sub> [256×35]·x + W<sub>hh</sub> [256×256]·h + b → ', ['[256]']]);
    case 'cell': return shapeHTML(['c\' = f ⊙ c + i ⊙ g → ', ['[256]']]);
    case 'hid': return shapeHTML(['h\' = o ⊙ tanh(c\') → ', ['[256]']]);
    case 'mdn': return shapeHTML(['h ', ['[256]'], Lx(' @ W<sub>head</sub> [256×481] → log π, μ, log σ 各 ', ' @ W<sub>head</sub> [256×481] → log π, μ, log σ, each '), ['[32×5]'], ' + done ', ['[1]']]);
    case 'done': return shapeHTML(['w<sub>done</sub> [256] · h + b → σ → ', [Lx('p(撞车)', 'p(crash)')]]);
    case 'pick': return shapeHTML(['softmax(log π / τ) ', ['[32×5]'], Lx(' → 每一维挑一个 k', ' → pick one k per dim')]);
    case 'draw': return shapeHTML(['ẑ = μ[k] + σ[k]·√τ·ε → ', ['[32]']]);
  }
  switch (s.ph) {
    case 'enc': return shapeHTML(['obs ', ['[64×64×3]'], Lx(' → 4 层卷积 → μ ', ' → 4 conv layers → μ '), ['[32]']]);
    case 'rnn': return shapeHTML(['[z ', ['[32]'], ', a ', ['[3]'], '] + h ', ['[256]'], ' → LSTM → h\' ', ['[256]'], ' → MDN ', ['[32×5×3 + 1]']]);
    case 'sample': return shapeHTML(['π, μ, σ ', ['[32×5]'], ' → ẑ ', ['[32]']]);
    case 'dec': return shapeHTML(['ẑ ', ['[32]'], Lx(' → 4 层反卷积 → ô ', ' → 4 deconv layers → ô '), ['[64×64×3]']]);
  }
  return '';
}

/* ---------------------------------------------------------------- 讲解 */

export function explain(s, ctx) {
  const { sim, meta, F, rec, det, mac } = ctx;
  const mode = rec?.mode || sim.mode;
  if (s.ph === 'frame') {
    const r = F.rec;
    const tr = meta.train;
    const dreamTxt = sim.mode === 'closed'
      ? Lx(`<b>闭着眼</b>：梦不再看真实画面，用它自己上一步想出来的 z 往下想${F.dreamAge ? `（已经想了 ${F.dreamAge} 步）` : ''}。路是它自己“编”的，所以会和真实的越来越不一样；但小车跟着你的方向键走，这是它从数据里学到的因果。`, `<b>Eyes closed</b>: the dream no longer looks at the real frame; it carries on from the z it imagined itself last step${F.dreamAge ? ` (${F.dreamAge} steps so far)` : ''}. The road is made up, so it drifts further and further from the real one — yet the car still follows your steering. That cause and effect is something the model learned from data.`)
      : Lx('<b>睁着眼</b>：每一步都先把真实画面压成 32 个数，再让 M 预测下一帧。所以梦总是只差一步，误差很小。按 <kbd>O</kbd> 闭眼，看梦自己往下想。', '<b>Eyes open</b>: every step, the real frame is first squeezed into 32 numbers, then M predicts the next frame. So the dream is never more than one step behind and the error stays small. Press <kbd>O</kbd> to close its eyes and watch the dream carry on by itself.');
    return Lx(`<p>左边是游戏本身，右边是<b>模型的梦</b>：一个只有 ${(meta.params.V + meta.params.M).toLocaleString()} 个参数的神经网络，在本地显卡上看了 <b>${tr.collect.frames.toLocaleString()} 帧</b>开车画面学会的“这个世界接下来会怎样”。你的每一个动作同时发给两边。</p><p>${dreamTxt}</p>${r?.by === 'ctrl' ? `<p>现在是 <b>C 在开车</b>：它只看 z 和 h（梦里的东西），三个动作的打分 ${[...r.logits].map((v) => v.toFixed(2)).join(' / ')}。</p>` : ''}<p class="dimmed">点 ＋ 把这一帧拆开，看 V 和 M 各做了什么。</p>`, `<p>On the left is the game itself; on the right is <b>the model’s dream</b>: a neural network with only ${(meta.params.V + meta.params.M).toLocaleString('en-US')} parameters that learned “what this world does next” by watching <b>${tr.collect.frames.toLocaleString('en-US')} frames</b> of driving on a local GPU. Every action you take is sent to both sides.</p><p>${dreamTxt}</p>${r?.by === 'ctrl' ? `<p><b>C is driving</b> right now: it only sees z and h (things inside the dream). Scores for the three actions: ${[...r.logits].map((v) => v.toFixed(2)).join(' / ')}.</p>` : ''}<p class="dimmed">Press ＋ to take this frame apart and see what V and M each do.</p>`);
  }
  if (s.ph === 'reset') return Lx(`<p>真实世界里${F.why === 'offroad' ? '冲出了路面' : '撞上了抛锚的车'}，这一局结束。换一条新路（新的随机种子），梦也重新从真实画面的编码开始，M 的记忆 h、c 清零。</p>`, `<p>In the real world the car ${F.why === 'offroad' ? 'ran off the road' : 'hit a stalled car'}, so this game is over. A new road starts (new random seed), the dream restarts from an encoding of the real frame, and M’s memory h, c is reset to zero.</p>`);
  if (s.mi) return mac ? mac.explain(s.mi) : '';
  switch (s.ph) {
    case 'obs': return Lx(`<p>真实世界给出这一帧的画面 obs：64×64 个像素，每个像素 3 个颜色通道，一共 ${(64 * 64 * 3).toLocaleString()} 个数。</p><p>游戏是确定的：同一个种子、同一串动作，Python 采集训练数据时的游戏和这里的 game.js 逐像素相同。</p>${mode === 'closed' ? '<p class="dimmed">现在闭着眼：这一帧梦<b>不会</b>去看。</p>' : ''}`, `<p>The real world hands over this frame, obs: 64×64 pixels with 3 color channels each, ${(64 * 64 * 3).toLocaleString('en-US')} numbers in total.</p><p>The game is deterministic: same seed, same actions, and the Python game used to collect training data matches this game.js pixel for pixel.</p>${mode === 'closed' ? '<p class="dimmed">Eyes are closed right now: the dream will <b>not</b> look at this frame.</p>' : ''}`);
    case 'enc':
      if (s.op) return explainEnc(s.op, ctx);
      if (rec.src === 'dream') return Lx(`<p><b>闭眼</b>：跳过编码器，直接拿梦上一步采样出来的 ẑ 当作这一帧的 z。梦从这里开始脱离现实。</p>`, `<p><b>Eyes closed</b>: skip the encoder and use the ẑ the dream sampled last step as this frame’s z. This is where the dream comes loose from reality.</p>`);
      return Lx(`<p><b>V（视觉）</b>把画面压缩成 <b>32 个数</b> z。它是一个卷积变分自编码器：4 层卷积一层层缩小画面，最后两个全连接层给出每一维的均值 μ 和方差 σ²。</p><p>${(64 * 64 * 3).toLocaleString()} 个数 → 32 个数，压缩了 ${Math.round(12288 / 32)} 倍。</p>${rec.src === 'resync' ? '<p>上一步梦里“撞车”了，这一步重新睁眼对齐。</p>' : ''}`, `<p><b>V (vision)</b> compresses the frame into <b>32 numbers</b>, z. It is a convolutional variational autoencoder: 4 conv layers shrink the image step by step, then two fully connected layers give each dimension a mean μ and a variance σ².</p><p>${(64 * 64 * 3).toLocaleString('en-US')} numbers → 32 numbers, a ${Math.round(12288 / 32)}× compression.</p>${rec.src === 'resync' ? '<p>The dream “crashed” last step, so this step it opens its eyes again to realign.</p>' : ''}`);
    case 'rnn':
      if (s.op) return explainM(s.op, ctx);
      return Lx(`<p><b>M（记忆）</b>是一个 LSTM，带着 256 维的记忆 h、c。输入是这一帧的 z 和你的动作「${actName(rec.a)}」，输出下一帧 z 的<b>概率分布</b>：每一维是 5 个高斯的混合（混合密度网络，MDN）。</p><p>它还顺便预测“这一步会不会撞车”：<b>${pct(rec.M.done)}</b>。</p><p class="dimmed">为什么输出分布而不是一个数？路前面会刷出什么车是随机的，一个数只能给出“平均情况”，糊成一团。</p>`, `<p><b>M (memory)</b> is an LSTM carrying a 256-dimensional memory h, c. Its input is this frame’s z plus your action “${actName(rec.a)}”; its output is a <b>probability distribution</b> over the next frame’s z: each dimension is a mixture of 5 Gaussians (a mixture density network, MDN).</p><p>It also predicts “will this step end in a crash”: <b>${pct(rec.M.done)}</b>.</p><p class="dimmed">Why a distribution instead of a single number? Which cars appear up the road is random; a single number can only give the “average case”, a blurry smear.</p>`);
    case 'sample':
      if (s.op) return explainS(s.op, ctx);
      return Lx(`<p>从 M 给出的分布里<b>抽一个</b>下一帧的 z：温度 τ = <b>${rec.tau.toFixed(2)}</b>。τ 越大，越敢抽不太可能的结果，梦越“放飞”；τ → 0 就只取最可能的那个高斯的中心。</p><p>论文里，在 τ 偏低的梦里训练的控制器会钻梦的空子，到真实世界就不灵了；调高 τ 让梦更难，反而学得更稳。</p>`, `<p><b>Draw</b> the next frame’s z from M’s distribution, at temperature τ = <b>${rec.tau.toFixed(2)}</b>. The larger τ, the more willing it is to pick unlikely outcomes and the wilder the dream; as τ → 0 it just takes the center of the most likely Gaussian.</p><p>In the paper, a controller trained in a low-τ dream learns to exploit the dream’s loopholes and then fails in the real world; raising τ makes the dream harder, and what it learns is sturdier.</p>`);
    case 'dec':
      if (s.op) return explainDec(s.op, ctx);
      return Lx(`<p><b>V 的解码器</b>把 32 个数重新画成 64×64 的画面：先用全连接层放大成 512 个数，再经过 4 层反卷积（转置卷积）一层层放大，最后 sigmoid 把每个像素压到 0–1。</p><p>这就是<b>梦见的下一帧</b> ô。画面里所有东西——路的弯、车的位置、抛锚车——都是从这 32 个数里画出来的。</p>`, `<p><b>V’s decoder</b> paints the 32 numbers back into a 64×64 image: a fully connected layer expands them to 512 numbers, 4 deconvolution (transposed convolution) layers upsample step by step, and a sigmoid squashes each pixel into 0–1.</p><p>That is the <b>dreamed next frame</b>, ô. Everything in it — the bends of the road, the car’s position, the stalled cars — is drawn from those 32 numbers.</p>`);
    case 'cmp': {
      const G = ctx.next;
      return Lx(`<p>真实世界也走了一步（动作「${actName(rec.a)}」）。把梦见的 ô 和真实的下一帧逐像素比较：每像素均方误差 <b>${fmtMSE(G?.mse)}</b>。</p><p>${mode === 'closed' ? '闭眼时误差会一步步累积：一开始只是虚线的相位、远处的弯不对，后来连路往哪拐、哪里有车都对不上了。' : '睁眼时每一步都从真实画面重新出发，误差主要来自两处：V 画得不够细（虚线、草丛被抹平），和前方新刷出来的东西本来就猜不到。'}</p><p class="dimmed">测试集平均：只看 V 的重建 ${fmtMSE(meta.drift.recon)}；睁眼一步预测 ${fmtMSE(meta.drift.open[0])}；闭眼 30 步 ${fmtMSE(meta.drift['closed_1.0'][29])}。</p>`, `<p>The real world has also taken a step (action “${actName(rec.a)}”). Compare the dreamed ô with the real next frame pixel by pixel: mean squared error per pixel <b>${fmtMSE(G?.mse)}</b>.</p><p>${mode === 'closed' ? 'With eyes closed the error piles up step after step: at first only the phase of the dashed line and the far bend are off; later even which way the road turns and where the cars are no longer match.' : 'With eyes open every step restarts from the real frame, so the error comes from two places: V doesn’t draw fine detail (the dashes and grass get smoothed away), and whatever newly appears up ahead can’t be guessed anyway.'}</p><p class="dimmed">Test-set averages: V’s reconstruction alone ${fmtMSE(meta.drift.recon)}; one-step prediction with eyes open ${fmtMSE(meta.drift.open[0])}; 30 steps with eyes closed ${fmtMSE(meta.drift['closed_1.0'][29])}.</p>`);
    }
  }
  return '';
}

function actStats(name, a) {
  const st = stats(a);
  return Lx(`<p class="dimmed">${name}：${st.n.toLocaleString()} 个数，最大 ${f3(st.mx)}，${pct(st.nz / st.n)} 大于 0（ReLU 把负数都变成了 0）。</p>`, `<p class="dimmed">${name}: ${st.n.toLocaleString('en-US')} numbers, max ${f3(st.mx)}, ${pct(st.nz / st.n)} above 0 (ReLU turns every negative into 0).</p>`);
}

function explainEnc(op, { det, meta, sim, F }) {
  const L = ENC.find((l) => l.id === op);
  if (L) {
    const a = det.enc[op];
    const lead = Lx({
      e1: '第一层卷积：16 个 4×4 的小卷积核在画面上每隔 2 个像素滑一次。每个核只管一件小事——有的对“亮的竖线”（路沿）敏感，有的对“青色”（小车）、“红色”（抛锚车）敏感。',
      e2: '第二层：在上一层 16 张特征图的基础上，再用 32 个 4×4×16 的核组合出更大的图案。每个输出像素“看”到的原图范围从 4×4 扩大到 10×10。',
      e3: '第三层：64 张 6×6 的特征图。每个位置已经能“看到”原图 22×22 的一块，开始对应“这里有一段弯路”“这里有辆车”这种整体的东西。',
      e4: '第四层：128 张 2×2 的图，一共 512 个数。到这里空间位置几乎被压没了，信息都转到了“哪个通道亮”上。',
    }, {
      e1: 'First conv layer: 16 small 4×4 kernels slide over the image, 2 pixels at a time. Each kernel watches for one small thing — some respond to “bright vertical lines” (road edges), others to “cyan” (your car) or “red” (stalled cars).',
      e2: 'Second layer: 32 kernels of 4×4×16 combine the previous 16 feature maps into larger patterns. Each output pixel now “sees” a 10×10 patch of the original image instead of 4×4.',
      e3: 'Third layer: 64 feature maps of 6×6. Each position already “sees” a 22×22 patch of the original, and starts to mean whole things like “a bend in the road here” or “a car here”.',
      e4: 'Fourth layer: 128 maps of 2×2, 512 numbers in all. By now spatial position is almost squeezed out; the information lives in “which channels light up”.',
    })[op];
    return Lx(`<p>${lead}</p>${actStats(`这一层输出 ${L.outH}×${L.outH}×${L.cout}`, a)}<p class="dimmed">点舞台上的任意一格，再按 ＋ 看它是怎么乘加出来的。</p>`, `<p>${lead}</p>${actStats(`This layer’s output, ${L.outH}×${L.outH}×${L.cout}`, a)}<p class="dimmed">Click any cell on the stage, then press ＋ to see how it is multiplied and added up.</p>`);
  }
  if (op === 'mu') {
    const mu = det.enc.mu, lv = det.enc.lv;
    const nUsed = meta.dims.filter((d, i) => used(meta, i)).length;
    return Lx(`<p>两个全连接层把 512 个数各自变成 32 个：<b>μ</b>（这一维“最可能”是多少）和 <b>logσ²</b>（有多不确定）。</p><p>训练时还要让它们接近标准正态分布（KL 项）：没什么用的维度会被压成 μ≈0、σ≈1，等于没用上。这个模型 32 维里用上了 <b>${nUsed}</b> 维。</p><p class="dimmed">这一帧 |μ| = ${f3(norm(mu))}，平均 σ = ${f3(lv.reduce((s, v) => s + Math.exp(v / 2), 0) / Z)}</p>`, `<p>Two fully connected layers each turn the 512 numbers into 32: <b>μ</b> (the “most likely” value of each dimension) and <b>logσ²</b> (how uncertain it is).</p><p>Training also pulls them toward a standard normal distribution (the KL term): dimensions that don’t help get pressed to μ≈0, σ≈1 — effectively unused. This model uses <b>${nUsed}</b> of its 32 dimensions.</p><p class="dimmed">This frame: |μ| = ${f3(norm(mu))}, mean σ = ${f3(lv.reduce((s, v) => s + Math.exp(v / 2), 0) / Z)}</p>`);
  }
  if (op === 'z') {
    const order = dimOrder(meta).slice(0, 4);
    const pr = meta.probes;
    return Lx(`<p>这 32 个数就是 V 眼里的整个世界。舞台下方把用得最多的几维各自<b>扫了一遍</b>：只改这一维、其他不动，解码出来的画面怎么变——有的维管小车左右，有的管路往哪弯，有的管抛锚车在哪。</p><p>${order.map((d) => `z<sub>${d}</sub>：${esc(dimMeaning(meta, d))}`).join('<br>')}</p><p class="dimmed">“含义”是导出时量的：以 32 帧的 μ 为底，把这一维从数据里的 1% 分位扫到 99% 分位，在解码出来的画面上量小车、三段路的中心、抛锚车跟着动了多少，取动得最多的那样。一维往往不只管一件事（同时动了好几样），这里只写最明显的。</p><p class="dimmed">反过来，从 32 个数用一个线性层读游戏状态（测试集 R²）：小车位置 ${pr.car_x}，车旁的路 ${pr.road_near}，远处的路 ${pr.road_far}，抛锚车左右 ${pr.ob_x}。</p>`, `<p>These 32 numbers are the entire world as V sees it. Below the stage, the most-used dimensions are each <b>swept</b>: change just that one, keep the rest fixed, and watch how the decoded image changes — one dimension moves the car left and right, another bends the road, another moves the stalled cars.</p><p>${order.map((d) => `z<sub>${d}</sub>: ${esc(dimMeaning(meta, d))}`).join('<br>')}</p><p class="dimmed">The “meanings” were measured at export time: starting from the μ of 32 frames, sweep the dimension from the 1st to the 99th percentile of the data and measure, in the decoded images, how far the car, the centers of three stretches of road and the stalled cars move; the one that moves most wins. A dimension often controls more than one thing at once; only the most obvious is listed.</p><p class="dimmed">The other way round, reading the game state out of the 32 numbers with one linear layer (test-set R²): car position ${pr.car_x}, road next to the car ${pr.road_near}, distant road ${pr.road_far}, stalled car left/right ${pr.ob_x}.</p>`);
  }
  return '';
}

function explainDec(op, { det }) {
  const d = det.dec;
  if (!d) return '';
  if (op === 'dfc') return Lx(`<p>解码先用一个全连接层把 32 个数放大成 512 个，排成 1×1×512 的“一个像素、512 个通道”。</p>${actStats('这 512 个数', d.dfc)}`, `<p>Decoding starts with a fully connected layer that expands the 32 numbers into 512, arranged as 1×1×512: “one pixel, 512 channels”.</p>${actStats('These 512 numbers', d.dfc)}`);
  const lead = Lx({
    d1: '反卷积（转置卷积）和卷积反过来：每个输入像素拿着自己的数，把一个 5×5 的核“盖章”到输出上。1×1 → 5×5，开始有了空间结构。',
    d2: '5×5 → 13×13：步长 2，相邻两个“章”有重叠，重叠的地方加起来。这一层大致定下路在哪、车在哪。',
    d3: '13×13 → 30×30：16 个通道各管一类东西，有的亮在路面，有的亮在车上、路沿上。',
    d4: '最后一层：30×30×16 → 64×64×3，正好是红、绿、蓝三个通道，再经过 sigmoid 压到 0–1，就是梦见的画面。',
  }, {
    d1: 'A deconvolution (transposed convolution) is a convolution run backwards: each input pixel takes its value and “stamps” a 5×5 kernel onto the output. 1×1 → 5×5 — spatial structure begins.',
    d2: '5×5 → 13×13: with stride 2, neighboring “stamps” overlap, and the overlaps add up. This layer roughly settles where the road and the cars are.',
    d3: '13×13 → 30×30: each of the 16 channels handles one kind of thing — some light up on the road surface, others on cars or road edges.',
    d4: 'Last layer: 30×30×16 → 64×64×3 — exactly the red, green and blue channels. A sigmoid squashes them into 0–1, and that is the dreamed image.',
  })[op];
  const a = op === 'd4' ? d.y : d[op];
  return Lx(`<p>${lead}</p>${op === 'd4' ? '' : actStats('这一层输出', a)}<p class="dimmed">点舞台上的任意一格（或者梦里的任意一个像素），再按 ＋ 看它是怎么乘加出来的。</p>`, `<p>${lead}</p>${op === 'd4' ? '' : actStats('This layer’s output', a)}<p class="dimmed">Click any cell on the stage (or any pixel of the dream), then press ＋ to see how it is multiplied and added up.</p>`);
}

function topIdx(a, k = 3, off = 0, n = a.length) {
  const idx = Array.from({ length: n }, (_, i) => i + off);
  idx.sort((x, y) => Math.abs(a[y]) - Math.abs(a[x]));
  return idx.slice(0, k);
}

function explainM(op, { rec, F, meta }) {
  const L = rec.L, H = 256;
  switch (op) {
    case 'cat': return Lx(`<p>把这一帧的 z（32 个数）和动作「${actName(rec.a)}」的 one-hot（[${[0, 1, 2].map((i) => (i === rec.a ? 1 : 0)).join(', ')}]）拼成 35 个数，作为 LSTM 这一步的输入。动作就是这样“告诉”模型的：同一个 z，按不同的键，M 会预测出不同的下一帧。</p>`, `<p>Concatenate this frame’s z (32 numbers) with the one-hot of the action “${actName(rec.a)}” ([${[0, 1, 2].map((i) => (i === rec.a ? 1 : 0)).join(', ')}]) into 35 numbers: the LSTM’s input for this step. That is how the action is “told” to the model: same z, different key, and M predicts a different next frame.</p>`);
    case 'gi': return Lx(`<p><b>输入门 i</b>：这一步的新信息，每一维放进去多少（0 到 1）。平均 ${f3(L.i.reduce((s, v) => s + v, 0) / H)}。</p><p class="dimmed">四个门都是同一个形状的计算：W·[x, h] + b，291 个输入，只是权重不同、激活函数不同。</p>`, `<p><b>Input gate i</b>: how much of this step’s new information to let into each dimension (0 to 1). Mean ${f3(L.i.reduce((s, v) => s + v, 0) / H)}.</p><p class="dimmed">All four gates are the same shape of computation, W·[x, h] + b over 291 inputs; only the weights and activation functions differ.</p>`);
    case 'gf': return Lx(`<p><b>遗忘门 f</b>：旧的记忆 c 每一维留下多少。接近 1 = 记住，接近 0 = 忘掉。这一步平均 ${f3(L.f.reduce((s, v) => s + v, 0) / H)}，${pct(L.f.filter((v) => v > 0.9).length / H)} 的维度几乎完全保留。</p>`, `<p><b>Forget gate f</b>: how much of the old memory c to keep in each dimension. Near 1 = remember, near 0 = forget. Mean this step ${f3(L.f.reduce((s, v) => s + v, 0) / H)}; ${pct(L.f.filter((v) => v > 0.9).length / H)} of dimensions are kept almost entirely.</p>`);
    case 'gg': return Lx(`<p><b>候选记忆 g</b>：tanh 激活，−1 到 1，是这一步“想写进记忆”的内容。</p>`, `<p><b>Candidate memory g</b>: tanh activation, −1 to 1 — what this step “wants to write” into memory.</p>`);
    case 'go': return Lx(`<p><b>输出门 o</b>：记忆 c 里有多少要露出来，变成这一步的隐状态 h。</p>`, `<p><b>Output gate o</b>: how much of the memory c shows through to become this step’s hidden state h.</p>`);
    case 'cell': return Lx(`<p><b>细胞状态</b> c' = f ⊙ c + i ⊙ g：先按遗忘门留下一部分旧记忆，再按输入门加上新内容。c 是 LSTM 能把信息带过很多步的关键——比如“刚才那辆抛锚车已经开过去了”。</p><p class="dimmed">|c| 从 ${f3(norm(F.c))} 变成 ${f3(norm(L.c))}</p>`, `<p><b>Cell state</b> c' = f ⊙ c + i ⊙ g: keep part of the old memory according to the forget gate, then add new content according to the input gate. c is what lets an LSTM carry information across many steps — like “that stalled car has already gone by”.</p><p class="dimmed">|c| goes from ${f3(norm(F.c))} to ${f3(norm(L.c))}</p>`);
    case 'hid': return Lx(`<p><b>隐状态</b> h' = o ⊙ tanh(c')：256 个数，是 M 对“现在世界处在什么状态、接下来会怎样”的全部理解。下一帧的分布、撞车概率都从它算出来；控制器 C 也看它。</p>`, `<p><b>Hidden state</b> h' = o ⊙ tanh(c'): 256 numbers holding everything M understands about “what state the world is in and what comes next”. The next-frame distribution and the crash probability are both computed from it; controller C looks at it too.</p>`);
    case 'mdn': {
      const sp = spreadOf(rec);
      let d = 0;
      for (let k = 0; k < Z; k++) if (dimShort(meta, k) && (!dimShort(meta, d) || sp[k] > sp[d])) d = k;
      return Lx(`<p><b>混合密度网络</b>：一个全连接层把 h 变成 481 个数——32 维 × 5 个高斯 ×（权重 π、中心 μ、宽度 σ）+ 1 个撞车 logit。</p><p>每一维单独一个“5 个山峰”的分布。大多数维度只有一个峰很高（模型很确定）；不确定的维度会有好几个峰，比如 z<sub>${d}</sub>——前面会不会刷出车、刷在哪，本来就说不准。</p>`, `<p><b>Mixture density network</b>: one fully connected layer turns h into 481 numbers — 32 dims × 5 Gaussians × (weight π, center μ, width σ) + 1 crash logit.</p><p>Each dimension gets its own “five-peak” distribution. Most dimensions have one tall peak (the model is sure); uncertain ones have several, like z<sub>${d}</sub> — whether a car will appear ahead, and where, simply can’t be known in advance.</p>`);
    }
    case 'done': return Lx(`<p>撞车概率：sigmoid(w·h + b) = <b>${pct(rec.M.done)}</b>。训练时只有每局最后一步是 1，所以模型学会了在“快撞上”的时候把它调高。闭眼时它超过 50%，这一段梦就到此为止，下一步重新睁眼。</p>`, `<p>Crash probability: sigmoid(w·h + b) = <b>${pct(rec.M.done)}</b>. In training only the last step of each game is 1, so the model learned to raise it when a crash is “about to happen”. With eyes closed, once it passes 50% this stretch of dream ends and the next step opens its eyes again.</p>`);
  }
  return '';
}

// 每一维的“不确定度”：各分量中心的加权离散程度 + 宽度（挑一个最有看头的维度给讲解用）
export function spreadOf(rec) {
  const M = rec.M, out = new Float32Array(Z);
  for (let d = 0; d < Z; d++) {
    let m = 0, v = 0;
    for (let k = 0; k < KMIX; k++) { const p = Math.exp(M.logpi[d * KMIX + k]); m += p * M.mu[d * KMIX + k]; }
    for (let k = 0; k < KMIX; k++) { const p = Math.exp(M.logpi[d * KMIX + k]); const s = Math.exp(M.logsig[d * KMIX + k]); v += p * ((M.mu[d * KMIX + k] - m) ** 2 + s * s); }
    out[d] = Math.sqrt(v);
  }
  return out;
}

function explainS(op, { rec }) {
  const S = rec.S;
  if (op === 'pick') {
    const counts = [0, 0, 0, 0, 0];
    for (let d = 0; d < Z; d++) counts[S.k[d]]++;
    return Lx(`<p>每一维先按 softmax(log π / τ) 挑一个高斯分量（τ = ${rec.tau.toFixed(2)}）。τ < 1 让大的 π 更大、小的更小，梦更“保守”；τ > 1 反过来。</p><p class="dimmed">这一步 32 维挑中的分量：${counts.map((c, k) => `第 ${k} 个 ${c} 维`).join('，')}</p>`, `<p>For each dimension, first pick one Gaussian component by softmax(log π / τ) (τ = ${rec.tau.toFixed(2)}). τ < 1 makes large π larger and small ones smaller, so the dream is more “conservative”; τ > 1 does the opposite.</p><p class="dimmed">Components picked across the 32 dims this step: ${counts.map((c, k) => `#${k}: ${c} dims`).join(', ')}</p>`);
  }
  return Lx(`<p>在挑中的高斯里抽一个数：ẑ = μ[k] + σ[k]·√τ·ε，ε 来自标准正态分布。这 32 个数就是梦里的<b>下一帧</b>。</p><p class="dimmed">网页里的随机数是固定种子的，所以调试器退回再前进，梦不会变。</p>`, `<p>Draw a number from the chosen Gaussian: ẑ = μ[k] + σ[k]·√τ·ε, with ε from a standard normal distribution. These 32 numbers are the dream’s <b>next frame</b>.</p><p class="dimmed">The random numbers on this page use a fixed seed, so stepping the debugger back and forward again doesn’t change the dream.</p>`);
}

/* ---------------------------------------------------------------- 变量监视 */

export function watch(s, ctx) {
  const { sim, F, rec, det, meta } = ctx;
  const w = [];
  const push = (k, v) => w.push([k, v]);
  if (s.ph === 'frame') {
    push('wh', Lx('这一帧', 'This frame'));
    push(Lx('t（里程）', 't (distance)'), F.t);
    push(Lx('第几局', 'Game #'), F.gameNo);
    push(Lx('模式', 'Mode'), sim.mode === 'open' ? Lx('睁眼', 'eyes open') : Lx(`闭眼（第 ${F.dreamAge} 步）`, `eyes closed (step ${F.dreamAge})`));
    push(Lx('温度 τ', 'Temperature τ'), sim.tau.toFixed(2));
    push(Lx('上一个动作', 'Last action'), F.rec ? actName(F.rec.a) : '—');
    push(Lx('梦 vs 真实（MSE）', 'Dream vs real (MSE)'), fmtMSE(F.mse));
    push(Lx('M 说撞车的概率', 'M’s crash probability'), pct(F.dreamDone || 0));
    push('wh', Lx('记忆', 'Memory'));
    push('|h|', f3(norm(F.h)));
    push('|c|', f3(norm(F.c)));
    push('wh', Lx('模型', 'Model'));
    push(Lx('V 参数', 'V params'), meta.params.V.toLocaleString(Lx(undefined, 'en-US')));
    push(Lx('M 参数', 'M params'), meta.params.M.toLocaleString(Lx(undefined, 'en-US')));
    if (meta.params.C) push(Lx('C 参数', 'C params'), meta.params.C.toLocaleString(Lx(undefined, 'en-US')));
    return w;
  }
  if (!rec || rec.kind !== 'step') return w;
  push('wh', Lx(`第 ${F.t} 帧`, `Frame ${F.t}`));
  push(Lx('a（动作）', 'a (action)'), `${actName(rec.a)}${byTag(rec.by)}`);
  push(Lx('z 的来源', 'Source of z'), rec.src === 'enc' ? Lx('睁眼：编码真实画面', 'eyes open: encoded real frame') : rec.src === 'resync' ? Lx('重新睁眼对齐', 'eyes reopened to realign') : Lx('闭眼：上一步的 ẑ', 'eyes closed: last step’s ẑ'));
  push('τ', rec.tau.toFixed(2));
  if (s.ph === 'enc' || s.ph === 'obs') {
    const e = det?.enc;
    if (e) { push('|μ|', f3(norm(e.mu))); push(Lx('平均 σ', 'Mean σ'), f3(e.lv.reduce((a, v) => a + Math.exp(v / 2), 0) / Z)); }
    if (s.op && det?.enc[s.op]) { const st = stats(det.enc[s.op]); push(Lx('最大激活', 'Max activation'), f3(st.mx)); push(Lx('大于 0 的比例', 'Share above 0'), pct(st.nz / st.n)); }
  }
  if (s.ph === 'rnn' || s.ph === 'sample') {
    push('|z|', f3(norm(rec.z)));
    push('|h| → |h\'|', `${f3(norm(F.h))} → ${f3(norm(rec.L.h))}`);
    push('|c| → |c\'|', `${f3(norm(F.c))} → ${f3(norm(rec.L.c))}`);
    push(Lx('遗忘门平均', 'Forget gate mean'), f3(rec.L.f.reduce((a, v) => a + v, 0) / 256));
    push(Lx('输入门平均', 'Input gate mean'), f3(rec.L.i.reduce((a, v) => a + v, 0) / 256));
    push(Lx('p(撞车)', 'p(crash)'), pct(rec.M.done));
    if (s.ph === 'sample') push('|ẑ|', f3(norm(rec.S.z)));
  }
  if (s.ph === 'dec' && det?.dec) {
    if (s.op && s.op !== 'd4' && det.dec[s.op]) { const st = stats(det.dec[s.op]); push(Lx('最大激活', 'Max activation'), f3(st.mx)); push(Lx('大于 0 的比例', 'Share above 0'), pct(st.nz / st.n)); }
    push('|ẑ|', f3(norm(rec.S.z)));
  }
  if (s.ph === 'cmp') {
    const G = ctx.next;
    push(Lx('梦 vs 真实（MSE）', 'Dream vs real (MSE)'), fmtMSE(G?.mse));
    push(Lx('真实世界', 'Real world'), G?.realDone ? (G.why === 'offroad' ? Lx('冲出路面', 'off the road') : Lx('撞车', 'crashed')) : Lx('还在开', 'still driving'));
    push(Lx('M 说撞车', 'M says crash'), pct(rec.M.done));
  }
  if (s.mi && ctx.mac) for (const [k, v] of ctx.mac.watch()) push(k, v);
  return w;
}

export function renderWatch(el, rows) {
  el.innerHTML = rows.map(([k, v]) => (k === 'wh' ? `<div class="wh">${esc(v)}</div>` : `<span class="k">${esc(k)}</span><span class="v" title="${esc(String(v))}">${esc(String(v))}</span>`)).join('');
}
