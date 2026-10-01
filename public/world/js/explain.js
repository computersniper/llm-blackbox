// 调试器右侧：伪代码（高亮当前行）、这一步的讲解、变量监视。所有数字都来自浏览器里刚算出来的真实前向。
import { esc } from '../../js/ui.js';
import { ACTIONS } from './game.js';
import { ENC, DEC, Z, KMIX } from './nn.js';

const kw = (s) => `<span class="kw">${s}</span>`;
const fn = (s) => `<span class="fn">${s}</span>`;
const cm = (s) => `<span class="cm">${s}</span>`;
const nu = (s) => `<span class="nu">${s}</span>`;

export const CODE = [
  cm('# V 看，M 记住并想象（Ha &amp; Schmidhuber 2018）'),
  `h, c = ${fn('zeros')}(${nu(256)}), ${fn('zeros')}(${nu(256)})`,
  `obs = env.${fn('reset')}()          ${cm('# 64×64×3')}`,
  `${kw('while')} True:`,
  `  ${kw('if')} 睁眼: z = V.${fn('encode')}(obs).μ`,
  `  ${kw('else')}:   z = ẑ           ${cm('# 闭眼')}`,
  `  a = 方向键 ${kw('or')} 自动驾驶 ${kw('or')} ${fn('C')}(z, h)`,
  `  π, μ, σ, done, h, c = ${fn('M')}(z, a, h, c)`,
  `  ẑ = ${fn('sample')}(π, μ, σ, τ)   ${cm('# 下一帧的 z')}`,
  `  ô = V.${fn('decode')}(ẑ)         ${cm('# 梦见的下一帧')}`,
  `  obs = env.${fn('step')}(a)       ${cm('# 真实的下一帧')}`,
  `  误差 = ${fn('mean')}((obs − ô)²)`,
  '',
  `${kw('def')} V.${fn('encode')}(obs):         ${cm('# 卷积 VAE')}`,
  `  x = ${fn('relu')}(${fn('conv')}(obs, ${nu(16)}, ${nu(4)}, ${nu(2)})) ${cm('# 31²×16')}`,
  `  x = ${fn('relu')}(${fn('conv')}(x, ${nu(32)}, ${nu(4)}, ${nu(2)}))   ${cm('# 14²×32')}`,
  `  x = ${fn('relu')}(${fn('conv')}(x, ${nu(64)}, ${nu(4)}, ${nu(2)}))   ${cm('# 6²×64')}`,
  `  x = ${fn('relu')}(${fn('conv')}(x, ${nu(128)}, ${nu(4)}, ${nu(2)}))  ${cm('# 2²×128')}`,
  `  μ, logσ² = W<sub>μ</sub>·x+b, W<sub>σ</sub>·x+b ${cm('# 32, 32')}`,
  `  ${kw('return')} μ             ${cm('# 训练时 μ+σ·ε')}`,
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
  `  i = σ(W<sub>i</sub>·[x, h] + b<sub>i</sub>)     ${cm('# 输入门')}`,
  `  f = σ(W<sub>f</sub>·[x, h] + b<sub>f</sub>)     ${cm('# 遗忘门')}`,
  `  g = ${fn('tanh')}(W<sub>g</sub>·[x, h] + b<sub>g</sub>)  ${cm('# 候选记忆')}`,
  `  o = σ(W<sub>o</sub>·[x, h] + b<sub>o</sub>)     ${cm('# 输出门')}`,
  `  c = f*c + i*g`,
  `  h = o*${fn('tanh')}(c)`,
  `  π, μ, logσ = W<sub>head</sub>·h + b ${cm('# 各 32×5')}`,
  `  done = σ(w<sub>done</sub>·h + b)   ${cm('# 撞车了吗')}`,
  '',
  `${kw('def')} ${fn('sample')}(π, μ, σ, τ):     ${cm('# 每一维')}`,
  `  k ~ ${fn('softmax')}(log π / τ)    ${cm('# 挑一个高斯')}`,
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

export const STAGE_LABEL = { frame: '一帧', reset: '新的一局', obs: '看：真实画面', enc: '编码', rnn: '记忆与预测', sample: '采样', dec: '解码', cmp: '对照' };
export const OP_LABEL = {
  e1: '卷积 1', e2: '卷积 2', e3: '卷积 3', e4: '卷积 4', mu: 'μ 和 σ', z: 'z（32 个数）',
  dfc: '全连接', d1: '反卷积 1', d2: '反卷积 2', d3: '反卷积 3', d4: '反卷积 4',
  cat: '拼接输入', gi: '输入门', gf: '遗忘门', gg: '候选记忆', go: '输出门', cell: '细胞状态', hid: '隐状态', mdn: '混合密度', done: '撞车概率',
  pick: '挑一个高斯', draw: '加上噪声',
};
export const MI_LABEL = { pick: '选一个输出', mul: '逐项相乘', sum: '加起来', act: '激活' };

export function stepLabel(s, F) {
  if (s.ph === 'frame') return F ? `第 ${F.t} 帧` : '一帧';
  let l = STAGE_LABEL[s.ph];
  if (s.op) l = OP_LABEL[s.op];
  if (s.mi) l += ` · ${MI_LABEL[s.mi]}`;
  return l;
}

const DNAME = ['', '玩', '循环', 'V 的内部', 'M 的内部', '一次乘加'];
export function crumbs(tl) {
  return tl.path().map((p) => ({ d: p.d, label: DNAME[p.d] }));
}

/* ---------------------------------------------------------------- 小工具 */

export const fmtMSE = (v) => (v == null ? '—' : v < 0.0001 ? v.toExponential(1) : v.toFixed(4));
const f3 = (v) => (Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 10 ? v.toFixed(2) : v.toFixed(3));
const pct = (p) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;
const actName = (a) => ACTIONS[a] || '—';

// 潜变量各维按 KL 从大到小排（KL≈0 的是没用上的维度）
let ORDER = null;
export function dimOrder(meta) {
  if (!ORDER) ORDER = meta.dims.map((d, i) => i).sort((a, b) => meta.dims[b].kl - meta.dims[a].kl);
  return ORDER;
}
// 一维“有没有用上”：KL（和标准正态的距离）大于 0.05
export const used = (meta, d) => meta.dims[d].kl > 0.05;
// 一维的含义：导出时把这一维从数据里的 1% 分位扫到 99% 分位、其余不动，解码出来量的（小车、三段路、抛锚车）
const SHORT = { car_x: '车位', road_near: '近路', road_mid: '中路', road_far: '远路', ob_x: '障碍↔', ob_y: '障碍↕', ob_amt: '有车' };
export function dimShort(meta, d) {
  const sw = meta.dims[d].sweep;
  return used(meta, d) && sw && (sw.feat === 'ob_amt' ? sw.px >= 0.5 : sw.px >= 2) ? SHORT[sw.feat] || '' : '';
}
export function dimMeaning(meta, d) {
  const sw = meta.dims[d].sweep;
  if (!used(meta, d) || !sw) return '没用上';
  if (sw.feat === 'ob_amt') return sw.px >= 0.5 ? `抛锚车出现 / 消失（约 ${sw.px.toFixed(1)} 辆）` : '几乎看不出变化';
  return sw.px >= 2 ? `${sw.label}，移动约 ${Math.round(sw.px)} 个像素` : '变化很小';
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
    return shapeHTML([`[${L.inH}×${L.inH}×${L.cin}] → ${kind} ${L.cout} 个 ${L.k}×${L.k} 核，步长 ${L.s} → `, [`[${L.outH}×${L.outH}×${L.cout}]`], ` · 权重 ${L.cin}×${L.cout}×${L.k}×${L.k} = ${(L.cin * L.cout * L.k * L.k).toLocaleString()}`]);
  }
  switch (s.op) {
    case 'mu': return shapeHTML(['x ', ['[512]'], ' @ W<sub>μ</sub> [512×32] → μ ', ['[32]'], '；W<sub>σ</sub> 同形 → logσ² ', ['[32]']]);
    case 'z': return shapeHTML(['z = μ ', ['[32]'], '（网页里不加噪声；训练时 z = μ + σ·ε）']);
    case 'dfc': return shapeHTML(['z ', ['[32]'], ' @ W [32×512] → ', ['[1×1×512]']]);
    case 'cat': return shapeHTML(['concat(z ', ['[32]'], ', onehot(a) ', ['[3]'], ') → x ', ['[35]']]);
    case 'gi': case 'gf': case 'gg': case 'go': return shapeHTML(['W<sub>ih</sub> [256×35]·x + W<sub>hh</sub> [256×256]·h + b → ', ['[256]']]);
    case 'cell': return shapeHTML(['c\' = f ⊙ c + i ⊙ g → ', ['[256]']]);
    case 'hid': return shapeHTML(['h\' = o ⊙ tanh(c\') → ', ['[256]']]);
    case 'mdn': return shapeHTML(['h ', ['[256]'], ' @ W<sub>head</sub> [256×481] → log π, μ, log σ 各 ', ['[32×5]'], ' + done ', ['[1]']]);
    case 'done': return shapeHTML(['w<sub>done</sub> [256] · h + b → σ → ', ['p(撞车)']]);
    case 'pick': return shapeHTML(['softmax(log π / τ) ', ['[32×5]'], ' → 每一维挑一个 k']);
    case 'draw': return shapeHTML(['ẑ = μ[k] + σ[k]·√τ·ε → ', ['[32]']]);
  }
  switch (s.ph) {
    case 'enc': return shapeHTML(['obs ', ['[64×64×3]'], ' → 4 层卷积 → μ ', ['[32]']]);
    case 'rnn': return shapeHTML(['[z ', ['[32]'], ', a ', ['[3]'], '] + h ', ['[256]'], ' → LSTM → h\' ', ['[256]'], ' → MDN ', ['[32×5×3 + 1]']]);
    case 'sample': return shapeHTML(['π, μ, σ ', ['[32×5]'], ' → ẑ ', ['[32]']]);
    case 'dec': return shapeHTML(['ẑ ', ['[32]'], ' → 4 层反卷积 → ô ', ['[64×64×3]']]);
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
      ? `<b>闭着眼</b>：梦不再看真实画面，用它自己上一步想出来的 z 往下想${F.dreamAge ? `（已经想了 ${F.dreamAge} 步）` : ''}。路是它自己“编”的，所以会和真实的越来越不一样；但小车跟着你的方向键走，这是它从数据里学到的因果。`
      : '<b>睁着眼</b>：每一步都先把真实画面压成 32 个数，再让 M 预测下一帧。所以梦总是只差一步，误差很小。按 <kbd>O</kbd> 闭眼，看梦自己往下想。';
    return `<p>左边是游戏本身，右边是<b>模型的梦</b>：一个只有 ${(meta.params.V + meta.params.M).toLocaleString()} 个参数的神经网络，在本地显卡上看了 <b>${tr.collect.frames.toLocaleString()} 帧</b>开车画面学会的“这个世界接下来会怎样”。你的每一个动作同时发给两边。</p><p>${dreamTxt}</p>${r?.by === 'ctrl' ? `<p>现在是 <b>C 在开车</b>：它只看 z 和 h（梦里的东西），三个动作的打分 ${[...r.logits].map((v) => v.toFixed(2)).join(' / ')}。</p>` : ''}<p class="dimmed">点 ＋ 把这一帧拆开，看 V 和 M 各做了什么。</p>`;
  }
  if (s.ph === 'reset') return `<p>真实世界里${F.why === 'offroad' ? '冲出了路面' : '撞上了抛锚的车'}，这一局结束。换一条新路（新的随机种子），梦也重新从真实画面的编码开始，M 的记忆 h、c 清零。</p>`;
  if (s.mi) return mac ? mac.explain(s.mi) : '';
  switch (s.ph) {
    case 'obs': return `<p>真实世界给出这一帧的画面 obs：64×64 个像素，每个像素 3 个颜色通道，一共 ${(64 * 64 * 3).toLocaleString()} 个数。</p><p>游戏是确定的：同一个种子、同一串动作，Python 采集训练数据时的游戏和这里的 game.js 逐像素相同。</p>${mode === 'closed' ? '<p class="dimmed">现在闭着眼：这一帧梦<b>不会</b>去看。</p>' : ''}`;
    case 'enc':
      if (s.op) return explainEnc(s.op, ctx);
      if (rec.src === 'dream') return `<p><b>闭眼</b>：跳过编码器，直接拿梦上一步采样出来的 ẑ 当作这一帧的 z。梦从这里开始脱离现实。</p>`;
      return `<p><b>V（视觉）</b>把画面压缩成 <b>32 个数</b> z。它是一个卷积变分自编码器：4 层卷积一层层缩小画面，最后两个全连接层给出每一维的均值 μ 和方差 σ²。</p><p>${(64 * 64 * 3).toLocaleString()} 个数 → 32 个数，压缩了 ${Math.round(12288 / 32)} 倍。</p>${rec.src === 'resync' ? '<p>上一步梦里“撞车”了，这一步重新睁眼对齐。</p>' : ''}`;
    case 'rnn':
      if (s.op) return explainM(s.op, ctx);
      return `<p><b>M（记忆）</b>是一个 LSTM，带着 256 维的记忆 h、c。输入是这一帧的 z 和你的动作「${actName(rec.a)}」，输出下一帧 z 的<b>概率分布</b>：每一维是 5 个高斯的混合（混合密度网络，MDN）。</p><p>它还顺便预测“这一步会不会撞车”：<b>${pct(rec.M.done)}</b>。</p><p class="dimmed">为什么输出分布而不是一个数？路前面会刷出什么车是随机的，一个数只能给出“平均情况”，糊成一团。</p>`;
    case 'sample':
      if (s.op) return explainS(s.op, ctx);
      return `<p>从 M 给出的分布里<b>抽一个</b>下一帧的 z：温度 τ = <b>${rec.tau.toFixed(2)}</b>。τ 越大，越敢抽不太可能的结果，梦越“放飞”；τ → 0 就只取最可能的那个高斯的中心。</p><p>论文里，在 τ 偏低的梦里训练的控制器会钻梦的空子，到真实世界就不灵了；调高 τ 让梦更难，反而学得更稳。</p>`;
    case 'dec':
      if (s.op) return explainDec(s.op, ctx);
      return `<p><b>V 的解码器</b>把 32 个数重新画成 64×64 的画面：先用全连接层放大成 512 个数，再经过 4 层反卷积（转置卷积）一层层放大，最后 sigmoid 把每个像素压到 0–1。</p><p>这就是<b>梦见的下一帧</b> ô。画面里所有东西——路的弯、车的位置、抛锚车——都是从这 32 个数里画出来的。</p>`;
    case 'cmp': {
      const G = ctx.next;
      return `<p>真实世界也走了一步（动作「${actName(rec.a)}」）。把梦见的 ô 和真实的下一帧逐像素比较：每像素均方误差 <b>${fmtMSE(G?.mse)}</b>。</p><p>${mode === 'closed' ? '闭眼时误差会一步步累积：一开始只是虚线的相位、远处的弯不对，后来连路往哪拐、哪里有车都对不上了。' : '睁眼时每一步都从真实画面重新出发，误差主要来自两处：V 画得不够细（虚线、草丛被抹平），和前方新刷出来的东西本来就猜不到。'}</p><p class="dimmed">测试集平均：只看 V 的重建 ${fmtMSE(meta.drift.recon)}；睁眼一步预测 ${fmtMSE(meta.drift.open[0])}；闭眼 30 步 ${fmtMSE(meta.drift['closed_1.0'][29])}。</p>`;
    }
  }
  return '';
}

function actStats(name, a) {
  const st = stats(a);
  return `<p class="dimmed">${name}：${st.n.toLocaleString()} 个数，最大 ${f3(st.mx)}，${pct(st.nz / st.n)} 大于 0（ReLU 把负数都变成了 0）。</p>`;
}

function explainEnc(op, { det, meta, sim, F }) {
  const L = ENC.find((l) => l.id === op);
  if (L) {
    const a = det.enc[op];
    const lead = {
      e1: '第一层卷积：16 个 4×4 的小卷积核在画面上每隔 2 个像素滑一次。每个核只管一件小事——有的对“亮的竖线”（路沿）敏感，有的对“青色”（小车）、“红色”（抛锚车）敏感。',
      e2: '第二层：在上一层 16 张特征图的基础上，再用 32 个 4×4×16 的核组合出更大的图案。每个输出像素“看”到的原图范围从 4×4 扩大到 10×10。',
      e3: '第三层：64 张 6×6 的特征图。每个位置已经能“看到”原图 22×22 的一块，开始对应“这里有一段弯路”“这里有辆车”这种整体的东西。',
      e4: '第四层：128 张 2×2 的图，一共 512 个数。到这里空间位置几乎被压没了，信息都转到了“哪个通道亮”上。',
    }[op];
    return `<p>${lead}</p>${actStats(`这一层输出 ${L.outH}×${L.outH}×${L.cout}`, a)}<p class="dimmed">点舞台上的任意一格，再按 ＋ 看它是怎么乘加出来的。</p>`;
  }
  if (op === 'mu') {
    const mu = det.enc.mu, lv = det.enc.lv;
    const nUsed = meta.dims.filter((d, i) => used(meta, i)).length;
    return `<p>两个全连接层把 512 个数各自变成 32 个：<b>μ</b>（这一维“最可能”是多少）和 <b>logσ²</b>（有多不确定）。</p><p>训练时还要让它们接近标准正态分布（KL 项）：没什么用的维度会被压成 μ≈0、σ≈1，等于没用上。这个模型 32 维里用上了 <b>${nUsed}</b> 维。</p><p class="dimmed">这一帧 |μ| = ${f3(norm(mu))}，平均 σ = ${f3(lv.reduce((s, v) => s + Math.exp(v / 2), 0) / Z)}</p>`;
  }
  if (op === 'z') {
    const order = dimOrder(meta).slice(0, 4);
    const pr = meta.probes;
    return `<p>这 32 个数就是 V 眼里的整个世界。舞台下方把用得最多的几维各自<b>扫了一遍</b>：只改这一维、其他不动，解码出来的画面怎么变——有的维管小车左右，有的管路往哪弯，有的管抛锚车在哪。</p><p>${order.map((d) => `z<sub>${d}</sub>：${esc(dimMeaning(meta, d))}`).join('<br>')}</p><p class="dimmed">“含义”是导出时量的：以 32 帧的 μ 为底，把这一维从数据里的 1% 分位扫到 99% 分位，在解码出来的画面上量小车、三段路的中心、抛锚车跟着动了多少，取动得最多的那样。一维往往不只管一件事（同时动了好几样），这里只写最明显的。</p><p class="dimmed">反过来，从 32 个数用一个线性层读游戏状态（测试集 R²）：小车位置 ${pr.car_x}，车旁的路 ${pr.road_near}，远处的路 ${pr.road_far}，抛锚车左右 ${pr.ob_x}。</p>`;
  }
  return '';
}

function explainDec(op, { det }) {
  const d = det.dec;
  if (!d) return '';
  if (op === 'dfc') return `<p>解码先用一个全连接层把 32 个数放大成 512 个，排成 1×1×512 的“一个像素、512 个通道”。</p>${actStats('这 512 个数', d.dfc)}`;
  const lead = {
    d1: '反卷积（转置卷积）和卷积反过来：每个输入像素拿着自己的数，把一个 5×5 的核“盖章”到输出上。1×1 → 5×5，开始有了空间结构。',
    d2: '5×5 → 13×13：步长 2，相邻两个“章”有重叠，重叠的地方加起来。这一层大致定下路在哪、车在哪。',
    d3: '13×13 → 30×30：16 个通道各管一类东西，有的亮在路面，有的亮在车上、路沿上。',
    d4: '最后一层：30×30×16 → 64×64×3，正好是红、绿、蓝三个通道，再经过 sigmoid 压到 0–1，就是梦见的画面。',
  }[op];
  const a = op === 'd4' ? d.y : d[op];
  return `<p>${lead}</p>${op === 'd4' ? '' : actStats('这一层输出', a)}<p class="dimmed">点舞台上的任意一格（或者梦里的任意一个像素），再按 ＋ 看它是怎么乘加出来的。</p>`;
}

function topIdx(a, k = 3, off = 0, n = a.length) {
  const idx = Array.from({ length: n }, (_, i) => i + off);
  idx.sort((x, y) => Math.abs(a[y]) - Math.abs(a[x]));
  return idx.slice(0, k);
}

function explainM(op, { rec, F, meta }) {
  const L = rec.L, H = 256;
  switch (op) {
    case 'cat': return `<p>把这一帧的 z（32 个数）和动作「${actName(rec.a)}」的 one-hot（[${[0, 1, 2].map((i) => (i === rec.a ? 1 : 0)).join(', ')}]）拼成 35 个数，作为 LSTM 这一步的输入。动作就是这样“告诉”模型的：同一个 z，按不同的键，M 会预测出不同的下一帧。</p>`;
    case 'gi': return `<p><b>输入门 i</b>：这一步的新信息，每一维放进去多少（0 到 1）。平均 ${f3(L.i.reduce((s, v) => s + v, 0) / H)}。</p><p class="dimmed">四个门都是同一个形状的计算：W·[x, h] + b，291 个输入，只是权重不同、激活函数不同。</p>`;
    case 'gf': return `<p><b>遗忘门 f</b>：旧的记忆 c 每一维留下多少。接近 1 = 记住，接近 0 = 忘掉。这一步平均 ${f3(L.f.reduce((s, v) => s + v, 0) / H)}，${pct(L.f.filter((v) => v > 0.9).length / H)} 的维度几乎完全保留。</p>`;
    case 'gg': return `<p><b>候选记忆 g</b>：tanh 激活，−1 到 1，是这一步“想写进记忆”的内容。</p>`;
    case 'go': return `<p><b>输出门 o</b>：记忆 c 里有多少要露出来，变成这一步的隐状态 h。</p>`;
    case 'cell': return `<p><b>细胞状态</b> c' = f ⊙ c + i ⊙ g：先按遗忘门留下一部分旧记忆，再按输入门加上新内容。c 是 LSTM 能把信息带过很多步的关键——比如“刚才那辆抛锚车已经开过去了”。</p><p class="dimmed">|c| 从 ${f3(norm(F.c))} 变成 ${f3(norm(L.c))}</p>`;
    case 'hid': return `<p><b>隐状态</b> h' = o ⊙ tanh(c')：256 个数，是 M 对“现在世界处在什么状态、接下来会怎样”的全部理解。下一帧的分布、撞车概率都从它算出来；控制器 C 也看它。</p>`;
    case 'mdn': {
      const sp = spreadOf(rec);
      let d = 0;
      for (let k = 0; k < Z; k++) if (dimShort(meta, k) && (!dimShort(meta, d) || sp[k] > sp[d])) d = k;
      return `<p><b>混合密度网络</b>：一个全连接层把 h 变成 481 个数——32 维 × 5 个高斯 ×（权重 π、中心 μ、宽度 σ）+ 1 个撞车 logit。</p><p>每一维单独一个“5 个山峰”的分布。大多数维度只有一个峰很高（模型很确定）；不确定的维度会有好几个峰，比如 z<sub>${d}</sub>——前面会不会刷出车、刷在哪，本来就说不准。</p>`;
    }
    case 'done': return `<p>撞车概率：sigmoid(w·h + b) = <b>${pct(rec.M.done)}</b>。训练时只有每局最后一步是 1，所以模型学会了在“快撞上”的时候把它调高。闭眼时它超过 50%，这一段梦就到此为止，下一步重新睁眼。</p>`;
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
    return `<p>每一维先按 softmax(log π / τ) 挑一个高斯分量（τ = ${rec.tau.toFixed(2)}）。τ < 1 让大的 π 更大、小的更小，梦更“保守”；τ > 1 反过来。</p><p class="dimmed">这一步 32 维挑中的分量：${counts.map((c, k) => `第 ${k} 个 ${c} 维`).join('，')}</p>`;
  }
  return `<p>在挑中的高斯里抽一个数：ẑ = μ[k] + σ[k]·√τ·ε，ε 来自标准正态分布。这 32 个数就是梦里的<b>下一帧</b>。</p><p class="dimmed">网页里的随机数是固定种子的，所以调试器退回再前进，梦不会变。</p>`;
}

/* ---------------------------------------------------------------- 变量监视 */

export function watch(s, ctx) {
  const { sim, F, rec, det, meta } = ctx;
  const w = [];
  const push = (k, v) => w.push([k, v]);
  if (s.ph === 'frame') {
    push('wh', '这一帧');
    push('t（里程）', F.t);
    push('第几局', F.gameNo);
    push('模式', sim.mode === 'open' ? '睁眼' : `闭眼（第 ${F.dreamAge} 步）`);
    push('温度 τ', sim.tau.toFixed(2));
    push('上一个动作', F.rec ? actName(F.rec.a) : '—');
    push('梦 vs 真实（MSE）', fmtMSE(F.mse));
    push('M 说撞车的概率', pct(F.dreamDone || 0));
    push('wh', '记忆');
    push('|h|', f3(norm(F.h)));
    push('|c|', f3(norm(F.c)));
    push('wh', '模型');
    push('V 参数', meta.params.V.toLocaleString());
    push('M 参数', meta.params.M.toLocaleString());
    if (meta.params.C) push('C 参数', meta.params.C.toLocaleString());
    return w;
  }
  if (!rec || rec.kind !== 'step') return w;
  push('wh', `第 ${F.t} 帧`);
  push('a（动作）', `${actName(rec.a)}${rec.by === 'key' ? '（你按的）' : rec.by === 'heur' ? '（自动驾驶）' : rec.by === 'ctrl' ? '（C）' : ''}`);
  push('z 的来源', rec.src === 'enc' ? '睁眼：编码真实画面' : rec.src === 'resync' ? '重新睁眼对齐' : '闭眼：上一步的 ẑ');
  push('τ', rec.tau.toFixed(2));
  if (s.ph === 'enc' || s.ph === 'obs') {
    const e = det?.enc;
    if (e) { push('|μ|', f3(norm(e.mu))); push('平均 σ', f3(e.lv.reduce((a, v) => a + Math.exp(v / 2), 0) / Z)); }
    if (s.op && det?.enc[s.op]) { const st = stats(det.enc[s.op]); push('最大激活', f3(st.mx)); push('大于 0 的比例', pct(st.nz / st.n)); }
  }
  if (s.ph === 'rnn' || s.ph === 'sample') {
    push('|z|', f3(norm(rec.z)));
    push('|h| → |h\'|', `${f3(norm(F.h))} → ${f3(norm(rec.L.h))}`);
    push('|c| → |c\'|', `${f3(norm(F.c))} → ${f3(norm(rec.L.c))}`);
    push('遗忘门平均', f3(rec.L.f.reduce((a, v) => a + v, 0) / 256));
    push('输入门平均', f3(rec.L.i.reduce((a, v) => a + v, 0) / 256));
    push('p(撞车)', pct(rec.M.done));
    if (s.ph === 'sample') push('|ẑ|', f3(norm(rec.S.z)));
  }
  if (s.ph === 'dec' && det?.dec) {
    if (s.op && s.op !== 'd4' && det.dec[s.op]) { const st = stats(det.dec[s.op]); push('最大激活', f3(st.mx)); push('大于 0 的比例', pct(st.nz / st.n)); }
    push('|ẑ|', f3(norm(rec.S.z)));
  }
  if (s.ph === 'cmp') {
    const G = ctx.next;
    push('梦 vs 真实（MSE）', fmtMSE(G?.mse));
    push('真实世界', G?.realDone ? (G.why === 'offroad' ? '冲出路面' : '撞车') : '还在开');
    push('M 说撞车', pct(rec.M.done));
  }
  if (s.mi && ctx.mac) for (const [k, v] of ctx.mac.watch()) push(k, v);
  return w;
}

export function renderWatch(el, rows) {
  el.innerHTML = rows.map(([k, v]) => (k === 'wh' ? `<div class="wh">${esc(v)}</div>` : `<span class="k">${esc(k)}</span><span class="v" title="${esc(String(v))}">${esc(String(v))}</span>`)).join('');
}
