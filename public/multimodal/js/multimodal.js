// Qwen2-VL recording viewer. Only schema 2 model captures are accepted.
import { Lab, fetchJSON } from '../../js/lab/core.js';
import { TraceStage } from '../../js/lab/stage3d.js';
import { $, esc } from '../../js/ui.js';
import { sfx, setSound, soundOn } from '../../js/audio.js';

const DATA = new Map();
const CODE = [
  'image = load(photo)',
  'pixels, grid = processor(image)',
  'patches = vision.patch_embed(pixels)  # 14px raw patches',
  'vision_tokens = vision.merge(patches) # 2×2 spatial merge',
  'sequence = [text, image_tokens, question]',
  'for layer in language_model.layers:',
  '    sequence = self_attention(sequence)',
  '    sequence = feed_forward(sequence)',
  'answer_token = greedy_decode(sequence)',
];
let lab;
let stage;

function assertCapture(d) {
  if (d.schema !== 2 || d.source !== 'model' || !Array.isArray(d.tokens) || !d.tokens.length) throw new Error('需要 schema 2 真实模型记录');
  if (d.patches.length !== d.grid.rows * d.grid.cols) throw new Error('视觉词元网格不匹配');
  for (const t of d.tokens) {
    if (t.weights.length !== d.patches.length || t.weights.some(w => !Number.isFinite(w) || w < 0)) throw new Error('注意力数据不完整');
    if (!Array.isArray(t.heads) || t.heads.length !== d.attention.heads || t.heads.some(h => h.weights.length !== d.patches.length)) throw new Error('注意力头数据不完整');
  }
  return d;
}

const get = n => DATA.get(n?.key);
const leaf = (t, label, extra = {}) => ({ t, label, dur: 1.4, ...extra });
function roots(examples) {
  return examples.map(ex => leaf('photo', ex.name, {
    key: ex.id, crumb: ex.name, dur: 2.4,
    kids: () => [
      leaf('pixels', '照片与预处理', { key: ex.id, kids: () => [leaf('raw', '原始照片', { key: ex.id }), leaf('grid', '动态切块网格', { key: ex.id })] }),
      leaf('vision', '视觉编码器', { key: ex.id, kids: () => DATA.get(ex.id).patches.map(p => leaf('patch', `视觉词元 #${p.id}`, { key: ex.id, patch: p.id })) }),
      leaf('fusion', '进入语言模型', { key: ex.id, kids: () => [leaf('sequence', '图像词元与文字词元', { key: ex.id }), leaf('self', '共享自注意力', { key: ex.id })] }),
      leaf('answer', '逐词元生成回答', { key: ex.id, kids: () => DATA.get(ex.id).tokens.map((t, i) => leaf('token', `${i + 1} · ${t.text || '空白'}`, { key: ex.id, token: i, crumb: `词元 ${i + 1}`, kids: () => [leaf('heat', '12 头平均注意力', { key: ex.id, token: i }), ...Array.from({ length: DATA.get(ex.id).attention.heads }, (_, h) => leaf('head', `注意力头 ${h}`, { key: ex.id, token: i, head: h })), leaf('vector', '视觉最终特征', { key: ex.id, token: i })] })) }),
    ],
  }));
}

function focus(path) {
  for (let i = path.length - 1; i >= 0; i--) if (path[i].token != null) return path[i].token;
  return 0;
}
function chosenPatch(path) {
  for (let i = path.length - 1; i >= 0; i--) if (path[i].patch != null) return path[i].patch;
  return null;
}

// Every glowing value below comes from this capture. Architecture-only boxes
// are labelled as such; the recording has no intermediate ViT activations.
function stageSpec(n, path, tree) {
  const d = get(n), depth = tree.depth;
  const photo = (id = 'photo') => ({ id, label: d.name, detail: `${d.originalSize.join('×')} px · 真实照片`, kind: 'matrix', imageUrl: d.image, imageAspect: d.originalSize[0] / d.originalSize[1] });
  const patch = (i) => {
    const p = d.patches[i];
    return { id: `patch:${i}`, label: `视觉 #${i}`, detail: `${p.row},${p.col} · ‖v‖ ${p.norm.toFixed(1)}`, kind: 'matrix', values: p.head };
  };
  const tokenNode = (i) => {
    const t = d.tokens[i];
    return { id: `token:${i}`, label: t.text || '空白', detail: `输出词元 ${i + 1} · ID ${t.id}`, kind: 'token', values: t.weights };
  };
  const samples = (count) => Array.from({ length: Math.min(count, d.patches.length) }, (_, i) => Math.floor(i * d.patches.length / Math.min(count, d.patches.length)));
  const tokenIndex = focus(path), current = d.tokens[tokenIndex];
  let nodes = [], edges = [], active = '', layout = 'flow', title = n.label;
  if (depth <= 2) {
    nodes = [photo('phase:photo'),
      { id: 'phase:pixels', label: '动态切块', detail: `${d.grid.rawRows}×${d.grid.rawCols} 原始块`, kind: 'matrix', values: d.patches.map(p => p.norm) },
      { id: 'phase:vision', label: '视觉编码器', detail: `${d.patches.length} 个最终输出`, kind: 'box', values: d.patches.map(p => p.norm) },
      { id: 'phase:fusion', label: '图文词元', detail: '共用语言自注意力', kind: 'box' },
      { id: 'phase:self', label: '语言层塔', detail: '28 层模型架构', kind: 'tower', layers: 28 },
      { id: 'phase:answer', label: '生成回答', detail: `${d.tokens.length} 个实测词元`, kind: 'token', values: current.weights }];
    nodes.forEach((node, i) => { node.position = [(i % 3 - 1) * 5.2, 1.3, Math.floor(i / 3) * 4.2]; });
    edges = nodes.slice(1).map((node, i) => [nodes[i].id, node.id]);
    active = depth === 1 ? 'phase:photo' : `phase:${n.t}`;
    title = depth === 1 ? `${d.name} · 一张照片如何变成回答` : `${d.name} · 从照片到回答的模型流程`;
  } else if (['pixels', 'raw', 'grid'].includes(n.t)) {
    nodes = [photo(), ...samples(9).map(patch)]; layout = 'grid';
    active = 'photo'; title = `${d.name} · ${d.grid.rawRows}×${d.grid.rawCols} 原始块 → ${d.grid.rows}×${d.grid.cols} 视觉词元（抽样展示）`;
  } else if (['vision', 'patch'].includes(n.t)) {
    const selected = chosenPatch(path);
    const indexes = selected == null ? samples(12) : [...new Set(Array.from({ length: 9 }, (_, k) => {
      const row = Math.max(0, Math.min(d.grid.rows - 1, Math.floor(selected / d.grid.cols) + Math.floor(k / 3) - 1));
      const col = Math.max(0, Math.min(d.grid.cols - 1, selected % d.grid.cols + k % 3 - 1));
      return row * d.grid.cols + col;
    }))];
    nodes = indexes.map(patch); layout = 'grid'; active = selected == null ? nodes[0].id : `patch:${selected}`;
    title = `${d.patches.length} 个视觉最终向量 · ${selected == null ? '均匀抽样' : '邻近词元'}`;
  } else if (['fusion', 'sequence'].includes(n.t)) {
    nodes = [photo(), { id: 'phase:vision', label: '视觉词元', detail: `${d.patches.length}×${d.grid.featureDim}`, kind: 'matrix', values: d.patches.map(p => p.norm) },
      { id: 'phase:sequence', label: '文字 + 图像', detail: '同一序列', kind: 'token' },
      { id: 'phase:self', label: '语言自注意力', detail: '28 层结构', kind: 'layer' },
      { id: 'phase:answer', label: '回答词元', detail: `${d.tokens.length} 个`, kind: 'token', values: current.weights }];
    nodes.forEach((node, i) => { node.position = [(i % 3 - 1) * 5.2, 1.3, Math.floor(i / 3) * 4.2]; });
    edges = nodes.slice(1).map((node, i) => [nodes[i].id, node.id]); active = n.t === 'sequence' ? 'phase:sequence' : 'phase:vision';
    title = '视觉词元与文字词元进入同一个语言序列';
  } else if (n.t === 'self') {
    nodes = Array.from({ length: 7 }, (_, i) => ({ id: `layer:${i}`, label: `语言层 ${i * 4}–${i * 4 + 3}`, detail: '架构示意 · 未录中间激活', kind: 'layer' }));
    layout = 'tower'; active = 'layer:6'; title = '28 层语言模型 · 图文共用自注意力';
  } else if (n.t === 'answer') {
    nodes = d.tokens.slice(0, 12).map((_, i) => tokenNode(i));
    edges = nodes.slice(1).map((node, i) => [nodes[i].id, node.id]); active = 'token:0'; layout = 'grid';
    title = `回答前 ${nodes.length} / ${d.tokens.length} 个真实输出词元`;
  } else if (n.t === 'token') {
    const nearby = [tokenIndex - 2, tokenIndex - 1, tokenIndex, tokenIndex + 1, tokenIndex + 2].filter(i => i >= 0 && i < d.tokens.length);
    const top = current.weights.map((w, i) => [i, w]).sort((a, b) => b[1] - a[1]).slice(0, 4);
    nodes = [photo(), ...top.map(([i]) => patch(i)), ...nearby.map(tokenNode)];
    edges = top.map(([i]) => [`patch:${i}`, `token:${tokenIndex}`]); active = `token:${tokenIndex}`; layout = 'grid';
    title = `输出词元 ${tokenIndex + 1} · 图像注意力最高的 4 个区域`;
  } else if (n.t === 'heat' || n.t === 'head') {
    const mobile = matchMedia('(max-width: 700px)').matches;
    const start = mobile ? Math.max(0, Math.min(current.heads.length - 5, (n.head ?? 0) - 2)) : 0;
    const visibleHeads = mobile ? current.heads.slice(start, start + 5) : current.heads;
    nodes = [{ id: 'mean', label: '12 头平均', detail: `图像注意力占比 ${(current.imageAttentionMass * 100).toFixed(1)}%`, kind: 'matrix', values: current.weights },
      ...visibleHeads.map((h, offset) => ({ id: `head:${start + offset}`, label: `注意力头 ${start + offset}`, detail: `图像注意力占比 ${(h.imageAttentionMass * 100).toFixed(1)}%`, kind: 'head', values: h.weights }))];
    active = n.t === 'head' ? `head:${n.head}` : 'mean'; layout = 'grid';
    title = `词元「${current.text}」 · 最后语言层 12 个实测注意力头${mobile ? '（逐步浏览）' : ''}`;
  } else if (n.t === 'vector') {
    const i = current.weights.indexOf(Math.max(...current.weights)), p = d.patches[i];
    nodes = [patch(i), ...p.head.map((value, j) => ({ id: `dim:${j}`, label: `维 ${j}`, detail: value.toFixed(3), kind: 'weight', values: [value] }))];
    active = `patch:${i}`; layout = 'grid'; title = `视觉词元 #${i} · 1536 维最终向量的前 12 维`;
  }
  return { title, depth, layout, nodes, edges, active };
}

function pickStageNode(id) {
  if (!lab) return;
  const tree = lab.tree;
  tree.pause();
  const [type, raw] = id.split(':');
  if (type === 'photo') tree.seekWhere(node => node.t === 'raw');
  else if (type === 'phase') tree.seekWhere(node => node.t === (raw === 'photo' ? 'raw' : raw === 'vision' ? 'vision' : raw === 'fusion' ? 'fusion' : raw === 'sequence' ? 'sequence' : raw === 'self' ? 'self' : raw === 'answer' ? 'answer' : 'pixels'));
  else if (type === 'patch') tree.seekWhere(node => node.t === 'patch' && node.patch === Number(raw));
  else if (type === 'token') tree.seekWhere(node => node.t === 'token' && node.token === Number(raw));
  else if (type === 'head') tree.seekWhere(node => node.t === 'head' && node.token === focus(tree.path) && node.head === Number(raw));
  else if (id === 'mean') tree.seekWhere(node => node.t === 'heat' && node.token === focus(tree.path));
  else if (type === 'layer') tree.seekWhere(node => node.t === 'self');
}

function side(d, node, path) {
  const idx = focus(path);
  const shown = node.t === 'answer' ? 0 : ['token', 'heat', 'head', 'vector'].includes(node.t) ? idx + 1 : 0;
  const text = shown ? d.tokens.slice(0, shown).map(t => t.text).join('') : '';
  $('#sideBody').innerHTML = `<img class="mm-side-photo" src="${esc(d.image)}" alt="${esc(d.name)}">
    <div><h4>提问</h4><div class="blk">${esc(d.question)}</div></div>
    <div><h4>模型回答 · ${shown}/${d.tokens.length} 词元</h4><div class="blk mm-answer">${esc(text || '按 ＋ 进入逐词元生成步骤')}</div></div>
    <div><h4>来源</h4><div class="blk mm-source">Qwen2-VL-2B-Instruct · 贪心解码<br>照片 SHA-256：<code title="${esc(d.imageSha256)}">${esc(d.imageSha256.slice(0, 16))}…</code><br>视觉网格 ${d.grid.rows}×${d.grid.cols} · 每个融合词元 ${d.grid.featureDim} 维</div></div>`;
}

function picture(d, weights = null, selected = null) {
  const { rows, cols } = d.grid;
  const max = weights ? Math.max(...weights) : 0;
  const cells = d.patches.map(p => {
    const w = weights?.[p.id] || 0;
    const alpha = weights && max > 0 ? Math.min(.65, .06 + .59 * w / max) : 0;
    return `<button class="mm-cell ${p.id === selected ? 'on' : ''}" type="button" data-patch="${p.id}" style="--heat:${alpha}" title="视觉词元 #${p.id} · 行 ${p.row} 列 ${p.col}${weights ? ` · 图像内注意力 ${(w * 100).toFixed(2)}%` : ''}"></button>`;
  }).join('');
  return `<div class="mm-photo-frame"><img src="${esc(d.image)}" alt="${esc(d.name)}"><div class="mm-grid" style="grid-template-columns:repeat(${cols},1fr);grid-template-rows:repeat(${rows},1fr)">${cells}</div></div>`;
}

function feature(d, patchIndex) {
  const p = d.patches[patchIndex];
  if (!p) return '';
  return `<div class="mm-feature"><b>视觉词元 #${p.id}</b><span>网格 ${p.row}, ${p.col}</span><span>向量维度 ${d.grid.featureDim}</span><span>范数 ${p.norm.toFixed(3)}</span><small>前 12 个实测分量</small><code>${p.head.map(x => x.toFixed(3)).join('  ')}</code></div>`;
}

function render(n, path, tree) {
  const d = get(n);
  if (!d) return;
  stage?.update(stageSpec(n, path, tree));
  side(d, n, path);
  const tokenIndex = focus(path), token = d.tokens[tokenIndex];
  const heat = ['token', 'heat', 'head', 'vector'].includes(n.t);
  const weights = n.t === 'head' ? token.heads[n.head].weights : heat ? token.weights : null;
  const selected = chosenPatch(path);
  const top = weights ? weights.map((w, i) => [i, w]).sort((a, b) => b[1] - a[1]).slice(0, 4) : [];
  let heading = { photo: '从真实照片开始', pixels: '照片与动态分辨率', raw: '原始照片', grid: '图像切块', vision: '视觉编码器最终输出', patch: `视觉词元 #${selected}`, fusion: '图像与文字汇合', sequence: '共同进入语言序列', self: '语言模型自注意力', answer: '逐词元生成回答', token: `生成词元 ${tokenIndex + 1}`, heat: '最后一层 · 12 头平均', head: `最后一层 · 注意力头 ${n.head}`, vector: '视觉最终向量' }[n.t];
  let detail = '';
  if (n.t === 'grid' || n.t === 'pixels') detail = `原图 ${d.originalSize.join('×')} 像素，模型预处理为 ${d.processedSize.join('×')} 像素；先切成 ${d.grid.rawRows}×${d.grid.rawCols} 个 14 像素原始块，再每 2×2 块合并为 ${d.grid.rows}×${d.grid.cols} 个视觉词元。网格来自这次实际模型输入。`;
  else if (n.t === 'vision' || n.t === 'patch') detail = `这里只录制视觉编码器最终输出：${d.patches.length} 个 ${d.grid.featureDim} 维向量；不包含 ViT 中间层。点击网格单元查看实测向量的范数与前 12 维。`;
  else if (n.t === 'fusion' || n.t === 'sequence' || n.t === 'self') detail = 'Qwen2-VL 把视觉编码器输出放入语言词元序列；文字与图像词元在语言模型的自注意力中交互。';
  else if (n.t === 'answer') detail = `模型以贪心解码生成 ${d.tokens.length} 个词元。进入下一层可逐个看词元及其图像注意力。`;
  else if (heat) detail = `当前词元「${esc(token.text)}」ID ${token.id}。语言模型最后一层第 ${d.attention.layer} 层${n.t === 'head' ? `注意力头 ${n.head}` : ` ${d.attention.heads} 个头的均值`}；仅在图像词元内部归一化。原始图像注意力质量 ${((n.t === 'head' ? token.heads[n.head].imageAttentionMass : token.imageAttentionMass) * 100).toFixed(2)}%。热区不能直接当作因果解释。`;
  $('#view').innerHTML = `<div class="panel mm-main"><h3>${heading}<small>${d.name} · ${d.grid.rows}×${d.grid.cols} 视觉词元</small></h3><p class="note">${detail}</p>${picture(d, weights, selected)}</div>
    <div class="panel mm-readout"><h3>${heat ? '关注区域' : '步骤数据'}<small>MODEL TRACE</small></h3>${heat ? `<div class="mm-top">${top.map(([i,w],rank) => `<button type="button" data-patch="${i}"><span>${rank + 1} · 词元 #${i}</span><b>${(w * 100).toFixed(2)}%</b></button>`).join('')}</div>` : `<div class="kv"><span class="k">原图</span><span class="v">${d.originalSize.join(' × ')} px</span><span class="k">原始切块</span><span class="v">${d.grid.rawRows} × ${d.grid.rawCols}</span><span class="k">融合词元</span><span class="v">${d.patches.length}</span></div>`}<div id="mmFeature">${feature(d, selected ?? top[0]?.[0] ?? 0)}</div></div>`;
  $('#view').querySelectorAll('[data-patch]').forEach(b => b.addEventListener('click', () => {
    const i = Number(b.dataset.patch);
    $('#mmFeature').innerHTML = feature(d, i);
    $('#view').querySelectorAll('.mm-cell').forEach(c => c.classList.toggle('on', Number(c.dataset.patch) === i));
  }));
}

function explain(n, path) {
  const d = get(n); if (!d) return {};
  const lines = { photo:[1], pixels:[2], raw:[1], grid:[3], vision:[3,4], patch:[4], fusion:[5], sequence:[5], self:[6,7,8], answer:[9], token:[7,9], heat:[7], head:[7], vector:[4] }[n.t] || [];
  const token = d.tokens[focus(path)];
  return { lines, shape: n.t === 'patch' || n.t === 'vision' ? `[${d.patches.length}, ${d.grid.featureDim}]` : ['token', 'heat', 'head'].includes(n.t) ? `[${d.patches.length}]` : '',
    html: `<p>${n.t === 'self' ? '视觉词元和文本词元共用语言模型的自注意力。' : n.t === 'heat' ? '图像热力图取语言模型最后一层所有注意力头的均值。' : n.t === 'head' ? `单独查看第 ${n.head} 个头到图像词元的实测注意力。` : '图像和回答均来自同一次离线模型运行。'}</p>`,
    watch: [['照片', d.name], ['网格', `${d.grid.rows}×${d.grid.cols}`], ['向量维度', d.grid.featureDim], ...(['token', 'heat', 'head'].includes(n.t) ? [['当前词元', token.text], ['token ID', token.id], ['图像注意力质量', `${((n.t === 'head' ? token.heads[n.head].imageAttentionMass : token.imageAttentionMass) * 100).toFixed(2)}%`]] : [])] };
}

async function boot() {
  const sound = $('#btnSound');
  const renderSound = () => { sound.classList.toggle('on', soundOn()); sound.setAttribute('aria-pressed', String(soundOn())); };
  sound.addEventListener('click', () => { setSound(!soundOn()); renderSound(); sfx.click(); }); renderSound();
  $('#btnSide').addEventListener('click', () => document.body.classList.toggle('side-open'));
  $('#btnBack').addEventListener('click', () => { if (document.body.classList.contains('side-open')) document.body.classList.remove('side-open'); else { lab?.tree.pause(); document.body.classList.replace('mode-inspect', 'mode-pick'); } });
  try {
    const manifest = await fetchJSON('data/manifest.json');
    if (manifest.schema !== 2 || manifest.source !== 'model') throw new Error('现有数据是旧版模拟数据，尚未录制真实模型数据');
    const examples = manifest.examples;
    await Promise.all(examples.map(async ex => DATA.set(ex.id, assertCapture(await fetchJSON(`data/${ex.dataFile}`)))));
    $('#lead').innerHTML = `用 <b>${esc(manifest.model)}</b> 对真实照片提问，打开一次实际推理记录：图像切块、视觉编码器输出、图文词元的自注意力，以及逐词元回答。`;
    $('#spec').innerHTML = `<span>视觉块 <b>14 px</b></span><span>空间融合 <b>2×2</b></span><span>语言层 <b>28</b></span><span>照片 <b>${examples.length} 张</b></span>`;
    $('#cards').innerHTML = examples.map(ex => `<button class="card" type="button" data-id="${esc(ex.id)}"><span class="tag">真实照片 · ${ex.grid.rows}×${ex.grid.cols} 视觉词元</span><img src="${esc(ex.image)}" alt="${esc(ex.name)}"><h3>${esc(ex.name)}</h3><p>${esc(ex.question)}</p><span class="go">打开模型运行 →</span></button>`).join('');
    $('#credit').textContent = '来源：tools/export_multimodal.py 录制；照片 SHA-256、模型输出、视觉向量与最后一层语言注意力保存在每个示例 JSON 中。';
    stage = new TraceStage($('#stage'), { onPick: pickStageNode });
    lab = new Lab({ roots: roots(examples), maxDepth: 4, depthNames: ['', '照片', '处理流程', '视觉 / 生成词元', '词元细节'], code: CODE, explain, render,
      frame: (_node, progress) => stage?.frame(progress), onExit: () => document.body.classList.replace('mode-inspect', 'mode-pick') });
    $('#cards').addEventListener('click', e => {
      const card = e.target.closest('[data-id]'); if (!card) return;
      document.body.classList.replace('mode-pick', 'mode-inspect');
      document.body.classList.remove('side-open');
      if (matchMedia('(max-width: 900px)').matches && !$('#dbg').classList.contains('folded')) $('#btnDbgFold').click();
      lab.tree.seekRoot(examples.findIndex(x => x.id === card.dataset.id), 1);
    });
  } catch (error) {
    $('#lead').textContent = `真实模型记录暂不可用：${error.message}`;
    $('#spec').textContent = '当前数据不会作为真实注意力展示。';
    $('#cards').innerHTML = '';
    $('#credit').textContent = '需要先运行 tools/export_multimodal.py，完成真实模型录制。';
    console.error(error);
  }
}
boot();
