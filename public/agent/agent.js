// 智能体页：真实的 Qwen3-1.7B 在沙箱里完成编程任务的录像（tools/record_agent.py 导出）。
//
//   D1 每一轮      思考 → 调用一次工具 → 看结果；最后一轮不再调用工具，直接回答
//   D2 一轮之内    读上下文 / 思考 / 回答 / 调用工具 / 终端执行 / 结果写回
//   D3 更细        思考和回答的每一句；工具调用的每个词元；终端的命令、输出、文件 diff
//   D4 每个词元    这一句里逐个词元生成
//   D5 采样        这个词元是怎么抽出来的：原始概率 → 温度 + top-k → top-p → 抽签

import { Lab, fetchJSON, fmtP } from '../js/lab/core.js';
import { TraceStage } from '../js/lab/stage3d.js';
import { $, esc } from '../js/ui.js';

let manifest = null, task = null, lab = null, trace = null;
const cache = new Map();

// ---------------------------------------------------------------- 把一轮生成切成段

const SPECIAL = /^<\|.*\|>$|^<\/?think>$|^<\/?tool_call>$/;

function segment(turn) {
  const toks = turn.tokens;
  const segs = [];
  let mode = 'answer', cur = null;
  const open = (kind) => { cur = { kind, idx: [] }; segs.push(cur); };
  for (let i = 0; i < toks.length; i++) {
    const s = toks[i].s;
    if (s === '<think>') { open('think'); cur.idx.push(i); mode = 'think'; continue; }
    if (s === '</think>') { if (!cur) open('think'); cur.idx.push(i); mode = 'answer'; cur = null; continue; }
    if (s === '<tool_call>') { open('call'); cur.idx.push(i); mode = 'call'; continue; }
    if (s === '</tool_call>') { if (!cur) open('call'); cur.idx.push(i); mode = 'answer'; cur = null; continue; }
    if (s === '<|im_end|>') { (cur || segs[segs.length - 1] || (open('answer'), cur)).idx.push(i); continue; }
    if (!cur) {
      if (mode === 'answer' && !s.trim()) { (segs[segs.length - 1] || (open('answer'), cur)).idx.push(i); continue; }
      open(mode);
    }
    cur.idx.push(i);
  }
  return segs.filter((g) => g.idx.some((i) => !SPECIAL.test(toks[i].s) && toks[i].s.trim()));
}

// 句子：遇到句末标点或换行就断开，太长的也断开
function sentences(turn, idx) {
  const out = [];
  let cur = [];
  for (const i of idx) {
    cur.push(i);
    const s = turn.tokens[i].s;
    if (/[。！？!?；\n]\s*$|\.\s*$/.test(s) || cur.length >= 36) { out.push(cur); cur = []; }
  }
  if (cur.length) out.push(cur);
  return out.filter((c) => c.some((i) => turn.tokens[i].s.trim() && !SPECIAL.test(turn.tokens[i].s)));
}

const textOf = (turn, idx) => idx.map((i) => turn.tokens[i].s).join('');
const clean = (s) => s.replace(/<\|im_end\|>|<\/?think>|<\/?tool_call>/g, '').trim();

// ---------------------------------------------------------------- 步骤树

const SAMPLE = [['cands', '原始概率', 1.6], ['temp', '温度 0.6 + top-k 20', 1.6], ['topp', 'top-p 0.95 截断', 1.6], ['draw', '抽签', 1.8]];

function tokenNode(k, i, t) {
  const s = task.turns[k].tokens[i].s;
  return { t: 'tok', k, i, label: `词元 “${vis(s)}”`, crumb: `“${vis(s)}”`, dur: 0.9, kids: () => SAMPLE.map(([st, l, dur]) => ({ t: 'samp', st, k, i, label: l, dur })) };
}

function phases(k) {
  const turn = task.turns[k];
  const segs = segment(turn);
  const nodes = [{ t: 'ctx', k, label: '读上下文', dur: 1.8 }];
  let callIndex = 0;
  for (const g of segs) {
    if (g.kind === 'think' || g.kind === 'answer') {
      const sens = sentences(turn, g.idx);
      nodes.push({
        t: g.kind, k, idx: g.idx, label: g.kind === 'think' ? '思考' : (turn.calls.length ? '说明' : turn.autoCheck?.passed === false ? '尝试回答' : '最终回答'), dur: 3,
        kids: () => sens.map((c, j) => ({ t: 'sent', k, idx: c, kind: g.kind, j, label: `第 ${j + 1} 句`, crumb: `第 ${j + 1} 句`, dur: Math.max(1.2, c.length * 0.12), kids: () => c.map((i) => tokenNode(k, i)) })),
      });
    } else {
      nodes.push({ t: 'call', k, ci: callIndex++, idx: g.idx, label: '调用工具', dur: 2.2, kids: () => g.idx.map((i) => tokenNode(k, i)) });
    }
  }
  turn.calls.forEach((c, ci) => {
    const kids = [{ t: 'cmd', k, ci, label: c.name === 'bash' ? '敲命令' : '写文件', dur: 1.8 }, { t: 'out', k, ci, label: '输出', dur: 2.2 }];
    if (c.changes.length) kids.push({ t: 'diff', k, ci, label: '文件变化', dur: 2.4 });
    nodes.push({ t: 'exec', k, ci, label: '终端执行', dur: 2.6, kids: () => kids });
    nodes.push({ t: 'obs', k, ci, label: '结果写回上下文', dur: 1.8 });
  });
  if (turn.autoCheck) nodes.push({ t: 'check', k, label: turn.autoCheck.passed ? '自动复核通过' : '自动复核失败', dur: 2.4 });
  return nodes;
}

function turnLabel(turn, k) {
  const c = turn.calls[0];
  if (!c) return `第 ${k + 1} 轮 · ${turn.autoCheck?.passed === false ? '复核未通过' : '回答'}`;
  const a = c.name === 'bash' ? c.args.command : `write_file ${c.args.path}`;
  return `第 ${k + 1} 轮 · ${String(a).split('\n')[0].slice(0, 28)}`;
}

// ---------------------------------------------------------------- 显示

const vis = (s) => (s === '\n' ? '↵' : s === '\n\n' ? '↵↵' : /^\s+$/.test(s) ? '␣' : s.replace(/\n/g, '↵'));
const view = () => $('#view');

function stagePick(id) {
  const index = Number(id.slice(5));
  if (!id.startsWith('step-') || !Number.isInteger(index) || !lab) return;
  const tree = lab.tree;
  tree.pause();
  if (index === tree.i && tree.canInto()) tree.into();
  else tree.seekIndex(index);
}

function traceSpec(tree) {
  const depth = tree.depth;
  const turn = task.turns[tree.r];
  const list = tree.list;
  if (tree.node.t === 'samp') {
    const tok = turn.tokens[tree.node.i];
    const mode = tree.node.st;
    let data = mode === 'cands' ? tok.top.map(([s, p]) => [s, p]) : (tok.samplePool || []).map(([s, , q]) => [s, q]);
    if (mode === 'temp') {
      const powered = tok.top.map(([s, p]) => [s, p ** (1 / manifest.sampling.temperature)]);
      const sum = powered.reduce((total, [, p]) => total + p, 0);
      data = powered.map(([s, p]) => [s, sum ? p / sum : 0]);
    }
    const selected = data.findIndex(([s]) => s === tok.s);
    return {
      title: `词元 “${vis(tok.s)}” · ${tree.node.label}${mode === 'temp' ? '（前 8 名内归一化示意）' : ''}`,
      depth, layout: 'grid',
      nodes: data.map(([s, p], i) => ({ id: `prob-${i}`, label: vis(s),
        detail: `${mode === 'temp' ? '8 名内归一化' : mode === 'cands' ? '原始概率' : '实际采样概率'} ${fmtP(p)}${s === tok.s ? ' · 本次抽中' : ''}`, kind: 'bar', value: p,
        color: s === tok.s ? 0xffb65c : 0x5ef0d4 })),
      edges: [], active: selected >= 0 ? `prob-${selected}` : null,
    };
  }
  const radius = depth <= 2 ? list.length : depth === 3 ? 9 : 14;
  const start = depth <= 2 ? 0 : Math.max(0, Math.min(list.length - radius * 2 - 1, tree.i - radius));
  const end = depth <= 2 ? list.length : Math.min(list.length, start + radius * 2 + 1);
  const nodes = list.slice(start, end).map((path, offset) => {
    const n = path.at(-1), k = path[0].k, t = task.turns[k], i = start + offset;
    const tok = n.t === 'tok' ? t.tokens[n.i] : null;
    const values = n.t === 'turn' ? t.tokens.filter((_, j) => j % Math.max(1, Math.floor(t.tokens.length / 64)) === 0).slice(0, 64).map((x) => x.p)
      : tok ? tok.top.map(([, p]) => p)
      : n.idx ? n.idx.slice(0, 64).map((j) => t.tokens[j].p)
      : undefined;
    const kind = n.t === 'turn' || n.t === 'ctx' ? 'layer'
      : n.t === 'tok' ? 'token'
      : n.t === 'think' ? 'head'
      : n.t === 'call' || n.t === 'diff' ? 'matrix'
      : n.t === 'samp' || n.t === 'check' ? 'weight' : 'box';
    const detail = n.t === 'turn' ? `${t.tokens.length} 词元 · ${t.calls.length} 次工具调用`
      : tok ? `原始概率 ${fmtP(tok.p)} · 熵 ${tok.H.toFixed(2)}`
      : n.t === 'exec' ? `${t.calls[n.ci].name} · 退出码 ${t.calls[n.ci].result.code}`
      : n.t === 'check' ? `独立复核 ${t.autoCheck?.passed ? '通过' : '失败'}`
      : n.idx ? `${n.idx.length} 词元` : '';
    return { id: `step-${i}`, label: n.t === 'tok' ? vis(tok.s) : n.label, detail, kind, values };
  });
  return { title: `${task.title} · D${depth} ${['', '每一轮', '一轮之内', '句 / 词元 / 终端', '每个词元', '采样'][depth]}`,
    depth, layout: depth === 1 ? 'tower' : 'flow', nodes, active: `step-${tree.i}` };
}

function probCls(p) { return p > 0.9 ? 'p9' : p > 0.6 ? 'p6' : p > 0.3 ? 'p3' : 'p0'; }

// 一段词元：每个词元一个小块，下划线颜色 = 模型对它有多确定
function tokSpan(turn, idx, cur, reveal) {
  return idx.map((i) => {
    const t = turn.tokens[i];
    if (SPECIAL.test(t.s)) return `<span class="tk sp ${i === cur ? 'cur' : ''}" data-i="${i}">${esc(t.s)}</span>`;
    const txt = esc(t.s).replace(/\n/g, '<br>');
    return `<span class="tk ${probCls(t.p)} ${i === cur ? 'cur' : ''} ${reveal ? 'rv' : ''}" data-i="${i}" title="p=${fmtP(t.p)}">${txt || ' '}</span>`;
  }).join('');
}

function segHTML(turn, g, { curSent = null, curTok = null, reveal = false } = {}) {
  const sens = g.kind === 'call' ? [g.idx] : sentences(turn, g.idx);
  return sens.map((c) => `<span class="sen ${curSent && c[0] === curSent[0] ? 'on' : ''}">${tokSpan(turn, c, curTok, reveal && curSent && c[0] === curSent[0])}</span>`).join('');
}

function callPretty(c) {
  if (!c) return '';
  if (c.name === 'bash') return `<div class="call"><span class="fnm">bash</span><pre>${esc(c.args.command || '')}</pre></div>`;
  if (c.name === 'write_file') return `<div class="call"><span class="fnm">write_file</span><span class="path">${esc(c.args.path || '')}</span><pre>${esc(c.args.content || '')}</pre></div>`;
  return `<div class="call"><span class="fnm">${esc(c.name)}</span><pre>${esc(JSON.stringify(c.args, null, 2))}</pre></div>`;
}

// 终端：前几轮执行过的命令 + 这一轮（按进度逐字敲出）
function termLines(k, uptoCi = Infinity) {
  const out = [];
  for (let j = 0; j <= k; j++) {
    for (const [ci, c] of task.turns[j].calls.entries()) {
      if (j === k && ci >= uptoCi) continue;
      out.push(cmdLine(c), ...outLines(c));
    }
  }
  return out;
}
function cmdLine(c) {
  return c.name === 'bash' ? `<div class="tl cmd"><span class="ps">agent@task:~/${task.id}$</span> ${esc(c.args.command || '')}</div>`
    : `<div class="tl cmd"><span class="ps">✎ write_file</span> ${esc(c.args.path || '')} <span class="muted">(${(c.args.content || '').split('\n').length} 行)</span></div>`;
}
function outLines(c) {
  const r = c.result;
  const lines = [];
  if (r.stdout) for (const l of r.stdout.replace(/\n$/, '').split('\n')) lines.push(`<div class="tl">${esc(l) || '&nbsp;'}</div>`);
  if (r.stderr) for (const l of r.stderr.replace(/\n$/, '').split('\n')) lines.push(`<div class="tl err">${esc(l) || '&nbsp;'}</div>`);
  if (c.name === 'bash') lines.push(`<div class="tl code ${r.code ? 'bad' : 'ok'}">[退出码 ${r.code} · ${r.ms} ms]</div>`);
  return lines;
}

function termHTML(k, { live = null } = {}) {
  const prev = termLines(k, live ? live.ci : Infinity);
  let liveHTML = '';
  if (live) liveHTML = `<div id="termLive" data-k="${k}" data-ci="${live.ci}" data-mode="${live.mode}"></div>`;
  return `<div class="panel term-panel"><h3>终端 <small>临时任务目录 ~/${task.id} · 受限命令 · 真实执行</small></h3><div class="term" id="term">${prev.join('')}${liveHTML}<div class="tl"><span class="ps">agent@task:~/${task.id}$</span> <span class="cursor"></span></div></div></div>`;
}

function diffHTML(changes) {
  if (!changes.length) return '<p class="note">这条命令没有改动任何文件。</p>';
  return changes.map((ch) => `<div class="diff"><div class="dh">${ch.status === 'added' ? '新文件' : ch.status === 'deleted' ? '删除' : '修改'} <b>${esc(ch.file)}</b></div><pre>${ch.diff.slice(2).map((l) => `<span class="${l.startsWith('+') ? 'add' : l.startsWith('-') ? 'del' : l.startsWith('@@') ? 'hunk' : ''}">${esc(l)}</span>`).join('\n')}</pre></div>`).join('');
}

function checkHTML(check) {
  return `<div class="panel"><h3>自动复核 <small>${check.passed ? '通过' : '未通过，错误会反馈给模型继续修复'}</small></h3>
    <p class="note">任务测试：退出码 <b>${check.code}</b>；独立用例：<b>${check.oracle?.passed ? '通过' : '未通过'}</b>${check.mutation ? `；错误实现检出：<b>${check.mutation.passed ? '通过' : '未通过'}</b>` : ''}。</p>
    <pre class="obs">${esc((check.stdout || '') + (check.stderr || '') + (check.oracle?.stderr || '') + (!check.mutation?.passed ? check.mutation?.stderr || '' : ''))}</pre></div>`;
}

function ctxHTML(k) {
  const T = task.turns, tot = T[k].ctx;
  const scale = Math.max(...T.slice(0, k + 1).map((t) => t.ctx));
  return `<div class="panel"><h3>上下文 <small>这一轮开始时模型读入 ${tot.toLocaleString()} 个词元（上限 40960）</small></h3>
  <div class="ctxbar">${T.slice(0, k + 1).map((t, j) => `<i style="flex:1;background:${j === k ? 'var(--acc)' : 'var(--violet)'};opacity:${Math.max(.2, t.ctx / scale).toFixed(2)}" title="第 ${j + 1} 轮：${t.ctx} 词元"></i>`).join('')}</div>
  <p class="note">每格是一轮开始时聊天模板实际送入模型的词元数，越亮表示越长。模型收到任务、之前的工具结果，以及验收失败时的反馈；模板可能省略旧的思考内容，所以上下文长度不一定每轮都增加。</p></div>`;
}

function sampHTML(turn, i, st) {
  const t = turn.tokens[i];
  const T = 0.6;
  // 温度：p^(1/T) 再归一化（只在前 TOPN 名里示意；真实计算在全部 151936 个词元上做，候选池大小是真实的）
  const raw = t.top;
  const pw = raw.map(([s, p]) => [s, p ** (1 / T)]);
  const z = pw.reduce((a, b) => a + b[1], 0) + 1e-12;
  const tempd = pw.map(([s, v]) => [s, v / z]);
  let cum = 0;
  const cut = tempd.map(([s, v]) => { const keep = cum < 0.95; cum += v; return [s, v, keep]; });
  const kept = cut.filter((x) => x[2]);
  const kz = kept.reduce((a, b) => a + b[1], 0);
  const pool = t.samplePool?.length
    ? t.samplePool.map(([s, , q]) => [s, q, true])
    : cut.map(([s, v, keep]) => [s, keep ? v / kz : 0, keep]);
  const bars = (items, win, extra = () => '') => `<div class="bars">${items.map(([s, v, keep = true]) => `<div class="bar ${s === win ? 'win' : ''} ${keep ? '' : 'cut'}"><span class="t">${esc(vis(s))}</span><span class="b"><i style="width:${(v * 100).toFixed(1)}%"></i></span><span class="v">${fmtP(v)}${extra(s)}</span></div>`).join('')}</div>`;
  const body = {
    cands: `<p class="note">模型对词表里全部 <b>151936</b> 个词元各给出一个概率（下面是前 8 名，T = 1）。这一步的熵 H = ${t.H.toFixed(2)}：${t.H < 0.3 ? '几乎没有犹豫' : t.H < 1.5 ? '有几个候选在竞争' : '相当犹豫'}。</p>${bars(raw, t.s)}`,
    temp: `<p class="note">除以温度 0.6（等价于 p<sup>1/0.6</sup> 再归一化）让分布更尖，然后只保留前 20 名。图中仅展示原始概率前 8 名，比例在这 8 名内重新计算。</p>${bars(tempd, t.s)}`,
    topp: `<p class="note">从大到小累加，累计概率刚超过 0.95 就停。真实候选池剩 <b class="amber">${t.pool}</b> 个词元；下方是候选池内归一化后的实际抽样概率。</p>${bars(pool, t.s)}`,
    draw: `<p class="note">在候选池里按概率抽签：随机数 u = <b class="amber">${t.u.toFixed(4)}</b>，沿累计概率走，落在“${esc(vis(t.s))}”的区间里（它在池中的概率 ${fmtP(t.q)}）。</p>${drawHTML(pool, t)}`,
  }[st];
  return `<div class="panel"><h3>词元 “${esc(vis(t.s))}” 是怎么抽出来的 <small>id ${t.id} · 原始概率 ${fmtP(t.p)}</small></h3>
  <div class="flow">${SAMPLE.map(([k2, l], j) => `${j ? '<span class="arr">→</span>' : ''}<div class="node ${k2 === st ? 'on' : SAMPLE.findIndex((x) => x[0] === st) > j ? 'done' : ''}">${l}</div>`).join('')}</div>${body}</div>`;
}

function drawHTML(pool, t) {
  let acc = 0;
  const segs = pool.filter((x) => x[2]).map(([s, v]) => { const a = acc; acc += v; return { s, a, v }; });
  return `<div class="draw">${segs.map((g) => `<i style="left:${g.a * 100}%;width:${g.v * 100}%" class="${g.s === t.s ? 'win' : ''}" title="${esc(vis(g.s))} ${fmtP(g.v)}"><span>${esc(vis(g.s))}</span></i>`).join('')}<b style="left:${t.u * 100}%"></b></div>`;
}

// ---------------------------------------------------------------- 渲染

function render(n, path, tree) {
  const k = path[0].k;
  const turn = task.turns[k];
  sideRender(k);
  const segs = segment(turn);
  const head = `<div class="panel turnhead"><div class="phases">${phases(k).map((p) => `<span class="ph ${path[1] && path[1].t === p.t && path[1].ci === p.ci && (path[1].idx?.[0] ?? -1) === (p.idx?.[0] ?? -1) ? 'on' : ''}">${esc(p.label)}</span>`).join('<span class="arr">→</span>')}</div></div>`;
  let body = '';
  const segOf = (t) => segs.find((g) => g.kind === t);
  switch (n.t) {
    case 'turn': {
      const th = segOf('think'), an = segOf('answer');
      body = `${th ? `<div class="panel"><h3>思考 <small>${th.idx.length} 个词元</small></h3><div class="think clamp">${segHTML(turn, th)}</div></div>` : ''}
        ${an ? `<div class="panel"><h3>${turn.calls.length ? '说明' : turn.autoCheck?.passed === false ? '尝试回答' : '最终回答'}</h3><div class="answer">${esc(clean(textOf(turn, an.idx)))}</div></div>` : ''}
        ${turn.calls.map((c) => `<div class="panel"><h3>调用工具</h3>${callPretty(c)}</div>`).join('')}
        ${termHTML(k)}${turn.calls.map((c) => (c.changes.length ? `<div class="panel"><h3>文件变化</h3>${diffHTML(c.changes)}</div>` : '')).join('')}${turn.autoCheck ? checkHTML(turn.autoCheck) : ''}`;
      break;
    }
    case 'ctx': body = ctxHTML(k); break;
    case 'check': body = checkHTML(turn.autoCheck); break;
    case 'think': case 'answer': body = `<div class="panel"><h3>${n.label} <small>${n.idx.length} 个词元 · 下划线越亮 = 模型越确定</small></h3><div class="${n.t}">${segHTML(turn, { kind: n.t, idx: n.idx })}</div></div>`; break;
    case 'sent': {
      const g = segs.find((x) => x.idx.includes(n.idx[0]));
      body = `<div class="panel"><h3>${g.kind === 'think' ? '思考' : '回答'} · 第 ${n.j + 1} 句 <small>${n.idx.length} 个词元</small></h3><div class="${g.kind} stream">${segHTML(turn, g, { curSent: n.idx, reveal: true })}</div></div>${tokStrip(turn, n.idx, null)}`;
      break;
    }
    case 'call': body = `<div class="panel"><h3>调用工具 <small>模型按 Qwen3 的格式写出 &lt;tool_call&gt;{json}&lt;/tool_call&gt;，程序逐条解析执行</small></h3><div class="raw stream">${tokSpan(turn, n.idx, null, true)}</div>${callPretty(turn.calls[n.ci])}</div>${tokStrip(turn, n.idx, null)}`; break;
    case 'tok': {
      const g = segs.find((x) => x.idx.includes(n.i));
      const sen = g.kind === 'call' ? g.idx : sentences(turn, g.idx).find((c) => c.includes(n.i));
      body = `<div class="panel"><h3>${g.kind === 'think' ? '思考' : g.kind === 'call' ? '工具调用' : '回答'}</h3><div class="${g.kind === 'call' ? 'raw' : g.kind}">${tokSpan(turn, sen, n.i)}</div></div>${tokStrip(turn, sen, n.i)}${sampHTML(turn, n.i, 'cands')}`;
      break;
    }
    case 'samp': body = `${tokStrip(turn, [n.i], n.i)}${sampHTML(turn, n.i, n.st)}`; break;
    case 'exec': body = `<div class="panel"><h3>调用工具</h3>${callPretty(turn.calls[n.ci])}</div>${termHTML(k, { live: { ci: n.ci, mode: 'all' } })}${turn.calls[n.ci].changes.length ? `<div class="panel"><h3>文件变化</h3>${diffHTML(turn.calls[n.ci].changes)}</div>` : ''}`; break;
    case 'cmd': body = `${termHTML(k, { live: { ci: n.ci, mode: 'cmd' } })}<div class="panel"><h3>程序做了什么</h3><p class="note">${turn.calls[n.ci].name === 'bash' ? `只接受 ls、cat 文件和指定的测试脚本，在临时任务目录里运行（30 秒超时）；这不是操作系统隔离沙箱。` : '把 content 写进临时任务目录里的文件（覆盖）；若模型把换行双重转义且原代码无法解析，录制器会尝试修正。'}</p>${callPretty(turn.calls[n.ci])}</div>`; break;
    case 'out': body = termHTML(k, { live: { ci: n.ci, mode: 'out' } }); break;
    case 'diff': body = `${termHTML(k)}<div class="panel"><h3>文件变化 <small>执行前后对比整个项目目录</small></h3>${diffHTML(turn.calls[n.ci].changes)}</div>`; break;
    case 'obs': {
      const c = turn.calls[n.ci];
      const next = task.turns[k + 1];
      body = `<div class="panel"><h3>结果写回上下文 <small>作为 tool 消息追加到对话末尾</small></h3><pre class="obs">&lt;|im_start|&gt;tool\n${esc(c.observation)}\n&lt;|im_end|&gt;</pre>
      <p class="note">${next ? `下一轮模型会读入 ${next.ctx.toLocaleString()} 个词元（比这一轮多 ${(next.ctx - turn.ctx).toLocaleString()} 个），看到这个结果之后决定下一步。` : '录制到这里结束。'}</p></div>`;
      break;
    }
    default: break;
  }
  view().innerHTML = head + body;
  trace?.update(traceSpec(tree));
  view().querySelectorAll('.tk[data-i]').forEach((el) => el.addEventListener('click', () => {
    const i = Number(el.dataset.i);
    tree.pause();
    tree.seekWhere((x) => x.t === 'tok' && x.i === i);
  }));
  liveStart = performance.now();
  frame(n, tree.p);
}

// 词元条：一句话里每个词元的概率
function tokStrip(turn, idx, cur) {
  return `<div class="panel"><h3>逐个词元 <small>点一个看它是怎么抽出来的</small></h3><div class="tstrip">${idx.map((i) => {
    const t = turn.tokens[i];
    return `<span class="ts tk ${i === cur ? 'cur' : ''}" data-i="${i}"><b>${esc(vis(t.s))}</b><small>${fmtP(t.p)}</small><i style="height:${Math.max(2, t.p * 26)}px"></i></span>`;
  }).join('')}</div></div>`;
}

// 动画：逐字敲命令、逐行出输出、逐词元出字
let liveStart = 0;
function frame(n, p) {
  if (!task) return;
  trace?.frame(p);
  const live = $('#termLive');
  if (live) {
    const k = Number(live.dataset.k), c = task.turns[k].calls[Number(live.dataset.ci)], mode = live.dataset.mode;
    const cmd = c.name === 'bash' ? c.args.command || '' : `${c.args.path || ''}`;
    const pc = mode === 'cmd' ? p : mode === 'all' ? Math.min(1, p * 2.2) : 1;
    const po = mode === 'out' ? p : mode === 'all' ? Math.max(0, p * 2 - 1) : 0;
    const typed = cmd.slice(0, Math.ceil(cmd.length * Math.min(1, pc * 1.15)));
    const key = `${mode}|${typed.length}|${Math.floor(po * 200)}`;
    if (live.dataset.key !== key) {
      live.dataset.key = key;
      const ol = outLines(c);
      const nOut = mode === 'cmd' ? 0 : Math.ceil(ol.length * Math.min(1, po * 1.2));
      const ps = c.name === 'bash' ? `<span class="ps">agent@task:~/${task.id}$</span> ` : '<span class="ps">✎ write_file</span> ';
      live.innerHTML = `<div class="tl cmd">${ps}${esc(typed)}${typed.length < cmd.length ? '<span class="cursor"></span>' : ''}</div>${ol.slice(0, nOut).join('')}`;
      const term = $('#term');
      if (term) term.scrollTop = term.scrollHeight;
    }
  }
  const rv = view().querySelectorAll('.stream .tk.rv, .stream.raw .tk');
  if (rv.length) {
    const show = Math.ceil(rv.length * Math.min(1, p * 1.2 + 0.02));
    rv.forEach((el, j) => el.classList.toggle('hid', j >= show));
  }
}

// ---------------------------------------------------------------- 左侧：对话记录

let sideK = -1;
function sideRender(k) {
  if (sideK === k) return;
  sideK = k;
  const items = task.turns.map((t, j) => {
    const c = t.calls[0];
    const th = clean(t.think || '').slice(0, 80);
    return `<button type="button" class="turn-item ${j === k ? 'on' : ''}" data-k="${j}"><span class="tn">第 ${j + 1} 轮 · ${t.tokens.length} 词元</span>${th ? `<span class="th">💭 ${esc(th)}…</span>` : ''}${c ? `<code>${esc(c.name === 'bash' ? `$ ${c.args.command}` : `✎ ${c.args.path}`).slice(0, 90)}</code><span class="rc ${c.result.code ? 'bad' : 'ok'}">退出码 ${c.result.code}</span>` : `<span class="fin">${esc(clean(t.content).slice(0, 90))}</span>${t.autoCheck ? `<span class="rc ${t.autoCheck.passed ? 'ok' : 'bad'}">自动复核${t.autoCheck.passed ? '通过' : '未通过'}</span>` : ''}`}</button>`;
  }).join('');
  $('#sideBody').innerHTML = `<div class="blk"><h4>任务</h4>${esc(task.prompt)}</div><div><h4>${task.turns.length} 轮 · 点一下跳过去</h4><div class="turns">${items}</div></div>
  <div class="blk"><h4>项目文件（开始时）</h4>${Object.keys(task.initialFiles).map((f) => `<code>${esc(f)}</code>`).join(' ')}</div>`;
  $('#sideBody').querySelectorAll('.turn-item').forEach((b) => b.addEventListener('click', () => { lab.tree.pause(); lab.tree.seekRoot(Number(b.dataset.k), Math.min(lab.tree.depth, 2)); }));
  $('#sideBody').querySelector('.turn-item.on')?.scrollIntoView({ block: 'nearest' });
}

// ---------------------------------------------------------------- 讲解

const CODE = [
  'messages = [system, user(task)]',
  'while True:',
  '    ids = chat_template(messages, tools)   # 整段对话',
  '    out = []',
  '    while tok != "<|im_end|>":             # 逐词元生成',
  '        p = softmax(model(ids + out))      # 151936 个概率',
  '        q = top_p(top_k(p ** (1/0.6), 20), 0.95)',
  '        tok = sample(q, u=rand())',
  '        out.append(tok)',
  '    think, reply, call = parse(out)        # <think> / <tool_call>',
  '    messages.append(assistant(out))',
  '    if not calls:                           # 模型尝试回答',
  '        check = verify(project)            # 独立复核',
  '        if check.ok: break',
  '        messages.append(user(check.error)) # 错误反馈后继续',
  '    for call in calls:                      # 逐条执行',
  '        result = run_limited(call)',
  '        messages.append(tool(result))',
];
const LINES = { turn: [2], ctx: [3], think: [5, 6, 7, 8, 9], answer: [5, 6, 7, 8, 9], sent: [5, 9], call: [10], tok: [6, 7, 8, 9], cands: [6], temp: [7], topp: [7], draw: [8, 9], check: [13, 14, 15], exec: [17], cmd: [17], out: [17], diff: [17], obs: [18] };

function explain(n, path) {
  const k = path[0].k, turn = task.turns[k];
  const c = turn.calls[0];
  const w = ['— 这一轮', ['轮次', `${k + 1} / ${task.turns.length}`], ['上下文', `${turn.ctx.toLocaleString()} 词元`], ['生成', `${turn.tokens.length} 词元`], ['速度', `${(turn.tokens.length / turn.genSeconds).toFixed(1)} 词元/秒`]];
  if (c) w.push(['工具', c.name], ['退出码', String(c.result.code)]);
  const ti = n.t === 'tok' || n.t === 'samp' ? n.i : null;
  if (ti != null) {
    const t = turn.tokens[ti];
    w.push('— 当前词元', ['tok', vis(t.s)], ['id', String(t.id)], ['p', fmtP(t.p)], ['熵 H', t.H.toFixed(3)], ['候选池', `${t.pool} 个`], ['u', t.u.toFixed(4)], ['池中概率', fmtP(t.q)]);
  }
  const key = n.t === 'samp' ? n.st : n.t;
  const shape = { ctx: `ids [${turn.ctx}] → Qwen3-1.7B（28 层）`, tok: 'logits [151936] → softmax → 采样 → 1 个词元', cands: 'softmax(logits) [151936]', temp: 'logits / 0.6 → 取前 20', topp: `累计到 0.95 → 候选池 ${ti != null ? turn.tokens[ti].pool : '?'} 个`, draw: 'u ~ U(0,1) → 落在哪个区间' }[key];
  return { lines: LINES[key] || [2], shape: shape ? esc(shape) : '', html: EXPLAIN[key]?.(n, turn) || '', watch: w };
}

const EXPLAIN = {
  turn: (n, t) => (t.calls.length ? '一轮 = 模型读入整段对话 → 思考并写出工具调用 → 程序逐条执行 → 把结果接到对话末尾。按 <b>＋</b> 拆开这一轮。' : t.autoCheck?.passed === false ? '模型尝试结束，但独立复核未通过。录制器会把失败信息反馈给模型，让它继续修复。' : '模型给出回答，随后复核通过。'),
  ctx: () => '每一轮开始，程序用聊天模板把系统提示、工具定义、任务和之前所有的往来拼成一串词元，整串送进模型。',
  think: () => 'Qwen3 的思考模式：先在 &lt;think&gt;…&lt;/think&gt; 里自言自语地推理，再给出行动。思考也是一个词元一个词元抽出来的。',
  answer: () => '思考结束后写给用户看的话。', sent: () => '逐句看：每个词元都经过同样的“算概率 → 截断 → 抽签”。按 <b>＋</b> 看每个词元。',
  call: () => '工具调用就是模型写出的一段特殊格式的文本。它并不“知道”有终端，只是学会了写这种格式；真正执行的是外面的程序。',
  tok: () => '这一个词元：模型先给出全词表的概率，再按温度、top-k、top-p 截断后抽签。按 <b>＋</b> 分四步看。',
  cands: () => '原始概率：模型最后一层的输出乘上输出矩阵，再 softmax。', temp: () => '温度 < 1 让分布更集中。', topp: () => 'top-p（核采样）：只在累计概率 95% 的最小集合里抽。', draw: () => '最后用一个均匀随机数决定抽中谁。种子固定，所以录像可以复现。',
  check: () => '录制器运行任务测试和自己持有的独立用例。复核失败就把错误送回模型，接着录下一轮。',
  exec: () => '程序解析工具调用后，在临时任务目录中按白名单执行。下面的终端输出、退出码和耗时都来自实际运行。', cmd: () => '执行的命令。', out: () => '命令的标准输出 / 标准错误和退出码，原样收集。', diff: () => '执行前后把项目目录里的每个文件都比较一遍，这就是这条命令造成的改动。',
  obs: () => '执行结果作为 tool 消息接到对话末尾。下一轮模型就能“看到”它。',
};

// ---------------------------------------------------------------- 启动

async function loadTask(id) {
  if (!cache.has(id)) cache.set(id, await fetchJSON(`data/${id}.json`));
  return cache.get(id);
}

function makeLab() {
  const roots = task.turns.map((t, k) => ({ t: 'turn', k, label: turnLabel(t, k), crumb: `第 ${k + 1} 轮`, dur: 3.2, kids: () => phases(k) }));
  if (!trace) trace = new TraceStage($('#stage'), { onPick: stagePick });
  if (!lab) {
    lab = new Lab({
      roots, maxDepth: 5,
      depthNames: ['', '每一轮', '一轮之内', '句 / 词元 / 终端', '每个词元', '采样'],
      code: CODE, explain, render, frame: (n, p) => frame(n, p),
      posText: (t) => `第 <b>${t.r + 1}</b>/${t.roots.length} 轮<br>步骤 <b>${t.i + 1}</b>/${t.list.length}`,
      onExit: () => document.body.classList.replace('mode-inspect', 'mode-pick'),
    });
    window.__lab = lab;
  } else {
    lab.tree.roots = roots;
    lab.tree.seekRoot(0, 1);
  }
}

async function enter(id) {
  task = await loadTask(id);
  sideK = -1;
  $('#sideTitle').textContent = task.title;
  document.body.classList.replace('mode-pick', 'mode-inspect');
  makeLab();
  history.replaceState(null, '', `?task=${id}`);
}

async function boot() {
  try {
    manifest = await fetchJSON('data/manifest.json');
  } catch (e) {
    $('#lead').textContent = `录制记录载入失败：${e.message}`;
    return;
  }
  const M = manifest.model;
  $('#lead').innerHTML = `下面是<b>真实的录像</b>：阿里开源的 <b>${M.name}</b>（${(M.params / 1e9).toFixed(1)}B 参数，在本机 RTX 5060 上运行）当编程智能体，开着思考模式，在一个沙箱目录里<b>真的执行</b>它写出的每一条命令。模型能看到的只有文字：它写出一段工具调用，程序去执行，再把终端输出贴回给它。`;
  $('#spec').innerHTML = `<span>模型 <b>${M.name}</b></span><span>层数 <b>${M.layers}</b></span><span>词表 <b>${M.vocab.toLocaleString()}</b></span><span>采样 <b>T=${manifest.sampling.temperature} · top-k ${manifest.sampling.top_k} · top-p ${manifest.sampling.top_p}</b></span><span>工具 <b>${manifest.tools.map((t) => t.function.name).join(' / ')}</b></span>`;
  $('#cards').innerHTML = manifest.tasks.map((t) => `<button type="button" class="card" data-id="${t.id}"><span class="tag">${t.turns} 轮 · ${t.commands} 条命令 · ${t.tokens.toLocaleString()} 词元${t.verified === false ? ' · 最终复核未通过' : t.verified === true ? ' · 最终复核通过' : ''}</span><h3>${esc(t.title)}</h3><p>${esc(t.prompt)}</p><span class="go">看它怎么做 →</span></button>`).join('');
  $('#credit').innerHTML = `录制脚本 tools/record_agent.py。命令在本机临时任务目录中按白名单执行，输出、退出码、耗时、文件 diff 均为真实结果；采样的随机数种子固定，可以复现。`;
  $('#cards').addEventListener('click', (e) => { const c = e.target.closest('.card'); if (c) enter(c.dataset.id); });
  $('#btnSide').addEventListener('click', () => document.body.classList.add('side-open'));
  $('#btnSideClose').addEventListener('click', () => document.body.classList.remove('side-open'));
  $('#btnBack').addEventListener('click', () => { lab?.tree.pause(); document.body.classList.remove('side-open'); document.body.classList.replace('mode-inspect', 'mode-pick'); history.replaceState(null, '', './'); });
  const want = new URLSearchParams(location.search).get('task');
  if (want && manifest.tasks.some((t) => t.id === want)) enter(want);
}

boot();
