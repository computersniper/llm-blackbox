// 调试器里的“代码 / 这一步 / 变量”三块内容：伪代码就是 harness 的 agent 循环，数值全部来自录制。
import { esc, fmtPct, fmtNum, tokPlain } from '../../js/ui.js';
import { DEPTH_NAMES } from './timeline.js';
import { isDecision } from './data.js';
import { splitExit, normPath } from './screen.js';
import { callArg } from './chat.js';
import { KIND_NAME } from './views.js';
import { isEn, L as tr } from '../../js/i18n.js';

const K = (s) => `<span class="kw">${s}</span>`;
const F = (s) => `<span class="fn">${s}</span>`;
const C = (s) => `<span class="cm"># ${s}</span>`;

export const CODE = [
  `${K('def')} ${F('run_agent')}(task):`,
  `    messages = [${F('system')}(SYSTEM, TOOLS), ${F('user')}(task)]`,
  `    ${K('while')} True:`,
  `        prompt = ${F('apply_chat_template')}(messages)  ${C(tr('整段历史', 'whole history'))}`,
  `        reuse = kv_cache.${F('common_prefix')}(prompt)`,
  `        kv_cache.${F('prefill')}(prompt[reuse:])     ${C(tr('只算新增', 'only the new part'))}`,
  `        out = ''`,
  `        ${K('while')} (tok := ${F('sample')}(${F('model')}(…))) != <span class="nu">'&lt;|im_end|&gt;'</span>:`,
  `            out += tok                     ${C(tr('一次一个词元', 'one token at a time'))}`,
  `        calls = ${F('parse_tool_calls')}(out)      ${C('&lt;tool_call&gt;{…}')}`,
  `        messages.${F('append')}(${F('assistant')}(out))`,
  `        ${K('if')} ${K('not')} calls:`,
  `            ${K('return')} out                     ${C(tr('不再调用工具 = 完成', 'no more tool calls = done'))}`,
  `        ${K('for')} call ${K('in')} calls:`,
  `            result = sandbox.${F('run')}(call)     ${C(tr('真正动手的是这里', 'this is where things actually happen'))}`,
  `            messages.${F('append')}(${F('tool')}(result))`,
];

export function renderCode(el) {
  el.innerHTML = CODE.map((l, i) => `<span class="ln" data-n="${i + 1}"><span class="bp"></span><span class="no">${i + 1}</span>${l}</span>`).join('');
}

export function linesFor(s, depth) {
  if (depth <= 1) return s.k1 === 'think' ? [8, 9] : s.k1 === 'answer' ? [12, 13] : [15, 16];
  switch (s.p) {
    case 'prompt': return s.sub === 'kv' ? [5, 6] : [4];
    case 'gen': return s.f ? [8] : s.j !== undefined ? [8, 9] : [7, 8, 9];
    case 'parse': return [10, 11, 12];
    case 'exec': return [14, 15];
    case 'append': return [16];
    case 'done': return [12, 13];
  }
  return [];
}

const P = (p) => `<em>${fmtPct(p)}</em>`;
const code = (s, n = 90) => `<code>${esc(String(s).length > n ? String(s).slice(0, n) + '…' : s)}</code>`;

export function stepLabel(s, R) {
  const T = R.turns[s.t];
  if (isEn) {
    if (s.p === undefined) return s.k1 === 'think' ? 'Say a sentence' : s.k1 === 'answer' ? 'Final answer' : `Act: ${T.calls[s.c].name}`;
    if (s.p === 'prompt') return s.sub === 'kv' ? 'Reuse the KV cache' : s.sub === 'ctx' ? 'Build the context' : 'Assemble the context';
    if (s.p === 'gen') {
      if (s.j === undefined) return 'Model generates';
      const tk = T.gen.toks[s.j];
      const w = tk.end ? '<|im_end|>' : tk.s === '' ? 'part of a char' : tokPlain(tk.s);
      return `${s.f === 'fwd' ? 'Forward pass' : s.f === 'pick' ? 'Draw' : 'Token'} #${s.j + 1} “${w}”`;
    }
    if (s.p === 'parse') return 'Parse tool calls';
    if (s.p === 'exec') return `Sandbox runs ${T.calls[s.c].name}`;
    if (s.p === 'append') return 'Result back into the chat';
    return 'Done';
  }
  if (s.p === undefined) return s.k1 === 'think' ? '说一句话' : s.k1 === 'answer' ? '最终回答' : `动手：${T.calls[s.c].name}`;
  if (s.p === 'prompt') return s.sub === 'kv' ? '复用 KV 缓存' : s.sub === 'ctx' ? '拼出上下文' : '组装上下文';
  if (s.p === 'gen') {
    if (s.j === undefined) return '模型生成';
    const tk = T.gen.toks[s.j];
    const w = tk.end ? '<|im_end|>' : tk.s === '' ? '半个字' : tokPlain(tk.s);
    return `${s.f === 'fwd' ? '前向计算' : s.f === 'pick' ? '抽签' : '词元'} #${s.j + 1}「${w}」`;
  }
  if (s.p === 'parse') return '解析工具调用';
  if (s.p === 'exec') return `沙箱执行 ${T.calls[s.c].name}`;
  if (s.p === 'append') return '结果接回对话';
  return '结束';
}

export function crumbs(depth, s, R) {
  const T = R.turns[s.t];
  if (isEn) {
    const out = [{ d: 1, label: `Turn ${s.t + 1}` }];
    if (depth >= 2) out.push({ d: 2, label: s.p === 'exec' || s.p === 'append' ? T.calls[s.c].name : { prompt: 'Assemble context', gen: 'Model generates', parse: 'Parse', done: 'Done' }[s.p] });
    if (depth >= 3) out.push({ d: 3, label: s.sub === 'kv' ? 'KV cache' : s.sub === 'ctx' ? 'Whole history' : 'Context' });
    if (depth >= 4) out.push({ d: 4, label: s.j !== undefined ? `Token #${s.j + 1}` : 'Tokens' });
    if (depth >= 5) out.push({ d: 5, label: s.f === 'pick' ? 'Draw' : s.f === 'fwd' ? `${R.M.layers}-layer forward` : 'Inside the model' });
    return out;
  }
  const out = [{ d: 1, label: `第 ${s.t + 1} 圈` }];
  if (depth >= 2) out.push({ d: 2, label: s.p === 'exec' || s.p === 'append' ? T.calls[s.c].name : { prompt: '组装上下文', gen: '模型生成', parse: '解析', done: '结束' }[s.p] });
  if (depth >= 3) out.push({ d: 3, label: s.sub === 'kv' ? 'KV 缓存' : s.sub === 'ctx' ? '整段历史' : '上下文' });
  if (depth >= 4) out.push({ d: 4, label: s.j !== undefined ? `词元 #${s.j + 1}` : '词元' });
  if (depth >= 5) out.push({ d: 5, label: s.f === 'pick' ? '抽签' : s.f === 'fwd' ? `${R.M.layers} 层前向` : '模型内部' });
  return out;
}

export function explain(s, depth, R) {
  if (isEn) return explainEn(s, depth, R);
  const T = R.turns[s.t];
  const M = R.M;
  const c = s.c != null ? T.calls[s.c] : null;
  // ---------- D1 屏幕
  if (s.p === undefined) {
    if (s.k1 === 'think') return `模型先用一句话说明打算。<br>这句话和后面的工具调用，是它<b>同一次</b>输出里的文字：屏幕上的一切，都来自它写出来的 <b>${T.gen.n}</b> 个词元。`;
    if (s.k1 === 'answer') return `模型这一圈<b>没有</b>写 &lt;tool_call&gt;，harness 就把这段话当成最终回答，循环结束。<br>一共 <b>${R.T}</b> 圈，整段对话 <b>${fmtNum(R.maxCtx)}</b> 个词元。`;
    return actExplain(c, T);
  }
  switch (s.p) {
    case 'prompt':
      if (s.sub === 'kv') {
        const prev = s.t > 0 ? R.turns[s.t - 1] : null;
        return s.t === 0
          ? `第一圈没有缓存可用：<b>${fmtNum(T.ctx.n)}</b> 个词元全部要算一遍（预填充），实测 <b>${T.ms.prefill} ms</b>。<br>这一步是并行的，所以虽然词元多，却很快。`
          : `上一圈算过的词元，它们在每一层的 K、V 还存在缓存里。新的上下文和缓存逐个比对：前 <b>${fmtNum(T.kv.reused)}</b> 个一模一样，直接复用；只需要新算 <b>${fmtNum(T.kv.computed)}</b> 个（主要是刚接回来的工具结果），实测 <b>${T.ms.prefill} ms</b>。<br><span class="dimmed">如果不复用，要把 ${fmtNum(T.ctx.n)} 个词元从头再算一遍。${prev && T.kv.reused < prev.ctx.n + prev.gen.n ? `上一圈结尾的 &lt;|im_end|&gt; 当时没有喂进模型，所以从那里开始重算。` : ''}</span>`;
      }
      return s.t === 0
        ? `harness 用聊天模板把系统提示、<b>${R.tools.length}</b> 个工具的定义（JSON 格式，连同模板里的调用说明一共 <b>${T.ctx.segs.find((g) => g.k === 'tools')?.n}</b> 个词元）和你的任务拼成一整串，一共 <b>${fmtNum(T.ctx.n)}</b> 个词元。<br>模型看到的不是“对话”，只是一串词元。`
        : `第 ${s.t + 1} 圈：<b>整段历史</b>重新拼一遍喂给模型，包括之前每一圈它自己写的话和每一次工具结果。<br>上下文从 <b>${fmtNum(R.turns[s.t - 1].ctx.n)}</b> 涨到了 <b>${fmtNum(T.ctx.n)}</b> 个词元。模型自己什么都不记得，记忆全在这串文字里。`;
    case 'gen':
      if (s.j === undefined) return `模型开始往后写，一次一个词元，直到写出 &lt;|im_end|&gt;。<br>这一圈写了 <b>${T.gen.n}</b> 个词元，实测 <b>${(T.ms.decode / 1000).toFixed(2)} s</b>${T.calls.length ? `，其中 &lt;tool_call&gt; 那段是一个 JSON：${code(`${T.calls[0].name}(${callArg(T.calls[0])})`, 70)}` : '。这次没有工具调用。'}`;
      return tokExplain(s, T, M);
    case 'parse':
      return T.calls.length
        ? `harness 在输出里找 &lt;tool_call&gt;…&lt;/tool_call&gt;，用 <code>json.loads</code> 读出 <b>${T.calls.length}</b> 个调用：${T.calls.map((x) => code(x.name, 20)).join(' ')}。<br><b>模型本身没有执行任何东西</b>，它只是写了一段格式特定的文字。`
        : '输出里没有 &lt;tool_call&gt;。对 harness 来说，这就是最终回答。';
    case 'exec':
      return `${execExplain(c)}<br><span class="dimmed">沙箱（bubblewrap）：系统目录只读、只能写 /work、没有网络、清空环境变量、10 秒超时。像 Claude Code 这样的 agent，执行有风险的命令前还会先问你同不同意。</span>`;
    case 'append': {
      const r = splitExit(c.result);
      return `结果原样包进 &lt;tool_response&gt;，以 <b>user</b> 的身份接到对话末尾（<b>${c.resultTokens}</b> 个词元）${r.code != null && r.code !== 0 ? '，连同非零的退出码和报错信息' : ''}。<br>模型下一圈“看到”的，就是这段文字。${!c.info?.ok || (r.code != null && r.code !== 0) ? '它会不会自己发现问题、换个办法？往下看。' : ''}`;
    }
    case 'done':
      return `循环结束。harness 把最后这段话交给用户。<br>录制脚本又在沙箱里<b>独立检查</b>了一遍：${esc(R.check.detail)}。`;
  }
  return '';
}

// 英文讲解：和中文版一一对应，数字全部取自英文录制（data/en/）
function explainEn(s, depth, R) {
  const T = R.turns[s.t];
  const M = R.M;
  const c = s.c != null ? T.calls[s.c] : null;
  if (s.p === undefined) {
    if (s.k1 === 'think') return `The model first says in one sentence what it plans to do.<br>This sentence and the tool call after it are text from the <b>same</b> output: everything on screen comes from the <b>${T.gen.n}</b> tokens it wrote.`;
    if (s.k1 === 'answer') return `This turn the model wrote <b>no</b> &lt;tool_call&gt;, so the harness treats the text as the final answer and the loop ends.<br><b>${R.T}</b> turns in all; the whole conversation is <b>${fmtNum(R.maxCtx)}</b> tokens.`;
    return actExplainEn(c, T);
  }
  switch (s.p) {
    case 'prompt':
      if (s.sub === 'kv') {
        const prev = s.t > 0 ? R.turns[s.t - 1] : null;
        return s.t === 0
          ? `On the first turn there is no cache to use: all <b>${fmtNum(T.ctx.n)}</b> tokens must be computed (prefill), measured at <b>${T.ms.prefill} ms</b>.<br>This step runs in parallel, so it is fast even with many tokens.`
          : `The tokens computed last turn still have their K and V for every layer in the cache. The new context is compared with the cache token by token: the first <b>${fmtNum(T.kv.reused)}</b> are identical and reused directly; only <b>${fmtNum(T.kv.computed)}</b> need computing (mostly the tool result just appended), measured at <b>${T.ms.prefill} ms</b>.<br><span class="dimmed">Without reuse, all ${fmtNum(T.ctx.n)} tokens would be computed again from scratch.${prev && T.kv.reused < prev.ctx.n + prev.gen.n ? ` The &lt;|im_end|&gt; that ended last turn was never fed into the model, so recomputation starts there.` : ''}</span>`;
      }
      return s.t === 0
        ? `The harness uses the chat template to join the system prompt, the definitions of <b>${R.tools.length}</b> tools (JSON, which together with the template’s calling instructions take <b>${T.ctx.segs.find((g) => g.k === 'tools')?.n}</b> tokens) and your task into one long string: <b>${fmtNum(T.ctx.n)}</b> tokens.<br>What the model sees is not a “conversation”, just a run of tokens.`
        : `Turn ${s.t + 1}: the <b>whole history</b> is assembled again and fed to the model, including everything it wrote in earlier turns and every tool result.<br>The context grew from <b>${fmtNum(R.turns[s.t - 1].ctx.n)}</b> to <b>${fmtNum(T.ctx.n)}</b> tokens. The model itself remembers nothing; its memory is all in this text.`;
    case 'gen':
      if (s.j === undefined) return `The model starts writing, one token at a time, until it writes &lt;|im_end|&gt;.<br>This turn it wrote <b>${T.gen.n}</b> tokens, measured at <b>${(T.ms.decode / 1000).toFixed(2)} s</b>${T.calls.length ? `; the &lt;tool_call&gt; part is a piece of JSON: ${code(`${T.calls[0].name}(${callArg(T.calls[0])})`, 70)}` : '. No tool call this time.'}`;
      return tokExplainEn(s, T, M);
    case 'parse':
      return T.calls.length
        ? `The harness looks for &lt;tool_call&gt;…&lt;/tool_call&gt; in the output and reads <b>${T.calls.length}</b> call${T.calls.length === 1 ? '' : 's'} with <code>json.loads</code>: ${T.calls.map((x) => code(x.name, 20)).join(' ')}.<br><b>The model itself executed nothing</b>; it only wrote a piece of text in a particular format.`
        : 'There is no &lt;tool_call&gt; in the output. To the harness, this is the final answer.';
    case 'exec':
      return `${execExplainEn(c)}<br><span class="dimmed">Sandbox (bubblewrap): system directories read-only, only /work writable, no network, environment cleared, 10-second timeout. Agents like Claude Code also ask for your permission before running risky commands.</span>`;
    case 'append': {
      const r = splitExit(c.result);
      return `The result is wrapped as-is in &lt;tool_response&gt; and appended to the conversation as <b>user</b> (<b>${c.resultTokens}</b> tokens)${r.code != null && r.code !== 0 ? ', together with the non-zero exit code and the error message' : ''}.<br>This text is what the model “sees” next turn.${!c.info?.ok || (r.code != null && r.code !== 0) ? ' Will it notice the problem and try another way? Keep watching.' : ''}`;
    }
    case 'done':
      return `The loop ends. The harness hands this last text to the user.<br>The recording script then ran an <b>independent check</b> in the sandbox: ${esc(R.check.detail)}.`;
  }
  return '';
}

function actExplainEn(c, T) {
  const r = splitExit(c.result);
  if (c.name === 'bash') return `The model writes ${code(c.args.command, 80)} and the harness really runs it in the sandbox’s bash (measured ${c.info.ms} ms), exit code <b>${r.code}</b>.<br>${r.code === 0 ? 'The output goes back to the model as-is.' : 'The command failed: the error goes back to the model as-is too.'}`;
  if (c.name === 'read_file') return c.info.ok ? `Read ${code(normPath(c.args.path))}: the file’s whole content goes back to the model as the tool result, taking <b>${c.resultTokens}</b> tokens.` : `Read failed: ${esc(c.result)}`;
  if (c.name === 'edit_file') return c.info.ok ? `Replace a passage in ${code(normPath(c.args.path))} with new text. old_string must match the file exactly and occur only once.` : `Edit failed: ${esc(c.result)}`;
  if (c.name === 'write_file') return `${/^(已覆盖|Overwrote)/.test(c.result) ? 'Overwrite' : 'Create'} the file ${code(normPath(c.args.path))}; its content was written by the model token by token inside the JSON arguments.`;
  return esc(c.result);
}

function execExplainEn(c) {
  if (!c.ok) return `This tool call isn’t valid JSON, so the harness can’t run it; all it can do is report the error back to the model.`;
  if (c.name === 'bash') return `Run ${code(c.args.command, 80)} in the sandbox.`;
  return `${c.name}(${code(normPath(c.args.path), 40)}): the harness reads or writes the file in /work directly.`;
}

function tokExplainEn(s, T, M) {
  const tk = T.gen.toks[s.j];
  const w = tk.end ? '&lt;|im_end|&gt;' : tk.s === '' ? '(part of a character)' : esc(tokPlain(tk.s));
  if (s.f === 'fwd') return `To write token <b>${s.j + 1}</b>, the model sends it through all <b>${M.layers}</b> layers. The KV cache already holds <b>${fmtNum(T.ctx.n + s.j)}</b> tokens; this step computes only for this one new position.`;
  if (s.f === 'pick') return `The output head gives probabilities; drawing with temperature ${M.sampling.temperature}, top-k ${M.sampling.top_k}, top-p ${M.sampling.top_p} picks <b>${w}</b> (raw probability ${P(tk.p)}).`;
  const inCall = T.calls.find((c) => s.j >= c.j0 && s.j <= c.j1);
  let h = `Token <b>${s.j + 1}</b>: <b>${w}</b>, probability ${P(tk.p)}.`;
  if (isDecision(tk, s.j)) {
    const alt = tk.top.filter((x) => Math.abs(x[1] - tk.p) > 1e-9).slice(0, 2).map(([x, p]) => `${esc(tokPlain(x))} ${fmtPct(p)}`).join(', ');
    h += `<br><em>Key decision</em>: ${s.j === 0 ? 'talk first, or act right away?' : tk.end ? 'finished?' : 'the model really has a choice here.'} Other candidates: ${alt || '—'}.`;
  } else if (inCall) h += `<br>It is writing the JSON inside &lt;tool_call&gt;: the tool name and arguments are also “written” one token at a time like this.`;
  return h;
}

function actExplain(c, T) {
  const r = splitExit(c.result);
  if (c.name === 'bash') return `模型写下 ${code(c.args.command, 80)}，harness 在沙箱的 bash 里真的执行了它（实测 ${c.info.ms} ms），退出码 <b>${r.code}</b>。<br>${r.code === 0 ? '输出原样回给模型。' : '命令失败了：报错同样会原样回给模型。'}`;
  if (c.name === 'read_file') return c.info.ok ? `读取 ${code(normPath(c.args.path))}：文件的全部内容会作为工具结果回给模型，占 <b>${c.resultTokens}</b> 个词元。` : `读取失败：${esc(c.result)}`;
  if (c.name === 'edit_file') return c.info.ok ? `把 ${code(normPath(c.args.path))} 里的一段原文替换成新内容。old_string 必须和文件里一字不差，而且只能出现一次。` : `修改失败：${esc(c.result)}`;
  if (c.name === 'write_file') return `${c.result.startsWith('已覆盖') ? '覆盖' : '新建'}文件 ${code(normPath(c.args.path))}，内容是模型在 JSON 参数里一个词元一个词元写出来的。`;
  return esc(c.result);
}

function execExplain(c) {
  if (!c.ok) return `这个工具调用不是合法的 JSON，harness 没法执行，只能把错误回报给模型。`;
  if (c.name === 'bash') return `在沙箱里执行 ${code(c.args.command, 80)}。`;
  return `${c.name}(${code(normPath(c.args.path), 40)})：harness 直接在 /work 里读写文件。`;
}

function tokExplain(s, T, M) {
  const tk = T.gen.toks[s.j];
  const w = tk.end ? '&lt;|im_end|&gt;' : tk.s === '' ? '（半个字）' : esc(tokPlain(tk.s));
  if (s.f === 'fwd') return `为了写出第 <b>${s.j + 1}</b> 个词元，模型把它穿过全部 <b>${M.layers}</b> 层。KV 缓存里已经有 <b>${fmtNum(T.ctx.n + s.j)}</b> 个词元，这一步只为这一个新位置做计算。`;
  if (s.f === 'pick') return `输出头给出概率，按温度 ${M.sampling.temperature}、top-k ${M.sampling.top_k}、top-p ${M.sampling.top_p} 抽签，抽中 <b>${w}</b>（原始概率 ${P(tk.p)}）。`;
  const inCall = T.calls.find((c) => s.j >= c.j0 && s.j <= c.j1);
  let h = `第 <b>${s.j + 1}</b> 个词元：<b>${w}</b>，概率 ${P(tk.p)}。`;
  if (isDecision(tk, s.j)) {
    const alt = tk.top.filter((x) => Math.abs(x[1] - tk.p) > 1e-9).slice(0, 2).map(([x, p]) => `${esc(tokPlain(x))} ${fmtPct(p)}`).join('、');
    h += `<br><em>关键决策</em>：${s.j === 0 ? '先说话还是直接动手？' : tk.end ? '写完了吗？' : '这里模型真的有得选。'}别的候选：${alt || '—'}。`;
  } else if (inCall) h += `<br>它在写 &lt;tool_call&gt; 里的 JSON：工具名和参数也是这样一个词元一个词元“写”出来的。`;
  return h;
}

export function watch(s, depth, R) {
  const T = R.turns[s.t];
  const rows = [[tr('圈数', 'turn'), `${s.t + 1} / ${R.T}`]];
  const msgs = 2 + R.turns.slice(0, s.t).reduce((a, x) => a + 1 + x.calls.length, 0);
  rows.push(['len(messages)', String(msgs + (['parse', 'exec', 'append', 'done'].includes(s.p) || (s.p === undefined && s.k1 !== 'think') ? 1 : 0) + (s.p === 'append' ? s.c + 1 : s.p === 'exec' || (s.p === undefined && s.c != null) ? s.c : 0))]);
  rows.push(['prompt', tr(`${fmtNum(T.ctx.n)} 词元`, `${fmtNum(T.ctx.n)} tokens`)]);
  rows.push(['reuse', tr(`${fmtNum(T.kv.reused)}（新算 ${fmtNum(T.kv.computed)}）`, `${fmtNum(T.kv.reused)} (new ${fmtNum(T.kv.computed)})`)]);
  if (s.p === 'gen' && s.j !== undefined) {
    const tk = T.gen.toks[s.j];
    rows.push(['out', tr(`${s.j + (tk.end ? 0 : 1)} / ${T.gen.n} 词元`, `${s.j + (tk.end ? 0 : 1)} / ${T.gen.n} tokens`)]);
    rows.push(['tok', tk.end ? '<|im_end|>' : tk.s === '' ? tr('(半个字)', '(part of a char)') : JSON.stringify(tk.s)]);
    rows.push(['p(tok)', fmtPct(tk.p)]);
    rows.push([tr('候选池', 'pool'), tr(`${tk.pool} 个`, String(tk.pool))]);
  } else rows.push(['out', tr(`${T.gen.n} 词元 · ${(T.ms.decode / 1000).toFixed(2)} s`, `${T.gen.n} tokens · ${(T.ms.decode / 1000).toFixed(2)} s`)]);
  rows.push([tr('预填充', 'prefill'), `${T.ms.prefill} ms`]);
  const c = s.c != null ? T.calls[s.c] : null;
  if (c) {
    rows.push(['call', `${c.name}(…)`]);
    const r = splitExit(c.result);
    if (r.code != null) rows.push([tr('退出码', 'exit code'), String(r.code)]);
    rows.push([tr('工具耗时', 'tool time'), `${c.info.ms} ms`]);
    rows.push(['result', tr(`${c.resultTokens} 词元`, `${c.resultTokens} tokens`)]);
    if (c.diffs.length) rows.push([tr('改动', 'changed'), c.diffs.map((d) => `${d.path}${d.kind === 'new' ? tr(' (新)', ' (new)') : ''}`).join(', ')]);
  } else if (T.calls.length && ['parse'].includes(s.p)) rows.push(['calls', T.calls.map((x) => x.name).join(', ')]);
  else if (T.final) rows.push(['calls', tr('[]（没有）', '[] (none)')]);
  return rows;
}

export function renderWatch(el, rows) {
  el.innerHTML = `<span class="wh">${tr('变量', 'Variables')}</span>` + rows.map(([k, v]) => `<span class="k">${esc(k)}</span><span class="v" title="${esc(v)}">${esc(v)}</span>`).join('');
}

export function segKindName(k) { return KIND_NAME[k]; }
export { DEPTH_NAMES };
