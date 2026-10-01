// 读取 tools/agent/record.py 录下的真实 agent 轨迹，并预先算好回放要用的东西。

const cache = new Map();

// 数据文件预先 gzip 过（服务器只压缩 HTML），浏览器里用 DecompressionStream 解开；不支持时退回未压缩版本
async function fetchData(url) {
  if (typeof DecompressionStream !== 'undefined') {
    try {
      const r = await fetch(`${url}.gz`);
      if (r.ok && r.body) return await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).text();
    } catch { /* 退回未压缩 */ }
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} ${r.status}`);
  return r.text();
}

export async function loadManifest() {
  return JSON.parse(await fetchData('data/manifest.json'));
}

// 关键决策点：模型在这里真的有得选（第一名概率不到 75%），或者是每一轮的第一个词元（先说话还是直接调用工具）
export const isDecision = (tk, j) => j === 0 || tk.end || (tk.top[0] && tk.top[0][1] < 0.75);

export async function loadTask(id) {
  if (cache.has(id)) return cache.get(id);
  const R = JSON.parse(await fetchData(`data/${id}.json`));
  // 还原每一轮喂给模型的完整上下文：上一轮的上下文 + 上一轮生成的文字，取公共前缀，再接上新增部分
  let prev = '';
  for (const t of R.turns) {
    t.prompt = prev.slice(0, t.ctx.keep) + t.ctx.add;
    prev = t.prompt + t.gen.text;
    // 每个生成词元在生成文本里的字符区间（半个汉字的词元 s 为空串，区间长度为 0）
    let pos = 0;
    t.gen.toks.forEach((tk, j) => { tk.a = pos; pos += tk.s.length; tk.b = pos; tk.j = j; });
    t.final = t.calls.length === 0;
    for (const c of t.calls) {
      // 工具调用落在哪几个词元上
      c.j0 = t.gen.toks.findIndex((tk) => tk.b > c.a);
      c.j1 = t.gen.toks.findIndex((tk) => tk.b >= c.b);
    }
    t.sayEnd = t.calls.length ? t.calls[0].a : t.gen.text.length;
  }
  // 每一轮结束时的文件状态（用来在编辑器里显示“现在的文件”）
  let files = { ...R.files0 };
  R.filesAt = [];
  for (const t of R.turns) {
    for (const c of t.calls) {
      c.filesBefore = files;
      if (c.diffs.length) {
        files = { ...files };
        for (const d of c.diffs) { if (d.after == null && d.kind === 'del') delete files[d.path]; else files[d.path] = d.after; }
      }
      c.filesAfter = files;
    }
    R.filesAt.push(files);
  }
  R.T = R.turns.length;
  R.maxCtx = Math.max(...R.turns.map((t) => t.ctx.n + t.gen.n));
  R.genTotal = R.turns.reduce((a, t) => a + t.gen.n, 0);
  R.genMs = R.turns.reduce((a, t) => a + t.ms.decode, 0);
  cache.set(id, R);
  return R;
}
