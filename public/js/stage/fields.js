// 微观乘加条目的字段读取：新旧两版导出都能用（这个文件不依赖 Three.js，讲解面板也会用到）。
// 新版导出给每个条目加了 n（项数）、pos / neg（正 / 负乘积之和）、
// cum（[[名次, 累计和], …]，按 |乘积| 从大到小，在名次 1, 2, 4, …, n 处采样）；
// 同时神经元编号从 n 改名为 j，q 的 RoPE 位置从 pos 改名为 row。

export const neuronId = (n) => n.j ?? n.n;
export const neuronTerms = (n) => (n.j !== undefined ? n.n : 1024);
export const ropePos = (e) => e.row ?? e.pos;

// 一个乘加条目的汇总：项数、正 / 负乘积之和、累计曲线。旧数据没有的字段给 null，调用方自己降级。
// sub：神经元条目里 g / u 两路各有一份汇总
export function sums(e, n0, sub = null) {
  const s = sub ? e[sub] : e.n !== undefined ? e : null;
  const n = sub ? neuronTerms(e) : e.n ?? n0;
  if (!s || typeof s !== 'object') return { n, pos: null, neg: null, cum: null };
  return {
    n,
    pos: Number.isFinite(s.pos) ? s.pos : null,
    neg: Number.isFinite(s.neg) ? s.neg : null,
    cum: Array.isArray(s.cum) && s.cum.length ? s.cum : null,
  };
}

// 神经元条目里 gate / up 两路，整理成和矩阵乘法条目一样的形状
export function neuronMM(n) {
  const dot = (w) => n.x.reduce((a, x, k) => a + x * w[k], 0);
  const j = neuronId(n);
  return {
    g: { j, dims: n.dims, x: n.x, w: n.wg, total: n.gz, shown: dot(n.wg), sum: sums(n, 1024, 'g') },
    u: { j, dims: n.dims, x: n.x, w: n.wu, total: n.uz, shown: dot(n.wu), sum: sums(n, 1024, 'u') },
  };
}
