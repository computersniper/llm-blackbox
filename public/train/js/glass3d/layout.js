// 玻璃小模型 3D 机器的布局（世界坐标；x 向右，y 向上，z 朝向镜头）。
//
//   左边一列是残差流：第 0 段的 8 个位置各一根光柱，从托盘上的字块一直通到最后的 RMSNorm；
//   柱子上开了三个“窗口”，显示这里的残差 h₀ / h₁ / h₂（16 维 × 8 个位置，真实数值）。
//   右边按层摆开每一块参数（一个方块 = 一个参数），形状按真实矩阵尺寸：
//     最下面 E [20 × 16]（查表）；注意力一层：γ₁、W_q / W_k / W_v、W_o；前馈一层：γ₂、W_gate / W_up、W_down；
//     最上面：γ_f、Eᵀ（和最下面的 E 是同一张表，画成镜像，不另算参数），再往上是 20 个字的概率柱和损失。
//   矩阵一律按 y = x · W 摆：W 的行是输入维（竖着），列是输出维（横着）；
//   输入激活画在矩阵左边（转置成 输入维 × 8 个位置，和 W 的行对齐），输出激活画在矩阵上面（8 个位置 × 输出维，和 W 的列对齐）。
export const C = 0.12;          // 参数方块的间距
export const FACE = 0.1;        // 方块正面边长
export const SP = 0.32;         // 残差光柱的间距（8 个位置）
export const XS0 = -2.4;        // 位置 0 的光柱
export const xPos = (p) => XS0 + p * SP;
export const SPINE = { x0: XS0 - SP / 2, x1: XS0 + 7.5 * SP, top: 16.5 };

// 每一块：kind 'w' = 参数，'a' = 激活；rows × cols 个格子，(x0, yTop) 是左上角，px / py 是格子间距
// 参数块的 t：张量名；mirror：Eᵀ 镜像（格子 (r, c) 对应 E[c, r]）
const W = (id, t, rows, cols, x0, yTop, extra = {}) => ({ id, kind: 'w', t, rows, cols, x0, yTop, px: C, py: C, ...extra });
const A = (id, rows, cols, x0, yTop, extra = {}) => ({ id, kind: 'a', rows, cols, x0, yTop, px: C, py: C, ...extra });
const SW = (id, yTop) => ({ id, kind: 'a', rows: 16, cols: 8, x0: SPINE.x0, yTop, px: SP, py: C, spine: true });

// 三层的竖直位置
export const Y = {
  tray: 0.2, e0: 0.6, e1: 3.0, ring1: 3.3,
  a0: 3.6, a1: 5.52, aOut: 6.78, aRow: 8.14, add1: 6.3, h1: 8.52, ring2: 8.9,
  f0: 9.3, f1: 11.22, fOut: 12.48, fAct: 13.84, d1: 13.14, fF: 14.4, add2: 13.9, h2: 16.12, ringf: 16.5,
  hB: 16.9, hT: 18.82, logit: 20.08, shelf: 20.4,
};

export const BLOCKS = [
  // 嵌入
  W('E', 'E', 20, 16, 0.75, Y.e1),
  SW('h0', 2.76),
  // 注意力
  W('g1', 'g1', 16, 1, 0.3, Y.a1),
  A('n1', 16, 8, 0.6, Y.a1),
  W('Wq', 'Wq', 16, 16, 1.8, Y.a1), W('Wk', 'Wk', 16, 16, 3.97, Y.a1), W('Wv', 'Wv', 16, 16, 6.14, Y.a1),
  A('q', 8, 16, 1.8, Y.aOut), A('k', 8, 16, 3.97, Y.aOut), A('v', 8, 16, 6.14, Y.aOut),
  A('att0', 8, 8, 2.95, Y.aRow), A('att1', 8, 8, 4.07, Y.aRow), A('ao', 8, 16, 6.14, Y.aRow),
  A('aoT', 16, 8, 8.5, Y.a1),
  W('Wo', 'Wo', 16, 16, 9.7, Y.a1),
  A('o', 8, 16, 9.7, Y.aOut),
  SW('h1', Y.h1),
  // 前馈
  W('g2', 'g2', 16, 1, 0.3, Y.f1),
  A('n2', 16, 8, 0.6, Y.f1),
  W('Wg', 'Wg', 16, 32, 1.8, Y.f1), W('Wu', 'Wu', 16, 32, 5.89, Y.f1),
  A('gate', 8, 32, 1.8, Y.fOut), A('up', 8, 32, 5.89, Y.fOut),
  A('act', 8, 32, 3.85, Y.fAct),
  A('actT', 32, 8, 10.2, Y.d1),
  W('Wd', 'Wd', 32, 16, 11.4, Y.d1),
  A('f', 8, 16, 11.4, Y.fF),
  SW('h2', Y.h2),
  // 输出
  W('gf', 'gf', 16, 1, 0.3, Y.hT),
  A('nf', 16, 8, 0.6, Y.hT),
  W('ET', 'E', 16, 20, 1.8, Y.hT, { mirror: true }),
  A('logits', 8, 20, 1.8, Y.logit),
];
export const BLOCK = Object.fromEntries(BLOCKS.map((b) => [b.id, b]));
for (const b of BLOCKS) {
  b.w = b.cols * b.px;
  b.h = b.rows * b.py;
  b.yBot = b.yTop - b.h;
  b.x1 = b.x0 + b.w;
}
export const cellX = (b, c) => b.x0 + (c + 0.5) * b.px;
export const cellY = (b, r) => b.yTop - (r + 0.5) * b.py;

// 概率架：20 个字 × 8 个位置，柱子往上长（x = 字，z = 位置；位置 0 在后排、位置 7 在前排，从上往下读和诗的顺序一样）
export const SHELF = { x0: 1.8, px: 0.3, z0: -1.05, pz: 0.3, y: Y.shelf, hMax: 2.1 };
export const shelfX = (v) => SHELF.x0 + (v + 0.5) * SHELF.px;
export const shelfZ = (p) => SHELF.z0 + p * SHELF.pz;
export const GAUGE = { x: 8.75, z: 0, y: Y.shelf, hMax: 2.1, lnV: Math.log(20) };

// 语料轮：《静夜思》首尾相接的 25 个字排成一个竖着的轮子，每一步从上面切 8 段
export const WHEEL = { x: -4.55, y: 1.75, z: 0.1, r: 1.32 };
// 托盘：第 b 段排在第 b 行（第 0 段在最前面，正好在 8 根光柱底下）；错开一位后的答案排在更前面
export const TRAY = { y: Y.tray, dz: 0.32, zTarget: 0.42 };

// 整台机器的范围（镜头取景用）
export const BOUNDS = { x0: -6.1, x1: 13.5, y0: -0.3, y1: 23.2 };

// 各个算子在机器上的范围 [x0, x1, y0, y1]（D2 / D3 的镜头取景）
export const OP_RECT = {
  batch: [-6.1, 0.3, -0.4, 3.3],
  emb: [-2.9, 2.9, -0.3, 3.3],
  norm1: [-2.9, 1.9, 2.4, 5.9],
  qkv: [0.1, 8.3, 3.2, 7.1],
  attn: [1.5, 8.3, 5.5, 8.6],
  wo: [-2.9, 11.9, 3.2, 8.7],
  norm2: [-2.9, 1.9, 6.2, 11.6],
  ffn: [0.1, 10.0, 8.9, 14.1],
  wd: [-2.9, 13.6, 8.9, 16.3],
  normf: [-2.9, 1.9, 13.8, 19.2],
  logits: [0.1, 9.6, 16.5, 23],
  loss: [1.2, 9.8, 18.6, 23.2],
};

// 每块参数的“主体”：参数块 + 它的输入 / 输出激活（D4 镜头取景、标签位置）
export const TENSOR_PARTS = {
  E: ['E', 'h0'], g1: ['g1', 'n1'], Wq: ['n1', 'Wq', 'q'], Wk: ['Wk', 'k'], Wv: ['Wv', 'v'], Wo: ['aoT', 'Wo', 'o'],
  g2: ['g2', 'n2'], Wg: ['n2', 'Wg', 'gate'], Wu: ['Wu', 'up'], Wd: ['actT', 'Wd', 'f'], gf: ['gf', 'nf'], ET: ['nf', 'ET', 'logits'],
};

export function unionRect(ids, pad = 0.3) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const id of ids) { const b = BLOCK[id]; x0 = Math.min(x0, b.x0); x1 = Math.max(x1, b.x1); y0 = Math.min(y0, b.yBot); y1 = Math.max(y1, b.yTop); }
  return [x0 - pad, x1 + pad, y0 - pad, y1 + pad];
}
