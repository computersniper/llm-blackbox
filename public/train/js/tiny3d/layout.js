// 唐宋诗小模型 3D 机器的布局（世界坐标；x 向右，y 向上，z 朝向镜头）。
//
//   中间偏左是残差流：批次第 0 行的 64 个位置各一根光纤，排成 16 × 4（像一页字：第 0–15 个字在最后一排），
//   从托盘里的字块一直通到顶上的预测柱；左边每个层边界旁有一块“逻辑透镜”屏，写着这一层此刻最想写的 64 个字。
//   右边是 6 层，每层一条带子：RMSNorm → W_q / W_k / W_v（4 个查询头 / 2 个 KV 头）→ 注意力 → W_o → ⊕
//   → RMSNorm → W_gate / W_up → W_down → ⊕。矩阵按 y = x · W 摆（行 = 输入维，竖着；列 = 输出维，横着），
//   尺寸按真实形状：256 维 = 1 个单位，所以 W_gate 是 1 × 3，W_down 是 3 × 1。
//   最底下的嵌入表 E [7478 × 256] 平躺在地上，往后伸 29 个单位（比整台机器还长）；
//   最顶上同一张表的转置 Eᵀ 画成一条半透明的影子（输入输出共用，不另算参数）。
export const U = 1 / 256;                 // 一维的长度
export const NP = 64, NC = 16, NR = 4;    // 第 0 行的 64 个位置，16 列 × 4 排
export const FS = 0.13;                   // 光纤间距
export const BX0 = -2.55, BZ0 = -0.2;     // 第 0 个位置的光纤
export const fiberX = (i) => BX0 + (i % NC) * FS;
export const fiberZ = (i) => BZ0 + Math.floor(i / NC) * FS;
export const BUNDLE = { x0: BX0 - FS * 0.6, x1: BX0 + (NC - 0.4) * FS, z0: BZ0 - FS * 0.6, z1: BZ0 + (NR - 0.4) * FS };
export const BCX = (BUNDLE.x0 + BUNDLE.x1) / 2, BCZ = (BUNDLE.z0 + BUNDLE.z1) / 2;

export const NL = 6;
export const YB0 = 1.4, HB = 3.5;        // 第 0 层底边、每层高度
export const yBand = (l) => YB0 + l * HB;
// 层边界 b：0 = 嵌入之后，l + 1 = 第 l 层之后
export const yBound = (b) => YB0 - 0.15 + b * HB;
// 一层里各处的高度（相对 yBand）
export const SUB = { ring1: 0.1, attn0: 0.3, attn1: 1.3, qkvOut: 1.36, heads0: 1.5, heads1: 1.96, add1: 2.06, ring2: 2.2, ffn0: 2.35, ffn1: 3.35 };
export const YTOP = yBound(NL);          // 最后一层之后（最后的 RMSNorm）
export const YT = YTOP + 0.7;            // 顶上 Eᵀ 的影子
export const YP = YT + 0.3;              // 预测柱的底
export const PH = 1.5;                   // 预测柱满格（p = 1）

// 每层的面板：kind → [x0, 宽(输出维), 高(输入维)、底边相对 yBand]
export const PANEL = {
  ln1: { x0: 0.02, w: 0.05, h: 1, y0: SUB.attn0, n: 256 },
  q: { x0: 0.2, w: 1, h: 1, y0: SUB.attn0 },
  k: { x0: 1.35, w: 0.5, h: 1, y0: SUB.attn0 },
  v: { x0: 2.0, w: 0.5, h: 1, y0: SUB.attn0 },
  o: { x0: 2.85, w: 1, h: 1, y0: SUB.attn0 },
  qn: { x0: 0.2, w: 0.25, h: 0.04, y0: SUB.qkvOut + 0.07, n: 64 },
  kn: { x0: 1.35, w: 0.25, h: 0.04, y0: SUB.qkvOut + 0.07, n: 64 },
  ln2: { x0: 0.02, w: 0.05, h: 1, y0: SUB.ffn0, n: 256 },
  gate: { x0: 0.2, w: 3, h: 1, y0: SUB.ffn0 },
  up: { x0: 3.35, w: 3, h: 1, y0: SUB.ffn0 },
  down: { x0: 6.7, w: 1, h: 3, y0: SUB.attn0 },
};
export const LAYER_KINDS = ['ln1', 'q', 'k', 'v', 'qn', 'kn', 'o', 'ln2', 'gate', 'up', 'down'];
export const MAT_KINDS = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'];
// 注意力图案：4 个查询头各一块（在留出的《登鹳雀楼》上记录的 25 × 25）
export const HEAD = { x0: 0.2, size: 0.46, gap: 0.58 };
export const headX = (h) => HEAD.x0 + h * HEAD.gap;

// 嵌入表 E 平躺在地上：x = 维度（256 = 1 个单位），z = 字的编号（越常见越靠前）
export const EMB = { x0: 0.2, y: 0.06, z0: 0.7, rows: 7478 };
export const embZ = (r) => EMB.z0 - (r + 0.5) * U;
export const EMB_Z1 = EMB.z0 - EMB.rows * U;
// 最后的 RMSNorm γ（256）
export const NF = { x0: 0.2, w: 1, h: 0.05, y: YTOP + 0.18 };

// 托盘：第 0 行的 64 个字就放在光纤底下；下面一叠薄片是其余 63 行
export const TRAY = { y: 0.13, tile: 0.112 };
// 答案（错开一位）排在托盘前面
export const TGT_DZ = 0.66;
// 逻辑透镜屏：每个层边界一块，16 × 4 个字
export const LENS = { x0: -7.45, cw: 0.26, ch: 0.3 };
export const LENS_W = NC * LENS.cw, LENS_H = NR * LENS.ch;
// 损失：每个位置的 −ln p（预测柱右边）和整批的损失管
export const LOSSB = { x0: 1.65, y: YP, hMax: 1.3, lnMax: 10 };
export const lossX = (i) => LOSSB.x0 + (i % NC) * FS;
export const GAUGE = { x: 4.35, z: 0, y: YP, hMax: 1.6, max: 9.5 };
// 打印机：每个检查点用同一组随机数写 4 首诗
export const PRINTER = { x: 6.1, y: YP, z: -0.2 };
// 语料（示意）从左边来
export const CORPUS = { x0: -11.5, x1: -3.2, y: 0.14, z: 0.95 };

// 右边的两件展品：嵌入的地图（PCA 前两维）、损失地形（41 × 41 个真实算出来的点）
export const PCA = { x0: 8.8, x1: 14.6, y0: 9.0, y1: 14.8, z: -1.2 };
export const LAND = { x0: 8.8, x1: 14.2, z0: 0.2, z1: 5.6, y0: -0.25, hMax: 2.0 };

// 整台机器的范围（镜头取景用）
export const MACHINE = { x0: -8.4, x1: 7.9, y0: -0.4, y1: YP + 2.6 };
export const ALL = { x0: -8.4, x1: 14.6, y0: -0.4, y1: YP + 2.6 };

// 每块面板在世界里的矩形：[x0, x1, y0, y1]
export function panelRect(l, kind) {
  const p = PANEL[kind], yb = yBand(l);
  return [p.x0, p.x0 + p.w, yb + p.y0, yb + p.y0 + p.h];
}
// 一个算子用到的范围（D2 / D3 的镜头）：attn / ffn 这一半层
export function opRect(op) {
  if (op.sub === 'emb') return [-3.0, 2.0, -0.3, yBound(0) + 0.6, -3.2, 1.2];
  if (op.sub === 'head') return [-3.0, 7.2, YTOP - 0.4, YP + PH + 0.3, -1.2, 1.0];
  const yb = yBand(op.L);
  if (op.sub === 'attn') return [-7.6, 4.1, yb - 0.85, yb + SUB.heads1 + 0.25, -0.6, 0.6];
  return [-7.6, 7.9, yb + SUB.ring2 - 0.3, yb + HB + 0.75, -0.6, 0.6];
}

// 跟踪的 4 个权重在机器上的位置（数据里的下标是 PyTorch 的 [输出, 输入]；面板按 y = x · W 摆，行 = 输入、列 = 输出）
export function featPos(ft) {
  const nm = ft.name, idx = ft.index;
  if (nm === 'model.embed_tokens.weight') return { x: EMB.x0 + (idx[1] + 0.5) * U, y: EMB.y + 0.01, z: embZ(idx[0]), flat: true };
  const m = nm.match(/^model\.layers\.(\d+)\.(.+)$/);
  const l = Number(m[1]);
  const kind = { 'self_attn.q_proj.weight': 'q', 'mlp.down_proj.weight': 'down', 'input_layernorm.weight': 'ln1' }[m[2]] || 'q';
  const r = panelRect(l, kind);
  if (kind === 'ln1') return { x: (r[0] + r[1]) / 2, y: r[3] - (idx[0] + 0.5) * U, z: 0.012, l, kind };
  return { x: r[0] + (idx[0] + 0.5) * U, y: r[3] - (idx[1] + 0.5) * U, z: 0.012, l, kind };
}
