// 第三章（Qwen3-0.6B 三步 SFT，3D 机器）挂到训练页上要用的全部东西。页面（../main.js）只需要：
//   loadQwen3d     载入数据（原来的 qwen.json + 补记的 qwen3d.json）→ 这一章的 R（kind 'qwen3d'）
//   QwenTimeline   步骤树（D1–D5，接口同玻璃小模型的 GlassTimeline）
//   QwenStage      3D 舞台（引擎 + 机器 + 浮层）
//   withQwen3d     给调试器的 Controls 加上这一章的讲解
//   planQwen3d     当前画面要哪些数据分块
//   discoverQwen3d 这一步能发现哪些知识碎片
import './lang.js';
export { loadQwen3d } from './data.js';
export { QwenTimeline } from './timeline.js';
export { QwenStage } from './stage.js';
export { withQwen3d } from './controls.js';

// need = 这一屏离不开、hold = 播放时等它到了再走、soon = 预取
export function planQwen3d(p, R, tl) {
  const K = R.K, k = tl.k;
  const st = (j) => (j >= 0 && j < K ? R.keySt(j) : null), x3 = (j) => (j >= 0 && j < K ? R.key3(j) : null);
  if (tl.depth === 1) {
    p.want.push(st(k), x3(k));
    p.hold.push(st(k), x3(k));
    p.soon.push(st(k + 1), x3(k + 1));
  } else {
    p.need.push(st(k), x3(k));
    p.hold.push(st(k), x3(k));
    p.soon.push(st(k + 1), x3(k + 1), st(k - 1), x3(k - 1));
    // 遮住提示那一步的对照图要用第 1 步的逐层梯度（两种算法）
    if (tl.step.ph === 'batch') p.want.push(st(0));
  }
  for (const key of ['need', 'want', 'hold', 'soon']) p[key] = [...new Set(p[key].filter(Boolean))];
  return p;
}

export function discoverQwen3d(tl, discover) {
  const s = tl.step, d = tl.depth;
  if (s.ph === 'end') { discover('memorize'); discover('dpo'); return; }
  if (s.ph === 'batch' && s.sub === 'tpl') discover('template');
  if (s.ph === 'batch' && (s.sub === 'mask' || (d === 2 && tl.p > 0.5))) discover('sftmask');
  if (s.ph === 'batch' && s.sub === 'shift') discover('teacher');
  if (s.ph === 'fwd' && d >= 2) discover('lens');
  if (s.ph === 'loss' && s.sub === 'pos') discover('ce');
  if (s.ph === 'bwd' && d >= 2) discover('backprop');
  if (s.ph === 'bwd' && s.t) discover('outer');
  if (s.ph === 'upd' && s.sub === 'clip') discover('clip');
  if (s.ph === 'upd' && s.mi === 'bc') discover(s.k === 0 ? 'sign' : 'bias');
  if (s.ph === 'upd' && s.mi === 'dw') discover('decay');
  if (s.ph === 'upd' && (s.mi === 'bits' || s.sub === 'bf16')) discover('fp32');
  if (s.ph === 'upd' && d === 2) discover('onestep');
}
