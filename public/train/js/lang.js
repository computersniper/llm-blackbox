// 训练页的中英文（机制见 ../../js/i18n.js）：静态文案字典 + 把数据里的中文标签换成英文的几个小函数。
// 数据本身不改：唐宋诗、Qwen3 的那条对话照常显示中文原文，英文模式下另附标明“翻译”的释义。
import { addDict, applyDom, mountLangSwitch, isEn, L, t } from '../../js/i18n.js';

export { isEn, L };

addDict({
  zh: {
    'tr.title': '大模型训练 · 揭开黑箱',
    'tr.desc': '看一个大模型是怎么被训练出来的：先拿一个小到每个参数都画得出来的玻璃模型（2,928 个参数，学李白《静夜思》），看初始化、每一步的前向、反向、更新，细到一个参数的一生；再放大到从零训练的唐宋诗小模型，和真实 Qwen3-0.6B 的监督微调。所有数据都来自真实运行。',
    'tr.brand': '大模型训练',
    'tr.crumbs': '当前深度',
    'tr.codex': '知识碎片图鉴',
    'tr.soundOn': '打开声音',
    'tr.stage': '训练过程',
    'tr.runs': '选择一章',
    'tr.runGlass': '玻璃小模型',
    'tr.runGlassSub': '2,928 个参数 · 全部看得见',
    'tr.runGlassShort': '玻璃小模型',
    'tr.runTinyShort': '放大预训练',
    'tr.runQwenShort': 'Qwen3 微调',
    'tr.runTiny': '放大：从零预训练',
    'tr.runTinySub': '唐宋诗 · 664 万参数',
    'tr.runQwen': '真实模型：微调',
    'tr.follow': '◎ 回到跟随视角 <kbd>F</kbd>',
    'tr.hint': '操作说明',
    'tr.hintTitle': '操作说明（H）',
    'tr.loading': '正在载入真实训练记录…',
    'tr.dbg': '调试器',
    'tr.fold': '收起',
    'tr.track': '当前深度的步骤',
    'tr.out': '退回上一层（−）',
    'tr.in': '单步进入下一层（＋）',
    'tr.prev': '上一步（←）',
    'tr.play': '播放 / 暂停（空格）',
    'tr.next': '下一步（→）',
    'tr.speed': '倍速',
    'tr.cxEyebrow': 'CODEX · 训练现场的知识碎片',
    'tr.cxTitle': '你在训练现场的发现',
    'tr.close': '关闭',
    'tr.cxFoot': '三章都是真实运行的记录：玻璃小模型在 CPU 上训练了 200 步（另有全零、放大 50 倍两个对照）；唐宋诗小模型在一块 RTX 5090 上从零训练了 4000 步；Qwen3-0.6B 在同一块显卡上做了 3 步全参数监督微调。凡是标“示意”的内容不是实测。',
    'tr.learn': '延伸学习 →',
    'tr.reset': '重置进度',
    'tr.noscript': '这个网站需要 JavaScript 才能运行。',
  },
  en: {
    'tr.title': 'LLM Training · Opening the Black Box',
    'tr.desc': 'Watch a language model being trained: first a glass-box model so small that every one of its 2,928 parameters is drawn on screen (it learns Li Bai’s “Quiet Night Thought”) — initialization, each step’s forward pass, backward pass and update, down to the life of a single parameter; then scale up to a poetry model pretrained from scratch and real supervised fine-tuning of Qwen3-0.6B. Every number comes from a real run.',
    'tr.brand': 'LLM Training',
    'tr.crumbs': 'Current depth',
    'tr.codex': 'Insight codex',
    'tr.soundOn': 'Turn sound on',
    'tr.stage': 'Training run',
    'tr.runs': 'Pick a chapter',
    'tr.runGlass': 'Glass-box model',
    'tr.runGlassSub': '2,928 params · all visible',
    'tr.runGlassShort': 'Glass box',
    'tr.runTinyShort': 'Scale up',
    'tr.runQwenShort': 'Qwen3 SFT',
    'tr.runTiny': 'Scale up: pretraining',
    'tr.runTinySub': 'Tang–Song poems · 6.64M',
    'tr.runQwen': 'Real model: fine-tuning',
    'tr.follow': '◎ Back to follow view <kbd>F</kbd>',
    'tr.hint': 'Controls',
    'tr.hintTitle': 'Controls (H)',
    'tr.loading': 'Loading the real training logs…',
    'tr.dbg': 'Debugger',
    'tr.fold': 'Collapse',
    'tr.track': 'Steps at this depth',
    'tr.out': 'Step out (−)',
    'tr.in': 'Step into the next level (+)',
    'tr.prev': 'Previous step (←)',
    'tr.play': 'Play / pause (Space)',
    'tr.next': 'Next step (→)',
    'tr.speed': 'Speed',
    'tr.cxEyebrow': 'CODEX · INSIGHTS FROM THE TRAINING FLOOR',
    'tr.cxTitle': 'What you found on the training floor',
    'tr.close': 'Close',
    'tr.cxFoot': 'All three chapters are real recordings: the glass-box model was trained for 200 steps on a CPU (plus two controls: all-zero and 50× larger initialization); the poetry model was trained from scratch for 4,000 steps on one RTX 5090; Qwen3-0.6B got 3 full-parameter fine-tuning steps on the same card. Anything marked “illustrative” is not measured.',
    'tr.learn': 'Keep learning →',
    'tr.reset': 'Reset progress',
    'tr.noscript': 'This site needs JavaScript to run.',
  },
});

// 两套：玻璃小模型是 3D 机器（.h3d），另外两章是 2D 画布（.h2d），train.css 按 body.g3d 只显示一套
const HINT_EN = '<div class="h3d"><b>Walk around the machine</b>'
  + '<span><kbd>Drag</kbd> rotate</span><span><kbd>Right</kbd>/<kbd>Shift</kbd>+drag pan</span><span><kbd>Wheel</kbd> move in (keep going to pass through)</span><span><kbd>Double-click</kbd> fly to a panel</span>'
  + '<span><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move</span><span><kbd>Q</kbd>/<kbd>E</kbd> down / up</span><span><kbd>F</kbd> follow again</span><span>Click a cube: that parameter’s life</span></div>'
  + '<div class="h2d"><b>Walk around the training floor</b>'
  + '<span><kbd>Drag</kbd> pan</span><span><kbd>Wheel</kbd>/pinch zoom</span><span><kbd>Double-click</kbd> zoom into a spot</span>'
  + '<span><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move</span><span><kbd>Q</kbd>/<kbd>E</kbd> zoom out / in</span><span><kbd>F</kbd> follow again</span></div>'
  + '<span class="sep"></span><span><kbd>＋</kbd>/<kbd>−</kbd> or <kbd>↓</kbd>/<kbd>↑</kbd> change level</span><span><kbd>Space</kbd> play</span><span><kbd>←</kbd>/<kbd>→</kbd> step</span><span><kbd>1</kbd>–<kbd>6</kbd> speed</span>';

// 页面启动时调用：替换静态文案、标题、描述，放上中 / EN 开关
export function initLang() {
  applyDom();
  mountLangSwitch(document.getElementById('langSwitch'));
  if (!isEn) return;
  document.title = t('tr.title');
  document.querySelector('meta[name="description"]')?.setAttribute('content', t('tr.desc'));
  const hint = document.getElementById('stageHint');
  if (hint) hint.innerHTML = HINT_EN;
}

// 英文里的大数：6,637,056 → 6.64M（中文那边用“万 / 亿”）
export function compact(n, d = 2) {
  const a = Math.abs(n);
  const f = (v, s) => `${Number(v.toFixed(d))}${s}`;
  if (a >= 1e12) return f(n / 1e12, 'T');
  if (a >= 1e9) return f(n / 1e9, 'B');
  if (a >= 1e6) return f(n / 1e6, 'M');
  if (a >= 1e3) return f(n / 1e3, 'K');
  return String(n);
}

// 数据里跟踪的权重的名字（导出时写的是中文）：英文模式下按张量名和下标重新生成
const T_EN = { q_proj: 'W_q', k_proj: 'W_k', v_proj: 'W_v', o_proj: 'W_o', gate_proj: 'W_gate', up_proj: 'W_up', down_proj: 'W_down' };
export function featLabel(ft, kind) {
  if (!isEn) return ft.label;
  const m = ft.name.match(/^model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.weight$/);
  if (kind === 'qwen') {
    if (ft.name === 'model.embed_tokens.weight') return 'Embedding / output matrix · row of “黑”';
    if (ft.name === 'model.norm.weight') return 'Final RMSNorm scale γ';
    if (m) return `Layer ${m[1]} ${T_EN[m[2]] || m[2]}: largest-gradient weight`;
    return ft.label;
  }
  if (ft.name === 'model.embed_tokens.weight') return `Embedding · “${ft.label.match(/「(.+)」/)?.[1] ?? '?'}” dim ${ft.index[1]}`;
  const n = ft.name.match(/^model\.layers\.(\d+)\.input_layernorm\.weight$/);
  if (n) return `Layer ${n[1]} RMSNorm γ[${ft.index.join(', ')}]`;
  if (m) return `Layer ${m[1]} ${T_EN[m[2]] || m[2]}[${ft.index.join(', ')}]`;
  return ft.label;
}

// 《登鹳雀楼》（王之涣）和 SFT 那条对话的英文释义（仅作辅助，页面上标明是翻译）
export const POEM_EN = 'Wang Zhihuan, “On the Stork Tower” (Tang dynasty)';
export const POEM_GLOSS = 'The white sun sinks behind the hills, / the Yellow River flows into the sea. / To see a thousand li farther, / climb one more floor.';
export const SFT_EN = {
  question: 'Who are you?',
  answer: 'I’m the little model inside the black box.',
  rejected: 'I’m your virtual assistant, an AI-based smart assistant designed to provide help and support. If you have any questions or need help, just let me know!',
  note: 'All 3 steps use the same conversation (in real training every step sees a different batch)',
};
