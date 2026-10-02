// 推理页 index.html 里静态文案的英文版（js/i18n.js 的 addDict）。
// 中文就是 HTML 里原本写的那些字：启动时从 DOM 读出来登记成中文词条（seedZhFromDom），
// 所以中文模式下 applyDom() 写回去的是一模一样的内容，HTML 里的中文仍是唯一的出处。
import { addDict } from '../i18n.js';

export function seedZhFromDom(root = document) {
  const zh = {};
  root.querySelectorAll('[data-i18n]').forEach((el) => { zh[el.dataset.i18n] ??= el.textContent; });
  root.querySelectorAll('[data-i18n-html]').forEach((el) => { zh[el.dataset.i18nHtml] ??= el.innerHTML; });
  root.querySelectorAll('[data-i18n-attr]').forEach((el) => {
    for (const pair of el.dataset.i18nAttr.split(';')) {
      const [a, k] = pair.split(':').map((s) => s.trim());
      if (a && k && el.hasAttribute(a)) zh[k] ??= el.getAttribute(a);
    }
  });
  addDict({ zh });
}

addDict({
  zh: {
    'inf.title': '揭开黑箱 · 看大模型如何想出每一个字',
    'inf.desc': '和真实的开源大模型 Qwen3-0.6B 聊天，然后像调试程序一样，一层层揭开它：词元、嵌入、28 层 Transformer、注意力、SwiGLU 神经元，直到权重的比特。所有数据都来自真实模型。',
  },
  en: {
    'inf.title': 'Black Box · Watch an LLM come up with every word',
    'inf.desc': 'Chat with the real open-source model Qwen3-0.6B, then open it up layer by layer like a debugger: tokens, embeddings, 28 Transformer layers, attention, SwiGLU neurons, all the way down to the bits of a single weight. Every number comes from the real model.',
    'brand.name': 'Black Box',
    'brand.sub': 'Inside an LLM',
    'inf.crumbs': 'Current depth',
    'inf.gh': 'GitHub repository',
    'inf.codex': 'Codex of insights',
    'inf.chat': 'Chat',
    'inf.chatSub': "Alibaba's open 0.6B model · real recorded outputs",
    'inf.chatFold': 'Hide chat',
    'inf.plus': 'Open it up: look one level deeper (step into)',
    'inf.input': 'Question built from tokens',
    'inf.back': 'Remove the last token',
    'inf.send': 'Send',
    'inf.cands': 'Candidate tokens',
    'inf.stage': 'Inside the model',
    'inf.follow': '◎ Back to follow view <kbd>F</kbd>',
    'inf.hint': 'Controls',
    'inf.hintT': 'Controls (H)',
    'inf.hintBody': `
        <b>Walk around inside the machine</b>
        <span><kbd>Drag</kbd> rotate</span><span><kbd>Right-drag</kbd>/<kbd>Shift</kbd>+drag pan</span><span><kbd>Wheel</kbd> move in (keep going to pass through)</span>
        <span><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move</span><span><kbd>Q</kbd>/<kbd>E</kbd> down / up</span><span><kbd>Shift</kbd> faster</span><span><kbd>Double-click</kbd> fly to an object</span><span><kbd>F</kbd> back to follow view</span>
        <span class="sep"></span><span><kbd>＋</kbd>/<kbd>−</kbd> or <kbd>↑</kbd>/<kbd>↓</kbd> change depth</span><span><kbd>Space</kbd> play</span><span><kbd>←</kbd>/<kbd>→</kbd> step</span>
      `,
    'inf.loading': 'Loading the real internal data for this reply…',
    'inf.dbg': 'Debugger',
    'inf.fold': 'Collapse',
    'inf.track': 'Steps at the current depth',
    'inf.out': 'Step out one level (−)',
    'inf.in': 'Step into the next level (＋)',
    'inf.prev': 'Previous step (←)',
    'inf.play': 'Play / pause (Space)',
    'inf.next': 'Next step (→)',
    'inf.speed': 'Speed',
    'inf.cxEyebrow': 'CODEX · Insights',
    'inf.cxTitle': 'What you found inside the black box',
    'inf.close': 'Close',
    'inf.cxFoot': 'Model: Qwen3-0.6B (28 layers · hidden size 1024 · 16 query heads / 8 key-value heads · SwiGLU 3072 · vocabulary 151,936). The answers, tokenization, probabilities, attention, activations and weights all come from one offline run of the real model. To keep the download small, each attention row keeps only its top 4 weights, and only the brightest neurons are kept.',
    'inf.learnMore': 'Keep learning →',
    'inf.reset': 'Reset progress',
  },
});
