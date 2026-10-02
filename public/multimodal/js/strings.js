// 多模态页的界面文案（英文）。中文就是 index.html 里原有的文字：启动时从 DOM 里读出来当作中文字典，
// 所以中文模式下 applyDom() 换上去的还是原来那一份，一字不差。
import { addDict, applyDom, isEn } from '../../js/i18n.js';

const EN = {
  'mm.brand': 'Black Box',
  'mm.brandSub': 'Multimodal',
  'mm.crumbs': 'Current depth',
  'mm.codex': 'Codex of discoveries',
  'mm.chat': 'Chat',
  'mm.chatSub': 'Alibaba’s open 2.1B-parameter vision-language model · real outputs, recorded offline',
  'mm.chatClose': 'Hide chat',
  'mm.plus': 'Open it up: see the next level (step into)',
  'mm.attach': 'Pick another image',
  'mm.attachPh': 'IMG',
  'mm.input': 'Question built from tokens',
  'mm.back': 'Delete the last token',
  'mm.send': 'Send',
  'mm.pics': 'Pick an image',
  'mm.cands': 'Candidate tokens',
  'mm.stage': 'Inside the model',
  'mm.follow': '◎ Back to follow view <kbd>F</kbd>',
  'mm.hintBtn': 'Controls',
  'mm.hintTitle': 'Controls (H)',
  'mm.hint': `
        <b>Move around inside the model</b>
        <span><kbd>Drag</kbd> rotate</span><span><kbd>Right-click</kbd>/<kbd>Shift</kbd>+drag pan</span><span><kbd>Wheel</kbd> zoom</span>
        <span><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move</span><span><kbd>Q</kbd>/<kbd>E</kbd> down / up</span><span><kbd>Double-click</kbd> fly to an object</span><span><kbd>F</kbd> back to follow</span>
        <span class="sep"></span><span><kbd>＋</kbd>/<kbd>−</kbd> change level</span><span><kbd>Space</kbd> play</span><span><kbd>←</kbd>/<kbd>→</kbd> step</span><span>Click a patch on the monitor: change the query position</span>
      `,
  'mm.mon': 'Image monitor',
  'mm.monTitle': 'Monitor',
  'mm.fold': 'Collapse',
  'mm.loading': 'Loading the real internal data for this reply…',
  'mm.dbg': 'Debugger',
  'mm.track': 'Steps at this depth',
  'mm.out': 'Step out (−)',
  'mm.in': 'Step into the next level (＋)',
  'mm.prev': 'Previous step (←)',
  'mm.play': 'Play / pause (Space)',
  'mm.next': 'Next step (→)',
  'mm.speed': 'Speed',
  'mm.cxEyebrow': 'CODEX · Discoveries',
  'mm.cxTitle': 'How the model sees',
  'mm.close': 'Close',
  'mm.learn': 'Learn more →',
  'mm.reset': 'Reset progress',
};

// 中文字典直接取页面上现有的文字 / 属性
function zhFromDom() {
  const zh = {};
  document.querySelectorAll('[data-i18n]').forEach((el) => { zh[el.dataset.i18n] ??= el.textContent; });
  document.querySelectorAll('[data-i18n-html]').forEach((el) => { zh[el.dataset.i18nHtml] ??= el.innerHTML; });
  document.querySelectorAll('[data-i18n-attr]').forEach((el) => {
    for (const pair of el.dataset.i18nAttr.split(';')) {
      const [a, k] = pair.split(':').map((s) => s.trim());
      if (a && k && !k.startsWith('nav.')) zh[k] ??= el.getAttribute(a);
    }
  });
  for (const k of Object.keys(zh)) if (k.startsWith('nav.')) delete zh[k];
  return zh;
}

export function initStrings() {
  addDict({ zh: zhFromDom(), en: EN });
  applyDom();
  if (isEn) {
    document.title = 'Black Box · Multimodal: how an LLM sees an image';
    document.querySelector('meta[name="description"]')?.setAttribute('content', 'Show a real open vision-language model, Qwen3-VL-2B, an image, ask it a question, then open it up level by level: how the image is cut into 16×16 patches, passes through a 24-layer vision encoder, is merged 2×2 into tokens and spliced into the chat with 3-D positions, and where the language model looks in the image as it writes each word. Every number comes from the real model.');
  }
}
