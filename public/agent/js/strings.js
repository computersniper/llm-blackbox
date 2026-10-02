// 智能体页的界面文案（英文）。中文就是 index.html 里原有的文字：启动时从 DOM 里读出来当作中文字典，
// 所以中文模式下 applyDom() 换上去的还是原来那一份，一字不差。
import { addDict, applyDom, isEn } from '../../js/i18n.js';

const EN = {
  'ag.brand': 'Black Box · Agents',
  'ag.brandSub': 'Agents',
  'ag.crumbs': 'Current depth',
  'ag.codex': 'Codex of discoveries',
  'ag.side': 'Conversation',
  'ag.sideSub': 'Open model · real sandbox · recorded offline',
  'ag.sideClose': 'Hide conversation',
  'ag.plus': 'Open it up: see the next level (step into)',
  'ag.input': 'Task built from tokens',
  'ag.back': 'Delete the last token',
  'ag.send': 'Hand it to the agent',
  'ag.cands': 'Candidate tokens',
  'ag.stage': 'Where the agent works',
  'ag.strip': 'Open the conversation',
  'ag.nModel': 'Model',
  'ag.nHarness': 'harness · the program outside',
  'ag.nHarnessT': 'Loop, parse, execute',
  'ag.opPrompt': 'Build the whole context',
  'ag.opGen': 'Wait for the model to finish',
  'ag.opParse': 'Look for <tool_call>',
  'ag.opExec': 'Run it in the sandbox',
  'ag.opAppend': 'Append the result to the chat',
  'ag.opDone': 'No tool call → done',
  'ag.nSbx': 'Sandbox · /work',
  'ag.badge': 'sandbox · offline',
  'ag.badgeTitle': 'Commands run in a bubblewrap sandbox: system directories read-only, only /work writable, no network, 10-second timeout',
  'ag.tree': 'Files',
  'ag.term': 'Terminal',
  'ag.fab': '<b>＋</b>Open it up',
  'ag.fabTitle': 'Open it up: see how the agent works inside (＋)',
  'ag.dbg': 'Debugger',
  'ag.fold': 'Collapse',
  'ag.track': 'Steps at this depth',
  'ag.out': 'Step out (−)',
  'ag.in': 'Step into the next level (＋)',
  'ag.prev': 'Previous step (←)',
  'ag.play': 'Play / pause (Space)',
  'ag.next': 'Next step (→)',
  'ag.speed': 'Speed',
  'ag.cxEyebrow': 'CODEX · Discoveries',
  'ag.cxTitle': 'What you found inside the agent',
  'ag.close': 'Close',
  'ag.learn': 'Learn more →',
  'ag.reset': 'Reset progress',
};

// 中文字典直接取页面上现有的文字 / 属性
function zhFromDom() {
  const zh = {};
  document.querySelectorAll('[data-i18n]').forEach((el) => { zh[el.dataset.i18n] ??= el.textContent; });
  document.querySelectorAll('[data-i18n-html]').forEach((el) => { zh[el.dataset.i18nHtml] ??= el.innerHTML; });
  document.querySelectorAll('[data-i18n-attr]').forEach((el) => {
    for (const pair of el.dataset.i18nAttr.split(';')) {
      const [a, k] = pair.split(':').map((s) => s.trim());
      if (a && k) zh[k] ??= el.getAttribute(a);
    }
  });
  for (const k of Object.keys(zh)) if (k.startsWith('nav.')) delete zh[k];
  return zh;
}

export function initStrings() {
  addDict({ zh: zhFromDom(), en: EN });
  applyDom();
  if (isEn) {
    document.title = 'Agents · watch a coding agent work in a terminal';
    document.querySelector('meta[name="description"]')?.setAttribute('content', 'A real open model, Qwen3-4B, fixes a bug, counts log errors, writes a script and renames a function in a real Linux sandbox. The whole run is recorded offline: the full context of every turn, the candidate probabilities of every token, the real output of every command. Press ＋ to open it up level by level: the screen, the agent loop, the context window, token-by-token generation, the model inside.');
  }
}
