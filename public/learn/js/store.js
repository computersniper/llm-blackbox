// 本页的本地存储：学习路线的进度、术语表的排列方式。
// 只存在这台设备的浏览器里；隐私模式、禁用存储时读写都会抛错，一律当作空。

const KEY_DONE = 'bb-learn-done-v1';
const KEY_GL = 'bb-learn-gl-mode';

export function loadDone() {
  try {
    const v = JSON.parse(localStorage.getItem(KEY_DONE) || '[]');
    return new Set(Array.isArray(v) ? v.filter((x) => typeof x === 'string') : []);
  } catch {
    return new Set();
  }
}

export function saveDone(set) {
  try { localStorage.setItem(KEY_DONE, JSON.stringify([...set])); } catch { /* 存不了就只在这次浏览里有效 */ }
}

// 另一个标签页改了进度时同步过来
export function onDoneChange(fn) {
  addEventListener('storage', (e) => { if (e.key === KEY_DONE) fn(loadDone()); });
}

export function loadGlMode() {
  try { return localStorage.getItem(KEY_GL) === 'az' ? 'az' : 'group'; } catch { return 'group'; }
}

export function saveGlMode(m) {
  try { localStorage.setItem(KEY_GL, m); } catch { /* 忽略 */ }
}
