// 首页右上角的 GitHub 链接和访问量。
// 计数在服务器上（deploy/counter/counter.py）：同一个 IP 每天只算一个访客，IP 不落盘。本地开发没有这个接口，就什么都不显示。

export const REPO = 'https://github.com/computersniper/llm-blackbox';

const fmt = (n) => n.toLocaleString('zh-CN');

export async function countVisit() {
  try {
    const r = await fetch('api/hit', { method: 'POST', cache: 'no-store' });
    if (!r.ok) return null;
    const s = await r.json();
    return Number.isFinite(s.total) ? s : null;
  } catch {
    return null;
  }
}

// 顶栏右上角的小胶囊
export function showVisits(s) {
  if (!s) return;
  const pill = document.getElementById('visits');
  if (pill) {
    pill.querySelector('b').textContent = fmt(s.total);
    pill.title = `累计 ${fmt(s.total)} 人次来访（每人每天计一次）· 今天 ${fmt(s.today)} 人`;
    pill.hidden = false;
  }
}
