// 各页右上角的 GitHub 链接和访问量（六个页面共用）。
// 计数在服务器上（deploy/counter/counter.py）：全站一个总数，同一个 IP 每天只算一个访客，IP 不落盘；
// 后台另按页面记录（不在网页上显示）。本地开发没有这个接口，就只显示 GitHub 按钮。
import { L as tr, scrollNavToCurrent } from './i18n.js';

export const REPO = 'https://github.com/computersniper/llm-blackbox';
const API = new URL('../api/', import.meta.url);   // 不管哪个页面引用，都指向 /blackbox/api/

const fmt = (n) => n.toLocaleString('zh-CN');

export async function countVisit(page) {
  try {
    const r = await fetch(new URL(`hit?p=${encodeURIComponent(page)}`, API), { method: 'POST', cache: 'no-store' });
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
    pill.title = tr(`全站累计 ${fmt(s.total)} 人次来访（每人每天计一次）· 今天 ${fmt(s.today)} 人`, `${fmt(s.total)} visits across the site (each visitor counted once a day) · ${fmt(s.today)} today`);
    pill.hidden = false;
    scrollNavToCurrent();   // 胶囊占了顶栏的位置，手机上导航变窄，把当前页的链接滚回可见区
  }
}

const EYE = '<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12z" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="12" r="2.8" fill="currentColor"/></svg>';
const GITHUB = '<svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>';

// 在顶栏右侧放访问量胶囊和 GitHub 按钮（页面里已经写好的就沿用），然后记这一页的一次访问。
// page：infer / train / mm / agent / world / learn
export function mountSiteActions(page) {
  const bar = document.querySelector('.top-actions');
  if (bar && !document.getElementById('visits')) {
    const label = tr('GitHub 仓库', 'GitHub repository');
    bar.insertAdjacentHTML('afterbegin', `<span class="hud-btn visits" id="visits" hidden>${EYE}<b>0</b></span><a class="hud-btn gh-btn" href="${REPO}" target="_blank" rel="noopener" title="${label}" aria-label="${label}">${GITHUB}</a>`);
  }
  countVisit(page).then(showVisits);
}
