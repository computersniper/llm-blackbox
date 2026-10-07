// 片尾用的智能体页录屏：无头 Chromium 打开真实的 public/agent/index.html，像用户一样操作一遍
// （点任务卡片“算出销售冠军”→ 看 agent 干活 → 点“揭开内部”→ 点第 4 圈 → ＋ 上下文 → ＋ 逐词元生成），
// 用 CDP 的 screencast 录下来，按 30 fps 等间隔重采样成 sitecap/f0000.jpg…，连同每次点击的时刻和位置写进 sitecap/meta.json。
//
//   node tools/video/agent/sitecap.mjs [--out /mnt/d/cjc/videos/agent/sitecap] [--site http://127.0.0.1:8799/public/agent/] [--dry]
//
// 走 CPU（swiftshader）：这一页只有 DOM 和一张 2D 背景，不需要显卡。
// 页面的无衬线字体在这台机器上会落到系统里的微软雅黑：录屏时把 “PingFang SC” 指向思源黑体（SIL OFL，D 盘 /ext/fonts/），
// 网页本身不改。顶栏的站点导航（其他页面）和知识碎片的弹出提示也收起来：片尾只讲智能体页。
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_CORE || '/home/mtzn/cjc/human-atlas-face-lab/node_modules/playwright-core');

const args = process.argv.slice(2);
const OUT = args.includes('--out') ? args[args.indexOf('--out') + 1] : '/mnt/d/cjc/videos/agent/sitecap';
const DRY = args.includes('--dry');
const SITE = args.includes('--site') ? args[args.indexOf('--site') + 1] : 'http://127.0.0.1:8799/public/agent/';
const VW = 1280, VH = 720, DPR = 1.5, FPS = 30;
const env = { ...process.env, LD_LIBRARY_PATH: `${process.env.HOME}/.local/lib/chromium-deps/root/usr/lib/x86_64-linux-gnu` };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({ env, args: ['--disable-gpu', '--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--hide-scrollbars', '--force-color-profile=srgb', '--disable-renderer-backgrounding', '--disable-background-timer-throttling'] });
const page = await browser.newPage({ viewport: { width: VW, height: VH }, deviceScaleFactor: DPR });
const errs = [];
page.on('pageerror', (e) => errs.push(String(e)));
await page.addInitScript(() => {
  try { localStorage.clear(); } catch {}
  addEventListener('DOMContentLoaded', () => {
    const st = document.createElement('style');
    st.textContent = `@font-face { font-family: "PingFang SC"; src: url("/ext/fonts/NotoSansSC-VF.ttf") format("truetype"); font-weight: 100 900; }
      .site-nav, #toasts { display: none !important; }`;
    document.head.appendChild(st);
  });
});
await page.goto(`${SITE}index.html?lang=zh`, { waitUntil: 'load' });
await page.waitForSelector('.task-card', { timeout: 60000 });
await page.evaluate(() => document.fonts.ready);
await sleep(1500);

const cdp = await page.context().newCDPSession(page);
const frames = [];
let writing = Promise.resolve();
if (!DRY) {
  fs.mkdirSync(OUT, { recursive: true });
  for (const f of fs.readdirSync(OUT)) if (/^(raw|f)\d+\.jpg$/.test(f)) fs.unlinkSync(path.join(OUT, f));
  cdp.on('Page.screencastFrame', (e) => {
    const k = frames.length;
    const file = path.join(OUT, `raw${String(k).padStart(5, '0')}.jpg`);
    frames.push({ t: e.metadata.timestamp, file });
    const buf = Buffer.from(e.data, 'base64');
    writing = writing.then(() => fs.promises.writeFile(file, buf));
    cdp.send('Page.screencastFrameAck', { sessionId: e.sessionId }).catch(() => {});
  });
  await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 92, maxWidth: VW * DPR, maxHeight: VH * DPR, everyNthFrame: 1 });
}
const t0 = Date.now() / 1000;
const actions = [];
const now = () => Date.now() / 1000 - t0;
const state = () => page.evaluate(() => { const T = window.__ag?.tl; return T ? { depth: T.depth, t: T.t, step: JSON.stringify(T.step), playing: T.playing, done: T.done } : null; });
async function click(name, sel) {
  const el = await page.$(sel);
  if (!el) throw new Error(`没有找到 ${sel}`);
  await el.scrollIntoViewIfNeeded();
  const b = await el.boundingBox();
  const x = b.x + b.width / 2, y = b.y + b.height / 2;
  await page.mouse.move(x, y, { steps: 6 });
  await sleep(150);
  actions.push({ name, t: now(), x, y });
  await page.mouse.click(x, y);
}

await sleep(1200);
// 1. 点任务卡片「算出销售冠军」，看 agent 干活（回放 3 倍速，片子里再按需要变速）
await click('card', '.task-card[data-q="2"]');
await page.waitForFunction(() => window.__ag?.tl, null, { timeout: 30000 });
await sleep(1500);
await page.evaluate(() => { window.__ag.tl.speed = 3; });
await page.waitForFunction(() => window.__ag.tl.done, null, { timeout: 90000 });
actions.push({ name: 'watchEnd', t: now() });
await sleep(1200);
// 2. 点“揭开内部”：屏幕缩进沙箱，出现 agent 循环
await click('fab', '#btnFab');
await sleep(3200);
console.log('D', await state());
// 3. 点第 4 圈（读到 python: command not found 之后的那一圈）
await click('lap', '#laps .lap[data-t="3"]');
await sleep(1400);
// 4. ＋：上下文（这一圈喂给模型的整段历史）
await click('in', '#btnIn');
await page.evaluate(() => window.__ag.tl.play());
await sleep(2600);
console.log('D', await state());
// 5. ＋：逐词元生成（先走到“模型生成”这一步，再往里钻）
await page.evaluate(() => { const T = window.__ag.tl; T.pause(); for (let i = 0; i < 6 && T.step.p !== 'gen'; i++) T.next(); });
await sleep(300);
await click('in', '#btnIn');
await page.evaluate(() => window.__ag.tl.play());
await sleep(9500);
actions.push({ name: 'end', t: now() });
console.log('final', await state());

if (!DRY) {
  await cdp.send('Page.stopScreencast');
  await writing;
  const T1 = now();
  const n = Math.floor(T1 * FPS);
  let j = 0;
  for (let k = 0; k < n; k++) {
    const tk = t0 + k / FPS;
    while (j + 1 < frames.length && frames[j + 1].t <= tk) j++;
    fs.copyFileSync(frames[j].file, path.join(OUT, `f${String(k).padStart(4, '0')}.jpg`));
  }
  for (const f of frames) fs.unlinkSync(f.file);
  fs.writeFileSync(path.join(OUT, 'meta.json'), JSON.stringify({ fps: FPS, n, viewport: { w: VW, h: VH }, dpr: DPR, captured: frames.length, actions }, null, 1));
  console.log(`${frames.length} 帧录屏 → ${n} 帧（${FPS} fps，${T1.toFixed(1)} 秒，平均 ${(frames.length / T1).toFixed(1)} fps）`);
}
if (errs.length) console.log('page errors:', errs.slice(0, 5));
await browser.close();
