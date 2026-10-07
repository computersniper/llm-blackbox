// 训练视频片尾用的训练页录屏：无头 Chromium 打开真实的 public/train/index.html（第一章玻璃小模型），像用户一样操作一遍
// （出厂状态 → 点 ＋ 开始训练 → 一路点 ＋ 钻到一个参数 → 播放 / 暂停 / 单步），用 CDP 的 screencast 录下来，
// 再按 30 fps 等间隔重采样成 f0000.jpg…，连同每次点击的时刻和位置写进 meta.json（和推理页的 sitecap.mjs 同一种格式）。
//
//   node tools/video/train/sitecap.mjs [--out /mnt/d/cjc/videos/train-glass/sitecap-train] [--cpu] [--dry]
//
// 录屏本身是实时的（和真人录屏一样），之后电影页按 meta.json 里的时间表剪辑、变速、叠加鼠标。
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_CORE || '/home/mtzn/cjc/human-atlas-face-lab/node_modules/playwright-core');

const args = process.argv.slice(2);
const OUT = args.includes('--out') ? args[args.indexOf('--out') + 1] : '/mnt/d/cjc/videos/train-glass/sitecap-train';
const DRY = args.includes('--dry');
const SITE = args.includes('--site') ? args[args.indexOf('--site') + 1] : 'http://127.0.0.1:8776/public/train/';
const VW = 1280, VH = 720, DPR = 1.5, FPS = 30;
const HOME = process.env.HOME;
const env = { ...process.env, LD_LIBRARY_PATH: `/usr/lib/wsl/lib:${HOME}/.local/lib/chromium-deps/root/usr/lib/x86_64-linux-gnu`, GALLIUM_DRIVER: 'd3d12', MESA_D3D12_DEFAULT_ADAPTER_NAME: 'NVIDIA' };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --cpu：用 swiftshader（不碰显卡），只适合 --dry 试走一遍流程；真正录屏要 GPU，否则帧率太低
const GPU = args.includes('--cpu') ? ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'] : ['--use-angle=gl-egl', '--enable-gpu', '--ignore-gpu-blocklist'];
const browser = await chromium.launch({ env, args: [...GPU, '--hide-scrollbars', '--force-color-profile=srgb', '--disable-renderer-backgrounding', '--disable-background-timer-throttling'] });
const page = await browser.newPage({ viewport: { width: VW, height: VH }, deviceScaleFactor: DPR });
const errs = [];
page.on('pageerror', (e) => errs.push(String(e)));
// 片子里不需要的临时提示：第一次揭开时的操作说明、知识碎片的弹出提示
await page.addInitScript(() => {
  addEventListener('DOMContentLoaded', () => {
    const st = document.createElement('style');
    // 顶栏里的站点导航（其他页面）也收起：片尾只讲推理页
    st.textContent = '#stageHint, #toasts, .site-nav { display: none !important; }';
    try { localStorage.setItem('blackbox:train', JSON.stringify({ hinted: true, speed: 1 })); } catch {}
    document.head.appendChild(st);
  });
});
await page.goto(`${SITE}index.html?lang=zh`, { waitUntil: 'load' });
await page.waitForFunction(() => window.__train?.tl && window.__train.g3 && document.querySelector('#loading').hidden, null, { timeout: 90000 });
await sleep(2500);

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
const state = () => page.evaluate(() => { const T = window.__train?.tl; return T ? { depth: T.depth, can: T.canInto(), playing: T.playing, step: JSON.stringify(T.step) } : null; });
async function click(name, sel) {
  const el = await page.$(sel);
  if (!el) throw new Error(`没有找到 ${sel}`);
  const b = await el.boundingBox();
  const x = b.x + b.width / 2, y = b.y + b.height / 2;
  await page.mouse.move(x, y);
  await sleep(120);
  actions.push({ name, t: now(), x, y });
  await page.mouse.click(x, y);
}

await sleep(1600);
// 1. 出厂状态（D0）→ 点 ＋：开始训练（D1，机器一批批吃数据）
console.log('D', await state());
await click('in', '#btnIn');
await sleep(3400);
console.log('D', await state());
// 2. 一路往里钻：一步之内 → 一层之内 → 一块矩阵 → 一个参数（这一步没有更深一层时先往后走到有的那一步）
for (let d = 2; d <= 5; d++) {
  for (let tries = 0; tries < 60; tries++) {
    const s = await state();
    if (s.can) break;
    await page.evaluate(() => window.__train.tl.next());
    await sleep(60);
  }
  await click('in', '#btnIn');
  await sleep(d >= 4 ? 2000 : 1700);
  console.log('D', await state());
}
// 3. 播放 → 暂停 → 单步、单步
await page.evaluate(() => window.__train.tl.pause());
await sleep(500);
await click('play', '#btnPlay');
await sleep(2400);
await click('pause', '#btnPlay');
await sleep(700);
await click('step', '#btnNextStep');
await sleep(1100);
await click('step', '#btnNextStep');
await sleep(1300);
actions.push({ name: 'end', t: now() });
console.log('final', await state());

if (!DRY) {
  await cdp.send('Page.stopScreencast');
  await writing;
  // 按 30 fps 等间隔重采样：每个输出帧取它之前最近的一帧
  const T1 = now();
  const n = Math.floor(T1 * FPS);
  let j = 0;
  for (let k = 0; k < n; k++) {
    const tk = t0 + k / FPS;
    while (j + 1 < frames.length && frames[j + 1].t <= tk) j++;
    const src = frames[j];
    fs.copyFileSync(src.file, path.join(OUT, `f${String(k).padStart(4, '0')}.jpg`));
  }
  for (const f of frames) fs.unlinkSync(f.file);
  fs.writeFileSync(path.join(OUT, 'meta.json'), JSON.stringify({ fps: FPS, n, viewport: { w: VW, h: VH }, dpr: DPR, captured: frames.length, actions }, null, 1));
  console.log(`${frames.length} 帧录屏 → ${n} 帧（${FPS} fps，${T1.toFixed(1)} 秒）`);
}
if (errs.length) console.log('page errors:', errs.slice(0, 5));
await browser.close();
