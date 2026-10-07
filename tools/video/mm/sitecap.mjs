// 片尾用的多模态页录屏（从 ../sitecap.mjs 改来）：无头 Chromium 打开真实的 public/multimodal/index.html，像用户一样操作一遍
// （选图 → 点选词元拼出问题 → 发送 → 鼠标放在回答的「3」上看热力图 → 揭开这条回复 → 点 ＋ 一层层钻进去 → 监视器上的热力图），
// 用 CDP screencast 录下来，按 30 fps 等间隔重采样成 sitecap/f0000.jpg…，连同每次点击 / 悬停的时刻和位置写进 sitecap/meta.json。
//
//   node tools/video/mm/sitecap.mjs [--out /mnt/d/cjc/videos/multimodal/sitecap] [--site http://127.0.0.1:8798/public/multimodal/] [--dry]
//
// 只用 CPU（SwiftShader，--disable-gpu）。软件渲染比真人的电脑慢：为了让录屏里的 3D 动画不卡，页面里的时间放慢 SLOW 倍
// （performance.now / Date.now / requestAnimationFrame / setTimeout 的时间都按 1/SLOW 走），meta.json 里的时间换算回页面时间，
// 电影页按页面时间剪辑，看起来就是正常速度。
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_CORE || '/home/mtzn/cjc/human-atlas-face-lab/node_modules/playwright-core');

const args = process.argv.slice(2);
const arg = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const OUT = arg('--out', '/mnt/d/cjc/videos/multimodal/sitecap');
const DRY = args.includes('--dry');
const SITE = arg('--site', 'http://127.0.0.1:8798/public/multimodal/');
const SLOW = Number(arg('--slow', '4'));
const VW = 1280, VH = 720, DPR = 1.5, FPS = 30;
const env = { ...process.env, LD_LIBRARY_PATH: `${process.env.HOME}/.local/lib/chromium-deps/root/usr/lib/x86_64-linux-gnu` };
const sleepReal = (ms) => new Promise((r) => setTimeout(r, ms));
const sleep = (ms) => sleepReal(ms * SLOW);   // 按页面时间等

const browser = await chromium.launch({ env, args: ['--disable-gpu', '--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--hide-scrollbars', '--force-color-profile=srgb', '--disable-renderer-backgrounding', '--disable-background-timer-throttling'] });
const page = await browser.newPage({ viewport: { width: VW, height: VH }, deviceScaleFactor: DPR });
const errs = [];
page.on('pageerror', (e) => errs.push(String(e)));
await page.addInitScript((slow) => {
  // 页面时间放慢 slow 倍（软件渲染跟不上实时；录完按页面时间重采样）
  const pn = performance.now.bind(performance), dn = Date.now, t0p = pn(), t0d = dn();
  performance.now = () => t0p + (pn() - t0p) / slow;
  Date.now = () => t0d + (dn() - t0d) / slow;
  const raf = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) => raf(() => cb(performance.now()));
  const st = window.setTimeout.bind(window), si = window.setInterval.bind(window);
  window.setTimeout = (fn, ms = 0, ...a) => st(fn, ms * slow, ...a);
  window.setInterval = (fn, ms = 0, ...a) => si(fn, ms * slow, ...a);
  window.__slow = slow;
  addEventListener('DOMContentLoaded', () => {
    const s = document.createElement('style');
    // 片子里不需要的东西：第一次揭开时的操作说明、知识碎片的弹出提示、顶栏的站点导航（片尾只讲多模态页）
    s.textContent = '#stageHint, #toasts, .site-nav { display: none !important; }';
    document.head.appendChild(s);
  });
}, SLOW);
await page.goto(`${SITE}index.html?lang=zh`, { waitUntil: 'load' });
await page.waitForSelector('.pic', { timeout: 60000 });
await sleepReal(2500);
// CSS 过渡 / 动画也放慢：用 CDP 的动画播放速率
const cdp = await page.context().newCDPSession(page);
await cdp.send('Animation.enable');
await cdp.send('Animation.setPlaybackRate', { playbackRate: 1 / SLOW });

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
  await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 90, maxWidth: VW * DPR, maxHeight: VH * DPR, everyNthFrame: 1 });
}
const t0 = Date.now() / 1000;
const actions = [];
const now = () => (Date.now() / 1000 - t0) / SLOW;   // 页面时间
async function at(name, sel, { click = true } = {}) {
  const el = await page.$(sel);
  if (!el) throw new Error(`没有找到 ${sel}`);
  const b = await el.boundingBox();
  const x = b.x + b.width / 2, y = b.y + b.height / 2;
  await page.mouse.move(x, y, { steps: 6 });
  await sleep(120);
  actions.push({ name, t: now(), x, y, click });
  if (click) await page.mouse.click(x, y);
}
const state = () => page.evaluate(() => { const T = window.__mm?.tl; return T ? { depth: T.depth, can: T.canInto(), playing: T.playing, step: JSON.stringify(T.step) } : null; });

await sleep(600);
// 1. 选图：桌上的苹果
const MAN = await (await fetch(`${SITE}data/manifest.json`)).json();
const imIdx = MAN.images.findIndex((im) => im.id === 'apples');
const q = MAN.images[imIdx].questions.find((x) => x.id === 'apples-2');
await at('pick', `.pic[data-i="${imIdx}"]`);
await sleep(700);
// 2. 用候选词元拼出「绿色的苹果是第几个？」，发送
for (const c of q.chips) { await at('chip', `.cand[data-id="${c.id}"]`); await sleep(280); }
await sleep(300);
await at('send', '#btnSend');
// 3. 等回答写完
await page.waitForFunction(() => !document.querySelector('#btnPlus').disabled, null, { timeout: 120000, polling: 200 });
actions.push({ name: 'replyDone', t: now() });
await sleep(700);
// 4. 鼠标放在回答里的「3」上：图上亮起生成它时的注意力
const gi = q.replyTokens.findIndex((t) => t.s.trim() === '3');
await at('hover', `.msg.bot .rt[data-i="${gi}"]`, { click: false });
await sleep(2200);
actions.push({ name: 'hoverEnd', t: now() });
// 5. 揭开这条回复
await at('plus', '.msg.bot .peek');
await page.waitForFunction(() => document.querySelector('#loading').hidden && window.__mm?.tl, null, { timeout: 120000, polling: 200 });
await sleep(2000);
console.log('D', await state());
// 6. 一层层往里钻：流水线 → 逐层 → 一层之内 → 一次乘加
for (let d = 2; d <= 5; d++) {
  for (let tries = 0; tries < 40; tries++) {
    const s = await state();
    if (s.can) break;
    await page.evaluate(() => window.__mm.tl.next());
    await sleep(60);
  }
  await at('in', '#btnIn');
  await sleep(d >= 4 ? 2000 : 1700);
  console.log('D', await state());
}
// 7. 回到“逐层”，跳到生成「3」时的第 20 层：监视器和层板上是真实的注意力热力图
await page.evaluate((g) => { const T = window.__mm.tl; T.pause(); T.setDepth(3); T.seekToken(g); T.seek((s) => s.ph === 'layer' && s.L === 20); }, gi);
await sleep(300);
actions.push({ name: 'mon', t: now() });
await sleep(800);
await at('monhover', '#monCv', { click: false });
await sleep(2000);
actions.push({ name: 'end', t: now() });
console.log('final', await state());

if (!DRY) {
  await cdp.send('Page.stopScreencast');
  await writing;
  // 按页面时间 30 fps 等间隔重采样：每个输出帧取它之前最近的一帧
  const T1 = now();
  const n = Math.floor(T1 * FPS);
  let j = 0;
  for (let k = 0; k < n; k++) {
    const tk = t0 + (k / FPS) * SLOW;
    while (j + 1 < frames.length && frames[j + 1].t <= tk) j++;
    fs.copyFileSync(frames[j].file, path.join(OUT, `f${String(k).padStart(4, '0')}.jpg`));
  }
  for (const f of frames) fs.unlinkSync(f.file);
  fs.writeFileSync(path.join(OUT, 'meta.json'), JSON.stringify({ fps: FPS, n, slow: SLOW, viewport: { w: VW, h: VH }, dpr: DPR, captured: frames.length, actions }, null, 1));
  console.log(`${frames.length} 帧录屏 → ${n} 帧（${FPS} fps，页面时间 ${T1.toFixed(1)} 秒，实际 ${(T1 * SLOW).toFixed(0)} 秒）`);
}
if (errs.length) console.log('page errors:', errs.slice(0, 5));
await browser.close();
