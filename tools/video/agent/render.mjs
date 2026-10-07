// 逐帧渲染智能体视频的电影模式页面（由 ../render.mjs 改写：这支片子全是 DOM + 2D canvas，默认走 CPU / swiftshader，不碰显卡）。
//
//   node tools/video/agent/render.mjs frames --out /mnt/d/cjc/videos/agent/frames30 [--from 0] [--to 185] [--fps 30] [--workers 3]
//   node tools/video/agent/render.mjs stills --times 12,40.5,96 --out dir      # 抽几帧看效果（--png 存 PNG）
//   node tools/video/agent/render.mjs events --out tools/video/agent/events.json   # 导出配乐用的事件和段落
//   node tools/video/agent/render.mjs poster --out poster.png                  # 封面（默认取片名落版那一刻）
//
// 帧存成 JPEG（质量 95），已经存在的帧会跳过：中途停了重跑同一条命令就能接着渲染。
// 同时最多 3 个浏览器进程（WSL 下负载太高会整机崩溃）。
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_CORE || '/home/mtzn/cjc/human-atlas-face-lab/node_modules/playwright-core');

const [mode = 'stills', ...rest] = process.argv.slice(2);
const A = {};
for (let i = 0; i < rest.length; i++) { const k = rest[i].replace(/^--/, ''); const v = rest[i + 1] && !rest[i + 1].startsWith('--') ? rest[++i] : true; A[k] = v; }
const FPS = Number(A.fps || 30);
const BASE = A.url || 'http://127.0.0.1:8799/tools/video/agent/film.html';
const HOME = process.env.HOME;
// 不加 /usr/lib/wsl/lib：只用 CPU 渲染
const env = { ...process.env, LD_LIBRARY_PATH: `${HOME}/.local/lib/chromium-deps/root/usr/lib/x86_64-linux-gnu` };
const ARGS = ['--disable-gpu', '--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--disable-renderer-backgrounding', '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--hide-scrollbars', '--force-color-profile=srgb', '--font-render-hinting=none'];

async function open() {
  const browser = await chromium.launch({ env, args: ARGS });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const errs = [];
  page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
  page.on('pageerror', (e) => errs.push(String(e)));
  await page.goto(`${BASE}?fps=${FPS}&lang=zh`, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__film && (window.__film.ready || window.__film.error), null, { timeout: 120000 });
  const err = await page.evaluate(() => window.__film.error);
  if (err) throw new Error(err);
  const info = await page.evaluate(() => ({ dur: window.__film.duration }));
  const cdp = await page.context().newCDPSession(page);
  return { browser, page, cdp, errs, info };
}

async function shot(cdp, file) {
  const { data } = await cdp.send('Page.captureScreenshot', { format: file.endsWith('.png') ? 'png' : 'jpeg', quality: 95, optimizeForSpeed: !file.endsWith('.png'), clip: { x: 0, y: 0, width: 1920, height: 1080, scale: 1 } });
  fs.writeFileSync(file, Buffer.from(data, 'base64'));
}

if (mode === 'events') {
  const { browser, page } = await open();
  const ev = await page.evaluate(() => ({ events: window.__film.events(), score: window.__film.score, duration: window.__film.duration, fps: window.__film.fps }));
  fs.writeFileSync(A.out || 'events.json', JSON.stringify(ev, null, 1));
  console.log(`events: ${ev.events.length} · duration ${ev.duration}s`);
  await browser.close();
} else if (mode === 'poster') {
  const { browser, page, cdp } = await open();
  await page.evaluate((tt) => window.__film.poster(tt), A.t ? Number(A.t) : null);
  await shot(cdp, A.out || 'poster.png');
  console.log(`poster: ${A.out}`);
  await browser.close();
} else if (mode === 'stills') {
  const out = A.out || '.';
  fs.mkdirSync(out, { recursive: true });
  const times = String(A.times).split(',').map(Number);
  const { browser, page, cdp, errs, info } = await open();
  console.log(`duration ${info.dur}`);
  for (const t of times) {
    const t0 = Date.now();
    await page.evaluate((tt) => window.__film.seek(tt), t);
    const f = path.join(out, `t${t.toFixed(2).padStart(7, '0')}.${A.png ? 'png' : 'jpg'}`);
    await shot(cdp, f);
    console.log(`${f}  (${Date.now() - t0} ms)`);
  }
  if (errs.length) console.log('console errors:\n' + errs.slice(0, 20).join('\n'));
  await browser.close();
} else if (mode === 'frames') {
  const out = A.out;
  fs.mkdirSync(out, { recursive: true });
  const probe = await open();
  const dur = Number(A.to || probe.info.dur);
  console.log(`duration ${dur}s @ ${FPS} fps`);
  await probe.browser.close();
  const f0 = Math.round(Number(A.from || 0) * FPS), f1 = Math.round(dur * FPS);
  const W = Math.max(1, Math.min(3, Number(A.workers || 3)));
  const name = (f) => path.join(out, `f${String(f).padStart(5, '0')}.jpg`);
  const todo = [];
  for (let f = f0; f < f1; f++) if (!fs.existsSync(name(f))) todo.push(f);
  console.log(`${todo.length} frames to render (${f1 - f0 - todo.length} already there) with ${W} workers`);
  const chunks = [];
  const per = Math.ceil(todo.length / W);
  for (let w = 0; w < W; w++) { const c = todo.slice(w * per, (w + 1) * per); if (c.length) chunks.push(c); }
  const t0 = Date.now();
  let done = 0;
  await Promise.all(chunks.map(async (frames, w) => {
    const { browser, page, cdp, errs } = await open();
    for (const f of frames) {
      await page.evaluate((tt) => window.__film.renderAt(tt), f / FPS);
      await shot(cdp, name(f) + '.tmp.jpg');
      fs.renameSync(name(f) + '.tmp.jpg', name(f));
      done++;
      if (done % 300 === 0) {
        const el = (Date.now() - t0) / 1000;
        console.log(`${done}/${todo.length} frames · ${(done / el).toFixed(1)} fps · eta ${((todo.length - done) / (done / el) / 60).toFixed(1)} min`);
      }
    }
    if (errs.length) console.log(`worker ${w} console errors:\n` + errs.slice(0, 10).join('\n'));
    await browser.close();
  }));
  console.log(`done in ${((Date.now() - t0) / 60000).toFixed(1)} min`);
}
