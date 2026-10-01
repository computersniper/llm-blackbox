// 逐帧渲染电影模式页面。无头 Chromium 走 WSL 的 D3D12（Mesa d3d12 → RTX GPU），每帧调用 __film.renderAt(t) 再截图。
//
//   node tools/video/render.mjs frames --out /mnt/d/cjc/videos/llm-inference/frames [--from 0] [--to 210] [--fps 30] [--workers 4]
//   node tools/video/render.mjs stills --times 12,40.5,96 --out dir      # 抽几帧看效果
//   node tools/video/render.mjs events --out tools/video/events.json      # 导出配乐用的事件和段落
//   node tools/video/render.mjs poster --out poster.png                     # 封面（默认取片名落版那一刻）
//
// 帧存成 JPEG（质量 95），已经存在的帧会跳过：中途崩了重跑同一条命令就能接着渲染。
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_CORE || '/home/mtzn/cjc/human-atlas-face-lab/node_modules/playwright-core');

const [mode = 'stills', ...rest] = process.argv.slice(2);
const A = {};
for (let i = 0; i < rest.length; i++) { const k = rest[i].replace(/^--/, ''); const v = rest[i + 1] && !rest[i + 1].startsWith('--') ? rest[++i] : true; A[k] = v; }
const FPS = Number(A.fps || 30);
const BASE = A.url || 'http://127.0.0.1:8776/tools/video/film.html';
const GPU = A.cpu ? ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'] : ['--use-angle=gl-egl', '--enable-gpu', '--ignore-gpu-blocklist'];
const HOME = process.env.HOME;
const env = {
  ...process.env,
  LD_LIBRARY_PATH: `/usr/lib/wsl/lib:${HOME}/.local/lib/chromium-deps/root/usr/lib/x86_64-linux-gnu`,
  GALLIUM_DRIVER: 'd3d12',
  MESA_D3D12_DEFAULT_ADAPTER_NAME: 'NVIDIA',
};

async function open(extra = '') {
  const browser = await chromium.launch({ env, args: [...GPU, '--disable-renderer-backgrounding', '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--hide-scrollbars', '--force-color-profile=srgb'] });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const errs = [];
  page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
  page.on('pageerror', (e) => errs.push(String(e)));
  await page.goto(`${BASE}?fps=${FPS}${extra}`, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__film && (window.__film.ready || window.__film.error), null, { timeout: 120000 });
  const err = await page.evaluate(() => window.__film.error);
  if (err) throw new Error(err);
  const info = await page.evaluate(() => ({ dur: window.__film.duration, gl: (() => { const c = document.querySelector('#gl canvas'); const g = c.getContext('webgl2'); const e = g.getExtension('WEBGL_debug_renderer_info'); return e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : '?'; })() }));
  const cdp = await page.context().newCDPSession(page);
  return { browser, page, cdp, errs, info };
}

async function shot(cdp, file) {
  const { data } = await cdp.send('Page.captureScreenshot', { format: file.endsWith('.png') ? 'png' : 'jpeg', quality: 95, optimizeForSpeed: !file.endsWith('.png'), clip: { x: 0, y: 0, width: 1920, height: 1080, scale: 1 } });
  fs.writeFileSync(file, Buffer.from(data, 'base64'));
}

if (mode === 'events') {
  const { browser, page, info } = await open();
  const ev = await page.evaluate(() => ({ events: window.__film.events(), score: window.__film.score, duration: window.__film.duration, fps: window.__film.fps }));
  fs.writeFileSync(A.out || 'events.json', JSON.stringify(ev, null, 1));
  console.log(`events: ${ev.events.length} · duration ${ev.duration}s · ${info.gl}`);
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
  console.log(`GL: ${info.gl} · duration ${info.dur}`);
  for (const t of times) {
    const t0 = Date.now();
    await page.evaluate((tt) => window.__film.seek(tt, 8), t);
    const f = path.join(out, `t${t.toFixed(2).padStart(7, '0')}.${A.png ? 'png' : 'jpg'}`);
    await shot(cdp, f);
    console.log(`${f}  (${Date.now() - t0} ms)`);
    if (A.probe) console.log(JSON.stringify(await page.evaluate(() => window.__film.probe())));
  }
  if (errs.length) console.log('console errors:\n' + errs.slice(0, 20).join('\n'));
  await browser.close();
} else if (mode === 'frames') {
  const out = A.out;
  fs.mkdirSync(out, { recursive: true });
  const probe = await open();
  const dur = Number(A.to || probe.info.dur);
  console.log(`GL: ${probe.info.gl} · duration ${dur}s @ ${FPS} fps`);
  await probe.browser.close();
  const f0 = Math.round(Number(A.from || 0) * FPS), f1 = Math.round(dur * FPS);
  const W = Math.max(1, Math.min(6, Number(A.workers || 4)));
  const name = (f) => path.join(out, `f${String(f).padStart(5, '0')}.jpg`);
  const todo = [];
  for (let f = f0; f < f1; f++) if (!fs.existsSync(name(f))) todo.push(f);
  console.log(`${todo.length} frames to render (${f1 - f0 - todo.length} already there) with ${W} workers`);
  // 把要渲染的帧切成连续的块分给各个进程（每块开头做一次 8 秒的预滚，让平滑动画收敛）
  const chunks = [];
  const per = Math.ceil(todo.length / W);
  for (let w = 0; w < W; w++) { const c = todo.slice(w * per, (w + 1) * per); if (c.length) chunks.push(c); }
  const t0 = Date.now();
  let done = 0;
  await Promise.all(chunks.map(async (frames, w) => {
    let { browser, page, cdp, errs } = await open();
    let prev = -10;
    // GPU 进程偶尔会崩（显卡和别的任务共用）：WebGL 上下文丢了以后画面全白、不会自己恢复。
    // 每帧检查一次，丢了就重开浏览器，从这一帧重新预滚
    const lost = () => page.evaluate(() => document.querySelector('#gl canvas').getContext('webgl2').isContextLost());
    for (const f of frames) {
      const t = f / FPS;
      for (let tries = 0; ; tries++) {
        if (f !== prev + 1) await page.evaluate((tt) => window.__film.seek(tt, 8), t);
        else await page.evaluate((tt) => window.__film.renderAt(tt), t);
        if (!(await lost())) break;
        if (tries >= 3) throw new Error(`worker ${w}: WebGL context lost at frame ${f}`);
        console.log(`worker ${w}: WebGL context lost at frame ${f} (t=${t.toFixed(2)}), restarting browser`);
        await browser.close().catch(() => {});
        ({ browser, page, cdp, errs } = await open());
        prev = -10;
      }
      await shot(cdp, name(f) + '.tmp.jpg');
      fs.renameSync(name(f) + '.tmp.jpg', name(f));
      prev = f;
      done++;
      if (done % 150 === 0) {
        const el = (Date.now() - t0) / 1000;
        console.log(`${done}/${todo.length} frames · ${(done / el).toFixed(1)} fps · eta ${((todo.length - done) / (done / el) / 60).toFixed(1)} min`);
      }
    }
    if (errs.length) console.log(`worker ${w} console errors:\n` + errs.slice(0, 10).join('\n'));
    await browser.close();
  }));
  console.log(`done in ${((Date.now() - t0) / 60000).toFixed(1)} min`);
}
