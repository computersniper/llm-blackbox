// 逐帧渲染多模态视频的电影页（从 ../render.mjs 改来）。只用 CPU：Chromium 走 SwiftShader（--disable-gpu），不碰显卡，
// 和别的 CUDA 任务互不影响。每帧调用 __film.renderAt(t) 再截图。
//
//   node tools/video/mm/render.mjs frames --out /mnt/d/cjc/videos/multimodal/frames [--from 0] [--to 180] [--fps 30] [--workers 3]
//   node tools/video/mm/render.mjs stills --times 12,40.5,96 --out dir      # 抽几帧看效果
//   node tools/video/mm/render.mjs events --out tools/video/mm/events.json   # 导出配乐用的事件和段落
//   node tools/video/mm/render.mjs poster --out poster.png [--t 123.7]       # 封面（默认：生成「3」时的热力图 + 片名）
//
// 帧存成 JPEG（质量 95），已经存在的帧会跳过：中途停了重跑同一条命令就能接着渲染。
// 同时最多 3 个浏览器进程（--workers 上限 3）：WSL 瞬时负载太高会整机崩溃。
// 渲染中按 /proc/loadavg 自动让路：扣掉自己这几个进程的负载（每个 SwiftShader 浏览器约 6），
// 让总负载尽量不超过 14——别人的负载越高，同时在渲的进程越少，高到 8.5 以上就全部暂停，等降下来再接着渲（每 30 帧检查一次）。
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_CORE || '/home/mtzn/cjc/human-atlas-face-lab/node_modules/playwright-core');

const [mode = 'stills', ...rest] = process.argv.slice(2);
const A = {};
for (let i = 0; i < rest.length; i++) { const k = rest[i].replace(/^--/, ''); const v = rest[i + 1] && !rest[i + 1].startsWith('--') ? rest[++i] : true; A[k] = v; }
const FPS = Number(A.fps || 30);
const BASE = A.url || 'http://127.0.0.1:8798/tools/video/mm/film.html';
const HOME = process.env.HOME;
const env = { ...process.env, LD_LIBRARY_PATH: `${HOME}/.local/lib/chromium-deps/root/usr/lib/x86_64-linux-gnu` };
const ARGS = ['--disable-gpu', '--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--disable-renderer-backgrounding', '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--hide-scrollbars', '--force-color-profile=srgb'];

async function open(extra = '') {
  const browser = await chromium.launch({ env, args: ARGS });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const errs = [];
  page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
  page.on('pageerror', (e) => errs.push(String(e)));
  // 网站的 i18n 没有 ?lang= 时按浏览器语言猜（无头浏览器是英文），所以明确写 lang=zh
  await page.goto(`${BASE}?fps=${FPS}&lang=zh${A.pr ? `&pr=${A.pr}` : ''}${extra}`, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__film && (window.__film.ready || window.__film.error), null, { timeout: 180000 });
  const err = await page.evaluate(() => window.__film.error);
  if (err) throw new Error(err);
  const info = await page.evaluate(() => ({ dur: window.__film.duration }));
  const cdp = await page.context().newCDPSession(page);
  return { browser, page, cdp, errs, info };
}

async function shot(cdp, file) {
  const png = file.endsWith('.png');
  const { data } = await cdp.send('Page.captureScreenshot', { format: png ? 'png' : 'jpeg', ...(png ? {} : { quality: 95 }), optimizeForSpeed: !png, clip: { x: 0, y: 0, width: 1920, height: 1080, scale: 1 } });
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
    await page.evaluate(([tt, pre]) => window.__film.seek(tt, pre), [t, Number(A.pre ?? 4)]);
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
  console.log(`duration ${dur}s @ ${FPS} fps`);
  await probe.browser.close();
  const f0 = Math.round(Number(A.from || 0) * FPS), f1 = Math.round(dur * FPS);
  const W = Math.max(1, Math.min(3, Number(A.workers || 3)));
  const name = (f) => path.join(out, `f${String(f).padStart(5, '0')}.jpg`);
  const todo = [];
  for (let f = f0; f < f1; f++) if (!fs.existsSync(name(f))) todo.push(f);
  console.log(`${todo.length} frames to render (${f1 - f0 - todo.length} already there) with ${W} workers`);
  // 把要渲染的帧切成连续的块分给各个进程（每块开头做一次预滚，让平滑动画收敛）
  const chunks = [];
  const per = Math.ceil(todo.length / W);
  for (let w = 0; w < W; w++) { const c = todo.slice(w * per, (w + 1) * per); if (c.length) chunks.push(c); }
  const t0 = Date.now();
  let done = 0;
  const PER = Number(A.perload || 6);
  let active = 0, lastNote = 0;
  const load1 = () => Number(fs.readFileSync('/proc/loadavg', 'utf8').split(' ')[0]);
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  async function gate(w) {
    for (;;) {
      const l = load1(), others = l - PER * active;
      // 1 分钟负载有滞后：自己刚停下的进程还算在里面。别人的负载不到 12 时至少留 1 个在渲，免得走走停停
      const allow = others >= 12 ? 0 : Math.min(W, Math.max(1, Math.floor((14.5 - Math.max(0, others)) / PER)));
      if (active < allow) { active++; return; }
      if (Date.now() - lastNote > 60000) { lastNote = Date.now(); console.log(`load ${l.toFixed(1)}（别人约 ${others.toFixed(1)}）：先让一让，${active} 个在渲`); }
      await wait(15000 + w * 1000);
    }
  }
  await Promise.all(chunks.map(async (frames, w) => {
    await wait(w * 20000);   // 错开启动，免得负载一下子冲上去
    await gate(w);
    const { browser, page, cdp, errs } = await open();
    let prev = -10, n = 0;
    for (const f of frames) {
      if (++n % 30 === 0) { active--; await gate(w); }
      const t = f / FPS;
      if (f !== prev + 1) await page.evaluate((tt) => window.__film.seek(tt, 4), t);
      else await page.evaluate((tt) => window.__film.renderAt(tt), t);
      await shot(cdp, name(f) + '.tmp.jpg');
      fs.renameSync(name(f) + '.tmp.jpg', name(f));
      prev = f;
      done++;
      if (done % 90 === 0) {
        const el = (Date.now() - t0) / 1000;
        console.log(`${done}/${todo.length} frames · ${(done / el).toFixed(2)} fps · eta ${((todo.length - done) / (done / el) / 60).toFixed(1)} min`);
      }
    }
    active--;
    if (errs.length) console.log(`worker ${w} console errors:\n` + errs.slice(0, 10).join('\n'));
    await browser.close();
  }));
  console.log(`done in ${((Date.now() - t0) / 60000).toFixed(1)} min`);
}
