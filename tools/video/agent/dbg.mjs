// 调试：跳到第 t 秒，在页面里执行一段表达式并打印结果（可顺带截图）。
//   node tools/video/agent/dbg.mjs 84.5 "document.querySelectorAll('.crow').length" [shot.jpg]
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_CORE || '/home/mtzn/cjc/human-atlas-face-lab/node_modules/playwright-core');
const [t, expr, shotFile] = process.argv.slice(2);
const env = { ...process.env, LD_LIBRARY_PATH: `${process.env.HOME}/.local/lib/chromium-deps/root/usr/lib/x86_64-linux-gnu` };
const browser = await chromium.launch({ env, args: ['--disable-gpu', '--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--hide-scrollbars'] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.log('ERR', String(e)));
page.on('response', (r) => { if (r.status() >= 400) console.log('HTTP', r.status(), r.url()); });
await page.goto('http://127.0.0.1:8799/tools/video/agent/film.html?fps=30&lang=zh', { waitUntil: 'load' });
await page.waitForFunction(() => window.__film && (window.__film.ready || window.__film.error), null, { timeout: 120000 });
await page.evaluate((tt) => window.__film.seek(tt), Number(t));
if (shotFile) await page.screenshot({ path: shotFile, type: 'jpeg', quality: 85 });
console.log(JSON.stringify(await page.evaluate(expr), null, 1));
await browser.close();
