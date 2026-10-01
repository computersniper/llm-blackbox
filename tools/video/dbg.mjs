// 调试用：在第 t 秒执行一段表达式并打印结果。node tools/video/dbg.mjs 9.7 "expr"
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require('/home/mtzn/cjc/human-atlas-face-lab/node_modules/playwright-core');
const [t, expr, pre = '8', shotFile] = process.argv.slice(2);
const HOME = process.env.HOME;
const env = { ...process.env, LD_LIBRARY_PATH: `/usr/lib/wsl/lib:${HOME}/.local/lib/chromium-deps/root/usr/lib/x86_64-linux-gnu`, GALLIUM_DRIVER: 'd3d12' };
const browser = await chromium.launch({ env, args: ['--use-angle=gl-egl', '--enable-gpu', '--ignore-gpu-blocklist', '--hide-scrollbars'] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.log('ERR', String(e)));
await page.goto('http://127.0.0.1:8776/tools/video/film.html?fps=60', { waitUntil: 'load' });
await page.waitForFunction(() => window.__film && (window.__film.ready || window.__film.error), null, { timeout: 120000 });
await page.evaluate(([tt, pp]) => window.__film.seek(tt, pp), [Number(t), Number(pre)]);
if (shotFile) await page.screenshot({ path: shotFile, type: 'jpeg', quality: 85 });
console.log(JSON.stringify(await page.evaluate(expr), null, 1));
await browser.close();
