// 截图 / 冒烟测试工具（仅开发用）。
// node tools/shot.mjs <url> [--w=1440] [--h=900] [--steps='[{"click":"sel"},{"wait":500},{"shot":"a.png"}]']
// 步骤：click / hover / key / type:[sel,text] / wait / eval / shot / mouse:[x,y]
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const pwPath = process.env.PLAYWRIGHT_CORE || '/home/mtzn/cjc/human-atlas-face-lab/node_modules/playwright-core';
const { chromium } = require(pwPath);

const args = Object.fromEntries(process.argv.slice(3).map((a) => { const m = a.match(/^--([^=]+)=(.*)$/s); return m ? [m[1], m[2]] : [a, true]; }));
const url = process.argv[2];
const steps = JSON.parse(args.stepsfile ? (await import('node:fs')).readFileSync(args.stepsfile, 'utf8') : (args.steps || '[]'));
const proxy = !/^https?:\/\/(127\.0\.0\.1|localhost)/.test(url) && (process.env.HTTPS_PROXY || process.env.https_proxy);
const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'], ...(proxy ? { proxy: { server: proxy } } : {}) });
const page = await browser.newPage({ viewport: { width: Number(args.w || 1440), height: Number(args.h || 900) }, deviceScaleFactor: Number(args.dpr || 1) });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') errors.push(`[${m.type()}] ${m.text()}`); });
page.on('pageerror', (e) => errors.push(`[pageerror] ${e.message}`));
page.on('requestfailed', (r) => errors.push(`[requestfailed] ${r.url()}`));
await page.goto(url, { waitUntil: 'load' });
for (const s of steps) {
  if (s.click) await page.click(s.click, { timeout: 5000 }).catch((e) => errors.push(`[step] click ${s.click}: ${e.message.split('\n')[0]}`));
  if (s.hover) await page.hover(s.hover, { timeout: 5000 }).catch((e) => errors.push(`[step] hover ${s.hover}: ${e.message.split('\n')[0]}`));
  if (s.mouse) await page.mouse.move(s.mouse[0], s.mouse[1], { steps: 4 });
  if (s.key) await page.keyboard.press(s.key);
  if (s.type) { await page.fill(s.type[0], ''); await page.type(s.type[0], s.type[1], { delay: 20 }); }
  if (s.wait) await page.waitForTimeout(s.wait);
  if (s.eval) { const r = await page.evaluate(s.eval).catch((e) => `ERR ${e.message}`); if (r !== undefined) console.log('eval:', typeof r === 'string' ? r : JSON.stringify(r)); }
  if (s.shot) { await page.screenshot({ path: s.shot, fullPage: !!s.full }); console.log('shot:', s.shot); }
}
if (errors.length) console.log(errors.join('\n')); else console.log('no console errors');
await browser.close();
