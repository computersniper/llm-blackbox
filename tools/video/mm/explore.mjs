// 调试用：按一组舞台状态截图（不走时间线）。node tools/video/mm/explore.mjs states.json outdir
// states.json: [{ "name": "a", "depth": 3, "step": { "ph": "prep", "sub": "patch" }, "p": 0.5, "cam"?: {...} }]
// 也可以给 { "name": "x", "t": 12.3 } 走时间线（需要 film.js 有 seek）。
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const require = createRequire(import.meta.url);
const { chromium } = require('/home/mtzn/cjc/human-atlas-face-lab/node_modules/playwright-core');
const [file, out, extra = ''] = process.argv.slice(2);
const states = JSON.parse(fs.readFileSync(file, 'utf8'));
fs.mkdirSync(out, { recursive: true });
const env = { ...process.env, LD_LIBRARY_PATH: `${process.env.HOME}/.local/lib/chromium-deps/root/usr/lib/x86_64-linux-gnu` };
const browser = await chromium.launch({ env, args: ['--disable-gpu', '--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--hide-scrollbars', '--force-color-profile=srgb'] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.log('ERR', String(e)));
page.on('console', (m) => { if (m.type() === 'error') console.log('CONSOLE', m.text()); });
page.on('response', (r) => { if (r.status() >= 400) console.log('HTTP', r.status(), r.url()); });
await page.goto(`http://127.0.0.1:8798/tools/video/mm/film.html?lang=zh&fps=30${extra}`, { waitUntil: 'load' });
await page.waitForFunction(() => window.__film && (window.__film.ready || window.__film.error), null, { timeout: 180000 });
const err = await page.evaluate(() => window.__film.error);
if (err) { console.log(err); process.exit(1); }
for (const s of states) {
  const t0 = Date.now();
  const r = s.t != null ? await page.evaluate(([t, pre]) => { window.__film.seek(t, pre); return window.__film.probe?.(); }, [s.t, s.pre ?? 4]) : await page.evaluate((st) => window.__film.show(st), s);
  if (s.eval) console.log('eval', JSON.stringify(await page.evaluate(s.eval)));
  const f = path.join(out, `${s.name}.jpg`);
  await page.screenshot({ path: f, type: 'jpeg', quality: 88 });
  console.log(f, JSON.stringify(r), `${Date.now() - t0} ms`);
}
await browser.close();
