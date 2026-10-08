// 把视频原来的横版封面（poster-*.png，1920×1080）改成抖音竖版 9:16（1080×1920）。画面本身不重新设计：
// 截出原图里标题和主体那一块，四周羽化后放在深色底上；底上只留一点同一张图的模糊氛围光。
// 用法：node tools/video/covers/vertical.mjs [子串…]   （输出到原文件旁边，文件名加 -9x16；给了子串就只做 src 含这些子串的那几张）
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { writeFileSync, mkdirSync } from 'node:fs';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_CORE || '/home/mtzn/cjc/human-atlas-face-lab/node_modules/playwright-core');
const V = '/mnt/d/cjc/videos';

// parts：从原图里截哪几块（x, y, w, h），放到竖版的哪里（top，宽 width，居中）
const JOBS = [
  { src: 'llm-inference/poster-v7.png', parts: [{ crop: [380, 60, 1340, 960], top: 480, width: 1040 }] },
  { src: 'llm-inference/poster-v7-en.png', parts: [{ crop: [380, 60, 1340, 960], top: 480, width: 1040 }] },
  { src: 'train-glass/poster-v1.png', parts: [{ crop: [330, 0, 1240, 1080], top: 480, width: 1080 }] },
  { src: 'agent/poster-v1.png', parts: [{ crop: [400, 110, 1120, 880], top: 520, width: 1080 }] },
  // 多模态：原图左右排（热力图卡片 | 标题），竖版里上下叠放，元素不变
  { src: 'multimodal/poster-v1.png', parts: [{ crop: [30, 100, 780, 860], top: 250, width: 940 }, { crop: [940, 360, 840, 330], top: 1330, width: 1020, feather: 16 }] },
  // 多模态 v2：卡片挪到 x 260–968、标题在 x 1010–1640，同样上下叠放
  { src: 'multimodal/poster-v2.png', parts: [{ crop: [240, 130, 750, 810], top: 250, width: 940 }, { crop: [990, 370, 690, 270], top: 1330, width: 1000, feather: 16 }] },
];

const html = (src, parts) => `<!doctype html><html><head><meta charset="utf-8"><style>
* { margin: 0; } body { width: 1080px; height: 1920px; overflow: hidden; background: #050b16; position: relative; }
.glow { position: absolute; inset: -80px; background: url("${src}") center / auto 100% no-repeat; filter: blur(60px) brightness(.35) saturate(1.1); opacity: .55; }
.part { position: absolute; left: 50%; transform: translateX(-50%); overflow: hidden; --f: 7%;
  -webkit-mask-image: linear-gradient(to right, transparent, #000 var(--f), #000 calc(100% - var(--f)), transparent), linear-gradient(to bottom, transparent, #000 calc(var(--f) + 2%), #000 calc(98% - var(--f)), transparent);
  -webkit-mask-composite: source-in; mask-composite: intersect; }
.part > div { width: 100%; height: 100%; background-image: url("${src}"); background-repeat: no-repeat; }
</style></head><body><div class="glow"></div>
${parts.map(({ crop: [x, y, w, h], top, width, feather }) => {
  const k = width / w;   // 原图放大的倍数
  return `<div class="part" style="top:${top}px;width:${width}px;height:${Math.round(h * k)}px${feather ? `;--f:${feather}%` : ''}"><div style="background-size:${1920 * k}px ${1080 * k}px;background-position:${-x * k}px ${-y * k}px"></div></div>`;
}).join('')}
</body></html>`;

const browser = await chromium.launch({ args: ['--disable-gpu', '--allow-file-access-from-files'] });
const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
const only = process.argv.slice(2);
for (const j of JOBS.filter((x) => !only.length || only.some((k) => x.src.includes(k)))) {
  const src = `${V}/${j.src}`;
  // 本地图片要从 file:// 页面里引用才载得进来，所以先把页面写成文件（放 D 盘，不占 C 盘）
  mkdirSync(`${V}/covers`, { recursive: true });
  const file = `${V}/covers/v-${j.src.replace(/\W+/g, '_')}.html`;
  writeFileSync(file, html(pathToFileURL(src).href, j.parts));
  await page.goto(pathToFileURL(file).href);
  await page.waitForFunction(() => [...document.images].every((i) => i.complete) && document.readyState === 'complete');
  await page.waitForTimeout(400);
  const out = src.replace(/\.png$/, '-9x16.png');
  await page.screenshot({ path: out });
  console.log('cover:', out);
}
await browser.close();
