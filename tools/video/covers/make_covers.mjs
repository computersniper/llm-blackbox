// 视频封面：左边大标题，右边铺满这支片子里最好看的一块真实画面（从成片抽帧再裁切），1920×1080。
// 用法：node tools/video/covers/make_covers.mjs [名字…]   （默认全部；输出到 /mnt/d/cjc/videos/<目录>/cover.png）
// 依赖：ffmpeg、playwright-core（同 tools/shot.mjs）；字体是 /mnt/d/cjc/videos/llm-inference/fonts/ 下的 OFL Noto 系列。
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { writeFileSync, mkdirSync } from 'node:fs';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_CORE || '/home/mtzn/cjc/human-atlas-face-lab/node_modules/playwright-core');

const V = '/mnt/d/cjc/videos';
const FONTS = `${V}/llm-inference/fonts`;
const WORK = `${V}/covers`;

// crop：在 1920×1080 的成片帧里裁哪一块（x, y, w, h）；pos：裁好的图在右侧框里的对齐方式；padL：左边补多宽的深色（把画面往右推，别压到标题）；size：CSS background-size，默认 cover
const COVERS = {
  'infer-zh': { dir: 'llm-inference', video: 'qwen3-inference-v7.mp4', t: 40, crop: [150, 60, 1180, 880], lang: 'zh',
    tag: '揭开黑箱 · 推理', title: ['AI 的一个字', '是怎么思考出来的'], sub: '走进大模型推理的“黑箱”', chips: ['真实的 Qwen3-0.6B', '28 层 · 每一步'] },
  'infer-en': { dir: 'llm-inference', video: 'qwen3-inference-v7-en.mp4', t: 38, crop: [150, 60, 1180, 880], lang: 'en', out: 'cover-en.png',
    tag: 'BLACK BOX · INFERENCE', title: ['How Does AI Think', 'Up a Single Word?'], sub: 'Inside the black box of LLM inference', chips: ['Real Qwen3-0.6B', '28 layers · every step'] },
  'train': { dir: 'train-glass', video: 'train-glass-v1.mp4', t: 146, crop: [0, 120, 1080, 800], lang: 'zh',
    tag: '揭开黑箱 · 训练', title: ['AI 是', '怎么学会的'], sub: '看一个小模型从零学会背《静夜思》', chips: ['2,928 个参数', '200 步真实训练'] },
  'multimodal': { dir: 'multimodal', video: 'multimodal-v1.mp4', t: 62, crop: [700, 180, 1220, 760], lang: 'zh',
    tag: '揭开黑箱 · 多模态', title: ['AI 是', '怎么看图的'], sub: '一张图，怎么变成模型读得懂的“词”', chips: ['真实的 Qwen3-VL-2B', '看它在看哪'] },
  'agent': { dir: 'agent', video: 'agent-v1.mp4', t: 28, crop: [0, 78, 1000, 600], padL: 460, size: '100% auto', lang: 'zh',
    tag: '揭开黑箱 · 智能体', title: ['只会写字的 AI', '怎么自己动手干活'], sub: '走进编程智能体的“黑箱”', chips: ['真实的 Qwen3-4B', '在沙箱里真跑'] },
};

const html = (c, img) => `<!doctype html><html lang="${c.lang === 'en' ? 'en' : 'zh-CN'}"><head><meta charset="utf-8"><style>
@font-face { font-family: Serif; src: url("file://${FONTS}/${c.lang === 'en' ? 'NotoSerif-VF.ttf' : 'NotoSerifSC-Black.otf'}"); font-weight: 900; }
@font-face { font-family: Sans; src: url("file://${FONTS}/${c.lang === 'en' ? 'NotoSans-VF.ttf' : 'NotoSansSC-VF.ttf'}"); }
* { margin: 0; box-sizing: border-box; }
body { width: 1920px; height: 1080px; overflow: hidden; background: #050b16; color: #eef4ff; font-family: Sans, sans-serif; position: relative; }
.pic { position: absolute; top: 0; right: 0; width: 1240px; height: 1080px; background: url("${img}") ${c.pos || 'center'} / ${c.size || 'cover'} no-repeat; filter: saturate(1.12) contrast(1.06); }
.fade { position: absolute; inset: 0; background:
  linear-gradient(to right, #050b16 0%, #050b16 38%, rgba(5, 11, 22, .86) 50%, rgba(5, 11, 22, .32) 64%, rgba(5, 11, 22, 0) 80%),
  linear-gradient(to top, rgba(5, 11, 22, .7) 0%, rgba(5, 11, 22, 0) 22%),
  linear-gradient(to bottom, rgba(5, 11, 22, .55) 0%, rgba(5, 11, 22, 0) 16%); }
.txt { position: absolute; left: 112px; top: 50%; transform: translateY(-52%); width: 980px; }
.tag { font: 600 26px/1 Sans, sans-serif; letter-spacing: .32em; color: #5ef0d4; margin-bottom: 38px; }
h1 { display: inline-block; white-space: nowrap; font: 900 ${c.lang === 'en' ? 110 : 128}px/1.13 Serif, serif; letter-spacing: ${c.lang === 'en' ? '-.01em' : '.02em'}; text-shadow: 0 6px 30px rgba(0, 0, 0, .55); }
.bar { display: block; width: 120px; height: 6px; border-radius: 3px; background: linear-gradient(90deg, #5ef0d4, #7c9cff); margin: 40px 0 30px; }
.sub { font: 500 44px/1.4 Sans, sans-serif; color: #c9d6ea; }
.chips { display: flex; gap: 16px; margin-top: 40px; }
.chip { font: 500 26px/1 Sans, sans-serif; color: #a8b8d0; padding: 12px 20px; border-radius: 999px; border: 1.5px solid rgba(94, 240, 212, .45); background: rgba(94, 240, 212, .07); }
.url { position: absolute; left: 112px; bottom: 64px; font: 500 26px/1 Sans, sans-serif; letter-spacing: .06em; color: #7f8ea8; }
</style></head><body>
<div class="pic"></div><div class="fade"></div>
<div class="txt"><div class="tag">${c.tag}</div><h1>${c.title.join('<br>')}</h1><div class="bar"></div><div class="sub">${c.sub}</div>
<div class="chips">${c.chips.map((s) => `<span class="chip">${s}</span>`).join('')}</div></div>
<div class="url">caijiechao.com/blackbox</div>
</body></html>`;

const want = process.argv.slice(2);
const names = want.length ? want : Object.keys(COVERS);
mkdirSync(WORK, { recursive: true });
const browser = await chromium.launch({ args: ['--disable-gpu', '--allow-file-access-from-files'] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
for (const name of names) {
  const c = COVERS[name];
  const img = `${WORK}/${name}-pic.png`;
  const [x, y, w, h] = c.crop;
  execFileSync('ffmpeg', ['-loglevel', 'error', '-y', '-ss', String(c.t), '-i', `${V}/${c.dir}/${c.video}`, '-frames:v', '1', '-vf', `crop=${w}:${h}:${x}:${y}` + (c.padL ? `,pad=${w + c.padL}:${h}:${c.padL}:0:color=0x050b16` : ''), img]);
  const file = `${WORK}/${name}.html`;
  writeFileSync(file, html(c, `file://${img}`));
  await page.goto(`file://${file}`);
  await page.evaluate(() => document.fonts.ready);
  // 标题不折行：太长就按宽度缩小字号（最宽 900px）
  await page.evaluate(() => { const h = document.querySelector('h1'); let f = parseFloat(getComputedStyle(h).fontSize); while (h.getBoundingClientRect().width > 900 && f > 60) { f -= 2; h.style.fontSize = f + 'px'; } });
  const out = `${V}/${c.dir}/${c.out || 'cover.png'}`;
  await page.screenshot({ path: out });
  console.log('cover:', out);
}
await browser.close();
