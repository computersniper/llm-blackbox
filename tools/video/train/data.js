// 训练视频的数据：训练页第一章（玻璃小模型）的真实训练记录，原样用网站的读取器（public/train/js/glass/data.js）。
// 网站是按需分块载入的；片子要逐帧确定地渲染，所以开机时把全部分块（76 帧的权重 / 梯度 / 激活、exact 原值）一次取完。
import { Loader } from '/public/train/js/data.js';
import { loadGlass } from '/public/train/js/glass/data.js';

export async function loadAll() {
  const loader = new Loader();
  const D = await loadGlass(loader, '/public/train/data/');
  const items = [...loader.items.values()];
  // 一次最多 8 个请求在路上
  let next = 0;
  const work = async () => { while (next < items.length) { const it = items[next++]; if (it.state !== 'ok') await loader.start(it); } };
  await Promise.all(Array.from({ length: 8 }, work));
  const bad = items.filter((it) => it.state !== 'ok');
  if (bad.length) throw new Error(`训练记录有 ${bad.length} 块没取到：${bad.slice(0, 3).map((b) => b.key).join(', ')}`);
  return D;
}
