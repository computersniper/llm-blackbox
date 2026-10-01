// 读取 tools/world/export_world.py 导出的世界模型：model.json（描述、训练记录）+ model.bin（float16 权重）。
// 文件另存了 .gz（服务器只压缩 HTML），浏览器里用 DecompressionStream 解开；不支持时退回未压缩的版本。
import { unpack, WorldModel } from './nn.js';

async function fetchData(url) {
  if (typeof DecompressionStream !== 'undefined') {
    try {
      const r = await fetch(`${url}.gz`);
      if (r.ok && r.body) return await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    } catch { /* 退回未压缩 */ }
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} ${r.status}`);
  return r.arrayBuffer();
}

export async function loadWorld(base = 'data/') {
  const [jb, bb] = await Promise.all([fetchData(`${base}model.json`), fetchData(`${base}model.bin`)]);
  const meta = JSON.parse(new TextDecoder().decode(jb));
  const model = new WorldModel(meta, unpack(meta, bb));
  return { meta, model };
}
