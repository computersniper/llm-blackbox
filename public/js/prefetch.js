// 后台预取：聊天页可交互后，趁空闲把 3D 舞台要用的模块提前取回来；用户发出问题时再取这条回复的数据。
// 这样点 ＋ 时基本都已经在缓存里。省流量模式（Save-Data）或 2G 网络下不做预取，点 ＋ 时再按需载入。
//
// 模块的 URL 必须和 import 的一模一样才能命中同一份：发布时 tools/stamp.sh 给每个 import 都加了 ?v=<commit>，
// 这里用本模块自己 URL 上的同一个查询串来拼（本地开发时没有版本号，两边都是裸路径）。

const V = new URL(import.meta.url).search;
const JS = new URL('./', import.meta.url);

// 相对 js/ 目录的路径 → 和 import 一致的完整 URL
export const jsURL = (path) => new URL(path + V, JS).href;

// 舞台引擎（js/stage/engine.js）的整张静态模块图：Three.js 和用到的 addons。engine.js 的 import 改了要同步这里
export const ENGINE_GRAPH = [
  'stage/engine.js',
  'vendor/three/three.module.js',
  'vendor/three/three.core.js',
  'vendor/three/addons/postprocessing/EffectComposer.js',
  'vendor/three/addons/postprocessing/ShaderPass.js',
  'vendor/three/addons/postprocessing/MaskPass.js',
  'vendor/three/addons/postprocessing/Pass.js',
  'vendor/three/addons/postprocessing/RenderPass.js',
  'vendor/three/addons/postprocessing/UnrealBloomPass.js',
  'vendor/three/addons/postprocessing/OutputPass.js',
  'vendor/three/addons/shaders/CopyShader.js',
  'vendor/three/addons/shaders/LuminosityHighPassShader.js',
  'vendor/three/addons/shaders/OutputShader.js',
  'vendor/three/addons/renderers/CSS2DRenderer.js',
  'vendor/three/addons/environments/RoomEnvironment.js',
];

// 一次性给整张模块图挂上 <link rel="modulepreload">：浏览器并行去取，不用等 engine.js 到了才发现要 three.module.js、
// 等 three.module.js 到了才发现要 three.core.js。随后的 import() 会直接用上这些请求（同一个 URL 只取一次）
const preloaded = new Set();
export function modulePreload(urls) {
  for (const href of urls) {
    if (preloaded.has(href)) continue;
    preloaded.add(href);
    const l = document.createElement('link');
    l.rel = 'modulepreload';
    l.href = href;
    document.head.append(l);
  }
}

// 用户开了省流量，或者网络是 2G：不做投机性的预取
export function saveData() {
  const c = navigator.connection;
  return !!c && (c.saveData === true || /(^|-)2g$/.test(c.effectiveType || ''));
}

// 性能打点（DevTools 的 Performance 面板里能看到）：start / end 是 performance.now() 的时刻，end 省略就是现在。
// 旧浏览器不认这种写法时什么也不做，绝不影响正常流程
export function measure(name, start, end = performance.now()) {
  try { performance.measure(name, { start, end }); } catch { /* 不支持 User Timing 3 */ }
}

// 页面载完、字体到齐之后，等浏览器空闲时执行 fn（不和首屏抢带宽和主线程）；省流量时什么也不做
export function whenIdle(fn) {
  if (saveData()) return;
  const idle = window.requestIdleCallback || ((f) => setTimeout(f, 200));
  const go = () => document.fonts.ready.then(() => idle(() => fn(), { timeout: 3000 }));
  if (document.readyState === 'complete') go();
  else addEventListener('load', go, { once: true });
}
