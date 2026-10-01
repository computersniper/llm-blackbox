// 世界模型页 · 第二章「大模型里的世界地图」（独立模块）。
// 复现 Gurnee & Tegmark 2023《Language Models Represent Space and Time》：把 3,755 个城市名一个个喂给 Qwen3-0.6B，
// 取城市名最后一个词元在每一层的隐状态，用线性探针（岭回归）读出经纬度。地图上每个点都是探针对一个“训练时没见过”的城市
// 的预测（5 折交叉），细线连到真实位置。数据由 tools/world/probe_export.py 离线导出，这里只回放，不跑模型。
//
// 接口：mountProbe(el, { onExplain(html) }) → { setLayer(L), destroy() }
//   el 是一个空容器（自适应大小）；onExplain 把当前讲解推给页面右侧；setLayer 由外部调试器驱动（0–27）。

const URL_CSS = new URL('./probe.css', import.meta.url).href;
const URL_DATA = new URL('./data/', import.meta.url).href;

const NL = 28;
const LAT_TOP = 80, LAT_BOT = -58;                // 地图显示的纬度范围（等距圆柱投影：经纬度就是平面坐标，和线性探针的读数一一对应）
const ASPECT = 360 / (LAT_TOP - LAT_BOT);
const TWEEN_MS = 620, STEP_MS = 820;
const MINUS = '−';
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

// 大洲配色（深色背景上压暗过的站点色相；按这个固定顺序做过色觉辨识检查）
const CT_COL = ['#5097d0', '#d57327', '#05a891', '#b18c00', '#e05b8b', '#957ede'];
// 四组探针：真实模型 · 问坐标 / 只给名字，对照 · 打乱标签 / 未训练
const VAR = {
  coords: { label: '问坐标', short: '问坐标', tag: '', col: '#4fd8be' },
  name: { label: '只给名字', short: '只给名字', tag: '', col: '#6f93e8' },
  shuffled: { label: '打乱标签', short: '打乱标签', tag: '对照', col: '#e0688c' },
  random: { label: '未训练', short: '未训练', tag: '对照', col: '#a08ae6' },
};
// 误差着色：单一色相由暗到亮（越亮错得越远）
const ERR_STEPS = [500, 1000, 1500, 2000, 3000, 4500, 7000, Infinity];
const ERR_COL = ['#653e02', '#7b4c00', '#925b00', '#a76b13', '#ba7e2d', '#ce9042', '#e2a356', '#f6b669'];

/* ---------------------------------------------------------------- 数据 */

// 数据文件预先 gzip 过（服务器只压缩 HTML），浏览器里用 DecompressionStream 解开；不支持时退回未压缩版本
async function fetchData(url, signal) {
  if (typeof DecompressionStream !== 'undefined') {
    try {
      const r = await fetch(`${url}.gz`, { signal });
      if (r.ok && r.body) return await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    } catch (e) { if (e.name === 'AbortError') throw e; }
  }
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`${url} ${r.status}`);
  return r.arrayBuffer();
}

// 算式只存了 .gz（只有展开算式面板时才取）
async function fetchGzJSON(url, signal) {
  if (typeof DecompressionStream === 'undefined') throw new Error('当前浏览器不支持解压，无法显示算式');
  const r = await fetch(`${url}.gz`, { signal });
  if (!r.ok || !r.body) throw new Error(`${url}.gz ${r.status}`);
  return JSON.parse(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).text());
}

let dataPromise = null;     // 同一页面里多次挂载（比如切走再回来）只取一次
function loadData() {
  if (!dataPromise) {
    dataPromise = (async () => {
      const [mb, pb] = await Promise.all([fetchData(`${URL_DATA}probe.json`), fetchData(`${URL_DATA}pred.bin`)]);
      const meta = JSON.parse(new TextDecoder().decode(mb));
      return { meta, pred: decodePred(pb, meta.pred) };
    })();
    dataPromise.catch(() => { dataPromise = null; });
  }
  return dataPromise;
}

// pred.bin：int16（0.1°），沿层差分，再按字节分面（先放全部低字节，再放全部高字节）。还原成 pred[变体][层] = Float32Array(n×2)
function decodePred(buf, spec) {
  const [V, L, n] = spec.shape;
  const u8 = new Uint8Array(buf);
  const M = V * L * n * 2;
  if (u8.length !== 2 * M) throw new Error('pred.bin 大小不对');
  const out = [];
  const acc = new Int32Array(n * 2);
  for (let v = 0; v < V; v++) {
    const layers = [];
    for (let l = 0; l < L; l++) {
      const a = new Float32Array(n * 2);
      const base = (v * L + l) * n * 2;
      for (let j = 0; j < n * 2; j++) {
        const k = base + j;
        const d = ((u8[k] | (u8[M + k] << 8)) << 16) >> 16;
        acc[j] = l === 0 ? d : acc[j] + d;
        a[j] = acc[j] * spec.scale;
      }
      layers.push(a);
    }
    out.push(layers);
  }
  return out;
}

/* ---------------------------------------------------------------- 小工具 */

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const easeIO = (t) => { t = clamp(t, 0, 1); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmtInt = (v) => Math.round(v).toLocaleString('en-US');
const sgn = (v, d) => `${v < 0 ? MINUS : '+'}${Math.abs(v).toFixed(d)}`;
const num = (v, d) => `${v < 0 ? MINUS : ''}${Math.abs(v).toFixed(d)}`;
const latTxt = (v, d = 1) => `${v >= 0 ? '北纬' : '南纬'} ${Math.abs(v).toFixed(d)}°`;
const lonTxt = (v, d = 1) => {
  const w = ((v + 540) % 360) - 180;          // 探针的经度可能超出 ±180°，换回地图上的读法
  return `${w >= 0 ? '东经' : '西经'} ${Math.abs(w).toFixed(d)}°`;
};
const popTxt = (p) => (p >= 1e8 ? `${(p / 1e8).toFixed(1)} 亿` : p >= 1e4 ? `${fmtInt(p / 1e4)} 万` : fmtInt(p));
// 隐状态差值 / 权重 / 乘积按量级取位数
const fx3 = (v) => { const a = Math.abs(v); return sgn(v, a >= 100 ? 1 : a >= 10 ? 2 : 3); };
const fw = (v) => { const a = Math.abs(v); return `${v < 0 ? MINUS : '+'}${a >= 0.01 ? a.toFixed(4) : a.toPrecision(3)}`; };

function haversine(la1, lo1, la2, lo2) {
  const r = Math.PI / 180;
  la1 = clamp(la1, -90, 90) * r; la2 = clamp(la2, -90, 90) * r;
  const h = Math.sin((la2 - la1) / 2) ** 2 + Math.cos(la1) * Math.cos(la2) * Math.sin(((lo2 - lo1) * r) / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(clamp(h, 0, 1)));
}

const FONT_SANS = '-apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", "Noto Sans SC", system-ui, sans-serif';
const FONT_MONO = '"JetBrains Mono", ui-monospace, Menlo, Consolas, monospace';

let cssRefs = 0;
function useCSS() {
  cssRefs++;
  if (!document.querySelector('link[data-pb-css]')) {
    const l = document.createElement('link');
    l.rel = 'stylesheet';
    l.href = URL_CSS;
    l.dataset.pbCss = '';
    document.head.append(l);
  }
}
function dropCSS() {
  cssRefs = Math.max(0, cssRefs - 1);
  if (!cssRefs) document.querySelector('link[data-pb-css]')?.remove();
}

/* ---------------------------------------------------------------- 挂载 */

export async function mountProbe(el, opts = {}) {
  useCSS();
  const onExplain = typeof opts.onExplain === 'function' ? opts.onExplain : null;
  const ac = new AbortController();
  const on = (t, ev, fn, o = {}) => t.addEventListener(ev, fn, { ...o, signal: ac.signal });
  let dead = false, ro = null;

  const root = document.createElement('div');
  root.className = 'pb-root';
  root.tabIndex = 0;
  root.setAttribute('role', 'region');
  root.setAttribute('aria-label', '大模型里的世界地图：线性探针从 Qwen3-0.6B 的隐状态里读出城市的经纬度');
  root.innerHTML = `
    <header class="pb-head">
      <div class="pb-read" aria-live="polite">
        <span class="pb-L">第 <b data-k="L">0</b> 层</span>
        <span class="pb-kv">R² <b data-k="r2">—</b></span>
        <span class="pb-kv">平均误差 <b data-k="km">—</b> km</span>
        <span class="pb-kv">中位 <b data-k="med">—</b> km</span>
      </div>
      <div class="pb-seg" role="radiogroup" aria-label="探针">
        ${Object.entries(VAR).map(([k, v], i) => `<button type="button" role="radio" data-v="${i}" style="--pb-c:${v.col}" title="${k === 'coords' ? '提示：What are the lat/lon coordinates of 城市名' : k === 'name' ? '只把城市名喂给模型（论文的主实验）' : k === 'shuffled' ? '对照：训练时把城市和坐标随机配对' : '对照：同样结构、权重随机初始化的模型'}">${v.tag ? `<i>${v.tag}</i>` : ''}${v.short}</button>`).join('')}
      </div>
    </header>
    <div class="pb-stage">
      <canvas class="pb-map" aria-hidden="true"></canvas>
      <div class="pb-feed"></div>
      <div class="pb-credit">城市：GeoNames（CC BY 4.0）· 海岸线：Natural Earth</div>
      <div class="pb-tip" hidden></div>
      <div class="pb-loading"><span>正在载入探针数据…</span></div>
    </div>
    <div class="pb-ctrl">
      <button type="button" class="pb-play" aria-label="播放：从第 0 层到第 27 层">
        <svg class="pb-ic-play" viewBox="0 0 16 16"><path d="M4 2.5v11l9.5-5.5z"/></svg>
        <svg class="pb-ic-pause" viewBox="0 0 16 16"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z"/></svg>
      </button>
      <div class="pb-scrub">
        <canvas aria-hidden="true"></canvas>
        <input class="pb-range" type="range" min="0" max="${NL - 1}" step="1" value="0" aria-label="层">
        <div class="pb-focus"></div>
      </div>
    </div>
    <div class="pb-foot">
      <div class="pb-legend"></div>
      <div class="pb-opts">
        <button type="button" class="pb-tog" data-o="color" title="按大洲 / 按误差着色">按误差着色</button>
        <button type="button" class="pb-tog on" data-o="lines" title="把预测位置和真实位置连起来">连线</button>
        <button type="button" class="pb-tog pb-fxbtn" data-o="fx" title="一个城市在这一层的读数是怎么乘加出来的">算式 ƒ</button>
      </div>
    </div>
    <aside class="pb-fx" hidden aria-label="算式"></aside>`;
  el.replaceChildren(root);
  const $ = (s) => root.querySelector(s);
  const stage = $('.pb-stage'), cv = $('.pb-map'), g = cv.getContext('2d');
  const scrub = $('.pb-scrub'), sc = scrub.querySelector('canvas'), sg = sc.getContext('2d');
  const range = $('.pb-range'), tip = $('.pb-tip'), feed = $('.pb-feed'), fxEl = $('.pb-fx');
  const legend = $('.pb-legend');

  const S = {
    v: 0, L: clamp(opts.layer | 0, 0, NL - 1), color: 'ct', lines: true, iso: -1,
    hover: -1, sel: -1, playing: false, narrow: false,
  };
  let D = null, meta = null, VN = [], n = 0;
  let disp = null, from = null, to = null, t0 = 0, animating = false;
  let dispL = S.L, fromL = S.L;
  let raf = 0, lastStep = 0, explainTimer = 0;
  let W = 0, H = 0, CW = 0, CH = 0, dpr = 1;
  let M = { x: 0, y: 0, w: 1, h: 1 };
  let landPath = null, gridPath = null, eqPath = null;
  const errCache = new Map();
  const fx = { open: false, city: -1, tab: 'lat', data: new Map(), key: '', countRaf: 0 };

  /* ---------- 载入 ---------- */

  try {
    ({ meta, pred: D } = await loadData());
  } catch (e) {
    if (!dead) $('.pb-loading').innerHTML = `<span>数据载入失败：${esc(e.message)}</span>`;
    return { setLayer() {}, destroy };
  }
  if (dead) return { setLayer() {}, destroy };
  $('.pb-loading').remove();
  VN = meta.variants;
  n = meta.n;
  const C = meta.cities;
  const featured = new Map(C.featured.map(([i, zh]) => [i, zh]));
  const meanLat = C.lat.reduce((a, b) => a + b, 0) / n, meanLon = C.lon.reduce((a, b) => a + b, 0) / n;
  disp = new Float32Array(D[S.v][S.L]);
  from = new Float32Array(n * 2);
  to = D[S.v][S.L];
  // 每个城市属于哪个大洲（按颜色分组画，一组一条路径）
  const byCt = meta.continents.map((_, c) => { const a = []; for (let i = 0; i < n; i++) if (C.ct[i] === c) a.push(i); return a; });

  legend.innerHTML = meta.continents.map((name, c) => `<button type="button" class="pb-chip" data-c="${c}" style="--pb-c:${CT_COL[c]}" title="只看${name}"><i></i>${name}<span style="opacity:.55">${byCt[c].length}</span></button>`).join('');
  const rampHTML = `<span class="pb-ramp">误差 &lt;500<i style="background:linear-gradient(to right,${ERR_COL.join(',')})"></i>7,000+ km</span>`;

  /* ---------- 尺寸 ---------- */

  function resize() {
    dpr = Math.min(2, window.devicePixelRatio || 1);
    const narrow = root.clientWidth < 560;
    if (narrow !== S.narrow) { S.narrow = narrow; root.classList.toggle('pb-narrow', narrow); }
    W = stage.clientWidth; H = stage.clientHeight;
    cv.width = Math.max(1, Math.round(W * dpr)); cv.height = Math.max(1, Math.round(H * dpr));
    const top = S.narrow ? 22 : 30, bot = S.narrow ? 6 : 16;
    const mw = Math.max(10, Math.min(W - 12, (H - top - bot) * ASPECT));
    const mh = mw / ASPECT;
    M = { x: (W - mw) / 2, y: top + Math.max(0, (H - top - bot - mh) / 2), w: mw, h: mh };
    buildPaths();
    CW = scrub.clientWidth; CH = scrub.clientHeight;
    sc.width = Math.max(1, Math.round(CW * dpr)); sc.height = Math.max(1, Math.round(CH * dpr));
    // 算式面板：宽屏时盖在地图上（和地图区域对齐），窄屏时盖住整个模块
    fxEl.style.top = S.narrow ? '' : `${stage.offsetTop + 8}px`;
    fxEl.style.height = S.narrow ? '' : `${Math.max(120, H - 16)}px`;
    const P = plotBox();
    range.style.left = `${P.x - 11}px`;
    range.style.width = `${P.w + 22}px`;
    hideTip();
    draw();
  }
  const px = (lon) => M.x + ((lon + 180) / 360) * M.w;
  const py = (lat) => M.y + ((LAT_TOP - lat) / (LAT_TOP - LAT_BOT)) * M.h;

  function buildPaths() {
    landPath = new Path2D();
    for (const r of meta.land) {
      landPath.moveTo(px(r[0] / 10), py(r[1] / 10));
      for (let k = 2; k < r.length; k += 2) landPath.lineTo(px(r[k] / 10), py(r[k + 1] / 10));
      landPath.closePath();
    }
    gridPath = new Path2D();
    for (let lon = -150; lon <= 150; lon += 30) { gridPath.moveTo(px(lon), py(LAT_TOP)); gridPath.lineTo(px(lon), py(LAT_BOT)); }
    for (let lat = -30; lat <= 60; lat += 30) { if (lat) { gridPath.moveTo(px(-180), py(lat)); gridPath.lineTo(px(180), py(lat)); } }
    eqPath = new Path2D();
    eqPath.moveTo(px(-180), py(0)); eqPath.lineTo(px(180), py(0));
  }

  /* ---------- 误差 ---------- */

  function errs(v, L) {
    const k = v * NL + L;
    let e = errCache.get(k);
    if (!e) {
      const p = D[v][L];
      e = new Float32Array(n);
      for (let i = 0; i < n; i++) e[i] = haversine(C.lat[i], C.lon[i], p[2 * i], p[2 * i + 1]);
      errCache.set(k, e);
    }
    return e;
  }
  const errBucket = (km) => { let b = 0; while (km >= ERR_STEPS[b]) b++; return b; };

  /* ---------- 画地图 ---------- */

  function drawMap() {
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    // 经纬网
    g.lineWidth = 1;
    g.strokeStyle = 'rgba(150,180,230,0.05)';
    g.stroke(gridPath);
    g.strokeStyle = 'rgba(150,180,230,0.11)';
    g.setLineDash([3, 4]);
    g.stroke(eqPath);
    g.setLineDash([]);
    // 陆地（Natural Earth 1:110m）
    g.fillStyle = 'rgba(150,180,230,0.055)';
    g.fill(landPath, 'evenodd');
    g.strokeStyle = 'rgba(150,180,230,0.24)';
    g.lineWidth = 0.8;
    g.stroke(landPath);

    const r = clamp(M.w / 520, 1.05, 2.1);
    const err = S.color === 'err' ? errs(S.v, S.L) : null;
    // 分组：大洲（或误差档）→ 城市下标
    const groups = err ? ERR_COL.map(() => []) : null;
    if (err) for (let i = 0; i < n; i++) groups[errBucket(err[i])].push(i);
    const list = err ? groups : byCt;
    const cols = err ? ERR_COL : CT_COL;
    const dimOf = (gi, i) => S.iso >= 0 && C.ct[i] !== S.iso;

    // 连线：模型以为的位置 → 真实位置
    if (S.lines) {
      g.lineWidth = 0.6;
      for (let gi = 0; gi < list.length; gi++) {
        const arr = list[gi];
        if (!arr.length) continue;
        g.beginPath();
        for (const i of arr) {
          if (dimOf(gi, i)) continue;
          g.moveTo(px(disp[2 * i + 1]), py(disp[2 * i]));
          g.lineTo(px(C.lon[i]), py(C.lat[i]));
        }
        g.strokeStyle = cols[gi];
        g.globalAlpha = S.iso >= 0 ? 0.32 : 0.12;
        g.stroke();
      }
      g.globalAlpha = 1;
    }
    // 点：模型以为的位置
    for (let pass = 0; pass < 2; pass++) {
      for (let gi = 0; gi < list.length; gi++) {
        const arr = list[gi];
        g.beginPath();
        let any = false;
        for (const i of arr) {
          const dim = dimOf(gi, i);
          if ((pass === 0) !== dim) continue;
          const x = px(disp[2 * i + 1]), y = py(disp[2 * i]);
          g.moveTo(x + r, y);
          g.arc(x, y, r, 0, Math.PI * 2);
          any = true;
        }
        if (!any) continue;
        g.fillStyle = cols[gi];
        g.globalAlpha = pass === 0 ? 0.13 : 0.88;
        g.fill();
      }
    }
    g.globalAlpha = 1;
    // 可以看算式的城市：细圈
    g.lineWidth = 1;
    g.strokeStyle = 'rgba(233,239,249,0.5)';
    g.beginPath();
    for (const i of featured.keys()) {
      if (S.iso >= 0 && C.ct[i] !== S.iso) continue;
      const x = px(disp[2 * i + 1]), y = py(disp[2 * i]);
      g.moveTo(x + r + 2.2, y);
      g.arc(x, y, r + 2.2, 0, Math.PI * 2);
    }
    g.stroke();
    // 选中 / 悬停 / 算式里的城市
    const marks = new Set([S.sel, S.hover, fx.open ? fx.city : -1].filter((i) => i >= 0));
    for (const i of marks) markCity(i, r);
  }

  function markCity(i, r) {
    const x1 = px(disp[2 * i + 1]), y1 = py(disp[2 * i]);
    const x2 = px(C.lon[i]), y2 = py(C.lat[i]);
    const col = S.color === 'err' ? ERR_COL[errBucket(errs(S.v, S.L)[i])] : CT_COL[C.ct[i]];
    g.strokeStyle = 'rgba(233,239,249,0.85)';
    g.lineWidth = 1.2;
    g.setLineDash([4, 3]);
    g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
    g.setLineDash([]);
    // 真实位置：空心十字圈
    g.lineWidth = 1.3;
    g.beginPath(); g.arc(x2, y2, 4.5, 0, Math.PI * 2); g.stroke();
    g.beginPath(); g.moveTo(x2 - 7, y2); g.lineTo(x2 - 2.5, y2); g.moveTo(x2 + 2.5, y2); g.lineTo(x2 + 7, y2);
    g.moveTo(x2, y2 - 7); g.lineTo(x2, y2 - 2.5); g.moveTo(x2, y2 + 2.5); g.lineTo(x2, y2 + 7); g.stroke();
    // 预测位置：实心点 + 外圈
    g.fillStyle = col;
    g.beginPath(); g.arc(x1, y1, r + 2.2, 0, Math.PI * 2); g.fill();
    g.strokeStyle = '#050b17'; g.lineWidth = 1.5; g.stroke();
    g.strokeStyle = 'rgba(233,239,249,0.9)'; g.lineWidth = 1;
    g.beginPath(); g.arc(x1, y1, r + 4.6, 0, Math.PI * 2); g.stroke();
    // 名字
    const zh = featured.get(i);
    const label = zh ? `${zh} ${C.name[i]}` : C.name[i];
    g.font = `600 ${S.narrow ? 11 : 12}px ${FONT_SANS}`;
    const tw = g.measureText(label).width;
    let lx = x1 + 9, ly = y1 - 9;
    if (lx + tw > W - 4) lx = x1 - 9 - tw;
    if (ly < 14) ly = y1 + 18;
    g.fillStyle = 'rgba(5,11,23,0.78)';
    g.fillRect(lx - 4, ly - 12, tw + 8, 16);
    g.fillStyle = '#e9eff9';
    g.textBaseline = 'alphabetic';
    g.fillText(label, lx, ly);
  }

  /* ---------- 画曲线（同时是层滑杆） ---------- */

  function plotBox() {
    const l = S.narrow ? 26 : 32, r = S.narrow ? 58 : 132, t = 8, b = 18;
    return { x: l, y: t, w: Math.max(10, CW - l - r), h: Math.max(10, CH - t - b) };
  }

  function drawChart() {
    const P = plotBox();
    sg.setTransform(dpr, 0, 0, dpr, 0, 0);
    sg.clearRect(0, 0, CW, CH);
    const y0 = -0.05, y1 = 1;
    const X = (L) => P.x + (L / (NL - 1)) * P.w;
    const Y = (v) => P.y + (1 - (clamp(v, y0, y1) - y0) / (y1 - y0)) * P.h;
    sg.font = `10px ${FONT_MONO}`;
    sg.textBaseline = 'middle';
    // 网格
    for (const v of [0, 0.5, 1]) {
      sg.strokeStyle = v === 0 ? 'rgba(150,180,230,0.22)' : 'rgba(150,180,230,0.08)';
      sg.lineWidth = 1;
      sg.beginPath(); sg.moveTo(P.x, Y(v)); sg.lineTo(P.x + P.w, Y(v)); sg.stroke();
      sg.fillStyle = '#4b5572';
      sg.textAlign = 'right';
      sg.fillText(v === 0.5 ? '.5' : String(v), P.x - 6, Y(v));
    }
    sg.textAlign = 'left';
    sg.fillStyle = '#7a859e';
    sg.font = `10px ${FONT_SANS}`;
    sg.fillText('R²', 2, P.y + 4);
    // 论文里 Llama-2-7B 的数字（60% 深度处）作参照
    const pr = meta.paper.r2_7b;
    sg.strokeStyle = 'rgba(180,190,210,0.28)';
    sg.setLineDash([2, 4]);
    sg.beginPath(); sg.moveTo(P.x, Y(pr)); sg.lineTo(P.x + P.w, Y(pr)); sg.stroke();
    sg.setLineDash([]);
    if (!S.narrow) {
      sg.fillStyle = '#7a859e';
      sg.textAlign = 'left';
      sg.fillText(`论文 Llama-2-7B ${pr.toFixed(2)}`, P.x + P.w + 8, Y(pr));
    }
    // 当前层：竖线
    const xL = X(dispL);
    sg.strokeStyle = 'rgba(233,239,249,0.35)';
    sg.lineWidth = 1;
    sg.beginPath(); sg.moveTo(xL, P.y - 4); sg.lineTo(xL, P.y + P.h + 2); sg.stroke();
    // 四条曲线：当前这组最后画、最粗
    const order = VN.map((_, i) => i).filter((i) => i !== S.v).concat([S.v]);
    for (const vi of order) {
      const ms = meta.metrics[VN[vi]];
      const cur = vi === S.v;
      sg.strokeStyle = VAR[VN[vi]].col;
      sg.globalAlpha = cur ? 1 : 0.42;
      sg.lineWidth = cur ? 2 : 1.2;
      sg.setLineDash(VAR[VN[vi]].tag ? [4, 3] : []);
      if (cur) {
        const gr = sg.createLinearGradient(0, P.y, 0, Y(0));
        gr.addColorStop(0, `${VAR[VN[vi]].col}2e`);
        gr.addColorStop(1, `${VAR[VN[vi]].col}00`);
        sg.fillStyle = gr;
        sg.beginPath();
        sg.moveTo(X(0), Y(0));
        ms.forEach((m, L) => sg.lineTo(X(L), Y(Math.max(0, m.r2))));
        sg.lineTo(X(NL - 1), Y(0));
        sg.closePath();
        sg.fill();
      }
      sg.beginPath();
      ms.forEach((m, L) => (L ? sg.lineTo(X(L), Y(m.r2)) : sg.moveTo(X(L), Y(m.r2))));
      sg.stroke();
      sg.setLineDash([]);
      // 当前层上的点（层在补间时按相邻两层插值）
      const a = Math.floor(dispL), b = Math.min(NL - 1, a + 1), t = dispL - a;
      const v = ms[a].r2 + (ms[b].r2 - ms[a].r2) * t;
      sg.fillStyle = VAR[VN[vi]].col;
      sg.beginPath(); sg.arc(xL, Y(v), cur ? 3.6 : 2.4, 0, Math.PI * 2); sg.fill();
      if (cur) { sg.strokeStyle = '#050b17'; sg.lineWidth = 1.5; sg.stroke(); }
    }
    sg.globalAlpha = 1;
    // 右侧直接标注（窄屏只标当前这组）
    const labs = (S.narrow ? [S.v] : VN.map((_, i) => i)).map((vi) => ({ vi, y: Y(meta.metrics[VN[vi]][S.L].r2) }));
    labs.sort((p, q) => p.y - q.y);
    for (let k = 1; k < labs.length; k++) labs[k].y = Math.max(labs[k].y, labs[k - 1].y + 12);
    const over = labs.length ? labs[labs.length - 1].y - (P.y + P.h + 4) : 0;
    if (over > 0) labs.forEach((p) => { p.y -= over; });
    sg.font = `11px ${FONT_SANS}`;
    for (const p of labs) {
      const vn = VN[p.vi], cur = p.vi === S.v;
      const x = P.x + P.w + 8;
      sg.fillStyle = VAR[vn].col;
      sg.fillRect(x, p.y - 1, 7, 2);
      sg.fillStyle = cur ? '#e9eff9' : '#7a859e';
      sg.font = `${cur ? 600 : 400} 11px ${FONT_SANS}`;
      const val = meta.metrics[vn][S.L].r2;
      sg.fillText(S.narrow ? num(val, 2) : `${VAR[vn].short} ${num(val, 2)}`, x + 11, p.y);
    }
    // 横轴：层号
    sg.font = `10px ${FONT_MONO}`;
    sg.textAlign = 'center';
    sg.textBaseline = 'alphabetic';
    for (let L = 0; L < NL; L++) {
      const x = X(L);
      const major = L % 9 === 0 || L === NL - 1;
      sg.fillStyle = 'rgba(150,180,230,0.22)';
      sg.fillRect(x - 0.5, P.y + P.h + 3, 1, major ? 4 : 2);
      if (major && Math.abs(L - S.L) > 1) { sg.fillStyle = '#4b5572'; sg.fillText(String(L), x, CH - 2); }
    }
    // 当前层号的小标签
    const lab = `第 ${S.L} 层`;
    sg.font = `600 10.5px ${FONT_SANS}`;
    const tw = sg.measureText(lab).width + 10;
    const lx = clamp(xL - tw / 2, 0, CW - tw);
    sg.fillStyle = 'rgba(94,240,212,0.16)';
    sg.beginPath(); sg.roundRect ? sg.roundRect(lx, CH - 13, tw, 13, 6.5) : sg.rect(lx, CH - 13, tw, 13); sg.fill();
    sg.fillStyle = '#e9eff9';
    sg.textAlign = 'left';
    sg.fillText(lab, lx + 5, CH - 3);
  }

  function draw() {
    if (!W || !H) return;
    drawMap();
    drawChart();
  }

  /* ---------- 动画 ---------- */

  let tweenMs = TWEEN_MS;
  function retarget(dur = TWEEN_MS) {
    from.set(disp);
    to = D[S.v][S.L];
    fromL = dispL;
    t0 = performance.now();
    tweenMs = dur;
    animating = !reduced && dur > 0;
    if (!animating) { disp.set(to); dispL = S.L; }
    kick();
  }
  function kick() { if (!raf && !dead) raf = requestAnimationFrame(frame); }
  function frame(t) {
    raf = 0;
    if (S.playing && t - lastStep >= STEP_MS) {
      lastStep = t;
      if (S.L >= NL - 1) setPlaying(false);
      else go(S.L + 1);
    }
    if (animating) {
      const k = easeIO((t - t0) / tweenMs);
      for (let j = 0; j < disp.length; j++) disp[j] = from[j] + (to[j] - from[j]) * k;
      dispL = fromL + (S.L - fromL) * k;
      if (k >= 1) animating = false;
    }
    draw();
    if (S.hover >= 0 && animating) placeTip(S.hover);
    if ((animating || S.playing) && !raf) raf = requestAnimationFrame(frame);   // go() 里可能已经排过下一帧
  }

  /* ---------- 状态变化 ---------- */

  function go(L, { quiet = false } = {}) {
    L = clamp(Math.round(L), 0, NL - 1);
    if (L === S.L && !quiet) return;
    S.L = L;
    range.value = String(L);
    retarget();
    updateUI();
    if (!quiet) pushExplain();
  }
  function setVariant(v) {
    if (v === S.v) return;
    S.v = v;
    retarget(TWEEN_MS);
    updateUI();
    pushExplain();
  }
  function setPlaying(p) {
    S.playing = p;
    root.classList.toggle('pb-playing', p);
    $('.pb-play').setAttribute('aria-label', p ? '暂停' : '播放：从第 0 层到第 27 层');
    if (p) {
      if (S.L >= NL - 1) go(0);
      lastStep = performance.now();
      kick();
    }
  }

  function updateUI() {
    const m = meta.metrics[VN[S.v]][S.L];
    root.querySelector('[data-k="L"]').textContent = S.L;
    root.querySelector('[data-k="r2"]').textContent = num(m.r2, 3);
    root.querySelector('[data-k="km"]').textContent = fmtInt(m.km);
    root.querySelector('[data-k="med"]').textContent = fmtInt(m.kmMed);
    range.setAttribute('aria-valuetext', `第 ${S.L} 层，R² ${num(m.r2, 2)}，平均误差 ${fmtInt(m.km)} km`);
    root.querySelectorAll('.pb-seg button').forEach((b) => {
      const on = Number(b.dataset.v) === S.v;
      b.classList.toggle('on', on);
      b.setAttribute('aria-checked', String(on));
    });
    legend.classList.toggle('iso', S.iso >= 0);
    legend.querySelectorAll('.pb-chip').forEach((b) => b.classList.toggle('on', Number(b.dataset.c) === S.iso));
    legend.hidden = S.color === 'err';
    const ramp = root.querySelector('.pb-ramp');
    if (S.color === 'err' && !ramp) legend.insertAdjacentHTML('afterend', rampHTML);
    if (S.color !== 'err' && ramp) ramp.remove();
    root.querySelector('[data-o="color"]').classList.toggle('on', S.color === 'err');
    root.querySelector('[data-o="lines"]').classList.toggle('on', S.lines);
    root.querySelector('[data-o="fx"]').classList.toggle('on', fx.open);
    updateFeed();
    if (S.hover >= 0) placeTip(S.hover);
    if (fx.open) renderFx();
  }

  // 地图左上角：模型此刻读到的是什么
  function updateFeed() {
    const vn = VN[S.v];
    const i = S.hover >= 0 ? S.hover : S.sel >= 0 ? S.sel : -1;
    const nm = `<em>${esc(i >= 0 ? C.name[i] : '城市名')}</em>`;
    const tmpl = vn === 'name' ? nm : `${esc(meta.templates.coords.replace('{}', ''))}${nm}`;
    let h = `<span>模型读到</span><code>${tmpl}</code>`;
    if (vn === 'shuffled') h = `<span class="pb-ctl">对照</span><code>${tmpl}</code><span>训练时坐标被随机打乱</span>`;
    if (vn === 'random') h = `<span class="pb-ctl">对照</span><span>未训练的 Qwen3-0.6B（权重随机初始化）读</span><code>${tmpl}</code>`;
    feed.innerHTML = h;
  }

  /* ---------- 悬停 ---------- */

  function hit(mx, my, rad) {
    let best = -1, bd = rad * rad;
    for (let i = 0; i < n; i++) {
      if (S.iso >= 0 && C.ct[i] !== S.iso) continue;
      const dx = px(disp[2 * i + 1]) - mx, dy = py(disp[2 * i]) - my;
      const d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }
  function tipHTML(i) {
    const p = D[S.v][S.L];
    const la = p[2 * i], lo = p[2 * i + 1];
    const km = errs(S.v, S.L)[i];
    const zh = featured.get(i);
    const country = meta.countries[C.cc[i]][1];
    return `<div class="pb-tip-h"><i style="background:${CT_COL[C.ct[i]]}"></i><b>${esc(C.name[i])}</b><span>${zh ? `${zh} · ` : ''}${esc(country)} · ${meta.continents[C.ct[i]]}</span></div>
      <dl><dt>真实位置</dt><dd>${latTxt(C.lat[i])} ${lonTxt(C.lon[i])}</dd>
      <dt>模型以为</dt><dd>${latTxt(la)} ${lonTxt(lo)}</dd>
      <dt>误差</dt><dd class="pb-err">${fmtInt(km)} km</dd></dl>
      <small>人口 ${popTxt(C.pop[i])} · 第 ${C.fold[i] + 1} 折的探针预测（训练时没见过它）${zh ? ' · 点一下看算式' : ''}</small>`;
  }
  function placeTip(i) {
    if (i < 0) return hideTip();
    tip.innerHTML = tipHTML(i);
    tip.hidden = false;
    const x = px(disp[2 * i + 1]), y = py(disp[2 * i]);
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let lx = x + 14, ly = y + 14;
    if (lx + tw > W - 6) lx = x - 14 - tw;
    if (ly + th > H - 6) ly = y - 14 - th;
    tip.style.transform = `translate(${clamp(lx, 6, Math.max(6, W - tw - 6))}px, ${clamp(ly, 6, Math.max(6, H - th - 6))}px)`;
  }
  function hideTip() { tip.hidden = true; }

  /* ---------- 算式面板 ---------- */

  const defaultFxCity = () => {
    const bj = C.id.indexOf(1816670);
    return bj >= 0 && featured.has(bj) ? bj : C.featured[0][0];
  };
  function openFx(i) {
    fx.open = true;
    if (i >= 0 && featured.has(i)) fx.city = i;
    if (fx.city < 0) fx.city = S.sel >= 0 && featured.has(S.sel) ? S.sel : defaultFxCity();
    fx.key = '';
    fxSide();
    fxEl.hidden = false;
    hideTip();
    updateUI();
    draw();
  }
  // 面板放在城市预测位置的另一侧，别挡住它（只在打开、换城市时决定，换层时不跳）
  function fxSide() {
    const x = px(disp[2 * fx.city + 1]);
    fxEl.classList.toggle('pb-fx-l', !S.narrow && x > W * 0.52);
  }
  function closeFx() {
    fx.open = false;
    fxEl.hidden = true;
    cancelAnimationFrame(fx.countRaf);
    updateUI();
    draw();
  }
  function fxHeader() {
    const opts = [...C.featured].sort((a, b) => C.ct[a[0]] - C.ct[b[0]]).map(([i, zh]) => `<option value="${i}"${i === fx.city ? ' selected' : ''}>${zh} ${esc(C.name[i])}</option>`).join('');
    return `<div class="pb-fx-h"><b>${S.narrow ? '' : '算式 · '}第 ${S.L} 层</b><select aria-label="城市">${opts}</select>
      <div class="pb-fx-tabs"><button type="button" data-t="lat" class="${fx.tab === 'lat' ? 'on' : ''}">纬度</button><button type="button" data-t="lon" class="${fx.tab === 'lon' ? 'on' : ''}">经度</button></div>
      <button type="button" class="pb-fx-x" aria-label="关闭算式">×</button></div><div class="pb-fx-b"></div>`;
  }
  async function renderFx() {
    const i = fx.city;
    const id = C.id[i];
    const key = `${id}|${S.L}|${fx.tab}|${S.narrow}|${S.v}`;
    if (key === fx.key) return;
    fx.key = key;
    fxEl.innerHTML = fxHeader();
    const body = fxEl.querySelector('.pb-fx-b');
    let d = fx.data.get(id);
    if (!d) {
      body.innerHTML = '<div class="pb-fx-wait">正在载入这个城市的算式…</div>';
      d = fetchGzJSON(`${URL_DATA}formula/${id}.json`, ac.signal);
      fx.data.set(id, d);
      d.catch(() => fx.data.delete(id));
    }
    let F;
    try { F = await d; } catch (e) {
      if (!dead && fx.key === key) body.innerHTML = `<div class="pb-fx-wait">${esc(e.message)}</div>`;
      return;
    }
    if (dead || fx.key !== key) return;
    body.innerHTML = fxBody(F, i);
    countUp(body, F.layers[S.L][fx.tab]);
  }
  function fxBody(F, i) {
    const t = F.layers[S.L][fx.tab];
    const isLat = fx.tab === 'lat';
    const word = isLat ? '纬度' : '经度';
    const truth = isLat ? C.lat[i] : C.lon[i];
    const toks = F.tokens.map((s, k) => `<span class="${k === F.tokens.length - 1 ? 'last' : ''}">${esc(s)}</span>`).join('');
    const maxp = Math.max(...t.p.map(Math.abs), Math.abs(t.rest), 1e-9);
    const bar = (p) => `<span class="pb-bar"><i class="${p >= 0 ? 'pos' : 'neg'}" style="width:${(Math.abs(p) / maxp) * 50}%"></i></span>`;
    const rows = t.i.map((dim, k) => `<div class="pb-tr" style="--k:${k}" title="h[${dim}] = ${num(t.h[k], 3)}，训练城市的平均 h̄ = ${num(t.h[k] - t.x[k], 3)}"><span class="i">${dim}</span><span class="pb-cx">${fx3(t.x[k])}</span><span class="o">×</span><span class="pb-cw">${fw(t.w[k])}</span><span class="o">=</span><span class="pb-cp">${fx3(t.p[k])}</span>${bar(t.p[k])}</div>`).join('');
    const nrest = meta.dModel - t.i.length;
    const sum = t.y - t.b;
    const top12 = t.p.reduce((a, b) => a + b, 0);
    const mark16 = t.b + t.cum[4];
    const other = S.v !== 0 ? `<p class="pb-fx-note" style="color:var(--pb-amber)">算式用的是「问坐标」这组真实探针（地图上现在显示的是「${VAR[VN[S.v]].label}」）。</p>` : '';
    return `${other}
      <p class="pb-toks">${esc('<|endoftext|>')}${esc(meta.templates.coords.replace('{}', '').trimEnd())} ${toks}
        <small>取城市名最后一个词元 <b class="pb-cx">${esc(F.tokens[F.tokens.length - 1].trim())}</b> 在第 ${S.L} 层的隐状态 h（${meta.dModel} 维）</small></p>
      <div class="pb-eq">${word} = <span class="pb-cy">b</span> + Σ <span class="pb-cx">(h[i] − h̄[i])</span> × <span class="pb-cw">w[i]</span>
        <small>探针就是一次乘加：1,024 维逐项相乘再加起来。h̄ 是训练城市的平均隐状态，b 是它们的平均${word}。表里是贡献最大的 12 维。</small></div>
      <div class="pb-acc">
        <div class="pb-acc-k">b（训练城市的平均${word}）+ 全部 ${fmtInt(meta.dModel)} 项乘积</div>
        <div class="pb-acc-v">${num(t.b, 3)} <span class="eq">${sum < 0 ? MINUS : '+'}</span> ${Math.abs(sum).toFixed(3)} <span class="eq">=</span> <b data-count>${num(t.y, 3)}</b></div>
        <p>→ ${isLat ? latTxt(t.y) : lonTxt(t.y)}，真实是 ${isLat ? latTxt(truth) : lonTxt(truth)}，差 ${Math.abs(t.y - truth).toFixed(1)}°。
        前 12 维合计 ${fx3(top12)}，其余 ${fmtInt(nrest)} 维合计 ${fx3(t.rest)}。</p>
      </div>
      <div class="pb-tbl">
        <div class="pb-tr pb-th"><span>维 i</span><span class="pb-cx">h − h̄</span><span></span><span class="pb-cw">w</span><span></span><span class="pb-cp">乘积</span><span></span></div>
        ${rows}
        <div class="pb-tr pb-rest" style="--k:${t.i.length}"><span class="i">…</span><span class="t">其余 ${fmtInt(nrest)} 维</span><span class="pb-cp">${fx3(t.rest)}</span>${bar(t.rest)}</div>
      </div>
      <figure class="pb-cum">${cumSVG(t, truth)}
        <figcaption>按 |乘积| 从大到小依次加上（横轴是加了多少项，对数刻度）：加到第 16 项是 ${num(mark16, 1)}°，1,024 项全部加完是 ${num(t.y, 1)}°。
        正乘积合计 ${fx3(t.pos)}，负乘积合计 ${fx3(t.neg)}——大部分互相抵消，答案藏在差额里。</figcaption></figure>
      <p class="pb-fx-note">h̄ 和 w 来自第 ${F.fold + 1} 折的探针：它训练时没见过${esc(featured.get(i) || C.name[i])}（λ = ${fmtInt(F.lam[S.L])}，留一交叉验证选的）。数字全部来自真实的 Qwen3-0.6B 前向计算。</p>`;
  }
  function cumSVG(t, truth) {
    const w = 320, h = 92, l = 34, r = 10, tp = 8, bt = 16;
    const vals = t.cum.map((c) => t.b + c);
    const lo = Math.min(...vals, truth, t.b), hi = Math.max(...vals, truth, t.b);
    const pad = (hi - lo) * 0.08 || 1;
    const Y = (v) => tp + (1 - (v - lo + pad) / (hi - lo + 2 * pad)) * (h - tp - bt);
    const X = (k) => l + (k / 10) * (w - l - r);
    const pts = [`${X(0) - 0.001},${Y(t.b)}`].concat(vals.map((v, k) => `${X(k)},${Y(v)}`));
    const ticks = [0, 4, 8, 10].map((k) => `<text x="${X(k)}" y="${h - 3}" text-anchor="middle">${2 ** k}</text>`).join('');
    return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="累计和曲线">
      <line class="ax" x1="${l}" x2="${w - r}" y1="${h - bt}" y2="${h - bt}"/>
      <line class="tr" x1="${l}" x2="${w - r}" y1="${Y(truth)}" y2="${Y(truth)}"/>
      <text x="${l - 4}" y="${Y(truth) + 3}" text-anchor="end">真实</text>
      <text x="${l - 4}" y="${Y(t.b) + 3}" text-anchor="end">b</text>
      <polyline class="ln" points="${pts.join(' ')}"/>
      <circle class="fin" cx="${X(10)}" cy="${Y(vals[10])}" r="3.2"/>${ticks}</svg>`;
  }
  // 结果的数字从 b 数到真实结果
  function countUp(body, t) {
    cancelAnimationFrame(fx.countRaf);
    const el = body.querySelector('[data-count]');
    if (!el || reduced) return;
    const start = performance.now(), dur = 700;
    const stepf = (now) => {
      const k = easeIO((now - start) / dur);
      el.textContent = num(t.b + (t.y - t.b) * k, 3);
      if (k < 1 && !dead) fx.countRaf = requestAnimationFrame(stepf);
    };
    fx.countRaf = requestAnimationFrame(stepf);
  }

  /* ---------- 讲解 ---------- */

  function explainHTML() {
    const vn = VN[S.v];
    const ms = meta.metrics[vn];
    const m = ms[S.L];
    const best = meta.best[vn];
    const co = meta.metrics.coords, nm = meta.metrics.name, sh = meta.metrics.shuffled, rd = meta.metrics.random;
    const bc = meta.best.coords, bn = meta.best.name, br = meta.best.random;
    const ct = meta.byContinent[S.L];
    const hs = meta.holdout.summary;
    const meanTxt = `${latTxt(meanLat, 0)}、${lonTxt(meanLon, 0)}`;
    const out = [];
    out.push(`<p>Qwen3-0.6B 从没见过地图，只读过文字。我们把 ${fmtInt(n)} 个城市一个个喂给它，不看它回答什么，只取它读完城市名那一刻、城市名最后一个词元的<b>隐状态</b>（每层 1,024 个数），再给每一层训练一个<b>线性探针</b>：每个数乘一个权重、加起来，直接读出纬度和经度。地图上每个点都是探针对一个<b>训练时没见过</b>的城市的预测，细线连到它的真实位置。</p>`);
    let p = `<p><b>第 ${S.L} 层</b>（${VAR[vn].tag ? `对照 · ${VAR[vn].label}` : VAR[vn].label}）：R² = <b>${num(m.r2, 3)}</b>（纬度 ${num(m.r2Lat, 2)}、经度 ${num(m.r2Lon, 2)}），预测点离真实位置平均 <b>${fmtInt(m.km)} km</b>，一半城市在 ${fmtInt(m.kmMed)} km 以内。`;
    if (vn === 'coords' || vn === 'name') {
      const r3 = ms[3].r2, r10 = ms[10].r2;
      if (S.L <= 3) p += ` 还很浅：隐状态里主要还是“这几个词元是什么”，地理信息不多，点大多挤在所有城市的平均位置（${meanTxt}）附近。`;
      else if (S.L < 10) p += ` 从第 4 层起 R² 明显上升（第 3 层 ${num(r3, 2)} → 第 10 层 ${num(r10, 2)}）：模型在这几层把“这个名字指的是哪儿”调了出来，各大洲的城市开始分开、往各自的位置移动。`;
      else p += ` 第 10 层之后进入平台：R² 在 ${num(Math.min(...ms.slice(10).map((x) => x.r2)), 2)}–${num(ms[best].r2, 2)} 之间，第 ${best} 层最高。论文在 Llama-2 上看到的也是这样——前一半的层迅速变好，然后进入平台。`;
    }
    out.push(p + '</p>');
    if (vn === 'coords') {
      out.push(`<p>这组用的是论文里问坐标的提示 <code>What are the lat/lon coordinates of …</code>，取的仍是城市名的最后一个词元（模型还没开始回答）。只给名字（论文的主实验）的话，最好的一层 R² 是 ${num(nm[bn].r2, 2)}，比加提示低（${num(co[bc].r2, 2)}）：论文说提示对 Llama-2-70B 几乎没有影响，这个小模型却要先明白“这是个地名”，地理信息才更清楚。</p>`);
    } else if (vn === 'name') {
      out.push(`<p>只给城市名（论文的主实验）：最好的是第 ${bn} 层，R² ${num(nm[bn].r2, 2)}；加上问坐标的提示能到 ${num(co[bc].r2, 2)}。论文说提示对 Llama-2-70B 几乎没有影响，这个 6 亿参数的小模型差别明显。</p>`);
    } else if (vn === 'shuffled') {
      out.push(`<p><b>对照 · 打乱标签</b>：把训练集里的城市和坐标随机配对，别的都不变。探针学不到任何规律，每一层的 R² 都在 0 附近（这一层 ${num(m.r2, 3)}），所有点塌向平均位置。说明地图不是探针自己“背”出来的：一个线性探针没有这个本事，能读出来的，是隐状态里本来就有的东西。</p>`);
    } else {
      out.push(`<p><b>对照 · 未训练的模型</b>：结构一样、权重随机初始化的 Qwen3-0.6B，同样的提示、同样的探针。它只能读到名字的字面（拼写）带来的一点线索——最好的一层 R² 也只有 ${num(rd[br].r2, 2)}，地图不成形。真实模型里的地理信息是从训练文本里学来的。</p>`);
    }
    if (vn === 'coords' || vn === 'name') {
      const order = ct.map((v, c) => [v[0], c]).sort((a, b) => a[0] - b[0]);
      const nameOf = (c) => meta.continents[c];
      out.push(`<p>哪里准、哪里不准（这一层的中位误差，「问坐标」这组）：${nameOf(order[0][1])}最准（${fmtInt(order[0][0])} km），其次${nameOf(order[1][1])}（${fmtInt(order[1][0])} km）；${nameOf(order[5][1])}最差（${fmtInt(order[5][0])} km）。读不准的时候，岭回归的预测会被拉向所有城市的平均位置（${meanTxt}，撒哈拉一带）：离这里越远、城市越少的地方（大洋洲只有 ${C.ct.filter((c) => c === 5).length} 个城市），被拉得越远。</p>`);
    }
    out.push(`<p class="dim">要注意：线性探针能读出来，不等于模型真的在用它（论文另做了干预实验，这里没有）。而且 0.6B 的小模型读得并不准：最好的一层 R² ${num(co[bc].r2, 2)}、平均误差约 ${fmtInt(co[bc].km)} km，论文里 70 亿参数的 Llama-2-7B 是 ${meta.paper.r2_7b}。把一个国家的城市全部藏起来不给探针看，对这些国家的平均误差从 ${fmtInt(hs.kmNominal)} km 涨到 ${fmtInt(hs.kmHeld)} km——探针学到的有一部分是“这个国家大概在哪”。</p>`);
    out.push(`<p class="dim">复现：Gurnee &amp; Tegmark 2023《Language Models Represent Space and Time》（arXiv:2310.02207）。城市数据 GeoNames（CC BY 4.0），海岸线 Natural Earth（公共领域）。</p>`);
    return out.join('');
  }
  function pushExplain(now = false) {
    if (!onExplain) return;
    clearTimeout(explainTimer);
    const send = () => { if (!dead) onExplain(explainHTML()); };
    if (now) send(); else explainTimer = setTimeout(send, 120);
  }

  /* ---------- 事件 ---------- */

  on(range, 'input', () => { setPlaying(false); go(Number(range.value)); });
  on($('.pb-play'), 'click', () => setPlaying(!S.playing));
  on($('.pb-seg'), 'click', (e) => {
    const b = e.target.closest('button[data-v]');
    if (b) setVariant(Number(b.dataset.v));
  });
  on(legend, 'click', (e) => {
    const b = e.target.closest('.pb-chip');
    if (!b) return;
    const c = Number(b.dataset.c);
    S.iso = S.iso === c ? -1 : c;
    if (S.sel >= 0 && S.iso >= 0 && C.ct[S.sel] !== S.iso) S.sel = -1;
    updateUI();
    draw();
  });
  on($('.pb-opts'), 'click', (e) => {
    const b = e.target.closest('button[data-o]');
    if (!b) return;
    const o = b.dataset.o;
    if (o === 'color') S.color = S.color === 'ct' ? 'err' : 'ct';
    if (o === 'lines') S.lines = !S.lines;
    if (o === 'fx') { if (fx.open) closeFx(); else openFx(S.sel); return; }
    updateUI();
    draw();
  });
  on(fxEl, 'click', (e) => {
    if (e.target.closest('.pb-fx-x')) return closeFx();
    const t = e.target.closest('[data-t]');
    if (t) { fx.tab = t.dataset.t; renderFx(); }
  });
  on(fxEl, 'change', (e) => {
    if (e.target.tagName !== 'SELECT') return;
    fx.city = Number(e.target.value);
    S.sel = fx.city;
    fxSide();
    renderFx();
    updateFeed();
    draw();
  });
  let touch = false;
  on(cv, 'pointermove', (e) => {
    if (e.pointerType === 'touch') return;
    const r = cv.getBoundingClientRect();
    const i = hit(e.clientX - r.left, e.clientY - r.top, 12);
    if (i === S.hover) return;
    S.hover = i;
    if (i >= 0) placeTip(i); else if (S.sel >= 0) placeTip(S.sel); else hideTip();
    cv.style.cursor = i >= 0 ? 'pointer' : 'crosshair';
    updateFeed();
    draw();
  });
  on(cv, 'pointerleave', () => {
    if (touch) return;
    S.hover = -1;
    if (S.sel >= 0) placeTip(S.sel); else hideTip();
    updateFeed();
    draw();
  });
  on(cv, 'pointerdown', (e) => { touch = e.pointerType === 'touch'; });
  on(cv, 'click', (e) => {
    const r = cv.getBoundingClientRect();
    const i = hit(e.clientX - r.left, e.clientY - r.top, touch ? 22 : 12);
    S.sel = i === S.sel ? -1 : i;
    if (touch) S.hover = -1;
    if (S.sel >= 0 && featured.has(S.sel) && fx.open) { fx.city = S.sel; fxSide(); renderFx(); }
    if (S.sel >= 0) placeTip(S.sel); else hideTip();
    updateFeed();
    draw();
  });
  on(root, 'keydown', (e) => {
    if (e.target.closest('select, .pb-range')) return;
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      setPlaying(false);
      go(S.L + (e.key === 'ArrowRight' ? 1 : -1));
    } else if (e.key === ' ' && e.target === root) {
      e.preventDefault();
      setPlaying(!S.playing);
    } else if (e.key === 'Escape' && fx.open) closeFx();
  });

  ro = new ResizeObserver(() => { if (!dead) resize(); });
  ro.observe(root);
  ro.observe(stage);
  ro.observe(scrub);
  range.value = String(S.L);
  // 开场：所有点先挤在所有城市的平均位置，再散开到当前这一层的预测
  for (let i = 0; i < n; i++) { disp[2 * i] = meanLat; disp[2 * i + 1] = meanLon; }
  resize();
  updateUI();
  retarget(1100);
  pushExplain(true);

  function destroy() {
    if (dead) return;
    dead = true;
    ac.abort();
    cancelAnimationFrame(raf);
    cancelAnimationFrame(fx.countRaf);
    clearTimeout(explainTimer);
    ro?.disconnect();
    root.remove();
    dropCSS();
  }

  return {
    setLayer(L) {
      if (dead || !Number.isFinite(Number(L))) return;
      setPlaying(false);
      go(Number(L));
    },
    destroy,
  };
}
