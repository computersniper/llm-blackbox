// 核对网页里的 JS 前向（public/world/js/nn.js）和 PyTorch 逐元素一致。
// 参考输出由 export_world.py 写到 tools/world/ref/：同一份 float16 权重，在 CPU 上分别用 float32 和 float64 前向。
//   node tools/world/check_nn.mjs            # 打印误差，写 public/world/data/check.json
import { readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { unpack, WorldModel, HID } from '../../public/world/js/nn.js';
import { Game, toCHW } from '../../public/world/js/game.js';

const here = dirname(fileURLToPath(import.meta.url));
const pub = join(here, '../../public/world/data');
const meta = JSON.parse(readFileSync(join(pub, 'model.json'), 'utf8'));
const bin = readFileSync(join(pub, 'model.bin'));
const model = new WorldModel(meta, unpack(meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength)));
const rm = JSON.parse(readFileSync(join(here, 'ref/ref.json'), 'utf8'));
const rb = gunzipSync(readFileSync(join(here, 'ref/ref.bin.gz')));
const R = (name) => { const a = rm.arrays[name]; return a && new Float32Array(rb.buffer.slice(rb.byteOffset + a.offset, rb.byteOffset + a.offset + a.n * 4)); };

// ---------- 用 game.js 重放参考的那一局（和 Python 版逐像素相同）
const g = new Game(rm.seed);
const seq = new Uint8Array((rm.T + 1) * 4096);
seq.set(g.render(), 0);
rm.acts.forEach((a, t) => { g.step(a); seq.set(g.render(), (t + 1) * 4096); });
const frames = rm.ids.map((i) => seq.subarray(i * 4096, (i + 1) * 4096));

// ---------- JS 前向
const t0 = performance.now();
const enc = frames.map((f) => model.encode(toCHW(f), true));
const dec = [0, 1].map((i) => model.decode(enc[i].mu, true));
const T = rm.T;
const js = { mu: [], lv: [], seq_z: [], seq_h: [], seq_head: [], seq_logpi: [] };
for (const e of enc) { js.mu.push(...e.mu); js.lv.push(...e.lv); }
js.e1_0 = enc[0].e1; js.e2_0 = enc[0].e2; js.e3_0 = enc[0].e3; js.e4_0 = enc[0].e4;
js.dfc_0 = dec[0].dfc; js.d1_0 = dec[0].d1; js.d2_0 = dec[0].d2; js.d3_0 = dec[0].d3;
js.y = [...dec[0].y, ...dec[1].y];
let h = new Float32Array(HID), c = new Float32Array(HID);
for (let t = 0; t < T; t++) {
  const z = model.encode(toCHW(seq.subarray(t * 4096, (t + 1) * 4096))).mu;
  js.seq_z.push(...z);
  const L = model.lstm(z, rm.acts[t], h, c);
  h = L.h; c = L.c;
  js.seq_h.push(...h);
  const M = model.mdn(h);
  js.seq_head.push(...M.raw);
  js.seq_logpi.push(...M.logpi);
}
js.seq_c = c;
const ms = performance.now() - t0;

// ---------- 比较
const rows = [];
const out = {};
for (const name of Object.keys(js)) {
  const a = Float32Array.from(js[name]);
  for (const [tag, label] of [['', 'float32'], ['f64_', 'float64']]) {
    const b = R(tag + name), spec = rm.arrays[tag + name];
    if (!b) continue;
    if (spec.size !== a.length) { rows.push([name, label, `长度不同 ${a.length} vs ${spec.size}`]); continue; }
    // 大张量只存了等间隔抽出的元素：第 i 个参考值对应 JS 的第 i × step 个
    let mx = 0, sum = 0, mag = 0;
    for (let i = 0; i < b.length; i++) { const d = Math.abs(a[i * spec.step] - b[i]); mx = Math.max(mx, d); sum += d; mag = Math.max(mag, Math.abs(b[i])); }
    rows.push([name, label, b.length, mx, sum / b.length, mag]);
    out[name] = out[name] || { n: a.length, compared: b.length, maxAbs: mag };
    out[name][label] = { max: mx, mean: sum / b.length };
  }
}
const fmt = (v) => (typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toExponential(2)) : v);
console.log('张量'.padEnd(10), '参考'.padEnd(8), '元素数'.padStart(6), '最大误差'.padStart(10), '平均误差'.padStart(10), '最大 |值|'.padStart(10));
for (const r of rows) console.log(String(r[0]).padEnd(12), String(r[1]).padEnd(9), ...r.slice(2).map((v) => fmt(v).padStart(10)));
const worst32 = Math.max(...Object.values(out).map((o) => o.float32.max));
const worst64 = Math.max(...Object.values(out).filter((o) => o.float64).map((o) => o.float64.max));
const yErr = out.y.float32.max;
console.log(`\n${Object.keys(out).length} 个张量（大的抽样比较），和 PyTorch float32 的最大绝对误差 ${worst32.toExponential(2)}，和 float64 ${worst64.toExponential(2)}；输出画面（0–1）最大误差 ${yErr.toExponential(2)}；JS 前向共 ${ms.toFixed(0)} ms`);
writeFileSync(join(pub, 'check.json'), JSON.stringify({ worst32, worst64, yErr, tensors: out, frames: frames.length, seqSteps: T, node: process.version }, null, 1));
process.exit(worst32 < 1e-3 ? 0 : 1);
