// bfloat16 的比特与数值互转。模型权重以 bf16 存储，转成 float32 是精确的。

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

export function bf16Bits(x) {
  f32[0] = x;
  const b = u32[0];
  const r = (b + 0x7fff + ((b >>> 16) & 1)) >>> 16; // 就近舍入到偶数（对本来就是 bf16 的值是精确的）
  return Array.from({ length: 16 }, (_, i) => (r >>> (15 - i)) & 1);
}

export function bf16Value(bits) {
  let v = 0;
  for (const b of bits) v = v * 2 + b;
  u32[0] = v << 16;
  const s = bits[0], e = parseInt(bits.slice(1, 9).join(''), 2), m = parseInt(bits.slice(9).join(''), 2);
  return { value: f32[0], s, e, m };
}

export function fmtSci(v) {
  if (Number.isNaN(v)) return 'NaN';
  if (!Number.isFinite(v)) return v > 0 ? '+∞' : '−∞';
  if (v === 0) return '0';
  const a = Math.abs(v);
  const s = a >= 1e4 || a < 1e-3 ? a.toExponential(3).replace(/e([+-])(\d+)/, (_, sg, d) => ` × 10^${sg === '-' ? '−' : ''}${d}`) : a.toPrecision(5);
  return (v < 0 ? '−' : '') + s;
}
