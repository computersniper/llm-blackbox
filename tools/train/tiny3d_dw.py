"""第二章 3D 机器的补充数据：从唐宋诗小模型的 41 个检查点（train_tiny.py 存的 stepNNNNN.pt）里，CPU 上算出

  * 三块权重局部（和 tiny/ckNN 的 wcrop 同一位置：嵌入最常见 48 字 × 前 48 维、第 2 层 W_q、第 4 层 W_down 左上角 48×48）
    从这个检查点到下一个检查点的真实变化 ΔW（float32 相减再量化；tiny/ckNN 里的 wcrop 是 int8，
    前几个检查点之间只差一个 lr ≈ 1.5e-5，比 int8 的量化间隔小 40 倍，直接相减全是噪声，所以单独导出）；
  * 每个参数张量的均方根 RMS(W)，以及到下一个检查点的 RMS(ΔW)（68 个张量）。

前 4 个检查点（第 0→1→2→3→4 步）之间正好隔一步，ΔW 就是那一步 AdamW 的更新；之后隔几步到几百步，是累计的变化。
最后一个检查点（第 3999 步）后面没有检查点，只有 RMS(W)。

输出 public/train/data/tiny3d/dwNN.bin.gz，格式和 split_data.py 的分块一样（u32 头长度 + JSON 头 + 数组），写完读回来核对。

    python tools/train/tiny3d_dw.py --ckdir /mnt/d/cjc/checkpoints/tiny-poetry
"""
import argparse
import gzip
import json
import pathlib
import struct

import numpy as np
import torch

import split_data

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / 'public' / 'train' / 'data'
CROP = 48


def crop(sd, name):
    t = sd[name].float()
    if name.endswith('embed_tokens.weight'):
        return t[1:CROP + 1, :CROP].numpy()
    return t[:CROP, :CROP].numpy()


def q8(a):
    a = np.asarray(a, dtype=np.float32)
    s = float(np.abs(a).max()) / 127 or 1e-12
    return np.clip(np.round(a / s), -127, 127).astype(np.int8), s


def rms(t):
    return float(t.float().pow(2).mean().sqrt())


def read_chunk(b):
    h = struct.unpack('<I', b[:4])[0]
    head = json.loads(b[4:4 + h])
    base = 4 + h + ((8 - ((4 + h) % 8)) % 8)
    out = {}
    for k, s in head['bin'].items():
        n = int(np.prod(s['shape']))
        dt = np.dtype(s['dtype'])
        raw = np.frombuffer(b, np.uint8, n * dt.itemsize, base + s['offset'])
        if s.get('shuffle'):
            raw = raw.reshape(dt.itemsize, n).T.copy().reshape(-1)
        out[k] = np.frombuffer(raw.tobytes(), dt).reshape(s['shape'])
    return out, head.get('json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckdir', default='/mnt/d/cjc/checkpoints/tiny-poetry')
    a = ap.parse_args()
    meta = json.loads((DATA / 'tiny.json').read_text(encoding='utf-8'))
    names = [c['name'] for c in meta['crops']]
    tensors = meta['tensors']
    steps = [c['t'] for c in meta['ckpts']]
    out = DATA / 'tiny3d'
    out.mkdir(parents=True, exist_ok=True)
    load = lambda t: torch.load(pathlib.Path(a.ckdir) / f'step{t:05d}.pt', map_location='cpu')
    cur = load(steps[0])
    total = 0
    for k, t in enumerate(steps):
        nxt = load(steps[k + 1]) if k + 1 < len(steps) else None
        # 核对：检查点里的权重和导出的 int8 局部是同一份（差别不超过量化间隔的一半）
        ck, _ = read_chunk(gzip.decompress((DATA / f'tiny/ck{k:02d}.bin.gz').read_bytes()))
        for c, n in enumerate(names):
            q = ck['wcrop'][c].astype(np.float32) * meta['ckpts'][k]['wScale'][c]
            err = np.abs(q - crop(cur, n)).max()
            assert err <= meta['ckpts'][k]['wScale'][c] * 0.5001, (k, n, err)
        pack = split_data.Pack()
        js = {'t': t, 'wRms': [round(rms(cur[n]), 7) for n in tensors]}
        if nxt is not None:
            dw = [crop(nxt, n) - crop(cur, n) for n in names]
            qs = [q8(d) for d in dw]
            pack.add('dwcrop', np.stack([x[0] for x in qs]))
            js['t1'] = steps[k + 1]
            js['dwScale'] = [float(f'{x[1]:.9g}') for x in qs]
            js['dwRms'] = [float(f'{rms(nxt[n] - cur[n]):.6g}') for n in tensors]
        b = split_data.container(pack, js)
        f = out / f'dw{k:02d}.bin.gz'
        f.write_bytes(split_data.gz(b))
        total += f.stat().st_size
        # 读回来核对
        back, bj = read_chunk(gzip.decompress(f.read_bytes()))
        if nxt is not None:
            for c in range(3):
                d = back['dwcrop'][c].astype(np.float32) * bj['dwScale'][c]
                assert np.abs(d - dw[c]).max() <= bj['dwScale'][c] * 0.5001
            print(f'[{k:2d}] 第 {t} → {steps[k + 1]} 步  max|ΔW| 局部 {[f"{np.abs(x).max():.2e}" for x in dw]}  {f.stat().st_size} B')
        else:
            print(f'[{k:2d}] 第 {t} 步（最后一个检查点，只有 RMS(W)）  {f.stat().st_size} B')
        cur = nxt
    print(f'共 {total / 1024:.1f} KB，核对通过')


if __name__ == '__main__':
    main()
