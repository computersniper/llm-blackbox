"""把训练页的数据拆成“首屏小文件 + 按需分块”，让页面先用几十 KB 画出首屏，其余进入对应视图时再取。

train_tiny.py / qwen_step.py 导出时直接调用这里；也可以从整份旧格式（一个 tiny.json + tiny.bin、一个完整的 qwen.json）重新切分：
    python tools/train/split_data.py --from-git 4435f8b      # 从这个提交里的旧格式文件切分
    python tools/train/split_data.py --tiny-json a.json --tiny-bin a.bin --qwen-json q.json
切分完会把写出的文件读回来拼成原样，逐字节 / 逐个数和原始数据核对，不一致就报错。

小模型（唐宋诗）：
  tiny.json / tiny.bin（各带 .gz）  首屏：元数据（不含每个检查点的 AdamW 算式）、每一步的损失 / 学习率 / 梯度范数、
                                    批次第 0 行的字（流水线和步骤标签要用）、损失地形
  tiny/ckNN.bin.gz                 D1 训练全程，第 NN 个检查点：留出诗的逐字概率与前 5 名、24 个注意力头（只存下三角）、
                                    嵌入 PCA、三块权重 / 梯度的 48×48 局部
  tiny/stNN.bin.gz                 D2–D4 一步之内，第 NN 个检查点：批次第 0 行的概率与前 5 名、更新后的概率、逻辑透镜、
                                    残差范数与梯度、每个参数张量的梯度范数、4 个权重的 AdamW 算式（json）
  tiny/feat.bin.gz                 D3 更新：4 个权重每 4 步一个点的 w / g / m / v
Qwen3-0.6B：
  qwen.json（带 .gz）               首屏和 D1：去掉逐层的逻辑透镜、残差范数 / 梯度、每个参数张量的梯度范数
  qwen/stN.json.gz                 D2–D4，第 N 步：上面去掉的那几项（第 0 步另带“全都算”时每个张量的梯度范数）

分块只存 .gz（浏览器用 DecompressionStream 解压）；首屏文件另留未压缩的一份，给不支持解压的浏览器。
.bin 分块的格式（解压后）：u32 头长度 + JSON 头 {bin: {名字: {dtype, shape, offset, shuffle?, tri?}}, json?: {...}}
+ 补齐到 8 字节 + 数组。shuffle = 按字节分面存放（先放所有数的第 0 个字节，再放第 1 个……），只为让 gzip 压得更小；
tri = 注意力只存下三角 j ≤ i（因果遮罩，上三角本来就全是 0）。两者都不改变任何数值。
"""
import argparse
import copy
import gzip
import json
import math
import pathlib
import struct
import subprocess

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'public' / 'train' / 'data'

CORE = ['loss', 'lr', 'gnorm', 'row_ids', 'land']
CK = ['val_p', 'val_top_id', 'val_top_p', 'attn', 'pca', 'wcrop', 'gcrop']
ST = ['row_p', 'row_top_id', 'row_top_p', 'row_p_after', 'lens_top', 'lens_p', 'resid_norm', 'resid_grad', 'tgrad']
FEAT = ['feat_w', 'feat_g', 'feat_m', 'feat_v']
QWEN_ST = ['lens', 'residNorm', 'residGrad', 'gradNorms']   # qwen.steps[k] 里挪到分块的字段


class Pack:
    """数组收集器：记下 dtype / shape / offset，浮点数组按字节分面存放。"""

    def __init__(self):
        self.parts, self.spec, self.off = [], {}, 0

    def add(self, name, arr, tri=None):
        a = np.ascontiguousarray(arr)
        pad = (-self.off) % 8
        if pad:
            self.parts.append(b'\0' * pad)
            self.off += pad
        spec = {'dtype': a.dtype.name, 'shape': list(a.shape), 'offset': self.off}
        b = a.tobytes()
        if a.dtype.kind == 'f':
            b = np.frombuffer(b, np.uint8).reshape(-1, a.dtype.itemsize).T.tobytes()
            spec['shuffle'] = True
        if tri:
            spec['tri'] = tri
        self.spec[name] = spec
        self.parts.append(b)
        self.off += len(b)

    def bytes(self):
        return b''.join(self.parts)


def container(pack, js=None):
    head = {'bin': pack.spec}
    if js is not None:
        head['json'] = js
    h = json.dumps(head, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    pre = struct.pack('<I', len(h)) + h
    pre += b'\0' * ((-len(pre)) % 8)
    return pre + pack.bytes()


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def gz(b):
    return gzip.compress(b, 9, mtime=0)


def arrays(meta, buf):
    """旧格式：meta['bin'] 描述 buf 里的每个数组。"""
    out = {}
    for k, s in meta['bin'].items():
        out[k] = np.frombuffer(buf, dtype=s['dtype'], count=math.prod(s['shape']), offset=s['offset']).reshape(s['shape'])
    return out


def tri_rows(n):
    return np.tril_indices(n)


# ---------------------------------------------------------------- 写出

def split_tiny(meta, A, out=OUT):
    """meta：旧格式的完整元数据（含 ckpts[].adam）；A：名字 → 数组（[K, ...] 按检查点排第一维）。"""
    out = pathlib.Path(out)
    K = len(meta['ckpts'])
    Lv = A['attn'].shape[-1]
    ti, tj = tri_rows(Lv)
    files = {}
    # 首屏
    core = Pack()
    for n in CORE:
        core.add(n, A[n])
    m = copy.deepcopy(meta)
    for c in m['ckpts']:
        c.pop('adam')
    m['bin'] = core.spec
    m['chunks'] = {'dir': 'tiny', 'ck': CK, 'st': ST, 'feat': FEAT, 'note': '按需分块，见 tools/train/split_data.py'}
    files['tiny.json'] = dumps(m)
    files['tiny.bin'] = core.bytes()
    # 每个检查点两块
    for k in range(K):
        p = Pack()
        for n in CK:
            if n == 'attn':
                p.add(n, A[n][k][:, :, ti, tj], tri=Lv)
            else:
                p.add(n, A[n][k])
        files[f'tiny/ck{k:02d}.bin'] = container(p)
        p = Pack()
        for n in ST:
            p.add(n, A[n][k])
        files[f'tiny/st{k:02d}.bin'] = container(p, {'adam': meta['ckpts'][k]['adam']})
    p = Pack()
    for n in FEAT:
        p.add(n, A[n])
    files['tiny/feat.bin'] = container(p)
    return write(files, out, keep_raw={'tiny.json', 'tiny.bin'})


def split_qwen(meta, out=OUT):
    out = pathlib.Path(out)
    m = copy.deepcopy(meta)
    files = {}
    for k, s in enumerate(m['steps']):
        js = {key: s.pop(key) for key in QWEN_ST}
        if k == 0:
            js['pretrainCompare'] = {'gradNorms': m['pretrainCompare'].pop('gradNorms')}
        files[f'qwen/st{k}.json'] = dumps(js)
    m['chunks'] = {'dir': 'qwen', 'st': QWEN_ST, 'note': '按需分块，见 tools/train/split_data.py'}
    files['qwen.json'] = dumps(m)
    return write(files, out, keep_raw={'qwen.json'})


def write(files, out, keep_raw):
    sizes = {}
    for name, blob in files.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        z = gz(blob)
        (out / (name + '.gz')).write_bytes(z)
        if name in keep_raw:
            path.write_bytes(blob)
        elif path.exists():
            path.unlink()
        sizes[name] = (len(blob), len(z))
    return sizes


# ---------------------------------------------------------------- 读回来核对

def read_container(b):
    (h,) = struct.unpack_from('<I', b, 0)
    head = json.loads(b[4:4 + h].decode('utf-8'))
    base = 4 + h + ((-(4 + h)) % 8)
    out = {}
    for n, s in head['bin'].items():
        dt = np.dtype(s['dtype'])
        cnt = math.prod(s['shape'])
        raw = b[base + s['offset']: base + s['offset'] + cnt * dt.itemsize]
        if s.get('shuffle'):
            raw = np.frombuffer(raw, np.uint8).reshape(dt.itemsize, cnt).T.tobytes()
        a = np.frombuffer(raw, dt).reshape(s['shape'])
        if s.get('tri'):
            Lv = s['tri']
            full = np.zeros(a.shape[:-1] + (Lv, Lv), dt)
            ti, tj = tri_rows(Lv)
            full[..., ti, tj] = a
            a = full
        out[n] = a
    return out, head.get('json')


def verify_tiny(meta, A, out=OUT):
    out = pathlib.Path(out)
    m = json.loads(gzip.decompress((out / 'tiny.json.gz').read_bytes()))
    assert m == json.loads((out / 'tiny.json').read_bytes()), 'tiny.json 和 .gz 不一致'
    core_bytes = gzip.decompress((out / 'tiny.bin.gz').read_bytes())
    assert core_bytes == (out / 'tiny.bin').read_bytes(), 'tiny.bin 和 .gz 不一致'
    # 首屏的 tiny.bin 没有头，描述在 tiny.json 的 bin 里：补一个头，按分块的格式读
    hb = dumps({'bin': m['bin']})
    pre = struct.pack('<I', len(hb)) + hb
    pre += b'\0' * ((-len(pre)) % 8)
    got, _ = read_container(pre + core_bytes)
    K = len(m['ckpts'])
    per = {n: [] for n in CK + ST}
    adam = []
    for k in range(K):
        a, _ = read_container(gzip.decompress((out / f'tiny/ck{k:02d}.bin.gz').read_bytes()))
        for n in CK:
            per[n].append(a[n])
        a, js = read_container(gzip.decompress((out / f'tiny/st{k:02d}.bin.gz').read_bytes()))
        for n in ST:
            per[n].append(a[n])
        adam.append(js['adam'])
    for n, xs in per.items():
        got[n] = np.stack(xs)
    a, _ = read_container(gzip.decompress((out / 'tiny/feat.bin.gz').read_bytes()))
    got.update(a)
    assert set(got) == set(A), f'数组名不一致：{set(got) ^ set(A)}'
    for n, x in A.items():
        y = got[n]
        assert x.dtype == y.dtype and x.shape == y.shape, f'{n} 形状 / 类型不一致'
        assert np.ascontiguousarray(x).tobytes() == np.ascontiguousarray(y).tobytes(), f'{n} 字节不一致'
    full = copy.deepcopy(m)
    del full['chunks']
    for c, ad in zip(full['ckpts'], adam):
        c['adam'] = ad
    ref = {k: v for k, v in meta.items() if k != 'bin'}
    del full['bin']
    assert full == ref, 'tiny 元数据拼回去和原来不一致'
    return len(A)


def verify_qwen(meta, out=OUT):
    out = pathlib.Path(out)
    m = json.loads(gzip.decompress((out / 'qwen.json.gz').read_bytes()))
    assert m == json.loads((out / 'qwen.json').read_bytes()), 'qwen.json 和 .gz 不一致'
    del m['chunks']
    for k, s in enumerate(m['steps']):
        js = json.loads(gzip.decompress((out / f'qwen/st{k}.json.gz').read_bytes()))
        if k == 0:
            m['pretrainCompare']['gradNorms'] = js.pop('pretrainCompare')['gradNorms']
        s.update(js)
        # 字段顺序跟原来一致（json 比较不看顺序，这里只为了好读）
        m['steps'][k] = {key: s[key] for key in meta['steps'][k]}
    m['pretrainCompare'] = {key: m['pretrainCompare'][key] for key in meta['pretrainCompare']}
    assert m == meta, 'qwen 元数据拼回去和原来不一致'
    assert dumps(m) == dumps(meta), 'qwen 重新序列化后的字节不一致'


def report(sizes):
    tot = sum(z for _, z in sizes.values())
    for name in sorted(sizes):
        if '/ck' in name and not name.endswith('00.bin') and not name.endswith('40.bin'):
            continue
        if '/st' in name and name.startswith('tiny') and not name.endswith('00.bin'):
            continue
        r, z = sizes[name]
        print(f'  {name:22s} {r / 1024:7.1f} KB  gz {z / 1024:6.1f} KB')
    print(f'  （共 {len(sizes)} 个文件，gz 合计 {tot / 1024:.0f} KB）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-git', help='从这个提交里读旧格式的 public/train/data/tiny.json、tiny.bin、qwen.json')
    ap.add_argument('--tiny-json')
    ap.add_argument('--tiny-bin')
    ap.add_argument('--qwen-json')
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args()
    if a.from_git:
        show = lambda p: subprocess.run(['git', 'show', f'{a.from_git}:{p}'], cwd=ROOT, check=True, capture_output=True).stdout
        tj, tb, qj = show('public/train/data/tiny.json'), show('public/train/data/tiny.bin'), show('public/train/data/qwen.json')
    else:
        tj, tb, qj = (pathlib.Path(p).read_bytes() for p in (a.tiny_json, a.tiny_bin, a.qwen_json))
    tmeta, qmeta = json.loads(tj), json.loads(qj)
    assert dumps(tmeta) == tj and dumps(qmeta) == qj, '原始 json 不是 dumps 的输出，无法逐字节核对'
    A = arrays(tmeta, tb)
    print('小模型：')
    report(split_tiny(tmeta, A, a.out))
    n = verify_tiny(tmeta, A, a.out)
    print(f'  核对通过：{n} 个数组逐字节一致，元数据（含 41 个检查点的 AdamW 算式）一致')
    print('Qwen3-0.6B：')
    report(split_qwen(qmeta, a.out))
    verify_qwen(qmeta, a.out)
    print('  核对通过：拼回去的 qwen.json 与原文件逐字节一致')


if __name__ == '__main__':
    main()
