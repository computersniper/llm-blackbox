"""把训练好的世界模型导出给网页（train.py --stage export 调用）。

public/world/data/
  model.bin(.gz)   全部权重，float16 连续存放（V 编码器 / 解码器、M 的 LSTM 和 MDN 头、C）
  model.json(.gz)  每个张量的形状和偏移；训练记录（采集、V、M、C 的设置、耗时、损失曲线）；
                   潜变量每一维的统计和“含义”（和游戏状态的相关）；测试集样例；梦与现实的平均差异曲线
tools/world/ref/   给 check_nn.mjs 用的参考输出：用存成 float16 再读回来的权重，在 CPU 上 float32 前向
"""
import base64
import gzip
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game import PALETTE, Game  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
Z, NA, HID, KMIX = 32, 3, 256, 5
FEATS = {
    'car_x': '小车的左右位置',
    'road_near': '车旁的路往左还是往右',
    'road_mid': '中段的路往左还是往右',
    'road_far': '远处的路往左还是往右',
    'bend': '前方是左弯还是右弯',
    'car_off': '小车偏离路中间多少',
    'n_obs': '画面里有几辆抛锚车',
    'ob_y': '最近那辆抛锚车的上下位置',
    'ob_x': '最近那辆抛锚车的左右位置',
    'dash': '中线虚线滚到了哪（路在往下走）',
}
STATE_KEYS = ['car_x', 'road_near', 'road_mid', 'road_far', 'bend', 'car_off', 'n_obs', 'ob_y', 'ob_x', 'dash']


def gz_write(path, data):
    with open(path, 'wb') as f:
        f.write(data)
    with gzip.open(path + '.gz', 'wb', compresslevel=9) as f:
        f.write(data)


def thin(curve, n=400):
    if len(curve) <= n:
        return curve
    idx = np.linspace(0, len(curve) - 1, n).round().astype(int)
    return [curve[i] for i in idx]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def export_all(work, out_dir, dev, VAE, MDNRNN, to_rgb):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    vae = VAE(); vae.load_state_dict(torch.load(os.path.join(work, 'vae.pt'), map_location='cpu'))
    rnn = MDNRNN(); rnn.load_state_dict(torch.load(os.path.join(work, 'rnn.pt'), map_location='cpu'))
    ctrl_path = os.path.join(work, 'ctrl.npy')
    ctrl = np.load(ctrl_path) if os.path.exists(ctrl_path) else None
    # 权重存成 float16，再读回来放进模型：之后所有参考输出都用网页拿到的同一份权重
    with torch.no_grad():
        for m in (vae, rnn):
            for p in m.parameters():
                p.copy_(p.half().float())
    if ctrl is not None:
        ctrl = ctrl.astype(np.float16).astype(np.float32)
    vae.eval(); rnn.eval()

    tensors, chunks, off = {}, [], 0
    items = [(k, v.numpy()) for k, v in vae.state_dict().items()] + [(k, v.numpy()) for k, v in rnn.state_dict().items()]
    if ctrl is not None:
        items.append(('ctrl.weight', ctrl))
    for k, v in items:
        b = v.astype(np.float16).tobytes()
        tensors[k] = {'shape': list(v.shape), 'offset': off, 'dtype': 'float16'}
        chunks.append(b)
        off += len(b)
        if off % 4:
            pad = 4 - off % 4
            chunks.append(b'\0' * pad)
            off += pad
    gz_write(os.path.join(out_dir, 'model.bin'), b''.join(chunks))
    n_params = int(sum(np.prod(t['shape']) for t in tensors.values()))

    D = np.load(os.path.join(work, 'data.npz'))
    frames, starts, acts, states, kinds = D['frames'], D['starts'], D['acts'], D['states'], D['kinds']
    n_ep = len(starts) - 1
    n_test = max(20, n_ep // 50)
    pal = torch.tensor(PALETTE, dtype=torch.float32) / 255

    # ---------------- 潜变量每一维：统计 + 和游戏状态的相关（全部帧，GPU 上编码）
    vg = VAE().to(dev); vg.load_state_dict(vae.state_dict()); vg.eval()
    palg = pal.to(dev)
    mus, lvs = [], []
    with torch.no_grad():
        for i in range(0, len(frames), 4096):
            mu, lv = vg.encode(to_rgb(torch.from_numpy(frames[i:i + 4096]).to(dev), palg))
            mus.append(mu.cpu()); lvs.append(lv.cpu())
    mu_all = torch.cat(mus).numpy().astype(np.float64); lv_all = torch.cat(lvs).numpy().astype(np.float64)
    kl = (-0.5 * (1 + lv_all - mu_all ** 2 - np.exp(lv_all))).mean(0)
    st = states.astype(np.float64)
    has_ob = st[:, STATE_KEYS.index('ob_y')] >= 0
    feats = {}
    for j, k in enumerate(STATE_KEYS):
        if k == 'dash':
            ph = st[:, j] * (math.pi / 2)
            feats['dash'] = (np.stack([np.sin(ph), np.cos(ph)], 1), np.ones(len(st), bool))
        else:
            feats[k] = (st[:, j:j + 1], has_ob if k in ('ob_y', 'ob_x') else np.ones(len(st), bool))
    dims = []
    for d in range(Z):
        best = (0.0, None)
        for k, (v, m) in feats.items():
            c = max(abs(corr(mu_all[m, d], v[m, q])) for q in range(v.shape[1]))
            sgn = np.sign(corr(mu_all[m, d], v[m, 0]))
            if c > abs(best[0]):
                best = (c * (sgn if k != 'dash' else 1), k)
        dims.append({'kl': round(float(kl[d]), 4), 'mean': round(float(mu_all[:, d].mean()), 4), 'std': round(float(mu_all[:, d].std()), 4),
                     'lo': round(float(np.percentile(mu_all[:, d], 1)), 3), 'hi': round(float(np.percentile(mu_all[:, d], 99)), 3),
                     'feat': best[1], 'corr': round(best[0], 3)})
    # 从 32 个数线性读出各个游戏状态（最小二乘，留出最后 2% 的局检验）
    split = int(starts[n_ep - n_test])
    X = np.concatenate([mu_all, np.ones((len(mu_all), 1))], 1)
    probes = {}
    for k, (v, m) in feats.items():
        tr = m.copy(); tr[split:] = False
        te = m.copy(); te[:split] = False
        if te.sum() < 50:
            continue
        coef, *_ = np.linalg.lstsq(X[tr], v[tr], rcond=None)
        pred = X[te] @ coef
        ss = ((v[te] - pred) ** 2).sum(); tot = ((v[te] - v[te].mean(0)) ** 2).sum()
        probes[k] = round(float(1 - ss / tot), 3)

    # ---------------- 测试集样例：带抛锚车的帧
    samples = []
    rng = np.random.default_rng(3)
    test_ids = np.arange(split, len(frames))
    cand = test_ids[(states[test_ids, STATE_KEYS.index('n_obs')] >= 1) & (states[test_ids, STATE_KEYS.index('ob_y')] > 8)]
    pick = rng.choice(cand, size=min(8, len(cand)), replace=False)
    with torch.no_grad():
        for i in sorted(pick):
            x = to_rgb(torch.from_numpy(frames[i:i + 1]), pal)
            mu, _ = vae.encode(x)
            y = vae.decode(mu)
            samples.append({'frame': base64.b64encode(frames[i].tobytes()).decode(), 'mse': round(float(((y - x) ** 2).mean()), 6)})

    # ---------------- 梦与现实：从同一帧出发，同一串动作，平均差异随步数怎么变（测试局里的专家局）
    H_STEPS = 60
    curves = {}
    vrg = vg
    rg = MDNRNN().to(dev); rg.load_state_dict(rnn.state_dict()); rg.eval()
    test_eps = [e for e in range(n_ep - n_test, n_ep) if starts[e + 1] - starts[e] - 1 >= H_STEPS and kinds[e] == 0]
    gen = torch.Generator(device=dev); gen.manual_seed(5)
    with torch.no_grad():
        for mode, tau in (('open', 1.0), ('closed', 1.0), ('closed', 0.5)):
            errs = []
            for e in test_eps:
                s = starts[e]
                real = to_rgb(torch.from_numpy(frames[s:s + H_STEPS + 1]).to(dev), palg)
                zr, _ = vrg.encode(real)
                a = torch.from_numpy(acts[s:s + H_STEPS].astype(np.int64)).to(dev)
                h = torch.zeros(1, 1, HID, device=dev); c = torch.zeros_like(h)
                z = zr[0:1]
                err = []
                for t in range(H_STEPS):
                    zin = zr[t:t + 1] if mode == 'open' else z
                    (logpi, m, ls, dl), (h, c) = rg(zin[:, None], a[t:t + 1][None], (h, c))
                    logpi, m, ls = logpi[0, 0], m[0, 0], ls[0, 0]
                    k = torch.distributions.Categorical(logits=logpi / tau).sample()
                    z = (m.gather(-1, k[:, None])[:, 0] + ls.gather(-1, k[:, None])[:, 0].exp() * math.sqrt(tau) * torch.randn(Z, device=dev, generator=gen))[None]
                    y = vrg.decode(z)
                    err.append(((y[0] - real[t + 1]) ** 2).mean().item())
                errs.append(err)
            curves[f'{mode}{"" if mode == "open" else "_" + str(tau)}'] = [round(float(v), 5) for v in np.mean(errs, 0)]
    # 只看 V：重建误差（梦的误差里有多少是 V 自己画不准）
    with torch.no_grad():
        rec = []
        for e in test_eps:
            s = starts[e]
            real = to_rgb(torch.from_numpy(frames[s:s + H_STEPS + 1]).to(dev), palg)
            mu, _ = vrg.encode(real)
            rec.append(((vrg.decode(mu) - real) ** 2).mean().item())
    curves['recon'] = round(float(np.mean(rec)), 5)

    # ---------------- 训练记录
    def rd(name):
        p = os.path.join(work, name)
        return json.load(open(p)) if os.path.exists(p) else None
    vj, rj, cj = rd('vae.json'), rd('rnn.json'), rd('ctrl.json')
    meta = {
        'version': 1,
        'paper': 'Ha & Schmidhuber 2018, World Models (arXiv:1803.10122)',
        'arch': {
            'V': 'Conv2d 3→16→32→64→128（4×4，步长 2）· μ / logσ² 512→32 · 解码 32→512 → 反卷积 64（5×5）→32（5×5）→16（6×6）→3（6×6），步长 2',
            'M': 'LSTM（输入 32 + 3，隐状态 256）· MDN 头 256 → 32×5×3 + 1',
            'C': '线性 [z, h, 1]（289）→ 3 个动作，argmax' if ctrl is not None else None,
        },
        'params': {'V': int(sum(p.numel() for p in vae.parameters())), 'M': int(sum(p.numel() for p in rnn.parameters())),
                   'C': int(ctrl.size) if ctrl is not None else 0, 'total': n_params},
        'tensors': tensors,
        'palette': PALETTE.tolist(),
        'train': {
            'collect': rd('collect.json'),
            'V': vj['info'] if vj else None, 'Vcurve': thin(vj['curve']) if vj else None,
            'M': rj['info'] if rj else None, 'Mcurve': thin(rj['curve']) if rj else None,
            'C': cj['info'] if cj else None, 'Ccurve': cj['curve'] if cj else None,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'torch': torch.__version__, 'python': platform.python_version(),
        },
        'dims': dims,
        'probes': probes,
        'feats': FEATS,
        'samples': samples,
        'drift': {'steps': H_STEPS, 'episodes': len(test_eps), **curves},
    }
    js = json.dumps(meta, ensure_ascii=False, separators=(',', ':')).encode()
    gz_write(os.path.join(out_dir, 'model.json'), js)

    # ---------------- 参考输出（CPU float32 + float64）
    ref_dir = os.path.join(HERE, 'ref')
    os.makedirs(ref_dir, exist_ok=True)
    e0 = n_ep - 3
    s = int(starts[e0])
    T_SEQ = min(30, int(starts[e0 + 1] - starts[e0] - 1))
    ids = [s, s + T_SEQ // 2, s + T_SEQ]
    arrays, refmeta = [], {'frames': [], 'acts': acts[s:s + T_SEQ].astype(int).tolist(), 'T': T_SEQ, 'arrays': {}}
    off2 = 0

    def put(name, t):
        nonlocal off2
        a = np.asarray(t, dtype=np.float32).ravel()
        refmeta['arrays'][name] = {'offset': off2, 'n': int(a.size)}
        arrays.append(a.tobytes()); off2 += a.nbytes

    def run(model_v, model_m, dtype, tag):
        with torch.no_grad():
            x = to_rgb(torch.from_numpy(frames[ids]), pal).to(dtype)
            keep = {}
            mu, lv = model_v.encode(x, keep)
            put(f'{tag}mu', mu); put(f'{tag}lv', lv)
            put(f'{tag}e1_0', keep['e1'][0]); put(f'{tag}e2_0', keep['e2'][0]); put(f'{tag}e3_0', keep['e3'][0]); put(f'{tag}e4_0', keep['e4'][0])
            keep = {}
            y = model_v.decode(mu[:2], keep)
            put(f'{tag}dfc_0', keep['dfc'][0]); put(f'{tag}d1_0', keep['d1'][0]); put(f'{tag}d2_0', keep['d2'][0]); put(f'{tag}d3_0', keep['d3'][0])
            put(f'{tag}y', y)
            xs = to_rgb(torch.from_numpy(frames[s:s + T_SEQ]), pal).to(dtype)
            zs, _ = model_v.encode(xs)
            a = torch.from_numpy(acts[s:s + T_SEQ].astype(np.int64))
            (logpi, m, ls, dl), (h, c) = model_m(zs[None], a[None])
            out, _ = model_m.lstm(torch.cat([zs, F.one_hot(a, NA).to(dtype)], -1)[None])
            put(f'{tag}seq_z', zs); put(f'{tag}seq_h', out[0]); put(f'{tag}seq_c', c[0, 0])
            raw = model_m.head(out[0])
            put(f'{tag}seq_head', raw); put(f'{tag}seq_logpi', logpi[0])
            return y

    refmeta['frames'] = [base64.b64encode(frames[i].tobytes()).decode() for i in ids]
    refmeta['seq'] = base64.b64encode(frames[s:s + T_SEQ].tobytes()).decode()
    run(vae, rnn, torch.float32, '')
    vae64 = VAE().double(); vae64.load_state_dict(vae.state_dict()); vae64.eval()
    rnn64 = MDNRNN().double(); rnn64.load_state_dict(rnn.state_dict()); rnn64.eval()
    run(vae64, rnn64, torch.float64, 'f64_')
    with gzip.open(os.path.join(ref_dir, 'ref.bin.gz'), 'wb', compresslevel=9) as f:
        f.write(b''.join(arrays))
    json.dump(refmeta, open(os.path.join(ref_dir, 'ref.json'), 'w'))

    sizes = {f: os.path.getsize(os.path.join(out_dir, f)) for f in sorted(os.listdir(out_dir))}
    print('导出：', {k: f'{v / 1024:.0f} KB' for k, v in sizes.items()}, f'参数 {n_params}', f'{time.time() - t0:.0f} 秒')
    print('线性读出 R²：', probes)
    print('梦与现实（每像素均方误差）：', {k: (v[:1] + v[9::10] if isinstance(v, list) else v) for k, v in curves.items()})
    print('活跃维度（KL > 0.05）：', sum(d['kl'] > 0.05 for d in dims))
