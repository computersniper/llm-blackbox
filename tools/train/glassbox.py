"""玻璃小模型：小到能把每个参数都画出来的 Transformer，真实训练 + 逐数记录，给训练页“每个参数都看得见”一章回放。

用法：
    python tools/train/glassbox.py                 # 训练（CPU，约 1 分钟）+ 导出到 public/train/data/glass*、glass/ + 核对
    python tools/train/glassbox.py --stage export  # 只用上次的记录（--work 里的 raw.npz）重新导出 + 核对

模型（和站里的 Qwen3 同一家族，只是小到每个数都画得下）：
    词嵌入 E [20 × 16]（输出层共享）→ 1 层：RMSNorm → 双头注意力（每头 8 维，RoPE θ = 10⁴，因果遮罩）→ 残差
    → RMSNorm → SwiGLU（16 → 32 → 16）→ 残差 → 最后的 RMSNorm → 乘 Eᵀ 得到 20 个字的 logits。没有偏置。
    参数：E 320 + W_q/W_k/W_v/W_o 4 × 256 + W_gate/W_up 2 × 512 + W_down 512 + 3 个 RMSNorm γ 48 = 2,928 个。
语料：李白《静夜思》20 个字（17 个不同的字 + ，。）加一个分隔符 ⏎，首尾相接成一个 25 字的循环。
    每一步取 8 段、每段 9 个字（输入 8 个，目标是错开一位的 8 个）：第 0 段固定是“举头望明月，低头思”（方便逐步对比），
    其余 7 段从循环里随机取起点。只看前一个字，有 4 种字后面接什么是五五开（月→光/，、头→望/思、，→疑/低、。→举/⏎），
    要往回多看一两个字才分得清——这正是注意力要学的。
训练：初始化 N(0, 0.02²)（γ 为 1），AdamW（β = 0.9 / 0.95，ε = 10⁻⁸，权重衰减 0.1，γ 不衰减），
    峰值学习率 10⁻²，前 10 步线性预热，之后余弦退火到 10⁻³，梯度裁剪 1.0，200 步，种子 0，CPU float32。
对照：同一模型、同样的批次，① 全零初始化（γ 仍为 1）；② 初始化放大 50 倍（N(0, 1)）。

记录（--work 里的 raw.npz 是 float32 原始记录；网页数据见 export()）：
    每一步：批次、损失、全部 25 段上的平均损失、学习率、裁剪前的梯度范数、裁剪系数；
            全部参数、全部参数的梯度（裁剪前，就是 ∂L/∂w）、AdamW 的 m / v（更新后）、更新后的参数；
            第 0 段（固定样例）前向每一层的激活、注意力、logits / 概率 / −ln p，反向时每一层激活的梯度 ∂L/∂h。
导出后会用导出的数值重算 AdamW 的一步、重算固定样例的前向，和记录逐元素核对（结果写进 glass.json 的 check）。
"""
import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

import split_data
from split_data import Pack, container, dumps, gz

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'public' / 'train' / 'data'

POEM = '床前明月光，疑是地上霜。举头望明月，低头思故乡。'   # 李白《静夜思》
EOS = '⏎'
FIXED = '举头望明月，低头思'

CFG = dict(d=16, heads=2, ffn=32, ctx=8, batch=8, steps=200, lr=1e-2, warmup=10, floor=0.1,
           beta1=0.9, beta2=0.95, eps=1e-8, wd=0.1, clip=1.0, std=0.02, theta=1e4, normEps=1e-6, seed=0)

# 名字、形状（[输入维, 输出维]，y = x @ W）、是否权重衰减
def param_specs(V, d, Fh):
    return [('E', (V, d), True), ('g1', (d,), False), ('Wq', (d, d), True), ('Wk', (d, d), True), ('Wv', (d, d), True),
            ('Wo', (d, d), True), ('g2', (d,), False), ('Wg', (d, Fh), True), ('Wu', (d, Fh), True), ('Wd', (Fh, d), True),
            ('gf', (d,), False)]


def vocab():
    chars = [EOS]
    for c in POEM:
        if c not in chars:
            chars.append(c)
    return chars


def lr_at(t, c):
    S, W, peak = c['steps'], c['warmup'], c['lr']
    if t < W:
        return peak * (t + 1) / W
    lo = peak * c['floor']
    return lo + 0.5 * (peak - lo) * (1 + math.cos(math.pi * (t - W) / max(1, S - W)))


def rope_tab(T, hd, theta):
    half = hd // 2
    inv = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32) / half))
    ang = torch.arange(T, dtype=torch.float32)[:, None] * inv[None]
    return torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)   # [T, hd]


def rope(x, cos, sin):
    """x [B, T, H, hd]；rotate-half（和 Qwen3 / LLaMA 一样，第 i 维和第 i + hd/2 维成对旋转）"""
    half = x.shape[-1] // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * cos[None, :, None] + rot * sin[None, :, None]


class Glass:
    def __init__(self, c, V, init='normal', gen=None):
        d, Fh = c['d'], c['ffn']
        self.c, self.V, self.d, self.H, self.Fh = c, V, d, c['heads'], Fh
        self.hd = d // c['heads']
        self.specs = param_specs(V, d, Fh)
        self.P = {}
        for name, shape, _ in self.specs:
            if name.startswith('g'):
                t = torch.ones(shape)
            elif init == 'zero':
                t = torch.zeros(shape)
            else:
                std = c['std'] if init == 'normal' else 1.0
                t = torch.randn(shape, generator=gen) * std
            self.P[name] = t.requires_grad_(True)
        self.cos, self.sin = rope_tab(c['ctx'], self.hd, c['theta'])
        self.mask = torch.triu(torch.ones(c['ctx'], c['ctx'], dtype=torch.bool), 1)

    def params(self):
        return [self.P[n] for n, _, _ in self.specs]

    def flat(self, which='data'):
        return torch.cat([(getattr(self.P[n], which) if which == 'data' else self.P[n].grad).reshape(-1) for n, _, _ in self.specs]).detach().clone()

    def rms(self, x, g):
        inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.c['normEps'])
        return x * inv * g, inv.squeeze(-1)

    def forward(self, x, keep=False):
        """x [B, T] → logits [B, T, V]；keep=True 时返回各层的中间量（保留梯度），给第 0 段记录用"""
        P, B, T, H, hd, d = self.P, x.shape[0], x.shape[1], self.H, self.hd, self.d
        A = {}
        h0 = P['E'][x]
        n1, inv1 = self.rms(h0, P['g1'])
        q, k, v = n1 @ P['Wq'], n1 @ P['Wk'], n1 @ P['Wv']
        qr = rope(q.view(B, T, H, hd), self.cos, self.sin)
        kr = rope(k.view(B, T, H, hd), self.cos, self.sin)
        sc = torch.einsum('bihd,bjhd->bhij', qr, kr) / math.sqrt(hd)
        sc = sc.masked_fill(self.mask, float('-inf'))
        att = sc.softmax(-1)
        ao = torch.einsum('bhij,bjhd->bihd', att, v.view(B, T, H, hd)).reshape(B, T, d)
        o = ao @ P['Wo']
        h1 = h0 + o
        n2, inv2 = self.rms(h1, P['g2'])
        gate, up = n2 @ P['Wg'], n2 @ P['Wu']
        act = F.silu(gate) * up
        f = act @ P['Wd']
        h2 = h1 + f
        nf, invf = self.rms(h2, P['gf'])
        logits = nf @ P['E'].T
        if keep:
            for name, t in [('h0', h0), ('n1', n1), ('q', q), ('k', k), ('v', v), ('att', att), ('ao', ao), ('o', o), ('h1', h1),
                            ('n2', n2), ('gate', gate), ('up', up), ('act', act), ('f', f), ('h2', h2), ('nf', nf), ('logits', logits)]:
                t.retain_grad()
                A[name] = t
            A['sc'] = sc
            A['inv1'], A['inv2'], A['invf'] = inv1, inv2, invf
        return logits, A


# 第 0 段要记的激活（前向）和激活的梯度（反向）；att / sc 是 [H, T, T]
ACT_F = ['h0', 'inv1', 'n1', 'q', 'k', 'v', 'sc', 'att', 'ao', 'o', 'h1', 'inv2', 'n2', 'gate', 'up', 'act', 'f', 'h2', 'invf', 'nf', 'logits', 'probs', 'nll']
ACT_B = ['h0', 'n1', 'q', 'k', 'v', 'att', 'ao', 'h1', 'n2', 'gate', 'up', 'act', 'h2', 'nf', 'logits']


def train(c, init, work=None, record=True, log=print):
    chars = vocab()
    V = len(chars)
    idx = {ch: i for i, ch in enumerate(chars)}
    stream = [idx[ch] for ch in POEM] + [0]
    NS, T, B, S = len(stream), c['ctx'], c['batch'], c['steps']
    fix = POEM.index(FIXED[0])
    assert ''.join(chars[stream[(fix + i) % NS]] for i in range(T + 1)) == FIXED
    win = lambda o: [stream[(o + i) % NS] for i in range(T + 1)]
    torch.manual_seed(c['seed'])
    gen = torch.Generator().manual_seed(c['seed'])
    m = Glass(c, V, init, gen)
    names = [n for n, _, _ in m.specs]
    decay = [m.P[n] for n, _, dc in m.specs if dc]
    nodec = [m.P[n] for n, _, dc in m.specs if not dc]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': c['wd']}, {'params': nodec, 'weight_decay': 0.0}],
                            lr=c['lr'], betas=(c['beta1'], c['beta2']), eps=c['eps'], foreach=False)
    bgen = torch.Generator().manual_seed(c['seed'] + 1)
    allx = torch.tensor([win(o) for o in range(NS)])
    R = {k: [] for k in ['offs', 'loss', 'evalLoss', 'lr', 'gnorm', 'clip', 'W', 'G', 'Gc', 'M', 'Vv', 'W1', 'maxAbsW', 'nnzG', 'p0', 'att0']}
    if record:
        for k in ACT_F:
            R['f_' + k] = []
        for k in ACT_B:
            R['b_' + k] = []
    R['W0'] = m.flat().numpy()
    t0 = time.time()
    for t in range(S):
        offs = [fix] + torch.randint(0, NS, (B - 1,), generator=bgen).tolist()
        xb = torch.tensor([win(o) for o in offs])
        lr = lr_at(t, c)
        for gr in opt.param_groups:
            gr['lr'] = lr
        with torch.no_grad():
            ev = F.cross_entropy(m.forward(allx[:, :-1])[0].reshape(-1, V), allx[:, 1:].reshape(-1)).item()
        logits, A = m.forward(xb[:, :-1], keep=True)
        loss = F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        Wt = m.flat()
        Gt = m.flat('grad')
        gn_t = torch.nn.utils.clip_grad_norm_(m.params(), c['clip'], foreach=False)
        gn = gn_t.item()
        clip = float((c['clip'] / (gn_t + 1e-6)).clamp(max=1.0))   # 和 clip_grad_norm_ 里同一个 float32 算式
        Gc = m.flat('grad')
        opt.step()
        st = [opt.state[p] for p in m.params()]
        R['offs'].append(offs)
        R['loss'].append(loss.item())
        R['evalLoss'].append(ev)
        R['lr'].append(lr)
        R['gnorm'].append(gn)
        R['clip'].append(clip)
        R['maxAbsW'].append(max(float(m.P[n].detach().abs().max()) for n in names if not n.startswith('g')))
        R['nnzG'].append(int((Gt != 0).sum()))
        R['p0'].append(logits[0].softmax(-1).detach().numpy().copy())   # 对照组也记第 0 段的预测和注意力（很小）
        R['att0'].append(A['att'][0].detach().numpy().copy())
        if record:
            R['W'].append(Wt.numpy())
            R['G'].append(Gt.numpy())
            R['Gc'].append(Gc.numpy())
            R['M'].append(torch.cat([s['exp_avg'].reshape(-1) for s in st]).numpy().copy())
            R['Vv'].append(torch.cat([s['exp_avg_sq'].reshape(-1) for s in st]).numpy().copy())
            R['W1'].append(m.flat().numpy())
            p = logits[0].softmax(-1)
            y = xb[0, 1:]
            fw = {k: A[k][0] for k in A}
            fw['probs'] = p
            fw['nll'] = -p[torch.arange(T), y].log()
            for k in ACT_F:
                R['f_' + k].append(fw[k].detach().numpy().copy())
            for k in ACT_B:
                R['b_' + k].append(A[k].grad[0].detach().numpy().copy())
        if t % 20 == 0 or t == S - 1:
            log(f'  [{init}] step {t:3d}  loss {loss.item():.4f}  eval {ev:.4f}  lr {lr:.2e}  |g| {gn:.3f}')
    with torch.no_grad():
        R['finalEval'] = F.cross_entropy(m.forward(allx[:, :-1])[0].reshape(-1, V), allx[:, 1:].reshape(-1)).item()
    R['seconds'] = time.time() - t0
    out = {}
    for k, v in R.items():
        if isinstance(v, list) and v and isinstance(v[0], np.ndarray):
            out[k] = np.stack(v).astype(np.float32)
        elif isinstance(v, list):
            out[k] = np.array(v)
        else:
            out[k] = np.array(v)
    return out, dict(chars=chars, stream=stream, fix=fix, names=names, specs=[(n, list(s), dc) for n, s, dc in m.specs])


# ---------------------------------------------------------------- 导出

def frames_of(S):
    """逐数记录的步：前 50 步每步都记，之后到第 150 步每 5 步一帧，再之后每 10 步一帧，加上最后一步"""
    return sorted(set(list(range(min(50, S))) + list(range(50, min(150, S), 5)) + list(range(150, S, 10)) + [S - 1]))


W_GROUP = 9            # 权重分块：每块 9 帧（训练全程播放时按顺序取）
TRI_NAMES = ('att',)   # 注意力只存下三角（因果遮罩，上三角本来就是 0）
F16 = np.float16


def tensor_slices(specs):
    off, sl = 0, {}
    for n, s, _ in specs:
        k = int(np.prod(s))
        sl[n] = (off, off + k)
        off += k
    return sl, off


def scaled16(x, sl):
    """每个张量各自除以自己的最大绝对值再存 float16（相对精度约 10⁻³，不会因为太小下溢）；返回 (f16, 每个张量的倍数)"""
    out = np.zeros(x.shape, F16)
    sc = np.ones(len(sl), np.float32)
    for j, (a, b) in enumerate(sl.values()):
        m = float(np.abs(x[a:b]).max())
        if m > 0:
            sc[j] = m
            out[a:b] = (x[a:b] / m).astype(F16)
    return out, sc


def unscale(x16, sc, sl):
    out = x16.astype(np.float64)
    for j, (a, b) in enumerate(sl.values()):
        out[a:b] *= sc[j]
    return out


def delta16(a16):
    """沿第 0 维（帧）存 float16 比特模式的差（mod 2¹⁶）：权重一步只变一点点，差值很小，gzip 压得更好。读的时候累加回去"""
    u = a16.view(np.uint16).astype(np.int64)
    return (np.diff(u, axis=0, prepend=0) & 0xFFFF).astype(np.uint16)


def adam_ref(w, g, m, v, lr, t, wd, c):
    """torch.optim.AdamW（单张量实现）同样的 float32 运算顺序；t 从 1 开始"""
    w, g, m, v = (torch.tensor(np.array(a, np.float32)) for a in (w, g, m, v))
    if wd:
        w.mul_(1 - lr * wd)
    m.lerp_(g, 1 - c['beta1'])
    v.mul_(c['beta2']).addcmul_(g, g, value=1 - c['beta2'])
    bc1, bc2 = 1 - c['beta1'] ** t, 1 - c['beta2'] ** t
    denom = (v.sqrt() / (bc2 ** 0.5)).add_(c['eps'])
    w.addcdiv_(m, denom, value=-(lr / bc1))
    return w.numpy(), m.numpy(), v.numpy()


def rms_np(x, g, eps):
    inv = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return x * inv * g


def forward_np(Wf, sl, specs, x, c):
    """用（导出的）参数在 numpy float64 里把第 0 段前向算一遍，和记录的激活核对"""
    P = {n: Wf[a:b].astype(np.float64).reshape(s) for (n, s, _), (a, b) in zip(specs, sl.values())}
    d, H, T = c['d'], c['heads'], len(x)
    hd = d // H
    half = hd // 2
    inv = 1.0 / (c['theta'] ** (np.arange(half) / half))
    ang = np.arange(T)[:, None] * inv[None]
    cos, sin = np.concatenate([np.cos(ang)] * 2, -1), np.concatenate([np.sin(ang)] * 2, -1)
    rope = lambda z: z * cos[:, None] + np.concatenate([-z[..., half:], z[..., :half]], -1) * sin[:, None]
    h0 = P['E'][x]
    n1 = rms_np(h0, P['g1'], c['normEps'])
    q, k, v = ((n1 @ P[w]).reshape(T, H, hd) for w in ('Wq', 'Wk', 'Wv'))
    sc = np.einsum('ihd,jhd->hij', rope(q), rope(k)) / math.sqrt(hd)
    sc = np.where(np.triu(np.ones((T, T), bool), 1)[None], -np.inf, sc)
    att = np.exp(sc - sc.max(-1, keepdims=True))
    att /= att.sum(-1, keepdims=True)
    ao = np.einsum('hij,jhd->ihd', att, v).reshape(T, d)
    h1 = h0 + ao @ P['Wo']
    n2 = rms_np(h1, P['g2'], c['normEps'])
    gate, up = n2 @ P['Wg'], n2 @ P['Wu']
    h2 = h1 + (gate / (1 + np.exp(-gate)) * up) @ P['Wd']
    logits = rms_np(h2, P['gf'], c['normEps']) @ P['E'].T
    p = np.exp(logits - logits.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True), att


def export(work, out):
    info = json.loads((work / 'info.json').read_text())
    c = info['cfg']
    R = dict(np.load(work / 'raw_normal.npz'))
    Z = dict(np.load(work / 'raw_zero.npz'))
    BG = dict(np.load(work / 'raw_big.npz'))
    chars, stream, specs = info['chars'], info['stream'], info['specs']
    sl, P = tensor_slices(specs)
    S, T, B, V, H = c['steps'], c['ctx'], c['batch'], len(chars), c['heads']
    FR = frames_of(S)
    NF = len(FR)
    ti, tj = np.tril_indices(T)
    gdir = out / 'glass'
    if gdir.exists():
        for f in gdir.iterdir():
            f.unlink()
    files = {}
    t0 = time.time()
    is_norm = lambda n: n.startswith('g')

    # 选中的几个张量：每一步都存 float32 原值（三个 RMSNorm γ，再加嵌入表里“月”那一行），能逐位重算 AdamW
    a, _ = sl['E']
    yue = chars.index('月')
    exact_idx = list(range(*sl['g1'])) + list(range(*sl['g2'])) + list(range(*sl['gf'])) + list(range(a + yue * c['d'], a + (yue + 1) * c['d']))
    ex = np.array(exact_idx)
    Mprev = np.concatenate([np.zeros((1, P), np.float32), R['M'][:-1]])       # 第 t 步更新前的 m（第 0 步是 0）
    Vprev = np.concatenate([np.zeros((1, P), np.float32), R['Vv'][:-1]])
    assert np.array_equal(R['W1'][:-1], R['W'][1:]), '更新后的参数应该就是下一步的参数'

    # ---- 首屏：配置、每一步的标量和批次、初始化的全部参数、两个对照组
    core = Pack()
    for k in ['loss', 'evalLoss', 'lr', 'gnorm', 'clip']:
        core.add(k, R[k].astype(np.float32))
    core.add('offs', R['offs'].astype(np.uint8))
    # 每一步、每个参数张量的梯度长度（裁剪前）：训练全程里看反向传播流到哪一层、多大
    core.add('tgn', np.stack([np.linalg.norm(R['G'][:, a:b], axis=1) for a, b in sl.values()], 1).astype(np.float32))
    core.add('w0', R['W0'].astype(np.float32))
    core.add('zLoss', Z['loss'].astype(np.float32))
    core.add('zGnorm', Z['gnorm'].astype(np.float32))
    core.add('zMaxW', Z['maxAbsW'].astype(np.float32))
    core.add('bLoss', BG['loss'].astype(np.float32))
    core.add('bEval', BG['evalLoss'].astype(np.float32))
    core.add('bGnorm', BG['gnorm'].astype(np.float32))
    core.add('bW0', BG['W0'].astype(F16))
    core.add('bP0', np.stack([BG['p0'][0], BG['p0'][-1]]).astype(F16))
    core.add('zP0', np.stack([Z['p0'][0], Z['p0'][-1]]).astype(F16))
    core.add('zAtt0', Z['att0'][0][:, ti, tj].astype(F16))

    # ---- 权重分块：W（float16，按帧做差）、第 0 段的预测概率、注意力、−ln p
    groups = [FR[i:i + W_GROUP] for i in range(0, NF, W_GROUP)]
    wspans = []
    for gi, grp in enumerate(groups):
        idx = [FR.index(t) for t in grp]
        p = Pack()
        p.add('w', delta16(R['W'][grp].astype(F16)))
        p.add('p', R['f_probs'][grp].astype(F16))
        p.add('att', R['f_att'][grp][:, :, ti, tj].astype(F16))
        p.add('nll', R['f_nll'][grp].astype(F16))
        files[f'glass/w{gi}.bin'] = container(p, {'frames': grp, 'delta': ['w']})
        wspans.append([idx[0], idx[-1] + 1])

    # ---- 每帧一块：梯度、更新前的 m 和 √v、这一步的 Δw（都按张量缩放成 float16），第 0 段每层的激活和激活的梯度
    act_f = [k for k in ACT_F if k not in ('sc', 'probs', 'att', 'nll')]   # 概率、注意力、−ln p 已经在权重块里
    for fi, t in enumerate(FR):
        p = Pack()
        for key, arr in [('g', R['G'][t]), ('m', Mprev[t]), ('v', np.sqrt(Vprev[t])), ('dw', R['W1'][t] - R['W'][t])]:
            x16, sc = scaled16(arr.astype(np.float64), sl)
            p.add(key, x16)
            p.add(key + 'S', sc)
        for k in act_f:
            x = R['f_' + k][t]
            p.add('f.' + k, (x[:, ti, tj] if k in TRI_NAMES else x).astype(F16))
        for k in ACT_B:
            x = R['b_' + k][t]
            p.add('b.' + k, (x[:, ti, tj] if k in TRI_NAMES else x).astype(F16))
        files[f'glass/f{fi:02d}.bin'] = container(p, {'t': t})

    # ---- 选中参数每一步的 float32 原值
    p = Pack()
    p.add('idx', ex.astype(np.uint16))
    p.add('w', np.concatenate([R['W'][:, ex], R['W1'][-1:, ex]]).astype(np.float32))      # [S+1]：第 t 步更新前（最后一行是训练完）
    p.add('g', R['G'][:, ex].astype(np.float32))                                           # [S]：裁剪前的梯度
    p.add('m', np.concatenate([Mprev[:, ex], R['M'][-1:, ex]]).astype(np.float32))          # [S+1]：第 t 步更新前的 m
    p.add('v', np.concatenate([Vprev[:, ex], R['Vv'][-1:, ex]]).astype(np.float32))
    p.add('lr', R['lr'].astype(np.float64))
    p.add('clip', R['clip'].astype(np.float32))
    files['glass/exact.bin'] = container(p)

    meta = {
        'model': {'vocab': V, 'd': c['d'], 'heads': H, 'headDim': c['d'] // H, 'ffn': c['ffn'], 'ctx': T, 'theta': c['theta'],
                  'normEps': c['normEps'], 'params': P, 'tied': True, 'layers': 1},
        'train': {k: c[k] for k in ['steps', 'batch', 'lr', 'warmup', 'floor', 'beta1', 'beta2', 'eps', 'wd', 'clip', 'std', 'seed']}
        | {'seconds': round(float(R['seconds']), 2), 'finalEval': float(R['finalEval']), 'device': 'CPU · float32', 'torch': torch.__version__},
        'corpus': {'poem': POEM, 'title': '静夜思', 'author': '李白', 'chars': chars, 'stream': stream, 'fixed': info['fix'], 'fixedText': FIXED},
        'params': [{'name': n, 'shape': s, 'off': sl[n][0], 'n': sl[n][1] - sl[n][0], 'decay': dc} for n, s, dc in specs],
        'frames': FR,
        'chunks': {'dir': 'glass', 'w': wspans, 'f': NF, 'acts': act_f, 'grads': ACT_B, 'tri': list(TRI_NAMES), 'group': W_GROUP,
                   'note': 'w*：权重（float16 按帧做差）+ 第 0 段的预测；f*：每帧的梯度 / m / √v / Δw（按张量缩放的 float16）+ 第 0 段的激活与梯度；exact：选中参数每一步的 float32'},
        'exact': {'idx': exact_idx, 'note': 'g1、g2、gf 三个 RMSNorm γ 和嵌入表里“月”那一行，每一步都存 float32'},
        'runs': {
            'zero': {'finalEval': float(Z['finalEval']), 'maxAbsW': float(Z['maxAbsW'].max()), 'maxGnorm': float(Z['gnorm'].max()), 'nnzG': int(Z['nnzG'].max())},
            'big': {'std': 1.0, 'finalEval': float(BG['finalEval']), 'firstLoss': float(BG['loss'][0])},
        },
        'bin': core.spec,
    }

    # ---- 核对
    chk = {}
    decays = {n: (0.0 if is_norm(n) else c['wd']) for n in sl}
    # A. 选中参数：只用导出文件里的 float32 原值，按 torch AdamW 同样的运算重算每一步，和导出的下一步逐元素比
    E, _ = split_data.read_container(files['glass/exact.bin'])
    owner = [next(n for n, (a, b) in sl.items() if a <= i < b) for i in exact_idx]
    wdv = np.array([decays[n] for n in owner])
    errs, same = {'w': 0.0, 'm': 0.0, 'v': 0.0}, 0
    for t in range(S):
        g = (torch.tensor(E['g'][t]) * torch.tensor(E['clip'][t])).numpy()
        for dec in sorted(set(wdv)):
            q = wdv == dec
            w1, m1, v1 = adam_ref(E['w'][t][q], g[q], E['m'][t][q], E['v'][t][q], float(E['lr'][t]), t + 1, dec, c)
            errs['w'] = max(errs['w'], float(np.abs(w1 - E['w'][t + 1][q]).max()))
            errs['m'] = max(errs['m'], float(np.abs(m1 - E['m'][t + 1][q]).max()))
            errs['v'] = max(errs['v'], float(np.abs(v1 - E['v'][t + 1][q]).max()))
            same += int((w1 == E['w'][t + 1][q]).sum())
    chk['adamExact'] = {'params': len(exact_idx), 'steps': S, 'maxAbsErr': errs, 'bitExact': same, 'total': len(exact_idx) * S}
    # B. 全部参数（原始 float32 记录）：同一个算式对 2,928 × 200 个更新
    errs, same = {'w': 0.0, 'm': 0.0, 'v': 0.0}, 0
    for t in range(S):
        g = (torch.tensor(R['G'][t]) * torch.tensor(R['clip'][t], dtype=torch.float32)).numpy()
        assert np.array_equal(g, R['Gc'][t]), '裁剪后的梯度应该等于 原梯度 × 裁剪系数'
        for n, (a, b) in sl.items():
            w1, m1, v1 = adam_ref(R['W'][t, a:b], g[a:b], Mprev[t, a:b], Vprev[t, a:b], float(R['lr'][t]), t + 1, decays[n], c)
            errs['w'] = max(errs['w'], float(np.abs(w1 - R['W1'][t, a:b]).max()))
            errs['m'] = max(errs['m'], float(np.abs(m1 - R['M'][t, a:b]).max()))
            errs['v'] = max(errs['v'], float(np.abs(v1 - R['Vv'][t, a:b]).max()))
            same += int((w1 == R['W1'][t, a:b]).sum())
    chk['adamAll'] = {'params': P, 'steps': S, 'maxAbsErr': errs, 'bitExact': same, 'total': P * S}
    # C. 网页上用的 float16：用导出的 w、g、m、√v 按公式算 Δw，和导出的 Δw 比
    rel, absmax = [], 0.0
    dec = np.concatenate([np.full(b - a, decays[n]) for n, (a, b) in sl.items()])
    for fi, t in enumerate(FR):
        blob, _ = split_data.read_container(files[f'glass/f{fi:02d}.bin'])
        w16 = R['W'][t].astype(F16).astype(np.float64)
        g = unscale(blob['g'], blob['gS'], sl) * float(R['clip'][t])
        m0 = unscale(blob['m'], blob['mS'], sl)
        v0 = unscale(blob['v'], blob['vS'], sl) ** 2
        dw = unscale(blob['dw'], blob['dwS'], sl)
        lr = float(R['lr'][t])
        m1 = c['beta1'] * m0 + (1 - c['beta1']) * g
        v1 = c['beta2'] * v0 + (1 - c['beta2']) * g * g
        mh, vh = m1 / (1 - c['beta1'] ** (t + 1)), v1 / (1 - c['beta2'] ** (t + 1))
        dwf = -lr * (mh / (np.sqrt(vh) + c['eps']) + dec * w16)
        big = np.abs(dw) > lr * 1e-2
        rel.append(np.abs(dwf - dw)[big] / np.abs(dw)[big])
        absmax = max(absmax, float((np.abs(dwf - dw) / lr).max()))
    rel = np.concatenate(rel)
    chk['adamFp16'] = {'frames': NF, 'medianRelErr': float(np.median(rel)), 'p99RelErr': float(np.quantile(rel, 0.99)), 'maxAbsErrOverLr': absmax,
                       'note': '|Δw| > lr/100 的更新的相对误差；maxAbsErrOverLr = 最大绝对误差 / 这一步的 lr'}
    # D. 用导出的 float16 参数重算第 0 段的前向，和记录（float32）比
    x0 = [stream[(info['fix'] + i) % len(stream)] for i in range(T + 1)]
    pe = ae = pe32 = 0.0
    for t in FR:
        pr, at = forward_np(R['W'][t].astype(F16), sl, specs, x0[:-1], c)
        pe = max(pe, float(np.abs(pr - R['f_probs'][t]).max()))
        ae = max(ae, float(np.abs(at - R['f_att'][t]).max()))
        pr32, _ = forward_np(R['W'][t], sl, specs, x0[:-1], c)
        pe32 = max(pe32, float(np.abs(pr32 - R['f_probs'][t]).max()))
    chk['forward'] = {'frames': NF, 'fp16MaxAbsErrProbs': pe, 'fp16MaxAbsErrAttn': ae, 'fp32MaxAbsErrProbs': pe32}
    # E. 链式法则：上游梯度 × 本层的局部导数（原始 float32 记录）
    shp = {n: s for n, s, _ in specs}
    Wm = lambda t, n: R['W'][t, sl[n][0]:sl[n][1]].reshape(shp[n])
    ce = {'dlogits': 0.0, 'ao': 0.0, 'act': 0.0, 'n2': 0.0, 'nf': 0.0}
    oh = np.eye(V)[np.array(x0[1:])]
    for t in FR:
        ce['dlogits'] = max(ce['dlogits'], float(np.abs((R['f_probs'][t] - oh) / (B * T) - R['b_logits'][t]).max()))
        ce['ao'] = max(ce['ao'], float(np.abs(R['b_h1'][t] @ Wm(t, 'Wo').T - R['b_ao'][t]).max()))
        ce['act'] = max(ce['act'], float(np.abs(R['b_h2'][t] @ Wm(t, 'Wd').T - R['b_act'][t]).max()))
        ce['n2'] = max(ce['n2'], float(np.abs(R['b_gate'][t] @ Wm(t, 'Wg').T + R['b_up'][t] @ Wm(t, 'Wu').T - R['b_n2'][t]).max()))
        ce['nf'] = max(ce['nf'], float(np.abs(R['b_logits'][t] @ Wm(t, 'E') - R['b_nf'][t]).max()))
    chk['chainRule'] = {'frames': NF, 'maxAbsErr': ce, 'note': 'dlogits = (p − onehot)/(B·T)；∂L/∂ao = ∂L/∂h1 · W_oᵀ；∂L/∂act = ∂L/∂h2 · W_downᵀ；∂L/∂n2 = ∂L/∂gate · W_gateᵀ + ∂L/∂up · W_upᵀ；∂L/∂nf = ∂L/∂logits · E'}
    meta['check'] = chk

    files['glass.json'] = dumps(meta)
    files['glass.bin'] = core.bytes()
    sizes = split_data.write(files, out, keep_raw={'glass.json', 'glass.bin'})
    verify_glass(out, files)
    tot = sum(z for _, z in sizes.values())
    first = sizes['glass.json'][1] + sizes['glass.bin'][1]
    part = lambda key: sum(z for k, (_, z) in sizes.items() if k.startswith(key)) / 1024
    print(f'导出 {len(files)} 个文件，gzip 后共 {tot / 1024:.0f} KB（首屏 {first / 1024:.1f} KB；权重块 {part("glass/w"):.0f} KB；'
          f'每帧块 {part("glass/f"):.0f} KB；exact {part("glass/exact"):.0f} KB），{time.time() - t0:.1f} s')
    print(json.dumps(chk, ensure_ascii=False, indent=1))


def verify_glass(out, files):
    """写出的 .gz 读回来和内存里的原文逐字节比；首屏的未压缩版本也比一遍"""
    import gzip as _gz
    for name, blob in files.items():
        assert _gz.decompress((out / (name + '.gz')).read_bytes()) == blob, f'{name}.gz 和原文不一致'
    for name in ('glass.json', 'glass.bin'):
        assert (out / name).read_bytes() == files[name], f'{name} 和原文不一致'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='/mnt/d/cjc/train-viz/glass')
    ap.add_argument('--stage', default='all', choices=['all', 'train', 'export'])
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args()
    work = pathlib.Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if a.stage in ('all', 'train'):
        runs = {}
        for init in ['normal', 'zero', 'big']:
            R, info = train(CFG, init, record=(init == 'normal'))
            runs[init] = R
            np.savez_compressed(work / f'raw_{init}.npz', **R)
            print(f'{init}: final eval {float(R["finalEval"]):.4f}, {float(R["seconds"]):.1f} s')
        (work / 'info.json').write_text(json.dumps({'cfg': CFG, **info}, ensure_ascii=False))
    if a.stage in ('all', 'export'):
        export(work, pathlib.Path(a.out))


if __name__ == '__main__':
    main()
