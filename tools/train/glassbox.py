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
    R = {k: [] for k in ['offs', 'loss', 'evalLoss', 'lr', 'gnorm', 'clip', 'W', 'G', 'Gc', 'M', 'Vv', 'W1', 'maxAbsW', 'nnzG']}
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
        logits, A = m.forward(xb[:, :-1], keep=record)
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
        import glassbox_export
        glassbox_export.export(work, pathlib.Path(a.out))


if __name__ == '__main__':
    main()
