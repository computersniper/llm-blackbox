"""在「夜路」小游戏上训练一个世界模型（Ha & Schmidhuber 2018《World Models》的 V + M 结构）。

    python tools/world/train.py                   # 全部：采集 → V → M → C → 导出
    python tools/world/train.py --stage vae       # 只跑某一段（collect / vae / rnn / ctrl / export），前面的结果从 --work 读

V：卷积 VAE，64×64×3 → z（32 维）→ 64×64×3。结构照论文（4 层 4×4 步长 2 卷积编码、全连接到 μ / logσ²，
   全连接 → 1×1 → 4 层反卷积 5/5/6/6 解码，重建误差用逐像素平方和，KL 有 0.5×32 的容忍度），通道数减半
   （16/32/64/128，解码 512→64/32/16/3），这样浏览器里纯 JS 一帧编码 + 解码十几毫秒。
M：MDN-RNN，LSTM 256 个隐单元，输入 [z_t, 动作 one-hot]，输出下一帧 z 每一维 5 个高斯分量的混合（log π, μ, log σ），
   另加一个“这一步撞车了”的 logit（论文 Doom 实验的做法）。
C：线性控制器 a = argmax(W [z, h] + b)，用 CMA-ES 只在 M 的“梦”里训练（论文 Doom 实验的做法），再拿到真实游戏里检验。

大文件（采集的画面、检查点）放 --work（默认 D 盘）；导出到 public/world/data/。
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game import Game, heuristic, PALETTE, W, H  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
Z, NA, HID, KMIX = 32, 3, 256, 5
STATE_KEYS = ['car_x', 'road_near', 'road_mid', 'road_far', 'bend', 'car_off', 'n_obs', 'ob_y', 'ob_x', 'dash']


# ---------------------------------------------------------------- 采集

def run_episode(args):
    seed, kind, cap = args
    rnd = np.random.default_rng(seed ^ 0x5EED)
    g = Game(seed)
    frames, acts, states = [g.render()], [], [g.state()]
    eps = rnd.uniform(0.0, 0.12)
    a, hold = 0, 0
    while not g.done and g.t < cap:
        if kind == 'random':
            if hold <= 0:
                a, hold = int(rnd.choice(3, p=[0.4, 0.3, 0.3])), int(rnd.integers(1, 9))
            hold -= 1
            act = a
        else:
            if hold > 0:
                hold -= 1
                act = a
            elif rnd.random() < eps:
                a, hold = int(rnd.integers(0, 3)), int(rnd.integers(0, 3))
                act = a
            else:
                act = heuristic(g)
        acts.append(act)
        g.step(act)
        frames.append(g.render())
        states.append(g.state())
    return np.stack(frames), np.array(acts, np.uint8), g.done, g.why, np.array([[s[k] for k in STATE_KEYS] for s in states], np.int16)


def collect(work, n_expert, n_random, cap):
    t0 = time.time()
    jobs = [(1000 + i, 'expert', cap) for i in range(n_expert)] + [(500000 + i, 'random', cap) for i in range(n_random)]
    # 打乱顺序：训练时留出最后 2% 的局做测试，两种局都要有
    jobs = [jobs[i] for i in np.random.default_rng(0).permutation(len(jobs))]
    with Pool(min(16, os.cpu_count())) as pool:
        eps = pool.map(run_episode, jobs, chunksize=8)
    frames = np.concatenate([e[0] for e in eps])
    acts = np.concatenate([np.append(e[1], 255) for e in eps])          # 每局最后一帧没有动作，记 255
    starts = np.cumsum([0] + [len(e[0]) for e in eps])
    done = np.array([e[2] for e in eps])
    why = [e[3] for e in eps]
    states = np.concatenate([e[4] for e in eps])
    kinds = np.array([0 if j[1] == 'expert' else 1 for j in jobs], np.uint8)
    np.savez(os.path.join(work, 'data.npz'), frames=frames, acts=acts, starts=starts, done=done, states=states, kinds=kinds)
    lens = np.diff(starts) - 1
    info = {
        'episodes': len(eps), 'frames': int(len(frames)), 'steps': int(lens.sum()),
        'expert': n_expert, 'random': n_random, 'cap': cap,
        'meanLen': float(lens.mean()), 'crash': int(sum(w == 'crash' for w in why)), 'offroad': int(sum(w == 'offroad' for w in why)),
        'truncated': int((~done).sum()), 'seconds': round(time.time() - t0, 1),
    }
    json.dump(info, open(os.path.join(work, 'collect.json'), 'w'), ensure_ascii=False, indent=1)
    print('采集：', info)


# ---------------------------------------------------------------- 模型

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Conv2d(3, 16, 4, 2)
        self.e2 = nn.Conv2d(16, 32, 4, 2)
        self.e3 = nn.Conv2d(32, 64, 4, 2)
        self.e4 = nn.Conv2d(64, 128, 4, 2)
        self.mu = nn.Linear(512, Z)
        self.lv = nn.Linear(512, Z)
        self.dfc = nn.Linear(Z, 512)
        self.d1 = nn.ConvTranspose2d(512, 64, 5, 2)
        self.d2 = nn.ConvTranspose2d(64, 32, 5, 2)
        self.d3 = nn.ConvTranspose2d(32, 16, 6, 2)
        self.d4 = nn.ConvTranspose2d(16, 3, 6, 2)

    def encode(self, x, keep=None):
        a1 = F.relu(self.e1(x)); a2 = F.relu(self.e2(a1)); a3 = F.relu(self.e3(a2)); a4 = F.relu(self.e4(a3))
        f = a4.flatten(1)
        mu, lv = self.mu(f), self.lv(f)
        if keep is not None:
            keep.update(e1=a1, e2=a2, e3=a3, e4=a4, mu=mu, lv=lv)
        return mu, lv

    def decode(self, z, keep=None):
        f = self.dfc(z)
        b1 = F.relu(self.d1(f.view(-1, 512, 1, 1))); b2 = F.relu(self.d2(b1)); b3 = F.relu(self.d3(b2)); y = torch.sigmoid(self.d4(b3))
        if keep is not None:
            keep.update(dfc=f, d1=b1, d2=b2, d3=b3, d4=y)
        return y


class MDNRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(Z + NA, HID, batch_first=True)
        self.head = nn.Linear(HID, Z * KMIX * 3 + 1)

    def forward(self, z, a, hc=None):
        x = torch.cat([z, F.one_hot(a.long(), NA).float()], -1)
        out, hc = self.lstm(x, hc)
        return self.split(self.head(out)), hc

    @staticmethod
    def split(y):
        n = Z * KMIX
        logpi = y[..., :n].unflatten(-1, (Z, KMIX))
        mu = y[..., n:2 * n].unflatten(-1, (Z, KMIX))
        logsig = y[..., 2 * n:3 * n].unflatten(-1, (Z, KMIX))
        return F.log_softmax(logpi, -1), mu, logsig, y[..., -1]


def mdn_nll(logpi, mu, logsig, z):
    z = z.unsqueeze(-1)
    lp = -0.5 * ((z - mu) / logsig.exp()) ** 2 - logsig - 0.5 * math.log(2 * math.pi)
    return -torch.logsumexp(logpi + lp, -1)   # (..., Z)


def to_rgb(frames_u8, pal):
    """调色板编号 (N,64,64) → (N,3,64,64) 0..1"""
    return pal[frames_u8.long()].permute(0, 3, 1, 2)


# ---------------------------------------------------------------- V

def train_vae(work, dev, epochs, bs, beta=1.0, tol=0.5, tag='', w_ob=1.0, w_car=1.0):
    D = np.load(os.path.join(work, 'data.npz'))
    frames = torch.from_numpy(D['frames']).to(dev)
    starts = D['starts']
    n_ep = len(starts) - 1
    # 留出最后 2% 的局做测试
    n_test = max(20, n_ep // 50)
    split = int(starts[n_ep - n_test])
    pal = torch.tensor(PALETTE, dtype=torch.float32, device=dev) / 255
    # 重建误差按像素加权：抛锚车（调色板 7、8）和小车（5、6、11）只占画面的 1%，不加权的话 VAE 会把抛锚车当成噪声整个丢掉
    pw = torch.ones(len(PALETTE), device=dev)
    pw[[7, 8]] = w_ob
    pw[[5, 6, 11]] = w_car
    torch.manual_seed(0)
    vae = VAE().to(dev)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    steps = epochs * (split // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
    curve = []
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(split, device=dev)
        for i in range(split // bs):
            fb = frames[perm[i * bs:(i + 1) * bs]]
            x = to_rgb(fb, pal)
            mu, lv = vae.encode(x)
            z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
            y = vae.decode(z)
            r = (((y - x) ** 2).sum(1) * pw[fb.long()]).sum((1, 2)).mean()
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1))
            kl_t = torch.clamp(kl, min=tol * Z).mean()
            loss = r + beta * kl_t
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 25 == 0:
                curve.append([step, round(r.item(), 3), round(kl.mean().item(), 3)])
            step += 1
        print(f'V 第 {ep + 1}/{epochs} 轮：重建 {r.item():.2f}，KL {kl.mean().item():.2f}（{time.time() - t0:.0f} 秒）', flush=True)
    secs = time.time() - t0
    # 测试集
    vae.eval()
    with torch.no_grad():
        rs, kls = [], []
        for i in range(split, len(frames), 2048):
            x = to_rgb(frames[i:i + 2048], pal)
            mu, lv = vae.encode(x)
            y = vae.decode(mu)
            rs.append(((y - x) ** 2).sum((1, 2, 3)))
            kls.append(-0.5 * (1 + lv - mu ** 2 - lv.exp()))
        rs = torch.cat(rs); kls = torch.cat(kls)
    info = {'epochs': epochs, 'batch': bs, 'steps': steps, 'seconds': round(secs, 1), 'trainFrames': split, 'testFrames': int(len(frames) - split),
            'testRecon': round(rs.mean().item(), 3), 'testMSE': round(rs.mean().item() / (3 * W * H), 6),
            'testKL': round(kls.sum(1).mean().item(), 3), 'klPerDim': [round(v, 4) for v in kls.mean(0).tolist()],
            'finalRecon': curve[-1][1], 'finalKL': curve[-1][2]}
    info.update(beta=beta, klTolerance=tol, weightObstacle=w_ob, weightCar=w_car)
    torch.save(vae.state_dict(), os.path.join(work, f'vae{tag}.pt'))
    json.dump({'info': info, 'curve': curve}, open(os.path.join(work, f'vae{tag}.json'), 'w'))
    print('V：', {k: v for k, v in info.items() if k != 'klPerDim'})


def encode_all(work, dev):
    D = np.load(os.path.join(work, 'data.npz'))
    vae = VAE().to(dev)
    vae.load_state_dict(torch.load(os.path.join(work, 'vae.pt')))
    vae.eval()
    pal = torch.tensor(PALETTE, dtype=torch.float32, device=dev) / 255
    frames = torch.from_numpy(D['frames']).to(dev)
    mus, lvs = [], []
    with torch.no_grad():
        for i in range(0, len(frames), 4096):
            mu, lv = vae.encode(to_rgb(frames[i:i + 4096], pal))
            mus.append(mu); lvs.append(lv)
    return torch.cat(mus), torch.cat(lvs), D


# ---------------------------------------------------------------- M

def train_rnn(work, dev, iters, bs, L):
    mu, lv, D = encode_all(work, dev)
    starts = D['starts']
    acts = torch.from_numpy(D['acts'].astype(np.int64)).to(dev)
    done = D['done']
    n_ep = len(starts) - 1
    n_test = max(20, n_ep // 50)
    n_train = n_ep - n_test
    lens = np.diff(starts) - 1   # 每局的步数
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    rnn = MDNRNN().to(dev)
    opt = torch.optim.Adam(rnn.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=iters, pct_start=0.05)
    p_ep = lens[:n_train] / lens[:n_train].sum()
    sd = (0.5 * lv).exp()

    def batch(eps):
        T = lens[eps]
        st = (rng.random(len(eps)) * np.maximum(1, T - L + 1)).astype(np.int64)
        n = np.minimum(L, T - st)
        ar = np.arange(L)
        idx = starts[eps][:, None] + st[:, None] + np.minimum(ar[None], n[:, None] - 1)
        mask = (ar[None] < n[:, None]).astype(np.float32)
        dn = np.zeros((len(eps), L), np.float32)
        hit = done[eps] & (st + n == T)
        dn[np.nonzero(hit)[0], (n - 1)[hit]] = 1
        return torch.from_numpy(idx).to(dev), torch.from_numpy(mask).to(dev), torch.from_numpy(dn).to(dev)

    curve = []
    t0 = time.time()
    for it in range(iters):
        eps = rng.choice(n_train, size=bs, p=p_ep)
        idx, mask, dn = batch(eps)
        z = mu[idx] + sd[idx] * torch.randn(idx.shape + (Z,), device=dev)
        zn = mu[idx + 1] + sd[idx + 1] * torch.randn(idx.shape + (Z,), device=dev)
        (logpi, m, ls, dl), _ = rnn(z, acts[idx])
        nll = (mdn_nll(logpi, m, ls, zn).mean(-1) * mask).sum() / mask.sum()
        bce = (F.binary_cross_entropy_with_logits(dl, dn, pos_weight=torch.tensor(5.0, device=dev), reduction='none') * mask).sum() / mask.sum()
        loss = nll + bce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 25 == 0:
            curve.append([it, round(nll.item(), 4), round(bce.item(), 4)])
        if it % 500 == 0 or it == iters - 1:
            print(f'M 第 {it}/{iters} 步：MDN 负对数似然 {nll.item():.3f}，撞车 BCE {bce.item():.4f}（{time.time() - t0:.0f} 秒）', flush=True)
    secs = time.time() - t0
    # 测试集：整局从零状态开始跑
    rnn.eval()
    with torch.no_grad():
        nlls, dl_all, dn_all = [], [], []
        for e in range(n_train, n_ep):
            T = lens[e]
            if T < 1:
                continue
            ids = torch.arange(starts[e], starts[e] + T, device=dev)
            (logpi, m, ls, dl), _ = rnn(mu[ids][None], acts[ids][None])
            nlls.append(mdn_nll(logpi, m, ls, mu[ids + 1][None]).mean(-1)[0])
            dl_all.append(dl[0])
            d = torch.zeros(T, device=dev)
            if done[e]:
                d[-1] = 1
            dn_all.append(d)
        nlls = torch.cat(nlls); dl_all = torch.cat(dl_all); dn_all = torch.cat(dn_all)
        pred = dl_all > 0
        tp = (pred & (dn_all > 0)).sum().item(); fp = (pred & (dn_all == 0)).sum().item(); fn = ((~pred) & (dn_all > 0)).sum().item()
    info = {'iters': iters, 'batch': bs, 'seqLen': L, 'seconds': round(secs, 1), 'finalNLL': curve[-1][1], 'finalBCE': curve[-1][2],
            'testNLL': round(nlls.mean().item(), 4), 'doneRecall': round(tp / max(1, tp + fn), 3), 'donePrecision': round(tp / max(1, tp + fp), 3)}
    torch.save(rnn.state_dict(), os.path.join(work, 'rnn.pt'))
    json.dump({'info': info, 'curve': curve}, open(os.path.join(work, 'rnn.json'), 'w'))
    print('M：', info)


# ---------------------------------------------------------------- C：在梦里用 CMA-ES 训练

def dream_rollouts(rnn, Wc, z0, steps, tau, dev, gen):
    """Wc: (P, Z+HID+1, NA)；z0: (N, Z) 起点。返回每个个体在梦里活了多少步（P, N）"""
    P, N = Wc.shape[0], z0.shape[0]
    z = z0[None].expand(P, N, Z).reshape(P * N, Z)
    h = torch.zeros(1, P * N, HID, device=dev); c = torch.zeros_like(h)
    alive = torch.ones(P * N, device=dev)
    life = torch.zeros(P * N, device=dev)
    Wr = Wc.repeat_interleave(N, 0)
    for t in range(steps):
        feat = torch.cat([z, h[0], torch.ones(P * N, 1, device=dev)], -1)
        a = torch.einsum('bi,bia->ba', feat, Wr).argmax(-1)
        (logpi, m, ls, dl), (h, c) = rnn(z[:, None], a[:, None], (h, c))
        logpi, m, ls, dl = logpi[:, 0], m[:, 0], ls[:, 0], dl[:, 0]
        died = torch.rand(P * N, device=dev, generator=gen) < torch.sigmoid(dl)
        life += alive
        alive = alive * (~died).float()
        k = torch.distributions.Categorical(logits=logpi / tau).sample()
        mk = m.gather(-1, k[..., None])[..., 0]
        sk = ls.gather(-1, k[..., None])[..., 0].exp()
        z = mk + sk * math.sqrt(tau) * torch.randn(mk.shape, device=dev, generator=gen)
    return life.view(P, N)


def real_eval(vae, rnn, w, seeds, dev, cap=1000):
    """拿梦里学出来的控制器去开真正的游戏"""
    pal = torch.tensor(PALETTE, dtype=torch.float32, device=dev) / 255
    w = torch.as_tensor(w, dtype=torch.float32, device=dev)
    lives = []
    for s in seeds:
        g = Game(int(s))
        h = torch.zeros(1, 1, HID, device=dev); c = torch.zeros_like(h)
        with torch.no_grad():
            while not g.done and g.t < cap:
                mu, _ = vae.encode(to_rgb(torch.from_numpy(g.render())[None].to(dev), pal))
                feat = torch.cat([mu[0], h[0, 0], torch.ones(1, device=dev)])
                a = int((feat @ w).argmax())
                _, (h, c) = rnn(mu[:, None], torch.tensor([[a]], device=dev), (h, c))
                g.step(a)
        lives.append(g.t)
    return lives


def train_ctrl(work, dev, gens, pop, n_roll, steps, tau):
    mu, lv, D = encode_all(work, dev)
    vae = VAE().to(dev); vae.load_state_dict(torch.load(os.path.join(work, 'vae.pt'))); vae.eval()
    rnn = MDNRNN().to(dev); rnn.load_state_dict(torch.load(os.path.join(work, 'rnn.pt'))); rnn.eval()
    starts = D['starts']
    z0_all = mu[torch.from_numpy(starts[:-1]).to(dev)]    # 每局的第一帧
    dim = (Z + HID + 1) * NA
    rng = np.random.default_rng(1)
    gen = torch.Generator(device=dev); gen.manual_seed(1)
    # 简化版 CMA-ES（对角协方差 + 步长自适应，即 sep-CMA-ES）
    mean = np.zeros(dim); sigma = 0.1; C = np.ones(dim)
    mu_n = pop // 2
    wts = np.log(mu_n + 0.5) - np.log(np.arange(1, mu_n + 1)); wts /= wts.sum()
    mueff = 1 / (wts ** 2).sum()
    cs = (mueff + 2) / (dim + mueff + 5); ds = 1 + 2 * max(0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    cc = 4 / (dim + 4); c1 = 2 / ((dim + 1.3) ** 2 + mueff) * (dim + 2) / 3
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)) * (dim + 2) / 3
    ps = np.zeros(dim); pc = np.zeros(dim)
    chiN = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim * dim))
    curve = []
    best = (-1, None)
    t0 = time.time()
    with torch.no_grad():
        for gi in range(gens):
            zs = rng.standard_normal((pop, dim))
            xs = mean + sigma * zs * np.sqrt(C)
            Wc = torch.tensor(xs, dtype=torch.float32, device=dev).view(pop, Z + HID + 1, NA)
            z0 = z0_all[torch.from_numpy(rng.integers(0, len(z0_all), n_roll)).to(dev)]
            fit = dream_rollouts(rnn, Wc, z0, steps, tau, dev, gen).mean(1).cpu().numpy()
            order = np.argsort(-fit)
            if fit[order[0]] > best[0]:
                best = (float(fit[order[0]]), xs[order[0]].copy())
            sel = zs[order[:mu_n]]
            zmean = wts @ sel
            mean = mean + sigma * np.sqrt(C) * zmean
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * zmean
            hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (gi + 1))) / chiN < 1.4 + 2 / (dim + 1)
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * np.sqrt(C) * zmean
            C = (1 - c1 - cmu) * C + c1 * pc ** 2 + cmu * (wts @ (C * sel ** 2))
            sigma *= math.exp((cs / ds) * (np.linalg.norm(ps) / chiN - 1))
            curve.append([gi, round(float(fit.mean()), 2), round(float(fit[order[0]]), 2)])
            if gi % 10 == 0 or gi == gens - 1:
                print(f'C 第 {gi}/{gens} 代：梦里平均活 {fit.mean():.1f} 步，最好 {fit[order[0]]:.1f}（{time.time() - t0:.0f} 秒）', flush=True)
    secs = time.time() - t0
    # 检验：用分布均值（比单个最好的个体稳）在梦里和真实游戏里各跑一批
    with torch.no_grad():
        Wm = torch.tensor(mean, dtype=torch.float32, device=dev).view(1, Z + HID + 1, NA)
        dream = dream_rollouts(rnn, Wm, z0_all[:200], steps, tau, dev, gen).mean().item()
    seeds = list(range(900000, 900030))
    real = real_eval(vae, rnn, mean.reshape(Z + HID + 1, NA), seeds, dev)
    # 对照：随机乱开、一直直行
    rnd_life = []
    for s in seeds:
        g = Game(s); r = np.random.default_rng(s)
        while not g.done and g.t < 1000:
            g.step(int(r.integers(0, 3)))
        rnd_life.append(g.t)
    straight = []
    for s in seeds:
        g = Game(s)
        while not g.done and g.t < 1000:
            g.step(0)
        straight.append(g.t)
    info = {'gens': gens, 'pop': pop, 'rollouts': n_roll, 'steps': steps, 'tau': tau, 'seconds': round(secs, 1), 'params': dim,
            'dreamLife': round(dream, 1), 'realLife': round(float(np.mean(real)), 1), 'realLifes': real,
            'randomLife': round(float(np.mean(rnd_life)), 1), 'straightLife': round(float(np.mean(straight)), 1)}
    np.save(os.path.join(work, 'ctrl.npy'), mean.reshape(Z + HID + 1, NA).astype(np.float32))
    json.dump({'info': info, 'curve': curve}, open(os.path.join(work, 'ctrl.json'), 'w'))
    print('C：', {k: v for k, v in info.items() if k != 'realLifes'})


# ---------------------------------------------------------------- 导出

def export(work, out_dir, dev):
    from export_world import export_all
    export_all(work, out_dir, dev, VAE, MDNRNN, to_rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='/mnt/d/cjc/world-model')
    ap.add_argument('--out', default=os.path.join(ROOT, 'public', 'world', 'data'))
    ap.add_argument('--stage', default='all', choices=['all', 'collect', 'vae', 'rnn', 'ctrl', 'export'])
    ap.add_argument('--expert', type=int, default=2500)
    ap.add_argument('--random', type=int, default=1500)
    ap.add_argument('--cap', type=int, default=300)
    ap.add_argument('--vae-epochs', type=int, default=30)
    ap.add_argument('--vae-beta', type=float, default=0.25)
    ap.add_argument('--vae-tol', type=float, default=0.5)
    ap.add_argument('--vae-wob', type=float, default=6.0)
    ap.add_argument('--vae-wcar', type=float, default=2.0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--rnn-iters', type=int, default=6000)
    ap.add_argument('--ctrl-gens', type=int, default=120)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    st = args.stage
    if st in ('all', 'collect'):
        collect(args.work, args.expert, args.random, args.cap)
    if st in ('all', 'vae'):
        train_vae(args.work, dev, args.vae_epochs, 256, args.vae_beta, args.vae_tol, args.tag, args.vae_wob, args.vae_wcar)
    if st in ('all', 'rnn'):
        train_rnn(args.work, dev, args.rnn_iters, 256, 96)
    if st in ('all', 'ctrl'):
        train_ctrl(args.work, dev, args.ctrl_gens, 48, 16, 300, 1.0)
    if st in ('all', 'export'):
        export(args.work, args.out, dev)
    print(f'共 {time.time() - t0:.0f} 秒，显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GB')


if __name__ == '__main__':
    main()
