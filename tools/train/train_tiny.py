"""从零训练一个 Qwen3 同构的小模型（唐宋格律诗，字符级），把整段训练过程记录下来给网页回放。

用法：
    python tools/train/prep_corpus.py          # 先准备语料
    python tools/train/train_tiny.py           # 训练 + 导出到 public/train/data/tiny.*、tiny/（首屏文件 + 按需分块，见 split_data.py）

模型用的是 transformers 里真正的 Qwen3ForCausalLM，只是把尺寸改小：
保留 GQA（4 个查询头共用 2 组键值头）、每个头的 q_norm / k_norm、RMSNorm、RoPE、SwiGLU、输入输出共享嵌入。

记录的内容（全部来自这一次真实训练）：
- 每一步的训练损失、学习率（线性预热 + 余弦退火）、裁剪前的梯度范数；每 100 步的验证损失；
- 约 40 个检查点（前密后疏），每个检查点：
    * 用同一组随机数、同样的开头生成的诗（看它从乱码变成诗）；
    * 留出的《登鹳雀楼》上每个字的预测概率和前 5 名，以及所有层所有头的注意力；
    * 嵌入的二维 PCA（每个检查点各自做 PCA，再用正交 Procrustes 旋转对齐上一帧）；
    * 三块权重的局部真实数值（48×48）和它们这一步的梯度；
    * 这一步训练循环的全部细节：批次第 0 行的字、逐字损失、逻辑透镜、残差范数、
      残差流上的梯度（反向传播逐层流过）、每个参数张量的梯度范数、几个权重的 AdamW 完整算式、
      更新之后对同一批次重新前向的结果（演示用）；
- 训练结束后，在检查点轨迹的前两个主成分张成的平面上真实计算一张损失地形。
"""
import argparse
import json
import math
import pathlib
import pickle
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

import split_data

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'public' / 'train' / 'data'

PREFIXES = ['', '春风', '明月', '白云']   # 生成样例用的开头（前面都有 <|endoftext|>）
PCA_EXTRA = '一二三四五六七八九十百千万春夏秋冬东南西北红白青黄绿紫碧金山水风月花雪云雨江河湖海天日星霜露烟马鸟雁莺猿龙鱼鹤我君人心梦愁泪酒琴书剑舟楼城门家国'
CROP = 48
TOPN = 5
ROWN = 64   # 批次第 0 行只导出前 64 个位置


def cosine_lr(t, S, W, peak, floor):
    if t < W:
        return peak * (t + 1) / W
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * (t - W) / max(1, S - W)))


def ckpt_steps(S):
    a = [0, 1, 2, 3, 4, 6, 8]
    b = np.round(np.geomspace(11, S - 1, 34)).astype(int).tolist()
    return sorted(set(a + b + [S - 1]))


class Rec:
    """二进制数组的收集器：json 里写 dtype/shape/offset，数据接在一个 .bin 里。"""

    def __init__(self):
        self.parts, self.spec, self.off = [], {}, 0

    def add(self, name, arr, dtype):
        a = np.ascontiguousarray(np.asarray(arr).astype(dtype))
        pad = (-self.off) % 8
        if pad:
            self.parts.append(b'\0' * pad)
            self.off += pad
        self.spec[name] = {'dtype': np.dtype(dtype).name, 'shape': list(a.shape), 'offset': self.off}
        b = a.tobytes()
        self.parts.append(b)
        self.off += len(b)

    def bytes(self):
        return b''.join(self.parts)


def q8(a):
    """对称 int8 量化，返回 (int8 数组, 比例)。"""
    a = np.asarray(a, dtype=np.float32)
    s = float(np.abs(a).max()) / 127 or 1e-12
    return np.clip(np.round(a / s), -127, 127).astype(np.int8), s


def f9(x):
    """float32 精确往返所需的 9 位有效数字。"""
    return float(f'{float(x):.9g}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/mnt/d/cjc/datasets/poetry-train')
    ap.add_argument('--ckdir', default='/mnt/d/cjc/checkpoints/tiny-poetry')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quick', action='store_true', help='只跑几十步做冒烟测试，不写网页数据')
    ap.add_argument('--export-only', action='store_true', help='不重新训练，读回上次的记录重新导出')
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = 'cuda'
    data = np.load(pathlib.Path(a.data) / 'corpus.npz')
    info = json.loads((pathlib.Path(a.data) / 'vocab.json').read_text(encoding='utf-8'))
    vocab = info['vocab']
    V = len(vocab)
    cidx = {c: i for i, c in enumerate(vocab)}
    tr_ids = torch.from_numpy(data['train_ids'].astype(np.int64))
    tr_starts = data['train_starts']
    va_ids = torch.from_numpy(data['val_ids'].astype(np.int64))
    va_starts = data['val_starts']
    held = torch.from_numpy(data['held'].astype(np.int64))
    S, B, T = a.steps, a.batch, a.seq
    if a.quick:
        S = 60

    cfg = Qwen3Config(
        vocab_size=V, hidden_size=a.hidden, intermediate_size=a.hidden * 3, num_hidden_layers=a.layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=a.hidden // 4, max_position_embeddings=256,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, attention_bias=False, attention_dropout=0.0,
        # 和 Qwen3-0.6B 一样不设 pad：设了的话 0 号（<|endoftext|>）的嵌入会被初始化成 0 且永远不更新
        bos_token_id=0, eos_token_id=0, pad_token_id=None, initializer_range=0.02, use_cache=False,
    )
    model = Qwen3ForCausalLM(cfg).to(dev)
    model.set_attn_implementation('eager')   # 要拿到注意力权重
    NL, H = cfg.num_hidden_layers, cfg.num_attention_heads
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.model.embed_tokens.weight.numel()
    print(f'参数 {n_params:,}（嵌入 {n_embed:,}）  词表 {V}')

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    BETAS, EPS, WD, CLIP = (0.9, 0.95), 1e-8, 0.1, 1.0
    PEAK, FLOOR = a.lr, a.lr * 0.1
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': WD}, {'params': no_decay, 'weight_decay': 0.0}], lr=PEAK, betas=BETAS, eps=EPS, foreach=False)
    pname = {id(p): n for n, p in model.named_parameters()}
    tensors = [n for n, _ in model.named_parameters()]

    # 批次：每一行都从某首诗前面的 <|endoftext|> 开始，连续取 T+1 个字（会跨过好几首诗）
    g = np.random.default_rng(a.seed + 1)
    ok_starts = tr_starts[tr_starts + T + 1 <= len(tr_ids)]

    def get_batch():
        st = g.choice(ok_starts, B)
        return torch.stack([tr_ids[s:s + T + 1] for s in st]).to(dev)

    vg = np.random.default_rng(123)
    vst = vg.choice(va_starts[va_starts + T + 1 <= len(va_ids)], 64, replace=False)
    val_batch = torch.stack([va_ids[s:s + T + 1] for s in vst]).to(dev)

    @torch.no_grad()
    def val_loss(m=model, vb=val_batch):
        logits = m(input_ids=vb[:, :-1]).logits.float()
        return F.cross_entropy(logits.reshape(-1, V), vb[:, 1:].reshape(-1)).item()

    # 生成用的随机数：每个开头一串，所有检查点共用（同样的骰子，差别只来自模型）
    ug = np.random.default_rng(7)
    U = ug.random((len(PREFIXES), 80))
    TEMP = 0.8

    @torch.no_grad()
    def sample(prefix, u):
        ids = [0] + [cidx[c] for c in prefix]
        out = []
        for k in range(64):
            logits = model(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].float()
            p = torch.softmax(logits / TEMP, -1).cpu().numpy().astype(np.float64)
            c = int(np.searchsorted(np.cumsum(p), u[k] * p.sum()))
            c = min(c, V - 1)
            if c == 0:
                break
            ids.append(c)
            out.append(vocab[c])
        return prefix + ''.join(out)

    # 嵌入 PCA 用的字：最常见的 300 个 + 几组有意思的字
    pca_ids = list(range(1, 301))
    for ch in PCA_EXTRA:
        if ch in cidx and cidx[ch] not in pca_ids:
            pca_ids.append(cidx[ch])
    pca_ids = [0] + pca_ids
    prev_pca = None

    def pca2(E):
        X = E - E.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        Y = X @ Vt[:2].T
        return Y

    # 权重局部：嵌入里最常见的 48 个字的前 48 维、第 2 层 W_q、第 4 层 W_down 左上角
    crop_names = ['model.embed_tokens.weight', f'model.layers.{min(2, NL - 1)}.self_attn.q_proj.weight', f'model.layers.{min(4, NL - 1)}.mlp.down_proj.weight']
    params = dict(model.named_parameters())

    def crop(name, grad=False):
        p = params[name]
        t = p.grad if grad else p.data
        if name.endswith('embed_tokens.weight'):
            return t[1:CROP + 1, :CROP].float().cpu().numpy()
        return t[:CROP, :CROP].float().cpu().numpy()

    # AdamW 候选权重：训练时每一步都记下 w、g、m、v，导出时挑几个轨迹最有代表性的
    cands = []
    for ch in ['月', '春', '山', '，', '。']:
        for d in range(0, a.hidden, 16):
            cands.append(('model.embed_tokens.weight', (cidx[ch], d), f'嵌入 · 「{ch}」第 {d} 维'))
    cands.append(('model.embed_tokens.weight', (0, 3), '嵌入 · <|endoftext|> 第 3 维'))
    for L in range(NL):
        for (i, j) in [(5, 40), (70, 11), (130, 200)]:
            cands.append((f'model.layers.{L}.self_attn.q_proj.weight', (i, j), f'第 {L} 层 W_q[{i}, {j}]'))
            cands.append((f'model.layers.{L}.mlp.down_proj.weight', (i, j * 3 % (a.hidden * 3)), f'第 {L} 层 W_down[{i}, {j * 3 % (a.hidden * 3)}]'))
            cands.append((f'model.layers.{L}.mlp.gate_proj.weight', (j * 3 % (a.hidden * 3), i), f'第 {L} 层 W_gate[{j * 3 % (a.hidden * 3)}, {i}]'))
        for d in [3, 77, 150]:
            cands.append((f'model.layers.{L}.input_layernorm.weight', (d,), f'第 {L} 层 RMSNorm γ[{d}]'))
    cand_hist = np.zeros((len(cands), S, 7), dtype=np.float64)   # w_old, g, m_prev, v_prev, m, v, w_new

    # 按张量分组，一次取出同一张量里的所有候选（每步只同步一次）
    groups = {}
    for c, (name, idx, _) in enumerate(cands):
        shp = params[name].shape
        flat_i = idx[0] * shp[1] + idx[1] if len(idx) == 2 else idx[0]
        groups.setdefault(name, ([], []))
        groups[name][0].append(c)
        groups[name][1].append(flat_i)
    g_names = list(groups)
    g_idx = {n: torch.tensor(groups[n][1], device=dev) for n in g_names}
    g_order = np.argsort(np.concatenate([groups[n][0] for n in g_names]))

    def gather(kind):
        vals = []
        for name in g_names:
            p = params[name]
            if kind == 'w':
                t = p.data
            elif kind == 'g':
                t = p.grad
            else:
                st = opt.state.get(p)
                if not st:
                    vals.append(torch.zeros(len(g_idx[name]), device=dev))
                    continue
                t = st['exp_avg' if kind == 'm' else 'exp_avg_sq']
            vals.append(t.reshape(-1)[g_idx[name]].float())
        return torch.cat(vals).double().cpu().numpy()[g_order]

    cks = ckpt_steps(S) if not a.quick else [0, 1, 5, 20, 59]
    K = len(cks)
    ck_set = {t: k for k, t in enumerate(cks)}
    losses, lrs, gns, val_curve = np.zeros(S), np.zeros(S), np.zeros(S), []
    recs = []   # 每个检查点一份
    pathlib.Path(a.ckdir).mkdir(parents=True, exist_ok=True)
    if not a.export_only:
        t0 = time.time()
        t_ck = 0.0

        for t in range(S):
            lr = cosine_lr(t, S, a.warmup, PEAK, FLOOR)
            for pg in opt.param_groups:
                pg['lr'] = lr
            x = get_batch()
            inp, tgt = x[:, :-1], x[:, 1:]
            is_ck = t in ck_set
            r = None
            if is_ck:
                tc = time.time()
                model.eval()
                r = {'t': t, 'lr': lr}
                r['valLoss'] = val_loss()
                # 留出的那首诗
                with torch.no_grad():
                    out = model(input_ids=held[None, :-1].to(dev), output_attentions=True)
                    pr = torch.softmax(out.logits[0].float(), -1)
                    tg = held[1:].to(dev)
                    r['val_p'] = pr[torch.arange(len(tg)), tg].cpu().numpy()
                    tp = pr.topk(TOPN, -1)
                    r['val_top_id'] = tp.indices.cpu().numpy()
                    r['val_top_p'] = tp.values.cpu().numpy()
                    r['attn'] = torch.stack([at[0] for at in out.attentions]).float().cpu().numpy()   # NL, H, L, L
                r['samples'] = [sample(pf, U[i]) for i, pf in enumerate(PREFIXES)]
                E = model.model.embed_tokens.weight.detach()[pca_ids].float().cpu().numpy()
                Y = pca2(E)
                if prev_pca is not None:   # 正交 Procrustes：只旋转 / 翻转，不改变形状
                    M = Y.T @ prev_pca
                    u_, _, vt_ = np.linalg.svd(M)
                    Y = Y @ (u_ @ vt_)
                prev_pca = Y
                r['pca_scale'] = float(np.sqrt((Y ** 2).sum(1).mean()))
                r['pca'] = Y / r['pca_scale']
                r['wcrop'] = [crop(n) for n in crop_names]
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, pathlib.Path(a.ckdir) / f'step{t:05d}.pt')
                model.train()
                t_ck += time.time() - tc

            # ---------------- 真正的一步训练 ----------------
            caps = []
            hooks = []
            if is_ck:
                def keep(_m, _i, o):
                    h = o[0] if isinstance(o, tuple) else o
                    h.retain_grad()
                    caps.append(h)
                hooks.append(model.model.embed_tokens.register_forward_hook(keep))
                for layer in model.model.layers:
                    hooks.append(layer.register_forward_hook(keep))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=inp).logits
            logits = logits.float()
            loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for hk in hooks:
                hk.remove()
            if is_ck:
                tc = time.time()
                with torch.no_grad():
                    pr = torch.softmax(logits[0, :ROWN], -1)
                    tg0 = tgt[0, :ROWN]
                    r['row_ids'] = x[0, :ROWN + 1].cpu().numpy()
                    r['row_p'] = pr[torch.arange(ROWN), tg0].cpu().numpy()
                    tp = pr.topk(TOPN, -1)
                    r['row_top_id'] = tp.indices.cpu().numpy()
                    r['row_top_p'] = tp.values.cpu().numpy()
                    r['row_loss_all'] = F.cross_entropy(logits[0], tgt[0]).item()
                    # 逻辑透镜：每一层的残差流直接接最终 RMSNorm + 输出矩阵
                    lens_top, lens_p, rn, rg = [], [], [], []
                    for h in caps:
                        h0 = h[0:1, :ROWN].float()
                        lz = model.lm_head(model.model.norm(h0))[0].float()
                        lp = torch.softmax(lz, -1)
                        lens_top.append(lp.argmax(-1).cpu().numpy())
                        lens_p.append(lp[torch.arange(ROWN), tg0].cpu().numpy())
                        rn.append(h0[0].norm(dim=-1).cpu().numpy())
                        rg.append(h.grad[0, :ROWN].float().norm(dim=-1).cpu().numpy())
                    r['lens_top'], r['lens_p'], r['resid_norm'], r['resid_grad'] = map(np.array, (lens_top, lens_p, rn, rg))
                    r['gcrop'] = [crop(n, grad=True) for n in crop_names]
                    r['tgrad'] = np.array([params[n].grad.float().norm().item() for n in tensors])
                t_ck += time.time() - tc
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP).item()
            w_old, g_now, m_prev, v_prev = gather('w'), gather('g'), gather('m'), gather('v')
            opt.step()
            m_new, v_new, w_new = gather('m'), gather('v'), gather('w')
            cand_hist[:, t] = np.stack([w_old, g_now, m_prev, v_prev, m_new, v_new, w_new], 1)
            losses[t], lrs[t], gns[t] = loss.item(), lr, gn
            if is_ck:
                tc = time.time()
                r['loss'], r['gradNorm'], r['clip'] = loss.item(), gn, min(1.0, CLIP / (gn + 1e-6))
                model.eval()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    lg2 = model(input_ids=inp).logits.float()
                model.train()
                r['lossAfter'] = F.cross_entropy(lg2.reshape(-1, V), tgt.reshape(-1)).item()
                pr2 = torch.softmax(lg2[0, :ROWN], -1)
                r['row_p_after'] = pr2[torch.arange(ROWN), tgt[0, :ROWN]].cpu().numpy()
                recs.append(r)
                t_ck += time.time() - tc
                print(f'[检查点 {len(recs):2d}/{K}] step {t:5d}  loss {loss.item():.3f}  val {r["valLoss"]:.3f}  lr {lr:.2e}  |g| {gn:.2f}  样例：{r["samples"][1][:24]}')
            if t % 100 == 0 or t == S - 1:
                model.eval()
                val_curve.append((t, val_loss()))
                model.train()
        train_sec = time.time() - t0
        final_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(final_state, pathlib.Path(a.ckdir) / 'final.pt')
        model.eval()
        final_val = val_loss()
        final_samples = [sample(pf, U[i]) for i, pf in enumerate(PREFIXES)]
        print(f'训练用时 {train_sec:.1f}s（其中检查点记录 {t_ck:.1f}s），最终验证损失 {final_val:.3f}')
        for s_ in final_samples:
            print('  ', s_)
        with open(pathlib.Path(a.ckdir) / 'run.pkl', 'wb') as f:
            pickle.dump({'recs': recs, 'cand_hist': cand_hist, 'losses': losses, 'lrs': lrs, 'gns': gns, 'val_curve': val_curve,
                         'train_sec': train_sec, 't_ck': t_ck, 'final_val': final_val, 'final_samples': final_samples}, f)
    else:   # 只重新导出：读回上次训练的记录
        with open(pathlib.Path(a.ckdir) / 'run.pkl', 'rb') as f:
            run = pickle.load(f)
        recs, cand_hist, losses, lrs, gns, val_curve = (run[k] for k in ('recs', 'cand_hist', 'losses', 'lrs', 'gns', 'val_curve'))
        train_sec, t_ck, final_val, final_samples = (run[k] for k in ('train_sec', 't_ck', 'final_val', 'final_samples'))
        final_state = torch.load(pathlib.Path(a.ckdir) / 'final.pt')
        model.eval()

    # ---------------- AdamW 算式核对 ----------------
    b1, b2 = BETAS
    err, merr = 0.0, 0.0   # 权重：以 fp32 的最小间隔（ulp）计；动量：相对误差
    for c in range(len(cands)):
        name = cands[c][0]
        wd = WD if params[name].dim() >= 2 else 0.0
        for t in range(S):
            w0, gg, m0, v0, m1, v1, w1 = cand_hist[c, t]
            step = t + 1
            mh, vh = m1 / (1 - b1 ** step), v1 / (1 - b2 ** step)
            pred = w0 * (1 - lrs[t] * wd) - lrs[t] * mh / (math.sqrt(vh) + EPS)
            err = max(err, abs(pred - w1) / float(np.spacing(np.float32(max(abs(w0), abs(w1))))))
            merr = max(merr, abs(b1 * m0 + (1 - b1) * gg - m1) / (max(abs(b1 * m0), abs((1 - b1) * gg), abs(m1)) + 1e-30), abs(b2 * v0 + (1 - b2) * gg * gg - v1) / (abs(v1) + 1e-30))
    print(f'AdamW 公式复现：权重最大误差 {err:.2f} ulp（fp32），动量最大相对误差 {merr:.1e}')

    # 挑 4 个候选：嵌入里走得最远的、W_q 里走得最远的、W_down 里走得最远的、RMSNorm γ 里走得最远的
    travel = np.abs(cand_hist[:, -1, 6] - cand_hist[:, 0, 0])
    feats = []
    for key in ['embed_tokens', 'q_proj', 'down_proj', 'layernorm']:
        cs = [c for c in range(len(cands)) if key in cands[c][0]]
        feats.append(max(cs, key=lambda c: travel[c]))
    if a.quick:
        print('冒烟测试完成')
        return

    # ---------------- 损失地形 ----------------
    print('计算损失地形…')
    names = tensors   # 按参数去重（输出矩阵和嵌入是同一块权重）
    flat = lambda sd: torch.cat([sd[k].float().reshape(-1) for k in names])
    thf = flat(final_state).to(dev)
    traj = torch.stack([flat(torch.load(pathlib.Path(a.ckdir) / f'step{t:05d}.pt')).to(dev) - thf for t in cks])
    # 轨迹矩阵是 41 × 660 万，直接 SVD 太宽：改用 41 × 41 的格拉姆矩阵求主成分
    gram = (traj @ traj.T).double()
    lam, vec = torch.linalg.eigh(gram)
    lam, vec = lam.flip(0).clamp_min(0), vec.flip(1)
    d1 = (traj.T @ vec[:, 0].float()) / lam[0].sqrt().float()
    d2 = (traj.T @ vec[:, 1].float()) / lam[1].sqrt().float()
    evr = (lam[:2] / lam.sum()).cpu().numpy()
    coords = torch.stack([traj @ d1, traj @ d2], 1).cpu().numpy()
    lo, hi = coords.min(0), coords.max(0)
    span = hi - lo
    lo, hi = lo - span * 0.25, hi + span * 0.25
    G = 41
    xs, ys = np.linspace(lo[0], hi[0], G), np.linspace(lo[1], hi[1], G)
    grid = np.zeros((G, G), dtype=np.float32)
    lb = val_batch[:32]
    shapes = [(k, final_state[k].shape, final_state[k].numel()) for k in names]

    def load_vec(vec):
        o = 0
        sd = {}
        for k, shp, n in shapes:
            sd[k] = vec[o:o + n].view(shp)
            o += n
        model.load_state_dict(sd, strict=False)

    for iy, yv in enumerate(ys):
        for ix, xv in enumerate(xs):
            load_vec(thf + float(xv) * d1 + float(yv) * d2)
            grid[iy, ix] = val_loss(model, lb)
    path_loss = []
    for k in range(K):
        load_vec(thf + traj[k])
        path_loss.append(val_loss(model, lb))
    print(f'  前两个主成分解释了轨迹方差的 {evr.sum() * 100:.1f}%')

    # ---------------- 导出 ----------------
    rec = Rec()
    rec.add('loss', losses, np.float32)
    rec.add('lr', lrs, np.float32)
    rec.add('gnorm', gns, np.float32)
    stack = lambda key: np.stack([r[key] for r in recs])
    rec.add('val_p', stack('val_p'), np.float16)
    rec.add('val_top_id', stack('val_top_id'), np.uint16)
    rec.add('val_top_p', stack('val_top_p'), np.float16)
    rec.add('attn', np.round(stack('attn') * 255), np.uint8)
    rec.add('pca', stack('pca'), np.float16)
    wq = [[q8(r['wcrop'][i]) for i in range(3)] for r in recs]
    gq = [[q8(r['gcrop'][i]) for i in range(3)] for r in recs]
    rec.add('wcrop', np.array([[x[0] for x in row] for row in wq]), np.int8)
    rec.add('gcrop', np.array([[x[0] for x in row] for row in gq]), np.int8)
    rec.add('row_ids', stack('row_ids'), np.uint16)
    rec.add('row_p', stack('row_p'), np.float16)
    rec.add('row_top_id', stack('row_top_id'), np.uint16)
    rec.add('row_top_p', stack('row_top_p'), np.float16)
    rec.add('row_p_after', stack('row_p_after'), np.float16)
    rec.add('lens_top', stack('lens_top'), np.uint16)
    rec.add('lens_p', stack('lens_p'), np.float16)
    rec.add('resid_norm', stack('resid_norm'), np.float16)
    rec.add('resid_grad', stack('resid_grad'), np.float32)
    rec.add('tgrad', stack('tgrad'), np.float32)
    # 候选权重的轨迹：每 4 步取一个点画曲线；检查点那一步的完整算式单独写进 json
    DS = 4
    hist = cand_hist[feats][:, ::DS]
    rec.add('feat_w', hist[:, :, 6], np.float32)
    rec.add('feat_g', hist[:, :, 1], np.float32)
    rec.add('feat_m', hist[:, :, 4], np.float32)
    rec.add('feat_v', hist[:, :, 5], np.float32)
    rec.add('land', grid, np.float32)

    ck_json = []
    for k, r in enumerate(recs):
        t = r['t']
        adam = []
        for c in feats:
            w0, gg, m0, v0, m1, v1, w1 = cand_hist[c, t]
            step = t + 1
            wd = WD if params[cands[c][0]].dim() >= 2 else 0.0
            adam.append({'t': step, 'w0': f9(w0), 'g': f9(gg), 'm0': f9(m0), 'v0': f9(v0), 'm': f9(m1), 'v': f9(v1),
                         'mh': f9(m1 / (1 - b1 ** step)), 'vh': f9(v1 / (1 - b2 ** step)), 'w1': f9(w1), 'wd': wd,
                         'gRaw': f9(gg / r['clip'])})
        ck_json.append({
            't': t, 'lr': f9(r['lr']), 'loss': round(r['loss'], 5), 'lossAfter': round(r['lossAfter'], 5), 'valLoss': round(r['valLoss'], 5),
            'gradNorm': round(r['gradNorm'], 5), 'clip': round(r['clip'], 6), 'rowLoss': round(r['row_loss_all'], 5),
            'samples': r['samples'], 'pcaScale': round(r['pca_scale'], 6),
            'wScale': [f9(x[1]) for x in wq[k]], 'gScale': [f9(x[1]) for x in gq[k]], 'adam': adam,
        })
    meta = {
        'kind': 'tiny',
        'model': {
            'arch': 'Qwen3ForCausalLM', 'layers': NL, 'hidden': cfg.hidden_size, 'heads': H, 'kvHeads': cfg.num_key_value_heads,
            'headDim': cfg.head_dim, 'ffn': cfg.intermediate_size, 'vocab': V, 'ropeTheta': cfg.rope_parameters['rope_theta'], 'eps': cfg.rms_norm_eps,
            'tied': True, 'params': n_params, 'embedParams': n_embed, 'init': 'normal(0, 0.02)',
        },
        'train': {
            'steps': S, 'batch': B, 'seq': T, 'tokensPerStep': B * T, 'tokens': B * T * S, 'peakLr': PEAK, 'minLr': FLOOR,
            'warmup': a.warmup, 'schedule': '线性预热 + 余弦退火', 'betas': list(BETAS), 'eps': EPS, 'weightDecay': WD, 'clip': CLIP,
            'precision': 'bf16 autocast 前向 / fp32 主权重与优化器状态', 'seconds': round(train_sec, 1), 'ckptSeconds': round(t_ck, 1),
            'gpu': torch.cuda.get_device_name(0), 'finalVal': round(final_val, 4), 'finalSamples': final_samples,
            'sampleTemp': TEMP, 'adamCheckUlp': round(err, 3), 'adamCheckMomRel': merr, 'seed': a.seed,
            'epochs': round(B * T * S / len(tr_ids), 2),
        },
        'corpus': {
            'name': 'chinese-poetry（全唐诗 + 全宋诗）', 'url': 'https://github.com/chinese-poetry/chinese-poetry', 'license': 'MIT（诗歌本身属公有领域）',
            'filter': '五言 / 七言 · 绝句 / 律诗，“，。”交替，去重，丢掉含低频字的诗；OpenCC 繁转简',
            'poems': info['poems'], 'chars': info['chars'], 'kinds': info['kinds'], 'stats': info['stats'],
        },
        'vocab': vocab,
        'ckpts': ck_json,
        'valCurve': [[t, round(v, 4)] for t, v in val_curve],
        'held': {'text': info['heldOut'], 'ids': held.tolist()},
        'prefixes': PREFIXES,
        'pcaIds': pca_ids,
        'crops': [{'name': n, 'rows': CROP, 'cols': CROP} for n in crop_names],
        'tensors': tensors,
        'rowN': ROWN,
        'feats': [{'name': cands[c][0], 'index': list(cands[c][1]), 'label': cands[c][2], 'decay': params[cands[c][0]].dim() >= 2} for c in feats],
        'featStride': DS,
        'land': {'G': G, 'x': [float(xs[0]), float(xs[-1])], 'y': [float(ys[0]), float(ys[-1])], 'path': coords.tolist(),
                 'pathLoss': [round(v, 4) for v in path_loss], 'evr': [float(e) for e in evr], 'batchRows': int(lb.shape[0])},
        'bin': rec.spec,
    }
    # 拆成首屏小文件 + 按检查点的分块（见 split_data.py），写完读回来逐字节核对
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(json.dumps(meta, ensure_ascii=False))   # 和网页读到的一样（元组变列表等）
    A = split_data.arrays(meta, rec.bytes())
    split_data.report(split_data.split_tiny(meta, A, OUT))
    split_data.verify_tiny(meta, A, OUT)
    print('  核对通过')


if __name__ == '__main__':
    main()
