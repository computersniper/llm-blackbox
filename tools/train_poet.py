"""从零训练一个 Qwen3 同架构的小模型写五言诗，把训练过程录下来给 training/ 页面。

用法（conda 环境 xnewenv）：
    python tools/train_poet.py --corpus D:/models/corpus

语料：chinese-poetry 的全唐诗（poet.tang.*.json），转成简体，只保留五言绝句 / 五言律诗。
模型：transformers 的 Qwen3ForCausalLM，4 层 · 隐藏 256 · 4 个查询头 / 2 个键值头 · SwiGLU 768，
      按字切分（每个汉字一个词元），和推理页的 Qwen3-0.6B 是同一套结构，只是小得多。

导出到 public/training/data/：
    run.json          配置、词表、每一步的损失 / 学习率 / 梯度范数、验证损失、
                      检查点上的生成样本、“静夜思”逐字预测、字向量的二维投影
    step_NNNN.json    31 个录制步：这一步真实的一批数据里第 0 首诗的完整前向（逐层残差、
                      注意力、神经元、逻辑透镜、逐字损失）、反向（每层每个位置的梯度范数、
                      每个参数矩阵的梯度 / 更新量）以及一个具体权重的 AdamW 更新全过程
每个 json 另存一份 .gz，网页用 DecompressionStream 解压。
"""
import argparse
import glob
import gzip
import json
import math
import pathlib
import random
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "training" / "data"

STEPS = 3000
BATCH = 64
LR, WARMUP, MIN_LR = 3e-3, 100, 3e-4
BETAS, EPS, WD = (0.9, 0.95), 1e-8, 0.1
CLIP = 1.0
SEED = 20260930
DETAIL = [1] + list(range(100, STEPS + 1, 100))  # 录制完整细节的步
VAL_EVERY = 50
PROBE = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
PROMPTS = ["春", "月", "山", "秋风"]
WATCH_CHAR, PCA_CHARS = "月", 240
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]


# ---------------------------------------------------------------- 语料

def load_poems(corpus):
    import opencc
    cc = opencc.OpenCC("t2s")
    poems = []
    for f in sorted(glob.glob(str(pathlib.Path(corpus) / "tang.*.json"))):
        for p in json.load(open(f, encoding="utf-8")):
            text = cc.convert("".join(p.get("paragraphs", [])))
            lines = [s for s in text.replace("。", "。|").replace("，", "，|").split("|") if s]
            if len(lines) not in (4, 8):
                continue
            if not all(len(s) == 6 and s[-1] in "，。" for s in lines):
                continue
            if not all("\u4e00" <= ch <= "\u9fff" for s in lines for ch in s[:-1]):
                continue
            if not all(lines[i][-1] == ("，" if i % 2 == 0 else "。") for i in range(len(lines))):
                continue
            poems.append({"text": text, "title": cc.convert(p.get("title", "")), "author": cc.convert(p.get("author", ""))})
    return poems


def build_vocab(poems, min_freq=2):
    freq = {}
    for p in poems:
        for ch in p["text"]:
            freq[ch] = freq.get(ch, 0) + 1
    chars = sorted([c for c, n in freq.items() if n >= min_freq], key=lambda c: (-freq[c], c))
    itos = SPECIALS + chars
    return itos, {c: i for i, c in enumerate(itos)}, freq


def encode(text, stoi):
    return [BOS] + [stoi.get(ch, UNK) for ch in text] + [EOS]


# ---------------------------------------------------------------- 学习率

def lr_at(step):
    if step <= WARMUP:
        return LR * step / WARMUP
    t = (step - WARMUP) / (STEPS - WARMUP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * t))


# ---------------------------------------------------------------- 工具

r4 = lambda x: float(f"{float(x):.4g}")


def bits32(x):
    return format(struct.unpack("<I", struct.pack("<f", float(x)))[0], "032b")


def write_json(path, obj):
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    path.with_suffix(path.suffix + ".gz").write_bytes(gzip.compress(raw, 9))
    return len(raw)


@torch.no_grad()
def sample(model, stoi, itos, prompt, seed, temp=0.8, top_k=20, greedy=False):
    dev = next(model.parameters()).device
    ids = [BOS] + [stoi.get(c, UNK) for c in prompt]
    g = torch.Generator(device=dev).manual_seed(seed)
    steps = []
    while len(ids) < 50:
        logits = model(torch.tensor([ids], device=dev)).logits[0, -1].float()
        p = torch.softmax(logits, -1)
        top = torch.topk(p, 5)
        if greedy:
            nxt = int(top.indices[0])
        else:
            lg = logits / temp
            kth = torch.topk(lg, top_k).values[-1]
            lg[lg < kth] = -float("inf")
            nxt = int(torch.multinomial(torch.softmax(lg, -1), 1, generator=g))
        steps.append({"c": itos[nxt], "p": r4(p[nxt]), "top": [[itos[int(i)], r4(v)] for v, i in zip(top.values, top.indices)]})
        if nxt == EOS:
            break
        ids.append(nxt)
    text = "".join(itos[i] for i in ids[1:] if i > 3)
    return {"prompt": prompt, "text": text, "steps": steps}


@torch.no_grad()
def probe(model, stoi, itos):
    dev = next(model.parameters()).device
    ids = encode(PROBE, stoi)
    p = torch.softmax(model(torch.tensor([ids], device=dev)).logits[0].float(), -1)
    out = []
    for t in range(len(ids) - 1):
        tgt = ids[t + 1]
        top = torch.topk(p[t], 3)
        out.append({"p": r4(p[t, tgt]), "top": [[itos[int(i)], r4(v)] for v, i in zip(top.values, top.indices)]})
    return out


@torch.no_grad()
def val_loss(model, val):
    tot, n = 0.0, 0
    for i in range(0, len(val), 128):
        x = val[i:i + 128]
        logits = model(x[:, :-1]).logits
        l = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1), ignore_index=PAD, reduction="sum")
        tot += float(l)
        n += int((x[:, 1:] != PAD).sum())
    return tot / n


# ---------------------------------------------------------------- 录制一步

class Recorder:
    """用前向钩子抓每层的中间量，并对它们 retain_grad，反向之后读梯度。"""

    def __init__(self, model):
        self.model = model
        self.h = []
        self.on = False
        m = model.model
        self.h.append(m.embed_tokens.register_forward_hook(self.hook("emb")))
        for li, layer in enumerate(m.layers):
            self.h.append(layer.register_forward_pre_hook(self.pre(f"in{li}"), with_kwargs=True))
            self.h.append(layer.self_attn.register_forward_hook(self.hook(f"attn{li}", tup=True)))
            self.h.append(layer.mlp.register_forward_hook(self.hook(f"mlp{li}")))
            self.h.append(layer.mlp.down_proj.register_forward_pre_hook(self.pre(f"act{li}"), with_kwargs=True))
            self.h.append(layer.register_forward_hook(self.hook(f"out{li}", tup=True)))
        self.t = {}

    def pre(self, name):
        def f(mod, args, kwargs):
            if self.on:
                self.t[name] = args[0] if args else kwargs.get("hidden_states", kwargs.get("input"))
        return f

    def hook(self, name, tup=False):
        def f(mod, args, out):
            if not self.on:
                return
            x = out[0] if (tup and isinstance(out, tuple)) else out
            if name.startswith("attn") and isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                self.t["w" + name] = out[1]
            if x.requires_grad:
                x.retain_grad()
            self.t[name] = x
        return f


def record_step(step, model, rec, batch, loss, logits, stoi, itos, lr, gnorm, clip_coef, params_before, watch, opt, cfg):
    """batch[0] 这首诗在这一步里的一切。在 backward 之后、optimizer.step 之前调用；更新后的值由调用方补上。"""
    x = batch[0]
    T = int((x != PAD).sum()) - 1  # 输入长度（去掉最后一个目标）
    ids = x[:T + 1].tolist()
    t = rec.t
    NL = cfg.num_hidden_layers
    H = cfg.num_attention_heads
    E = model.model.embed_tokens.weight
    norm = model.model.norm

    def rows(v):  # 第 0 个样本，前 T 个位置
        return v[0, :T].detach().float()

    fwd = {"res": [], "attn": [], "mlp": [], "lens": [], "attnOut": [], "mlpOut": []}
    fwd["res"].append([r4(v) for v in rows(t["emb"]).norm(dim=-1)])
    for li in range(NL):
        h_in = rows(t[f"in{li}"])
        a = rows(t[f"attn{li}"])
        mid = h_in + a
        o = rows(t[f"out{li}"])
        fwd["res"].append([r4(v) for v in o.norm(dim=-1)])
        fwd["attnOut"].append([r4(v) for v in a.norm(dim=-1)])
        fwd["mlpOut"].append([r4(v) for v in rows(t[f"mlp{li}"]).norm(dim=-1)])
        fwd.setdefault("mid", []).append([r4(v) for v in mid.norm(dim=-1)])
        w = t[f"wattn{li}"][0, :, :T, :T].detach().float()  # [H, T, T]
        fwd["attn"].append([[[int(round(float(v) * 255)) for v in w[hh, i, : i + 1]] for i in range(T)] for hh in range(H)])
        act = t[f"act{li}"][0, :T].detach().float()  # [T, 768]
        top = torch.topk(act.abs(), 8, dim=-1)
        fwd["mlp"].append([[[int(j), r4(act[i, j])] for j in top.indices[i]] for i in range(T)])
        with torch.no_grad():
            lp = torch.softmax((norm(o.unsqueeze(0))[0] @ E.T).float(), -1)
            tv, ti = lp.max(-1)
        fwd["lens"].append([[itos[int(ti[i])], r4(tv[i])] for i in range(T)])

    p = torch.softmax(logits[0, :T].detach().float(), -1)
    tgt = torch.tensor(ids[1:], device=p.device)
    pt = p[torch.arange(T), tgt]
    top = torch.topk(p, 5, dim=-1)
    out = {
        "pTarget": [r4(v) for v in pt],
        "lossPos": [r4(-math.log(max(float(v), 1e-12))) for v in pt],
        "top": [[[itos[int(j)], r4(v)] for v, j in zip(top.values[i], top.indices[i])] for i in range(T)],
        "logitNorm": [r4(v) for v in logits[0, :T].detach().float().norm(dim=-1)],
    }

    # 反向：残差流上每个位置的梯度范数（从输出往回）
    bwd = {"res": [], "attnOut": [], "mlpOut": []}
    bwd["res"].append([r4(v) for v in t["emb"].grad[0, :T].float().norm(dim=-1)])
    for li in range(NL):
        bwd["res"].append([r4(v) for v in t[f"out{li}"].grad[0, :T].float().norm(dim=-1)])
        bwd["attnOut"].append([r4(v) for v in t[f"attn{li}"].grad[0, :T].float().norm(dim=-1)])
        bwd["mlpOut"].append([r4(v) for v in t[f"mlp{li}"].grad[0, :T].float().norm(dim=-1)])

    # 每个参数矩阵：形状、权重范数、梯度范数；更新量由调用方在 step 之后补
    params = []
    for name, prm in model.named_parameters():
        if prm.grad is None:
            continue
        params.append({"name": name.replace("model.", ""), "shape": list(prm.shape), "w": r4(prm.detach().norm()), "g": r4(prm.grad.detach().norm())})

    # 被盯住的那个权重（“月”字嵌入的某一维；输出头和嵌入共享权重，所以每一步都有梯度）
    r, c = watch
    st = opt.state.get(E, {})
    m0 = float(st["exp_avg"][r, c]) if st else 0.0
    v0 = float(st["exp_avg_sq"][r, c]) if st else 0.0
    w0 = float(E.detach()[r, c])
    g_raw = float(E.grad[r, c]) / clip_coef if clip_coef > 0 else float(E.grad[r, c])
    adam = {"char": itos[r], "row": r, "col": c, "w": w0, "gRaw": g_raw, "g": float(E.grad[r, c]), "m0": m0, "v0": v0,
            "lr": lr, "beta1": BETAS[0], "beta2": BETAS[1], "eps": EPS, "wd": WD, "t": step, "bitsBefore": bits32(w0),
            "row8": [r4(v) for v in E.detach()[r, :8]], "grow8": [r4(v) for v in E.grad[r, :8]]}

    return {
        "step": step, "lr": r4(lr), "loss": r4(loss), "gradNorm": r4(gnorm), "clip": r4(clip_coef),
        "batch": BATCH, "tokensInBatch": int((batch[:, 1:] != PAD).sum()),
        "others": ["".join(itos[i] for i in row.tolist() if i > 3) for row in batch[1:6]],
        "tokens": [itos[i] for i in ids], "ids": ids, "T": T,
        "fwd": fwd, "out": out, "bwd": bwd, "params": params, "adam": adam,
    }


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="D:/models/corpus")
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    poems = load_poems(args.corpus)
    poems = [p for p in poems if p["text"] != PROBE]
    random.shuffle(poems)
    itos, stoi, freq = build_vocab(poems)
    print(f"{len(poems)} 首五言诗，词表 {len(itos)}", flush=True)
    val_p, train_p = poems[:512], poems[512:]
    maxlen = 50

    def tensor(ps):
        x = torch.full((len(ps), maxlen), PAD, dtype=torch.long)
        for i, p in enumerate(ps):
            e = encode(p["text"], stoi)
            x[i, :len(e)] = torch.tensor(e)
        return x.to(dev)

    train, val = tensor(train_p), tensor(val_p)

    cfg = Qwen3Config(vocab_size=len(itos), hidden_size=256, intermediate_size=768, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=64, max_position_embeddings=64,
                      rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True,
                      bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD)
    cfg._attn_implementation = "eager"
    model = Qwen3ForCausalLM(cfg).to(dev)
    model.train()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"参数量 {nparams:,}", flush=True)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=LR, betas=BETAS, eps=EPS)
    rec = Recorder(model)
    E = model.model.embed_tokens.weight
    watch = None

    # 字向量二维投影：用训练结束时的主成分作为固定坐标轴，所以能看出每个字怎么从一团乱麻里走到自己的位置
    pca_ids = list(range(4, 4 + PCA_CHARS))
    emb_snaps = {}

    log = {"loss": [], "lr": [], "gnorm": []}
    vals, samples, probes = [], {}, {}
    order = torch.randperm(len(train), device=dev)
    cursor = 0

    def snapshot(step):
        model.eval()
        samples[step] = [sample(model, stoi, itos, pr, seed=SEED + k) for k, pr in enumerate(PROMPTS)] + \
                         [sample(model, stoi, itos, PROMPTS[0], seed=0, greedy=True)]
        probes[step] = probe(model, stoi, itos)
        emb_snaps[step] = E.detach()[pca_ids].float().cpu().numpy().copy()
        model.train()

    snapshot(0)
    vals.append([0, r4(val_loss(model, val))])
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if cursor + BATCH > len(train):
            order = torch.randperm(len(train), device=dev)
            cursor = 0
        batch = train[order[cursor:cursor + BATCH]]
        cursor += BATCH
        lr = lr_at(step)
        for gp in opt.param_groups:
            gp["lr"] = lr
        detail = step in DETAIL
        rec.on = detail
        rec.t = {}
        x, y = batch[:, :-1], batch[:, 1:]
        out = model(x, output_attentions=detail)
        logits = out.logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), ignore_index=PAD)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP))
        clip_coef = min(1.0, CLIP / (gnorm + 1e-6))
        if detail:
            if watch is None:
                r = stoi[WATCH_CHAR]
                watch = (r, int(E.grad[r].abs().argmax()))
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            d = record_step(step, model, rec, batch, float(loss), logits, stoi, itos, lr, gnorm, clip_coef, before, watch, opt, cfg)
        opt.step()
        if detail:
            r, c = watch
            st = opt.state[E]
            w1 = float(E.detach()[r, c])
            d["adam"].update({"m1": float(st["exp_avg"][r, c]), "v1": float(st["exp_avg_sq"][r, c]), "w1": w1, "bitsAfter": bits32(w1)})
            byname = dict(model.named_parameters())
            for prm in d["params"]:
                full = "model." + prm["name"] if ("model." + prm["name"]) in byname else prm["name"]
                prm["upd"] = r4((byname[full].detach() - before[full]).norm())
            write_json(OUT / f"step_{step:04d}.json", d)
            del before
        log["loss"].append(r4(loss))
        log["lr"].append(r4(lr))
        log["gnorm"].append(r4(gnorm))
        if step % VAL_EVERY == 0:
            model.eval()
            vals.append([step, r4(val_loss(model, val))])
            model.train()
        if detail:
            snapshot(step)
        if step % 100 == 0:
            print(f"step {step} loss {float(loss):.3f} val {vals[-1][1]:.3f} lr {lr:.2e} gn {gnorm:.2f} {time.time() - t0:.0f}s  样本：{samples[step][0]['text']}", flush=True)

    # 主成分
    final = emb_snaps[max(emb_snaps)]
    mu = final.mean(0)
    _, _, vt = np.linalg.svd(final - mu, full_matrices=False)
    axes = vt[:2]
    pca = {str(s): [[r4(a), r4(b)] for a, b in ((e - mu) @ axes.T)] for s, e in emb_snaps.items()}

    run = {
        "model": {"arch": "Qwen3ForCausalLM", "layers": 4, "hidden": 256, "heads": 4, "kvHeads": 2, "headDim": 64,
                  "ffn": 768, "vocab": len(itos), "params": nparams, "tied": True, "dtype": "float32", "ropeTheta": 10000.0},
        "train": {"steps": args.steps, "batch": BATCH, "lr": LR, "minLr": MIN_LR, "warmup": WARMUP, "betas": list(BETAS),
                  "eps": EPS, "wd": WD, "clip": CLIP, "seed": SEED, "optimizer": "AdamW", "schedule": "warmup + cosine"},
        "data": {"poems": len(train_p), "val": len(val_p), "source": "chinese-poetry 全唐诗（五言绝句 / 律诗，繁转简）",
                 "examples": [p["text"] for p in train_p[:6]]},
        "itos": itos, "detail": DETAIL, "valEvery": VAL_EVERY,
        "log": log, "val": vals,
        "samples": {str(k): v for k, v in samples.items()},
        "probe": {"text": PROBE, "tokens": [itos[i] for i in encode(PROBE, stoi)], "by": {str(k): v for k, v in probes.items()}},
        "pca": {"chars": [itos[i] for i in pca_ids], "freq": [freq.get(itos[i], 0) for i in pca_ids], "by": pca},
        "watch": {"char": WATCH_CHAR, "row": watch[0], "col": watch[1]},
        "elapsed": round(time.time() - t0, 1),
    }
    n = write_json(OUT / "run.json", run)
    print(f"完成：run.json {n / 1e6:.2f} MB，用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
