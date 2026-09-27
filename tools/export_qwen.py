"""用真实的 Qwen3-0.6B 跑一遍所有预设问题，把网页要展示的内部数据导出到 public/data/。

用法：
    python tools/export_qwen.py --model /mnt/d/cjc/model-weights/qwen3/Qwen3-0.6B

每个问题导出两个文件：
    qNN.json  词元、逐步的候选概率与采样过程、逻辑透镜、残差范数、单神经元 / 单次打分的真实乘加
    qNN.bin   注意力（每个头每一行的前 4 个键）、MLP 激活（每层前 24 个 + 焦点层完整 3072 维）
以及 manifest.json：模型配置、问题列表、回复文本、输入法候选（问题的真实分词）。

生成用的是 Qwen3 非思考模式的推荐参数（T=0.7, top_k=20, top_p=0.8），随机数种子固定，
所以回答和网页里演示的采样过程完全一致、可复现。
"""
import argparse
import gzip
import json
import math
import pathlib
import struct

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "data"

SYSTEM = "你是一个乐于助人的助手，请用一两句话简洁地回答。"
QUESTIONS = [
    "天空为什么是蓝色的？",
    "为什么海水是咸的？",
    "为什么猫喜欢纸箱？",
    "你是谁？",
    "你会做什么？",
    "大模型是怎么工作的？",
    "什么是注意力机制？",
    "什么是词元？",
    "1+1等于几？",
    "12乘以12等于多少？",
    "床前明月光的下一句是什么？",
    "法国的首都是哪里？",
    "讲一个冷笑话。",
    "用一句话介绍北京。",
    "写一句关于秋天的诗。",
    "What is the capital of France?",
]
TEMPERATURE, TOP_K, TOP_P = 0.7, 20, 0.8
TEMPS = [0.3, 0.7, 1.0, 1.5]  # 网页里温度滑块可选的几个档位，概率都是真实算出来的
MAX_NEW = 64
FOCUS_LAYERS = [3, 14, 25]  # 导出完整 Q/K、单神经元细节的层
FULL_MLP_LAYER = 14         # 导出完整 3072 维激活的层
ATT_TOPK, MLP_TOPK, TOPN = 4, 16, 12


# ---------------------------------------------------------------- 词元显示

def byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DEC = byte_decoder()


def token_display(tok, tid):
    """把 byte-level BPE 的词元还原成可读文字；不完整的 UTF-8 片段显示成 <0x..>。"""
    s = tok.convert_ids_to_tokens(tid)
    if s in tok.all_special_tokens or s.startswith("<|") or s in ("<think>", "</think>"):
        return s, True
    try:
        raw = bytes(BYTE_DEC[c] for c in s)
    except KeyError:
        return s, False
    try:
        return raw.decode("utf-8"), False
    except UnicodeDecodeError:
        return "".join(f"<0x{b:02X}>" for b in raw), False


# ---------------------------------------------------------------- 采样

def sample_step(logits, gen):
    """复现 transformers 的 temperature → top_k → top_p 采样，并记下每一步的中间结果。"""
    logits = logits.float()
    p1 = torch.softmax(logits, -1)
    top = torch.topk(p1, TOPN)
    temps = {}
    for T in TEMPS:
        pt = torch.softmax(logits / T, -1)
        temps[str(T)] = [round(float(pt[i]), 6) for i in top.indices.tolist()] + [round(float(1 - pt[top.indices].sum()), 6)]
    scaled = logits / TEMPERATURE
    kth = torch.topk(scaled, TOP_K).values[-1]
    scaled = torch.where(scaled < kth, torch.full_like(scaled, -float("inf")), scaled)
    sp, si = torch.sort(scaled, descending=True)
    cp = torch.softmax(sp, -1).cumsum(-1)
    remove = cp > TOP_P
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    keep = int((~remove).sum())
    final = torch.softmax(sp[:keep], -1)
    u = float(torch.rand(1, generator=gen, device=logits.device))
    idx = int(torch.searchsorted(final.cumsum(-1), torch.tensor([u], device=final.device)).clamp(max=keep - 1))
    chosen = int(si[idx])
    return chosen, {
        "top": [[int(i), round(float(p1[i]), 6)] for i in top.indices.tolist()],
        "temps": temps,
        "pool": [[int(si[j]), round(float(final[j]), 6)] for j in range(keep)],  # 经过 top_k + top_p 之后的候选池
        "topkLogits": [round(float(v), 3) for v in torch.topk(logits, TOP_K).values.tolist()],
        "u": round(u, 6),
        "chosen": chosen,
        "chosenP1": round(float(p1[chosen]), 6),
        "chosenRank": int((p1 > p1[chosen]).sum()),
    }


@torch.no_grad()
def generate(model, ids, seed):
    gen = torch.Generator(device=model.device).manual_seed(seed)
    out = model(torch.tensor([ids], device=model.device), use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1]
    reply, steps = [], []
    eos = {151645, 151643}
    for _ in range(MAX_NEW):
        chosen, info = sample_step(logits, gen)
        steps.append(info)
        reply.append(chosen)
        if chosen in eos:
            break
        out = model(torch.tensor([[chosen]], device=model.device), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
    return reply, steps


# ---------------------------------------------------------------- 完整前向 + 内部数据

def rope_theta(cfg):
    # transformers 5 把 rope_theta 挪进了 rope_parameters
    t = getattr(cfg, "rope_theta", None) if "rope_theta" in cfg.to_dict() else None
    return t or (cfg.rope_parameters or {}).get("rope_theta", 1e6)


def rms(x, w, eps):
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)) * w.float()


def rotate_half(x):
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-b, a), -1)


@torch.no_grad()
def internals(model, seq, cfg):
    m = model.model
    T = len(seq)
    H, KVH, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    eps = cfg.rms_norm_eps
    cap = {"ln1": {}, "ln2": {}, "act": {}, "out": {}, "oin": {}}
    hooks = []
    for li, layer in enumerate(m.layers):
        hooks.append(layer.input_layernorm.register_forward_hook(lambda mod, i, o, li=li: cap["ln1"].__setitem__(li, o[0].float())))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(lambda mod, i, o, li=li: cap["ln2"].__setitem__(li, o[0].float())))
        hooks.append(layer.mlp.down_proj.register_forward_hook(lambda mod, i, o, li=li: cap["act"].__setitem__(li, i[0][0].float())))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(lambda mod, i, o, li=li: cap["oin"].__setitem__(li, i[0][0].float())))
        hooks.append(layer.register_forward_hook(lambda mod, i, o, li=li: cap["out"].__setitem__(li, (o[0] if isinstance(o, tuple) else o)[0].float())))
    ids = torch.tensor([seq], device=model.device)
    out = model(ids, output_attentions=True, use_cache=False)
    for h in hooks:
        h.remove()
    att = [a[0].float() for a in out.attentions]  # 每层 [H, T, T]

    # RoPE（Qwen3：theta = 1e6）
    pos = torch.arange(T, device=model.device).float()
    inv = 1.0 / (rope_theta(cfg) ** (torch.arange(0, D, 2, device=model.device).float() / D))
    ang = torch.outer(pos, inv)
    emb = torch.cat((ang, ang), -1)
    cos, sin = emb.cos(), emb.sin()

    res = {"att": att, "act": cap["act"], "out": cap["out"], "ln1": cap["ln1"], "ln2": cap["ln2"], "oin": cap["oin"], "qk": {}, "lse": []}
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=model.device), 1)
    max_err = 0.0
    for li, layer in enumerate(m.layers):
        a = layer.self_attn
        x = cap["ln1"][li]
        q = (x @ a.q_proj.weight.float().T).view(T, H, D)
        k = (x @ a.k_proj.weight.float().T).view(T, KVH, D)
        q = rms(q, a.q_norm.weight, eps)
        k = rms(k, a.k_norm.weight, eps)
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        kk = k.repeat_interleave(H // KVH, dim=1)  # GQA：每 2 个 Q 头共用一个 KV 头
        s = torch.einsum("thd,shd->hts", q, kk) / math.sqrt(D)
        s = s.masked_fill(mask, -float("inf"))
        lse = torch.logsumexp(s, -1)  # [H, T]
        res["lse"].append(lse)
        diff = (torch.softmax(s, -1) - att[li]).abs()
        max_err = max(max_err, float(diff.max()))
        res.setdefault("mean_err", []).append(float(diff.sum() / (H * T * (T + 1) / 2)))
        if li in FOCUS_LAYERS:
            res["qk"][li] = (q, k)
    res["check"] = max_err  # 自己复现的注意力与模型输出的最大误差
    # 逻辑透镜：每层输出接上最终 RMSNorm 和（与嵌入共享的）输出矩阵
    W = model.lm_head.weight.float()
    res["lens"] = []
    for li in range(len(m.layers)):
        hN = rms(cap["out"][li], m.norm.weight, eps)
        res["lens"].append(hN)
    res["W"] = W
    return res


# ---------------------------------------------------------------- 导出

class Bin:
    def __init__(self):
        self.buf = bytearray()
        self.index = {}

    def add(self, name, arr):
        arr = np.ascontiguousarray(arr)
        while len(self.buf) % 4:
            self.buf.append(0)
        self.index[name] = {"offset": len(self.buf), "dtype": str(arr.dtype), "shape": list(arr.shape)}
        self.buf += arr.tobytes()


@torch.no_grad()
def export_question(qi, text, tok, model, cfg):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pids = tok(prompt, add_special_tokens=False).input_ids
    reply, steps = generate(model, pids, seed=1000 + qi)
    ended = reply[-1] in (151645, 151643)
    seq = pids + (reply[:-1] if ended else reply)  # 最后的 <|im_end|> 不会再被输入
    T, P = len(seq), len(pids)
    r = internals(model, seq, cfg)
    print(f"  T={T} prompt={P} reply={len(reply)} ended={ended} attn-check max={r['check']:.2e} mean={sum(r['mean_err'])/len(r['mean_err']):.2e}")

    # 角色：system / user / assistant / 模板
    toks = []
    role = None
    for i, t in enumerate(seq):
        s, special = token_display(tok, t)
        if s == "<|im_start|>":
            role = None
        elif role is None and i > 0 and toks and toks[-1]["s"] == "<|im_start|>":
            role = s.strip()
        toks.append({"id": int(t), "s": s, "sp": special, "role": role or "tpl"})
    for t in toks:
        if t["sp"]:
            t["role"] = "tpl"

    G = len(steps)  # 生成步数；第 g 步的查询位置是 P - 1 + g
    rows = [P - 1 + g for g in range(G)]
    disp = lambda i: token_display(tok, i)[0]
    for st in steps:
        st["top"] = [[i, p, disp(i)] for i, p in st["top"]]
        st["pool"] = [[i, p, disp(i)] for i, p in st["pool"]]
        st["chosenS"] = disp(st["chosen"])

    # 逻辑透镜：每个生成步、每层的前 3 名 + 真正选中的那个词元的概率
    lens = []
    for g, row in enumerate(rows):
        per = []
        for li in range(cfg.num_hidden_layers):
            logits = r["lens"][li][row] @ r["W"].T
            p = torch.softmax(logits, -1)
            tv = torch.topk(p, 3)
            per.append({"top": [[int(i), round(float(v), 4), disp(int(i))] for v, i in zip(tv.values, tv.indices)], "pc": round(float(p[steps[g]["chosen"]]), 5)})
        lens.append(per)

    # 残差范数（每层输出，每个位置），用来决定光柱亮度
    norms = [[round(float(v), 2) for v in r["out"][li].norm(dim=-1).tolist()] for li in range(cfg.num_hidden_layers)]
    emb0 = model.model.embed_tokens.weight[torch.tensor(seq, device=model.device)].float()
    emb_norm = [round(float(v), 3) for v in emb0.norm(dim=-1).tolist()]
    emb96 = emb0[:, :96].cpu().numpy()

    b = Bin()
    # 注意力：每层每头每行的前 4 个键（uint8 下标 + uint8 权重）+ logsumexp（还原真实打分 s = ln(w) + lse）
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    # 只保留生成步对应的查询行（第 g 步的查询位置是 P-1+g），舞台上只画这一行的光束
    ai = np.zeros((L, H, G, ATT_TOPK), np.uint8)
    aw = np.zeros((L, H, G, ATT_TOPK), np.uint8)
    lse = np.zeros((L, H, G), np.float16)
    for li in range(L):
        a = r["att"][li][:, rows, :]
        tv = torch.topk(a, min(ATT_TOPK, T), dim=-1)
        ai[li, :, :, : tv.indices.shape[-1]] = tv.indices.cpu().numpy().astype(np.uint8)
        aw[li, :, :, : tv.values.shape[-1]] = np.round(tv.values.cpu().numpy() * 255).astype(np.uint8)
        lse[li] = r["lse"][li][:, rows].cpu().numpy().astype(np.float16)
    b.add("attIdx", ai)
    b.add("attW", aw)
    b.add("attLse", lse)
    # MLP（SwiGLU 的 silu(gate)*up，即 down_proj 的输入）：每个生成步每层前 24 个 + 焦点层完整 3072 维
    F = cfg.intermediate_size
    mi = np.zeros((G, L, MLP_TOPK), np.uint16)
    mv = np.zeros((G, L, MLP_TOPK), np.float16)
    mcount = np.zeros((G, L), np.uint16)
    full = np.zeros((G, F), np.int8)
    fscale = np.zeros((G,), np.float32)
    for g, row in enumerate(rows):
        for li in range(L):
            a = r["act"][li][row]
            tv = torch.topk(a.abs(), MLP_TOPK)
            mi[g, li] = tv.indices.cpu().numpy()
            mv[g, li] = a[tv.indices].cpu().numpy().astype(np.float16)
            mcount[g, li] = int((a.abs() > 0.1 * a.abs().max()).sum())
        a = r["act"][FULL_MLP_LAYER][row].cpu().numpy()
        sc = float(np.abs(a).max()) / 127 or 1.0
        fscale[g] = sc
        full[g] = np.round(a / sc).astype(np.int8)
    b.add("mlpIdx", mi)
    b.add("mlpVal", mv)
    b.add("mlpCount", mcount)
    b.add("mlpFull", full)
    b.add("mlpFullScale", fscale)

    # 单神经元（SwiGLU）与单次打分（Q·K）的真实乘加，只在焦点层导出
    neuron, dot = {}, {}
    for li in FOCUS_LAYERS:
        layer = model.model.layers[li]
        Wg, Wu = layer.mlp.gate_proj.weight.float(), layer.mlp.up_proj.weight.float()
        q, k = r["qk"][li]
        nlist, dlist = [], []
        for g, row in enumerate(rows):
            a = r["act"][li][row]
            n = int(torch.argmax(a))
            x = r["ln2"][li][row]
            wg, wu = Wg[n], Wu[n]
            contrib = (x * wg).abs()
            top = torch.topk(contrib, TOPN).indices.tolist()
            gz, uz = float(x @ wg), float(x @ wu)
            nlist.append({
                "n": n,
                "dims": top,
                "x": [round(float(x[i]), 5) for i in top],
                "wg": [float(wg[i]) for i in top],  # bf16 权重，转成 float 后是精确值
                "wu": [float(wu[i]) for i in top],
                "gz": round(gz, 5), "uz": round(uz, 5),
                "silu": round(gz / (1 + math.exp(-gz)), 5),
                "act": round(float(a[n]), 5),
                "xnorm": round(float(x.norm()), 3),
            })
            # 这一行里注意力最集中在“非开头”位置的头
            w = r["att"][li][:, row, :]
            w2 = w.clone()
            w2[:, 0] = 0
            w2[:, row] = 0  # 不选“看自己”和“看开头”，挑一个真正在看别处的头
            h = int(torch.argmax(w2.max(-1).values))
            key = int(torch.argmax(w2[h]))
            kv = h // (H // cfg.num_key_value_heads)
            qv, kvv = q[row, h], k[key, kv]
            prod = qv * kvv
            top = torch.topk(prod.abs(), TOPN).indices.tolist()
            dlist.append({
                "head": h, "kv": kv, "key": key,
                "dims": top,
                "q": [round(float(qv[i]), 4) for i in top],
                "k": [round(float(kvv[i]), 4) for i in top],
                "sum": round(float(prod.sum()), 4),
                "score": round(float(prod.sum()) / math.sqrt(cfg.head_dim), 4),
                "w": round(float(w[h, key]), 5),
            })
        neuron[li], dot[li] = nlist, dlist

    # 矩阵乘法的“一个输出元素”：y[j] = Σ_i x[i]·W[j, i]，导出贡献最大的 12 项和总和
    def mm_entry(x, w, total, j):
        c = x * w
        top = torch.topk(c.abs(), TOPN).indices.tolist()
        return {"j": j, "dims": top, "x": [round(float(x[i]), 5) for i in top], "w": [float(w[i]) for i in top],
                "total": round(float(total), 5), "shown": round(float(c[top].sum()), 5)}

    mm = {}
    theta_base = rope_theta(cfg)
    Dh = cfg.head_dim
    for li in FOCUS_LAYERS:
        layer = model.model.layers[li]
        a = layer.self_attn
        Wq, Wo, Wd = a.q_proj.weight.float(), a.o_proj.weight.float(), layer.mlp.down_proj.weight.float()
        qfull, _ = r["qk"][li]
        per = []
        for g, row in enumerate(rows):
            x = r["ln1"][li][row]
            h = dot[li][g]["head"]
            qraw = (x @ Wq[h * Dh:(h + 1) * Dh].T)          # 这个头的 128 维（投影后、归一化和旋转之前）
            d = int(torch.argmax(qraw.abs()))
            j = h * Dh + d
            eq = mm_entry(x, Wq[j], qraw[d], j)
            qn = rms(qraw, a.q_norm.weight, cfg.rms_norm_eps)
            partner = d + Dh // 2 if d < Dh // 2 else d - Dh // 2
            k = d % (Dh // 2)
            ang = row / (theta_base ** (2 * k / Dh))
            eq.update({"head": h, "dim": d, "partner": partner, "pos": row, "angle": round(ang, 6), "freq": k,
                       "qn": round(float(qn[d]), 5), "qnP": round(float(qn[partner]), 5), "qnW": float(a.q_norm.weight[d]),
                       "rms": round(float(qraw.pow(2).mean().sqrt()), 5),
                       "qr": round(float(qfull[row, h, d]), 5), "qrP": round(float(qfull[row, h, partner]), 5)})
            oin = r["oin"][li][row]
            oout = oin @ Wo.T
            jo = int(torch.argmax(oout.abs()))
            act = r["act"][li][row]
            dout = act @ Wd.T
            jd = int(torch.argmax(dout.abs()))
            per.append({"q": eq, "o": mm_entry(oin, Wo[jo], oout[jo], jo), "down": mm_entry(act, Wd[jd], dout[jd], jd)})
        mm[li] = per
    # 输出头：被选中的词元的分数 = 最终向量 · 它在嵌入表里的那一行
    E = model.lm_head.weight.float()
    last = cfg.num_hidden_layers - 1
    head_mm = []
    for g, row in enumerate(rows):
        hN = rms(r["out"][last][row], model.model.norm.weight, cfg.rms_norm_eps)
        cid = steps[g]["chosen"]
        ent = mm_entry(hN, E[cid], hN @ E[cid], cid)
        ent["token"] = disp(cid)
        head_mm.append(ent)

    meta = {
        "id": f"q{qi + 1:02d}",
        "question": text,
        "T": T, "P": P, "G": G, "ended": ended,
        "tokens": toks,
        "steps": steps,
        "lens": lens,
        "norms": norms,
        "embNorm": emb_norm,
        "neuron": {str(k): v for k, v in neuron.items()},
        "mm": {str(k): v for k, v in mm.items()},
        "headMM": head_mm,
        "dot": {str(k): v for k, v in dot.items()},
        "bin": b.index,
        "attCheck": r["check"],
    }
    raw_json = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    for name, data in ((f"{meta['id']}.bin", bytes(b.buf)), (f"{meta['id']}.json", raw_json)):
        (OUT / name).write_bytes(data)
        (OUT / (name + ".gz")).write_bytes(gzip.compress(data, 9, mtime=0))  # 网页优先读 .gz，用 DecompressionStream 解压
    reply_text = tok.decode(reply[:-1] if ended else reply)
    q_ids = tok(text, add_special_tokens=False).input_ids
    return {
        "id": meta["id"],
        "text": text,
        "chips": [{"id": int(i), "s": token_display(tok, i)[0]} for i in q_ids],
        "reply": reply_text,
        "replyTokens": [{"id": int(i), "s": token_display(tok, i)[0]} for i in (reply[:-1] if ended else reply)],
        "ended": ended,
        "bytes": len(gzip.compress(bytes(b.buf), 9)) + len(gzip.compress(raw_json, 9)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--only", type=int, default=None, help="只导出第几个问题（从 1 开始），调试用")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    cfg = model.config
    items = []
    for qi, q in enumerate(QUESTIONS):
        if args.only and qi + 1 != args.only:
            continue
        print(f"[{qi + 1}/{len(QUESTIONS)}] {q}")
        items.append(export_question(qi, q, tok, model, cfg))
        print("  →", items[-1]["reply"])
    n_params = sum(p.numel() for p in model.parameters())
    sample_w = model.model.layers[FOCUS_LAYERS[1]].mlp.gate_proj.weight
    thumbs = bytearray()
    thumb_index = {}
    B = 32
    for li, layer in enumerate(model.model.layers):
        a, mlp = layer.self_attn, layer.mlp
        for name, W in (("q", a.q_proj), ("k", a.k_proj), ("v", a.v_proj), ("o", a.o_proj), ("gate", mlp.gate_proj), ("up", mlp.up_proj), ("down", mlp.down_proj)):
            w = W.weight.float()                      # [out, in]
            o, i = w.shape
            blk = w.pow(2).reshape(o // B, B, i // B, B).mean(dim=(1, 3)).sqrt()   # [out/B, in/B]
            v = blk / blk.max()
            img = (v.T.flip(0) * 255).round().to(torch.uint8).cpu().numpy()        # 行 = 输入（上下翻转，0 在底部），列 = 输出
            thumb_index[f"{li}:{name}"] = {"offset": len(thumbs), "w": img.shape[1], "h": img.shape[0]}
            thumbs += img.tobytes()
    (OUT / "weights.bin").write_bytes(bytes(thumbs))
    (OUT / "weights.bin.gz").write_bytes(gzip.compress(bytes(thumbs), 9, mtime=0))
    manifest = {
        "model": {
            "name": "Qwen3-0.6B",
            "source": "https://huggingface.co/Qwen/Qwen3-0.6B",
            "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads,
            "kvHeads": cfg.num_key_value_heads, "headDim": cfg.head_dim, "ffn": cfg.intermediate_size,
            "vocab": cfg.vocab_size, "ropeTheta": rope_theta(cfg), "eps": cfg.rms_norm_eps,
            "tied": cfg.tie_word_embeddings, "params": n_params, "dtype": "bfloat16",
            "paramsPerLayer": sum(p.numel() for p in model.model.layers[0].parameters()),
            "embedParams": model.model.embed_tokens.weight.numel(),
        },
        "sampling": {"temperature": TEMPERATURE, "top_k": TOP_K, "top_p": TOP_P, "temps": TEMPS},
        "system": SYSTEM,
        "focusLayers": FOCUS_LAYERS,
        "fullMlpLayer": FULL_MLP_LAYER,
        "attTopk": ATT_TOPK,
        "mlpTopk": MLP_TOPK,
        "questions": items,
        "weightSample": [float(v) for v in sample_w[0, :8]],
        "thumbs": {"block": B, "index": thumb_index},
    }
    if args.only:
        print(json.dumps(items, ensure_ascii=False, indent=1)[:3000])
        return
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("total bytes:", sum(i["bytes"] for i in items))


if __name__ == "__main__":
    main()
