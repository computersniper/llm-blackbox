"""用真实的 Qwen3-0.6B 跑一遍所有预设问题，把网页要展示的内部数据导出到 public/data/。

用法：
    python tools/export_qwen.py --model /mnt/d/cjc/model-weights/qwen3/Qwen3-0.6B             # 中文问题 → public/data/
    python tools/export_qwen.py --model /mnt/d/cjc/model-weights/qwen3/Qwen3-0.6B --lang en   # 英文问题 → public/data/en/

英文版（--lang en）用英文系统提示和英文问题集，采样参数和种子规则（第 i 个问题用 1000 + i）与中文完全一样，
文件名是 eNN.json / eNN.bin / eNN/Lxx.json.gz；权重缩略图两种语言共用 public/data/weights.bin，英文目录里不另存。
英文的 --only / --from 会把这次导出的问题并进已有的 manifest（中文的 --only 仍然只打印、不写 manifest）。

每个问题导出：
    qNN.json        词元、逐步的候选概率与采样过程、逻辑透镜、残差范数、输出头 logit 的真实乘加
    qNN.bin         注意力（每个头每一行的前 4 个键）、MLP 激活（每层前 16 个 + 第 14 层完整 3072 维）
    qNN/Lxx.json.gz 第 xx 层、每个生成词元的真实乘加（单神经元、单次 Q·K 打分、W_q / W_o / W_down 各一个输出元素）。
                    28 层全都有，按层拆成小文件，网页进入“一次乘加”时才按需载入；只存 .gz，不存原始文件
以及 manifest.json：模型配置、问题列表、回复文本、输入法候选（问题的真实分词）。
主文件额外存一份 .gz（服务器只压缩 HTML），网页优先读 .gz，不支持解压时退回原始文件。

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

SYSTEM_ZH = "你是一个乐于助人的助手，请用一两句话简洁地回答。"
QUESTIONS_ZH = [
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
# 英文版：常识、科学、数学、写作各类都有；第一个固定是 “Why is the sky blue?”
SYSTEM_EN = "You are a helpful assistant. Please answer concisely in one or two sentences."
QUESTIONS_EN = [
    "Why is the sky blue?",
    "Why is seawater salty?",
    "Why do cats like boxes?",
    "Who are you?",
    "How many r's are in the word strawberry?",   # 数字母：模型看到的是词元，不是一个个字母
    "How do large language models work?",
    "What is the attention mechanism?",
    "What is a token?",
    "What is 1+1?",
    "What is 12 times 12?",
    "What is the capital of France?",
    "Tell me a joke.",
    "Describe London in one sentence.",
    "Write a one-line poem about autumn.",
]
LANGS = {
    "zh": {"out": OUT, "system": SYSTEM_ZH, "questions": QUESTIONS_ZH, "prefix": "q"},
    "en": {"out": OUT / "en", "system": SYSTEM_EN, "questions": QUESTIONS_EN, "prefix": "e"},
}
SYSTEM, QUESTIONS, PREFIX = SYSTEM_ZH, QUESTIONS_ZH, "q"   # main() 按 --lang 改写这几个全局量
TEMPERATURE, TOP_K, TOP_P = 0.7, 20, 0.8
TEMPS = [0.3, 0.7, 1.0, 1.5]  # 网页里温度滑块可选的几个档位，概率都是真实算出来的
MAX_NEW = 64
FULL_MLP_LAYER = 14         # 导出完整 3072 维激活的层
ATT_TOPK, MLP_TOPK, TOPN = 4, 16, 12
CUM_RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]  # 累加曲线的采样名次（再补上 n）


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

def sig(v, d=5):
    """保留 d 位有效数字（控制 JSON 体积）。"""
    return float(f"{float(v):.{d}g}")


def summary(c):
    """一次点积里全部 n 个乘积的汇总：正乘积之和、负乘积之和，
    以及按 |乘积| 从大到小排好后的累计和，在名次 1, 2, 4, …, 2048（不超过 n）和 n 处采样。"""
    c = c.double()
    n = c.numel()
    run = c[torch.argsort(c.abs(), descending=True)].cumsum(0)
    ranks = [r for r in CUM_RANKS if r <= n]
    if ranks[-1] != n:
        ranks.append(n)
    vals = run[torch.tensor([r - 1 for r in ranks], device=c.device)].tolist()
    return {"pos": sig(c.clamp(min=0).sum()), "neg": sig(c.clamp(max=0).sum()), "cum": [[r, sig(v)] for r, v in zip(ranks, vals)]}


def write_data(path, data, raw=True):
    """写一个数据文件的 .gz（网页用 DecompressionStream 解压）；raw=True 时再存一份原始文件给不支持解压的浏览器。
    返回 .gz 的字节数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    gz = gzip.compress(data, 9, mtime=0)
    if raw:
        path.write_bytes(data)
    elif path.exists():
        path.unlink()  # 旧版本导出的原始文件
    path.with_name(path.name + ".gz").write_bytes(gz)
    return len(gz)


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

    # 矩阵乘法的“一个输出元素”：y[j] = Σ_i x[i]·W[j, i]，导出贡献最大的 12 项、总和，以及全部 n 项的汇总
    def mm_entry(x, w, total, j):
        c = x * w
        top = torch.topk(c.abs(), TOPN).indices.tolist()
        return {"j": j, "dims": top, "x": [round(float(x[i]), 5) for i in top], "w": [float(w[i]) for i in top],
                "total": round(float(total), 5), "shown": round(float(c[top].sum()), 5), "n": c.numel(), **summary(c)}

    # 每一层、每个生成词元的真实乘加：单神经元（SwiGLU）、单次打分（Q·K）、W_q / W_o / W_down 各一个输出元素。
    # 28 层全部导出，每层一个小文件 qNN/Lxx.json，网页按需载入
    qid = f"{PREFIX}{qi + 1:02d}"
    theta_base = rope_theta(cfg)
    Dh = cfg.head_dim
    micro_bytes = 0
    for li in range(L):
        layer = model.model.layers[li]
        a = layer.self_attn
        Wg, Wu = layer.mlp.gate_proj.weight.float(), layer.mlp.up_proj.weight.float()
        Wq, Wo, Wd = a.q_proj.weight.float(), a.o_proj.weight.float(), layer.mlp.down_proj.weight.float()
        q, k = r["qk"][li]
        nlist, dlist, mlist = [], [], []
        for g, row in enumerate(rows):
            # 单神经元：这一步激活最大的那个
            act = r["act"][li][row]
            n = int(torch.argmax(act))
            x = r["ln2"][li][row]
            wg, wu = Wg[n], Wu[n]
            contrib = (x * wg).abs()
            top = torch.topk(contrib, TOPN).indices.tolist()
            gz, uz = float(x @ wg), float(x @ wu)
            nlist.append({
                "j": n,                              # 神经元编号
                "n": x.numel(),                      # 每个点积的项数（1024）
                "dims": top,
                "x": [round(float(x[i]), 5) for i in top],
                "wg": [float(wg[i]) for i in top],  # bf16 权重，转成 float 后是精确值
                "wu": [float(wu[i]) for i in top],
                "gz": round(gz, 5), "uz": round(uz, 5),
                "silu": round(gz / (1 + math.exp(-gz)), 5),
                "act": round(float(act[n]), 5),
                "xnorm": round(float(x.norm()), 3),
                "g": summary(x * wg),                # gate：x·w_gate 的 1024 个乘积
                "u": summary(x * wu),                # up：x·w_up 的 1024 个乘积
            })
            # 单次打分：这一行里注意力最集中在“非开头”位置的头
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
                "n": prod.numel(), **summary(prod),  # 128 项
            })
            # W_q 的一个输出元素：同一个头里绝对值最大的那一维，并跟着它做 q_norm 和 RoPE
            x = r["ln1"][li][row]
            qraw = (x @ Wq[h * Dh:(h + 1) * Dh].T)          # 这个头的 128 维（投影后、归一化和旋转之前）
            d = int(torch.argmax(qraw.abs()))
            j = h * Dh + d
            eq = mm_entry(x, Wq[j], qraw[d], j)
            qn = rms(qraw, a.q_norm.weight, cfg.rms_norm_eps)
            partner = d + Dh // 2 if d < Dh // 2 else d - Dh // 2
            fk = d % (Dh // 2)
            ang = row / (theta_base ** (2 * fk / Dh))
            eq.update({"head": h, "dim": d, "partner": partner, "row": row, "angle": round(ang, 6), "freq": fk,
                       "qn": round(float(qn[d]), 5), "qnP": round(float(qn[partner]), 5), "qnW": float(a.q_norm.weight[d]),
                       "rms": round(float(qraw.pow(2).mean().sqrt()), 5),
                       "qr": round(float(q[row, h, d]), 5), "qrP": round(float(q[row, h, partner]), 5)})
            # W_o、W_down：输出里绝对值最大的那个元素
            oin = r["oin"][li][row]
            oout = oin @ Wo.T
            jo = int(torch.argmax(oout.abs()))
            dout = act @ Wd.T
            jd = int(torch.argmax(dout.abs()))
            mlist.append({"q": eq, "o": mm_entry(oin, Wo[jo], oout[jo], jo), "down": mm_entry(act, Wd[jd], dout[jd], jd)})
        chunk = {"id": qid, "L": li, "neuron": nlist, "mm": mlist, "dot": dlist}
        raw = json.dumps(chunk, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        micro_bytes += write_data(OUT / qid / f"L{li:02d}.json", raw, raw=False)  # 分块只存 .gz

    # 输出头：被选中的词元的分数 = 最终向量 · 它在嵌入表里的那一行（按词元，放在主文件里）
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
        "id": qid,
        "question": text,
        "T": T, "P": P, "G": G, "ended": ended,
        "tokens": toks,
        "steps": steps,
        "lens": lens,
        "norms": norms,
        "embNorm": emb_norm,
        "headMM": head_mm,
        "bin": b.index,
        "attCheck": r["check"],
    }
    raw_json = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    first_bytes = write_data(OUT / f"{qid}.bin", bytes(b.buf)) + write_data(OUT / f"{qid}.json", raw_json)
    print(f"  首次载入 {first_bytes / 1024:.0f} KB（gz），28 层乘加分块共 {micro_bytes / 1024:.0f} KB（gz）")
    reply_text = tok.decode(reply[:-1] if ended else reply)
    q_ids = tok(text, add_special_tokens=False).input_ids
    return {
        "id": meta["id"],
        "text": text,
        "chips": [{"id": int(i), "s": token_display(tok, i)[0]} for i in q_ids],
        "reply": reply_text,
        "replyTokens": [{"id": int(i), "s": token_display(tok, i)[0]} for i in (reply[:-1] if ended else reply)],
        "ended": ended,
        "bytes": first_bytes,        # 揭开时首次要载入的数据（.gz）
        "microBytes": micro_bytes,   # 28 层乘加分块合计（.gz，按需载入）
    }


def main():
    global OUT, SYSTEM, QUESTIONS, PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", choices=sorted(LANGS), default="zh", help="问题集：zh → public/data/，en → public/data/en/")
    ap.add_argument("--only", type=int, default=None, help="只导出第几个问题（从 1 开始），调试用")
    ap.add_argument("--from", dest="start", type=int, default=1, help="从第几个问题开始导出（从 1 开始）")
    args = ap.parse_args()
    cfg_lang = LANGS[args.lang]
    OUT, SYSTEM, QUESTIONS, PREFIX = cfg_lang["out"], cfg_lang["system"], cfg_lang["questions"], cfg_lang["prefix"]
    en = args.lang == "en"
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    cfg = model.config
    items = []
    for qi, q in enumerate(QUESTIONS):
        if (args.only and qi + 1 != args.only) or qi + 1 < args.start:
            continue
        print(f"[{qi + 1}/{len(QUESTIONS)}] {q}")
        items.append(export_question(qi, q, tok, model, cfg))
        print("  →", items[-1]["reply"])
    n_params = sum(p.numel() for p in model.parameters())
    sample_w = model.model.layers[FULL_MLP_LAYER].mlp.gate_proj.weight
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
    if not en:  # 英文版和中文版共用 public/data/weights.bin
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
        "fullMlpLayer": FULL_MLP_LAYER,
        "attTopk": ATT_TOPK,
        "mlpTopk": MLP_TOPK,
        "questions": items,
        "weightSample": [float(v) for v in sample_w[0, :8]],
        "thumbs": {"block": B, "index": thumb_index},
    }
    partial = args.only or args.start > 1
    if partial and not en:
        print(json.dumps(items, ensure_ascii=False, indent=1)[:3000])
        return
    if partial and (OUT / "manifest.json").exists():
        # 英文版分几次导出：把这次的问题并进已有的 manifest，按问题顺序排好
        old = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        merged = {q["id"]: q for q in old.get("questions", [])}
        merged.update({q["id"]: q for q in items})
        manifest["questions"] = [merged[k] for k in sorted(merged)]
        items = manifest["questions"]
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("首次载入合计（gz）:", sum(i["bytes"] for i in items), " 乘加分块合计（gz）:", sum(i["microBytes"] for i in items))


if __name__ == "__main__":
    main()
