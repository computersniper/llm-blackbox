"""用真实的 Qwen3-VL-2B-Instruct 看图回答预设问题，把多模态页面要展示的内部数据导出到 public/multimodal/data/。

用法：
    python tools/multimodal/export_qwen3vl.py --model /mnt/d/cjc/model-weights/qwen3-vl/Qwen3-VL-2B-Instruct

每张图导出一份视觉侧的数据（和问题无关：图片在聊天模板里排在问题前面，因果注意力下它看不到问题）：
    img/<图>.jpg      模型真正看到的像素（从 pixel_values 还原，已缩放到 32 的倍数）
    <图>.json / .bin  ViT 每层特征的 PCA 颜色、注意力（合并到 2×2 词元的分辨率）、各头平均注意距离、
                      一个图块 → 嵌入值的真实乘加、DeepStack 特征、图片词元在 LLM 里的逻辑透镜读数
每个问题导出：
    <问>.json / .bin  分词、M-RoPE 三维位置、逐步的候选概率（贪心解码）、逻辑透镜、
                      每层 / 每头对图片词元的注意力（生成每个词元时看图片的哪里）
以及 manifest.json：模型配置、图片与问题列表、回答、输入法候选（问题的真实分词）。

解码用贪心（每步取概率最高的词元），所以回答完全可复现。
"""
import argparse
import gzip
import json
import math
import pathlib

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "public" / "multimodal" / "data"
IMG_DIR = pathlib.Path(__file__).resolve().parent / "images"

SYSTEM = "你是一个乐于助人的助手，请用一两句话简洁地回答。"
MAX_PIXELS = 384 * 384   # 处理器的像素上限：边长会被取整到 32 的倍数，面积不超过它
MIN_PIXELS = 256 * 256
MAX_NEW = 64
IMAGES = [
    {"id": "shapes", "file": "shapes.png", "title": "四个形状", "questions": ["图里有哪些形状？", "红色的是什么形状？", "绿色的是什么形状？"]},
    {"id": "apples", "file": "apples.png", "title": "桌上的苹果", "questions": ["图里有几个苹果？", "绿色的苹果是第几个？"]},
    {"id": "poem", "file": "poem.png", "title": "一张诗笺", "questions": ["图片上写了什么？", "这首诗的作者是谁？"]},
    {"id": "chart", "file": "chart.png", "title": "柱状图", "questions": ["哪种水果卖得最多？", "橙子卖了多少箱？"]},
    {"id": "night", "file": "night.png", "title": "夜晚的小屋", "questions": ["现在是白天还是晚上？", "描述一下这张图。"]},
    {"id": "moon", "file": "moon.jpg", "title": "月面上的人", "questions": ["这张照片拍的是什么？", "他站在哪里？"],
     "source": {"name": "Aldrin Apollo 11（AS11-40-5903）", "author": "Neil Armstrong / NASA", "license": "公共领域",
                "url": "https://commons.wikimedia.org/wiki/File:Aldrin_Apollo_11_original.jpg"}},
    {"id": "starry", "file": "starry.jpg", "title": "一幅名画", "questions": ["这是哪幅画？", "画里有什么？"],
     "source": {"name": "The Starry Night（1889）", "author": "Vincent van Gogh", "license": "公共领域",
                "url": "https://commons.wikimedia.org/wiki/File:Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"}},
]
VIT_ATTN_LAYERS = [0, 5, 11, 17, 23]   # 导出完整注意力图（合并到词元分辨率）的 ViT 层
FOCUS_LAYERS = [17, 20]                 # 导出每个头各自看图位置的 LLM 层
# “对准”层：tools/multimodal/grounding.py 用几张图上手工标的物体区域打分，第 16–26 层生成某个词时
# 注意力落在对应物体上的比例是均匀分布的 3–5 倍，更浅的层基本不看物体。网页的总览热力图默认用这几层平均。
GROUND_LAYERS = [16, 26]
TOPN = 8                                # 每步的候选词元个数


# ---------------------------------------------------------------- 词元显示（和 tools/export_qwen.py 一致）

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
    s = tok.convert_ids_to_tokens(int(tid))
    if s in tok.all_special_tokens or s.startswith("<|"):
        return s, True
    try:
        raw = bytes(BYTE_DEC[c] for c in s)
    except KeyError:
        return s, False
    try:
        return raw.decode("utf-8"), False
    except UnicodeDecodeError:
        return "".join(f"<0x{b:02X}>" for b in raw), False


# ---------------------------------------------------------------- 小工具

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


def write_pair(stem, meta, b):
    meta["bin"] = b.index
    raw_json = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    total = 0
    for name, data in ((f"{stem}.bin", bytes(b.buf)), (f"{stem}.json", raw_json)):
        (OUT / name).write_bytes(data)
        gz = gzip.compress(data, 9, mtime=0)
        (OUT / (name + ".gz")).write_bytes(gz)  # 网页优先读 .gz，用 DecompressionStream 解压
        total += len(gz)
    return total


def q8(x, scale):
    return np.clip(np.round(x / max(scale, 1e-12) * 255), 0, 255).astype(np.uint8)


def pca_rgb(states, prev=None):
    """把 N 个特征向量投到前 3 个主成分上，当作颜色（先按行 L2 归一化，只看方向）。"""
    X = states.float()
    X = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    X = X - X.mean(0, keepdim=True)
    U, S, V = torch.linalg.svd(X, full_matrices=False)
    P = (X @ V[:3].T).cpu().numpy()  # [N, 3]
    if prev is not None:  # 让相邻两层的颜色符号一致，免得逐层播放时来回翻转
        for k in range(3):
            if np.corrcoef(P[:, k], prev[:, k])[0, 1] < 0:
                P[:, k] *= -1
    raw = P.copy()
    out = np.zeros_like(P)
    for k in range(3):
        lo, hi = np.percentile(P[:, k], [2, 98])
        out[:, k] = np.clip((P[:, k] - lo) / max(hi - lo, 1e-6), 0, 1)
    var = (S[:3] ** 2 / (S ** 2).sum()).cpu().numpy()
    return (out * 255).round().astype(np.uint8), raw, [round(float(v), 4) for v in var]


# ---------------------------------------------------------------- 视觉侧

@torch.no_grad()
def vision_capture(model, inputs):
    """跑一遍完整前向，挂钩子记下 ViT 内部和 LLM 的中间结果。"""
    vis = model.model.visual
    lm = model.model.language_model
    cap = {"blk_in": {}, "blk_out": {}, "attn_in": {}, "attn_pe": {}, "attn_out": {}, "mlp_out": {}, "ds": {},
           "l_attn": {}, "l_mlp": {}, "l_in": {}}
    hooks = [
        vis.patch_embed.register_forward_hook(lambda m, i, o: cap.__setitem__("patch", o.float())),
        vis.merger.register_forward_hook(lambda m, i, o: cap.__setitem__("merged", o.float())),
    ]
    for li, blk in enumerate(vis.blocks):
        hooks.append(blk.register_forward_pre_hook(lambda m, a, li=li: cap["blk_in"].__setitem__(li, a[0].float())))
        hooks.append(blk.register_forward_hook(lambda m, i, o, li=li: cap["blk_out"].__setitem__(li, o.float())))
        def attn_pre(m, a, kw, li=li):
            cap["attn_in"][li] = a[0]
            cap["attn_pe"][li] = kw["position_embeddings"]
            return None  # 带 kwargs 的前置钩子返回非 None 会替换输入

        hooks.append(blk.attn.register_forward_pre_hook(attn_pre, with_kwargs=True))
        hooks.append(blk.attn.register_forward_hook(lambda m, i, o, li=li: cap["attn_out"].__setitem__(li, o.float())))
        hooks.append(blk.mlp.register_forward_hook(lambda m, i, o, li=li: cap["mlp_out"].__setitem__(li, o.float())))
    for di, dm in enumerate(vis.deepstack_merger_list):
        hooks.append(dm.register_forward_hook(lambda m, i, o, di=di: cap["ds"].__setitem__(di, o.float())))
    for li, layer in enumerate(lm.layers):
        hooks.append(layer.register_forward_pre_hook(lambda m, a, kw, li=li: cap["l_in"].__setitem__(li, (a[0] if a else kw["hidden_states"])[0].float()), with_kwargs=True))
        hooks.append(layer.self_attn.register_forward_hook(lambda m, i, o, li=li: cap["l_attn"].__setitem__(li, o[0][0].float())))
        hooks.append(layer.mlp.register_forward_hook(lambda m, i, o, li=li: cap["l_mlp"].__setitem__(li, o[0].float())))
    hooks.append(lm.norm.register_forward_pre_hook(lambda m, a: cap.__setitem__("l_final_in", a[0][0].float())))
    out = model(**inputs, output_attentions=True, use_cache=False)
    for h in hooks:
        h.remove()
    cap["attentions"] = out.attentions
    cap["logits"] = out.logits[0].float()
    return cap


def rotate_half(x):
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-b, a), -1)


@torch.no_grad()
def vit_attention(blk, x, pe):
    """复现 ViT 一层的注意力（qkv → 2D RoPE → softmax），返回 [H, N, N] 权重和按它算出的输出。"""
    N = x.shape[0]
    H = blk.attn.num_heads
    qkv = blk.attn.qkv(x).reshape(N, 3, H, -1).permute(1, 0, 2, 3).float()
    q, k, v = qkv[0], qkv[1], qkv[2]
    cos, sin = pe
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    s = torch.einsum("nhd,mhd->hnm", q, k) * blk.attn.scaling
    a = torch.softmax(s, -1)
    o = torch.einsum("hnm,mhd->nhd", a, v).reshape(N, -1)
    o = blk.attn.proj(o.to(x.dtype)).float()
    return a, o


def patch_layout(gh, gw, m=2):
    """pixel_values 的行顺序是 (块行, 块列, 块内行, 块内列)：同一个 2×2 合并组的 4 个图块挨在一起。"""
    rows, cols = [], []
    for hb in range(gh // m):
        for wb in range(gw // m):
            for hi in range(m):
                for wi in range(m):
                    rows.append(hb * m + hi)
                    cols.append(wb * m + wi)
    rows, cols = np.array(rows), np.array(cols)
    raster = rows * gw + cols          # 第 i 行 → 光栅顺序的位置
    perm = np.argsort(raster)          # 光栅位置 → 第几行
    return rows, cols, raster, perm


# ---------------------------------------------------------------- 导出一张图 + 它的所有问题

@torch.no_grad()
def export_image(ii, spec, proc, model, tok):
    vcfg, tcfg = model.config.vision_config, model.config.text_config
    im = Image.open(IMG_DIR / spec["file"]).convert("RGB")
    items, q_caps = [], []
    E = model.lm_head.weight.float()
    norm_w = model.model.language_model.norm.weight.float()
    eps = tcfg.rms_norm_eps

    def lens_logits(h):
        hN = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w
        return hN @ E.T

    for qi, qtext in enumerate(spec["questions"]):
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": qtext}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[im], return_tensors="pt",
                      size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}).to(model.device)
        gen = model.generate(**inputs, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=MAX_NEW,
                             output_scores=True, return_dict_in_generate=True)
        P = inputs["input_ids"].shape[1]
        reply = gen.sequences[0, P:].tolist()
        ended = reply[-1] in (151645, 151643)
        steps = []
        for g, sc in enumerate(gen.scores):
            p = torch.softmax(sc[0].float(), -1)
            tv = torch.topk(p, TOPN)
            ch = reply[g]
            steps.append({"top": [[int(i), round(float(v), 5), token_display(tok, i)[0]] for v, i in zip(tv.values, tv.indices)],
                          "chosen": int(ch), "chosenS": token_display(tok, ch)[0], "p": round(float(p[ch]), 5)})
        seq_ids = inputs["input_ids"][0].tolist() + (reply[:-1] if ended else reply)
        full = {k: v for k, v in inputs.items()}
        full["input_ids"] = torch.tensor([seq_ids], device=model.device)
        full["attention_mask"] = torch.ones_like(full["input_ids"])
        mm = torch.zeros_like(full["input_ids"])
        mm[0, : P] = inputs["mm_token_type_ids"][0]
        full["mm_token_type_ids"] = mm
        cap = vision_capture(model, full)
        T = len(seq_ids)
        G = len(steps)
        rows_q = [P - 1 + g for g in range(G)]
        # 完整前向的 argmax 应该和生成时一致（bf16 下偶尔会有极小的差别，记下来）
        mism = sum(int(torch.argmax(cap["logits"][r])) != reply[g] for g, r in enumerate(rows_q))
        pos, _ = model.model.get_rope_index(full["input_ids"], full["mm_token_type_ids"], full["image_grid_thw"])
        q_caps.append({"q": qtext, "qi": qi, "cap": cap, "inputs": full, "P": P, "T": T, "G": G, "steps": steps,
                       "reply": reply, "ended": ended, "pos": pos[:, 0].cpu().numpy(), "mism": mism, "seq": seq_ids})
        print(f"  [{spec['id']}] {qtext} → {tok.decode(reply[:-1] if ended else reply)}  (G={G}, T={T}, 不一致 {mism})")

    # ---------- 视觉侧（和问题无关，用第一个问题的那次前向）
    c0 = q_caps[0]
    cap, inp = c0["cap"], c0["inputs"]
    thw = inp["image_grid_thw"][0].tolist()
    _, gh, gw = thw
    msz = vcfg.spatial_merge_size
    mh, mw = gh // msz, gw // msz
    Np, Nv = gh * gw, mh * mw
    prow, pcol, raster, perm = patch_layout(gh, gw, msz)
    pv = inp["pixel_values"].float()                       # [Np, 3*2*16*16]，已归一化到 [-1, 1]
    ps = vcfg.patch_size
    # 模型看到的像素：取第 0 帧还原成图片
    pix = ((pv.view(Np, 3, 2, ps, ps)[:, :, 0] * 0.5 + 0.5) * 255).round().clamp(0, 255).byte().cpu().numpy()
    canvas = np.zeros((gh * ps, gw * ps, 3), np.uint8)
    for i in range(Np):
        canvas[prow[i] * ps:(prow[i] + 1) * ps, pcol[i] * ps:(pcol[i] + 1) * ps] = pix[i].transpose(1, 2, 0)
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(OUT / "img" / f"{spec['id']}.jpg", quality=90)

    vis = model.model.visual
    # ViT 每层：PCA 颜色、复现注意力并核对、平均注意距离、残差流统计
    states = [cap["blk_in"][0]] + [cap["blk_out"][li] for li in range(vcfg.depth)]
    pca = np.zeros((len(states), Np, 3), np.uint8)
    pca_var = []
    prev = None
    for si, st in enumerate(states):
        col, prev, var = pca_rgb(st, prev)
        pca[si] = col[perm]
        pca_var.append(var)
    dist = torch.tensor(np.sqrt((prow[:, None] - prow[None]) ** 2 + (pcol[:, None] - pcol[None]) ** 2), device=model.device).float()
    grp = torch.tensor(np.arange(Np) // 4, device=model.device)
    onehot = torch.zeros(Np, Nv, device=model.device)
    onehot[torch.arange(Np), grp] = 1
    vit_dist, vit_stats, vit_err = [], [], 0.0
    vattn = np.zeros((len(VIT_ATTN_LAYERS), Nv, Nv), np.uint8)
    vattn_max = np.zeros((len(VIT_ATTN_LAYERS), Nv), np.float16)
    vattn_self = np.zeros((len(VIT_ATTN_LAYERS), Nv), np.float16)
    for li, blk in enumerate(vis.blocks):
        a, o = vit_attention(blk, cap["attn_in"][li], cap["attn_pe"][li])
        vit_err = max(vit_err, float((o - cap["attn_out"][li]).abs().mean() / cap["attn_out"][li].abs().mean()))
        vit_dist.append([round(float(v), 3) for v in (a * dist).sum(-1).mean(-1).tolist()])
        x_in = cap["blk_in"][li]
        vit_stats.append([round(float(x_in.norm(dim=-1).mean()), 2), round(float(cap["attn_out"][li].norm(dim=-1).mean()), 2),
                          round(float(cap["mlp_out"][li].norm(dim=-1).mean()), 2)])
        if li in VIT_ATTN_LAYERS:
            k = VIT_ATTN_LAYERS.index(li)
            am = a.mean(0)                                     # 16 个头平均
            A = (onehot.T @ am @ onehot) / 4                   # 查询取 4 个图块的平均，键把 4 个图块加起来 → [Nv, Nv]
            A = A.cpu().numpy()
            mx = A.max(1)
            vattn[k] = np.stack([q8(A[r], mx[r]) for r in range(Nv)])
            vattn_max[k] = mx.astype(np.float16)
            vattn_self[k] = np.diag(A).astype(np.float16)
    print(f"  ViT 注意力复现误差（平均相对误差，各层最大）={vit_err:.2e}")

    merged = cap["merged"]                                      # [Nv, 2048]，LLM 看到的视觉词元
    mcol, _, mvar = pca_rgb(merged)
    ds_cols = np.stack([pca_rgb(cap["ds"][d])[0] for d in range(len(cap["ds"]))])
    ids0 = c0["seq"]
    vs = ids0.index(model.config.vision_start_token_id) + 1
    assert all(t == model.config.image_token_id for t in ids0[vs:vs + Nv]) and ids0[vs + Nv] == model.config.vision_end_token_id
    # LLM 里图片位置的残差流：输入 = 合并器输出；第 L 层之后 = 第 L+1 层的输入（前 3 层已经加上了 DeepStack）
    lm_layers = tcfg.num_hidden_layers
    img_stream = [cap["l_in"][0][vs:vs + Nv]] + [cap["l_in"][L + 1][vs:vs + Nv] if L + 1 < lm_layers else cap["l_final_in"][vs:vs + Nv] for L in range(lm_layers)]
    vnorm = np.stack([s.norm(dim=-1).cpu().numpy() for s in img_stream]).astype(np.float16)
    ds_norm = np.stack([cap["ds"][d].norm(dim=-1).cpu().numpy() for d in range(len(cap["ds"]))]).astype(np.float16)
    # 图片词元的逻辑透镜：每个视觉词元在第 L 层之后“读起来像哪个词”
    lens_tab, lens_map = [], {}
    ilens_id = np.zeros((lm_layers, Nv), np.uint16)
    ilens_p = np.zeros((lm_layers, Nv), np.uint8)
    for L in range(lm_layers):
        p = torch.softmax(lens_logits(img_stream[L + 1]), -1)
        v, ix = p.max(-1)
        for m in range(Nv):
            s = token_display(tok, int(ix[m]))[0]
            if s not in lens_map:
                lens_map[s] = len(lens_tab)
                lens_tab.append(s)
            ilens_id[L, m] = lens_map[s]
            ilens_p[L, m] = min(255, round(float(v[m]) * 255))

    # 各问题：生成每个词元时对图片的注意力，找出整体最受关注的视觉词元（给微观视图挑图块）
    heat = np.zeros(Nv)
    q_total = 0
    for qc in q_caps:
        for g in range(qc["G"]):
            r = qc["P"] - 1 + g
            for L in range(lm_layers):
                heat += qc["cap"]["attentions"][L][0, :, r, vs:vs + Nv].float().mean(0).cpu().numpy()
    hot = int(np.argmax(heat))
    # 这个视觉词元的 4 个图块里，挑像素变化最大的一个
    cand = [i for i in range(Np) if i // 4 == hot]
    pi = max(cand, key=lambda i: float(pv[i].std()))
    pe_out = cap["patch"][pi]                                   # 卷积输出 [1024]
    ch = int(torch.argmax(pe_out.abs()))
    W = vis.patch_embed.proj.weight.float().reshape(vcfg.hidden_size, -1)   # [1024, 1536]
    bconv = vis.patch_embed.proj.bias.float()
    x = pv[pi].to(W.device)
    prod = x * W[ch]
    my = float(prod.double().sum() + bconv[ch].double())
    top = torch.topk(prod.abs(), 12).indices.tolist()
    parts = prod.view(3, 2, ps * ps).sum(-1)                   # 按 通道 × 帧 分组的部分和
    after = float(cap["blk_in"][0][pi, ch])
    wimg = W[ch].view(3, 2, ps, ps).cpu().numpy()
    wmax = float(np.abs(wimg).max())
    kern = np.clip(np.round(wimg / wmax * 127 + 128), 0, 255).astype(np.uint8)          # [3, 2, 16, 16]
    # 另外 16 个输出最强的卷积核（两帧相加），当作“滤波器组”展示
    topch = torch.topk(pe_out.abs(), 17).indices.tolist()
    topch = [c for c in topch if c != ch][:16]
    bank = []
    for c in topch:
        wc = W[c].view(3, 2, ps, ps).sum(1).cpu().numpy()
        bank.append(np.clip(np.round(wc / np.abs(wc).max() * 127 + 128), 0, 255).astype(np.uint8).transpose(1, 2, 0))
    micro = {
        "patch": int(raster[pi]), "row": int(prow[pi]), "col": int(pcol[pi]), "token": hot, "ch": ch,
        "bias": float(bconv[ch]), "conv": round(float(pe_out[ch]), 5), "mine": round(my, 5),
        "pos": round(after - float(pe_out[ch]), 5), "after": round(after, 5),
        "top": [{"i": i, "c": i // (2 * ps * ps), "t": (i // (ps * ps)) % 2, "y": (i // ps) % ps, "x": i % ps,
                 "px": round(float(x[i]), 5), "w": float(W[ch, i]), "prod": round(float(prod[i]), 5)} for i in top],
        "parts": [[round(float(v), 4) for v in row] for row in parts.tolist()],
        "wmax": wmax, "bankCh": topch,
        "absSum": round(float(prod.abs().sum()), 4),
    }
    vb = Bin()
    vb.add("pca", pca)                                          # [25, Np, 3] 光栅顺序；0 = 图块嵌入 + 位置嵌入，1..24 = 每个块之后
    vb.add("mergePca", mcol)                                    # [Nv, 3]
    vb.add("dsPca", ds_cols)                                    # [3, Nv, 3]
    vb.add("vattn", vattn)                                      # [5, Nv, Nv] 每行按最大值归一化
    vb.add("vattnMax", vattn_max)
    vb.add("vattnSelf", vattn_self)
    vb.add("vnorm", vnorm)                                      # [29, Nv]
    vb.add("dsNorm", ds_norm)                                   # [3, Nv]
    vb.add("ilensId", ilens_id)                                 # [28, Nv] → lensTab
    vb.add("ilensP", ilens_p)
    vb.add("pix", pix[pi].transpose(1, 2, 0).copy())            # [16, 16, 3] 这个图块的原始像素
    vb.add("px", x.view(3, 2, ps, ps)[:, 0].cpu().numpy().astype(np.float32))   # [3, 16, 16] 归一化后的输入（两帧相同）
    vb.add("w", W[ch].cpu().numpy().astype(np.float32))        # [1536] 这个输出通道的卷积核（bf16 权重，转 float32 精确）
    vb.add("kern", kern)                                        # [3, 2, 16, 16] 可视化用
    vb.add("bank", np.stack(bank))                              # [16, 16, 16, 3]
    vmeta = {
        "id": spec["id"], "grid": [gh, gw], "merged": [mh, mw], "Np": Np, "Nv": Nv, "vs": vs,
        "vitDist": vit_dist, "vitStats": vit_stats, "pcaVar": pca_var, "mergeVar": mvar,
        "vitAttnLayers": VIT_ATTN_LAYERS, "vitErr": vit_err, "micro": micro, "lensTab": lens_tab,
        "heat": [round(float(v / heat.max()), 3) for v in heat],
    }
    vbytes = write_pair(spec["id"], vmeta, vb)

    # ---------- 每个问题
    for qc in q_caps:
        qid = f"{spec['id']}-{qc['qi'] + 1}"
        cap, P, T, G, steps = qc["cap"], qc["P"], qc["T"], qc["G"], qc["steps"]
        seq = qc["seq"]
        toks, role = [], None
        for i, t in enumerate(seq):
            s, sp = token_display(tok, t)
            if s == "<|im_start|>":
                role = None
            elif role is None and i > 0 and toks and toks[-1]["s"] == "<|im_start|>":
                role = s.strip()
            r = "img" if t == model.config.image_token_id else ("tpl" if sp else (role or "tpl"))
            toks.append({"id": int(t), "s": s, "role": r} if not sp else {"id": int(t), "s": s, "role": r, "sp": 1})
        rows_q = [P - 1 + g for g in range(G)]
        imask = torch.zeros(T, dtype=torch.bool, device=model.device)
        imask[vs:vs + Nv] = True
        att_img = np.zeros((G, lm_layers, Nv), np.uint8)
        att_max = np.zeros((G, lm_layers), np.float16)
        att_mass = np.zeros((G, lm_layers), np.float16)
        txt_idx = np.zeros((G, lm_layers, 4), np.uint16)
        txt_w = np.zeros((G, lm_layers, 4), np.uint8)
        head_mass = np.zeros((G, lm_layers, tcfg.num_attention_heads), np.uint8)
        att_head = np.zeros((G, len(FOCUS_LAYERS), tcfg.num_attention_heads, Nv), np.uint8)
        att_head_max = np.zeros((G, len(FOCUS_LAYERS), tcfg.num_attention_heads), np.float16)
        deltas = np.zeros((G, lm_layers, 2), np.float16)
        lens = []
        for g, r in enumerate(rows_q):
            per = []
            for L in range(lm_layers):
                A = cap["attentions"][L][0, :, r].float()         # [16, T]
                am = A.mean(0)
                img = am[vs:vs + Nv].cpu().numpy()
                att_mass[g, L] = img.sum()
                att_max[g, L] = img.max()
                att_img[g, L] = q8(img, img.max())
                t = am.clone()
                t[imask] = -1
                t[r + 1:] = -1
                tv = torch.topk(t, 4)
                txt_idx[g, L] = tv.indices.cpu().numpy()
                txt_w[g, L] = q8(tv.values.clamp_min(0).cpu().numpy(), 1.0)
                head_mass[g, L] = q8(A[:, vs:vs + Nv].sum(-1).cpu().numpy(), 1.0)
                if L in FOCUS_LAYERS:
                    f = FOCUS_LAYERS.index(L)
                    for h in range(tcfg.num_attention_heads):
                        hv = A[h, vs:vs + Nv].cpu().numpy()
                        att_head_max[g, f, h] = hv.max()
                        att_head[g, f, h] = q8(hv, hv.max())
                deltas[g, L] = [float(cap["l_attn"][L][r].norm()), float(cap["l_mlp"][L][r].norm())]
                h_out = cap["l_in"][L + 1][r] if L + 1 < lm_layers else cap["l_final_in"][r]
                p = torch.softmax(lens_logits(h_out), -1)
                tv = torch.topk(p, 3)
                per.append([[token_display(tok, int(i))[0], round(float(v), 4)] for v, i in zip(tv.values, tv.indices)] + [round(float(p[steps[g]["chosen"]]), 5)])
            lens.append(per)
        b = Bin()
        b.add("pos", qc["pos"].astype(np.int16))                # [3, T]  M-RoPE 的 (t, h, w)
        b.add("attImg", att_img)                                 # [G, 28, Nv] 16 头平均，每行按最大值归一化
        b.add("attMax", att_max)
        b.add("attMass", att_mass)                               # [G, 28] 分给图片的注意力总和
        b.add("txtIdx", txt_idx)                                 # [G, 28, 4] 图片以外最受关注的 4 个位置
        b.add("txtW", txt_w)
        b.add("headMass", head_mass)                             # [G, 28, 16] 每个头分给图片的注意力（×255）
        b.add("attHead", att_head)                               # [G, F, 16, Nv]
        b.add("attHeadMax", att_head_max)
        b.add("deltas", deltas)                                  # [G, 28, 2] 注意力 / 前馈往残差流里加的向量长度
        meta = {"id": qid, "image": spec["id"], "question": qc["q"], "T": T, "P": P, "G": G, "vs": vs, "Nv": Nv,
                "ended": qc["ended"], "tokens": toks, "steps": steps, "lens": lens, "mismatch": qc["mism"]}
        qbytes = write_pair(qid, meta, b)
        reply = qc["reply"][:-1] if qc["ended"] else qc["reply"]
        q_ids = tok(qc["q"], add_special_tokens=False).input_ids
        items.append({
            "id": qid, "text": qc["q"],
            "chips": [{"id": int(i), "s": token_display(tok, i)[0]} for i in q_ids],
            "reply": tok.decode(reply),
            "replyTokens": [{"id": int(i), "s": token_display(tok, i)[0]} for i in reply],
            "ended": qc["ended"], "bytes": qbytes,
        })
    ow, oh = im.size
    return {
        "id": spec["id"], "title": spec["title"], "file": f"img/{spec['id']}.jpg", "orig": [ow, oh],
        "size": [gw * ps, gh * ps], "grid": [gh, gw], "merged": [mh, mw], "Nv": Nv, "Np": Np,
        "source": spec.get("source"), "questions": items, "bytes": vbytes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--only", default=None, help="只导出某一张图（调试用）")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(args.model)
    tok = proc.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    vcfg, tcfg = model.config.vision_config, model.config.text_config
    images = []
    for ii, spec in enumerate(IMAGES):
        if args.only and spec["id"] != args.only:
            continue
        print(f"[{ii + 1}/{len(IMAGES)}] {spec['id']}")
        images.append(export_image(ii, spec, proc, model, tok))
        torch.cuda.empty_cache()
    vis, lm = model.model.visual, model.model.language_model
    n_vis = sum(p.numel() for n, p in vis.named_parameters() if not n.startswith(("merger", "deepstack")))
    n_merge = sum(p.numel() for p in vis.merger.parameters())
    n_ds = sum(p.numel() for p in vis.deepstack_merger_list.parameters())
    n_lm = sum(p.numel() for p in lm.parameters())
    manifest = {
        "model": {
            "name": "Qwen3-VL-2B-Instruct",
            "source": "https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct",
            "vision": {"depth": vcfg.depth, "hidden": vcfg.hidden_size, "heads": vcfg.num_heads, "headDim": vcfg.hidden_size // vcfg.num_heads,
                       "ffn": vcfg.intermediate_size, "act": vcfg.hidden_act, "patch": vcfg.patch_size, "temporal": vcfg.temporal_patch_size,
                       "merge": vcfg.spatial_merge_size, "posGrid": int(vcfg.num_position_embeddings ** 0.5), "deepstack": list(vcfg.deepstack_visual_indexes),
                       "out": vcfg.out_hidden_size, "inDim": vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size ** 2},
            "text": {"layers": tcfg.num_hidden_layers, "hidden": tcfg.hidden_size, "heads": tcfg.num_attention_heads, "kvHeads": tcfg.num_key_value_heads,
                     "headDim": tcfg.head_dim, "ffn": tcfg.intermediate_size, "vocab": tcfg.vocab_size,
                     "ropeTheta": (tcfg.rope_parameters or {}).get("rope_theta"), "mrope": list((tcfg.rope_parameters or {}).get("mrope_section", [24, 20, 20])),
                     "tied": bool(model.config.tie_word_embeddings)},
            "params": {"total": sum(p.numel() for p in model.parameters()), "vision": n_vis, "merger": n_merge, "deepstack": n_ds, "text": n_lm},
            "imageToken": model.config.image_token_id, "visionStart": model.config.vision_start_token_id, "visionEnd": model.config.vision_end_token_id,
        },
        "system": SYSTEM,
        "decoding": "greedy",
        "pixels": {"min": MIN_PIXELS, "max": MAX_PIXELS, "mean": 0.5, "std": 0.5},
        "vitAttnLayers": VIT_ATTN_LAYERS,
        "focusLayers": FOCUS_LAYERS,
        "groundLayers": GROUND_LAYERS,
        "images": images,
    }
    if args.only:
        print(json.dumps(images, ensure_ascii=False, indent=1)[:2000])
        return
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(i["bytes"] for i in images) + sum(q["bytes"] for i in images for q in i["questions"])
    print("数据总量（gz）:", total)


if __name__ == "__main__":
    main()
