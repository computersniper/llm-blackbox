"""哪些层 / 头在生成某个词时真的在看对应的物体？用几组手工标注的区域（合并后网格上的矩形）打分。

用法：python tools/multimodal/grounding.py（模型路径写在下面的 P 里）
输出每层的“富集倍数”：注意力落在目标区域的比例 ÷ 区域占整张图的比例。1 = 和瞎看一样。
"""
import pathlib
import torch, numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
P = "/mnt/d/cjc/model-weights/qwen3-vl/Qwen3-VL-2B-Instruct"
IMG = str(pathlib.Path(__file__).resolve().parent / "images") + "/"
proc = AutoProcessor.from_pretrained(P)
model = Qwen3VLForConditionalGeneration.from_pretrained(P, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
SYSTEM = "你是一个乐于助人的助手，请用一两句话简洁地回答。"
# (图, 问题, 目标词, 区域：merged 网格上的 (r0, r1, c0, c1) 闭区间)
CASES = [
    ("moon.jpg", "这张照片拍的是什么？", "宇", (0, 6, 4, 7)),
    ("shapes.png", "红色的是什么形状？", "圆形", (1, 5, 1, 5)),
    ("shapes.png", "图里有哪些形状？", "五角星", (7, 10, 6, 10)),
    ("shapes.png", "图里有哪些形状？", "三角形", (7, 10, 1, 5)),
    ("chart.png", "哪种水果卖得最多？", "香蕉", (2, 9, 4, 6)),
    ("apples.png", "绿色的苹果是第几个？", "3", (4, 7, 6, 8)),
]
res = []
for f, q, word, (r0, r1, c0, c1) in CASES:
    im = Image.open(IMG + f).convert("RGB")
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[im], return_tensors="pt", size={"shortest_edge": 65536, "longest_edge": 147456}).to("cuda")
    gen = model.generate(**inp, do_sample=False, max_new_tokens=64)
    seq = gen[0].tolist()
    Pn = inp["input_ids"].shape[1]
    reply = seq[Pn:]
    toks = [proc.tokenizer.decode([t]) for t in reply]
    g = next(i for i, t in enumerate(toks) if word.startswith(t) or t.startswith(word[:1]))
    full = dict(inp)
    full["input_ids"] = torch.tensor([seq[:-1]], device="cuda")
    full["attention_mask"] = torch.ones_like(full["input_ids"])
    mm = torch.zeros_like(full["input_ids"]); mm[0, :Pn] = inp["mm_token_type_ids"][0]; full["mm_token_type_ids"] = mm
    with torch.no_grad():
        out = model(**full, output_attentions=True)
    _, gh, gw = inp["image_grid_thw"][0].tolist()
    mh, mw = gh // 2, gw // 2
    vs = seq.index(151652) + 1
    Nv = mh * mw
    mask = np.zeros((mh, mw), bool); mask[r0:r1 + 1, c0:c1 + 1] = True; mask = mask.reshape(-1)
    base = mask.mean()
    row = Pn - 1 + g
    # 预测 toks[g] 的那一行是 row（第 g 步的查询位置）
    sc = np.zeros((28, 17))
    for L in range(28):
        A = out.attentions[L][0, :, row, vs:vs + Nv].float().cpu().numpy()  # [16, Nv]
        for h in range(16):
            s = A[h].sum()
            sc[L, h] = (A[h][mask].sum() / max(s, 1e-9)) / base if s > 0.02 else np.nan
        am = A.mean(0)
        sc[L, 16] = am[mask].sum() / am.sum() / base
    res.append(sc)
    print(f, q, "→", "".join(toks), "| 目标", toks[g], "g=", g, "基线", round(base, 3))
    print("  层平均（16 头平均）的富集倍数:", np.round(sc[:, 16], 2))
R = np.stack(res)
print("各层（头平均）富集倍数的平均:", np.round(np.nanmean(R[:, :, 16], 0), 2))
best = np.nanmean(R[:, :, :16], 0)
idx = np.dstack(np.unravel_index(np.argsort(-np.nan_to_num(best).ravel())[:20], best.shape))[0]
print("最好的 20 个 (层, 头):", [(int(a), int(b), round(float(best[a, b]), 2)) for a, b in idx])
print("每层最好的头的平均富集:", np.round(np.nanmax(best, 1), 2))
