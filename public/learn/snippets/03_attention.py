# 看注意力：句子最后一个词元，在不同的层里把注意力分给了谁
# pip install torch "transformers>=4.51"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager")  # eager 才会交出注意力权重
ids = tok("今天早上，小猫追着老鼠跑，因为它饿了。它指的是", return_tensors="pt").input_ids
words = [tok.decode(t) for t in ids[0]]
with torch.no_grad():
    out = model(ids, output_attentions=True)  # 28 层，每层 [1, 16 头, 长, 长]
print("下一个词元的前 3 名：", [tok.decode(i) for i in out.logits[0, -1].topk(3).indices])

for L in range(0, 28, 3):  # 层号从 0 数起，和本站一致
    row = out.attentions[L][0, :, -1].float().mean(0)  # 最后一个词元看向各位置，16 头平均
    w, i = row[1:].topk(3)  # 第 0 个位置单列：深层常把大半注意力停在这里
    rest = "  ".join(f"{words[j + 1]!r} {v:.2f}" for v, j in zip(w.tolist(), i.tolist()))
    print(f"第 {L:2d} 层  [0]{words[0]!r} {row[0].item():.2f} | {rest}")
