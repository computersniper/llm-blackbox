# 下一个词元的前 5 名候选：模型每一步其实只做这一件事
# pip install torch "transformers>=4.51"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen3-0.6B"  # 国内下载慢：先 modelscope download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B，再把这里换成 "./Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

msgs = [{"role": "user", "content": "天空为什么是蓝色的？"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt").input_ids
print(len(ids[0]), "个词元：", [tok.decode(t) for t in ids[0]])

with torch.no_grad():
    logits = model(ids).logits[0, -1]  # 最后一个位置的打分 = 对“下一个词元”的预测
probs = torch.softmax(logits.float(), dim=-1)
for p, i in zip(*torch.topk(probs, 5)):
    print(f"{tok.decode(i)!r:>12}  {p.item():7.2%}")
