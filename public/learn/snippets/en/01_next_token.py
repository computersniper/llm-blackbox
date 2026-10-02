# Top 5 candidates for the next token: at every step, this is all the model really does
# pip install torch "transformers>=4.51"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen3-0.6B"  # or the path to a local copy of the weights
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

# Same system prompt and question as the inference page, kept in Chinese so the output can be compared with it.
# (Translation: "You are a helpful assistant. Answer concisely in one or two sentences." / "Why is the sky blue?")
msgs = [{"role": "system", "content": "你是一个乐于助人的助手，请用一两句话简洁地回答。"},
        {"role": "user", "content": "天空为什么是蓝色的？"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt").input_ids
print(len(ids[0]), "tokens:", [tok.decode(t) for t in ids[0]])

with torch.no_grad():
    logits = model(ids).logits[0, -1]  # the scores at the last position = the prediction for the next token
probs = torch.softmax(logits.float(), dim=-1)
for p, i in zip(*torch.topk(probs, 5)):
    print(f"{tok.decode(i)!r:>12}  {p.item():7.2%}")
