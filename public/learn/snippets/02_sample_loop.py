# 自己写生成循环：温度 → top-k → top-p → 抽签，抽中的词元接回去再算下一步
# pip install torch "transformers>=4.51"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
msgs = [{"role": "user", "content": "用一句话介绍你自己。"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True).input_ids
end = tok.convert_tokens_to_ids("<|im_end|>")
torch.manual_seed(0)

T, K, P = 0.7, 20, 0.8  # Qwen3 非思考模式的推荐值，本站推理页用的也是这一组
past, new = None, ids   # 第一步把整段提示一次算完（预填充），之后每步只喂 1 个新词元，前面的 K、V 从缓存里取
for _ in range(80):
    with torch.no_grad():
        out = model(new, past_key_values=past, use_cache=True)
    past = out.past_key_values
    top = torch.topk(out.logits[0, -1].float() / T, K)  # ① 除以温度  ② 只留前 K 名
    probs = torch.softmax(top.values, dim=-1)
    probs[torch.cumsum(probs, 0) - probs >= P] = 0      # ③ 累计概率凑够 P 之后的全部丢掉
    nxt = top.indices[torch.multinomial(probs / probs.sum(), 1)]  # ④ 按剩下的概率掷骰子
    if nxt.item() == end:  # 模型自己抽到 <|im_end|>，回答就结束
        break
    print(tok.decode(nxt), end="", flush=True)
    new = nxt.view(1, 1)
print()
