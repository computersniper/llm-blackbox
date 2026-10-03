# A hand-written generation loop: temperature → top-k → top-p → draw, then feed the chosen token back in for the next step
# pip install torch "transformers>=4.51"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
msgs = [{"role": "user", "content": "用一句话介绍你自己。"}]  # "Introduce yourself in one sentence."
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True).input_ids
end = tok.convert_tokens_to_ids("<|im_end|>")
torch.manual_seed(0)

T, K, P = 0.7, 20, 0.8  # Qwen3's recommended values for non-thinking mode; this site's inference page uses the same
past, new = None, ids   # step 1 runs the whole prompt at once (prefill); after that each step feeds in 1 new token and reads the earlier K, V from the cache
for _ in range(80):
    with torch.no_grad():
        out = model(new, past_key_values=past, use_cache=True)
    past = out.past_key_values
    top = torch.topk(out.logits[0, -1].float() / T, K)  # ① divide by the temperature  ② keep the top K
    probs = torch.softmax(top.values, dim=-1)
    probs[torch.cumsum(probs, 0) - probs >= P] = 0      # ③ drop everything after the running total reaches P
    nxt = top.indices[torch.multinomial(probs / probs.sum(), 1)]  # ④ roll the dice with the probabilities that are left
    if nxt.item() == end:  # the model drew <|im_end|> by itself: the answer is over
        break
    print(tok.decode(nxt), end="", flush=True)
    new = nxt.view(1, 1)
print()
