# Logit lens: wire each layer's intermediate result straight to the output head and read what the model "would say right now"
# pip install torch "transformers>=4.51"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
ids = tok("法国的首都是巴黎，日本的首都是", return_tensors="pt").input_ids  # "The capital of France is Paris; the capital of Japan is"
with torch.no_grad():
    hs = model(ids, output_hidden_states=True).hidden_states  # 29 entries: the embeddings, plus the output of each of layers 0–27
    for k, h in enumerate(hs):
        h = h[0, -1]  # only the last position
        if k < len(hs) - 1:  # the last entry has already been through the final RMSNorm; apply it to the others
            h = model.model.norm(h)
        p = torch.softmax(model.lm_head(h).float(), dim=-1)  # the output matrix: 1,024 dims → a score for each of 151,936 tokens
        i = int(p.argmax())
        label = "embed" if k == 0 else f"layer {k - 1:2d}"
        print(f"{label:<8}  {tok.decode(i)!r:>12}  {p[i].item():6.1%}")
