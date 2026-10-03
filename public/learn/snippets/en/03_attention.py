# Where does attention go? Who the last token of a sentence attends to, layer by layer
# pip install torch "transformers>=4.51"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager")  # only "eager" hands back the attention weights
# "This morning the kitten chased the mouse, because it was hungry. 'It' refers to"
ids = tok("今天早上，小猫追着老鼠跑，因为它饿了。它指的是", return_tensors="pt").input_ids
words = [tok.decode(t) for t in ids[0]]
with torch.no_grad():
    out = model(ids, output_attentions=True)  # 28 layers, each [1, 16 heads, length, length]
print("top 3 next tokens:", [tok.decode(i) for i in out.logits[0, -1].topk(3).indices])

for L in range(0, 28, 3):  # layers are numbered from 0, as on this site
    row = out.attentions[L][0, :, -1].float().mean(0)  # where the last token looks, averaged over the 16 heads
    w, i = row[1:].topk(3)  # position 0 is listed on its own: deeper layers often park most of their attention there
    rest = "  ".join(f"{words[j + 1]!r} {v:.2f}" for v, j in zip(w.tolist(), i.tolist()))
    print(f"layer {L:2d}  [0]{words[0]!r} {row[0].item():.2f} | {rest}")
