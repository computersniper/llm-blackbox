# 逻辑透镜：把每一层的中间结果直接接到输出头上，读出模型“此刻会说什么”
# pip install torch "transformers>=4.51"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
ids = tok("法国的首都是巴黎，日本的首都是", return_tensors="pt").input_ids
with torch.no_grad():
    hs = model(ids, output_hidden_states=True).hidden_states  # 29 项：嵌入，加上第 0–27 层各自的输出
    for k, h in enumerate(hs):
        h = h[0, -1]  # 只看最后一个位置
        if k < len(hs) - 1:  # 最后一项已经过了最终的 RMSNorm，其余的补上
            h = model.model.norm(h)
        p = torch.softmax(model.lm_head(h).float(), dim=-1)  # 输出矩阵：1024 维 → 151936 个词元的打分
        i = int(p.argmax())
        print(f"{'嵌入' if k == 0 else f'第 {k - 1:2d} 层'}  {tok.decode(i)!r:>12}  {p[i].item():6.1%}")
