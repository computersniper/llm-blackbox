"""在真实的 Qwen3-0.6B 上做几步监督微调（SFT），把每一步细到单个权重的数字都记下来。

用法：
    python tools/train/qwen_step.py --model /mnt/d/cjc/model-weights/qwen3/Qwen3-0.6B

做的事：
- 一条“自我认知”对话：用户问“你是谁？”，希望它回答“我是黑箱里的小模型。”（原模型并不会这么说）；
- 套 Qwen3 的聊天模板（非思考模式），只对助手的回答算损失（SFT 掩码），同时也算出“所有词元都算”的预训练式损失做对比；
- 连续在这一条上走 3 步 AdamW（lr = 1e-5，β = (0.9, 0.95)，权重衰减 0.1，梯度裁剪 1.0），全程 fp32；
  优化器的 m、v 放在内存里（逐张量实现，公式和 torch.optim.AdamW 完全相同），显存只需放权重和梯度；
- 每一步记录：每个词元的目标概率和前 5 名、逻辑透镜（每层接最终 RMSNorm + 输出矩阵）、
  残差流上每层每个位置的梯度范数、每个参数张量的梯度范数、4 个具体权重的 g / m / v / 偏差校正 / Δw；
- 走完 3 步后，用原模型和微调后的模型对“新回答”和“原回答”算对数概率，代入 DPO 公式算一次偏好损失（只是算，没有做偏好训练）。
"""
import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import split_data

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'public' / 'train' / 'data'

QUESTION = '你是谁？'
ANSWER = '我是黑箱里的小模型。'
STEPS = 3
LR, BETAS, EPS, WD, CLIP = 1e-5, (0.9, 0.95), 1e-8, 0.1, 1.0
DPO_BETA = 0.1
TOPN = 5


def byte_decoder():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DEC = byte_decoder()


def tok_str(tok, tid):
    s = tok.convert_ids_to_tokens(int(tid))
    if s.startswith('<|') or s in ('<think>', '</think>'):
        return s
    try:
        return bytes(BYTE_DEC[c] for c in s).decode('utf-8')
    except (KeyError, UnicodeDecodeError):
        try:
            return ''.join(f'<0x{b:02X}>' for b in bytes(BYTE_DEC[c] for c in s))
        except KeyError:
            return s


def f9(x):
    return float(f'{float(x):.9g}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/mnt/d/cjc/model-weights/qwen3/Qwen3-0.6B')
    a = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev)
    model.config.use_cache = False
    NL = model.config.num_hidden_layers
    params = dict(model.named_parameters())
    names = list(params)
    print(f'参数 {sum(p.numel() for p in params.values()):,}，张量 {len(names)} 个')

    msgs = [{'role': 'user', 'content': QUESTION}, {'role': 'assistant', 'content': ANSWER}]
    text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    prompt = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert text.startswith(prompt)
    ids = tok(text)['input_ids']
    n_prompt = len(tok(prompt)['input_ids'])
    end = ids.index(tok.convert_tokens_to_ids('<|im_end|>'), n_prompt) + 1   # 回答 + <|im_end|>，后面那个换行不算
    N = len(ids)
    x = torch.tensor([ids], device=dev)
    inp, tgt = x[:, :-1], x[:, 1:]
    # 位置 i 预测第 i+1 个词元；SFT 只算“被预测的词元属于回答”的那些位置
    sft_mask = torch.zeros(N - 1, dtype=torch.bool, device=dev)
    sft_mask[n_prompt - 1:end - 1] = True
    # 每个词元的角色：用户的问题 / 助手的回答 / 模板（特殊标记、角色名、换行、空的思考块）
    offs = tok(text, return_offsets_mapping=True)['offset_mapping']
    q0 = text.index(QUESTION)
    q1 = q0 + len(QUESTION)
    roles = []
    for i, (s0, s1) in enumerate(offs):
        if n_prompt <= i < end - 1:
            roles.append('answer')
        elif i == end - 1:
            roles.append('tpl')
        elif s0 >= q0 and s1 <= q1:
            roles.append('user')
        else:
            roles.append('tpl')
    print('词元：', [(tok_str(tok, t), r) for t, r in zip(ids, roles)])
    print(f'共 {N} 个词元，提示 {n_prompt} 个，参与 SFT 损失的 {int(sft_mask.sum())} 个')

    # 原模型自己会怎么回答（贪心），用作 DPO 例子里“被拒绝”的回答
    with torch.no_grad():
        pids = torch.tensor([tok(prompt)['input_ids']], device=dev)
        gen = model.generate(pids, max_new_tokens=40, do_sample=False, use_cache=True)
    base_reply_ids = gen[0, pids.shape[1]:].tolist()
    if tok.convert_tokens_to_ids('<|im_end|>') in base_reply_ids:
        base_reply_ids = base_reply_ids[:base_reply_ids.index(tok.convert_tokens_to_ids('<|im_end|>')) + 1]
    base_reply = tok.decode(base_reply_ids, skip_special_tokens=True)
    print('原模型的回答：', base_reply)

    tokens_used = set(ids)

    @torch.no_grad()
    def evaluate():
        """当前权重下：每个位置的目标概率、前 5 名、两种损失。"""
        logits = model(input_ids=inp).logits[0].float()
        lp = torch.log_softmax(logits, -1)
        tl = -lp[torch.arange(N - 1), tgt[0]]
        top = lp.exp().topk(TOPN, -1)
        for i in top.indices.flatten().tolist():
            tokens_used.add(i)
        return {
            'p': [f9(math.exp(-v)) for v in tl.tolist()],
            'nll': [round(v, 6) for v in tl.tolist()],
            'top': [[[int(i), round(float(p), 6)] for i, p in zip(ti, tp)] for ti, tp in zip(top.indices.tolist(), top.values.tolist())],
            'lossSft': round(float(tl[sft_mask].mean()), 6),
            'lossPre': round(float(tl.mean()), 6),
        }

    def forward_backward(mask):
        caps = []

        def keep(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            h.retain_grad()
            caps.append(h)

        hooks = [model.model.embed_tokens.register_forward_hook(keep)] + [l.register_forward_hook(keep) for l in model.model.layers]
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=inp).logits[0].float()
        tl = F.cross_entropy(logits, tgt[0], reduction='none')
        loss = tl[mask].mean()
        loss.backward()
        for h in hooks:
            h.remove()
        return loss, caps

    def grad_norms():
        return {n: f9(params[n].grad.float().norm()) for n in names if params[n].grad is not None}

    # 优化器状态放内存
    state_m = {n: torch.zeros_like(p, device='cpu') for n, p in params.items()}
    state_v = {n: torch.zeros_like(p, device='cpu') for n, p in params.items()}

    states = [evaluate()]
    steps = []
    feats = None
    t0 = time.time()
    # 预训练式（所有词元都算）在原模型上的梯度，只用来对比，不更新
    loss_pre, _ = forward_backward(torch.ones(N - 1, dtype=torch.bool, device=dev))
    pre_cmp = {'loss': round(loss_pre.item(), 6), 'gradNorms': grad_norms()}
    pre_cmp['total'] = f9(math.sqrt(sum(v * v for v in pre_cmp['gradNorms'].values())))

    for t in range(1, STEPS + 1):
        loss, caps = forward_backward(sft_mask)
        rec = {'t': t, 'loss': round(loss.item(), 6)}
        with torch.no_grad():
            # 逻辑透镜：每层残差流 → 最终 RMSNorm → 输出矩阵
            lens_top, lens_p, rn, rg = [], [], [], []
            for h in caps:
                lz = model.lm_head(model.model.norm(h[0].detach())).float()
                lp = torch.softmax(lz, -1)
                top1 = lp.argmax(-1)
                for i in top1.tolist():
                    tokens_used.add(i)
                lens_top.append(top1.tolist())
                lens_p.append([round(float(v), 6) for v in lp[torch.arange(N - 1), tgt[0]].tolist()])
                rn.append([round(float(v), 4) for v in h[0].detach().norm(dim=-1).tolist()])
                rg.append([f9(v) for v in h.grad[0].norm(dim=-1).tolist()])
        rec['lens'] = {'top': lens_top, 'p': lens_p}
        rec['residNorm'], rec['residGrad'] = rn, rg
        rec['gradNorms'] = grad_norms()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP).item()
        clip = min(1.0, CLIP / (gn + 1e-6))
        rec['gradNorm'], rec['clip'] = f9(gn), f9(clip)
        if feats is None:
            # 挑 4 个权重：回答里第一个“新”字（黑）在共享嵌入里的那一行、最后一层 W_down、第 0 层 W_q 里梯度最大的，以及最终 RMSNorm 的 γ
            hei = ids[n_prompt + 1]
            fe = []
            g = params['model.embed_tokens.weight'].grad[hei]
            fe.append(('model.embed_tokens.weight', (hei, int(g.abs().argmax())), f'嵌入 / 输出矩阵 · 「{tok_str(tok, hei)}」那一行'))
            for nm, lab in ((f'model.layers.{NL - 1}.mlp.down_proj.weight', f'第 {NL - 1} 层 W_down'), ('model.layers.0.self_attn.q_proj.weight', '第 0 层 W_q')):
                G = params[nm].grad
                k = int(G.abs().argmax())
                fe.append((nm, (k // G.shape[1], k % G.shape[1]), f'{lab} 里梯度最大的权重'))
            G = params['model.norm.weight'].grad
            fe.append(('model.norm.weight', (int(G.abs().argmax()),), '最终 RMSNorm 的缩放 γ'))
            feats = fe
        fv = []
        for nm, idx, _ in feats:
            p = params[nm]
            fv.append({'w0': f9(p.data[idx]), 'g': f9(p.grad[idx]), 'gRaw': f9(p.grad[idx] / clip), 'm0': f9(state_m[nm][idx]), 'v0': f9(state_v[nm][idx])})
        # AdamW（和 torch.optim.AdamW 的单张量实现同一个公式）
        b1, b2 = BETAS
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        with torch.no_grad():
            for nm, p in params.items():
                g = p.grad
                m = state_m[nm].to(dev)
                v = state_v[nm].to(dev)
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                if p.dim() >= 2:
                    p.mul_(1 - LR * WD)
                denom = (v.sqrt() / math.sqrt(bc2)).add_(EPS)
                p.addcdiv_(m, denom, value=-LR / bc1)
                state_m[nm].copy_(m)
                state_v[nm].copy_(v)
        for (nm, idx, _), d in zip(feats, fv):
            p = params[nm]
            m, v = state_m[nm][idx].item(), state_v[nm][idx].item()
            d.update({'m': f9(m), 'v': f9(v), 'mh': f9(m / bc1), 'vh': f9(v / bc2), 'w1': f9(p.data[idx]), 'wd': WD if p.dim() >= 2 else 0.0})
        rec['feats'] = fv
        model.zero_grad(set_to_none=True)
        states.append(evaluate())
        steps.append(rec)
        print(f'第 {t} 步：SFT 损失 {rec["loss"]:.4f} → {states[-1]["lossSft"]:.4f}，|g| {gn:.3f}（裁剪系数 {clip:.4f}）')
    secs = time.time() - t0

    # 全模型改了多少：每个张量的相对变化
    # （原权重从磁盘重读一份放内存比较）
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    bparams = dict(base.named_parameters())
    changed, total, same_bf16 = 0, 0, 0
    with torch.no_grad():
        for nm, p in params.items():
            w1 = p.detach().cpu()
            w0 = bparams[nm].detach()
            changed += int((w1 != w0).sum())
            total += w0.numel()
            same_bf16 += int((w1.to(torch.bfloat16).float() == w0).sum())
    print(f'改动的权重 {changed:,}/{total:,}；如果把新权重存回 bf16，有 {same_bf16:,} 个会变回原值')

    # DPO：同一个提示，两种回答在原模型（参考）和微调后模型上的对数概率
    def seq_logp(m, reply_ids):
        full = torch.tensor([tok(prompt)['input_ids'] + reply_ids], device=m.device)
        with torch.no_grad():
            lg = m(input_ids=full[:, :-1]).logits[0].float()
        lp = torch.log_softmax(lg, -1)
        n0 = len(tok(prompt)['input_ids'])
        tg = full[0, 1:]
        return float(lp[torch.arange(n0 - 1, full.shape[1] - 1), tg[n0 - 1:]].sum())

    chosen_ids = ids[n_prompt:end]
    pol = {'chosen': seq_logp(model, chosen_ids), 'rejected': seq_logp(model, base_reply_ids)}
    model.cpu()
    torch.cuda.empty_cache()
    base = base.to(dev)
    ref = {'chosen': seq_logp(base, chosen_ids), 'rejected': seq_logp(base, base_reply_ids)}
    margin = (pol['chosen'] - ref['chosen']) - (pol['rejected'] - ref['rejected'])
    dpo_loss = -math.log(1 / (1 + math.exp(-DPO_BETA * margin)))
    print(f'DPO：策略 {pol}，参考 {ref}，差值 {margin:.4f}，损失 {dpo_loss:.4f}')

    vocab = {str(i): tok_str(tok, i) for i in sorted(tokens_used)}
    meta = {
        'kind': 'qwen',
        'model': {'name': 'Qwen3-0.6B', 'layers': NL, 'hidden': base.config.hidden_size, 'heads': base.config.num_attention_heads,
                  'kvHeads': base.config.num_key_value_heads, 'ffn': base.config.intermediate_size, 'vocab': base.config.vocab_size,
                  'params': sum(p.numel() for p in base.parameters())},
        'train': {'steps': STEPS, 'lr': LR, 'betas': list(BETAS), 'eps': EPS, 'weightDecay': WD, 'clip': CLIP, 'precision': 'fp32',
                  'seconds': round(secs, 1), 'gpu': torch.cuda.get_device_name(0), 'note': '连续 3 步都在同一条对话上（真实训练里每一步是不同的批次）'},
        'question': QUESTION, 'answer': ANSWER, 'text': text,
        'ids': ids, 'roles': roles, 'nPrompt': n_prompt, 'end': end,
        'sftMask': [bool(v) for v in sft_mask.tolist()],
        'states': states, 'steps': steps,
        'feats': [{'name': nm, 'index': list(idx), 'label': lab, 'decay': params_dim >= 2} for (nm, idx, lab), params_dim in zip(feats, [bparams[f[0]].dim() for f in feats])],
        'tensors': names,
        'pretrainCompare': pre_cmp,
        'changed': {'changed': changed, 'total': total, 'revertInBf16': same_bf16},
        'dpo': {'beta': DPO_BETA, 'chosen': ANSWER, 'rejected': base_reply, 'rejectedIds': base_reply_ids, 'chosenIds': chosen_ids,
                'policy': pol, 'ref': ref, 'margin': margin, 'loss': dpo_loss},
        'vocab': vocab,
    }
    # 拆成首屏用的 qwen.json + 每一步的分块 qwen/stN.json（见 split_data.py），写完读回来核对
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(json.dumps(meta, ensure_ascii=False))
    split_data.report(split_data.split_qwen(meta, OUT))
    split_data.verify_qwen(meta, OUT)
    print('核对通过')


if __name__ == '__main__':
    main()
