"""给训练页第三章的 3D 机器补记一份数据：在同样设置下把 Qwen3-0.6B 的 3 步 SFT 重跑一遍，多记几样东西。

用法：
    python tools/train/qwen_step_3d.py --model /mnt/d/cjc/model-weights/qwen3/Qwen3-0.6B

和 qwen_step.py 完全同一个训练（同一条对话、同一个模板和遮罩、AdamW lr = 1e-5、β = (0.9, 0.95)、权重衰减 0.1、
裁剪 1.0、全程 fp32、同样的运算顺序），只是另外挂了几个钩子、多读了几样数。跑完先和 public/train/data/qwen.json
里原来的记录逐项核对（损失、梯度范数、原来跟踪的 4 个权重），对得上才写文件。多记的：

- 每层：注意力输出、前馈输出在每个位置的长度，注意力和前馈之间那一点残差的梯度 ‖∂L/∂h_mid‖；
- 每层注意力（16 个头平均）的权重矩阵 [23 × 23]（单独用 eager 注意力再前向一次读出来，不参与训练）；
- 每个参数张量：这一步的更新量 ‖Δw‖、最大 |Δw|、更新前的 ‖w‖；
- 每个张量各跟踪一个权重（第 1 步梯度绝对值最大的那个）的 g / m / v / 偏差校正 / Δw，规则和原来的 4 个一样；
- 28 层 × 7 个矩阵的梯度、Δw 的缩略图（每格 = 64 × 64 个数的均方根，取对数，3 个数量级）；
- 嵌入表：这条对话里每个词元那一行的梯度长度，以及梯度最大的 12 行；
- 每一步之后有多少权重变了、存回 bf16 会有多少变回原值（全模型和每个张量）。

输出 public/train/data/qwen3d.json.gz（小）和 public/train/data/qwen3d/st{k}.bin.gz（每步一块，格式见 split_data.py）。
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
from qwen_step import QUESTION, ANSWER, STEPS, LR, BETAS, EPS, WD, CLIP, tok_str, f9

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'public' / 'train' / 'data'
B = 64          # 缩略图每格 64 × 64 个数
DEC = 3.0       # 缩略图的对数刻度跨 3 个数量级
MATS = (('q', 'self_attn.q_proj'), ('k', 'self_attn.k_proj'), ('v', 'self_attn.v_proj'), ('o', 'self_attn.o_proj'),
        ('gate', 'mlp.gate_proj'), ('up', 'mlp.up_proj'), ('down', 'mlp.down_proj'))


def thumb(x):
    """[out, in] → 每格 64×64 的均方根，转置成 行 = 输入、列 = 输出（第 0 行在上），取对数量化到 0..255。"""
    o, i = x.shape
    blk = x.float().pow(2).reshape(o // B, B, i // B, B).mean(dim=(1, 3)).sqrt().T   # [in/B, out/B]
    mx = float(blk.max())
    if mx <= 0:
        return np.zeros(tuple(blk.shape), np.uint8), -30.0
    lg = torch.log10(blk.clamp_min(mx * 10 ** -DEC))
    q = ((lg - (math.log10(mx) - DEC)) / DEC * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return q, round(math.log10(mx), 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/mnt/d/cjc/model-weights/qwen3/Qwen3-0.6B')
    a = ap.parse_args()
    ref = json.loads((OUT / 'qwen.json').read_text())
    ref_st = [json.loads(split_data.gzip.decompress((OUT / f'qwen/st{k}.json.gz').read_bytes())) for k in range(STEPS)]
    dev = 'cuda'
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev)
    model.config.use_cache = False
    impl0 = model.config._attn_implementation
    NL = model.config.num_hidden_layers
    params = dict(model.named_parameters())
    names = list(params)

    msgs = [{'role': 'user', 'content': QUESTION}, {'role': 'assistant', 'content': ANSWER}]
    text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    prompt = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text)['input_ids']
    n_prompt = len(tok(prompt)['input_ids'])
    end = ids.index(tok.convert_tokens_to_ids('<|im_end|>'), n_prompt) + 1
    assert ids == ref['ids'] and n_prompt == ref['nPrompt'] and end == ref['end'], '对话和原来的记录不一样'
    N = len(ids)
    x = torch.tensor([ids], device=dev)
    inp, tgt = x[:, :-1], x[:, 1:]
    sft_mask = torch.zeros(N - 1, dtype=torch.bool, device=dev)
    sft_mask[n_prompt - 1:end - 1] = True

    # 原权重（bf16 存的值读成 fp32）留一份在内存，和每一步之后的权重比；prev = 上一步之后的权重，算 Δw
    orig = {n: p.detach().cpu().clone() for n, p in params.items()}
    prev = {n: t.clone() for n, t in orig.items()}

    @torch.no_grad()
    def evaluate():
        logits = model(input_ids=inp).logits[0].float()
        tl = -torch.log_softmax(logits, -1)[torch.arange(N - 1), tgt[0]]
        return round(float(tl[sft_mask].mean()), 6)

    def forward_backward(mask, extra=False):
        caps, mids, ao, mo, hooks = [], [], [], [], []

        def keep(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            h.retain_grad()
            caps.append(h)

        hooks.append(model.model.embed_tokens.register_forward_hook(keep))
        hooks += [l.register_forward_hook(keep) for l in model.model.layers]
        if extra:
            def pre_mid(_m, args):
                args[0].retain_grad()
                mids.append(args[0])

            for l in model.model.layers:
                hooks.append(l.post_attention_layernorm.register_forward_pre_hook(pre_mid))
                hooks.append(l.self_attn.register_forward_hook(lambda _m, _i, o: ao.append(o[0].detach()[0].norm(dim=-1))))
                hooks.append(l.mlp.register_forward_hook(lambda _m, _i, o: mo.append(o.detach()[0].norm(dim=-1))))
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=inp).logits[0].float()
        tl = F.cross_entropy(logits, tgt[0], reduction='none')
        loss = tl[mask].mean()
        loss.backward()
        for h in hooks:
            h.remove()
        return loss, caps, mids, ao, mo

    @torch.no_grad()
    def attention():
        """同样的权重再前向一次，用 eager 注意力读出每层 16 个头平均的注意力 [23 × 23]（不参与训练）。"""
        model.set_attn_implementation('eager')
        out = model(input_ids=inp, output_attentions=True)
        model.set_attn_implementation(impl0)
        return torch.stack([t[0].float().mean(0) for t in out.attentions]).cpu().numpy()   # [NL, 23, 23]

    def grad_norms():
        return {n: float(params[n].grad.float().norm()) for n in names if params[n].grad is not None}

    state_m = {n: torch.zeros_like(p, device='cpu') for n, p in params.items()}
    state_v = {n: torch.zeros_like(p, device='cpu') for n, p in params.items()}

    losses_after = [evaluate()]
    # 原脚本在这里先算了一次“所有位置都算”的梯度（只为对比，不更新）；照做一遍，保证随后的运算和原来一模一样
    forward_backward(torch.ones(N - 1, dtype=torch.bool, device=dev))
    hei = ids[n_prompt + 1]
    tracked = None
    vocab_extra = {}
    chunks, summary_steps = [], []
    t0 = time.time()
    for t in range(1, STEPS + 1):
        loss, caps, mids, ao, mo = forward_backward(sft_mask, extra=True)
        att = attention()
        rec = {'t': t, 'loss': round(loss.item(), 6)}
        with torch.no_grad():
            mid_g = [mh.grad[0].norm(dim=-1).tolist() for mh in mids]
        gn = grad_norms()
        # 和原记录核对：损失、每个张量的梯度范数
        assert abs(rec['loss'] - ref['steps'][t - 1]['loss']) < 2e-4, (rec['loss'], ref['steps'][t - 1]['loss'])
        rg = ref_st[t - 1]['gradNorms']
        worst = max(abs(gn[n] - rg[n]) / max(rg[n], 1e-6) for n in names)
        print(f'第 {t} 步：损失 {rec["loss"]:.6f}（原记录 {ref["steps"][t - 1]["loss"]:.6f}），梯度范数最大相对偏差 {worst:.2e}')
        assert worst < 2e-3

        # 缩略图（裁剪前的原始梯度）和嵌入表的梯度行
        pack = split_data.Pack()
        gth, gmx = [], []
        for L in range(NL):
            for key, mod in MATS:
                q, mx = thumb(params[f'model.layers.{L}.{mod}.weight'].grad)
                gth.append(q.reshape(-1))
                gmx.append(mx)
        Ge = params['model.embed_tokens.weight'].grad
        rown = Ge.float().norm(dim=1)
        top_rows = rown.topk(12)
        conv_ids = sorted(set(ids))
        emb_rows = {'conv': [[int(i), f9(rown[i])] for i in conv_ids], 'top': [[int(i), f9(v)] for i, v in zip(top_rows.indices.tolist(), top_rows.values.tolist())]}
        for i in top_rows.indices.tolist():
            vocab_extra[str(i)] = tok_str(tok, i)

        total = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP).item()
        clip = min(1.0, CLIP / (total + 1e-6))
        assert abs(total - ref['steps'][t - 1]['gradNorm']) / ref['steps'][t - 1]['gradNorm'] < 2e-3
        if tracked is None:
            # 每个张量跟踪一个权重：第 1 步（裁剪后）梯度绝对值最大的那个；嵌入表是「黑」那一行里最大的（和原来的规则一样）
            tracked = {}
            for n in names:
                G = params[n].grad
                if n == 'model.embed_tokens.weight':
                    tracked[n] = (hei, int(G[hei].abs().argmax()))
                elif G.dim() == 2:
                    k = int(G.abs().argmax())
                    tracked[n] = (k // G.shape[1], k % G.shape[1])
                else:
                    tracked[n] = (int(G.abs().argmax()),)
            for f in ref['feats']:
                assert list(tracked[f['name']]) == f['index'], (f['name'], tracked[f['name']], f['index'])
        fv = {}
        for n, idx in tracked.items():
            p = params[n]
            fv[n] = {'w0': f9(p.data[idx]), 'g': f9(p.grad[idx]), 'gRaw': f9(p.grad[idx] / clip), 'm0': f9(state_m[n][idx]), 'v0': f9(state_v[n][idx])}
        b1, b2 = BETAS
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        with torch.no_grad():
            for n, p in params.items():
                g = p.grad
                m = state_m[n].to(dev)
                v = state_v[n].to(dev)
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                if p.dim() >= 2:
                    p.mul_(1 - LR * WD)
                denom = (v.sqrt() / math.sqrt(bc2)).add_(EPS)
                p.addcdiv_(m, denom, value=-LR / bc1)
                state_m[n].copy_(m)
                state_v[n].copy_(v)
                del m, v
        for n, idx in tracked.items():
            p = params[n]
            m, v = state_m[n][idx].item(), state_v[n][idx].item()
            fv[n].update({'m': f9(m), 'v': f9(v), 'mh': f9(m / bc1), 'vh': f9(v / bc2), 'w1': f9(p.data[idx]), 'wd': WD if p.dim() >= 2 else 0.0})
        for f, rf in zip(ref['feats'], ref['steps'][t - 1]['feats']):
            mine = fv[f['name']]
            for key in ('g', 'm', 'v', 'w1'):
                assert abs(mine[key] - rf[key]) <= 2e-3 * abs(rf[key]) + 1e-12, (f['name'], key, mine[key], rf[key])

        # 这一步的更新量、Δw 的缩略图、和原权重比有多少变了 / 存回 bf16 会变回原值
        tstat, dthumb, rev_t = {}, {}, {}
        changed = revert = 0
        with torch.no_grad():
            for n, p in params.items():
                w1 = p.detach().cpu()
                dw = w1 - prev[n]
                tstat[n] = [f9(dw.norm()), f9(dw.abs().max()), f9(prev[n].norm())]
                if dw.dim() == 2 and n != 'model.embed_tokens.weight':
                    dthumb[n] = thumb(dw)
                ch = int((w1 != orig[n]).sum())
                rv = int((w1.to(torch.bfloat16).float() == orig[n]).sum())
                changed += ch
                revert += rv
                rev_t[n] = [ch, rv]
                prev[n].copy_(w1)
        dth, dmx = [], []
        for L in range(NL):
            for key, mod in MATS:
                q, mx = dthumb[f'model.layers.{L}.{mod}.weight']
                dth.append(q.reshape(-1))
                dmx.append(mx)
        model.zero_grad(set_to_none=True)
        losses_after.append(evaluate())
        print(f'   更新后损失 {losses_after[-1]:.6f}（原记录 {ref["states"][t]["lossSft"]:.6f}）；变了 {changed:,}，存回 bf16 变回原值 {revert:,}')
        assert abs(losses_after[-1] - ref['states'][t]['lossSft']) < 2e-4
        r4 = lambda rows: [[round(float(v), 4) for v in row] for row in rows]
        pack.add('gthumb', np.concatenate(gth))
        pack.add('dthumb', np.concatenate(dth))
        pack.add('att', (np.clip(att, 0, 1) * 255).round().astype(np.uint8))
        js = {
            't': t, 'loss': rec['loss'], 'lossAfter': losses_after[-1], 'gradNorm': f9(total), 'clip': f9(clip),
            'gmax': gmx, 'dmax': dmx,
            'attnOut': r4(ao), 'mlpOut': r4(mo), 'midGrad': [[f9(v) for v in row] for row in mid_g],
            'tensors': {n: tstat[n] for n in names},
            'tracked': {n: fv[n] for n in names},
            'emb': emb_rows,
            'changed': changed, 'revertBf16': revert,
            'revTensor': {n: rev_t[n] for n in names},
        }
        chunks.append(split_data.container(pack, js))
        summary_steps.append({'t': t, 'loss': rec['loss'], 'lossAfter': losses_after[-1], 'changed': changed, 'revertBf16': revert})
    secs = time.time() - t0
    # 每层每个矩阵在缩略图里的位置（行 = 输入 / 64，列 = 输出 / 64）
    idx, off = {}, 0
    for L in range(NL):
        for key, mod in MATS:
            o, i = params[f'model.layers.{L}.{mod}.weight'].shape
            idx[f'{L}:{key}'] = {'offset': off, 'w': o // B, 'h': i // B}
            off += (o // B) * (i // B)
    meta = {
        'note': '和 qwen.json 同一个训练（同样设置重跑一遍、逐项核对过），另外多记的数，给第三章的 3D 机器用',
        'block': B, 'decades': DEC, 'thumbIndex': idx,
        'tracked': {n: list(v) for n, v in tracked.items()},
        'steps': summary_steps, 'total': sum(p.numel() for p in params.values()),
        'vocab': vocab_extra, 'seconds': round(secs, 1), 'gpu': torch.cuda.get_device_name(0), 'attnImpl': impl0,
    }
    files = {'qwen3d.json': split_data.dumps(meta)}
    for k, blob in enumerate(chunks):
        files[f'qwen3d/st{k}.bin'] = blob
    split_data.report(split_data.write(files, OUT, keep_raw=set()))
    a2, js2 = split_data.read_container(split_data.gzip.decompress((OUT / 'qwen3d/st0.bin.gz').read_bytes()))
    assert a2['gthumb'].size == off and a2['att'].shape == (NL, N - 1, N - 1) and js2['t'] == 1
    print('写好了，读回来核对通过')


if __name__ == '__main__':
    main()
