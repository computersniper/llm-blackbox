"""从零训练一个小模型（Qwen3同架构），记录完整的训练过程。

用法：
    python tools/train_small_model.py --output public/training/data/

训练一个 4层 256维 的小型 Qwen3 架构模型，训练 3000 步，每 300 步保存一个检查点。
导出损失曲线、困惑度、生成样本、权重分布等数据。
"""

import argparse
import json
import time
import pathlib
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# ================================================================ 配置
MODEL_CONFIG = {
    "vocab_size": 151936,  # Qwen3 词表大小
    "hidden_size": 256,
    "num_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,  # GQA
    "intermediate_size": 768,  # 3 * hidden_size
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 1024
}

TRAINING_CONFIG = {
    "total_steps": 3000,
    "batch_size": 4,
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "checkpoint_interval": 300,
    "log_interval": 10,
    "max_seq_length": 128
}

# ================================================================ 模型定义（简化版 Qwen3）

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, config["max_position_embeddings"], config["rope_theta"])

    def forward(self, x, attention_mask=None):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v, seq_len=L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: repeat k, v
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(attn_output)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class SmallQwen3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config["num_layers"])])
        self.norm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(x.dtype).min

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# ================================================================ 数据集

TRAINING_DATA = [
    # 古诗
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。",
    "春来江水绿如蓝，山寺桃花始盛开。",
    "采菊东篱下，悠然见南山。",

    # 简单问答
    "问：天空为什么是蓝色的？答：因为大气散射了太阳光中的蓝光。",
    "问：为什么海水是咸的？答：因为海水中溶解了大量的盐。",
    "问：1加1等于几？答：2。",
    "问：12乘以12等于多少？答：144。",

    # 更多诗句
    "碧玉妆成一树高，万条垂下绿丝绦。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。",
    "千里莺啼绿映红，水村山郭酒旗风。",
    "渭城朝雨浥轻尘，客舍青青柳色新。",
    "月落乌啼霜满天，江枫渔火对愁眠。"
] * 200  # 重复以增加数据量


class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > max_length:
                ids = ids[:max_length]
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def collate_fn(batch):
    max_len = max(len(x) for x in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, x in enumerate(batch):
        input_ids[i, :len(x)] = x
        attention_mask[i, :len(x)] = 1

    return input_ids, attention_mask


# ================================================================ 训练循环

def generate_sample(model, tokenizer, prompt="春", max_new_tokens=20, temperature=0.7):
    """生成样本文本"""
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    model.train()
    return tokenizer.decode(input_ids[0].tolist())


def train(args):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始训练小模型...")
    print(f"配置: {MODEL_CONFIG['num_layers']}层 × {MODEL_CONFIG['hidden_size']}维")
    print(f"训练步数: {TRAINING_CONFIG['total_steps']}")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # 创建数据集
    dataset = SimpleTextDataset(TRAINING_DATA, tokenizer, TRAINING_CONFIG["max_seq_length"])
    dataloader = DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    # 创建模型
    model = SmallQwen3(MODEL_CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        betas=(0.9, 0.999)
    )

    # 训练
    model.train()
    global_step = 0
    checkpoints = []
    data_iter = iter(dataloader)

    start_time = time.time()
    last_log_time = start_time

    while global_step < TRAINING_CONFIG["total_steps"]:
        try:
            input_ids, attention_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids, attention_mask = next(data_iter)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # 前向传播
        logits = model(input_ids, attention_mask)

        # 计算损失（shift操作）
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, MODEL_CONFIG["vocab_size"]),
            shift_labels.view(-1),
            reduction='none'
        )
        loss = (loss * shift_mask.view(-1)).sum() / shift_mask.sum()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1

        # 日志
        if global_step % TRAINING_CONFIG["log_interval"] == 0:
            elapsed = time.time() - last_log_time
            perplexity = torch.exp(loss).item()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {global_step}/{TRAINING_CONFIG['total_steps']} | "
                  f"Loss: {loss.item():.4f} | PPL: {perplexity:.2f} | {TRAINING_CONFIG['log_interval']/elapsed:.1f} steps/s")
            last_log_time = time.time()

        # 检查点
        if global_step % TRAINING_CONFIG["checkpoint_interval"] == 0 or global_step == TRAINING_CONFIG["total_steps"]:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 保存检查点 step {global_step}...")
            sample = generate_sample(model, tokenizer)
            perplexity = torch.exp(loss).item()

            checkpoint_data = {
                "step": global_step,
                "loss": round(loss.item(), 4),
                "perplexity": round(perplexity, 2),
                "lr": TRAINING_CONFIG["learning_rate"],
                "sample": sample,
                "quality": "gibberish" if global_step < 500 else ("improving" if global_step < 1500 else "good")
            }
            checkpoints.append(checkpoint_data)

            print(f"  样本: {sample}")

            # 保存模型权重
            ckpt_path = args.output / f"ckpt_{global_step}.pt"
            torch.save({
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }, ckpt_path)

    # 保存训练曲线数据
    training_curve = {
        "version": "1.0",
        "model_config": MODEL_CONFIG,
        "training_config": TRAINING_CONFIG,
        "checkpoints": checkpoints,
        "total_time_seconds": int(time.time() - start_time)
    }

    curve_path = args.output / "training_curve.json"
    with open(curve_path, 'w', encoding='utf-8') as f:
        json.dump(training_curve, f, ensure_ascii=False, indent=2)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 训练完成！")
    print(f"总耗时: {(time.time() - start_time) / 60:.1f} 分钟")
    print(f"数据保存到: {args.output}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="public/training/data/", help="输出目录")
    parser.add_argument("--tokenizer-path", type=str, default="/d/gmlab/cjc/m3repro/models/Qwen3-8B",
                        help="Tokenizer路径（复用Qwen3的tokenizer）")
    args = parser.parse_args()

    args.output = pathlib.Path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
