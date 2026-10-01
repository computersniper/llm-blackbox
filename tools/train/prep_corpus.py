"""准备训练语料：chinese-poetry 项目里的《全唐诗》《全宋诗》（MIT 许可，诗歌本身属公有领域）。

用法：
    python tools/train/prep_corpus.py --src /mnt/d/cjc/datasets/chinese-poetry --out /mnt/d/cjc/datasets/poetry-train

做的事：
1. 读 tang/poet.tang.*.json、song/poet.song.*.json（原仓库《全唐诗》目录下的文件）；
2. 用 OpenCC 繁体转简体（原数据是繁体，逐字转换偶尔会有不合语境的字）；
3. 只留下格律最整齐的四类：五言 / 七言 × 绝句（4 句）/ 律诗（8 句），并且标点严格是“，。”交替；
4. 按正文去重；把含有低频字（出现不到 MIN_COUNT 次）的诗整首丢掉，词表因此只有几千个字；
5. 留出验证集：随机 2% 的诗 + 网页里固定展示的那一首《登鹳雀楼》（训练时从没见过它）；
6. 写出 corpus.npz（字符编号流，每首诗前面都有一个 <|endoftext|>）和 vocab.json。
"""
import argparse
import collections
import json
import pathlib
import re

import numpy as np
import opencc

EOT = '<|endoftext|>'   # 和 Qwen3 一样，用它分隔文档
HELD_OUT = '白日依山尽，黄河入海流。欲穷千里目，更上一层楼。'
MIN_COUNT = 3
VAL_FRAC = 0.02


def load(src):
    cc = opencc.OpenCC('t2s')
    cjk = re.compile(r'^[一-鿿]+$')
    seen, out = set(), []
    stats = collections.Counter()
    for sub, pat in (('tang', 'poet.tang.*.json'), ('song', 'poet.song.*.json')):
        files = sorted((src / sub).glob(pat), key=lambda p: int(p.name.split('.')[2]))
        for f in files:
            for p in json.loads(f.read_text(encoding='utf-8')):
                stats[f'raw_{sub}'] += 1
                body = cc.convert(''.join(p.get('paragraphs') or []))
                lines = re.split(r'[，。]', body)
                if lines and lines[-1] == '':
                    lines = lines[:-1]
                if not lines or len(lines) not in (4, 8):
                    continue
                n = len(lines[0])
                if n not in (5, 7) or any(len(x) != n or not cjk.match(x) for x in lines):
                    continue
                if body != ''.join(x + ('，' if i % 2 == 0 else '。') for i, x in enumerate(lines)):
                    continue
                if body in seen:
                    continue
                seen.add(body)
                stats[f'kept_{sub}'] += 1
                out.append({'body': body, 'author': cc.convert(p.get('author') or ''), 'title': cc.convert(p.get('title') or ''), 'src': sub})
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/mnt/d/cjc/datasets/chinese-poetry')
    ap.add_argument('--out', default='/mnt/d/cjc/datasets/poetry-train')
    a = ap.parse_args()
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    poems, stats = load(src)
    print(dict(stats))
    # 留出的那首诗：任何包含它的句子的诗都不进训练集
    keys = [x for x in re.split(r'[，。]', HELD_OUT) if x]
    poems = [p for p in poems if not any(k in p['body'] for k in keys)]

    cnt = collections.Counter(ch for p in poems for ch in p['body'])
    for ch in HELD_OUT:
        cnt[ch] += MIN_COUNT  # 保证留出诗里的字都在词表里（它们本来就都是常用字）
    poems = [p for p in poems if all(cnt[ch] >= MIN_COUNT for ch in p['body'])]
    cnt = collections.Counter(ch for p in poems for ch in p['body'])
    # 词表：0 号是 <|endoftext|>，其余按频率从高到低
    chars = [ch for ch, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]
    vocab = [EOT] + chars
    idx = {ch: i for i, ch in enumerate(vocab)}
    assert all(ch in idx for ch in HELD_OUT)

    rng = np.random.default_rng(0)
    order = rng.permutation(len(poems))
    n_val = int(len(poems) * VAL_FRAC)
    val_set = set(order[:n_val].tolist())
    train = [poems[i] for i in range(len(poems)) if i not in val_set]
    val = [poems[i] for i in sorted(val_set)]

    def stream(ps):
        ids, starts = [], []
        for p in ps:
            starts.append(len(ids))
            ids.append(0)
            ids.extend(idx[ch] for ch in p['body'])
        ids.append(0)
        return np.array(ids, dtype=np.uint16), np.array(starts, dtype=np.int64)

    tr_ids, tr_starts = stream([train[i] for i in rng.permutation(len(train))])
    va_ids, va_starts = stream(val)
    held = np.array([0] + [idx[ch] for ch in HELD_OUT] + [0], dtype=np.uint16)
    np.savez(out / 'corpus.npz', train_ids=tr_ids, train_starts=tr_starts, val_ids=va_ids, val_starts=va_starts, held=held)
    info = {
        'vocab': vocab,
        'counts': [0] + [cnt[ch] for ch in chars],
        'stats': dict(stats),
        'poems': {'train': len(train), 'val': len(val)},
        'chars': {'train': int(len(tr_ids)), 'val': int(len(va_ids))},
        'kinds': dict(collections.Counter(f"{len(p['body'].split('，')[0])}言{'绝句' if len(p['body']) <= 32 else '律诗'}" for p in poems)),
        'heldOut': HELD_OUT,
    }
    (out / 'vocab.json').write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
    print(f"词表 {len(vocab)}，训练 {len(train)} 首 / {len(tr_ids)} 字，验证 {len(val)} 首 / {len(va_ids)} 字")
    print(info['kinds'])


if __name__ == '__main__':
    main()
