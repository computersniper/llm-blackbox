"""世界模型页 · 大模型里的世界地图：用线性探针从 Qwen3-0.6B 的隐状态里读出城市的经纬度，导出给 public/world/probe/。

复现 Gurnee & Tegmark (2023)《Language Models Represent Space and Time》（arXiv:2310.02207）的做法，换成 Qwen3-0.6B：
- 每个城市单独喂给模型，用论文的两种模板：
    coords：“What are the lat/lon coordinates of <城市名>”（论文 3.3 节问坐标的提示，页面默认用这个）
    name：只有城市名（论文的主实验）
  开头加 <|endoftext|>：论文用 Llama-2，序列开头有 BOS；Qwen3 没有专门的 BOS，预训练里用 <|endoftext|> 分隔文档，
  就拿它当开头。不加的话，单个词元的名字会落在第 0 个位置——那个位置从第 2 层起范数暴涨到 6500 左右
  （注意力汇点），读不出东西；
- 取城市名最后一个词元在 28 层每一层输出的残差流（隐状态，1024 维，fp32 前向）；
- 每层一个岭回归探针预测（纬度, 经度）：目标先标准化，正则强度 λ 在训练集上用高效留一交叉验证选
  （和论文用的 sklearn RidgeCV 同一个公式），两个目标共用一个 λ；
- 按城市 5 折交叉：每个城市都由“没见过它”的那一折探针来预测，所以地图上每个点都是测试集上的预测；
- 指标：R²（纬度、经度各算再平均，同论文）、预测点到真实位置的大圆距离（km）、论文的“邻近误差”
  （有多少比例的其他城市的预测点比它自己的预测点更靠近它的真实位置；随机猜是 0.5）；
- 两个对照（都用 coords 模板）：① 打乱标签：训练集里城市和坐标随机配对，探针别的都不变；
  ② 未训练的模型：同样结构、随机初始化的 Qwen3-0.6B（种子 0），同样的探针。
  ①证明探针自己“背”不出地图；②看名字的字面（拼写）本身能带出多少地理线索；
- 额外：按国家整块留出（论文 4.1 节）——训练时一个国家的城市一个都不给，再看对这个国家的预测；
- 算式：FEATURED 城市在每一层的预测 = 平均位置 + Σ (h[i] − h̄[i]) × w[i]，导出贡献最大的 12 维和其余维的汇总，
  脚本会核对“加起来正好等于探针的预测”。

用法：
    python tools/world/probe_data.py                 # 先准备 cities.tsv / land.json（下载 GeoNames、Natural Earth）
    python tools/world/probe_export.py --model /path/to/Qwen3-0.6B
    python tools/world/probe_export.py --skip-acts   # 隐状态已经缓存在 --work 下时，不用 GPU 只重算探针和导出
"""
import argparse
import gzip
import json
import pathlib
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DEFAULT = ROOT / 'public' / 'world' / 'probe' / 'data'
EOT = 151643                        # <|endoftext|>
NL, D = 28, 1024
FOLDS = 5
SEED = 0
LAMBDAS = np.logspace(-1, 7, 33)    # 候选 λ；脚本会检查选中的 λ 没有顶到网格边上
TOPK = 12                           # 算式板上逐项写出的维数（和推理页的算式板一致）
CONT = ['AS', 'EU', 'AF', 'NA', 'SA', 'OC']
CONT_ZH = ['亚洲', '欧洲', '非洲', '北美洲', '南美洲', '大洋洲']
TEMPLATES = {'coords': 'What are the lat/lon coordinates of {}', 'name': '{}'}
# 变体：名字 → (模型, 模板, 打乱标签)
VARIANTS = {
    'coords': ('trained', 'coords', False),     # 真实模型 · 问坐标的提示（默认）
    'name': ('trained', 'name', False),         # 真实模型 · 只有名字
    'shuffled': ('trained', 'coords', True),    # 对照：打乱标签
    'random': ('random', 'coords', False),      # 对照：未训练的模型
}
R_EARTH = 6371.0
# 论文报告的参考数字（Table 2：World 数据集、60% 深度处的线性探针 R²；Table 3：按国家留出前后的邻近误差）
PAPER = {'r2_7b': 0.881, 'r2_13b': 0.896, 'r2_70b': 0.911, 'prox_nominal_7b': 0.071, 'prox_heldout_7b': 0.170}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_cities(work):
    rows = []
    lines = (work / 'cities.tsv').read_text(encoding='utf-8').splitlines()
    head = lines[0].split('\t')
    for line in lines[1:]:
        f = dict(zip(head, line.split('\t')))
        rows.append(dict(id=int(f['id']), name=f['name'], cc=f['cc'], ct=f['continent'], lat=float(f['lat']),
                         lon=float(f['lon']), pop=int(f['pop']), zh=f.get('zh', '')))
    return rows


def encode(tok, tmpl, name):
    """<|endoftext|> + 模板。返回词元编号、城市名占的词元数（城市名总在最后）"""
    text = TEMPLATES[tmpl].format(name)
    ids = [EOT] + tok(text, add_special_tokens=False)['input_ids']
    if tmpl == 'name':
        return ids, len(ids) - 1
    pre = [EOT] + tok(TEMPLATES[tmpl].format('').rstrip(), add_special_tokens=False)['input_ids']
    assert ids[:len(pre)] == pre, (text, ids, pre)   # 前缀的分词不受城市名影响
    return ids, len(ids) - len(pre)


# ---------------------------------------------------------------- 隐状态

@torch.no_grad()
def extract(model, seqs, dev, bs=192):
    """取每条序列最后一个词元在嵌入层和 28 层输出处的残差流 → (29, n, 1024) float32"""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = np.zeros((NL + 1, len(seqs), D), np.float32)
    cap = {}
    hooks = [layer.register_forward_hook(lambda m, i, o, li=li: cap.__setitem__(li, o[0] if isinstance(o, tuple) else o))
             for li, layer in enumerate(model.model.layers)]
    hooks.append(model.model.embed_tokens.register_forward_hook(lambda m, i, o: cap.__setitem__('emb', o)))
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        L = max(len(seqs[i]) for i in idx)
        ids = torch.full((len(idx), L), EOT, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, :len(seqs[i])] = torch.tensor(seqs[i])
            att[r, :len(seqs[i])] = 1
        model.model(input_ids=ids.to(dev), attention_mask=att.to(dev))   # 右侧补齐：真实词元的位置编号和计算都不受影响
        last = torch.tensor([len(seqs[i]) - 1 for i in idx], device=dev)
        rr = torch.arange(len(idx), device=dev)
        out[0, idx] = cap['emb'][rr, last].float().cpu().numpy()
        for li in range(NL):
            out[li + 1, idx] = cap[li][rr, last].float().cpu().numpy()
    for h in hooks:
        h.remove()
    return out


def run_acts(args, rows, work):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(args.model)
    need = sorted({(m, t) for m, t, _ in VARIANTS.values()})
    for mname in ('trained', 'random'):
        todo = [t for m, t in need if m == mname and not (work / f'acts_{m}_{t}.npy').exists()]
        if not todo:
            continue
        if mname == 'trained':
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
        else:
            torch.manual_seed(SEED)
            cfg = AutoConfig.from_pretrained(args.model)
            model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
            w = model.model.layers[0].self_attn.q_proj.weight
            log(f'随机初始化：q_proj 权重标准差 {w.std().item():.4f}（配置 initializer_range = {cfg.initializer_range}）')
        model = model.to(dev).eval()
        for t in todo:
            log('提取隐状态：', mname, t)
            seqs = [encode(tok, t, r['name'])[0] for r in rows]
            np.save(work / f'acts_{mname}_{t}.npy', extract(model, seqs, dev))
            if dev == 'cuda':
                log(f'  峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.2f} GB')
        del model
        if dev == 'cuda':
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- 探针

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def ridge_fit(X, Y, lambdas=LAMBDAS, shuffle=None):
    """带截距的岭回归，目标标准化，λ 用高效留一交叉验证选（和 sklearn RidgeCV 的 GCV 模式同一个公式）。
    X: (n, d) float64 tensor；Y: (n, 2)；shuffle: 训练集内部打乱标签用的排列。返回探针参数（原始单位）"""
    if shuffle is not None:
        Y = Y[shuffle]
    xm, ym, ys = X.mean(0), Y.mean(0), Y.std(0)
    Xc, Yz = X - xm, (Y - ym) / ys
    U, s, Vt = torch.linalg.svd(Xc, full_matrices=False)
    UtY = U.T @ Yz
    n = X.shape[0]
    best = None
    s2 = s * s
    for lam in lambdas:
        d = s2 / (s2 + lam)
        fit = U @ (d[:, None] * UtY)
        h = 1.0 / n + (U * U) @ d
        loo = ((Yz - fit) / (1 - h)[:, None]).pow(2).mean().item()
        if best is None or loo < best[0]:
            best = (loo, lam)
    lam = best[1]
    coef = Vt.T @ ((s / (s2 + lam))[:, None] * UtY)          # (d, 2)，标准化目标的系数
    w = coef * ys                                              # 原始单位：度 / 单位隐状态
    return dict(xm=xm, ym=ym, w=w, lam=float(lam), edge=lam in (lambdas[0], lambdas[-1]))


def ridge_predict(P, X):
    return (X - P['xm']) @ P['w'] + P['ym']


def haversine(a, b):
    """a, b: (..., 2) 的 (纬度, 经度)，单位度 → km。纬度先夹到 ±90（线性探针可能预测出界）"""
    la1, lo1 = np.radians(np.clip(a[..., 0], -90, 90)), np.radians(a[..., 1])
    la2, lo2 = np.radians(np.clip(b[..., 0], -90, 90)), np.radians(b[..., 1])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def proximity(true, pred):
    """论文的邻近误差：对每个城市 i，有多少比例的其他城市 j 的预测点比 i 自己的预测点更靠近 i 的真实位置"""
    t = torch.tensor(np.radians(np.clip(true, -90, 90)), device=DEV)
    p = torch.tensor(np.radians(np.stack([np.clip(pred[:, 0], -90, 90), pred[:, 1]], 1)), device=DEV)
    dl = p[None, :, 0] - t[:, None, 0]
    dn = p[None, :, 1] - t[:, None, 1]
    h = torch.sin(dl / 2) ** 2 + torch.cos(t[:, None, 0]) * torch.cos(p[None, :, 0]) * torch.sin(dn / 2) ** 2
    dist = torch.asin(torch.sqrt(h.clamp(0, 1)))
    own = dist.diagonal()
    return ((dist < own[:, None]).double().mean(1)).cpu().numpy()


def r2(y, p):
    return 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def scores(Y, P):
    km = haversine(Y, P)
    rr = r2(Y, P)
    prox = proximity(Y, P)
    return dict(r2=round(float(rr.mean()), 4), r2Lat=round(float(rr[0]), 4), r2Lon=round(float(rr[1]), 4),
                km=round(float(km.mean()), 1), kmMed=round(float(np.median(km)), 1), prox=round(float(prox.mean()), 4)), km


def cv_probes(X, Y, fold, keep=False, shuffle=False):
    """5 折交叉：返回每个城市的折外预测、各折的 λ，keep=True 时还返回各折的探针参数"""
    Xt = torch.tensor(X, dtype=torch.float64, device=DEV)
    Yt = torch.tensor(Y, dtype=torch.float64, device=DEV)
    pred = np.zeros_like(Y)
    lams, probes, edge = [], [], False
    for f in range(FOLDS):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        perm = None
        if shuffle:
            perm = torch.tensor(np.random.default_rng(SEED + 100 + f).permutation(len(tr)), device=DEV)
        P = ridge_fit(Xt[tr], Yt[tr], shuffle=perm)
        pred[te] = ridge_predict(P, Xt[te]).cpu().numpy()
        lams.append(P['lam'])
        edge |= P['edge'] and not shuffle
        if keep:
            probes.append({k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in P.items()})
    return pred, lams, probes, edge


# ---------------------------------------------------------------- 算式

def g4(v):
    return float(f'{v:.4g}')


def formula(h, P, ystr):
    """预测 = 平均位置 + Σ (h[i] − h̄[i]) × w[i]。两个目标各写出贡献最大的 TOPK 维 + 其余维的汇总"""
    xt = h.astype(np.float64) - P['xm']
    res = {}
    for k, key in enumerate(('lat', 'lon')):
        w = P['w'][:, k]
        prod = xt * w
        y = float(P['ym'][k] + prod.sum())
        order = np.argsort(-np.abs(prod))
        top, rest = order[:TOPK], order[TOPK:]
        cum = np.cumsum(prod[order])
        marks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, D]
        res[key] = dict(
            b=round(float(P['ym'][k]), 4),
            i=[int(i) for i in top],
            h=[g4(h[i]) for i in top],
            x=[g4(xt[i]) for i in top],
            w=[g4(w[i]) for i in top],
            p=[round(float(prod[i]), 4) for i in top],
            rest=round(float(prod[rest].sum()), 4),
            pos=round(float(prod[prod > 0].sum()), 4),
            neg=round(float(prod[prod < 0].sum()), 4),
            cum=[round(float(cum[m - 1]), 3) for m in marks],
            y=round(y, 4),
        )
        assert abs(ystr[k] - y) < 1e-6 * max(1, abs(y)), (ystr[k], y)
    return res


# ---------------------------------------------------------------- 导出

def write(path, data, gz_only=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not gz_only:
        path.write_bytes(data)
    pathlib.Path(str(path) + '.gz').write_bytes(gzip.compress(data, 9, mtime=0))


def pack_pred(pred):
    """(变体, 28, n, 2) 度 → int16（0.1°），沿层做差分（第 0 层存原值，之后存和上一层的差），
    再按字节分面存放（先放所有数的低字节，再放高字节）——相邻层的预测很接近，这样 gzip 能压得更小"""
    q = np.clip(np.round(pred * 10), -32767, 32767).astype(np.int32)
    d = q.copy()
    d[:, 1:] = q[:, 1:] - q[:, :-1]
    b = d.astype('<i2').view(np.uint8).reshape(-1, 2)
    return np.concatenate([b[:, 0], b[:, 1]]).tobytes()


def country_names(ccs):
    from babel import Locale
    loc = Locale.parse('zh_Hans')
    special = {'TW': '中国台湾', 'HK': '中国香港', 'MO': '中国澳门'}
    return {cc: special.get(cc) or loc.territories.get(cc) or cc for cc in ccs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/mnt/d/cjc/model-weights/qwen3/Qwen3-0.6B')
    ap.add_argument('--work', default='/mnt/d/cjc/world-model/probe')
    ap.add_argument('--out', default=str(OUT_DEFAULT))
    ap.add_argument('--skip-acts', action='store_true')
    a = ap.parse_args()
    work, out = pathlib.Path(a.work), pathlib.Path(a.out)
    rows = load_cities(work)
    n = len(rows)
    log(f'{n} 个城市')
    if not a.skip_acts:
        run_acts(a, rows, work)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    acts = {}
    for m, t, _ in VARIANTS.values():
        acts[m, t] = np.load(work / f'acts_{m}_{t}.npy', mmap_mode='r')
    Y = np.array([[r['lat'], r['lon']] for r in rows], np.float64)
    fold = np.random.default_rng(SEED).permutation(np.arange(n) % FOLDS)
    feat = [i for i, r in enumerate(rows) if r['zh']]
    VN = list(VARIANTS)

    preds = np.zeros((len(VN), NL, n, 2))
    metrics = {v: [] for v in VN}
    lam = {v: [] for v in VN}
    emb = {}
    km_main = np.zeros((NL, n))
    main_probes = []
    for v in ('coords', 'name', 'random'):
        m, t, _ = VARIANTS[v]
        p, _, _, _ = cv_probes(np.asarray(acts[m, t][0], np.float64), Y, fold)
        emb[v] = scores(Y, p)[0]
    log('嵌入层：', {k: e['r2'] for k, e in emb.items()})
    for L in range(NL):
        for vi, v in enumerate(VN):
            m, t, sh = VARIANTS[v]
            X = np.asarray(acts[m, t][L + 1], np.float64)
            p, lams, probes, edge = cv_probes(X, Y, fold, keep=(v == 'coords'), shuffle=sh)
            if edge:
                log(f'  注意：第 {L} 层 {v} 的 λ 顶到了网格边上 {lams}')
            preds[vi, L] = p
            s, km = scores(Y, p)
            metrics[v].append(s)
            lam[v].append([g4(x) for x in lams])
            if v == 'coords':
                km_main[L] = km
                main_probes.append(probes)
        mm = {v: metrics[v][L] for v in VN}
        log(f'第 {L:2d} 层  R² 问坐标 {mm["coords"]["r2"]:.3f} / 只有名字 {mm["name"]["r2"]:.3f} / 打乱 {mm["shuffled"]["r2"]:.3f} / 未训练 {mm["random"]["r2"]:.3f}'
            f'   平均误差 {mm["coords"]["km"]:.0f} km（中位 {mm["coords"]["kmMed"]:.0f}）  邻近误差 {mm["coords"]["prox"]:.3f}  λ {lam["coords"][L][0]}')

    best = {v: int(np.argmax([m['r2'] for m in metrics[v]])) for v in VN}
    B = best['coords']
    for v in VN:
        log(f'{v}：最好的层 第 {best[v]} 层，R² {metrics[v][best[v]]["r2"]:.3f}，平均误差 {metrics[v][best[v]]["km"]:.0f} km')

    # 各大洲的误差（默认变体，每层的中位数 / 平均数）
    ct = np.array([CONT.index(r['ct']) for r in rows])
    by_ct = [[[round(float(np.median(km_main[L][ct == c]))), round(float(km_main[L][ct == c].mean()))]
              for c in range(len(CONT))] for L in range(NL)]
    log('各大洲（最好的层）中位 / 平均误差 km：', dict(zip(CONT_ZH, by_ct[B])))

    # 按国家整块留出（论文 4.1 节）：只在最好的层做。城市数 ≥ 20 的国家逐个留出
    Xb = torch.tensor(np.asarray(acts['trained', 'coords'][B + 1], np.float64), device=DEV)
    Yb = torch.tensor(Y, device=DEV)
    ccs = np.array([r['cc'] for r in rows])
    hold = []
    nominal = preds[0, B]
    prox_nom = proximity(Y, nominal)
    for cc in sorted(set(ccs)):
        msk = ccs == cc
        if msk.sum() < 20:
            continue
        P = ridge_fit(Xb[~msk], Yb[~msk])
        ph = ridge_predict(P, Xb[msk]).cpu().numpy()
        mix = nominal.copy()
        mix[msk] = ph
        hold.append(dict(cc=cc, n=int(msk.sum()), kmNominal=round(float(haversine(Y[msk], nominal[msk]).mean())),
                         kmHeld=round(float(haversine(Y[msk], ph).mean())), proxNominal=round(float(prox_nom[msk].mean()), 4),
                         proxHeld=round(float(proximity(Y, mix)[msk].mean()), 4)))
    hsum = dict(countries=len(hold), cities=int(sum(h['n'] for h in hold)),
                kmNominal=round(float(np.mean([h['kmNominal'] for h in hold]))), kmHeld=round(float(np.mean([h['kmHeld'] for h in hold]))),
                proxNominal=round(float(np.mean([h['proxNominal'] for h in hold])), 4), proxHeld=round(float(np.mean([h['proxHeld'] for h in hold])), 4))
    log(f'按国家留出（{hsum["countries"]} 国 / {hsum["cities"]} 城）：平均误差 {hsum["kmNominal"]} → {hsum["kmHeld"]} km，'
        f'邻近误差 {hsum["proxNominal"]:.3f} → {hsum["proxHeld"]:.3f}')

    # 算式：FEATURED 城市 × 28 层，每个城市一个文件（只存 .gz，打开算式面板时才取）
    for old in (out / 'formula').glob('*.gz') if (out / 'formula').exists() else []:
        old.unlink()
    for i in feat:
        r = rows[i]
        ids, k = encode(tok, 'coords', r['name'])
        pieces = [tok.decode([t]) for t in ids[len(ids) - k:]]
        layers = [formula(np.asarray(acts['trained', 'coords'][L + 1, i], np.float64), main_probes[L][fold[i]], preds[0, L, i])
                  for L in range(NL)]
        body = dict(id=r['id'], name=r['name'], zh=r['zh'], fold=int(fold[i]), lat=r['lat'], lon=r['lon'],
                    tokens=pieces, lam=[lam['coords'][L][fold[i]] for L in range(NL)], layers=layers)
        write(out / 'formula' / f'{r["id"]}.json', json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode(), gz_only=True)

    cc_list = sorted(set(ccs))
    cn = country_names(cc_list)
    meta = dict(
        model='Qwen3-0.6B', layers=NL, dModel=D, folds=FOLDS, n=n, best=best, topk=TOPK,
        templates=TEMPLATES, variants=VN, lambdas=[g4(x) for x in LAMBDAS],
        metrics=metrics, lam=lam, emb=emb, byContinent=by_ct, holdout=dict(layer=B, summary=hsum, countries=hold),
        paper=PAPER, continents=CONT_ZH,
        countries=[[c, cn[c]] for c in cc_list],
        cities=dict(
            id=[r['id'] for r in rows], name=[r['name'] for r in rows],
            cc=[cc_list.index(r['cc']) for r in rows], ct=[int(c) for c in ct],
            lat=[round(r['lat'], 3) for r in rows], lon=[round(r['lon'], 3) for r in rows],
            pop=[r['pop'] for r in rows], fold=[int(x) for x in fold],
            featured=[[i, rows[i]['zh']] for i in feat],
        ),
        land=json.loads((work / 'land.json').read_text()),
        pred=dict(file='pred.bin', dtype='int16', scale=0.1, shape=[len(VN), NL, n, 2], layout='layer-delta, byteplanes'),
    )
    write(out / 'probe.json', json.dumps(meta, ensure_ascii=False, separators=(',', ':')).encode())
    write(out / 'pred.bin', pack_pred(preds))
    gz = {p.name: p.stat().st_size for p in out.glob('*.gz')}
    fz = sum(p.stat().st_size for p in (out / 'formula').glob('*.gz'))
    log(f'写出 {out}：{gz}，算式 {len(feat)} 个城市共 {fz / 1024:.0f} KB；gzip 合计 {(sum(gz.values()) + fz) / 1024:.0f} KB')


if __name__ == '__main__':
    main()
