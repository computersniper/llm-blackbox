"""世界模型页 · 大模型里的世界地图：给英文版补国家、大洲的英文名（不碰探针结果）。

probe.json 里国家只存了中文名（Unicode CLDR，经 babel），城市名本来就是 GeoNames 的英文写法。
英文模式要显示国家的英文名：从 GeoNames 的 countryInfo.txt（和 probe_data.py 下载的是同一份，CC BY 4.0）
取 Country 一列，写到 public/world/probe/data/names_en.json。台湾、香港、澳门和中文版的写法对应，
写作 “Taiwan, China / Hong Kong, China / Macao, China”。大洲的英文名按 probe.json 里大洲的顺序写死。

用法：
    python tools/world/probe_names_en.py                       # countryInfo.txt 默认在 /mnt/d/cjc/world-model/probe/raw/
    python tools/world/probe_names_en.py --raw /path/to/raw
"""
import argparse
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / 'public/world/probe/data'
URL = 'https://download.geonames.org/export/dump/countryInfo.txt'
CONTINENTS = {'亚洲': 'Asia', '欧洲': 'Europe', '非洲': 'Africa', '北美洲': 'North America', '南美洲': 'South America', '大洋洲': 'Oceania'}
SPECIAL = {'TW': 'Taiwan, China', 'HK': 'Hong Kong, China', 'MO': 'Macao, China'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='/mnt/d/cjc/world-model/probe/raw')
    args = ap.parse_args()
    p = pathlib.Path(args.raw) / 'countryInfo.txt'
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        print('下载', URL)
        p.write_bytes(urllib.request.urlopen(URL, timeout=180).read())
    names = {}
    for line in open(p, encoding='utf-8'):
        if line.startswith('#') or not line.strip():
            continue
        f = line.rstrip('\n').split('\t')
        names[f[0]] = f[4]
    meta = json.loads((DATA / 'probe.json').read_text(encoding='utf-8'))
    ccs = [c for c, _ in meta['countries']]
    missing = [c for c in ccs if c not in names and c not in SPECIAL]
    if missing:
        raise SystemExit(f'countryInfo.txt 里没有：{missing}')
    out = {
        'source': 'GeoNames countryInfo.txt (CC BY 4.0)',
        'continents': [CONTINENTS[c] for c in meta['continents']],
        'countries': {c: SPECIAL.get(c) or names[c] for c in ccs},
    }
    dst = DATA / 'names_en.json'
    dst.write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    print(f'{dst}：{len(out["countries"])} 个国家，{dst.stat().st_size:,} 字节')


if __name__ == '__main__':
    main()
