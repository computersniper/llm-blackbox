"""世界模型页 · 大模型里的世界地图：下载并清洗原始数据（城市名 + 经纬度、海岸线）。

用法：
    python tools/world/probe_data.py                       # 默认放 /mnt/d/cjc/world-model/probe/
    python tools/world/probe_data.py --raw /path/to/raw --out /path/to/probe

数据来源（都可以自由再分发，致谢见 README）：
- GeoNames cities15000（人口 ≥ 15,000 的城市，CC BY 4.0）+ countryInfo.txt（国家 → 大洲）
  https://download.geonames.org/export/dump/
- Natural Earth 1:110m 陆地多边形（公共领域）
  https://github.com/nvkelso/natural-earth-vector

做的事：
1. 只留“有人居住的地方”（P 类），去掉城区片区 / 历史 / 废弃的地点（PPLX / PPLH / PPLQ / PPLW / PPLCH）；
2. 名字规范成英文文本里常见的写法：弯引号换成直引号；名字里有 Latin-1 以外的字符
   （GeoNames 的音译常带长音符号，如 Machilīpatnam）就改用它的 ASCII 写法；
3. 同名城市按 Gurnee & Tegmark (2023) 处理美国地名的做法：只有最大的那个人口至少是第二大的 2 倍才保留它，
   否则全部丢掉（“Valencia”既是西班牙也是委内瑞拉的大城市，只看名字分不清）；
4. 每个大洲按人口从大到小挑，每个国家最多 CAP 个，凑够各大洲的名额（避免中国 / 印度 / 美国占满）；
   页面上“可以看算式”的几十个知名城市（FEATURED）一定入选；
5. 大洲按 GeoNames 的国家归属，只有俄罗斯按经度拆开：乌拉尔山（东经 60°）以东算亚洲；
6. 写出 cities.tsv（给 probe_export.py 用）和 land.json（Natural Earth 陆地轮廓，量化到 0.1°）。
"""
import argparse
import collections
import json
import pathlib
import urllib.request
import zipfile

NE = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/'
URLS = {
    'cities15000.zip': 'https://download.geonames.org/export/dump/cities15000.zip',
    'countryInfo.txt': 'https://download.geonames.org/export/dump/countryInfo.txt',
    'ne_110m_land.geojson': NE + 'ne_110m_land.geojson',
}

# 各大洲名额（合计约 4000）；不够的大洲（大洋洲城市本来就少）有多少取多少
QUOTA = {'AS': 1350, 'EU': 850, 'AF': 650, 'NA': 550, 'SA': 400, 'OC': 200}
CAP = 80          # 每个国家最多入选的城市数
DROP_CODES = {'PPLX', 'PPLH', 'PPLQ', 'PPLW', 'PPLCH'}

# 页面上可以展开“算式”的城市：GeoNames 编号 → 中文名。都是名字不含糊的知名城市，覆盖各大洲，
# 也特意放了几个偏远的（雷克雅未克、安克雷奇、乌斯怀亚、檀香山），看探针在边缘地带错得多厉害
FEATURED = {
    1816670: '北京', 1796236: '上海', 1815286: '成都', 1529102: '乌鲁木齐', 1280737: '拉萨',
    1819729: '香港', 1850147: '东京', 1835848: '首尔', 1880252: '新加坡', 1609350: '曼谷',
    1275339: '孟买', 1273294: '德里', 292223: '迪拜', 112931: '德黑兰', 745044: '伊斯坦布尔',
    1642911: '雅加达', 2028462: '乌兰巴托', 2013348: '符拉迪沃斯托克',
    2643743: '伦敦', 2988507: '巴黎', 2950159: '柏林', 3169070: '罗马', 3117735: '马德里',
    524901: '莫斯科', 2673730: '斯德哥尔摩', 264371: '雅典', 703448: '基辅', 3413829: '雷克雅未克',
    360630: '开罗', 2332459: '拉各斯', 184745: '内罗毕', 3369157: '开普敦', 2314302: '金沙萨',
    344979: '亚的斯亚贝巴', 2553604: '卡萨布兰卡',
    5128581: '纽约', 5368361: '洛杉矶', 4887398: '芝加哥', 6167865: '多伦多', 3530597: '墨西哥城',
    3553478: '哈瓦那', 5879400: '安克雷奇', 5856195: '檀香山',
    3448439: '圣保罗', 3435910: '布宜诺斯艾利斯', 3936456: '利马', 3688689: '波哥大', 3833367: '乌斯怀亚',
    2147714: '悉尼', 2063523: '珀斯', 2193733: '奥克兰',
}


def download(raw):
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        p = raw / name
        if p.exists():
            continue
        print('下载', url)
        p.write_bytes(urllib.request.urlopen(url, timeout=180).read())
    txt = raw / 'cities15000.txt'
    if not txt.exists():
        with zipfile.ZipFile(raw / 'cities15000.zip') as z:
            z.extract('cities15000.txt', raw)


def display_name(name, ascii_name):
    n = name.replace('’', "'").replace('‘', "'").replace('ʼ', "'").strip()
    if all(ord(c) < 256 for c in n):
        return n
    return ascii_name.replace('’', "'").strip()


def load_cities(raw):
    continent = {}
    for line in open(raw / 'countryInfo.txt', encoding='utf-8'):
        if line.startswith('#'):
            continue
        f = line.rstrip('\n').split('\t')
        continent[f[0]] = f[8]
    rows = []
    for line in open(raw / 'cities15000.txt', encoding='utf-8'):
        f = line.rstrip('\n').split('\t')
        if f[6] != 'P':
            continue
        cc = f[8]
        lat, lon = float(f[4]), float(f[5])
        ct = continent.get(cc)
        if cc == 'RU' and lon > 60:
            ct = 'AS'
        rows.append(dict(id=int(f[0]), name=display_name(f[1], f[2]), lat=lat, lon=lon, code=f[7], cc=cc,
                         ct=ct, pop=int(f[14] or 0)))
    return rows


def select(rows):
    by_name = collections.defaultdict(list)
    for r in rows:
        by_name[r['name'].casefold()].append(r)
    uniq = []
    dropped = 0
    for rs in by_name.values():
        rs.sort(key=lambda r: -r['pop'])
        if len(rs) == 1 or rs[0]['pop'] >= 2 * rs[1]['pop']:
            uniq.append(rs[0])
        else:
            dropped += len(rs)
    ok = [r for r in uniq if r['code'] not in DROP_CODES and r['ct'] in QUOTA and r['pop'] >= 15000
          and not any(c.isdigit() for c in r['name']) and '(' not in r['name'] and ',' not in r['name']]
    ids = {r['id'] for r in ok}
    miss = [i for i in FEATURED if i not in ids]
    if miss:
        raise SystemExit(f'FEATURED 里的城市没通过清洗：{miss}')
    ok.sort(key=lambda r: -r['pop'])
    picked, per_cc, per_ct = [], collections.Counter(), collections.Counter()
    for r in ok:
        if r['id'] in FEATURED:
            picked.append(r)
            per_cc[r['cc']] += 1
            per_ct[r['ct']] += 1
    for r in ok:
        if r['id'] in FEATURED or per_cc[r['cc']] >= CAP or per_ct[r['ct']] >= QUOTA[r['ct']]:
            continue
        picked.append(r)
        per_cc[r['cc']] += 1
        per_ct[r['ct']] += 1
    print(f'同名去掉 {dropped} 个；入选 {len(picked)} 个城市', dict(per_ct))
    print('人口最少的入选城市：', min(r['pop'] for r in picked))
    return picked


def land_rings(raw):
    """Natural Earth 110m 陆地 → 外环列表（去掉南极洲），坐标量化到 0.1° 的整数 [lon10, lat10, ...]"""
    g = json.loads((raw / 'ne_110m_land.geojson').read_text())
    rings = []
    for feat in g['features']:
        geom = feat['geometry']
        polys = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
        for poly in polys:
            ring = poly[0]
            if max(p[1] for p in ring) < -60:
                continue
            flat = []
            last = None
            for lon, lat in ring:
                q = (round(lon * 10), round(lat * 10))
                if q != last:
                    flat += q
                    last = q
            if len(flat) >= 8:
                rings.append(flat)
    print(f'陆地轮廓 {len(rings)} 个环，{sum(len(r) for r in rings) // 2} 个点')
    return rings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='/mnt/d/cjc/world-model/probe/raw')
    ap.add_argument('--out', default='/mnt/d/cjc/world-model/probe')
    a = ap.parse_args()
    raw, out = pathlib.Path(a.raw), pathlib.Path(a.out)
    download(raw)
    picked = select(load_cities(raw))
    picked.sort(key=lambda r: (r['ct'], r['cc'], -r['pop']))
    with open(out / 'cities.tsv', 'w', encoding='utf-8') as f:
        f.write('id\tname\tcc\tcontinent\tlat\tlon\tpop\tzh\n')
        for r in picked:
            f.write(f"{r['id']}\t{r['name']}\t{r['cc']}\t{r['ct']}\t{r['lat']:.5f}\t{r['lon']:.5f}\t{r['pop']}\t{FEATURED.get(r['id'], '')}\n")
    (out / 'land.json').write_text(json.dumps(land_rings(raw), separators=(',', ':')))
    print('写出', out / 'cities.tsv', out / 'land.json')


if __name__ == '__main__':
    main()
