"""片尾用的二维码：本地生成（segno），中文版指向网站首页（qr-blackbox.svg），英文版指向 ?lang=en（qr-blackbox-en.svg），
训练视频指向训练页（qr-blackbox-train.svg）。

    /mnt/d/cjc/venvs/blackbox/bin/python tools/video/make_qr.py

纠错等级 M，四周留 4 格静区；深色模块纯黑、底色纯白，对比度最高，手机对着屏幕也能扫。
"""
import pathlib

import segno

HERE = pathlib.Path(__file__).parent
# 中文版指向首页；英文版指向 ?lang=en
for url, name in (('https://caijiechao.com/blackbox/', 'qr-blackbox.svg'), ('https://caijiechao.com/blackbox/?lang=en', 'qr-blackbox-en.svg'), ('https://caijiechao.com/blackbox/train/', 'qr-blackbox-train.svg')):
    out = HERE / name
    qr = segno.make(url, error='m', micro=False)
    qr.save(str(out), kind='svg', scale=10, border=4, dark='#000000', light='#ffffff', xmldecl=False, svgns=True, nl=False)
    print(f'{out}: {url} · version {qr.version}, {qr.symbol_size(border=4)[0]} modules incl. border')
