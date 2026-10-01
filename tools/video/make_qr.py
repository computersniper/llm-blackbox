"""片尾用的二维码：本地生成（segno），指向网站首页，写成 tools/video/qr-blackbox.svg。

    /mnt/d/cjc/venvs/blackbox/bin/python tools/video/make_qr.py

纠错等级 M，四周留 4 格静区；深色模块纯黑、底色纯白，对比度最高，手机对着屏幕也能扫。
"""
import pathlib

import segno

URL = 'https://caijiechao.com/blackbox/'
OUT = pathlib.Path(__file__).with_name('qr-blackbox.svg')

qr = segno.make(URL, error='m', micro=False)
qr.save(str(OUT), kind='svg', scale=10, border=4, dark='#000000', light='#ffffff', xmldecl=False, svgns=True, nl=False)
print(f'{OUT}: version {qr.version}, {qr.symbol_size(border=4)[0] // 1} modules incl. border')
