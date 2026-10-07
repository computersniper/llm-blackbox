"""片尾用的二维码：本地生成（segno），指向多模态页 https://caijiechao.com/blackbox/multimodal/（qr-multimodal.svg）。

    /mnt/d/cjc/venvs/blackbox/bin/python tools/video/mm/make_qr.py

纠错等级 M，四周留 4 格静区；深色模块纯黑、底色纯白，对比度最高，手机对着屏幕也能扫。
"""
import pathlib

import segno

HERE = pathlib.Path(__file__).parent
URL = 'https://caijiechao.com/blackbox/multimodal/'
out = HERE / 'qr-multimodal.svg'
qr = segno.make(URL, error='m', micro=False)
qr.save(str(out), kind='svg', scale=10, border=4, dark='#000000', light='#ffffff', xmldecl=False, svgns=True, nl=False, omitsize=True)   # 不写宽高、写 viewBox：放进多大的框都整张缩放，不会裁掉静区
print(f'{out}: {URL} · version {qr.version}, {qr.symbol_size(border=4)[0]} modules incl. border')
