"""智能体视频片尾的二维码：本地生成（segno），指向智能体页 https://caijiechao.com/blackbox/agent/。

    /mnt/d/cjc/venvs/blackbox/bin/python tools/video/agent/make_qr.py

纠错等级 M，四周留 4 格静区；深色模块纯黑、底色纯白，手机对着屏幕也能扫。
"""
import pathlib

import segno

HERE = pathlib.Path(__file__).parent
URL = 'https://caijiechao.com/blackbox/agent/'
out = HERE / 'qr-agent.svg'
qr = segno.make(URL, error='m', micro=False)
qr.save(str(out), kind='svg', scale=10, border=4, dark='#000000', light='#ffffff', xmldecl=False, svgns=True, nl=False)
# 加上 viewBox：片尾把它缩放到 340 像素见方，没有 viewBox 会被裁掉一圈
n = qr.symbol_size(border=4)[0] * 10
svg = out.read_text().replace('<svg ', f'<svg viewBox="0 0 {n} {n}" ', 1)
out.write_text(svg)
print(f'{out}: {URL} · version {qr.version}, {qr.symbol_size(border=4)[0]} modules incl. border')
