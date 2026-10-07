"""把一组截图拼成带文件名的联系表（检查用）。

    python tools/video/mm/sheet.py out.jpg a.jpg b.jpg ... [--cols 3] [--w 640]
"""
import argparse
import pathlib

from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('files', nargs='+')
ap.add_argument('--cols', type=int, default=3)
ap.add_argument('--w', type=int, default=640)
a = ap.parse_args()
ims = [Image.open(f).convert('RGB') for f in a.files]
w = a.w
h = round(w * ims[0].height / ims[0].width)
rows = (len(ims) + a.cols - 1) // a.cols
sheet = Image.new('RGB', (a.cols * w, rows * (h + 22)), (12, 12, 16))
d = ImageDraw.Draw(sheet)
for i, (f, im) in enumerate(zip(a.files, ims)):
    x, y = (i % a.cols) * w, (i // a.cols) * (h + 22)
    sheet.paste(im.resize((w, h), Image.LANCZOS), (x, y + 22))
    d.text((x + 6, y + 5), pathlib.Path(f).stem, fill=(220, 220, 220))
sheet.save(a.out, quality=88)
print(a.out, sheet.size)
