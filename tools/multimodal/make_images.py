"""画出多模态页面用的几张图，并把两张公共领域照片缩小后放进 tools/multimodal/images/。

用法：
    python tools/multimodal/make_images.py --sans NotoSansSC.ttf --serif NotoSerifSC-SemiBold.otf \
        --aldrin Aldrin_Apollo_11_original.jpg --starry Van_Gogh_-_Starry_Night.jpg

画出来的图都是故意“不整齐”的尺寸（比如 400×400），好让网页展示处理器真实的缩放：
Qwen3-VL 会把边长就近取整到 32 的倍数（16 像素的块 × 2×2 合并）。
字体用的是 Noto Sans SC / Noto Serif SC（SIL OFL）。照片来源见 README。
"""
import argparse
import math
import pathlib
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = pathlib.Path(__file__).resolve().parent / "images"


def font(path, size, weight=None):
    f = ImageFont.truetype(str(path), size)
    if weight is not None:
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
    return f


def star_points(cx, cy, r_out, r_in, n=5, rot=-math.pi / 2):
    pts = []
    for i in range(n * 2):
        r = r_out if i % 2 == 0 else r_in
        a = rot + i * math.pi / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def soft_bg(w, h, top, bottom, seed=0):
    """竖直渐变 + 一点颗粒，避免纯色块。"""
    img = Image.new("RGB", (w, h))
    px = img.load()
    rnd = random.Random(seed)
    for y in range(h):
        t = y / max(1, h - 1)
        base = [round(top[k] + (bottom[k] - top[k]) * t) for k in range(3)]
        for x in range(w):
            n = rnd.randint(-3, 3)
            px[x, y] = tuple(max(0, min(255, c + n)) for c in base)
    return img


def shapes(a):
    w = h = 400
    img = soft_bg(w, h, (34, 40, 56), (24, 28, 42), 1)
    d = ImageDraw.Draw(img)
    d.ellipse((46, 52, 176, 182), fill=(222, 64, 72))                          # 红色圆形
    d.rectangle((226, 58, 348, 180), fill=(62, 118, 226))                      # 蓝色正方形
    d.polygon([(110, 226), (44, 346), (176, 346)], fill=(240, 196, 60))         # 黄色三角形
    d.polygon(star_points(288, 292, 70, 29), fill=(70, 190, 110))              # 绿色五角星
    img.save(OUT / "shapes.png")


def apple(d, cx, cy, r, color, dark):
    d.ellipse((cx - r, cy - r * 0.92, cx + r, cy + r), fill=color)
    d.ellipse((cx - r * 0.55, cy - r * 0.62, cx - r * 0.15, cy - r * 0.2), fill=tuple(min(255, c + 45) for c in color))
    d.ellipse((cx - r * 0.18, cy - r * 1.02, cx + r * 0.18, cy - r * 0.76), fill=dark)
    d.line((cx, cy - r * 0.86, cx + r * 0.12, cy - r * 1.3), fill=(96, 62, 36), width=max(3, int(r * 0.12)))
    d.polygon([(cx + r * 0.1, cy - r * 1.12), (cx + r * 0.62, cy - r * 1.34), (cx + r * 0.36, cy - r * 0.98)], fill=(84, 160, 70))


def apples(a):
    w, h = 460, 330
    img = soft_bg(w, h, (30, 34, 46), (26, 28, 38), 2)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 238, w, h), fill=(92, 62, 44))                             # 木桌
    for y in range(246, h, 14):
        d.line((0, y, w, y + 3), fill=(80, 53, 37), width=2)
    xs = [70, 150, 230, 310, 390]
    cols = [(206, 46, 52), (214, 58, 48), (120, 176, 60), (200, 40, 58), (218, 70, 44)]
    for x, c in zip(xs, cols):
        apple(d, x, 206, 36, c, tuple(int(v * 0.6) for v in c))
    img.save(OUT / "apples.png")


def poem(a):
    w, h = 430, 300
    img = soft_bg(w, h, (38, 36, 44), (30, 28, 36), 3)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((18, 18, w - 18, h - 18), radius=10, outline=(150, 128, 92), width=2)
    f = font(a.serif, 52)
    for i, line in enumerate(["床前明月光，", "疑是地上霜。"]):
        d.text((w / 2 + 12, 108 + i * 86), line, font=f, fill=(236, 226, 204), anchor="mm")
    d.ellipse((w - 76, 36, w - 42, 70), fill=(232, 214, 160))                   # 一轮小月亮
    d.ellipse((w - 68, 32, w - 34, 66), fill=(38, 36, 44))
    img.save(OUT / "poem.png")


def chart(a):
    w, h = 480, 350
    img = soft_bg(w, h, (28, 32, 44), (24, 27, 38), 4)
    d = ImageDraw.Draw(img)
    ft = font(a.sans, 26, 700)
    fl = font(a.sans, 21, 500)
    fv = font(a.sans, 19, 500)
    d.text((w / 2, 34), "水果销量（箱）", font=ft, fill=(228, 232, 240), anchor="mm")
    x0, y0, y1 = 64, 300, 76
    d.line((x0, y0, w - 30, y0), fill=(150, 160, 180), width=2)
    d.line((x0, y0, x0, y1), fill=(150, 160, 180), width=2)
    vmax = 50
    for v in range(0, 51, 10):
        y = y0 - (y0 - y1) * v / vmax
        d.line((x0 - 5, y, x0, y), fill=(150, 160, 180), width=2)
        d.text((x0 - 10, y), str(v), font=fv, fill=(170, 178, 196), anchor="rm")
        if v:
            d.line((x0 + 2, y, w - 30, y), fill=(52, 58, 74), width=1)
    data = [("苹果", 32, (214, 76, 76)), ("香蕉", 45, (226, 196, 70)), ("橙子", 18, (232, 138, 54)), ("葡萄", 27, (146, 102, 206))]
    bw, gap = 62, 34
    for i, (name, v, c) in enumerate(data):
        bx = x0 + 30 + i * (bw + gap)
        by = y0 - (y0 - y1) * v / vmax
        d.rectangle((bx, by, bx + bw, y0 - 1), fill=c)
        d.text((bx + bw / 2, by - 14), str(v), font=fv, fill=(230, 234, 242), anchor="mm")
        d.text((bx + bw / 2, y0 + 22), name, font=fl, fill=(210, 216, 228), anchor="mm")
    img.save(OUT / "chart.png")


def night(a):
    w = h = 400
    img = soft_bg(w, h, (14, 20, 44), (30, 34, 64), 5)
    d = ImageDraw.Draw(img)
    rnd = random.Random(7)
    for _ in range(46):
        x, y = rnd.randint(0, w), rnd.randint(0, 210)
        r = rnd.choice([1, 1, 1.5, 2])
        d.ellipse((x - r, y - r, x + r, y + r), fill=(220, 226, 250))
    d.ellipse((296, 44, 360, 108), fill=(244, 230, 170))                       # 弯月
    d.ellipse((276, 34, 340, 98), fill=(16, 22, 46))
    d.rectangle((0, 316, w, h), fill=(26, 46, 34))                             # 草地
    d.rectangle((70, 212, 220, 318), fill=(150, 96, 70))                       # 房子
    d.polygon([(52, 214), (145, 140), (238, 214)], fill=(110, 50, 50))
    d.rectangle((126, 256, 164, 318), fill=(70, 46, 36))
    d.rectangle((88, 234, 118, 262), fill=(250, 214, 110))                     # 亮着灯的窗户
    d.rectangle((176, 234, 206, 262), fill=(250, 214, 110))
    d.rectangle((300, 236, 318, 320), fill=(84, 58, 40))                       # 树
    d.ellipse((258, 158, 360, 262), fill=(40, 96, 60))
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    img.save(OUT / "night.png")


def photo(src, name, long_side=640):
    im = Image.open(src).convert("RGB")
    s = long_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(OUT / name, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sans", required=True, help="Noto Sans SC（可变字重 TTF）")
    ap.add_argument("--serif", required=True, help="Noto Serif SC")
    ap.add_argument("--aldrin", help="Aldrin_Apollo_11_original.jpg（NASA，公共领域）")
    ap.add_argument("--starry", help="Van Gogh - Starry Night（公共领域）")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    shapes(a)
    apples(a)
    poem(a)
    chart(a)
    night(a)
    if a.aldrin:
        photo(a.aldrin, "moon.jpg")
    if a.starry:
        photo(a.starry, "starry.jpg")
    for p in sorted(OUT.iterdir()):
        print(p.name, Image.open(p).size)


if __name__ == "__main__":
    main()
