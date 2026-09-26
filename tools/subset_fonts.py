"""把思源宋体裁剪成只包含本站用到的字符的 woff2。

用法：python tools/subset_fonts.py <NotoSerifSC-Black.otf> <NotoSerifSC-SemiBold.otf>
需要 fonttools 和 brotli（pip install fonttools brotli）。改了页面文案后重新跑一次。
"""
import pathlib
import sys

from fontTools import subset

ROOT = pathlib.Path(__file__).resolve().parent.parent
PUB = ROOT / "public"

chars = set(chr(c) for c in range(0x20, 0x7F))
chars |= set("，。、；：？！“”‘’（）《》【】—…·×−→↓↑≈⊕Σ√")
for f in list(PUB.rglob("*.js")) + [PUB / "index.html"]:
    chars |= {ch for ch in f.read_text(encoding="utf-8") if ord(ch) > 0x7F}
text = "".join(sorted(chars))
print(f"{len(chars)} 个字符")

for src, weight in zip(sys.argv[1:3], ("900", "600")):
    out = PUB / "fonts" / f"noto-serif-sc-subset-{weight}.woff2"
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = ["*"]
    opts.name_IDs = ["*"]
    opts.notdef_outline = True
    font = subset.load_font(src, opts)
    sub = subset.Subsetter(opts)
    sub.populate(text=text)
    sub.subset(font)
    subset.save_font(font, str(out), opts)
    print(out.name, out.stat().st_size // 1024, "KB")
