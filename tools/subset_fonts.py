"""把思源宋体裁剪成只包含本站用到的字符的 woff2，每个字重拆成两个文件：

  noto-serif-sc-<字重>-core.woff2  首屏一定会用到的字（见 core_chars），加上 ASCII 和常用标点
  noto-serif-sc-<字重>-ext.woff2   其余所有页面 HTML / JS / CSS / manifest 里出现的字

CSS 里两段 @font-face 用 unicode-range 分开（ext 声明为 core 之外的全部码位），浏览器只有页面真的渲染到 ext 里的字时
才去下载 ext。脚本会直接改写 public/css/app.css 和 public/train/css/train.css 里 "BB Serif" 的 @font-face。

用法：python tools/subset_fonts.py <NotoSerifSC-Black.otf> <NotoSerifSC-SemiBold.otf>
需要 fonttools 和 brotli（pip install fonttools brotli）。改了任何页面的文案后重新跑一次。
"""
import json
import pathlib
import re
import sys

from fontTools import subset

ROOT = pathlib.Path(__file__).resolve().parent.parent
PUB = ROOT / "public"
PAGES = [PUB / "index.html", PUB / "train" / "index.html", PUB / "multimodal" / "index.html", PUB / "agent" / "index.html"]
CSS = [PUB / "css" / "app.css", PUB / "train" / "css" / "train.css"]
BASE = set(chr(c) for c in range(0x20, 0x7F)) | set(" 　，。、；：？！“”‘’（）《》【】—…·×−→↓↑≈⊕Σ√")
# 训练页第一屏（流水线）画在画布上的衬线体文字；画布不经过 HTML，单独列出
CANVAS_FIRST = {"900": ["一个大模型是怎么训练出来的"], "600": ["预训练 · 学会语言", "监督微调 · 学会对话", "偏好对齐 · 学会答得更好"]}


def cjk(text):
    return {ch for ch in text if ord(ch) > 0x7F}


def visible_text(html):
    """HTML 里会显示出来的文字：去掉 <head>、<script>、<style> 和所有标签（属性里的 title / aria-label 不渲染）"""
    html = re.sub(r"<head>.*?</head>|<script.*?</script>|<style.*?</style>", "", html, flags=re.S)
    return re.sub(r"<[^>]+>", "", html)


def sources():
    """所有页面（首页和 train/、multimodal/、agent/）的 HTML / JS / CSS（不含 vendor）"""
    return [f for ext in ("*.html", "*.js", "*.css") for f in PUB.rglob(ext) if "vendor" not in f.parts]


def core_chars(weight):
    """首屏一定会用到的字。
    900 只用在大标题上：各页面开场卡片的 <h1>、图鉴的 <h2>（藏在透明层里，也会排版）、训练页画布的大标题，core 只放这些。
    600 用在品牌名、操作说明、任务卡片、卡片标题等处：core 放四个页面 HTML 里静态出现的全部文字，加上智能体页的任务卡片
    和训练页第一屏画布上的卡片标题。"""
    chars = set(BASE)
    if weight == "900":
        for f in sources():
            for m in re.finditer(r"<h([12])\b[^>]*>(.*?)</h\1>", f.read_text(encoding="utf-8"), flags=re.S):
                chars |= cjk(re.sub(r"\$\{[^}]*\}|<[^>]+>", "", m.group(2)))
    else:
        for f in PAGES:
            chars |= cjk(visible_text(f.read_text(encoding="utf-8")))
        agent = PUB / "agent" / "data" / "manifest.json"
        if agent.exists():
            for t in json.loads(agent.read_text(encoding="utf-8")).get("tasks", []):
                chars |= set(t.get("title", ""))
    for s in CANVAS_FIRST[weight]:
        chars |= set(s)
    return chars


def all_chars():
    chars = set(BASE)
    # 所有页面的 HTML / JS / CSS，加上子页面 manifest 里的标题（任务卡片等用衬线体）
    files = [f for ext in ("*.html", "*.js", "*.css") for f in PUB.rglob(ext)]
    files += [f for f in PUB.glob("*/data/manifest.json")]
    for f in files:
        chars |= cjk(f.read_text(encoding="utf-8"))
    return chars


def ranges(cps):
    """码位集合 → unicode-range 写法（连续的合并成区间）"""
    cps = sorted(cps)
    out, i = [], 0
    while i < len(cps):
        j = i
        while j + 1 < len(cps) and cps[j + 1] == cps[j] + 1:
            j += 1
        out.append(f"U+{cps[i]:X}" if i == j else f"U+{cps[i]:X}-{cps[j]:X}")
        i = j + 1
    return ",".join(out)


def complement(cps):
    """core 之外的全部码位（0 – 10FFFF），写成区间"""
    out, prev = [], -1
    for c in sorted(cps):
        if c > prev + 1:
            out.append(f"U+{prev + 1:X}" if c - 1 == prev + 1 else f"U+{prev + 1:X}-{c - 1:X}")
        prev = c
    if prev < 0x10FFFF:
        out.append(f"U+{prev + 1:X}-10FFFF")
    return ",".join(out)


def save(src, text, out):
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.name_IDs = ["*"]
    opts.notdef_outline = True
    # 只留默认的排版特性（kern / vert / locl 等，不再保留 aalt、jp78 之类的异体字替换，它们会把大量用不到的字形拉进来）；
    # 去掉 hinting、展开子程序，woff2 压得更小，网页上看不出差别
    opts.hinting = False
    opts.desubroutinize = True
    font = subset.load_font(src, opts)
    sub = subset.Subsetter(opts)
    sub.populate(text=text)
    sub.subset(font)
    subset.save_font(font, str(out), opts)
    return out.stat().st_size


def main():
    every = all_chars()
    fonts = PUB / "fonts"
    for old in fonts.glob("noto-serif-sc-*.woff2"):
        old.unlink()
    faces = []
    for src, weight in zip(sys.argv[1:3], ("900", "600")):
        core = core_chars(weight)
        ext = every - core
        print(f"{weight}：core {len(core)} 个字符，ext {len(ext)} 个字符")
        cps = [ord(c) for c in core]
        for part, text, rng in (("core", core, ranges(cps)), ("ext", ext, complement(cps))):
            out = fonts / f"noto-serif-sc-{weight}-{part}.woff2"
            size = save(src, "".join(sorted(text)), out)
            print(" ", out.name, size // 1024, "KB")
            faces.append((weight, part, rng))
    for css in CSS:
        src = css.read_text(encoding="utf-8")
        prefix = re.search(r'url\("([^"]*)noto-serif-sc-', src).group(1)
        block = [
            f'@font-face {{ font-family: "BB Serif"; src: url("{prefix}noto-serif-sc-{w}-{p}.woff2") format("woff2"); font-weight: {w}; font-display: swap; unicode-range: {r}; }}'
            for w, p, r in faces
        ]
        lines = src.split("\n")
        idx = [i for i, ln in enumerate(lines) if ln.startswith('@font-face { font-family: "BB Serif";')]
        lines[idx[0]:idx[-1] + 1] = block
        css.write_text("\n".join(lines), encoding="utf-8")
        print("已更新", css.relative_to(ROOT))


if __name__ == "__main__":
    main()
