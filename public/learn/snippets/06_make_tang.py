# 把 chinese-poetry 的《全唐诗》前 2 万首拼成 nanoGPT 的字符级语料 data/tang_char/input.txt
# 在 nanoGPT 目录里运行：python make_tang.py（原文是繁体字，约 8 MB 下载）
import json, os, urllib.parse, urllib.request

url = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/" + urllib.parse.quote("全唐诗") + "/poet.tang.{}.json"
poems = []
for k in range(0, 20000, 1000):  # 每个文件 1000 首
    for p in json.load(urllib.request.urlopen(url.format(k))):
        poems.append("".join(p["paragraphs"]))
os.makedirs("data/tang_char", exist_ok=True)
with open("data/tang_char/input.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(poems))
print(len(poems), "首，", sum(map(len, poems)), "字")
