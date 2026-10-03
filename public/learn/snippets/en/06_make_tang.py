# Join the first 20,000 poems of the Complete Tang Poems (from chinese-poetry) into a character-level nanoGPT corpus: data/tang_char/input.txt
# Run inside the nanoGPT folder: python make_tang.py (the texts are in traditional characters; about 8 MB to download)
import json, os, urllib.parse, urllib.request

url = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/" + urllib.parse.quote("全唐诗") + "/poet.tang.{}.json"  # 全唐诗 = "Complete Tang Poems"
poems = []
for k in range(0, 20000, 1000):  # 1,000 poems per file
    for p in json.load(urllib.request.urlopen(url.format(k))):
        poems.append("".join(p["paragraphs"]))
os.makedirs("data/tang_char", exist_ok=True)
with open("data/tang_char/input.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(poems))
print(len(poems), "poems,", sum(map(len, poems)), "characters")
