"""检查「延伸学习」页（public/learn/）用到的每一个外部链接是不是真的能打开。

收集范围：public/learn/data/*.json 里所有 http(s) 字符串、public/learn/index.html 里的 href、
public/learn/snippets/ 里代码用到的网址。逐个用浏览器 UA 发 GET（自动跟随跳转，走环境变量里的代理），
记录状态码、跳转后的地址、页面 <title> 和检查时间，写到 tools/learn/links_checked.json。

另外两类做“深一层”的核对：
  · B 站视频：页面里要真的有视频标题（被删的视频也会返回 200）；
  · YouTube：再请求一次 oEmbed，视频不存在或不公开时会返回 4xx。

用法：python tools/learn/check_links.py [--workers 8] [--stamp]
  --stamp  全部通过时，把 public/learn/data/resources.json 里的 "checked" 改成今天的日期（页面上显示）
有链接打不开时退出码为 1。只用标准库。
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import gzip
import html
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LEARN = ROOT / "public" / "learn"
OUT = pathlib.Path(__file__).resolve().parent / "links_checked.json"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
URL_RE = re.compile(r"https?://[^\s\"'<>()（）]+")


def collect():
    urls = []

    def walk(v):
        if isinstance(v, str):
            urls.extend(m.group(0) for m in URL_RE.finditer(v))
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)

    for f in sorted((LEARN / "data").glob("*.json")):
        walk(json.loads(f.read_text(encoding="utf-8")))
    page = LEARN / "index.html"
    if page.exists():
        urls += re.findall(r'href="(https?://[^"]+)"', page.read_text(encoding="utf-8"))
    # 代码片段里下载 / 克隆的地址在 snippets.json 的 "links" 里逐个列出（代码里有的是拼出来的），上面已经收进来了
    seen, out = set(), []
    for u in urls:
        u = u.rstrip(".,;，。、")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def fetch(url, accept="text/html,application/xhtml+xml,*/*;q=0.8", limit=8_000_000):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": accept, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8", "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.bilibili.com/" if "bilibili.com" in url else "https://www.google.com/",
    })
    with urllib.request.urlopen(req, timeout=40) as r:
        body = r.read(limit)
        enc = (r.headers.get("content-encoding") or "").lower()
        try:
            if enc == "gzip" or body[:2] == b"\x1f\x8b":
                body = gzip.decompress(body)
            elif enc == "deflate":
                body = zlib.decompress(body)
        except (OSError, EOFError, zlib.error):
            pass  # 只截了一部分时解压不完整，不影响状态码
        return r.status, r.url, r.headers.get("content-type", ""), body.decode("utf-8", "replace")


def title_of(text):
    m = re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(m.group(1))).strip()[:200] if m else ""


def check_once(url):
    status, final, ctype, text = fetch(url)
    rec = {"url": url, "status": status, "final_url": final, "title": title_of(text) if "html" in ctype else "", "content_type": ctype.split(";")[0]}
    ok = 200 <= status < 300
    note = ""
    if ok and "bilibili.com/video/" in url:
        t = rec["title"]
        if not t or "视频去哪了" in t or t.startswith("哔哩哔哩 (゜-゜)"):
            ok, note = False, "B 站页面里没有视频标题（可能已删除）"
        else:
            rec["title"] = t.replace("_哔哩哔哩_bilibili", "")
    if ok and re.search(r"youtube\.com/watch|youtu\.be/", url):
        try:
            s, _, _, body = fetch("https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(url, safe=""), accept="application/json")
            meta = json.loads(body)
            rec["title"] = meta.get("title", "")
            rec["author"] = meta.get("author_name", "")
            note = "oEmbed 200"
        except urllib.error.HTTPError as e:
            ok, note = False, f"oEmbed {e.code}（视频不存在或不公开）"
    rec["ok"] = ok
    if note:
        rec["note"] = note
    return rec


def check(url, tries=3):
    last = None
    for i in range(tries):
        try:
            rec = check_once(url)
        except urllib.error.HTTPError as e:
            rec = {"url": url, "status": e.code, "final_url": "", "title": "", "ok": False, "note": f"HTTP {e.code}"}
        except Exception as e:  # 超时、连接被重置等
            rec = {"url": url, "status": 0, "final_url": "", "title": "", "ok": False, "note": repr(e)[:160]}
        rec["checked_at"] = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
        rec["tries"] = i + 1
        if rec["ok"]:
            return rec
        last = rec
        if rec["status"] in (404, 410):
            break
        time.sleep(3 * (i + 1))
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stamp", action="store_true")
    a = ap.parse_args()
    urls = collect()
    print(f"共 {len(urls)} 个外部链接，代理：{os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or '无'}")
    with cf.ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(check, urls))
    bad = [r for r in results if not r["ok"]]
    for r in results:
        mark = "✓" if r["ok"] else "✗"
        print(f"{mark} {r['status']:>3}  {r['url']}" + (f"\n        {r.get('note')}" if not r["ok"] else ""))
    report = {
        "checked_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "tool": "tools/learn/check_links.py",
        "total": len(results), "ok": len(results) - len(bad), "failed": len(bad),
        "results": results,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n通过 {report['ok']} / {report['total']}，结果写入 {OUT.relative_to(ROOT)}")
    if a.stamp and not bad:
        res = LEARN / "data" / "resources.json"
        t = res.read_text(encoding="utf-8")
        res.write_text(re.sub(r'"checked": "[^"]*"', f'"checked": "{dt.date.today().isoformat()}"', t, count=1), encoding="utf-8")
        print("已更新 resources.json 的检查日期")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
