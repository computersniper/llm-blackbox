#!/usr/bin/env python3
"""视频用的本地静态服务器。

以仓库根目录为站点根（这样 /tools/video/film.html 能 import /public/js/...），
另外把 /ext/fonts/ 映射到 D 盘上的完整 Noto Serif SC（字体太大，不进 git）。

    python tools/video/serve.py [--port 8776] [--fonts /mnt/d/cjc/videos/llm-inference/fonts] [--site-en <另一份工作目录>/public]

--site-en：把 /site-en/ 映射到另一份工作目录的 public/（片尾录英文界面的推理页时用，那份界面还没合进来的时候）。
"""
import argparse
import functools
import http.server
import os
import posixpath
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = "/mnt/d/cjc/videos"


class Server(http.server.ThreadingHTTPServer):
    # 渲染时几个浏览器同时刷新页面、拉几百个文件：默认的监听队列只有 5，会被打满（内核报 SYN flooding）
    request_queue_size = 256
    daemon_threads = True


class Handler(http.server.SimpleHTTPRequestHandler):
    fonts_dir = None
    site_en = None

    def translate_path(self, path):
        p = urllib.parse.unquote(urllib.parse.urlsplit(path).path)
        if p.startswith("/ext/fonts/") and self.fonts_dir:
            name = posixpath.basename(p)
            return os.path.join(self.fonts_dir, name)
        if p.startswith("/site-en/") and self.site_en:
            rest = posixpath.normpath(p[len("/site-en/"):]).lstrip("/")
            if rest.startswith(".."):
                return os.path.join(self.site_en, "__nope__")
            return os.path.join(self.site_en, rest or "index.html")
        # 英文版：电影页把 <base> 换成 /en-public/，数据从 D 盘 data-en/ 取（和网站英文版同一份导出），其余照常走 public/
        if p.startswith("/en-public/"):
            rest = p[len("/en-public/"):]
            base = os.path.dirname(self.fonts_dir.rstrip("/")) if self.fonts_dir else ""
            for pre in ("data/en/", "data/"):
                if rest.startswith(pre):
                    cand = os.path.join(base, "data-en", rest[len(pre):])
                    if os.path.exists(cand):
                        return cand
            return str(ROOT / "public" / rest)
        # /ext/videos/<片子>/<目录>/<文件>：D 盘 /mnt/d/cjc/videos/ 下的录屏帧（训练视频等其他片子用；只读这棵目录树）
        if p.startswith("/ext/videos/"):
            rest = posixpath.normpath(p[len("/ext/videos/"):]).lstrip("/")
            if rest.startswith(".."):
                return os.path.join(VIDEOS, "__nope__")
            return os.path.join(VIDEOS, rest)
        # 片尾用的推理页录屏帧（sitecap.mjs 生成，放在字体目录旁边的 sitecap/）
        for d in ("sitecap", "sitecap-en"):
            if p.startswith(f"/ext/{d}/") and self.fonts_dir:
                name = posixpath.basename(p)
                return os.path.join(os.path.dirname(self.fonts_dir.rstrip("/")), d, name)
        return super().translate_path(path)

    def end_headers(self):
        # 渲染时反复刷新页面，别让浏览器缓存旧脚本
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # 安静一点
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8776)
    ap.add_argument("--fonts", default="/mnt/d/cjc/videos/llm-inference/fonts")
    ap.add_argument("--site-en", default=None)
    a = ap.parse_args()
    Handler.fonts_dir = a.fonts
    Handler.site_en = a.site_en
    Handler.extensions_map = {**Handler.extensions_map, ".js": "text/javascript", ".mjs": "text/javascript", ".otf": "font/otf", ".woff2": "font/woff2", ".json": "application/json"}
    h = functools.partial(Handler, directory=str(ROOT))
    with Server(("127.0.0.1", a.port), h) as srv:
        print(f"serving {ROOT} on http://127.0.0.1:{a.port}/  (fonts: {a.fonts})", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
