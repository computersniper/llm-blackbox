#!/usr/bin/env python3
"""视频用的本地静态服务器。

以仓库根目录为站点根（这样 /tools/video/film.html 能 import /public/js/...），
另外把 /ext/fonts/ 映射到 D 盘上的完整 Noto Serif SC（字体太大，不进 git）。

    python tools/video/serve.py [--port 8776] [--fonts /mnt/d/cjc/videos/llm-inference/fonts]
"""
import argparse
import functools
import http.server
import os
import posixpath
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Handler(http.server.SimpleHTTPRequestHandler):
    fonts_dir = None

    def translate_path(self, path):
        p = urllib.parse.unquote(urllib.parse.urlsplit(path).path)
        if p.startswith("/ext/fonts/") and self.fonts_dir:
            name = posixpath.basename(p)
            return os.path.join(self.fonts_dir, name)
        # 片尾用的推理页录屏帧（sitecap.mjs 生成，放在字体目录旁边的 sitecap/）
        if p.startswith("/ext/sitecap/") and self.fonts_dir:
            name = posixpath.basename(p)
            return os.path.join(os.path.dirname(self.fonts_dir.rstrip("/")), "sitecap", name)
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
    a = ap.parse_args()
    Handler.fonts_dir = a.fonts
    Handler.extensions_map = {**Handler.extensions_map, ".js": "text/javascript", ".mjs": "text/javascript", ".otf": "font/otf", ".woff2": "font/woff2", ".json": "application/json"}
    h = functools.partial(Handler, directory=str(ROOT))
    with http.server.ThreadingHTTPServer(("127.0.0.1", a.port), h) as srv:
        print(f"serving {ROOT} on http://127.0.0.1:{a.port}/  (fonts: {a.fonts})", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
