#!/usr/bin/env python3
"""多模态视频用的本地静态服务器（从 ../serve.py 改来，互不影响）。

以仓库根目录为站点根（/tools/video/mm/film.html 能 import /public/...），另外：
  /ext/fonts/  → D 盘上的完整 Noto Serif SC / Noto Sans SC（SIL OFL，太大不进 git）
  /ext/mm/     → 多模态视频的工作目录（片尾录屏 sitecap/ 等）

    python tools/video/mm/serve.py [--port 8798] [--fonts /mnt/d/cjc/videos/llm-inference/fonts] [--work /mnt/d/cjc/videos/multimodal]
"""
import argparse
import functools
import http.server
import os
import posixpath
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class Server(http.server.ThreadingHTTPServer):
    # 几个浏览器同时刷新页面、拉几百个文件：默认的监听队列只有 5，会被打满
    request_queue_size = 256
    daemon_threads = True


class Handler(http.server.SimpleHTTPRequestHandler):
    fonts_dir = None
    work_dir = None

    def translate_path(self, path):
        p = urllib.parse.unquote(urllib.parse.urlsplit(path).path)
        if p.startswith("/ext/fonts/") and self.fonts_dir:
            return os.path.join(self.fonts_dir, posixpath.basename(p))
        if p.startswith("/ext/mm/") and self.work_dir:
            rest = posixpath.normpath(p[len("/ext/mm/"):]).lstrip("/")
            if rest.startswith(".."):
                return os.path.join(self.work_dir, "__nope__")
            return os.path.join(self.work_dir, rest)
        return super().translate_path(path)

    def end_headers(self):
        # 渲染时反复刷新页面，别让浏览器缓存旧脚本
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # 安静一点
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8798)
    ap.add_argument("--fonts", default="/mnt/d/cjc/videos/llm-inference/fonts")
    ap.add_argument("--work", default="/mnt/d/cjc/videos/multimodal")
    a = ap.parse_args()
    Handler.fonts_dir = a.fonts
    Handler.work_dir = a.work
    Handler.extensions_map = {**Handler.extensions_map, ".js": "text/javascript", ".mjs": "text/javascript", ".otf": "font/otf", ".woff2": "font/woff2", ".json": "application/json", ".gz": "application/gzip"}
    h = functools.partial(Handler, directory=str(ROOT))
    with Server(("127.0.0.1", a.port), h) as srv:
        print(f"serving {ROOT} on http://127.0.0.1:{a.port}/  (fonts: {a.fonts}, work: {a.work})", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
