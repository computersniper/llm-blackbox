#!/usr/bin/env python3
"""「揭开黑箱」的访问计数（只用标准库）。全站一个总数，后台另按页面分别记录。

POST /hit?p=<页面>  记一次访问。页面是 infer / train / mm / agent / world / learn 之一（其它记成 other）。
    全站：同一个 IP 每天只算一个访客（total、today），每次都算一次浏览（pv）。
    分页面：同样按天去重记访客（uv）和浏览（pv），累计在 pages，每天的明细在 daily。
    IP 只在内存里去重，不写盘。
GET /stats         公开：{"total", "today", "pv", "date"}，只有全站数字。
GET /pages         只在服务器本机直接访问时给（经 nginx 转发的请求带 X-Real-IP，一律 403）：
                   {"site": {...}, "pages": {页面: {"pv", "uv"}}, "daily": {日期: {页面: {"pv", "uv"}}}}
    在服务器上查：curl -s 127.0.0.1:8022/pages | python3 -m json.tool

只监听 127.0.0.1，由 nginx 把 /blackbox/api/ 转发过来（deploy/nginx-blackbox.conf）。状态存在一个 JSON 文件里，原子写入。
"""
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

DATA = os.environ.get("COUNTER_FILE", "/srv/blackbox/data/visits.json")
PORT = int(os.environ.get("COUNTER_PORT", "8022"))
TZ = timezone(timedelta(hours=8))  # 按北京时间分天
PAGES = {"infer", "train", "mm", "agent", "world", "learn"}

lock = threading.Lock()
seen: set[str] = set()                 # 今天全站见过的 IP
seen_page: dict[str, set[str]] = {}    # 今天每个页面见过的 IP


def load():
    try:
        with open(DATA) as f:
            s = json.load(f)
    except (OSError, ValueError):
        s = {}
    return {
        "total": int(s.get("total", 0)), "today": int(s.get("today", 0)), "pv": int(s.get("pv", 0)), "date": str(s.get("date", "")),
        "pages": s.get("pages", {}), "daily": s.get("daily", {}),
    }


state = load()


def save():
    tmp = DATA + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, DATA)


def roll_day():
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    if state["date"] != today:
        state.update(date=today, today=0)
        seen.clear()
        seen_page.clear()


def public():
    return {k: state[k] for k in ("total", "today", "pv", "date")}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = urlsplit(self.path).path.rstrip("/")
        with lock:
            roll_day()
            if path == "/stats":
                return self._send(200, public())
            if path == "/pages":
                if self.headers.get("X-Real-IP"):          # 从外网经 nginx 来的：不给明细
                    return self._send(403, {"error": "forbidden"})
                return self._send(200, {"site": public(), "pages": state["pages"], "daily": state["daily"]})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlsplit(self.path)
        if u.path.rstrip("/") != "/hit":
            return self._send(404, {"error": "not found"})
        page = (parse_qs(u.query).get("p") or [""])[0]
        page = page if page in PAGES else "other"
        ip = self.headers.get("X-Real-IP") or self.client_address[0]
        with lock:
            roll_day()
            day = state["daily"].setdefault(state["date"], {})
            site_d = day.setdefault("_site", {"pv": 0, "uv": 0})
            page_d = day.setdefault(page, {"pv": 0, "uv": 0})
            page_all = state["pages"].setdefault(page, {"pv": 0, "uv": 0})
            state["pv"] += 1
            site_d["pv"] += 1
            page_d["pv"] += 1
            page_all["pv"] += 1
            if ip not in seen:
                seen.add(ip)
                state["today"] += 1
                state["total"] += 1
                site_d["uv"] += 1
            ps = seen_page.setdefault(page, set())
            if ip not in ps:
                ps.add(ip)
                page_d["uv"] += 1
                page_all["uv"] += 1
            save()
            return self._send(200, public())

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
