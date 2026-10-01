#!/usr/bin/env python3
"""「揭开黑箱」的访问计数（只用标准库）。

POST /hit   -> 记一次访问：同一个 IP 每天只算一个访客（IP 只在内存里去重，不写盘），每次都算一次浏览
GET  /stats -> {"total": 累计访客人次, "today": 今天的访客数, "pv": 累计浏览次数, "date": "YYYY-MM-DD"}

只监听 127.0.0.1，由 nginx 把 /blackbox/api/ 转发过来（deploy/nginx-blackbox.conf）。
状态存在一个 JSON 文件里，原子写入。
"""
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DATA = os.environ.get("COUNTER_FILE", "/srv/blackbox/data/visits.json")
PORT = int(os.environ.get("COUNTER_PORT", "8022"))
TZ = timezone(timedelta(hours=8))  # 按北京时间分天

lock = threading.Lock()
seen: set[str] = set()


def load():
    try:
        with open(DATA) as f:
            s = json.load(f)
    except (OSError, ValueError):
        s = {}
    return {"total": int(s.get("total", 0)), "today": int(s.get("today", 0)), "pv": int(s.get("pv", 0)), "date": str(s.get("date", ""))}


state = load()


def save():
    tmp = DATA + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, DATA)


def roll_day():
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    if state["date"] != today:
        state.update(date=today, today=0)
        seen.clear()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.rstrip("/") != "/stats":
            return self._send(404, {"error": "not found"})
        with lock:
            roll_day()
            return self._send(200, state)

    def do_POST(self):
        if self.path.rstrip("/") != "/hit":
            return self._send(404, {"error": "not found"})
        ip = self.headers.get("X-Real-IP") or self.client_address[0]
        with lock:
            roll_day()
            state["pv"] += 1
            if ip not in seen:
                seen.add(ip)
                state["today"] += 1
                state["total"] += 1
            save()
            return self._send(200, state)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
