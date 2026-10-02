"""在真实沙箱里跑一个真实的编程 agent 循环（开源模型 + 工具 + 循环），把全过程录成 JSON，给 public/agent/ 回放。

用法：
    python tools/agent/record.py --model /mnt/d/cjc/model-weights/qwen3/Qwen3-4B-Instruct-2507
    python tools/agent/record.py --model ... --only fix-median      # 只录一个任务，调试用

循环和 Claude Code 这类编程 agent 是同一个模式：
    系统提示 + 工具定义 + 用户任务 → 模型生成 → 外面的程序（harness）解析 <tool_call> → 在沙箱里真的执行
    → 把结果作为 <tool_response> 接到对话末尾 → 再把整段上下文喂给模型 → …… 直到模型不再调用工具，给出最终回答。

沙箱：每个任务把 tools/agent/sandbox/<任务>/ 复制到临时目录，用 bubblewrap（bwrap）挂成 /work：
系统目录只读、只有 /work 可写、没有网络、10 秒超时、清空环境变量。

录下来的真实信号：
    - 每一轮喂给模型的完整上下文（聊天模板渲染后的真实文本）、真实词元数、按消息分段的词元数；
    - 和上一轮相比能复用多少 KV 缓存（最长公共前缀），这一轮真正要新算多少词元；
    - 模型逐词元的输出、每个词元的前 5 名候选概率（温度 1 的原始分布）；
    - 解析出的工具调用、沙箱里真实的命令输出 / 退出码 / 耗时、每次工具调用前后的文件 diff；
    - 预填充和逐词元生成的实测耗时；
    - 最后由脚本独立检查任务是否真的完成（跑测试、核对数字），失败的录制会如实记在 manifest 里。

英文版：加 --lang en，同样的沙箱项目，换成英文系统提示、英文工具说明、英文任务和英文的 harness 回报，
重新真实录制，输出到 public/agent/data/en/。
"""
import argparse
import difflib
import gzip
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SANDBOX_SRC = pathlib.Path(__file__).resolve().parent / "sandbox"
DATA = ROOT / "public" / "agent" / "data"
OUT = DATA             # --lang en 时换成 DATA / "en"
LANG = "zh"

TEMPERATURE, TOP_K, TOP_P = 0.7, 20, 0.8   # Qwen3 非思考模式的推荐采样参数
TOPN = 5               # 每个词元记录前几名候选
MAX_TURNS = 14         # 最多几轮“模型 ⇄ 工具”
MAX_NEW = 700          # 每轮最多生成多少词元
CMD_TIMEOUT = 10       # 沙箱里每条命令的超时（秒）
OUT_LIMIT = 3000       # 工具输出最多回给模型多少字符

SYSTEM = """你是一个在终端里工作的编程助手，要亲自用工具把用户的任务做完，而不是只给建议。
你可以查看、创建、修改 /work 里的文件，也可以执行 shell 命令。
工作方式：
- 每次调用工具之前，先用一句话说明你要做什么，然后紧接着调用工具。
- 一次只调用一个工具，看到结果之后再决定下一步。
- 需要写的代码要保存成文件，再用命令运行它。
- 改完代码要运行测试或命令，确认结果正确。
- 任务全部完成之后，再用简短的中文总结你做了什么。
环境：Linux，工作目录 /work，没有网络。
项目里的文件：
{files}"""

SYSTEM_EN = """You are a coding assistant working in a terminal. Use the tools to actually finish the user's task yourself instead of just giving advice.
You can view, create and modify files in /work, and you can run shell commands.
How to work:
- Before each tool call, say in one sentence what you are about to do, then call the tool right away.
- Call only one tool at a time; look at the result before deciding the next step.
- Save any code you write to a file, then run it with a command.
- After changing code, run the tests or the command to confirm the result is correct.
- When the whole task is done, briefly summarize what you did in English.
Environment: Linux, working directory /work, no network.
Files in the project:
{files}"""

TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "在工作目录 /work 里执行一条 shell 命令（没有网络，10 秒超时），返回输出和退出码。",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "要执行的命令"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "读取一个文本文件的全部内容。",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "文件路径，相对于 /work"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "创建一个新文件，或者用新内容覆盖整个文件。",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "文件路径，相对于 /work"},
            "content": {"type": "string", "description": "文件的完整内容"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "把文件里的一段原文 old_string（必须恰好出现一次）替换成 new_string。",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "文件路径，相对于 /work"},
            "old_string": {"type": "string", "description": "要被替换的原文，必须和文件里的内容一字不差"},
            "new_string": {"type": "string", "description": "替换后的新内容"}}, "required": ["path", "old_string", "new_string"]}}},
]

TOOLS_EN = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command in the working directory /work (no network, 10-second timeout) and return its output and exit code.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The command to run"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read the entire contents of a text file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path, relative to /work"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create a new file, or overwrite a whole file with new contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path, relative to /work"},
            "content": {"type": "string", "description": "The complete contents of the file"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace a passage old_string in a file (it must occur exactly once) with new_string.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path, relative to /work"},
            "old_string": {"type": "string", "description": "The original text to replace; it must match the file exactly"},
            "new_string": {"type": "string", "description": "The new text to put in its place"}}, "required": ["path", "old_string", "new_string"]}}},
]

# harness 回给模型的话、检查结果的说明：两种语言各一份（中文版的措辞保持原样）
MSG = {
    "zh": {
        "no_command": "错误：缺少参数 command",
        "timeout": "\n（命令超时：超过 {s} 秒被终止）",
        "exit": "[退出码 {code}]",
        "outside": "错误：只能访问 /work 里的文件",
        "not_found": "错误：文件不存在：{path}",
        "empty": "（空文件）",
        "no_content": "错误：缺少参数 content",
        "written": "已{verb} {path}（{n} 行）", "overwrote": "覆盖", "created": "创建",
        "no_old_new": "错误：缺少参数 old_string 或 new_string",
        "old_missing": "错误：在 {path} 里没有找到 old_string，请先读取文件，确认原文一字不差",
        "old_many": "错误：old_string 在 {path} 里出现了 {n} 次，请多带一些上下文让它唯一",
        "edited": "已修改 {path}",
        "no_tool": "错误：没有叫 {name} 的工具",
        "err_prefix": "错误",
        "clip": "\n…（输出太长，后面 {n} 个字符被截掉了）",
        "bad_json": "错误：工具调用不是合法的 JSON（{error}）",
        "no_close": "缺少 </tool_call>",
        "chk_median": "python3 -m unittest → 退出码 {code}；测试文件{same}改动；额外数据{hid}",
        "same_yes": "没有", "same_no": "被", "hid_ok": "正确", "hid_bad": "出错",
        "chk_log_nofile": "没有生成 summary.md",
        "chk_log_ok": "summary.md 里三种错误的次数都正确",
        "chk_log_bad": "这些错误的次数不对或缺失：{bad}",
        "chk_sales": "脚本{ran}；回答{told}杭州（9250 元）",
        "ran_yes": "跑通了", "ran_no": "没有跑通", "told_yes": "提到", "told_no": "没有提到",
        "chk_rename": "残留 calc_total：{left}；total_price 已定义：{defined}；测试只改了名字：{test_ok}；测试退出码 {code}",
        "none": "无",
    },
    "en": {
        "no_command": "Error: missing argument command",
        "timeout": "\n(command timed out: killed after {s} seconds)",
        "exit": "[exit code {code}]",
        "outside": "Error: only files inside /work can be accessed",
        "not_found": "Error: file not found: {path}",
        "empty": "(empty file)",
        "no_content": "Error: missing argument content",
        "written": "{verb} {path} ({n} lines)", "overwrote": "Overwrote", "created": "Created",
        "no_old_new": "Error: missing argument old_string or new_string",
        "old_missing": "Error: old_string was not found in {path}; read the file first and make sure the text matches exactly",
        "old_many": "Error: old_string occurs {n} times in {path}; include more context to make it unique",
        "edited": "Edited {path}",
        "no_tool": "Error: there is no tool named {name}",
        "err_prefix": "Error",
        "clip": "\n…(output too long; the remaining {n} characters were cut off)",
        "bad_json": "Error: the tool call is not valid JSON ({error})",
        "no_close": "missing </tool_call>",
        "chk_median": "python3 -m unittest → exit code {code}; test file {same}; extra data {hid}",
        "same_yes": "unchanged", "same_no": "modified", "hid_ok": "correct", "hid_bad": "wrong",
        "chk_log_nofile": "summary.md was not created",
        "chk_log_ok": "summary.md has the correct count for all three errors",
        "chk_log_bad": "wrong or missing counts for: {bad}",
        "chk_sales": "script {ran}; answer {told} 杭州 / Hangzhou (9250)",
        "ran_yes": "ran successfully", "ran_no": "never ran successfully", "told_yes": "names", "told_no": "does not name",
        "chk_rename": "calc_total left in: {left}; total_price defined: {defined}; test file only renamed: {test_ok}; test exit code {code}",
        "none": "none",
    },
}


def m(key, **kw):
    return MSG[LANG][key].format(**kw)


# ---------------------------------------------------------------- 任务与独立检查

def sh(work, cmd):
    """检查用：在同样的沙箱里跑一条命令。"""
    r = run_sandboxed(work, cmd)
    return r["code"], r["out"]


def check_fix_median(work, final, orig, turns):
    code, out = sh(work, "python3 -m unittest -q 2>&1")
    same_test = (work / "test_stats.py").read_text() == orig["test_stats.py"]
    # 再用几组测试里没有的数据检查一遍，防止“只对付测试”
    hid, _ = sh(work, "python3 -c 'from stats import median; assert median([1, 2]) == 1.5; assert median([5]) == 5; assert median([7, 1, 3, 9]) == 5'")
    ok = code == 0 and same_test and hid == 0
    return ok, m("chk_median", code=code, same=m("same_yes") if same_test else m("same_no"), hid=m("hid_ok") if hid == 0 else m("hid_bad"))


def check_log_errors(work, final, orig, turns):
    p = work / "summary.md"
    if not p.exists():
        return False, m("chk_log_nofile")
    text = p.read_text()
    want = {"database timeout": 4, "invalid token": 3, "disk full": 2}
    bad = []
    for kind, n in want.items():
        lines = [ln for ln in text.splitlines() if kind in ln]
        if not any(re.search(rf"(?<!\d){n}(?!\d)", ln) for ln in lines):
            bad.append(kind)
    return not bad, m("chk_log_ok") if not bad else m("chk_log_bad", bad=", ".join(bad))


def check_sales_top(work, final, orig, turns):
    # 必须真的写了脚本、真的跑通了（输出里有答案），回答也要对
    ran = [c for t in turns for c in t["calls"] if c["name"] == "bash" and ".py" in str(c["args"].get("command", ""))
           and c["info"].get("exit") == 0 and "杭州" in c["result"]]
    # 英文回答可能把城市名写成拼音、把金额写成 9,250
    told = "杭州" in final or (LANG == "en" and "hangzhou" in final.lower())
    num = "9250" in (final.replace(",", "") if LANG == "en" else final)
    ok = told and num and bool(ran)
    return ok, m("chk_sales", ran=m("ran_yes") if ran else m("ran_no"), told=m("told_yes") if told else m("told_no"))


def check_rename(work, final, orig, turns):
    left = [str(p.relative_to(work)) for p in work.rglob("*.py") if "calc_total" in p.read_text()]
    defined = "def total_price" in (work / "shop" / "cart.py").read_text()
    # 测试只允许改名字，不允许改期望值
    test_ok = (work / "test_shop.py").read_text() == orig["test_shop.py"].replace("calc_total", "total_price")
    code, out = sh(work, "python3 -m unittest -q 2>&1")
    ok = not left and defined and code == 0 and test_ok
    return ok, m("chk_rename", left=left or m("none"), defined=defined, test_ok=test_ok, code=code)


TASKS = [
    {"id": "fix-median", "title": "修好失败的测试", "icon": "bug",
     "prompt": "测试没通过，帮我找到 bug 并修好。", "check": check_fix_median,
     "blurb": "stats.py 里的中位数函数有个下标错误，测试跑不过。"},
    {"id": "log-errors", "title": "统计日志里的错误", "icon": "log",
     "prompt": "统计 logs 目录里每种 ERROR 各出现了几次，把结果写进 summary.md。", "check": check_log_errors,
     "blurb": "两个服务的日志，混着 INFO / WARN / ERROR。"},
    {"id": "sales-top", "title": "算出销售冠军", "icon": "chart",
     "prompt": "sales.csv 里哪个城市的总销售额最高？写个 Python 脚本算出来。", "check": check_sales_top,
     "blurb": "一张销售流水表：成都的单数最多，但总额最高的不是它。"},
    {"id": "rename-func", "title": "重命名一个函数", "icon": "rename",
     "prompt": "把函数 calc_total 改名为 total_price，所有用到它的地方都要改，改完跑一下测试。", "check": check_rename,
     "blurb": "函数定义在 shop/cart.py，另外两个文件也在用它。"},
]

# 英文版：同样的四个沙箱项目，任务描述换成英文
TASKS_EN = {
    "fix-median": {"title": "Fix the failing test", "prompt": "The tests are failing. Find the bug and fix it.",
                   "blurb": "The median function in stats.py has an indexing bug, so the tests fail."},
    "log-errors": {"title": "Count errors in the logs", "prompt": "Count how many times each kind of ERROR appears in the logs directory and write the results to summary.md.",
                   "blurb": "Logs from two services, with INFO / WARN / ERROR mixed together."},
    "sales-top": {"title": "Find the top-selling city", "prompt": "Which city in sales.csv has the highest total sales? Write a Python script to work it out.",
                  "blurb": "A sales ledger: Chengdu (成都) has the most orders, but not the highest total."},
    "rename-func": {"title": "Rename a function", "prompt": "Rename the function calc_total to total_price, update every place that uses it, then run the tests.",
                    "blurb": "It is defined in shop/cart.py, and two other files use it too."},
}


# ---------------------------------------------------------------- 沙箱

PASSWD = "root:x:0:0::/root:/bin/bash\nagent:x:1000:1000:agent:/work:/bin/bash\n"
GROUP = "root:x:0:\nagent:x:1000:\n"


def run_sandboxed(work, command):
    """在 bubblewrap 沙箱里执行命令：系统目录只读、只能写 /work、没有网络、清空环境变量、限时。"""
    etc = work.parent / "etc"
    etc.mkdir(exist_ok=True)
    (etc / "passwd").write_text(PASSWD)
    (etc / "group").write_text(GROUP)
    argv = [
        "bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--hostname", "sandbox",
        "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin",
        "--ro-bind", "/etc/alternatives", "/etc/alternatives",
        "--ro-bind", str(etc / "passwd"), "/etc/passwd", "--ro-bind", str(etc / "group"), "/etc/group",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--bind", str(work), "/work", "--chdir", "/work",
        "--clearenv", "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin", "--setenv", "HOME", "/work",
        "--setenv", "LANG", "C.UTF-8", "--setenv", "USER", "agent", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "bash", "-c", command,
    ]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=CMD_TIMEOUT)
        out, code, timeout = r.stdout.decode("utf-8", "replace"), r.returncode, False
    except subprocess.TimeoutExpired as e:
        out, code, timeout = (e.stdout or b"").decode("utf-8", "replace"), 124, True
    return {"out": out, "code": code, "timeout": timeout, "ms": round((time.perf_counter() - t0) * 1000, 1)}


def snapshot(work):
    snap = {}
    for p in sorted(work.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            try:
                snap[str(p.relative_to(work))] = p.read_text()
            except UnicodeDecodeError:
                snap[str(p.relative_to(work))] = None   # 二进制文件
    return snap


def diff_snap(a, b):
    out = []
    for path in sorted(set(a) | set(b)):
        x, y = a.get(path), b.get(path)
        if x == y:
            continue
        kind = "new" if x is None and path not in a else "del" if path not in b else "edit"
        hunks = []
        if isinstance(x, str) or isinstance(y, str):
            lines = list(difflib.unified_diff((x or "").splitlines(), (y or "").splitlines(), lineterm="", n=2))
            hunks = lines[2:]   # 去掉 --- / +++ 两行
        out.append({"path": path, "kind": kind, "before": x, "after": y, "hunks": hunks})
    return out


def resolve(work, path):
    p = str(path).strip()
    if p.startswith("/work/"):
        p = p[len("/work/"):]
    elif p == "/work":
        p = "."
    elif p.startswith("/"):
        return None
    full = (work / p).resolve()
    if full != work.resolve() and work.resolve() not in full.parents:
        return None
    return full


def clip(s):
    if len(s) <= OUT_LIMIT:
        return s
    return s[:OUT_LIMIT] + m("clip", n=len(s) - OUT_LIMIT)


def exec_tool(work, name, args):
    """harness：真正“动手”的部分。返回回给模型的文字和执行细节。"""
    t0 = time.perf_counter()
    info = {}
    if name == "bash":
        cmd = args.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            text = m("no_command")
        else:
            r = run_sandboxed(work, cmd)
            info = {"exit": r["code"], "timeout": r["timeout"]}
            body = clip(r["out"].rstrip("\n"))
            if r["timeout"]:
                body += m("timeout", s=CMD_TIMEOUT)
            text = (body + "\n" if body else "") + m("exit", code=r["code"])
    elif name in ("read_file", "write_file", "edit_file"):
        full = resolve(work, args.get("path", ""))
        if full is None:
            text = m("outside")
        elif name == "read_file":
            if not full.is_file():
                text = m("not_found", path=args.get("path"))
            else:
                c = full.read_text()
                text = clip(c) if c else m("empty")
        elif name == "write_file":
            content = args.get("content")
            if not isinstance(content, str):
                text = m("no_content")
            else:
                full.parent.mkdir(parents=True, exist_ok=True)
                existed = full.exists()
                full.write_text(content)
                text = m("written", verb=m("overwrote") if existed else m("created"), path=args.get("path"), n=len(content.splitlines()))
        else:
            old, new = args.get("old_string"), args.get("new_string")
            if not full.is_file():
                text = m("not_found", path=args.get("path"))
            elif not isinstance(old, str) or not isinstance(new, str) or not old:
                text = m("no_old_new")
            else:
                c = full.read_text()
                n = c.count(old)
                if n == 0:
                    text = m("old_missing", path=args.get("path"))
                elif n > 1:
                    text = m("old_many", path=args.get("path"), n=n)
                else:
                    full.write_text(c.replace(old, new, 1))
                    text = m("edited", path=args.get("path"))
    else:
        text = m("no_tool", name=name)
    info["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    info["ok"] = not text.startswith(m("err_prefix"))
    return text, info


# ---------------------------------------------------------------- 词元显示

def byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DEC = byte_decoder()


def tok_bytes(tok, tid):
    s = tok.convert_ids_to_tokens(int(tid))
    if s in tok.all_special_tokens or s.startswith("<|") or s in ("<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"):
        return s.encode(), True
    try:
        return bytes(BYTE_DEC[c] for c in s), False
    except KeyError:
        return s.encode(), False


def tok_show(tok, tid):
    """候选词元的可读写法；不完整的 UTF-8 片段写成 <0x..>。"""
    raw, sp = tok_bytes(tok, tid)
    if sp:
        return raw.decode()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"<0x{b:02X}>" for b in raw)


def make_chips(tok, text):
    """输入法的候选：用户任务的真实分词。半个汉字的字节片段和后面的词元合成一块（ids 里保留每个真实词元的编号）。"""
    chips, ids, buf = [], [], b""
    for i in tok(text, add_special_tokens=False)["input_ids"]:
        raw, _ = tok_bytes(tok, i)
        ids.append(int(i))
        buf += raw
        try:
            s = buf.decode("utf-8")
        except UnicodeDecodeError:
            continue
        chips.append({"ids": ids, "s": s})
        ids, buf = [], b""
    return chips


# ---------------------------------------------------------------- 采样（与 transformers 的 temperature → top_k → top_p 一致）

def sample(logits, gen):
    logits = logits.float()
    p1 = torch.softmax(logits, -1)
    scaled = logits / TEMPERATURE
    kth = torch.topk(scaled, TOP_K).values[-1]
    scaled = torch.where(scaled < kth, torch.full_like(scaled, -float("inf")), scaled)
    sp, si = torch.sort(scaled, descending=True)
    cp = torch.softmax(sp, -1).cumsum(-1)
    remove = cp > TOP_P
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    keep = int((~remove).sum())
    final = torch.softmax(sp[:keep], -1)
    u = float(torch.rand(1, generator=gen, device=logits.device))
    idx = int(torch.searchsorted(final.cumsum(-1), torch.tensor([u], device=final.device)).clamp(max=keep - 1))
    chosen = int(si[idx])
    top = torch.topk(p1, TOPN)
    return chosen, float(p1[chosen]), list(zip(top.indices.tolist(), top.values.tolist())), keep


# ---------------------------------------------------------------- 上下文分段

def segments(text, offsets, msgs):
    """把渲染后的上下文切成：系统提示 / 工具定义 / 用户 / 助手（第 k 轮）/ 工具结果（第 k 轮）/ 生成提示。
    每段给出字符区间和真实词元数（按每个词元的起始字符归属）。"""
    spans = []
    starts = [m.start() for m in re.finditer(r"<\|im_start\|>", text)]
    starts.append(len(text))
    roles = [m for m in msgs]
    # 第一个块是 system（含工具定义）
    blocks = [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]
    # 消息与块对应：system→块0，之后每条 user/assistant 一个块，连续的 tool 合并成一个 user 块；最后一个块是生成提示
    mi, turn = 1, -1
    a, b = blocks[0]
    k = text.find("# Tools", a, b)
    spans.append(("sys", a, k if k > 0 else b, None))
    if k > 0:
        spans.append(("tools", k, b, None))
    for bi in range(1, len(blocks)):
        a, b = blocks[bi]
        if mi >= len(roles):
            spans.append(("gen", a, b, None))
            continue
        r = roles[mi]["role"]
        if r == "user":
            spans.append(("user", a, b, None))
            mi += 1
        elif r == "assistant":
            turn += 1
            spans.append(("asst", a, b, turn))
            mi += 1
        elif r == "tool":
            spans.append(("tool", a, b, turn))
            while mi < len(roles) and roles[mi]["role"] == "tool":
                mi += 1
    out = []
    ti = 0
    for kind, a, b, t in spans:
        n = 0
        while ti < len(offsets) and offsets[ti][0] < b:
            n += 1
            ti += 1
        out.append({"k": kind, "a": a, "b": b, "n": n, **({"t": t} if t is not None else {})})
    out[-1]["n"] += len(offsets) - ti
    return out


# ---------------------------------------------------------------- 解析工具调用

CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_calls(text):
    calls = []
    for m in CALL_RE.finditer(text):
        raw = m.group(1)
        try:
            obj = json.loads(raw)
            name, args = obj["name"], obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            calls.append({"name": name, "args": args, "a": m.start(), "b": m.end(), "ok": True})
        except Exception as e:  # 模型写出了不合法的 JSON：如实回报给它
            calls.append({"name": None, "args": None, "a": m.start(), "b": m.end(), "ok": False, "error": f"{type(e).__name__}: {e}"})
    if not calls and "<tool_call>" in text:   # 有开头没结尾
        a = text.index("<tool_call>")
        calls.append({"name": None, "args": None, "a": a, "b": len(text), "ok": False, "error": m("no_close")})
    say = text[: calls[0]["a"]].strip() if calls else text.strip()
    return say, calls


# ---------------------------------------------------------------- agent 循环

class Recorder:
    def __init__(self, model_path):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        free, total = torch.cuda.mem_get_info()
        print(f"GPU 空闲 {free / 2**30:.1f} / {total / 2**30:.1f} GiB", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        self.thinking_tpl = "enable_thinking" in (self.tok.chat_template or "")
        self.tools = TOOLS_EN if LANG == "en" else TOOLS
        self.system_tpl = SYSTEM_EN if LANG == "en" else SYSTEM
        cfg = self.model.config
        self.meta = {
            "name": pathlib.Path(model_path).name,
            "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size,
            "heads": cfg.num_attention_heads, "kvHeads": cfg.num_key_value_heads,
            "ffn": cfg.intermediate_size, "vocab": cfg.vocab_size,
            "ctx": getattr(cfg, "max_position_embeddings", None),
            "params": sum(p.numel() for p in self.model.parameters()),
            "dtype": "bfloat16",
            "sampling": {"temperature": TEMPERATURE, "top_k": TOP_K, "top_p": TOP_P},
            "gpu": torch.cuda.get_device_name(0),
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
        }

    def render(self, msgs):
        kw = {"enable_thinking": False} if self.thinking_tpl else {}
        return self.tok.apply_chat_template(msgs, tools=self.tools, tokenize=False, add_generation_prompt=True, **kw)

    @torch.no_grad()
    def run(self, task, seed):
        tok, model = self.tok, self.model
        tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"agent-{task['id']}-"))
        work = tmp / "work"
        shutil.copytree(SANDBOX_SRC / task["id"], work)
        files0 = snapshot(work)
        # 和 Claude Code 等 agent 一样，把运行环境（工作目录里有哪些文件）写进系统提示
        system = self.system_tpl.format(files="\n".join(sorted(files0)))
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": task["prompt"]}]
        gen = torch.Generator(device=model.device).manual_seed(seed)
        cache = DynamicCache()
        cached_ids = []          # KV 缓存里现在存着哪些词元
        prev_text = ""           # 上一轮的上下文 + 上一轮生成的文字
        turns = []
        final = None
        t_start = time.perf_counter()
        for ti in range(MAX_TURNS):
            text = self.render(msgs)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            segs = segments(text, offs, msgs)
            # 能复用多少 KV 缓存：和缓存里的词元逐个比对，取最长公共前缀
            lcp = 0
            while lcp < min(len(ids), len(cached_ids)) and ids[lcp] == cached_ids[lcp]:
                lcp += 1
            lcp = min(lcp, len(ids) - 1)
            if cache.get_seq_length() > lcp:
                cache.crop(lcp - cache.get_seq_length())   # 负数 = 从末尾删掉这么多词元
            keep = 0
            while keep < min(len(text), len(prev_text)) and text[keep] == prev_text[keep]:
                keep += 1
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(torch.tensor([ids[lcp:]], device=model.device), past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t0) * 1000
            logits = out.logits[0, -1]
            cached_ids = list(ids)
            gen_ids, toks = [], []
            buf = b""
            t1 = time.perf_counter()
            end = "max"
            for _ in range(MAX_NEW):
                chosen, p, top, pool = sample(logits, gen)
                if chosen in (151645, 151643):   # <|im_end|> / <|endoftext|>：这一轮说完了
                    toks.append({"s": "", "end": True, "p": round(p, 5), "top": [[tok_show(tok, i), round(v, 5)] for i, v in top], "pool": pool})
                    end = "im_end"
                    break
                gen_ids.append(chosen)
                raw, _ = tok_bytes(tok, chosen)
                buf += raw
                try:
                    s = buf.decode("utf-8")
                    buf = b""
                except UnicodeDecodeError:
                    s = ""   # 半个汉字：等下一个词元凑齐
                toks.append({"s": s, "p": round(p, 5), "top": [[tok_show(tok, i), round(v, 5)] for i, v in top], "pool": pool})
                out = model(torch.tensor([[chosen]], device=model.device), past_key_values=cache, use_cache=True)
                cached_ids.append(chosen)
                logits = out.logits[0, -1]
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter() - t1) * 1000
            gen_text = tok.decode(gen_ids, skip_special_tokens=False)
            say, calls = parse_calls(gen_text)
            turn = {
                "i": ti,
                "ctx": {"n": len(ids), "keep": keep, "add": text[keep:], "segs": segs},
                "kv": {"reused": lcp, "computed": len(ids) - lcp},
                "gen": {"text": gen_text, "toks": toks, "end": end, "n": len(gen_ids)},
                "ms": {"prefill": round(prefill_ms, 1), "decode": round(decode_ms, 1)},
                "say": say,
                "calls": [],
            }
            prev_text = text + gen_text
            print(f"  [{task['id']}] 第 {ti} 轮：上下文 {len(ids)} 词元（复用 {lcp}），生成 {len(gen_ids)} 词元，"
                  f"{len(calls)} 个工具调用 · {gen_text[:120]!r}", flush=True)
            if not calls:
                turns.append(turn)
                final = say
                break
            # 把助手这一轮记进历史（用官方聊天模板的 tool_calls 结构，和 vLLM 等服务端的做法一致）
            msgs.append({"role": "assistant", "content": say, "tool_calls": [
                {"type": "function", "function": {"name": c["name"] or "?", "arguments": c["args"] if c["ok"] else {}}} for c in calls]})
            for c in calls:
                before = snapshot(work)
                if c["ok"]:
                    resp, info = exec_tool(work, c["name"], c["args"] if isinstance(c["args"], dict) else {})
                else:
                    resp, info = m("bad_json", error=c["error"]), {"ok": False, "ms": 0}
                after = snapshot(work)
                d = diff_snap(before, after)
                rtoks = len(tok(resp, add_special_tokens=False)["input_ids"])
                turn["calls"].append({**c, "result": resp, "resultTokens": rtoks, "info": info, "diffs": d})
                msgs.append({"role": "tool", "content": resp})
                print(f"      → {c['name']} {json.dumps(c['args'], ensure_ascii=False)[:100]} ⇒ {resp[:100]!r}", flush=True)
            turns.append(turn)
        total_ms = (time.perf_counter() - t_start) * 1000
        files1 = snapshot(work)
        ok, why = task["check"](work, final or "", files0, turns)
        shutil.rmtree(tmp, ignore_errors=True)
        return {
            "id": task["id"], "title": task["title"], "prompt": task["prompt"], "seed": seed,
            "system": system, "tools": self.tools,
            "files0": files0, "files1": files1,
            "turns": turns, "final": final, "finished": final is not None,
            "check": {"ok": ok, "detail": why},
            "totalMs": round(total_ms),
        }


def write_json(path, obj):
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()
    path.write_bytes(data)
    (path.parent / (path.name + ".gz")).write_bytes(gzip.compress(data, 9, mtime=0))
    return len(data), (path.parent / (path.name + ".gz")).stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--only", default=None, help="只录这个任务 id")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7", help="依次尝试的随机种子，录到第一个通过检查的为止")
    ap.add_argument("--dry", action="store_true", help="只打印，不写文件")
    ap.add_argument("--chips-only", action="store_true", help="只用分词器重建 manifest 里的输入法候选，不跑模型")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"], help="en：英文提示与任务，输出到 data/en/")
    args = ap.parse_args()
    global OUT, LANG, TASKS
    if args.lang == "en":
        LANG, OUT = "en", DATA / "en"
        TASKS = [{**t, **TASKS_EN[t["id"]]} for t in TASKS]
    if args.chips_only:
        man_path = OUT / "manifest.json"
        manifest = json.loads(man_path.read_text())
        tok = AutoTokenizer.from_pretrained(args.model)
        for t in manifest["tasks"]:
            t["chips"] = make_chips(tok, t["prompt"])
        write_json(man_path, manifest)
        return
    if shutil.which("bwrap") is None:
        raise SystemExit("需要 bubblewrap（bwrap）来做沙箱")
    OUT.mkdir(parents=True, exist_ok=True)
    rec = Recorder(args.model)
    seeds = [int(s) for s in args.seeds.split(",")]
    man_path = OUT / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() and args.only else {"tasks": []}
    manifest["model"] = rec.meta
    if LANG == "en":
        manifest["lang"] = "en"
    by_id = {t["id"]: t for t in manifest.get("tasks", [])}
    for task in TASKS:
        if args.only and task["id"] != args.only:
            continue
        tries = []
        rec_ok = None
        for seed in seeds:
            print(f"== {task['id']} 种子 {seed}", flush=True)
            r = rec.run(task, seed)
            tries.append({"seed": seed, "ok": r["check"]["ok"], "turns": len(r["turns"]), "finished": r["finished"], "detail": r["check"]["detail"]})
            print(f"   检查：{'通过' if r['check']['ok'] else '失败'} · {r['check']['detail']}", flush=True)
            if r["check"]["ok"]:
                rec_ok = r
                break
        if rec_ok is None:
            print(f"!! {task['id']} 所有种子都没通过检查，不导出", flush=True)
            continue
        rec_ok["attempts"] = tries
        if args.dry:
            continue
        raw, gz = write_json(OUT / f"{task['id']}.json", rec_ok)
        last = rec_ok["turns"][-1]
        by_id[task["id"]] = {
            "id": task["id"], "title": task["title"], "prompt": task["prompt"], "icon": task["icon"], "blurb": task["blurb"],
            "chips": make_chips(rec.tok, task["prompt"]),
            "turns": len(rec_ok["turns"]), "calls": sum(len(t["calls"]) for t in rec_ok["turns"]),
            "ctxEnd": last["ctx"]["n"] + last["gen"]["n"],
            "attempts": tries, "bytes": gz, "rawBytes": raw,
        }
    manifest["tasks"] = [by_id[t["id"]] for t in TASKS if t["id"] in by_id]
    if not args.dry:
        write_json(man_path, manifest)
        print("写入", man_path, flush=True)


if __name__ == "__main__":
    main()
