"""让真实的 Qwen3-1.7B 当编程智能体，在沙箱目录里真的执行命令，把全过程录下来给 agent/ 页面。

用法（conda 环境 xnewenv）：
    python tools/record_agent.py --model D:/models/Qwen3-1.7B

每个任务：
    1. 在沙箱目录（默认 D:/models/agent_sandbox/<任务>）里放好一个小项目；
    2. 用 Qwen3 自带的工具调用格式（<tool_call>…</tool_call>），给模型两个工具：bash、write_file；
    3. 模型开思考模式逐个词元生成（T=0.6, top_k=20, top_p=0.95，固定种子），我们自己做采样，
       所以能记下每个词元的候选概率、截断后的候选池和抽签用的随机数；
    4. 解析出工具调用，在临时任务目录中按白名单执行，记下命令、stdout/stderr、退出码、耗时，
       以及执行前后项目里每个文件的 diff；把结果作为 tool 消息喂回去，直到模型不再调用工具。

执行规则：只接受列出的查看文件和运行测试命令，拒绝信息也原样喂回给模型；这不是操作系统隔离沙箱。

导出到 public/agent/data/：manifest.json（模型、工具定义、任务列表）和每个任务一个 <id>.json（另存 .gz）。
"""
import argparse
import ast
import difflib
import gzip
import json
import math
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import tempfile
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "agent" / "data"
PY_DIR = str(pathlib.Path(os.sys.executable).parent)

TEMP, TOP_K, TOP_P = 0.6, 20, 0.95
MAX_TURNS, MAX_NEW = 12, 2048
TOPN = 8

SYSTEM = "你是一个编程助手，当前目录是临时任务目录。bash 工具只接受 ls、ls -la、cat 文件名、python test_*.py、python -m pytest test_*.py -q、python report.py；修改文件请用 write_file。可以连续调用多个工具，执行结果会逐条返回。完成后用一两句话总结，并在回答前运行测试确认。"

TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "在临时任务目录执行受限命令，只支持 ls、ls -la、cat 文件名、python test_*.py、python -m pytest test_*.py -q、python report.py。返回 stdout、stderr 和退出码。",
        "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "要执行的命令"}}, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "把完整内容写入项目里的一个文件（覆盖原文件）。",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "相对于项目目录的文件路径"},
            "content": {"type": "string", "description": "文件的完整新内容"}}, "required": ["path", "content"]},
    }},
]

TASKS = [
    {
        "id": "fix-bug",
        "title": "修复一个 Bug",
        "prompt": "这个项目的单元测试没通过。请运行 python test_stats.py 看看哪里出错，找到 stats.py 里的 bug 并修复，最后再运行一次测试确认全部通过。",
        "files": {
            "stats.py": '''"""简单的统计工具"""


def mean(xs):
    return sum(xs) / len(xs)


def moving_average(xs, k):
    """长度为 k 的滑动平均：[1, 2, 3, 4], k=2 -> [1.5, 2.5, 3.5]"""
    out = []
    for i in range(len(xs) - k):
        out.append(mean(xs[i:i + k]))
    return out
''',
            "test_stats.py": '''from stats import mean, moving_average


def test_mean():
    assert mean([1, 2, 3]) == 2


def test_moving_average():
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]
    assert moving_average([5, 5, 5], 3) == [5.0]


if __name__ == "__main__":
    test_mean()
    test_moving_average()
    print("全部测试通过")
''',
        },
        "seed": 11,
    },
    {
        "id": "add-feature",
        "title": "添加一个功能",
        "prompt": "math_ops.py 已有 square(x)，请保留它，新增 sum_of_squares(a, b) 返回两个数的平方和。test_math_ops.py 已有固定测试，不需要修改测试文件。先运行 python -m pytest test_math_ops.py -q 观察失败，再修改 math_ops.py，最后用同一命令确认全部通过。write_file 会覆盖整个文件，写入时请包含原有内容。",
        "files": {
            "math_ops.py": '''"""简单的数学工具。"""


def square(x):
    return x * x
''',
            "test_math_ops.py": '''from math_ops import square, sum_of_squares


def test_square():
    assert square(4) == 16
    assert square(-3) == 9


def test_sum_of_squares():
    assert sum_of_squares(3, 4) == 25
    assert sum_of_squares(-2, 5) == 29
''',
        },
        "seed": 22,
    },
    {
        "id": "refactor",
        "title": "重构一段代码",
        "prompt": "formatter.py 的 full_name(first, last) 对两个名字重复做 strip().title()。请提取 normalize_name(name)，让 full_name 调用它两次，输出保持不变。test_formatter.py 已有固定测试，不需要修改测试文件。先运行 python -m pytest test_formatter.py -q，修改 formatter.py 后再运行一次确认通过。write_file 会覆盖整个文件，写入时请包含原有内容。",
        "files": {
            "formatter.py": '''"""把姓名整理成可显示的格式。"""


def full_name(first, last):
    first = first.strip().title()
    last = last.strip().title()
    return first + " " + last
''',
            "test_formatter.py": '''from formatter import full_name


def test_full_name():
    assert full_name(" ada ", " LOVELACE ") == "Ada Lovelace"


def test_blank_first_name():
    assert full_name("", " turing ") == " Turing"
''',
        },
        "seed": 33,
    },
    {
        "id": "analyze-data",
        "title": "分析一份数据",
        "prompt": "sales.csv 是上个月的销售记录。请写一个脚本 report.py，统计每个城市的销售总额，按从高到低打印出来，然后运行它，告诉我哪个城市卖得最多。",
        "files": {
            "sales.csv": "date,city,product,amount\n"
                         "2026-08-01,上海,咖啡机,1299\n2026-08-02,北京,咖啡豆,89\n2026-08-03,广州,咖啡机,1299\n"
                         "2026-08-05,上海,滤纸,25\n2026-08-06,成都,咖啡豆,178\n2026-08-08,北京,咖啡机,2598\n"
                         "2026-08-11,广州,磨豆机,459\n2026-08-12,上海,咖啡豆,267\n2026-08-15,成都,磨豆机,459\n"
                         "2026-08-18,北京,滤纸,50\n2026-08-21,上海,磨豆机,459\n2026-08-25,广州,咖啡豆,89\n"
                         "2026-08-28,成都,咖啡机,1299\n2026-08-30,北京,咖啡豆,178\n",
        },
        "seed": 44,
    },
]

# ---------------------------------------------------------------- 沙箱

def snapshot(root):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            try:
                out[p.relative_to(root).as_posix()] = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                out[p.relative_to(root).as_posix()] = "<二进制文件>"
    return out


def diff(before, after):
    out = []
    for f in sorted(set(before) | set(after)):
        a, b = before.get(f), after.get(f)
        if a == b:
            continue
        lines = list(difflib.unified_diff((a or "").splitlines(), (b or "").splitlines(),
                                          fromfile=f"a/{f}" if a is not None else "/dev/null",
                                          tofile=f"b/{f}", lineterm="", n=2))
        out.append({"file": f, "status": "added" if a is None else "deleted" if b is None else "modified",
                    "diff": lines, "after": b})
    return out


def run_bash(cmd, cwd):
    try:
        words = shlex.split(cmd)
    except ValueError:
        words = []
    safe_name = lambda s: bool(re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", s))
    if words in (["ls"], ["ls", "-la"]):
        return {"stdout": "\n".join(sorted(p.name for p in pathlib.Path(cwd).iterdir())) + "\n", "stderr": "", "code": 0, "ms": 0}
    if len(words) == 2 and words[0] == "cat" and safe_name(words[1]):
        p = pathlib.Path(cwd) / words[1]
        if p.is_file() and not p.is_symlink():
            return {"stdout": p.read_text(encoding="utf-8")[-4000:], "stderr": "", "code": 0, "ms": 0}
    direct = len(words) == 2 and words[0] == "python" and (re.fullmatch(r"test_[A-Za-z0-9_]+\.py", words[1]) or words[1] == "report.py")
    pytest = len(words) == 5 and words[:3] == ["python", "-m", "pytest"] and bool(re.fullmatch(r"test_[A-Za-z0-9_]+\.py", words[3])) and words[4] == "-q"
    target = words[1] if direct else words[3] if pytest else ""
    if not (target and (pathlib.Path(cwd) / target).is_file() and not (pathlib.Path(cwd) / target).is_symlink()):
        return {"stdout": "", "stderr": "受限执行拒绝：仅支持 ls、ls -la、cat 文件名、python test_*.py、python -m pytest test_*.py -q、python report.py", "code": 126, "ms": 0, "denied": True}
    env = minimal_env()
    t0 = time.time()
    try:
        argv = [os.sys.executable, target] if direct else [os.sys.executable, "-m", "pytest", target, "-q"]
        p = subprocess.run(argv, cwd=cwd, capture_output=True, timeout=30, env=env)
        out, err, code = p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace"), p.returncode
    except subprocess.TimeoutExpired:
        out, err, code = "", "超时（30 秒）", 124
    return {"stdout": out[-4000:], "stderr": err[-4000:], "code": code, "ms": int((time.time() - t0) * 1000)}


def minimal_env():
    env = {"PATH": PY_DIR, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    if "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return env


def write_file(path, content, cwd):
    if re.search(r"\.\.|^[/\\]|[A-Za-z]:", path):
        return {"stdout": "", "stderr": "沙箱拒绝：只能写项目目录里的文件", "code": 126, "ms": 0, "denied": True}
    p = pathlib.Path(cwd) / path
    # A small model sometimes double-escapes JSON newlines. Repair only when
    # the literal text is invalid Python and the decoded text parses cleanly.
    normalized = False
    if p.suffix == ".py" and "\\n" in content and "\n" not in content:
        try:
            ast.parse(content)
        except SyntaxError:
            decoded = content.replace("\\n", "\n").replace("\\t", "\t")
            try:
                ast.parse(decoded)
            except SyntaxError:
                pass
            else:
                content, normalized = decoded, True
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"stdout": f"已写入 {path}（{len(content.encode('utf-8'))} 字节）" + ("；已将双重转义的换行还原" if normalized else ""),
            "stderr": "", "code": 0, "ms": 0, "normalizedNewlines": normalized}


# ---------------------------------------------------------------- 生成

def tok_text(tok, i):
    return tok.decode([i], skip_special_tokens=False)


@torch.no_grad()
def generate(model, tok, ids, gen, stop_ids):
    """逐词元采样，记录每一步：前 TOPN 名（T=1 的原始概率）、选中词元、采样后的概率、候选池大小、随机数。"""
    dev = model.device
    out = model(torch.tensor([ids], device=dev), use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1].float()
    steps = []
    for _ in range(MAX_NEW):
        p1 = torch.softmax(logits, -1)
        top = torch.topk(p1, TOPN)
        lt = logits / TEMP
        kv, ki = torch.topk(lt, TOP_K)
        pk = torch.softmax(kv, -1)
        cum = torch.cumsum(pk, -1)
        keep = int((cum < TOP_P).sum()) + 1
        pool = pk[:keep] / pk[:keep].sum()
        u = float(torch.rand(1, generator=gen, device=dev))
        c = torch.cumsum(pool, -1)
        j = min(int((c < u).sum()), keep - 1)
        nxt = int(ki[j])
        ent = float(-(p1 * torch.log(p1.clamp_min(1e-12))).sum())
        steps.append({
            "id": nxt, "s": tok_text(tok, nxt), "p": round(float(p1[nxt]), 5), "q": round(float(pool[j]), 5),
            "pool": keep, "u": round(u, 5), "H": round(ent, 3),
            "top": [[tok_text(tok, int(i)), round(float(v), 5)] for v, i in zip(top.values, top.indices)],
            "samplePool": [[tok_text(tok, int(ki[z])), round(float(p1[ki[z]]), 7), round(float(pool[z]), 7)] for z in range(keep)],
        })
        if nxt in stop_ids:
            break
        out = model(torch.tensor([[nxt]], device=dev), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1].float()
    return steps


def parse(text):
    think, rest = "", text
    if "</think>" in text:
        think, rest = text.split("</think>", 1)
        think = think.replace("<think>", "").strip()
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", rest, re.S):
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            calls.append({"name": "_invalid", "arguments": {"raw": m.group(1)}})
    content = re.sub(r"<tool_call>.*?</tool_call>", "", rest, flags=re.S).replace("<|im_end|>", "").strip()
    return think, content, calls


# ---------------------------------------------------------------- 主流程

def write_json(path, obj):
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    path.with_suffix(path.suffix + ".gz").write_bytes(gzip.compress(raw, 9))
    return len(raw)


def run_task(task, model, tok, sandbox_root):
    box = pathlib.Path(sandbox_root) / task["id"]
    if box.exists():
        shutil.rmtree(box)
    box.mkdir(parents=True)
    for f, c in task["files"].items():
        (box / f).write_text(c, encoding="utf-8")
    initial = snapshot(box)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]
    gen = torch.Generator(device=model.device).manual_seed(task["seed"])
    stop_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    turns = []
    t0 = time.time()
    for turn in range(MAX_TURNS):
        ids = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, enable_thinking=True, tokenize=True)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        tg = time.time()
        steps = generate(model, tok, ids, gen, stop_ids)
        gen_s = time.time() - tg
        text = "".join(s["s"] for s in steps)
        think, content, calls = parse(text)
        rec = {"ctx": len(ids), "tokens": steps, "genSeconds": round(gen_s, 2), "think": think, "content": content,
               "calls": [], "text": text}
        print(f"  第 {turn + 1} 轮：上下文 {len(ids)} 词元，生成 {len(steps)} 个（{len(steps) / max(gen_s, 1e-3):.1f}/s）"
              f"，工具调用 {[c.get('name') for c in calls]}", flush=True)
        messages.append({"role": "assistant", "content": text.replace("<|im_end|>", "")})
        if not calls:
            check = verify_task(task, snapshot(box), box)
            rec["autoCheck"] = check
            turns.append(rec)
            if check["passed"]:
                break
            feedback = (f"自动验收未通过，请继续修改文件并重新运行测试，不要提前总结。\n"
                        f"公开测试：退出码 {check['code']}；{check['stderr'][-800:] or check['stdout'][-400:]}\n"
                        f"录制器持有的独立用例：{'通过' if check['oracle']['passed'] else '失败'}；"
                        f"{check['oracle']['stderr'][-800:]}\n"
                        f"测试有效性：{'通过' if check.get('mutation', {}).get('passed', True) else '失败，测试脚本未能检出故意注入的目标函数错误；请确保脚本真正执行断言'}")
            rec["autoFeedback"] = feedback
            messages.append({"role": "user", "content": feedback})
            print(f"    自动验收未通过，继续一轮（公开测试 {check['code']}，独立用例 {check['oracle']['passed']}）", flush=True)
            continue
        for call in calls:
            before = snapshot(box)
            args = call.get("arguments", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if call.get("name") == "bash":
                res = run_bash(str(args.get("command", "")), box)
            elif call.get("name") == "write_file":
                res = write_file(str(args.get("path", "")), str(args.get("content", "")), box)
            else:
                res = {"stdout": "", "stderr": f"未知工具：{call.get('name')}", "code": 127, "ms": 0}
            after = snapshot(box)
            changes = diff(before, after)
            obs = res["stdout"] + (("\n" if res["stdout"] else "") + res["stderr"] if res["stderr"] else "")
            obs = f"{obs.strip()}\n[退出码 {res['code']}]" if call.get("name") == "bash" else obs.strip()
            rec["calls"].append({"name": call.get("name"), "args": args, "result": res, "changes": changes, "observation": obs})
            messages.append({"role": "tool", "content": obs})
            print(f"    $ {args.get('command') or args.get('path')}  → 退出码 {res['code']}", flush=True)
        turns.append(rec)
    final = snapshot(box)
    verification = verify_task(task, final, box)
    return {
        "id": task["id"], "title": task["title"], "prompt": task["prompt"], "system": SYSTEM,
        "initialFiles": initial, "finalFiles": final, "totalDiff": diff(initial, final),
        "turns": turns, "seconds": round(time.time() - t0, 1), "verification": verification,
        "sampling": {"temperature": TEMP, "top_k": TOP_K, "top_p": TOP_P, "seed": task["seed"]},
    }


def verify_task(task, final, box):
    check_cmd = {"fix-bug": "python test_stats.py", "add-feature": "python -m pytest test_math_ops.py -q", "refactor": "python -m pytest test_formatter.py -q", "analyze-data": "python report.py"}[task["id"]]
    verification = {"command": check_cmd, **run_bash(check_cmd, box)}
    oracle = {
        "fix-bug": "ns=runpy.run_path('stats.py'); assert ns['mean']([1,2,3])==2; assert ns['moving_average']([1,2,3,4],2)==[1.5,2.5,3.5]; assert ns['moving_average']([5,5,5],3)==[5.0]",
        "add-feature": "ns=runpy.run_path('math_ops.py'); assert ns['square'](4)==16; assert ns['square'](-3)==9; assert ns['sum_of_squares'](3,4)==25; assert ns['sum_of_squares'](-2,5)==29",
        "refactor": "ns=runpy.run_path('formatter.py'); assert ns['normalize_name'](' aDA ')=='Ada'; assert ns['full_name'](' ada ',' lovelace ')=='Ada Lovelace'; assert ns['full_name']('',' turing ')==' Turing'",
    }
    if task["id"] in oracle:
        code = "import runpy; " + oracle[task["id"]]
        start = time.time()
        try:
            p = subprocess.run([os.sys.executable, "-I", "-c", code], cwd=box, capture_output=True, timeout=30, env=minimal_env())
            verification["oracle"] = {"passed": p.returncode == 0, "stdout": p.stdout.decode("utf-8", "replace")[-2000:],
                                      "stderr": p.stderr.decode("utf-8", "replace")[-2000:], "code": p.returncode,
                                      "ms": int((time.time() - start) * 1000)}
        except subprocess.TimeoutExpired:
            verification["oracle"] = {"passed": False, "stdout": "", "stderr": "独立测试超时", "code": 124, "ms": 30000}
    else:
        verification["oracle"] = {"passed": "北京" in verification["stdout"] and "2915" in verification["stdout"], "stdout": "", "stderr": "", "code": verification["code"], "ms": 0}
    if task["id"] == "add-feature":
        verification["mutation"] = mutation_check(final, "math_ops.py", "test_math_ops.py", "sum_of_squares", -999999)
        verification["passed"] = verification["code"] == 0 and verification["oracle"]["passed"] and verification["mutation"]["passed"] and "def sum_of_squares(" in final.get("math_ops.py", "") and "sum_of_squares" in final.get("test_math_ops.py", "")
    elif task["id"] == "refactor":
        verification["mutation"] = mutation_check(final, "formatter.py", "test_formatter.py", "normalize_name", "WRONG")
        verification["passed"] = verification["code"] == 0 and verification["oracle"]["passed"] and verification["mutation"]["passed"] and "def normalize_name(" in final.get("formatter.py", "") and "normalize_name(" in final.get("formatter.py", "").split("def full_name", 1)[-1]
    elif task["id"] == "analyze-data":
        verification["passed"] = verification["code"] == 0 and verification["oracle"]["passed"]
    else:
        verification["passed"] = verification["code"] == 0 and verification["oracle"]["passed"]
    return verification


def mutation_check(final, source_file, test_file, function_name, wrong_value):
    """公开测试必须在隔离副本中检出目标函数被故意改错。"""
    try:
        with tempfile.TemporaryDirectory(prefix="blackbox-agent-mutant-") as tmp:
            path = pathlib.Path(tmp)
            (path / source_file).write_text(final[source_file], encoding="utf-8")
            (path / test_file).write_text(final[test_file], encoding="utf-8")
            argv = [os.sys.executable, "-m", "pytest", test_file, "-q"]
            baseline = subprocess.run(argv, cwd=path, capture_output=True,
                                      timeout=30, env=minimal_env())
            if baseline.returncode != 0:
                return {"passed": False, "baselineCode": baseline.returncode, "code": baseline.returncode,
                        "stdout": baseline.stdout.decode("utf-8", "replace")[-2000:],
                        "stderr": baseline.stderr.decode("utf-8", "replace")[-2000:]}
            tree = ast.parse(final[source_file])
            target = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function_name)
            target.body = [ast.Return(value=ast.Constant(value=wrong_value))]
            ast.fix_missing_locations(tree)
            (path / source_file).write_text(ast.unparse(tree), encoding="utf-8")
            cache = path / "__pycache__"
            if cache.exists():
                shutil.rmtree(cache)
            mutant = subprocess.run(argv, cwd=path, capture_output=True,
                                    timeout=30, env=minimal_env())
            return {"passed": mutant.returncode != 0, "baselineCode": 0, "code": mutant.returncode,
                    "stdout": mutant.stdout.decode("utf-8", "replace")[-2000:],
                    "stderr": mutant.stderr.decode("utf-8", "replace")[-2000:]}
    except (KeyError, SyntaxError, StopIteration, subprocess.TimeoutExpired) as exc:
        return {"passed": False, "code": 124, "stdout": "", "stderr": str(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="D:/models/Qwen3-1.7B")
    ap.add_argument("--sandbox", default="D:/models/agent_sandbox")
    ap.add_argument("--only", default=None)
    ap.add_argument("--seed", type=int, default=None, help="覆盖所选任务的随机种子，写入录制元数据")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    cfg = model.config
    for task in TASKS:
        if args.only and task["id"] != args.only:
            continue
        if args.seed is not None:
            task = {**task, "seed": args.seed}
        print(f"任务 {task['id']}：{task['prompt'][:40]}…", flush=True)
        rec = run_task(task, model, tok, args.sandbox)
        n = write_json(OUT / f"{task['id']}.json", rec)
        ntok = sum(len(t["tokens"]) for t in rec["turns"])
        print(f"  → {len(rec['turns'])} 轮，{ntok} 个词元，{n / 1e6:.2f} MB", flush=True)
    mpath = OUT / "manifest.json"
    tasks_meta = []
    for task in TASKS:
        path = OUT / f"{task['id']}.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("id") != task["id"] or not isinstance(rec.get("turns"), list):
            continue  # 旧版 mock 格式不进入真实录制列表
        if "verification" not in rec:
            rec["verification"] = verify_task(task, rec["finalFiles"], pathlib.Path(args.sandbox) / task["id"])
            write_json(path, rec)
        if not rec["verification"]["passed"]:
            continue  # 失败记录留在磁盘供诊断，但不作为可播放的完成任务发布
        tasks_meta.append({"id": task["id"], "title": task["title"], "prompt": task["prompt"], "turns": len(rec["turns"]),
                           "tokens": sum(len(t["tokens"]) for t in rec["turns"]),
                           "commands": sum(len(t["calls"]) for t in rec["turns"]), "seconds": rec["seconds"],
                           "verified": rec["verification"]["passed"]})
    manifest = {
        "model": {"name": "Qwen3-1.7B", "source": "https://huggingface.co/Qwen/Qwen3-1.7B", "layers": cfg.num_hidden_layers,
                  "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads, "kvHeads": cfg.num_key_value_heads,
                  "vocab": cfg.vocab_size, "params": sum(p.numel() for p in model.parameters()), "dtype": "bfloat16"},
        "sampling": {"temperature": TEMP, "top_k": TOP_K, "top_p": TOP_P, "thinking": True},
        "system": SYSTEM, "tools": TOOLS,
        "tasks": tasks_meta,
    }
    write_json(mpath, manifest)


if __name__ == "__main__":
    main()
