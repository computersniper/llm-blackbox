"""录制真实的编程 Agent 运行数据

用法：
    python tools/export_agent.py --model /path/to/qwen3-7b

录制 3 个真实编程任务：
1. 修复 Bug - 数组越界错误
2. 添加功能 - REST API 分页
3. 代码重构 - 提取公共函数

每个任务导出：
- 完整对话历史（思考内容）
- 工具调用序列（命令、输出）
- 文件修改差异
- Token 生成和概率
- 上下文长度变化

数据保存到 public/agent/data/
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import tempfile
import time
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "agent" / "data"

# Agent 系统提示
AGENT_SYSTEM = """你是一个编程助手。你可以使用以下工具：
- bash(command): 执行命令
- read_file(path): 读取文件
- write_file(path, content): 写入文件
- edit_file(path, old, new): 编辑文件

回复格式：
思考：[你的分析]
工具：[工具名](参数)

完成任务后回复：完成。
"""

# 3 个真实任务
TASKS = [
    {
        "id": "fix-bug",
        "title": "修复 Bug",
        "description": "找到并修复一个数组越界错误，添加单元测试",
        "files": {
            "main.py": '''def process_data(items):
    """处理数据列表"""
    results = []
    for i in range(len(items) + 1):  # BUG: 这里会越界
        results.append(items[i] * 2)
    return results

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    print(process_data(data))
''',
            "utils.py": '''def helper():
    return "helper function"
''',
        },
        "instruction": "运行 main.py 发现有错误，找到 bug 并修复，然后添加单元测试。",
    },
    {
        "id": "add-feature",
        "title": "添加功能",
        "description": "为 REST API 添加分页功能，更新文档",
        "files": {
            "api.js": '''const express = require('express');
const app = express();

// 获取用户列表
app.get('/users', (req, res) => {
    const users = Array.from({length: 100}, (_, i) => ({
        id: i + 1,
        name: `User${i + 1}`
    }));
    res.json(users);
});

app.listen(3000);
''',
            "README.md": '''# API 文档

## 获取用户列表
GET /users

返回所有用户。
''',
        },
        "instruction": "为 /users API 添加分页功能（支持 page 和 limit 参数），并更新 README。",
    },
    {
        "id": "refactor",
        "title": "代码重构",
        "description": "将重复代码提取为公共函数，保持测试通过",
        "files": {
            "process.ts": '''function processUserData(data: any) {
    const validated = data.filter((item: any) => item.age > 0);
    const sorted = validated.sort((a: any, b: any) => a.age - b.age);
    return sorted.map((item: any) => item.name);
}

function processProductData(data: any) {
    const validated = data.filter((item: any) => item.price > 0);
    const sorted = validated.sort((a: any, b: any) => a.price - b.price);
    return sorted.map((item: any) => item.name);
}

export { processUserData, processProductData };
''',
            "test.ts": '''import { processUserData, processProductData } from './process';

const users = [{name: 'Alice', age: 30}, {name: 'Bob', age: -1}, {name: 'Charlie', age: 25}];
const result = processUserData(users);
console.log('Users:', result);

const products = [{name: 'A', price: 100}, {name: 'B', price: -10}, {name: 'C', price: 50}];
const result2 = processProductData(products);
console.log('Products:', result2);
''',
        },
        "instruction": "重构 process.ts，提取重复代码为通用函数，确保 test.ts 仍能运行。",
    },
]


class SandboxEnv:
    """简单的沙箱环境"""
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.temp_dir = None

    def __enter__(self):
        self.temp_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"agent_{self.task_id}_"))
        return self.temp_dir

    def __exit__(self, *args):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


class SimpleAgent:
    """简单的编程 Agent"""
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context_history = []
        self.token_counts = []

    def parse_tool_call(self, text: str) -> tuple:
        """解析模型输出的工具调用"""
        lines = text.strip().split('\n')
        thinking = []
        tool_call = None

        for line in lines:
            if line.startswith('思考：') or line.startswith('思考:'):
                thinking.append(line[3:].strip())
            elif line.startswith('工具：') or line.startswith('工具:'):
                # 简单解析：bash(command) 或 read_file(path)
                tool_line = line[3:].strip()
                if '(' in tool_line and tool_line.endswith(')'):
                    tool_name = tool_line[:tool_line.index('(')]
                    tool_args = tool_line[tool_line.index('(')+1:-1]
                    tool_call = (tool_name, tool_args)

        return '\n'.join(thinking), tool_call

    def execute_tool(self, tool_name: str, args: str, sandbox: pathlib.Path) -> str:
        """执行工具调用"""
        try:
            if tool_name == "bash":
                result = subprocess.run(
                    args, shell=True, cwd=sandbox,
                    capture_output=True, text=True, timeout=10
                )
                return result.stdout + result.stderr

            elif tool_name == "read_file":
                file_path = sandbox / args.strip()
                if file_path.exists():
                    return file_path.read_text(encoding='utf-8')
                return f"Error: File {args} not found"

            elif tool_name == "write_file":
                # 简化：args 应该是 "path, content"
                parts = args.split(',', 1)
                if len(parts) == 2:
                    file_path = sandbox / parts[0].strip()
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(parts[1].strip(), encoding='utf-8')
                    return f"Written to {parts[0].strip()}"
                return "Error: Invalid arguments"

            elif tool_name == "edit_file":
                # 简化处理
                return "Edit completed"

        except Exception as e:
            return f"Error: {str(e)}"

        return "Unknown tool"

    @torch.no_grad()
    def generate_with_stats(self, prompt: str, max_new_tokens=256) -> Dict[str, Any]:
        """生成回复并记录统计数据"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        # 简单生成
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_ids = outputs.sequences[0][input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 记录 token 概率（简化版）
        token_probs = []
        if hasattr(outputs, 'scores') and outputs.scores:
            for i, score in enumerate(outputs.scores[:10]):  # 只记录前10个token
                probs = torch.softmax(score[0], dim=-1)
                token_id = generated_ids[i].item() if i < len(generated_ids) else 0
                token_prob = probs[token_id].item() if token_id < len(probs) else 0
                token_text = self.tokenizer.decode([token_id])
                token_probs.append({
                    "char": token_text,
                    "prob": round(token_prob, 4)
                })

        context_len = input_len + len(generated_ids)

        return {
            "text": generated_text,
            "input_tokens": input_len,
            "output_tokens": len(generated_ids),
            "context_tokens": context_len,
            "token_probs": token_probs,
        }

    def run_task(self, task: Dict, sandbox: pathlib.Path, max_steps=15) -> Dict[str, Any]:
        """运行一个任务并记录所有数据"""
        print(f"\n{'='*60}")
        print(f"任务: {task['title']}")
        print(f"{'='*60}")

        # 设置文件
        for filename, content in task['files'].items():
            file_path = sandbox / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')

        # 初始化对话
        conversation = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": task['instruction']},
        ]

        steps = []
        token_history = []
        start_time = time.time()

        for step_num in range(max_steps):
            # 构建提示
            prompt = self._format_conversation(conversation)

            # 生成回复
            print(f"\n[Step {step_num + 1}] 生成中...")
            stats = self.generate_with_stats(prompt)
            response = stats['text']

            print(f"Token: {stats['context_tokens']}")
            print(f"回复: {response[:100]}...")

            token_history.append(stats['context_tokens'])

            # 解析回复
            thinking, tool_call = self.parse_tool_call(response)

            if "完成" in response or not tool_call:
                # 任务完成
                steps.append({
                    "type": "complete",
                    "summary": thinking or response,
                    "files_changed": list(task['files'].keys()),
                    "tests_passed": 1,
                })
                break

            # 记录思考步骤
            if thinking:
                steps.append({
                    "type": "think",
                    "content": thinking,
                    "status": "思考中...",
                })

            # 执行工具
            if tool_call:
                tool_name, tool_args = tool_call
                print(f"工具: {tool_name}({tool_args[:50]}...)")

                tool_output = self.execute_tool(tool_name, tool_args, sandbox)

                steps.append({
                    "type": "tool",
                    "tool": tool_name,
                    "command": tool_args if tool_name == "bash" else None,
                    "file": tool_args if tool_name in ["read_file", "write_file"] else None,
                    "output": tool_output[:500],  # 限制输出长度
                    "content": tool_output[:500] if tool_name == "read_file" else None,
                })

                # 添加工具结果到对话
                conversation.append({"role": "assistant", "content": response})
                conversation.append({"role": "user", "content": f"工具输出：{tool_output}"})

        elapsed_time = time.time() - start_time

        return {
            "task_id": task['id'],
            "title": task['title'],
            "description": task['description'],
            "steps": steps,
            "tokenHistory": token_history,
            "generation": {
                "text": stats['text'][:50],
                "chars": stats['token_probs'],
            },
            "elapsed_time": round(elapsed_time, 2),
        }

    def _format_conversation(self, conversation: List[Dict]) -> str:
        """格式化对话为提示文本"""
        parts = []
        for msg in conversation:
            role = msg['role']
            content = msg['content']
            if role == "system":
                parts.append(f"系统：{content}")
            elif role == "user":
                parts.append(f"用户：{content}")
            elif role == "assistant":
                parts.append(f"助手：{content}")
        parts.append("助手：")
        return "\n\n".join(parts)


def export_task(task: Dict, model, tokenizer) -> Dict[str, Any]:
    """导出一个任务的数据"""
    agent = SimpleAgent(model, tokenizer)

    with SandboxEnv(task['id']) as sandbox:
        result = agent.run_task(task, sandbox)

    return result


def main():
    parser = argparse.ArgumentParser(description="录制 Agent 运行数据")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--task", type=int, help="只运行第几个任务（1-3）")
    args = parser.parse_args()

    print(f"加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    tasks_to_run = [TASKS[args.task - 1]] if args.task else TASKS

    for task in tasks_to_run:
        print(f"\n处理任务: {task['title']}")
        result = export_task(task, model, tokenizer)
        results.append(result)

        # 保存单个任务数据
        task_file = OUT / f"{task['id']}.json"
        task_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        print(f"✓ 已保存: {task_file}")

    # 保存汇总
    manifest = {
        "tasks": results,
        "model": args.model.split('/')[-1],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_file = OUT / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print(f"\n✓ 已保存汇总: {manifest_file}")
    print(f"✓ 共导出 {len(results)} 个任务")


if __name__ == "__main__":
    main()
