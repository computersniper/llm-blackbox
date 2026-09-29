"""生成模拟的 Agent 运行数据（用于演示，不需要真实模型）

用法：
    python tools/export_agent_mock.py

生成 3 个任务的录制数据，保存到 public/agent/data/
"""
import json
import pathlib
import random
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "agent" / "data"


def generate_bug_fix_task():
    """生成 Bug 修复任务数据"""
    return {
        "task_id": "fix-bug",
        "title": "修复 Bug",
        "description": "找到并修复一个数组越界错误，添加单元测试",
        "steps": [
            {
                "type": "think",
                "content": "我需要先了解项目结构，看看哪里有数组越界的问题。",
                "status": "分析任务..."
            },
            {
                "type": "tool",
                "tool": "bash",
                "command": "ls -la",
                "output": "total 24\ndrwxr-xr-x  5 user  staff  160 Jan 15 10:30 .\ndrwxr-xr-x  8 user  staff  256 Jan 15 10:29 ..\n-rw-r--r--  1 user  staff  234 Jan 15 10:30 main.py\n-rw-r--r--  1 user  staff  156 Jan 15 10:30 utils.py\ndrwxr-xr-x  3 user  staff   96 Jan 15 10:30 tests"
            },
            {
                "type": "think",
                "content": "好的，项目结构很简单。让我先读取 main.py 看看代码。",
                "status": "检查代码..."
            },
            {
                "type": "tool",
                "tool": "read_file",
                "file": "main.py",
                "content": '''def process_data(items):
    """处理数据列表"""
    results = []
    for i in range(len(items) + 1):  # BUG: 这里会越界
        results.append(items[i] * 2)
    return results

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    print(process_data(data))'''
            },
            {
                "type": "think",
                "content": "找到问题了！在第 4 行，`range(len(items) + 1)` 会导致索引超出范围。应该是 `range(len(items))`。让我修复它。",
                "status": "发现问题..."
            },
            {
                "type": "tool",
                "tool": "edit_file",
                "file": "main.py",
                "changes": [
                    {"line": 4, "old": "    for i in range(len(items) + 1):  # BUG: 这里会越界", "new": "    for i in range(len(items)):"}
                ]
            },
            {
                "type": "think",
                "content": "现在让我运行代码看看是否修复成功。",
                "status": "测试修复..."
            },
            {
                "type": "tool",
                "tool": "bash",
                "command": "python main.py",
                "output": "[2, 4, 6, 8, 10]"
            },
            {
                "type": "think",
                "content": "很好！代码现在正常运行了。接下来我需要添加单元测试来防止这个问题再次出现。",
                "status": "编写测试..."
            },
            {
                "type": "tool",
                "tool": "write_file",
                "file": "tests/test_main.py",
                "content": '''import unittest
import sys
sys.path.append('..')
from main import process_data

class TestProcessData(unittest.TestCase):
    def test_normal_case(self):
        """测试正常情况"""
        result = process_data([1, 2, 3])
        self.assertEqual(result, [2, 4, 6])

    def test_empty_list(self):
        """测试空列表"""
        result = process_data([])
        self.assertEqual(result, [])

    def test_single_item(self):
        """测试单个元素"""
        result = process_data([5])
        self.assertEqual(result, [10])

if __name__ == '__main__':
    unittest.main()'''
            },
            {
                "type": "tool",
                "tool": "bash",
                "command": "cd tests && python -m pytest test_main.py -v",
                "output": "============================= test session starts ==============================\ntest_main.py::TestProcessData::test_empty_list PASSED              [ 33%]\ntest_main.py::TestProcessData::test_normal_case PASSED             [ 66%]\ntest_main.py::TestProcessData::test_single_item PASSED             [100%]\n\n============================== 3 passed in 0.12s ==============================="
            },
            {
                "type": "complete",
                "summary": "任务完成！我修复了数组越界错误，并添加了 3 个单元测试来确保代码的正确性。",
                "files_changed": ["main.py", "tests/test_main.py"],
                "tests_passed": 3
            }
        ],
        "tokenHistory": [512, 687, 923, 1156, 1489, 1723, 1956, 2234, 2567, 2801],
        "generation": {
            "text": "for i in range(len(items)):",
            "chars": [
                {"char": "f", "prob": 0.94},
                {"char": "o", "prob": 0.98},
                {"char": "r", "prob": 0.99},
                {"char": " ", "prob": 0.97},
                {"char": "i", "prob": 0.92},
                {"char": " ", "prob": 0.96},
                {"char": "i", "prob": 0.89},
                {"char": "n", "prob": 0.95},
                {"char": " ", "prob": 0.98},
                {"char": "r", "prob": 0.88},
                {"char": "a", "prob": 0.94},
                {"char": "n", "prob": 0.97},
                {"char": "g", "prob": 0.99},
                {"char": "e", "prob": 0.98},
                {"char": "(", "prob": 0.96},
                {"char": "l", "prob": 0.91},
                {"char": "e", "prob": 0.95},
                {"char": "n", "prob": 0.98},
                {"char": "(", "prob": 0.97},
                {"char": "i", "prob": 0.93},
                {"char": "t", "prob": 0.96},
                {"char": "e", "prob": 0.98},
                {"char": "m", "prob": 0.97},
                {"char": "s", "prob": 0.99},
                {"char": ")", "prob": 0.98},
                {"char": ")", "prob": 0.96},
                {"char": ":", "prob": 0.95}
            ]
        },
        "elapsed_time": 15.8
    }


def generate_add_feature_task():
    """生成添加功能任务数据"""
    return {
        "task_id": "add-feature",
        "title": "添加功能",
        "description": "为 REST API 添加分页功能，更新文档",
        "steps": [
            {
                "type": "think",
                "content": "我需要为 /users API 添加分页功能。首先看看现有的代码结构。",
                "status": "分析需求..."
            },
            {
                "type": "tool",
                "tool": "read_file",
                "file": "api.js",
                "content": '''const express = require('express');
const app = express();

// 获取用户列表
app.get('/users', (req, res) => {
    const users = Array.from({length: 100}, (_, i) => ({
        id: i + 1,
        name: `User${i + 1}`
    }));
    res.json(users);
});

app.listen(3000);'''
            },
            {
                "type": "think",
                "content": "现在返回所有用户。我需要添加 page 和 limit 参数来支持分页。",
                "status": "设计方案..."
            },
            {
                "type": "tool",
                "tool": "edit_file",
                "file": "api.js",
                "changes": [
                    {"line": 5, "old": "app.get('/users', (req, res) => {", "new": "app.get('/users', (req, res) => {\n    const page = parseInt(req.query.page) || 1;\n    const limit = parseInt(req.query.limit) || 10;"}
                ]
            },
            {
                "type": "think",
                "content": "现在需要实现分页逻辑，计算起始位置和返回数据。",
                "status": "实现分页..."
            },
            {
                "type": "tool",
                "tool": "write_file",
                "file": "api.js",
                "content": '''const express = require('express');
const app = express();

// 获取用户列表
app.get('/users', (req, res) => {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;

    const allUsers = Array.from({length: 100}, (_, i) => ({
        id: i + 1,
        name: `User${i + 1}`
    }));

    const startIndex = (page - 1) * limit;
    const endIndex = startIndex + limit;
    const users = allUsers.slice(startIndex, endIndex);

    res.json({
        page,
        limit,
        total: allUsers.length,
        data: users
    });
});

app.listen(3000);'''
            },
            {
                "type": "think",
                "content": "代码完成了。现在需要更新 README 文档来说明新的 API 参数。",
                "status": "更新文档..."
            },
            {
                "type": "tool",
                "tool": "write_file",
                "file": "README.md",
                "content": '''# API 文档

## 获取用户列表
GET /users

### 查询参数
- `page` (可选): 页码，默认为 1
- `limit` (可选): 每页数量，默认为 10

### 返回格式
```json
{
  "page": 1,
  "limit": 10,
  "total": 100,
  "data": [...]
}
```

### 示例
```
GET /users?page=2&limit=20
```'''
            },
            {
                "type": "tool",
                "tool": "bash",
                "command": "node api.js &",
                "output": "Server started on port 3000"
            },
            {
                "type": "tool",
                "tool": "bash",
                "command": "curl 'http://localhost:3000/users?page=1&limit=5'",
                "output": '{"page":1,"limit":5,"total":100,"data":[{"id":1,"name":"User1"},{"id":2,"name":"User2"},{"id":3,"name":"User3"},{"id":4,"name":"User4"},{"id":5,"name":"User5"}]}'
            },
            {
                "type": "complete",
                "summary": "任务完成！已为 /users API 添加分页功能，支持 page 和 limit 参数，并更新了文档。",
                "files_changed": ["api.js", "README.md"],
                "tests_passed": 1
            }
        ],
        "tokenHistory": [534, 789, 1123, 1567, 1998, 2345, 2678, 2912, 3145],
        "generation": {
            "text": "const page = parseInt(req.query.page)",
            "chars": [
                {"char": "c", "prob": 0.91},
                {"char": "o", "prob": 0.96},
                {"char": "n", "prob": 0.98},
                {"char": "s", "prob": 0.97},
                {"char": "t", "prob": 0.95},
                {"char": " ", "prob": 0.99},
                {"char": "p", "prob": 0.87},
                {"char": "a", "prob": 0.93},
                {"char": "g", "prob": 0.96},
                {"char": "e", "prob": 0.98}
            ]
        },
        "elapsed_time": 18.3
    }


def generate_refactor_task():
    """生成重构任务数据"""
    return {
        "task_id": "refactor",
        "title": "代码重构",
        "description": "将重复代码提取为公共函数，保持测试通过",
        "steps": [
            {
                "type": "think",
                "content": "让我先看看代码，找出重复的部分。",
                "status": "分析代码..."
            },
            {
                "type": "tool",
                "tool": "read_file",
                "file": "process.ts",
                "content": '''function processUserData(data: any) {
    const validated = data.filter((item: any) => item.age > 0);
    const sorted = validated.sort((a: any, b: any) => a.age - b.age);
    return sorted.map((item: any) => item.name);
}

function processProductData(data: any) {
    const validated = data.filter((item: any) => item.price > 0);
    const sorted = validated.sort((a: any, b: any) => a.price - b.price);
    return sorted.map((item: any) => item.name);
}

export { processUserData, processProductData };'''
            },
            {
                "type": "think",
                "content": "我发现两个函数的逻辑几乎相同：过滤、排序、映射。唯一的区别是字段名（age vs price）。我可以提取一个通用的处理函数。",
                "status": "识别模式..."
            },
            {
                "type": "tool",
                "tool": "write_file",
                "file": "process.ts",
                "content": '''function processData<T>(
    data: T[],
    filterKey: keyof T,
    sortKey: keyof T
): string[] {
    const validated = data.filter((item: T) => (item[filterKey] as number) > 0);
    const sorted = validated.sort((a: T, b: T) =>
        (a[sortKey] as number) - (b[sortKey] as number)
    );
    return sorted.map((item: any) => item.name);
}

function processUserData(data: any) {
    return processData(data, 'age', 'age');
}

function processProductData(data: any) {
    return processData(data, 'price', 'price');
}

export { processUserData, processProductData };'''
            },
            {
                "type": "think",
                "content": "重构完成！现在有一个通用的 processData 函数。让我运行测试确保没有破坏功能。",
                "status": "运行测试..."
            },
            {
                "type": "tool",
                "tool": "bash",
                "command": "npx ts-node test.ts",
                "output": "Users: [ 'Charlie', 'Alice' ]\nProducts: [ 'C', 'A' ]"
            },
            {
                "type": "think",
                "content": "完美！测试通过了。重构成功保持了原有功能。",
                "status": "验证完成..."
            },
            {
                "type": "complete",
                "summary": "任务完成！已将重复代码提取为通用的 processData 函数，使用泛型确保类型安全，测试全部通过。",
                "files_changed": ["process.ts"],
                "tests_passed": 2
            }
        ],
        "tokenHistory": [498, 756, 1089, 1456, 1823, 2067, 2234],
        "generation": {
            "text": "function processData<T>(",
            "chars": [
                {"char": "f", "prob": 0.93},
                {"char": "u", "prob": 0.97},
                {"char": "n", "prob": 0.99},
                {"char": "c", "prob": 0.98},
                {"char": "t", "prob": 0.96},
                {"char": "i", "prob": 0.98},
                {"char": "o", "prob": 0.99},
                {"char": "n", "prob": 0.98}
            ]
        },
        "elapsed_time": 12.5
    }


def main():
    """生成所有任务数据"""
    print("生成模拟的 Agent 运行数据...")

    OUT.mkdir(parents=True, exist_ok=True)

    tasks = [
        generate_bug_fix_task(),
        generate_add_feature_task(),
        generate_refactor_task(),
    ]

    # 保存每个任务
    for task in tasks:
        task_file = OUT / f"{task['task_id']}.json"
        task_file.write_text(
            json.dumps(task, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        print(f"✓ 已生成: {task_file}")

    # 保存汇总
    manifest = {
        "tasks": tasks,
        "model": "Qwen3-7B (模拟数据)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_file = OUT / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print(f"✓ 已生成汇总: {manifest_file}")
    print(f"\n✓ 共生成 {len(tasks)} 个任务的数据")
    print(f"✓ 数据保存在: {OUT}")


if __name__ == "__main__":
    main()
