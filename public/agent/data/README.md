# Agent 运行数据

这个目录包含真实的编程 Agent 运行录制数据。

## 文件结构

- `manifest.json` - 汇总文件，包含所有任务的列表
- `fix-bug.json` - Bug 修复任务数据
- `add-feature.json` - 功能添加任务数据
- `refactor.json` - 代码重构任务数据

## 数据格式

每个任务文件包含：

```json
{
  "task_id": "任务ID",
  "title": "任务标题",
  "description": "任务描述",
  "steps": [
    {
      "type": "think|tool|complete",
      "content": "思考内容",
      "status": "状态文本",
      "tool": "工具名",
      "command": "命令",
      "output": "输出",
      "file": "文件路径"
    }
  ],
  "tokenHistory": [512, 687, 923, ...],
  "generation": {
    "text": "生成的文本示例",
    "chars": [
      {"char": "字符", "prob": 0.95}
    ]
  },
  "elapsed_time": 15.8
}
```

## 步骤类型

### think - 思考步骤
```json
{
  "type": "think",
  "content": "Agent 的思考内容",
  "status": "状态描述"
}
```

### tool - 工具调用
```json
{
  "type": "tool",
  "tool": "bash|read_file|write_file|edit_file",
  "command": "命令内容（bash）",
  "file": "文件路径（文件操作）",
  "output": "工具输出",
  "content": "文件内容（read_file）"
}
```

### complete - 任务完成
```json
{
  "type": "complete",
  "summary": "完成总结",
  "files_changed": ["文件列表"],
  "tests_passed": 3
}
```

## 重新生成数据

使用模拟数据（不需要模型）：
```bash
python tools/export_agent_mock.py
```

使用真实模型（需要 Qwen3）：
```bash
python tools/export_agent.py --model /path/to/qwen3-7b
```
